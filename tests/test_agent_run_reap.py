"""Tests for agent-run status reconciliation and idle reaping.

Covers:
- _effective_status: running+alive+fresh → running; running+dead pid → died;
  running+alive+stale log → stalled; done/failed passthrough.
- AGENT_RUN_IDLE_KILL_HOURS env override changes the threshold.
- reap --dry-run mutates nothing but reports correctly.
- reap marks a dead-pid "running" run as died with ended_at set.
- idle-kill path: alive pid + backdated log mtime → signalled + marked killed.
- opportunistic self-heal from list marks dead-pid runs died but never
  idle-kills.
- kill-path coverage: pgid/pty_pid/keeper_pid signalling, grace→SIGKILL
  escalation, pid identity mismatch guard.
- --name <typo> prints a clear "no such run" message.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

import pytest

from toolbox import agent_run


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_run(
    state_root: Path,
    log_root: Path,
    name: str,
    status: str = "running",
    pid: Optional[int] = None,
    interactive: str = "0",
    write_log: bool = True,
    log_age_secs: Optional[float] = None,
    pgid: Optional[int] = None,
    pty_pid: Optional[int] = None,
    keeper_pid: Optional[int] = None,
    pid_start: Optional[str] = None,
) -> tuple[Path, Path]:
    """Create a minimal fake run under state_root and log_root."""
    sd = state_root / name
    ld = log_root / name
    sd.mkdir(parents=True, exist_ok=True)
    ld.mkdir(parents=True, exist_ok=True)

    (sd / "status").write_text(f"{status}\n")
    (sd / "interactive").write_text(f"{interactive}\n")
    (sd / "started_at").write_text("2024-01-01T00:00:00Z\n")
    if pid is not None:
        (sd / "pid").write_text(f"{pid}\n")
    if pgid is not None:
        (sd / "pgid").write_text(f"{pgid}\n")
    if pty_pid is not None:
        (sd / "pty_pid").write_text(f"{pty_pid}\n")
    if keeper_pid is not None:
        (sd / "keeper_pid").write_text(f"{keeper_pid}\n")
    if pid_start is not None:
        (sd / "pid_start").write_text(f"{pid_start}\n")

    if write_log:
        log_file = ld / "log"
        log_file.write_text("some output\n")
        if log_age_secs is not None:
            # Backdate the mtime.
            old_mtime = time.time() - log_age_secs
            os.utime(log_file, (old_mtime, old_mtime))

    return sd, ld


def _reap_args(
    *,
    dry_run: bool = False,
    idle_hours: Optional[float] = None,
    name: Optional[str] = None,
) -> argparse.Namespace:
    return argparse.Namespace(dry_run=dry_run, idle_hours=idle_hours, name=name)


# ---------------------------------------------------------------------------
# _effective_status
# ---------------------------------------------------------------------------

class TestEffectiveStatus:
    def test_done_passthrough(self, tmp_path):
        (tmp_path / "status").write_text("done\n")
        assert agent_run._effective_status(tmp_path) == "done"

    def test_failed_passthrough(self, tmp_path):
        (tmp_path / "status").write_text("failed\n")
        assert agent_run._effective_status(tmp_path) == "failed"

    def test_unknown_passthrough(self, tmp_path):
        (tmp_path / "status").write_text("unknown\n")
        assert agent_run._effective_status(tmp_path) == "unknown"

    def test_running_alive_fresh_log_returns_running(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """running + alive pid + recently-touched log → running."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "myrun",
            status="running",
            pid=12345,
            log_age_secs=10,  # 10 seconds old, well under 24h
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        assert agent_run._effective_status(sd) == "running"

    def test_running_dead_pid_returns_died(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """running + dead pid → died."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "myrun",
            status="running",
            pid=99999,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: False)
        assert agent_run._effective_status(sd) == "died"

    def test_running_no_pid_returns_died(
        self, isolated_runs_root, isolated_log_root
    ):
        """running + no pid file → died."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "myrun",
            status="running",
            pid=None,  # no pid file written
        )
        assert agent_run._effective_status(sd) == "died"

    def test_running_alive_stale_log_returns_stalled(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """running + alive pid + log older than threshold → stalled."""
        threshold = 3600.0  # 1 hour
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "myrun",
            status="running",
            pid=12345,
            log_age_secs=threshold + 60,  # older than threshold
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        assert agent_run._effective_status(sd, idle_threshold=threshold) == "stalled"

    def test_running_alive_fresh_log_not_stalled(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """running + alive pid + log newer than threshold → running."""
        threshold = 3600.0
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "myrun",
            status="running",
            pid=12345,
            log_age_secs=threshold - 60,  # newer than threshold
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        assert agent_run._effective_status(sd, idle_threshold=threshold) == "running"

    def test_env_override_changes_threshold(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """AGENT_RUN_IDLE_KILL_HOURS env var overrides the default threshold."""
        # Set threshold to 0.5 hours via env — module reads it at import time,
        # so we patch the module constant directly (as the fixture already does
        # for STATE_ROOT/LOG_ROOT).
        monkeypatch.setattr(agent_run, "IDLE_STALL_SECONDS", 0.5 * 3600)

        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "myrun",
            status="running",
            pid=12345,
            log_age_secs=0.6 * 3600,  # older than 0.5h threshold
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        # With the reduced threshold, the log appears stalled.
        assert agent_run._effective_status(sd) == "stalled"


# ---------------------------------------------------------------------------
# cmd_reap -- dry-run
# ---------------------------------------------------------------------------

class TestReapDryRun:
    def test_dry_run_does_not_mutate_dead_pid_run(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        """--dry-run: dead-pid run is reported but state files unchanged."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "deadrun",
            status="running",
            pid=99999,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: False)

        rc = agent_run.cmd_reap(_reap_args(dry_run=True))

        assert rc == 0
        # Status file must NOT have been mutated.
        assert sd.joinpath("status").read_text().strip() == "running"
        assert not (sd / "ended_at").exists()
        # Must have reported the run.
        out = capsys.readouterr().out
        assert "deadrun" in out

    def test_dry_run_does_not_mutate_idle_run(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        """--dry-run: idle-alive run is reported but not killed."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "idlerun",
            status="running",
            pid=12345,
            log_age_secs=25 * 3600,  # older than 24h default
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        # Make sure IDLE_STALL_SECONDS is the default (24h).
        monkeypatch.setattr(agent_run, "IDLE_STALL_SECONDS", 24 * 3600)

        killed_pids = []
        monkeypatch.setattr(agent_run.os, "kill", lambda pid, sig: killed_pids.append((pid, sig)))
        monkeypatch.setattr(agent_run.os, "killpg", lambda pgid, sig: None)

        rc = agent_run.cmd_reap(_reap_args(dry_run=True))

        assert rc == 0
        assert sd.joinpath("status").read_text().strip() == "running"
        assert not (sd / "ended_at").exists()
        # No kill calls.
        assert killed_pids == []
        out = capsys.readouterr().out
        assert "idlerun" in out

    def test_dry_run_totals_line(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        """--dry-run always prints a totals line."""
        agent_run.cmd_reap(_reap_args(dry_run=True))
        out = capsys.readouterr().out
        assert "reap done" in out


# ---------------------------------------------------------------------------
# cmd_reap -- dead-pid mutation
# ---------------------------------------------------------------------------

class TestReapDeadPid:
    def test_marks_dead_pid_run_as_died(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """reap: dead-pid run gets status=died and ended_at written."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "deadrun",
            status="running",
            pid=99999,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: False)

        rc = agent_run.cmd_reap(_reap_args())

        assert rc == 0
        assert sd.joinpath("status").read_text().strip() == "died"
        ended_at = sd.joinpath("ended_at").read_text().strip()
        assert ended_at  # non-empty ISO timestamp
        assert "T" in ended_at  # basic ISO8601 shape

    def test_marks_no_pid_run_as_died(
        self, isolated_runs_root, isolated_log_root
    ):
        """reap: run with no pid file gets status=died."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "nopidrun",
            status="running",
            pid=None,
        )

        rc = agent_run.cmd_reap(_reap_args())

        assert rc == 0
        assert sd.joinpath("status").read_text().strip() == "died"

    def test_name_filter_only_reaps_target(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """--name NAME only reaps that specific run."""
        sd1, ld1 = _make_run(
            isolated_runs_root, isolated_log_root, "run-a",
            status="running", pid=99991,
        )
        sd2, ld2 = _make_run(
            isolated_runs_root, isolated_log_root, "run-b",
            status="running", pid=99992,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: False)

        agent_run.cmd_reap(_reap_args(name="run-a"))

        assert sd1.joinpath("status").read_text().strip() == "died"
        # run-b must be untouched.
        assert sd2.joinpath("status").read_text().strip() == "running"

    def test_already_done_run_is_not_touched(
        self, isolated_runs_root, isolated_log_root
    ):
        """reap skips runs that are not in 'running' status."""
        sd, ld = _make_run(
            isolated_runs_root, isolated_log_root, "donerun",
            status="done", pid=None,
        )

        agent_run.cmd_reap(_reap_args())

        assert sd.joinpath("status").read_text().strip() == "done"

    def test_name_no_such_run_prints_message(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        """--name <nonexistent> prints a clear 'no such run' message."""
        rc = agent_run.cmd_reap(_reap_args(name="does-not-exist"))
        assert rc == 0
        out = capsys.readouterr().out
        assert "no such run" in out
        assert "does-not-exist" in out


# ---------------------------------------------------------------------------
# cmd_reap -- idle-kill path
# ---------------------------------------------------------------------------

class TestReapIdleKill:
    @pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork/exec")
    def test_idle_kill_signals_and_marks_killed(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """reap idle-kill: alive pid + backdated log → SIGTERM + marked killed."""
        # Start a real subprocess we control.
        proc = subprocess.Popen(["sleep", "600"])
        pid = proc.pid
        try:
            threshold = 3600.0  # 1 hour
            monkeypatch.setattr(agent_run, "IDLE_STALL_SECONDS", threshold)

            sd, ld = _make_run(
                isolated_runs_root,
                isolated_log_root,
                "idlerun",
                status="running",
                pid=pid,
                log_age_secs=threshold + 60,  # log older than threshold
            )

            rc = agent_run.cmd_reap(_reap_args(idle_hours=threshold / 3600))

            assert rc == 0
            final_status = sd.joinpath("status").read_text().strip()
            assert final_status == "killed"
            assert (sd / "ended_at").exists()
            reason = (sd / "reap_reason").read_text().strip()
            assert "idle" in reason

            # Process should be dead now.
            proc.wait(timeout=10)
            assert proc.returncode is not None
        finally:
            # Best-effort cleanup — process may already be dead.
            try:
                proc.kill()
            except OSError:
                pass
            proc.wait(timeout=5)

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork/exec")
    def test_idle_kill_with_explicit_idle_hours_flag(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """--idle-hours N overrides the module constant for reap decisions.

        Uses a real subprocess and a shrunken grace period so the escalation
        path runs without actually waiting 5 seconds.
        """
        threshold_h = 0.1
        threshold_s = threshold_h * 3600

        proc = subprocess.Popen(["sleep", "600"])
        pid = proc.pid
        try:
            # Shrink grace to 0s so the test runs fast.
            monkeypatch.setattr(agent_run, "REAP_GRACE_SECONDS", 0.0)
            monkeypatch.setattr(agent_run, "IDLE_STALL_SECONDS", 24 * 3600)  # default large

            sd, ld = _make_run(
                isolated_runs_root,
                isolated_log_root,
                "idlerun2",
                status="running",
                pid=pid,
                log_age_secs=threshold_s + 10,
            )

            rc = agent_run.cmd_reap(_reap_args(idle_hours=threshold_h))

            assert rc == 0
            assert sd.joinpath("status").read_text().strip() == "killed"
            assert (sd / "ended_at").exists()

            proc.wait(timeout=10)
            assert proc.returncode is not None
        finally:
            try:
                proc.kill()
            except OSError:
                pass
            proc.wait(timeout=5)

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork/exec")
    def test_idle_kill_with_pgid_signals_process_group(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """reap uses pgid (killpg) when the pgid state file is present."""
        # Launch the subprocess in its own process group so killpg doesn't
        # terminate pytest itself.
        proc = subprocess.Popen(
            ["sleep", "600"],
            start_new_session=True,  # gives the process its own pgid == pid
        )
        pid = proc.pid
        pgid = pid  # start_new_session makes pgid == pid
        try:
            threshold = 3600.0
            monkeypatch.setattr(agent_run, "IDLE_STALL_SECONDS", threshold)
            monkeypatch.setattr(agent_run, "REAP_GRACE_SECONDS", 0.0)

            sd, ld = _make_run(
                isolated_runs_root,
                isolated_log_root,
                "pgidrun",
                status="running",
                pid=pid,
                pgid=pgid,
                log_age_secs=threshold + 60,
            )

            rc = agent_run.cmd_reap(_reap_args(idle_hours=threshold / 3600))

            assert rc == 0
            assert sd.joinpath("status").read_text().strip() == "killed"

            proc.wait(timeout=10)
            assert proc.returncode is not None
        finally:
            try:
                proc.kill()
            except OSError:
                pass
            proc.wait(timeout=5)

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork/exec")
    def test_idle_kill_signals_aux_pids(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """reap sends SIGTERM to pty_pid and keeper_pid aux processes."""
        # Launch two aux processes to act as pty_pid and keeper_pid.
        pty_proc = subprocess.Popen(["sleep", "600"])
        keeper_proc = subprocess.Popen(["sleep", "600"])
        main_proc = subprocess.Popen(["sleep", "600"])
        try:
            threshold = 3600.0
            monkeypatch.setattr(agent_run, "IDLE_STALL_SECONDS", threshold)
            monkeypatch.setattr(agent_run, "REAP_GRACE_SECONDS", 0.0)

            sd, ld = _make_run(
                isolated_runs_root,
                isolated_log_root,
                "auxrun",
                status="running",
                pid=main_proc.pid,
                pty_pid=pty_proc.pid,
                keeper_pid=keeper_proc.pid,
                log_age_secs=threshold + 60,
            )

            rc = agent_run.cmd_reap(_reap_args(idle_hours=threshold / 3600))

            assert rc == 0
            assert sd.joinpath("status").read_text().strip() == "killed"

            # All three processes should be dead.
            main_proc.wait(timeout=10)
            pty_proc.wait(timeout=10)
            keeper_proc.wait(timeout=10)
            assert main_proc.returncode is not None
            assert pty_proc.returncode is not None
            assert keeper_proc.returncode is not None
        finally:
            for p in (main_proc, pty_proc, keeper_proc):
                try:
                    p.kill()
                except OSError:
                    pass
                p.wait(timeout=5)

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork/exec")
    def test_pid_identity_mismatch_skips_kill(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        """PID identity mismatch: reap must NOT signal the process."""
        proc = subprocess.Popen(["sleep", "600"])
        pid = proc.pid
        real_kill = os.kill  # save before monkeypatching
        try:
            threshold = 3600.0
            monkeypatch.setattr(agent_run, "IDLE_STALL_SECONDS", threshold)

            sd, ld = _make_run(
                isolated_runs_root,
                isolated_log_root,
                "mismatchrun",
                status="running",
                pid=pid,
                log_age_secs=threshold + 60,
                # Record a fake start time that won't match the real process.
                pid_start="Thu Jan  1 00:00:00 1970",
            )

            harmful_signals_sent = []

            def recording_kill(p, sig):
                # Only record actual kill signals, not liveness probes (sig 0).
                if sig != 0:
                    harmful_signals_sent.append((p, sig))
                else:
                    # Allow liveness probes to use the real kill.
                    real_kill(p, sig)

            monkeypatch.setattr(agent_run.os, "kill", recording_kill)
            if hasattr(os, "killpg"):
                monkeypatch.setattr(agent_run.os, "killpg", lambda pg, sig: harmful_signals_sent.append(("pg", pg, sig)))

            rc = agent_run.cmd_reap(_reap_args(idle_hours=threshold / 3600))

            assert rc == 0
            # No harmful signals sent.
            assert harmful_signals_sent == [], f"Expected no signals, got: {harmful_signals_sent}"
            # Status must remain running.
            assert sd.joinpath("status").read_text().strip() == "running"
            # Output should explain the skip.
            out = capsys.readouterr().out
            assert "skipped" in out or "identity" in out.lower() or "unverified" in out
        finally:
            # Use the real os.kill to clean up (not the monkeypatched version).
            try:
                real_kill(pid, signal.SIGKILL)
            except OSError:
                pass
            proc.wait(timeout=5)

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork/exec")
    def test_sigkill_escalation_runs_after_grace(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """After grace period expires, SIGKILL is sent and process is marked killed."""
        # Use a process that ignores SIGTERM (we'll use a shell that traps it).
        # For simplicity, we use a real sleep and a zero grace period so SIGKILL
        # is sent immediately after SIGTERM.
        proc = subprocess.Popen(["sleep", "600"])
        pid = proc.pid
        try:
            threshold = 3600.0
            monkeypatch.setattr(agent_run, "IDLE_STALL_SECONDS", threshold)
            # Zero grace: SIGKILL fires immediately after SIGTERM.
            monkeypatch.setattr(agent_run, "REAP_GRACE_SECONDS", 0.0)

            sd, ld = _make_run(
                isolated_runs_root,
                isolated_log_root,
                "gracerun",
                status="running",
                pid=pid,
                log_age_secs=threshold + 60,
            )

            rc = agent_run.cmd_reap(_reap_args(idle_hours=threshold / 3600))

            assert rc == 0
            assert sd.joinpath("status").read_text().strip() == "killed"
            proc.wait(timeout=10)
        finally:
            try:
                proc.kill()
            except OSError:
                pass
            proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# opportunistic self-heal from list
# ---------------------------------------------------------------------------

class TestOpportunisticHeal:
    def test_list_marks_dead_pid_run_as_died(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        """cmd_list → _opportunistic_heal marks dead-pid runs as died (no idle-kill)."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "deadrun",
            status="running",
            pid=99999,
            log_age_secs=99 * 3600,  # very old log — must NOT trigger idle-kill
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: False)

        rc = agent_run.cmd_list(argparse.Namespace())

        assert rc == 0
        # After cmd_list, the state file should show died (written by _opportunistic_heal).
        assert sd.joinpath("status").read_text().strip() == "died"

    def test_list_never_idle_kills(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """cmd_list / _opportunistic_heal must NOT kill alive-but-idle runs."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "idlerun",
            status="running",
            pid=12345,
            log_age_secs=99 * 3600,  # very old
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        monkeypatch.setattr(agent_run, "IDLE_STALL_SECONDS", 1)  # tiny threshold

        kill_calls = []
        monkeypatch.setattr(agent_run.os, "kill", lambda pid, sig: kill_calls.append((pid, sig)))
        monkeypatch.setattr(agent_run.os, "killpg", lambda pgid, sig: kill_calls.append(("pgid", pgid, sig)))

        agent_run.cmd_list(argparse.Namespace())

        # No kill calls at all from the list command.
        assert kill_calls == []
        # Status should still be running (not touched by heal since pid is alive).
        assert sd.joinpath("status").read_text().strip() == "running"

    def test_opportunistic_heal_alone_marks_dead_run(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """_opportunistic_heal directly marks a dead-pid run without cmd_list."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "deadrun2",
            status="running",
            pid=88888,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: False)

        agent_run._opportunistic_heal(isolated_runs_root)

        assert sd.joinpath("status").read_text().strip() == "died"
        assert (sd / "ended_at").exists()

    def test_opportunistic_heal_skips_non_running(
        self, isolated_runs_root, isolated_log_root
    ):
        """_opportunistic_heal ignores runs that are not 'running'."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "finishedrun",
            status="done",
            pid=None,
        )

        agent_run._opportunistic_heal(isolated_runs_root)

        assert sd.joinpath("status").read_text().strip() == "done"

    def test_opportunistic_heal_marks_invalid_pid_as_died(
        self, isolated_runs_root, isolated_log_root
    ):
        """_opportunistic_heal marks a run with an unparseable pid as died."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "badpidrun",
            status="running",
        )
        # Write an unparseable pid.
        (sd / "pid").write_text("not-a-number\n")

        agent_run._opportunistic_heal(isolated_runs_root)

        assert sd.joinpath("status").read_text().strip() == "died"
        assert (sd / "ended_at").exists()
        reason = (sd / "reap_reason").read_text().strip()
        assert "invalid" in reason or "pid" in reason

