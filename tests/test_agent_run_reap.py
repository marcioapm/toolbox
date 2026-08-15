"""Tests for agent-run status reconciliation and idle reaping.

Covers:
- _effective_status: running+alive+fresh → running; running+dead pid → died;
  running+alive+stale log → stalled; starting states; done/failed passthrough.
- AGENT_RUN_IDLE_KILL_HOURS env override changes the threshold.
- reap --dry-run mutates nothing but reports correctly.
- reap marks a dead-pid "running" run as died with ended_at set.
- idle-kill path: alive pid + backdated log mtime → identity-verified,
  routed through _force_kill → marked killed.
- opportunistic self-heal from list marks dead-pid runs died but never
  idle-kills; correctly treats "starting" as a transient, not dead, state.
- kill-path coverage: process-identity verification, grace→SIGKILL
  escalation via _force_kill, identity mismatch guard (no signal sent).
- --name <typo> prints a clear "no such run" message.
- terminal-state GC: reap removes state dir + scratch dir (TMPDIR) for
  old done/failed/died/killed runs, honours --min-age-hours/env override,
  never touches persistent log/log.clean/prompt, never collects a live or
  unverifiable run, and is fully inert under --dry-run.
- launch creates and exports a per-run scratch dir as TMPDIR, inherited by
  the launched command and its descendants.
- list defaults to hiding terminal runs; --all / --status override that.

Reconciled against the R1-R10+N1+N2 kill/identity hardening: reap never
does a raw, unverified killpg/os.kill sweep on an alive run. It reuses the
same `process_identity`/`_process_identity` primitives and the same
`_force_kill` escalation machinery as `agent-run kill <name> KILL`.
"""
from __future__ import annotations

import argparse
import os
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pytest

from toolbox import agent_run


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _read_status(state_dir: Path) -> str:
    try:
        return (state_dir / "status").read_text().strip()
    except FileNotFoundError:
        return ""


def _process_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


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
    process_identity: Optional[str] = None,
    ended_at_age_secs: Optional[float] = None,
    make_scratch: bool = False,
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
    if process_identity is not None:
        (sd / "process_identity").write_text(f"{process_identity}\n")
    if ended_at_age_secs is not None:
        ended = datetime.fromtimestamp(
            time.time() - ended_at_age_secs, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        (sd / "ended_at").write_text(ended + "\n")

    if write_log:
        log_file = ld / "log"
        log_file.write_text("some output\n")
        if log_age_secs is not None:
            # Backdate the mtime.
            old_mtime = time.time() - log_age_secs
            os.utime(log_file, (old_mtime, old_mtime))

    if make_scratch:
        scratch = ld / "tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        os.chmod(scratch, 0o700)
        (scratch / "leaked-workspace").mkdir(exist_ok=True)
        (sd / "tmp_dir").write_text(str(scratch) + "\n")

    return sd, ld


def _reap_args(
    *,
    dry_run: bool = False,
    idle_hours: Optional[float] = None,
    min_age_hours: Optional[float] = None,
    name: Optional[str] = None,
    force_unknown: bool = False,
    include_logs: bool = False,
    log_min_age_hours: Optional[float] = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        dry_run=dry_run,
        idle_hours=idle_hours,
        min_age_hours=min_age_hours,
        name=name,
        force_unknown=force_unknown,
        include_logs=include_logs,
        log_min_age_hours=log_min_age_hours,
    )


def _shrink_escalation(monkeypatch, escalation=0.2, poll=0.05, reap_timeout=2.0):
    """Speed up _force_kill's escalation windows for tests."""
    monkeypatch.setattr(agent_run, "KILL_ESCALATION_TIMEOUT_SECONDS", escalation)
    monkeypatch.setattr(agent_run, "KILL_POLL_INTERVAL_SECONDS", poll)
    monkeypatch.setattr(agent_run, "KILL_CHILD_REAP_TIMEOUT_SECONDS", reap_timeout)


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

    def test_starting_no_pid_is_not_died(
        self, isolated_runs_root, isolated_log_root
    ):
        """starting + no pid yet (normal pre-publication window) → "starting",
        never "died". A launch racing _effective_status must not appear
        dead before the detached runner has even published its pid."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "startingrun",
            status="starting",
            pid=None,
            write_log=False,
        )
        assert agent_run._effective_status(sd) == "starting"

    def test_starting_alive_pid_is_starting(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """starting + a live pid already published → still "starting" (not
        yet promoted to "running" by the runner)."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "startingrun",
            status="starting",
            pid=12345,
            write_log=False,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        assert agent_run._effective_status(sd) == "starting"

    def test_starting_dead_pid_returns_died(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """starting + a published pid that is now dead → "died" (the launch
        crashed after publishing its pid but before promoting to running)."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "startingrun",
            status="starting",
            pid=99999,
            write_log=False,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: False)
        assert agent_run._effective_status(sd) == "died"


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

        kill_calls = []
        monkeypatch.setattr(agent_run.os, "kill", lambda pid, sig: kill_calls.append((pid, sig)))
        monkeypatch.setattr(agent_run, "_force_kill", lambda *a, **k: kill_calls.append(("force_kill", a)))

        rc = agent_run.cmd_reap(_reap_args(dry_run=True))

        assert rc == 0
        assert sd.joinpath("status").read_text().strip() == "running"
        assert not (sd / "ended_at").exists()
        # No kill calls, and _force_kill (our verified kill path) never invoked.
        assert kill_calls == []
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

    def test_starting_run_is_not_idle_killed_or_marked_died_by_reap(
        self, isolated_runs_root, isolated_log_root
    ):
        """reap only reconciles "running" state; a "starting" run (mid
        publication, no runner yet) must be left alone entirely — matches
        our launch model where "starting" is a transient pre-runner state,
        not something reap's dead-pid/idle-kill logic should touch."""
        sd, ld = _make_run(
            isolated_runs_root, isolated_log_root, "startingrun",
            status="starting", pid=None,
        )

        agent_run.cmd_reap(_reap_args())

        assert sd.joinpath("status").read_text().strip() == "starting"

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
#
# Reap's idle-kill routes through the same _force_kill machinery cmd_kill
# uses: identity-verified SIGTERM, bounded escalation window, then
# parentage-verified SIGKILL of the runner and its recorded children. No
# raw killpg/os.kill sweep of an unverified pid.
# ---------------------------------------------------------------------------

class TestReapIdleKill:
    @pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork/exec")
    def test_idle_kill_signals_and_marks_killed(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """reap idle-kill: alive pid + backdated log + verified identity →
        signalled via _force_kill + marked killed."""
        proc = subprocess.Popen(["sleep", "600"])
        pid = proc.pid
        try:
            threshold = 3600.0  # 1 hour
            monkeypatch.setattr(agent_run, "IDLE_STALL_SECONDS", threshold)
            _shrink_escalation(monkeypatch)

            identity = agent_run._process_identity(pid)
            assert identity is not None

            sd, ld = _make_run(
                isolated_runs_root,
                isolated_log_root,
                "idlerun",
                status="running",
                pid=pid,
                log_age_secs=threshold + 60,  # log older than threshold
                process_identity=identity,
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
        """--idle-hours N overrides the module constant for reap decisions."""
        threshold_h = 0.1
        threshold_s = threshold_h * 3600

        proc = subprocess.Popen(["sleep", "600"])
        pid = proc.pid
        try:
            _shrink_escalation(monkeypatch)
            monkeypatch.setattr(agent_run, "IDLE_STALL_SECONDS", 24 * 3600)  # default large

            identity = agent_run._process_identity(pid)
            assert identity is not None

            sd, ld = _make_run(
                isolated_runs_root,
                isolated_log_root,
                "idlerun2",
                status="running",
                pid=pid,
                log_age_secs=threshold_s + 10,
                process_identity=identity,
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
    def test_idle_kill_ignores_pgid_and_uses_verified_identity_kill(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """A stale/irrelevant pgid state file must not be relied on for
        signalling — reap's idle-kill goes through _force_kill, which
        targets the identity-verified pid directly rather than killpg'ing
        a recorded process group."""
        proc = subprocess.Popen(
            ["sleep", "600"],
            start_new_session=True,  # gives the process its own pgid == pid
        )
        pid = proc.pid
        pgid = pid  # start_new_session makes pgid == pid
        try:
            threshold = 3600.0
            monkeypatch.setattr(agent_run, "IDLE_STALL_SECONDS", threshold)
            _shrink_escalation(monkeypatch)

            identity = agent_run._process_identity(pid)
            assert identity is not None

            killpg_calls = []
            if hasattr(os, "killpg"):
                monkeypatch.setattr(
                    agent_run.os, "killpg",
                    lambda pg, sig: killpg_calls.append((pg, sig)),
                )

            sd, ld = _make_run(
                isolated_runs_root,
                isolated_log_root,
                "pgidrun",
                status="running",
                pid=pid,
                pgid=pgid,
                log_age_secs=threshold + 60,
                process_identity=identity,
            )

            rc = agent_run.cmd_reap(_reap_args(idle_hours=threshold / 3600))

            assert rc == 0
            assert sd.joinpath("status").read_text().strip() == "killed"
            # _force_kill never calls killpg — it signals verified pids directly.
            assert killpg_calls == []

            proc.wait(timeout=10)
            assert proc.returncode is not None
        finally:
            try:
                proc.kill()
            except OSError:
                pass
            proc.wait(timeout=5)

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork/exec")
    def test_idle_kill_reaps_parentage_verified_aux_pids_when_wedged(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """reap's idle-kill, routed through _force_kill, SIGKILLs
        parentage-verified aux pids (pty_pid/keeper_pid) once the runner
        fails to exit within the escalation window — not a raw signal to
        whatever pid happens to be recorded in state. Uses a shell that
        ignores SIGTERM to simulate a wedged runner, matching the same
        escalation scenario the R1-R10 _force_kill tests exercise."""
        script = (
            "trap '' TERM\n"
            "sleep 600 &\n"
            "p1=$!\n"
            "sleep 600 &\n"
            "p2=$!\n"
            "echo $p1\n"
            "echo $p2\n"
            "wait\n"
        )
        main_proc = subprocess.Popen(
            ["bash", "-c", script],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        pty_pid = keeper_pid = None
        try:
            pty_pid = int(main_proc.stdout.readline().strip())
            keeper_pid = int(main_proc.stdout.readline().strip())

            threshold = 3600.0
            monkeypatch.setattr(agent_run, "IDLE_STALL_SECONDS", threshold)
            _shrink_escalation(monkeypatch, escalation=0.3, poll=0.05, reap_timeout=2.0)

            identity = agent_run._process_identity(main_proc.pid)
            assert identity is not None

            sd, ld = _make_run(
                isolated_runs_root,
                isolated_log_root,
                "auxrun",
                status="running",
                pid=main_proc.pid,
                pty_pid=pty_pid,
                keeper_pid=keeper_pid,
                log_age_secs=threshold + 60,
                process_identity=identity,
            )

            rc = agent_run.cmd_reap(_reap_args(idle_hours=threshold / 3600))

            assert rc == 0
            assert sd.joinpath("status").read_text().strip() == "killed"

            # All three processes should be dead.
            assert _wait_until(lambda: _process_gone(main_proc.pid), timeout=5)
            assert _wait_until(lambda: _process_gone(pty_pid), timeout=5)
            assert _wait_until(lambda: _process_gone(keeper_pid), timeout=5)
        finally:
            for p in (main_proc.pid, pty_pid, keeper_pid):
                if p is None:
                    continue
                try:
                    os.kill(p, signal.SIGKILL)
                except OSError:
                    pass
            try:
                main_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork/exec")
    def test_pid_identity_mismatch_skips_kill(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        """PID identity mismatch: reap must NOT signal the process at all."""
        proc = subprocess.Popen(["sleep", "600"])
        pid = proc.pid
        real_kill = os.kill  # save before monkeypatching
        try:
            threshold = 3600.0
            monkeypatch.setattr(agent_run, "IDLE_STALL_SECONDS", threshold)
            _shrink_escalation(monkeypatch)

            sd, ld = _make_run(
                isolated_runs_root,
                isolated_log_root,
                "mismatchrun",
                status="running",
                pid=pid,
                log_age_secs=threshold + 60,
                # Record a fake identity that won't match the real process.
                process_identity="linux:0",
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
    def test_missing_identity_skips_kill(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        """Legacy state with no process_identity recorded at all: reap must
        refuse to signal rather than falling back to an unverified kill."""
        proc = subprocess.Popen(["sleep", "600"])
        pid = proc.pid
        real_kill = os.kill
        try:
            threshold = 3600.0
            monkeypatch.setattr(agent_run, "IDLE_STALL_SECONDS", threshold)
            _shrink_escalation(monkeypatch)

            sd, ld = _make_run(
                isolated_runs_root,
                isolated_log_root,
                "legacyrun",
                status="running",
                pid=pid,
                log_age_secs=threshold + 60,
                # No process_identity written at all.
            )

            harmful_signals_sent = []

            def recording_kill(p, sig):
                if sig != 0:
                    harmful_signals_sent.append((p, sig))
                else:
                    real_kill(p, sig)

            monkeypatch.setattr(agent_run.os, "kill", recording_kill)

            rc = agent_run.cmd_reap(_reap_args(idle_hours=threshold / 3600))

            assert rc == 0
            assert harmful_signals_sent == []
            assert sd.joinpath("status").read_text().strip() == "running"
            out = capsys.readouterr().out
            assert "skipped" in out or "identity" in out.lower()
        finally:
            try:
                real_kill(pid, signal.SIGKILL)
            except OSError:
                pass
            proc.wait(timeout=5)

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork/exec")
    def test_sigkill_escalation_runs_after_grace(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """A runner that ignores SIGTERM must still end up killed once the
        escalation window expires — exercises _force_kill's grace→SIGKILL
        escalation from within reap's idle-kill path."""
        proc = subprocess.Popen(
            ["bash", "-c", "trap '' TERM; sleep 600"],
            start_new_session=True,
        )
        pid = proc.pid
        try:
            threshold = 3600.0
            monkeypatch.setattr(agent_run, "IDLE_STALL_SECONDS", threshold)
            # Tight escalation window: SIGKILL fires shortly after SIGTERM
            # is ignored, so the test runs fast without weakening the
            # guarantee that an unresponsive run still eventually dies.
            _shrink_escalation(monkeypatch, escalation=0.3, poll=0.05, reap_timeout=2.0)

            identity = agent_run._process_identity(pid)
            assert identity is not None

            sd, ld = _make_run(
                isolated_runs_root,
                isolated_log_root,
                "gracerun",
                status="running",
                pid=pid,
                log_age_secs=threshold + 60,
                process_identity=identity,
            )

            rc = agent_run.cmd_reap(_reap_args(idle_hours=threshold / 3600))

            assert rc == 0
            assert sd.joinpath("status").read_text().strip() == "killed"
            assert _wait_until(lambda: _process_gone(pid), timeout=5)
        finally:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


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

    def test_opportunistic_heal_leaves_starting_with_no_pid_alone(
        self, isolated_runs_root, isolated_log_root
    ):
        """A "starting" run that hasn't published its pid yet is a normal
        transient pre-runner window, not evidence of death — heal must not
        touch it."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "startingrun",
            status="starting",
            pid=None,
        )

        agent_run._opportunistic_heal(isolated_runs_root)

        assert sd.joinpath("status").read_text().strip() == "starting"
        assert not (sd / "ended_at").exists()

    def test_opportunistic_heal_marks_starting_dead_pid_as_died(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """A "starting" run whose published pid is already dead (crashed
        right after publishing pid, before promotion to "running") must
        still be healed to "died"."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "startingdead",
            status="starting",
            pid=77777,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: False)

        agent_run._opportunistic_heal(isolated_runs_root)

        assert sd.joinpath("status").read_text().strip() == "died"
        assert (sd / "ended_at").exists()


# ---------------------------------------------------------------------------
# per-run scratch dir (TMPDIR)
# ---------------------------------------------------------------------------

class TestScratchDir:
    def test_launch_creates_scratch_dir_mode_0700(self, isolated_runs_root, isolated_log_root):
        name = "scratchrun"
        args = argparse.Namespace(
            name=name,
            command=[sys.executable, "-c", "pass"],
            interactive=False,
            prompt_file=None,
            echo=False,
            echo_interval=2.0,
            submit_mode=None,
        )
        rc = agent_run.cmd_launch(args)
        assert rc == 0

        scratch = isolated_log_root / name / "tmp"
        assert scratch.is_dir()
        mode = scratch.stat().st_mode & 0o777
        assert mode == 0o700, f"expected 0700, got {oct(mode)}"

    def test_launch_records_tmp_dir_in_state(self, isolated_runs_root, isolated_log_root):
        name = "scratchrun2"
        args = argparse.Namespace(
            name=name,
            command=[sys.executable, "-c", "pass"],
            interactive=False,
            prompt_file=None,
            echo=False,
            echo_interval=2.0,
            submit_mode=None,
        )
        agent_run.cmd_launch(args)

        recorded = (isolated_runs_root / name / "tmp_dir").read_text().strip()
        assert recorded == str(isolated_log_root / name / "tmp")

    def test_tmpdir_exported_and_inherited_by_child(self, isolated_runs_root, isolated_log_root):
        """The launched command (and therefore its descendants) must see
        TMPDIR pointing at this run's own scratch dir, not the ambient
        system TMPDIR."""
        name = "scratchrun3"
        args = argparse.Namespace(
            name=name,
            command=[sys.executable, "-c", "import os, sys; sys.stdout.write(os.environ.get('TMPDIR', '<unset>'))"],
            interactive=False,
            prompt_file=None,
            echo=False,
            echo_interval=2.0,
            submit_mode=None,
        )
        rc = agent_run.cmd_launch(args)
        assert rc == 0

        state_dir = isolated_runs_root / name
        assert _wait_until(lambda: _read_status(state_dir) in {"done", "failed"}, timeout=10)

        log_path = isolated_log_root / name / "log"
        assert _wait_until(lambda: log_path.exists() and log_path.read_text() != "", timeout=10)
        content = log_path.read_text()
        expected = str(isolated_log_root / name / "tmp")
        assert content.strip() == expected, f"child saw TMPDIR={content!r}, expected {expected!r}"

    def test_scratch_dir_not_deleted_on_normal_exit(self, isolated_runs_root, isolated_log_root):
        """Scratch is postmortem material — it must survive the run ending,
        only reap removes it."""
        name = "scratchrun4"
        args = argparse.Namespace(
            name=name,
            command=[
                sys.executable,
                "-c",
                "import os; open(os.path.join(os.environ['TMPDIR'], 'leftover'), 'w').close()",
            ],
            interactive=False,
            prompt_file=None,
            echo=False,
            echo_interval=2.0,
            submit_mode=None,
        )
        rc = agent_run.cmd_launch(args)
        assert rc == 0

        state_dir = isolated_runs_root / name
        scratch = isolated_log_root / name / "tmp"
        assert _wait_until(lambda: _read_status(state_dir) in {"done", "failed"}, timeout=10)
        assert (scratch / "leftover").exists(), "child could not write into its own TMPDIR"
        # Still there after the run has finished — nothing deletes it on exit.
        assert scratch.is_dir()
        assert (scratch / "leftover").exists()

    def test_argv_not_touched_by_scratch_dir(self, isolated_runs_root, isolated_log_root):
        """Only the environment carries TMPDIR — argv/command must be
        recorded and exec'd unmodified."""
        name = "scratchrun5"
        argv = [sys.executable, "-c", "pass"]
        args = argparse.Namespace(
            name=name,
            command=list(argv),
            interactive=False,
            prompt_file=None,
            echo=False,
            echo_interval=2.0,
            submit_mode=None,
        )
        agent_run.cmd_launch(args)
        import json as _json
        recorded_argv = _json.loads((isolated_runs_root / name / "argv").read_text())
        assert recorded_argv == argv


# ---------------------------------------------------------------------------
# reap -- terminal-state garbage collection
# ---------------------------------------------------------------------------

class TestReapTerminalStateGC:
    @pytest.mark.parametrize("status", ["done", "failed", "died", "killed"])
    def test_old_terminal_run_is_collected(
        self, isolated_runs_root, isolated_log_root, status
    ):
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            f"old-{status}",
            status=status,
            pid=None,
            ended_at_age_secs=200 * 3600,  # well past default 168h
            make_scratch=True,
        )
        scratch = ld / "tmp"
        assert scratch.is_dir()

        rc = agent_run.cmd_reap(_reap_args(min_age_hours=168))

        assert rc == 0
        assert not sd.exists(), "state dir must be removed"
        assert not scratch.exists(), "scratch dir must be removed"
        # Persistent log must survive.
        assert (ld / "log").exists()

    def test_young_terminal_run_is_not_collected(
        self, isolated_runs_root, isolated_log_root
    ):
        """A done run younger than the min-age threshold must be left alone
        entirely — this is what closed issue #2 in the task (reap only
        reconciling "running" and never GC'ing anything) but conservatively:
        a *fresh* terminal run's state/scratch must not vanish."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "freshdone",
            status="done",
            pid=None,
            ended_at_age_secs=1 * 3600,  # 1 hour old
            make_scratch=True,
        )
        scratch = ld / "tmp"

        rc = agent_run.cmd_reap(_reap_args(min_age_hours=168))

        assert rc == 0
        assert sd.exists()
        assert scratch.exists()

    def test_dry_run_does_not_delete_anything(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "olddone",
            status="done",
            pid=None,
            ended_at_age_secs=200 * 3600,
            make_scratch=True,
        )
        scratch = ld / "tmp"

        rc = agent_run.cmd_reap(_reap_args(dry_run=True, min_age_hours=168))

        assert rc == 0
        assert sd.exists()
        assert scratch.exists()
        out = capsys.readouterr().out
        assert "olddone" in out
        assert "terminal" in out
        assert "collected=1" in out

    def test_min_age_hours_flag_overrides_default(
        self, isolated_runs_root, isolated_log_root
    ):
        """--min-age-hours lets a shorter threshold collect a run that the
        168h default would still preserve."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "shortthreshold",
            status="failed",
            pid=None,
            ended_at_age_secs=2 * 3600,  # 2 hours old
            make_scratch=True,
        )
        scratch = ld / "tmp"

        # Default threshold: not old enough.
        agent_run.cmd_reap(_reap_args())
        assert sd.exists()
        assert scratch.exists()

        # Explicit 1-hour threshold: now old enough.
        rc = agent_run.cmd_reap(_reap_args(min_age_hours=1))
        assert rc == 0
        assert not sd.exists()
        assert not scratch.exists()

    def test_env_override_changes_min_age_threshold(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """AGENT_RUN_REAP_MIN_AGE_HOURS is re-read at call time by cmd_reap
        (same pattern as AGENT_RUN_IDLE_KILL_HOURS), so setting the env var
        — not the module constant — is what --min-age-hours falls back to."""
        monkeypatch.setenv("AGENT_RUN_REAP_MIN_AGE_HOURS", "1")
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "envthreshold",
            status="done",
            pid=None,
            ended_at_age_secs=2 * 3600,
            make_scratch=True,
        )

        rc = agent_run.cmd_reap(_reap_args())

        assert rc == 0
        assert not sd.exists()

    def test_live_pid_is_never_collected_even_if_status_says_terminal(
        self, isolated_runs_root, isolated_log_root
    ):
        """A "terminal" status with a live pid recorded is a contradiction
        (should never happen in practice — the runner only writes done/
        failed after it is genuinely finished) but reap's GC pass must
        refuse to act on it rather than trust the stale status label."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "contradictrun",
            status="done",
            pid=os.getpid(),  # definitely alive (this test process)
            ended_at_age_secs=200 * 3600,
            make_scratch=True,
        )
        scratch = ld / "tmp"

        rc = agent_run.cmd_reap(_reap_args(min_age_hours=168))

        assert rc == 0
        assert sd.exists(), "must not remove state for a live pid"
        assert scratch.exists()

    def test_stalled_run_is_never_collected(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """"stalled" (a live pid, idle log) is not in TERMINAL_STATUSES and
        must never be GC'd, no matter how old — only reconciliation (which
        requires raw status=="running") can eventually move it to killed."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "stalledrun",
            status="stalled",
            pid=None,
            ended_at_age_secs=500 * 3600,
            make_scratch=True,
        )
        scratch = ld / "tmp"

        rc = agent_run.cmd_reap(_reap_args(min_age_hours=1))

        assert rc == 0
        assert sd.exists()
        assert scratch.exists()

    def test_running_run_is_never_collected(
        self, isolated_runs_root, isolated_log_root
    ):
        """A "running" status is never in TERMINAL_STATUSES regardless of
        age — GC must leave it alone entirely. Uses a genuinely live pid so
        the reconciliation pass earlier in the same cmd_reap call doesn't
        first turn it into "died" (which is itself covered by the
        dead-pid-then-too-young-to-GC case elsewhere)."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "runningrun",
            status="running",
            pid=os.getpid(),
            ended_at_age_secs=500 * 3600,
            make_scratch=True,
        )
        scratch = ld / "tmp"

        rc = agent_run.cmd_reap(_reap_args(min_age_hours=1))

        assert rc == 0
        assert sd.exists()
        assert scratch.exists()

    def test_unknown_legacy_status_is_skipped(
        self, isolated_runs_root, isolated_log_root
    ):
        """A status outside the recognized set (legacy/corrupt/unknown)
        must be left alone by GC — never treated as an implicit terminal."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "weirdrun",
            status="frobnicated",
            pid=None,
            ended_at_age_secs=500 * 3600,
            make_scratch=True,
        )
        scratch = ld / "tmp"

        rc = agent_run.cmd_reap(_reap_args(min_age_hours=1))

        assert rc == 0
        assert sd.exists()
        assert scratch.exists()

    def test_logs_survive_reap_gc(self, isolated_runs_root, isolated_log_root):
        """log / log.clean / prompt must never be touched by the
        terminal-state GC pass — only the ephemeral state dir and the
        scratch dir are removed."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "logsurvive",
            status="done",
            pid=None,
            ended_at_age_secs=200 * 3600,
            make_scratch=True,
        )
        (ld / "log.clean").write_text("cleaned transcript\n")
        (ld / "prompt").write_text("original prompt\n")

        rc = agent_run.cmd_reap(_reap_args(min_age_hours=168))

        assert rc == 0
        assert not sd.exists()
        assert (ld / "log").exists()
        assert (ld / "log.clean").read_text() == "cleaned transcript\n"
        assert (ld / "prompt").read_text() == "original prompt\n"

    def test_gc_skips_when_no_scratch_dir_present(
        self, isolated_runs_root, isolated_log_root
    ):
        """A run launched before this feature existed (or one whose scratch
        was already removed) has no tmp/ dir at all — GC must still remove
        the state dir cleanly without erroring on the missing scratch."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "noscratch",
            status="done",
            pid=None,
            ended_at_age_secs=200 * 3600,
            make_scratch=False,
        )
        assert not (ld / "tmp").exists()

        rc = agent_run.cmd_reap(_reap_args(min_age_hours=168))

        assert rc == 0
        assert not sd.exists()
        assert (ld / "log").exists()

    def test_gc_respects_name_filter(self, isolated_runs_root, isolated_log_root):
        sd1, ld1 = _make_run(
            isolated_runs_root, isolated_log_root, "gc-a",
            status="done", pid=None, ended_at_age_secs=200 * 3600, make_scratch=True,
        )
        sd2, ld2 = _make_run(
            isolated_runs_root, isolated_log_root, "gc-b",
            status="done", pid=None, ended_at_age_secs=200 * 3600, make_scratch=True,
        )

        agent_run.cmd_reap(_reap_args(min_age_hours=168, name="gc-a"))

        assert not sd1.exists()
        assert sd2.exists(), "gc-b must be untouched by --name gc-a"

    def test_gc_inode_swap_race_rejected(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """If the state dir is replaced (same name, new inode) between
        reap's scan and its deletion under the per-name lock, the
        replacement must survive untouched — the same inode-reverification
        guarantee _safe_rmtree and _prune_old_logs already provide."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "raceswap",
            status="done",
            pid=None,
            ended_at_age_secs=200 * 3600,
            make_scratch=True,
        )
        original_lock = agent_run._launch_lock

        @agent_run.contextmanager
        def swap_while_locked(name):
            with original_lock(name) as fd:
                if name == "raceswap":
                    import shutil as _shutil
                    _shutil.rmtree(sd)
                    sd.mkdir()
                    (sd / "status").write_text("running\n")
                    (sd / "marker").write_text("new run, do not touch\n")
                yield fd

        monkeypatch.setattr(agent_run, "_launch_lock", swap_while_locked)

        rc = agent_run.cmd_reap(_reap_args(min_age_hours=168))

        assert rc == 0
        # The replacement (now "running") must survive intact.
        assert sd.exists()
        assert (sd / "marker").read_text() == "new run, do not touch\n"
        assert (sd / "status").read_text().strip() == "running"

    def test_existing_reap_reconciliation_paths_unchanged(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """Sanity check that adding the GC pass didn't disturb the existing
        dead-pid reconciliation behavior in the same invocation."""
        sd, ld = _make_run(
            isolated_runs_root,
            isolated_log_root,
            "stilldeadpid",
            status="running",
            pid=99999,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: False)

        rc = agent_run.cmd_reap(_reap_args())

        assert rc == 0
        assert sd.joinpath("status").read_text().strip() == "died"
        assert (sd / "ended_at").exists()
        # Freshly died — must NOT be GC'd in the same pass (too young).
        assert sd.exists()


# ---------------------------------------------------------------------------
# list -- filtering
# ---------------------------------------------------------------------------

class TestListFiltering:
    def test_default_hides_terminal_statuses(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        sd1, ld1 = _make_run(
            isolated_runs_root, isolated_log_root, "livevrun",
            status="running", pid=os.getpid(),
        )
        sd2, ld2 = _make_run(
            isolated_runs_root, isolated_log_root, "donerun",
            status="done", pid=None,
        )

        rc = agent_run.cmd_list(argparse.Namespace(all=False, status=None))

        assert rc == 0
        out = capsys.readouterr().out
        assert "livevrun" in out
        assert "donerun" not in out

    def test_all_flag_shows_terminal_statuses(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        sd1, ld1 = _make_run(
            isolated_runs_root, isolated_log_root, "livevrun2",
            status="running", pid=os.getpid(),
        )
        sd2, ld2 = _make_run(
            isolated_runs_root, isolated_log_root, "donerun2",
            status="done", pid=None,
        )

        rc = agent_run.cmd_list(argparse.Namespace(all=True, status=None))

        assert rc == 0
        out = capsys.readouterr().out
        assert "livevrun2" in out
        assert "donerun2" in out

    def test_status_filter_selects_only_matching(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        sd1, ld1 = _make_run(
            isolated_runs_root, isolated_log_root, "died-run",
            status="died", pid=None,
        )
        sd2, ld2 = _make_run(
            isolated_runs_root, isolated_log_root, "killed-run",
            status="killed", pid=None,
        )
        sd3, ld3 = _make_run(
            isolated_runs_root, isolated_log_root, "done-run",
            status="done", pid=None,
        )

        rc = agent_run.cmd_list(argparse.Namespace(all=False, status="died,killed"))

        assert rc == 0
        out = capsys.readouterr().out
        assert "died-run" in out
        assert "killed-run" in out
        assert "done-run" not in out

    def test_status_and_all_mutually_exclusive(
        self, isolated_runs_root, isolated_log_root
    ):
        with pytest.raises(SystemExit):
            agent_run.cmd_list(argparse.Namespace(all=True, status="done"))

    def test_default_namespace_without_all_or_status_attrs_still_works(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        """`agent-run` with no args (bare cmd_list(argparse.Namespace()))
        must keep working — getattr-based defaults, not required attrs."""
        _make_run(
            isolated_runs_root, isolated_log_root, "bareinvoke",
            status="running", pid=os.getpid(),
        )
        rc = agent_run.cmd_list(argparse.Namespace())
        assert rc == 0
        out = capsys.readouterr().out
        assert "bareinvoke" in out


# ---------------------------------------------------------------------------
# list --include-logs
# ---------------------------------------------------------------------------

def _list_args(*, all=False, status=None, include_logs=False) -> argparse.Namespace:
    return argparse.Namespace(all=all, status=status, include_logs=include_logs)


class TestListIncludeLogs:
    def test_hidden_by_default(self, isolated_runs_root, isolated_log_root, capsys):
        (isolated_log_root / "preservedrun").mkdir()
        (isolated_log_root / "preservedrun" / "log").write_text("x\n")

        rc = agent_run.cmd_list(_list_args())

        assert rc == 0
        out = capsys.readouterr().out
        assert "preservedrun" not in out
        assert "Preserved logs" not in out

    def test_shown_with_include_logs(self, isolated_runs_root, isolated_log_root, capsys):
        (isolated_log_root / "preservedrun2").mkdir()
        (isolated_log_root / "preservedrun2" / "log").write_text("x\n")

        rc = agent_run.cmd_list(_list_args(include_logs=True))

        assert rc == 0
        out = capsys.readouterr().out
        assert "preservedrun2" in out
        assert "Preserved logs" in out

    def test_combinable_with_all(self, isolated_runs_root, isolated_log_root, capsys):
        _make_run(
            isolated_runs_root, isolated_log_root, "donerun3",
            status="done", pid=None,
        )
        (isolated_log_root / "preservedrun3").mkdir()
        (isolated_log_root / "preservedrun3" / "log").write_text("x\n")

        rc = agent_run.cmd_list(_list_args(all=True, include_logs=True))

        assert rc == 0
        out = capsys.readouterr().out
        assert "donerun3" in out
        assert "preservedrun3" in out

    def test_all_alone_still_hides_preserved_logs(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        (isolated_log_root / "preservedrun4").mkdir()
        (isolated_log_root / "preservedrun4" / "log").write_text("x\n")

        rc = agent_run.cmd_list(_list_args(all=True, include_logs=False))

        assert rc == 0
        out = capsys.readouterr().out
        assert "preservedrun4" not in out

    def test_combinable_with_status(self, isolated_runs_root, isolated_log_root, capsys):
        _make_run(
            isolated_runs_root, isolated_log_root, "killedrun4",
            status="killed", pid=None,
        )
        (isolated_log_root / "preservedrun5").mkdir()
        (isolated_log_root / "preservedrun5" / "log").write_text("x\n")

        rc = agent_run.cmd_list(_list_args(status="killed", include_logs=True))

        assert rc == 0
        out = capsys.readouterr().out
        assert "killedrun4" in out
        assert "preservedrun5" in out

    def test_hint_on_stderr_only_when_hidden_logs_exist(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        (isolated_log_root / "hiddenpreserved").mkdir()
        (isolated_log_root / "hiddenpreserved" / "log").write_text("x\n")

        rc = agent_run.cmd_list(_list_args())

        assert rc == 0
        err = capsys.readouterr().err
        assert "preserved log(s) hidden" in err
        assert "--include-logs" in err

    def test_no_hint_when_no_preserved_logs(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        rc = agent_run.cmd_list(_list_args())

        assert rc == 0
        err = capsys.readouterr().err
        assert "preserved log(s) hidden" not in err

    def test_no_hint_printed_when_include_logs_shown(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        (isolated_log_root / "shownpreserved").mkdir()
        (isolated_log_root / "shownpreserved" / "log").write_text("x\n")

        rc = agent_run.cmd_list(_list_args(include_logs=True))

        assert rc == 0
        err = capsys.readouterr().err
        assert "preserved log(s) hidden" not in err

    def test_env_var_opts_in_by_default(
        self, isolated_runs_root, isolated_log_root, capsys, monkeypatch
    ):
        monkeypatch.setenv("AGENT_RUN_LIST_INCLUDE_LOGS", "1")
        (isolated_log_root / "envshown").mkdir()
        (isolated_log_root / "envshown" / "log").write_text("x\n")

        rc = agent_run.cmd_list(_list_args())

        assert rc == 0
        out = capsys.readouterr().out
        assert "envshown" in out

    @pytest.mark.parametrize("raw", ["true", "yes", "TRUE", "Yes"])
    def test_env_var_accepts_true_yes_case_insensitive(
        self, isolated_runs_root, isolated_log_root, capsys, monkeypatch, raw
    ):
        monkeypatch.setenv("AGENT_RUN_LIST_INCLUDE_LOGS", raw)
        (isolated_log_root / "envshown2").mkdir()
        (isolated_log_root / "envshown2" / "log").write_text("x\n")

        rc = agent_run.cmd_list(_list_args())

        assert rc == 0
        assert "envshown2" in capsys.readouterr().out

    def test_explicit_flag_wins_over_env_false(
        self, isolated_runs_root, isolated_log_root, capsys, monkeypatch
    ):
        monkeypatch.setenv("AGENT_RUN_LIST_INCLUDE_LOGS", "0")
        (isolated_log_root / "explicitwins").mkdir()
        (isolated_log_root / "explicitwins" / "log").write_text("x\n")

        rc = agent_run.cmd_list(_list_args(include_logs=True))

        assert rc == 0
        assert "explicitwins" in capsys.readouterr().out

    def test_invalid_env_value_warns_and_defaults_to_hidden(
        self, isolated_runs_root, isolated_log_root, capsys, monkeypatch
    ):
        monkeypatch.setenv("AGENT_RUN_LIST_INCLUDE_LOGS", "maybe")
        (isolated_log_root / "invalidenvlog").mkdir()
        (isolated_log_root / "invalidenvlog" / "log").write_text("x\n")

        rc = agent_run.cmd_list(_list_args())

        assert rc == 0
        out, err = capsys.readouterr()
        assert "invalidenvlog" not in out
        assert "AGENT_RUN_LIST_INCLUDE_LOGS" in err


# ---------------------------------------------------------------------------
# reviewer regression coverage
# ---------------------------------------------------------------------------

class TestReapReviewerRegressions:
    @pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "-1", "0"])
    def test_invalid_hour_env_values_fall_back_to_safe_defaults(self, monkeypatch, capsys, raw):
        monkeypatch.setenv("AGENT_RUN_IDLE_KILL_HOURS", raw)
        assert agent_run._parse_idle_stall_seconds() == 24 * 3600
        assert "not a finite, positive" in capsys.readouterr().err

        monkeypatch.setenv("AGENT_RUN_REAP_MIN_AGE_HOURS", raw)
        assert agent_run._parse_reap_min_age_seconds() == 168 * 3600
        assert "not a finite, positive" in capsys.readouterr().err

    @pytest.mark.parametrize("flag", ["--idle-hours", "--min-age-hours"])
    @pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "-1", "0"])
    def test_reap_cli_rejects_invalid_hour_flags(self, flag, raw, capsys):
        option = f"{flag}={raw}" if raw == "-inf" else flag
        argv = ["reap", option] if raw == "-inf" else ["reap", flag, raw]
        with pytest.raises(SystemExit) as exc:
            agent_run._build_parser().parse_args(argv)
        assert exc.value.code == 2
        assert "must be finite and greater than 0" in capsys.readouterr().err

    def test_symlinked_log_dir_never_deletes_outside_log_root(
        self, isolated_runs_root, isolated_log_root, tmp_path, capsys
    ):
        sd, _ = _make_run(
            isolated_runs_root, isolated_log_root, "linked-log", status="done",
            ended_at_age_secs=200 * 3600,
        )
        outside = tmp_path / "outside"
        outside.mkdir()
        protected = outside / "protected"
        protected.write_text("must survive")
        log_dir = isolated_log_root / "linked-log"
        (log_dir / "log").unlink()
        log_dir.rmdir()
        log_dir.symlink_to(outside, target_is_directory=True)

        agent_run.cmd_reap(_reap_args(min_age_hours=168))

        assert sd.exists()
        assert protected.read_text() == "must survive"
        assert "log path is not a real directory" in capsys.readouterr().out

    def test_safe_rmtree_refuses_root_replaced_by_symlink(
        self, isolated_log_root, tmp_path
    ):
        """A root swap immediately before deletion cannot redirect its child."""
        log_dir = isolated_log_root / "raced-log"
        scratch = log_dir / "tmp"
        scratch.mkdir(parents=True)
        (scratch / "safe-artifact").write_text("remove only this")
        outside = tmp_path / "outside"
        victim_scratch = outside / "tmp"
        victim_scratch.mkdir(parents=True)
        protected = victim_scratch / "protected"
        protected.write_text("must survive")
        moved = isolated_log_root / "raced-log-old"
        log_dir.rename(moved)
        log_dir.symlink_to(outside, target_is_directory=True)

        with pytest.raises(SystemExit):
            agent_run._safe_rmtree(log_dir / "tmp", log_dir)

        assert protected.read_text() == "must survive"
        assert (moved / "tmp" / "safe-artifact").exists()

    def test_bad_scratch_does_not_abort_later_gc(
        self, isolated_runs_root, isolated_log_root, tmp_path, capsys
    ):
        bad_state, bad_log = _make_run(
            isolated_runs_root, isolated_log_root, "bad-scratch", status="done",
            ended_at_age_secs=200 * 3600,
        )
        outside = tmp_path / "outside"
        outside.mkdir()
        protected = outside / "protected"
        protected.write_text("must survive")
        (bad_log / "tmp").symlink_to(outside, target_is_directory=True)
        good_state, good_log = _make_run(
            isolated_runs_root, isolated_log_root, "good-scratch", status="done",
            ended_at_age_secs=200 * 3600, make_scratch=True,
        )

        agent_run.cmd_reap(_reap_args(min_age_hours=168))

        assert bad_state.exists()
        assert protected.read_text() == "must survive"
        assert not good_state.exists()
        assert not (good_log / "tmp").exists()
        assert "scratch path is not a real directory" in capsys.readouterr().out

    def test_reconciled_old_ended_at_is_not_collected_in_same_pass(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        sd, ld = _make_run(
            isolated_runs_root, isolated_log_root, "old-reconciled", status="running",
            pid=99999, ended_at_age_secs=200 * 3600, make_scratch=True,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: False)

        agent_run.cmd_reap(_reap_args(min_age_hours=168))

        assert sd.exists()
        assert _read_status(sd) == "died"
        assert (ld / "tmp").exists()

    def test_orphan_scratch_is_collected_without_state(
        self, isolated_runs_root, isolated_log_root
    ):
        scratch = isolated_log_root / "orphan" / "tmp"
        nested = scratch / "nested"
        nested.mkdir(parents=True)
        artifact = nested / "artifact"
        artifact.write_text("old")
        old = time.time() - 200 * 3600
        os.utime(artifact, (old, old))
        os.utime(nested, (old, old))
        os.utime(scratch, (old, old))

        agent_run.cmd_reap(_reap_args(min_age_hours=168))

        assert not scratch.exists()
        assert (isolated_log_root / "orphan").is_dir()

    def test_dry_run_does_not_predict_gc_for_live_recorded_runner(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        identity = agent_run._process_identity(os.getpid())
        assert identity is not None
        sd, ld = _make_run(
            isolated_runs_root, isolated_log_root, "live-dry-run", status="done",
            pid=os.getpid(), process_identity=identity,
            ended_at_age_secs=200 * 3600, make_scratch=True,
        )

        agent_run.cmd_reap(_reap_args(dry_run=True, min_age_hours=168))

        assert sd.exists()
        assert (ld / "tmp").exists()
        out = capsys.readouterr().out
        assert "recorded live runner" in out
        assert "collected=0" in out

    def test_recycled_pid_identity_mismatch_allows_gc(
        self, isolated_runs_root, isolated_log_root
    ):
        sd, ld = _make_run(
            isolated_runs_root, isolated_log_root, "recycled", status="done",
            pid=os.getpid(), process_identity="different-birth-token",
            ended_at_age_secs=200 * 3600, make_scratch=True,
        )

        agent_run.cmd_reap(_reap_args(min_age_hours=168))

        assert not sd.exists()
        assert not (ld / "tmp").exists()

    @pytest.mark.parametrize("ended_at, warning", [
        ("not-a-timestamp", "is malformed"),
        ("2999-01-01T00:00:00Z", "is in the future"),
    ])
    def test_malformed_or_future_ended_at_warns_and_preserves_state(
        self, isolated_runs_root, isolated_log_root, capsys, ended_at, warning
    ):
        sd, ld = _make_run(
            isolated_runs_root, isolated_log_root, "bad-ended-at", status="done",
            make_scratch=True,
        )
        (sd / "ended_at").write_text(ended_at + "\n")

        agent_run.cmd_reap(_reap_args(min_age_hours=168))

        assert sd.exists()
        assert (ld / "tmp").exists()
        assert warning in capsys.readouterr().err

    def test_list_buckets_unknown_and_force_unknown_gc_requires_opt_in(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        sd, ld = _make_run(
            isolated_runs_root, isolated_log_root, "legacy", status="frobnicated",
            ended_at_age_secs=200 * 3600, make_scratch=True,
        )

        agent_run.cmd_list(argparse.Namespace(all=False, status=None))
        assert "Unrecognized / needs attention" in capsys.readouterr().out
        assert sd.exists()

        agent_run.cmd_reap(_reap_args(min_age_hours=168))
        assert sd.exists()

        agent_run.cmd_reap(_reap_args(min_age_hours=168, force_unknown=True))
        assert not sd.exists()
        assert not (ld / "tmp").exists()

    def test_reap_cleans_crash_deletion_sentinel(self, isolated_runs_root, isolated_log_root):
        sentinel = isolated_runs_root / ".reaping-interrupted.1"
        sentinel.mkdir()
        (sentinel / "partial").write_text("left by interrupted deletion")

        agent_run.cmd_reap(_reap_args())

        assert not sentinel.exists()


# ---------------------------------------------------------------------------
# _gc_live_runner_pid: strict pid read for the GC path (absent vs unreadable)
# ---------------------------------------------------------------------------

class TestGcLiveRunnerPidStrictRead:
    """The pid file is read strictly on the GC path, not through the lenient
    shared `_read`. GC's fail-safe direction is a spurious skip, never a
    spurious delete, so an unreadable pid file must report as still-live
    (refusing GC) rather than as no-pid-recorded (permitting it)."""

    ROOT_SKIP = pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="root ignores permission bits",
    )

    # Ways the pid file can be present but unreadable. `pid=99999999` would
    # otherwise resolve to a dead pid, so a lenient read would permit GC.
    UNREADABLE_PID = [
        pytest.param(
            lambda p: p.chmod(0o000), 99999999, id="mode_000", marks=ROOT_SKIP,
        ),
        pytest.param(lambda p: p.mkdir(), None, id="directory_in_place"),
    ]

    def test_readable_pid_file_with_a_live_pid_returns_that_pid(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        sd, ld = _make_run(
            isolated_runs_root, isolated_log_root, "live", status="done",
            pid=os.getpid(),
        )
        monkeypatch.setattr(agent_run, "_process_identity", lambda _p: None)
        assert agent_run._gc_live_runner_pid(sd) == os.getpid()

    def test_absent_pid_file_returns_none(
        self, isolated_runs_root, isolated_log_root
    ):
        sd, ld = _make_run(
            isolated_runs_root, isolated_log_root, "absent", status="done",
        )
        assert not (sd / "pid").exists()
        assert agent_run._gc_live_runner_pid(sd) is None

    @pytest.mark.parametrize("damage, pid", UNREADABLE_PID)
    def test_unreadable_pid_file_reports_live(
        self, isolated_runs_root, isolated_log_root, damage, pid
    ):
        sd, ld = _make_run(
            isolated_runs_root, isolated_log_root, "unreadable", status="done",
            pid=pid,
        )
        damage(sd / "pid")
        try:
            assert agent_run._gc_live_runner_pid(sd) is not None
        finally:
            if (sd / "pid").is_file():
                (sd / "pid").chmod(0o644)

    @pytest.mark.parametrize("damage, pid", UNREADABLE_PID)
    def test_reap_refuses_to_collect_when_pid_file_is_unreadable(
        self, isolated_runs_root, isolated_log_root, damage, pid
    ):
        sd, ld = _make_run(
            isolated_runs_root, isolated_log_root, "unreadablereap", status="done",
            pid=pid, ended_at_age_secs=200 * 3600, make_scratch=True,
        )
        damage(sd / "pid")
        try:
            rc = agent_run.cmd_reap(_reap_args(min_age_hours=168))
        finally:
            if (sd / "pid").is_file():
                (sd / "pid").chmod(0o644)
        assert rc == 0
        assert sd.exists(), "an unreadable pid file must block GC, not permit it"
        assert (ld / "tmp").exists()

    def test_reap_still_collects_when_pid_is_genuinely_absent(
        self, isolated_runs_root, isolated_log_root
    ):
        sd, ld = _make_run(
            isolated_runs_root, isolated_log_root, "absentreap", status="done",
            ended_at_age_secs=200 * 3600, make_scratch=True,
        )
        assert not (sd / "pid").exists()
        rc = agent_run.cmd_reap(_reap_args(min_age_hours=168))
        assert rc == 0
        assert not sd.exists()
        assert not (ld / "tmp").exists()


# ---------------------------------------------------------------------------
# reap --include-logs: preserved-log-only GC
# ---------------------------------------------------------------------------

def _make_preserved_log(
    log_root: Path, name: str, *, age_secs: float, make_scratch: bool = False
) -> Path:
    """A log-only dir (no state dir) with every file backdated to age_secs."""
    ld = log_root / name
    ld.mkdir(parents=True, exist_ok=True)
    (ld / "log").write_text("preserved output\n")
    if make_scratch:
        scratch = ld / "tmp"
        scratch.mkdir(exist_ok=True)
        (scratch / "leftover").write_text("x")
    old = time.time() - age_secs
    for p in ld.rglob("*"):
        os.utime(p, (old, old))
    os.utime(ld, (old, old))
    return ld


class TestReapIncludeLogs:
    def test_old_log_removed_with_include_logs(
        self, isolated_runs_root, isolated_log_root
    ):
        ld = _make_preserved_log(
            isolated_log_root, "oldlog", age_secs=(agent_run.PRUNE_AFTER_DAYS + 1) * 86400
        )

        rc = agent_run.cmd_reap(_reap_args(include_logs=True))

        assert rc == 0
        assert not ld.exists()

    def test_old_log_kept_without_include_logs(
        self, isolated_runs_root, isolated_log_root
    ):
        ld = _make_preserved_log(
            isolated_log_root, "oldlog2", age_secs=(agent_run.PRUNE_AFTER_DAYS + 1) * 86400
        )

        rc = agent_run.cmd_reap(_reap_args(include_logs=False))

        assert rc == 0
        assert ld.exists()

    def test_young_log_kept_with_include_logs(
        self, isolated_runs_root, isolated_log_root
    ):
        ld = _make_preserved_log(isolated_log_root, "younglog", age_secs=3600)

        rc = agent_run.cmd_reap(_reap_args(include_logs=True))

        assert rc == 0
        assert ld.exists()

    def test_log_with_live_state_dir_is_never_touched(
        self, isolated_runs_root, isolated_log_root
    ):
        sd, ld = _make_run(
            isolated_runs_root, isolated_log_root, "hasstate",
            status="running", pid=os.getpid(),
        )
        old = time.time() - (agent_run.PRUNE_AFTER_DAYS + 1) * 86400
        os.utime(ld / "log", (old, old))
        os.utime(ld, (old, old))

        rc = agent_run.cmd_reap(_reap_args(include_logs=True))

        assert rc == 0
        assert ld.exists()
        assert sd.exists()

    def test_dry_run_mutates_nothing(self, isolated_runs_root, isolated_log_root, capsys):
        ld = _make_preserved_log(
            isolated_log_root, "dryrunlog", age_secs=(agent_run.PRUNE_AFTER_DAYS + 1) * 86400
        )

        rc = agent_run.cmd_reap(_reap_args(include_logs=True, dry_run=True))

        assert rc == 0
        assert ld.exists()
        out = capsys.readouterr().out
        assert "dryrunlog" in out
        assert "logs_collected=1" in out

    def test_dry_run_no_double_count_with_scratch(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        """--dry-run must not count a state-less run with tmp/ as both a
        collected log (pass 2.5) and an orphaned scratch (pass 3).

        In a real run pass 2.5 removes the whole log dir so pass 3 finds
        no scratch.  In --dry-run nothing is removed; without the fix pass 3
        re-reports the same directory under a second heading."""
        old_secs = (agent_run.PRUNE_AFTER_DAYS + 1) * 86400
        ld = _make_preserved_log(
            isolated_log_root, "drydbldbl", age_secs=old_secs, make_scratch=True
        )
        # Backdate the scratch dir so it is old enough for pass 3.
        scratch = ld / "tmp"
        old_ts = time.time() - old_secs
        for p in scratch.rglob("*"):
            try:
                os.utime(p, (old_ts, old_ts))
            except OSError:
                pass
        os.utime(scratch, (old_ts, old_ts))

        rc = agent_run.cmd_reap(_reap_args(include_logs=True, dry_run=True,
                                            min_age_hours=0.001))

        assert rc == 0
        out = capsys.readouterr().out
        # The run must appear once (as a log collection) and never as orphaned scratch.
        assert "logs_collected=1" in out, f"expected logs_collected=1; output: {out}"
        assert "orphaned_scratch=0" in out, (
            f"dry-run must not count same run as orphaned_scratch; output: {out}"
        )

    def test_name_targeting_only_reaps_named_log(
        self, isolated_runs_root, isolated_log_root
    ):
        old_secs = (agent_run.PRUNE_AFTER_DAYS + 1) * 86400
        ld1 = _make_preserved_log(isolated_log_root, "logtarget-a", age_secs=old_secs)
        ld2 = _make_preserved_log(isolated_log_root, "logtarget-b", age_secs=old_secs)

        rc = agent_run.cmd_reap(_reap_args(include_logs=True, name="logtarget-a"))

        assert rc == 0
        assert not ld1.exists()
        assert ld2.exists()

    def test_counter_appears_in_summary(self, isolated_runs_root, isolated_log_root, capsys):
        _make_preserved_log(
            isolated_log_root, "countedlog", age_secs=(agent_run.PRUNE_AFTER_DAYS + 1) * 86400
        )

        rc = agent_run.cmd_reap(_reap_args(include_logs=True))

        assert rc == 0
        out = capsys.readouterr().out
        assert "logs_collected=1" in out

    def test_counter_present_but_zero_without_include_logs(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        _make_preserved_log(
            isolated_log_root, "nocountlog", age_secs=(agent_run.PRUNE_AFTER_DAYS + 1) * 86400
        )

        rc = agent_run.cmd_reap(_reap_args(include_logs=False))

        assert rc == 0
        out = capsys.readouterr().out
        assert "logs_collected=0" in out

    def test_symlinked_log_dir_is_refused(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        outside = isolated_log_root.parent / "outside-log-target"
        outside.mkdir()
        (outside / "secret").write_text("must not be touched")
        link = isolated_log_root / "symlog"
        link.symlink_to(outside, target_is_directory=True)

        # Backdate both the target and the symlink past the collection threshold
        # so the age gate cannot protect the target — only the S_ISDIR symlink
        # refusal can.  With a fresh mtime the age gate fires first, making the
        # test vacuous (the symlink guard is never reached).
        old = time.time() - (agent_run.PRUNE_AFTER_DAYS + 1) * 86400
        os.utime(outside / "secret", (old, old))
        os.utime(outside, (old, old))
        os.utime(link, (old, old), follow_symlinks=False)

        rc = agent_run.cmd_reap(_reap_args(include_logs=True))

        assert rc == 0
        assert outside.exists()
        assert (outside / "secret").exists()
        assert link.is_symlink()
        out = capsys.readouterr().out
        assert "logs_collected=0" in out

    def test_include_logs_works_when_state_root_absent(
        self, isolated_runs_root, isolated_log_root
    ):
        """The `reap: no state or log root, nothing to do.` guard must not
        short-circuit --include-logs when STATE_ROOT is gone but LOG_ROOT
        has content."""
        shutil.rmtree(isolated_runs_root)
        ld = _make_preserved_log(
            isolated_log_root, "noStateRoot", age_secs=(agent_run.PRUNE_AFTER_DAYS + 1) * 86400
        )

        rc = agent_run.cmd_reap(_reap_args(include_logs=True))

        assert rc == 0
        assert not ld.exists()

    def test_log_root_sentinel_reaped_by_next_invocation(
        self, isolated_runs_root, isolated_log_root
    ):
        """A .reaping-* sentinel left under LOG_ROOT by an interrupted reap
        must be cleaned up by the next reap invocation and must never appear
        in list output."""
        # Simulate a crash-safe-rmtree sentinel: the directory was renamed
        # to .reaping-<name>.<pid>.<ns> but not yet emptied.
        sentinel_name = ".reaping-oldrun.9999.123456789000000000"
        sentinel = isolated_log_root / sentinel_name
        sentinel.mkdir()
        (sentinel / "log").write_text("leftover\n")

        rc = agent_run.cmd_reap(_reap_args())

        assert rc == 0
        assert not sentinel.exists(), "stale LOG_ROOT sentinel was not cleaned up"

    def test_list_does_not_show_log_root_sentinel(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        """A .reaping-* sentinel under LOG_ROOT must be invisible to list,
        even with --include-logs."""
        sentinel = isolated_log_root / ".reaping-badrun.9999.123456789000000000"
        sentinel.mkdir()
        (sentinel / "log").write_text("leftover\n")

        agent_run.cmd_list(_list_args(include_logs=True))

        out = capsys.readouterr().out
        assert ".reaping-" not in out


class TestReapLogMinAgeThreshold:
    """--log-min-age-hours is independent of --min-age-hours, defaults to
    PRUNE_AFTER_DAYS*24 (21d), and is resolved flag > env > default, exactly
    like --min-age-hours/AGENT_RUN_MIN_AGE_HOURS."""

    def test_default_is_21_days_not_7(self, isolated_runs_root, isolated_log_root):
        ten_days = 10 * 86400
        ld = _make_preserved_log(isolated_log_root, "tendays", age_secs=ten_days)

        # 10 days old: kept by the 7-day (168h) state default were it (wrongly)
        # reused, but the log default is 21 days, so it must survive.
        rc = agent_run.cmd_reap(_reap_args(include_logs=True))

        assert rc == 0
        assert ld.exists()

    def test_flag_overrides_default_to_collect_10_day_old_log(
        self, isolated_runs_root, isolated_log_root
    ):
        ld = _make_preserved_log(isolated_log_root, "tendays2", age_secs=10 * 86400)

        rc = agent_run.cmd_reap(_reap_args(include_logs=True, log_min_age_hours=24))

        assert rc == 0
        assert not ld.exists()

    def test_env_var_honoured(self, isolated_runs_root, isolated_log_root, monkeypatch):
        monkeypatch.setenv("AGENT_RUN_LOG_MIN_AGE_HOURS", "24")
        ld = _make_preserved_log(isolated_log_root, "envlog", age_secs=10 * 86400)

        rc = agent_run.cmd_reap(_reap_args(include_logs=True))

        assert rc == 0
        assert not ld.exists()

    def test_flag_beats_env(self, isolated_runs_root, isolated_log_root, monkeypatch):
        monkeypatch.setenv("AGENT_RUN_LOG_MIN_AGE_HOURS", "24")
        ld = _make_preserved_log(isolated_log_root, "flagbeatsenv", age_secs=10 * 86400)

        # Env would collect it (24h < 10d); the explicit flag raises the bar
        # back above 10 days, so it must survive.
        rc = agent_run.cmd_reap(_reap_args(include_logs=True, log_min_age_hours=500))

        assert rc == 0
        assert ld.exists()

    @pytest.mark.parametrize("raw", ["not-a-number", "-5", "0", "nan", "inf"])
    def test_invalid_env_value_warns_and_falls_back_to_default(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys, raw
    ):
        monkeypatch.setenv("AGENT_RUN_LOG_MIN_AGE_HOURS", raw)
        ld = _make_preserved_log(isolated_log_root, "badenvlog", age_secs=10 * 86400)

        rc = agent_run.cmd_reap(_reap_args(include_logs=True))

        assert rc == 0
        assert ld.exists(), "invalid env value must fall back to the 21d default, not 0"
        assert "AGENT_RUN_LOG_MIN_AGE_HOURS" in capsys.readouterr().err

    def test_min_age_hours_alone_does_not_affect_log_gc(
        self, isolated_runs_root, isolated_log_root
    ):
        """--min-age-hours (state-dir threshold) must never influence
        preserved-log GC eligibility."""
        ld = _make_preserved_log(isolated_log_root, "stateThresholdOnly", age_secs=10 * 86400)

        # A tiny --min-age-hours would collect state dirs almost instantly,
        # but must have zero effect on this log-only run.
        rc = agent_run.cmd_reap(_reap_args(include_logs=True, min_age_hours=0.001))

        assert rc == 0
        assert ld.exists()


# ---------------------------------------------------------------------------
# S5: _dir_size_bytes tolerates OSError beyond the original caught set
# ---------------------------------------------------------------------------

class TestDirSizeBytesOSError:
    """_dir_size_bytes must not raise when os.scandir returns an OSError
    outside the original (FileNotFoundError, NotADirectoryError,
    PermissionError) set — e.g. ENAMETOOLONG (errno 63) from a deeply
    nested scratch tree.  It must return a partial count and allow
    reap --include-logs to complete normally."""

    def test_enametoolong_returns_partial_count_not_raises(
        self, tmp_path, monkeypatch
    ):
        """Simulate ENAMETOOLONG from os.scandir on a nested directory.

        The outer directory has one countable file and one subdirectory
        whose scan raises ENAMETOOLONG.  _dir_size_bytes must return the
        outer file's size (not raise) and continue past the bad entry."""
        import errno as _errno

        d = tmp_path / "log"
        d.mkdir()
        (d / "outer.txt").write_bytes(b"hello")  # 5 bytes, always countable
        deep = d / "deep"
        deep.mkdir()

        real_scandir = os.scandir

        def fake_scandir(path):
            path_str = str(path)
            if path_str == str(deep):
                raise OSError(_errno.ENAMETOOLONG, "File name too long", path_str)
            return real_scandir(path)

        monkeypatch.setattr(os, "scandir", fake_scandir)

        result = agent_run._dir_size_bytes(d)

        # Must return the outer file size without raising.
        assert result == 5

    def test_enametoolong_does_not_abort_reap_include_logs(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """An ENAMETOOLONG on os.scandir inside a log dir must not abort
        reap --include-logs for other candidates."""
        import errno as _errno

        old_secs = (agent_run.PRUNE_AFTER_DAYS + 1) * 86400
        good = _make_preserved_log(isolated_log_root, "goodlog", age_secs=old_secs)
        bad = _make_preserved_log(isolated_log_root, "badlog", age_secs=old_secs)
        deep = bad / "deep"
        deep.mkdir(exist_ok=True)

        real_scandir = os.scandir

        def fake_scandir(path):
            if str(path) == str(deep):
                raise OSError(_errno.ENAMETOOLONG, "File name too long", str(path))
            return real_scandir(path)

        monkeypatch.setattr(os, "scandir", fake_scandir)

        rc = agent_run.cmd_reap(_reap_args(include_logs=True))

        # reap must complete normally; the good log is collected.
        assert rc == 0
        assert not good.exists(), "goodlog must be collected despite bad neighbour"


# ---------------------------------------------------------------------------
# C7: pass 2 scratch deletion must not defer same-invocation log GC
# ---------------------------------------------------------------------------

class TestLogGCAfterScratchDeletion:
    """Deleting tmp/ in pass 2 bumps log_d's mtime to now.

    _newest_mtime_recursive for pass 2.5 must ignore the top-level
    directory's own mtime so a 30-day-old log is still eligible in the
    same invocation as the state-dir GC that removed tmp/."""

    def test_log_collected_same_invocation_as_state_with_scratch(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        """A terminal run with a tmp/ scratch dir: both state and log are
        collected in one reap --include-logs invocation."""
        old_secs = (agent_run.PRUNE_AFTER_DAYS + 1) * 86400
        log_age_secs = old_secs  # log older than log_min_age_threshold

        sd, ld = _make_run(
            isolated_runs_root, isolated_log_root, "c7scratch",
            status="done",
            ended_at_age_secs=old_secs,
            make_scratch=True,
        )
        # Backdate log file and log dir to old_secs ago.
        old_ts = time.time() - old_secs
        for p in [ld / "log"]:
            if p.exists():
                os.utime(p, (old_ts, old_ts))
        os.utime(ld, (old_ts, old_ts))
        # Also backdate the scratch dir contents.
        scratch = ld / "tmp"
        for p in scratch.rglob("*"):
            try:
                os.utime(p, (old_ts, old_ts))
            except OSError:
                pass
        os.utime(scratch, (old_ts, old_ts))

        rc = agent_run.cmd_reap(_reap_args(
            include_logs=True,
            min_age_hours=0.001,          # state threshold: collect immediately
            log_min_age_hours=old_secs / 3600 * 0.9,  # just under old_secs
        ))

        assert rc == 0
        out = capsys.readouterr().out
        assert not sd.exists(), "state dir must be collected by pass 2"
        assert not ld.exists(), (
            "log dir must be collected by pass 2.5 in the same invocation; "
            f"output was:\n{out}"
        )
        assert "logs_collected=1" in out
