"""Tests for the orphan-process termination pass in cmd_reap (Task 3).

All tests use synthetic process tables and a fake signal sender — no real
processes are spawned or killed.  Assertions are on what signals would have
been sent (captured by the fake sender) and on counter values in the
"reap done:" summary line.

The helpers under test:
  cmd_reap (Pass 4 / --orphan-processes)
  ORPHAN_KILL_GRACE_SECONDS
  _parse_orphan_min_age_seconds (via CLI flag/env independence)
"""
from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path
from typing import List, Optional, Tuple
from unittest.mock import patch

import pytest

from toolbox import agent_run
from toolbox.agent_run import (
    ORPHAN_KILL_GRACE_SECONDS,
    _OrphanCandidate,
    _OrphanSkip,
    _ProcEntry,
    _parse_orphan_min_age_seconds,
    cmd_reap,
)


# ---------------------------------------------------------------------------
# Module-level safety guards
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_real_signals(monkeypatch):
    """Any test reaching a real os.kill/os.killpg is a bug: fail loudly instead
    of terminating a live process.  Tests that legitimately assert on signalling
    install their own recording stubs via monkeypatch, which override these."""
    monkeypatch.setattr(os, "kill", lambda *a, **k: pytest.fail(f"real os.kill{a}"))
    monkeypatch.setattr(os, "killpg", lambda *a, **k: pytest.fail(f"real os.killpg{a}"))


@pytest.fixture(autouse=True)
def _no_real_scan(monkeypatch):
    """Prevent any test from accidentally enumerating the host process table.
    Tests that need a table install their own stub via monkeypatch, overriding
    this default."""
    monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

MY_UID = os.getuid()
FAR_PAST = time.time() - 100_000   # ~28 h ago — always older than any threshold


def _make_proc_entry(
    pid: int,
    name: str,
    *,
    ppid: int = 999,
    uid: int = MY_UID,
    start_time: float = FAR_PAST,
    identity: str = "",
    pgid: int = 0,
) -> _ProcEntry:
    ident = identity or f"darwin:{pid}-test-identity"
    argv = ["agent-run", name, "claude"]
    return _ProcEntry(
        pid=pid, ppid=ppid, uid=uid,
        argv=argv, start_time=start_time,
        identity=ident, pgid=pgid,
    )


def _reap_args(
    *,
    dry_run: bool = False,
    idle_hours: Optional[float] = None,
    min_age_hours: Optional[float] = None,
    name: Optional[str] = None,
    force_unknown: bool = False,
    include_logs: bool = False,
    log_min_age_hours: Optional[float] = None,
    orphan_processes: bool = False,
    orphan_min_age_hours: Optional[float] = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        dry_run=dry_run,
        idle_hours=idle_hours,
        min_age_hours=min_age_hours,
        name=name,
        force_unknown=force_unknown,
        include_logs=include_logs,
        log_min_age_hours=log_min_age_hours,
        orphan_processes=orphan_processes,
        orphan_min_age_hours=orphan_min_age_hours,
    )


class _FakeSignalSender:
    """Captures (pid, signal) pairs sent by the orphan termination pass."""

    def __init__(self):
        self.calls: List[Tuple[int, int]] = []
        self.killpg_calls: List[Tuple[int, int]] = []

    def kill(self, pid: int, sig: int) -> None:
        self.calls.append((pid, sig))

    def killpg(self, pgid: int, sig: int) -> None:
        self.killpg_calls.append((pgid, sig))

    @property
    def all_sigs_sent(self) -> List[int]:
        return [sig for _pid, sig in self.calls + self.killpg_calls]

    def sent_to(self, pid: int) -> List[int]:
        return [sig for p, sig in self.calls if p == pid]


def _extract_summary(capsys) -> str:
    """Return the 'reap done:' summary line from captured stdout."""
    out = capsys.readouterr().out
    for line in out.splitlines():
        if "reap done:" in line:
            return line
    return ""


def _summary_field(line: str, field: str) -> int:
    """Parse an integer field from the 'reap done:' summary line."""
    for token in line.split():
        if token.startswith(f"{field}="):
            return int(token.split("=", 1)[1])
    raise KeyError(f"field {field!r} not found in: {line!r}")


# ---------------------------------------------------------------------------
# --orphan-processes absent → behaviour byte-identical to today
# ---------------------------------------------------------------------------

class TestOrphanProcessesAbsent:
    """Without --orphan-processes, no scan occurs and counters are zero."""

    def test_no_scan_no_signals(self, isolated_runs_root, monkeypatch, capsys):
        """--orphan-processes not given → _scan_process_table never called."""
        scan_called = []
        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: scan_called.append(1) or [])
        cmd_reap(_reap_args())
        assert not scan_called

    def test_summary_has_zero_orphan_counters(self, isolated_runs_root, capsys):
        cmd_reap(_reap_args())
        line = _extract_summary(capsys)
        assert "orphan_procs_killed=0" in line
        assert "orphan_procs_skipped=0" in line


# ---------------------------------------------------------------------------
# --dry-run sends no signals at all
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_sends_no_signals(self, isolated_runs_root, monkeypatch, capsys):
        entry = _make_proc_entry(9100, "dryrun")
        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        monkeypatch.setattr(agent_run, "_process_identity",
                            lambda pid: f"darwin:{pid}-test-identity")
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)

        signals_sent: List[Tuple[int, int]] = []
        monkeypatch.setattr(os, "kill", lambda pid, sig: signals_sent.append((pid, sig)))
        monkeypatch.setattr(os, "killpg", lambda pgid, sig: signals_sent.append((-pgid, sig)))

        cmd_reap(_reap_args(dry_run=True, orphan_processes=True))
        assert not signals_sent

    def test_dry_run_counter_increments(self, isolated_runs_root, monkeypatch, capsys):
        entry = _make_proc_entry(9101, "drycount")
        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        monkeypatch.setattr(agent_run, "_process_identity",
                            lambda pid: f"darwin:{pid}-test-identity")
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        monkeypatch.setattr(os, "kill", lambda *_a: None)
        monkeypatch.setattr(os, "killpg", lambda *_a: None)

        cmd_reap(_reap_args(dry_run=True, orphan_processes=True))
        line = _extract_summary(capsys)
        assert _summary_field(line, "orphan_procs_killed") == 1
        assert _summary_field(line, "orphan_procs_skipped") == 0


# ---------------------------------------------------------------------------
# TERM sent, process dies within grace → no KILL
# ---------------------------------------------------------------------------

class TestTermGrace:
    def test_term_sent_dies_within_grace_no_kill(
        self, isolated_runs_root, monkeypatch, capsys
    ):
        pid = 9200
        entry = _make_proc_entry(pid, "gracerun")
        identity = f"darwin:{pid}-test-identity"

        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        monkeypatch.setattr(agent_run, "_process_identity", lambda p: identity)
        monkeypatch.setattr(agent_run, "ORPHAN_KILL_GRACE_SECONDS", 0.1)
        monkeypatch.setattr(agent_run, "KILL_POLL_INTERVAL_SECONDS", 0.01)

        signals_sent: List[Tuple[int, int]] = []
        alive = [True]  # mutable so the closure can set it

        def fake_kill(p, sig):
            signals_sent.append((p, sig))
            if sig == signal.SIGTERM:
                alive[0] = False  # process "dies" immediately on TERM

        monkeypatch.setattr(os, "kill", fake_kill)
        monkeypatch.setattr(os, "killpg", lambda pgid, sig: signals_sent.append((-pgid, sig)))
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: alive[0])
        # pgid != pid → signal pid only
        monkeypatch.setattr(os, "getpgid", lambda p: p + 1)

        cmd_reap(_reap_args(orphan_processes=True))
        sigs = [sig for _p, sig in signals_sent]
        assert signal.SIGTERM in sigs
        assert signal.SIGKILL not in sigs

    def test_term_killed_counter(self, isolated_runs_root, monkeypatch, capsys):
        pid = 9201
        entry = _make_proc_entry(pid, "gracecount")
        identity = f"darwin:{pid}-test-identity"

        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        monkeypatch.setattr(agent_run, "_process_identity", lambda p: identity)
        monkeypatch.setattr(agent_run, "ORPHAN_KILL_GRACE_SECONDS", 0.1)
        monkeypatch.setattr(agent_run, "KILL_POLL_INTERVAL_SECONDS", 0.01)

        alive = [True]

        def fake_kill(p, sig):
            if sig == signal.SIGTERM:
                alive[0] = False

        monkeypatch.setattr(os, "kill", fake_kill)
        monkeypatch.setattr(os, "killpg", lambda *_a: None)
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: alive[0])
        monkeypatch.setattr(os, "getpgid", lambda p: p + 1)

        cmd_reap(_reap_args(orphan_processes=True))
        line = _extract_summary(capsys)
        assert _summary_field(line, "orphan_procs_killed") == 1


# ---------------------------------------------------------------------------
# TERM sent, process survives grace → KILL follows
# ---------------------------------------------------------------------------

class TestTermKillEscalation:
    def test_term_survives_grace_kill_follows(
        self, isolated_runs_root, monkeypatch, capsys
    ):
        pid = 9300
        entry = _make_proc_entry(pid, "stubborn")
        identity = f"darwin:{pid}-test-identity"

        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        monkeypatch.setattr(agent_run, "_process_identity", lambda p: identity)
        monkeypatch.setattr(agent_run, "ORPHAN_KILL_GRACE_SECONDS", 0.05)
        monkeypatch.setattr(agent_run, "KILL_POLL_INTERVAL_SECONDS", 0.01)

        signals_sent: List[Tuple[int, int]] = []

        monkeypatch.setattr(os, "kill",
                            lambda p, sig: signals_sent.append((p, sig)))
        monkeypatch.setattr(os, "killpg", lambda pgid, sig: signals_sent.append((-pgid, sig)))
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)  # never dies
        monkeypatch.setattr(os, "getpgid", lambda p: p + 1)  # pgid != pid

        cmd_reap(_reap_args(orphan_processes=True))
        sigs = [sig for _p, sig in signals_sent]
        assert signal.SIGTERM in sigs
        assert signal.SIGKILL in sigs

    def test_kill_after_grace_counted(self, isolated_runs_root, monkeypatch, capsys):
        pid = 9301
        entry = _make_proc_entry(pid, "stubborn2")
        identity = f"darwin:{pid}-test-identity"

        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        monkeypatch.setattr(agent_run, "_process_identity", lambda p: identity)
        monkeypatch.setattr(agent_run, "ORPHAN_KILL_GRACE_SECONDS", 0.05)
        monkeypatch.setattr(agent_run, "KILL_POLL_INTERVAL_SECONDS", 0.01)

        monkeypatch.setattr(os, "kill", lambda *_a: None)
        monkeypatch.setattr(os, "killpg", lambda *_a: None)
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        monkeypatch.setattr(os, "getpgid", lambda p: p + 1)

        cmd_reap(_reap_args(orphan_processes=True))
        line = _extract_summary(capsys)
        assert _summary_field(line, "orphan_procs_killed") == 1


# ---------------------------------------------------------------------------
# Identity changed between discovery and signalling → skipped
# ---------------------------------------------------------------------------

class TestIdentityChanged:
    def test_identity_changed_aborts_candidate(
        self, isolated_runs_root, monkeypatch, capsys
    ):
        pid = 9400
        original_identity = f"darwin:{pid}-original"
        new_identity = f"darwin:{pid}-recycled"
        entry = _make_proc_entry(pid, "recycled", identity=original_identity)

        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        # _process_identity returns a different token every call (PID recycled).
        monkeypatch.setattr(agent_run, "_process_identity", lambda p: new_identity)
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)

        signals_sent: List[Tuple[int, int]] = []
        monkeypatch.setattr(os, "kill",
                            lambda p, sig: signals_sent.append((p, sig)))
        monkeypatch.setattr(os, "killpg", lambda pgid, sig: signals_sent.append((-pgid, sig)))

        cmd_reap(_reap_args(orphan_processes=True))
        assert not signals_sent

    def test_identity_changed_counted_skipped(
        self, isolated_runs_root, monkeypatch, capsys
    ):
        pid = 9401
        entry = _make_proc_entry(pid, "recycled2", identity=f"darwin:{pid}-orig")

        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        monkeypatch.setattr(agent_run, "_process_identity", lambda p: f"darwin:{p}-new")
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        monkeypatch.setattr(os, "kill", lambda *_a: None)

        cmd_reap(_reap_args(orphan_processes=True))
        line = _extract_summary(capsys)
        assert _summary_field(line, "orphan_procs_skipped") == 1
        assert _summary_field(line, "orphan_procs_killed") == 0


# ---------------------------------------------------------------------------
# State dir reappeared before lock → skipped
# ---------------------------------------------------------------------------

class TestStateDirReappeared:
    def test_state_dir_reappeared_skipped(
        self, isolated_runs_root, monkeypatch, capsys
    ):
        pid = 9500
        name = "reappeared"
        entry = _make_proc_entry(pid, name)

        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        monkeypatch.setattr(agent_run, "_process_identity",
                            lambda p: f"darwin:{p}-test-identity")
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)

        # Create the state dir so it "reappears" before the lock.
        (isolated_runs_root / name).mkdir(parents=True, exist_ok=True)

        signals_sent: List[Tuple[int, int]] = []
        monkeypatch.setattr(os, "kill",
                            lambda p, sig: signals_sent.append((p, sig)))

        cmd_reap(_reap_args(orphan_processes=True))
        assert not signals_sent

    def test_state_dir_reappeared_counted_skipped(
        self, isolated_runs_root, monkeypatch, capsys
    ):
        pid = 9501
        name = "reappeared2"
        entry = _make_proc_entry(pid, name)

        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        monkeypatch.setattr(agent_run, "_process_identity",
                            lambda p: f"darwin:{p}-test-identity")
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        (isolated_runs_root / name).mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(os, "kill", lambda *_a: None)

        cmd_reap(_reap_args(orphan_processes=True))
        line = _extract_summary(capsys)
        assert _summary_field(line, "orphan_procs_skipped") == 1

    def test_state_dir_reappeared_after_discovery_skipped(
        self, isolated_runs_root, monkeypatch, capsys
    ):
        """State dir that appears between discovery and the in-lock re-check
        causes the candidate to be skipped with the in-lock reason string.

        _path_entry_exists returns False on the first call (discovery) so the
        entry becomes a candidate, then True on the second call (inside the lock)
        simulating a concurrent launch that created the state dir in the window
        between discovery and the lock."""
        pid = 9502
        name = "reappeared3"
        entry = _make_proc_entry(pid, name)

        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        monkeypatch.setattr(agent_run, "_process_identity",
                            lambda p: f"darwin:{p}-test-identity")
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)

        call_count: List[int] = [0]
        real_exists = agent_run._path_entry_exists

        def patched_exists(path):
            call_count[0] += 1
            # First call: discovery — state dir absent, entry becomes a candidate.
            # Second call: inside lock — state dir has appeared.
            if call_count[0] == 1:
                return False
            return True

        monkeypatch.setattr(agent_run, "_path_entry_exists", patched_exists)

        cmd_reap(_reap_args(orphan_processes=True))
        out = capsys.readouterr().out
        line = next((l for l in out.splitlines() if "reap done:" in l), "")
        assert _summary_field(line, "orphan_procs_killed") == 0
        assert _summary_field(line, "orphan_procs_skipped") == 1
        assert "state dir appeared after discovery" in out


# ---------------------------------------------------------------------------
# os.getpgid(pid) != pid → pid signalled, group not
# ---------------------------------------------------------------------------

class TestPgidMismatch:
    def test_pgid_ne_pid_signals_pid_only(
        self, isolated_runs_root, monkeypatch, capsys
    ):
        pid = 9600
        entry = _make_proc_entry(pid, "pgidrun")
        identity = f"darwin:{pid}-test-identity"

        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        monkeypatch.setattr(agent_run, "_process_identity", lambda p: identity)
        monkeypatch.setattr(agent_run, "ORPHAN_KILL_GRACE_SECONDS", 0.05)
        monkeypatch.setattr(agent_run, "KILL_POLL_INTERVAL_SECONDS", 0.01)

        # pgid != pid → must not kill group.
        monkeypatch.setattr(os, "getpgid", lambda p: p + 99)

        kill_calls: List[Tuple[int, int]] = []
        killpg_calls: List[Tuple[int, int]] = []
        monkeypatch.setattr(os, "kill",
                            lambda p, sig: kill_calls.append((p, sig)))
        monkeypatch.setattr(os, "killpg",
                            lambda pgid, sig: killpg_calls.append((pgid, sig)))
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)

        cmd_reap(_reap_args(orphan_processes=True))
        # Signals must go to the pid, not a process group.
        assert any(p == pid for p, _ in kill_calls)
        assert not killpg_calls

    def test_pgid_eq_pid_signals_group(
        self, isolated_runs_root, monkeypatch, capsys
    ):
        pid = 9601
        # Set pgid=pid: the entry is a group leader (its own pgid != self_pgid=0).
        entry = _make_proc_entry(pid, "pgidsession", pgid=pid)
        identity = f"darwin:{pid}-test-identity"

        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        monkeypatch.setattr(agent_run, "_process_identity", lambda p: identity)
        monkeypatch.setattr(agent_run, "ORPHAN_KILL_GRACE_SECONDS", 0.05)
        monkeypatch.setattr(agent_run, "KILL_POLL_INTERVAL_SECONDS", 0.01)

        # pgid == pid at runtime → runner is a session leader, kill the group.
        monkeypatch.setattr(os, "getpgid", lambda p: p)

        kill_calls: List[Tuple[int, int]] = []
        killpg_calls: List[Tuple[int, int]] = []
        monkeypatch.setattr(os, "kill",
                            lambda p, sig: kill_calls.append((p, sig)))
        monkeypatch.setattr(os, "killpg",
                            lambda pgid, sig: killpg_calls.append((pgid, sig)))
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)

        cmd_reap(_reap_args(orphan_processes=True))
        # killpg must have been used, not bare kill, for the group.
        assert any(pgid == pid for pgid, _ in killpg_calls)


# ---------------------------------------------------------------------------
# Counters in summary and existing fields retain their order
# ---------------------------------------------------------------------------

class TestSummaryCounters:
    def test_existing_fields_before_orphan_fields(
        self, isolated_runs_root, monkeypatch, capsys
    ):
        """orphan_procs_* must appear after all pre-existing summary fields.

        An empty process table avoids any real scan; the ordering assertion is
        independent of whether any orphan is found.
        """
        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [])
        cmd_reap(_reap_args(orphan_processes=True))
        line = _extract_summary(capsys)
        pre_existing = [
            "died=", "killed=", "skipped=", "collected=",
            "orphaned_scratch=", "gc_skipped=", "resumed=", "logs_collected=",
        ]
        orphan_fields = ["orphan_procs_killed=", "orphan_procs_skipped="]
        for field in pre_existing + orphan_fields:
            assert field in line, f"field {field!r} missing from: {line!r}"
        # Ordering: every pre-existing field must appear before orphan fields.
        for pre in pre_existing:
            for orph in orphan_fields:
                assert line.index(pre) < line.index(orph), (
                    f"{pre!r} must appear before {orph!r}"
                )

    def test_multiple_candidates_counted_correctly(
        self, isolated_runs_root, monkeypatch, capsys
    ):
        entries = [_make_proc_entry(9700 + i, f"multi{i}") for i in range(3)]
        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: entries)
        monkeypatch.setattr(agent_run, "_process_identity",
                            lambda pid: f"darwin:{pid}-test-identity")
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        monkeypatch.setattr(agent_run, "ORPHAN_KILL_GRACE_SECONDS", 0.05)
        monkeypatch.setattr(agent_run, "KILL_POLL_INTERVAL_SECONDS", 0.01)
        monkeypatch.setattr(os, "kill", lambda *_a: None)
        monkeypatch.setattr(os, "killpg", lambda *_a: None)
        monkeypatch.setattr(os, "getpgid", lambda p: p + 1)

        cmd_reap(_reap_args(orphan_processes=True))
        line = _extract_summary(capsys)
        assert _summary_field(line, "orphan_procs_killed") == 3


# ---------------------------------------------------------------------------
# --name targeting
# ---------------------------------------------------------------------------

class TestNameTargeting:
    def test_name_targets_only_matching_run(
        self, isolated_runs_root, monkeypatch, capsys
    ):
        entries = [
            _make_proc_entry(9800, "run-a"),
            _make_proc_entry(9801, "run-b"),
        ]
        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: entries)
        monkeypatch.setattr(agent_run, "_process_identity",
                            lambda pid: f"darwin:{pid}-test-identity")
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        monkeypatch.setattr(agent_run, "ORPHAN_KILL_GRACE_SECONDS", 0.05)
        monkeypatch.setattr(agent_run, "KILL_POLL_INTERVAL_SECONDS", 0.01)
        monkeypatch.setattr(os, "kill", lambda *_a: None)
        monkeypatch.setattr(os, "killpg", lambda *_a: None)
        monkeypatch.setattr(os, "getpgid", lambda p: p + 1)

        cmd_reap(_reap_args(orphan_processes=True, name="run-a"))
        line = _extract_summary(capsys)
        assert _summary_field(line, "orphan_procs_killed") == 1


# ---------------------------------------------------------------------------
# Discovery-level safety refusals still hold end-to-end through cmd_reap
# ---------------------------------------------------------------------------

class TestDiscoveryRefusals:
    """Safety rules from _find_orphan_runners still apply end-to-end."""

    def test_self_is_never_killed(self, isolated_runs_root, monkeypatch, capsys):
        entry = _make_proc_entry(os.getpid(), "selfrun")
        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        monkeypatch.setattr(agent_run, "_process_identity",
                            lambda p: f"darwin:{p}-test-identity")

        cmd_reap(_reap_args(orphan_processes=True))
        out = capsys.readouterr().out
        line = next((l for l in out.splitlines() if "reap done:" in l), "")
        assert _summary_field(line, "orphan_procs_killed") == 0
        assert _summary_field(line, "orphan_procs_skipped") == 1
        assert "skipped: self" in out

    def test_pid_1_is_never_killed(self, isolated_runs_root, monkeypatch, capsys):
        entry = _make_proc_entry(1, "initrun")
        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        monkeypatch.setattr(agent_run, "_process_identity",
                            lambda p: f"darwin:{p}-test-identity")

        cmd_reap(_reap_args(orphan_processes=True))
        out = capsys.readouterr().out
        line = next((l for l in out.splitlines() if "reap done:" in l), "")
        assert _summary_field(line, "orphan_procs_killed") == 0
        assert _summary_field(line, "orphan_procs_skipped") == 1
        assert "skipped: pid_1" in out

    def test_bash_lc_false_positive_not_killed(
        self, isolated_runs_root, monkeypatch, capsys
    ):
        """bash -lc 'cat /var/tmp/agent-runs/foo/log' must never be a candidate.

        It must not even appear in the skip log — it is not an agent-run runner
        and is excluded before any safety check runs.
        """
        entry = _ProcEntry(
            pid=9900, ppid=999, uid=MY_UID,
            argv=["bash", "-lc", "cat /var/tmp/agent-runs/foo/log"],
            start_time=FAR_PAST,
            identity="darwin:9900-test",
            pgid=0,
        )
        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        monkeypatch.setattr(agent_run, "_process_identity",
                            lambda p: f"darwin:{p}-test-identity")

        cmd_reap(_reap_args(orphan_processes=True))
        out = capsys.readouterr().out
        line = next((l for l in out.splitlines() if "reap done:" in l), "")
        # Not a runner: counts as neither killed nor skipped.
        assert _summary_field(line, "orphan_procs_killed") == 0
        assert _summary_field(line, "orphan_procs_skipped") == 0
        assert "9900" not in out

    def test_state_dir_present_skips_process(
        self, isolated_runs_root, monkeypatch, capsys
    ):
        """A process whose state dir exists is tracked by other passes — skip."""
        name = "tracked"
        entry = _make_proc_entry(9901, name)
        (isolated_runs_root / name).mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        monkeypatch.setattr(agent_run, "_process_identity",
                            lambda p: f"darwin:{p}-test-identity")

        cmd_reap(_reap_args(orphan_processes=True))
        out = capsys.readouterr().out
        line = next((l for l in out.splitlines() if "reap done:" in l), "")
        assert _summary_field(line, "orphan_procs_killed") == 0
        assert _summary_field(line, "orphan_procs_skipped") == 1
        assert "skipped: state_dir_exists" in out


# ---------------------------------------------------------------------------
# --orphan-min-age-hours flag/env/default resolution; independence from other thresholds
# ---------------------------------------------------------------------------

class TestOrphanMinAgeHours:
    def test_default_is_24h(self, monkeypatch):
        monkeypatch.delenv("AGENT_RUN_ORPHAN_MIN_AGE_HOURS", raising=False)
        assert _parse_orphan_min_age_seconds() == pytest.approx(24 * 3600)

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("AGENT_RUN_ORPHAN_MIN_AGE_HOURS", "48")
        assert _parse_orphan_min_age_seconds() == pytest.approx(48 * 3600)

    def test_flag_overrides_env(self, isolated_runs_root, monkeypatch, capsys):
        """--orphan-min-age-hours flag takes precedence over the env var."""
        now = time.time()
        # Process is 2 h old; env says 3 h threshold, flag says 1 h.
        entry = _ProcEntry(
            pid=9990, ppid=999, uid=MY_UID,
            argv=["agent-run", "agetest", "claude"],
            start_time=now - 7200.0,  # 2 h old
            identity="darwin:9990-test",
        )
        monkeypatch.setenv("AGENT_RUN_ORPHAN_MIN_AGE_HOURS", "3")
        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        monkeypatch.setattr(agent_run, "_process_identity", lambda p: "darwin:9990-test")
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        monkeypatch.setattr(agent_run, "ORPHAN_KILL_GRACE_SECONDS", 0.05)
        monkeypatch.setattr(agent_run, "KILL_POLL_INTERVAL_SECONDS", 0.01)
        monkeypatch.setattr(os, "kill", lambda *_a: None)
        monkeypatch.setattr(os, "killpg", lambda *_a: None)
        monkeypatch.setattr(os, "getpgid", lambda p: p + 1)

        # With --orphan-min-age-hours 1, the 2 h old process IS a candidate.
        cmd_reap(_reap_args(orphan_processes=True, orphan_min_age_hours=1.0))
        line = _extract_summary(capsys)
        assert _summary_field(line, "orphan_procs_killed") == 1

    def test_too_young_not_candidate(self, isolated_runs_root, monkeypatch, capsys):
        now = time.time()
        entry = _ProcEntry(
            pid=9991, ppid=999, uid=MY_UID,
            argv=["agent-run", "youngrun", "claude"],
            start_time=now - 60.0,   # 1 min old
            identity="darwin:9991-test",
        )
        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        monkeypatch.setattr(os, "kill", lambda *_a: None)

        # threshold=1h, process is only 60s old → skip
        cmd_reap(_reap_args(orphan_processes=True, orphan_min_age_hours=1.0))
        line = _extract_summary(capsys)
        assert _summary_field(line, "orphan_procs_killed") == 0

    def test_independent_of_min_age_hours(self, monkeypatch):
        """Changing AGENT_RUN_MIN_AGE_HOURS must not affect the orphan threshold."""
        monkeypatch.setenv("AGENT_RUN_MIN_AGE_HOURS", "1")
        monkeypatch.delenv("AGENT_RUN_ORPHAN_MIN_AGE_HOURS", raising=False)
        assert _parse_orphan_min_age_seconds() == pytest.approx(24 * 3600)

    def test_independent_of_log_min_age_hours(self, monkeypatch):
        """Changing AGENT_RUN_LOG_MIN_AGE_HOURS must not affect the orphan threshold."""
        monkeypatch.setenv("AGENT_RUN_LOG_MIN_AGE_HOURS", "1")
        monkeypatch.delenv("AGENT_RUN_ORPHAN_MIN_AGE_HOURS", raising=False)
        assert _parse_orphan_min_age_seconds() == pytest.approx(24 * 3600)

    def test_orphan_min_age_hours_flag_parsed_without_orphan_processes(
        self, isolated_runs_root, monkeypatch, capsys
    ):
        """--orphan-min-age-hours is parsed/validated even without --orphan-processes."""
        # Just verify cmd_reap doesn't error out when the flag is provided alone.
        # The flag only gate-keeps; no scan happens.
        scan_called = []
        monkeypatch.setattr(agent_run, "_scan_process_table",
                            lambda: scan_called.append(1) or [])
        cmd_reap(_reap_args(orphan_min_age_hours=48.0))  # no orphan_processes=True
        assert not scan_called
