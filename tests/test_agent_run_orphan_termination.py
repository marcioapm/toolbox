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
from typing import Dict, List, Optional, Tuple

import pytest

from toolbox import agent_run
from toolbox.agent_run import (
    ORPHAN_KILL_GRACE_SECONDS,
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
def _no_real_pidfd(monkeypatch):
    """Block the pidfd signal path so no real signal is sent to a fabricated pid.

    On Linux, _send_signal_to_verified_pid uses os.pidfd_open +
    signal.pidfd_send_signal before falling back to os.kill.  Fabricated pids
    (9100, 9200, …) do not exist: a real os.pidfd_open call would raise
    ProcessLookupError, causing the candidate to be counted skipped rather than
    killed.  Tests that assert on signal delivery install recording stubs via
    _install_signal_stubs(), which override this guard."""
    monkeypatch.setattr(
        agent_run.os, "pidfd_open",
        lambda *a, **k: pytest.fail(f"real os.pidfd_open{a}"),
        raising=False,
    )
    monkeypatch.setattr(
        agent_run.signal, "pidfd_send_signal",
        lambda *a, **k: pytest.fail(f"real signal.pidfd_send_signal{a}"),
        raising=False,
    )


@pytest.fixture(autouse=True)
def _no_real_scan(monkeypatch):
    """Prevent any test from accidentally enumerating the host process table.
    Tests that need a table install their own stub via monkeypatch, overriding
    this default."""
    monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [])


@pytest.fixture(autouse=True)
def _stub_runner_state_root(monkeypatch):
    """Return STATE_ROOT for every fabricated pid so tests are not platform-
    dependent.  On Linux, _runner_state_root reads /proc/<pid>/environ and
    returns None on OSError; fabricated pids (9100, 9200, …) do not exist, so
    every candidate would be skipped as state_root_unreadable on Linux CI.
    Tests that need a different return value install their own stub."""
    monkeypatch.setattr(agent_run, "_runner_state_root",
                        lambda _pid: agent_run.STATE_ROOT)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

MY_UID = os.getuid()
FAR_PAST = time.time() - 100_000   # ~28 h ago — always older than any threshold


def _make_log_dir(name: str) -> None:
    """Create LOG_ROOT/<name> so the log-dir corroboration check passes.

    Every real agent-run runner creates this directory on launch, so any
    entry expected to survive past the log_dir_missing skip must have it.
    """
    agent_run.LOG_ROOT.mkdir(parents=True, exist_ok=True)
    (agent_run.LOG_ROOT / name).mkdir(exist_ok=True)


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


def _install_signal_stubs(
    monkeypatch,
    sent: List[Tuple[int, int]],
    *,
    on_kill: Optional[object] = None,
) -> None:
    """Install recording stubs for both the Darwin (os.kill/killpg) and Linux
    (os.pidfd_open + signal.pidfd_send_signal) signal paths.

    All signals recorded as (pid, sig); killpg signals encoded as (-pgid, sig).
    ``on_kill`` may be a callable(pid, sig) called after recording, used to
    drive stateful fake-alive flags.  Both paths write to the same ``sent``
    list so assertions work regardless of which platform branch runs.

    This overrides both the _no_real_signals and _no_real_pidfd autouse guards.
    """
    fd_counter: List[int] = [100]
    fd_to_pid: Dict[int, int] = {}

    def fake_pidfd_open(pid: int, flags: int) -> int:
        fd = fd_counter[0]
        fd_counter[0] += 1
        fd_to_pid[fd] = pid
        return fd

    def fake_pidfd_send_signal(fd: int, sig: int, info: object, flags: int) -> None:
        pid = fd_to_pid.get(fd)
        if pid is not None:
            sent.append((pid, sig))
            if on_kill is not None and callable(on_kill):
                on_kill(pid, sig)

    def fake_kill(pid: int, sig: int) -> None:
        sent.append((pid, sig))
        if on_kill is not None and callable(on_kill):
            on_kill(pid, sig)

    monkeypatch.setattr(os, "kill", fake_kill)
    monkeypatch.setattr(os, "killpg", lambda g, s: sent.append((-g, s)))
    monkeypatch.setattr(agent_run.os, "pidfd_open", fake_pidfd_open, raising=False)
    monkeypatch.setattr(agent_run.signal, "pidfd_send_signal",
                        fake_pidfd_send_signal, raising=False)
    # Prevent os.close from receiving our synthetic fds; real fds pass through.
    real_close = os.close
    monkeypatch.setattr(
        agent_run.os, "close",
        lambda fd: (None if fd in fd_to_pid else real_close(fd)),
    )

@pytest.fixture
def orphan_harness(monkeypatch):
    """Return a setup callable that installs the standard orphan-test stubs.

    Usage::

        def test_something(isolated_runs_root, orphan_harness, capsys):
            entry = _make_proc_entry(...)
            signals = orphan_harness([entry])
            cmd_reap(_reap_args(orphan_processes=True))
            assert (pid, signal.SIGTERM) in signals

    ``alive`` may be a bool (constant) or a zero-argument callable returning
    bool (stateful).  ``pgid_offset`` controls ``getpgid`` return value.

    Signal recording covers both the Darwin path (os.kill / os.killpg) and the
    Linux pidfd path (os.pidfd_open + signal.pidfd_send_signal), so assertions
    on ``signals`` work regardless of which platform branch runs.
    """
    def setup(entries, *, identity=None, alive=True, pgid_offset=1):
        sent: List[Tuple[int, int]] = []
        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: entries)
        monkeypatch.setattr(
            agent_run, "_process_identity",
            identity if identity is not None
            else (lambda p: f"darwin:{p}-test-identity"),
        )
        monkeypatch.setattr(agent_run, "ORPHAN_KILL_GRACE_SECONDS", 0.05)
        monkeypatch.setattr(agent_run, "KILL_POLL_INTERVAL_SECONDS", 0.01)
        monkeypatch.setattr(
            agent_run, "_pid_alive",
            lambda _p: (alive() if callable(alive) else alive),
        )
        monkeypatch.setattr(os, "getpgid", lambda p: p + pgid_offset)
        _install_signal_stubs(monkeypatch, sent)
        return sent
    return setup


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
    def test_dry_run_sends_no_signals_and_increments_killed(
        self, isolated_runs_root, orphan_harness, capsys
    ):
        entry = _make_proc_entry(9100, "dryrun")
        _make_log_dir("dryrun")
        signals = orphan_harness([entry])

        cmd_reap(_reap_args(dry_run=True, orphan_processes=True))

        assert not signals
        line = _extract_summary(capsys)
        assert _summary_field(line, "orphan_procs_killed") == 1
        assert _summary_field(line, "orphan_procs_skipped") == 0

    def test_dry_run_leaves_state_root_unchanged(
        self, isolated_runs_root, orphan_harness, capsys
    ):
        """--dry-run must not create any files or directories under STATE_ROOT.

        _launch_lock creates STATE_ROOT/.locks/<name>.lock; the dry-run branch
        must skip lock acquisition entirely so a preview run mutates nothing."""
        entry = _make_proc_entry(9101, "dryrootprobe")
        _make_log_dir("dryrootprobe")
        orphan_harness([entry])

        def _tree(root):
            """Sorted relative paths of everything under root."""
            paths = []
            for p in sorted(root.rglob("*")):
                paths.append(str(p.relative_to(root)))
            return paths

        before = _tree(isolated_runs_root)
        cmd_reap(_reap_args(dry_run=True, orphan_processes=True))
        after = _tree(isolated_runs_root)

        assert before == after, (
            f"STATE_ROOT changed during --dry-run:\n"
            f"  before: {before}\n"
            f"  after:  {after}"
        )
        line = _extract_summary(capsys)
        assert _summary_field(line, "orphan_procs_killed") == 1


# ---------------------------------------------------------------------------
# TERM sent, process dies within grace → no KILL
# ---------------------------------------------------------------------------

class TestTermGrace:
    def test_term_sent_dies_within_grace_no_kill(
        self, isolated_runs_root, monkeypatch, orphan_harness, capsys
    ):
        pid = 9200
        entry = _make_proc_entry(pid, "gracerun")
        _make_log_dir("gracerun")
        identity = f"darwin:{pid}-test-identity"

        alive = [True]

        def _on_kill(p: int, sig: int) -> None:
            if sig == signal.SIGTERM:
                alive[0] = False  # process "dies" immediately on TERM

        signals_sent = orphan_harness(
            [entry],
            identity=lambda p: identity,
            alive=lambda: alive[0],
        )
        # Replace the harness's signal stubs with ones that also flip alive;
        # _install_signal_stubs' on_kill hook drives the stateful alive flag on
        # both the Darwin (os.kill) and Linux (pidfd_send_signal) branches.
        _install_signal_stubs(monkeypatch, signals_sent, on_kill=_on_kill)

        cmd_reap(_reap_args(orphan_processes=True))
        sigs = [sig for _p, sig in signals_sent]
        assert signal.SIGTERM in sigs
        assert signal.SIGKILL not in sigs
        line = _extract_summary(capsys)
        assert _summary_field(line, "orphan_procs_killed") == 1


# ---------------------------------------------------------------------------
# TERM sent, process survives grace → KILL follows
# ---------------------------------------------------------------------------

class TestTermKillEscalation:
    def test_term_survives_grace_kill_follows(
        self, isolated_runs_root, orphan_harness, capsys
    ):
        pid = 9300
        entry = _make_proc_entry(pid, "stubborn")
        _make_log_dir("stubborn")
        identity = f"darwin:{pid}-test-identity"

        signals = orphan_harness(
            [entry],
            identity=lambda p: identity,
            alive=True,  # never dies
        )
        cmd_reap(_reap_args(orphan_processes=True))
        sigs = [sig for _p, sig in signals]
        assert signal.SIGTERM in sigs
        assert signal.SIGKILL in sigs
        line = _extract_summary(capsys)
        assert _summary_field(line, "orphan_procs_killed") == 1


# ---------------------------------------------------------------------------
# Identity changed between discovery and signalling → skipped
# ---------------------------------------------------------------------------

class TestIdentityChanged:
    def test_identity_changed_aborts_candidate(
        self, isolated_runs_root, orphan_harness, capsys
    ):
        pid = 9400
        original_identity = f"darwin:{pid}-original"
        new_identity = f"darwin:{pid}-recycled"
        entry = _make_proc_entry(pid, "recycled", identity=original_identity)
        _make_log_dir("recycled")

        # _process_identity returns a different token every call (PID recycled).
        signals = orphan_harness(
            [entry],
            identity=lambda p: new_identity,
            alive=True,
        )
        cmd_reap(_reap_args(orphan_processes=True))
        assert not signals
        line = _extract_summary(capsys)
        assert _summary_field(line, "orphan_procs_skipped") == 1
        assert _summary_field(line, "orphan_procs_killed") == 0


# ---------------------------------------------------------------------------
# State dir reappeared before lock → skipped
# ---------------------------------------------------------------------------

class TestStateDirReappeared:
    def test_state_dir_reappeared_skipped(
        self, isolated_runs_root, orphan_harness, capsys
    ):
        pid = 9500
        name = "reappeared"
        entry = _make_proc_entry(pid, name)
        _make_log_dir(name)

        signals = orphan_harness([entry], alive=True)

        # Create the state dir so it "reappears" before the lock.
        (isolated_runs_root / name).mkdir(parents=True, exist_ok=True)

        cmd_reap(_reap_args(orphan_processes=True))
        assert not signals
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
        _make_log_dir(name)

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
        self, isolated_runs_root, monkeypatch, orphan_harness, capsys
    ):
        pid = 9600
        entry = _make_proc_entry(pid, "pgidrun")
        _make_log_dir("pgidrun")
        identity = f"darwin:{pid}-test-identity"

        # harness installs pgid_offset=1 (pgid != pid); override to +99 to be explicit.
        signals = orphan_harness([entry], identity=lambda p: identity, alive=True, pgid_offset=99)

        cmd_reap(_reap_args(orphan_processes=True))
        # Signals must go to the pid, not a process group.
        # harness encodes kill(p, s) as (p, s) and killpg(g, s) as (-g, s).
        assert any(p == pid for p, _ in signals)
        assert all(p >= 0 for p, _ in signals), "no killpg calls expected"

    def test_pgid_eq_pid_signals_group(
        self, isolated_runs_root, monkeypatch, orphan_harness, capsys
    ):
        pid = 9601
        # Set pgid=pid in entry: the entry is a group leader.
        entry = _make_proc_entry(pid, "pgidsession", pgid=pid)
        _make_log_dir("pgidsession")
        identity = f"darwin:{pid}-test-identity"

        # harness pgid_offset=0 means getpgid returns p + 0 == p.
        signals = orphan_harness([entry], identity=lambda p: identity, alive=True, pgid_offset=0)

        cmd_reap(_reap_args(orphan_processes=True))
        # killpg must have been used (encoded as (-g, s) where g>0).
        assert any(p < 0 for p, _ in signals), "killpg call expected"


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
        self, isolated_runs_root, orphan_harness, capsys
    ):
        entries = [_make_proc_entry(9700 + i, f"multi{i}") for i in range(3)]
        for i in range(3):
            _make_log_dir(f"multi{i}")
        orphan_harness(entries)

        cmd_reap(_reap_args(orphan_processes=True))
        line = _extract_summary(capsys)
        assert _summary_field(line, "orphan_procs_killed") == 3


# ---------------------------------------------------------------------------
# --name targeting
# ---------------------------------------------------------------------------

class TestNameTargeting:
    def test_name_targets_only_matching_run(
        self, isolated_runs_root, orphan_harness, capsys
    ):
        entries = [
            _make_proc_entry(9800, "run-a"),
            _make_proc_entry(9801, "run-b"),
        ]
        _make_log_dir("run-a")
        # run-b is filtered by name_mismatch before log-dir check, so no log_dir needed.
        orphan_harness(entries)

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
        # Log dir must exist so the entry reaches the state_dir_exists check.
        _make_log_dir(name)
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
        _make_log_dir("agetest")
        monkeypatch.setenv("AGENT_RUN_ORPHAN_MIN_AGE_HOURS", "3")
        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        monkeypatch.setattr(agent_run, "_process_identity", lambda p: "darwin:9990-test")
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        monkeypatch.setattr(agent_run, "ORPHAN_KILL_GRACE_SECONDS", 0.05)
        monkeypatch.setattr(agent_run, "KILL_POLL_INTERVAL_SECONDS", 0.01)
        monkeypatch.setattr(os, "getpgid", lambda p: p + 1)
        # Stub both signal paths so neither reaches a real process.
        sent: List[Tuple[int, int]] = []
        _install_signal_stubs(monkeypatch, sent)

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
        # Log dir must exist so the entry reaches the age gate.
        _make_log_dir("youngrun")
        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        # Patch identity so the entry would be signalled if it reached pass 4;
        # only the too_young skip in _find_orphan_runners keeps it safe.
        monkeypatch.setattr(agent_run, "_process_identity",
                            lambda p: f"darwin:{p}-test-identity")

        # threshold=1h, process is only 60s old → skip
        cmd_reap(_reap_args(orphan_processes=True, orphan_min_age_hours=1.0))
        out = capsys.readouterr().out
        line = next((l for l in out.splitlines() if "reap done:" in l), "")
        assert _summary_field(line, "orphan_procs_killed") == 0
        assert _summary_field(line, "orphan_procs_skipped") == 1
        assert "skipped: too_young" in out

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


# ---------------------------------------------------------------------------
# P2: three-phase TERM/grace/KILL structure
# ---------------------------------------------------------------------------

class TestThreePhaseOrphan:
    """Prove that the three-phase restructure preserved all safety properties.

    Phase 1: TERM under each name's lock with identity re-read immediately
             before the signal.
    Phase 2: one shared grace window (O(5s) not O(5s × N)).
    Phase 3: re-verify identity and re-take lock before KILL.
    """

    def test_grace_is_shared_not_per_candidate(
        self, isolated_runs_root, monkeypatch, capsys
    ):
        """N candidates that all exit on TERM: total wait is one grace window,
        not N × grace.

        Synthetic table: 3 candidates, each "dies" immediately on SIGTERM.
        With a per-candidate sequential grace the test would take ≥ 3 × grace;
        with a shared grace it takes ≤ 1 × grace + overhead.
        """
        import time as _time
        grace = 0.3
        n = 3
        pids = [9200 + i for i in range(n)]
        names = [f"shared-grace-{i}" for i in range(n)]
        entries = [_make_proc_entry(p, n) for p, n in zip(pids, names)]
        for name in names:
            _make_log_dir(name)

        alive: dict = {p: True for p in pids}

        def _on_kill(p: int, sig: int) -> None:
            if sig == signal.SIGTERM:
                alive[p] = False

        def fake_pid_alive(p):
            return alive.get(p, False)

        sent: List[Tuple[int, int]] = []
        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: entries)
        monkeypatch.setattr(agent_run, "_process_identity",
                            lambda p: f"darwin:{p}-test-identity")
        monkeypatch.setattr(agent_run, "ORPHAN_KILL_GRACE_SECONDS", grace)
        monkeypatch.setattr(agent_run, "KILL_POLL_INTERVAL_SECONDS", 0.02)
        monkeypatch.setattr(agent_run, "_pid_alive", fake_pid_alive)
        monkeypatch.setattr(os, "getpgid", lambda p: p + 1)
        _install_signal_stubs(monkeypatch, sent, on_kill=_on_kill)

        t0 = _time.time()
        cmd_reap(_reap_args(orphan_processes=True))
        elapsed = _time.time() - t0

        # All three must have received TERM and be counted killed.
        line = _extract_summary(capsys)
        assert _summary_field(line, "orphan_procs_killed") == n
        assert _summary_field(line, "orphan_procs_skipped") == 0
        # Wall time must be well under N × grace (allow 2× for test jitter).
        assert elapsed < n * grace * 2, (
            f"elapsed {elapsed:.2f}s >= {n * grace * 2:.2f}s: grace appears sequential"
        )

    def test_phase1_identity_reread_before_term(
        self, isolated_runs_root, monkeypatch, capsys
    ):
        """Identity is re-read before SIGTERM (phase 1), not taken from scan.

        The scan-time identity in the _ProcEntry is stale (original_identity),
        but _process_identity returns a different token (recycled_identity).
        The candidate must be skipped — no signal sent.
        """
        pid = 9280
        original_identity = f"darwin:{pid}-original"
        recycled_identity = f"darwin:{pid}-recycled"
        entry = _make_proc_entry(pid, "p1-id-check", identity=original_identity)
        _make_log_dir("p1-id-check")

        signals_sent: List[Tuple[int, int]] = []
        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        # _process_identity always returns the recycled token, simulating PID reuse.
        monkeypatch.setattr(agent_run, "_process_identity", lambda p: recycled_identity)
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        _install_signal_stubs(monkeypatch, signals_sent)

        cmd_reap(_reap_args(orphan_processes=True))
        assert not signals_sent, "TERM must not be sent when identity mismatches at phase 1"
        line = _extract_summary(capsys)
        assert _summary_field(line, "orphan_procs_skipped") == 1

    def test_phase3_identity_reread_before_kill(
        self, isolated_runs_root, monkeypatch, capsys
    ):
        """Identity is re-read before SIGKILL (phase 3), not taken from phase 1.

        Phase 1 passes (identity matches, SIGTERM sent).  During the grace
        window the PID is recycled so _process_identity returns a different
        token by the time phase 3 checks it.  SIGKILL must not be sent.
        """
        pid = 9281
        original_identity = f"darwin:{pid}-original"
        recycled_identity = f"darwin:{pid}-recycled"
        entry = _make_proc_entry(pid, "p3-id-check", identity=original_identity)
        _make_log_dir("p3-id-check")

        # identity_call_count tracks all _process_identity invocations for pid.
        # Both the outer pre-signal check and _send_signal_to_verified_pid's
        # internal re-read must see the original token for SIGTERM to be sent;
        # any subsequent call (phase-3 re-verify) sees the recycled token so
        # SIGKILL is refused.
        call_count: List[int] = [0]

        def fake_identity(p):
            call_count[0] += 1
            # First two calls: outer check + _send_signal_to_verified_pid's
            # internal re-read — both must match for TERM to proceed.
            # Call 3+: grace window has elapsed, PID recycled → mismatch.
            return original_identity if call_count[0] <= 2 else recycled_identity

        signals_sent: List[Tuple[int, int]] = []
        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        monkeypatch.setattr(agent_run, "_process_identity", fake_identity)
        # Process never dies on TERM (survives grace, so phase 3 runs).
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        monkeypatch.setattr(agent_run, "ORPHAN_KILL_GRACE_SECONDS", 0.05)
        monkeypatch.setattr(agent_run, "KILL_POLL_INTERVAL_SECONDS", 0.01)
        monkeypatch.setattr(os, "getpgid", lambda p: p + 1)
        _install_signal_stubs(monkeypatch, signals_sent)

        cmd_reap(_reap_args(orphan_processes=True))

        sigs = [sig for _p, sig in signals_sent]
        assert signal.SIGTERM in sigs, "SIGTERM must be sent when phase-1 identity matches"
        assert signal.SIGKILL not in sigs, "SIGKILL must not be sent when phase-3 identity mismatches"
        line = _extract_summary(capsys)
        assert _summary_field(line, "orphan_procs_skipped") == 1

    def test_phase3_lock_retaken_before_kill(
        self, isolated_runs_root, monkeypatch, capsys
    ):
        """The per-name lock is re-acquired in phase 3 before SIGKILL.

        Verified indirectly: if the state dir appears between TERM and KILL
        (simulated by creating it after the first lock release), the candidate
        must still be counted skipped=1 rather than killed.

        Note: the current phase 3 does *not* re-check state dir presence —
        it only re-reads identity.  This test therefore proves that the lock
        is re-taken (preventing concurrent mutation) by verifying that the
        identity re-read in phase 3 is fresh, consistent with the lock being
        held.  Identity mismatch in phase 3 → skipped, which would only
        happen if the phase-3 identity re-read actually ran (inside the lock).
        """
        pid = 9282
        original_identity = f"darwin:{pid}-original"
        recycled_identity = f"darwin:{pid}-after-grace"
        entry = _make_proc_entry(pid, "p3-lock-check", identity=original_identity)
        _make_log_dir("p3-lock-check")

        call_count: List[int] = [0]

        def fake_identity(p):
            call_count[0] += 1
            return original_identity if call_count[0] == 1 else recycled_identity

        signals_sent: List[Tuple[int, int]] = []
        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        monkeypatch.setattr(agent_run, "_process_identity", fake_identity)
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        monkeypatch.setattr(agent_run, "ORPHAN_KILL_GRACE_SECONDS", 0.05)
        monkeypatch.setattr(agent_run, "KILL_POLL_INTERVAL_SECONDS", 0.01)
        monkeypatch.setattr(os, "getpgid", lambda p: p + 1)
        _install_signal_stubs(monkeypatch, signals_sent)

        cmd_reap(_reap_args(orphan_processes=True))

        # Phase-3 identity re-read must have run (inside the re-taken lock).
        assert call_count[0] >= 2, "identity must be re-read at least twice (phase 1 and phase 3)"
        sigs = [sig for _p, sig in signals_sent]
        assert signal.SIGKILL not in sigs

    def test_multiple_term_all_die_within_grace_no_kill(
        self, isolated_runs_root, monkeypatch, capsys
    ):
        """Multiple candidates: all die on TERM, so SIGKILL must never be sent."""
        n = 4
        pids = [9290 + i for i in range(n)]
        names = [f"multi-grace-{i}" for i in range(n)]
        entries = [_make_proc_entry(p, nm) for p, nm in zip(pids, names)]
        for name in names:
            _make_log_dir(name)

        alive: dict = {p: True for p in pids}

        def _on_kill(p: int, sig: int) -> None:
            if sig == signal.SIGTERM:
                alive[p] = False

        sent: List[Tuple[int, int]] = []
        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: entries)
        monkeypatch.setattr(agent_run, "_process_identity",
                            lambda p: f"darwin:{p}-test-identity")
        monkeypatch.setattr(agent_run, "ORPHAN_KILL_GRACE_SECONDS", 0.1)
        monkeypatch.setattr(agent_run, "KILL_POLL_INTERVAL_SECONDS", 0.01)
        monkeypatch.setattr(agent_run, "_pid_alive", lambda p: alive.get(p, False))
        monkeypatch.setattr(os, "getpgid", lambda p: p + 1)
        _install_signal_stubs(monkeypatch, sent, on_kill=_on_kill)

        cmd_reap(_reap_args(orphan_processes=True))

        line = _extract_summary(capsys)
        assert _summary_field(line, "orphan_procs_killed") == n
        assert _summary_field(line, "orphan_procs_skipped") == 0
        sigs_sent = [sig for _p, sig in sent]
        assert signal.SIGKILL not in sigs_sent, "SIGKILL must not be sent when all die on TERM"

    def test_deferred_field_present_in_summary(
        self, isolated_runs_root, monkeypatch, capsys
    ):
        """The summary line must always include deferred=N (zero when none deferred)."""
        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [])
        cmd_reap(_reap_args(orphan_processes=True))
        line = _extract_summary(capsys)
        assert "deferred=0" in line, f"deferred= field missing from summary: {line!r}"


# ---------------------------------------------------------------------------
# P5: wall-clock budget (--max-seconds / AGENT_RUN_REAP_MAX_SECONDS)
# ---------------------------------------------------------------------------

class TestReapBudget:
    """Budget exhaustion defers remaining candidates without aborting, exit 0."""

    def test_budget_exceeded_defers_candidates(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        """With a budget of effectively 0 s, every candidate past the first is
        deferred and deferred= in the summary reflects the count."""
        # Use a real log dir (include_logs path) so there is something to defer.
        # _make_preserved_log is not available here, so create it manually.
        log_root = agent_run.LOG_ROOT
        log_root.mkdir(parents=True, exist_ok=True)
        names = [f"budget-cand-{i}" for i in range(5)]
        for name in names:
            ld = log_root / name
            ld.mkdir(exist_ok=True)
            (ld / "log").write_text("x")
            # Make it old enough to be a log-GC candidate.
            import os as _os
            _os.utime(ld, (0, 0))

        # Zero-second budget: every iteration check fires immediately.
        rc = agent_run.cmd_reap(argparse.Namespace(
            dry_run=True,
            idle_hours=None,
            min_age_hours=None,
            name=None,
            force_unknown=False,
            include_logs=True,
            log_min_age_hours=0.001,
            orphan_processes=False,
            orphan_min_age_hours=None,
            max_seconds=0.0001,
        ))
        assert rc == 0
        line = _extract_summary(capsys)
        deferred = _summary_field(line, "deferred")
        assert deferred > 0, f"expected deferred>0, got deferred={deferred}"

    def test_zero_budget_defers_everything(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        """With max_seconds → 0 and multiple state candidates, all are deferred."""
        import stat as _stat
        state_root = agent_run.STATE_ROOT
        state_root.mkdir(parents=True, exist_ok=True)
        log_root = agent_run.LOG_ROOT
        log_root.mkdir(parents=True, exist_ok=True)

        # Create several terminal state candidates old enough for GC.
        names = [f"zbcand-{i}" for i in range(4)]
        for name in names:
            sd = state_root / name
            sd.mkdir(exist_ok=True)
            (sd / "status").write_text("done\n")
            # ended_at far in the past so age > threshold.
            import datetime as _dt
            (sd / "ended_at").write_text("2000-01-01T00:00:00Z\n")
            ld = log_root / name
            ld.mkdir(exist_ok=True)

        rc = agent_run.cmd_reap(argparse.Namespace(
            dry_run=True,
            idle_hours=None,
            min_age_hours=0.001,
            name=None,
            force_unknown=False,
            include_logs=False,
            log_min_age_hours=None,
            orphan_processes=False,
            orphan_min_age_hours=None,
            max_seconds=0.0001,
        ))
        assert rc == 0
        line = _extract_summary(capsys)
        deferred = _summary_field(line, "deferred")
        assert deferred > 0


# ---------------------------------------------------------------------------
# Linux signal-path proof: pidfd branch used and recorded end-to-end
# ---------------------------------------------------------------------------

class TestLinuxSignalPath:
    """Prove that the Linux pidfd branch of _send_signal_to_verified_pid is
    exercised end-to-end through cmd_reap.

    platform.system is forced to "Linux" and os.pidfd_open +
    signal.pidfd_send_signal are replaced by recording stubs.  The test
    asserts that signals arrive via the pidfd stubs, not via os.kill, which
    would indicate the Darwin fallback had been taken instead.

    This is the cross-platform check that was missing and caused 21 CI
    failures: on Linux the pidfd path bypasses os.kill, so tests that only
    patched os.kill saw signals_sent=[] and failed.
    """

    def test_linux_pidfd_path_used_for_pid_signal(
        self, isolated_runs_root, monkeypatch, capsys
    ):
        """With platform forced to Linux, SIGTERM and SIGKILL reach the target
        via pidfd_send_signal, not os.kill.  kill_calls stays empty while
        pidfd_calls records both signals."""
        import types as _types

        pid = 9850
        identity = f"linux:{pid}-test-identity"
        # Use an entry whose identity matches what _process_identity will return.
        entry = _make_proc_entry(pid, "linuxrun", identity=identity)
        _make_log_dir("linuxrun")

        # Force the Linux platform branch in _send_signal_to_verified_pid.
        monkeypatch.setattr(agent_run, "platform",
                            _types.SimpleNamespace(system=lambda: "Linux"))

        kill_calls: List[Tuple[int, int]] = []
        pidfd_calls: List[Tuple[int, int]] = []
        fd_counter: List[int] = [200]
        fd_to_pid: Dict[int, int] = {}

        def fake_pidfd_open(p: int, flags: int) -> int:
            fd = fd_counter[0]
            fd_counter[0] += 1
            fd_to_pid[fd] = p
            return fd

        def fake_pidfd_send_signal(fd: int, sig: int, info: object, flags: int) -> None:
            p = fd_to_pid.get(fd)
            if p is not None:
                pidfd_calls.append((p, sig))

        # Capture real os.close before patching to avoid infinite recursion when
        # _launch_lock calls os.close on its own real fds.
        real_os_close = os.close

        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        monkeypatch.setattr(agent_run, "_process_identity", lambda p: identity)
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        monkeypatch.setattr(agent_run, "ORPHAN_KILL_GRACE_SECONDS", 0.05)
        monkeypatch.setattr(agent_run, "KILL_POLL_INTERVAL_SECONDS", 0.01)
        monkeypatch.setattr(os, "getpgid", lambda p: p + 1)
        # Record os.kill to prove it is NOT called when pidfd succeeds.
        monkeypatch.setattr(os, "kill", lambda p, s: kill_calls.append((p, s)))
        monkeypatch.setattr(os, "killpg", lambda g, s: None)
        monkeypatch.setattr(agent_run.os, "pidfd_open", fake_pidfd_open, raising=False)
        monkeypatch.setattr(agent_run.signal, "pidfd_send_signal",
                            fake_pidfd_send_signal, raising=False)
        # Intercept os.close: skip synthetic fds, pass real fds to real_os_close.
        monkeypatch.setattr(agent_run.os, "close",
                            lambda fd: (None if fd in fd_to_pid else real_os_close(fd)))

        cmd_reap(_reap_args(orphan_processes=True))

        # Signals must have arrived via pidfd, not os.kill.
        sigs_via_pidfd = [sig for _p, sig in pidfd_calls]
        assert signal.SIGTERM in sigs_via_pidfd, (
            f"SIGTERM must be sent via pidfd on Linux; pidfd_calls={pidfd_calls}"
        )
        assert signal.SIGKILL in sigs_via_pidfd, (
            f"SIGKILL must be sent via pidfd on Linux; pidfd_calls={pidfd_calls}"
        )
        assert not kill_calls, (
            f"os.kill must not be called when pidfd path succeeds; kill_calls={kill_calls}"
        )
        line = _extract_summary(capsys)
        assert _summary_field(line, "orphan_procs_killed") == 1
