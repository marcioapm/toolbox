"""Tests for orphan-process discovery helpers (Task 2).

All tests use synthetic _ProcEntry tables — no real processes are spawned
or killed.  The helpers under test:
  _argv_is_agent_run_runner
  _run_name_from_argv
  _find_orphan_runners
  _parse_orphan_min_age_seconds
  _scan_process_table (pid-vanishing race only)
  _darwin_lstart_normalize
"""
from __future__ import annotations

import os
import subprocess
import time
import types as _types
from datetime import datetime
from typing import List, Optional

import pytest

from toolbox import agent_run
from toolbox.agent_run import (
    _ProcEntry,
    _OrphanCandidate,
    _OrphanSkip,
    _argv_is_agent_run_runner,
    _darwin_lstart_normalize,
    _run_name_from_argv,
    _find_orphan_runners,
    _parse_orphan_min_age_seconds,
    _runner_state_root as _runner_state_root_real,
    cmd_reap,
)


# ---------------------------------------------------------------------------
# Module-level safety guards
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_real_signals(monkeypatch):
    """Any test reaching a real os.kill/os.killpg is a bug: fail loudly
    instead of terminating a live process."""
    monkeypatch.setattr(os, "kill", lambda *a, **k: pytest.fail(f"real os.kill{a}"))
    monkeypatch.setattr(os, "killpg", lambda *a, **k: pytest.fail(f"real os.killpg{a}"))


@pytest.fixture(autouse=True)
def _no_real_scan(monkeypatch):
    """Prevent accidental enumeration of the host process table.  Tests that
    exercise the scanner directly install their own subprocess stub."""
    monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [])


@pytest.fixture(autouse=True)
def _stub_runner_state_root(monkeypatch):
    """Return STATE_ROOT for every fabricated pid so tests are not platform-
    dependent.  On Linux, _runner_state_root reads /proc/<pid>/environ and
    returns None on OSError; fabricated pids (9001, 9002, …) do not exist, so
    every candidate would be skipped as state_root_unreadable on Linux CI.
    Tests that specifically exercise the state-root guard (TestStateRootGuard)
    install their own stub via monkeypatch, overriding this default."""
    monkeypatch.setattr(agent_run, "_runner_state_root",
                        lambda _pid: agent_run.STATE_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MY_UID = os.getuid()
FAR_PAST = time.time() - 100_000   # ~28 h ago — always older than any threshold


def _make_log_dir(name: str) -> None:
    """Create LOG_ROOT/<name> as a real directory so the log-dir corroboration
    check in _find_orphan_runners passes for tests that expect a candidate or
    a skip reason other than log_dir_missing."""
    agent_run.LOG_ROOT.mkdir(parents=True, exist_ok=True)
    (agent_run.LOG_ROOT / name).mkdir(exist_ok=True)


def _make_entry(
    pid: int,
    argv: List[str],
    *,
    ppid: int = 1,
    uid: int = MY_UID,
    start_time: float = FAR_PAST,
    identity: str = "",
    pgid: int = 0,
) -> _ProcEntry:
    ident = identity or f"darwin:{pid}-test-identity"
    return _ProcEntry(
        pid=pid, ppid=ppid, uid=uid,
        argv=argv, start_time=start_time,
        identity=ident, pgid=pgid,
    )


def _find(
    table: List[_ProcEntry],
    *,
    min_age_seconds: float = 3600.0,
    now: Optional[float] = None,
    self_pid: int = os.getpid(),
    self_pgid: int = os.getpgid(0),
    target_name: Optional[str] = None,
) -> tuple[List[_OrphanCandidate], List[_OrphanSkip]]:
    return _find_orphan_runners(
        table,
        min_age_seconds=min_age_seconds,
        now=now if now is not None else time.time(),
        self_pid=self_pid,
        self_pgid=self_pgid,
        target_name=target_name,
    )


# ---------------------------------------------------------------------------
# _argv_is_agent_run_runner — identification
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _argv_is_agent_run_runner — identification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("argv,expected", [
    (["agent-run", "myrun", "cmd"], True),
    (["/usr/local/bin/agent-run", "myrun", "cmd"], True),
    (["python3", "/usr/local/bin/agent-run", "myrun", "cmd"], True),
    (["python", "/opt/bin/agent-run", "myrun"], True),
    (["python3", "-u", "/path/agent-run", "myrun", "cmd"], True),
    (["/bin/bash", "-lc", "cat /var/tmp/agent-runs/myrun/log false"], False),
    (["grep", "agent-run", "/var/log/syslog"], False),
    (["vim", "/var/tmp/agent-runs/myrun/log"], False),
    ([], False),
    (["tail", "-f", "/var/tmp/agent-runs/something/log"], False),
    (["sh", "-c", "echo agent-run"], False),
    (["/usr/bin/agent-runner"], False),
    (["python3", "-u", "-O"], False),
])
def test_argv_is_agent_run_runner(argv, expected):
    assert _argv_is_agent_run_runner(argv) is expected


def test_bash_lc_cat_log_is_not_runner():
    """bash -lc 'cat /var/tmp/agent-runs/foo/log' must return False.

    This exact argv exists on production hosts and must never be treated
    as a runner — killing it would destroy unrelated work.
    """
    argv = ["bash", "-lc", "cat /var/tmp/agent-runs/foo/log"]
    assert _argv_is_agent_run_runner(argv) is False


# ---------------------------------------------------------------------------
# _run_name_from_argv — name recovery
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("argv,expected_name", [
    (["agent-run", "myrun", "claude"], "myrun"),
    (["agent-run", "-i", "myrun", "cmd"], "myrun"),
    (["agent-run", "-f", "/p.md", "myrun", "cmd"], "myrun"),
    (["agent-run", "--prompt-file=/p.md", "myrun", "cmd"], "myrun"),
    (["agent-run", "--echo", "myrun", "cmd"], "myrun"),
    (["agent-run", "--echo=2", "myrun", "cmd"], "myrun"),
    (["agent-run", "--submit-mode=cr", "myrun", "cmd"], "myrun"),
    (["agent-run", "--submit-mode=crlf", "myrun", "cmd"], "myrun"),
    (["agent-run", "--idle-timeout", "30", "myrun", "cmd"], "myrun"),
    (["agent-run", "myrun", "--", "opencode", "--print"], "myrun"),
    ([
        "agent-run", "-i", "-f", "/p.md", "--echo=3",
        "--submit-mode=crlf", "--idle-timeout", "60",
        "myrun", "--", "cmd",
    ], "myrun"),
    (["agent-run", "reap", "--dry-run"], None),
    (["agent-run", "list"], None),
    (["agent-run", "status", "myrun"], None),
    (["agent-run"], None),
    (["agent-run", "my/run", "cmd"], None),
    (["python3", "/usr/local/bin/agent-run", "myrun", "cmd"], "myrun"),
    (["bash", "-lc", "cat /var/tmp/agent-runs/foo/log"], None),
])
def test_run_name_from_argv(argv, expected_name):
    assert _run_name_from_argv(argv) == expected_name


# ---------------------------------------------------------------------------
# _find_orphan_runners — selection and skip reasons
# ---------------------------------------------------------------------------

class TestFindOrphanRunners:
    """Verify candidate selection and every skip-reason path."""

    def test_basic_candidate(self, isolated_runs_root):
        _make_log_dir("myrun")
        entry = _make_entry(9001, ["agent-run", "myrun", "claude"])
        candidates, skips = _find([entry])
        assert len(candidates) == 1
        assert candidates[0].pid == 9001
        assert candidates[0].name == "myrun"

    def test_state_dir_present_is_skipped(self, isolated_runs_root):
        # Create both the log dir (corroboration) and the state dir (the guard
        # being tested) so the entry reaches the state_dir_exists check.
        _make_log_dir("myrun")
        (isolated_runs_root / "myrun").mkdir(parents=True, exist_ok=True)
        entry = _make_entry(9002, ["agent-run", "myrun", "claude"])
        candidates, skips = _find([entry])
        assert not candidates
        assert any(s.reason == "state_dir_exists" for s in skips)

    def test_too_young_is_skipped(self, isolated_runs_root):
        # Log dir must exist so the entry reaches the age gate.
        _make_log_dir("youngrun")
        now = time.time()
        entry = _make_entry(9003, ["agent-run", "youngrun", "cmd"],
                            start_time=now - 60.0)  # 60 s old
        candidates, skips = _find([entry], min_age_seconds=3600.0, now=now)
        assert not candidates
        assert any(s.reason == "too_young" for s in skips)

    def test_old_enough_is_candidate(self, isolated_runs_root):
        _make_log_dir("oldrun")
        now = time.time()
        entry = _make_entry(9004, ["agent-run", "oldrun", "cmd"],
                            start_time=now - 7200.0)  # 2 h old
        candidates, skips = _find([entry], min_age_seconds=3600.0, now=now)
        assert len(candidates) == 1
        assert candidates[0].pid == 9004

    def test_self_pid_is_skipped(self, isolated_runs_root):
        self_pid = os.getpid()
        entry = _make_entry(self_pid, ["agent-run", "selfrun", "cmd"])
        candidates, skips = _find([entry], self_pid=self_pid)
        assert not candidates
        assert any(s.reason == "self" for s in skips)

    def test_ancestor_is_skipped(self, isolated_runs_root):
        # pid 9010 is the "current process"; pid 9005 is its parent.
        parent = _make_entry(9005, ["agent-run", "ancestorrun", "cmd"], ppid=1)
        child = _make_entry(9010, ["irrelevant"], ppid=9005)
        table = [parent, child]
        candidates, skips = _find(table, self_pid=9010)
        # 9005 is an ancestor of 9010; must be skipped, not a candidate.
        assert not any(c.pid == 9005 for c in candidates)
        assert any(s.pid == 9005 and s.reason == "ancestor" for s in skips)

    def test_same_pgid_is_skipped(self, isolated_runs_root):
        # Use a synthetic pgid that is not the real one so we don't collide
        # with any legitimately running test process.
        fake_pgid = 88888
        entry = _make_entry(9006, ["agent-run", "pgidrun", "cmd"], pgid=fake_pgid)
        candidates, skips = _find([entry], self_pgid=fake_pgid, self_pid=os.getpid() + 1000)
        assert any(s.reason == "same_pgid" for s in skips)

    def test_same_pgid_sibling_is_skipped(self, isolated_runs_root):
        """A sibling runner sharing the reaper's pgid but with a different ppid
        must be skipped.  This is the real-world shape: on production hosts
        each run is pid=child ppid=runner pgid=<shared group leader>, so
        ppid != self_pgid and pid != self_pgid but pgid == self_pgid."""
        fake_pgid = 49440
        # ppid is the runner parent, not the reaper — the old ppid-based
        # check would have missed this entry.
        entry = _make_entry(
            9006, ["agent-run", "sibling", "cmd"],
            ppid=49441, pgid=fake_pgid,
        )
        candidates, skips = _find([entry], self_pgid=fake_pgid, self_pid=70426)
        assert not candidates
        assert any(s.pid == 9006 and s.reason == "same_pgid" for s in skips)

    def test_pid_1_is_skipped(self, isolated_runs_root):
        entry = _make_entry(1, ["agent-run", "initrun", "cmd"])
        candidates, skips = _find([entry])
        assert not candidates
        assert any(s.reason == "pid_1" for s in skips)

    def test_foreign_uid_is_skipped(self, isolated_runs_root):
        other_uid = MY_UID + 1
        entry = _make_entry(9007, ["agent-run", "foreignrun", "cmd"], uid=other_uid)
        candidates, skips = _find([entry])
        assert not candidates
        assert any(s.reason == "foreign_uid" for s in skips)

    def test_unrecoverable_name_is_skipped(self, isolated_runs_root):
        # bash -lc "..." looks like a runner if we were doing substring matching
        # but _argv_is_agent_run_runner returns False, so it won't even be a skip.
        # Instead test a process that IS a runner but whose name can't be parsed.
        entry = _make_entry(9008, ["agent-run"])  # no name or command
        candidates, skips = _find([entry])
        assert not candidates
        # name_unrecoverable or skipped silently (no runner match at all)

    def test_target_name_filters_others(self, isolated_runs_root):
        # run-a becomes a candidate; run-b gets name_mismatch before log-dir check.
        _make_log_dir("run-a")
        e1 = _make_entry(9020, ["agent-run", "run-a", "cmd"])
        e2 = _make_entry(9021, ["agent-run", "run-b", "cmd"])
        candidates, skips = _find([e1, e2], target_name="run-a")
        assert len(candidates) == 1
        assert candidates[0].name == "run-a"
        assert any(s.reason == "name_mismatch" for s in skips)

    def test_non_runner_argv_ignored_entirely(self, isolated_runs_root):
        # grep agent-run is not a runner; it should not appear in skips either.
        entry = _make_entry(9030, ["grep", "agent-run", "/var/log/syslog"])
        candidates, skips = _find([entry])
        assert not candidates
        assert not any(s.pid == 9030 for s in skips)

    def test_bash_lc_cat_log_never_candidate(self, isolated_runs_root):
        """The bash -lc 'cat /var/tmp/agent-runs/foo/log' false positive
        must never appear as a candidate — it is the highest-priority safety
        invariant in this module."""
        entry = _make_entry(9031, ["bash", "-lc", "cat /var/tmp/agent-runs/foo/log"])
        candidates, skips = _find([entry])
        assert not candidates
        assert not any(s.pid == 9031 for s in skips)

    def test_candidates_carry_identity(self, isolated_runs_root):
        _make_log_dir("idrun")
        entry = _make_entry(9040, ["agent-run", "idrun", "cmd"],
                            identity="darwin:Wed Jan 01 00:00:00 2025")
        candidates, _ = _find([entry])
        assert candidates[0].identity == "darwin:Wed Jan 01 00:00:00 2025"

    def test_multiple_candidates_all_returned(self, isolated_runs_root):
        entries = [
            _make_entry(9050 + i, ["agent-run", f"run{i}", "cmd"])
            for i in range(5)
        ]
        for i in range(5):
            _make_log_dir(f"run{i}")
        candidates, _ = _find(entries)
        assert len(candidates) == 5


# ---------------------------------------------------------------------------
# _parse_orphan_min_age_seconds — threshold parsing
# ---------------------------------------------------------------------------

class TestParseOrphanMinAgeSeconds:
    def test_default_is_24h(self, monkeypatch):
        monkeypatch.delenv("AGENT_RUN_ORPHAN_MIN_AGE_HOURS", raising=False)
        assert _parse_orphan_min_age_seconds() == pytest.approx(24 * 3600)

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("AGENT_RUN_ORPHAN_MIN_AGE_HOURS", "48")
        assert _parse_orphan_min_age_seconds() == pytest.approx(48 * 3600)

    def test_invalid_env_falls_back_to_default(self, monkeypatch, capsys):
        monkeypatch.setenv("AGENT_RUN_ORPHAN_MIN_AGE_HOURS", "notanumber")
        result = _parse_orphan_min_age_seconds()
        assert result == pytest.approx(24 * 3600)
        assert "AGENT_RUN_ORPHAN_MIN_AGE_HOURS" in capsys.readouterr().err

    def test_independent_of_reap_min_age(self, monkeypatch):
        """Changing AGENT_RUN_MIN_AGE_HOURS must not affect the orphan threshold."""
        monkeypatch.setenv("AGENT_RUN_MIN_AGE_HOURS", "1")
        monkeypatch.delenv("AGENT_RUN_ORPHAN_MIN_AGE_HOURS", raising=False)
        assert _parse_orphan_min_age_seconds() == pytest.approx(24 * 3600)

    def test_independent_of_log_min_age(self, monkeypatch):
        """Changing AGENT_RUN_LOG_MIN_AGE_HOURS must not affect the orphan threshold."""
        monkeypatch.setenv("AGENT_RUN_LOG_MIN_AGE_HOURS", "1")
        monkeypatch.delenv("AGENT_RUN_ORPHAN_MIN_AGE_HOURS", raising=False)
        assert _parse_orphan_min_age_seconds() == pytest.approx(24 * 3600)

    def test_zero_invalid_falls_back_to_default(self, monkeypatch, capsys):
        monkeypatch.setenv("AGENT_RUN_ORPHAN_MIN_AGE_HOURS", "0")
        result = _parse_orphan_min_age_seconds()
        assert result == pytest.approx(24 * 3600)
        assert "AGENT_RUN_ORPHAN_MIN_AGE_HOURS" in capsys.readouterr().err

    def test_negative_invalid_falls_back_to_default(self, monkeypatch, capsys):
        monkeypatch.setenv("AGENT_RUN_ORPHAN_MIN_AGE_HOURS", "-5")
        result = _parse_orphan_min_age_seconds()
        assert result == pytest.approx(24 * 3600)


# ---------------------------------------------------------------------------
# _scan_process_table — vanishing-pid race tolerance
# ---------------------------------------------------------------------------

class TestScanProcessTableRace:
    def test_darwin_scanner_tolerates_subprocess_failure(self, monkeypatch):
        """_scan_process_table_darwin returns [] when ps fails rather than raising."""
        import subprocess as _subprocess

        def patched_run(*args, **kwargs):
            raise _subprocess.SubprocessError("simulated failure")

        monkeypatch.setattr(_subprocess, "run", patched_run)
        result = agent_run._scan_process_table_darwin()
        assert result == []

    def test_linux_ghost_pid_skipped(self, monkeypatch):
        """A pid listed in /proc that vanishes before its files are read is skipped.

        Runs on all platforms: the Linux scanner is called directly with os.listdir
        stubbed to inject a ghost pid.  On Linux, any real /proc entries are also
        returned; on non-Linux platforms the real os.listdir is replaced entirely.
        """
        import types as _types

        original_listdir = os.listdir

        def fake_listdir(path):
            if str(path) == "/proc":
                # On Linux, include real pids so the scanner still returns entries;
                # on other platforms /proc doesn't exist so we synthesise the list.
                try:
                    real = original_listdir(path)
                except OSError:
                    real = []
                return list(real) + ["999999999"]   # ghost pid
            return original_listdir(path)

        monkeypatch.setattr(os, "listdir", fake_listdir)
        # Should not raise even though /proc/999999999/* do not exist.
        entries = agent_run._scan_process_table_linux()
        assert isinstance(entries, list)
        assert not any(e.pid == 999999999 for e in entries)


# ---------------------------------------------------------------------------
# _darwin_lstart_normalize — whitespace canonicalisation
# ---------------------------------------------------------------------------

class TestDarwinLstartNormalise:
    """Regression for the single-digit-day whitespace mismatch.

    ``ps -axo lstart=`` collapses the fixed-width kernel date format to a
    single space between tokens, while ``ps -p PID -o lstart=`` preserves the
    right-padding used for days 1-9 (e.g. ``Mon Jun  8`` — two spaces).
    Both call sites now run through ``_darwin_lstart_normalize`` so the
    identity tokens they produce compare equal.
    """

    def test_single_digit_day_with_double_space_normalised(self):
        """The ps -p form (double-space before single-digit day) must equal the
        ps -axo form (single space) after normalisation."""
        ps_p_form = "Mon Jun  8 12:29:39 2026"   # right-padded, from ps -p
        ps_axo_form = "Mon Jun 8 12:29:39 2026"   # collapsed, from ps -axo split
        assert _darwin_lstart_normalize(ps_p_form) == _darwin_lstart_normalize(ps_axo_form)

    def test_double_digit_day_unchanged(self):
        """Days 10-31 have no extra padding; normalisation must be a no-op."""
        lstart = "Mon Jun 12 14:00:00 2026"
        assert _darwin_lstart_normalize(lstart) == lstart

    def test_already_normalised_idempotent(self):
        """Calling normalise on already-normalised output is idempotent."""
        lstart = "Mon Jun 8 12:29:39 2026"
        assert _darwin_lstart_normalize(_darwin_lstart_normalize(lstart)) == lstart

    def test_identity_token_equality_across_ps_forms(self, monkeypatch):
        """Simulate the two ps call paths and assert identity tokens compare equal.

        This is the end-to-end regression: the token stored by
        _scan_process_table_darwin (built from the ps -axo column) must equal
        the token returned by _process_identity (built from ps -p lstart=) for
        the same process with a single-digit start day.

        Runs on all platforms via a platform.system stub — no Darwin dependency.
        """
        import subprocess as _subprocess
        import types as _types

        pid = 12345
        my_uid = os.getuid()

        # ps -axo output with column order: pid ppid pgid uid lstart command.
        # Whitespace-collapsed by split(None, N) parsing.
        ps_axo_line = (
            f"  {pid}     1  {pid} {my_uid} Mon Jun  8 12:29:39 2026"
            f" /usr/bin/agent-run myrun claude"
        )

        # ps -p output: raw fixed-width kernel format with two-space padding.
        ps_p_output = "Mon Jun  8 12:29:39 2026"

        def fake_subprocess_run(cmd, **kwargs):
            class R:
                stdout = ""
                returncode = 0
            r = R()
            if cmd[:2] == ["ps", "-axo"]:
                r.stdout = ps_axo_line + "\n"
            elif cmd[:2] == ["ps", "-p"] and "lstart=" in " ".join(cmd):
                r.stdout = ps_p_output + "\n"
            return r

        monkeypatch.setattr(_subprocess, "run", fake_subprocess_run)
        # Redirect agent_run's platform reference so the Darwin branch runs
        # on Linux CI as well.
        monkeypatch.setattr(agent_run, "platform", _types.SimpleNamespace(system=lambda: "Darwin"))

        # Identity from the scanner (ps -axo path).
        entries = agent_run._scan_process_table_darwin()
        assert entries, "scanner returned no entries from fake ps"
        scanner_identity = entries[0].identity

        # Identity from _process_identity (ps -p path).
        verify_identity = agent_run._process_identity(pid)

        assert scanner_identity == verify_identity, (
            f"scanner identity {scanner_identity!r} != "
            f"verify identity {verify_identity!r}"
        )


# ---------------------------------------------------------------------------
# _scan_process_table_darwin — start_time correctness (C1: TZ, C2: locale)
# ---------------------------------------------------------------------------

def _fake_ps_run_for_lstart(lstart: str, pid: int = 4242):
    """Return a fake subprocess.run stub that emits one ps -axo line with
    the given lstart string.  Used to test start_time parsing in isolation."""
    my_uid = os.getuid()

    def fake_run(cmd, **kwargs):
        class R:
            stdout = ""
            returncode = 0
        r = R()
        if cmd[:2] == ["ps", "-axo"]:
            # Column order: pid ppid pgid uid lstart command
            r.stdout = (
                f"  {pid}     1  {pid} {my_uid} {lstart}"
                f" /usr/bin/agent-run testrun claude\n"
            )
        return r

    return fake_run


class TestDarwinStartTimeParsing:
    """Verify that _scan_process_table_darwin converts lstart to the correct
    epoch, independent of the host's TZ and locale (C1, C2).

    All tests run on every platform via a subprocess.run stub — no Darwin
    dependency.  The TZ environment variable and time.tzset() are used to
    change the interpreter's local timezone so that datetime.timestamp()
    computes the right epoch for each case.
    """

    def _parse_lstart_in_tz(self, lstart: str, tz: str, monkeypatch) -> Optional[float]:
        """Run the Darwin scanner with a fake ps line and the given TZ,
        return the start_time from the first entry, or None if empty."""
        old_tz = os.environ.get("TZ")
        monkeypatch.setenv("TZ", tz)
        time.tzset()
        try:
            monkeypatch.setattr(subprocess, "run", _fake_ps_run_for_lstart(lstart))
            monkeypatch.setattr(
                agent_run, "platform",
                _types.SimpleNamespace(system=lambda: "Darwin"),
            )
            entries = agent_run._scan_process_table_darwin()
            return entries[0].start_time if entries else None
        finally:
            if old_tz is None:
                monkeypatch.delenv("TZ", raising=False)
            else:
                monkeypatch.setenv("TZ", old_tz)
            time.tzset()

    @pytest.mark.parametrize("tz,lstart,expected_utc_hour", [
        # UTC: local == UTC, so 10:00 local == 10:00 UTC.
        ("UTC", "Mon Aug 10 10:00:00 2026", 10),
        # America/Los_Angeles is PDT (UTC-7) in August.
        # 03:00 local == 10:00 UTC.
        ("America/Los_Angeles", "Mon Aug 10 03:00:00 2026", 10),
        # Asia/Tokyo is JST (UTC+9) year-round.
        # 19:00 local == 10:00 UTC.
        ("Asia/Tokyo", "Mon Aug 10 19:00:00 2026", 10),
    ])
    def test_lstart_parsed_as_local_time(
        self, tz, lstart, expected_utc_hour, monkeypatch
    ):
        """lstart is wall-clock local time; the parsed epoch must equal the
        corresponding UTC instant regardless of the host timezone."""
        # Reference UTC epoch: 2026-08-10 10:00:00 UTC
        reference_utc = datetime(2026, 8, 10, 10, 0, 0)
        # Compute expected epoch in UTC (force UTC for this calculation).
        old_tz = os.environ.get("TZ")
        monkeypatch.setenv("TZ", "UTC")
        time.tzset()
        expected_epoch = reference_utc.timestamp()
        # Restore before calling _parse_lstart_in_tz.
        if old_tz is None:
            monkeypatch.delenv("TZ", raising=False)
        else:
            monkeypatch.setenv("TZ", old_tz)
        time.tzset()

        result = self._parse_lstart_in_tz(lstart, tz, monkeypatch)
        assert result is not None, f"scanner returned no entry for TZ={tz}"
        assert result == pytest.approx(expected_epoch, abs=2.0), (
            f"TZ={tz}: expected epoch {expected_epoch} (UTC 10:00), got {result}"
        )

    def test_dst_boundary_spring_forward_america_los_angeles(self, monkeypatch):
        """Across a DST transition the epoch must still be correct.

        2026-03-08 02:00 clocks spring forward to 03:00 in America/Los_Angeles,
        so 03:30 local that day is PDT (UTC-7), not PST (UTC-8).
        03:30 PDT == 10:30 UTC.
        """
        lstart = "Sun Mar  8 03:30:00 2026"
        result = self._parse_lstart_in_tz(lstart, "America/Los_Angeles", monkeypatch)
        assert result is not None

        # Compute the expected epoch: 2026-03-08 10:30:00 UTC.
        old_tz = os.environ.get("TZ")
        monkeypatch.setenv("TZ", "UTC")
        time.tzset()
        expected_epoch = datetime(2026, 3, 8, 10, 30, 0).timestamp()
        if old_tz is None:
            monkeypatch.delenv("TZ", raising=False)
        else:
            monkeypatch.setenv("TZ", old_tz)
        time.tzset()

        assert result == pytest.approx(expected_epoch, abs=2.0), (
            f"DST boundary: expected {expected_epoch} (2026-03-08 10:30 UTC), "
            f"got {result}"
        )


class TestDarwinLocaleHandling:
    """Verify that a non-C locale in ps output produces start_time=None and
    that _find_orphan_runners emits start_time_unknown (fail-closed) (C2)."""

    def test_unparsable_lstart_yields_start_time_none(self, monkeypatch):
        """A localised lstart (e.g. French 'ven. 14 août 23:00:18 2026')
        must produce start_time=None, not 0.0 (the epoch)."""
        # A localised lstart that strptime cannot parse with "%a %b %d %H:%M:%S %Y".
        localised_lstart = "ven. 14 aout 23:00:18 2026"
        monkeypatch.setattr(subprocess, "run",
                            _fake_ps_run_for_lstart(localised_lstart))
        monkeypatch.setattr(
            agent_run, "platform",
            _types.SimpleNamespace(system=lambda: "Darwin"),
        )
        entries = agent_run._scan_process_table_darwin()
        assert entries, "scanner returned no entries"
        assert entries[0].start_time is None, (
            f"expected None for unparsable lstart, got {entries[0].start_time!r}"
        )

    def test_start_time_none_skipped_not_candidate(self, isolated_runs_root):
        """An entry with start_time=None must be emitted as start_time_unknown,
        never as a candidate — an unknown age cannot satisfy any threshold."""
        _make_log_dir("localerun")
        entry = _ProcEntry(
            pid=8800, ppid=1, uid=MY_UID,
            argv=["agent-run", "localerun", "claude"],
            start_time=None,
            identity="darwin:8800-test",
            pgid=0,
        )
        candidates, skips = _find([entry])
        assert not candidates, "start_time=None entry must never be a candidate"
        assert any(s.pid == 8800 and s.reason == "start_time_unknown" for s in skips), (
            f"expected start_time_unknown skip, got: {skips}"
        )

    def test_start_time_none_never_satisfies_any_threshold(self, isolated_runs_root):
        """Even with min_age_seconds=0, start_time=None is not a candidate."""
        _make_log_dir("localerun2")
        entry = _ProcEntry(
            pid=8801, ppid=1, uid=MY_UID,
            argv=["agent-run", "localerun2", "claude"],
            start_time=None,
            identity="darwin:8801-test",
            pgid=0,
        )
        candidates, skips = _find([entry], min_age_seconds=0.0)
        assert not candidates
        assert any(s.reason == "start_time_unknown" for s in skips)


# ---------------------------------------------------------------------------
# C3: log-dir corroboration and argv ambiguity detection
# ---------------------------------------------------------------------------

class TestArgvCorroboration:
    """Verify the log-dir existence check and argv-ambiguity guard (C3/S1)."""

    def test_spaced_path_misparse_realrun_not_candidate(self, isolated_runs_root):
        """The space-in-path scenario from S1:

        True argv: ['agent-run', '-f', '/tmp/my prompt.md', 'realrun', '--', 'claude']
        ps output: 'agent-run -f /tmp/my prompt.md realrun -- claude'
        shlex recovers name='prompt.md' (wrong); plain split also gets 'prompt.md'.

        Neither 'realrun' nor 'prompt.md' should be a candidate:
        - 'realrun' is never seen by the scanner (the recovered name is 'prompt.md').
        - 'prompt.md' has no LOG_ROOT entry → log_dir_missing skip.
        """
        # Create the log and state dirs for 'realrun' (the real, tracked run).
        _make_log_dir("realrun")
        (isolated_runs_root / "realrun").mkdir(parents=True, exist_ok=True)

        # Entry as the scanner would produce it after shlex-splitting the ps output.
        # ps joins with spaces: 'agent-run -f /tmp/my prompt.md realrun -- claude'
        # shlex.split produces: ['agent-run','-f','/tmp/my','prompt.md','realrun','--','claude']
        misparse_argv = ["agent-run", "-f", "/tmp/my", "prompt.md", "realrun", "--", "claude"]
        entry = _make_entry(7777, misparse_argv, start_time=FAR_PAST)
        candidates, skips = _find([entry])

        assert not candidates, (
            f"mis-parsed entry must not be a candidate; candidates={candidates}"
        )
        # The recovered name is 'prompt.md'; it should be skipped for log_dir_missing.
        assert any(s.pid == 7777 and s.reason == "log_dir_missing" for s in skips), (
            f"expected log_dir_missing for 'prompt.md', got: {skips}"
        )
        # 'realrun' must not appear anywhere — it was never the recovered name.
        assert not any(
            hasattr(x, "name") and getattr(x, "name", None) == "realrun"
            for x in candidates + skips
        )

    def test_log_dir_missing_skips_candidate(self, isolated_runs_root):
        """A well-formed entry whose LOG_ROOT/<name> does not exist is skipped."""
        # Do NOT create LOG_ROOT/missinglog.
        entry = _make_entry(7778, ["agent-run", "missinglog", "claude"])
        candidates, skips = _find([entry])
        assert not candidates
        assert any(s.pid == 7778 and s.reason == "log_dir_missing" for s in skips)

    def test_log_dir_symlink_skips_candidate(self, isolated_runs_root):
        """A LOG_ROOT/<name> that is a symlink (not a real directory) is rejected.

        A real runner always creates a plain directory; a symlink is not a
        trustworthy corroboration because it can point anywhere."""
        outside = isolated_runs_root.parent / "outside-dir"
        outside.mkdir(exist_ok=True)
        symlink_log = agent_run.LOG_ROOT / "symlinkrun"
        agent_run.LOG_ROOT.mkdir(parents=True, exist_ok=True)
        symlink_log.symlink_to(outside, target_is_directory=True)

        entry = _make_entry(7779, ["agent-run", "symlinkrun", "claude"])
        candidates, skips = _find([entry])
        assert not candidates
        assert any(s.pid == 7779 and s.reason == "log_dir_missing" for s in skips), (
            f"symlink should produce log_dir_missing, got: {skips}"
        )

    def test_argv_ambiguous_when_shlex_and_plain_split_disagree(self, isolated_runs_root):
        """When re-joining entry.argv and re-splitting with str.split() yields a
        different name than entry.argv itself, emit argv_ambiguous and skip.

        This covers the case where a pre-name flag value carries an embedded
        space inside an argv token — the re-join loses the token boundary and
        plain split misidentifies the name.

        argv = ['agent-run', '-f', '/tmp/my file.md', 'realrun', 'claude']
        shlex name (from entry.argv directly) = 'realrun'
        re-join = 'agent-run -f /tmp/my file.md realrun claude'
        plain.split() = ['agent-run','-f','/tmp/my','file.md','realrun','claude']
        plain name = 'file.md'   (file.md != realrun → argv_ambiguous)
        """
        # 'realrun' and 'file.md' would both need log dirs if either were to
        # become a candidate, but the ambiguity check fires before the log-dir
        # check, so neither log dir is needed here.
        ambiguous_argv = ["agent-run", "-f", "/tmp/my file.md", "realrun", "claude"]
        entry = _make_entry(7780, ambiguous_argv)
        candidates, skips = _find([entry])
        assert not candidates
        assert any(s.pid == 7780 and s.reason == "argv_ambiguous" for s in skips), (
            f"expected argv_ambiguous, got: {skips}"
        )

    def test_log_dir_present_allows_candidate(self, isolated_runs_root):
        """A well-formed entry with an existing LOG_ROOT/<name> real directory
        passes the corroboration check and becomes a candidate."""
        _make_log_dir("corrobrun")
        entry = _make_entry(7790, ["agent-run", "corrobrun", "claude"])
        candidates, skips = _find([entry])
        assert any(c.pid == 7790 and c.name == "corrobrun" for c in candidates), (
            f"entry with valid log dir should be a candidate; got candidates={candidates}"
        )

    def test_spaced_path_end_to_end_through_cmd_reap(
        self, isolated_runs_root, monkeypatch, capsys
    ):
        """End-to-end: the S1 mis-parse scenario kills no process through cmd_reap.

        'realrun' has a state dir (tracked) and a log dir. 'prompt.md' has
        neither.  The scanner returns the mis-parsed argv.  cmd_reap must not
        signal any process and must not produce a candidate for 'realrun' or
        'prompt.md'.
        """
        import argparse

        _make_log_dir("realrun2")
        (isolated_runs_root / "realrun2").mkdir(parents=True, exist_ok=True)

        # Mis-parsed argv as the scanner would produce it.
        misparse_argv = ["agent-run", "-f", "/tmp/my", "prompt.md", "realrun2", "--", "claude"]
        entry = _ProcEntry(
            pid=7800, ppid=1, uid=MY_UID,
            argv=misparse_argv,
            start_time=FAR_PAST,
            identity="darwin:7800-test",
            pgid=0,
        )

        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        monkeypatch.setattr(agent_run, "_process_identity",
                            lambda p: f"darwin:{p}-test")
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)

        signals_sent = []
        monkeypatch.setattr(os, "kill", lambda *a: signals_sent.append(a))
        monkeypatch.setattr(os, "killpg", lambda *a: signals_sent.append(a))

        reap_args = argparse.Namespace(
            dry_run=False, idle_hours=None, min_age_hours=None,
            name=None, force_unknown=False, include_logs=False,
            log_min_age_hours=None, orphan_processes=True,
            orphan_min_age_hours=None,
        )
        agent_run.cmd_reap(reap_args)

        assert not signals_sent, (
            f"cmd_reap must not signal any process in S1 scenario; sent={signals_sent}"
        )
        out = capsys.readouterr().out
        assert "realrun2" not in out or "state_dir_exists" in out, (
            "realrun2 must not appear as a candidate"
        )


# ---------------------------------------------------------------------------
# S3: state-root guard — runner with foreign AGENT_RUN_STATE_DIR
# ---------------------------------------------------------------------------

class TestStateRootGuard:
    """A runner started with a non-default AGENT_RUN_STATE_DIR must be
    skipped even when its name has no state dir under the reaper's root.

    Verified via _runner_state_root (Linux path mocked; Darwin returns
    STATE_ROOT unconditionally so the check is a no-op there)."""

    def test_foreign_state_root_skipped(self, isolated_runs_root, monkeypatch):
        """On Linux, a process whose AGENT_RUN_STATE_DIR differs from the
        reaper's STATE_ROOT is skipped with reason foreign_state_root."""
        import platform as _platform

        entry = _make_entry(
            pid=7900,
            argv=["agent-run", "foreignrun", "claude"],
            start_time=FAR_PAST,
        )
        _make_log_dir("foreignrun")

        # Simulate Linux: _runner_state_root reads /proc/<pid>/environ and
        # finds a different root.
        other_root = isolated_runs_root.parent / "other-state"
        monkeypatch.setattr(agent_run, "_runner_state_root",
                            lambda _pid: other_root)

        candidates, skips = _find_orphan_runners(
            [entry],
            min_age_seconds=0,
            now=time.time(),
            self_pid=os.getpid() + 1000,
            self_pgid=os.getpgid(0) + 1000,
        )

        assert not candidates
        assert any(s.reason == "foreign_state_root" for s in skips), (
            f"expected foreign_state_root skip, got: {[s.reason for s in skips]}"
        )

    def test_unreadable_env_skipped(self, isolated_runs_root, monkeypatch):
        """If the runner's environment cannot be read, skip the candidate
        (fail-closed: unknown root must not be treated as matching)."""
        entry = _make_entry(
            pid=7901,
            argv=["agent-run", "envfailrun", "claude"],
            start_time=FAR_PAST,
        )
        _make_log_dir("envfailrun")

        # _runner_state_root returns None → unreadable env → skip.
        monkeypatch.setattr(agent_run, "_runner_state_root", lambda _pid: None)

        candidates, skips = _find_orphan_runners(
            [entry],
            min_age_seconds=0,
            now=time.time(),
            self_pid=os.getpid() + 1000,
            self_pgid=os.getpgid(0) + 1000,
        )

        assert not candidates
        assert any(s.reason == "state_root_unreadable" for s in skips), (
            f"expected state_root_unreadable skip, got: {[s.reason for s in skips]}"
        )

    def test_matching_state_root_is_candidate(self, isolated_runs_root, monkeypatch):
        """A process whose recovered state root matches STATE_ROOT proceeds
        to become a candidate (if all other checks pass)."""
        entry = _make_entry(
            pid=7902,
            argv=["agent-run", "samerootrun", "claude"],
            start_time=FAR_PAST,
        )
        _make_log_dir("samerootrun")

        # _runner_state_root returns the same path as agent_run.STATE_ROOT.
        monkeypatch.setattr(agent_run, "_runner_state_root",
                            lambda _pid: agent_run.STATE_ROOT)

        candidates, skips = _find_orphan_runners(
            [entry],
            min_age_seconds=0,
            now=time.time(),
            self_pid=os.getpid() + 1000,
            self_pgid=os.getpgid(0) + 1000,
        )

        assert len(candidates) == 1
        assert candidates[0].name == "samerootrun"


# ---------------------------------------------------------------------------
# Linux path proof: _runner_state_root with forced Linux branch
# ---------------------------------------------------------------------------

class TestRunnerStateRootLinuxBranch:
    """Directly exercise the Linux branch of _runner_state_root with the
    platform check forced.  These tests run on every platform via a
    platform.system stub so Linux CI behaviour is verifiable on macOS.

    Tests call _runner_state_root_real (the function object imported before any
    monkeypatching) so the _stub_runner_state_root autouse fixture — which
    replaces the module attribute — does not interfere.  monkeypatch controls
    both the platform branch and the /proc read.
    """

    def test_linux_branch_returns_state_root_when_var_absent(
        self, isolated_runs_root, monkeypatch
    ):
        """When AGENT_RUN_STATE_DIR is absent from the process environ, the
        Linux branch returns STATE_ROOT (the compiled-in default)."""
        import pathlib
        import types as _types

        monkeypatch.setattr(agent_run, "platform",
                            _types.SimpleNamespace(system=lambda: "Linux"))
        # NUL-separated environ with no AGENT_RUN_STATE_DIR key.
        fake_environ = b"PATH=/usr/bin\x00HOME=/root\x00"
        monkeypatch.setattr(pathlib.Path, "read_bytes", lambda _self: fake_environ)

        result = _runner_state_root_real(99999)
        assert result == agent_run.STATE_ROOT, (
            f"expected STATE_ROOT when AGENT_RUN_STATE_DIR absent, got {result!r}"
        )

    def test_linux_branch_returns_custom_root_when_var_present(
        self, isolated_runs_root, monkeypatch
    ):
        """When AGENT_RUN_STATE_DIR is set in the process environ, the Linux
        branch returns that path rather than STATE_ROOT."""
        import pathlib
        from pathlib import Path
        import types as _types

        custom_root = "/custom/state/root"
        monkeypatch.setattr(agent_run, "platform",
                            _types.SimpleNamespace(system=lambda: "Linux"))
        fake_environ = (
            b"PATH=/usr/bin\x00"
            + b"AGENT_RUN_STATE_DIR=" + custom_root.encode() + b"\x00"
            + b"HOME=/root\x00"
        )
        monkeypatch.setattr(pathlib.Path, "read_bytes", lambda _self: fake_environ)

        result = _runner_state_root_real(99999)
        assert result == Path(custom_root), (
            f"expected custom root {custom_root!r}, got {result!r}"
        )

    def test_linux_branch_returns_none_on_oserror(self, monkeypatch):
        """An OSError reading /proc/<pid>/environ causes the Linux branch to
        return None (fail-closed: unreadable env → skip the candidate)."""
        import pathlib
        import types as _types

        monkeypatch.setattr(agent_run, "platform",
                            _types.SimpleNamespace(system=lambda: "Linux"))

        def _raise_oserror(self):
            raise OSError("No such file or directory")

        monkeypatch.setattr(pathlib.Path, "read_bytes", _raise_oserror)

        result = _runner_state_root_real(99999)
        assert result is None, (
            f"expected None on OSError (fail-closed), got {result!r}"
        )

    def test_state_root_unreadable_printed_not_swallowed(
        self, isolated_runs_root, monkeypatch, capsys
    ):
        """When _runner_state_root returns None, _find_orphan_runners emits
        state_root_unreadable and cmd_reap prints it — the skip is visible
        to the operator, not silently absorbed."""
        import argparse

        entry = _make_entry(
            pid=7910,
            argv=["agent-run", "unreadrun", "claude"],
            start_time=FAR_PAST,
        )
        _make_log_dir("unreadrun")

        # Override the autouse stub: None simulates an unreadable /proc entry.
        monkeypatch.setattr(agent_run, "_runner_state_root", lambda _pid: None)
        monkeypatch.setattr(agent_run, "_scan_process_table", lambda: [entry])
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        monkeypatch.setattr(agent_run, "_process_identity",
                            lambda p: f"darwin:{p}-test-identity")

        agent_run.cmd_reap(argparse.Namespace(
            dry_run=False, idle_hours=None, min_age_hours=None,
            name=None, force_unknown=False, include_logs=False,
            log_min_age_hours=None, orphan_processes=True,
            orphan_min_age_hours=None,
        ))

        out = capsys.readouterr().out
        assert "state_root_unreadable" in out, (
            f"state_root_unreadable must appear in reap output; got:\n{out}"
        )
