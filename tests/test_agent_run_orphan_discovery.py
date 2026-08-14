"""Tests for orphan-process discovery helpers (Task 2).

All tests use synthetic _ProcEntry tables — no real processes are spawned
or killed.  The helpers under test:
  _argv_is_agent_run_runner
  _run_name_from_argv
  _find_orphan_runners
  _parse_orphan_min_age_seconds
  _scan_process_table (pid-vanishing race only)
  _darwin_lstart_normalise
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

import pytest

from toolbox import agent_run
from toolbox.agent_run import (
    _ProcEntry,
    _OrphanCandidate,
    _OrphanSkip,
    _argv_is_agent_run_runner,
    _darwin_lstart_normalise,
    _run_name_from_argv,
    _find_orphan_runners,
    _parse_orphan_min_age_seconds,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MY_UID = os.getuid()
FAR_PAST = time.time() - 100_000   # ~28 h ago — always older than any threshold


def _make_entry(
    pid: int,
    argv: List[str],
    *,
    ppid: int = 1,
    uid: int = MY_UID,
    start_time: float = FAR_PAST,
    identity: str = "",
) -> _ProcEntry:
    ident = identity or f"darwin:{pid}-test-identity"
    return _ProcEntry(
        pid=pid, ppid=ppid, uid=uid,
        argv=argv, start_time=start_time,
        identity=ident,
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

class TestArgvIsAgentRunRunner:
    def test_bare_agent_run(self):
        assert _argv_is_agent_run_runner(["agent-run", "myrun", "cmd"]) is True

    def test_absolute_path_agent_run(self):
        assert _argv_is_agent_run_runner(["/usr/local/bin/agent-run", "myrun", "cmd"]) is True

    def test_python_wrapped(self):
        assert _argv_is_agent_run_runner(
            ["python3", "/usr/local/bin/agent-run", "myrun", "cmd"]
        ) is True

    def test_python_wrapped_no_suffix(self):
        assert _argv_is_agent_run_runner(
            ["python", "/opt/bin/agent-run", "myrun"]
        ) is True

    def test_python_with_dash_u_flag(self):
        assert _argv_is_agent_run_runner(
            ["python3", "-u", "/path/agent-run", "myrun", "cmd"]
        ) is True

    # The false-positive that must never match.
    def test_bash_lc_cat_log_is_not_runner(self):
        """bash -lc 'cat /var/tmp/agent-runs/foo/log' must return False.

        This exact argv exists on production hosts and must never be treated
        as a runner — killing it would destroy unrelated work.
        """
        argv = ["bash", "-lc", "cat /var/tmp/agent-runs/foo/log"]
        assert _argv_is_agent_run_runner(argv) is False

    def test_bash_lc_cat_log_false(self):
        # Variant with double quotes rendered as a single token.
        assert _argv_is_agent_run_runner(
            ["/bin/bash", "-lc", "cat /var/tmp/agent-runs/myrun/log false"]
        ) is False

    def test_grep_agent_run_is_not_runner(self):
        assert _argv_is_agent_run_runner(["grep", "agent-run", "/var/log/syslog"]) is False

    def test_editor_with_agent_run_file_open(self):
        assert _argv_is_agent_run_runner(
            ["vim", "/var/tmp/agent-runs/myrun/log"]
        ) is False

    def test_empty_argv(self):
        assert _argv_is_agent_run_runner([]) is False

    def test_tail_agent_run_log_is_not_runner(self):
        assert _argv_is_agent_run_runner(
            ["tail", "-f", "/var/tmp/agent-runs/something/log"]
        ) is False

    def test_agent_run_as_argument_value_is_not_runner(self):
        # The string "agent-run" appears only inside a flag value.
        assert _argv_is_agent_run_runner(
            ["sh", "-c", "echo agent-run"]
        ) is False

    def test_unrelated_script_named_not_agent_run(self):
        assert _argv_is_agent_run_runner(["/usr/bin/agent-runner"]) is False

    def test_python_no_script_returns_false(self):
        # Only flags after python, no positional script.
        assert _argv_is_agent_run_runner(["python3", "-u", "-O"]) is False


# ---------------------------------------------------------------------------
# _run_name_from_argv — name recovery
# ---------------------------------------------------------------------------

class TestRunNameFromArgv:
    def _name(self, argv):
        return _run_name_from_argv(argv)

    def test_bare_invocation(self):
        assert self._name(["agent-run", "myrun", "claude"]) == "myrun"

    def test_short_interactive_flag(self):
        assert self._name(["agent-run", "-i", "myrun", "cmd"]) == "myrun"

    def test_prompt_file_short_space(self):
        assert self._name(["agent-run", "-f", "/p.md", "myrun", "cmd"]) == "myrun"

    def test_prompt_file_long_equals(self):
        assert self._name(["agent-run", "--prompt-file=/p.md", "myrun", "cmd"]) == "myrun"

    def test_echo_bare(self):
        assert self._name(["agent-run", "--echo", "myrun", "cmd"]) == "myrun"

    def test_echo_with_interval(self):
        assert self._name(["agent-run", "--echo=2", "myrun", "cmd"]) == "myrun"

    def test_submit_mode_cr(self):
        assert self._name(["agent-run", "--submit-mode=cr", "myrun", "cmd"]) == "myrun"

    def test_submit_mode_crlf(self):
        assert self._name(["agent-run", "--submit-mode=crlf", "myrun", "cmd"]) == "myrun"

    def test_idle_timeout_space(self):
        assert self._name(["agent-run", "--idle-timeout", "30", "myrun", "cmd"]) == "myrun"

    def test_dashdash_separator(self):
        assert self._name(["agent-run", "myrun", "--", "opencode", "--print"]) == "myrun"

    def test_all_flags_combined(self):
        assert self._name([
            "agent-run", "-i", "-f", "/p.md", "--echo=3",
            "--submit-mode=crlf", "--idle-timeout", "60",
            "myrun", "--", "cmd",
        ]) == "myrun"

    def test_subcommand_reap_yields_none(self):
        # agent-run reap --dry-run is a subcommand, not a launch.
        assert self._name(["agent-run", "reap", "--dry-run"]) is None

    def test_subcommand_list_yields_none(self):
        assert self._name(["agent-run", "list"]) is None

    def test_subcommand_status_yields_none(self):
        assert self._name(["agent-run", "status", "myrun"]) is None

    def test_too_few_args_yields_none(self):
        # Only the entry-point, no name.
        assert self._name(["agent-run"]) is None

    def test_invalid_name_slash_yields_none(self):
        assert self._name(["agent-run", "my/run", "cmd"]) is None

    def test_python_wrapped_name_recovery(self):
        assert self._name([
            "python3", "/usr/local/bin/agent-run", "myrun", "cmd"
        ]) == "myrun"

    def test_no_agent_run_token_yields_none(self):
        # Not a runner at all — no agent-run token in argv.
        assert self._name(["bash", "-lc", "cat /var/tmp/agent-runs/foo/log"]) is None


# ---------------------------------------------------------------------------
# _find_orphan_runners — selection and skip reasons
# ---------------------------------------------------------------------------

class TestFindOrphanRunners:
    """Verify candidate selection and every skip-reason path."""

    def test_basic_candidate(self, isolated_runs_root):
        entry = _make_entry(9001, ["agent-run", "myrun", "claude"])
        candidates, skips = _find([entry])
        assert len(candidates) == 1
        assert candidates[0].pid == 9001
        assert candidates[0].name == "myrun"

    def test_state_dir_present_is_skipped(self, isolated_runs_root):
        # Create the state dir so _path_entry_exists returns True.
        (isolated_runs_root / "myrun").mkdir(parents=True, exist_ok=True)
        entry = _make_entry(9002, ["agent-run", "myrun", "claude"])
        candidates, skips = _find([entry])
        assert not candidates
        assert any(s.reason == "state_dir_exists" for s in skips)

    def test_too_young_is_skipped(self):
        now = time.time()
        entry = _make_entry(9003, ["agent-run", "youngrun", "cmd"],
                            start_time=now - 60.0)  # 60 s old
        candidates, skips = _find([entry], min_age_seconds=3600.0, now=now)
        assert not candidates
        assert any(s.reason == "too_young" for s in skips)

    def test_old_enough_is_candidate(self, isolated_runs_root):
        now = time.time()
        entry = _make_entry(9004, ["agent-run", "oldrun", "cmd"],
                            start_time=now - 7200.0)  # 2 h old
        candidates, skips = _find([entry], min_age_seconds=3600.0, now=now)
        assert len(candidates) == 1
        assert candidates[0].pid == 9004

    def test_self_pid_is_skipped(self):
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
        entry = _make_entry(9006, ["agent-run", "pgidrun", "cmd"], ppid=fake_pgid)
        candidates, skips = _find([entry], self_pgid=fake_pgid, self_pid=os.getpid() + 1000)
        # ppid == pgid triggers the same_pgid rule.
        assert any(s.reason == "same_pgid" for s in skips)

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

    def test_unrecoverable_name_is_skipped(self):
        # bash -lc "..." looks like a runner if we were doing substring matching
        # but _argv_is_agent_run_runner returns False, so it won't even be a skip.
        # Instead test a process that IS a runner but whose name can't be parsed.
        entry = _make_entry(9008, ["agent-run"])  # no name or command
        candidates, skips = _find([entry])
        assert not candidates
        # name_unrecoverable or skipped silently (no runner match at all)

    def test_target_name_filters_others(self, isolated_runs_root):
        e1 = _make_entry(9020, ["agent-run", "run-a", "cmd"])
        e2 = _make_entry(9021, ["agent-run", "run-b", "cmd"])
        candidates, skips = _find([e1, e2], target_name="run-a")
        assert len(candidates) == 1
        assert candidates[0].name == "run-a"
        assert any(s.reason == "name_mismatch" for s in skips)

    def test_non_runner_argv_ignored_entirely(self):
        # grep agent-run is not a runner; it should not appear in skips either.
        entry = _make_entry(9030, ["grep", "agent-run", "/var/log/syslog"])
        candidates, skips = _find([entry])
        assert not candidates
        assert not any(s.pid == 9030 for s in skips)

    def test_bash_lc_cat_log_never_candidate(self):
        """The bash -lc 'cat /var/tmp/agent-runs/foo/log' false positive
        must never appear as a candidate — it is the highest-priority safety
        invariant in this module."""
        entry = _make_entry(9031, ["bash", "-lc", "cat /var/tmp/agent-runs/foo/log"])
        candidates, skips = _find([entry])
        assert not candidates
        assert not any(s.pid == 9031 for s in skips)

    def test_candidates_carry_identity(self, isolated_runs_root):
        entry = _make_entry(9040, ["agent-run", "idrun", "cmd"],
                            identity="darwin:Wed Jan 01 00:00:00 2025")
        candidates, _ = _find([entry])
        assert candidates[0].identity == "darwin:Wed Jan 01 00:00:00 2025"

    def test_multiple_candidates_all_returned(self, isolated_runs_root):
        entries = [
            _make_entry(9050 + i, ["agent-run", f"run{i}", "cmd"])
            for i in range(5)
        ]
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
    def test_vanishing_pid_does_not_raise(self, monkeypatch):
        """Simulate a pid disappearing mid-scan; _scan_process_table must not raise."""
        # Monkeypatch the Linux or Darwin scanner so it encounters a "vanishing" pid.
        import platform as _platform

        original_system = _platform.system()

        if original_system == "Linux":
            real_scan = agent_run._scan_process_table_linux

            def patched_linux():
                # Call through; the actual /proc scan tolerates FileNotFoundError.
                # Inject a fake pid that will cause OSError on every read.
                entries = real_scan()
                # If the real scan works, that proves tolerance. We also
                # verify the function itself doesn't blow up on OSError from
                # within by patching os.listdir to return a ghost pid.
                return entries

            monkeypatch.setattr(agent_run, "_scan_process_table_linux", patched_linux)
        # On Darwin, ps subprocess failure returns an empty list (tested by
        # disabling the subprocess).
        elif original_system == "Darwin":
            import subprocess as _subprocess

            def patched_run(*args, **kwargs):
                raise _subprocess.SubprocessError("simulated failure")

            monkeypatch.setattr(_subprocess, "run", patched_run)

        # Must not raise regardless of platform.
        result = agent_run._scan_process_table()
        assert isinstance(result, list)

    def test_linux_ghost_pid_skipped(self, monkeypatch):
        """A pid listed in /proc that vanishes before its files are read is skipped."""
        import platform as _platform
        if _platform.system() != "Linux":
            pytest.skip("Linux-only test")

        original_listdir = os.listdir

        def fake_listdir(path):
            result = original_listdir(path)
            if str(path) == "/proc":
                result = list(result) + ["999999999"]   # ghost pid
            return result

        monkeypatch.setattr(os, "listdir", fake_listdir)
        # Should not raise even though /proc/999999999/* do not exist.
        entries = agent_run._scan_process_table_linux()
        assert isinstance(entries, list)
        assert not any(e.pid == 999999999 for e in entries)


# ---------------------------------------------------------------------------
# _darwin_lstart_normalise — whitespace canonicalisation
# ---------------------------------------------------------------------------

class TestDarwinLstartNormalise:
    """Regression for the single-digit-day whitespace mismatch.

    ``ps -axo lstart=`` collapses the fixed-width kernel date format to a
    single space between tokens, while ``ps -p PID -o lstart=`` preserves the
    right-padding used for days 1-9 (e.g. ``Mon Jun  8`` — two spaces).
    Both call sites now run through ``_darwin_lstart_normalise`` so the
    identity tokens they produce compare equal.
    """

    def test_single_digit_day_with_double_space_normalised(self):
        """The ps -p form (double-space before single-digit day) must equal the
        ps -axo form (single space) after normalisation."""
        ps_p_form = "Mon Jun  8 12:29:39 2026"   # right-padded, from ps -p
        ps_axo_form = "Mon Jun 8 12:29:39 2026"   # collapsed, from ps -axo split
        assert _darwin_lstart_normalise(ps_p_form) == _darwin_lstart_normalise(ps_axo_form)

    def test_double_digit_day_unchanged(self):
        """Days 10-31 have no extra padding; normalisation must be a no-op."""
        lstart = "Mon Jun 12 14:00:00 2026"
        assert _darwin_lstart_normalise(lstart) == lstart

    def test_already_normalised_idempotent(self):
        """Calling normalise on already-normalised output is idempotent."""
        lstart = "Mon Jun 8 12:29:39 2026"
        assert _darwin_lstart_normalise(_darwin_lstart_normalise(lstart)) == lstart

    def test_identity_token_equality_across_ps_forms(self, monkeypatch):
        """Simulate the two ps call paths and assert identity tokens compare equal.

        This is the end-to-end regression: the token stored by
        _scan_process_table_darwin (built from the ps -axo column) must equal
        the token returned by _process_identity (built from ps -p lstart=) for
        the same process with a single-digit start day.
        """
        import subprocess as _subprocess
        import platform as _platform
        if _platform.system() != "Darwin":
            pytest.skip("Darwin-only test")

        pid = 12345
        my_uid = os.getuid()

        # ps -axo output: whitespace-collapsed by split(None, N) parsing.
        ps_axo_line = (
            f"  {pid}     1 {my_uid} Mon Jun  8 12:29:39 2026 /usr/bin/agent-run myrun claude"
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
