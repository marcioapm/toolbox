"""Tests for the agent-run CLI ergonomics feature set:

- `_parse_launch_argv`: the pure top-level flag/name/command parser.
- `agent-run du` (per-status rollup and --by-run, --top, --bytes, --json).
- The `--` separator between agent-run's own flags/name and the launch
  command, including its interaction with the pre-existing "flag typed
  after the name" rejection and the known-subcommand dispatch path.
- Exporting BUN_TMPDIR alongside TMPDIR into the launched command's
  environment.

Follows the tmp-dir + monkeypatched STATE_ROOT/LOG_ROOT pattern used by
tests/test_agent_run_reap.py and tests/conftest.py.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
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
    *,
    status: str = "done",
    pid: Optional[int] = None,
    state_bytes: int = 0,
    log_bytes: int = 0,
    scratch_bytes: int = 0,
    state_dir: bool = True,
    log_dir: bool = True,
) -> tuple[Path, Path]:
    """Create a fake run with sized state/log/scratch payloads. Returns
    (state_dir, log_dir); callers compute expected byte totals with
    `_actual_bytes` independently of `_make_run`'s own bookkeeping files
    (status/pid), so assertions never hardcode those files' exact sizes."""
    sd = state_root / name
    ld = log_root / name
    if state_dir:
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "status").write_text(f"{status}\n")
        if pid is not None:
            (sd / "pid").write_text(f"{pid}\n")
        if state_bytes:
            (sd / "padding").write_bytes(b"x" * state_bytes)
    if log_dir:
        ld.mkdir(parents=True, exist_ok=True)
        if log_bytes:
            (ld / "log").write_bytes(b"x" * log_bytes)
        if scratch_bytes:
            scratch = ld / "tmp"
            scratch.mkdir(exist_ok=True)
            (scratch / "leaked").write_bytes(b"x" * scratch_bytes)
    return sd, ld


def _actual_bytes(d: Path, *, exclude: Optional[Path] = None) -> int:
    """Independently sum regular-file sizes under d (excluding `exclude`),
    without going through agent_run's own `_dir_size_bytes` — so a bug in
    that function wouldn't make its own tests pass."""
    total = 0
    if not d.is_dir():
        return 0
    for root, dirs, files in os.walk(d):
        root_path = Path(root)
        if exclude is not None and (root_path == exclude or exclude in root_path.parents):
            dirs[:] = []
            continue
        for f in files:
            total += (root_path / f).stat().st_size
    return total


def _du_args(
    *,
    by_run: bool = False,
    top: Optional[int] = None,
    bytes_: bool = False,
    json_: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(by_run=by_run, top=top, bytes=bytes_, json=json_)


# ---------------------------------------------------------------------------
# du -- rollup / --by-run grouping and byte totals
# ---------------------------------------------------------------------------

class TestDuGrouping:
    def test_empty_roots_report_zero_total(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        rc = agent_run.cmd_du(_du_args())

        assert rc == 0
        out = capsys.readouterr().out
        assert "TOTAL" in out
        assert "0B" in out

    def test_per_status_rollup_totals(self, isolated_runs_root, isolated_log_root, capsys):
        sd1, ld1 = _make_run(
            isolated_runs_root, isolated_log_root, "run-done-1",
            status="done", state_bytes=100, log_bytes=200, scratch_bytes=50,
        )
        sd2, ld2 = _make_run(
            isolated_runs_root, isolated_log_root, "run-done-2",
            status="done", state_bytes=10, log_bytes=20, scratch_bytes=5,
        )
        sd3, ld3 = _make_run(
            isolated_runs_root, isolated_log_root, "run-running",
            status="running", pid=os.getpid(), state_bytes=7, log_bytes=3,
        )
        expected_state = _actual_bytes(sd1) + _actual_bytes(sd2)
        expected_log = _actual_bytes(ld1, exclude=ld1 / "tmp") + _actual_bytes(
            ld2, exclude=ld2 / "tmp"
        )
        expected_scratch = _actual_bytes(ld1 / "tmp") + _actual_bytes(ld2 / "tmp")

        rc = agent_run.cmd_du(_du_args(bytes_=True))

        assert rc == 0
        out = capsys.readouterr().out
        lines = {ln.split()[0]: ln for ln in out.splitlines()}
        assert "done" in lines
        done_fields = lines["done"].split()
        assert done_fields[1] == "2"
        assert done_fields[2] == str(expected_state)
        assert done_fields[3] == str(expected_log)
        assert done_fields[4] == str(expected_scratch)
        assert done_fields[-1] == str(expected_state + expected_log + expected_scratch)
        assert "running" in lines
        total_fields = lines["TOTAL"].split()
        assert total_fields[1] == "3"
        grand_total = (
            _actual_bytes(sd1) + _actual_bytes(ld1, exclude=ld1 / "tmp") + _actual_bytes(ld1 / "tmp")
            + _actual_bytes(sd2) + _actual_bytes(ld2, exclude=ld2 / "tmp") + _actual_bytes(ld2 / "tmp")
            + _actual_bytes(sd3) + _actual_bytes(ld3, exclude=ld3 / "tmp") + _actual_bytes(ld3 / "tmp")
        )
        assert total_fields[-1] == str(grand_total)

    def test_preserved_log_only_group(self, isolated_runs_root, isolated_log_root, capsys):
        _make_run(
            isolated_runs_root, isolated_log_root, "orphanlog",
            state_dir=False, log_bytes=42,
        )

        rc = agent_run.cmd_du(_du_args(bytes_=True))

        assert rc == 0
        out = capsys.readouterr().out
        lines = {ln.split()[0]: ln for ln in out.splitlines()}
        assert "preserved-log-only" in lines
        assert lines["preserved-log-only"].split()[3] == "42"

    def test_scratch_bytes_excluded_from_log_bytes(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        """Log bytes must exclude tmp/ scratch — total = state + log + scratch,
        never double-counting the scratch subtree."""
        sd, ld = _make_run(
            isolated_runs_root, isolated_log_root, "scratchsplit",
            status="done", log_bytes=100, scratch_bytes=900,
        )
        expected_state = _actual_bytes(sd)
        expected_log = _actual_bytes(ld, exclude=ld / "tmp")
        expected_scratch = _actual_bytes(ld / "tmp")

        rc = agent_run.cmd_du(_du_args(bytes_=True))

        out = capsys.readouterr().out
        fields = next(ln for ln in out.splitlines() if ln.startswith("done")).split()
        assert fields[2] == str(expected_state)
        assert fields[3] == str(expected_log) == "100"  # LOG excludes scratch
        assert fields[4] == str(expected_scratch) == "900"  # SCRATCH
        assert fields[-1] == str(expected_state + expected_log + expected_scratch)

    def test_by_run_one_row_per_run(self, isolated_runs_root, isolated_log_root, capsys):
        sdA, ldA = _make_run(
            isolated_runs_root, isolated_log_root, "runA", status="done", log_bytes=500,
        )
        _make_run(
            isolated_runs_root, isolated_log_root, "runB", status="failed", log_bytes=100,
        )
        expected_total_a = _actual_bytes(sdA) + _actual_bytes(ldA)

        rc = agent_run.cmd_du(_du_args(by_run=True, bytes_=True))

        assert rc == 0
        out = capsys.readouterr().out
        assert "runA" in out
        assert "runB" in out
        a_fields = next(ln for ln in out.splitlines() if ln.startswith("runA")).split()
        assert a_fields[-1] == str(expected_total_a)

    def test_sorted_total_descending_then_name(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        _make_run(isolated_runs_root, isolated_log_root, "small", status="done", log_bytes=1)
        _make_run(isolated_runs_root, isolated_log_root, "big", status="done", log_bytes=1000)
        _make_run(isolated_runs_root, isolated_log_root, "medium", status="done", log_bytes=100)

        rc = agent_run.cmd_du(_du_args(by_run=True, bytes_=True))

        out = capsys.readouterr().out
        order = [
            ln.split()[0]
            for ln in out.splitlines()
            if ln.split() and ln.split()[0] in {"small", "big", "medium"}
        ]
        assert order == ["big", "medium", "small"]

    def test_human_readable_by_default(self, isolated_runs_root, isolated_log_root, capsys):
        _make_run(
            isolated_runs_root, isolated_log_root, "hrsize", status="done",
            log_bytes=2 * 1024 * 1024,
        )

        rc = agent_run.cmd_du(_du_args())

        out = capsys.readouterr().out
        assert "2.0M" in out
        assert "2097152" not in out

    def test_no_markdown_table_syntax(self, isolated_runs_root, isolated_log_root, capsys):
        _make_run(isolated_runs_root, isolated_log_root, "plain", status="done", log_bytes=1)

        agent_run.cmd_du(_du_args())

        out = capsys.readouterr().out
        assert "|" not in out
        assert "---" not in out

    def test_du_never_mutates_anything(self, isolated_runs_root, isolated_log_root):
        sd, ld = _make_run(
            isolated_runs_root, isolated_log_root, "untouched",
            status="running", pid=999999999, log_bytes=10,
        )
        before_status = (sd / "status").read_text()

        agent_run.cmd_du(_du_args(by_run=True))
        agent_run.cmd_du(_du_args())

        # A live-looking "running" status with a bogus pid would be healed
        # by _opportunistic_heal if du called it; it must not have.
        assert (sd / "status").read_text() == before_status
        assert sd.exists()
        assert ld.exists()

    @pytest.mark.parametrize("root_kind", ["state", "log"])
    def test_du_ignores_top_level_symlink(self, isolated_runs_root, isolated_log_root, root_kind):
        outside = isolated_log_root.parent / f"outside-{root_kind}"
        outside.mkdir()
        (outside / "payload").write_bytes(b"x" * 123)
        root = isolated_runs_root if root_kind == "state" else isolated_log_root
        (root / "linked").symlink_to(outside, target_is_directory=True)

        assert all(row.key != "linked" for row in agent_run._du_collect_rows())

    @pytest.mark.parametrize("root_kind", ["state", "log"])
    def test_du_per_entry_lstat_error_skips_that_entry_only(
        self, isolated_runs_root, isolated_log_root, monkeypatch, root_kind
    ):
        """An entry whose lstat raises between iterdir() and lstat() must be
        skipped; healthy sibling entries must still appear with their real sizes."""
        root = isolated_runs_root if root_kind == "state" else isolated_log_root
        # Create two real run dirs.
        healthy = root / "healthy"
        doomed = root / "doomed"
        healthy.mkdir()
        doomed.mkdir()
        if root_kind == "log":
            (healthy / "log").write_bytes(b"x" * 100)
            (doomed / "log").write_bytes(b"x" * 100)

        real_lstat = Path.lstat

        def failing_lstat(self):
            if self.name == "doomed":
                raise FileNotFoundError(2, "No such file or directory", str(self))
            return real_lstat(self)

        monkeypatch.setattr(Path, "lstat", failing_lstat)

        rows = {r.key: r for r in agent_run._du_collect_rows()}

        # "healthy" must appear; "doomed" may be absent but must not wipe "healthy".
        assert "healthy" in rows, "healthy sibling must still be reported"


# ---------------------------------------------------------------------------
# cmd_list symlink exclusion (M3 parity with du)
# ---------------------------------------------------------------------------

class TestListSymlinkExclusion:
    @pytest.mark.parametrize("root_kind", ["state", "log"])
    def test_list_ignores_top_level_symlink(
        self, isolated_runs_root, isolated_log_root, capsys, root_kind
    ):
        """A symlink under STATE_ROOT or LOG_ROOT must not appear in cmd_list output."""
        outside = isolated_log_root.parent / f"outside-list-{root_kind}"
        outside.mkdir()
        (outside / "log").write_text("data\n")
        root = isolated_runs_root if root_kind == "state" else isolated_log_root
        (root / "linkedrun").symlink_to(outside, target_is_directory=True)

        agent_run.cmd_list(argparse.Namespace(all=True, status=None, include_logs=True))

        out = capsys.readouterr().out
        assert "linkedrun" not in out


# ---------------------------------------------------------------------------
# _dir_size_bytes top-argument symlink exclusion (M3)
# ---------------------------------------------------------------------------

class TestDirSizeBytesTopSymlink:
    def test_symlink_top_returns_zero(self, tmp_path):
        """_dir_size_bytes must return 0 when ``d`` itself is a symlink,
        even when the target directory contains files."""
        target = tmp_path / "real"
        target.mkdir()
        (target / "payload").write_bytes(b"x" * 500)
        link = tmp_path / "link"
        link.symlink_to(target, target_is_directory=True)

        assert agent_run._dir_size_bytes(link) == 0

    def test_real_dir_is_still_counted(self, tmp_path):
        d = tmp_path / "real"
        d.mkdir()
        (d / "f").write_bytes(b"y" * 300)
        assert agent_run._dir_size_bytes(d) == 300


# ---------------------------------------------------------------------------
# du -- --top truncation
# ---------------------------------------------------------------------------

class TestDuTop:
    def _make_three(self, state_root, log_root):
        dirs = [
            _make_run(state_root, log_root, "top-a", status="done", log_bytes=300),
            _make_run(state_root, log_root, "top-b", status="done", log_bytes=200),
            _make_run(state_root, log_root, "top-c", status="done", log_bytes=100),
        ]
        return sum(_actual_bytes(sd) + _actual_bytes(ld) for sd, ld in dirs)

    def test_top_truncates_by_run_rows(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        self._make_three(isolated_runs_root, isolated_log_root)

        rc = agent_run.cmd_du(_du_args(by_run=True, top=1, bytes_=True))

        assert rc == 0
        out = capsys.readouterr().out
        assert "top-a" in out
        assert "top-b" not in out
        assert "top-c" not in out
        assert "2 more row(s) omitted" in out

    def test_top_total_still_covers_all_runs(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        grand_total = self._make_three(isolated_runs_root, isolated_log_root)

        rc = agent_run.cmd_du(_du_args(by_run=True, top=1, bytes_=True))

        out = capsys.readouterr().out
        total_line = next(ln for ln in out.splitlines() if ln.startswith("TOTAL"))
        assert total_line.split()[-1] == str(grand_total)

    def test_top_shown_plus_omitted_equals_total(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        self._make_three(isolated_runs_root, isolated_log_root)

        rc = agent_run.cmd_du(_du_args(by_run=True, top=2, bytes_=True))

        out = capsys.readouterr().out
        shown_total = sum(
            int(ln.split()[-1])
            for ln in out.splitlines()
            if ln.split() and ln.split()[0] in {"top-a", "top-b", "top-c"}
        )
        omitted_line = next(ln for ln in out.splitlines() if "more row(s) omitted" in ln)
        # "... 1 more row(s) omitted by --top (sum <N>; ...)" (--bytes: no unit suffix)
        omitted_sum = int(omitted_line.split("sum ")[1].split(";")[0].rstrip("B"))
        total_line = next(ln for ln in out.splitlines() if ln.startswith("TOTAL"))
        assert shown_total + omitted_sum == int(total_line.split()[-1])

    def test_top_with_rollup_mode(self, isolated_runs_root, isolated_log_root, capsys):
        _make_run(isolated_runs_root, isolated_log_root, "d1", status="done", log_bytes=1)
        _make_run(isolated_runs_root, isolated_log_root, "f1", status="failed", log_bytes=1)
        _make_run(
            isolated_runs_root, isolated_log_root, "r1", status="running",
            pid=os.getpid(), log_bytes=1,
        )

        rc = agent_run.cmd_du(_du_args(top=1, bytes_=True))

        assert rc == 0
        out = capsys.readouterr().out
        assert "more row(s) omitted" in out

    def test_invalid_top_rejected_by_argparse(self):
        with pytest.raises(SystemExit):
            agent_run._build_parser().parse_args(["du", "--top", "0"])
        with pytest.raises(SystemExit):
            agent_run._build_parser().parse_args(["du", "--top", "-1"])
        with pytest.raises(SystemExit):
            agent_run._build_parser().parse_args(["du", "--top", "notanumber"])


# ---------------------------------------------------------------------------
# du -- --json / --bytes
# ---------------------------------------------------------------------------

class TestDuJson:
    def test_json_shape_rollup(self, isolated_runs_root, isolated_log_root, capsys):
        sd, ld = _make_run(
            isolated_runs_root, isolated_log_root, "jsonrun", status="done", log_bytes=64
        )
        expected_total = _actual_bytes(sd) + _actual_bytes(ld)

        rc = agent_run.cmd_du(_du_args(json_=True))

        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "groups" in payload
        assert "runs" not in payload
        assert payload["groups"]["done"]["log_bytes"] == 64
        assert payload["total"]["total_bytes"] == expected_total
        assert payload["state_root"] == str(isolated_runs_root)
        assert payload["log_root"] == str(isolated_log_root)
        assert isinstance(payload["groups"]["done"]["log_bytes"], int)

    def test_json_shape_by_run(self, isolated_runs_root, isolated_log_root, capsys):
        _make_run(isolated_runs_root, isolated_log_root, "jsonrun2", status="done", log_bytes=64)

        rc = agent_run.cmd_du(_du_args(by_run=True, json_=True))

        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "runs" in payload
        assert "groups" not in payload
        assert payload["runs"][0]["name"] == "jsonrun2"
        assert payload["runs"][0]["log_bytes"] == 64

    def test_json_with_top_includes_omitted(self, isolated_runs_root, isolated_log_root, capsys):
        sdA, ldA = _make_run(
            isolated_runs_root, isolated_log_root, "ja", status="done", log_bytes=200
        )
        sdB, ldB = _make_run(
            isolated_runs_root, isolated_log_root, "jb", status="done", log_bytes=100
        )
        total_a = _actual_bytes(sdA) + _actual_bytes(ldA)
        total_b = _actual_bytes(sdB) + _actual_bytes(ldB)

        rc = agent_run.cmd_du(_du_args(by_run=True, top=1, json_=True))

        payload = json.loads(capsys.readouterr().out)
        assert len(payload["runs"]) == 1
        assert payload["omitted"]["count"] == 1
        assert payload["omitted"]["total_bytes"] == total_b
        assert payload["total"]["total_bytes"] == total_a + total_b

    def test_bytes_and_json_together_rejected(self, isolated_runs_root, isolated_log_root):
        with pytest.raises(SystemExit):
            agent_run.cmd_du(_du_args(bytes_=True, json_=True))

    def test_json_always_emits_integers_without_bytes_flag(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        _make_run(
            isolated_runs_root, isolated_log_root, "intcheck", status="done",
            log_bytes=3 * 1024 * 1024,
        )

        rc = agent_run.cmd_du(_du_args(json_=True))

        payload = json.loads(capsys.readouterr().out)
        assert payload["groups"]["done"]["log_bytes"] == 3 * 1024 * 1024


# ---------------------------------------------------------------------------
# `--` separator
# ---------------------------------------------------------------------------

class TestDashDashSeparator:
    def test_error_when_name_missing_before_dashdash(self):
        with pytest.raises(SystemExit) as exc:
            agent_run.main(["--", "echo", "hi"])
        assert "before" in str(exc.value)
        assert "--" in str(exc.value)

    def test_error_on_empty_command_after_dashdash(self):
        with pytest.raises(SystemExit) as exc:
            agent_run.main(["mytask", "--"])
        assert "empty command" in str(exc.value)

    def test_existing_flag_after_name_without_dashdash_still_rejected(self):
        with pytest.raises(SystemExit) as exc:
            agent_run.main(["mytask", "--foo"])
        msg = str(exc.value)
        assert "looks like an agent-run flag" in msg
        assert "--" in msg


# ---------------------------------------------------------------------------
# BUN_TMPDIR export alongside TMPDIR
# ---------------------------------------------------------------------------

def _run_and_read_log(
    state_root: Path,
    log_root: Path,
    name: str,
    script: str,
    timeout: float = 10.0,
) -> str:
    """Launch ``python -c script`` as ``name``, wait for a terminal status,
    and return the captured log contents."""
    args = argparse.Namespace(
        name=name,
        command=[sys.executable, "-c", script],
        interactive=False,
        prompt_file=None,
        submit_mode=None,
    )
    rc = agent_run.cmd_launch(args)
    assert rc == 0

    state_dir = state_root / name
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status = (state_dir / "status").read_text().strip()
        except FileNotFoundError:
            status = ""
        if status in {"done", "failed"}:
            break
        time.sleep(0.05)

    log_path = log_root / name / "log"
    deadline = time.monotonic() + timeout
    content = ""
    while time.monotonic() < deadline:
        if log_path.exists():
            content = log_path.read_text()
            if content:
                break
        time.sleep(0.05)
    return content


class TestBunTmpdirExport:
    def test_bun_tmpdir_equals_tmpdir_equals_scratch_dir(
        self, isolated_runs_root, isolated_log_root
    ):
        name = "buntmpdirrun"
        script = (
            "import os, sys; sys.stdout.write("
            "os.environ.get('TMPDIR', '<unset>') + '|' + "
            "os.environ.get('BUN_TMPDIR', '<unset>'))"
        )
        content = _run_and_read_log(isolated_runs_root, isolated_log_root, name, script)

        expected_scratch = str(isolated_log_root / name / "tmp")
        tmpdir_seen, bun_tmpdir_seen = content.strip().split("|")
        assert tmpdir_seen == expected_scratch
        assert bun_tmpdir_seen == expected_scratch

    def test_inherited_bun_tmpdir_is_overridden(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        monkeypatch.setenv("BUN_TMPDIR", "/some/ambient/bun/tmpdir")
        name = "buntmpdiroverride"
        script = "import os, sys; sys.stdout.write(os.environ.get('BUN_TMPDIR', '<unset>'))"
        content = _run_and_read_log(isolated_runs_root, isolated_log_root, name, script)

        expected_scratch = str(isolated_log_root / name / "tmp")
        assert content.strip() == expected_scratch
        assert content.strip() != "/some/ambient/bun/tmpdir"


# ---------------------------------------------------------------------------
# _parse_launch_argv — pure helper unit tests
# ---------------------------------------------------------------------------

class TestParseLaunchArgv:
    """Direct unit tests for _parse_launch_argv: flags, --, errors, and a
    table-driven comparison against the pre-refactor main() behaviour."""

    def _parse(self, argv):
        return agent_run._parse_launch_argv(argv)

    # --- individual flag forms ---

    def test_interactive_short(self):
        r = self._parse(["-i", "myrun", "cmd"])
        assert r.interactive is True
        assert r.name == "myrun"
        assert r.command == ["cmd"]

    def test_interactive_long(self):
        r = self._parse(["--interactive", "myrun", "cmd"])
        assert r.interactive is True

    def test_prompt_file_short_space(self):
        r = self._parse(["-f", "/path/to/p.md", "myrun", "cmd"])
        assert r.prompt_file == "/path/to/p.md"

    def test_prompt_file_long_space(self):
        r = self._parse(["--prompt-file", "/path.md", "myrun", "cmd"])
        assert r.prompt_file == "/path.md"

    def test_prompt_file_equals(self):
        r = self._parse(["--prompt-file=/path.md", "myrun", "cmd"])
        assert r.prompt_file == "/path.md"

    def test_submit_mode_cr(self):
        r = self._parse(["--submit-mode=cr", "myrun", "cmd"])
        assert r.submit_mode == "cr"

    def test_submit_mode_crlf(self):
        r = self._parse(["--submit-mode=crlf", "myrun", "cmd"])
        assert r.submit_mode == "crlf"

    def test_idle_timeout_space(self):
        r = self._parse(["--idle-timeout", "60", "myrun", "cmd"])
        assert r.idle_timeout == 60.0

    def test_idle_timeout_equals(self):
        r = self._parse(["--idle-timeout=120", "myrun", "cmd"])
        assert r.idle_timeout == 120.0

    # --- combined and varied order ---

    def test_all_flags_combined_any_order(self):
        r = self._parse([
            "-i", "--submit-mode=crlf",
            "-f", "/p.md", "--idle-timeout=90",
            "myrun", "--", "opencode",
        ])
        assert r.interactive is True
        assert r.prompt_file == "/p.md"
        assert r.submit_mode == "crlf"
        assert r.idle_timeout == 90.0
        assert r.name == "myrun"
        assert r.command == ["opencode"]

    def test_flags_interleaved_order(self):
        r = self._parse([
            "--idle-timeout", "45", "--interactive",
            "myrun", "--", "cmd", "arg",
        ])
        assert r.idle_timeout == 45.0
        assert r.interactive is True
        assert r.command == ["cmd", "arg"]

    # --- -- separator semantics ---

    def test_dashdash_passes_leading_dash_tokens_verbatim(self):
        r = self._parse(["myrun", "--", "-not-a-flag", "--also-not"])
        assert r.command == ["-not-a-flag", "--also-not"]
        assert r.subcommand_tokens is None

    def test_dashdash_passes_second_dashdash_verbatim(self):
        r = self._parse(["myrun", "--", "foo", "--", "bar"])
        assert r.command == ["foo", "--", "bar"]

    def test_dashdash_no_subcommand_dispatch(self):
        r = self._parse(["myrun", "--", "list", "foo"])
        assert r.name == "myrun"
        assert r.command == ["list", "foo"]
        assert r.subcommand_tokens is None

    @pytest.mark.parametrize("name", sorted(agent_run._KNOWN_SUBCOMMANDS))
    def test_subcommand_name_with_dashdash_dispatches_as_subcommand(self, name):
        """A known subcommand token always dispatches, even with "--" following it."""
        r = self._parse([name, "--", "myrun"])
        assert r.subcommand_tokens == [name, "--", "myrun"]

    def test_dashdash_flags_before_name(self):
        r = self._parse(["--idle-timeout", "30", "myrun", "--", "some-cmd", "--flag"])
        assert r.idle_timeout == 30.0
        assert r.name == "myrun"
        assert r.command == ["some-cmd", "--flag"]

    # --- error cases with exact message strings ---

    def test_error_prompt_file_missing_path(self):
        with pytest.raises(agent_run._LaunchArgvError) as exc:
            self._parse(["-f"])
        assert str(exc.value) == "agent-run: -f/--prompt-file requires a path"

    def test_error_prompt_file_long_missing_path(self):
        with pytest.raises(agent_run._LaunchArgvError) as exc:
            self._parse(["--prompt-file"])
        assert str(exc.value) == "agent-run: -f/--prompt-file requires a path"

    def test_error_idle_timeout_missing_value(self):
        with pytest.raises(agent_run._LaunchArgvError) as exc:
            self._parse(["--idle-timeout"])
        assert str(exc.value) == "agent-run: --idle-timeout requires a value in seconds"

    def test_error_submit_mode_invalid_value(self):
        with pytest.raises(agent_run._LaunchArgvError) as exc:
            self._parse(["--submit-mode=lf", "myrun", "cmd"])
        assert str(exc.value) == "agent-run: --submit-mode must be cr or crlf"

    def test_error_name_missing_before_dashdash_exact(self):
        with pytest.raises(agent_run._LaunchArgvError) as exc:
            self._parse(["--", "echo"])
        assert str(exc.value) == (
            "agent-run: the run name must appear before '--'; shape is "
            "'agent-run [flags] NAME -- <command> [args...]'"
        )

    def test_error_empty_command_after_dashdash_exact(self):
        with pytest.raises(agent_run._LaunchArgvError) as exc:
            self._parse(["myrun", "--"])
        assert str(exc.value) == (
            "agent-run: empty command after '--'; provide a command to "
            "launch: 'agent-run [flags] NAME -- <command> [args...]'"
        )

    def test_error_flag_after_name_without_dashdash(self):
        with pytest.raises(agent_run._LaunchArgvError) as exc:
            self._parse(["myrun", "--foo"])
        msg = str(exc.value)
        assert "looks like an agent-run flag" in msg
        assert "--" in msg

    def test_error_flag_after_name_contains_offending_token(self):
        with pytest.raises(agent_run._LaunchArgvError) as exc:
            self._parse(["myrun", "--unknown-flag"])
        assert "'--unknown-flag'" in str(exc.value)

    def test_error_invalid_name_with_slash(self):
        with pytest.raises(agent_run._LaunchArgvError) as exc:
            self._parse(["my/run", "cmd"])
        assert "invalid name" in str(exc.value)

    def test_error_invalid_name_leading_dash(self):
        with pytest.raises(agent_run._LaunchArgvError) as exc:
            self._parse(["-myrun", "cmd"])
        assert "invalid name" in str(exc.value)

    def test_error_empty_name_raises_not_help(self):
        # agent-run "" echo hi — empty name must reject with "invalid name",
        # not silently signal the help path.  Old main() reached _validate_run_name
        # and exited with the same message; _parse_launch_argv must match.
        for argv in [["", "echo", "hi"], ["", "--", "echo"], ["-i", "", "cmd"]]:
            with pytest.raises(agent_run._LaunchArgvError) as exc:
                self._parse(argv)
            assert "invalid name" in str(exc.value), argv

    # --- subcommand dispatch signal ---

    def test_subcommand_returns_subcommand_tokens(self):
        r = self._parse(["status", "myrun"])
        assert r.subcommand_tokens == ["status", "myrun"]

    def test_subcommand_with_launch_flag_not_dispatched(self):
        r = self._parse(["-i", "status", "myrun"])
        # -i was consumed; "status" becomes the name, "myrun" the command
        assert r.subcommand_tokens is None
        assert r.name == "status"
        assert r.command == ["myrun"]
        assert r.interactive is True

    def test_all_known_subcommands_dispatched(self):
        for sub in ["status", "watch", "logs", "tail", "attach", "steer",
                    "kill", "list", "reap", "du", "help"]:
            r = self._parse([sub])
            assert r.subcommand_tokens is not None, f"{sub!r} did not dispatch"
            assert r.subcommand_tokens[0] == sub

    # --- insufficient args ---

    def test_too_few_args_returns_empty_name(self):
        r = self._parse(["myrun"])
        assert r.name == ""
        assert r.subcommand_tokens is None

    # --- table-driven match against main() ---

    @pytest.mark.parametrize("argv,expected", [
        (
            ["-i", "myrun", "--", "claude", "--print"],
            dict(interactive=True, name="myrun", command=["claude", "--print"],
                 submit_mode=None, idle_timeout=None,
                 prompt_file=None),
        ),
        (
            ["--idle-timeout", "30", "myrun", "--", "cmd", "--flag"],
            dict(interactive=False, name="myrun", command=["cmd", "--flag"],
                 submit_mode=None, idle_timeout=30.0,
                 prompt_file=None),
        ),
        (
            ["-f", "/some/prompt.md", "myrun", "cmd"],
            dict(interactive=False, prompt_file="/some/prompt.md", name="myrun",
                 command=["cmd"],
                 submit_mode=None, idle_timeout=None),
        ),
        (
            ["--submit-mode=crlf", "myrun", "--", "opencode"],
            dict(interactive=False, submit_mode="crlf", name="myrun",
                 command=["opencode"],
                 idle_timeout=None, prompt_file=None),
        ),
        (
            ["myrun", "echo", "hello"],
            dict(interactive=False, name="myrun", command=["echo", "hello"],
                 submit_mode=None,
                 idle_timeout=None, prompt_file=None),
        ),
        (
            ["myrun", "--", "list", "foo"],
            dict(interactive=False, name="myrun", command=["list", "foo"],
                 submit_mode=None,
                 idle_timeout=None, prompt_file=None),
        ),
        (
            ["myrun", "--", "foo", "--", "bar"],
            dict(interactive=False, name="myrun", command=["foo", "--", "bar"],
                 submit_mode=None,
                 idle_timeout=None, prompt_file=None),
        ),
        (
            ["myrun", "--", "-not-a-flag", "--also-not"],
            dict(interactive=False, name="myrun", command=["-not-a-flag", "--also-not"],
                 submit_mode=None,
                 idle_timeout=None, prompt_file=None),
        ),
        (
            ["-i", "mytask", "--", "claude", "--permission-mode", "bypassPermissions", "--print"],
            dict(interactive=True, name="mytask",
                 command=["claude", "--permission-mode", "bypassPermissions", "--print"],
                 submit_mode=None,
                 idle_timeout=None, prompt_file=None),
        ),
    ])
    def test_table_driven_matches_captured_main(self, argv, expected, monkeypatch):
        """_parse_launch_argv output matches what main() forwarded to cmd_launch."""
        captured = {}
        monkeypatch.setattr(
            agent_run, "cmd_launch", lambda args: captured.update(vars(args)) or 0
        )
        agent_run.main(argv)
        r = self._parse(argv)
        assert r.subcommand_tokens is None
        for key, val in expected.items():
            assert getattr(r, key) == val, f"{key}: expected {val!r}, got {getattr(r, key)!r}"
            if key in captured:
                assert captured[key] == val, (
                    f"main() produced {captured[key]!r} for {key!r}, "
                    f"helper produced {val!r}"
                )

    # --- --worktree* flags: argv-level parsing ---

    def test_worktree_flags_space_form(self):
        r = self._parse([
            "--worktree", "/tmp/wt", "--worktree-base", "main",
            "--worktree-branch", "feature", "--worktree-repo", "/tmp/repo",
            "--worktree-reuse", "myrun", "cmd",
        ])
        assert r.worktree == "/tmp/wt"
        assert r.worktree_base == "main"
        assert r.worktree_branch == "feature"
        assert r.worktree_repo == "/tmp/repo"
        assert r.worktree_reuse is True
        assert r.name == "myrun"
        assert r.command == ["cmd"]

    def test_worktree_flags_equals_form(self):
        r = self._parse([
            "--worktree=/tmp/wt", "--worktree-base=main",
            "--worktree-branch=feature", "--worktree-repo=/tmp/repo",
            "myrun", "cmd",
        ])
        assert r.worktree == "/tmp/wt"
        assert r.worktree_base == "main"
        assert r.worktree_branch == "feature"
        assert r.worktree_repo == "/tmp/repo"

    def test_worktree_flags_default_to_none_or_false(self):
        r = self._parse(["myrun", "cmd"])
        assert r.worktree is None
        assert r.worktree_base is None
        assert r.worktree_branch is None
        assert r.worktree_repo is None
        assert r.worktree_reuse is False

    def test_worktree_flag_reaches_main_and_cmd_launch(self, monkeypatch):
        """The five --worktree* fields survive the full main() -> cmd_launch
        Namespace path, not only the pure parser."""
        captured = {}
        monkeypatch.setattr(
            agent_run, "cmd_launch", lambda args: captured.update(vars(args)) or 0
        )
        agent_run.main([
            "--worktree", "/tmp/wt", "--worktree-base", "main",
            "--worktree-branch", "feature", "--worktree-repo", "/tmp/repo",
            "--worktree-reuse", "myrun", "cmd",
        ])
        assert captured["worktree"] == "/tmp/wt"
        assert captured["worktree_base"] == "main"
        assert captured["worktree_branch"] == "feature"
        assert captured["worktree_repo"] == "/tmp/repo"
        assert captured["worktree_reuse"] is True

    @pytest.mark.parametrize("flag", [
        "--worktree", "--worktree-base", "--worktree-branch", "--worktree-repo",
    ])
    def test_worktree_value_flag_missing_value_is_rejected(self, flag):
        with pytest.raises(agent_run._LaunchArgvError):
            self._parse([flag])

    @pytest.mark.parametrize("flag", [
        "--worktree", "--worktree-base", "--worktree-branch", "--worktree-repo",
    ])
    def test_worktree_value_flag_swallowing_next_flag_is_rejected(self, flag):
        """A typo omitting DIR (e.g. `--worktree --worktree-base main n -- cmd`)
        must not silently take the next flag as this flag's value and then
        fail pointing at the wrong flag."""
        with pytest.raises(agent_run._LaunchArgvError) as exc:
            self._parse([flag, "--worktree-base", "n", "--", "true"])
        assert flag in str(exc.value)
        assert "requires a value" in str(exc.value)
