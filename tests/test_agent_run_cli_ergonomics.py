"""Tests for the agent-run CLI ergonomics feature set:

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
import sys
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
    *, by_run: bool = False, top: Optional[int] = None, bytes_: bool = False, json_: bool = False
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
        assert done_fields[5] == str(expected_state + expected_log + expected_scratch)
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
        assert fields[5] == str(expected_state + expected_log + expected_scratch)

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
    def test_flags_name_dashdash_command_with_dash_laden_args(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            agent_run, "cmd_launch", lambda args: captured.update(vars(args)) or 0
        )

        rc = agent_run.main(
            ["-i", "mytask", "--", "claude", "--permission-mode", "bypassPermissions", "--print"]
        )

        assert rc == 0
        assert captured["name"] == "mytask"
        assert captured["command"] == [
            "claude", "--permission-mode", "bypassPermissions", "--print",
        ]
        assert captured["interactive"] is True

    def test_error_when_name_missing_before_dashdash(self):
        with pytest.raises(SystemExit) as exc:
            agent_run.main(["--", "echo", "hi"])
        assert "before" in str(exc.value)
        assert "--" in str(exc.value)

    def test_error_on_empty_command_after_dashdash(self):
        with pytest.raises(SystemExit) as exc:
            agent_run.main(["mytask", "--"])
        assert "empty command" in str(exc.value)

    def test_no_subcommand_dispatch_after_dashdash(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            agent_run, "cmd_launch", lambda args: captured.update(vars(args)) or 0
        )

        rc = agent_run.main(["mytask", "--", "list", "foo"])

        assert rc == 0
        assert captured["name"] == "mytask"
        assert captured["command"] == ["list", "foo"]

    def test_literal_dashdash_expressible_after_separator(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            agent_run, "cmd_launch", lambda args: captured.update(vars(args)) or 0
        )

        rc = agent_run.main(["mytask", "--", "foo", "--", "bar"])

        assert rc == 0
        assert captured["command"] == ["foo", "--", "bar"]

    def test_existing_no_dashdash_behavior_unchanged_plain_command(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            agent_run, "cmd_launch", lambda args: captured.update(vars(args)) or 0
        )

        rc = agent_run.main(["mytask", "echo", "hi"])

        assert rc == 0
        assert captured["name"] == "mytask"
        assert captured["command"] == ["echo", "hi"]

    def test_existing_flag_after_name_without_dashdash_still_rejected(self):
        with pytest.raises(SystemExit) as exc:
            agent_run.main(["mytask", "--foo"])
        msg = str(exc.value)
        assert "looks like an agent-run flag" in msg
        assert "--" in msg  # now also suggests the separator

    def test_dashdash_command_leading_dash_token_not_rejected(self, monkeypatch):
        """Before `--`, a leading-dash command token is rejected; after it,
        it must be accepted verbatim."""
        captured = {}
        monkeypatch.setattr(
            agent_run, "cmd_launch", lambda args: captured.update(vars(args)) or 0
        )

        rc = agent_run.main(["mytask", "--", "-not-a-flag", "--also-not"])

        assert rc == 0
        assert captured["command"] == ["-not-a-flag", "--also-not"]

    def test_agent_run_flags_before_name_still_consumed_with_dashdash(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            agent_run, "cmd_launch", lambda args: captured.update(vars(args)) or 0
        )

        rc = agent_run.main(
            ["--echo", "--idle-timeout", "30", "mytask", "--", "some-cmd", "--flag"]
        )

        assert rc == 0
        assert captured["echo"] is True
        assert captured["idle_timeout"] == 30.0
        assert captured["name"] == "mytask"
        assert captured["command"] == ["some-cmd", "--flag"]


# ---------------------------------------------------------------------------
# BUN_TMPDIR export alongside TMPDIR
# ---------------------------------------------------------------------------

class TestBunTmpdirExport:
    def test_bun_tmpdir_equals_tmpdir_equals_scratch_dir(
        self, isolated_runs_root, isolated_log_root
    ):
        name = "buntmpdirrun"
        args = argparse.Namespace(
            name=name,
            command=[
                sys.executable, "-c",
                "import os, sys; sys.stdout.write(os.environ.get('TMPDIR', '<unset>') + "
                "'|' + os.environ.get('BUN_TMPDIR', '<unset>'))",
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
        deadline_ok = False
        import time as _time
        deadline = _time.monotonic() + 10
        while _time.monotonic() < deadline:
            try:
                status = (state_dir / "status").read_text().strip()
            except FileNotFoundError:
                status = ""
            if status in {"done", "failed"}:
                deadline_ok = True
                break
            _time.sleep(0.05)
        assert deadline_ok

        log_path = isolated_log_root / name / "log"
        deadline = _time.monotonic() + 10
        content = ""
        while _time.monotonic() < deadline:
            if log_path.exists():
                content = log_path.read_text()
                if content:
                    break
            _time.sleep(0.05)

        expected_scratch = str(isolated_log_root / name / "tmp")
        tmpdir_seen, bun_tmpdir_seen = content.strip().split("|")
        assert tmpdir_seen == expected_scratch
        assert bun_tmpdir_seen == expected_scratch

    def test_inherited_bun_tmpdir_is_overridden(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        monkeypatch.setenv("BUN_TMPDIR", "/some/ambient/bun/tmpdir")
        name = "buntmpdiroverride"
        args = argparse.Namespace(
            name=name,
            command=[
                sys.executable, "-c",
                "import os, sys; sys.stdout.write(os.environ.get('BUN_TMPDIR', '<unset>'))",
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
        import time as _time
        deadline = _time.monotonic() + 10
        while _time.monotonic() < deadline:
            try:
                status = (state_dir / "status").read_text().strip()
            except FileNotFoundError:
                status = ""
            if status in {"done", "failed"}:
                break
            _time.sleep(0.05)

        log_path = isolated_log_root / name / "log"
        deadline = _time.monotonic() + 10
        content = ""
        while _time.monotonic() < deadline:
            if log_path.exists():
                content = log_path.read_text()
                if content:
                    break
            _time.sleep(0.05)

        expected_scratch = str(isolated_log_root / name / "tmp")
        assert content.strip() == expected_scratch
        assert content.strip() != "/some/ambient/bun/tmpdir"
