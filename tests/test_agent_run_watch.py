"""Tests for `agent-run watch` — the stateless, read-only JSON fact contract.

Covers: terminal vs running status mapping, missing state dir with a
surviving log dir, a missing `cwd` state file (repo/git → null), git
subprocess failure (git → null), repeated-error detection at and below the
3+ threshold, repeated-read detection, and that `watch` never writes to the
state dir.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pytest

from toolbox import agent_run


WATCH_CONTRACT_KEYS = {
    "schema", "name", "observed_at", "status", "exit_code", "pid",
    "interactive", "started_at", "ended_at", "elapsed_s", "terminal",
    "launch_error", "log", "repo", "git", "signals", "observation_error",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_run(
    state_root: Path,
    log_root: Path,
    name: str,
    *,
    status: str = "running",
    pid: Optional[int] = None,
    interactive: str = "0",
    started_age_secs: float = 100.0,
    ended_age_secs: Optional[float] = None,
    exit_code: Optional[int] = None,
    cwd: Optional[str] = None,
    log_text: str = "some output\n",
    log_age_secs: Optional[float] = None,
    write_state: bool = True,
    write_log: bool = True,
) -> tuple[Path, Path]:
    sd = state_root / name
    ld = log_root / name
    ld.mkdir(parents=True, exist_ok=True)
    if write_state:
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "status").write_text(f"{status}\n")
        (sd / "interactive").write_text(f"{interactive}\n")
        started = datetime.now(timezone.utc) - timedelta(seconds=started_age_secs)
        (sd / "started_at").write_text(started.strftime("%Y-%m-%dT%H:%M:%SZ") + "\n")
        if pid is not None:
            (sd / "pid").write_text(f"{pid}\n")
        if ended_age_secs is not None:
            ended = datetime.now(timezone.utc) - timedelta(seconds=ended_age_secs)
            (sd / "ended_at").write_text(ended.strftime("%Y-%m-%dT%H:%M:%SZ") + "\n")
        if exit_code is not None:
            (sd / "exit_code").write_text(f"{exit_code}\n")
        if cwd is not None:
            (sd / "cwd").write_text(cwd + "\n")
    if write_log:
        log_file = ld / "log"
        log_file.write_text(log_text)
        if log_age_secs is not None:
            old_mtime = time.time() - log_age_secs
            os.utime(log_file, (old_mtime, old_mtime))
    return sd, ld


def _watch_args(name: str, *, as_json: bool = True, repo: Optional[str] = None) -> argparse.Namespace:
    return argparse.Namespace(name=name, json=as_json, repo=repo)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t.com",
             "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t.com"},
    )


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    (path / "a.txt").write_text("hello\n")
    _git(path, "add", "a.txt")
    _git(path, "commit", "-q", "-m", "initial")


# ---------------------------------------------------------------------------
# status / terminal mapping
# ---------------------------------------------------------------------------

class TestStatusMapping:
    def test_running_alive_fresh_log_is_not_terminal(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        _make_run(
            isolated_runs_root, isolated_log_root, "r1",
            status="running", pid=111, log_age_secs=1,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        rc = agent_run.cmd_watch(_watch_args("r1"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "running"
        assert payload["terminal"] is False
        assert payload["ended_at"] is None

    def test_done_status_is_terminal(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        _make_run(
            isolated_runs_root, isolated_log_root, "r2",
            status="done", pid=222, exit_code=0, ended_age_secs=5,
        )
        rc = agent_run.cmd_watch(_watch_args("r2"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "done"
        assert payload["terminal"] is True
        assert payload["exit_code"] == 0
        assert payload["elapsed_s"] is not None

    def test_dead_pid_running_status_reports_died_and_terminal(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        _make_run(
            isolated_runs_root, isolated_log_root, "r3",
            status="running", pid=99999,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: False)
        rc = agent_run.cmd_watch(_watch_args("r3"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "died"
        assert payload["terminal"] is True

    def test_schema_and_exact_keys(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        _make_run(
            isolated_runs_root, isolated_log_root, "r4",
            status="running", pid=111, log_age_secs=1,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        agent_run.cmd_watch(_watch_args("r4"))
        payload = json.loads(capsys.readouterr().out)
        assert payload["schema"] == "agent-run.watch.v1"
        assert set(payload.keys()) == WATCH_CONTRACT_KEYS
        assert set(payload["log"].keys()) == {
            "path", "bytes", "lines", "mtime_age_s", "growing",
        }
        assert set(payload["signals"].keys()) == {
            "repeated_error", "distinct_files_read", "top_repeated_read",
        }

    def test_human_output_is_not_json(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        _make_run(
            isolated_runs_root, isolated_log_root, "r5",
            status="running", pid=111, log_age_secs=1,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        rc = agent_run.cmd_watch(_watch_args("r5", as_json=False))
        assert rc == 0
        out = capsys.readouterr().out
        assert out.startswith("name=r5 status=running")
        with pytest.raises(json.JSONDecodeError):
            json.loads(out)


# ---------------------------------------------------------------------------
# missing state dir / missing run
# ---------------------------------------------------------------------------

class TestUnresolvable:
    def test_unknown_run_exits_nonzero(self, isolated_runs_root, isolated_log_root):
        with pytest.raises(SystemExit) as exc_info:
            agent_run.cmd_watch(_watch_args("nope"))
        assert exc_info.value.code != 0

    def test_missing_state_dir_surviving_log_dir(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        _make_run(
            isolated_runs_root, isolated_log_root, "r6",
            write_state=False, log_text="line1\nline2\n",
        )
        rc = agent_run.cmd_watch(_watch_args("r6"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "not running (log preserved)"
        assert payload["terminal"] is True
        assert payload["pid"] is None
        assert payload["started_at"] is None
        assert payload["repo"] is None
        assert payload["git"] is None
        assert payload["log"]["lines"] == 2

    def test_missing_state_dir_with_explicit_repo_still_reads_git(
        self, isolated_runs_root, isolated_log_root, tmp_path, capsys
    ):
        repo = tmp_path / "repo"
        _init_repo(repo)
        _make_run(
            isolated_runs_root, isolated_log_root, "r7",
            write_state=False,
        )
        rc = agent_run.cmd_watch(_watch_args("r7", repo=str(repo)))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["repo"] == str(repo)
        assert payload["git"] is not None
        assert payload["git"]["head"]


# ---------------------------------------------------------------------------
# repo / git resolution
# ---------------------------------------------------------------------------

class TestRepoResolution:
    def test_missing_cwd_file_yields_null_repo_and_git(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        _make_run(
            isolated_runs_root, isolated_log_root, "r8",
            status="done", pid=1, exit_code=0, ended_age_secs=1,
            cwd=None,
        )
        rc = agent_run.cmd_watch(_watch_args("r8"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["repo"] is None
        assert payload["git"] is None
        # every other field still populated
        assert payload["status"] == "done"
        assert payload["exit_code"] == 0

    def test_cwd_file_used_when_no_repo_flag(
        self, isolated_runs_root, isolated_log_root, tmp_path, capsys
    ):
        repo = tmp_path / "repo"
        _init_repo(repo)
        _make_run(
            isolated_runs_root, isolated_log_root, "r9",
            status="done", pid=1, exit_code=0, ended_age_secs=1,
            cwd=str(repo),
        )
        rc = agent_run.cmd_watch(_watch_args("r9"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["repo"] == str(repo)
        assert payload["git"]["dirty"] is False
        assert payload["git"]["files_changed"] == 0

    def test_repo_flag_overrides_cwd_file(
        self, isolated_runs_root, isolated_log_root, tmp_path, capsys
    ):
        repo_a = tmp_path / "repo_a"
        repo_b = tmp_path / "repo_b"
        _init_repo(repo_a)
        _init_repo(repo_b)
        _make_run(
            isolated_runs_root, isolated_log_root, "r10",
            status="done", pid=1, exit_code=0, ended_age_secs=1,
            cwd=str(repo_a),
        )
        rc = agent_run.cmd_watch(_watch_args("r10", repo=str(repo_b)))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["repo"] == str(repo_b)

    def test_git_subprocess_failure_yields_null_git(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        not_a_repo = isolated_runs_root.parent / "not-a-repo"
        not_a_repo.mkdir(parents=True, exist_ok=True)
        _make_run(
            isolated_runs_root, isolated_log_root, "r11",
            status="done", pid=1, exit_code=0, ended_age_secs=1,
            cwd=str(not_a_repo),
        )
        rc = agent_run.cmd_watch(_watch_args("r11"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["repo"] == str(not_a_repo)
        assert payload["git"] is None

    def test_git_timeout_yields_null_git(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch, capsys
    ):
        repo = tmp_path / "repo"
        _init_repo(repo)
        _make_run(
            isolated_runs_root, isolated_log_root, "r12",
            status="done", pid=1, exit_code=0, ended_age_secs=1,
            cwd=str(repo),
        )

        def _wedged(*_args, **_kwargs):
            raise subprocess.TimeoutExpired(cmd="git", timeout=2.5)

        monkeypatch.setattr(agent_run.subprocess, "run", _wedged)
        rc = agent_run.cmd_watch(_watch_args("r12"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["git"] is None
        # everything else stays populated
        assert payload["status"] == "done"

    def test_commits_since_start_counts_only_newer_commits(
        self, isolated_runs_root, isolated_log_root, tmp_path, capsys
    ):
        repo = tmp_path / "repo"
        _init_repo(repo)
        # Run started now; make one more commit "after" the run started.
        _make_run(
            isolated_runs_root, isolated_log_root, "r13",
            status="done", pid=1, exit_code=0, ended_age_secs=1,
            started_age_secs=5, cwd=str(repo),
        )
        (repo / "b.txt").write_text("more\n")
        _git(repo, "add", "b.txt")
        _git(repo, "commit", "-q", "-m", "second")
        rc = agent_run.cmd_watch(_watch_args("r13"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["git"]["commits_since_start"] >= 1
        assert payload["git"]["last_commit_age_s"] is not None


# ---------------------------------------------------------------------------
# git subprocess hardening
# ---------------------------------------------------------------------------

class TestGitHardening:
    def test_fsmonitor_is_not_executed(self, tmp_path):
        repo = tmp_path / "repo"
        _init_repo(repo)
        marker = tmp_path / "marker"
        script = tmp_path / "fsmonitor.sh"
        script.write_text(f"#!/bin/sh\ntouch '{marker}'\nprintf '2\\n'\n")
        script.chmod(0o755)
        _git(repo, "config", "core.fsmonitor", str(script))
        agent_run._watch_run_git(repo, ["status", "--porcelain"])
        assert marker.exists() is False

    def test_git_dir_env_is_ignored(self, tmp_path, monkeypatch):
        repo_a = tmp_path / "repo_a"
        repo_b = tmp_path / "repo_b"
        _init_repo(repo_a)
        _init_repo(repo_b)
        (repo_b / "b.txt").write_text("more\n")
        _git(repo_b, "add", "b.txt")
        _git(repo_b, "commit", "-q", "-m", "second")

        expected_head = agent_run._watch_run_git(
            repo_a, ["rev-parse", "--short", "HEAD"]
        ).strip()

        monkeypatch.setenv("GIT_DIR", str(repo_b / ".git"))
        monkeypatch.setenv("GIT_WORK_TREE", str(repo_b))
        result = agent_run._watch_run_git(repo_a, ["rev-parse", "--short", "HEAD"])
        assert result.strip() == expected_head

    def test_undecodable_output_does_not_raise(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        _init_repo(repo)

        class _FakeResult:
            stdout = b"A\xff\xfeB"

        def _fake_run(*_args, **_kwargs):
            return _FakeResult()

        monkeypatch.setattr(agent_run.subprocess, "run", _fake_run)
        result = agent_run._watch_run_git(repo, ["status", "--porcelain"])
        assert result == "A\ufffd\ufffdB"


# ---------------------------------------------------------------------------
# log facts
# ---------------------------------------------------------------------------

class TestLogFacts:
    def test_recent_log_reports_growing_true(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        _make_run(
            isolated_runs_root, isolated_log_root, "r14",
            status="running", pid=1, log_age_secs=1,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        agent_run.cmd_watch(_watch_args("r14"))
        payload = json.loads(capsys.readouterr().out)
        assert payload["log"]["growing"] is True
        assert payload["log"]["mtime_age_s"] < agent_run.WATCH_LOG_GROWING_MAX_AGE_SECONDS

    def test_stale_log_reports_growing_false(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        _make_run(
            isolated_runs_root, isolated_log_root, "r15",
            status="running", pid=1,
            log_age_secs=agent_run.WATCH_LOG_GROWING_MAX_AGE_SECONDS + 30,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        agent_run.cmd_watch(_watch_args("r15"))
        payload = json.loads(capsys.readouterr().out)
        assert payload["log"]["growing"] is False

    def test_missing_log_file_yields_null_log(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        _make_run(
            isolated_runs_root, isolated_log_root, "r16",
            status="running", pid=1, write_log=False,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        rc = agent_run.cmd_watch(_watch_args("r16"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["log"] is None
        assert payload["signals"] == {
            "repeated_error": None,
            "distinct_files_read": 0,
            "top_repeated_read": None,
        }


# ---------------------------------------------------------------------------
# signals: repeated-error / repeated-read
# ---------------------------------------------------------------------------

class TestSignals:
    def test_repeated_error_at_threshold_is_detected(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        text = "ok\n" + "Error: boom\n" * agent_run.WATCH_REPEAT_THRESHOLD
        _make_run(
            isolated_runs_root, isolated_log_root, "r17",
            status="running", pid=1, log_text=text, log_age_secs=1,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        agent_run.cmd_watch(_watch_args("r17"))
        payload = json.loads(capsys.readouterr().out)
        assert payload["signals"]["repeated_error"] == "Error: boom"

    def test_repeated_error_below_threshold_is_null(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        text = "ok\n" + "Error: boom\n" * (agent_run.WATCH_REPEAT_THRESHOLD - 1)
        _make_run(
            isolated_runs_root, isolated_log_root, "r18",
            status="running", pid=1, log_text=text, log_age_secs=1,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        agent_run.cmd_watch(_watch_args("r18"))
        payload = json.loads(capsys.readouterr().out)
        assert payload["signals"]["repeated_error"] is None

    def test_repeated_error_line_truncated_to_200_chars(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        long_line = "Error: " + ("x" * 300)
        text = (long_line + "\n") * agent_run.WATCH_REPEAT_THRESHOLD
        _make_run(
            isolated_runs_root, isolated_log_root, "r19",
            status="running", pid=1, log_text=text, log_age_secs=1,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        agent_run.cmd_watch(_watch_args("r19"))
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["signals"]["repeated_error"]) == 200

    def test_repeated_read_detected_at_threshold(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        lines = ["Read /a/one.py"] * agent_run.WATCH_REPEAT_THRESHOLD
        lines += ["Read /a/two.py", "Read /a/three.py"]
        text = "\n".join(lines) + "\n"
        _make_run(
            isolated_runs_root, isolated_log_root, "r20",
            status="running", pid=1, log_text=text, log_age_secs=1,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        agent_run.cmd_watch(_watch_args("r20"))
        payload = json.loads(capsys.readouterr().out)
        assert payload["signals"]["distinct_files_read"] == 3
        assert payload["signals"]["top_repeated_read"] == {
            "path": "/a/one.py",
            "count": agent_run.WATCH_REPEAT_THRESHOLD,
        }

    def test_repeated_read_below_threshold_yields_null_top(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        lines = ["Read /a/one.py"] * (agent_run.WATCH_REPEAT_THRESHOLD - 1)
        text = "\n".join(lines) + "\n"
        _make_run(
            isolated_runs_root, isolated_log_root, "r21",
            status="running", pid=1, log_text=text, log_age_secs=1,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        agent_run.cmd_watch(_watch_args("r21"))
        payload = json.loads(capsys.readouterr().out)
        assert payload["signals"]["top_repeated_read"] is None
        assert payload["signals"]["distinct_files_read"] == 1

    def test_signals_scan_only_last_200_lines(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        # Errors far in the tail's discard zone must not be counted.
        old_errors = "Error: old\n" * agent_run.WATCH_REPEAT_THRESHOLD
        filler = "noise\n" * (agent_run.WATCH_TAIL_LINES + 10)
        text = old_errors + filler
        _make_run(
            isolated_runs_root, isolated_log_root, "r22",
            status="running", pid=1, log_text=text, log_age_secs=1,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        agent_run.cmd_watch(_watch_args("r22"))
        payload = json.loads(capsys.readouterr().out)
        assert payload["signals"]["repeated_error"] is None


# ---------------------------------------------------------------------------
# read-only guarantee
# ---------------------------------------------------------------------------

class TestReadOnly:
    def test_watch_writes_nothing_to_state_dir(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        sd, ld = _make_run(
            isolated_runs_root, isolated_log_root, "r23",
            status="running", pid=1, log_age_secs=1,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        before = {
            p.name: (p.stat().st_mtime, p.read_bytes()) for p in sd.iterdir()
        }
        agent_run.cmd_watch(_watch_args("r23"))
        capsys.readouterr()
        after = {
            p.name: (p.stat().st_mtime, p.read_bytes()) for p in sd.iterdir()
        }
        assert before == after
        assert set(p.name for p in sd.iterdir()) == set(before.keys())

    def test_watch_does_not_create_new_files_in_log_dir(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys, tmp_path
    ):
        repo = tmp_path / "repo"
        _init_repo(repo)
        sd, ld = _make_run(
            isolated_runs_root, isolated_log_root, "r24",
            status="done", pid=1, exit_code=0, ended_age_secs=1, cwd=str(repo),
        )
        before = set(p.name for p in ld.iterdir())
        agent_run.cmd_watch(_watch_args("r24"))
        capsys.readouterr()
        after = set(p.name for p in ld.iterdir())
        assert before == after


# ---------------------------------------------------------------------------
# launch writes cwd
# ---------------------------------------------------------------------------

class TestNeverRaise:
    """The only documented failure mode is an unresolvable run name; every
    other observation failure — corrupt values, unreadable state, an
    unanticipated exception — must degrade individual fields and still
    exit 0 with the full contract."""

    ROOT_SKIP = pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="root ignores permission bits",
    )

    @staticmethod
    def _payload(capsys) -> dict:
        payload = json.loads(capsys.readouterr().out)
        assert set(payload.keys()) == WATCH_CONTRACT_KEYS
        return payload

    def test_unicode_digit_pid_is_null(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        sd, ld = _make_run(
            isolated_runs_root, isolated_log_root, "u1", status="running",
        )
        (sd / "pid").write_text("\u00b2\n")
        rc = agent_run.cmd_watch(_watch_args("u1"))
        assert rc == 0
        payload = self._payload(capsys)
        assert payload["pid"] is None
        assert payload["observation_error"] is None

    def test_unicode_digit_exit_code_is_null(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        sd, ld = _make_run(
            isolated_runs_root, isolated_log_root, "u2",
            status="done", ended_age_secs=1,
        )
        (sd / "exit_code").write_text("\u00b2\n")
        rc = agent_run.cmd_watch(_watch_args("u2"))
        assert rc == 0
        payload = self._payload(capsys)
        assert payload["exit_code"] is None
        assert payload["observation_error"] is None

    def test_state_file_that_is_a_directory_degrades_to_default(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        sd, ld = _make_run(
            isolated_runs_root, isolated_log_root, "u3",
            status="running", pid=111,
        )
        (sd / "pid").unlink()
        (sd / "pid").mkdir()
        rc = agent_run.cmd_watch(_watch_args("u3"))
        assert rc == 0
        payload = self._payload(capsys)
        assert payload["pid"] is None
        assert payload["observation_error"] is None

    @ROOT_SKIP
    def test_state_file_at_mode_000_degrades_to_default(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        sd, ld = _make_run(
            isolated_runs_root, isolated_log_root, "u4",
            status="running", pid=111,
        )
        target = sd / "pid"
        target.chmod(0o000)
        try:
            rc = agent_run.cmd_watch(_watch_args("u4"))
        finally:
            target.chmod(0o644)
        assert rc == 0
        payload = self._payload(capsys)
        assert payload["pid"] is None
        assert payload["observation_error"] is None

    def test_state_file_with_invalid_utf8_degrades_to_default(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        sd, ld = _make_run(
            isolated_runs_root, isolated_log_root, "u5",
            status="running", pid=111,
        )
        (sd / "pid").write_bytes(b"\xff\xfe123")
        rc = agent_run.cmd_watch(_watch_args("u5"))
        assert rc == 0
        payload = self._payload(capsys)
        assert payload["pid"] is None
        assert payload["observation_error"] is None

    @ROOT_SKIP
    def test_log_file_at_mode_000_degrades_line_count(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        sd, ld = _make_run(
            isolated_runs_root, isolated_log_root, "u6",
            status="running", pid=111, log_age_secs=1,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        log_file = ld / "log"
        log_file.chmod(0o000)
        try:
            rc = agent_run.cmd_watch(_watch_args("u6"))
        finally:
            log_file.chmod(0o644)
        assert rc == 0
        payload = self._payload(capsys)
        assert payload["log"]["lines"] == 0
        assert payload["signals"] == {
            "repeated_error": None,
            "distinct_files_read": 0,
            "top_repeated_read": None,
        }
        assert payload["observation_error"] is None

    @ROOT_SKIP
    def test_state_dir_at_mode_000_degrades_every_field(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        sd, ld = _make_run(
            isolated_runs_root, isolated_log_root, "u7",
            status="running", pid=111, log_age_secs=1,
        )
        sd.chmod(0o000)
        try:
            rc = agent_run.cmd_watch(_watch_args("u7"))
        finally:
            sd.chmod(0o755)
        assert rc == 0
        payload = self._payload(capsys)
        assert payload["status"] == "unknown"
        assert payload["terminal"] is False
        assert payload["pid"] is None
        assert payload["started_at"] is None
        assert payload["ended_at"] is None
        assert payload["exit_code"] is None
        assert payload["interactive"] is None
        assert payload["launch_error"] is None
        assert payload["repo"] is None
        assert payload["git"] is None
        assert payload["observation_error"] is None

    def test_observation_error_is_null_on_a_healthy_run(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        _make_run(
            isolated_runs_root, isolated_log_root, "u8",
            status="running", pid=111, log_age_secs=1,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        rc = agent_run.cmd_watch(_watch_args("u8"))
        assert rc == 0
        payload = self._payload(capsys)
        assert payload["observation_error"] is None

    def test_observation_error_set_when_top_level_guard_fires(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        _make_run(
            isolated_runs_root, isolated_log_root, "u9",
            status="running", pid=111, log_age_secs=1,
        )

        def _boom(*_args, **_kwargs):
            raise RuntimeError("synthetic failure for guard test")

        monkeypatch.setattr(agent_run, "_effective_status", _boom)
        rc = agent_run.cmd_watch(_watch_args("u9"))
        assert rc == 0
        payload = self._payload(capsys)
        assert payload["observation_error"] == "RuntimeError: synthetic failure for guard test"
        assert payload["status"] == "unknown"
        assert payload["terminal"] is False
        assert payload["pid"] is None
        assert payload["log"] is None


class TestLaunchWritesCwd:
    def test_launch_records_absolute_cwd(
        self, isolated_runs_root, isolated_log_root, monkeypatch, tmp_path
    ):
        workdir = tmp_path / "launch_from_here"
        workdir.mkdir()
        monkeypatch.chdir(workdir)
        ns = argparse.Namespace(
            command=["true"],
            interactive=False,
            prompt_file=None,
            echo=False,
            echo_interval=2.0,
            submit_mode=None,
            idle_timeout=None,
            name="launched",
        )
        agent_run.cmd_launch(ns)
        sd = isolated_runs_root / "launched"
        deadline = time.monotonic() + 5.0
        while not (sd / "cwd").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (sd / "cwd").read_text().strip() == str(workdir.resolve())
