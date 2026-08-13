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
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pytest

from toolbox import agent_run


WATCH_CONTRACT_KEYS = {
    "schema", "name", "observed_at", "status", "exit_code", "pid",
    "interactive", "started_at", "ended_at", "elapsed_s", "terminal",
    "launch_error", "log", "repo", "git", "git_error", "signals", "observation_error",
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
    process_identity: Optional[str] = None,
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
        if process_identity is not None:
            (sd / "process_identity").write_text(process_identity + "\n")
    if write_log:
        log_file = ld / "log"
        log_file.write_text(log_text)
        if log_age_secs is not None:
            old_mtime = time.time() - log_age_secs
            os.utime(log_file, (old_mtime, old_mtime))
    return sd, ld


def _watch_args(name: str, *, as_json: bool = True, repo: Optional[str] = None) -> argparse.Namespace:
    return argparse.Namespace(name=name, json=as_json, repo=repo)


def _mock_verified_alive(monkeypatch, token: str = "linux:1000") -> str:
    """Mock a pid as alive with a live identity of ``token``; the caller is
    responsible for recording the same token as the run's process_identity
    so watch's identity check matches."""
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
    monkeypatch.setattr(agent_run, "_process_identity", lambda _p: token)
    return token


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
        token = _mock_verified_alive(monkeypatch)
        _make_run(
            isolated_runs_root, isolated_log_root, "r1",
            status="running", pid=111, log_age_secs=1, process_identity=token,
        )
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

    def test_identity_match_reports_running(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        token = _mock_verified_alive(monkeypatch, token="linux:5000")
        _make_run(
            isolated_runs_root, isolated_log_root, "r25",
            status="running", pid=111, log_age_secs=1, process_identity=token,
        )
        rc = agent_run.cmd_watch(_watch_args("r25"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "running"
        assert payload["terminal"] is False

    def test_identity_mismatch_reports_died_and_terminal(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        monkeypatch.setattr(agent_run, "_process_identity", lambda _p: "linux:9999")
        _make_run(
            isolated_runs_root, isolated_log_root, "r26",
            status="running", pid=111, log_age_secs=1, process_identity="linux:1111",
        )
        rc = agent_run.cmd_watch(_watch_args("r26"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "died"
        assert payload["terminal"] is True

    def test_no_identity_recorded_reports_unverified(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        _make_run(
            isolated_runs_root, isolated_log_root, "r27",
            status="running", pid=111, log_age_secs=1,
        )
        rc = agent_run.cmd_watch(_watch_args("r27"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "unverified"
        assert payload["terminal"] is False

    def test_unreadable_identity_reports_unverified(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        monkeypatch.setattr(agent_run, "_process_identity", lambda _p: None)
        _make_run(
            isolated_runs_root, isolated_log_root, "r28",
            status="running", pid=111, log_age_secs=1, process_identity="linux:1111",
        )
        rc = agent_run.cmd_watch(_watch_args("r28"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "unverified"
        assert payload["terminal"] is False

    def test_status_cmd_unchanged_for_run_with_no_process_identity(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        """Proves `_effective_status` itself was not altered: `cmd_status`
        must still print "running" for a live pid with no identity token,
        even though `watch` reports "unverified" for the same run."""
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        _make_run(
            isolated_runs_root, isolated_log_root, "r29",
            status="running", pid=111, log_age_secs=1,
        )
        rc = agent_run.cmd_status(argparse.Namespace(name="r29"))
        assert rc == 0
        out = capsys.readouterr().out
        assert out.startswith("name=r29 status=running ")

    def test_logs_cmd_unchanged_for_run_with_no_process_identity(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        """`cmd_logs` never calls `_effective_status` (it has no status line
        to print); this proves it is unaffected by identity verification
        by printing its usual log-tail output for a run with no identity
        token recorded, the same as it would before this change."""
        _make_run(
            isolated_runs_root, isolated_log_root, "r30",
            status="running", pid=111, log_text="line-a\nline-b\n",
        )
        rc = agent_run.cmd_logs(argparse.Namespace(name="r30", n=50))
        assert rc == 0
        out = capsys.readouterr().out
        assert out == "line-a\nline-b\n"

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
        token = _mock_verified_alive(monkeypatch)
        _make_run(
            isolated_runs_root, isolated_log_root, "r5",
            status="running", pid=111, log_age_secs=1, process_identity=token,
        )
        rc = agent_run.cmd_watch(_watch_args("r5", as_json=False))
        assert rc == 0
        out = capsys.readouterr().out
        assert out.startswith("name=r5 status=running")
        with pytest.raises(json.JSONDecodeError):
            json.loads(out)


# ---------------------------------------------------------------------------
# zombie detection: watch-local liveness must not trust a matching identity
# for a pid that exited but was never reaped
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not hasattr(os, "fork") or not hasattr(os, "waitpid"),
    reason="requires a POSIX fork/waitpid to construct a real zombie",
)
class TestZombieDetection:
    def test_unwaited_zombie_child_reports_died_and_terminal(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        try:
            # Give the child time to exit and settle into state Z before we
            # probe it; never reaped by this test, so it stays a zombie for
            # the duration of the assertions below.
            deadline = time.time() + 5
            while time.time() < deadline:
                try:
                    stat_text = Path(f"/proc/{pid}/stat").read_text()
                    _before, after = stat_text.rsplit(")", 1)
                    if after.split()[0] == "Z":
                        break
                except OSError:
                    pass
                if sys.platform == "darwin":
                    result = subprocess.run(
                        ["ps", "-o", "stat=", "-p", str(pid)],
                        capture_output=True, text=True,
                    )
                    if result.stdout.strip().startswith("Z"):
                        break
                time.sleep(0.05)
            else:
                pytest.skip("could not observe child in zombie state on this platform")

            token = agent_run._process_identity(pid)
            assert token is not None
            _make_run(
                isolated_runs_root, isolated_log_root, "zombie1",
                status="running", pid=pid, log_age_secs=1, process_identity=token,
            )
            rc = agent_run.cmd_watch(_watch_args("zombie1"))
            assert rc == 0
            payload = json.loads(capsys.readouterr().out)
            assert payload["status"] == "died"
            assert payload["terminal"] is True
        finally:
            os.waitpid(pid, 0)

    def test_live_non_zombie_child_reports_running(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        pid = os.fork()
        if pid == 0:
            time.sleep(5)
            os._exit(0)
        try:
            token = agent_run._process_identity(pid)
            assert token is not None
            _make_run(
                isolated_runs_root, isolated_log_root, "zombie2",
                status="running", pid=pid, log_age_secs=1, process_identity=token,
            )
            rc = agent_run.cmd_watch(_watch_args("zombie2"))
            assert rc == 0
            payload = json.loads(capsys.readouterr().out)
            assert payload["status"] == "running"
            assert payload["terminal"] is False
        finally:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)

    def test_zombie_probe_failure_falls_through_without_raising(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        """The zombie probe's own OS-level read failing (unreadable /proc
        entry, ps failure) must degrade to "not a known zombie" rather than
        propagate, so a probe failure falls through to the existing
        identity-based verdict instead of crashing the poll."""
        token = _mock_verified_alive(monkeypatch, token="linux:7000")
        monkeypatch.setattr(agent_run.platform, "system", lambda: "Linux")

        real_read_text = Path.read_text

        def boom(self, *a, **k):
            if str(self).startswith("/proc/"):
                raise OSError("forced zombie-probe failure")
            return real_read_text(self, *a, **k)

        monkeypatch.setattr(Path, "read_text", boom)
        assert agent_run._watch_pid_is_zombie(222) is False

        _make_run(
            isolated_runs_root, isolated_log_root, "zombie3",
            status="running", pid=222, log_age_secs=1, process_identity=token,
        )
        rc = agent_run.cmd_watch(_watch_args("zombie3"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "running"
        assert payload["terminal"] is False


# ---------------------------------------------------------------------------
# terminal / status derivability
# ---------------------------------------------------------------------------

class TestTerminalDerivability:
    """`terminal` must be exactly derivable from `status` via
    `_watch_is_terminal`, watch-local and independent from the shared
    `TERMINAL_STATUSES`/`KNOWN_NONTERMINAL_STATUSES` reap vocabulary."""

    @pytest.mark.parametrize(
        "status",
        sorted(agent_run.TERMINAL_STATUSES)
        + sorted(agent_run.KNOWN_NONTERMINAL_STATUSES)
        + ["unknown", "some_garbage_status"],
    )
    def test_terminal_matches_watch_is_terminal_for_every_emittable_status(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys, status
    ):
        """Drives `watch` end-to-end for every status it can emit via a raw
        status file (all TERMINAL_STATUSES and KNOWN_NONTERMINAL_STATUSES
        members, "unknown", and one garbage string) and asserts
        `payload["terminal"] == _watch_is_terminal(payload["status"])`
        exactly. The log-preserved, "unverified", and top-level-guard
        "unknown" cases are covered by the dedicated tests above/below
        since they are not reachable via a raw status file alone."""
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        monkeypatch.setattr(agent_run, "_process_identity", lambda _p: "linux:1")
        _make_run(
            isolated_runs_root, isolated_log_root, "tdmatrix",
            status=status, pid=111, log_age_secs=1, process_identity="linux:1",
        )
        rc = agent_run.cmd_watch(_watch_args("tdmatrix"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == status
        assert payload["terminal"] == agent_run._watch_is_terminal(payload["status"])

    def test_log_preserved_branch_status_and_terminal(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        _make_run(
            isolated_runs_root, isolated_log_root, "td1",
            write_state=False, log_text="line1\n",
        )
        rc = agent_run.cmd_watch(_watch_args("td1"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "not running (log preserved)"
        assert payload["status"] == agent_run.WATCH_STATUS_LOG_PRESERVED
        assert payload["terminal"] is True

    @pytest.mark.parametrize(
        "mutate, expected_status",
        [
            (lambda p: p.unlink(), "unknown"),
            (lambda p: p.write_text(""), ""),
        ],
        ids=["missing_status_file", "empty_status_file"],
    )
    def test_unreadable_status_file_is_terminal_as_unrecognized(
        self, isolated_runs_root, isolated_log_root, capsys, mutate, expected_status
    ):
        """A missing status file reads as `"unknown"` and an empty one as
        `""`. Both are outside TERMINAL_STATUSES/KNOWN_NONTERMINAL_STATUSES,
        so both must be terminal — a poller must not wait forever on them."""
        _make_run(
            isolated_runs_root, isolated_log_root, "td2",
            status="running", pid=111,
        )
        mutate(isolated_runs_root / "td2" / "status")
        rc = agent_run.cmd_watch(_watch_args("td2"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == expected_status
        assert payload["terminal"] is True

    def test_unverified_status_is_not_terminal(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        _make_run(
            isolated_runs_root, isolated_log_root, "td4",
            status="running", pid=111, log_age_secs=1,
        )
        rc = agent_run.cmd_watch(_watch_args("td4"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "unverified"
        assert payload["terminal"] is False

    def test_running_status_is_not_terminal(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        token = _mock_verified_alive(monkeypatch)
        _make_run(
            isolated_runs_root, isolated_log_root, "td5",
            status="running", pid=111, log_age_secs=1, process_identity=token,
        )
        rc = agent_run.cmd_watch(_watch_args("td5"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "running"
        assert payload["terminal"] is False

    def test_shared_gc_status_sets_are_not_widened(self):
        """Reap's GC vocabulary must stay exactly as narrow as it is:
        widening either set would make `reap` start deleting state for
        statuses it currently refuses to touch."""
        assert agent_run.TERMINAL_STATUSES == frozenset(
            {"done", "failed", "launch_failed", "died", "killed"}
        )
        assert agent_run.KNOWN_NONTERMINAL_STATUSES == frozenset(
            {"starting", "running", "stalled"}
        )


# ---------------------------------------------------------------------------
# missing state dir / missing run
# ---------------------------------------------------------------------------

class TestUnresolvable:
    def test_unknown_run_exits_2_with_empty_stdout(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        rc = agent_run.cmd_watch(_watch_args("nope"))
        assert rc == 2
        assert capsys.readouterr().out == ""

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
        assert payload["git_error"] == "no_repo_path"
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
        assert payload["git_error"] is None


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
        assert payload["git_error"] == "no_repo_path"
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
        assert payload["git_error"] is None

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
        assert payload["git_error"] == "not_a_repo"

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
        assert payload["git_error"] == "timeout"
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
# git_error discriminator
# ---------------------------------------------------------------------------

def _repo_missing(tmp_path: Path) -> Path:
    return tmp_path / "does-not-exist"


def _repo_plain_dir(tmp_path: Path) -> Path:
    plain = tmp_path / "plain-dir"
    plain.mkdir()
    return plain


def _repo_no_commits(tmp_path: Path) -> Path:
    repo = tmp_path / "empty-repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    return repo


def _repo_dangling_head(tmp_path: Path) -> Path:
    repo = tmp_path / "dangling-head-repo"
    _init_repo(repo)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/does-not-exist\n")
    return repo


def _repo_corrupt_index(tmp_path: Path) -> Path:
    repo = tmp_path / "corrupt-index-repo"
    _init_repo(repo)
    (repo / ".git" / "index").write_text("garbage-not-an-index")
    return repo


def _repo_refs_wiped(tmp_path: Path) -> Path:
    repo = tmp_path / "refs-wiped-repo"
    _init_repo(repo)
    for entry in (repo / ".git" / "refs").rglob("*"):
        if entry.is_file():
            entry.unlink()
    packed_refs = repo / ".git" / "packed-refs"
    if packed_refs.exists():
        packed_refs.unlink()
    return repo


def _repo_objects_removed(tmp_path: Path) -> Path:
    repo = tmp_path / "objects-removed-repo"
    _init_repo(repo)
    for entry in (repo / ".git" / "objects").rglob("*"):
        if entry.is_file():
            entry.unlink()
    return repo


def _repo_blob_removed(tmp_path: Path) -> Path:
    """Only the blob under HEAD is gone: the stat-cache-driven reads can skip
    re-reading it, so this is the case that requires the connectivity fsck."""
    repo = tmp_path / "blob-removed-repo"
    _init_repo(repo)
    blob = subprocess.run(
        ["git", "rev-parse", "HEAD:a.txt"],
        cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()
    (repo / ".git" / "objects" / blob[:2] / blob[2:]).unlink()
    return repo


class TestGitErrorDiscriminator:
    # Each entry reproduces a distinct way a repo can be unobservable, paired
    # with the discriminator it must produce. ANY_ERROR means "some non-null
    # discriminator" — the damage is detected, but git's exact wording for it
    # is not part of the contract.
    ANY_ERROR = object()

    @pytest.mark.parametrize(
        "damage, expected",
        [
            pytest.param(_repo_missing, "no_repo_path", id="nonexistent_path"),
            pytest.param(_repo_plain_dir, "not_a_repo", id="plain_directory"),
            pytest.param(_repo_no_commits, "no_commits", id="init_zero_commits"),
            pytest.param(_repo_dangling_head, "git_failed", id="dangling_head"),
            pytest.param(_repo_corrupt_index, "git_failed", id="corrupt_index"),
            pytest.param(_repo_refs_wiped, ANY_ERROR, id="refs_wiped"),
            pytest.param(_repo_objects_removed, ANY_ERROR, id="objects_removed"),
            pytest.param(_repo_blob_removed, ANY_ERROR, id="tracked_blob_removed"),
        ],
    )
    def test_unobservable_repo_yields_null_git_and_a_discriminator(
        self, isolated_runs_root, isolated_log_root, tmp_path, capsys, damage, expected
    ):
        repo = damage(tmp_path)
        _make_run(
            isolated_runs_root, isolated_log_root, "ge1",
            status="done", pid=1, exit_code=0, ended_age_secs=1,
            cwd=str(repo),
        )
        rc = agent_run.cmd_watch(_watch_args("ge1"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["git"] is None
        if expected is self.ANY_ERROR:
            assert payload["git_error"] is not None
        else:
            assert payload["git_error"] == expected

    def test_timeout_expired_yields_timeout(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch, capsys
    ):
        repo = tmp_path / "repo"
        _init_repo(repo)
        _make_run(
            isolated_runs_root, isolated_log_root, "ge4",
            status="done", pid=1, exit_code=0, ended_age_secs=1,
            cwd=str(repo),
        )

        def _wedged(*_args, **_kwargs):
            raise subprocess.TimeoutExpired(cmd="git", timeout=2.5)

        monkeypatch.setattr(agent_run.subprocess, "run", _wedged)
        rc = agent_run.cmd_watch(_watch_args("ge4"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["git"] is None
        assert payload["git_error"] == "timeout"

    def test_git_binary_missing_yields_git_missing(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch, capsys
    ):
        repo = tmp_path / "repo"
        _init_repo(repo)
        _make_run(
            isolated_runs_root, isolated_log_root, "ge5",
            status="done", pid=1, exit_code=0, ended_age_secs=1,
            cwd=str(repo),
        )

        def _no_git(*_args, **_kwargs):
            raise FileNotFoundError("git")

        monkeypatch.setattr(agent_run.subprocess, "run", _no_git)
        rc = agent_run.cmd_watch(_watch_args("ge5"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["git"] is None
        assert payload["git_error"] == "git_missing"

    def test_healthy_repo_yields_fully_populated_git_and_null_git_error(
        self, isolated_runs_root, isolated_log_root, tmp_path, capsys
    ):
        repo = tmp_path / "repo"
        _init_repo(repo)
        _make_run(
            isolated_runs_root, isolated_log_root, "ge6",
            status="done", pid=1, exit_code=0, ended_age_secs=1,
            cwd=str(repo),
        )
        rc = agent_run.cmd_watch(_watch_args("ge6"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["git"] is not None
        assert payload["git"]["head"]
        assert payload["git_error"] is None

    def test_git_error_present_in_missing_state_dir_branch(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        _make_run(
            isolated_runs_root, isolated_log_root, "ge7",
            write_state=False, log_text="line1\n",
        )
        rc = agent_run.cmd_watch(_watch_args("ge7"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "git_error" in payload
        assert payload["git"] is None
        assert payload["git_error"] == "no_repo_path"

    def test_git_error_present_in_observation_error_guard_branch(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        _make_run(
            isolated_runs_root, isolated_log_root, "ge8",
            status="running", pid=111, log_age_secs=1,
        )

        def _boom(*_args, **_kwargs):
            raise RuntimeError("synthetic failure for git_error guard test")

        monkeypatch.setattr(agent_run, "_effective_status", _boom)
        rc = agent_run.cmd_watch(_watch_args("ge8"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "git_error" in payload
        assert payload["git"] is None
        assert payload["git_error"] is None
        assert payload["observation_error"] == "RuntimeError: synthetic failure for git_error guard test"


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
        agent_run._watch_run_git_checked(repo, ["status", "--porcelain"])
        assert marker.exists() is False

    def test_git_dir_env_is_ignored(self, tmp_path, monkeypatch):
        repo_a = tmp_path / "repo_a"
        repo_b = tmp_path / "repo_b"
        _init_repo(repo_a)
        _init_repo(repo_b)
        (repo_b / "b.txt").write_text("more\n")
        _git(repo_b, "add", "b.txt")
        _git(repo_b, "commit", "-q", "-m", "second")

        expected_head = agent_run._watch_run_git_checked(
            repo_a, ["rev-parse", "--short", "HEAD"]
        ).stdout.strip()

        monkeypatch.setenv("GIT_DIR", str(repo_b / ".git"))
        monkeypatch.setenv("GIT_WORK_TREE", str(repo_b))
        result = agent_run._watch_run_git_checked(repo_a, ["rev-parse", "--short", "HEAD"])
        assert result.stdout.strip() == expected_head

    def test_undecodable_output_does_not_raise(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        _init_repo(repo)

        class _FakeResult:
            stdout = b"A\xff\xfeB"

        def _fake_run(*_args, **_kwargs):
            return _FakeResult()

        monkeypatch.setattr(agent_run.subprocess, "run", _fake_run)
        result = agent_run._watch_run_git_checked(repo, ["status", "--porcelain"])
        assert result.stdout == "A\ufffd\ufffdB"


# ---------------------------------------------------------------------------
# shared time budget across the five git calls
# ---------------------------------------------------------------------------

class TestGitTotalBudget:
    def test_total_wall_time_is_bounded(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()

        class _FakeResult:
            stdout = b"0\n"

        def _slow_run(*_args, **kwargs):
            time.sleep(kwargs["timeout"])
            return _FakeResult()

        monkeypatch.setattr(agent_run.subprocess, "run", _slow_run)
        started_at = datetime.now(timezone.utc) - timedelta(hours=1)
        start = time.monotonic()
        agent_run._watch_git_facts_checked(repo, started_at)
        elapsed = time.monotonic() - start
        assert elapsed < agent_run.WATCH_GIT_TOTAL_BUDGET_SECONDS + 2.0

    def test_per_call_cap_holds(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()

        class _FakeResult:
            stdout = b"0\n"

        seen_timeouts: list[float] = []

        def _fast_run(*_args, **kwargs):
            seen_timeouts.append(kwargs["timeout"])
            return _FakeResult()

        monkeypatch.setattr(agent_run.subprocess, "run", _fast_run)
        started_at = datetime.now(timezone.utc) - timedelta(hours=1)
        result = agent_run._watch_git_facts_checked(repo, started_at).facts
        assert result is not None
        assert len(seen_timeouts) == 7
        assert all(t <= agent_run.WATCH_GIT_SUBPROCESS_TIMEOUT_SECONDS for t in seen_timeouts)

    def test_budget_exhaustion_returns_none(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()

        class _FakeResult:
            stdout = b"0\n"

        def _slow_run(*_args, **_kwargs):
            time.sleep(0.3)
            return _FakeResult()

        monkeypatch.setattr(agent_run.subprocess, "run", _slow_run)
        monkeypatch.setattr(agent_run, "WATCH_GIT_TOTAL_BUDGET_SECONDS", 0.05)
        started_at = datetime.now(timezone.utc) - timedelta(hours=1)
        result = agent_run._watch_git_facts_checked(repo, started_at).facts
        assert result is None

    def test_happy_path_returns_fully_populated_facts(self, tmp_path):
        repo = tmp_path / "repo"
        _init_repo(repo)
        expected_head = agent_run._watch_run_git_checked(
            repo, ["rev-parse", "--short", "HEAD"]
        ).stdout.strip()
        started_at = datetime.now(timezone.utc) - timedelta(hours=1)
        result = agent_run._watch_git_facts_checked(repo, started_at).facts
        assert result == {
            "head": expected_head,
            "dirty": False,
            "files_changed": 0,
            "insertions": 0,
            "deletions": 0,
            "commits_since_start": 1,
            "last_commit_age_s": result["last_commit_age_s"],
            "toplevel": str(repo.resolve()),
        }
        assert result["last_commit_age_s"] is not None
        assert result["last_commit_age_s"] >= 0.0


class TestGitToplevel:
    def test_repo_at_its_own_root(self, tmp_path):
        repo = tmp_path / "repo"
        _init_repo(repo)
        started_at = datetime.now(timezone.utc) - timedelta(hours=1)
        outcome = agent_run._watch_git_facts_checked(repo, started_at)
        assert outcome.git_error is None
        assert outcome.facts is not None
        assert outcome.facts["toplevel"] == str(repo.resolve())

    def test_subdirectory_of_its_own_repo_reports_repo_root(self, tmp_path):
        repo = tmp_path / "repo"
        _init_repo(repo)
        subdir = repo / "sub"
        subdir.mkdir()
        (subdir / "b.txt").write_text("more\n")
        _git(repo, "add", "sub/b.txt")
        _git(repo, "commit", "-q", "-m", "second")
        started_at = datetime.now(timezone.utc) - timedelta(hours=1)
        outcome = agent_run._watch_git_facts_checked(subdir, started_at)
        assert outcome.git_error is None
        assert outcome.facts is not None
        assert outcome.facts["toplevel"] == str(repo.resolve())

    def test_plain_subdir_inside_unrelated_outer_repo_reports_outer_toplevel(self, tmp_path):
        outer = tmp_path / "outer"
        _init_repo(outer)
        plain_subdir = outer / "plain-subdir"
        plain_subdir.mkdir()
        started_at = datetime.now(timezone.utc) - timedelta(hours=1)
        outcome = agent_run._watch_git_facts_checked(plain_subdir, started_at)
        assert outcome.git_error is None
        assert outcome.facts is not None
        assert outcome.facts["toplevel"] == str(outer.resolve())


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

    def test_future_mtime_beyond_tolerance_degrades_to_null_age_not_growing(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        # A log mtime an hour in the future (clock skew / NFS server-side
        # timestamp) must not read as freshly growing — that is precisely
        # the wrong direction for a contract that fails toward firing.
        _make_run(
            isolated_runs_root, isolated_log_root, "r15b",
            status="running", pid=1, log_age_secs=-3600.0,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        agent_run.cmd_watch(_watch_args("r15b"))
        payload = json.loads(capsys.readouterr().out)
        assert payload["log"]["mtime_age_s"] is None
        assert payload["log"]["growing"] is False

    def test_small_future_mtime_within_tolerance_still_clamps_to_zero(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        _make_run(
            isolated_runs_root, isolated_log_root, "r15c",
            status="running", pid=1,
            log_age_secs=-(agent_run.WATCH_LOG_FUTURE_MTIME_TOLERANCE_SECONDS / 2),
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        agent_run.cmd_watch(_watch_args("r15c"))
        payload = json.loads(capsys.readouterr().out)
        assert payload["log"]["mtime_age_s"] == 0.0
        assert payload["log"]["growing"] is True

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
# non-regular log paths (FIFO, directory, symlinks) must never be opened
# ---------------------------------------------------------------------------

_DEGRADED_SIGNALS = {
    "repeated_error": None,
    "distinct_files_read": 0,
    "top_repeated_read": None,
}


class TestRegularLogPath:
    def test_fifo_log_is_rejected_by_the_helper(
        self, isolated_runs_root, isolated_log_root
    ):
        if not hasattr(os, "mkfifo"):
            pytest.skip("os.mkfifo unavailable on this platform")
        _, ld = _make_run(
            isolated_runs_root, isolated_log_root, "fifo1", write_log=False,
        )
        os.mkfifo(ld / "log")
        with agent_run._watch_open_validated_log(agent_run._log_file_for("fifo1")) as f:
            assert f is None

    def test_fifo_log_end_to_end_returns_instead_of_hanging(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        if not hasattr(os, "mkfifo"):
            pytest.skip("os.mkfifo unavailable on this platform")
        _, ld = _make_run(
            isolated_runs_root, isolated_log_root, "fifo2",
            status="running", pid=1, write_log=False,
        )
        os.mkfifo(ld / "log")
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)

        result: dict = {}

        def _call() -> None:
            result["rc"] = agent_run.cmd_watch(_watch_args("fifo2"))

        thread = threading.Thread(target=_call, daemon=True)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive(), "watch hung reading a FIFO log with no writer"
        assert result["rc"] == 0
        payload = json.loads(capsys.readouterr().out)
        assert set(payload.keys()) == WATCH_CONTRACT_KEYS
        assert payload["log"] is None
        assert payload["signals"] == _DEGRADED_SIGNALS

    def test_directory_in_place_of_log_is_rejected(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        _, ld = _make_run(
            isolated_runs_root, isolated_log_root, "dirlog",
            status="running", pid=1, write_log=False,
        )
        (ld / "log").mkdir()
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        with agent_run._watch_open_validated_log(agent_run._log_file_for("dirlog")) as f:
            assert f is None
        rc = agent_run.cmd_watch(_watch_args("dirlog"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["log"] is None
        assert payload["signals"] == _DEGRADED_SIGNALS

    def test_symlink_to_a_regular_file_is_accepted(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        _, ld = _make_run(
            isolated_runs_root, isolated_log_root, "symreal",
            status="running", pid=1, log_age_secs=1, log_text="line one\nline two\n",
        )
        target = ld / "log.real"
        (ld / "log").rename(target)
        (ld / "log").symlink_to(target)
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        with agent_run._watch_open_validated_log(agent_run._log_file_for("symreal")) as f:
            assert f is not None
        rc = agent_run.cmd_watch(_watch_args("symreal"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["log"]["bytes"] == len("line one\nline two\n")
        assert payload["log"]["lines"] == 2

    def test_symlink_to_a_fifo_is_rejected(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        if not hasattr(os, "mkfifo"):
            pytest.skip("os.mkfifo unavailable on this platform")
        _, ld = _make_run(
            isolated_runs_root, isolated_log_root, "symfifo",
            status="running", pid=1, write_log=False,
        )
        fifo_path = ld / "log.fifo"
        os.mkfifo(fifo_path)
        (ld / "log").symlink_to(fifo_path)
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        with agent_run._watch_open_validated_log(agent_run._log_file_for("symfifo")) as f:
            assert f is None
        rc = agent_run.cmd_watch(_watch_args("symfifo"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["log"] is None
        assert payload["signals"] == _DEGRADED_SIGNALS

    def test_ordinary_regular_file_is_unchanged(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        _, ld = _make_run(
            isolated_runs_root, isolated_log_root, "regular1",
            status="running", pid=1, log_age_secs=1, log_text="line one\nline two\n",
        )
        with agent_run._watch_open_validated_log(agent_run._log_file_for("regular1")) as f:
            assert f is not None
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        rc = agent_run.cmd_watch(_watch_args("regular1"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["log"]["bytes"] == len("line one\nline two\n")
        assert payload["log"]["lines"] == 2

    def test_inode_swapped_between_stat_and_read_returns_instead_of_hanging(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        """TOCTOU regression: something replaces the regular log with a FIFO
        after validation would have passed by path but before the fd-based
        open+fstat in _watch_open_validated_log runs. Structured to fail
        (hang, caught by the join timeout) rather than wedge the suite if
        the descriptor-based validation regresses to path-based validation."""
        if not hasattr(os, "mkfifo"):
            pytest.skip("os.mkfifo unavailable on this platform")
        _, ld = _make_run(
            isolated_runs_root, isolated_log_root, "toctou1",
            status="done", pid=1, log_age_secs=1, log_text="line one\n",
        )
        log_path = ld / "log"
        real_open = agent_run._watch_open_validated_log

        def swap_then_open(path):
            if path == log_path and log_path.exists() and not log_path.is_symlink():
                log_path.unlink()
                os.mkfifo(log_path)
            return real_open(path)

        monkeypatch.setattr(agent_run, "_watch_open_validated_log", swap_then_open)

        result: dict = {}

        def _call() -> None:
            result["rc"] = agent_run.cmd_watch(_watch_args("toctou1"))

        thread = threading.Thread(target=_call, daemon=True)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive(), "watch_still_blocked"
        assert result["rc"] == 0
        payload = json.loads(capsys.readouterr().out)
        assert set(payload.keys()) == WATCH_CONTRACT_KEYS
        assert payload["log"] is None
        assert payload["signals"] == _DEGRADED_SIGNALS


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

    @pytest.mark.parametrize(
        "noisy_line",
        [
            "  0 errors, 0 warnings",
            "PASS test_exception_is_swallowed",
            "| Error | Count |",
            "step NOTFAILED ok",
            "Read /src/errors.py",
            "errors: 0",
        ],
    )
    def test_repeated_error_ignores_healthy_lookalikes(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys, noisy_line
    ):
        text = (noisy_line + "\n") * agent_run.WATCH_REPEAT_THRESHOLD
        _make_run(
            isolated_runs_root, isolated_log_root, "r23",
            status="running", pid=1, log_text=text, log_age_secs=1,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        agent_run.cmd_watch(_watch_args("r23"))
        payload = json.loads(capsys.readouterr().out)
        assert payload["signals"]["repeated_error"] is None

    @pytest.mark.parametrize(
        "error_line",
        [
            "Error: ENOENT: no such file",
            "src/foo.py: error: bad thing",
            "Traceback (most recent call last):",
        ],
    )
    def test_repeated_error_detects_genuine_errors(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys, error_line
    ):
        text = (error_line + "\n") * agent_run.WATCH_REPEAT_THRESHOLD
        _make_run(
            isolated_runs_root, isolated_log_root, "r24",
            status="running", pid=1, log_text=text, log_age_secs=1,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        agent_run.cmd_watch(_watch_args("r24"))
        payload = json.loads(capsys.readouterr().out)
        assert payload["signals"]["repeated_error"] == error_line

    def test_repeated_read_ansi_wrapped_lines_collapse_to_clean_path(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        plain = "Read /a/one.py"
        colour = "Read \x1b[36m/a/one.py\x1b[0m"
        lines = [plain, colour, plain, colour, plain, colour]
        text = "\n".join(lines) + "\n"
        _make_run(
            isolated_runs_root, isolated_log_root, "r25",
            status="running", pid=1, log_text=text, log_age_secs=1,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        agent_run.cmd_watch(_watch_args("r25"))
        payload = json.loads(capsys.readouterr().out)
        assert payload["signals"]["distinct_files_read"] == 1
        assert payload["signals"]["top_repeated_read"] == {
            "path": "/a/one.py",
            "count": 6,
        }

    def test_repeated_read_rejects_prose_target(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        text = ("Please Read carefully\n") * 3
        _make_run(
            isolated_runs_root, isolated_log_root, "r26",
            status="running", pid=1, log_text=text, log_age_secs=1,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        agent_run.cmd_watch(_watch_args("r26"))
        payload = json.loads(capsys.readouterr().out)
        assert payload["signals"]["top_repeated_read"] is None

    def test_repeated_error_detected_across_cr_redraws(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        # A TUI frame redrawing in place puts repeats on one physical line.
        text = "Error: boom\r" * 4 + "\n"
        _make_run(
            isolated_runs_root, isolated_log_root, "r27",
            status="running", pid=1, log_text=text, log_age_secs=1,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        agent_run.cmd_watch(_watch_args("r27"))
        payload = json.loads(capsys.readouterr().out)
        assert payload["signals"]["repeated_error"] == "Error: boom"


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

    @pytest.mark.parametrize(
        "corrupt",
        [
            pytest.param(lambda p: p.write_text("\u00b2\n"), id="non_ascii_digit"),
            pytest.param(lambda p: p.write_bytes(b"\xff\xfe123"), id="invalid_utf8"),
            pytest.param(lambda p: (p.unlink(), p.mkdir()), id="directory_in_place"),
            pytest.param(lambda p: p.chmod(0o000), id="mode_000", marks=ROOT_SKIP),
        ],
    )
    def test_unreadable_pid_file_degrades_to_null_pid(
        self, isolated_runs_root, isolated_log_root, capsys, corrupt
    ):
        """Every way a pid file can be unreadable or unparseable degrades
        `pid` to null and still emits the full contract at exit 0."""
        sd, _ld = _make_run(
            isolated_runs_root, isolated_log_root, "u1",
            status="running", pid=111,
        )
        target = sd / "pid"
        corrupt(target)
        try:
            rc = agent_run.cmd_watch(_watch_args("u1"))
        finally:
            if target.is_file():
                target.chmod(0o644)
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

    @ROOT_SKIP
    def test_log_file_at_mode_000_degrades_line_count(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        """Log facts and the signal scan both open through the same
        validated fd, so a permission-denied open degrades the whole `log`
        object to null rather than a byte count paired with lines=0."""
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
        assert payload["log"] is None
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
        assert payload["terminal"] is True
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
        assert payload["terminal"] is True
        assert payload["pid"] is None
        assert payload["log"] is None
        assert payload["git"] is None
        assert payload["git_error"] is None


class TestExitCodes:
    """The three ``cmd_watch`` exit codes must stay distinct: 2 for an
    unresolvable run name (nothing printed), 0 for a valid contract (even
    one with ``observation_error`` set), and cmd_status must be unaffected."""

    def test_unresolvable_name_exits_2_with_empty_stdout(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        rc = agent_run.cmd_watch(_watch_args("does-not-exist"))
        assert rc == 2
        assert capsys.readouterr().out == ""

    def test_internal_observation_failure_exits_0_with_valid_contract(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        _make_run(
            isolated_runs_root, isolated_log_root, "exitcode-crash",
            status="running", pid=111, log_age_secs=1,
        )

        def _boom(*_args, **_kwargs):
            raise RuntimeError("synthetic crash")

        monkeypatch.setattr(agent_run, "_effective_status", _boom)
        rc = agent_run.cmd_watch(_watch_args("exitcode-crash"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert set(payload.keys()) == WATCH_CONTRACT_KEYS
        assert payload["observation_error"] is not None

    def test_cmd_status_unaffected_by_watch_exit_code_split(
        self, isolated_runs_root, isolated_log_root
    ):
        with pytest.raises(SystemExit) as exc_info:
            agent_run.cmd_status(argparse.Namespace(name="does-not-exist"))
        assert exc_info.value.code != 0


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

    def test_unreadable_cwd_skips_cwd_file_without_stranding_run(
        self, isolated_runs_root, isolated_log_root, monkeypatch, tmp_path, capsys
    ):
        real_cwd = Path.cwd

        def _raising_cwd():
            raise FileNotFoundError("launch directory removed")

        monkeypatch.setattr(agent_run.Path, "cwd", staticmethod(_raising_cwd))
        ns = argparse.Namespace(
            command=["true"],
            interactive=False,
            prompt_file=None,
            echo=False,
            echo_interval=2.0,
            submit_mode=None,
            idle_timeout=None,
            name="launched_no_cwd",
        )
        agent_run.cmd_launch(ns)
        monkeypatch.setattr(agent_run.Path, "cwd", staticmethod(real_cwd))

        sd = isolated_runs_root / "launched_no_cwd"
        deadline = time.monotonic() + 5.0
        while not (sd / "pid").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not (sd / "cwd").exists()
        # The run must have fully published, not be stranded as "starting"
        # with no pid behind it.
        status = (sd / "status").read_text().strip()
        pid_exists = (sd / "pid").exists()
        assert not (status == "starting" and not pid_exists)
        capsys.readouterr()

        rc = agent_run.cmd_watch(_watch_args("launched_no_cwd"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["repo"] is None
        assert payload["git"] is None
        assert set(payload.keys()) == WATCH_CONTRACT_KEYS


# ---------------------------------------------------------------------------
# _watch_tail_lines: byte-bounded backward scan
# ---------------------------------------------------------------------------

class TestWatchTailLinesByteBound:
    def test_ordinary_tail_is_unchanged(self, tmp_path):
        lines = [f"line {i}\n" for i in range(500)]
        log = tmp_path / "log"
        log.write_text("".join(lines))
        result = agent_run._watch_tail_lines(log, 50)
        assert result == [f"line {i}" for i in range(450, 500)]

    def test_byte_cap_enforced_on_newline_free_file(self, tmp_path):
        log = tmp_path / "log"
        log.write_bytes(b"x" * (2 * 1024 * 1024))
        result = agent_run._watch_tail_lines(log, 200)
        total_chars = sum(len(line) for line in result)
        max_expected = agent_run.WATCH_TAIL_MAX_BYTES + agent_run.WATCH_TAIL_READ_BLOCK_BYTES
        assert total_chars <= max_expected

    def test_cap_with_sparse_newlines(self, tmp_path):
        log = tmp_path / "log"
        with log.open("wb") as f:
            f.write(b"first\n")
            f.write(b"second\n")
            f.write(b"y" * (2 * 1024 * 1024))
        result = agent_run._watch_tail_lines(log, 200)
        total_chars = sum(len(line) for line in result)
        max_expected = agent_run.WATCH_TAIL_MAX_BYTES + agent_run.WATCH_TAIL_READ_BLOCK_BYTES
        assert total_chars <= max_expected
        assert "first" not in result

    def test_small_file_returns_every_line(self, tmp_path):
        log = tmp_path / "log"
        log.write_text("a\nb\nc\n")
        result = agent_run._watch_tail_lines(log, 200)
        assert result == ["a", "b", "c"]

    def test_empty_file_returns_empty_list(self, tmp_path):
        log = tmp_path / "log"
        log.write_bytes(b"")
        result = agent_run._watch_tail_lines(log, 200)
        assert result == []


# ---------------------------------------------------------------------------
# _is_dir_safe: the existence check ahead of cmd_watch's/cmd_status's guard
# ---------------------------------------------------------------------------

def _write_file(parent: Path) -> Path:
    f = parent / "f"
    f.write_text("x")
    return f


class TestIsDirSafe:
    ROOT_SKIP = pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="root ignores permission bits",
    )

    def test_returns_true_for_a_real_directory(self, tmp_path):
        assert agent_run._is_dir_safe(tmp_path) is True

    @pytest.mark.parametrize(
        "make",
        [
            pytest.param(lambda p: p / "nope", id="missing_path"),
            pytest.param(_write_file, id="regular_file"),
        ],
    )
    def test_returns_false_for_a_non_directory(self, tmp_path, make):
        assert agent_run._is_dir_safe(make(tmp_path)) is False

    @ROOT_SKIP
    def test_returns_false_instead_of_raising_when_parent_is_unreadable(self, tmp_path):
        parent = tmp_path / "unreadable"
        parent.mkdir()
        target = parent / "child"
        target.mkdir()
        parent.chmod(0o000)
        try:
            assert agent_run._is_dir_safe(target) is False
        finally:
            parent.chmod(0o755)

    @ROOT_SKIP
    def test_cmd_watch_exits_zero_with_full_payload_when_state_root_unreadable(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        _make_run(
            isolated_runs_root, isolated_log_root, "w1",
            status="running", pid=111, log_age_secs=1,
        )
        isolated_runs_root.chmod(0o000)
        try:
            rc = agent_run.cmd_watch(_watch_args("w1"))
        finally:
            isolated_runs_root.chmod(0o755)
        assert rc == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert set(payload.keys()) == WATCH_CONTRACT_KEYS
        assert isinstance(payload["observation_error"], str)
        assert payload["observation_error"]

    def test_cmd_watch_unresolvable_run_name_exits_2_with_empty_stdout(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        rc = agent_run.cmd_watch(_watch_args("does-not-exist"))
        assert rc == 2
        assert capsys.readouterr().out == ""

    def test_cmd_watch_eloop_symlink_state_dir_treated_as_absent(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        state_link = isolated_runs_root / "w2"
        state_link.symlink_to(state_link)  # self-referential -> ELOOP on stat
        log_dir = isolated_log_root / "w2"
        log_dir.mkdir(parents=True)
        (log_dir / "log").write_text("line1\n")
        rc = agent_run.cmd_watch(_watch_args("w2"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "not running (log preserved)"
        assert payload["observation_error"] is None

    @ROOT_SKIP
    def test_cmd_status_does_not_raise_when_state_root_unreadable(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        _make_run(
            isolated_runs_root, isolated_log_root, "w3",
            status="running", pid=111, log_age_secs=1,
        )
        isolated_runs_root.chmod(0o000)
        try:
            rc = agent_run.cmd_status(argparse.Namespace(name="w3"))
        finally:
            isolated_runs_root.chmod(0o755)
        assert rc == 0
        out = capsys.readouterr().out
        assert "not running (log preserved)" in out
