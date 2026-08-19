"""Tests for `agent-run watch` — the stateless, read-only JSON fact contract."""
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
    "session", "scratch",
}

# The `signals` object emitted whenever the log could not be read at all.
_DEGRADED_SIGNALS = {
    "repeated_error": None,
    "distinct_files_read": 0,
    "top_repeated_read": None,
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
    launch_head: Optional[str] = None,
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
        if launch_head is not None:
            (sd / "launch_head").write_text(launch_head + "\n")
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


def _git(repo: Path, *args: str, date: Optional[str] = None) -> None:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t.com"}
    if date is not None:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = date
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, env=env)


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    (path / "a.txt").write_text("hello\n")
    _git(path, "add", "a.txt")
    _git(path, "commit", "-q", "-m", "initial")


def _repo_with_gitignored_finding(repo: Path, subdir: str) -> Path:
    """A clean repo whose only activity is a file under a gitignored dir.

    This is the blind spot the scratch scan exists for: git reports zero
    commits, zero untracked files and a clean worktree, yet the run is
    actively writing.
    """
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("tracked\n")
    (repo / ".gitignore").write_text(".taskdocs/\n")
    _git(repo, "add", "a.txt", ".gitignore")
    _git(repo, "commit", "-q", "-m", "initial")

    finding = repo / ".taskdocs" / subdir / "findings-001.md"
    finding.parent.mkdir(parents=True)
    finding.write_text("analysis\n" * 1000)
    return finding


_REAL_SCANDIR = os.scandir


def _patch_scandir(monkeypatch, hook) -> None:
    """Replace ``os.scandir`` with one that passes each entry through *hook*.

    The replacement stays lazy — one entry per ``__next__`` — so a scan that
    stops part-way through a directory stops doing work part-way too, which
    is what the budget tests measure.
    """

    class _HookedScandir:
        def __init__(self, path):
            self._it = _REAL_SCANDIR(path)

        def __iter__(self):
            return self

        def __next__(self):
            return hook(next(self._it))

        def __enter__(self):
            self._it.__enter__()
            return self

        def __exit__(self, *args):
            return self._it.__exit__(*args)

    monkeypatch.setattr(os, "scandir", _HookedScandir)


# ---------------------------------------------------------------------------
# status / terminal mapping
# ---------------------------------------------------------------------------

class TestStatusMapping:
    @pytest.mark.parametrize(
        "zombie,live_identity,expected",
        [(True, "linux:old", "died"), (False, "linux:new", "died"), (False, None, "unverified")],
    )
    def test_starting_pid_is_verified(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys,
        zombie, live_identity, expected,
    ):
        _make_run(
            isolated_runs_root, isolated_log_root, "startingcheck",
            status="starting", pid=111, process_identity="linux:old",
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: True)
        monkeypatch.setattr(agent_run, "_watch_pid_is_zombie", lambda _pid: zombie)
        monkeypatch.setattr(agent_run, "_process_identity", lambda _pid: live_identity)

        agent_run.cmd_watch(_watch_args("startingcheck"))

        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == expected
        assert payload["terminal"] is (expected == "died")

    @pytest.mark.parametrize("raw_pid", ["0", "-1", "-42"])
    def test_nonpositive_pid_is_not_probed_or_published(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys, raw_pid
    ):
        sd, _ = _make_run(
            isolated_runs_root, isolated_log_root, "badpid", status="running"
        )
        (sd / "pid").write_text(raw_pid)
        monkeypatch.setattr(
            agent_run.os, "kill", lambda *_args: pytest.fail("used special PID semantics")
        )

        agent_run.cmd_watch(_watch_args("badpid"))

        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "died"
        assert payload["pid"] is None
        assert payload["terminal"] is True

    def test_verified_alive_pid_reports_running_and_not_terminal(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        token = _mock_verified_alive(monkeypatch, token="linux:5000")
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

    @pytest.mark.parametrize(
        "recorded_identity, live_identity",
        [
            pytest.param(None, "linux:1111", id="no_token_recorded"),
            pytest.param("linux:1111", None, id="live_identity_unreadable"),
        ],
    )
    def test_unconfirmable_identity_reports_unverified_and_not_terminal(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys,
        recorded_identity, live_identity,
    ):
        """An alive pid whose identity cannot be confirmed — either way round
        — must never be reported as a healthy "running" run."""
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        monkeypatch.setattr(agent_run, "_process_identity", lambda _p: live_identity)
        _make_run(
            isolated_runs_root, isolated_log_root, "r27",
            status="running", pid=111, log_age_secs=1,
            process_identity=recorded_identity,
        )
        rc = agent_run.cmd_watch(_watch_args("r27"))
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
        """`cmd_logs` never calls `_effective_status`, so identity
        verification cannot affect it: it prints its usual log tail for a run
        with no identity token recorded."""
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

    def test_linux_zombie_detected_via_proc_stat(self, monkeypatch):
        """With platform forced to Linux, _watch_pid_is_zombie reads
        /proc/<pid>/stat and returns True when the state field is 'Z'.

        Runs on all platforms via a platform stub; no real zombie needed."""
        monkeypatch.setattr(agent_run.platform, "system", lambda: "Linux")
        # Minimal /proc/stat line: fields[0] after the comm field is the state.
        # Format: "<pid> (<comm>) <state> ..."
        fake_stat = "222 (myproc) Z 1 222 222 0\n"
        monkeypatch.setattr(Path, "read_text", lambda _self: fake_stat)
        assert agent_run._watch_pid_is_zombie(222) is True

    def test_darwin_zombie_detected_via_ps(self, monkeypatch):
        """With platform forced to Darwin, _watch_pid_is_zombie uses
        _ps_field(pid, 'stat') and returns True when the first character is 'Z'.

        Runs on all platforms via a platform stub and subprocess.run stub."""
        monkeypatch.setattr(agent_run.platform, "system", lambda: "Darwin")

        class PsResult:
            stdout = "Z+\n"
            returncode = 0

        monkeypatch.setattr(
            agent_run.subprocess, "run",
            lambda cmd, **kwargs: PsResult(),
        )
        assert agent_run._watch_pid_is_zombie(222) is True

    def test_darwin_non_zombie_returns_false(self, monkeypatch):
        """A running process (state 'S') must not be mistaken for a zombie."""
        monkeypatch.setattr(agent_run.platform, "system", lambda: "Darwin")

        class PsResult:
            stdout = "S\n"
            returncode = 0

        monkeypatch.setattr(
            agent_run.subprocess, "run",
            lambda cmd, **kwargs: PsResult(),
        )
        assert agent_run._watch_pid_is_zombie(222) is False


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
        """Drives `watch` end-to-end for every status reachable via a raw
        status file and asserts `payload["terminal"] ==
        _watch_is_terminal(payload["status"])` exactly. The log-preserved,
        "unverified" and top-level-guard "unknown" statuses are not reachable
        that way and are covered by their own tests."""
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
        assert payload["status"] == agent_run.WATCH_STATUS_LOG_PRESERVED == "not running (log preserved)"
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

    def test_commits_since_start_counts_by_head_movement_not_timestamps(
        self, isolated_runs_root, isolated_log_root, tmp_path, capsys
    ):
        """A commit's author/committer date is controlled by the committing
        process and can be backdated, so the count must come from HEAD
        movement since launch_head — the pre-launch commit excluded, the
        backdated post-launch one still counted."""
        repo = tmp_path / "repo"
        _init_repo(repo)
        launch_head = agent_run._watch_run_git_checked(
            repo, ["rev-parse", "HEAD"]
        ).stdout.strip()
        _make_run(
            isolated_runs_root, isolated_log_root, "r13",
            status="done", pid=1, exit_code=0, ended_age_secs=1,
            started_age_secs=5, cwd=str(repo), launch_head=launch_head,
        )
        (repo / "b.txt").write_text("more\n")
        _git(repo, "add", "b.txt")
        _git(
            repo, "commit", "-q", "-m", "backdated",
            date="2000-01-01T00:00:00",
        )
        rc = agent_run.cmd_watch(_watch_args("r13"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["git"]["commits_since_start"] == 1
        assert payload["git"]["last_commit_age_s"] is not None

    def test_commits_since_start_is_null_without_a_recorded_launch_head(
        self, isolated_runs_root, isolated_log_root, tmp_path, capsys
    ):
        repo = tmp_path / "repo"
        _init_repo(repo)
        _make_run(
            isolated_runs_root, isolated_log_root, "r13b",
            status="done", pid=1, exit_code=0, ended_age_secs=1,
            started_age_secs=5, cwd=str(repo),
        )
        rc = agent_run.cmd_watch(_watch_args("r13b"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "commits_since_start" in payload["git"]
        assert payload["git"]["commits_since_start"] is None


class TestRepoArgValidation:
    @pytest.mark.parametrize("bad_repo", ["", "   ", "\t"])
    def test_empty_or_whitespace_repo_is_rejected(
        self, isolated_runs_root, isolated_log_root, bad_repo
    ):
        _make_run(isolated_runs_root, isolated_log_root, "repoval1")
        with pytest.raises(SystemExit):
            agent_run.cmd_watch(_watch_args("repoval1", repo=bad_repo))

    def test_rejecting_empty_repo_mutates_nothing(
        self, isolated_runs_root, isolated_log_root
    ):
        sd, _ld = _make_run(isolated_runs_root, isolated_log_root, "repoval2")
        before = {p.name: p.read_text() for p in sd.iterdir()}
        with pytest.raises(SystemExit):
            agent_run.cmd_watch(_watch_args("repoval2", repo=""))
        after = {p.name: p.read_text() for p in sd.iterdir()}
        assert before == after

    def test_relative_repo_is_resolved_to_absolute(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch, capsys
    ):
        repo = tmp_path / "relrepo"
        _init_repo(repo)
        _make_run(isolated_runs_root, isolated_log_root, "repoval3")
        monkeypatch.chdir(tmp_path)
        rc = agent_run.cmd_watch(_watch_args("repoval3", repo="relrepo"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["repo"] == str(repo.resolve())
        assert Path(payload["repo"]).is_absolute()

    def test_cwd_file_first_line_only_is_echoed(
        self, isolated_runs_root, isolated_log_root, tmp_path, capsys
    ):
        repo = tmp_path / "repo"
        _init_repo(repo)
        sd, _ld = _make_run(
            isolated_runs_root, isolated_log_root, "repoval4",
            status="done", pid=1, exit_code=0, ended_age_secs=1,
        )
        (sd / "cwd").write_text(f"{repo}\nsome-second-line\n")
        rc = agent_run.cmd_watch(_watch_args("repoval4"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["repo"] == str(repo)
        assert "\n" not in payload["repo"]


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

    @pytest.mark.parametrize(
        "failure, expected",
        [
            pytest.param(
                subprocess.TimeoutExpired(cmd="git", timeout=2.5), "timeout", id="wedged_git",
            ),
            pytest.param(FileNotFoundError("git"), "git_missing", id="git_not_installed"),
        ],
    )
    def test_subprocess_level_failures_yield_their_discriminator(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch, capsys,
        failure, expected,
    ):
        repo = tmp_path / "repo"
        _init_repo(repo)
        _make_run(
            isolated_runs_root, isolated_log_root, "ge4",
            status="done", pid=1, exit_code=0, ended_age_secs=1,
            cwd=str(repo),
        )

        def _fail(*_args, **_kwargs):
            raise failure

        monkeypatch.setattr(agent_run.subprocess, "run", _fail)
        rc = agent_run.cmd_watch(_watch_args("ge4"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["git"] is None
        assert payload["git_error"] == expected
        # Only the git facts degrade; the rest of the contract stands.
        assert payload["status"] == "done"
        assert payload["exit_code"] == 0

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
        assert payload["git"] is None
        assert payload["git_error"] == "no_repo_path"


# ---------------------------------------------------------------------------
# git subprocess hardening
# ---------------------------------------------------------------------------

class TestGitHardening:
    def test_replacement_refs_are_disabled(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        _init_repo(repo)
        (repo / "a.txt").write_text("second\n")
        _git(repo, "add", "a.txt")
        _git(repo, "commit", "-q", "-m", "second")
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()
        parent = subprocess.run(
            ["git", "rev-parse", "HEAD^"], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()
        namespace = "refs/advrev-replace/"
        _git(repo, "update-ref", namespace + head, parent)
        monkeypatch.setenv("GIT_REPLACE_REF_BASE", namespace)

        result = agent_run._watch_git_facts_checked(repo, None)

        assert result.facts["dirty"] is False
        assert result.facts["files_changed"] == 0

    def test_git_replace_env_is_neutralised_at_the_subprocess_boundary(
        self, tmp_path, monkeypatch
    ):
        """Both replacement-ref defences are observable in the child env:
        GIT_REPLACE_REF_BASE is stripped and GIT_NO_REPLACE_OBJECTS is forced."""
        repo = tmp_path / "repo"
        _init_repo(repo)
        monkeypatch.setenv("GIT_REPLACE_REF_BASE", "refs/advrev-replace/")
        seen = []
        real_run = agent_run.subprocess.run

        def capturing_run(*args, **kwargs):
            seen.append(kwargs["env"])
            return real_run(*args, **kwargs)

        monkeypatch.setattr(agent_run.subprocess, "run", capturing_run)

        agent_run._watch_git_facts_checked(repo, None)

        assert seen
        for env in seen:
            assert "GIT_REPLACE_REF_BASE" not in env
            assert env.get("GIT_NO_REPLACE_OBJECTS") == "1"

    def test_git_graft_file_is_stripped(self, tmp_path, monkeypatch):
        """An inherited GIT_GRAFT_FILE rewrites parent pointers silently,
        making commits_since_start report a wrong value.  The variable must be
        stripped from every git subprocess env."""
        import tempfile

        repo = tmp_path / "repo"
        _init_repo(repo)
        # Commit A: this will be launch_head.
        (repo / "b.txt").write_text("second\n")
        _git(repo, "add", "b.txt")
        _git(repo, "commit", "-q", "-m", "second")
        commit_a = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()
        # Commit B: an intermediate commit.
        (repo / "c.txt").write_text("third\n")
        _git(repo, "add", "c.txt")
        _git(repo, "commit", "-q", "-m", "third")
        # Commit C (HEAD).
        (repo / "d.txt").write_text("fourth\n")
        _git(repo, "add", "d.txt")
        _git(repo, "commit", "-q", "-m", "fourth")
        commit_c = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()
        # Graft: HEAD's (C's) parent becomes A, cutting out B.
        # Without stripping: rev-list A..C = 1 (C only; B is gone from ancestry).
        # With stripping: rev-list A..C = 2 (B, C are both descendants of A).
        with tempfile.NamedTemporaryFile(mode="w", suffix=".grafts", delete=False) as gf:
            gf.write(f"{commit_c} {commit_a}\n")
            graft_path = gf.name
        monkeypatch.setenv("GIT_GRAFT_FILE", graft_path)

        result = agent_run._watch_git_facts_checked(repo, commit_a)

        assert result.facts is not None, f"git_error={result.git_error}"
        assert result.facts["commits_since_start"] == 2, (
            f"GIT_GRAFT_FILE not stripped: commits_since_start="
            f"{result.facts['commits_since_start']} (expected 2, got fewer if graft applied)"
        )

    def test_head_change_during_observation_degrades(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        _init_repo(repo)
        real = agent_run._watch_run_git_checked
        changed = False

        def run_git(path, args, timeout=agent_run.WATCH_GIT_SUBPROCESS_TIMEOUT_SECONDS):
            nonlocal changed
            outcome = real(path, args, timeout)
            if args == ["rev-parse", "HEAD"] and not changed:
                changed = True
                (repo / "b.txt").write_text("new\n")
                _git(repo, "add", "b.txt")
                _git(repo, "commit", "-q", "-m", "new")
            return outcome

        monkeypatch.setattr(agent_run, "_watch_run_git_checked", run_git)

        result = agent_run._watch_git_facts_checked(repo, None)

        assert result.facts is None
        assert result.git_error == "changed_during_observation"
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
        """Non-UTF-8 bytes in git output are preserved as surrogate escapes
        (PEP 383 / errors='surrogateescape') rather than silently replaced.
        Surrogates allow the original bytes to be recovered and detected by
        _has_surrogate(), which fails closed on non-round-trippable paths."""
        repo = tmp_path / "repo"
        _init_repo(repo)
        monkeypatch.setattr(
            agent_run.subprocess, "run", _fake_git_run(stdout=b"A\xff\xfeB"),
        )
        result = agent_run._watch_run_git_checked(repo, ["status", "--porcelain"])
        # \xff and \xfe are mapped to surrogate code points U+DCFF and U+DCFE.
        assert result.stdout == "A\udcff\udcfeB"


# ---------------------------------------------------------------------------
# shared time budget across the git calls
# ---------------------------------------------------------------------------

def _fake_git_run(*, stdout: bytes = b"0\n", sleep: Optional[float] = None,
                  sleep_full_timeout: bool = False, record: Optional[list] = None):
    """A `subprocess.run` stand-in returning fixed stdout, optionally
    sleeping (a fixed duration, or the whole timeout it was handed) and
    recording each call's timeout."""
    class _FakeResult:
        pass

    _FakeResult.stdout = stdout

    def _run(*_args, **kwargs):
        if record is not None:
            record.append(kwargs["timeout"])
        if sleep_full_timeout:
            time.sleep(kwargs["timeout"])
        elif sleep is not None:
            time.sleep(sleep)
        return _FakeResult()

    return _run


class TestGitTotalBudget:
    def test_total_wall_time_is_bounded(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
        monkeypatch.setattr(
            agent_run.subprocess, "run", _fake_git_run(sleep_full_timeout=True),
        )
        start = time.monotonic()
        agent_run._watch_git_facts_checked(repo, "0" * 40)
        elapsed = time.monotonic() - start
        assert elapsed < agent_run.WATCH_GIT_TOTAL_BUDGET_SECONDS + 2.0

    def test_per_call_cap_holds(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
        seen_timeouts: list[float] = []
        monkeypatch.setattr(
            agent_run.subprocess, "run", _fake_git_run(record=seen_timeouts),
        )
        assert agent_run._watch_git_facts_checked(repo, "0" * 40).facts is not None
        assert len(seen_timeouts) == 8
        assert all(t <= agent_run.WATCH_GIT_SUBPROCESS_TIMEOUT_SECONDS for t in seen_timeouts)

    def test_budget_exhaustion_returns_none(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
        monkeypatch.setattr(agent_run.subprocess, "run", _fake_git_run(sleep=0.3))
        monkeypatch.setattr(agent_run, "WATCH_GIT_TOTAL_BUDGET_SECONDS", 0.05)
        assert agent_run._watch_git_facts_checked(repo, "0" * 40).facts is None

    def test_happy_path_returns_fully_populated_facts(self, tmp_path):
        repo = tmp_path / "repo"
        _init_repo(repo)
        launch_head = agent_run._watch_run_git_checked(
            repo, ["rev-parse", "HEAD"]
        ).stdout.strip()
        (repo / "b.txt").write_text("more\n")
        _git(repo, "add", "b.txt")
        _git(repo, "commit", "-q", "-m", "second")
        expected_head = agent_run._watch_run_git_checked(
            repo, ["rev-parse", "--short", "HEAD"]
        ).stdout.strip()
        result = agent_run._watch_git_facts_checked(repo, launch_head).facts
        assert result == {
            "head": expected_head,
            "dirty": False,
            "files_changed": 0,
            "insertions": 0,
            "deletions": 0,
            "untracked_files": 0,
            "commits_since_start": 1,
            "last_commit_age_s": result["last_commit_age_s"],
            "toplevel": str(repo.resolve()),
        }
        assert result["last_commit_age_s"] is not None
        assert result["last_commit_age_s"] >= 0.0


class TestGitUntrackedFiles:
    def test_untracked_files_below_one_directory_are_counted_individually(self, tmp_path):
        repo = tmp_path / "repo"
        _init_repo(repo)
        new_dir = repo / "newdir"
        new_dir.mkdir()
        for index in range(3):
            (new_dir / f"file-{index}").write_text("work\n")

        result = agent_run._watch_git_facts_checked(repo, None)

        assert result.facts["untracked_files"] == 3

    def test_rename_origin_path_starting_with_question_marks_not_counted(self, tmp_path):
        """A rename/copy emits two NUL-separated records: the status record
        R  <new> and a bare origin-path record.  If the origin path starts
        with "?? " it must not be counted as an untracked file."""
        repo = tmp_path / "repo"
        _init_repo(repo)
        # Create and commit a tracked file whose name starts with "?? ".
        tricky = repo / "?? evil"
        tricky.write_text("data\n")
        _git(repo, "add", tricky.name)
        _git(repo, "commit", "-q", "-m", "add tricky name")
        # Rename it; the origin path "?? evil" will appear in the -z stream.
        renamed = repo / "renamed.txt"
        tricky.rename(renamed)
        _git(repo, "add", "-A")

        result = agent_run._watch_git_facts_checked(repo, None)

        # Zero untracked files: "?? evil" is the rename origin, not a new file.
        assert result.facts is not None, f"git_error={result.git_error}"
        assert result.facts["untracked_files"] == 0, (
            f"rename origin path counted as untracked: untracked_files="
            f"{result.facts['untracked_files']}"
        )

    def test_untracked_file_is_counted_and_repo_reads_dirty(
        self, isolated_runs_root, isolated_log_root, tmp_path, capsys
    ):
        repo = tmp_path / "repo"
        _init_repo(repo)
        (repo / "new-agent-file").write_text("work\n")
        _make_run(
            isolated_runs_root, isolated_log_root, "untracked1",
            status="done", pid=1, exit_code=0, ended_age_secs=1,
            cwd=str(repo),
        )
        rc = agent_run.cmd_watch(_watch_args("untracked1"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["git"]["dirty"] is True
        assert payload["git"]["untracked_files"] == 1
        # Tracked-file diff counters stay 0 — no tracked file changed.
        assert payload["git"]["files_changed"] == 0

    def test_ignored_file_is_not_counted(
        self, isolated_runs_root, isolated_log_root, tmp_path, capsys
    ):
        repo = tmp_path / "repo"
        _init_repo(repo)
        (repo / ".gitignore").write_text("generated/\n")
        _git(repo, "add", ".gitignore")
        _git(repo, "commit", "-q", "-m", "add gitignore")
        (repo / "generated").mkdir()
        (repo / "generated" / "work").write_text("build output\n")
        _make_run(
            isolated_runs_root, isolated_log_root, "ignored1",
            status="done", pid=1, exit_code=0, ended_age_secs=1,
            cwd=str(repo),
        )
        rc = agent_run.cmd_watch(_watch_args("ignored1"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["git"]["untracked_files"] == 0
        assert payload["git"]["dirty"] is False


class TestGitToplevel:
    def test_repo_at_its_own_root(self, tmp_path):
        repo = tmp_path / "repo"
        _init_repo(repo)
        outcome = agent_run._watch_git_facts_checked(repo, None)
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
        outcome = agent_run._watch_git_facts_checked(subdir, None)
        assert outcome.git_error is None
        assert outcome.facts is not None
        assert outcome.facts["toplevel"] == str(repo.resolve())

    def test_plain_subdir_inside_unrelated_outer_repo_reports_outer_toplevel(self, tmp_path):
        outer = tmp_path / "outer"
        _init_repo(outer)
        plain_subdir = outer / "plain-subdir"
        plain_subdir.mkdir()
        outcome = agent_run._watch_git_facts_checked(plain_subdir, None)
        assert outcome.git_error is None
        assert outcome.facts is not None
        assert outcome.facts["toplevel"] == str(outer.resolve())


# ---------------------------------------------------------------------------
# log facts
# ---------------------------------------------------------------------------

class TestLogFacts:
    def test_one_observation_opens_log_once(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        _make_run(
            isolated_runs_root, isolated_log_root, "oneopen",
            status="done", log_text="Error: boom\n" * 3,
        )
        real_open = agent_run._watch_open_validated_log
        calls = []

        def counted(path):
            calls.append(path)
            return real_open(path)

        monkeypatch.setattr(agent_run, "_watch_open_validated_log", counted)
        agent_run.cmd_watch(_watch_args("oneopen"))
        capsys.readouterr()

        assert len(calls) == 1

    def test_large_log_line_count_is_unknown_instead_of_scanned(self, tmp_path, monkeypatch):
        log = tmp_path / "log"
        log.write_text("one\ntwo\n")
        monkeypatch.setattr(agent_run, "WATCH_LINE_COUNT_MAX_BYTES", 1)

        with open(log, "rb") as f:
            import os as _os
            st = _os.fstat(f.fileno())
            facts = agent_run._watch_log_facts_from_file(f, log, st)
        assert facts["lines"] is None

    @pytest.mark.parametrize(
        "log_age_secs, check_age, expected_growing",
        [
            pytest.param(
                1, lambda age: 0 <= age < agent_run.WATCH_LOG_GROWING_MAX_AGE_SECONDS,
                True, id="fresh_log_is_growing",
            ),
            pytest.param(
                agent_run.WATCH_LOG_GROWING_MAX_AGE_SECONDS + 30,
                lambda age: age >= agent_run.WATCH_LOG_GROWING_MAX_AGE_SECONDS,
                False, id="stale_log_is_not_growing",
            ),
            # Clock skew / an NFS server-side timestamp, not freshness: a
            # future mtime must not read as freshly growing, which is the
            # wrong direction for a contract that fails toward firing.
            pytest.param(
                -3600.0, lambda age: age is None, False,
                id="far_future_mtime_degrades_to_null",
            ),
            pytest.param(
                -(agent_run.WATCH_LOG_FUTURE_MTIME_TOLERANCE_SECONDS / 2),
                lambda age: age == 0.0, True,
                id="jitter_within_tolerance_clamps_to_zero",
            ),
        ],
    )
    def test_mtime_age_and_growing(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys,
        log_age_secs, check_age, expected_growing,
    ):
        _make_run(
            isolated_runs_root, isolated_log_root, "r14",
            status="running", pid=1, log_age_secs=log_age_secs,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        assert agent_run.cmd_watch(_watch_args("r14")) == 0
        log = json.loads(capsys.readouterr().out)["log"]
        assert log["growing"] is expected_growing
        assert check_age(log["mtime_age_s"])

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
        assert payload["signals"] == _DEGRADED_SIGNALS


# ---------------------------------------------------------------------------
# non-regular log paths (FIFO, directory, symlinks) must never be opened
# ---------------------------------------------------------------------------

def _requires_mkfifo() -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("os.mkfifo unavailable on this platform")


def _watch_without_hanging(name: str, *, timeout: float = 5.0) -> int:
    """Run `cmd_watch` on a worker thread and fail if it does not return.

    A non-regular log that slips past validation blocks forever on open/read
    (a FIFO with no writer), so these cases must fail the test rather than
    wedge the whole suite.
    """
    result: dict = {}

    def _call() -> None:
        result["rc"] = agent_run.cmd_watch(_watch_args(name))

    thread = threading.Thread(target=_call, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    assert not thread.is_alive(), f"watch hung observing run {name!r}"
    return result["rc"]


def _log_fifo(ld: Path) -> None:
    _requires_mkfifo()
    os.mkfifo(ld / "log")


def _log_directory(ld: Path) -> None:
    (ld / "log").mkdir()


def _log_symlink_to_fifo(ld: Path) -> None:
    _requires_mkfifo()
    fifo_path = ld / "log.fifo"
    os.mkfifo(fifo_path)
    (ld / "log").symlink_to(fifo_path)


class TestRegularLogPath:
    @pytest.mark.parametrize(
        "make_log",
        [
            pytest.param(_log_fifo, id="fifo"),
            pytest.param(_log_directory, id="directory"),
            pytest.param(_log_symlink_to_fifo, id="symlink_to_fifo"),
        ],
    )
    def test_non_regular_log_is_rejected(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys, make_log
    ):
        """Rejected by the helper on the open descriptor, and degraded to a
        null `log` plus empty signals end to end."""
        _, ld = _make_run(
            isolated_runs_root, isolated_log_root, "nonreg",
            status="running", pid=1, write_log=False,
        )
        make_log(ld)
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        with agent_run._watch_open_validated_log(agent_run._log_file_for("nonreg")) as f:
            assert f is None
        assert _watch_without_hanging("nonreg") == 0
        payload = json.loads(capsys.readouterr().out)
        assert set(payload.keys()) == WATCH_CONTRACT_KEYS
        assert payload["log"] is None
        assert payload["signals"] == _DEGRADED_SIGNALS

    @pytest.mark.parametrize("via_symlink", [False, True], ids=["plain_file", "symlink"])
    def test_regular_log_is_accepted(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys, via_symlink
    ):
        _, ld = _make_run(
            isolated_runs_root, isolated_log_root, "reg",
            status="running", pid=1, log_age_secs=1, log_text="line one\nline two\n",
        )
        if via_symlink:
            target = ld / "log.real"
            (ld / "log").rename(target)
            (ld / "log").symlink_to(target)
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        with agent_run._watch_open_validated_log(agent_run._log_file_for("reg")) as f:
            assert f is not None
        rc = agent_run.cmd_watch(_watch_args("reg"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["log"]["bytes"] == len("line one\nline two\n")
        assert payload["log"]["lines"] == 2

    def test_inode_swapped_between_stat_and_read_returns_instead_of_hanging(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        """TOCTOU regression: something replaces the regular log with a FIFO
        after validation would have passed by path but before the fd-based
        open+fstat in _watch_open_validated_log runs."""
        _requires_mkfifo()
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
        assert _watch_without_hanging("toctou1") == 0
        payload = json.loads(capsys.readouterr().out)
        assert set(payload.keys()) == WATCH_CONTRACT_KEYS
        assert payload["log"] is None
        assert payload["signals"] == _DEGRADED_SIGNALS


# ---------------------------------------------------------------------------
# signals: repeated-error / repeated-read
# ---------------------------------------------------------------------------

@pytest.fixture
def signals_for(isolated_runs_root, isolated_log_root, monkeypatch, capsys):
    """Run `watch` end-to-end over a log with the given text and return the
    `signals` object from the emitted contract."""
    def _run(log_text: str) -> dict:
        _make_run(
            isolated_runs_root, isolated_log_root, "sig",
            status="running", pid=1, log_text=log_text, log_age_secs=1,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        assert agent_run.cmd_watch(_watch_args("sig")) == 0
        return json.loads(capsys.readouterr().out)["signals"]

    return _run


THRESHOLD = agent_run.WATCH_REPEAT_THRESHOLD


class TestSignals:
    @pytest.mark.parametrize(
        "repeats, expected_count",
        [
            pytest.param(THRESHOLD - 1, None, id="below_threshold"),
            pytest.param(THRESHOLD, THRESHOLD, id="at_threshold"),
        ],
    )
    def test_repeated_error_needs_the_threshold(self, signals_for, repeats, expected_count):
        signals = signals_for("ok\n" + "Error: boom\n" * repeats)
        if expected_count is None:
            assert signals["repeated_error"] is None
        else:
            assert signals["repeated_error"] == {"line": "Error: boom", "count": expected_count}

    @pytest.mark.parametrize(
        "repeats, expected_count",
        [
            pytest.param(THRESHOLD - 1, None, id="below_threshold"),
            pytest.param(THRESHOLD, THRESHOLD, id="at_threshold"),
        ],
    )
    def test_repeated_read_needs_the_threshold(self, signals_for, repeats, expected_count):
        signals = signals_for("Read /a/one.py\n" * repeats)
        assert signals["distinct_files_read"] == 1
        if expected_count is None:
            assert signals["top_repeated_read"] is None
        else:
            assert signals["top_repeated_read"] == {"path": "/a/one.py", "count": expected_count}

    def test_distinct_files_read_counts_every_target_not_just_repeated_ones(self, signals_for):
        signals = signals_for(
            "Read /a/one.py\n" * THRESHOLD + "Read /a/two.py\nRead /a/three.py\n"
        )
        assert signals["distinct_files_read"] == 3
        assert signals["top_repeated_read"] == {"path": "/a/one.py", "count": THRESHOLD}

    def test_repeated_error_line_is_truncated_to_the_contract_width(self, signals_for):
        signals = signals_for(("Error: " + "x" * 300 + "\n") * THRESHOLD)
        assert len(signals["repeated_error"]["line"]) == agent_run.WATCH_ERROR_LINE_MAX_CHARS

    def test_signals_scan_only_the_tail(self, signals_for):
        """Errors pushed past WATCH_TAIL_LINES by later output are outside
        the scan window and must not be counted."""
        old_errors = "Error: old\n" * THRESHOLD
        filler = "noise\n" * (agent_run.WATCH_TAIL_LINES + 10)
        assert signals_for(old_errors + filler)["repeated_error"] is None

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
    def test_repeated_error_ignores_healthy_lookalikes(self, signals_for, noisy_line):
        assert signals_for((noisy_line + "\n") * THRESHOLD)["repeated_error"] is None

    @pytest.mark.parametrize(
        "error_line",
        [
            "Error: ENOENT: no such file",
            "src/foo.py: error: bad thing",
            "Traceback (most recent call last):",
        ],
    )
    def test_repeated_error_detects_genuine_errors(self, signals_for, error_line):
        assert signals_for((error_line + "\n") * THRESHOLD)["repeated_error"] == {
            "line": error_line,
            "count": THRESHOLD,
        }

    def test_repeated_read_ansi_wrapped_lines_collapse_to_clean_path(self, signals_for):
        plain = "Read /a/one.py"
        colour = "Read \x1b[36m/a/one.py\x1b[0m"
        signals = signals_for("\n".join([plain, colour] * 3) + "\n")
        assert signals["distinct_files_read"] == 1
        assert signals["top_repeated_read"] == {"path": "/a/one.py", "count": 6}

    def test_repeated_read_rejects_prose_target(self, signals_for):
        assert signals_for("Please Read carefully\n" * 3)["top_repeated_read"] is None

    def test_repeated_error_detected_across_cr_redraws(self, signals_for):
        # A TUI frame redrawing in place puts repeats on one physical line.
        signals = signals_for("Error: boom\r" * 4 + "\n")
        assert signals["repeated_error"] == {"line": "Error: boom", "count": 4}

    def test_repeated_error_picks_max_count_not_first_encountered(self, signals_for):
        lines = ["Error: first"] * THRESHOLD + ["Error: loudest"] * (THRESHOLD + 5)
        assert signals_for("\n".join(lines) + "\n")["repeated_error"] == {
            "line": "Error: loudest",
            "count": THRESHOLD + 5,
        }


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
        assert payload["signals"] == _DEGRADED_SIGNALS
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

    def test_internal_observation_failure_exits_0_with_a_degraded_contract(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        """The top-level never-raise guard: an unanticipated exception still
        emits the full contract at exit 0, with every observed field at its
        null/unknown value and only ``observation_error`` set."""
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
        assert payload["observation_error"] == "RuntimeError: synthetic crash"
        assert payload["status"] == "unknown"
        assert payload["terminal"] is True
        assert payload["pid"] is None
        assert payload["log"] is None
        assert payload["git"] is None
        assert payload["git_error"] is None

    def test_cmd_status_unaffected_by_watch_exit_code_split(
        self, isolated_runs_root, isolated_log_root
    ):
        with pytest.raises(SystemExit) as exc_info:
            agent_run.cmd_status(argparse.Namespace(name="does-not-exist"))
        assert exc_info.value.code != 0

    @pytest.mark.parametrize("argv", [["watch"], ["watch", "--bad"], ["watch", "run", "--repo"]])
    def test_watch_parse_errors_exit_1(self, argv):
        with pytest.raises(SystemExit) as exc_info:
            agent_run.main(argv)
        assert exc_info.value.code == 1


def _launch(name: str) -> None:
    agent_run.cmd_launch(argparse.Namespace(
        command=["true"],
        interactive=False,
        prompt_file=None,
        echo=False,
        echo_interval=2.0,
        submit_mode=None,
        idle_timeout=None,
        name=name,
    ))


def _await_state_file(state_dir: Path, field: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not (state_dir / field).exists() and time.monotonic() < deadline:
        time.sleep(0.02)


class TestLaunchWritesCwd:
    def test_launch_records_absolute_cwd_and_launch_head_for_a_repo(
        self, isolated_runs_root, isolated_log_root, monkeypatch, tmp_path
    ):
        workdir = tmp_path / "launch_repo"
        _init_repo(workdir)
        expected_head = agent_run._watch_run_git_checked(
            workdir, ["rev-parse", "HEAD"]
        ).stdout.strip()
        monkeypatch.chdir(workdir)
        _launch("launched_repo")
        sd = isolated_runs_root / "launched_repo"
        _await_state_file(sd, "launch_head")
        assert (sd / "cwd").read_text().strip() == str(workdir.resolve())
        assert (sd / "launch_head").read_text().strip() == expected_head

    def test_launch_skips_launch_head_when_cwd_is_not_a_repo(
        self, isolated_runs_root, isolated_log_root, monkeypatch, tmp_path
    ):
        workdir = tmp_path / "launch_plain"
        workdir.mkdir()
        monkeypatch.chdir(workdir)
        _launch("launched_plain")
        sd = isolated_runs_root / "launched_plain"
        _await_state_file(sd, "pid")
        assert (sd / "cwd").read_text().strip() == str(workdir.resolve())
        assert not (sd / "launch_head").exists()

    def test_unreadable_cwd_skips_cwd_file_without_stranding_run(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        real_cwd = Path.cwd

        def _raising_cwd():
            raise FileNotFoundError("launch directory removed")

        monkeypatch.setattr(agent_run.Path, "cwd", staticmethod(_raising_cwd))
        _launch("launched_no_cwd")
        monkeypatch.setattr(agent_run.Path, "cwd", staticmethod(real_cwd))

        sd = isolated_runs_root / "launched_no_cwd"
        _await_state_file(sd, "pid")
        assert not (sd / "cwd").exists()
        # The run must have fully published, not be stranded as "starting"
        # with no pid behind it.
        status = (sd / "status").read_text().strip()
        assert not (status == "starting" and not (sd / "pid").exists())
        capsys.readouterr()

        rc = agent_run.cmd_watch(_watch_args("launched_no_cwd"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["repo"] is None
        assert payload["git"] is None
        assert set(payload.keys()) == WATCH_CONTRACT_KEYS


# ---------------------------------------------------------------------------
# _watch_tail_lines_from_file: byte-bounded backward scan (live path)
# ---------------------------------------------------------------------------

def _tail_from_file(log: Path, n: int) -> list:
    """Open log and call the live _watch_tail_lines_from_file."""
    with open(log, "rb") as f:
        import os as _os
        end = _os.fstat(f.fileno()).st_size
        return agent_run._watch_tail_lines_from_file(f, end, n)


class TestWatchTailLinesByteBound:
    def test_ordinary_tail_is_unchanged(self, tmp_path):
        lines = [f"line {i}\n" for i in range(500)]
        log = tmp_path / "log"
        log.write_text("".join(lines))
        result = _tail_from_file(log, 50)
        assert result == [f"line {i}" for i in range(450, 500)]

    def test_byte_cap_enforced_on_newline_free_file(self, tmp_path):
        log = tmp_path / "log"
        log.write_bytes(b"x" * (2 * 1024 * 1024))
        result = _tail_from_file(log, 200)
        total_chars = sum(len(line) for line in result)
        max_expected = agent_run.WATCH_TAIL_MAX_BYTES + agent_run.WATCH_TAIL_READ_BLOCK_BYTES
        assert total_chars <= max_expected

    def test_cap_with_sparse_newlines(self, tmp_path):
        log = tmp_path / "log"
        with log.open("wb") as f:
            f.write(b"first\n")
            f.write(b"second\n")
            f.write(b"y" * (2 * 1024 * 1024))
        result = _tail_from_file(log, 200)
        total_chars = sum(len(line) for line in result)
        max_expected = agent_run.WATCH_TAIL_MAX_BYTES + agent_run.WATCH_TAIL_READ_BLOCK_BYTES
        assert total_chars <= max_expected
        assert "first" not in result

    def test_small_file_returns_every_line(self, tmp_path):
        log = tmp_path / "log"
        log.write_text("a\nb\nc\n")
        result = _tail_from_file(log, 200)
        assert result == ["a", "b", "c"]

    def test_empty_file_returns_empty_list(self, tmp_path):
        log = tmp_path / "log"
        log.write_bytes(b"")
        result = _tail_from_file(log, 200)
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
    def test_cmd_watch_unstatable_state_and_missing_log_exits_zero_not_two(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        """An unstatable state dir with no log must not return exit 2 (stop polling).

        The old _is_dir_safe gate collapsed every OSError on the state dir to
        False.  If the log was also absent, the conjunction was True and cmd_watch
        returned exit 2, which tells the poller to stop polling this name —
        permanently hiding a healthy but temporarily unreadable run.

        With tri-state probes, UNSTATABLE + MISSING must emit the degraded
        contract (observation_error set) and return exit 0.
        """
        # Create only the state dir, make the parent unreadable so stat fails.
        state_dir = isolated_runs_root / "w4"
        state_dir.mkdir(parents=True)
        (state_dir / "status").write_text("running\n")
        # No log dir created → log path is MISSING.
        isolated_runs_root.chmod(0o000)
        try:
            rc = agent_run.cmd_watch(_watch_args("w4"))
        finally:
            isolated_runs_root.chmod(0o755)
        assert rc == 0, (
            "UNSTATABLE state + MISSING log must return exit 0, not exit 2; "
            "exit 2 permanently stops polling the run"
        )
        payload = json.loads(capsys.readouterr().out)
        assert set(payload.keys()) == WATCH_CONTRACT_KEYS
        assert isinstance(payload["observation_error"], str)
        assert payload["observation_error"]

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


# ---------------------------------------------------------------------------
# scratch file-activity facts
# ---------------------------------------------------------------------------

class TestScratchFacts:
    """Tests for _watch_scratch_facts and scratch field in the watch contract."""

    SCRATCH_KEYS = {"newest_mtime_age_s", "files_modified_recent", "scanned", "truncated", "error"}

    # ------------------------------------------------------------------
    # unit-level tests of _watch_scratch_facts
    # ------------------------------------------------------------------

    def test_file_under_working_dir_is_reflected(self, tmp_path):
        """A file written under the working dir shows up in the scratch facts."""
        f = tmp_path / "findings.md"
        f.write_text("review output\n")
        result = agent_run._watch_scratch_facts(str(tmp_path))
        assert set(result.keys()) == self.SCRATCH_KEYS
        assert result["error"] is None
        assert result["scanned"] >= 1
        assert result["newest_mtime_age_s"] is not None
        assert result["newest_mtime_age_s"] >= 0.0
        # File was just written so it must count as recent.
        assert result["files_modified_recent"] >= 1

    def test_gitignored_scratch_file_is_included(self, tmp_path):
        """The scan deliberately includes gitignored paths — that is the point.

        Regression test for the blind spot: zero git facts but real activity.
        """
        repo = tmp_path / "repo"
        _repo_with_gitignored_finding(repo, "probe")

        # Git confirms zero activity: no commits since start, no untracked files.
        launch_head = agent_run._watch_run_git_checked(
            repo, ["rev-parse", "HEAD"]
        ).stdout.strip()
        git_result = agent_run._watch_git_facts_checked(repo, launch_head)
        assert git_result.facts is not None
        assert git_result.facts["commits_since_start"] == 0
        assert git_result.facts["untracked_files"] == 0
        assert git_result.facts["files_changed"] == 0

        # Scratch scan must see the finding despite git seeing nothing.
        result = agent_run._watch_scratch_facts(str(repo))
        assert result["error"] is None
        assert result["scanned"] is not None
        assert result["scanned"] >= 1
        # The scratch file was just written: it must count as recent.
        assert result["files_modified_recent"] >= 1
        assert result["newest_mtime_age_s"] is not None
        assert result["newest_mtime_age_s"] < agent_run.WATCH_SCRATCH_RECENT_SECONDS

    def test_pruned_dirs_are_not_descended(self, tmp_path):
        """Directories in WATCH_SCRATCH_PRUNE_DIRS must not be entered."""
        for dirname in agent_run.WATCH_SCRATCH_PRUNE_DIRS:
            pruned = tmp_path / dirname
            pruned.mkdir()
            (pruned / "heavy.json").write_text("x" * 100)

        # Put one real file outside the pruned dirs so we scan something.
        (tmp_path / "real.py").write_text("code\n")

        result = agent_run._watch_scratch_facts(str(tmp_path))
        assert result["error"] is None
        # Only the one real file outside pruned dirs counts.
        assert result["scanned"] == 1

    def test_file_count_bound_sets_truncated(self, tmp_path, monkeypatch):
        """Hitting WATCH_SCRATCH_MAX_FILES with zero recent files degrades to null + file_limit.

        When the cap fires and files_modified_recent is still zero, the zero
        is absence of evidence (unseen files may be recent).  The result must
        have null decision fields, truncated=True, and error="file_limit".
        """
        monkeypatch.setattr(agent_run, "WATCH_SCRATCH_MAX_FILES", 3)
        # Only old files — the cap fires before any recent file could be found.
        old_time = time.time() - 7200
        for i in range(10):
            f = tmp_path / f"f{i}.txt"
            f.write_text("data\n")
            os.utime(f, (old_time, old_time))

        result = agent_run._watch_scratch_facts(str(tmp_path))
        assert result["truncated"] is True
        assert result["error"] == "file_limit"
        assert result["files_modified_recent"] is None
        assert result["newest_mtime_age_s"] is None
        assert result["scanned"] is None

    def test_truncated_with_recent_file_keeps_positive_evidence(self, tmp_path, monkeypatch):
        """Hitting a bound after finding a recent file keeps positive evidence.

        files_modified_recent >= 1 under truncation is sound positive evidence —
        the scan saw activity.  The result must keep the count, truncated=True,
        and error=None.  Test: pinning the asymmetric truncation contract.

        Uses depth_limit truncation with a known-recent file at depth 1 and a
        subdir at depth 2 that causes depth_limit to fire.  The recent file is
        always scanned; the depth-limited subdir is reliably not.
        """
        monkeypatch.setattr(agent_run, "WATCH_SCRATCH_MAX_DEPTH", 1)
        # Recent file at depth 1: always scanned.
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "recent.txt").write_text("data\n")
        # Another subdir at depth 1 that has a child at depth 2: fires depth_limit.
        deep_parent = tmp_path / "deep_parent"
        deep_parent.mkdir()
        deep_child = deep_parent / "nested"
        deep_child.mkdir()
        (deep_child / "unseen.txt").write_text("data\n")

        result = agent_run._watch_scratch_facts(str(tmp_path))
        assert result["truncated"] is True
        # The recent file was seen, so positive evidence must be kept.
        assert result["files_modified_recent"] is not None
        assert result["files_modified_recent"] >= 1, (
            "files_modified_recent must reflect the scanned recent file"
        )
        assert result["error"] is None, (
            "Positive evidence under truncation must keep error=None"
        )

    def test_missing_or_unreadable_dir_returns_null_with_error_not_zero(self, tmp_path):
        """A missing working directory must degrade to null fields + exact categorical code."""
        missing = tmp_path / "does-not-exist"
        result = agent_run._watch_scratch_facts(str(missing))
        assert result["error"] == "not_a_directory"
        # The numeric fields must be null, not a confident 0.
        assert result["newest_mtime_age_s"] is None
        assert result["files_modified_recent"] is None
        assert result["scanned"] is None

    def test_no_working_dir_returns_null_with_error(self):
        """A None or empty working_dir degrades gracefully with an exact categorical code."""
        for bad in (None, ""):
            result = agent_run._watch_scratch_facts(bad)
            assert result["error"] == "no_working_dir"
            assert result["newest_mtime_age_s"] is None
            assert result["files_modified_recent"] is None
            assert result["scanned"] is None

    def test_idle_working_dir_reports_large_age_not_error(self, tmp_path):
        """A real directory with only old files reports a large age, not an error."""
        old_file = tmp_path / "old.txt"
        old_file.write_text("old\n")
        # Age it to 1 hour ago.
        old_time = time.time() - 3600
        os.utime(old_file, (old_time, old_time))

        result = agent_run._watch_scratch_facts(str(tmp_path))
        assert result["error"] is None
        assert result["scanned"] == 1
        assert result["newest_mtime_age_s"] is not None
        assert result["newest_mtime_age_s"] >= agent_run.WATCH_SCRATCH_RECENT_SECONDS
        assert result["files_modified_recent"] == 0

    def test_future_mtime_beyond_tolerance_degrades_to_null_with_clock_skew(self, tmp_path):
        """A materially future mtime must degrade to null + error='clock_skew'.

        A confident zero is internally contradictory: newest_mtime_age_s is
        null (clock disagreement) while files_modified_recent=0 claims inactivity
        was observed.  On NFS/server-clock skew a freshly written file can have
        a future mtime, so the zero fails toward escalation of a healthy run.

        Choice matches the log path: minor jitter within
        WATCH_LOG_FUTURE_MTIME_TOLERANCE_SECONDS is clamped to 0 (still recent);
        beyond the tolerance the scan degrades.  Test: mutation replacing the
        clock_skew raise with a pass now reports files_modified_recent=0,
        error=None — a confident zero — and this test fails.
        """
        f = tmp_path / "future.txt"
        f.write_text("data\n")
        future_time = time.time() + agent_run.WATCH_LOG_FUTURE_MTIME_TOLERANCE_SECONDS + 10.0
        os.utime(f, (future_time, future_time))

        result = agent_run._watch_scratch_facts(str(tmp_path))
        assert result["error"] == "clock_skew", (
            f"Materially future mtime must set error='clock_skew' (got {result['error']!r})"
        )
        assert result["files_modified_recent"] is None
        assert result["newest_mtime_age_s"] is None
        assert result["scanned"] is None
        assert result["truncated"] is False

    def test_future_mtime_within_tolerance_counts_as_recent(self, tmp_path):
        """A mtime within WATCH_LOG_FUTURE_MTIME_TOLERANCE_SECONDS ahead is treated as 'now'.

        Minor NFS/server-clock jitter should not degrade the scan; a freshly
        written file with a slightly future timestamp must count as recent.
        """
        f = tmp_path / "jitter.txt"
        f.write_text("data\n")
        # Set mtime slightly in the future but within the tolerance.
        jitter = agent_run.WATCH_LOG_FUTURE_MTIME_TOLERANCE_SECONDS / 2
        os.utime(f, (time.time() + jitter, time.time() + jitter))

        result = agent_run._watch_scratch_facts(str(tmp_path))
        assert result["error"] is None, (
            "Minor future jitter within tolerance must not degrade the scan"
        )
        assert result["files_modified_recent"] is not None
        assert result["files_modified_recent"] >= 1, (
            "A file with minor future jitter must count as recent"
        )

    def test_nan_mtime_degrades_to_null_with_invalid_mtime(self, tmp_path, monkeypatch):
        """A NaN st_mtime must degrade to null fields + error='invalid_mtime'.

        NaN makes every ordered comparison false: the file is neither recent
        nor clock-skewed, and max(0.0, NaN) returns 0.0.  Without the
        isfinite guard this produces a confident zero, which can trigger
        escalation of a healthy run.  Mutation: remove the math.isfinite
        guard and the test fails because error is None and scanned=1.
        """
        import math as _math

        (tmp_path / "real.txt").write_text("data\n")

        class NaNMtimeEntry:
            """Proxy a real DirEntry but return NaN from stat().st_mtime."""

            def __init__(self, entry):
                self._entry = entry

            def stat(self, **kw):
                s = self._entry.stat(**kw)
                return type("FakeStat", (), {
                    "st_mtime": float("nan"),
                    "st_mode": s.st_mode,
                })()

            def __getattr__(self, name):
                return getattr(self._entry, name)

        _patch_scandir(monkeypatch, NaNMtimeEntry)

        result = agent_run._watch_scratch_facts(str(tmp_path))
        assert result["error"] == "invalid_mtime", (
            f"NaN mtime must set error='invalid_mtime' (got {result['error']!r})"
        )
        assert result["files_modified_recent"] is None
        assert result["newest_mtime_age_s"] is None
        assert result["scanned"] is None
        assert result["truncated"] is False

    def test_positive_infinity_mtime_degrades_to_null_with_invalid_mtime(self, tmp_path, monkeypatch):
        """+inf st_mtime must degrade to null fields + error='invalid_mtime'.

        Positive infinity reaches the clock_skew guard incidentally, but one
        uniform invalid_mtime category for all non-finite values is simpler
        and deterministic.  Mutation: remove the isfinite guard and the test
        fails because error changes to 'clock_skew' instead of 'invalid_mtime'.
        """
        (tmp_path / "real.txt").write_text("data\n")

        class PosInfMtimeEntry:
            def __init__(self, entry):
                self._entry = entry

            def stat(self, **kw):
                s = self._entry.stat(**kw)
                return type("FakeStat", (), {
                    "st_mtime": float("inf"),
                    "st_mode": s.st_mode,
                })()

            def __getattr__(self, name):
                return getattr(self._entry, name)

        _patch_scandir(monkeypatch, PosInfMtimeEntry)

        result = agent_run._watch_scratch_facts(str(tmp_path))
        assert result["error"] == "invalid_mtime", (
            f"+inf mtime must set error='invalid_mtime' (got {result['error']!r})"
        )
        assert result["files_modified_recent"] is None
        assert result["newest_mtime_age_s"] is None
        assert result["scanned"] is None
        assert result["truncated"] is False

    def test_negative_infinity_mtime_degrades_to_null_with_invalid_mtime(self, tmp_path, monkeypatch):
        """-inf st_mtime must degrade to null fields + error='invalid_mtime'.

        Negative infinity is treated as infinitely old without the guard:
        the file is not recent, age > 0, and the scan completes with a
        confident zero.  Mutation: remove the isfinite guard and the test
        fails because error is None and files_modified_recent=0.
        """
        (tmp_path / "real.txt").write_text("data\n")

        class NegInfMtimeEntry:
            def __init__(self, entry):
                self._entry = entry

            def stat(self, **kw):
                s = self._entry.stat(**kw)
                return type("FakeStat", (), {
                    "st_mtime": float("-inf"),
                    "st_mode": s.st_mode,
                })()

            def __getattr__(self, name):
                return getattr(self._entry, name)

        _patch_scandir(monkeypatch, NegInfMtimeEntry)

        result = agent_run._watch_scratch_facts(str(tmp_path))
        assert result["error"] == "invalid_mtime", (
            f"-inf mtime must set error='invalid_mtime' (got {result['error']!r})"
        )
        assert result["files_modified_recent"] is None
        assert result["newest_mtime_age_s"] is None
        assert result["scanned"] is None
        assert result["truncated"] is False

    def test_wall_clock_budget_sets_truncated_before_exhaustion(self, tmp_path, monkeypatch):
        """When the wall-clock budget is zero the scan truncates with error=timeout.

        With no recent files found before the deadline, the zero is unsound:
        result must have null fields, truncated=True, and error="timeout".
        """
        # Create only old files so the timeout fires with files_modified_recent==0.
        old_time = time.time() - 7200
        for i in range(5):
            f = tmp_path / f"f{i}.txt"
            f.write_text("data\n")
            os.utime(f, (old_time, old_time))
        monkeypatch.setattr(agent_run, "WATCH_SCRATCH_BUDGET_SECONDS", 0.0)
        result = agent_run._watch_scratch_facts(str(tmp_path))
        assert result["truncated"] is True
        assert result["error"] == "timeout"
        assert result["files_modified_recent"] is None
        assert result["newest_mtime_age_s"] is None
        assert result["scanned"] is None

    def test_outer_deadline_check_fires(self, tmp_path, monkeypatch):
        """The outer per-queue-item deadline check must set truncated=True.

        The outer check runs at the top of the while-queue loop before each
        directory is processed.  To isolate it from the inner per-entry check:
        two empty subdirectories are used (no entries → inner check never runs for
        them).  The deadline expires precisely on the outer check for the second
        empty subdirectory; with the outer check active, truncated=True.  With it
        disabled the scan processes both empty subdirs and exits with truncated=False.

        time.monotonic() call sequence (2 empty subdirs in root):
          call 1: setup (now = ...)
          call 2: outer check — root (proceed: return real time, deadline not yet)
          call 3: inner check — root entry 1 (aaa_sub1, dir, not a file, no stat)
          call 4: inner check — root entry 2 (zzz_sub2, dir, not a file, no stat)
          call 5: outer check — zzz_sub2 (proceed: empty, real time)
          call 6: outer check — aaa_sub1 (HERE: return past deadline → truncated=True)
        Without the outer check: call 6 does not happen; aaa_sub1 is processed
        (0 entries), queue empties, exits with truncated=False.

        Mutation: disable the outer ``if time.monotonic() >= deadline: break``
        block — this test then gets truncated=False and fails.
        """
        import time as _time

        # Two empty subdirectories: no files, so the inner per-entry check
        # for sub dirs never fires (their entries are dirs, counted but not
        # stat-called; the inner check does fire once per entry in root).
        (tmp_path / "aaa_sub1").mkdir()
        (tmp_path / "zzz_sub2").mkdir()

        call_count = [0]
        real_monotonic = _time.monotonic

        def patched_monotonic():
            call_count[0] += 1
            val = real_monotonic()
            # Call 6 is the outer check for aaa_sub1 (3rd queue item).
            # Returning past deadline here causes the outer check to truncate
            # before aaa_sub1 is processed.
            if call_count[0] >= 6:
                return val + 1000.0
            return val

        monkeypatch.setattr(agent_run.time, "monotonic", patched_monotonic)
        result = agent_run._watch_scratch_facts(str(tmp_path))
        assert result["truncated"] is True, (
            "Outer deadline check must set truncated=True; with it disabled the "
            "scan processes both empty directories and exits with truncated=False"
        )

    def test_slow_enumeration_hits_budget_before_iterator_exhausted(self, tmp_path, monkeypatch):
        """A slow-per-entry iterator must be cut off by the budget, not list()-materialised.

        WATCH_SCRATCH_BUDGET_SECONDS has to bound wall-clock time even when the
        cost is in yielding each entry.  Under list(os.scandir(...)) every entry
        is materialised before the first deadline check, so the scan would take
        ENTRY_COUNT * ENTRY_DELAY however small the budget.
        """
        import time as _time

        ENTRY_DELAY = 0.02   # 20 ms per entry
        ENTRY_COUNT = 20     # total would be 400 ms if list()-materialised
        BUDGET = 0.05        # 50 ms — enough for ~2 entries, not all 20

        for i in range(ENTRY_COUNT):
            (tmp_path / f"slow_{i}.txt").write_text("data\n")

        def slow(entry):
            _time.sleep(ENTRY_DELAY)
            return entry

        _patch_scandir(monkeypatch, slow)
        monkeypatch.setattr(agent_run, "WATCH_SCRATCH_BUDGET_SECONDS", BUDGET)

        start = _time.monotonic()
        result = agent_run._watch_scratch_facts(str(tmp_path))
        elapsed = _time.monotonic() - start

        # Must respect the budget: must not take nearly as long as list()-materialising all entries.
        assert elapsed < ENTRY_DELAY * ENTRY_COUNT * 0.5, (
            f"Scan took {elapsed:.3f}s; list()-materialisation would have taken "
            f"~{ENTRY_DELAY * ENTRY_COUNT:.3f}s — the budget is not being checked during enumeration"
        )
        assert result["truncated"] is True, "Budget hit must set truncated=True"

    def test_entries_visited_cap_truncates_independently_of_file_count(self, tmp_path, monkeypatch):
        """Entry cap with zero recent files degrades to null + entry_limit.

        WATCH_SCRATCH_MAX_ENTRIES bounds enumeration work independently of
        WATCH_SCRATCH_MAX_FILES: a wide directory of non-regular entries never
        trips the file cap.  When the cap fires with no recent files seen, the
        zero is unsound; the result must have null decision fields, truncated=True,
        and error="entry_limit".
        """
        # Create entries that are not regular files (empty sub-dirs) so the file cap won't trigger.
        for i in range(20):
            (tmp_path / f"subdir_{i}").mkdir()

        monkeypatch.setattr(agent_run, "WATCH_SCRATCH_MAX_ENTRIES", 5)
        result = agent_run._watch_scratch_facts(str(tmp_path))
        assert result["truncated"] is True, (
            "Entries-visited cap must set truncated=True when exceeded by non-file entries"
        )
        assert result["error"] == "entry_limit"
        assert result["files_modified_recent"] is None
        assert result["newest_mtime_age_s"] is None
        assert result["scanned"] is None

    def test_depth_limit_at_cap_sets_truncated(self, tmp_path, monkeypatch):
        """A non-pruned directory at the depth cap must set truncated=True + depth_limit.

        With WATCH_SCRATCH_MAX_DEPTH=1, a subdirectory encountered at depth=1
        cannot be descended (depth < MAX_DEPTH is False for depth=1).  If the
        only recent file is inside that subdirectory, the zero is unsound;
        the result must have null decision fields, truncated=True, and
        error="depth_limit".
        Mutation: replace ``depth < WATCH_SCRATCH_MAX_DEPTH`` with
        ``depth <= WATCH_SCRATCH_MAX_DEPTH`` — this test then fails because
        the file is found, positive evidence is kept, and truncated=False.
        """
        monkeypatch.setattr(agent_run, "WATCH_SCRATCH_MAX_DEPTH", 1)
        # File at depth 2: root/sub (depth=1) is descended, sub/inner (depth=2 dir) is not.
        # recent file is inside inner (depth 3 if we count files, depth 2 for the dir).
        # Actually: root (depth=0) → sub (depth=1, descended) → inner (depth=1 dir, NOT descended)
        # File must be inside a dir at depth=1 that tries to descend further.
        inner = tmp_path / "sub" / "inner"
        inner.mkdir(parents=True)
        (inner / "recent.txt").write_text("recent\n")

        result = agent_run._watch_scratch_facts(str(tmp_path))
        assert result["truncated"] is True, (
            "Non-pruned directory at max depth must set truncated=True"
        )
        assert result["error"] == "depth_limit"
        assert result["files_modified_recent"] is None
        assert result["newest_mtime_age_s"] is None
        assert result["scanned"] is None

    def test_depth_limit_file_at_cap_is_scanned(self, tmp_path, monkeypatch):
        """A file exactly at the depth cap must be scanned (no truncation from depth alone)."""
        monkeypatch.setattr(agent_run, "WATCH_SCRATCH_MAX_DEPTH", 1)
        # File at depth=1 (root/sub/file.txt): sub is enqueued from root (depth=0),
        # file.txt is inside sub (depth=1) which is at cap but still descended.
        at_cap = tmp_path / "sub" / "file.txt"
        at_cap.parent.mkdir()
        at_cap.write_text("recent\n")

        result = agent_run._watch_scratch_facts(str(tmp_path))
        # No directory beyond cap was encountered, so depth_limit must not fire.
        assert result["error"] is None, (
            "File at the depth cap must be scanned without triggering depth_limit"
        )
        assert result["files_modified_recent"] >= 1
        assert result["scanned"] >= 1

    def test_depth_propagation_regression(self, tmp_path, monkeypatch):
        """Depth must advance by exactly 1 per level; a non-advancing depth hides deep files.

        Mutation: replace ``depth + 1`` with ``depth`` in the queue append.
        With WATCH_SCRATCH_MAX_DEPTH=1 and a file three levels deep, the mutant
        never hits the depth cap (depth never grows past 0) and the file is
        scanned, making this test fail on error != depth_limit.
        """
        monkeypatch.setattr(agent_run, "WATCH_SCRATCH_MAX_DEPTH", 1)
        # Three-level tree: root/a/b/c.txt — only accessible if depth does not advance.
        deep = tmp_path / "a" / "b" / "c.txt"
        deep.parent.mkdir(parents=True)
        deep.write_text("data\n")

        result = agent_run._watch_scratch_facts(str(tmp_path))
        # With correct depth tracking: a/b is at depth 2 > MAX_DEPTH=1, so it is
        # not enqueued and the scan sees only the depth_limit truncation.
        assert result["truncated"] is True
        # depth_limit may be overridden by another code if multiple bounds fire,
        # but the scan must not report confident non-null counts.
        assert result["files_modified_recent"] is None, (
            "Depth propagation regression: depth not advancing lets deep files through"
        )

    def test_unreadable_subdir_returns_null_with_error(self, tmp_path):
        """A PermissionError on a discovered subdirectory must degrade to null + exact code.

        A confident zero here reads to the poller as observed inactivity and
        stalls a healthy run; only null decision fields say "unknown".
        """
        import stat as _stat

        subdir = tmp_path / "unreadable_subdir"
        subdir.mkdir()
        (subdir / "file.txt").write_text("data\n")
        subdir.chmod(0)  # remove all permissions

        try:
            result = agent_run._watch_scratch_facts(str(tmp_path))
            assert result["error"] == "scan_error", (
                "A PermissionError on a subdirectory must set error='scan_error'"
            )
            assert result["files_modified_recent"] is None, (
                "files_modified_recent must be null, not a confident 0"
            )
            assert result["newest_mtime_age_s"] is None
            assert result["scanned"] is None
            assert result["truncated"] is False
        finally:
            subdir.chmod(_stat.S_IRWXU)

    def test_file_vanishes_at_stat_returns_null_with_error(self, tmp_path, monkeypatch):
        """A FileNotFoundError while statting a file must degrade to null + error="stat_error".

        The counts gathered so far describe a tree that has already changed
        underneath the scan, so they must not be reported as observed.
        """
        (tmp_path / "real.txt").write_text("data\n")

        class VanishingEntry:
            """Proxy a real DirEntry but make stat() raise FileNotFoundError."""

            def __init__(self, entry):
                self._entry = entry

            def stat(self, **kw):
                raise FileNotFoundError(f"[Errno 2] No such file: '{self._entry.path}'")

            def __getattr__(self, name):
                return getattr(self._entry, name)

        _patch_scandir(monkeypatch, VanishingEntry)

        result = agent_run._watch_scratch_facts(str(tmp_path))
        assert result["error"] == "stat_error", (
            "A stat FileNotFoundError must set error='stat_error', not a different category"
        )
        assert result["files_modified_recent"] is None, (
            "files_modified_recent must be null after a stat error"
        )
        assert result["newest_mtime_age_s"] is None
        assert result["scanned"] is None
        assert result["truncated"] is False

    def test_file_vanishes_during_classification_returns_null_with_error(self, tmp_path, monkeypatch):
        """An entry that vanishes between enumeration and stat() must degrade to null + stat_error.

        With two separate predicate calls (is_dir then is_file), a file that
        disappears after enumeration returns false from both, taking the same
        path as an ignored FIFO/symlink and producing a confident zero.  The
        single no-follow stat at classification closes this: any OSError at
        that point raises stat_error rather than silently continuing.

        This regression is distinct from test_file_vanishes_at_stat: that test
        covers a file that passes is_file() and vanishes at stat(); this test
        covers the identical safety failure reached one gate earlier.
        Mutation: replacing S_ISREG with a constant-True check (so the entry
        is always classified as a regular file regardless of mode) is not the
        target here; the target is the OSError at stat() classification: if the
        stat() raise is swallowed (e.g. 'except OSError: continue'), the test
        fails because error is None and scanned=0 (confident zero).
        """
        (tmp_path / "real.txt").write_text("data\n")

        # A DirEntry that raises OSError on stat() simulates classification-time
        # disappearance: the entry existed during scandir() but is gone by stat().
        class VanishingDuringClassification:
            """Proxy a real DirEntry; raise OSError on stat() before classification."""

            def __init__(self, entry):
                self._entry = entry
                self._stat_count = 0

            def stat(self, **kw):
                self._stat_count += 1
                # Raise on the first stat() call (classification); the original
                # two-predicate code would have called is_dir/is_file first and
                # only stat()ed a surviving file.
                raise FileNotFoundError(
                    f"[Errno 2] No such file or directory: '{self._entry.path}'"
                )

            def __getattr__(self, name):
                return getattr(self._entry, name)

        _patch_scandir(monkeypatch, VanishingDuringClassification)

        result = agent_run._watch_scratch_facts(str(tmp_path))
        assert result["error"] == "stat_error", (
            "Entry vanishing during classification must set error='stat_error', "
            f"not produce a confident zero (got error={result['error']!r})"
        )
        assert result["files_modified_recent"] is None, (
            "files_modified_recent must be null when an entry vanishes during classification"
        )
        assert result["newest_mtime_age_s"] is None
        assert result["scanned"] is None
        assert result["truncated"] is False

    def test_root_deleted_after_validation_returns_null_with_error(self, tmp_path):
        """Root deleted between is_dir() and os.open() must degrade to null + stat_error.

        The root-existence check (is_dir) and the directory open are a TOCTOU
        pair.  With fd-based traversal the race window is narrower than with
        pathname-based scandir, but deleting the root before os.open still
        produces an OSError that must degrade to null decision fields, not zero.
        """
        target = tmp_path / "scanroot"
        target.mkdir()
        (target / "file.txt").write_text("data\n")

        # Delete the target between is_dir() and os.open() by removing it first.
        # We can simulate this deterministically: delete the directory before
        # calling _watch_scratch_facts (is_dir will return False and we get
        # not_a_directory) — that is the expected degradation path.
        import shutil as _shutil
        _shutil.rmtree(str(target))

        result = agent_run._watch_scratch_facts(str(target))
        assert result["error"] in ("not_a_directory", "stat_error", "scan_error"), (
            f"Missing root must set a categorical error "
            f"(got {result['error']!r})"
        )
        assert result["files_modified_recent"] is None, (
            "files_modified_recent must be null when root does not exist"
        )
        assert result["newest_mtime_age_s"] is None
        assert result["scanned"] is None
        assert result["truncated"] is False

    def test_error_field_contains_no_os_path_or_message(self, tmp_path, monkeypatch):
        """The 'error' field must carry categorical codes only, never OS messages.

        ``str(exc)`` on an OSError carries the filename it failed on (e.g.
        '/customer/merger-codename/secret.txt'), and this contract is rendered
        into Discord and persisted.  Plant a sentinel in the OS error message
        and assert it does not reach the error field.  Also checks all five
        scratch keys are present and decision fields are null.
        """
        SENTINEL = "SENTINEL_SECRET_PATH_XYZ987"

        class SentinelPath:
            """Proxy Path that raises OSError with the sentinel in the message on is_dir()."""

            def __init__(self, p):
                self._p = p

            def is_dir(self):
                raise OSError(f"[Errno 13] Permission denied: '{SENTINEL}/secret.txt'")

            def __str__(self):
                return str(self._p)

            def __fspath__(self):
                return str(self._p)

        monkeypatch.setattr(agent_run, "Path", lambda arg: SentinelPath(Path(arg)))

        result = agent_run._watch_scratch_facts(str(tmp_path))
        assert result["error"] == "stat_error"
        assert SENTINEL not in (result["error"] or ""), (
            f"Sentinel path '{SENTINEL}' leaked into error field: {result['error']!r}"
        )
        assert result["files_modified_recent"] is None
        assert result["newest_mtime_age_s"] is None
        assert result["scanned"] is None

    def test_error_field_contains_no_os_path_from_scan_error(self, tmp_path, monkeypatch):
        """The scandir failure path must set error='scan_error' and not leak the exception text.

        Deterministically requires scan_error: make scandir raise on the second
        call (the subdirectory).  All five scratch fields are asserted.
        """
        import os as _os

        SENTINEL = "SENTINEL_SCAN_PATH_ABC123"
        (tmp_path / "file.txt").write_text("data\n")

        real_scandir = _os.scandir

        call_count = [0]

        def scandir_raises_on_second(path):
            call_count[0] += 1
            if call_count[0] > 1:
                raise OSError(f"[Errno 5] I/O error: '{SENTINEL}/disk-failure'")
            return real_scandir(path)

        monkeypatch.setattr(_os, "scandir", scandir_raises_on_second)
        # Subdir ensures the walk has a second scandir call, making scan_error deterministic.
        (tmp_path / "subdir").mkdir()

        result = agent_run._watch_scratch_facts(str(tmp_path))
        assert result["error"] == "scan_error", (
            "scandir OSError must produce error='scan_error', not a different category"
        )
        assert SENTINEL not in result["error"], (
            f"Sentinel path '{SENTINEL}' leaked into error field: {result['error']!r}"
        )
        assert result["files_modified_recent"] is None
        assert result["newest_mtime_age_s"] is None
        assert result["scanned"] is None
        assert result["truncated"] is False

    def test_directory_symlinks_are_not_followed(self, tmp_path):
        """A directory symlink inside the scan root must not be descended.

        entry.is_dir(follow_symlinks=False) must be False for a symlink-to-dir,
        so the scanner skips it without following into the target.  Mutation:
        change follow_symlinks=False to follow_symlinks=True (or remove it) —
        the symlink is then classified as a directory and the external file is
        scanned, making this test fail because scanned > 0 from outside the root.
        """
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "external.txt").write_text("external data\n")

        scanroot = tmp_path / "scanroot"
        scanroot.mkdir()
        # Symlink to the outside directory — must not be followed.
        (scanroot / "link_to_outside").symlink_to(outside)
        # No regular files inside scanroot itself.

        result = agent_run._watch_scratch_facts(str(scanroot))
        assert result["error"] is None, (
            "A directory symlink must be skipped cleanly, not cause an error"
        )
        # No file inside scanroot (the symlink is not followed), so scanned=0
        # and no truncation.
        assert result["scanned"] == 0, (
            "Symlink-to-dir must not be followed; external file must not be scanned"
        )
        assert result["files_modified_recent"] == 0
        assert result["truncated"] is False

    def test_queued_dir_replaced_by_symlink_outside_injection(self, tmp_path):
        """A child dir replaced by a symlink after classification must degrade to error.

        Outside-positive injection: entry.stat() sees a real directory (cached
        from scandir enumeration), classification queues it for descent, but by
        the time os.open() runs the name has been replaced by a symlink to an
        outside tree with a recent file.  O_NOFOLLOW rejects the symlink and
        the scan degrades to a categorical error rather than counting an outside
        file as in-root positive evidence.

        This property is the confinement guarantee in the asymmetric truncation
        contract comment: files_modified_recent >= 1 is sound because the fd-
        based descent prevents outside files from being counted.

        Mutation: remove O_NOFOLLOW (or use a pathname instead of dir_fd) and
        the scan follows the symlink, files_modified_recent=1 from the outside
        file, error=None — this test fails.
        """
        import os as _os
        import shutil as _shutil

        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "outside_recent.txt").write_text("outside\n")

        scanroot = tmp_path / "scanroot"
        scanroot.mkdir()
        child = scanroot / "child"
        child.mkdir()
        # No files in child itself; the swap injects the outside recent file.

        real_scandir = _os.scandir
        swapped = [False]

        def scandir_swap_on_child(path):
            """Perform the swap right after priming the DirEntry stat cache.

            Prime entry.stat() before the swap so classification sees S_ISDIR
            (the cached pre-swap mode).  After the swap, os.open with O_NOFOLLOW
            sees the symlink and fails.  Without priming, entry.stat would do a
            live lstat after the swap, see S_ISLNK, and silently skip the entry
            — that path does not exercise the race this test covers.
            """
            it = real_scandir(path)

            class SwappingIter:
                def __iter__(self):
                    return self

                def __next__(self):
                    entry = next(it)
                    if not swapped[0] and entry.name == "child":
                        # Prime the stat cache: first call caches S_ISDIR.
                        entry.stat(follow_symlinks=False)
                        # Swap child for a symlink to the outside tree.
                        _shutil.rmtree(str(child))
                        child.symlink_to(outside)
                        swapped[0] = True
                    return entry

                def __enter__(self):
                    it.__enter__()
                    return self

                def __exit__(self, *args):
                    return it.__exit__(*args)

            return SwappingIter()

        original_scandir = _os.scandir
        _os.scandir = scandir_swap_on_child
        try:
            result = agent_run._watch_scratch_facts(str(scanroot))
        finally:
            _os.scandir = original_scandir

        assert result["error"] is not None, (
            "A child replaced by a symlink to an outside tree must degrade to "
            f"a categorical error (got error=None, files_modified_recent="
            f"{result['files_modified_recent']!r}) — outside file was counted"
        )
        assert result["files_modified_recent"] is None, (
            "files_modified_recent must be null when a queued dir was swapped for a symlink"
        )
        assert result["newest_mtime_age_s"] is None
        assert result["scanned"] is None

    def test_queued_dir_replaced_by_symlink_hides_activity(self, tmp_path):
        """A child dir with recent files replaced by symlink must degrade to error.

        Active-subtree hiding: child/recent.txt exists when child is enumerated
        (entry.stat() → S_ISDIR, cached), but child is replaced by a symlink
        to an old outside tree before descent.  O_NOFOLLOW rejects the symlink,
        returning a categorical error rather than a confident zero from the old
        outside tree.

        Mutation: remove O_NOFOLLOW from _DIR_OPEN_FLAGS and the scan follows
        the replacement symlink, sees only old files, reports
        files_modified_recent=0 with error=None — a confident zero from an
        unrelated tree — this test fails.
        """
        import os as _os
        import shutil as _shutil

        outside = tmp_path / "outside"
        outside.mkdir()
        old_time = __import__("time").time() - 7200
        old_file = outside / "old.txt"
        old_file.write_text("old\n")
        _os.utime(str(old_file), (old_time, old_time))

        scanroot = tmp_path / "scanroot"
        scanroot.mkdir()
        child = scanroot / "child"
        child.mkdir()
        (child / "recent.txt").write_text("recent activity\n")

        real_scandir = _os.scandir
        swapped = [False]

        def scandir_swap_on_child(path):
            it = real_scandir(path)

            class SwappingIter:
                def __iter__(self):
                    return self

                def __next__(self):
                    entry = next(it)
                    if not swapped[0] and entry.name == "child":
                        entry.stat(follow_symlinks=False)  # prime cache → S_ISDIR
                        _shutil.rmtree(str(child))
                        child.symlink_to(outside)
                        swapped[0] = True
                    return entry

                def __enter__(self):
                    it.__enter__()
                    return self

                def __exit__(self, *args):
                    return it.__exit__(*args)

            return SwappingIter()

        original_scandir = _os.scandir
        _os.scandir = scandir_swap_on_child
        try:
            result = agent_run._watch_scratch_facts(str(scanroot))
        finally:
            _os.scandir = original_scandir

        assert result["error"] is not None, (
            "A child swapped for a symlink to an old outside tree must degrade to "
            f"a categorical error, not a confident zero (got error=None, "
            f"files_modified_recent={result['files_modified_recent']!r})"
        )
        assert result["files_modified_recent"] is None, (
            "files_modified_recent must be null, not a confident zero from an unrelated tree"
        )
        assert result["newest_mtime_age_s"] is None
        assert result["scanned"] is None

    # ------------------------------------------------------------------
    # end-to-end tests through cmd_watch
    # ------------------------------------------------------------------

    def test_scratch_key_present_in_normal_branch(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch, capsys
    ):
        """Normal branch: the scan runs and reports the recent write.

        Asserts observed values, not key presence: the payload default is
        error='not_observed', so only a scan that actually ran gives None.
        """
        repo = tmp_path / "repo"
        _init_repo(repo)
        # Write a recent file so the scan has something to report.
        (repo / "recent_finding.md").write_text("analysis output\n")
        _make_run(
            isolated_runs_root, isolated_log_root, "sc1",
            status="running", pid=111, log_age_secs=1, cwd=str(repo),
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        agent_run.cmd_watch(_watch_args("sc1"))
        payload = json.loads(capsys.readouterr().out)
        assert set(payload.keys()) == WATCH_CONTRACT_KEYS
        scratch = payload["scratch"]
        assert set(scratch.keys()) == self.SCRATCH_KEYS
        # Branch-specific values: a real scan ran and saw the recent file.
        assert scratch["error"] is None, (
            f"Normal branch must run the scan (error={scratch['error']!r}); "
            "if _watch_scratch_facts is removed, error defaults to 'not_observed'"
        )
        assert scratch["files_modified_recent"] >= 1, (
            "Normal branch must report the recent file written above"
        )

    def test_scratch_key_present_in_missing_state_dir_branch(
        self, isolated_runs_root, isolated_log_root, tmp_path, capsys
    ):
        """Missing-state-dir branch: the scan runs off --repo, since the
        recorded cwd went with the state dir.

        Asserts observed values, not key presence: the payload default is
        error='not_observed', so only a scan that actually ran gives None.
        """
        workdir = tmp_path / "workdir"
        workdir.mkdir()
        # Write a recent file so the scan has something to report.
        (workdir / "finding.md").write_text("review notes\n")
        _make_run(
            isolated_runs_root, isolated_log_root, "sc2",
            write_state=False, log_text="line1\n",
        )
        agent_run.cmd_watch(_watch_args("sc2", repo=str(workdir)))
        payload = json.loads(capsys.readouterr().out)
        assert set(payload.keys()) == WATCH_CONTRACT_KEYS
        scratch = payload["scratch"]
        assert set(scratch.keys()) == self.SCRATCH_KEYS
        # Branch-specific values: the scan ran via --repo and saw the recent file.
        assert scratch["error"] is None, (
            f"Missing-state branch with --repo must run the scan (error={scratch['error']!r}); "
            "if _watch_scratch_facts is removed, error defaults to 'not_observed'"
        )
        assert scratch["files_modified_recent"] >= 1, (
            "Missing-state branch must report the recent file written above"
        )

    def test_scratch_key_present_in_observation_error_branch(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys
    ):
        """Observation-error branch: scratch defaults to not_observed with all five null/false fields.

        The most dangerous mutation replaces the not_observed default with
        files_modified_recent=0, scanned=0, error=null — this test must fail
        that mutant by checking all five values explicitly.
        """
        _make_run(
            isolated_runs_root, isolated_log_root, "sc3",
            status="running", pid=111, log_age_secs=1,
        )

        def _boom(*_a, **_kw):
            raise RuntimeError("synthetic crash")

        monkeypatch.setattr(agent_run, "_effective_status", _boom)
        agent_run.cmd_watch(_watch_args("sc3"))
        payload = json.loads(capsys.readouterr().out)
        assert set(payload.keys()) == WATCH_CONTRACT_KEYS
        scratch = payload["scratch"]
        assert set(scratch.keys()) == self.SCRATCH_KEYS
        # Observation-error path uses _watch_payload defaults = _scratch_unknown("not_observed").
        assert scratch["error"] == "not_observed", (
            "Observation-error branch must default to error='not_observed', "
            "not a confident zero with error=None"
        )
        assert scratch["files_modified_recent"] is None
        assert scratch["newest_mtime_age_s"] is None
        assert scratch["scanned"] is None
        assert scratch["truncated"] is False

    def test_review_shaped_regression(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch, capsys
    ):
        """Full end-to-end regression for the blind spot: a reviewer run with
        zero git evidence must still show scratch activity from a gitignored
        file written during the run."""
        repo = tmp_path / "repo"
        _repo_with_gitignored_finding(repo, "findings")
        launch_head = agent_run._watch_run_git_checked(
            repo, ["rev-parse", "HEAD"]
        ).stdout.strip()

        _make_run(
            isolated_runs_root, isolated_log_root, "sc4",
            status="running", pid=111, log_age_secs=1,
            cwd=str(repo), launch_head=launch_head,
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        agent_run.cmd_watch(_watch_args("sc4"))
        payload = json.loads(capsys.readouterr().out)

        # Confirm git sees nothing: no commits, no dirty files.
        git = payload["git"]
        assert git is not None
        assert git["commits_since_start"] == 0
        assert git["untracked_files"] == 0
        assert git["files_changed"] == 0

        # Scratch must report the gitignored file.
        scratch = payload["scratch"]
        assert scratch["error"] is None
        assert scratch["files_modified_recent"] >= 1
        assert scratch["newest_mtime_age_s"] is not None
        assert scratch["newest_mtime_age_s"] < agent_run.WATCH_SCRATCH_RECENT_SECONDS

    def test_repo_arg_does_not_redirect_scratch_scan(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch, capsys
    ):
        """--repo must not move the scratch scan off the run's recorded launch cwd.

        The active branch computes repo for git facts as repo_arg or recorded_cwd.
        Passing --repo must not silently replace the cwd used for the scratch scan:
        activity written to the recorded cwd must be visible even when --repo
        points at a different directory.
        """
        recorded_cwd = tmp_path / "recorded_cwd"
        recorded_cwd.mkdir()
        # Recent file in the recorded cwd — the scratch scan must see this.
        (recorded_cwd / "findings.md").write_text("analysis\n")

        repo_override = tmp_path / "repo_override"
        _init_repo(repo_override)
        # Only old file in the override directory.
        old_file = repo_override / "old.txt"
        old_file.write_text("old\n")
        old_time = time.time() - 7200
        os.utime(old_file, (old_time, old_time))

        _make_run(
            isolated_runs_root, isolated_log_root, "sc5",
            status="running", pid=111, log_age_secs=1,
            cwd=str(recorded_cwd),
        )
        monkeypatch.setattr(agent_run, "_pid_alive", lambda _p: True)
        # Pass --repo pointing at repo_override, which has no recent activity.
        agent_run.cmd_watch(_watch_args("sc5", repo=str(repo_override)))
        payload = json.loads(capsys.readouterr().out)

        scratch = payload["scratch"]
        # Git facts use repo_override (the --repo argument).
        assert payload["repo"] == str(repo_override)
        # Scratch must use the recorded cwd, not the --repo override.
        assert scratch["error"] is None, (
            f"Scratch scan must use recorded cwd, not --repo override "
            f"(error={scratch['error']!r})"
        )
        assert scratch["files_modified_recent"] >= 1, (
            "Scratch must see the recent file in recorded cwd, not the old file in repo_override"
        )
