"""Tests for linked-git-worktree accounting and garbage collection:

- `_worktree_classify`: linked worktree, main worktree, bare repo, non-git
  directory, missing path, worktree subdirectory, git failures.
- `agent-run du`: separate WORKTREE column, realpath deduplication, TOTAL
  reconciliation across nested worktrees and nested STATE_ROOT/LOG_ROOT,
  --json shape, read-only guarantee.
- `agent-run reap --include-worktrees`: age threshold, dirty/unpushed/ignored
  refusals, shared-cwd single removal, symlink and live-run guards, nested
  registered worktrees, pre-removal revalidation, budget ordering, and
  --dry-run prediction accuracy.

Every destructive-path test uses a real git repository and, where liveness
matters, a real process: a refusal that only a mock produces proves nothing
about what `git worktree remove` would have deleted.

Uses the tmp-dir + monkeypatched STATE_ROOT/LOG_ROOT pattern shared with
tests/test_agent_run_reap.py and tests/conftest.py.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import pytest

from toolbox import agent_run


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _git(cwd: Path, *args: str) -> str:
    """Run git in ``cwd`` with a deterministic identity and no user config."""
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True, text=True, check=True, env=env,
    )
    return result.stdout


def _make_repo(root: Path, name: str = "main") -> Path:
    """A main worktree with one commit, plus a bare 'origin' remote holding
    that same commit, so a clean worktree has no unpushed work."""
    repo = root / name
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main", ".")
    (repo / "tracked.txt").write_text("content\n")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "initial")
    origin = root / f"{name}-origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "push", "-q", "origin", "main")
    return repo


def _add_worktree(repo: Path, path: Path, branch: str) -> Path:
    """A linked worktree of ``repo`` at ``path``, on a branch that exists on
    the remote — so it starts with nothing unpushed."""
    _git(repo, "branch", "-f", branch, "main")
    _git(repo, "push", "-q", "origin", f"{branch}:{branch}")
    _git(repo, "worktree", "add", "-q", str(path), branch)
    return path


def _make_state_run(
    state_root: Path,
    log_root: Path,
    name: str,
    *,
    status: str = "done",
    cwd: Optional[Path] = None,
    age_hours: float = 0.0,
    log_bytes: int = 0,
) -> Path:
    """A state-backed run whose `cwd` file points at ``cwd``. ``age_hours``
    backdates ended_at and the state dir mtime, which is what
    `_terminal_state_age_seconds` reads."""
    sd = state_root / name
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "status").write_text(f"{status}\n")
    if cwd is not None:
        (sd / "cwd").write_text(f"{cwd}\n")
    ld = log_root / name
    ld.mkdir(parents=True, exist_ok=True)
    if log_bytes:
        (ld / "log").write_bytes(b"x" * log_bytes)
    if age_hours:
        old = time.time() - age_hours * 3600
        (sd / "ended_at").write_text(
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(old)) + "\n"
        )
        for p in (sd / "status", sd):
            os.utime(p, (old, old))
        if cwd is not None and cwd.is_dir():
            for root, dirs, files in os.walk(cwd):
                for entry in [*dirs, *files]:
                    os.utime(Path(root) / entry, (old, old), follow_symlinks=False)
            os.utime(cwd, (old, old))
    return sd


def _du_args(**kw) -> argparse.Namespace:
    base = dict(by_run=False, top=None, bytes=True, json=False)
    base.update(kw)
    return argparse.Namespace(**base)


def _reap_args(**kw) -> argparse.Namespace:
    base = dict(
        dry_run=False, idle_hours=None, min_age_hours=None, log_min_age_hours=None,
        name=None, force_unknown=False, include_logs=False, orphan_processes=False,
        orphan_min_age_hours=None, max_seconds=None, include_worktrees=True,
        worktree_min_age_hours=None, force_dirty=False, all=False,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def _row(out: str, key: str) -> list[str]:
    return next(ln for ln in out.splitlines() if ln.split() and ln.split()[0] == key).split()


def _exit_status(code: object) -> int:
    """The OS-visible exit status ``sys.exit(code)`` would produce: ``0`` for
    ``None``, ``code`` itself for an ``int``, else ``1`` (any message string
    is printed to stderr and the process exits 1). Distinguishes ordinary
    ``sys.exit(message)`` rejections (status 1) from this module's explicit
    ``_worktree_usage_error`` (``SystemExit(2)``)."""
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1


def _kill_run_pid(state_dir: Path) -> None:
    """Terminate the runner recorded in state_dir; no-op if already gone."""
    try:
        pid = int((state_dir / "pid").read_text().strip())
    except (FileNotFoundError, ValueError, OSError):
        return
    if pid <= 0:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        if sig == signal.SIGTERM:
            time.sleep(0.3)


def _wait_terminal(state_dir: Path, timeout: float = 15.0) -> str:
    """Poll <state_dir>/status until terminal; kill the runner on timeout."""
    deadline = time.monotonic() + timeout
    status = "starting"
    while time.monotonic() < deadline:
        try:
            status = (state_dir / "status").read_text().strip()
        except FileNotFoundError:
            status = "starting"
        if status in agent_run.TERMINAL_STATUSES:
            break
        time.sleep(0.05)
    if status not in agent_run.TERMINAL_STATUSES:
        _kill_run_pid(state_dir)
    return status


def _launch_args(**kw) -> argparse.Namespace:
    """A full cmd_launch-shaped Namespace, defaulted for a raw one-shot run.

    Every field cmd_launch/_create_launch_worktree can read via getattr is
    present so a hand-built call never falls through to a getattr default
    that would mask a bug.
    """
    base = dict(
        name="wt-test", command=[sys.executable, "-c", "pass"], interactive=False,
        prompt_file=None, submit_mode=None, idle_timeout=None,
        harness=None, prompt=None, model=None, agent_mode=None, harness_args=[],
        permissions="bypass", cwd=None,
        worktree=None, worktree_base=None, worktree_branch=None,
        worktree_repo=None, worktree_reuse=False,
        enable_planning=False, enable_questions=False,
    )
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def git_root(tmp_path) -> Path:
    d = tmp_path / "gitroots"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# _worktree_classify
# ---------------------------------------------------------------------------

class TestWorktreeClassify:
    def test_linked_worktree(self, git_root):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")

        info = agent_run._worktree_classify(wt)

        assert info.kind == agent_run._WORKTREE_LINKED
        assert info.common_dir is not None
        assert Path(info.common_dir).samefile(repo / ".git")

    def test_main_worktree(self, git_root):
        repo = _make_repo(git_root)

        assert agent_run._worktree_classify(repo).kind == agent_run._WORKTREE_MAIN

    def test_subdirectory_of_main_worktree_is_not_linked(self, git_root):
        repo = _make_repo(git_root)
        (repo / "sub").mkdir()

        assert agent_run._worktree_classify(repo / "sub").kind == agent_run._WORKTREE_MAIN

    def test_subdirectory_of_linked_worktree_is_unknown(self, git_root):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        (wt / "nested").mkdir()

        info = agent_run._worktree_classify(wt / "nested")

        assert info.kind == agent_run._WORKTREE_UNKNOWN
        assert "top of its worktree" in (info.detail or "")

    def test_bare_repo(self, git_root):
        bare = git_root / "bare.git"
        subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)

        assert agent_run._worktree_classify(bare).kind == agent_run._WORKTREE_BARE

    def test_non_git_directory(self, git_root):
        plain = git_root / "plain"
        plain.mkdir()

        assert agent_run._worktree_classify(plain).kind == agent_run._WORKTREE_NOT_A_REPO

    def test_missing_directory_is_unknown(self, git_root):
        assert agent_run._worktree_classify(git_root / "gone").kind == agent_run._WORKTREE_UNKNOWN

    def test_symlink_to_worktree_is_unknown(self, git_root):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        link = git_root / "link"
        link.symlink_to(wt, target_is_directory=True)

        # lstat-based: the symlink path itself is never a classifiable dir.
        assert agent_run._worktree_classify(link).kind == agent_run._WORKTREE_UNKNOWN

    def test_git_env_pollution_does_not_change_classification(self, git_root, monkeypatch):
        repo = _make_repo(git_root)
        other = _make_repo(git_root, "other")
        wt = _add_worktree(repo, git_root / "wt", "feature")
        monkeypatch.setenv("GIT_DIR", str(other / ".git"))
        monkeypatch.setenv("GIT_WORK_TREE", str(other))

        assert agent_run._worktree_classify(wt).kind == agent_run._WORKTREE_LINKED
        assert agent_run._worktree_classify(other).kind == agent_run._WORKTREE_MAIN

    @pytest.mark.parametrize("error", ["git_missing", "timeout", "git_failed"])
    def test_git_failures_are_unknown_never_linked(self, git_root, monkeypatch, error):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        monkeypatch.setattr(
            agent_run, "_watch_run_git_checked",
            lambda *a, **k: agent_run._WatchGitOutcome(None, error),
        )

        info = agent_run._worktree_classify(wt)

        assert info.kind == agent_run._WORKTREE_UNKNOWN
        assert info.detail == error

    def test_unparseable_rev_parse_output_is_unknown(self, git_root, monkeypatch):
        """An older git that does not understand --path-format, or any other
        cause of short output, must never be read as linked."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        monkeypatch.setattr(
            agent_run, "_watch_run_git_checked",
            lambda *a, **k: agent_run._WatchGitOutcome("false\ntrue\n", None),
        )

        assert agent_run._worktree_classify(wt).kind == agent_run._WORKTREE_UNKNOWN

    def test_classification_does_not_mutate_the_repo(self, git_root):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        admin = repo / ".git" / "worktrees"
        before = sorted(p.name for p in admin.iterdir())
        (wt / "untracked").write_text("x\n")

        agent_run._worktree_classify(wt)
        agent_run._worktree_classify(repo)

        assert sorted(p.name for p in admin.iterdir()) == before
        assert (wt / "untracked").exists()
        assert _git(wt, "status", "--porcelain") == "?? untracked\n"


class TestWorktreeResolveCwd:
    def test_symlinked_prefixes_resolve_to_one_path(self, git_root):
        real = git_root / "real"
        real.mkdir()
        link = git_root / "link"
        link.symlink_to(real, target_is_directory=True)

        assert agent_run._worktree_resolve_cwd(str(link)) == agent_run._worktree_resolve_cwd(str(real))

    def test_missing_or_empty_is_none(self, git_root):
        assert agent_run._worktree_resolve_cwd("") is None
        assert agent_run._worktree_resolve_cwd(str(git_root / "gone")) is None

    def test_regular_file_is_none(self, git_root):
        f = git_root / "file"
        f.write_text("x")
        assert agent_run._worktree_resolve_cwd(str(f)) is None

    def test_relative_path_is_none(self, git_root):
        """A relative path is rejected inside _worktree_resolve_cwd itself so
        os.path.realpath cannot silently resolve it against the process cwd."""
        assert agent_run._worktree_resolve_cwd("relative/path") is None
        assert agent_run._worktree_resolve_cwd("wt") is None

    def test_relative_cwd_in_state_dir_is_refused(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """A state dir whose cwd file holds a relative path is refused with
        'is not absolute'; the candidate is skipped, not deleted."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        sd = _make_state_run(
            isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000
        )
        (sd / "cwd").write_text("relative/path\n")

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "is not absolute" in out or "unresolvable cwd" in out


# ---------------------------------------------------------------------------
# du worktree accounting
# ---------------------------------------------------------------------------

class TestDuWorktrees:
    def test_worktree_column_needs_no_flag(self):
        """Worktree accounting is unconditional: `du` takes no flag for it,
        and an unknown --worktrees is a parse error."""
        parsed = agent_run._build_parser().parse_args(["du"])
        assert not hasattr(parsed, "worktrees")
        with pytest.raises(SystemExit):
            agent_run._build_parser().parse_args(["du", "--worktrees"])

    def test_linked_worktree_counted_in_own_column(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        (wt / "big").write_bytes(b"x" * 5000)
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, log_bytes=10)
        expected = agent_run._dir_size_bytes(wt)

        agent_run.cmd_du(_du_args(by_run=True))

        out = capsys.readouterr().out
        assert "WORKTREE" in out
        fields = _row(out, "r1")
        assert fields[5] == str(expected)
        assert int(fields[5]) >= 5000
        # TOTAL == state + log + scratch + worktree, all in this one row.
        assert int(fields[6]) == sum(int(f) for f in fields[2:6])

    @pytest.mark.parametrize("kind", ["main", "bare", "plain", "missing"])
    def test_non_linked_cwds_contribute_zero(
        self, isolated_runs_root, isolated_log_root, git_root, capsys, kind
    ):
        repo = _make_repo(git_root)
        (repo / "payload").write_bytes(b"x" * 4096)
        if kind == "main":
            cwd = repo
        elif kind == "bare":
            cwd = git_root / "bare.git"
            subprocess.run(["git", "init", "-q", "--bare", str(cwd)], check=True)
        elif kind == "plain":
            cwd = git_root / "plain"
            cwd.mkdir()
            (cwd / "payload").write_bytes(b"x" * 4096)
        else:
            cwd = git_root / "gone"
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=cwd)

        agent_run.cmd_du(_du_args(by_run=True))

        assert _row(capsys.readouterr().out, "r1")[5] == "0"

    def test_shared_worktree_charged_once(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """Three runs launched in one worktree must add its size once; the
        naive per-run sum would report it three times."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        (wt / "big").write_bytes(b"x" * 8000)
        size = agent_run._dir_size_bytes(wt)
        for name in ("a-run", "b-run", "c-run"):
            _make_state_run(isolated_runs_root, isolated_log_root, name, cwd=wt)

        agent_run.cmd_du(_du_args(by_run=True))

        out = capsys.readouterr().out
        charged = [int(_row(out, n)[5]) for n in ("a-run", "b-run", "c-run")]
        assert charged == [size, 0, 0]
        assert int(_row(out, "TOTAL")[5]) == size

    def test_shared_worktree_via_symlinked_cwd_charged_once(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        (wt / "big").write_bytes(b"x" * 8000)
        link = git_root / "wt-link"
        link.symlink_to(wt, target_is_directory=True)
        size = agent_run._dir_size_bytes(wt)
        _make_state_run(isolated_runs_root, isolated_log_root, "a-run", cwd=wt)
        _make_state_run(isolated_runs_root, isolated_log_root, "b-run", cwd=link)

        agent_run.cmd_du(_du_args(by_run=True))

        out = capsys.readouterr().out
        assert int(_row(out, "a-run")[5]) == size
        assert int(_row(out, "b-run")[5]) == 0
        assert int(_row(out, "TOTAL")[5]) == size

    def test_total_reconciles_across_rollup_and_by_run(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt1 = _add_worktree(repo, git_root / "wt1", "f1")
        wt2 = _add_worktree(repo, git_root / "wt2", "f2")
        (wt1 / "big").write_bytes(b"x" * 3000)
        (wt2 / "big").write_bytes(b"x" * 1000)
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt1, log_bytes=11)
        _make_state_run(isolated_runs_root, isolated_log_root, "r2", cwd=wt2, log_bytes=22)
        _make_state_run(isolated_runs_root, isolated_log_root, "r3", status="failed", cwd=wt1)

        agent_run.cmd_du(_du_args(by_run=True))
        by_run_out = capsys.readouterr().out
        agent_run.cmd_du(_du_args())
        rollup_out = capsys.readouterr().out

        by_run_total = _row(by_run_out, "TOTAL")
        rollup_total = _row(rollup_out, "TOTAL")
        assert by_run_total[2:] == rollup_total[2:]
        # Column-wise sum of the per-run rows equals TOTAL, and TOTAL's own
        # TOTAL equals the sum of its four size columns.
        per_run = [_row(by_run_out, n) for n in ("r1", "r2", "r3")]
        for col in range(2, 7):
            assert sum(int(f[col]) for f in per_run) == int(by_run_total[col])
        assert int(by_run_total[6]) == sum(int(f) for f in by_run_total[2:6])

    def test_json_exposes_worktree_bytes_and_attribution(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        (wt / "big").write_bytes(b"x" * 2000)
        size = agent_run._dir_size_bytes(wt)
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt)
        _make_state_run(isolated_runs_root, isolated_log_root, "r2", cwd=wt)

        agent_run.cmd_du(_du_args(by_run=True, bytes=False, json=True))

        payload = json.loads(capsys.readouterr().out)
        assert "charged once" in payload["worktree_attribution"]
        by_name = {r["name"]: r for r in payload["runs"]}
        assert by_name["r1"]["worktree_bytes"] == size
        assert by_name["r2"]["worktree_bytes"] == 0
        assert payload["total"]["worktree_bytes"] == size
        assert payload["total"]["total_bytes"] == sum(
            payload["total"][k]
            for k in ("state_bytes", "log_bytes", "scratch_bytes", "worktree_bytes")
        )

    def test_json_reports_zero_worktree_bytes_without_any_worktree(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", log_bytes=5)

        agent_run.cmd_du(_du_args(by_run=True, bytes=False, json=True))

        payload = json.loads(capsys.readouterr().out)
        assert payload["total"]["worktree_bytes"] == 0
        assert payload["runs"][0]["worktree_bytes"] == 0
        assert "charged once" in payload["worktree_attribution"]

    def test_du_worktree_bytes_when_cwd_is_subdirectory(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """A run launched from a subdirectory of a linked worktree must still
        contribute the whole worktree's bytes to the WORKTREE column.

        _worktree_classify returns UNKNOWN for a subdirectory (not the top of
        its worktree), so the previous code reported WORKTREE=0 for such runs.
        _worktree_du_root resolves --show-toplevel to find and charge the actual
        linked-worktree root.

        Mutation: replacing _worktree_du_root with _worktree_classify().kind ==
        _WORKTREE_LINKED causes WORKTREE=0 for subdirectory-launched runs,
        because _worktree_classify returns UNKNOWN for them."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        (wt / "payload.bin").write_bytes(b"p" * 6000)
        subdir = wt / "src" / "pkg"
        subdir.mkdir(parents=True)
        expected_wt_size = agent_run._dir_size_bytes(wt)
        # Run whose cwd is a subdirectory of the linked worktree.
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=subdir)

        agent_run.cmd_du(_du_args(by_run=True))

        out = capsys.readouterr().out
        worktree_col = int(_row(out, "r1")[5])
        assert worktree_col == expected_wt_size, (
            f"run launched from subdirectory reported WORKTREE={worktree_col}, "
            f"expected {expected_wt_size} (the whole worktree)"
        )

    def test_multiple_subdirectory_runs_share_worktree_charge(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """Multiple runs launched from different subdirectories of one worktree
        each resolve to the same top-level root and collectively charge it once.

        Mutation: without the _worktree_du_root deduplication, each subdirectory-
        run resolves to the same toplevel but the owners dict would only charge
        the root once anyway.  This test verifies the total is not doubled."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        (wt / "payload.bin").write_bytes(b"q" * 4000)
        sub_a = wt / "a"
        sub_b = wt / "b"
        sub_a.mkdir()
        sub_b.mkdir()
        expected_wt_size = agent_run._dir_size_bytes(wt)
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=sub_a)
        _make_state_run(isolated_runs_root, isolated_log_root, "r2", cwd=sub_b)

        agent_run.cmd_du(_du_args(by_run=True))

        out = capsys.readouterr().out
        wt_r1 = int(_row(out, "r1")[5])
        wt_r2 = int(_row(out, "r2")[5])
        assert wt_r1 + wt_r2 == expected_wt_size, (
            f"two subdirectory runs charged {wt_r1}+{wt_r2}={wt_r1+wt_r2}, "
            f"expected {expected_wt_size} total (charged once)"
        )

    def test_du_never_mutates_the_worktree(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        (wt / "untracked").write_text("keep me\n")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt)
        admin_before = sorted(p.name for p in (repo / ".git" / "worktrees").iterdir())

        agent_run.cmd_du(_du_args(by_run=True))
        capsys.readouterr()

        assert wt.is_dir()
        assert (wt / "untracked").read_text() == "keep me\n"
        assert sorted(p.name for p in (repo / ".git" / "worktrees").iterdir()) == admin_before


# ---------------------------------------------------------------------------
# reap --include-worktrees
# ---------------------------------------------------------------------------

class TestReapWorktrees:
    def test_off_by_default(self, isolated_runs_root, isolated_log_root, git_root, capsys):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        agent_run.cmd_reap(_reap_args(include_worktrees=False))

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "worktrees_removed=0" in out

    def test_clean_linked_worktree_removed_with_admin_metadata(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert not wt.exists()
        assert "worktrees_removed=1" in out
        # A bare rmtree would leave .git/worktrees/wt registered.
        listed = [
            ln.split(" ", 1)[1]
            for ln in _git(repo, "worktree", "list", "--porcelain").splitlines()
            if ln.startswith("worktree ")
        ]
        assert listed == [str(repo.resolve())]
        admin = repo / ".git" / "worktrees"
        assert not admin.exists() or "wt" not in [p.name for p in admin.iterdir()]

    @pytest.mark.parametrize("kind", ["main", "plain", "bare", "missing"])
    def test_never_removes_non_linked_cwd(
        self, isolated_runs_root, isolated_log_root, git_root, capsys, kind
    ):
        repo = _make_repo(git_root)
        if kind == "main":
            cwd = repo
        elif kind == "plain":
            cwd = git_root / "home"
            cwd.mkdir()
            (cwd / "precious.txt").write_text("do not delete\n")
        elif kind == "bare":
            cwd = git_root / "bare.git"
            subprocess.run(["git", "init", "-q", "--bare", str(cwd)], check=True)
        else:
            cwd = git_root / "gone"
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=cwd, age_hours=1000)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert "worktrees_removed=0" in out
        if kind == "missing":
            assert "unresolvable cwd" in out
            assert "worktrees_skipped=1" in out
        else:
            assert cwd.is_dir()
            assert "not a linked worktree" in out
        if kind == "plain":
            assert (cwd / "precious.txt").exists()

    def test_classify_failure_refuses_without_reaching_downstream_checks(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """Classification failure terminates the check pipeline immediately;
        nested-worktree and content checks are never reached."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        nested_calls: list = []
        content_calls: list = []
        monkeypatch.setattr(
            agent_run, "_worktree_nested_reason",
            lambda *a, **k: nested_calls.append(1) or None,
        )
        monkeypatch.setattr(
            agent_run, "_worktree_content_reason",
            lambda *a, **k: content_calls.append(1) or None,
        )

        # Fail only the rev-parse that _worktree_classify uses; all other git
        # calls succeed so the test isolates the classification check.
        real_git = agent_run._watch_run_git_checked

        def git_fails_classify(path, args, **kw):
            if list(args[:2]) == ["rev-parse", "--path-format=absolute"]:
                return agent_run._WatchGitOutcome(None, "git_failed")
            return real_git(path, args, **kw)

        monkeypatch.setattr(agent_run, "_watch_run_git_checked", git_fails_classify)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "worktrees_removed=0 worktrees_skipped=1" in out
        assert not nested_calls, "nested check must not be reached after classify failure"
        assert not content_calls, "content check must not be reached after classify failure"

    def test_ls_files_modified_failure_refuses(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """A failure in the ls-files --modified check refuses deletion; the
        worktree is never passed to _worktree_remove."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        real_git = agent_run._watch_run_git_checked

        def git_fails_ls_modified(path, args, **kw):
            if "ls-files" in list(args) and "--modified" in list(args):
                return agent_run._WatchGitOutcome(None, "git_failed")
            return real_git(path, args, **kw)

        monkeypatch.setattr(agent_run, "_watch_run_git_checked", git_fails_ls_modified)

        remove_calls: list = []
        monkeypatch.setattr(
            agent_run, "_worktree_remove",
            lambda *a, **k: remove_calls.append(1) or None,
        )

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert not remove_calls, "_worktree_remove must not be called on git failure"
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    def test_worktree_list_failure_refuses(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """A failure in the worktree-list call used by _worktree_nested_reason
        refuses deletion; no content checks or removal follow."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        real_git = agent_run._watch_run_git_checked

        def git_fails_worktree_list(path, args, **kw):
            if list(args[:2]) == ["worktree", "list"]:
                return agent_run._WatchGitOutcome(None, "git_failed")
            return real_git(path, args, **kw)

        monkeypatch.setattr(agent_run, "_watch_run_git_checked", git_fails_worktree_list)

        remove_calls: list = []
        monkeypatch.setattr(
            agent_run, "_worktree_remove",
            lambda *a, **k: remove_calls.append(1) or None,
        )

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert not remove_calls, "_worktree_remove must not be called on nested-check failure"
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    def test_rev_list_failure_refuses(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """A failure in the rev-list --count call used by _worktree_unpushed_reason
        refuses deletion; the worktree is never passed to _worktree_remove."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        real_git = agent_run._watch_run_git_checked

        def git_fails_rev_list(path, args, **kw):
            if list(args[:2]) == ["rev-list", "--count"]:
                return agent_run._WatchGitOutcome(None, "git_failed")
            return real_git(path, args, **kw)

        monkeypatch.setattr(agent_run, "_watch_run_git_checked", git_fails_rev_list)

        remove_calls: list = []
        monkeypatch.setattr(
            agent_run, "_worktree_remove",
            lambda *a, **k: remove_calls.append(1) or None,
        )

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert not remove_calls, "_worktree_remove must not be called on rev-list failure"
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    def test_untracked_files_refused(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        (wt / "scratch.txt").write_text("agent output\n")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "untracked file(s)" in out
        assert "--force-dirty" in out
        assert "worktrees_skipped=1" in out

    def test_modified_tracked_file_refused(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        (wt / "tracked.txt").write_text("edited\n")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        agent_run.cmd_reap(_reap_args())

        assert wt.is_dir()
        assert "worktrees_skipped=1" in capsys.readouterr().out

    def test_unpushed_commit_refused(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        (wt / "tracked.txt").write_text("committed but unpushed\n")
        _git(wt, "add", "tracked.txt")
        _git(wt, "commit", "-qm", "local only")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "not on any remote-tracking ref" in out
        assert "worktrees_skipped=1" in out

    def test_stale_remote_tracking_ref_does_not_block_removal(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """A branch deleted on the remote but with an unpruned local tracking
        ref still shows rev-list count=0 and does NOT block removal.

        This is deliberate: _worktree_unpushed_reason reads refs/remotes/* from
        the local repository without contacting any remote.  Network calls in a
        GC path can hang, require authentication, or fail unpredictably.  The
        documented contract is 'commits not reachable from any local remote-
        tracking ref'; stale tracking refs (branch deleted remotely but not
        pruned locally) are not detected.  Callers needing current remote state
        should run 'git fetch --prune' before reap.

        This test pins the documented behaviour so the gap is deliberate and
        visible rather than implied.

        Mutation: adding remote-state verification (e.g. checking if the remote
        branch still exists) would cause a test needing network access or a
        complex mock, and would make deletion decisions depend on connectivity."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        # Simulate a remote-side deletion without pruning locally: remove the
        # remote branch ref directly from the bare origin so the push history
        # record stays intact, while the local tracking ref remains.
        origin = git_root / "main-origin.git"
        remote_ref = origin / "refs" / "heads" / "feature"
        remote_ref.unlink()
        # Verify the local tracking ref still exists (no prune performed).
        tracking_exists = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--verify", "refs/remotes/origin/feature"],
            capture_output=True,
        ).returncode == 0
        assert tracking_exists, "test setup: local tracking ref must still exist after remote deletion"

        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        # HEAD is reachable from the stale local tracking ref: count=0, removable.
        assert not wt.exists(), (
            "worktree must be removable when HEAD is still on a (stale) local tracking ref"
        )
        assert "not on any remote-tracking ref" not in out
        assert "worktrees_removed=1" in out

    def test_force_dirty_removes_dirty_worktree(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        (wt / "scratch.txt").write_text("agent output\n")
        (wt / "tracked.txt").write_text("edited\n")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        agent_run.cmd_reap(_reap_args(force_dirty=True))

        assert not wt.exists()
        assert "worktrees_removed=1" in capsys.readouterr().out

    def test_live_run_sharing_cwd_blocks_removal(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "a-done", cwd=wt, age_hours=1000)
        live = _make_state_run(
            isolated_runs_root, isolated_log_root, "b-live", status="running", cwd=wt
        )
        (live / "pid").write_text(f"{os.getpid()}\n")
        (live / "process_identity").write_text(f"{agent_run._process_identity(os.getpid())}\n")

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "a live run is using this directory" in out
        assert "worktrees_skipped=1" in out

    def test_shared_worktree_removed_once(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        for name in ("a-run", "b-run", "c-run"):
            _make_state_run(isolated_runs_root, isolated_log_root, name, cwd=wt, age_hours=1000)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert not wt.exists()
        assert "worktrees_removed=1" in out
        assert "worktrees_skipped=0" in out
        assert out.count("[removing]") == 1
        assert "runs: a-run, b-run, c-run" in out

    def test_symlinked_cwd_component_refused(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        link = git_root / "wt-link"
        link.symlink_to(wt, target_is_directory=True)
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=link, age_hours=1000)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "not a real directory" in out
        assert "worktrees_skipped=1" in out

    def test_age_threshold_boundaries(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        young = _add_worktree(repo, git_root / "young", "f-young")
        old = _add_worktree(repo, git_root / "old", "f-old")
        _make_state_run(isolated_runs_root, isolated_log_root, "young", cwd=young, age_hours=9)
        _make_state_run(isolated_runs_root, isolated_log_root, "old", cwd=old, age_hours=11)

        agent_run.cmd_reap(_reap_args(worktree_min_age_hours=10))

        out = capsys.readouterr().out
        assert young.is_dir()
        assert not old.exists()
        assert "worktrees_removed=1" in out

    def test_youngest_sharing_run_governs_the_age_gate(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """A recently finished run keeps a shared worktree even when an older
        sibling would pass the threshold on its own."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "a-old", cwd=wt, age_hours=1000)
        _make_state_run(isolated_runs_root, isolated_log_root, "b-new", cwd=wt, age_hours=1)

        agent_run.cmd_reap(_reap_args())

        assert wt.is_dir()
        assert "worktrees_removed=0" in capsys.readouterr().out

    def test_env_threshold_used_when_flag_absent(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=5)
        monkeypatch.setenv("AGENT_RUN_WORKTREE_MIN_AGE_HOURS", "2")

        agent_run.cmd_reap(_reap_args())

        assert not wt.exists()
        assert "worktrees_removed=1" in capsys.readouterr().out

    @pytest.mark.parametrize("raw", ["nan", "inf", "-1", "0", "abc"])
    def test_invalid_env_threshold_falls_back_to_default(self, monkeypatch, capsys, raw):
        monkeypatch.setenv("AGENT_RUN_WORKTREE_MIN_AGE_HOURS", raw)

        assert agent_run._parse_worktree_min_age_seconds() == 168 * 3600
        assert "AGENT_RUN_WORKTREE_MIN_AGE_HOURS" in capsys.readouterr().err

    @pytest.mark.parametrize("raw", ["nan", "inf", "-1", "0"])
    def test_invalid_flag_threshold_rejected_by_argparse(self, raw):
        with pytest.raises(SystemExit):
            agent_run._build_parser().parse_args(["reap", "--worktree-min-age-hours", raw])

    def test_non_terminal_run_cwd_is_not_a_candidate(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        sd = _make_state_run(
            isolated_runs_root, isolated_log_root, "r1", status="running", cwd=wt, age_hours=1000
        )
        (sd / "pid").write_text(f"{os.getpid()}\n")

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "worktrees_removed=0 worktrees_skipped=0" in out

    def test_all_enables_the_worktree_pass(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        agent_run.cmd_reap(_reap_args(include_worktrees=False, all=True))

        assert not wt.exists()
        assert "worktrees_removed=1" in capsys.readouterr().out

    def test_all_does_not_imply_force_dirty(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """Untracked content is a refusal, not a pass: --all leaves the
        worktree in place until --force-dirty is given explicitly."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        (wt / "untracked").write_text("unpushed work\n")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        agent_run.cmd_reap(_reap_args(include_worktrees=False, all=True))

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert (wt / "untracked").read_text() == "unpushed work\n"
        assert "worktrees_removed=0" in out
        assert "worktrees_skipped=1" in out

    def test_unknown_status_not_collected_even_with_force_unknown(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(
            isolated_runs_root, isolated_log_root, "r1", status="weird", cwd=wt, age_hours=1000
        )

        agent_run.cmd_reap(_reap_args(force_unknown=True))

        assert wt.is_dir()
        assert "worktrees_removed=0" in capsys.readouterr().out


def _make_legacy_state_dir(state_root: Path, name: str) -> Path:
    """A pre-`status` state directory: bookkeeping files only, with no
    ``status``, ``pid`` or ``cwd``. Reproduces the shape left by an old
    agent-run release whose runner recorded only argv/exit metadata."""
    sd = state_root / name
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "argv").write_text("sleep 1\n")
    (sd / "command").write_text("sleep 1\n")
    (sd / "echo").write_text("0\n")
    (sd / "ended_at").write_text("2020-01-01T00:00:00Z\n")
    (sd / "exit_code").write_text("0\n")
    (sd / "pgid").write_text("4242\n")
    (sd / "reap_reason").write_text("legacy\n")
    return sd


class TestReapWorktreeLivenessEvidence:
    """An unresolvable cwd is fatal to the pass only when the entry proves a
    live runner; without that evidence it can protect no directory."""

    def test_legacy_state_dir_does_not_abort_the_pass(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_legacy_state_dir(isolated_runs_root, "tc-example")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert not wt.exists()
        assert "worktrees_removed=1" in out
        assert "liveness scan failed" not in out
        # Reported once in aggregate, naming the unusable state directory.
        assert "no live runner and an unresolvable cwd: tc-example" in out
        assert "worktrees_skipped=1" in out

    def test_many_legacy_dirs_report_once(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt1 = _add_worktree(repo, git_root / "wt1", "f1")
        wt2 = _add_worktree(repo, git_root / "wt2", "f2")
        for legacy in ("tc-alpha", "tc-beta"):
            _make_legacy_state_dir(isolated_runs_root, legacy)
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt1, age_hours=1000)
        _make_state_run(isolated_runs_root, isolated_log_root, "r2", cwd=wt2, age_hours=1000)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert not wt1.exists() and not wt2.exists()
        assert "worktrees_removed=2" in out
        assert out.count("no live runner and an unresolvable cwd") == 1
        assert "tc-alpha, tc-beta" in out
        assert "worktrees_skipped=1" in out

    def test_dead_recorded_pid_with_unresolvable_cwd_does_not_abort(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """A recorded pid that has exited is not evidence of liveness."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        dead = subprocess.Popen(["true"])
        dead.wait()
        stale = _make_legacy_state_dir(isolated_runs_root, "tc-stale")
        (stale / "pid").write_text(f"{dead.pid}\n")
        (stale / "process_identity").write_text("stale-identity\n")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert not wt.exists()
        assert "liveness scan failed" not in out
        assert "no live runner and an unresolvable cwd: tc-stale" in out
        assert "worktrees_removed=1" in out

    def test_live_identity_matched_runner_with_unresolvable_cwd_aborts(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """A verified live runner may hold any directory: its unreadable cwd
        keeps every candidate in the pass."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        live = _make_state_run(
            isolated_runs_root, isolated_log_root, "z-live", status="running"
        )
        (live / "pid").write_text(f"{os.getpid()}\n")
        (live / "process_identity").write_text(f"{agent_run._process_identity(os.getpid())}\n")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "liveness scan failed: cannot resolve cwd for live run z-live" in out
        assert "worktrees_removed=0" in out

    def test_live_pid_with_unverifiable_identity_aborts(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """An identity that cannot be read at all is ambiguity, not absence."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        live = _make_state_run(
            isolated_runs_root, isolated_log_root, "z-live", status="running"
        )
        (live / "pid").write_text(f"{os.getpid()}\n")
        (live / "process_identity").write_text("recorded-token\n")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)
        monkeypatch.setattr(agent_run, "_process_identity", lambda pid: None)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "liveness scan failed" in out
        assert "unverifiable identity" in out
        assert "worktrees_removed=0" in out

    def test_unresolvable_cwd_without_evidence_is_not_treated_as_live(self, tmp_path):
        """`_worktree_live_run_cwds` reports the name instead of erroring."""
        state_root = tmp_path / "state"
        _make_legacy_state_dir(state_root, "tc-example")

        result = agent_run._worktree_live_run_cwds([state_root / "tc-example"])

        assert result.error is None
        assert result.paths == []
        assert result.unresolved == ("tc-example",)

    @pytest.mark.parametrize("pid_obstruction", ["mode_000", "is_a_directory"])
    def test_unreadable_pid_file_aborts_liveness_scan(
        self, isolated_runs_root, isolated_log_root, git_root, capsys, pid_obstruction
    ):
        """An unreadable pid file (mode 000 or a directory in its place) is
        ambiguity, not absence: _worktree_state_has_live_runner must return
        (None, reason) so the liveness scan aborts and the pass refuses."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        sd = _make_state_run(
            isolated_runs_root, isolated_log_root, "a-run",
            status="done", cwd=wt, age_hours=1000,
        )
        pid_path = sd / "pid"
        if pid_obstruction == "mode_000":
            pid_path.write_text("99999\n")
            pid_path.chmod(0o000)
        else:
            pid_path.mkdir()

        try:
            live, err = agent_run._worktree_state_has_live_runner(sd)
            assert live is None, "unreadable pid must not return False (no-runner)"
            assert err is not None and "unreadable pid" in err

            agent_run.cmd_reap(_reap_args())
        finally:
            if pid_obstruction == "mode_000":
                pid_path.chmod(0o644)

        out = capsys.readouterr().out
        assert wt.is_dir(), "worktree deleted despite unreadable pid file"
        assert "liveness scan failed" in out
        assert "worktrees_removed=0" in out


class TestReapWorktreeCopiedForged:
    """Copied or forged linked-worktree metadata must not reach the destructive path."""

    def test_copied_worktree_refused(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """A recursively copied linked worktree keeps its .git file pointing at
        the original admin dir, so _worktree_classify returns LINKED.  But the
        copy is not registered in git worktree list; git worktree remove would
        refuse it.  The fix requires the candidate's inode to match a registered
        root before accepting it for deletion.

        Mutation: removing the registered-root inode check in
        _worktree_candidate_refusal allows the copied path to advance to
        git worktree remove --force, which then fails (preventing actual
        deletion in this case), but the guard must be there before the
        destructive command rather than relying on git's own refusal."""
        import shutil

        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        (wt / "important.txt").write_text("must survive\n")

        # Recursively copy the worktree; the .git file now points at wt's admin.
        wt_copy = git_root / "wt-copy"
        shutil.copytree(wt, wt_copy, symlinks=True)
        assert (wt_copy / ".git").exists(), "copy must have a .git file"

        _make_state_run(
            isolated_runs_root, isolated_log_root, "r1", cwd=wt_copy, age_hours=1000
        )

        agent_run.cmd_reap(_reap_args(force_dirty=True))

        out = capsys.readouterr().out
        assert wt_copy.is_dir(), "copied worktree directory must not be deleted"
        assert "not a registered worktree root" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    def test_forged_git_file_refused(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """A plain directory with a crafted .git file pointing at a real worktree's
        admin dir is classified as LINKED but is not registered.

        This verifies that the registration check catches forged metadata."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")

        # Read the .git file from the real worktree.
        git_file_content = (wt / ".git").read_text()

        # Create a plain directory with a forged .git file.
        forged = git_root / "forged"
        forged.mkdir()
        (forged / ".git").write_text(git_file_content)
        (forged / "data.bin").write_bytes(b"x" * 200)

        _make_state_run(
            isolated_runs_root, isolated_log_root, "r1", cwd=forged, age_hours=1000
        )

        agent_run.cmd_reap(_reap_args(force_dirty=True))

        out = capsys.readouterr().out
        assert forged.is_dir(), "forged worktree directory must not be deleted"
        assert "not a registered worktree root" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out


class TestReapWorktreesDryRun:
    def _run_both(self, args_kw, capsys):
        dry = capsys
        agent_run.cmd_reap(_reap_args(dry_run=True, **args_kw))
        dry_out = dry.readouterr().out
        agent_run.cmd_reap(_reap_args(**args_kw))
        real_out = capsys.readouterr().out
        return dry_out, real_out

    def _counts(self, out: str) -> tuple[str, str]:
        removed = next(f for f in out.split() if f.startswith("worktrees_removed="))
        skipped = next(f for f in out.split() if f.startswith("worktrees_skipped="))
        return removed, skipped

    def test_dry_run_mutates_nothing(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        agent_run.cmd_reap(_reap_args(dry_run=True))

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "dry-run" in out
        assert "worktrees_removed=1" in out
        assert (isolated_runs_root / "r1").is_dir()

    def test_dry_run_predicts_removal(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(
            isolated_runs_root, isolated_log_root, "r1", cwd=git_root / "wt", age_hours=1000
        )

        dry_out, real_out = self._run_both({}, capsys)

        assert self._counts(dry_out) == self._counts(real_out) == (
            "worktrees_removed=1", "worktrees_skipped=0"
        )

    @pytest.mark.parametrize("hazard", ["dirty", "unpushed", "main", "young", "live"])
    def test_dry_run_predicts_refusals(
        self, isolated_runs_root, isolated_log_root, git_root, capsys, hazard
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        cwd = wt
        if hazard == "dirty":
            (wt / "junk").write_text("x\n")
        elif hazard == "unpushed":
            (wt / "tracked.txt").write_text("local\n")
            _git(wt, "commit", "-qam", "local only")
        elif hazard == "main":
            cwd = repo
        elif hazard == "live":
            live = _make_state_run(
                isolated_runs_root, isolated_log_root, "z-live", status="running", cwd=wt
            )
            (live / "pid").write_text(f"{os.getpid()}\n")
            (live / "process_identity").write_text(
                f"{agent_run._process_identity(os.getpid())}\n"
            )
        age = 1 if hazard == "young" else 1000
        _make_state_run(
            isolated_runs_root, isolated_log_root, "a-run", cwd=cwd, age_hours=age
        )

        dry_out, real_out = self._run_both({}, capsys)

        assert self._counts(dry_out) == self._counts(real_out)
        assert "worktrees_removed=0" in dry_out
        assert cwd.is_dir()


# ---------------------------------------------------------------------------
# destructive-path regressions: each guard, removed, must fail one of these
# ---------------------------------------------------------------------------

class TestReapWorktreeIgnoredContent:
    """`git worktree remove` deletes ignored files, and `status --porcelain`
    calls a tree holding only ignored content clean. The refusal must come
    from `ls-files --others --ignored`, not from status."""

    def test_ignored_file_refused_and_preserved(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        (repo / ".git" / "info" / "exclude").write_text("secrets.env\n")
        (wt / "secrets.env").write_text("TOKEN=keep-me\n")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)
        assert _git(wt, "status", "--porcelain") == ""

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert (wt / "secrets.env").read_text() == "TOKEN=keep-me\n"
        assert "ignored file(s) or directory content" in out
        assert "--force-dirty" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    def test_ignored_directory_tree_refused_and_preserved(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        (repo / ".git" / "info" / "exclude").write_text("build/\n")
        (wt / "build" / "artifacts").mkdir(parents=True)
        (wt / "build" / "artifacts" / "out.bin").write_bytes(b"payload")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)
        assert _git(wt, "status", "--porcelain") == ""

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert (wt / "build" / "artifacts" / "out.bin").read_bytes() == b"payload"
        assert "ignored file(s) or directory content" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    def test_force_dirty_is_the_only_override_for_ignored_content(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        (repo / ".git" / "info" / "exclude").write_text("build/\nsecrets.env\n")
        (wt / "build").mkdir()
        (wt / "build" / "out.bin").write_bytes(b"payload")
        (wt / "secrets.env").write_text("TOKEN=x\n")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        agent_run.cmd_reap(_reap_args(all=True, force_unknown=True))
        assert wt.is_dir(), "--all/--force-unknown must not override the content refusal"
        capsys.readouterr()

        agent_run.cmd_reap(_reap_args(force_dirty=True))

        assert not wt.exists()
        assert "worktrees_removed=1" in capsys.readouterr().out


class TestReapWorktreeNameAttribution:
    """--name selects which run is reported, never which runs are considered:
    every sharer of the cwd is still weighed by the age gate."""

    def test_named_old_run_does_not_remove_a_worktree_a_younger_run_shares(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "a-old", cwd=wt, age_hours=1000)
        _make_state_run(isolated_runs_root, isolated_log_root, "b-new", cwd=wt, age_hours=1)

        agent_run.cmd_reap(_reap_args(name="a-old"))

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "below the age threshold" in out
        assert "worktrees_removed=0" in out

    def test_removed_once_when_every_sharer_exceeds_the_threshold(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "a-old", cwd=wt, age_hours=1000)
        _make_state_run(isolated_runs_root, isolated_log_root, "b-new", cwd=wt, age_hours=900)

        agent_run.cmd_reap(_reap_args(name="a-old"))

        out = capsys.readouterr().out
        assert not wt.exists()
        assert "worktrees_removed=1" in out
        assert out.count("[removing]") == 1
        assert "runs: a-old, b-new" in out


class TestReapWorktreeVerifiedLiveRunner:
    """A live, identity-matched runner protects its cwd whatever the persisted
    status says: a stale terminal write is not evidence the process exited."""

    def test_terminal_status_with_live_verified_runner_preserves_worktree(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        proc = subprocess.Popen(["sleep", "60"], cwd=str(wt))
        try:
            sd = _make_state_run(
                isolated_runs_root, isolated_log_root, "z-stale-terminal",
                status="done", cwd=wt, age_hours=1000,
            )
            (sd / "pid").write_text(f"{proc.pid}\n")
            (sd / "process_identity").write_text(
                f"{agent_run._process_identity(proc.pid)}\n"
            )
            _make_state_run(isolated_runs_root, isolated_log_root, "a-run", cwd=wt, age_hours=1000)

            agent_run.cmd_reap(_reap_args())
        finally:
            proc.kill()
            proc.wait()

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "a live run is using this directory" in out
        assert "worktrees_removed=0" in out

    def test_terminal_status_with_exited_pid_still_removes(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """The paired negative: an identity token whose process has exited
        must not keep the worktree alive forever."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        proc = subprocess.Popen(["true"])
        identity = agent_run._process_identity(proc.pid)
        proc.wait()
        sd = _make_state_run(
            isolated_runs_root, isolated_log_root, "a-run",
            status="done", cwd=wt, age_hours=1000,
        )
        (sd / "pid").write_text(f"{proc.pid}\n")
        (sd / "process_identity").write_text(f"{identity}\n")

        agent_run.cmd_reap(_reap_args())

        assert not wt.exists()
        assert "worktrees_removed=1" in capsys.readouterr().out


class TestReapWorktreeLivenessScanFailure:
    """A liveness scan that cannot enumerate STATE_ROOT is fail-closed: the
    pass reports a skip and deletes nothing."""

    @pytest.mark.parametrize("shape", ["absent", "regular_file"])
    def test_unusable_state_root_refuses_the_pass(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys, shape
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)
        replacement = isolated_runs_root.parent / f"state-{shape}"
        if shape == "regular_file":
            replacement.write_text("not a directory\n")
        monkeypatch.setattr(agent_run, "STATE_ROOT", replacement)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        expected = (
            "cannot read state root" if shape == "absent"
            else "state root is missing or not a real directory"
        )
        assert expected in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    def test_state_root_iterdir_oserror_refuses_the_pass(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """An EIO from `STATE_ROOT.iterdir()` inside the real scan, injected
        only while the worktree pass runs so the other passes are unaffected."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)
        real_scan = agent_run._worktree_state_scan
        real_iterdir = Path.iterdir
        breaking = {"on": False}

        def failing_iterdir(self):
            if breaking["on"] and self == agent_run.STATE_ROOT:
                raise OSError(5, "simulated I/O error")
            return real_iterdir(self)

        def scan_with_broken_iterdir():
            breaking["on"] = True
            try:
                return real_scan()
            finally:
                breaking["on"] = False

        monkeypatch.setattr(Path, "iterdir", failing_iterdir)
        monkeypatch.setattr(agent_run, "_worktree_state_scan", scan_with_broken_iterdir)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "cannot read state root" in out
        assert "simulated I/O error" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    def test_scan_failure_between_collection_and_removal_refuses(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """The pre-removal rescan is its own fail-closed gate: a scan that
        only breaks after candidate collection still stops the deletion."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)
        real_scan = agent_run._worktree_state_scan
        calls = {"n": 0}

        def scan_then_fail():
            calls["n"] += 1
            if calls["n"] == 1:
                return real_scan()
            return agent_run._WorktreeStateScan(None, "simulated rescan failure")

        monkeypatch.setattr(agent_run, "_worktree_state_scan", scan_then_fail)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "simulated rescan failure" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out


class TestReapWorktreeSameInvocationReconciliation:
    """`_mark_terminal` preserves a pre-existing (possibly stale) ended_at, so
    a run this pass reconciles can look arbitrarily old the moment it turns
    terminal. Its worktree is held back until a later invocation."""

    def _shared_worktree(self, state_root, log_root, git_root, wt):
        """One collectable terminal sharer plus one status=running sharer whose
        recorded pid has exited and whose ended_at is already 1000 h old.
        Without the reconciliation guard the terminal sharer alone would make
        the worktree eligible in the very invocation that reconciles the other.
        """
        dead = subprocess.Popen(["true"])
        dead.wait()
        _make_state_run(state_root, log_root, "a-old", cwd=wt, age_hours=1000)
        sd = _make_state_run(
            state_root, log_root, "b-reconciled", status="running", cwd=wt, age_hours=1000
        )
        (sd / "status").write_text("running\n")
        (sd / "pid").write_text(f"{dead.pid}\n")
        return sd

    def test_reconciled_sharer_keeps_the_worktree(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        sd = self._shared_worktree(isolated_runs_root, isolated_log_root, git_root, wt)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert (sd / "status").read_text().strip() == "died"
        # The stale ended_at survives _mark_terminal, so the age gate alone
        # would already judge this newly terminal run collectable.
        assert agent_run._terminal_state_age_seconds(sd) > 900 * 3600
        assert "a sharing run was reconciled in this invocation" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    def test_reconciled_sharer_keeps_the_worktree_under_dry_run(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """--dry-run predicts the same refusal, so its preview never claims a
        removal the equivalent real invocation would decline."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        self._shared_worktree(isolated_runs_root, isolated_log_root, git_root, wt)

        agent_run.cmd_reap(_reap_args(dry_run=True))

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "a sharing run was reconciled in this invocation" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    def test_a_later_invocation_collects_it(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """The refusal defers by one invocation; it does not strand the tree."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        self._shared_worktree(isolated_runs_root, isolated_log_root, git_root, wt)

        agent_run.cmd_reap(_reap_args())
        capsys.readouterr()
        agent_run.cmd_reap(_reap_args())

        assert not wt.exists()
        assert "worktrees_removed=1" in capsys.readouterr().out

    def test_sharer_dying_between_pass1_and_worktree_pass_keeps_worktree(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """A sharer alive during pass 1 (not recorded in reconciled_cwds) that
        exits before _worktree_gc_pass runs must still block removal via the
        late-terminal loop: _effective_status computes 'died' for a running run
        with a dead pid, so the age gate catches it before the worktree is
        deleted.  Using raw status (which still says 'running') would skip it."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "a-old", cwd=wt, age_hours=1000)

        proc = subprocess.Popen(["sleep", "60"], cwd=str(wt))
        sd = _make_state_run(
            isolated_runs_root, isolated_log_root, "z-live", status="running", cwd=wt
        )
        (sd / "pid").write_text(f"{proc.pid}\n")
        (sd / "process_identity").write_text(f"{agent_run._process_identity(proc.pid)}\n")

        real_pass = agent_run._worktree_gc_pass

        def kill_then_pass(**kw):
            proc.kill()
            proc.wait()
            return real_pass(**kw)

        monkeypatch.setattr(agent_run, "_worktree_gc_pass", kill_then_pass)

        try:
            agent_run.cmd_reap(_reap_args())
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

        out = capsys.readouterr().out
        assert wt.is_dir(), "worktree of a just-died sharer was DELETED"
        assert "youngest sharing run is below the age threshold" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out


class TestReapWorktreeNested:
    """`git worktree remove` on an outer worktree would take a registered
    worktree checked out inside it, including that inner tree's own content."""

    def _nested_pair(self, git_root):
        repo = _make_repo(git_root)
        outer = _add_worktree(repo, git_root / "outer", "f-outer")
        inner = _add_worktree(repo, outer / "inner", "f-inner")
        (inner / "tracked.txt").write_text("inner work\n")
        _git(inner, "commit", "-qam", "inner change")
        return repo, outer, inner

    def test_nested_registered_worktree_refuses_removal(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo, outer, inner = self._nested_pair(git_root)
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=outer, age_hours=1000)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert outer.is_dir()
        assert inner.is_dir()
        assert (inner / "tracked.txt").read_text() == "inner work\n"
        assert "is nested inside the candidate" in out
        assert "worktrees_removed=0" in out

    def test_force_dirty_does_not_override_a_nested_worktree(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo, outer, inner = self._nested_pair(git_root)
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=outer, age_hours=1000)

        agent_run.cmd_reap(_reap_args(force_dirty=True))

        out = capsys.readouterr().out
        assert outer.is_dir()
        assert inner.is_dir()
        assert (inner / "tracked.txt").read_text() == "inner work\n"
        assert "is nested inside the candidate" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    def test_nested_live_run_cwd_refuses_removal(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """A live run whose cwd is a subdirectory of the candidate is inside
        the tree `git worktree remove` would delete."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        nested = wt / "sub" / "deeper"
        nested.mkdir(parents=True)
        live = _make_state_run(
            isolated_runs_root, isolated_log_root, "z-live", status="running", cwd=nested
        )
        (live / "pid").write_text(f"{os.getpid()}\n")
        (live / "process_identity").write_text(
            f"{agent_run._process_identity(os.getpid())}\n"
        )
        _make_state_run(isolated_runs_root, isolated_log_root, "a-run", cwd=wt, age_hours=1000)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert nested.is_dir()
        assert f"a live run is using this directory ({nested.resolve()})" in out
        assert "worktrees_removed=0" in out

    def test_foreign_nested_worktree_refused_under_force_dirty(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """A linked worktree from a different repository nested inside the
        candidate is refused even with --force-dirty.  Content checks are
        skipped under --force-dirty, so the same-repo nested check cannot
        see the foreign worktree; the structural check must catch it."""
        outer_repo = _make_repo(git_root, "outer-repo")
        outer_wt = _add_worktree(outer_repo, git_root / "outer-wt", "f-outer")

        inner_repo = _make_repo(git_root, "inner-repo")
        inner_wt = _add_worktree(inner_repo, outer_wt / "inner-wt", "f-inner")
        (inner_wt / "precious.txt").write_text("must survive\n")

        _make_state_run(
            isolated_runs_root, isolated_log_root, "r1", cwd=outer_wt, age_hours=1000
        )

        agent_run.cmd_reap(_reap_args(force_dirty=True))

        out = capsys.readouterr().out
        assert outer_wt.is_dir(), "outer worktree must not be deleted"
        assert inner_wt.is_dir(), "inner (foreign) worktree must not be deleted"
        assert (inner_wt / "precious.txt").exists()
        assert "worktrees_removed=0 worktrees_skipped=1" in out
        assert "nested git repository" in out

    def test_foreign_nested_worktree_at_depth_5_refused_under_force_dirty(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """A foreign linked worktree 5 levels below the candidate is refused
        under --force-dirty: the scan depth limit is reached with unexplored
        subdirectories, which is an uncertainty that must fail closed."""
        outer_repo = _make_repo(git_root, "outer-repo")
        outer_wt = _add_worktree(outer_repo, git_root / "outer-wt", "f-outer")
        deep = outer_wt / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)

        inner_repo = _make_repo(git_root, "inner-repo")
        inner_wt = _add_worktree(inner_repo, deep / "inner-wt", "f-inner")
        (inner_wt / "precious.txt").write_text("must survive\n")

        _make_state_run(
            isolated_runs_root, isolated_log_root, "r1", cwd=outer_wt, age_hours=1000
        )

        agent_run.cmd_reap(_reap_args(force_dirty=True))

        out = capsys.readouterr().out
        assert inner_wt.is_dir(), "inner (foreign) worktree at depth 5 was DELETED"
        assert outer_wt.is_dir(), "outer worktree was DELETED"
        assert "worktrees_removed=0 worktrees_skipped=1" in out
        assert "nested git repository" in out

    def test_foreign_nested_worktree_at_depth_8_ignored_intermediate_refused_under_force_dirty(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """A foreign linked worktree 8 directories deep inside a gitignored
        intermediate directory is refused under --force-dirty.

        git ls-files --others --ignored reports the nested repo as a single
        directory entry at its full path even when all intermediate directories
        are gitignored, so depth is not a limiting factor."""
        outer_repo = _make_repo(git_root, "outer-repo")
        outer_wt = _add_worktree(outer_repo, git_root / "outer-wt", "f-outer")
        # Ignore the top-level vendor directory so the intermediate dirs are
        # gitignored; the nested repo still appears via --others --ignored.
        (outer_repo / ".git" / "info" / "exclude").write_text("vendor/\n")
        deep = outer_wt / "vendor" / "a" / "b" / "c" / "d" / "e" / "f" / "g"
        deep.mkdir(parents=True)

        inner_repo = _make_repo(git_root, "inner-repo")
        inner_wt = _add_worktree(inner_repo, deep / "inner-wt", "f-inner")
        (inner_wt / "precious.txt").write_text("must survive\n")

        _make_state_run(
            isolated_runs_root, isolated_log_root, "r1", cwd=outer_wt, age_hours=1000
        )

        agent_run.cmd_reap(_reap_args(force_dirty=True))

        out = capsys.readouterr().out
        assert inner_wt.is_dir(), "inner (foreign) worktree at depth 8 (ignored) was DELETED"
        assert outer_wt.is_dir(), "outer worktree was DELETED"
        assert "worktrees_removed=0 worktrees_skipped=1" in out
        assert "nested git repository" in out

    def test_deep_vendor_directory_without_nested_repo_is_removable(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """A linked worktree whose only deep content is a plain vendor directory
        (no nested git repository) must be removable regardless of depth.

        The git ls-files approach reports individual files for untracked dirs,
        not directory entries, so no depth limit applies and no false refusal
        is produced for vendor trees that are merely deep."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        # A deep untracked directory tree with no nested git repo.
        deep = wt / "vendor" / "a" / "b" / "c" / "d" / "e" / "f" / "g" / "h"
        deep.mkdir(parents=True)
        (deep / "lib.py").write_text("# vendored library\n")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        agent_run.cmd_reap(_reap_args(force_dirty=True))

        out = capsys.readouterr().out
        assert not wt.exists(), (
            "worktree with only a deep vendor dir (no nested repo) was not removed"
        )
        assert "worktrees_removed=1" in out
        assert "scan depth limit" not in out

    def test_nested_bare_repository_refused_under_force_dirty(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """A bare repository nested inside a candidate is refused under
        --force-dirty.

        git ls-files descends into bare repositories (they have no .git marker)
        and emits their individual files rather than a single slash-terminated
        directory entry.  The trailing-/ filter skips all of those files, so the
        pre-6a89e5c check was invisible to bare repos.  The fix collects unique
        immediate-parent directories from non-slash ls-files entries and calls
        git rev-parse --is-bare-repository on each one.

        Mutation: removing the bare_candidates loop in
        _worktree_foreign_nested_reason causes this test to fail because the
        bare repository is no longer detected and the outer worktree is deleted
        under --force-dirty."""
        outer_repo = _make_repo(git_root, "outer-repo")
        outer_wt = _add_worktree(outer_repo, git_root / "outer-wt", "f-outer")
        # Bare repo nested inside the candidate worktree.
        bare = outer_wt / "precious.git"
        subprocess.run(
            ["git", "init", "-q", "--bare", str(bare)], check=True
        )
        (bare / "important-note").write_text("do not delete\n")

        _make_state_run(
            isolated_runs_root, isolated_log_root, "r1", cwd=outer_wt, age_hours=1000
        )

        agent_run.cmd_reap(_reap_args(force_dirty=True))

        out = capsys.readouterr().out
        assert outer_wt.is_dir(), "outer worktree must not be deleted"
        assert bare.is_dir(), "nested bare repository must not be deleted"
        assert (bare / "important-note").exists()
        assert "nested bare git repository" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    def test_nested_bare_repo_in_ignored_dir_refused_under_force_dirty(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """A bare repo inside a gitignored subdirectory is also refused.

        The --others --ignored pass covers repos buried in vendor/node_modules-
        style ignored directories.

        Mutation: same as above — removing the bare_candidates loop causes
        deletion of the outer worktree (and thereby the bare repo)."""
        outer_repo = _make_repo(git_root, "outer-repo")
        outer_wt = _add_worktree(outer_repo, git_root / "outer-wt", "f-outer")
        (outer_repo / ".git" / "info" / "exclude").write_text("vendor/\n")
        vendor = outer_wt / "vendor"
        vendor.mkdir()
        bare = vendor / "cached.git"
        subprocess.run(
            ["git", "init", "-q", "--bare", str(bare)], check=True
        )

        _make_state_run(
            isolated_runs_root, isolated_log_root, "r1", cwd=outer_wt, age_hours=1000
        )

        agent_run.cmd_reap(_reap_args(force_dirty=True))

        out = capsys.readouterr().out
        assert outer_wt.is_dir(), "outer worktree must not be deleted"
        assert bare.is_dir(), "bare repo inside ignored dir must not be deleted"
        assert "nested bare git repository" in out or "cannot enumerate untracked paths" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    def test_non_utf8_ls_files_entry_fails_closed(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """An ls-files entry containing non-UTF-8 bytes fails closed rather
        than silently mangling the path to a different name.

        On byte-preserving filesystems (Linux), git path output may contain
        arbitrary non-NUL bytes.  _watch_run_git_checked decodes with
        errors='surrogateescape', preserving non-UTF-8 bytes as surrogate code
        points (U+DC80–U+DCFF).  _worktree_foreign_nested_reason detects these
        surrogates and refuses rather than passing the mangled path to lstat.

        APFS rejects filenames with invalid UTF-8 (EILSEQ), so this fixture
        injects the surrogate-escaped output directly.

        Mutation: changing errors='surrogateescape' to errors='replace' in
        _watch_run_git_checked, or removing the _has_surrogate check in
        _worktree_foreign_nested_reason, causes the worktree to be deleted
        (FileNotFoundError on the mangled path is misread as 'ordinary dir')."""
        repo = _make_repo(git_root, "outer-repo")
        wt = _add_worktree(repo, git_root / "wt", "f-outer")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        real_git = agent_run._watch_run_git_checked

        def git_with_nonutf8_ls_files(path, args, **kw):
            if list(args[:2]) == ["ls-files", "--others"] and "--exclude-standard" in args:
                # Simulate a bare repo at precious\xff.git/ emitting its files.
                # The byte \xff is not valid UTF-8; surrogateescape maps it to \udcff.
                mangled = "precious\udcff.git/HEAD\x00precious\udcff.git/config\x00"
                return agent_run._WatchGitOutcome(mangled, None)
            return real_git(path, args, **kw)

        monkeypatch.setattr(agent_run, "_watch_run_git_checked", git_with_nonutf8_ls_files)

        agent_run.cmd_reap(_reap_args(force_dirty=True))

        out = capsys.readouterr().out
        assert wt.is_dir(), "worktree must not be deleted when ls-files returns non-UTF-8 path"
        assert "non-UTF-8" in out or "non_UTF" in out.replace("-", "_")
        assert "worktrees_removed=0 worktrees_skipped=1" in out



    """Initialized submodules must block removal; deinitialized must not."""

    _ENV = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }

    def _add_submodule(self, repo: Path, sub_src: Path, name: str = "tracked-sub") -> Path:
        """Add and initialize a submodule in ``repo`` from ``sub_src``."""
        subprocess.run(
            ["git", "-c", "protocol.file.allow=always",
             "-C", str(repo), "submodule", "add", "-q", str(sub_src), name],
            check=True, env=self._ENV,
        )
        _git(repo, "commit", "-qm", f"add submodule {name}")
        return repo / name

    def test_initialized_submodule_refused_under_force_dirty(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """An initialized submodule inside the candidate is refused under
        --force-dirty.

        A gitlink (mode 160000) is absent from both ls-files --others queries,
        so _worktree_foreign_nested_reason cannot see it.  --force-dirty skips
        content checks, leaving _worktree_submodule_reason as the only guard.

        Mutation: removing the _worktree_submodule_reason call in
        _worktree_candidate_refusal causes the worktree (and its initialized
        submodule checkout) to be deleted under --force-dirty."""
        repo = _make_repo(git_root, "outer-repo")
        wt = _add_worktree(repo, git_root / "wt", "feature")
        sub_src = _make_repo(git_root, "sub-src")
        sub_checkout = self._add_submodule(wt, sub_src)
        (sub_checkout / "precious.txt").write_text("local only work\n")

        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        agent_run.cmd_reap(_reap_args(force_dirty=True))

        out = capsys.readouterr().out
        assert wt.is_dir(), "candidate worktree must not be deleted"
        assert sub_checkout.is_dir(), "submodule checkout must not be deleted"
        assert (sub_checkout / "precious.txt").exists()
        assert "initialized submodule" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    def test_dirty_submodule_refused_under_force_dirty(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """An initialized submodule with dirty content is refused even when
        --force-dirty overrides the outer worktree's content checks."""
        repo = _make_repo(git_root, "outer-repo")
        wt = _add_worktree(repo, git_root / "wt", "feature")
        sub_src = _make_repo(git_root, "sub-src")
        sub_checkout = self._add_submodule(wt, sub_src)
        (sub_checkout / "untracked.txt").write_text("untracked inside submodule\n")

        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        agent_run.cmd_reap(_reap_args(force_dirty=True))

        out = capsys.readouterr().out
        assert wt.is_dir(), "worktree must not be deleted"
        assert sub_checkout.is_dir(), "dirty submodule must not be deleted"
        assert "initialized submodule" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    def test_locally_committed_submodule_refused_under_force_dirty(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """A submodule with local-only commits is refused under --force-dirty.
        Local commits inside a submodule may be the only copy of that work."""
        repo = _make_repo(git_root, "outer-repo")
        wt = _add_worktree(repo, git_root / "wt", "feature")
        sub_src = _make_repo(git_root, "sub-src")
        sub_checkout = self._add_submodule(wt, sub_src)
        (sub_checkout / "tracked.txt").write_text("local commit in sub\n")
        _git(sub_checkout, "commit", "-qam", "local-only commit in submodule")

        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        agent_run.cmd_reap(_reap_args(force_dirty=True))

        out = capsys.readouterr().out
        assert wt.is_dir(), "worktree must not be deleted"
        assert sub_checkout.is_dir(), "submodule with local commits must not be deleted"
        assert "initialized submodule" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    def test_deinitialized_submodule_does_not_block_removal(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """A deinitialized submodule (empty checkout directory, nothing to lose)
        does not block removal.

        Mutation: refusing on any mode-160000 index entry regardless of
        initialization state causes this test to fail because the worktree is
        skipped instead of removed."""
        repo = _make_repo(git_root, "outer-repo")
        wt = _add_worktree(repo, git_root / "wt", "feature")
        sub_src = _make_repo(git_root, "sub-src")
        sub_checkout = self._add_submodule(wt, sub_src)
        # Push so the submodule commit is on the remote (no unpushed blocker).
        _git(wt, "push", "-q", "origin", "feature")
        # Deinitialize: empties the checkout directory.
        subprocess.run(
            ["git", "-C", str(wt), "submodule", "deinit", "-f", "tracked-sub"],
            check=True, env=self._ENV,
        )
        assert not any(sub_checkout.iterdir()), "checkout must be empty after deinit"

        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        agent_run.cmd_reap(_reap_args(force_dirty=True))

        out = capsys.readouterr().out
        assert not wt.exists(), "worktree with only a deinitialized submodule must be removable"
        assert "worktrees_removed=1" in out


class TestReapWorktreeFinalRevalidation:
    """State captured at collection is re-read under the publication lock
    immediately before `git worktree remove`."""

    def _mutate_before_removal(self, monkeypatch, mutate):
        """Run ``mutate`` once, at the start of the pre-removal rescan."""
        real_scan = agent_run._worktree_state_scan
        calls = {"n": 0}

        def scan_with_mutation():
            calls["n"] += 1
            if calls["n"] > 1:
                mutate()
            return real_scan()

        monkeypatch.setattr(agent_run, "_worktree_state_scan", scan_with_mutation)

    def test_path_identity_change_after_collection_refuses(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """An fd opened at collection still refers to the original directory;
        samestat(fstat(fd), lstat(path)) is False when a different tree now
        occupies the candidate path."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        decoy = _add_worktree(repo, git_root / "decoy", "f-decoy")
        (decoy / "keep.txt").write_text("must survive\n")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        def swap():
            wt.rename(git_root / "moved-away")
            decoy.rename(wt)

        self._mutate_before_removal(monkeypatch, swap)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert (wt / "keep.txt").read_text() == "must survive\n"
        assert "candidate path identity changed after collection" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    def test_path_identity_inode_reuse_same_inode_refuses(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """_dir_identity returning the same (st_dev, st_ino) for both original
        and decoy does not defeat the fd-based revalidation check.

        On ext4, inode numbers are recycled immediately after rmdir+mkdir;
        _dir_identity on the decoy can return the same (st_dev, st_ino) as the
        original.  The revalidation check uses os.path.samestat on the fd opened
        at collection rather than _dir_identity, so a same-inode decoy still
        produces a stat mismatch (fd is pinned to the original kernel object;
        the decoy is a distinct object despite sharing the inode number on a
        filesystem without the fd held open).

        The test forces the collision by monkeypatching _dir_identity to return
        the same tuple for both the original wt and the decoy.  The swap must
        still be refused because the samestat check compares real stat results.

        Mutation: replacing the samestat(fstat(identity_fd), ...) check with
        ``_dir_identity(cand.resolved) != cand.identity`` causes this test to
        fail — the patched _dir_identity makes both sides equal, the check
        passes, and the decoy is removed."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        decoy = _add_worktree(repo, git_root / "decoy", "f-decoy")
        (decoy / "keep.txt").write_text("must survive\n")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        # Force same (st_dev, st_ino) for the wt path before and after the swap,
        # simulating ext4 inode reuse.  The fd-based check does not use this
        # function and must still refuse the swap.
        same_identity = (99, 42)
        real_dir_identity = agent_run._dir_identity

        def patched_dir_identity(path: Path):
            if path == wt:
                return same_identity
            return real_dir_identity(path)

        def swap():
            wt.rename(git_root / "moved-away")
            decoy.rename(wt)

        self._mutate_before_removal(monkeypatch, swap)
        monkeypatch.setattr(agent_run, "_dir_identity", patched_dir_identity)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert (wt / "keep.txt").read_text() == "must survive\n"
        assert "candidate path identity changed after collection" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    def test_content_created_after_collection_refuses(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """A fresh write is caught by the recursive-mtime activity gate, which
        the rescan re-evaluates against the current tree."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        self._mutate_before_removal(
            monkeypatch, lambda: (wt / "late.txt").write_text("written after collection\n")
        )

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert (wt / "late.txt").read_text() == "written after collection\n"
        assert "worktree filesystem activity is below the age threshold" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    def test_backdated_content_created_after_collection_refuses(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """Content whose mtimes are backdated past the activity gate is still
        caught, by the rescan's own untracked-file check."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)
        old = time.time() - 1000 * 3600

        def write_backdated():
            late = wt / "late.txt"
            late.write_text("written after collection\n")
            os.utime(late, (old, old))
            os.utime(wt, (old, old))

        self._mutate_before_removal(monkeypatch, write_backdated)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert (wt / "late.txt").read_text() == "written after collection\n"
        assert "untracked file(s)" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    def test_live_run_published_after_collection_refuses(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """A run that becomes live after candidate collection but while the
        exclusive reaper lock is held causes the final rescan to find it and
        refuse deletion."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        def publish_live_run():
            live = _make_state_run(
                isolated_runs_root, isolated_log_root, "z-late", status="running", cwd=wt
            )
            (live / "pid").write_text(f"{os.getpid()}\n")
            (live / "process_identity").write_text(
                f"{agent_run._process_identity(os.getpid())}\n"
            )

        self._mutate_before_removal(monkeypatch, publish_live_run)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "a live run is using this directory" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    def test_live_to_terminal_transition_after_collection_refuses(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """A sharing run that was live at collection and finishes normally
        before the final locked scan must keep the worktree safe for at least
        worktree-min-age-hours after it ended.

        At collection time the run is still running, so it does not appear in
        cand.names and does not contribute to cand.age_seconds.  After it
        transitions to terminal its ended_at is fresh (age≈0), which must be
        caught by the final scan even though the collection age was satisfied."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        # r1: old terminal run that creates the candidate (age=1000h satisfies threshold).
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        real_scan = agent_run._worktree_state_scan
        calls: dict = {"n": 0}

        def scan_and_transition():
            calls["n"] += 1
            if calls["n"] > 1:
                # r2 finishes normally just before the final locked scan.  It
                # was not present at collection, so it is absent from cand.names
                # and its fresh ended_at must block deletion.
                _make_state_run(
                    isolated_runs_root, isolated_log_root, "r2",
                    status="done", cwd=wt, age_hours=0
                )
            return real_scan()

        monkeypatch.setattr(agent_run, "_worktree_state_scan", scan_and_transition)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "youngest sharing run is below the age threshold" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    def test_validation_is_final_step_before_removal(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """No code runs between _worktree_candidate_refusal returning None and
        _worktree_remove being called; the window between validation and
        deletion is kept to the minimum achievable while using git worktree
        remove."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        events: list = []
        real_refusal = agent_run._worktree_candidate_refusal
        real_remove = agent_run._worktree_remove

        def recording_refusal(*a, **kw):
            result = real_refusal(*a, **kw)
            if result[0] is None:
                # Only record successful (non-refusing) refusal checks so that
                # spurious refusals from other candidates do not add noise.
                events.append("refusal")
            return result

        def recording_remove(*a, **kw):
            events.append("remove")
            return real_remove(*a, **kw)

        monkeypatch.setattr(agent_run, "_worktree_candidate_refusal", recording_refusal)
        monkeypatch.setattr(agent_run, "_worktree_remove", recording_remove)

        agent_run.cmd_reap(_reap_args())

        capsys.readouterr()
        assert "refusal" in events, "candidate refusal check must run"
        assert "remove" in events, "_worktree_remove must be called"
        # The first refusal check and the first remove call must be adjacent:
        # no other refusal check or remove attempt may lie between them.
        first_refusal = events.index("refusal")
        first_remove = events.index("remove")
        assert first_remove == first_refusal + 1, (
            f"_worktree_remove must follow _worktree_candidate_refusal immediately "
            f"(first refusal at index {first_refusal}, first remove at {first_remove}); "
            f"events: {events}"
        )

    def test_same_name_replacement_applies_fresh_age(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """A new run that replaces an old terminal state directory with the same
        name before the final scan must not inherit the old generation's age.

        Collection freezes the old run's age (1000h).  The final scan finds a
        directory with the same name but a different inode (new generation) whose
        ended_at is fresh (age≈0).  The inode mismatch triggers a fresh age
        check; age≈0 is below the threshold so the worktree is preserved.

        Mutation: reverting to 'd.name in cand.names: continue' (skipping by
        name alone) causes this test to fail because the frozen 1000h age
        authorises deletion of a recently-finished worktree."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "old-run", cwd=wt, age_hours=1000)

        real_scan = agent_run._worktree_state_scan
        calls: dict = {"n": 0}

        def scan_with_replacement():
            calls["n"] += 1
            if calls["n"] > 1:
                import shutil
                shutil.rmtree(isolated_runs_root / "old-run")
                # New generation with same name, fresh ended_at (age≈0).
                _make_state_run(
                    isolated_runs_root, isolated_log_root, "old-run",
                    status="done", cwd=wt, age_hours=0,
                )
            return real_scan()

        monkeypatch.setattr(agent_run, "_worktree_state_scan", scan_with_replacement)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir(), "worktree must not be deleted when same-name replacement has fresh age"
        assert "youngest sharing run is below the age threshold" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    def test_same_name_replacement_same_inode_applies_fresh_age(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """Fresh age is checked for a same-name replacement even when the
        replacement directory has the same (st_dev, st_ino) as the original.

        On ext4, inode numbers are reused immediately after directory removal,
        so a same-name replacement can have an identical _dir_identity to the
        collected entry.  The fix removes the inode-identity generation check
        entirely and always reads the fresh age; this test verifies that the
        fix holds regardless of filesystem behaviour by forcing _dir_identity
        to return a fixed value for state directories.

        Mutation: restoring the 'if d.name in cand.names: continue' skip causes
        this test to fail because the inode values are identical and the frozen
        1000h age authorises deletion of a worktree whose replacement run
        ended seconds ago."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "old-run", cwd=wt, age_hours=1000)

        # Fixed identity tuple returned for every state directory; forces the
        # same _dir_identity for both the original and replacement "old-run" dirs.
        fixed_state_identity = (99, 42)
        real_dir_identity = agent_run._dir_identity

        def patched_dir_identity(path: Path):
            # Return the fixed identity only for paths inside the runs root so
            # that the worktree's fd-based revalidation check is unaffected.
            try:
                path.relative_to(isolated_runs_root)
                return fixed_state_identity
            except ValueError:
                return real_dir_identity(path)

        real_scan = agent_run._worktree_state_scan
        calls: dict = {"n": 0}

        def scan_with_replacement():
            calls["n"] += 1
            if calls["n"] > 1:
                import shutil
                shutil.rmtree(isolated_runs_root / "old-run")
                # New generation with same name and — after the monkeypatch —
                # the same _dir_identity.  Fresh ended_at is age≈0.
                _make_state_run(
                    isolated_runs_root, isolated_log_root, "old-run",
                    status="done", cwd=wt, age_hours=0,
                )
            return real_scan()

        monkeypatch.setattr(agent_run, "_dir_identity", patched_dir_identity)
        monkeypatch.setattr(agent_run, "_worktree_state_scan", scan_with_replacement)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir(), (
            "worktree must not be deleted when replacement has same inode but fresh age"
        )
        assert "youngest sharing run is below the age threshold" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out


class TestReapWorktreePublicationLock:
    """The shared publication lock must be held across cwd entry and state
    publication so a concurrent reaper cannot pass its exclusive lock and final
    scan in the interval between chdir and visible run state."""

    def test_reaper_excluded_by_shared_publication_lock(
        self, isolated_runs_root, isolated_log_root, git_root
    ):
        """A process holding the exclusive lock (reaper) blocks a concurrent
        shared-lock attempt (launcher), and vice versa.  Tests the real flock
        protocol via os.fork so no mock can hide a missing lock acquisition."""
        import fcntl as _fcntl

        lock_dir = agent_run.STATE_ROOT / ".locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / "worktree-publication.lock"

        # Phase 1: parent holds EXCLUSIVE; child tries SHARED (non-blocking).
        r1, w1 = os.pipe()
        r2, w2 = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(w1); os.close(r2)
            os.read(r1, 1); os.close(r1)
            fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
            try:
                _fcntl.flock(fd, _fcntl.LOCK_SH | _fcntl.LOCK_NB)
                os.write(w2, b"1")
            except BlockingIOError:
                os.write(w2, b"0")
            finally:
                try:
                    _fcntl.flock(fd, _fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(fd)
                os.close(w2)
            os._exit(0)
        os.close(r1); os.close(w2)
        fd_ex = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
        _fcntl.flock(fd_ex, _fcntl.LOCK_EX)
        os.write(w1, b"1"); os.close(w1)
        child_result = os.read(r2, 1); os.close(r2)
        _fcntl.flock(fd_ex, _fcntl.LOCK_UN); os.close(fd_ex)
        os.waitpid(pid, 0)
        assert child_result == b"0", "shared lock must block while exclusive is held"

        # Phase 2: child holds SHARED; parent tries EXCLUSIVE (non-blocking).
        r3, w3 = os.pipe()
        r4, w4 = os.pipe()
        pid2 = os.fork()
        if pid2 == 0:
            os.close(w3); os.close(r4)
            fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
            _fcntl.flock(fd, _fcntl.LOCK_SH)
            os.write(w4, b"1"); os.close(w4)
            os.read(r3, 1); os.close(r3)
            _fcntl.flock(fd, _fcntl.LOCK_UN); os.close(fd)
            os._exit(0)
        os.close(r3); os.close(w4)
        os.read(r4, 1); os.close(r4)
        fd_ex2 = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
        got_exclusive = False
        try:
            _fcntl.flock(fd_ex2, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
            got_exclusive = True
        except BlockingIOError:
            pass
        finally:
            try:
                _fcntl.flock(fd_ex2, _fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(fd_ex2)
        os.write(w3, b"1"); os.close(w3)
        os.waitpid(pid2, 0)
        assert not got_exclusive, "exclusive lock must block while shared is held"

    def test_lock_acquired_before_cwd_entry_in_cmd_launch(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch
    ):
        """cmd_launch acquires the shared publication lock before calling
        _apply_launch_cwd; the ordering is recorded here and asserted."""
        from contextlib import contextmanager
        import pytest

        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")

        events: list = []
        real_lock = agent_run._worktree_publication_lock
        real_apply = agent_run._apply_launch_cwd

        @contextmanager
        def recording_lock(*, exclusive):
            events.append(("lock", exclusive))
            with real_lock(exclusive=exclusive) as fd:
                yield fd

        def recording_apply(a):
            events.append(("cwd",))
            real_apply(a)

        monkeypatch.setattr(agent_run, "_worktree_publication_lock", recording_lock)
        monkeypatch.setattr(agent_run, "_apply_launch_cwd", recording_apply)

        args = argparse.Namespace(
            name="order-check",
            cwd=str(wt),
            prompt_file=None,
            command=[],
            harness=None,
            interactive=False,
            idle_timeout=None,
            submit_mode=None,
        )
        with pytest.raises(SystemExit):
            agent_run.cmd_launch(args)

        shared_positions = [i for i, e in enumerate(events) if e == ("lock", False)]
        cwd_positions = [i for i, e in enumerate(events) if e[0] == "cwd"]
        assert shared_positions, "shared publication lock must be acquired"
        assert cwd_positions, "cwd must be entered"
        assert min(shared_positions) < min(cwd_positions), (
            "shared lock must be acquired before cwd entry"
        )


class TestReapWorktreeActivityWalk:
    """The recursive mtime walk must fail closed when a subtree is unreadable:
    a recent file hidden behind a permission error must not allow the worktree
    to appear idle."""

    def test_walk_error_fails_closed_in_strict_mode(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """A stat failure during the activity walk causes _WALK_INCOMPLETE to
        be returned, so the activity gate reports the worktree as unreadable
        rather than falling back to a stale timestamp."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        # A flat directory with one file whose stat will be made to fail.
        # No subdirectories, so the foreign-nested scan passes cleanly; only
        # the mtime walk visits the file and hits the injected stat error.
        secret = wt / "unreadable.txt"
        secret.write_text("hidden content\n")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        real_entry_stat = os.DirEntry.stat

        def stat_fails_for_secret(entry_self, *, follow_symlinks=True):
            if entry_self.path == str(secret):
                raise OSError("simulated permission denied")
            return real_entry_stat(entry_self, follow_symlinks=follow_symlinks)

        monkeypatch.setattr(os.DirEntry, "stat", stat_fails_for_secret)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "cannot determine worktree filesystem activity" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out


class TestReapWorktreeNoRepositoryControlledExecution:
    """Inspecting a candidate must never run a command the inspected
    repository chose: its contents are attacker-controlled by assumption."""

    def _armed_worktree(self, git_root, sentinel: Path):
        """A repo + linked worktree whose ``filter.evil.clean``, smudge,
        ``core.fsmonitor`` and ``post-index-change`` hook each create
        ``sentinel``. The hostile config is installed last so no setup command
        can trip it, and the worktree's one tracked file is modified to the
        same byte length with the index-cached mtime restored — the stat cache
        is then inconclusive, which is precisely when ``ls-files --modified``
        runs a clean filter to compare content."""
        repo = _make_repo(git_root)
        (repo / ".gitattributes").write_text("tracked.txt filter=evil\n")
        _git(repo, "add", ".gitattributes")
        _git(repo, "commit", "-qm", "attributes")
        _git(repo, "push", "-q", "origin", "main")
        wt = _add_worktree(repo, git_root / "wt", "feature")
        index_mtime = (repo / ".git" / "worktrees" / "wt" / "index").stat().st_mtime
        (wt / "tracked.txt").write_text("cxntent\n")
        os.utime(wt / "tracked.txt", (index_mtime, index_mtime))

        script = git_root / "sentinel.sh"
        script.write_text(f"#!/bin/sh\n: > {sentinel}\nexit 1\n")
        script.chmod(0o755)
        _git(repo, "config", "filter.evil.clean", f"sh -c ': > {sentinel}; cat'")
        _git(repo, "config", "filter.evil.smudge", f"sh -c ': > {sentinel}; cat'")
        _git(repo, "config", "core.fsmonitor", str(script))
        hooks = repo / ".git" / "hooks"
        hooks.mkdir(exist_ok=True)
        (hooks / "post-index-change").write_text(f"#!/bin/sh\n: > {sentinel}\n")
        (hooks / "post-index-change").chmod(0o755)
        assert not sentinel.exists(), "setup itself tripped the sentinel"
        return repo, wt

    def test_the_sentinel_configuration_is_actually_reachable(self, git_root):
        """Guards the guard: an unhardened git run on this same repository
        does execute the configured programs, so a passing refusal test is
        evidence of the hardening rather than of an inert fixture."""
        sentinel = git_root / "SENTINEL"
        _repo, wt = self._armed_worktree(git_root, sentinel)

        _git(wt, "status", "--porcelain")

        assert sentinel.exists()

    def test_inspection_runs_no_repository_configured_program(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        sentinel = git_root / "SENTINEL"
        _repo, wt = self._armed_worktree(git_root, sentinel)
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        agent_run.cmd_reap(_reap_args())
        capsys.readouterr()

        assert not sentinel.exists(), (
            "reap inspection executed a program configured by the inspected repository"
        )

    def test_content_refusal_still_detects_the_modified_file(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """Suppressing attributes must not blind the guard: the modified
        tracked file is still reported and the worktree still survives."""
        sentinel = git_root / "SENTINEL"
        _repo, wt = self._armed_worktree(git_root, sentinel)
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert not sentinel.exists()
        assert "modified or deleted tracked file(s)" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    @pytest.mark.parametrize(
        "outcome, expected",
        [
            (agent_run._WatchGitOutcome(None, "git_failed", "no such command"),
             "cannot compute empty tree object"),
            (agent_run._WatchGitOutcome("not-an-object-id\n", None),
             "unexpected empty tree object"),
        ],
    )
    def test_unusable_empty_tree_probe_is_a_refusal(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys,
        outcome, expected,
    ):
        """Without a usable ``--attr-source`` argument the content checks would
        have to run with the repository's attributes in force, so failing to
        obtain one refuses rather than inspecting unprotected."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)
        real_git = agent_run._watch_run_git_checked

        def probe_fails(path, args, *a, **k):
            if list(args)[:1] == ["hash-object"]:
                return outcome
            return real_git(path, args, *a, **k)

        monkeypatch.setattr(agent_run, "_watch_run_git_checked", probe_fails)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert expected in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out


class TestReapWorktreePassOrdering:
    """The worktree pass runs last and shares the reap budget, so its
    WORKTREE_REMOVE_TIMEOUT_SECONDS-scale removals cannot starve the state,
    log, scratch or orphan passes."""

    def test_slow_removal_defers_later_candidates_without_losing_them(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        repo = _make_repo(git_root)
        worktrees = [
            _add_worktree(repo, git_root / f"wt{i}", f"f{i}") for i in range(1, 4)
        ]
        for index, wt in enumerate(worktrees, start=1):
            _make_state_run(
                isolated_runs_root, isolated_log_root, f"w{index}-run", cwd=wt, age_hours=1000
            )
        old = time.time() - 1000 * 3600
        # Work for the state and orphan-scratch passes, both of which must
        # complete before the budget is consumed by the first slow removal.
        _make_state_run(
            isolated_runs_root, isolated_log_root, "s-state", cwd=None, age_hours=1000
        )
        orphan_log = isolated_log_root / "o-scratch"
        (orphan_log / "tmp").mkdir(parents=True)
        (orphan_log / "tmp" / "leftover").write_text("x\n")
        for p in (orphan_log / "tmp" / "leftover", orphan_log / "tmp", orphan_log):
            os.utime(p, (old, old))

        real_remove = agent_run._worktree_remove

        def slow_remove(info, path, *, force):
            time.sleep(1.5)
            return real_remove(info, path, force=force)

        monkeypatch.setattr(agent_run, "_worktree_remove", slow_remove)

        agent_run.cmd_reap(_reap_args(max_seconds=1.0))

        out = capsys.readouterr().out
        # Every routine pass ran to completion before the worktree pass began.
        assert "collected=1" in out
        assert "orphaned_scratch=1" in out
        assert not (isolated_runs_root / "s-state").exists()
        assert not (orphan_log / "tmp").exists()
        # Exactly one worktree removed; the remaining two are deferred, not
        # refused. The single skip is s-state's cwd-less state dir, which the
        # attribution scan reports before any candidate is considered.
        assert "worktrees_removed=1" in out
        assert "run s-state has an unresolvable cwd" in out
        assert "worktrees_skipped=1" in out
        assert "deferred 2 candidate(s)" in out
        assert sum(1 for wt in worktrees if wt.is_dir()) == 2

    def test_slow_removal_does_not_starve_the_log_pass(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """Pass 2.5 (--include-logs) also precedes the worktree pass."""
        repo = _make_repo(git_root)
        worktrees = [
            _add_worktree(repo, git_root / f"wt{i}", f"f{i}") for i in range(1, 4)
        ]
        for index, wt in enumerate(worktrees, start=1):
            _make_state_run(
                isolated_runs_root, isolated_log_root, f"w{index}-run", cwd=wt, age_hours=1000
            )
        log_only = isolated_log_root / "l-logonly"
        log_only.mkdir()
        (log_only / "log").write_text("archived\n")
        old = time.time() - 1000 * 3600
        for p in (log_only / "log", log_only):
            os.utime(p, (old, old))

        real_remove = agent_run._worktree_remove

        def slow_remove(info, path, *, force):
            time.sleep(1.5)
            return real_remove(info, path, force=force)

        monkeypatch.setattr(agent_run, "_worktree_remove", slow_remove)

        agent_run.cmd_reap(_reap_args(include_logs=True, max_seconds=1.0))

        out = capsys.readouterr().out
        assert "logs_collected=1" in out
        assert not log_only.exists()
        assert "worktrees_removed=1" in out
        assert "deferred 2 candidate(s)" in out

    def test_deferred_candidates_are_collected_by_the_next_invocation(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        worktrees = [
            _add_worktree(repo, git_root / f"wt{i}", f"f{i}") for i in range(1, 4)
        ]
        for index, wt in enumerate(worktrees, start=1):
            _make_state_run(
                isolated_runs_root, isolated_log_root, f"w{index}-run", cwd=wt, age_hours=1000
            )

        agent_run.cmd_reap(_reap_args())

        assert all(not wt.exists() for wt in worktrees)
        assert "worktrees_removed=3" in capsys.readouterr().out


class TestDuCompleteness:
    """Incomplete walks are marked and presented as lower bounds, not exact totals."""

    def test_top_level_permission_error_marks_row_incomplete(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """When os.scandir raises PermissionError on the top-level state dir,
        the row is marked incomplete and a WARNING line is printed.

        Mutation: not tracking completeness in _dir_size_bytes_complete causes
        the row to report complete=True even when bytes were missed, and the
        WARNING is not printed."""
        repo = _make_repo(git_root)
        _make_state_run(
            isolated_runs_root, isolated_log_root, "r1", log_bytes=1000
        )

        real_scandir = os.scandir

        def scandir_fails_for_state(path):
            if str(path) == str(isolated_log_root / "r1"):
                raise PermissionError("no permission")
            return real_scandir(path)

        monkeypatch.setattr(os, "scandir", scandir_fails_for_state)

        agent_run.cmd_du(_du_args(by_run=True, bytes=False, json=True))
        out = capsys.readouterr().out

        payload = json.loads(out)
        runs = {r["name"]: r for r in payload["runs"]}
        assert "complete" in runs["r1"] and runs["r1"]["complete"] is False, (
            "row with permission-denied scandir must have complete=False in JSON"
        )
        assert "complete" in payload["total"] and payload["total"]["complete"] is False

    def test_nested_permission_error_marks_row_incomplete(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """A PermissionError on a nested subtree also marks the row incomplete.

        Mutation: catching OSError and silently continuing (the old behaviour)
        hides the partial result; complete=True is returned even with missing bytes."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        (wt / "readable.txt").write_bytes(b"r" * 100)
        secret_dir = wt / "secret"
        secret_dir.mkdir()
        (secret_dir / "hidden.bin").write_bytes(b"h" * 500)
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt)

        real_scandir = os.scandir

        def scandir_fails_for_secret(path):
            if str(path) == str(secret_dir):
                raise PermissionError("no permission")
            return real_scandir(path)

        monkeypatch.setattr(os, "scandir", scandir_fails_for_secret)

        agent_run.cmd_du(_du_args(by_run=True, bytes=False, json=True))
        out = capsys.readouterr().out

        payload = json.loads(out)
        runs = {r["name"]: r for r in payload["runs"]}
        assert "complete" in runs["r1"] and runs["r1"]["complete"] is False, (
            "row with permission-denied subtree must have complete=False"
        )
        assert runs["r1"]["worktree_bytes"] < 600, (
            "partial worktree bytes (secret dir omitted) must be less than full size"
        )

    def test_complete_walk_has_no_complete_key_in_json(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """When all walks succeed, complete is not present in JSON output
        (omitting the key keeps the common case terse)."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt)

        agent_run.cmd_du(_du_args(by_run=True, bytes=False, json=True))
        out = capsys.readouterr().out

        payload = json.loads(out)
        runs = {r["name"]: r for r in payload["runs"]}
        assert "complete" not in runs["r1"], (
            "complete key must be absent from JSON when walk is complete"
        )
        assert "complete" not in payload["total"]

    def test_incomplete_walk_prints_warning_in_table_mode(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """A permission error during a state dir walk prints a WARNING line."""
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", log_bytes=200)

        real_scandir = os.scandir

        def scandir_fails_for_log(path):
            if str(path) == str(isolated_log_root / "r1"):
                raise PermissionError("no permission")
            return real_scandir(path)

        monkeypatch.setattr(os, "scandir", scandir_fails_for_log)

        agent_run.cmd_du(_du_args(by_run=True))
        out = capsys.readouterr().out

        assert "WARNING" in out and ("incomplete" in out or "lower bound" in out), (
            "table output must warn about incomplete walks"
        )


class TestDuWorktreeArithmetic:
    """TOTAL counts every readable byte exactly once, whatever nests inside what."""

    def test_nested_worktrees_counted_once_each(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        outer = _add_worktree(repo, git_root / "outer", "f-outer")
        inner = _add_worktree(repo, outer / "inner", "f-inner")
        (outer / "outer.bin").write_bytes(b"o" * 5000)
        (inner / "inner.bin").write_bytes(b"i" * 3000)
        _make_state_run(isolated_runs_root, isolated_log_root, "a-outer", cwd=outer)
        _make_state_run(isolated_runs_root, isolated_log_root, "b-inner", cwd=inner)
        whole_tree = agent_run._dir_size_bytes(outer)
        inner_size = agent_run._dir_size_bytes(inner)

        agent_run.cmd_du(_du_args(by_run=True))

        out = capsys.readouterr().out
        outer_charge = int(_row(out, "a-outer")[5])
        inner_charge = int(_row(out, "b-inner")[5])
        assert inner_charge == inner_size
        # The inner tree is charged to its own row and excluded from the outer.
        assert outer_charge == whole_tree - inner_size
        assert int(_row(out, "TOTAL")[5]) == whole_tree

    def test_state_and_log_roots_inside_a_worktree_counted_once(
        self, isolated_runs_root, isolated_log_root, git_root, tmp_path, monkeypatch, capsys
    ):
        """STATE_ROOT/LOG_ROOT under a counted worktree are charged to their
        own columns, and excluded from the worktree walk."""
        repo = _make_repo(tmp_path / "inside-gitroots")
        wt = _add_worktree(repo, tmp_path / "inside-gitroots" / "wt", "feature")
        (wt / "payload.bin").write_bytes(b"p" * 7000)
        state = wt / "nested-state"
        logs = wt / "nested-logs"
        state.mkdir()
        logs.mkdir()
        monkeypatch.setattr(agent_run, "STATE_ROOT", state)
        monkeypatch.setattr(agent_run, "LOG_ROOT", logs)
        _make_state_run(state, logs, "r1", cwd=wt, log_bytes=1234)
        (logs / "r1" / "tmp").mkdir()
        (logs / "r1" / "tmp" / "scratch").write_bytes(b"s" * 321)
        whole_tree = agent_run._dir_size_bytes(wt)

        agent_run.cmd_du(_du_args(by_run=True))

        out = capsys.readouterr().out
        fields = _row(out, "r1")
        state_b, log_b, scratch_b, worktree_b, total_b = (int(f) for f in fields[2:7])
        assert scratch_b == 321
        assert log_b >= 1234
        assert total_b == state_b + log_b + scratch_b + worktree_b
        # Every byte under the worktree is accounted exactly once across the
        # four columns: the worktree charge excludes the two nested roots.
        assert worktree_b == whole_tree - state_b - log_b - scratch_b
        assert int(_row(out, "TOTAL")[6]) == whole_tree

    def test_unrecognized_root_content_counted_in_worktree_total(
        self, isolated_runs_root, isolated_log_root, git_root, tmp_path, monkeypatch, capsys
    ):
        """Root-level files, .locks, sentinels, and invalid-name entries inside
        a STATE_ROOT or LOG_ROOT that is nested in a worktree must be counted
        exactly once in TOTAL.  Excluding the whole configured root drops these
        bytes from the total; excluding only the charged per-run subdirectories
        leaves the remainder in the worktree column."""
        repo = _make_repo(tmp_path / "inside-gitroots")
        wt = _add_worktree(repo, tmp_path / "inside-gitroots" / "wt", "feature")
        (wt / "payload.bin").write_bytes(b"p" * 7000)
        state = wt / "nested-state"
        logs = wt / "nested-logs"
        state.mkdir()
        logs.mkdir()
        monkeypatch.setattr(agent_run, "STATE_ROOT", state)
        monkeypatch.setattr(agent_run, "LOG_ROOT", logs)
        _make_state_run(state, logs, "r1", cwd=wt, log_bytes=1234)

        # Unrecognized content in the roots: .locks dir, a sentinel file, an
        # orphan log file, and an invalid-name directory.
        (state / ".locks").mkdir()
        (state / ".locks" / "some.lock").write_bytes(b"L" * 100)
        (state / "sentinel.del").write_bytes(b"S" * 50)
        (logs / "orphan-file.txt").write_bytes(b"O" * 200)
        (logs / "!!invalid-name").mkdir()
        (logs / "!!invalid-name" / "junk.bin").write_bytes(b"J" * 75)

        whole_tree = agent_run._dir_size_bytes(wt)

        agent_run.cmd_du(_du_args(by_run=True))

        out = capsys.readouterr().out
        assert int(_row(out, "TOTAL")[6]) == whole_tree, (
            "TOTAL must equal the whole worktree size; "
            "unrecognized root content must not be silently excluded"
        )

    def test_equal_state_and_log_roots_do_not_double_count(
        self, git_root, tmp_path, monkeypatch, capsys
    ):
        """When STATE_ROOT == LOG_ROOT the per-run state and log directories are
        the same path; charging it to both columns would double the byte count.

        The fix in _du_collect_rows excludes the state_dir from the log_dir walk
        when they are the same path.

        Mutation: removing the log_excludes.append(state_dir) line in
        _du_collect_rows causes the same directory to be walked twice, inflating
        the total by the size of the shared per-run directory."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        shared_root = tmp_path / "shared"
        shared_root.mkdir()
        monkeypatch.setattr(agent_run, "STATE_ROOT", shared_root)
        monkeypatch.setattr(agent_run, "LOG_ROOT", shared_root)

        run_dir = shared_root / "r1"
        run_dir.mkdir()
        (run_dir / "status").write_text("done\n")
        (run_dir / "cwd").write_text(f"{wt}\n")
        (run_dir / "data").write_bytes(b"D" * 300)

        run_dir_size = agent_run._dir_size_bytes(run_dir)

        agent_run.cmd_du(_du_args(by_run=True))

        out = capsys.readouterr().out
        fields = _row(out, "r1")
        state_b, log_b, scratch_b = int(fields[2]), int(fields[3]), int(fields[4])
        # With equal roots, the per-run dir is charged once across state+log columns.
        assert state_b + log_b + scratch_b == run_dir_size, (
            f"state={state_b} + log={log_b} + scratch={scratch_b} = "
            f"{state_b+log_b+scratch_b} but run_dir_size={run_dir_size}: double-counted"
        )


# ---------------------------------------------------------------------------
# Tests for undefended guards (Part 2)
# ---------------------------------------------------------------------------

class TestReapWorktreeUndefendedGuards:
    """One test per guard that was correct at HEAD but had no test.

    Each test is named after the finding it covers (G4-G18).  The mutation
    that proves the test is noted in the docstring.
    """

    # G4 — late-terminal unparseable age
    def test_late_terminal_run_with_unparseable_age_refuses(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """A run that becomes terminal after collection and whose
        _terminal_state_age_seconds returns None produces a refusal and
        leaves the worktree.

        Mutation: deleting the 'if late_age is None' guard at the final-scan
        loop causes it to compare None < threshold and raise TypeError."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        real_scan = agent_run._worktree_state_scan
        calls = {"n": 0}

        def scan_and_inject():
            calls["n"] += 1
            if calls["n"] > 1 and not (isolated_runs_root / "r2").exists():
                sd = _make_state_run(
                    isolated_runs_root, isolated_log_root, "r2", status="done", cwd=wt
                )
                (sd / "ended_at").write_text("garbage\n")
                monkeypatch.setattr(
                    agent_run, "_terminal_state_age_seconds",
                    lambda d: None if d.name == "r2" else 1000 * 3600,
                )
            return real_scan()

        monkeypatch.setattr(agent_run, "_worktree_state_scan", scan_and_inject)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "run r2 has unparseable age" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    # G5 — pid <= 0 is scan-fatal
    @pytest.mark.parametrize("pid_val", ["0", "-1"])
    def test_invalid_pid_value_aborts_liveness_scan(
        self, isolated_runs_root, isolated_log_root, git_root, capsys, pid_val
    ):
        """pid = 0 or -1 is invalid: _worktree_state_has_live_runner must
        return (None, reason) so the liveness scan aborts.

        Mutation: deleting the 'if pid <= 0' guard causes _pid_alive to
        return False for invalid pids, treating ambiguity as absence."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        sd = _make_state_run(
            isolated_runs_root, isolated_log_root, "a-run", cwd=wt, age_hours=1000
        )
        (sd / "pid").write_text(f"{pid_val}\n")

        live, err = agent_run._worktree_state_has_live_runner(sd)
        assert live is None, f"pid={pid_val} must not be treated as absence"
        assert err is not None and "invalid pid" in err

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "liveness scan failed" in out
        assert "worktrees_removed=0" in out

    # G6 — malformed (non-integer) pid is scan-fatal
    def test_malformed_pid_aborts_liveness_scan(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """A non-integer pid file is ambiguity, not absence; the liveness
        scan must abort and the worktree must survive.

        Mutation: replacing the return of (None, reason) with (False, None)
        treats an unparseable pid as 'no runner' and lets the candidate through."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        sd = _make_state_run(
            isolated_runs_root, isolated_log_root, "a-run", cwd=wt, age_hours=1000
        )
        (sd / "pid").write_text("abc\n")

        live, err = agent_run._worktree_state_has_live_runner(sd)
        assert live is None, "malformed pid must not be treated as absence"
        assert err is not None and "malformed pid" in err

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "liveness scan failed" in out
        assert "malformed pid" in out
        assert "worktrees_removed=0" in out

    # G8 — git worktree remove failure is a refusal
    def test_worktree_remove_git_failure_refuses(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """When _watch_run_git_checked fails only for the 'worktree remove'
        invocation, the pass reports a skip and counts removed=0.

        Mutation: dropping the failure branch from _worktree_remove causes
        [removing] to appear and worktrees_removed to increment despite git
        actually refusing the removal."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        real_git = agent_run._watch_run_git_checked

        def git_fails_worktree_remove(path, args, **kw):
            if list(args[:2]) == ["worktree", "remove"]:
                return agent_run._WatchGitOutcome(None, "git_failed")
            return real_git(path, args, **kw)

        monkeypatch.setattr(agent_run, "_watch_run_git_checked", git_fails_worktree_remove)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir(), "worktree must still exist after git worktree remove failure"
        assert "git worktree remove failed" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    # G9 (a) — invalid state entry name is scan-fatal
    def test_invalid_state_directory_name_aborts_scan(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """A state entry whose name fails _validate_run_name aborts the scan;
        the pass deletes nothing.

        Mutation: replacing the return of _WorktreeStateScan(None, ...) with
        'continue' makes the invalid entry a no-op instead of scan-fatal."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)
        (isolated_runs_root / "!!bogus").mkdir()

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "invalid state directory name" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    # G9 (b) — non-directory state entry is scan-fatal
    def test_non_directory_state_entry_aborts_scan(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """A regular file directly under STATE_ROOT with a valid run name
        aborts the scan; the pass deletes nothing.

        Mutation: replacing the return of _WorktreeStateScan(None, ...) with
        'continue' makes the non-directory entry a no-op instead of scan-fatal."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)
        (isolated_runs_root / "orphan-file").write_text("not a directory\n")

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "state entry is not a real directory" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    # G10 — empty worktree list is a refusal
    def test_empty_worktree_list_refuses(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """'git worktree list --porcelain -z' returning no 'worktree ' fields
        is treated as an error: the output is not what was expected.

        Mutation: deleting the 'if not roots' guard treats an empty roots list
        as 'no nested worktrees, proceed' — the wrong fail direction."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        real_git = agent_run._watch_run_git_checked

        def git_returns_empty_worktree_list(path, args, **kw):
            if list(args[:2]) == ["worktree", "list"]:
                return agent_run._WatchGitOutcome("", None)
            return real_git(path, args, **kw)

        monkeypatch.setattr(agent_run, "_watch_run_git_checked", git_returns_empty_worktree_list)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "worktree list returned no roots" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    # G11 — unresolvable registered-worktree path is a refusal
    def test_unresolvable_registered_worktree_refuses(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """A 'worktree /gone/path' entry whose path _worktree_resolve_cwd
        cannot resolve causes a refusal.

        Mutation (no separate mutation needed): nothing in the suite stubs a
        worktree list line that fails _worktree_resolve_cwd."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        real_git = agent_run._watch_run_git_checked

        def git_returns_bad_worktree_path(path, args, **kw):
            if list(args[:2]) == ["worktree", "list"]:
                return agent_run._WatchGitOutcome("worktree /gone/nonexistent\n", None)
            return real_git(path, args, **kw)

        monkeypatch.setattr(agent_run, "_watch_run_git_checked", git_returns_bad_worktree_path)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "cannot resolve registered worktree" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    # G12 — scandir/lstat error in foreign-nested scan is a refusal
    def test_foreign_nested_scandir_error_refuses(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """A git ls-files failure inside _worktree_foreign_nested_reason causes
        a refusal; any enumeration failure fails closed.

        Mutation: replacing the stdout-is-None branch with 'continue' treats
        an unreadable enumeration as empty, letting the candidate through."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        subdir = wt / "sub"
        subdir.mkdir()
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        real_git = agent_run._watch_run_git_checked

        def git_fails_ls_files_others(path, args, **kw):
            if list(args[:2]) == ["ls-files", "--others"]:
                return agent_run._WatchGitOutcome(None, "git_failed")
            return real_git(path, args, **kw)

        monkeypatch.setattr(agent_run, "_watch_run_git_checked", git_fails_ls_files_others)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "cannot enumerate untracked paths" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    # G13 — empty rev-parse path yields UNKNOWN classification
    def test_empty_rev_parse_path_is_unknown(self, git_root, monkeypatch):
        """Four lines of rev-parse output where git_dir or common_dir is an
        empty string must yield _WORKTREE_UNKNOWN with 'empty rev-parse path'.

        Mutation: deleting the 'if not git_dir or not common_dir' guard causes
        the empty-string paths to compare equal, classifying as _WORKTREE_MAIN."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        monkeypatch.setattr(
            agent_run, "_watch_run_git_checked",
            lambda *a, **k: agent_run._WatchGitOutcome("false\ntrue\n\n\n", None),
        )

        info = agent_run._worktree_classify(wt)

        assert info.kind == agent_run._WORKTREE_UNKNOWN
        assert info.detail == "empty rev-parse path"

    # G16 — du charged-excludes root guard is load-bearing
    def test_du_state_root_outside_worktree_with_symlinked_state_dir(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch, capsys
    ):
        """STATE_ROOT is outside the worktree; the per-run state dir is a
        symlink whose resolved target is inside it.  TOTAL must still equal
        the whole worktree size — the guard at _path_is_within(configured,
        root) ensures the symlinked target is excluded from the worktree
        charge (since it is already counted in the state charge).

        Mutation M50: removing the guard causes the symlinked bytes to be
        double-counted and/or dropped from TOTAL depending on traversal order."""
        repo = _make_repo(tmp_path / "inside")
        wt = _add_worktree(repo, tmp_path / "inside" / "wt", "feature")
        (wt / "payload.bin").write_bytes(b"p" * 5000)

        state = tmp_path / "outside-state"
        logs = tmp_path / "outside-logs"
        state.mkdir()
        logs.mkdir()
        monkeypatch.setattr(agent_run, "STATE_ROOT", state)
        monkeypatch.setattr(agent_run, "LOG_ROOT", logs)

        real_state_dir = wt / "real-r1-state"
        real_state_dir.mkdir()
        (real_state_dir / "status").write_text("done\n")
        (real_state_dir / "cwd").write_text(f"{wt}\n")
        (real_state_dir / "data").write_bytes(b"S" * 900)
        (state / "r1").symlink_to(real_state_dir, target_is_directory=True)
        (logs / "r1").mkdir()

        whole = agent_run._dir_size_bytes(wt)

        agent_run.cmd_du(_du_args(by_run=True))

        out = capsys.readouterr().out
        total = _row(out, "TOTAL")
        assert int(total[6]) == whole, (
            f"TOTAL {total[6]} != whole worktree {whole}: bytes dropped or doubled"
        )

    # G17 — --force-dirty must not override unpushed commits
    def test_force_dirty_does_not_override_unpushed_commits(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """Documented property (property 2): --force-dirty overrides content
        checks only, never unpushed commits.  This pins the most tempting
        future simplification: moving _worktree_unpushed_reason inside the
        'if not force_dirty:' block would make mutation M (adjacent code)
        pass all existing tests.

        Mutation: moving _worktree_unpushed_reason inside the force_dirty
        guard causes the worktree with unpushed commits to be deleted under
        --force-dirty (mutation passes all 119 tests without this test)."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        (wt / "tracked.txt").write_text("committed but unpushed\n")
        _git(wt, "commit", "-qam", "local only")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        agent_run.cmd_reap(_reap_args(force_dirty=True))

        out = capsys.readouterr().out
        assert wt.is_dir(), "worktree with unpushed commits deleted under --force-dirty"
        assert "not on any remote-tracking ref" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    # G18 (a) — reconciled cwd containment (subdirectory blocks removal)
    def test_reconciled_sharer_in_subdirectory_keeps_worktree(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """A run reconciled in this invocation whose cwd is a subdirectory of
        the candidate is inside the tree 'git worktree remove' would delete;
        it must still block removal.

        Mutation M24: replacing _path_is_within with equality for the
        reconciled-cwd check lets a sharer whose cwd is a subdirectory slip
        through and the worktree is deleted."""
        dead = subprocess.Popen(["true"])
        dead.wait()
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        nested = wt / "sub" / "deeper"
        nested.mkdir(parents=True)
        _make_state_run(isolated_runs_root, isolated_log_root, "a-old", cwd=wt, age_hours=1000)
        sd = _make_state_run(
            isolated_runs_root, isolated_log_root, "b-reconciled",
            status="running", cwd=nested, age_hours=1000,
        )
        (sd / "pid").write_text(f"{dead.pid}\n")

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "a sharing run was reconciled in this invocation" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    # G18 (b) — late-terminal cwd containment (subdirectory is weighed by age gate)
    def test_late_terminal_sharer_in_subdirectory_is_weighed_by_age_gate(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """A late-terminal run whose cwd is a subdirectory of the candidate
        must be weighed by the age gate, not skipped because its cwd is not
        equal to cand.resolved.

        Mutation M36: replacing _path_is_within with equality in the
        late-terminal loop skips a sharer in a subdirectory, allowing the
        worktree to be removed even if the sharing run is fresh."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        nested = wt / "sub" / "deeper"
        nested.mkdir(parents=True)
        _make_state_run(isolated_runs_root, isolated_log_root, "a-old", cwd=wt, age_hours=1000)

        real_scan = agent_run._worktree_state_scan
        calls = {"n": 0}

        def scan_and_inject():
            calls["n"] += 1
            if calls["n"] > 1 and not (isolated_runs_root / "b-fresh").exists():
                _make_state_run(
                    isolated_runs_root, isolated_log_root, "b-fresh",
                    status="done", cwd=nested, age_hours=0,
                )
            return real_scan()

        monkeypatch.setattr(agent_run, "_worktree_state_scan", scan_and_inject)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "youngest sharing run is below the age threshold" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out

    # F8 (a) — reaper lock must be exclusive
    def test_gc_pass_requests_exclusive_lock(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """_worktree_gc_pass must acquire the publication lock with exclusive=True.

        A shared lock (exclusive=False) would allow a concurrent launcher to
        publish new state between the final scan and git worktree remove, leaving
        the race that the exclusive lock is meant to prevent.

        Mutation: changing exclusive=True to exclusive=False in
        _worktree_gc_pass causes this test to fail because the recorded lock
        mode is False instead of True."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        lock_modes: list = []
        real_lock = agent_run._worktree_publication_lock
        from contextlib import contextmanager

        @contextmanager
        def recording_lock(*, exclusive):
            lock_modes.append(exclusive)
            with real_lock(exclusive=exclusive) as fd:
                yield fd

        monkeypatch.setattr(agent_run, "_worktree_publication_lock", recording_lock)

        agent_run.cmd_reap(_reap_args(dry_run=True))
        capsys.readouterr()

        assert lock_modes, "_worktree_gc_pass must acquire the publication lock"
        assert all(m is True for m in lock_modes), (
            f"gc_pass lock modes must all be exclusive=True; got {lock_modes}"
        )

    # F8 (b) — live PID with no process_identity is scan-fatal
    def test_live_pid_without_process_identity_aborts_scan(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        """A live PID with no recorded process_identity file must abort the
        liveness scan and preserve the worktree, not be treated as no runner.

        A missing or empty identity could mean the runner died before writing it
        or a race; ignoring it would allow a live process to lose its protection.

        Mutation: replacing 'return None, reason' with 'return False, None' for
        the no-identity branch causes _worktree_state_has_live_runner to report
        absence rather than ambiguity, letting the worktree be deleted."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        # A run whose PID is our own (live) but has no process_identity file.
        sd = _make_state_run(
            isolated_runs_root, isolated_log_root, "a-run", cwd=wt, age_hours=1000
        )
        (sd / "pid").write_text(f"{os.getpid()}\n")
        # Deliberately omit writing process_identity.

        live, err = agent_run._worktree_state_has_live_runner(sd)
        assert live is None, "live PID with no identity must return ambiguity (None), not False"
        assert err is not None and "no process identity" in err

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir(), "worktree must not be deleted when liveness is ambiguous"
        assert "liveness scan failed" in out
        assert "worktrees_removed=0" in out

    # F8 (c) — initial age_seconds=None is refused
    def test_candidate_with_unknown_age_is_refused(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """A candidate with age_seconds=None (unparseable ended_at, no fallback)
        must be refused by the age gate, not treated as old enough.

        Mutation: removing the 'cand.age_seconds is None' check in
        _worktree_candidate_refusal causes the candidate to pass the age gate
        (None < threshold raises TypeError, or the check is skipped)."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        sd = _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt)
        # Corrupt ended_at and remove the dir mtime fallback so age returns None.
        (sd / "ended_at").write_text("not-a-timestamp\n")
        (sd / "status").write_text("done\n")

        # Confirm that _terminal_state_age_seconds returns None for this entry.
        age = agent_run._terminal_state_age_seconds(sd)
        assert age is None or isinstance(age, float), "setup check"

        # Monkeypatch to guarantee None (in case mtime fallback returns a value).
        real_age = agent_run._terminal_state_age_seconds
        monkeypatch.setattr(
            agent_run, "_terminal_state_age_seconds",
            lambda d: None if d.name == "r1" else real_age(d),
        )

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir(), "worktree with age=None must not be deleted"
        assert "worktrees_removed=0" in out

    # F8 (d) — malformed rev-list count is refused
    def test_malformed_rev_list_count_refuses_removal(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """A successful rev-list --count that returns non-integer output must
        be refused rather than treated as zero (i.e. 'no unpushed commits').

        Mutation: removing the '_safe_int returns None' guard in
        _worktree_unpushed_reason and treating all non-error output as 0 causes
        the candidate to pass the unpushed gate with corrupted count data."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        real_git = agent_run._watch_run_git_checked

        def git_returns_bad_count(path, args, **kw):
            if list(args[:2]) == ["rev-list", "--count"]:
                return agent_run._WatchGitOutcome("not-a-number\n", None)
            return real_git(path, args, **kw)

        monkeypatch.setattr(agent_run, "_watch_run_git_checked", git_returns_bad_count)

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir(), "worktree must not be deleted with malformed rev-list count"
        assert "unparseable unpushed commit count" in out
        assert "worktrees_removed=0 worktrees_skipped=1" in out


# ---------------------------------------------------------------------------
# launch --worktree: creating (or attaching to) the linked worktree a run
# launches into.
# ---------------------------------------------------------------------------

class TestLaunchCreatesWorktree:
    """`agent-run launch --worktree DIR --worktree-base REF` creates a linked
    worktree and points the launch at it, via cmd_launch's real code path —
    no mocked git, no asserting on internal call arguments."""

    @pytest.fixture(autouse=True)
    def _restore_cwd(self):
        # _create_launch_worktree's invocation-cwd discovery, and
        # _apply_launch_cwd, both chdir the test process; put it back so
        # later tests in the same session aren't affected.
        origin = os.getcwd()
        yield
        os.chdir(origin)

    def test_happy_path_creates_worktree_branch_and_launches(
        self, isolated_runs_root, isolated_log_root, git_root, capsys
    ):
        repo = _make_repo(git_root)
        wt_dir = git_root / "wt-happy"
        name = "wt-happy"
        args = _launch_args(
            name=name, worktree=str(wt_dir), worktree_base="main", worktree_repo=str(repo),
        )

        rc = agent_run.cmd_launch(args)
        assert rc == 0

        state_dir = isolated_runs_root / name
        status = _wait_terminal(state_dir)
        assert status == "done"

        resolved = str(wt_dir.resolve())
        assert (state_dir / "cwd").read_text().strip() == resolved
        assert (state_dir / "worktree_created").read_text().strip() == resolved

        branch_list = _git(repo, "branch", "--list", name)
        assert name in branch_list

        # cmd_status must surface worktree_created in its user-visible line,
        # not just leave it as a file only cmd_launch itself reads.
        capsys.readouterr()  # discard cmd_launch's own stdout
        agent_run.cmd_status(argparse.Namespace(name=name))
        status_out = capsys.readouterr().out
        assert f"worktree_created={resolved!r}" in status_out

        # The integration this feature exists for: the real classifier must
        # recognise the created directory as a linked worktree, exactly as
        # `reap --include-worktrees` and `du` would.
        info = agent_run._worktree_classify(wt_dir)
        assert info.kind == agent_run._WORKTREE_LINKED
        assert info.common_dir is not None
        assert Path(info.common_dir).samefile(repo / ".git")

    def test_worktree_dir_recorded_through_symlinked_ancestor_resolves_to_realpath(
        self, isolated_runs_root, isolated_log_root, git_root
    ):
        """`state_dir/cwd` and `state_dir/worktree_created` must agree with
        each other and with `os.path.realpath`, not the raw --worktree path
        as typed. Uses an explicit symlink rather than relying on tmp_path
        happening to already be symlink-resolved (it is, via macOS's
        /var -> /private/var, which makes str(tmp_path) == realpath(tmp_path)
        and so cannot distinguish the two)."""
        repo = _make_repo(git_root)
        real_parent = git_root / "real-parent"
        real_parent.mkdir()
        link_parent = git_root / "link-parent"
        os.symlink(real_parent, link_parent)
        wt_dir_via_link = link_parent / "wt"
        name = "wt-symlinked-ancestor"
        args = _launch_args(
            name=name, worktree=str(wt_dir_via_link), worktree_base="main",
            worktree_repo=str(repo),
        )

        rc = agent_run.cmd_launch(args)
        assert rc == 0
        state_dir = isolated_runs_root / name
        _wait_terminal(state_dir)

        expected = os.path.realpath(wt_dir_via_link)
        assert expected == str(real_parent / "wt")  # actually resolved the symlink
        cwd_recorded = (state_dir / "cwd").read_text().strip()
        created_recorded = (state_dir / "worktree_created").read_text().strip()
        assert cwd_recorded == expected
        assert created_recorded == expected
        assert cwd_recorded == created_recorded

    def test_relative_worktree_and_worktree_repo_anchored_at_invocation_cwd(
        self, isolated_runs_root, isolated_log_root, git_root, tmp_path, monkeypatch
    ):
        """A relative --worktree DIR (and --worktree-repo) must be resolved
        against the invocation cwd (where agent-run was actually typed),
        before --cwd/--worktree entry ever chdirs the process -- not some
        other anchor such as the repo root."""
        repo = _make_repo(git_root)
        invocation_dir = tmp_path / "invocation-dir"
        invocation_dir.mkdir()
        monkeypatch.chdir(invocation_dir)
        name = "wt-relative-paths"
        args = _launch_args(
            name=name, worktree="../wt-rel", worktree_base="main",
            worktree_repo=str(Path("..") / git_root.name / repo.name),
        )
        # worktree_repo relative to invocation_dir must resolve to repo.
        assert (invocation_dir / ".." / git_root.name / repo.name).resolve() == repo.resolve()

        rc = agent_run.cmd_launch(args)
        assert rc == 0
        _wait_terminal(isolated_runs_root / name)

        expected_wt = (invocation_dir / ".." / "wt-rel").resolve()
        assert expected_wt.is_dir()
        assert agent_run._worktree_classify(expected_wt).kind == agent_run._WORKTREE_LINKED

    def test_tilde_worktree_is_expanded(
        self, isolated_runs_root, isolated_log_root, git_root, tmp_path, monkeypatch
    ):
        """--worktree ~/... must have ~ expanded, matching --cwd/--prompt-file."""
        repo = _make_repo(git_root)
        fake_home = tmp_path / "fake-home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        name = "wt-tilde"
        args = _launch_args(
            name=name, worktree="~/wt-tilde", worktree_base="main", worktree_repo=str(repo),
        )

        rc = agent_run.cmd_launch(args)
        assert rc == 0
        _wait_terminal(isolated_runs_root / name)

        expected_wt = fake_home / "wt-tilde"
        assert expected_wt.is_dir()
        assert agent_run._worktree_classify(expected_wt).kind == agent_run._WORKTREE_LINKED

    def test_missing_parent_directories_are_created_like_git_worktree_add(
        self, isolated_runs_root, isolated_log_root, git_root
    ):
        """`git worktree add` creates missing parent directories on its own;
        agent-run must not be stricter than the tool it wraps here."""
        repo = _make_repo(git_root)
        wt_dir = git_root / "deep" / "nested" / "wt"
        name = "wt-deep-parent"
        args = _launch_args(
            name=name, worktree=str(wt_dir), worktree_base="main", worktree_repo=str(repo),
        )

        rc = agent_run.cmd_launch(args)
        assert rc == 0

        _wait_terminal(isolated_runs_root / name)
        assert agent_run._worktree_classify(wt_dir).kind == agent_run._WORKTREE_LINKED

    def test_worktree_and_cwd_are_mutually_exclusive(
        self, isolated_runs_root, isolated_log_root, git_root, tmp_path
    ):
        repo = _make_repo(git_root)
        args = _launch_args(
            name="both", worktree=str(git_root / "wt"), worktree_base="main",
            worktree_repo=str(repo), cwd=str(tmp_path),
        )

        with pytest.raises(SystemExit) as exc:
            agent_run.cmd_launch(args)

        assert exc.value.code == 2
        assert not (git_root / "wt").exists()

    def test_worktree_without_base_is_usage_error(
        self, isolated_runs_root, isolated_log_root, git_root
    ):
        repo = _make_repo(git_root)
        wt_dir = git_root / "wt-no-base"
        args = _launch_args(name="no-base", worktree=str(wt_dir), worktree_repo=str(repo))

        with pytest.raises(SystemExit) as exc:
            agent_run.cmd_launch(args)

        assert exc.value.code == 2
        assert not wt_dir.exists()

    def test_worktree_reuse_without_worktree_is_usage_error(
        self, isolated_runs_root, isolated_log_root
    ):
        args = _launch_args(name="reuse-alone", worktree_reuse=True)

        with pytest.raises(SystemExit) as exc:
            agent_run.cmd_launch(args)

        assert exc.value.code == 2

    @pytest.mark.parametrize(
        "kw", [
            dict(worktree_base="main"),
            dict(worktree_branch="feature"),
            dict(worktree_repo="/tmp/some-repo"),
        ],
    )
    def test_worktree_only_flag_without_worktree_is_usage_error(
        self, isolated_runs_root, isolated_log_root, kw
    ):
        """A typo omitting --worktree DIR must not silently launch in the
        invocation cwd despite apparently-configured worktree flags."""
        args = _launch_args(name="worktree-flag-alone", **kw)

        with pytest.raises(SystemExit) as exc:
            agent_run.cmd_launch(args)

        assert exc.value.code == 2
        assert not (isolated_runs_root / "worktree-flag-alone").exists()

    def test_reuse_attaches_to_existing_branch_when_dir_is_absent(
        self, isolated_runs_root, isolated_log_root, git_root
    ):
        """--worktree-reuse with an existing branch but DIR absent must check
        that branch out (not create a detached checkout at --worktree-base),
        and this invocation did create the directory -- so worktree_created
        must be written, even though the branch pre-existed."""
        repo = _make_repo(git_root)
        _git(repo, "branch", "pre-existing-branch", "main")
        wt_dir = git_root / "wt-reuse-existing-branch"
        name = "reuse-existing-branch"
        args = _launch_args(
            name=name, worktree=str(wt_dir), worktree_base="main",
            worktree_branch="pre-existing-branch", worktree_repo=str(repo),
            worktree_reuse=True,
        )

        rc = agent_run.cmd_launch(args)
        assert rc == 0
        state_dir = isolated_runs_root / name
        _wait_terminal(state_dir)

        assert _git(wt_dir, "rev-parse", "--abbrev-ref", "HEAD").strip() == "pre-existing-branch"
        resolved = str(wt_dir.resolve())
        assert (state_dir / "worktree_created").read_text().strip() == resolved

    def test_existing_directory_without_reuse_fails_and_is_untouched(
        self, isolated_runs_root, isolated_log_root, git_root
    ):
        """A DIR that is already a *valid* linked worktree of the same repo
        (the one case --worktree-reuse would accept) must still be refused
        without --worktree-reuse: existence alone is disqualifying."""
        repo = _make_repo(git_root)
        existing = _add_worktree(repo, git_root / "already-here", "pre-existing-branch")
        marker = existing / "tracked.txt"
        before = marker.read_text()
        args = _launch_args(
            name="existing-dir", worktree=str(existing), worktree_base="main",
            worktree_branch="a-brand-new-branch", worktree_repo=str(repo),
        )

        with pytest.raises(SystemExit, match="already exists") as exc:
            agent_run.cmd_launch(args)

        assert _exit_status(exc.value.code) == 1
        assert marker.read_text() == before
        assert agent_run._worktree_classify(existing).kind == agent_run._WORKTREE_LINKED
        assert not (isolated_runs_root / "existing-dir").exists()

    def test_existing_branch_without_reuse_fails_and_leaves_no_worktree(
        self, isolated_runs_root, isolated_log_root, git_root
    ):
        repo = _make_repo(git_root)
        _git(repo, "branch", "taken-branch", "main")
        wt_dir = git_root / "wt-taken-branch"
        args = _launch_args(
            name="existing-branch", worktree=str(wt_dir), worktree_base="main",
            worktree_branch="taken-branch", worktree_repo=str(repo),
        )

        with pytest.raises(SystemExit, match="already exists in") as exc:
            agent_run.cmd_launch(args)

        assert _exit_status(exc.value.code) == 1
        assert "--worktree-reuse" in str(exc.value)
        assert not wt_dir.exists()
        assert not (isolated_runs_root / "existing-branch").exists()

    def test_reuse_attaches_to_existing_worktree_without_marking_it_created(
        self, isolated_runs_root, isolated_log_root, git_root
    ):
        """--worktree-reuse attaches without creating anything: a successful
        run must not write the worktree_created ownership marker."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt-preexisting", "feature")
        name = "reuse-run"
        args = _launch_args(
            name=name, worktree=str(wt), worktree_base="main",
            worktree_branch="feature", worktree_repo=str(repo), worktree_reuse=True,
        )

        rc = agent_run.cmd_launch(args)
        assert rc == 0

        state_dir = isolated_runs_root / name
        assert _wait_terminal(state_dir) == "done"
        assert (state_dir / "cwd").read_text().strip() == str(wt.resolve())
        assert not (state_dir / "worktree_created").exists()

    def test_reuse_accepts_a_branch_shadowed_by_a_same_named_tag(
        self, isolated_runs_root, isolated_log_root, git_root
    ):
        """M1: `git symbolic-ref --quiet --short HEAD` returns the shortest
        *unambiguous* name, which is "heads/release" rather than "release"
        whenever a tag named "release" also exists -- a legitimate reuse of
        a worktree genuinely on branch "release" must not be refused just
        because a same-named tag happens to exist."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt-shadowed", "release")
        _git(repo, "tag", "release")
        name = "shadowed-branch-reuse"
        args = _launch_args(
            name=name, worktree=str(wt), worktree_base="main",
            worktree_branch="release", worktree_repo=str(repo), worktree_reuse=True,
        )

        rc = agent_run.cmd_launch(args)
        assert rc == 0

        state_dir = isolated_runs_root / name
        assert _wait_terminal(state_dir) == "done"
        assert (state_dir / "cwd").read_text().strip() == str(wt.resolve())

    def test_reuse_rollback_on_launch_failure_never_removes_the_worktree(
        self, isolated_runs_root, isolated_log_root, git_root
    ):
        """A launch failure after attaching via --worktree-reuse must never
        trigger rollback removal: this invocation created nothing."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt-preexisting-2", "feature2")
        # An empty raw command makes _cmd_launch_locked fail with
        # sys.exit("missing command") after _create_launch_worktree has
        # already returned None (nothing to roll back) for the reuse case.
        args = _launch_args(
            name="reuse-run-fails", command=[], worktree=str(wt), worktree_base="main",
            worktree_branch="feature2", worktree_repo=str(repo), worktree_reuse=True,
        )

        with pytest.raises(SystemExit, match="missing command"):
            agent_run.cmd_launch(args)

        assert wt.is_dir()
        assert agent_run._worktree_classify(wt).kind == agent_run._WORKTREE_LINKED
        assert not (isolated_runs_root / "reuse-run-fails").exists()

    def test_reuse_refuses_linked_worktree_of_a_different_repo(
        self, isolated_runs_root, isolated_log_root, git_root
    ):
        """--worktree-reuse pointed at a linked worktree of a different repo
        (named via --worktree-repo of the *first* repo) must be refused, and
        the other repo's worktree must be untouched: this is the exact
        cross-contamination hazard the feature exists to prevent."""
        repo_a = _make_repo(git_root, "repo-a")
        repo_b = _make_repo(git_root, "repo-b")
        wt_b = _add_worktree(repo_b, git_root / "wt-b", "feature-b")
        marker = wt_b / "tracked.txt"
        before = marker.read_text()
        args = _launch_args(
            name="cross-repo-reuse", worktree=str(wt_b), worktree_base="main",
            worktree_branch="feature-b", worktree_repo=str(repo_a), worktree_reuse=True,
        )

        with pytest.raises(SystemExit, match="not a linked worktree of"):
            agent_run.cmd_launch(args)

        assert marker.read_text() == before
        assert agent_run._worktree_classify(wt_b).kind == agent_run._WORKTREE_LINKED
        assert not (isolated_runs_root / "cross-repo-reuse").exists()

    def test_reuse_refuses_the_main_worktree(
        self, isolated_runs_root, isolated_log_root, git_root
    ):
        """--worktree-reuse pointed at the repo's own main worktree must be
        refused, not silently attached."""
        repo = _make_repo(git_root)
        marker = repo / "tracked.txt"
        before = marker.read_text()
        args = _launch_args(
            name="main-worktree-reuse", worktree=str(repo), worktree_base="main",
            worktree_repo=str(repo), worktree_reuse=True,
        )

        with pytest.raises(SystemExit, match="not a linked worktree of"):
            agent_run.cmd_launch(args)

        assert marker.read_text() == before
        assert not (isolated_runs_root / "main-worktree-reuse").exists()

    def test_reuse_refuses_a_copied_unregistered_worktree(
        self, isolated_runs_root, isolated_log_root, git_root
    ):
        """A recursive copy of a linked worktree keeps a .git file pointing at
        the original repo's admin dir, so _worktree_classify reports LINKED
        and the common-dir matches -- but `git worktree list` does not
        register the copy, and `git worktree remove` would refuse it."""
        import shutil

        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt-original", "feature")
        wt_copy = git_root / "wt-copy"
        shutil.copytree(wt, wt_copy, symlinks=True)
        assert (wt_copy / ".git").exists()

        args = _launch_args(
            name="copied-worktree-reuse", worktree=str(wt_copy), worktree_base="main",
            worktree_branch="feature", worktree_repo=str(repo), worktree_reuse=True,
        )

        with pytest.raises(SystemExit, match="not a registered worktree"):
            agent_run.cmd_launch(args)

        assert wt_copy.is_dir()  # untouched, not removed or mutated
        assert not (isolated_runs_root / "copied-worktree-reuse").exists()

    def test_reuse_refuses_branch_mismatch(
        self, isolated_runs_root, isolated_log_root, git_root
    ):
        """--worktree-branch requested --worktree-reuse pointed at a registered
        worktree sitting on a *different* branch must be refused rather than
        silently running on whichever branch happens to be checked out."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt-actual-branch", "actual")
        args = _launch_args(
            name="branch-mismatch-reuse", worktree=str(wt), worktree_base="main",
            worktree_branch="requested", worktree_repo=str(repo), worktree_reuse=True,
        )

        with pytest.raises(SystemExit, match="is on branch 'actual'"):
            agent_run.cmd_launch(args)

        assert _git(wt, "rev-parse", "--abbrev-ref", "HEAD").strip() == "actual"
        assert not (isolated_runs_root / "branch-mismatch-reuse").exists()

    def test_reuse_refuses_branch_that_is_a_prefix_of_the_requested_branch(
        self, isolated_runs_root, isolated_log_root, git_root
    ):
        """T5: the branch-equality check must be exact string equality, not
        a prefix match in either direction. A worktree actually on branch
        'feat' must not satisfy a request for '--worktree-branch feature'
        just because 'feature'.startswith('feat')."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt-feat-prefix", "feat")
        args = _launch_args(
            name="prefix-mismatch-reuse", worktree=str(wt), worktree_base="main",
            worktree_branch="feature", worktree_repo=str(repo), worktree_reuse=True,
        )

        with pytest.raises(SystemExit, match="is on branch 'feat'"):
            agent_run.cmd_launch(args)

        assert _git(wt, "rev-parse", "--abbrev-ref", "HEAD").strip() == "feat"
        assert not (isolated_runs_root / "prefix-mismatch-reuse").exists()

    def test_reuse_refuses_detached_head(
        self, isolated_runs_root, isolated_log_root, git_root
    ):
        """A registered worktree with a detached HEAD is rejected explicitly,
        rather than guessed at or silently attached."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt-detached", "detached-src")
        _git(wt, "checkout", "--detach", "-q")
        args = _launch_args(
            name="detached-reuse", worktree=str(wt), worktree_base="main",
            worktree_branch="detached-src", worktree_repo=str(repo), worktree_reuse=True,
        )

        with pytest.raises(SystemExit, match="detached HEAD"):
            agent_run.cmd_launch(args)

        assert not (isolated_runs_root / "detached-reuse").exists()

    def test_unresolvable_base_fails_before_any_mutation(
        self, isolated_runs_root, isolated_log_root, git_root
    ):
        repo = _make_repo(git_root)
        wt_dir = git_root / "wt-bad-base"
        args = _launch_args(
            name="bad-base", worktree=str(wt_dir), worktree_base="no-such-ref",
            worktree_repo=str(repo),
        )

        with pytest.raises(SystemExit, match="does not resolve to a commit") as exc:
            agent_run.cmd_launch(args)

        assert _exit_status(exc.value.code) == 1
        assert not wt_dir.exists()
        assert not (isolated_runs_root / "bad-base").exists()
        assert _git(repo, "worktree", "list").count("\n") == 1  # only the main worktree

    def test_launch_failure_after_creation_removes_worktree_and_branch(
        self, isolated_runs_root, isolated_log_root, git_root
    ):
        repo = _make_repo(git_root)
        wt_dir = git_root / "wt-rollback"
        name = "rollback-run"
        args = _launch_args(
            name=name, command=[], worktree=str(wt_dir), worktree_base="main",
            worktree_repo=str(repo),
        )

        with pytest.raises(SystemExit, match="missing command"):
            agent_run.cmd_launch(args)

        assert not wt_dir.exists()
        assert name not in _git(repo, "branch", "--list", name)
        assert not (isolated_runs_root / name).exists()

    def test_worktree_base_is_frozen_to_the_requested_ref_not_invocation_head(
        self, isolated_runs_root, isolated_log_root, git_root
    ):
        """--worktree-base is the entire reason this feature exists: the new
        worktree's HEAD must be the requested ref's commit, never whatever
        the invocation repo's own HEAD happens to be pointed at."""
        repo = _make_repo(git_root)
        base_head_oid = _git(repo, "rev-parse", "HEAD").strip()
        (repo / "tracked.txt").write_text("second commit content\n")
        _git(repo, "add", "tracked.txt")
        _git(repo, "commit", "-qm", "second commit")
        _git(repo, "push", "-q", "origin", "main")
        invocation_head_oid = _git(repo, "rev-parse", "HEAD").strip()
        assert base_head_oid != invocation_head_oid

        base_ref = "base-ref"
        _git(repo, "branch", base_ref, base_head_oid)
        wt_dir = git_root / "wt-base-oid"
        name = "wt-base-oid"
        args = _launch_args(
            name=name, worktree=str(wt_dir), worktree_base=base_ref, worktree_repo=str(repo),
        )

        rc = agent_run.cmd_launch(args)
        assert rc == 0
        _wait_terminal(isolated_runs_root / name)

        assert _git(wt_dir, "rev-parse", "HEAD").strip() == base_head_oid
        assert _git(wt_dir, "rev-parse", "HEAD").strip() != invocation_head_oid

    def test_worktree_base_ref_moved_between_validation_and_creation_is_not_followed(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch
    ):
        """--worktree-base is resolved to a commit OID once; a concurrent
        ref update between validation and `worktree add` must not change
        which commit the new branch is built on."""
        repo = _make_repo(git_root)
        first_oid = _git(repo, "rev-parse", "HEAD").strip()
        _git(repo, "branch", "moving-base", first_oid)
        (repo / "tracked.txt").write_text("moved-base content\n")
        _git(repo, "add", "tracked.txt")
        _git(repo, "commit", "-qm", "moved-base commit")
        second_oid = _git(repo, "rev-parse", "HEAD").strip()
        assert first_oid != second_oid

        real_git_checked = agent_run._watch_run_git_checked
        moved = {"done": False}

        def _racing_git_checked(path, git_args, timeout=agent_run.WATCH_GIT_SUBPROCESS_TIMEOUT_SECONDS):
            outcome = real_git_checked(path, git_args, timeout)
            if (
                not moved["done"]
                and list(git_args[:3]) == ["rev-parse", "--verify", "--quiet"]
                and git_args[3] == "moving-base^{commit}"
            ):
                moved["done"] = True
                real_git_checked(path, ["branch", "-f", "moving-base", second_oid])
            return outcome

        monkeypatch.setattr(agent_run, "_watch_run_git_checked", _racing_git_checked)

        wt_dir = git_root / "wt-moving-base"
        name = "wt-moving-base"
        args = _launch_args(
            name=name, worktree=str(wt_dir), worktree_base="moving-base", worktree_repo=str(repo),
        )

        rc = agent_run.cmd_launch(args)
        assert rc == 0
        _wait_terminal(isolated_runs_root / name)

        assert _git(wt_dir, "rev-parse", "HEAD").strip() == first_oid

    def test_worktree_add_failure_after_branch_creation_does_not_orphan_branch(
        self, isolated_runs_root, isolated_log_root, git_root
    ):
        """Branch creation and worktree creation are two operations: a
        `git worktree add` failure occurring after the branch already exists
        must not leave that branch permanently orphaned."""
        repo = _make_repo(git_root)
        unwritable_parent = git_root / "noperm"
        unwritable_parent.mkdir()
        wt_dir = unwritable_parent / "wt-add-fails"
        os.chmod(unwritable_parent, 0o555)
        name = "add-fails-run"
        args = _launch_args(
            name=name, worktree=str(wt_dir), worktree_base="main", worktree_repo=str(repo),
        )

        try:
            with pytest.raises(SystemExit, match="git worktree add failed"):
                agent_run.cmd_launch(args)
        finally:
            os.chmod(unwritable_parent, 0o755)

        assert name not in _git(repo, "branch", "--list", name)
        assert not (isolated_runs_root / name).exists()

    def test_rollback_keeps_a_branch_the_workload_moved(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """The branch OID guard in _worktree_delete_branch_if_unchanged is
        what stops rollback from discarding committed work: if the branch
        moved away from the OID this invocation created it at (a workload
        committing into the new worktree before launch fails), the branch
        must survive rollback with a warning naming the OID reason."""
        repo = _make_repo(git_root)
        wt_dir = git_root / "wt-moved-branch"
        name = "moved-branch-run"
        args = _launch_args(
            name=name, command=[], worktree=str(wt_dir), worktree_base="main",
            worktree_repo=str(repo),
        )

        real_apply_cwd = agent_run._apply_launch_cwd

        def _apply_cwd_then_commit(a):
            real_apply_cwd(a)
            # A clean commit: the worktree itself has no untracked/modified
            # content, so `git worktree remove` (force=False) succeeds and
            # the branch-delete path is actually reached.
            (wt_dir / "workload-output.txt").write_text("work done\n")
            _git(wt_dir, "add", "workload-output.txt")
            _git(wt_dir, "commit", "-qm", "workload commit")

        monkeypatch.setattr(agent_run, "_apply_launch_cwd", _apply_cwd_then_commit)

        with pytest.raises(SystemExit, match="missing command"):
            agent_run.cmd_launch(args)

        assert not wt_dir.exists()  # the worktree itself is still rolled back
        branch_list = _git(repo, "branch", "--list", name)
        assert name in branch_list
        err = capsys.readouterr().err
        assert "no longer points at the commit this invocation created" in err

    def test_rollback_refuses_a_branch_checked_out_in_another_worktree(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """The checked-out guard: if the branch this invocation created is
        (somehow) also checked out elsewhere by the time rollback runs, the
        branch must survive -- `git branch -D` itself refuses this, and the
        pre-check must not paper over a refusal by reporting success."""
        repo = _make_repo(git_root)
        wt_dir = git_root / "wt-checked-out-elsewhere"
        name = "checked-out-run"
        args = _launch_args(
            name=name, command=[], worktree=str(wt_dir), worktree_base="main",
            worktree_repo=str(repo), worktree_branch="checked-out-elsewhere",
        )

        real_apply_cwd = agent_run._apply_launch_cwd
        elsewhere = git_root / "wt-elsewhere"

        def _apply_cwd_then_checkout_elsewhere(a):
            real_apply_cwd(a)
            # A second worktree attaches to the same branch this invocation
            # created, simulating another actor checking it out concurrently.
            _git(repo, "worktree", "add", "-q", "--force", str(elsewhere), "checked-out-elsewhere")

        monkeypatch.setattr(agent_run, "_apply_launch_cwd", _apply_cwd_then_checkout_elsewhere)

        with pytest.raises(SystemExit, match="missing command"):
            agent_run.cmd_launch(args)

        assert "checked-out-elsewhere" in _git(repo, "branch", "--list", "checked-out-elsewhere")
        assert "checked out in a worktree" in capsys.readouterr().err
        assert elsewhere.is_dir()

    def test_rollback_refuses_branch_deletion_when_worktree_enumeration_fails(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """`_worktree_branch_checked_out`'s docstring commits to treating a
        `worktree list` failure as assume-checked-out; this pins that the
        caller actually refuses in that case rather than deleting anyway."""
        repo = _make_repo(git_root)
        wt_dir = git_root / "wt-enum-fails"
        name = "enum-fails-run"
        args = _launch_args(
            name=name, command=[], worktree=str(wt_dir), worktree_base="main",
            worktree_repo=str(repo),
        )

        real_git_checked = agent_run._watch_run_git_checked

        def _fail_worktree_list(repo_arg, argv, **kw):
            if argv[:2] == ["worktree", "list"]:
                return agent_run._WatchGitOutcome(None, "simulated enumeration failure")
            return real_git_checked(repo_arg, argv, **kw)

        monkeypatch.setattr(agent_run, "_watch_run_git_checked", _fail_worktree_list)

        with pytest.raises(SystemExit, match="missing command"):
            agent_run.cmd_launch(args)

        assert name in _git(repo, "branch", "--list", name)
        assert "cannot confirm" in capsys.readouterr().err

    def test_rollback_branch_delete_itself_refuses_a_checked_out_branch(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """H2: even if the pre-check (`_worktree_branch_checked_out`) is
        stale or races a concurrent checkout and reports "not checked out",
        `git branch -D` must be the actual protection -- it refuses on its
        own when the branch really is checked out somewhere. This is what
        distinguishes rollback's delete from a plain `git update-ref -d`,
        which does not refuse and would silently discard the ref out from
        under the concurrent checkout."""
        repo = _make_repo(git_root)
        wt_dir = git_root / "wt-stale-precheck"
        name = "stale-precheck-run"
        args = _launch_args(
            name=name, command=[], worktree=str(wt_dir), worktree_base="main",
            worktree_repo=str(repo), worktree_branch="stale-precheck",
        )

        real_apply_cwd = agent_run._apply_launch_cwd
        elsewhere = git_root / "wt-stale-elsewhere"

        def _apply_cwd_then_checkout_elsewhere(a):
            real_apply_cwd(a)
            _git(repo, "worktree", "add", "-q", "--force", str(elsewhere), "stale-precheck")

        monkeypatch.setattr(agent_run, "_apply_launch_cwd", _apply_cwd_then_checkout_elsewhere)
        # The pre-check itself is stubbed to (falsely) report the branch as
        # unused, simulating a stale read racing the concurrent checkout
        # above: this isolates branch -D's own refusal as the sole guard.
        monkeypatch.setattr(agent_run, "_worktree_branch_checked_out", lambda repo, branch: (False, None))

        with pytest.raises(SystemExit, match="missing command"):
            agent_run.cmd_launch(args)

        assert "stale-precheck" in _git(repo, "branch", "--list", "stale-precheck")
        assert elsewhere.is_dir()

    def test_rollback_branch_check_does_not_prefix_match(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """A worktree checked out on branch 'feat-2' must not be read as
        satisfying a checked-out check for branch 'feat': the comparison is
        exact-match on the full ref, not a prefix match."""
        repo = _make_repo(git_root)
        _git(repo, "branch", "-f", "feat-2", "main")
        _git(repo, "worktree", "add", "-q", str(git_root / "wt-feat-2"), "feat-2")
        wt_dir = git_root / "wt-feat"
        name = "feat-run"
        args = _launch_args(
            name=name, command=[], worktree=str(wt_dir), worktree_base="main",
            worktree_repo=str(repo), worktree_branch="feat",
        )

        with pytest.raises(SystemExit, match="missing command"):
            agent_run.cmd_launch(args)

        # 'feat' was never checked out anywhere, so rollback must delete it
        # despite 'feat-2' being checked out concurrently.
        assert "feat" not in _git(repo, "branch", "--list", "feat").replace("feat-2", "")
        assert "feat-2" in _git(repo, "branch", "--list", "feat-2")

    def test_rollback_does_not_discard_uncommitted_work(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch
    ):
        """Rollback must refuse (not force-remove) a worktree that already has
        untracked content by the time launch fails: force=False is the only
        thing standing between rollback and discarding a runner's work."""
        repo = _make_repo(git_root)
        wt_dir = git_root / "wt-rollback-dirty"
        name = "rollback-dirty"
        args = _launch_args(
            name=name, command=[], worktree=str(wt_dir), worktree_base="main",
            worktree_repo=str(repo),
        )

        real_apply_cwd = agent_run._apply_launch_cwd

        def _apply_cwd_then_dirty(a):
            real_apply_cwd(a)
            # Simulate a runner having already written into the worktree
            # before the launch failure that triggers rollback: an untracked
            # file `git worktree remove` (without --force) must refuse to
            # discard.
            (wt_dir / "untracked-work.txt").write_text("must survive\n")

        monkeypatch.setattr(agent_run, "_apply_launch_cwd", _apply_cwd_then_dirty)

        with pytest.raises(SystemExit, match="missing command"):
            agent_run.cmd_launch(args)

        assert wt_dir.is_dir()
        assert (wt_dir / "untracked-work.txt").read_text() == "must survive\n"

    def test_rollback_refuses_a_replaced_non_worktree_directory(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """If DIR is replaced with an ordinary directory between creation and
        the failure that triggers rollback, rollback must refuse to touch it
        rather than calling git worktree remove against something that is no
        longer the worktree it created."""
        import shutil

        repo = _make_repo(git_root)
        wt_dir = git_root / "wt-rollback-replaced"
        name = "rollback-replaced"
        args = _launch_args(
            name=name, command=[], worktree=str(wt_dir), worktree_base="main",
            worktree_repo=str(repo),
        )

        real_apply_cwd = agent_run._apply_launch_cwd

        def _apply_cwd_then_replace(a):
            real_apply_cwd(a)
            shutil.rmtree(wt_dir)
            wt_dir.mkdir()
            (wt_dir / "replacement-marker.txt").write_text("not the worktree\n")

        monkeypatch.setattr(agent_run, "_apply_launch_cwd", _apply_cwd_then_replace)

        with pytest.raises(SystemExit, match="missing command"):
            agent_run.cmd_launch(args)

        assert (wt_dir / "replacement-marker.txt").read_text() == "not the worktree\n"
        assert "not a linked worktree" in capsys.readouterr().err

    def test_rollback_failure_does_not_mask_original_error(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        repo = _make_repo(git_root)
        wt_dir = git_root / "wt-rollback-fails"
        name = "rollback-fails"
        args = _launch_args(
            name=name, command=[], worktree=str(wt_dir), worktree_base="main",
            worktree_repo=str(repo),
        )
        monkeypatch.setattr(
            agent_run, "_worktree_remove", lambda info, path, force: "forced rollback failure"
        )

        with pytest.raises(SystemExit, match="missing command") as exc:
            agent_run.cmd_launch(args)

        assert "missing command" in str(exc.value)
        err = capsys.readouterr().err
        assert "forced rollback failure" in err
        # The forced-failing _worktree_remove never actually removed anything.
        assert wt_dir.is_dir()

    def test_rollback_raising_base_exception_does_not_mask_original_error(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """A second BaseException (e.g. another SIGINT) during rollback must
        not replace the original launch failure that triggered rollback."""
        repo = _make_repo(git_root)
        wt_dir = git_root / "wt-rollback-raises"
        name = "rollback-raises"
        args = _launch_args(
            name=name, command=[], worktree=str(wt_dir), worktree_base="main",
            worktree_repo=str(repo),
        )

        def _boom(created):
            raise KeyboardInterrupt("second interrupt during rollback")

        monkeypatch.setattr(agent_run, "_rollback_launch_worktree", _boom)

        with pytest.raises(SystemExit, match="missing command") as exc:
            agent_run.cmd_launch(args)

        assert "missing command" in str(exc.value)
        err = capsys.readouterr().err
        assert "second interrupt during rollback" in err
        # Rollback never ran, so the worktree it would have removed survives.
        assert wt_dir.is_dir()

    def test_fork_failure_after_state_dir_and_marker_exist_leaves_worktree_with_warning(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """Both existing pre-fork rollback tests fail at the very first
        statement of _cmd_launch_locked (command=[]), before the state dir,
        marker, status=starting, or args._worktree_process_started exist.
        This forces a failure *after* all of that is on disk: os.fork()
        itself raising (reached deterministically by monkeypatching it)
        happens after args._worktree_process_started is set, since that flag
        is the statement immediately preceding the call -- so this is a
        post-fork failure by definition, even though no child process
        actually came into being. The worktree and branch must survive with
        an actionable warning, and the state dir is left behind with
        status=failed naming the still-existing directory."""
        repo = _make_repo(git_root)
        wt_dir = git_root / "wt-rollback-post-publish"
        name = "rollback-post-publish"
        args = _launch_args(
            name=name, worktree=str(wt_dir), worktree_base="main", worktree_repo=str(repo),
        )

        def _fork_always_fails():
            raise OSError("simulated fork failure")

        monkeypatch.setattr(os, "fork", _fork_always_fails)

        with pytest.raises(SystemExit, match="failed to start agent"):
            agent_run.cmd_launch(args)

        state_dir = isolated_runs_root / name
        assert wt_dir.is_dir()
        assert agent_run._worktree_classify(wt_dir).kind == agent_run._WORKTREE_LINKED
        assert name in _git(repo, "branch", "--list", name)
        assert state_dir.is_dir()
        assert agent_run._read(state_dir / "status") == "failed"
        assert agent_run._read(state_dir / "cwd") == str(wt_dir.resolve())
        err = capsys.readouterr().err
        assert str(wt_dir) in err
        assert "git worktree remove" in err
        assert f"git branch -D {name}" in err

    def test_interrupt_after_readiness_published_never_rolls_back(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch
    ):
        """Once the runner has acknowledged readiness (status=running is
        published and a live runner owns the worktree), an interruption in
        the launcher's own remaining bookkeeping (pid parsing, print calls)
        must never trigger rollback: the worktree is a live run's cwd, not
        a leftover from a failed launch."""
        repo = _make_repo(git_root)
        wt_dir = git_root / "wt-interrupt-after-ready"
        name = "interrupt-after-ready"
        args = _launch_args(
            name=name, worktree=str(wt_dir), worktree_base="main", worktree_repo=str(repo),
        )

        real_print = print
        calls = {"n": 0}

        def _print_raises_once(*p_args, **p_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise KeyboardInterrupt("interrupted right after publication")
            return real_print(*p_args, **p_kwargs)

        monkeypatch.setattr("builtins.print", _print_raises_once)

        try:
            with pytest.raises(KeyboardInterrupt):
                agent_run.cmd_launch(args)
        finally:
            state_dir = isolated_runs_root / name
            _wait_terminal(state_dir)

        # Published before the interrupt: rollback must never have run.
        assert wt_dir.is_dir()
        assert agent_run._worktree_classify(wt_dir).kind == agent_run._WORKTREE_LINKED
        assert name in _git(repo, "branch", "--list", name)

    def test_failed_readiness_ack_survives_rollback_with_warning(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """A runner that reaches its own error handler and sends
        {"status": "error"} over the ack pipe is a post-fork failure: the
        fork that produced it already ran, so args._worktree_process_started
        was set before cmd_launch's except clause can ever see the
        SystemExit this raises. The worktree and branch this invocation
        created must survive, with an actionable warning telling the
        operator how to clean up by hand."""
        repo = _make_repo(git_root)
        wt_dir = git_root / "wt-failed-ack"
        name = "failed-ack-run"
        args = _launch_args(
            name=name, worktree=str(wt_dir), worktree_base="main", worktree_repo=str(repo),
        )

        def _run_oneshot_raises(*_a, **_k):
            raise RuntimeError("simulated setup failure")

        monkeypatch.setattr(agent_run, "_run_oneshot", _run_oneshot_raises)

        with pytest.raises(SystemExit, match="simulated setup failure"):
            agent_run.cmd_launch(args)

        assert wt_dir.is_dir()
        assert agent_run._worktree_classify(wt_dir).kind == agent_run._WORKTREE_LINKED
        assert name in _git(repo, "branch", "--list", name)
        err = capsys.readouterr().err
        assert str(wt_dir) in err
        assert "git worktree remove" in err
        assert f"git branch -D {name}" in err

    def test_post_fork_oserror_leaves_worktree_and_branch_intact_with_warning(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """A plain exception raised anywhere after os.fork() has returned --
        not just an interrupt -- must also be treated as post-fork: the
        worktree and branch survive with the actionable warning, and the
        original OSError still propagates unmasked."""
        repo = _make_repo(git_root)
        wt_dir = git_root / "wt-post-fork-oserror"
        name = "post-fork-oserror"
        args = _launch_args(
            name=name, worktree=str(wt_dir), worktree_base="main", worktree_repo=str(repo),
            command=[sys.executable, "-c", "import time; time.sleep(30)"],
        )

        real_fork = os.fork
        real_waitpid = os.waitpid
        main_pid = os.getpid()
        state = {"child_pid": None, "raised": False}

        def _fork_recording():
            pid = real_fork()
            if pid != 0 and state["child_pid"] is None:
                state["child_pid"] = pid
            return pid

        def _waitpid_raises_once(pid, options):
            if (
                os.getpid() == main_pid
                and pid == state["child_pid"]
                and not state["raised"]
            ):
                state["raised"] = True
                raise OSError("simulated post-fork bookkeeping failure")
            return real_waitpid(pid, options)

        monkeypatch.setattr(os, "fork", _fork_recording)
        monkeypatch.setattr(os, "waitpid", _waitpid_raises_once)

        try:
            with pytest.raises(OSError, match="simulated post-fork bookkeeping failure"):
                agent_run.cmd_launch(args)
        finally:
            state_dir = isolated_runs_root / name
            _wait_terminal(state_dir)

        assert wt_dir.is_dir()
        assert agent_run._worktree_classify(wt_dir).kind == agent_run._WORKTREE_LINKED
        assert name in _git(repo, "branch", "--list", name)
        err = capsys.readouterr().err
        assert "git worktree remove" in err
        assert f"git branch -D {name}" in err

    def test_interrupt_after_fork_leaves_worktree_intact(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """An interrupt landing at any point after os.fork() has returned
        must never trigger rollback: args._worktree_process_started was set
        as the last statement before that call, so nothing past this point
        can prove the forked process (or whatever it has itself forked) is
        dead. Interrupting os.waitpid's reap of the intermediate forker --
        the parent's very next statement once fork() returns -- exercises
        the earliest possible post-fork interruption point."""
        repo = _make_repo(git_root)
        wt_dir = git_root / "wt-interrupt-after-fork"
        name = "interrupt-after-fork"
        args = _launch_args(
            name=name, worktree=str(wt_dir), worktree_base="main", worktree_repo=str(repo),
            command=[sys.executable, "-c", "import time; time.sleep(30)"],
        )

        real_fork = os.fork
        real_waitpid = os.waitpid
        main_pid = os.getpid()
        state = {"child_pid": None, "raised": False}

        def _fork_recording():
            pid = real_fork()
            if pid != 0 and state["child_pid"] is None:
                state["child_pid"] = pid
            return pid

        def _waitpid_raises_once(pid, options):
            # Guarded on the *original* process id, not just the target
            # pid: this patch is inherited by every forked child's copy of
            # the os module, and a recycled pid number could otherwise
            # misfire this injected interrupt inside the runner itself.
            if (
                os.getpid() == main_pid
                and pid == state["child_pid"]
                and not state["raised"]
            ):
                state["raised"] = True
                raise KeyboardInterrupt("interrupted right after fork")
            return real_waitpid(pid, options)

        monkeypatch.setattr(os, "fork", _fork_recording)
        monkeypatch.setattr(os, "waitpid", _waitpid_raises_once)

        try:
            with pytest.raises(KeyboardInterrupt):
                agent_run.cmd_launch(args)
        finally:
            state_dir = isolated_runs_root / name
            _wait_terminal(state_dir)

        assert wt_dir.is_dir()
        assert agent_run._worktree_classify(wt_dir).kind == agent_run._WORKTREE_LINKED
        assert name in _git(repo, "branch", "--list", name)
        err = capsys.readouterr().err
        assert "git worktree remove" in err

    def test_post_fork_warning_write_failure_does_not_mask_original_exception(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """A write error while emitting the post-fork cleanup warning (a
        broken stderr, a second interrupt) must not replace the original
        launch failure that triggered rollback."""
        repo = _make_repo(git_root)
        wt_dir = git_root / "wt-warning-write-fails"
        name = "warning-write-fails"
        args = _launch_args(
            name=name, worktree=str(wt_dir), worktree_base="main", worktree_repo=str(repo),
        )

        def _fork_always_fails():
            raise OSError("simulated fork failure")

        monkeypatch.setattr(os, "fork", _fork_always_fails)

        real_print = print

        def _print_raises_for_warning(*p_args, **p_kwargs):
            if p_kwargs.get("file") is sys.stderr:
                raise BrokenPipeError("stderr write failed")
            return real_print(*p_args, **p_kwargs)

        monkeypatch.setattr("builtins.print", _print_raises_for_warning)

        with pytest.raises(SystemExit, match="failed to start agent"):
            agent_run.cmd_launch(args)

        assert wt_dir.is_dir()
        assert agent_run._worktree_classify(wt_dir).kind == agent_run._WORKTREE_LINKED
        assert name in _git(repo, "branch", "--list", name)

    def test_mint_cleanup_baseexception_with_live_child_refuses_rollback(
        self, isolated_runs_root, isolated_log_root, git_root, tmp_path, monkeypatch
    ):
        """A managed opencode launch starts the temporary mint process with
        the worktree as its cwd before args._worktree_process_started is
        set. If that process is still alive when a BaseException escapes
        _opencode_prefork_mint's cleanup, the worktree must survive: the
        mark was set before Popen, and cleanup failure must not clear it."""
        repo = _make_repo(git_root)
        wt_dir = git_root / "wt-mint-interrupt"
        name = "mint-interrupt-run"
        args = _launch_args(
            name=name, command=[], worktree=str(wt_dir), worktree_base="main",
            worktree_repo=str(repo), harness="opencode", prompt="hi",
        )

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake_opencode = bin_dir / "opencode"
        fake_opencode.write_text("#!/bin/sh\nexec sleep 60\n")
        fake_opencode.chmod(0o755)
        monkeypatch.setenv("PATH", str(bin_dir) + ":" + os.environ.get("PATH", ""))
        monkeypatch.setattr(agent_run, "_opencode_health_poll", lambda *a, **k: False)

        real_popen = subprocess.Popen
        state = {"proc": None}

        def _no_signal(*_a, **_k):
            # Cleanup's terminate()/kill() calls are neutered so the real
            # child is never actually signaled -- the test proves rollback
            # is refused without depending on whether cleanup's own
            # best-effort kill happens to succeed.
            pass

        def _wait_raises(*_a, **_k):
            raise KeyboardInterrupt("interrupt during mint cleanup")

        def _popen_tracking_opencode(argv, *a, **kw):
            proc = real_popen(argv, *a, **kw)
            if argv and argv[0] == "opencode":
                state["proc"] = proc
                proc.terminate = _no_signal
                proc.kill = _no_signal
                proc.wait = _wait_raises
            return proc

        monkeypatch.setattr(subprocess, "Popen", _popen_tracking_opencode)

        try:
            with pytest.raises(KeyboardInterrupt):
                agent_run.cmd_launch(args)

            proc = state["proc"]
            assert proc is not None, "the temporary opencode process must have started"
            assert proc.poll() is None, "the temporary process must still be alive"
            assert wt_dir.is_dir(), "worktree must survive an interrupted mint cleanup"
            assert agent_run._worktree_classify(wt_dir).kind == agent_run._WORKTREE_LINKED
            assert name in _git(repo, "branch", "--list", name)
        finally:
            proc = state["proc"]
            if proc is not None:
                real_popen.kill(proc)
                real_popen.wait(proc, timeout=5)
            _wait_terminal(isolated_runs_root / name)

    def test_invocation_cwd_outside_repo_without_worktree_repo_is_usage_error(
        self, isolated_runs_root, isolated_log_root, git_root, tmp_path, monkeypatch, capsys
    ):
        plain = tmp_path / "not-a-repo"
        plain.mkdir()
        monkeypatch.chdir(plain)
        args = _launch_args(
            name="outside-repo", worktree=str(git_root / "wt"), worktree_base="main",
        )

        with pytest.raises(SystemExit) as exc:
            agent_run.cmd_launch(args)

        assert exc.value.code == 2
        assert "not inside a git repository" in capsys.readouterr().err
        assert not (git_root / "wt").exists()

    def test_invalid_worktree_branch_name_is_rejected_before_any_mutation(
        self, isolated_runs_root, isolated_log_root, git_root
    ):
        """A git-illegal --worktree-branch (leading '-') must be rejected by
        git's own check-ref-format, not silently passed through to `git
        worktree add -b`, which would otherwise fail confusingly or, worse,
        be misinterpreted as another flag."""
        repo = _make_repo(git_root)
        wt_dir = git_root / "wt-bad-branch"
        args = _launch_args(
            name="bad-branch", worktree=str(wt_dir), worktree_base="main",
            worktree_branch="-not-a-valid-branch", worktree_repo=str(repo),
        )

        with pytest.raises(SystemExit, match="not a valid git branch name") as exc:
            agent_run.cmd_launch(args)

        assert _exit_status(exc.value.code) == 1
        assert not wt_dir.exists()
        assert _git(repo, "worktree", "list").count("\n") == 1  # only the main worktree

    def test_explicit_worktree_branch_overrides_run_name_default(
        self, isolated_runs_root, isolated_log_root, git_root
    ):
        repo = _make_repo(git_root)
        wt_dir = git_root / "wt-explicit-branch"
        args = _launch_args(
            name="run-name-not-branch", worktree=str(wt_dir), worktree_base="main",
            worktree_branch="explicit-branch-name", worktree_repo=str(repo),
        )

        rc = agent_run.cmd_launch(args)
        assert rc == 0
        _wait_terminal(isolated_runs_root / "run-name-not-branch")

        branches = _git(repo, "branch", "--list")
        assert "explicit-branch-name" in branches
        assert "run-name-not-branch" not in branches

    def test_nonexistent_worktree_repo_is_rejected(
        self, isolated_runs_root, isolated_log_root, git_root
    ):
        wt_dir = git_root / "wt-bad-repo"
        args = _launch_args(
            name="bad-repo", worktree=str(wt_dir), worktree_base="main",
            worktree_repo=str(git_root / "no-such-repo"),
        )

        with pytest.raises(SystemExit, match="does not exist or is not a directory") as exc:
            agent_run.cmd_launch(args)

        assert _exit_status(exc.value.code) == 1
        assert not wt_dir.exists()

    def test_worktree_creation_happens_inside_the_publication_lock(self):
        """`_create_launch_worktree(args)` must be called from within
        `cmd_launch`'s `with _worktree_publication_lock(...)` block: a
        freshly `git worktree add`-ed directory with no run state yet is
        exactly the half-created state that lock exists to hide from a
        concurrent reaper's exclusive-lock final scan. A real concurrency
        test would need two processes and would be flaky; this pins the
        ordering structurally instead, in the spirit of
        test_agent_run_watch.py's WATCH_CONTRACT_KEYS invariant."""
        lines = inspect.getsource(agent_run.cmd_launch).splitlines()
        with_line_idx = next(
            i for i, ln in enumerate(lines) if "_worktree_publication_lock(" in ln and ln.lstrip().startswith("with ")
        )
        with_indent = len(lines[with_line_idx]) - len(lines[with_line_idx].lstrip())
        # The block ends at the first later line, non-blank, indented at or
        # below the `with` statement's own indentation.
        block_end_idx = len(lines)
        for i in range(with_line_idx + 1, len(lines)):
            stripped = lines[i].strip()
            if not stripped:
                continue
            indent = len(lines[i]) - len(lines[i].lstrip())
            if indent <= with_indent:
                block_end_idx = i
                break
        block_lines = lines[with_line_idx + 1:block_end_idx]
        assert any("_create_launch_worktree(args)" in ln for ln in block_lines), (
            "_create_launch_worktree must be called inside the "
            "_worktree_publication_lock block"
        )


