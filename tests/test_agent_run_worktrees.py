"""Tests for linked-git-worktree accounting and garbage collection:

- `_worktree_classify`: linked worktree, main worktree, bare repo, non-git
  directory, missing path, worktree subdirectory, git failures.
- `agent-run du`: separate WORKTREE column, realpath
  deduplication, TOTAL reconciliation, --json shape, read-only guarantee.
- `agent-run reap --include-worktrees`: age threshold, dirty/unpushed
  refusals, shared-cwd single removal, symlink and live-run guards,
  --dry-run prediction accuracy.

Uses the tmp-dir + monkeypatched STATE_ROOT/LOG_ROOT pattern shared with
tests/test_agent_run_reap.py and tests/conftest.py.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
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

    def test_git_failure_never_removes(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)
        monkeypatch.setattr(
            agent_run, "_watch_run_git_checked",
            lambda *a, **k: agent_run._WatchGitOutcome(None, "git_missing"),
        )

        agent_run.cmd_reap(_reap_args())

        out = capsys.readouterr().out
        assert wt.is_dir()
        assert "worktrees_removed=0" in out
        assert "worktrees_skipped=1" in out

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
        assert "not on any remote" in out
        assert "worktrees_skipped=1" in out

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
