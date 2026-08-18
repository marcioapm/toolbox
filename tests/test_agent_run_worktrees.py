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
        """st_dev/st_ino recorded at collection no longer name the same
        directory: a different tree now occupies the candidate path."""
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


class TestReapWorktreeActivityWalk:
    """The recursive mtime walk must fail closed when a subtree is unreadable:
    a recent file hidden behind a permission error must not allow the worktree
    to appear idle."""

    def test_walk_error_fails_closed_in_strict_mode(
        self, isolated_runs_root, isolated_log_root, git_root, monkeypatch, capsys
    ):
        """A scandir failure during the activity walk causes _WALK_INCOMPLETE to
        be returned, so the activity gate reports the worktree as unreadable
        rather than falling back to a stale timestamp."""
        repo = _make_repo(git_root)
        wt = _add_worktree(repo, git_root / "wt", "feature")
        # Add a subdirectory so that the walk recurses: the failure will occur
        # when scanning the subdirectory, after the top-level scan succeeds and
        # records its (old) mtime.
        sub = wt / "subdir"
        sub.mkdir()
        _make_state_run(isolated_runs_root, isolated_log_root, "r1", cwd=wt, age_hours=1000)

        real_scandir = os.scandir
        call_count: dict = {"n": 0}

        def scandir_that_fails_on_subdir(path):
            # Fail when scanning the subdirectory: the top-level scan succeeds
            # (recording the old mtime), but the subdir is unreadable.
            if str(path).startswith(str(sub)):
                raise OSError("simulated permission denied")
            return real_scandir(path)

        monkeypatch.setattr(os, "scandir", scandir_that_fails_on_subdir)

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


class TestDuWorktreeArithmetic:
    """TOTAL counts every byte exactly once, whatever nests inside what."""

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
