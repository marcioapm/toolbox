"""Unit tests for agent-run's split-storage path-resolution helpers:
state dir (ephemeral, /tmp) vs. log dir (persistent, /var/tmp), the
old-layout fallback, and the age-based log prune."""
from __future__ import annotations

import argparse
import fcntl
import os
import time

import pytest

from toolbox import agent_run


class TestRunNameSafety:
    @pytest.mark.parametrize(
        "name",
        ["", ".", "..", "../escape", "nested/run", r"nested\\run", "/tmp/escape", "-flag", "bad\x00name", "bad\nname"],
    )
    def test_validate_run_name_rejects_unsafe_names(self, name):
        with pytest.raises(SystemExit, match="invalid run name"):
            agent_run._validate_run_name(name)

    @pytest.mark.parametrize("name", ["run", "run.v2", "Run_2-test", "1.2.3"])
    def test_validate_run_name_accepts_safe_dotted_names(self, name):
        assert agent_run._validate_run_name(name) == name

    @pytest.mark.parametrize(
        ("command", "args"),
        [
            (agent_run.cmd_status, argparse.Namespace(name="..")),
            (agent_run.cmd_logs, argparse.Namespace(name="..", tail=1, head=None, plain=False, clean=False)),
            (agent_run.cmd_tail, argparse.Namespace(name="..")),
            (
                agent_run.cmd_steer,
                argparse.Namespace(name="..", message=["x"], raw=False, esc=False),
            ),
            (agent_run.cmd_kill, argparse.Namespace(name="..", signal="TERM")),
            (
                agent_run.cmd_launch,
                argparse.Namespace(
                    name="..",
                    command=["true"],
                    interactive=False,
                    prompt_file=None,
                ),
            ),
        ],
    )
    def test_every_named_command_validates_before_path_use(self, command, args):
        with pytest.raises(SystemExit, match="invalid run name"):
            command(args)

    def test_safe_rmtree_removes_direct_child_only(self, tmp_path):
        root = tmp_path / "root"
        child = root / "run"
        child.mkdir(parents=True)

        agent_run._safe_rmtree(child, root)

        assert root.is_dir()
        assert not child.exists()

    @pytest.mark.parametrize("relative", [".", "..", "nested/run"])
    def test_safe_rmtree_refuses_root_parent_and_nested_paths(self, tmp_path, relative):
        root = tmp_path / "root"
        root.mkdir()
        candidate = root / relative
        if relative == "nested/run":
            candidate.mkdir(parents=True)

        with pytest.raises(SystemExit, match="refusing to delete"):
            agent_run._safe_rmtree(candidate, root)

        assert root.is_dir()
    def test_safe_rmtree_refuses_top_level_sibling_symlink(self, tmp_path):
        root = tmp_path / "root"
        target = root / "run-b"
        target.mkdir(parents=True)
        (target / "data").write_text("preserve")
        link = root / "run-a"
        link.symlink_to(target, target_is_directory=True)

        with pytest.raises(SystemExit, match="not a real directory"):
            agent_run._safe_rmtree(link, root)

        assert link.is_symlink()
        assert (target / "data").read_text() == "preserve"

    def test_safe_rmtree_refuses_top_level_outside_symlink(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "data").write_text("preserve")
        link = root / "run"
        link.symlink_to(outside, target_is_directory=True)

        with pytest.raises(SystemExit, match="not a real directory"):
            agent_run._safe_rmtree(link, root)

        assert (outside / "data").read_text() == "preserve"


class TestStateAndLogDirs:
    def test_state_dir_under_state_root(self, isolated_runs_root):
        d = agent_run._state_dir("foo")
        assert d == agent_run.STATE_ROOT / "foo"

    def test_log_dir_under_log_root(self, isolated_runs_root):
        d = agent_run._log_dir("foo")
        assert d == agent_run.LOG_ROOT / "foo"

    def test_state_and_log_roots_differ(self, isolated_runs_root):
        assert agent_run.STATE_ROOT != agent_run.LOG_ROOT


class TestCmdList:
    def test_ignores_internal_lock_directory(self, isolated_runs_root, capsys):
        (isolated_runs_root / ".locks").mkdir()
        run = isolated_runs_root / "actual-run"
        run.mkdir()
        (run / "status").write_text("running\n")
        # Seed a live pid so _opportunistic_heal (which reconciles a running
        # run whose pid is dead/missing to "died") leaves this run as running;
        # this test is about ignoring the internal .locks dir, not status.
        (run / "pid").write_text(f"{os.getpid()}\n")

        assert agent_run.cmd_list(argparse.Namespace()) == 0

        output = capsys.readouterr().out
        assert "actual-run: status=running" in output
        assert ".locks" not in output

    def test_only_internal_lock_directory_is_not_a_run(self, isolated_runs_root, capsys):
        (isolated_runs_root / ".locks").mkdir()

        assert agent_run.cmd_list(argparse.Namespace()) == 0

        assert "  (none)" in capsys.readouterr().out


class TestLogFileFor:
    def test_prefers_new_layout(self, isolated_runs_root):
        name = "run1"
        log_d = agent_run.LOG_ROOT / name
        log_d.mkdir(parents=True)
        (log_d / "log").write_bytes(b"new layout\n")
        assert agent_run._log_file_for(name) == log_d / "log"

    def test_falls_back_to_old_layout(self, isolated_runs_root):
        name = "run2"
        state_d = agent_run.STATE_ROOT / name
        state_d.mkdir()
        (state_d / "log").write_bytes(b"old layout\n")
        assert agent_run._log_file_for(name) == state_d / "log"

    def test_new_layout_wins_over_old_if_both_present(self, isolated_runs_root):
        name = "run3"
        state_d = agent_run.STATE_ROOT / name
        state_d.mkdir()
        (state_d / "log").write_bytes(b"old\n")
        log_d = agent_run.LOG_ROOT / name
        log_d.mkdir(parents=True)
        (log_d / "log").write_bytes(b"new\n")
        assert agent_run._log_file_for(name) == log_d / "log"

    def test_missing_returns_none(self, isolated_runs_root):
        assert agent_run._log_file_for("does-not-exist") is None


class TestKnown:
    def test_known_via_state_only(self, isolated_runs_root):
        (agent_run.STATE_ROOT / "s").mkdir()
        assert agent_run._known("s") is True

    def test_known_via_log_only(self, isolated_runs_root):
        (agent_run.LOG_ROOT / "l").mkdir()
        assert agent_run._known("l") is True

    def test_unknown(self, isolated_runs_root):
        assert agent_run._known("nope") is False


class TestRequireHelpers:
    def test_require_state_exits_when_missing(self, isolated_runs_root):
        import pytest

        with pytest.raises(SystemExit):
            agent_run._require_state("missing")

    def test_require_log_exits_when_missing(self, isolated_runs_root):
        import pytest

        with pytest.raises(SystemExit):
            agent_run._require_log("missing")

    def test_require_log_finds_new_layout(self, isolated_runs_root):
        log_d = agent_run.LOG_ROOT / "r"
        log_d.mkdir(parents=True)
        (log_d / "log").write_bytes(b"hi\n")
        assert agent_run._require_log("r") == log_d / "log"


class TestPruneOldLogs:
    def test_prunes_dirs_older_than_cutoff(self, isolated_runs_root):
        old_dir = agent_run.LOG_ROOT / "ancient"
        old_dir.mkdir(parents=True)
        old_log = old_dir / "log"
        old_log.write_text("stale\n")
        old_time = time.time() - 30 * 86400
        os.utime(old_log, (old_time, old_time))

        fresh_dir = agent_run.LOG_ROOT / "fresh"
        fresh_dir.mkdir(parents=True)
        (fresh_dir / "log").write_text("recent\n")

        agent_run._prune_old_logs(max_age_days=21)

        assert not old_dir.exists()
        assert fresh_dir.exists()

    def test_noop_when_log_root_missing(self, tmp_path, monkeypatch):
        missing = tmp_path / "does-not-exist"
        monkeypatch.setattr(agent_run, "LOG_ROOT", missing)
        agent_run._prune_old_logs()  # should not raise

    def test_skips_live_old_log_dir(self, isolated_runs_root):
        old_dir = agent_run.LOG_ROOT / "live"
        old_dir.mkdir(parents=True)
        old_log = old_dir / "log"
        old_log.write_text("quiet but live\n")
        state = agent_run.STATE_ROOT / "live"
        state.mkdir()
        (state / "status").write_text("running\n")
        old_time = time.time() - 30 * 86400
        os.utime(old_log, (old_time, old_time))

        agent_run._prune_old_logs(max_age_days=21)

        assert old_dir.exists()

    def test_skips_replaced_prune_candidate(self, isolated_runs_root, monkeypatch):
        old_dir = agent_run.LOG_ROOT / "replace"
        old_dir.mkdir(parents=True)
        old_log = old_dir / "log"
        old_log.write_text("old\n")
        old_time = time.time() - 30 * 86400
        os.utime(old_log, (old_time, old_time))
        original_lock = agent_run._launch_lock

        @agent_run.contextmanager
        def replace_while_locked(name):
            with original_lock(name) as fd:
                old_dir.rename(agent_run.LOG_ROOT / "old-replaced")
                replacement = agent_run.LOG_ROOT / "replace"
                replacement.mkdir()
                (replacement / "log").write_text("new\n")
                yield fd

        monkeypatch.setattr(agent_run, "_launch_lock", replace_while_locked)
        agent_run._prune_old_logs(max_age_days=21)

        assert (agent_run.LOG_ROOT / "replace" / "log").read_text() == "new\n"

    def test_never_prunes_per_name_lock_files(self, isolated_runs_root):
        lock_dir = agent_run.STATE_ROOT / ".locks"
        lock_dir.mkdir()
        old_lock = lock_dir / "old.lock"
        old_lock.write_text("")
        old_time = time.time() - 30 * 86400
        os.utime(old_lock, (old_time, old_time))

        agent_run._prune_old_locks(max_age_days=21)

        assert old_lock.exists()

    def test_keeps_stale_lock_held_by_another_process(self, isolated_runs_root):
        lock_dir = agent_run.STATE_ROOT / ".locks"
        lock_dir.mkdir()
        lock = lock_dir / "active.lock"
        lock.write_text("")
        old_time = time.time() - 30 * 86400
        os.utime(lock, (old_time, old_time))
        fd = os.open(lock, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            agent_run._prune_old_locks(max_age_days=21)
            assert lock.exists()
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def test_prune_locks_noop_when_directory_missing(self, isolated_runs_root):
        agent_run._prune_old_locks()
