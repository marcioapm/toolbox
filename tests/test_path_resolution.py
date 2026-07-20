"""Unit tests for agent-run's split-storage path-resolution helpers:
state dir (ephemeral, /tmp) vs. log dir (persistent, /var/tmp), the
old-layout fallback, and the age-based log prune."""
from __future__ import annotations

import os
import time

from toolbox import agent_run


class TestStateAndLogDirs:
    def test_state_dir_under_state_root(self, isolated_runs_root):
        d = agent_run._state_dir("foo")
        assert d == agent_run.STATE_ROOT / "foo"

    def test_log_dir_under_log_root(self, isolated_runs_root):
        d = agent_run._log_dir("foo")
        assert d == agent_run.LOG_ROOT / "foo"

    def test_state_and_log_roots_differ(self, isolated_runs_root):
        assert agent_run.STATE_ROOT != agent_run.LOG_ROOT


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
