"""Tests for toolbox.thread_state: registry atomicity/corruption, rendering
(icon/color/title truncation), the PR-scan regex, gh JSON parsing, CI rollup
derivation, gh-failure staleness, auto status derivation, and account
switching. All gh interaction is exercised through an injected fake runner —
these tests never touch the network or a real `gh` binary."""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import threading
import time

import pytest

from toolbox import thread_state as ts


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry_path(tmp_path):
    return tmp_path / "sub" / "registry.json"


def run_cli(argv, registry_path, capsys):
    """Invoke thread_state.main with --registry inserted after the
    subcommand's positional args are still parseable — argparse accepts
    global-ish options anywhere after the subcommand name since each
    subparser owns its own --registry."""
    full = [argv[0], *argv[1:], "--registry", str(registry_path)]
    code = ts.main(full)
    out = capsys.readouterr()
    return code, out.out, out.err


# ---------------------------------------------------------------------------
# registry: atomic writes, locking, corruption
# ---------------------------------------------------------------------------

class TestRegistryBasics:
    def test_missing_file_reads_as_empty(self, registry_path):
        data = ts._parse_registry(registry_path)
        assert data == {"schemaVersion": 1, "accounts": {}, "threads": {}}

    def test_write_then_read_roundtrip(self, registry_path):
        data = ts._empty_registry()
        data["threads"]["1"] = {"title": "x"}
        ts._write_registry(registry_path, data)
        assert ts._parse_registry(registry_path) == data

    def test_write_creates_parent_dirs(self, registry_path):
        assert not registry_path.parent.exists()
        ts._write_registry(registry_path, ts._empty_registry())
        assert registry_path.exists()

    def test_write_is_atomic_no_partial_file_left_on_crash(self, registry_path, monkeypatch):
        ts._write_registry(registry_path, ts._empty_registry())
        original = registry_path.read_text()

        def _boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(os, "fsync", _boom)
        with pytest.raises(OSError):
            ts._write_registry(registry_path, {"schemaVersion": 1, "accounts": {}, "threads": {"x": 1}})
        # Original file untouched; no stray temp file left behind.
        assert registry_path.read_text() == original
        leftovers = [p for p in registry_path.parent.iterdir() if p.name != registry_path.name]
        assert leftovers == []


class TestCorruptRegistry:
    def test_corrupt_json_backed_up_and_raises(self, registry_path):
        registry_path.parent.mkdir(parents=True)
        registry_path.write_text("{not json")
        with pytest.raises(ts.RegistryError):
            ts._parse_registry(registry_path)
        backups = list(registry_path.parent.glob("registry.json.bad-*"))
        assert len(backups) == 1
        assert backups[0].read_text() == "{not json"
        # Original left untouched, not overwritten.
        assert registry_path.read_text() == "{not json"

    def test_wrong_shape_treated_as_corrupt(self, registry_path):
        registry_path.parent.mkdir(parents=True)
        registry_path.write_text(json.dumps(["not", "a", "dict"]))
        with pytest.raises(ts.RegistryError):
            ts._parse_registry(registry_path)

    def test_cli_never_overwrites_corrupt_registry(self, registry_path, capsys):
        registry_path.parent.mkdir(parents=True)
        registry_path.write_text("{not json")
        code, out, err = run_cli(["list"], registry_path, capsys)
        assert code == 3
        assert "corrupt" in err
        # File on disk is exactly what it was before.
        assert registry_path.read_text() == "{not json"


class TestLocking:
    def test_locked_registry_holds_exclusive_lock_during_block(self, registry_path):
        ts._write_registry(registry_path, ts._empty_registry())
        entered = threading.Event()
        release = threading.Event()

        def holder():
            with ts._locked_registry(registry_path) as data:
                data["threads"]["1"] = {"v": 1}
                entered.set()
                release.wait(timeout=5)

        t = threading.Thread(target=holder)
        t.start()
        assert entered.wait(timeout=5)

        # A second exclusive lock attempt should block until released.
        fd = os.open(registry_path, os.O_RDWR)
        try:
            got_immediately = True
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                got_immediately = False
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
            assert got_immediately is False
        finally:
            os.close(fd)
            release.set()
            t.join(timeout=5)

    def test_concurrent_writers_do_not_clobber_each_other(self, registry_path):
        ts._write_registry(registry_path, ts._empty_registry())

        def add_thread(n):
            with ts._locked_registry(registry_path) as data:
                data["threads"][str(n)] = {"v": n}

        threads = [threading.Thread(target=add_thread, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        data = ts._parse_registry(registry_path)
        assert set(data["threads"].keys()) == {str(i) for i in range(10)}

    def test_abort_inside_block_does_not_persist_changes(self, registry_path):
        ts._write_registry(registry_path, ts._empty_registry())
        with pytest.raises(ts._Abort):
            with ts._locked_registry(registry_path) as data:
                data["threads"]["x"] = {"v": 1}
                raise ts._Abort(1, "nope")
        data = ts._parse_registry(registry_path)
        assert data["threads"] == {}


# ---------------------------------------------------------------------------
# status / emoji rendering
# ---------------------------------------------------------------------------

class TestStatusEmoji:
    @pytest.mark.parametrize("status,icon", list(ts.STATUS_EMOJI.items()))
    def test_icon_matches_status_when_ci_not_failing(self, status, icon):
        entry = {"status": status, "prState": {"ci": "SUCCESS"}}
        assert ts._effective_icon(entry) == icon

    def test_ci_failure_overrides_active_status_to_red(self):
        entry = {"status": "active", "prState": {"ci": "FAILURE"}}
        assert ts._effective_icon(entry) == ts.STATUS_EMOJI["blocked"]

    @pytest.mark.parametrize("status", ["merged", "closed", "done"])
    def test_ci_failure_does_not_override_terminal_statuses(self, status):
        entry = {"status": status, "prState": {"ci": "FAILURE"}}
        assert ts._effective_icon(entry) == ts.STATUS_EMOJI[status]

    def test_no_prstate_uses_plain_status_icon(self):
        entry = {"status": "review"}
        assert ts._effective_icon(entry) == ts.STATUS_EMOJI["review"]


# ---------------------------------------------------------------------------
# title rendering + truncation
# ---------------------------------------------------------------------------

class TestRenderTitle:
    def test_basic_format(self):
        entry = {"title": "mSPRT", "status": "review", "pr": 4694, "prState": {}}
        assert ts.render_title(entry) == "🔵 mSPRT · PR#4694 · review"

    def test_no_pr_omits_pr_segment(self):
        entry = {"title": "Widget", "status": "active", "pr": None, "prState": {}}
        assert ts.render_title(entry) == "🟢 Widget · active"

    def test_title_never_exceeds_100_chars(self):
        entry = {"title": "x" * 200, "status": "review", "pr": 999999, "prState": {}}
        out = ts.render_title(entry)
        assert len(out) <= ts.THREAD_TITLE_MAX

    def test_long_title_truncated_with_ellipsis_suffix_preserved(self):
        entry = {"title": "y" * 200, "status": "blocked", "pr": 42, "prState": {}}
        out = ts.render_title(entry)
        assert out.endswith("· PR#42 · blocked")
        assert "…" in out

    def test_short_title_not_truncated(self):
        entry = {"title": "short", "status": "active", "pr": None, "prState": {}}
        out = ts.render_title(entry)
        assert "…" not in out


# ---------------------------------------------------------------------------
# PR-reference scanning
# ---------------------------------------------------------------------------

class TestScanText:
    def test_matches_full_url(self):
        text = "see https://github.com/acme/widgets/pull/42"
        matches = ts.scan_text(text, known_repo=None)
        assert matches == [{"repo": "acme/widgets", "pr": 42, "kind": "url", "pos": 4}]

    def test_bare_ref_ignored_without_known_repo(self):
        text = "see #42"
        matches = ts.scan_text(text, known_repo=None)
        assert matches == []

    def test_bare_ref_matched_with_known_repo(self):
        text = "see #42"
        matches = ts.scan_text(text, known_repo="acme/widgets")
        assert matches == [{"repo": "acme/widgets", "pr": 42, "kind": "bare", "pos": 4}]

    def test_dedupe_and_count(self):
        text = "https://github.com/a/b/pull/1 then #1 then https://github.com/a/b/pull/1"
        matches = ts.scan_text(text, known_repo="a/b")
        summary = ts.summarize_matches(matches)
        assert len(summary) == 1
        assert summary[0]["repo"] == "a/b"
        assert summary[0]["pr"] == 1
        assert summary[0]["count"] == 3
        assert set(summary[0]["kinds"]) == {"url", "bare"}

    def test_ordering_most_recent_last(self):
        text = "https://github.com/a/b/pull/1 ... https://github.com/a/b/pull/2"
        summary = ts.summarize_matches(ts.scan_text(text, known_repo="a/b"))
        assert [g["pr"] for g in summary] == [1, 2]

    def test_best_match_prefers_url_over_bare(self):
        text = "#5 then https://github.com/a/b/pull/5"
        matches = ts.scan_text(text, known_repo="a/b")
        bm = ts.best_match(matches)
        assert bm["kind"] == "url"

    def test_best_match_prefers_last_occurrence_among_urls(self):
        text = "https://github.com/a/b/pull/1 https://github.com/a/b/pull/2"
        matches = ts.scan_text(text, known_repo=None)
        bm = ts.best_match(matches)
        assert bm["pr"] == 2

    def test_best_match_none_when_no_matches(self):
        assert ts.best_match([]) is None

    def test_different_repos_not_merged(self):
        text = "https://github.com/a/b/pull/1 https://github.com/c/d/pull/1"
        summary = ts.summarize_matches(ts.scan_text(text, known_repo=None))
        assert len(summary) == 2


# ---------------------------------------------------------------------------
# gh JSON parsing / CI rollup derivation / account switching
# ---------------------------------------------------------------------------

class TestDeriveCi:
    def test_no_checks_is_none(self):
        assert ts._derive_ci([]) == "NONE"
        assert ts._derive_ci(None) == "NONE"

    def test_any_failure_wins(self):
        rollup = [{"conclusion": "SUCCESS"}, {"conclusion": "FAILURE"}]
        assert ts._derive_ci(rollup) == "FAILURE"

    def test_error_state_counts_as_failure(self):
        rollup = [{"state": "ERROR"}]
        assert ts._derive_ci(rollup) == "FAILURE"

    def test_pending_when_no_failure_but_in_progress(self):
        rollup = [{"conclusion": "SUCCESS"}, {"status": "IN_PROGRESS"}]
        assert ts._derive_ci(rollup) == "PENDING"

    def test_success_when_all_pass(self):
        rollup = [{"conclusion": "SUCCESS"}, {"conclusion": "SUCCESS"}]
        assert ts._derive_ci(rollup) == "SUCCESS"

    def test_failure_takes_priority_over_pending(self):
        rollup = [{"status": "QUEUED"}, {"conclusion": "FAILURE"}]
        assert ts._derive_ci(rollup) == "FAILURE"


class TestAccountSwitching:
    def test_known_owner_maps_to_account(self):
        accounts = dict(ts.DEFAULT_ACCOUNTS)
        assert ts._gh_account_for("absmartly/foo", accounts) == "marcio-absmartly"
        assert ts._gh_account_for("marcioapm/bar", accounts) == "marcioapm"

    def test_unknown_owner_returns_none(self):
        assert ts._gh_account_for("someoneelse/repo", ts.DEFAULT_ACCOUNTS) is None

    def test_custom_registry_accounts_extend_defaults(self):
        accounts = {**ts.DEFAULT_ACCOUNTS, "customorg": "custom-user"}
        assert ts._gh_account_for("customorg/repo", accounts) == "custom-user"

    def test_fetch_pr_state_switches_account_before_view(self):
        calls = []

        def fake_runner(cmd, timeout):
            calls.append(cmd)
            if cmd[:2] == ["gh", "auth"]:
                return subprocess.CompletedProcess(cmd, 0, "", "")
            payload = {"state": "OPEN", "title": "t", "url": "u", "mergedAt": None, "statusCheckRollup": []}
            return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

        pr_state, err = ts.fetch_pr_state("absmartly/foo", 1, ts.DEFAULT_ACCOUNTS, 20, fake_runner)
        assert err is None
        assert calls[0] == ["gh", "auth", "switch", "--user", "marcio-absmartly"]
        assert calls[1][0:3] == ["gh", "pr", "view"]

    def test_fetch_pr_state_skips_switch_for_unknown_owner(self):
        calls = []

        def fake_runner(cmd, timeout):
            calls.append(cmd)
            payload = {"state": "OPEN", "title": "t", "url": "u", "mergedAt": None, "statusCheckRollup": []}
            return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

        ts.fetch_pr_state("unknown/foo", 1, ts.DEFAULT_ACCOUNTS, 20, fake_runner)
        assert len(calls) == 1
        assert calls[0][0:3] == ["gh", "pr", "view"]


class TestFetchPrStateParsing:
    def test_parses_full_payload(self):
        def fake_runner(cmd, timeout):
            payload = {
                "state": "OPEN",
                "title": "Fix thing",
                "url": "https://github.com/a/b/pull/9",
                "mergedAt": None,
                "statusCheckRollup": [{"conclusion": "SUCCESS"}],
            }
            return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

        pr_state, err = ts.fetch_pr_state("a/b", 9, {}, 20, fake_runner)
        assert err is None
        assert pr_state["state"] == "OPEN"
        assert pr_state["ci"] == "SUCCESS"
        assert pr_state["title"] == "Fix thing"
        assert pr_state["stale"] is False
        assert "checkedAt" in pr_state

    def test_nonzero_exit_returns_error(self):
        def fake_runner(cmd, timeout):
            return subprocess.CompletedProcess(cmd, 1, "", "not found")

        pr_state, err = ts.fetch_pr_state("a/b", 9, {}, 20, fake_runner)
        assert pr_state is None
        assert "not found" in err

    def test_invalid_json_returns_error(self):
        def fake_runner(cmd, timeout):
            return subprocess.CompletedProcess(cmd, 0, "not json", "")

        pr_state, err = ts.fetch_pr_state("a/b", 9, {}, 20, fake_runner)
        assert pr_state is None
        assert "invalid JSON" in err

    def test_timeout_returns_error_never_raises(self):
        def fake_runner(cmd, timeout):
            raise subprocess.TimeoutExpired(cmd, timeout)

        pr_state, err = ts.fetch_pr_state("a/b", 9, {}, 20, fake_runner)
        assert pr_state is None
        assert "timed out" in err

    def test_oserror_returns_error_never_raises(self):
        def fake_runner(cmd, timeout):
            raise OSError("gh not found")

        pr_state, err = ts.fetch_pr_state("a/b", 9, {}, 20, fake_runner)
        assert pr_state is None
        assert "gh not found" in err


# ---------------------------------------------------------------------------
# auto status derivation
# ---------------------------------------------------------------------------

class TestAutoStatus:
    def test_merged_sets_status_merged(self):
        entry = {"status": "review"}
        ts._apply_auto_status(entry, {"state": "MERGED"})
        assert entry["status"] == "merged"

    def test_closed_sets_status_closed(self):
        entry = {"status": "review"}
        ts._apply_auto_status(entry, {"state": "CLOSED"})
        assert entry["status"] == "closed"

    def test_open_leaves_status_untouched(self):
        entry = {"status": "review"}
        ts._apply_auto_status(entry, {"state": "OPEN"})
        assert entry["status"] == "review"

    def test_open_does_not_downgrade_manual_blocked(self):
        entry = {"status": "blocked"}
        ts._apply_auto_status(entry, {"state": "OPEN"})
        assert entry["status"] == "blocked"

    def test_open_does_not_downgrade_manual_paused(self):
        entry = {"status": "paused"}
        ts._apply_auto_status(entry, {"state": "OPEN"})
        assert entry["status"] == "paused"

    def test_merged_overrides_manual_blocked(self):
        # A merge is real signal even if a human had marked it blocked earlier.
        entry = {"status": "blocked"}
        ts._apply_auto_status(entry, {"state": "MERGED"})
        assert entry["status"] == "merged"


# ---------------------------------------------------------------------------
# CLI: refresh command — gh failure -> stale not crash
# ---------------------------------------------------------------------------

class TestRefreshCli:
    def _seed(self, registry_path, thread_id="1", repo="absmartly/foo", pr=5, status="review"):
        data = ts._empty_registry()
        data["threads"][thread_id] = {
            "threadId": thread_id,
            "title": "T",
            "description": "",
            "repo": repo,
            "pr": pr,
            "branch": None,
            "status": status,
            "tags": [],
            "links": [],
            "notes": "",
            "prState": None,
            "createdAt": "2026-01-01T00:00:00Z",
            "updatedAt": "2026-01-01T00:00:00Z",
        }
        ts._write_registry(registry_path, data)

    def test_gh_failure_keeps_cache_and_marks_stale_single_thread_nonzero_exit(
        self, registry_path, capsys, monkeypatch
    ):
        self._seed(registry_path)
        data = ts._parse_registry(registry_path)
        data["threads"]["1"]["prState"] = {"state": "OPEN", "ci": "SUCCESS", "title": "old", "url": "u", "mergedAt": None, "checkedAt": "2026-01-01T00:00:00Z", "stale": False}
        ts._write_registry(registry_path, data)

        def failing_runner(cmd, timeout):
            if cmd[:2] == ["gh", "auth"]:
                return subprocess.CompletedProcess(cmd, 0, "", "")
            raise subprocess.TimeoutExpired(cmd, timeout)

        monkeypatch.setattr(ts, "_run_command", failing_runner)
        code, out, err = run_cli(["refresh", "1"], registry_path, capsys)
        assert code == 1  # explicit single-thread refresh: nonzero on error

        data = ts._parse_registry(registry_path)
        pr_state = data["threads"]["1"]["prState"]
        assert pr_state["state"] == "OPEN"  # cached value preserved
        assert pr_state["stale"] is True
        assert "error" in pr_state

    def test_gh_failure_with_all_exits_zero_reports_per_thread(self, registry_path, capsys, monkeypatch):
        self._seed(registry_path)

        def failing_runner(cmd, timeout):
            if cmd[:2] == ["gh", "auth"]:
                return subprocess.CompletedProcess(cmd, 0, "", "")
            raise subprocess.TimeoutExpired(cmd, timeout)

        monkeypatch.setattr(ts, "_run_command", failing_runner)
        code, out, err = run_cli(["refresh", "--all"], registry_path, capsys)
        assert code == 0

    def test_successful_refresh_updates_prstate_and_derives_status(self, registry_path, capsys, monkeypatch):
        self._seed(registry_path, status="review")

        def ok_runner(cmd, timeout):
            if cmd[:2] == ["gh", "auth"]:
                return subprocess.CompletedProcess(cmd, 0, "", "")
            payload = {"state": "MERGED", "title": "t", "url": "u", "mergedAt": "2026-01-02T00:00:00Z", "statusCheckRollup": [{"conclusion": "SUCCESS"}]}
            return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

        monkeypatch.setattr(ts, "_run_command", ok_runner)
        code, out, err = run_cli(["refresh", "1"], registry_path, capsys)
        assert code == 0

        data = ts._parse_registry(registry_path)
        entry = data["threads"]["1"]
        assert entry["status"] == "merged"
        assert entry["prState"]["ci"] == "SUCCESS"
        assert entry["prState"]["stale"] is False

    def test_refresh_never_downgrades_manual_blocked_on_open_pr(self, registry_path, capsys, monkeypatch):
        self._seed(registry_path, status="blocked")

        def ok_runner(cmd, timeout):
            if cmd[:2] == ["gh", "auth"]:
                return subprocess.CompletedProcess(cmd, 0, "", "")
            payload = {"state": "OPEN", "title": "t", "url": "u", "mergedAt": None, "statusCheckRollup": []}
            return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

        monkeypatch.setattr(ts, "_run_command", ok_runner)
        run_cli(["refresh", "1"], registry_path, capsys)
        data = ts._parse_registry(registry_path)
        assert data["threads"]["1"]["status"] == "blocked"

    def test_refresh_unknown_thread_id_errors(self, registry_path, capsys):
        ts._write_registry(registry_path, ts._empty_registry())
        code, out, err = run_cli(["refresh", "ghost"], registry_path, capsys)
        assert code == 2
        assert "unknown" in err

    def test_refresh_skips_entries_without_repo_or_pr(self, registry_path, capsys):
        data = ts._empty_registry()
        data["threads"]["1"] = {
            "threadId": "1", "title": "T", "description": "", "repo": None, "pr": None,
            "branch": None, "status": "active", "tags": [], "links": [], "notes": "",
            "prState": None, "createdAt": "x", "updatedAt": "x",
        }
        ts._write_registry(registry_path, data)
        code, out, err = run_cli(["refresh", "--all", "--json"], registry_path, capsys)
        assert code == 0
        results = json.loads(out)
        assert results[0]["ok"] is False
        assert "skipped" in results[0]["error"]


# ---------------------------------------------------------------------------
# CLI: end-to-end add/set/rm/list/show
# ---------------------------------------------------------------------------

class TestCliCrud:
    def test_add_then_show(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "Widget", "--status", "active"], registry_path, capsys)
        code, out, err = run_cli(["show", "1", "--json"], registry_path, capsys)
        assert code == 0
        entry = json.loads(out)
        assert entry["title"] == "Widget"
        assert entry["status"] == "active"

    def test_add_duplicate_without_force_fails(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "Widget"], registry_path, capsys)
        code, out, err = run_cli(["add", "1", "--title", "Widget2"], registry_path, capsys)
        assert code == 1
        assert "already exists" in err

    def test_add_duplicate_with_force_overwrites(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "Widget"], registry_path, capsys)
        code, out, err = run_cli(["add", "1", "--title", "Widget2", "--force"], registry_path, capsys)
        assert code == 0
        _, out2, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        assert json.loads(out2)["title"] == "Widget2"

    def test_add_invalid_status_rejected(self, registry_path, capsys):
        code, out, err = run_cli(["add", "1", "--title", "W", "--status", "bogus"], registry_path, capsys)
        assert code == 2
        assert "invalid status" in err

    def test_set_partial_update_only_changes_given_fields(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "Widget", "--desc", "orig"], registry_path, capsys)
        run_cli(["set", "1", "--status", "blocked"], registry_path, capsys)
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        assert entry["status"] == "blocked"
        assert entry["title"] == "Widget"
        assert entry["description"] == "orig"

    def test_set_missing_thread_errors(self, registry_path, capsys):
        code, out, err = run_cli(["set", "ghost", "--status", "active"], registry_path, capsys)
        assert code == 1

    def test_rm_removes_entry(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "Widget"], registry_path, capsys)
        code, _, _ = run_cli(["rm", "1"], registry_path, capsys)
        assert code == 0
        code, _, err = run_cli(["show", "1"], registry_path, capsys)
        assert code == 1

    def test_list_filters_by_status(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "A", "--status", "active"], registry_path, capsys)
        run_cli(["add", "2", "--title", "B", "--status", "blocked"], registry_path, capsys)
        _, out, _ = run_cli(["list", "--status", "blocked", "--json"], registry_path, capsys)
        entries = json.loads(out)
        assert len(entries) == 1
        assert entries[0]["threadId"] == "2"

    def test_list_filters_by_tag(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "A", "--tags", "foo,bar"], registry_path, capsys)
        run_cli(["add", "2", "--title", "B", "--tags", "baz"], registry_path, capsys)
        _, out, _ = run_cli(["list", "--tag", "foo", "--json"], registry_path, capsys)
        entries = json.loads(out)
        assert len(entries) == 1
        assert entries[0]["threadId"] == "1"
