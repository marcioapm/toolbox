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


def _pr(number, repo=None, primary=False, state=None, ci="NONE", stale=False,
        error=None, url="", title=""):
    """Build a `prs` list entry for tests, filling in the prState shape."""
    return {
        "repo": repo,
        "number": number,
        "primary": primary,
        "prState": {
            "state": state,
            "ci": ci,
            "title": title,
            "url": url,
            "mergedAt": None,
            "checkedAt": None,
            "stale": stale,
            "error": error,
        },
    }


# ---------------------------------------------------------------------------
# registry: atomic writes, locking, corruption
# ---------------------------------------------------------------------------

class TestRegistryBasics:
    def test_missing_file_reads_as_empty(self, registry_path):
        data = ts._parse_registry(registry_path)
        assert data == {"schemaVersion": 5, "accounts": {}, "threads": {}}

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
        entry = {"status": status, "prs": [_pr(1, state="OPEN", ci="SUCCESS")]}
        assert ts._effective_icon(entry) == icon

    def test_ci_failure_overrides_active_status_to_red(self):
        entry = {"status": "active", "prs": [_pr(1, state="OPEN", ci="FAILURE")]}
        assert ts._effective_icon(entry) == ts.STATUS_EMOJI["blocked"]

    @pytest.mark.parametrize("status", ["merged", "closed", "done"])
    def test_ci_failure_does_not_override_terminal_statuses(self, status):
        entry = {"status": status, "prs": [_pr(1, state="OPEN", ci="FAILURE")]}
        assert ts._effective_icon(entry) == ts.STATUS_EMOJI[status]

    def test_no_prs_uses_plain_status_icon(self):
        entry = {"status": "review", "prs": []}
        assert ts._effective_icon(entry) == ts.STATUS_EMOJI["review"]


# ---------------------------------------------------------------------------
# title rendering + truncation
# ---------------------------------------------------------------------------

class TestRenderTitle:
    def test_basic_format(self):
        entry = {"title": "mSPRT", "status": "review", "prs": [_pr(4694, primary=True)]}
        assert ts.render_title(entry) == "🔵 mSPRT · PR#4694 · review"

    def test_no_pr_omits_pr_segment(self):
        entry = {"title": "Widget", "status": "active", "prs": []}
        assert ts.render_title(entry) == "🟢 Widget · active"

    def test_title_never_exceeds_100_chars(self):
        entry = {"title": "x" * 200, "status": "review", "prs": [_pr(999999, primary=True)]}
        out = ts.render_title(entry)
        assert len(out) <= ts.THREAD_TITLE_MAX

    def test_long_title_truncated_with_ellipsis_suffix_preserved(self):
        entry = {"title": "y" * 200, "status": "blocked", "prs": [_pr(42, primary=True)]}
        out = ts.render_title(entry)
        assert out.endswith("· PR#42 · blocked")
        assert "…" in out

    def test_short_title_not_truncated(self):
        entry = {"title": "short", "status": "active", "prs": []}
        out = ts.render_title(entry)
        assert "…" not in out

    def test_multiple_prs_render_plus_n_suffix(self):
        entry = {
            "title": "mSPRT", "status": "review",
            "prs": [_pr(4694, primary=True), _pr(4700), _pr(4701)],
        }
        assert ts.render_title(entry) == "🔵 mSPRT · PR#4694 +2 · review"

    def test_plus_n_suffix_survives_truncation_at_100_chars(self):
        entry = {
            "title": "z" * 200, "status": "review",
            "prs": [_pr(4694, primary=True), _pr(4700), _pr(4701)],
        }
        out = ts.render_title(entry)
        assert len(out) <= ts.THREAD_TITLE_MAX
        assert out.endswith("· PR#4694 +2 · review")
        assert "…" in out

    def test_first_pr_is_primary_when_none_marked(self):
        entry = {"title": "T", "status": "active", "prs": [_pr(1), _pr(2)]}
        assert ts.render_title(entry) == "🟢 T · PR#1 +1 · active"


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

# ---------------------------------------------------------------------------
# derived thread status (multi-PR)
# ---------------------------------------------------------------------------

class TestDeriveThreadStatus:
    def test_no_prs_leaves_status_untouched(self):
        entry = {"status": "review", "prs": []}
        assert ts.derive_thread_status(entry) is None

    def test_any_open_ci_failure_is_blocked(self):
        entry = {"status": "active", "prs": [_pr(1, state="OPEN", ci="FAILURE")]}
        assert ts.derive_thread_status(entry) == "blocked"

    def test_one_open_failing_among_others_is_still_blocked(self):
        entry = {
            "status": "active",
            "prs": [_pr(1, state="OPEN", ci="SUCCESS"), _pr(2, state="OPEN", ci="FAILURE")],
        }
        assert ts.derive_thread_status(entry) == "blocked"

    def test_all_merged_is_merged(self):
        entry = {"status": "review", "prs": [_pr(1, state="MERGED"), _pr(2, state="MERGED")]}
        assert ts.derive_thread_status(entry) == "merged"

    def test_all_closed_no_open_is_closed(self):
        entry = {"status": "review", "prs": [_pr(1, state="CLOSED"), _pr(2, state="CLOSED")]}
        assert ts.derive_thread_status(entry) == "closed"

    def test_mix_of_closed_and_merged_no_open_is_closed(self):
        entry = {"status": "review", "prs": [_pr(1, state="CLOSED"), _pr(2, state="MERGED")]}
        assert ts.derive_thread_status(entry) == "closed"

    def test_open_pr_leaves_review_status_as_review(self):
        entry = {"status": "review", "prs": [_pr(1, state="OPEN", ci="SUCCESS")]}
        assert ts.derive_thread_status(entry) == "review"

    def test_open_pr_leaves_blocked_status_as_blocked(self):
        entry = {"status": "blocked", "prs": [_pr(1, state="OPEN", ci="SUCCESS")]}
        assert ts.derive_thread_status(entry) == "blocked"

    def test_open_pr_leaves_paused_status_as_paused(self):
        entry = {"status": "paused", "prs": [_pr(1, state="OPEN", ci="SUCCESS")]}
        assert ts.derive_thread_status(entry) == "paused"

    def test_open_pr_with_plain_active_status_stays_active(self):
        entry = {"status": "active", "prs": [_pr(1, state="OPEN", ci="SUCCESS")]}
        assert ts.derive_thread_status(entry) == "active"

    def test_open_pr_with_merged_status_becomes_active(self):
        # Not one of the "sticky" statuses (review/blocked/paused) -> active.
        entry = {"status": "merged", "prs": [_pr(1, state="OPEN", ci="SUCCESS")]}
        assert ts.derive_thread_status(entry) == "active"

    def test_never_auto_downgrades_manual_blocked_unless_all_resolved(self):
        entry = {"status": "blocked", "prs": [_pr(1, state="OPEN", ci="SUCCESS")]}
        assert ts.derive_thread_status(entry) == "blocked"

    def test_manual_blocked_cleared_once_all_prs_merged(self):
        entry = {"status": "blocked", "prs": [_pr(1, state="MERGED")]}
        assert ts.derive_thread_status(entry) == "merged"

    def test_manual_paused_cleared_once_all_prs_closed(self):
        entry = {"status": "paused", "prs": [_pr(1, state="CLOSED")]}
        assert ts.derive_thread_status(entry) == "closed"

    def test_mixed_open_and_merged_with_no_failure_leaves_sticky_status(self):
        entry = {
            "status": "blocked",
            "prs": [_pr(1, state="OPEN", ci="SUCCESS"), _pr(2, state="MERGED")],
        }
        assert ts.derive_thread_status(entry) == "blocked"


# ---------------------------------------------------------------------------
# CLI: refresh command — gh failure -> stale not crash
# ---------------------------------------------------------------------------

class TestRefreshCli:
    def _seed(self, registry_path, thread_id="1", repo="absmartly/foo", pr=5, status="review", extra_prs=None):
        data = ts._empty_registry()
        prs = [{"repo": None, "number": pr, "primary": True, "prState": dict(ts.EMPTY_PR_STATE)}]
        if extra_prs:
            prs.extend(extra_prs)
        data["threads"][thread_id] = {
            "threadId": thread_id,
            "title": "T",
            "description": "",
            "repo": repo,
            "branch": None,
            "status": status,
            "tags": [],
            "links": [],
            "notes": "",
            "prs": prs,
            "createdAt": "2026-01-01T00:00:00Z",
            "updatedAt": "2026-01-01T00:00:00Z",
        }
        ts._write_registry(registry_path, data)

    def test_gh_failure_keeps_cache_and_marks_stale_single_thread_nonzero_exit(
        self, registry_path, capsys, monkeypatch
    ):
        self._seed(registry_path)
        data = ts._parse_registry(registry_path)
        data["threads"]["1"]["prs"][0]["prState"] = {
            "state": "OPEN", "ci": "SUCCESS", "title": "old", "url": "u",
            "mergedAt": None, "checkedAt": "2026-01-01T00:00:00Z", "stale": False, "error": None,
        }
        ts._write_registry(registry_path, data)

        def failing_runner(cmd, timeout):
            if cmd[:2] == ["gh", "auth"]:
                return subprocess.CompletedProcess(cmd, 0, "", "")
            raise subprocess.TimeoutExpired(cmd, timeout)

        monkeypatch.setattr(ts, "_run_command", failing_runner)
        code, out, err = run_cli(["refresh", "1"], registry_path, capsys)
        assert code == 1  # explicit single-thread refresh: nonzero on error

        data = ts._parse_registry(registry_path)
        pr_state = data["threads"]["1"]["prs"][0]["prState"]
        assert pr_state["state"] == "OPEN"  # cached value preserved
        assert pr_state["stale"] is True
        assert pr_state["error"]

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
        assert entry["prs"][0]["prState"]["ci"] == "SUCCESS"
        assert entry["prs"][0]["prState"]["stale"] is False

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
            "threadId": "1", "title": "T", "description": "", "repo": None,
            "branch": None, "status": "active", "tags": [], "links": [], "notes": "",
            "prs": [], "createdAt": "x", "updatedAt": "x",
        }
        ts._write_registry(registry_path, data)
        code, out, err = run_cli(["refresh", "--all", "--json"], registry_path, capsys)
        assert code == 0
        results = json.loads(out)
        assert results[0]["ok"] is False
        assert "skipped" in results[0]["error"]

    def test_refresh_multi_pr_updates_each_pr_independently(self, registry_path, capsys, monkeypatch):
        extra = [{"repo": None, "number": 6, "primary": False, "prState": dict(ts.EMPTY_PR_STATE)}]
        self._seed(registry_path, pr=5, status="review", extra_prs=extra)

        def ok_runner(cmd, timeout):
            if cmd[:2] == ["gh", "auth"]:
                return subprocess.CompletedProcess(cmd, 0, "", "")
            pr_num = cmd[cmd.index("view") + 1]
            state = "MERGED" if pr_num == "5" else "OPEN"
            payload = {"state": state, "title": "t", "url": f"u{pr_num}", "mergedAt": None, "statusCheckRollup": []}
            return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

        monkeypatch.setattr(ts, "_run_command", ok_runner)
        code, out, err = run_cli(["refresh", "1"], registry_path, capsys)
        assert code == 0

        data = ts._parse_registry(registry_path)
        prs = {p["number"]: p for p in data["threads"]["1"]["prs"]}
        assert prs[5]["prState"]["state"] == "MERGED"
        assert prs[6]["prState"]["state"] == "OPEN"

    def test_refresh_per_pr_failure_marks_only_that_pr_stale(self, registry_path, capsys, monkeypatch):
        extra = [{"repo": None, "number": 6, "primary": False, "prState": dict(ts.EMPTY_PR_STATE)}]
        self._seed(registry_path, pr=5, status="review", extra_prs=extra)

        def flaky_runner(cmd, timeout):
            if cmd[:2] == ["gh", "auth"]:
                return subprocess.CompletedProcess(cmd, 0, "", "")
            pr_num = cmd[cmd.index("view") + 1]
            if pr_num == "6":
                raise subprocess.TimeoutExpired(cmd, timeout)
            payload = {"state": "OPEN", "title": "t", "url": "u5", "mergedAt": None, "statusCheckRollup": []}
            return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

        monkeypatch.setattr(ts, "_run_command", flaky_runner)
        code, out, err = run_cli(["refresh", "1"], registry_path, capsys)

        data = ts._parse_registry(registry_path)
        prs = {p["number"]: p for p in data["threads"]["1"]["prs"]}
        assert prs[5]["prState"]["stale"] is False
        assert prs[5]["prState"]["state"] == "OPEN"
        assert prs[6]["prState"]["stale"] is True
        assert prs[6]["prState"]["error"]
        # The healthy PR's own state must survive a sibling PR's gh failure.
        assert data["threads"]["1"]["status"] in ("review", "active")

    def test_refresh_never_crashes_and_preserves_cache_on_failure(self, registry_path, capsys, monkeypatch):
        self._seed(registry_path)
        data = ts._parse_registry(registry_path)
        data["threads"]["1"]["prs"][0]["prState"] = {
            "state": "OPEN", "ci": "SUCCESS", "title": "cached", "url": "u",
            "mergedAt": None, "checkedAt": "2026-01-01T00:00:00Z", "stale": False, "error": None,
        }
        ts._write_registry(registry_path, data)

        def crashy_runner(cmd, timeout):
            if cmd[:2] == ["gh", "auth"]:
                return subprocess.CompletedProcess(cmd, 0, "", "")
            raise OSError("gh binary missing")

        monkeypatch.setattr(ts, "_run_command", crashy_runner)
        code, out, err = run_cli(["refresh", "--all"], registry_path, capsys)
        assert code == 0  # --all always exits zero regardless of per-thread errors

        data = ts._parse_registry(registry_path)
        pr_state = data["threads"]["1"]["prs"][0]["prState"]
        assert pr_state["title"] == "cached"  # cache never wiped
        assert pr_state["stale"] is True


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


# ---------------------------------------------------------------------------
# v1 -> v2 schema migration
# ---------------------------------------------------------------------------

class TestSchemaMigration:
    def _write_v1(self, registry_path, pr=4694, pr_state=None, repo="acme/widgets"):
        raw = {
            "schemaVersion": 1,
            "accounts": {},
            "threads": {
                "1": {
                    "threadId": "1",
                    "title": "mSPRT",
                    "description": "",
                    "repo": repo,
                    "pr": pr,
                    "branch": None,
                    "status": "review",
                    "tags": [],
                    "links": [],
                    "notes": "",
                    "prState": pr_state,
                    "createdAt": "2026-01-01T00:00:00Z",
                    "updatedAt": "2026-01-01T00:00:00Z",
                }
            },
        }
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(json.dumps(raw))
        return raw

    def test_migrate_entry_converts_scalar_pr_to_prs_list(self):
        entry = {"repo": "acme/widgets", "pr": 4694, "prState": {"state": "OPEN", "ci": "SUCCESS"}}
        ts._migrate_entry_v1_to_v2(entry)
        assert "pr" not in entry
        assert "prState" not in entry
        assert entry["prs"] == [{
            "repo": "acme/widgets", "number": 4694, "primary": True,
            "prState": {**ts.EMPTY_PR_STATE, "state": "OPEN", "ci": "SUCCESS"},
        }]

    def test_migrate_entry_with_no_pr_gets_empty_prs_list(self):
        entry = {"repo": None, "pr": None, "prState": None}
        ts._migrate_entry_v1_to_v2(entry)
        assert entry["prs"] == []
        assert "pr" not in entry
        assert "prState" not in entry

    def test_migrate_entry_is_idempotent(self):
        entry = {"repo": "a/b", "pr": 1, "prState": {"state": "OPEN"}}
        ts._migrate_entry_v1_to_v2(entry)
        first = json.dumps(entry, sort_keys=True)
        ts._migrate_entry_v1_to_v2(entry)
        assert json.dumps(entry, sort_keys=True) == first

    def test_load_and_migrate_bumps_schema_version(self, registry_path):
        self._write_v1(registry_path)
        data, migrated = ts._load_and_migrate(registry_path)
        assert migrated is True
        assert data["schemaVersion"] == 5
        assert data["threads"]["1"]["prs"][0]["number"] == 4694

    def test_load_and_migrate_no_op_on_v2(self, registry_path):
        data = ts._empty_registry()
        data["threads"]["1"] = {"prs": []}
        ts._write_registry(registry_path, data)
        loaded, migrated = ts._load_and_migrate(registry_path)
        assert migrated is False
        assert loaded["schemaVersion"] == 5

    def test_cli_read_migrates_v1_transparently(self, registry_path, capsys):
        self._write_v1(registry_path, pr_state={"state": "OPEN", "ci": "SUCCESS", "url": "u", "title": "t", "mergedAt": None, "checkedAt": "x", "stale": False})
        code, out, err = run_cli(["show", "1", "--json"], registry_path, capsys)
        assert code == 0
        entry = json.loads(out)
        assert entry["prs"][0]["number"] == 4694
        assert entry["pr"] == 4694  # computed read-only convenience field

    def test_read_only_migration_does_not_persist(self, registry_path, capsys):
        raw_before = self._write_v1(registry_path)
        run_cli(["show", "1"], registry_path, capsys)
        on_disk = json.loads(registry_path.read_text())
        assert on_disk == raw_before  # a read-only command never migrates on disk

    def test_write_triggers_migration_and_backup(self, registry_path, capsys):
        self._write_v1(registry_path)
        code, out, err = run_cli(["set", "1", "--desc", "touched"], registry_path, capsys)
        assert code == 0

        on_disk = json.loads(registry_path.read_text())
        assert on_disk["schemaVersion"] == 5
        assert on_disk["threads"]["1"]["prs"][0]["number"] == 4694
        assert "pr" not in on_disk["threads"]["1"]

        backups = list(registry_path.parent.glob("registry.json.v1-backup-*"))
        assert len(backups) == 1
        backed_up = json.loads(backups[0].read_text())
        assert backed_up["schemaVersion"] == 1
        assert backed_up["threads"]["1"]["pr"] == 4694

    def test_migration_backup_only_happens_once(self, registry_path, capsys):
        self._write_v1(registry_path)
        run_cli(["set", "1", "--desc", "first touch"], registry_path, capsys)
        run_cli(["set", "1", "--desc", "second touch"], registry_path, capsys)
        backups = list(registry_path.parent.glob("registry.json.v1-backup-*"))
        assert len(backups) == 1

    def test_migration_preserves_all_other_entry_fields(self, registry_path, capsys):
        self._write_v1(registry_path)
        run_cli(["set", "1", "--desc", "touched"], registry_path, capsys)
        on_disk = json.loads(registry_path.read_text())
        entry = on_disk["threads"]["1"]
        assert entry["title"] == "mSPRT"
        assert entry["repo"] == "acme/widgets"
        assert entry["status"] == "review"
        assert entry["description"] == "touched"

    def test_migration_no_data_loss_across_multiple_threads(self, registry_path):
        raw = {
            "schemaVersion": 1,
            "accounts": {},
            "threads": {
                "1": {"threadId": "1", "title": "A", "repo": "a/b", "pr": 1, "prState": None, "status": "active"},
                "2": {"threadId": "2", "title": "B", "repo": "c/d", "pr": None, "prState": None, "status": "paused"},
            },
        }
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(json.dumps(raw))
        data, migrated = ts._load_and_migrate(registry_path)
        assert migrated is True
        assert data["threads"]["1"]["prs"][0]["number"] == 1
        assert data["threads"]["2"]["prs"] == []
        assert data["threads"]["1"]["title"] == "A"
        assert data["threads"]["2"]["status"] == "paused"


# ---------------------------------------------------------------------------
# PR reference parsing (--add-pr / --rm-pr / --primary-pr)
# ---------------------------------------------------------------------------

class TestParsePrRef:
    def test_bare_number(self):
        assert ts.parse_pr_ref("4694") == (None, 4694)

    def test_hash_number(self):
        assert ts.parse_pr_ref("#4694") == (None, 4694)

    def test_repo_hash_number(self):
        assert ts.parse_pr_ref("owner/name#4694") == ("owner/name", 4694)

    def test_full_url(self):
        assert ts.parse_pr_ref("https://github.com/owner/name/pull/4694") == ("owner/name", 4694)

    def test_bare_number_falls_back_to_default_repo(self):
        assert ts.parse_pr_ref("4694", default_repo="acme/widgets") == ("acme/widgets", 4694)

    def test_invalid_ref_raises(self):
        with pytest.raises(ValueError):
            ts.parse_pr_ref("not-a-pr-ref")


# ---------------------------------------------------------------------------
# multi-PR add/rm/primary
# ---------------------------------------------------------------------------

class TestMultiPrHelpers:
    def test_add_pr_appends(self):
        entry = {"repo": "a/b", "prs": []}
        ts.add_pr(entry, None, 1)
        ts.add_pr(entry, None, 2)
        assert [p["number"] for p in entry["prs"]] == [1, 2]

    def test_add_pr_dedupes_by_repo_and_number(self):
        entry = {"repo": "a/b", "prs": []}
        ts.add_pr(entry, None, 1)
        ts.add_pr(entry, "a/b", 1)
        assert len(entry["prs"]) == 1

    def test_add_pr_different_repo_same_number_not_deduped(self):
        entry = {"repo": "a/b", "prs": []}
        ts.add_pr(entry, None, 1)
        ts.add_pr(entry, "c/d", 1)
        assert len(entry["prs"]) == 2

    def test_first_pr_added_is_implicit_primary_via_fallback(self):
        entry = {"repo": "a/b", "prs": []}
        ts.add_pr(entry, None, 1)
        ts.add_pr(entry, None, 2)
        assert ts.primary_pr(entry)["number"] == 1

    def test_remove_pr_by_number(self):
        entry = {"repo": "a/b", "prs": [ts_pr(1), ts_pr(2)]}
        assert ts.remove_pr(entry, None, 1) is True
        assert [p["number"] for p in entry["prs"]] == [2]

    def test_remove_pr_not_found_returns_false(self):
        entry = {"repo": "a/b", "prs": [ts_pr(1)]}
        assert ts.remove_pr(entry, None, 99) is False

    def test_remove_primary_pr_promotes_next(self):
        entry = {"repo": "a/b", "prs": [ts_pr(1, primary=True), ts_pr(2)]}
        ts.remove_pr(entry, None, 1)
        assert entry["prs"][0]["primary"] is True
        assert entry["prs"][0]["number"] == 2

    def test_set_primary_pr_marks_exactly_one(self):
        entry = {"repo": "a/b", "prs": [ts_pr(1, primary=True), ts_pr(2)]}
        assert ts.set_primary_pr(entry, None, 2) is True
        assert entry["prs"][0]["primary"] is False
        assert entry["prs"][1]["primary"] is True

    def test_set_primary_pr_not_found_returns_false(self):
        entry = {"repo": "a/b", "prs": [ts_pr(1)]}
        assert ts.set_primary_pr(entry, None, 99) is False

    def test_set_prs_single_replaces_whole_list(self):
        entry = {"repo": "a/b", "prs": [ts_pr(1), ts_pr(2)]}
        ts.set_prs_single(entry, "a/b", 3)
        assert len(entry["prs"]) == 1
        assert entry["prs"][0]["number"] == 3
        assert entry["prs"][0]["primary"] is True

    def test_pr_entry_repo_override_inherits_thread_repo_when_absent(self):
        entry = {"repo": "thread/repo", "prs": [{"repo": None, "number": 1, "primary": True, "prState": {}}]}
        assert ts._pr_repo(entry["prs"][0], entry["repo"]) == "thread/repo"

    def test_pr_entry_repo_override_wins_when_present(self):
        entry = {"repo": "thread/repo", "prs": [{"repo": "other/repo", "number": 1, "primary": True, "prState": {}}]}
        assert ts._pr_repo(entry["prs"][0], entry["repo"]) == "other/repo"


def ts_pr(number, primary=False):
    return {"repo": None, "number": number, "primary": primary, "prState": dict(ts.EMPTY_PR_STATE)}


# ---------------------------------------------------------------------------
# CLI: --add-pr / --rm-pr / --primary-pr / --pr
# ---------------------------------------------------------------------------

class TestCliMultiPr:
    def test_add_with_pr_sets_single_primary_pr(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T", "--repo", "a/b", "--pr", "4694"], registry_path, capsys)
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        assert len(entry["prs"]) == 1
        assert entry["prs"][0]["number"] == 4694
        assert entry["prs"][0]["primary"] is True
        assert entry["pr"] == 4694

    def test_add_pr_flag_appends_multiple(self, registry_path, capsys):
        run_cli(
            ["add", "1", "--title", "T", "--repo", "a/b", "--add-pr", "4694", "--add-pr", "4700"],
            registry_path, capsys,
        )
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        assert sorted(p["number"] for p in entry["prs"]) == [4694, 4700]

    def test_add_pr_accepts_all_four_ref_forms(self, registry_path, capsys):
        run_cli([
            "add", "1", "--title", "T", "--repo", "a/b",
            "--add-pr", "1",
            "--add-pr", "#2",
            "--add-pr", "c/d#3",
            "--add-pr", "https://github.com/e/f/pull/4",
        ], registry_path, capsys)
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        by_number = {p["number"]: p for p in entry["prs"]}
        assert ts._pr_repo(by_number[1], entry["repo"]) == "a/b"
        assert ts._pr_repo(by_number[2], entry["repo"]) == "a/b"
        assert ts._pr_repo(by_number[3], entry["repo"]) == "c/d"
        assert ts._pr_repo(by_number[4], entry["repo"]) == "e/f"

    def test_set_rm_pr_removes_one(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T", "--repo", "a/b", "--add-pr", "1", "--add-pr", "2"], registry_path, capsys)
        run_cli(["set", "1", "--rm-pr", "1"], registry_path, capsys)
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        assert [p["number"] for p in entry["prs"]] == [2]

    def test_set_primary_pr_switches_primary(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T", "--repo", "a/b", "--add-pr", "1", "--add-pr", "2"], registry_path, capsys)
        run_cli(["set", "1", "--primary-pr", "2"], registry_path, capsys)
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        assert entry["pr"] == 2

    def test_set_primary_pr_unmatched_ref_errors(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T", "--repo", "a/b", "--add-pr", "1"], registry_path, capsys)
        code, out, err = run_cli(["set", "1", "--primary-pr", "999"], registry_path, capsys)
        assert code == 2

    def test_set_pr_flag_replaces_whole_list(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T", "--repo", "a/b", "--add-pr", "1", "--add-pr", "2"], registry_path, capsys)
        run_cli(["set", "1", "--pr", "9"], registry_path, capsys)
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        assert [p["number"] for p in entry["prs"]] == [9]

    def test_show_json_pr_field_null_when_no_prs(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T"], registry_path, capsys)
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        assert entry["pr"] is None

    def test_list_shows_primary_plus_count(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T", "--repo", "a/b", "--add-pr", "1", "--add-pr", "2", "--add-pr", "3"], registry_path, capsys)
        _, out, _ = run_cli(["list"], registry_path, capsys)
        assert "#1+2" in out


# ---------------------------------------------------------------------------
# scan --apply: append + dedupe rather than overwrite
# ---------------------------------------------------------------------------

class TestScanApplyAppends:
    def test_apply_appends_new_prs_without_dropping_existing(self, registry_path, capsys, tmp_path):
        run_cli(["add", "1", "--title", "T", "--repo", "a/b", "--add-pr", "1"], registry_path, capsys)
        f = tmp_path / "thread.txt"
        f.write_text("see https://github.com/a/b/pull/2")
        run_cli(["scan", "1", "--file", str(f), "--apply"], registry_path, capsys)
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        assert sorted(p["number"] for p in entry["prs"]) == [1, 2]

    def test_apply_dedupes_already_known_pr(self, registry_path, capsys, tmp_path):
        run_cli(["add", "1", "--title", "T", "--repo", "a/b", "--add-pr", "1"], registry_path, capsys)
        f = tmp_path / "thread.txt"
        f.write_text("see https://github.com/a/b/pull/1")
        run_cli(["scan", "1", "--file", str(f), "--apply"], registry_path, capsys)
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        assert len(entry["prs"]) == 1

    def test_apply_on_empty_entry_sets_primary_from_best_match(self, registry_path, capsys, tmp_path):
        run_cli(["add", "1", "--title", "T"], registry_path, capsys)
        f = tmp_path / "thread.txt"
        f.write_text("https://github.com/a/b/pull/1 https://github.com/a/b/pull/2")
        run_cli(["scan", "1", "--file", str(f), "--apply"], registry_path, capsys)
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        assert entry["pr"] == 2  # best_match: last occurrence among URL matches


# ---------------------------------------------------------------------------
# card rendering with 0 / 1 / 3 PRs
# ---------------------------------------------------------------------------

class TestRenderCardMultiPr:
    def test_card_with_no_prs_renders_branch_line(self):
        entry = {"title": "T", "status": "active", "branches": [{"repo": None, "name": "feat/x", "primary": True}], "prs": []}
        out = ts.render_card_markdown(entry)
        assert "branch: `feat/x` (primary)" in out
        assert "PR:" not in out

    def test_card_with_one_pr_renders_single_line(self):
        entry = {"title": "T", "status": "review", "prs": [_pr(4694, primary=True, state="OPEN", ci="SUCCESS", url="https://github.com/a/b/pull/4694")]}
        out = ts.render_card_markdown(entry)
        assert "PR: **[#4694](https://github.com/a/b/pull/4694)** OPEN · CI SUCCESS" in out

    def test_card_with_three_prs_renders_one_line_each(self):
        entry = {
            "title": "T", "status": "review",
            "prs": [
                _pr(1, primary=True, state="OPEN", ci="SUCCESS"),
                _pr(2, state="OPEN", ci="FAILURE"),
                _pr(3, state="MERGED", ci="SUCCESS"),
            ],
        }
        out = ts.render_card_markdown(entry)
        lines = [l for l in out.splitlines() if l.startswith("PR:")]
        assert len(lines) == 3
        assert "⚠️" in lines[1]  # the FAILURE line

    def test_card_marks_stale_pr(self):
        entry = {"title": "T", "status": "review", "prs": [_pr(1, primary=True, state="OPEN", ci="SUCCESS", stale=True)]}
        out = ts.render_card_markdown(entry)
        assert "stale" in out

    def test_card_embed_pr_field_shows_count_with_extras(self):
        entry = {
            "title": "T", "status": "review",
            "prs": [_pr(1, primary=True, ci="SUCCESS"), _pr(2), _pr(3)],
        }
        embed = ts.render_card_embed(entry)
        pr_field = next(f for f in embed["fields"] if f["name"] == "PR")
        assert pr_field["value"] == "#1 (+2)"


# ---------------------------------------------------------------------------
# v2 -> v3 schema migration: branches, runs, lastActivityAt
# ---------------------------------------------------------------------------

class TestSchemaMigrationV3:
    def _write_v2(self, registry_path, branch="feat/x", repo="acme/widgets"):
        raw = {
            "schemaVersion": 2,
            "accounts": {},
            "threads": {
                "1": {
                    "threadId": "1",
                    "title": "mSPRT",
                    "description": "",
                    "repo": repo,
                    "branch": branch,
                    "status": "review",
                    "tags": [],
                    "links": [],
                    "notes": "",
                    "prs": [],
                    "createdAt": "2026-01-01T00:00:00Z",
                    "updatedAt": "2026-01-01T00:00:00Z",
                }
            },
        }
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(json.dumps(raw))
        return raw

    def test_migrate_entry_v2_to_v3_converts_scalar_branch_to_list(self):
        entry = {"repo": "acme/widgets", "branch": "feat/x"}
        ts._migrate_entry_v2_to_v3(entry)
        assert "branch" not in entry
        assert entry["branches"] == [{"repo": "acme/widgets", "name": "feat/x", "primary": True}]
        assert entry["runs"] == []
        assert entry["lastActivityAt"] is None
        assert entry["lastActivitySource"] is None

    def test_migrate_entry_v2_to_v3_with_no_branch_gets_empty_list(self):
        entry = {"repo": None, "branch": None}
        ts._migrate_entry_v2_to_v3(entry)
        assert entry["branches"] == []
        assert "branch" not in entry

    def test_migrate_entry_v2_to_v3_is_idempotent(self):
        entry = {"repo": "a/b", "branch": "feat/x"}
        ts._migrate_entry_v2_to_v3(entry)
        first = json.dumps(entry, sort_keys=True)
        ts._migrate_entry_v2_to_v3(entry)
        assert json.dumps(entry, sort_keys=True) == first

    def test_load_and_migrate_v2_to_v3_bumps_schema_version_and_backs_up(self, registry_path, capsys):
        self._write_v2(registry_path)
        data, migrated = ts._load_and_migrate(registry_path)
        assert migrated is True
        assert data["schemaVersion"] == 5
        assert data["threads"]["1"]["branches"] == [{"repo": "acme/widgets", "name": "feat/x", "primary": True}]

    def test_write_from_v2_triggers_v2_backup(self, registry_path, capsys):
        self._write_v2(registry_path)
        code, out, err = run_cli(["set", "1", "--desc", "touched"], registry_path, capsys)
        assert code == 0

        on_disk = json.loads(registry_path.read_text())
        assert on_disk["schemaVersion"] == 5
        assert on_disk["threads"]["1"]["branches"][0]["name"] == "feat/x"
        assert "branch" not in on_disk["threads"]["1"]

        backups = list(registry_path.parent.glob("registry.json.v2-backup-*"))
        assert len(backups) == 1
        backed_up = json.loads(backups[0].read_text())
        assert backed_up["schemaVersion"] == 2
        assert backed_up["threads"]["1"]["branch"] == "feat/x"

    def test_v1_registry_migrates_all_the_way_to_v3_in_one_pass(self, registry_path):
        raw = {
            "schemaVersion": 1,
            "accounts": {},
            "threads": {
                "1": {
                    "threadId": "1", "title": "A", "repo": "a/b", "pr": 1,
                    "branch": "feat/y", "prState": None, "status": "active",
                }
            },
        }
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(json.dumps(raw))
        data, migrated = ts._load_and_migrate(registry_path)
        assert migrated is True
        assert data["schemaVersion"] == 5
        entry = data["threads"]["1"]
        assert entry["prs"][0]["number"] == 1
        assert entry["branches"] == [{"repo": "a/b", "name": "feat/y", "primary": True}]
        assert entry["runs"] == []
        assert entry["lastActivityAt"] is None
        # v1 -> v4 in one pass still only backs up the original v1 bytes once.
        backups = list(registry_path.parent.glob("registry.json.v1-backup-*"))
        assert backups == []  # _load_and_migrate alone never writes a backup

    def test_migration_no_data_loss_multiple_v2_threads(self, registry_path):
        raw = {
            "schemaVersion": 2,
            "accounts": {},
            "threads": {
                "1": {"threadId": "1", "title": "A", "repo": "a/b", "branch": "feat/1", "prs": [], "status": "active"},
                "2": {"threadId": "2", "title": "B", "repo": "c/d", "branch": None, "prs": [], "status": "paused"},
            },
        }
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(json.dumps(raw))
        data, migrated = ts._load_and_migrate(registry_path)
        assert migrated is True
        assert data["threads"]["1"]["branches"][0]["name"] == "feat/1"
        assert data["threads"]["2"]["branches"] == []
        assert data["threads"]["1"]["title"] == "A"
        assert data["threads"]["2"]["status"] == "paused"


# ---------------------------------------------------------------------------
# branch helpers: get/add/remove/primary + ref parsing
# ---------------------------------------------------------------------------

class TestParseBranchRef:
    def test_bare_name(self):
        assert ts.parse_branch_ref("feat/x") == (None, "feat/x")

    def test_repo_colon_name(self):
        assert ts.parse_branch_ref("owner/name:feat/x") == ("owner/name", "feat/x")

    def test_bare_name_falls_back_to_default_repo(self):
        assert ts.parse_branch_ref("feat/x", default_repo="acme/widgets") == ("acme/widgets", "feat/x")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ts.parse_branch_ref("")

    def test_missing_name_after_colon_raises(self):
        with pytest.raises(ValueError):
            ts.parse_branch_ref("owner/name:")


class TestBranchHelpers:
    def test_add_branch_appends_and_first_is_primary(self):
        entry = {"repo": "a/b", "branches": []}
        ts.add_branch(entry, None, "feat/1")
        ts.add_branch(entry, None, "feat/2")
        assert [b["name"] for b in entry["branches"]] == ["feat/1", "feat/2"]
        assert entry["branches"][0]["primary"] is True
        assert entry["branches"][1]["primary"] is False

    def test_add_branch_dedupes_by_repo_and_name(self):
        entry = {"repo": "a/b", "branches": []}
        ts.add_branch(entry, None, "feat/1")
        ts.add_branch(entry, "a/b", "feat/1")
        assert len(entry["branches"]) == 1

    def test_remove_branch_by_name(self):
        entry = {"repo": "a/b", "branches": [ts_branch("feat/1"), ts_branch("feat/2")]}
        assert ts.remove_branch(entry, None, "feat/1") is True
        assert [b["name"] for b in entry["branches"]] == ["feat/2"]

    def test_remove_branch_not_found_returns_false(self):
        entry = {"repo": "a/b", "branches": [ts_branch("feat/1")]}
        assert ts.remove_branch(entry, None, "nope") is False

    def test_remove_primary_branch_promotes_next(self):
        entry = {"repo": "a/b", "branches": [ts_branch("feat/1", primary=True), ts_branch("feat/2")]}
        ts.remove_branch(entry, None, "feat/1")
        assert entry["branches"][0]["primary"] is True
        assert entry["branches"][0]["name"] == "feat/2"

    def test_set_primary_branch_marks_exactly_one(self):
        entry = {"repo": "a/b", "branches": [ts_branch("feat/1", primary=True), ts_branch("feat/2")]}
        assert ts.set_primary_branch(entry, None, "feat/2") is True
        assert entry["branches"][0]["primary"] is False
        assert entry["branches"][1]["primary"] is True

    def test_set_primary_branch_not_found_returns_false(self):
        entry = {"repo": "a/b", "branches": [ts_branch("feat/1")]}
        assert ts.set_primary_branch(entry, None, "nope") is False

    def test_set_branches_single_replaces_whole_list(self):
        entry = {"repo": "a/b", "branches": [ts_branch("feat/1"), ts_branch("feat/2")]}
        ts.set_branches_single(entry, "a/b", "feat/3")
        assert len(entry["branches"]) == 1
        assert entry["branches"][0]["name"] == "feat/3"
        assert entry["branches"][0]["primary"] is True


def ts_branch(name, primary=False):
    return {"repo": None, "name": name, "primary": primary}


class TestCliBranches:
    def test_branch_flag_sets_single_primary_branch(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T", "--repo", "a/b", "--branch", "feat/x"], registry_path, capsys)
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        assert entry["branches"] == [{"repo": "a/b", "name": "feat/x", "primary": True}]
        assert entry["branch"] == "feat/x"

    def test_add_branch_flag_appends_multiple(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T", "--repo", "a/b", "--add-branch", "feat/1", "--add-branch", "feat/2"], registry_path, capsys)
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        assert sorted(b["name"] for b in entry["branches"]) == ["feat/1", "feat/2"]

    def test_add_branch_accepts_repo_colon_form(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T", "--repo", "a/b", "--add-branch", "c/d:feat/x"], registry_path, capsys)
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        assert entry["branches"][0]["repo"] == "c/d"

    def test_set_rm_branch_removes_one(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T", "--repo", "a/b", "--add-branch", "feat/1", "--add-branch", "feat/2"], registry_path, capsys)
        run_cli(["set", "1", "--rm-branch", "feat/1"], registry_path, capsys)
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        assert [b["name"] for b in entry["branches"]] == ["feat/2"]

    def test_set_primary_branch_switches_primary(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T", "--repo", "a/b", "--add-branch", "feat/1", "--add-branch", "feat/2"], registry_path, capsys)
        run_cli(["set", "1", "--primary-branch", "feat/2"], registry_path, capsys)
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        assert entry["branch"] == "feat/2"

    def test_set_primary_branch_unmatched_ref_errors(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T", "--repo", "a/b", "--add-branch", "feat/1"], registry_path, capsys)
        code, out, err = run_cli(["set", "1", "--primary-branch", "nope"], registry_path, capsys)
        assert code == 2


# ---------------------------------------------------------------------------
# run helpers: host:name parsing, get/add/remove, unknown-host rejection
# ---------------------------------------------------------------------------

class TestParseRunRef:
    def test_valid_host_colon_name(self):
        assert ts.parse_run_ref("vibes:tstate2") == ("vibes", "tstate2")

    def test_missing_colon_raises(self):
        with pytest.raises(ValueError):
            ts.parse_run_ref("vibes")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ts.parse_run_ref("")


class TestRunHelpers:
    def test_add_run_appends(self):
        entry = {"runs": []}
        ts.add_run(entry, "vibes", "tstate2")
        assert entry["runs"] == [{"host": "vibes", "name": "tstate2", "state": dict(ts.EMPTY_RUN_STATE)}]

    def test_add_run_dedupes_by_host_and_name(self):
        entry = {"runs": []}
        ts.add_run(entry, "vibes", "tstate2")
        ts.add_run(entry, "vibes", "tstate2")
        assert len(entry["runs"]) == 1

    def test_add_run_different_host_same_name_not_deduped(self):
        entry = {"runs": []}
        ts.add_run(entry, "vibes", "tstate2")
        ts.add_run(entry, "macmini", "tstate2")
        assert len(entry["runs"]) == 2

    def test_remove_run_by_host_and_name(self):
        entry = {"runs": []}
        ts.add_run(entry, "vibes", "tstate2")
        ts.add_run(entry, "macmini", "other")
        assert ts.remove_run(entry, "vibes", "tstate2") is True
        assert [r["name"] for r in entry["runs"]] == ["other"]

    def test_remove_run_not_found_returns_false(self):
        entry = {"runs": []}
        assert ts.remove_run(entry, "vibes", "nope") is False


class TestCliRuns:
    def test_add_run_flag_tracks_known_host(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T", "--add-run", "vibes:tstate2"], registry_path, capsys)
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        assert entry["runs"] == [{"host": "vibes", "name": "tstate2", "state": dict(ts.EMPTY_RUN_STATE)}]

    def test_add_run_unknown_host_rejected_without_force(self, registry_path, capsys):
        code, out, err = run_cli(["add", "1", "--title", "T", "--add-run", "bogus-host:tstate2"], registry_path, capsys)
        assert code == 2
        assert "unknown host" in err

    def test_add_run_unknown_host_allowed_with_force(self, registry_path, capsys):
        code, out, err = run_cli(["add", "1", "--title", "T", "--add-run", "bogus-host:tstate2", "--force"], registry_path, capsys)
        assert code == 0
        _, out2, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out2)
        assert entry["runs"][0]["host"] == "bogus-host"

    def test_set_rm_run_removes_one(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T", "--add-run", "vibes:a", "--add-run", "macmini:b"], registry_path, capsys)
        run_cli(["set", "1", "--rm-run", "vibes:a"], registry_path, capsys)
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        assert [r["name"] for r in entry["runs"]] == ["b"]


# ---------------------------------------------------------------------------
# Discord snowflake -> ISO-8601 decoding
# ---------------------------------------------------------------------------

class TestSnowflakeDecoding:
    def test_known_snowflake_decodes_to_expected_timestamp(self):
        # Verified independently: (1531198023408029736 >> 22) + 1420070400000
        # lands on 2026-07-27T07:14:17Z.
        iso = ts.snowflake_to_iso("1531198023408029736")
        expected = ts._parse_iso("2026-07-27T07:14:17Z")
        actual = ts._parse_iso(iso)
        assert abs((actual - expected).total_seconds()) <= 1

    def test_discord_epoch_zero_snowflake_is_discord_epoch(self):
        iso = ts.snowflake_to_iso("0")
        assert iso == "2015-01-01T00:00:00Z"

    def test_cli_last_message_id_sets_last_activity_and_source(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T", "--last-message-id", "1531198023408029736"], registry_path, capsys)
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        actual = ts._parse_iso(entry["lastActivityAt"])
        expected = ts._parse_iso("2026-07-27T07:14:17Z")
        assert abs((actual - expected).total_seconds()) <= 1
        assert entry["lastActivitySource"] == "discord"

    def test_cli_last_activity_sets_manual_source(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T", "--last-activity", "2026-01-01T00:00:00Z"], registry_path, capsys)
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        assert entry["lastActivityAt"] == "2026-01-01T00:00:00Z"
        assert entry["lastActivitySource"] == "manual"


# ---------------------------------------------------------------------------
# duration parsing
# ---------------------------------------------------------------------------

class TestParseDuration:
    @pytest.mark.parametrize("raw,seconds", [
        ("35m", 35 * 60),
        ("2h", 2 * 3600),
        ("90s", 90),
        ("1d", 86400),
        ("0s", 0),
    ])
    def test_valid_durations(self, raw, seconds):
        assert ts.parse_duration(raw) == seconds

    @pytest.mark.parametrize("raw", ["", "5", "5x", "m5", "-5m", "5 m", "5.5m"])
    def test_invalid_durations_raise(self, raw):
        with pytest.raises(ValueError):
            ts.parse_duration(raw)


# ---------------------------------------------------------------------------
# silent-thread detector: boundary conditions + --stalled
# ---------------------------------------------------------------------------

def _entry_for_silent(status="active", last_activity=None, runs=None):
    return {
        "threadId": "1", "title": "T", "status": status, "repo": "a/b",
        "description": "", "tags": [], "links": [], "notes": "",
        "prs": [], "branches": [], "runs": runs or [],
        "lastActivityAt": last_activity, "lastActivitySource": "manual" if last_activity else None,
        "createdAt": "2026-01-01T00:00:00Z", "updatedAt": "2026-01-01T00:00:00Z",
    }


class TestSilentInfo:
    def test_just_under_threshold_not_silent(self):
        now = ts._parse_iso("2026-07-27T12:00:00Z")
        last = "2026-07-27T11:31:00Z"  # 29 minutes ago
        entry = _entry_for_silent(status="active", last_activity=last)
        silent, silent_for, stalled = ts._silent_info(entry, 30 * 60, ["active", "review"], now=now)
        assert silent is False

    def test_just_over_threshold_is_silent(self):
        now = ts._parse_iso("2026-07-27T12:00:00Z")
        last = "2026-07-27T11:29:00Z"  # 31 minutes ago
        entry = _entry_for_silent(status="active", last_activity=last)
        silent, silent_for, stalled = ts._silent_info(entry, 30 * 60, ["active", "review"], now=now)
        assert silent is True
        assert silent_for == pytest.approx(31 * 60, abs=1)

    def test_null_last_activity_treated_as_silent(self):
        now = ts._parse_iso("2026-07-27T12:00:00Z")
        entry = _entry_for_silent(status="active", last_activity=None)
        silent, silent_for, stalled = ts._silent_info(entry, 30 * 60, ["active", "review"], now=now)
        assert silent is True
        assert silent_for is None  # distinct from a known age

    def test_excluded_status_never_silent(self):
        now = ts._parse_iso("2026-07-27T12:00:00Z")
        entry = _entry_for_silent(status="merged", last_activity=None)
        silent, silent_for, stalled = ts._silent_info(entry, 30 * 60, ["active", "review"], now=now)
        assert silent is False

    def test_status_outside_include_list_not_silent(self):
        now = ts._parse_iso("2026-07-27T12:00:00Z")
        entry = _entry_for_silent(status="blocked", last_activity=None)
        silent, silent_for, stalled = ts._silent_info(entry, 30 * 60, ["active", "review"], now=now)
        assert silent is False

    def test_running_with_stale_logmtime_is_stalled(self):
        now = ts._parse_iso("2026-07-27T12:00:00Z")
        run = {"host": "vibes", "name": "x", "state": {**dict(ts.EMPTY_RUN_STATE), "status": "running", "logMtime": "2026-07-27T11:00:00Z"}}
        entry = _entry_for_silent(status="active", last_activity="2026-07-27T11:59:00Z", runs=[run])
        silent, silent_for, stalled = ts._silent_info(entry, 30 * 60, ["active", "review"], now=now)
        assert stalled is True

    def test_running_with_fresh_logmtime_not_stalled(self):
        now = ts._parse_iso("2026-07-27T12:00:00Z")
        run = {"host": "vibes", "name": "x", "state": {**dict(ts.EMPTY_RUN_STATE), "status": "running", "logMtime": "2026-07-27T11:59:00Z"}}
        entry = _entry_for_silent(status="active", last_activity="2026-07-27T11:59:00Z", runs=[run])
        silent, silent_for, stalled = ts._silent_info(entry, 30 * 60, ["active", "review"], now=now)
        assert stalled is False

    def test_done_run_never_stalled_regardless_of_logmtime(self):
        now = ts._parse_iso("2026-07-27T12:00:00Z")
        run = {"host": "vibes", "name": "x", "state": {**dict(ts.EMPTY_RUN_STATE), "status": "done", "logMtime": "2026-07-27T01:00:00Z"}}
        entry = _entry_for_silent(status="active", last_activity=None, runs=[run])
        silent, silent_for, stalled = ts._silent_info(entry, 30 * 60, ["active", "review"], now=now)
        assert stalled is False


class TestRunIsStalled:
    def test_running_stale_logmtime(self):
        now = ts._parse_iso("2026-07-27T12:00:00Z")
        run = {"state": {"status": "running", "logMtime": "2026-07-27T11:00:00Z"}}
        assert ts.run_is_stalled(run, 15 * 60, now=now) is True

    def test_running_fresh_logmtime(self):
        now = ts._parse_iso("2026-07-27T12:00:00Z")
        run = {"state": {"status": "running", "logMtime": "2026-07-27T11:59:00Z"}}
        assert ts.run_is_stalled(run, 15 * 60, now=now) is False

    def test_not_running_never_stalled(self):
        now = ts._parse_iso("2026-07-27T12:00:00Z")
        run = {"state": {"status": "done", "logMtime": "2026-07-27T01:00:00Z"}}
        assert ts.run_is_stalled(run, 15 * 60, now=now) is False

    def test_running_without_logmtime_not_stalled(self):
        now = ts._parse_iso("2026-07-27T12:00:00Z")
        run = {"state": {"status": "running", "logMtime": None}}
        assert ts.run_is_stalled(run, 15 * 60, now=now) is False


class TestCliSilentAndListFiltering:
    def _seed_silent(self, registry_path):
        data = ts._empty_registry()
        data["threads"]["silent1"] = _entry_for_silent(
            status="active", last_activity="2020-01-01T00:00:00Z",
        )
        data["threads"]["silent1"]["threadId"] = "silent1"
        data["threads"]["fresh1"] = _entry_for_silent(
            status="active", last_activity=ts._now_iso(),
        )
        data["threads"]["fresh1"]["threadId"] = "fresh1"
        data["threads"]["done1"] = _entry_for_silent(
            status="done", last_activity="2020-01-01T00:00:00Z",
        )
        data["threads"]["done1"]["threadId"] = "done1"
        ts._write_registry(registry_path, data)

    def test_list_silent_for_filters_to_silent_entries_only(self, registry_path, capsys):
        self._seed_silent(registry_path)
        _, out, _ = run_cli(["list", "--silent-for", "30m", "--json"], registry_path, capsys)
        entries = json.loads(out)
        ids = {e["threadId"] for e in entries}
        assert ids == {"silent1"}

    def test_list_silent_for_json_includes_silentforseconds_and_stalled(self, registry_path, capsys):
        self._seed_silent(registry_path)
        _, out, _ = run_cli(["list", "--silent-for", "30m", "--json"], registry_path, capsys)
        entries = json.loads(out)
        assert "silentForSeconds" in entries[0]
        assert "stalled" in entries[0]

    def test_list_silent_for_human_output_has_silent_column(self, registry_path, capsys):
        self._seed_silent(registry_path)
        _, out, _ = run_cli(["list", "--silent-for", "30m"], registry_path, capsys)
        assert "SILENT" in out

    def test_silent_alias_command_matches_list_silent_for(self, registry_path, capsys):
        self._seed_silent(registry_path)
        _, out1, _ = run_cli(["list", "--silent-for", "30m", "--json"], registry_path, capsys)
        _, out2, _ = run_cli(["silent", "--for", "30m", "--json"], registry_path, capsys)
        entries1, entries2 = json.loads(out1), json.loads(out2)
        assert [e["threadId"] for e in entries1] == [e["threadId"] for e in entries2]
        assert [e["stalled"] for e in entries1] == [e["stalled"] for e in entries2]
        # silentForSeconds is computed against "now" independently on each
        # call -- assert closeness rather than exact equality.
        for e1, e2 in zip(entries1, entries2):
            assert abs(e1["silentForSeconds"] - e2["silentForSeconds"]) < 5

    def test_silent_alias_default_duration_is_30m(self, registry_path, capsys):
        self._seed_silent(registry_path)
        code, out, err = run_cli(["silent", "--json"], registry_path, capsys)
        assert code == 0
        entries = json.loads(out)
        ids = {e["threadId"] for e in entries}
        assert ids == {"silent1"}

    def test_stalled_flag_requires_running_run_with_stale_logmtime(self, registry_path, capsys):
        data = ts._empty_registry()
        stalled_run = {"host": "vibes", "name": "x", "state": {**dict(ts.EMPTY_RUN_STATE), "status": "running", "logMtime": "2020-01-01T00:00:00Z"}}
        e1 = _entry_for_silent(status="active", last_activity="2020-01-01T00:00:00Z", runs=[stalled_run])
        e1["threadId"] = "stalled1"
        e2 = _entry_for_silent(status="active", last_activity="2020-01-01T00:00:00Z")
        e2["threadId"] = "silentonly"
        data["threads"]["stalled1"] = e1
        data["threads"]["silentonly"] = e2
        ts._write_registry(registry_path, data)

        _, out, _ = run_cli(["list", "--silent-for", "30m", "--stalled", "--json"], registry_path, capsys)
        entries = json.loads(out)
        assert {e["threadId"] for e in entries} == {"stalled1"}

    def test_include_status_override_widens_silent_set(self, registry_path, capsys):
        data = ts._empty_registry()
        e = _entry_for_silent(status="blocked", last_activity="2020-01-01T00:00:00Z")
        e["threadId"] = "b1"
        data["threads"]["b1"] = e
        ts._write_registry(registry_path, data)

        _, out, _ = run_cli(["list", "--silent-for", "30m", "--json"], registry_path, capsys)
        assert json.loads(out) == []

        _, out2, _ = run_cli(["list", "--silent-for", "30m", "--include-status", "blocked", "--json"], registry_path, capsys)
        assert {e["threadId"] for e in json.loads(out2)} == {"b1"}

    def test_include_status_cannot_reintroduce_never_silent_statuses(self, registry_path, capsys):
        data = ts._empty_registry()
        e = _entry_for_silent(status="merged", last_activity="2020-01-01T00:00:00Z")
        e["threadId"] = "m1"
        data["threads"]["m1"] = e
        ts._write_registry(registry_path, data)

        _, out, _ = run_cli(["list", "--silent-for", "30m", "--include-status", "merged", "--json"], registry_path, capsys)
        assert json.loads(out) == []

    def test_invalid_silent_for_duration_errors(self, registry_path, capsys):
        code, out, err = run_cli(["list", "--silent-for", "bogus"], registry_path, capsys)
        assert code == 2


# ---------------------------------------------------------------------------
# card rendering with runs
# ---------------------------------------------------------------------------

class TestRenderCardRuns:
    def test_run_line_shows_host_name_and_status(self):
        entry = {
            "title": "T", "status": "active", "prs": [], "branches": [],
            "runs": [{"host": "vibes", "name": "tstate2", "state": {**dict(ts.EMPTY_RUN_STATE), "status": "running"}}],
        }
        out = ts.render_card_markdown(entry)
        assert "run: vibes/tstate2 running" in out

    def test_run_line_shows_log_age(self):
        now = ts._parse_iso("2026-07-27T12:00:00Z")
        state = {**dict(ts.EMPTY_RUN_STATE), "status": "running", "logMtime": "2026-07-27T11:56:00Z"}
        run = {"host": "vibes", "name": "tstate2", "state": state}
        line = ts._run_line(run, now=now)
        assert "log 4m ago" in line

    def test_run_line_shows_stall_warning_when_stalled(self):
        now = ts._parse_iso("2026-07-27T12:00:00Z")
        state = {**dict(ts.EMPTY_RUN_STATE), "status": "running", "logMtime": "2026-07-27T11:00:00Z"}
        run = {"host": "vibes", "name": "tstate2", "state": state}
        line = ts._run_line(run, now=now)
        assert "⚠️ no output for" in line

    def test_run_line_no_stall_warning_when_fresh(self):
        now = ts._parse_iso("2026-07-27T12:00:00Z")
        state = {**dict(ts.EMPTY_RUN_STATE), "status": "running", "logMtime": "2026-07-27T11:59:00Z"}
        run = {"host": "vibes", "name": "tstate2", "state": state}
        line = ts._run_line(run, now=now)
        assert "⚠️" not in line

    def test_run_line_no_stall_warning_when_done(self):
        now = ts._parse_iso("2026-07-27T12:00:00Z")
        state = {**dict(ts.EMPTY_RUN_STATE), "status": "done", "logMtime": "2026-07-27T01:00:00Z"}
        run = {"host": "vibes", "name": "tstate2", "state": state}
        line = ts._run_line(run, now=now)
        assert "⚠️" not in line


# ---------------------------------------------------------------------------
# probe command: SSH output parsing (macOS + Linux stat), failure isolation
# ---------------------------------------------------------------------------

class TestProbeRun:
    def test_running_task_parses_status_exit_bytes_mtime(self):
        def fake_runner(cmd, timeout):
            stdout = "STATUS:running\nEXIT:\nBYTES:1234\nMTIME:1785315257\n"
            return subprocess.CompletedProcess(cmd, 0, stdout, "")

        state = ts.probe_run("vibes", "tstate2", 8, fake_runner)
        assert state["status"] == "running"
        assert state["exitCode"] is None
        assert state["logBytes"] == 1234
        assert state["logMtime"] is not None
        assert state["stale"] is False
        assert state["error"] is None

    def test_macos_stat_dialect_epoch_parses_same_as_linux(self):
        # Both `stat -f %m` (macOS) and `stat -c %Y` (Linux) print a bare
        # epoch integer -- probe_run's parsing doesn't care which produced
        # it, only that MTIME is a number.
        def fake_runner_mac(cmd, timeout):
            return subprocess.CompletedProcess(cmd, 0, "STATUS:done\nEXIT:0\nBYTES:99\nMTIME:1785315257\n", "")

        def fake_runner_linux(cmd, timeout):
            return subprocess.CompletedProcess(cmd, 0, "STATUS:done\nEXIT:0\nBYTES:99\nMTIME:1785315257\n", "")

        state_mac = ts.probe_run("vibes", "x", 8, fake_runner_mac)
        state_linux = ts.probe_run("macmini", "x", 8, fake_runner_linux)
        assert state_mac["logMtime"] == state_linux["logMtime"]
        assert state_mac["exitCode"] == 0

    def test_missing_status_file_reports_missing(self):
        def fake_runner(cmd, timeout):
            stdout = f"STATUS:{ts.PROBE_MISSING_MARKER}\nEXIT:\nBYTES:\nMTIME:\n"
            return subprocess.CompletedProcess(cmd, 0, stdout, "")

        state = ts.probe_run("vibes", "gone", 8, fake_runner)
        assert state["status"] == "missing"
        assert state["exitCode"] is None
        assert state["error"] is None

    def test_ssh_timeout_marks_stale_never_raises(self):
        def fake_runner(cmd, timeout):
            raise subprocess.TimeoutExpired(cmd, timeout)

        state = ts.probe_run("vibes", "x", 8, fake_runner)
        assert state["stale"] is True
        assert "timed out" in state["error"]

    def test_ssh_oserror_marks_stale_never_raises(self):
        def fake_runner(cmd, timeout):
            raise OSError("ssh not found")

        state = ts.probe_run("vibes", "x", 8, fake_runner)
        assert state["stale"] is True
        assert "ssh not found" in state["error"]

    def test_ssh_nonzero_exit_marks_stale(self):
        def fake_runner(cmd, timeout):
            return subprocess.CompletedProcess(cmd, 255, "", "Connection refused")

        state = ts.probe_run("vibes", "x", 8, fake_runner)
        assert state["stale"] is True
        assert "Connection refused" in state["error"]

    def test_probe_command_tries_both_stat_dialects(self):
        cmd = ts._build_probe_command("tstate2")
        assert "stat -f %m" in cmd
        assert "stat -c %Y" in cmd


class TestCliProbe:
    def test_probe_updates_run_state_from_ssh(self, registry_path, capsys, monkeypatch):
        run_cli(["add", "1", "--title", "T", "--add-run", "vibes:tstate2"], registry_path, capsys)

        def fake_runner(cmd, timeout):
            stdout = "STATUS:running\nEXIT:\nBYTES:42\nMTIME:1785315257\n"
            return subprocess.CompletedProcess(cmd, 0, stdout, "")

        monkeypatch.setattr(ts, "_run_command", fake_runner)
        code, out, err = run_cli(["probe", "1"], registry_path, capsys)
        assert code == 0

        _, out2, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out2)
        assert entry["runs"][0]["state"]["status"] == "running"
        assert entry["runs"][0]["state"]["logBytes"] == 42

    def test_probe_per_run_failure_isolated_others_still_update(self, registry_path, capsys, monkeypatch):
        run_cli(["add", "1", "--title", "T", "--add-run", "vibes:a", "--add-run", "macmini:b"], registry_path, capsys)

        def flaky_runner(cmd, timeout):
            if "vibes" in cmd:
                raise subprocess.TimeoutExpired(cmd, timeout)
            return subprocess.CompletedProcess(cmd, 0, "STATUS:done\nEXIT:0\nBYTES:5\nMTIME:1785315257\n", "")

        monkeypatch.setattr(ts, "_run_command", flaky_runner)
        code, out, err = run_cli(["probe", "1"], registry_path, capsys)

        _, out2, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out2)
        by_host = {r["host"]: r for r in entry["runs"]}
        assert by_host["vibes"]["state"]["stale"] is True
        assert by_host["vibes"]["state"]["error"]
        assert by_host["macmini"]["state"]["stale"] is False
        assert by_host["macmini"]["state"]["status"] == "done"

    def test_probe_never_crashes_and_preserves_cache_on_failure(self, registry_path, capsys, monkeypatch):
        run_cli(["add", "1", "--title", "T", "--add-run", "vibes:a"], registry_path, capsys)
        data = ts._parse_registry(registry_path)
        data["threads"]["1"]["runs"][0]["state"] = {
            "status": "running", "exitCode": None, "logBytes": 999, "logMtime": "2026-01-01T00:00:00Z",
            "checkedAt": "2026-01-01T00:00:00Z", "stale": False, "error": None,
        }
        ts._write_registry(registry_path, data)

        def crashy_runner(cmd, timeout):
            raise OSError("ssh binary missing")

        monkeypatch.setattr(ts, "_run_command", crashy_runner)
        code, out, err = run_cli(["probe", "--all"], registry_path, capsys)
        assert code == 0  # --all always exits zero regardless of per-thread errors

        data = ts._parse_registry(registry_path)
        state = data["threads"]["1"]["runs"][0]["state"]
        assert state["logBytes"] == 999  # cache never wiped
        assert state["stale"] is True

    def test_probe_unknown_thread_id_errors(self, registry_path, capsys):
        ts._write_registry(registry_path, ts._empty_registry())
        code, out, err = run_cli(["probe", "ghost"], registry_path, capsys)
        assert code == 2
        assert "unknown" in err

    def test_probe_skips_entries_without_runs(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T"], registry_path, capsys)
        code, out, err = run_cli(["probe", "--all", "--json"], registry_path, capsys)
        assert code == 0
        results = json.loads(out)
        assert results[0]["ok"] is False
        assert "skipped" in results[0]["error"]


# ---------------------------------------------------------------------------
# v3 -> v4 schema migration: `updates` list
# ---------------------------------------------------------------------------

class TestSchemaMigrationV4:
    def _write_v3(self, registry_path):
        raw = {
            "schemaVersion": 3,
            "accounts": {},
            "threads": {
                "1": {
                    "threadId": "1",
                    "title": "mSPRT",
                    "description": "",
                    "repo": "acme/widgets",
                    "status": "review",
                    "tags": [],
                    "links": [],
                    "notes": "",
                    "prs": [],
                    "branches": [],
                    "runs": [],
                    "lastActivityAt": None,
                    "lastActivitySource": None,
                    "createdAt": "2026-01-01T00:00:00Z",
                    "updatedAt": "2026-01-01T00:00:00Z",
                }
            },
        }
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(json.dumps(raw))
        return raw

    def test_migrate_entry_v3_to_v4_adds_empty_updates_list(self):
        entry = {"title": "T"}
        ts._migrate_entry_v3_to_v4(entry)
        assert entry["updates"] == []

    def test_migrate_entry_v3_to_v4_is_idempotent_and_preserves_existing(self):
        entry = {"title": "T", "updates": [{"ts": "x", "text": "hi", "kind": "manual"}]}
        ts._migrate_entry_v3_to_v4(entry)
        assert entry["updates"] == [{"ts": "x", "text": "hi", "kind": "manual"}]

    def test_load_and_migrate_v3_to_v4_bumps_schema_and_backs_up(self, registry_path):
        self._write_v3(registry_path)
        data, migrated = ts._load_and_migrate(registry_path)
        assert migrated is True
        assert data["schemaVersion"] == 5
        assert data["threads"]["1"]["updates"] == []
        # No data loss: everything else survives untouched.
        assert data["threads"]["1"]["title"] == "mSPRT"
        assert data["threads"]["1"]["repo"] == "acme/widgets"

    def test_write_from_v3_triggers_v3_backup_once(self, registry_path, capsys):
        self._write_v3(registry_path)
        code, out, err = run_cli(["set", "1", "--desc", "touched"], registry_path, capsys)
        assert code == 0

        on_disk = json.loads(registry_path.read_text())
        assert on_disk["schemaVersion"] == 5
        assert on_disk["threads"]["1"]["updates"] == []

        backups = list(registry_path.parent.glob("registry.json.v3-backup-*"))
        assert len(backups) == 1
        backed_up = json.loads(backups[0].read_text())
        assert backed_up["schemaVersion"] == 3
        assert "updates" not in backed_up["threads"]["1"]

    def test_migration_is_idempotent_on_reload(self, registry_path):
        self._write_v3(registry_path)
        data1, migrated1 = ts._load_and_migrate(registry_path)
        assert migrated1 is True
        ts._write_registry(registry_path, data1)
        data2, migrated2 = ts._load_and_migrate(registry_path)
        assert migrated2 is False
        assert data2["threads"]["1"]["updates"] == []


# ---------------------------------------------------------------------------
# v4 -> v5 schema migration: `cardMessageId`/`parentChannelId`/`discordName`
# ---------------------------------------------------------------------------

class TestSchemaMigrationV5:
    def _write_v4(self, registry_path):
        raw = {
            "schemaVersion": 4,
            "accounts": {},
            "threads": {
                "1": {
                    "threadId": "1",
                    "title": "mSPRT",
                    "description": "",
                    "repo": "acme/widgets",
                    "status": "review",
                    "tags": [],
                    "links": [],
                    "notes": "",
                    "prs": [],
                    "branches": [],
                    "runs": [],
                    "lastActivityAt": None,
                    "lastActivitySource": None,
                    "updates": [],
                    "createdAt": "2026-01-01T00:00:00Z",
                    "updatedAt": "2026-01-01T00:00:00Z",
                }
            },
        }
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(json.dumps(raw))
        return raw

    def test_migrate_entry_v4_to_v5_adds_null_defaults(self):
        entry = {"title": "T"}
        ts._migrate_entry_v4_to_v5(entry)
        assert entry["cardMessageId"] is None
        assert entry["parentChannelId"] is None
        assert entry["discordName"] is None

    def test_migrate_entry_v4_to_v5_is_idempotent_and_preserves_existing(self):
        entry = {
            "title": "T",
            "cardMessageId": "123",
            "parentChannelId": "456",
            "discordName": "old-name",
        }
        ts._migrate_entry_v4_to_v5(entry)
        assert entry["cardMessageId"] == "123"
        assert entry["parentChannelId"] == "456"
        assert entry["discordName"] == "old-name"

    def test_load_and_migrate_v4_to_v5_bumps_schema_and_backs_up(self, registry_path):
        self._write_v4(registry_path)
        data, migrated = ts._load_and_migrate(registry_path)
        assert migrated is True
        assert data["schemaVersion"] == 5
        entry = data["threads"]["1"]
        assert entry["cardMessageId"] is None
        assert entry["parentChannelId"] is None
        assert entry["discordName"] is None
        # No data loss: everything else survives untouched.
        assert entry["title"] == "mSPRT"
        assert entry["repo"] == "acme/widgets"

    def test_write_from_v4_triggers_v4_backup_once(self, registry_path, capsys):
        self._write_v4(registry_path)
        code, out, err = run_cli(["set", "1", "--desc", "touched"], registry_path, capsys)
        assert code == 0

        on_disk = json.loads(registry_path.read_text())
        assert on_disk["schemaVersion"] == 5
        assert on_disk["threads"]["1"]["cardMessageId"] is None

        backups = list(registry_path.parent.glob("registry.json.v4-backup-*"))
        assert len(backups) == 1
        backed_up = json.loads(backups[0].read_text())
        assert backed_up["schemaVersion"] == 4
        assert "cardMessageId" not in backed_up["threads"]["1"]

    def test_migration_is_idempotent_on_reload(self, registry_path):
        self._write_v4(registry_path)
        data1, migrated1 = ts._load_and_migrate(registry_path)
        assert migrated1 is True
        ts._write_registry(registry_path, data1)
        data2, migrated2 = ts._load_and_migrate(registry_path)
        assert migrated2 is False
        assert data2["threads"]["1"]["cardMessageId"] is None


# ---------------------------------------------------------------------------
# updates: append/cap/truncate/dedupe/clear helpers
# ---------------------------------------------------------------------------

class TestUpdatesHelpers:
    def _entry(self):
        return {"title": "T", "updates": []}

    def test_append_update_adds_entry_with_kind_and_ts(self):
        entry = self._entry()
        ts.append_update(entry, "did a thing", "manual", ts="2026-01-01T00:00:00Z")
        assert entry["updates"] == [{"ts": "2026-01-01T00:00:00Z", "text": "did a thing", "kind": "manual"}]

    def test_append_update_defaults_ts_to_now(self):
        entry = self._entry()
        ts.append_update(entry, "did a thing", "manual")
        assert entry["updates"][0]["ts"]  # non-empty, filled by _now_iso()

    def test_append_update_caps_at_ten_evicting_oldest(self):
        entry = self._entry()
        for i in range(12):
            ts.append_update(entry, f"entry {i}", "manual", ts=f"2026-01-01T00:00:{i:02d}Z")
        assert len(entry["updates"]) == 10
        texts = [u["text"] for u in entry["updates"]]
        # Oldest two (entry 0, entry 1) evicted; newest-last order preserved.
        assert texts == [f"entry {i}" for i in range(2, 12)]

    def test_append_update_truncates_at_120_chars_with_ellipsis(self):
        entry = self._entry()
        long_text = "x" * 200
        ts.append_update(entry, long_text, "manual")
        text = entry["updates"][0]["text"]
        assert len(text) == 120
        assert text.endswith("…")
        assert text[:119] == "x" * 119

    def test_append_update_exactly_120_chars_not_truncated(self):
        entry = self._entry()
        text = "x" * 120
        ts.append_update(entry, text, "manual")
        assert entry["updates"][0]["text"] == text

    def test_append_update_strips_newlines_to_spaces(self):
        entry = self._entry()
        ts.append_update(entry, "line one\nline two\r\nline three", "manual")
        assert entry["updates"][0]["text"] == "line one line two line three"

    def test_append_update_collapses_internal_whitespace(self):
        entry = self._entry()
        ts.append_update(entry, "a   b\t\tc", "manual")
        assert entry["updates"][0]["text"] == "a b c"

    def test_append_update_rejects_empty_text(self):
        entry = self._entry()
        with pytest.raises(ValueError):
            ts.append_update(entry, "", "manual")
        assert entry["updates"] == []

    def test_append_update_rejects_whitespace_only_text(self):
        entry = self._entry()
        with pytest.raises(ValueError):
            ts.append_update(entry, "   \n\t  ", "manual")
        assert entry["updates"] == []

    def test_append_update_dedupes_identical_back_to_back_text(self):
        entry = self._entry()
        ts.append_update(entry, "same text", "manual", ts="2026-01-01T00:00:00Z")
        ts.append_update(entry, "same text", "transition", ts="2026-01-01T00:01:00Z")
        assert len(entry["updates"]) == 1
        assert entry["updates"][0]["ts"] == "2026-01-01T00:00:00Z"

    def test_append_update_does_not_dedupe_non_adjacent_repeats(self):
        entry = self._entry()
        ts.append_update(entry, "A", "manual", ts="2026-01-01T00:00:00Z")
        ts.append_update(entry, "B", "manual", ts="2026-01-01T00:01:00Z")
        ts.append_update(entry, "A", "manual", ts="2026-01-01T00:02:00Z")
        assert [u["text"] for u in entry["updates"]] == ["A", "B", "A"]

    def test_get_updates_on_missing_key_returns_empty_list(self):
        assert ts.get_updates({"title": "T"}) == []

    def test_clear_updates_wipes_list(self):
        entry = self._entry()
        ts.append_update(entry, "hi", "manual")
        ts.clear_updates(entry)
        assert entry["updates"] == []


# ---------------------------------------------------------------------------
# CLI: --progress, --clear-updates, `log`, `list --with-last`
# ---------------------------------------------------------------------------

class TestCliUpdates:
    def test_set_progress_appends_manual_entry(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T"], registry_path, capsys)
        code, out, err = run_cli(["set", "1", "--progress", "shipped the fix"], registry_path, capsys)
        assert code == 0
        _, out2, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out2)
        assert entry["updates"][-1]["text"] == "shipped the fix"
        assert entry["updates"][-1]["kind"] == "manual"

    def test_set_progress_repeatable_in_one_invocation(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T"], registry_path, capsys)
        run_cli(["set", "1", "--progress", "first", "--progress", "second"], registry_path, capsys)
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        assert [u["text"] for u in entry["updates"]] == ["first", "second"]

    def test_set_progress_empty_text_errors_and_does_not_write(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T"], registry_path, capsys)
        code, out, err = run_cli(["set", "1", "--progress", "   "], registry_path, capsys)
        assert code == 2
        _, out2, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out2)
        assert entry["updates"] == []

    def test_set_clear_updates_wipes_list(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T"], registry_path, capsys)
        run_cli(["set", "1", "--progress", "a"], registry_path, capsys)
        code, out, err = run_cli(["set", "1", "--clear-updates"], registry_path, capsys)
        assert code == 0
        _, out2, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out2)
        assert entry["updates"] == []

    def test_set_clear_updates_then_progress_in_same_invocation_keeps_new(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T"], registry_path, capsys)
        run_cli(["set", "1", "--progress", "old"], registry_path, capsys)
        run_cli(["set", "1", "--clear-updates", "--progress", "new"], registry_path, capsys)
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        assert [u["text"] for u in entry["updates"]] == ["new"]

    def test_log_prints_newest_first(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T"], registry_path, capsys)
        run_cli(["set", "1", "--progress", "first", "--progress", "second", "--progress", "third"], registry_path, capsys)
        code, out, err = run_cli(["log", "1"], registry_path, capsys)
        assert code == 0
        lines = [l for l in out.splitlines() if l.strip()]
        assert "third" in lines[0]
        assert "second" in lines[1]
        assert "first" in lines[2]

    def test_log_respects_limit(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T"], registry_path, capsys)
        for i in range(5):
            run_cli(["set", "1", "--progress", f"entry {i}"], registry_path, capsys)
        code, out, err = run_cli(["log", "1", "--limit", "2"], registry_path, capsys)
        assert code == 0
        lines = [l for l in out.splitlines() if l.strip()]
        assert len(lines) == 2
        assert "entry 4" in lines[0]
        assert "entry 3" in lines[1]

    def test_log_default_limit_is_ten(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T"], registry_path, capsys)
        for i in range(15):
            run_cli(["set", "1", "--progress", f"entry {i}"], registry_path, capsys)
        code, out, err = run_cli(["log", "1"], registry_path, capsys)
        lines = [l for l in out.splitlines() if l.strip()]
        assert len(lines) == 10  # cap already limits underlying storage to 10

    def test_log_json_output(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T"], registry_path, capsys)
        run_cli(["set", "1", "--progress", "a"], registry_path, capsys)
        code, out, err = run_cli(["log", "1", "--json"], registry_path, capsys)
        assert code == 0
        data = json.loads(out)
        assert data[0]["text"] == "a"
        assert data[0]["kind"] == "manual"

    def test_log_empty_updates_human_message(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T"], registry_path, capsys)
        code, out, err = run_cli(["log", "1"], registry_path, capsys)
        assert code == 0
        assert "no updates" in out

    def test_log_unknown_thread_id_errors(self, registry_path, capsys):
        ts._write_registry(registry_path, ts._empty_registry())
        code, out, err = run_cli(["log", "ghost"], registry_path, capsys)
        assert code == 1
        assert "no such thread" in err

    def test_list_with_last_off_by_default(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T"], registry_path, capsys)
        run_cli(["set", "1", "--progress", "did the thing"], registry_path, capsys)
        code, out, err = run_cli(["list"], registry_path, capsys)
        assert "LAST" not in out
        assert "did the thing" not in out

    def test_list_with_last_shows_trailing_column(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T"], registry_path, capsys)
        run_cli(["set", "1", "--progress", "did the thing"], registry_path, capsys)
        code, out, err = run_cli(["list", "--with-last"], registry_path, capsys)
        assert "LAST" in out
        assert "did the thing" in out

    def test_list_with_last_shows_dash_when_no_updates(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T"], registry_path, capsys)
        code, out, err = run_cli(["list", "--with-last"], registry_path, capsys)
        lines = [l for l in out.splitlines() if l.strip()]
        assert lines[1].rstrip().endswith("-")


# ---------------------------------------------------------------------------
# automatic transition detection: PR state/CI, thread status, run state
# ---------------------------------------------------------------------------

class TestPrTransitionTexts:
    def test_before_none_never_synthesizes(self):
        after = {**dict(ts.EMPTY_PR_STATE), "state": "OPEN"}
        assert ts._pr_transition_texts(None, after, 4712) == []

    def test_first_fetch_open_reports_opened(self):
        before = dict(ts.EMPTY_PR_STATE)
        after = {**dict(ts.EMPTY_PR_STATE), "state": "OPEN"}
        assert ts._pr_transition_texts(before, after, 4712) == ["PR #4712 opened"]

    def test_open_to_merged(self):
        before = {**dict(ts.EMPTY_PR_STATE), "state": "OPEN"}
        after = {**dict(ts.EMPTY_PR_STATE), "state": "MERGED"}
        assert ts._pr_transition_texts(before, after, 4694) == ["PR #4694 merged"]

    def test_open_to_closed(self):
        before = {**dict(ts.EMPTY_PR_STATE), "state": "OPEN"}
        after = {**dict(ts.EMPTY_PR_STATE), "state": "CLOSED"}
        assert ts._pr_transition_texts(before, after, 4700) == ["PR #4700 closed"]

    def test_closed_to_open_is_reopened(self):
        before = {**dict(ts.EMPTY_PR_STATE), "state": "CLOSED"}
        after = {**dict(ts.EMPTY_PR_STATE), "state": "OPEN"}
        assert ts._pr_transition_texts(before, after, 4700) == ["PR #4700 reopened"]

    def test_ci_success_to_failure(self):
        before = {**dict(ts.EMPTY_PR_STATE), "state": "OPEN", "ci": "SUCCESS"}
        after = {**dict(ts.EMPTY_PR_STATE), "state": "OPEN", "ci": "FAILURE"}
        assert ts._pr_transition_texts(before, after, 4700) == ["CI failed on PR #4700"]

    def test_ci_failure_recovers_to_success(self):
        before = {**dict(ts.EMPTY_PR_STATE), "state": "OPEN", "ci": "FAILURE"}
        after = {**dict(ts.EMPTY_PR_STATE), "state": "OPEN", "ci": "SUCCESS"}
        assert ts._pr_transition_texts(before, after, 4700) == ["CI green on PR #4700"]

    def test_ci_pending_to_success_is_not_reported(self):
        before = {**dict(ts.EMPTY_PR_STATE), "state": "OPEN", "ci": "PENDING"}
        after = {**dict(ts.EMPTY_PR_STATE), "state": "OPEN", "ci": "SUCCESS"}
        assert ts._pr_transition_texts(before, after, 4700) == []

    def test_no_change_reports_nothing(self):
        state = {**dict(ts.EMPTY_PR_STATE), "state": "OPEN", "ci": "SUCCESS"}
        assert ts._pr_transition_texts(dict(state), dict(state), 4700) == []

    def test_state_and_ci_change_together_state_first(self):
        before = {**dict(ts.EMPTY_PR_STATE), "state": "OPEN", "ci": "FAILURE"}
        after = {**dict(ts.EMPTY_PR_STATE), "state": "MERGED", "ci": "SUCCESS"}
        texts = ts._pr_transition_texts(before, after, 4700)
        assert texts == ["PR #4700 merged", "CI green on PR #4700"]


class TestRunTransitionTexts:
    def test_before_none_never_synthesizes(self):
        after = {**dict(ts.EMPTY_RUN_STATE), "status": "running"}
        assert ts._run_transition_texts(None, after, "vibes", "tstate2", 900) == []

    def test_missing_to_running_reports(self):
        before = dict(ts.EMPTY_RUN_STATE)
        after = {**dict(ts.EMPTY_RUN_STATE), "status": "running"}
        assert ts._run_transition_texts(before, after, "vibes", "tstate2", 900) == ["run vibes/tstate2 running"]

    def test_running_to_failed_reports_exit_code(self):
        before = {**dict(ts.EMPTY_RUN_STATE), "status": "running"}
        after = {**dict(ts.EMPTY_RUN_STATE), "status": "failed", "exitCode": 1}
        assert ts._run_transition_texts(before, after, "vibes", "tstate2", 900) == ["run vibes/tstate2 failed (exit 1)"]

    def test_running_to_done_reports_exit_code(self):
        before = {**dict(ts.EMPTY_RUN_STATE), "status": "running"}
        after = {**dict(ts.EMPTY_RUN_STATE), "status": "done", "exitCode": 0}
        assert ts._run_transition_texts(before, after, "macmini", "brruns", 900) == ["run macmini/brruns done (exit 0)"]

    def test_done_without_exit_code_omits_parens(self):
        before = {**dict(ts.EMPTY_RUN_STATE), "status": "running"}
        after = {**dict(ts.EMPTY_RUN_STATE), "status": "done", "exitCode": None}
        assert ts._run_transition_texts(before, after, "macmini", "brruns", 900) == ["run macmini/brruns done"]

    def test_no_status_change_reports_nothing(self):
        state = {**dict(ts.EMPTY_RUN_STATE), "status": "running"}
        assert ts._run_transition_texts(dict(state), dict(state), "vibes", "tstate2", 900) == []

    def test_going_stalled_reports_age(self):
        now = ts._parse_iso("2026-07-27T12:00:00Z")
        before = {**dict(ts.EMPTY_RUN_STATE), "status": "running", "logMtime": "2026-07-27T11:59:00Z"}
        after = {**dict(ts.EMPTY_RUN_STATE), "status": "running", "logMtime": "2026-07-27T11:19:00Z"}
        texts = ts._run_transition_texts(before, after, "vibes", "tstate2", 900, now=now)
        assert texts == ["run vibes/tstate2 stalled — no output for 41m"]

    def test_already_stalled_does_not_rereport(self):
        now = ts._parse_iso("2026-07-27T12:00:00Z")
        before = {**dict(ts.EMPTY_RUN_STATE), "status": "running", "logMtime": "2026-07-27T11:00:00Z"}
        after = {**dict(ts.EMPTY_RUN_STATE), "status": "running", "logMtime": "2026-07-27T11:00:00Z"}
        texts = ts._run_transition_texts(before, after, "vibes", "tstate2", 900, now=now)
        assert texts == []

    def test_status_change_and_stall_can_both_report(self):
        # A run transitions to failed AND was stalled just before failing --
        # both facts are worth a card line.
        now = ts._parse_iso("2026-07-27T12:00:00Z")
        before = {**dict(ts.EMPTY_RUN_STATE), "status": "running", "logMtime": "2026-07-27T11:00:00Z"}
        after = {**dict(ts.EMPTY_RUN_STATE), "status": "failed", "exitCode": 1, "logMtime": "2026-07-27T11:00:00Z"}
        texts = ts._run_transition_texts(before, after, "vibes", "tstate2", 900, now=now)
        assert texts[0] == "run vibes/tstate2 failed (exit 1)"


class TestStatusTransitionText:
    def test_status_change_reports(self):
        assert ts._status_transition_text("active", "blocked") == "status: active -> blocked"

    def test_no_change_reports_none(self):
        assert ts._status_transition_text("active", "active") is None

    def test_new_status_none_reports_none(self):
        assert ts._status_transition_text("active", None) is None


class TestRefreshProbeAppendTransitions:
    def _seed_pr_thread(self, registry_path, thread_id="1", pr=5, status="review", pr_state=None):
        data = ts._empty_registry()
        prs = [{"repo": None, "number": pr, "primary": True, "prState": pr_state or dict(ts.EMPTY_PR_STATE)}]
        data["threads"][thread_id] = {
            "threadId": thread_id, "title": "T", "description": "", "repo": "a/b",
            "branch": None, "status": status, "tags": [], "links": [], "notes": "",
            "prs": prs, "branches": [], "runs": [], "lastActivityAt": None, "lastActivitySource": None,
            "createdAt": "2026-01-01T00:00:00Z", "updatedAt": "2026-01-01T00:00:00Z", "updates": [],
        }
        ts._write_registry(registry_path, data)

    def test_refresh_appends_opened_transition_on_first_fetch(self, registry_path, capsys, monkeypatch):
        self._seed_pr_thread(registry_path, pr=4712, status="active")

        def ok_runner(cmd, timeout):
            if cmd[:2] == ["gh", "auth"]:
                return subprocess.CompletedProcess(cmd, 0, "", "")
            payload = {"state": "OPEN", "title": "t", "url": "u", "mergedAt": None, "statusCheckRollup": []}
            return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

        monkeypatch.setattr(ts, "_run_command", ok_runner)
        run_cli(["refresh", "1"], registry_path, capsys)
        data = ts._parse_registry(registry_path)
        texts = [u["text"] for u in data["threads"]["1"]["updates"]]
        assert "PR #4712 opened" in texts
        assert all(u["kind"] == "transition" for u in data["threads"]["1"]["updates"] if u["text"] == "PR #4712 opened")

    def test_refresh_appends_merged_transition(self, registry_path, capsys, monkeypatch):
        pr_state = {**dict(ts.EMPTY_PR_STATE), "state": "OPEN", "ci": "SUCCESS", "checkedAt": "2026-01-01T00:00:00Z"}
        self._seed_pr_thread(registry_path, pr=4694, status="review", pr_state=pr_state)

        def ok_runner(cmd, timeout):
            if cmd[:2] == ["gh", "auth"]:
                return subprocess.CompletedProcess(cmd, 0, "", "")
            payload = {"state": "MERGED", "title": "t", "url": "u", "mergedAt": "2026-01-02T00:00:00Z", "statusCheckRollup": [{"conclusion": "SUCCESS"}]}
            return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

        monkeypatch.setattr(ts, "_run_command", ok_runner)
        run_cli(["refresh", "1"], registry_path, capsys)
        data = ts._parse_registry(registry_path)
        texts = [u["text"] for u in data["threads"]["1"]["updates"]]
        assert "PR #4694 merged" in texts

    def test_refresh_appends_ci_failed_transition(self, registry_path, capsys, monkeypatch):
        pr_state = {**dict(ts.EMPTY_PR_STATE), "state": "OPEN", "ci": "SUCCESS", "checkedAt": "2026-01-01T00:00:00Z"}
        self._seed_pr_thread(registry_path, pr=4700, status="review", pr_state=pr_state)

        def ok_runner(cmd, timeout):
            if cmd[:2] == ["gh", "auth"]:
                return subprocess.CompletedProcess(cmd, 0, "", "")
            payload = {"state": "OPEN", "title": "t", "url": "u", "mergedAt": None, "statusCheckRollup": [{"conclusion": "FAILURE"}]}
            return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

        monkeypatch.setattr(ts, "_run_command", ok_runner)
        run_cli(["refresh", "1"], registry_path, capsys)
        data = ts._parse_registry(registry_path)
        texts = [u["text"] for u in data["threads"]["1"]["updates"]]
        assert "CI failed on PR #4700" in texts

    def test_refresh_appends_status_transition(self, registry_path, capsys, monkeypatch):
        pr_state = {**dict(ts.EMPTY_PR_STATE), "state": "OPEN", "ci": "SUCCESS", "checkedAt": "2026-01-01T00:00:00Z"}
        self._seed_pr_thread(registry_path, pr=4700, status="review", pr_state=pr_state)

        def ok_runner(cmd, timeout):
            if cmd[:2] == ["gh", "auth"]:
                return subprocess.CompletedProcess(cmd, 0, "", "")
            payload = {"state": "MERGED", "title": "t", "url": "u", "mergedAt": "2026-01-02T00:00:00Z", "statusCheckRollup": [{"conclusion": "SUCCESS"}]}
            return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

        monkeypatch.setattr(ts, "_run_command", ok_runner)
        run_cli(["refresh", "1"], registry_path, capsys)
        data = ts._parse_registry(registry_path)
        texts = [u["text"] for u in data["threads"]["1"]["updates"]]
        assert "status: review -> merged" in texts

    def test_refresh_no_transition_when_nothing_changed(self, registry_path, capsys, monkeypatch):
        pr_state = {**dict(ts.EMPTY_PR_STATE), "state": "OPEN", "ci": "SUCCESS", "checkedAt": "2026-01-01T00:00:00Z"}
        self._seed_pr_thread(registry_path, pr=4700, status="review", pr_state=pr_state)

        def ok_runner(cmd, timeout):
            if cmd[:2] == ["gh", "auth"]:
                return subprocess.CompletedProcess(cmd, 0, "", "")
            payload = {"state": "OPEN", "title": "t", "url": "u", "mergedAt": None, "statusCheckRollup": [{"conclusion": "SUCCESS"}]}
            return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

        monkeypatch.setattr(ts, "_run_command", ok_runner)
        run_cli(["refresh", "1"], registry_path, capsys)
        data = ts._parse_registry(registry_path)
        assert data["threads"]["1"]["updates"] == []

    def test_refresh_dedupes_repeated_status_transition_across_polls(self, registry_path, capsys, monkeypatch):
        pr_state = {**dict(ts.EMPTY_PR_STATE), "state": "OPEN", "ci": "SUCCESS", "checkedAt": "2026-01-01T00:00:00Z"}
        self._seed_pr_thread(registry_path, pr=4700, status="review", pr_state=pr_state)

        def merged_runner(cmd, timeout):
            if cmd[:2] == ["gh", "auth"]:
                return subprocess.CompletedProcess(cmd, 0, "", "")
            payload = {"state": "MERGED", "title": "t", "url": "u", "mergedAt": "2026-01-02T00:00:00Z", "statusCheckRollup": [{"conclusion": "SUCCESS"}]}
            return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

        monkeypatch.setattr(ts, "_run_command", merged_runner)
        run_cli(["refresh", "1"], registry_path, capsys)
        run_cli(["refresh", "1"], registry_path, capsys)  # second poll: already merged, no new status change
        data = ts._parse_registry(registry_path)
        status_texts = [u["text"] for u in data["threads"]["1"]["updates"] if u["text"].startswith("status:")]
        assert status_texts == ["status: review -> merged"]

    def _seed_run_thread(self, registry_path, thread_id="1", host="vibes", name="tstate2", state=None):
        data = ts._empty_registry()
        data["threads"][thread_id] = {
            "threadId": thread_id, "title": "T", "description": "", "repo": None,
            "branch": None, "status": "active", "tags": [], "links": [], "notes": "",
            "prs": [], "branches": [], "runs": [{"host": host, "name": name, "state": state or dict(ts.EMPTY_RUN_STATE)}],
            "lastActivityAt": None, "lastActivitySource": None,
            "createdAt": "2026-01-01T00:00:00Z", "updatedAt": "2026-01-01T00:00:00Z", "updates": [],
        }
        ts._write_registry(registry_path, data)

    def test_probe_appends_run_status_transition(self, registry_path, capsys, monkeypatch):
        self._seed_run_thread(registry_path)

        def fake_runner(cmd, timeout):
            return subprocess.CompletedProcess(cmd, 0, "STATUS:failed\nEXIT:1\nBYTES:10\nMTIME:1785315257\n", "")

        monkeypatch.setattr(ts, "_run_command", fake_runner)
        run_cli(["probe", "1"], registry_path, capsys)
        data = ts._parse_registry(registry_path)
        texts = [u["text"] for u in data["threads"]["1"]["updates"]]
        assert "run vibes/tstate2 failed (exit 1)" in texts

    def test_probe_no_transition_when_status_unchanged(self, registry_path, capsys, monkeypatch):
        state = {**dict(ts.EMPTY_RUN_STATE), "status": "running", "checkedAt": "2026-01-01T00:00:00Z"}
        self._seed_run_thread(registry_path, state=state)

        def fake_runner(cmd, timeout):
            return subprocess.CompletedProcess(cmd, 0, "STATUS:running\nEXIT:\nBYTES:99\nMTIME:1785315257\n", "")

        monkeypatch.setattr(ts, "_run_command", fake_runner)
        run_cli(["probe", "1"], registry_path, capsys)
        data = ts._parse_registry(registry_path)
        assert data["threads"]["1"]["updates"] == []


# ---------------------------------------------------------------------------
# card rendering: `recent:` block, relative ages, length-cap degradation
# ---------------------------------------------------------------------------

class TestRecentUpdatesLines:
    def test_no_updates_returns_empty(self):
        entry = {"title": "T", "updates": []}
        assert ts._recent_updates_lines(entry, 3) == []

    def test_relative_age_minutes(self):
        now = ts._parse_iso("2026-07-27T12:00:00Z")
        entry = {"title": "T", "updates": [{"ts": "2026-07-27T11:56:00Z", "text": "did a thing", "kind": "manual"}]}
        lines = ts._recent_updates_lines(entry, 3, now=now)
        assert lines == ["recent: 4m ago — did a thing"]

    def test_relative_age_hours(self):
        now = ts._parse_iso("2026-07-27T12:00:00Z")
        entry = {"title": "T", "updates": [{"ts": "2026-07-27T09:00:00Z", "text": "did a thing", "kind": "manual"}]}
        lines = ts._recent_updates_lines(entry, 3, now=now)
        assert lines == ["recent: 3h ago — did a thing"]

    def test_relative_age_days(self):
        now = ts._parse_iso("2026-07-27T12:00:00Z")
        entry = {"title": "T", "updates": [{"ts": "2026-07-24T12:00:00Z", "text": "did a thing", "kind": "manual"}]}
        lines = ts._recent_updates_lines(entry, 3, now=now)
        assert lines == ["recent: 3d ago — did a thing"]

    def test_newest_first_with_indented_continuation(self):
        now = ts._parse_iso("2026-07-27T12:00:00Z")
        entry = {
            "title": "T",
            "updates": [
                {"ts": "2026-07-27T11:00:00Z", "text": "oldest", "kind": "manual"},
                {"ts": "2026-07-27T11:30:00Z", "text": "middle", "kind": "manual"},
                {"ts": "2026-07-27T11:56:00Z", "text": "newest", "kind": "manual"},
            ],
        }
        lines = ts._recent_updates_lines(entry, 3, now=now)
        assert len(lines) == 3
        assert lines[0].startswith("recent: ")
        assert "newest" in lines[0]
        assert "middle" in lines[1]
        assert "oldest" in lines[2]
        assert lines[1].startswith(" " * len(ts.RECENT_PREFIX))
        assert lines[2].startswith(" " * len(ts.RECENT_PREFIX))

    def test_count_limits_to_n_most_recent(self):
        now = ts._parse_iso("2026-07-27T12:00:00Z")
        entry = {
            "title": "T",
            "updates": [{"ts": f"2026-07-27T11:{i:02d}:00Z", "text": f"e{i}", "kind": "manual"} for i in range(10)],
        }
        lines = ts._recent_updates_lines(entry, 3, now=now)
        assert len(lines) == 3
        assert "e9" in lines[0]
        assert "e8" in lines[1]
        assert "e7" in lines[2]


class TestRenderCardRecentBlock:
    def _entry(self, updates):
        return {"title": "T", "status": "active", "prs": [], "branches": [], "runs": [], "updates": updates}

    def test_zero_updates_omits_block(self):
        out = ts.render_card_markdown(self._entry([]))
        assert "recent:" not in out

    def test_one_update_renders_single_line(self):
        now = ts._utcnow()
        ts_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        out = ts.render_card_markdown(self._entry([{"ts": ts_str, "text": "did a thing", "kind": "manual"}]))
        assert "recent: " in out
        assert "did a thing" in out
        assert out.count("recent:") == 1

    def test_three_updates_render_three_lines(self):
        now = ts._utcnow()
        ts_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        updates = [{"ts": ts_str, "text": f"entry {i}", "kind": "manual"} for i in range(3)]
        out = ts.render_card_markdown(self._entry(updates))
        for i in range(3):
            assert f"entry {i}" in out
        # Only the 3 most recent are shown, newest first, one prefixed line.
        assert out.count("recent:") == 1

    def test_ten_updates_render_only_three_most_recent(self):
        now = ts._utcnow()
        ts_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        updates = [{"ts": ts_str, "text": f"entry {i}", "kind": "manual"} for i in range(10)]
        out = ts.render_card_markdown(self._entry(updates))
        assert "entry 9" in out
        assert "entry 8" in out
        assert "entry 7" in out
        assert "entry 6" not in out

    def test_card_degrades_from_three_to_fewer_when_over_length_cap(self):
        now = ts._utcnow()
        ts_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        # Each update text is near the max (120 chars) so 3 of them push the
        # card past CARD_MAX_CHARS, forcing the degradation ladder to drop
        # to fewer entries.
        long_text = "x" * 120
        updates = [{"ts": ts_str, "text": long_text, "kind": "manual"} for _ in range(3)]
        entry = self._entry(updates)
        entry["title"] = "T" * 1500  # inflate the base card so 3 entries would blow the cap
        out = ts.render_card_markdown(entry)
        assert len(out) <= ts.CARD_MAX_CHARS
        recent_line_count = sum(1 for l in out.splitlines() if "recent:" in l or l.startswith(" " * len(ts.RECENT_PREFIX)))
        assert recent_line_count < 3

    def test_card_never_omits_last_entry_even_if_over_cap(self):
        now = ts._utcnow()
        ts_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        long_text = "x" * 120
        updates = [{"ts": ts_str, "text": long_text, "kind": "manual"} for _ in range(3)]
        entry = self._entry(updates)
        entry["title"] = "T" * 2000  # push base itself over CARD_MAX_CHARS
        out = ts.render_card_markdown(entry)
        # Even when it can't fit under the cap, degrade to exactly 1 entry
        # rather than dropping the block entirely.
        assert "recent:" in out


# ---------------------------------------------------------------------------
# Discord card identity: cardMessageId/parentChannelId/discordName
# ---------------------------------------------------------------------------

class TestSnowflakeValidation:
    def test_validate_snowflake_accepts_decimal_digits(self):
        assert ts.validate_snowflake("1531198023408029736") == "1531198023408029736"

    def test_validate_snowflake_accepts_zero(self):
        assert ts.validate_snowflake("0") == "0"

    def test_validate_snowflake_rejects_non_digits(self):
        with pytest.raises(ValueError):
            ts.validate_snowflake("abc123")

    def test_validate_snowflake_rejects_empty(self):
        with pytest.raises(ValueError):
            ts.validate_snowflake("")

    def test_validate_snowflake_rejects_negative(self):
        with pytest.raises(ValueError):
            ts.validate_snowflake("-123")

    def test_validate_snowflake_rejects_leading_plus(self):
        with pytest.raises(ValueError):
            ts.validate_snowflake("+123")

    def test_validate_snowflake_rejects_whitespace(self):
        with pytest.raises(ValueError):
            ts.validate_snowflake(" 123")


class TestDiscordCardIdentityCli:
    def test_add_sets_card_message_id_and_parent_channel_id(self, registry_path, capsys):
        code, out, err = run_cli(
            [
                "add", "1", "--title", "T",
                "--card-message-id", "111111111111111111",
                "--parent-channel-id", "222222222222222222",
                "--discord-name", "raw-thread-name",
            ],
            registry_path, capsys,
        )
        assert code == 0
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        assert entry["cardMessageId"] == "111111111111111111"
        assert entry["parentChannelId"] == "222222222222222222"
        assert entry["discordName"] == "raw-thread-name"

    def test_add_defaults_card_identity_fields_to_null(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T"], registry_path, capsys)
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        assert entry["cardMessageId"] is None
        assert entry["parentChannelId"] is None
        assert entry["discordName"] is None

    def test_add_rejects_invalid_card_message_id(self, registry_path, capsys):
        code, out, err = run_cli(
            ["add", "1", "--title", "T", "--card-message-id", "not-a-snowflake"],
            registry_path, capsys,
        )
        assert code == 2
        assert "invalid snowflake" in err
        assert "Traceback" not in err

    def test_add_rejects_invalid_parent_channel_id(self, registry_path, capsys):
        code, out, err = run_cli(
            ["add", "1", "--title", "T", "--parent-channel-id", "xyz"],
            registry_path, capsys,
        )
        assert code == 2
        assert "invalid snowflake" in err
        assert "Traceback" not in err

    def test_set_updates_card_message_id(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T"], registry_path, capsys)
        code, out, err = run_cli(["set", "1", "--card-message-id", "333333333333333333"], registry_path, capsys)
        assert code == 0
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        assert entry["cardMessageId"] == "333333333333333333"

    def test_set_rejects_invalid_card_message_id(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T"], registry_path, capsys)
        code, out, err = run_cli(["set", "1", "--card-message-id", "abc"], registry_path, capsys)
        assert code == 2
        assert "invalid snowflake" in err
        assert "Traceback" not in err

    def test_set_clears_card_message_id(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T", "--card-message-id", "444444444444444444"], registry_path, capsys)
        code, out, err = run_cli(["set", "1", "--clear-card-message-id"], registry_path, capsys)
        assert code == 0
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        assert entry["cardMessageId"] is None

    def test_clear_then_set_in_same_invocation_keeps_new_value(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T", "--card-message-id", "555555555555555555"], registry_path, capsys)
        run_cli(
            ["set", "1", "--clear-card-message-id", "--card-message-id", "666666666666666666"],
            registry_path, capsys,
        )
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        assert entry["cardMessageId"] == "666666666666666666"

    def test_set_roundtrips_parent_channel_id_and_discord_name(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T"], registry_path, capsys)
        run_cli(
            ["set", "1", "--parent-channel-id", "777777777777777777", "--discord-name", "new-name"],
            registry_path, capsys,
        )
        _, out, _ = run_cli(["show", "1", "--json"], registry_path, capsys)
        entry = json.loads(out)
        assert entry["parentChannelId"] == "777777777777777777"
        assert entry["discordName"] == "new-name"

    def test_human_show_includes_discord_card_identity(self, registry_path, capsys):
        run_cli(
            [
                "add", "1", "--title", "T",
                "--card-message-id", "888888888888888888",
                "--parent-channel-id", "999999999999999999",
                "--discord-name", "thread-name",
            ],
            registry_path, capsys,
        )
        _, out, _ = run_cli(["show", "1"], registry_path, capsys)
        assert "888888888888888888" in out
        assert "999999999999999999" in out
        assert "thread-name" in out

    def test_human_show_renders_dash_when_unset(self, registry_path, capsys):
        run_cli(["add", "1", "--title", "T"], registry_path, capsys)
        _, out, _ = run_cli(["show", "1"], registry_path, capsys)
        discord_line = next(l for l in out.splitlines() if l.startswith("discord:"))
        assert discord_line == "discord: cardMessageId=- parentChannelId=- name=-"


