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
        assert data == {"schemaVersion": 2, "accounts": {}, "threads": {}}

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
        assert data["schemaVersion"] == 2
        assert data["threads"]["1"]["prs"][0]["number"] == 4694

    def test_load_and_migrate_no_op_on_v2(self, registry_path):
        data = ts._empty_registry()
        data["threads"]["1"] = {"prs": []}
        ts._write_registry(registry_path, data)
        loaded, migrated = ts._load_and_migrate(registry_path)
        assert migrated is False
        assert loaded["schemaVersion"] == 2

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
        assert on_disk["schemaVersion"] == 2
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
        entry = {"title": "T", "status": "active", "branch": "feat/x", "prs": []}
        out = ts.render_card_markdown(entry)
        assert "branch: `feat/x`" in out
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

