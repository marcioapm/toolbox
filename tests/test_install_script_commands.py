"""The install.sh command list is derived from the packaging metadata.

install.sh is bash by exception: it bootstraps before any Python tooling exists.
Its advertised command list is hand-written and therefore drifts from
[project.scripts]; this test is what makes the drift impossible.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
ADVERTISED = re.compile(r'^echo "   (\S+)\s+—', re.MULTILINE)


def _entrypoint_names() -> set[str]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return set(tomllib.load(handle)["project"]["scripts"])


def _advertised_names() -> set[str]:
    return set(ADVERTISED.findall((REPO_ROOT / "install.sh").read_text()))


def test_install_script_advertises_every_entrypoint():
    missing = _entrypoint_names() - _advertised_names()
    assert not missing, f"install.sh does not advertise: {sorted(missing)}"


def test_install_script_advertises_nothing_extra():
    extra = _advertised_names() - _entrypoint_names()
    assert not extra, f"install.sh advertises non-existent commands: {sorted(extra)}"


def test_advertised_pattern_matches_the_real_file():
    """Guard the regex itself: a parse returning nothing would make both
    checks above pass vacuously."""
    assert len(_advertised_names()) >= 8
