"""Safety tests for agent-run process-control state."""
from __future__ import annotations

import argparse
import os
import platform
import signal
from pathlib import Path

import pytest

from toolbox import agent_run


def _kill_args(name: str = "run") -> argparse.Namespace:
    return argparse.Namespace(name=name, signal="TERM")


def _seed_kill_state(root: Path, *, pid: str = "123", pgid: str = "123", identity: str | None = "linux:456") -> Path:
    state = root / "run"
    state.mkdir()
    (state / "status").write_text("running\n")
    (state / "pid").write_text(pid + "\n")
    (state / "pgid").write_text(pgid + "\n")
    if identity is not None:
        (state / "process_identity").write_text(identity + "\n")
    return state


@pytest.mark.parametrize("pid", ["", "junk", "0", "-7"])
def test_steer_rejects_invalid_pid_before_liveness_or_fifo_write(isolated_runs_root, monkeypatch, pid):
    state = isolated_runs_root / "run"
    state.mkdir()
    (state / "interactive").write_text("1\n")
    (state / "pid").write_text(pid)
    os.mkfifo(state / "stdin")
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: pytest.fail("liveness checked"))

    with pytest.raises(SystemExit, match="invalid pid state"):
        agent_run.cmd_steer(argparse.Namespace(name="run", message=["hello"], esc=False, raw=False))


@pytest.mark.parametrize("field,value", [("pid", ""), ("pid", "junk"), ("pid", "0"), ("pid", "-1"), ("pgid", ""), ("pgid", "junk"), ("pgid", "0"), ("pgid", "-1")])
def test_kill_rejects_invalid_control_state_without_signaling(isolated_runs_root, monkeypatch, field, value):
    state = _seed_kill_state(isolated_runs_root)
    (state / field).write_text(value)
    calls = []
    monkeypatch.setattr(agent_run.os, "killpg", lambda *args: calls.append(args))
    monkeypatch.setattr(agent_run.os, "kill", lambda *args: calls.append(args))

    with pytest.raises(SystemExit, match=f"invalid {field} state"):
        agent_run.cmd_kill(_kill_args())
    assert calls == []


def _make_verified_kill_state(isolated_runs_root, monkeypatch, *, stored="linux:old", current="linux:old"):
    _seed_kill_state(isolated_runs_root, identity=stored)
    monkeypatch.setattr(agent_run, "_process_identity", lambda _pid: current)
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(agent_run.os, "getpgid", lambda _pid: 123)


def test_kill_matching_identity_signals_verified_group(isolated_runs_root, monkeypatch):
    _make_verified_kill_state(isolated_runs_root, monkeypatch)
    groups = []
    monkeypatch.setattr(agent_run.os, "killpg", lambda pgid, sig: groups.append((pgid, sig)))

    assert agent_run.cmd_kill(_kill_args()) == 0
    assert groups == [(123, signal.SIGTERM)]


@pytest.mark.parametrize("stored,current,error", [("linux:old", "linux:new", "does not match"), (None, "linux:old", "no process identity"), ("bad-token", "linux:old", "corrupt process identity"), ("linux:old", None, "cannot verify")])
def test_kill_refuses_unverifiable_identity_without_signaling(isolated_runs_root, monkeypatch, stored, current, error):
    _seed_kill_state(isolated_runs_root, identity=stored)
    groups = []
    monkeypatch.setattr(agent_run, "_process_identity", lambda _pid: current)
    monkeypatch.setattr(agent_run.os, "killpg", lambda *args: groups.append(args))

    with pytest.raises(SystemExit, match=error):
        agent_run.cmd_kill(_kill_args())
    assert groups == []


def test_kill_refuses_pgid_mismatch_without_signaling(isolated_runs_root, monkeypatch):
    _make_verified_kill_state(isolated_runs_root, monkeypatch)
    groups = []
    monkeypatch.setattr(agent_run.os, "getpgid", lambda _pid: 456)
    monkeypatch.setattr(agent_run.os, "killpg", lambda *args: groups.append(args))

    with pytest.raises(SystemExit, match="process group does not match"):
        agent_run.cmd_kill(_kill_args())
    assert groups == []


def test_linux_process_identity_extracts_starttime(monkeypatch):
    monkeypatch.setattr(agent_run.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        agent_run.Path,
        "read_text",
        lambda _path: "42 (name with ) paren) S " + " ".join(str(n) for n in range(1, 21)),
    )

    assert agent_run._process_identity(42) == "linux:19"


def test_darwin_process_identity_uses_ps_start_token(monkeypatch):
    monkeypatch.setattr(agent_run.platform, "system", lambda: "Darwin")

    class Result:
        stdout = "Mon Jan  1 00:00:00 2024\n"

    captured = {}
    monkeypatch.setattr(
        agent_run.subprocess,
        "run",
        lambda command, **kwargs: captured.update(command=command, **kwargs) or Result(),
    )

    assert agent_run._process_identity(42) == "darwin:Mon Jan  1 00:00:00 2024"
    assert captured["command"] == ["ps", "-p", "42", "-o", "lstart="]


@pytest.mark.skipif(platform.system() not in {"Linux", "Darwin"}, reason="identity is platform-specific")
def test_current_process_identity_is_stable():
    first = agent_run._process_identity(os.getpid())
    assert first is not None
    assert agent_run._process_identity(os.getpid()) == first
