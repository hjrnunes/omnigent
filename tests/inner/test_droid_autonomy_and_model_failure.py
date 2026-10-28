"""Autonomy and model selection drive the real ACP transport against a fake peer.

The peer is a stdlib Python script written into ``tmp_path`` and spawned the way
``droid`` is: ``<script> exec --output-format acp --cwd <cwd>``. It replays the
shapes captured from droid 0.251.0 — ``session/new`` ``configOptions`` keyed
``id`` with ``options[].value``, ``session/set_config_option`` echoing the full
list back, and ``-32602`` errors for an unknown autonomy value or model id.
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from omnigent.inner import droid_executor
from omnigent.inner.droid_executor import DroidExecutor
from omnigent.inner.executor import ExecutorError, TurnComplete

_AUTONOMY_OPTIONS = ("normal", "spec", "auto-low", "auto-medium", "auto-high")

_FAKE_DROID = r"""
import json
import os
import sys

record_path = os.environ["FAKE_DROID_RECORD"]
set_autonomy_mode = os.environ.get("FAKE_DROID_SET_AUTONOMY", "ok")
set_model_mode = os.environ.get("FAKE_DROID_SET_MODEL", "ok")
autonomy_options = ["normal", "spec", "auto-low", "auto-medium", "auto-high"]
state = {
    "session": "fake-session-1",
    "autonomy": os.environ.get("FAKE_DROID_START_AUTONOMY", "auto-medium"),
    "model": "fake-default-model",
}
recorded_params = ("sessionId", "configId", "value", "modelId")


def send(value):
    print(json.dumps(value), flush=True)


def config_options():
    return [
        {
            "id": "autonomy_level",
            "type": "select",
            "currentValue": state["autonomy"],
            "options": [{"value": value, "name": value} for value in autonomy_options],
        },
        {
            "id": "model",
            "type": "select",
            "currentValue": state["model"],
            "options": [{"value": state["model"], "name": state["model"]}],
        },
    ]


for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    if "id" not in request:
        continue
    params = request.get("params") or {}
    with open(record_path, "a", encoding="utf-8") as record:
        record.write(
            json.dumps(
                {
                    "method": method,
                    "params": {key: params[key] for key in recorded_params if key in params},
                }
            )
            + "\n"
        )
    if method == "initialize":
        result = {"protocolVersion": 1, "agentCapabilities": {}}
    elif method == "session/new":
        result = {"sessionId": state["session"], "configOptions": config_options()}
    elif method == "session/set_config_option":
        if set_autonomy_mode == "silent":
            continue
        value = str(params.get("value"))
        if set_autonomy_mode == "error" or value not in autonomy_options:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "error": {
                        "code": -32602,
                        "message": "Invalid params: Invalid autonomy level value: " + value,
                    },
                }
            )
            continue
        if set_autonomy_mode != "mismatch":
            state["autonomy"] = value
        if set_autonomy_mode in ("empty-result", "notify"):
            # No echo in the reply: either nothing at all, or a
            # config_option_update notification carrying the new value first.
            if set_autonomy_mode == "notify":
                send(
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": state["session"],
                            "update": {
                                "sessionUpdate": "config_option_update",
                                "configOptions": config_options(),
                            },
                        },
                    }
                )
            result = {}
        elif set_autonomy_mode == "empty-options":
            result = {"configOptions": []}
        else:
            result = {"configOptions": config_options()}
    elif method == "session/set_model":
        if set_model_mode == "silent":
            continue
        if set_model_mode == "error":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "error": {"code": -32602, "message": "Invalid params: Model not recognized"},
                }
            )
            continue
        state["model"] = params["modelId"]
        result = {}
    elif method == "session/prompt":
        send(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": state["session"],
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "ok"},
                    },
                },
            }
        )
        result = {"stopReason": "end_turn"}
    else:
        send(
            {
                "jsonrpc": "2.0",
                "id": request["id"],
                "error": {"code": -32601, "message": "unsupported method"},
            }
        )
        continue
    send({"jsonrpc": "2.0", "id": request["id"], "result": result})
"""


def _install_peer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    set_autonomy: str = "ok",
    set_model: str = "ok",
) -> Path:
    """Write the fake ACP peer and point the executor's env knobs at it."""
    script = tmp_path / "fake-droid"
    script.write_text(f"#!{sys.executable}\n{_FAKE_DROID}", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("FAKE_DROID_RECORD", str(tmp_path / "requests.jsonl"))
    monkeypatch.setenv("FAKE_DROID_SET_AUTONOMY", set_autonomy)
    monkeypatch.setenv("FAKE_DROID_SET_MODEL", set_model)
    return script


def _requests(tmp_path: Path) -> list[dict[str, object]]:
    """Return the peer's request log, or ``[]`` when it saw nothing."""
    path = tmp_path / "requests.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _methods(tmp_path: Path) -> list[str]:
    return [str(request["method"]) for request in _requests(tmp_path)]


async def _collect(events: AsyncIterator[object]) -> list[object]:
    return [event async for event in events]


def _executor(peer: Path, tmp_path: Path, **kwargs: str | None) -> DroidExecutor:
    return DroidExecutor(droid_path=str(peer), cwd=str(tmp_path), **kwargs)


async def _turn(executor: DroidExecutor) -> list[object]:
    try:
        return await _collect(executor.run_turn([{"role": "user", "content": "hi"}], [], ""))
    finally:
        await executor.close()


def test_launch_argv_passes_only_the_acp_transport_flags() -> None:
    """``--auto`` / ``-m`` / ``-r`` are ignored in acp mode, so they are not sent."""
    executor = DroidExecutor(cwd="/w", model="gpt-6-luna", auto="high", reasoning="high")

    assert executor._launch_argv() == ["exec", "--output-format", "acp", "--cwd", "/w"]


def test_no_default_autonomy_constant_remains() -> None:
    assert not hasattr(droid_executor, "_DEFAULT_AUTO")


@pytest.mark.parametrize("auto", [None, "inherit"])
async def test_inherited_autonomy_sends_no_config_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, auto: str | None
) -> None:
    """A spec without autonomy leaves droid's settings-file level untouched."""
    peer = _install_peer(tmp_path, monkeypatch)

    events = await _turn(_executor(peer, tmp_path, auto=auto))

    assert not [event for event in events if isinstance(event, ExecutorError)]
    assert [event for event in events if isinstance(event, TurnComplete)]
    assert "session/set_config_option" not in _methods(tmp_path)
    assert _methods(tmp_path) == [
        "initialize",
        "session/new",
        "session/prompt",
    ]


async def test_configured_autonomy_is_applied_and_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    peer = _install_peer(tmp_path, monkeypatch)

    events = await _turn(_executor(peer, tmp_path, auto="normal"))

    assert not [event for event in events if isinstance(event, ExecutorError)]
    calls = [
        request
        for request in _requests(tmp_path)
        if request["method"] == "session/set_config_option"
    ]
    assert calls == [
        {
            "method": "session/set_config_option",
            "params": {
                "sessionId": "fake-session-1",
                "configId": "autonomy_level",
                "value": "normal",
            },
        }
    ]


async def test_unknown_autonomy_fails_session_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A value droid does not advertise fails before the turn is prompted."""
    peer = _install_peer(tmp_path, monkeypatch)

    events = await _turn(_executor(peer, tmp_path, auto="bogus-level"))

    errors = [event for event in events if isinstance(event, ExecutorError)]
    assert len(errors) == 1
    assert "bogus-level" in errors[0].message
    assert "auto-medium" in errors[0].message
    assert errors[0].retryable is False
    assert "session/set_config_option" not in _methods(tmp_path)
    assert "session/prompt" not in _methods(tmp_path)


async def test_rejected_autonomy_fails_session_start_with_droid_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    peer = _install_peer(tmp_path, monkeypatch, set_autonomy="error")

    events = await _turn(_executor(peer, tmp_path, auto="normal"))

    errors = [event for event in events if isinstance(event, ExecutorError)]
    assert len(errors) == 1
    assert "normal" in errors[0].message
    assert "Invalid autonomy level value" in errors[0].message
    assert "session/prompt" not in _methods(tmp_path)


async def test_autonomy_that_droid_does_not_echo_back_fails_session_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reply echoes the level; a different one means the session is wrong."""
    peer = _install_peer(tmp_path, monkeypatch, set_autonomy="mismatch")

    events = await _turn(_executor(peer, tmp_path, auto="normal"))

    errors = [event for event in events if isinstance(event, ExecutorError)]
    assert len(errors) == 1
    assert "'normal'" in errors[0].message
    assert "auto-medium" in errors[0].message
    assert "session/prompt" not in _methods(tmp_path)


@pytest.mark.parametrize("mode", ["empty-result", "empty-options"])
async def test_autonomy_is_rejected_without_a_fresh_setter_echo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """The session/new value is not an echo of the setter.

    Droid 0.251 repeats the full ``configOptions`` list in the setter reply.
    A reply without it leaves the ``session/new`` list retained, so a request
    that already matches droid's starting tier would otherwise be reported as
    applied while the setter said nothing at all.
    """
    peer = _install_peer(tmp_path, monkeypatch, set_autonomy=mode)

    # The peer starts on auto-medium, so ordering it to auto-medium matches
    # the retained session/new value.
    events = await _turn(_executor(peer, tmp_path, auto="auto-medium"))

    errors = [event for event in events if isinstance(event, ExecutorError)]
    assert len(errors) == 1
    assert "auto-medium" in errors[0].message
    assert "echo" in errors[0].message
    assert "session/prompt" not in _methods(tmp_path)


async def test_autonomy_is_accepted_from_a_correlated_config_option_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty setter reply is fine when a ``config_option_update`` confirms it."""
    peer = _install_peer(tmp_path, monkeypatch, set_autonomy="notify")

    events = await _turn(_executor(peer, tmp_path, auto="normal"))

    assert not [event for event in events if isinstance(event, ExecutorError)]
    assert [event for event in events if isinstance(event, TurnComplete)]
    assert "session/prompt" in _methods(tmp_path)


async def test_silent_autonomy_setter_fails_with_a_contextual_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A peer that never answers the setter fails loudly, not with a bare timeout."""
    monkeypatch.setattr(droid_executor, "_INIT_TIMEOUT_SECONDS", 0.3)
    peer = _install_peer(tmp_path, monkeypatch, set_autonomy="silent")

    events = await _turn(_executor(peer, tmp_path, auto="normal"))

    errors = [event for event in events if isinstance(event, ExecutorError)]
    assert len(errors) == 1
    assert "set_config_option" in errors[0].message
    assert "'normal'" in errors[0].message
    assert "session/prompt" not in _methods(tmp_path)


async def test_rejected_model_fails_the_turn_naming_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    peer = _install_peer(tmp_path, monkeypatch, set_model="error")

    events = await _turn(_executor(peer, tmp_path, model="no-such-model-xyz"))

    errors = [event for event in events if isinstance(event, ExecutorError)]
    assert len(errors) == 1
    assert "no-such-model-xyz" in errors[0].message
    assert "Model not recognized" in errors[0].message
    assert "session/prompt" not in _methods(tmp_path)


async def test_unanswered_model_selection_fails_the_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A silent peer must not let the turn run on droid's default model."""
    monkeypatch.setattr(droid_executor, "_INIT_TIMEOUT_SECONDS", 0.3)
    peer = _install_peer(tmp_path, monkeypatch, set_model="silent")

    events = await _turn(_executor(peer, tmp_path, model="no-such-model-xyz"))

    errors = [event for event in events if isinstance(event, ExecutorError)]
    assert len(errors) == 1
    assert "no-such-model-xyz" in errors[0].message
    assert "did not answer" in errors[0].message
    assert "session/prompt" not in _methods(tmp_path)


async def test_per_turn_model_override_rejection_fails_the_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The web picker's per-turn override fails the same way as the spec's model."""
    peer = _install_peer(tmp_path, monkeypatch, set_model="error")
    executor = _executor(peer, tmp_path)
    override = droid_executor.ExecutorConfig(model="no-such-model-xyz")

    try:
        events = await _collect(
            executor.run_turn([{"role": "user", "content": "hi"}], [], "", override)
        )
    finally:
        await executor.close()

    errors = [event for event in events if isinstance(event, ExecutorError)]
    assert len(errors) == 1
    assert "no-such-model-xyz" in errors[0].message
    assert "session/prompt" not in _methods(tmp_path)
