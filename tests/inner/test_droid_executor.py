"""Unit tests for DroidExecutor's ACP handshake and prompt deadlines.

Covers the hand-edited timeouts the installed Mac executor carries: a 120 s
handshake budget for ``initialize`` / ``session/new`` / ``session/set_model``,
plus an idle prompt deadline that resets on every inbound notification and a
separate total-turn cap. The executor is driven through a fake ``_send`` (no
real ``droid`` binary), mirroring ``tests/inner/test_goose_executor.py``.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

from omnigent.inner import droid_executor
from omnigent.inner.droid_executor import DroidExecutor
from omnigent.inner.executor import ExecutorError, TurnComplete

# ---------------------------------------------------------------------------
# Handshake / prompt timeouts
# ---------------------------------------------------------------------------


def test_timeout_constants_match_the_hand_edited_executor() -> None:
    assert droid_executor._INIT_TIMEOUT_SECONDS == 120.0
    assert droid_executor._PROMPT_IDLE_TIMEOUT_SECONDS == 900.0
    assert droid_executor._PROMPT_TOTAL_TIMEOUT_SECONDS == 3600.0


def test_rpc_default_timeout_is_the_init_timeout() -> None:
    assert inspect.signature(DroidExecutor._rpc).parameters["timeout"].default == 120.0


async def test_handshake_requests_use_the_init_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """initialize, session/new and session/set_model share the 120 s budget."""
    monkeypatch.setattr(droid_executor, "_INIT_TIMEOUT_SECONDS", 7.5)
    executor = DroidExecutor()
    calls: list[tuple[str, float]] = []

    async def fake_rpc(
        method: str, params: dict[str, Any], timeout: float = 0.0
    ) -> dict[str, Any]:
        calls.append((method, timeout))
        if method == "initialize":
            return {"result": {"agentCapabilities": {}}}
        if method == "session/new":
            return {"result": {"sessionId": "sess-1"}}
        if method == "session/set_model":
            return {"result": {}}
        raise AssertionError(f"unexpected rpc {method}")

    executor._rpc = fake_rpc  # type: ignore[method-assign]

    await executor._ensure_initialized()
    await executor._ensure_session()
    await executor._apply_model("sess-1", "custom:model-1")

    assert calls == [
        ("initialize", 7.5),
        ("session/new", 7.5),
        ("session/set_model", 7.5),
    ]


# ---------------------------------------------------------------------------
# Prompt deadline behavior
# ---------------------------------------------------------------------------


def _ready_executor() -> DroidExecutor:
    """Return an executor whose handshake/session are already done."""
    executor = DroidExecutor()
    executor._initialized = True
    executor._session_id = "sess-1"
    # run_turn only checks that a process object exists and hasn't exited; the
    # fake _send never touches the real pipes.
    executor._proc = type("_Proc", (), {"returncode": None})()  # type: ignore[assignment]
    return executor


def _resolve_prompt(executor: DroidExecutor, req_id: int) -> None:
    fut = executor._pending.get(req_id)
    if fut is not None and not fut.done():
        fut.set_result({"jsonrpc": "2.0", "id": req_id, "result": {"stopReason": "end_turn"}})


def _progress_notification() -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "tick"},
            }
        },
    }


async def _collect(executor: DroidExecutor) -> list[Any]:
    return [
        event async for event in executor.run_turn([{"role": "user", "content": "go"}], [], "")
    ]


async def test_idle_deadline_resets_on_every_inbound_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A turn that keeps emitting progress outlives the idle window."""
    monkeypatch.setattr(droid_executor, "_PROMPT_IDLE_TIMEOUT_SECONDS", 0.6)
    monkeypatch.setattr(droid_executor, "_PROMPT_TOTAL_TIMEOUT_SECONDS", 10.0)
    executor = _ready_executor()
    background: list[asyncio.Task[None]] = []

    async def fake_send(msg: dict[str, Any]) -> None:
        if msg.get("method") != "session/prompt":
            return
        req_id = msg["id"]

        async def pump() -> None:
            # Three progress notifications 0.35 s apart: total elapsed exceeds
            # the 0.6 s idle window, but no single gap does.
            for _ in range(3):
                await asyncio.sleep(0.35)
                await executor._queue.put(_progress_notification())
            # The prompt result resolves the pending future; an inbound message
            # wakes the loop so it observes that on its next pass (production
            # droid always keeps emitting updates while it works).
            await executor._queue.put(_progress_notification())
            _resolve_prompt(executor, req_id)

        background.append(asyncio.create_task(pump()))

    executor._send = fake_send  # type: ignore[method-assign]

    events = await _collect(executor)

    assert not [event for event in events if isinstance(event, ExecutorError)]
    assert [event for event in events if isinstance(event, TurnComplete)]


async def test_idle_timeout_has_its_own_error_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(droid_executor, "_PROMPT_IDLE_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(droid_executor, "_PROMPT_TOTAL_TIMEOUT_SECONDS", 10.0)
    executor = _ready_executor()
    executor._send = _noop_send  # type: ignore[method-assign]

    errors = [event for event in await _collect(executor) if isinstance(event, ExecutorError)]

    assert len(errors) == 1
    assert "idle" in errors[0].message
    assert "without progress" in errors[0].message
    assert errors[0].retryable is True


async def test_total_timeout_has_its_own_error_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(droid_executor, "_PROMPT_IDLE_TIMEOUT_SECONDS", 10.0)
    monkeypatch.setattr(droid_executor, "_PROMPT_TOTAL_TIMEOUT_SECONDS", 0.1)
    executor = _ready_executor()
    executor._send = _noop_send  # type: ignore[method-assign]

    errors = [event for event in await _collect(executor) if isinstance(event, ExecutorError)]

    assert len(errors) == 1
    assert "turn exceeded" in errors[0].message
    assert "idle" not in errors[0].message
    assert errors[0].retryable is True


async def _noop_send(msg: dict[str, Any]) -> None:
    return None
