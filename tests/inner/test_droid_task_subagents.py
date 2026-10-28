"""Unit tests for DroidExecutor's Task sub-agent lifecycle events.

Droid's ``Task`` tool is its sub-agent spawn. The parent ACP stream reports a
``tool_call`` titled ``Task`` (``rawInput`` carrying ``subagent_type`` /
``description`` / ``prompt``) and later a terminal ``tool_call_update`` with the
sub-agent's ``rawOutput``. :class:`~omnigent.inner.executor.SubAgentStarted` /
:class:`~omnigent.inner.executor.SubAgentCompleted` surface those edges so the
runner mints a child session for each Task.

The executor is driven through a fake ``_send`` (no real ``droid`` binary),
mirroring ``tests/inner/test_droid_executor.py``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from omnigent.inner import droid_executor
from omnigent.inner.droid_executor import DroidExecutor
from omnigent.inner.executor import (
    ExecutorError,
    SubAgentCompleted,
    SubAgentStarted,
    ToolCallRequest,
    TurnComplete,
)

TASK_DESCRIPTION = "Answer the arithmetic question"
TASK_PROMPT = "what is 2+2?"


def _ready_executor() -> DroidExecutor:
    """Return an executor whose handshake/session are already done."""
    executor = DroidExecutor()
    executor._initialized = True
    executor._session_id = "sess-1"
    # run_turn only checks that a process object exists and hasn't exited; the
    # fake _send never touches the real pipes.
    executor._proc = type("_Proc", (), {"returncode": None})()  # type: ignore[assignment]
    return executor


def _notification(update: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "method": "session/update", "params": {"update": update}}


def _task_tool_call(
    call_id: str = "tc-task-1",
    *,
    title: str = "Task",
    description: str = TASK_DESCRIPTION,
    prompt: str = TASK_PROMPT,
    subagent_type: str = "worker",
) -> dict[str, Any]:
    return {
        "sessionUpdate": "tool_call",
        "toolCallId": call_id,
        "title": title,
        "kind": "other",
        "status": "pending",
        "rawInput": {
            "subagent_type": subagent_type,
            "description": description,
            "prompt": prompt,
            "await": True,
            "complexity": "low",
        },
    }


def _task_tool_call_update(
    call_id: str = "tc-task-1",
    *,
    status: str = "completed",
    raw_output: Any = None,
) -> dict[str, Any]:
    update: dict[str, Any] = {
        "sessionUpdate": "tool_call_update",
        "toolCallId": call_id,
        "status": status,
    }
    if raw_output is not None:
        update["rawOutput"] = raw_output
    return update


def _resolve_prompt(executor: DroidExecutor, req_id: int) -> None:
    fut = executor._pending.get(req_id)
    if fut is not None and not fut.done():
        fut.set_result({"jsonrpc": "2.0", "id": req_id, "result": {"stopReason": "end_turn"}})


def _drive(executor: DroidExecutor, *updates: dict[str, Any]) -> None:
    """Install a fake ``_send`` that replays *updates* then ends the turn."""

    async def fake_send(msg: dict[str, Any]) -> None:
        if msg.get("method") != "session/prompt":
            return
        req_id = msg["id"]
        for update in updates:
            await executor._queue.put(_notification(update))
        _resolve_prompt(executor, req_id)

    executor._send = fake_send  # type: ignore[method-assign]


async def _collect(executor: DroidExecutor) -> list[Any]:
    return [
        event async for event in executor.run_turn([{"role": "user", "content": "go"}], [], "")
    ]


def _subagent_events(events: list[Any]) -> list[SubAgentStarted | SubAgentCompleted]:
    return [event for event in events if isinstance(event, (SubAgentStarted, SubAgentCompleted))]


# ---------------------------------------------------------------------------
# Start / completion edges
# ---------------------------------------------------------------------------


async def test_task_tool_call_starts_a_child_and_terminal_update_completes_it() -> None:
    executor = _ready_executor()
    _drive(
        executor,
        _task_tool_call(),
        _task_tool_call_update(raw_output={"text": "The answer is 4."}),
    )

    events = await _collect(executor)

    assert _subagent_events(events) == [
        SubAgentStarted(
            child_key="tc-task-1",
            title=TASK_DESCRIPTION,
            task=TASK_PROMPT,
        ),
        SubAgentCompleted(child_key="tc-task-1", ok=True, summary="The answer is 4."),
    ]
    # The parent still surfaces the Task tool call itself.
    assert [event for event in events if isinstance(event, ToolCallRequest)]
    assert [event for event in events if isinstance(event, TurnComplete)]


async def test_droid_251_title_shape_starts_a_child() -> None:
    """Droid 0.251 titles the Task call ``<subagent_type>: <description>``.

    VERIFIED live against droid 0.251.0: the call is titled
    ``"Worker: Compute 2 plus 2"``, not ``"Task"``, so detection keys on the
    ``subagent_type`` input.
    """
    executor = _ready_executor()
    _drive(
        executor,
        _task_tool_call(title=f"Worker: {TASK_DESCRIPTION}"),
        _task_tool_call_update(raw_output={"text": "4"}),
    )

    events = await _collect(executor)

    starts = [event for event in events if isinstance(event, SubAgentStarted)]
    assert starts == [
        SubAgentStarted(
            child_key="tc-task-1",
            title=TASK_DESCRIPTION,
            task=TASK_PROMPT,
        )
    ]


async def test_repeated_tool_call_for_the_same_id_starts_one_child() -> None:
    """Droid can re-send ``tool_call`` as the call's status advances."""
    executor = _ready_executor()
    _drive(
        executor,
        _task_tool_call(),
        _task_tool_call(),
        _task_tool_call_update(raw_output={"text": "4"}),
    )

    events = await _collect(executor)

    starts = [event for event in events if isinstance(event, SubAgentStarted)]
    assert len(starts) == 1


async def test_failed_task_completes_the_child_with_ok_false() -> None:
    executor = _ready_executor()
    _drive(
        executor,
        _task_tool_call(),
        _task_tool_call_update(status="failed", raw_output={"text": "subagent crashed"}),
    )

    events = await _collect(executor)

    completions = [event for event in events if isinstance(event, SubAgentCompleted)]
    assert completions == [
        SubAgentCompleted(child_key="tc-task-1", ok=False, summary="subagent crashed")
    ]
    assert [event for event in events if isinstance(event, TurnComplete)]


async def test_plain_tool_call_emits_no_subagent_events() -> None:
    executor = _ready_executor()
    _drive(
        executor,
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "tc-read-1",
            "title": "Read",
            "kind": "read",
            "status": "pending",
            "rawInput": {"file_path": "/tmp/notes.txt"},
        },
        _task_tool_call_update("tc-read-1", raw_output={"text": "notes"}),
    )

    events = await _collect(executor)

    assert _subagent_events(events) == []


async def test_terminal_update_without_a_start_emits_no_completion() -> None:
    executor = _ready_executor()
    _drive(executor, _task_tool_call_update("tc-unknown", raw_output={"text": "orphan"}))

    events = await _collect(executor)

    assert _subagent_events(events) == []


async def test_string_raw_output_becomes_the_completion_summary() -> None:
    executor = _ready_executor()
    _drive(
        executor,
        _task_tool_call(),
        _task_tool_call_update(raw_output="plain text summary"),
    )

    events = await _collect(executor)

    completions = [event for event in events if isinstance(event, SubAgentCompleted)]
    assert completions == [
        SubAgentCompleted(child_key="tc-task-1", ok=True, summary="plain text summary")
    ]


# ---------------------------------------------------------------------------
# Dangling children at turn end
# ---------------------------------------------------------------------------


async def test_task_left_open_at_turn_end_completes_with_ok_false() -> None:
    """An ``await: false`` Task outlives the prompt; its child must not stay running."""
    executor = _ready_executor()
    _drive(executor, _task_tool_call())

    events = await _collect(executor)

    completions = [event for event in events if isinstance(event, SubAgentCompleted)]
    assert len(completions) == 1
    assert completions[0].child_key == "tc-task-1"
    assert completions[0].ok is False
    assert "ended" in completions[0].summary
    # The child completes before the turn's terminal event so the runner's
    # per-turn sub-agent bookkeeping still sees it.
    completion_idx = next(
        idx for idx, event in enumerate(events) if isinstance(event, SubAgentCompleted)
    )
    turn_complete_idx = next(
        idx for idx, event in enumerate(events) if isinstance(event, TurnComplete)
    )
    assert completion_idx < turn_complete_idx


async def test_dangling_children_complete_before_a_timeout_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _ready_executor()
    monkeypatch.setattr(droid_executor, "_PROMPT_IDLE_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(droid_executor, "_PROMPT_TOTAL_TIMEOUT_SECONDS", 10.0)

    async def fake_send(msg: dict[str, Any]) -> None:
        if msg.get("method") != "session/prompt":
            return
        await executor._queue.put(_notification(_task_tool_call()))

    executor._send = fake_send  # type: ignore[method-assign]

    events = await _collect(executor)

    completion_idx = next(
        idx for idx, event in enumerate(events) if isinstance(event, SubAgentCompleted)
    )
    error_idx = next(idx for idx, event in enumerate(events) if isinstance(event, ExecutorError))
    assert completion_idx < error_idx
    assert events[completion_idx].ok is False


async def test_open_children_do_not_leak_across_turns() -> None:
    executor = _ready_executor()
    _drive(executor, _task_tool_call())
    await _collect(executor)

    # Second turn with no Task tool call: the first child is already closed, so
    # nothing is re-completed.
    executor._send = _noop_send  # type: ignore[method-assign]
    executor._pending.clear()
    executor._queue = asyncio.Queue()
    _drive(executor)
    events = await _collect(executor)

    assert _subagent_events(events) == []


async def _noop_send(msg: dict[str, Any]) -> None:
    return None
