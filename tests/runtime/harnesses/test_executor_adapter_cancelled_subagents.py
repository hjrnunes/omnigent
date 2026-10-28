"""Adapter-level regression: an interrupted turn still finishes its sub-agents.

Droid reports a Task sub-agent's outcome on the parent ACP stream, and the
runner only marks the child session finished when it receives
``subagent.completed``. On a cancel the executor winds down by draining the
Task it left open (``ok=False``), but the adapter used to return at the first
event it saw after ``ctx.cancelled`` was set — discarding that completion and
leaving the child row running forever.

The turn runs through the real :class:`DroidExecutor` against a fake ACP peer
subprocess (no droid binary, no tokens): ``session/prompt`` spawns a Task and
then waits for ``session/cancel`` to resolve the prompt with
``stopReason: cancelled``, which is what droid 0.251 does.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from omnigent.inner.droid_executor import DroidExecutor
from omnigent.runtime.harnesses._executor_adapter import ExecutorAdapter
from omnigent.runtime.harnesses._scaffold import TurnContext
from omnigent.server.schemas import CreateResponseRequest

_FAKE_DROID = r"""
import json
import os
import sys

record_path = os.environ["FAKE_DROID_RECORD"]
pending_prompt = None


def send(value):
    print(json.dumps(value), flush=True)


for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    if "id" in request:
        with open(record_path, "a", encoding="utf-8") as record:
            record.write(json.dumps({"method": method}) + "\n")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": request["id"], "result": {"protocolVersion": 1}})
    elif method == "session/new":
        send(
            {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {
                    "sessionId": "fake-session-1",
                    "configOptions": [
                        {
                            "id": "autonomy_level",
                            "type": "select",
                            "currentValue": "auto-medium",
                            "options": [{"value": "auto-medium", "name": "auto-medium"}],
                        }
                    ],
                },
            }
        )
    elif method == "session/prompt":
        pending_prompt = request["id"]
        send(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": "fake-session-1",
                    "update": {
                        "sessionUpdate": "tool_call",
                        "toolCallId": "tc-task-1",
                        "title": "Worker: Count to four",
                        "kind": "other",
                        "status": "pending",
                        "rawInput": {
                            "subagent_type": "worker",
                            "description": "Count to four",
                            "prompt": "count to four",
                            "await": False,
                        },
                    },
                },
            }
        )
    elif method == "session/cancel":
        with open(record_path, "a", encoding="utf-8") as record:
            record.write(json.dumps({"method": method}) + "\n")
        if pending_prompt is not None:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": pending_prompt,
                    "result": {"stopReason": "cancelled"},
                }
            )
            pending_prompt = None
    elif "id" in request:
        send(
            {
                "jsonrpc": "2.0",
                "id": request["id"],
                "error": {"code": -32601, "message": "unsupported method"},
            }
        )
"""


def _install_peer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write the fake ACP peer and point the recorder at *tmp_path*."""
    script = tmp_path / "fake-droid"
    script.write_text(f"#!{sys.executable}\n{_FAKE_DROID}", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("FAKE_DROID_RECORD", str(tmp_path / "requests.jsonl"))
    return script


def _methods(tmp_path: Path) -> list[str]:
    path = tmp_path / "requests.jsonl"
    if not path.exists():
        return []
    return [json.loads(line)["method"] for line in path.read_text().splitlines() if line]


async def _events_named(
    queue: asyncio.Queue[Any], name: str, *, timeout: float = 20.0
) -> list[Any]:
    """Drain *queue* until an event named *name* arrives; return it (and prior)."""
    seen: list[Any] = []
    while True:
        event = await asyncio.wait_for(queue.get(), timeout=timeout)
        if event is None:
            return seen
        seen.append(event)
        if getattr(event, "type", None) == name:
            return seen


@pytest.mark.asyncio
async def test_interrupted_turn_completes_the_open_task_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancel must still deliver the Task child's ``subagent.completed``.

    VERIFIED live: droid answers ``session/cancel`` by resolving the in-flight
    prompt with ``stopReason: "cancelled"`` and reporting no Task outcome, so
    the executor drains the open child as failed. The runner keys the child
    row's terminal update off ``subagent.completed`` only — dropping it leaves
    the Subagents panel spinning.
    """
    peer = _install_peer(tmp_path, monkeypatch)
    executor = DroidExecutor(droid_path=str(peer), cwd=str(tmp_path))
    adapter = ExecutorAdapter(executor_factory=lambda: executor, session_key="sk")
    queue: asyncio.Queue[Any] = asyncio.Queue()
    ctx = TurnContext(response_id="resp_interrupt", event_queue=queue, cancelled=asyncio.Event())
    request = CreateResponseRequest(model="test-agent", input="count to four")

    turn = asyncio.create_task(adapter.run_turn(request, ctx))
    await _events_named(queue, "subagent.started")

    adapter._in_flight[ctx.response_id] = ctx
    await adapter._handle_interrupt_event()
    await asyncio.wait_for(turn, timeout=20.0)

    remaining: list[Any] = []
    while not queue.empty():
        remaining.append(queue.get_nowait())
    completions = [
        event for event in remaining if getattr(event, "type", None) == "subagent.completed"
    ]

    assert "session/cancel" in _methods(tmp_path), (
        "the interrupt must reach the peer as a session/cancel notification; "
        f"peer saw {_methods(tmp_path)!r}"
    )
    assert len(completions) == 1, (
        "an interrupted turn must still report its open Task child as finished; "
        f"adapter emitted {[getattr(e, 'type', None) for e in remaining]!r}"
    )
    assert completions[0].child_key == "tc-task-1"
    assert completions[0].ok is False
    assert completions[0].summary, "the completion must explain why the child stopped"
