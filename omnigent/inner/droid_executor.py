"""DroidExecutor: run agents through Factory AI's ``droid`` CLI in ACP mode.

Spawns Factory's Droid (``droid exec --output-format acp``) as a subprocess and
communicates via the Agent Client Protocol (ACP) — a JSON-RPC 2.0 protocol over
newline-delimited JSON on stdin/stdout. This is the *headless* Droid harness
(``harness: droid``): output streams into the Omnigent conversation as chat, and
Droid's mid-turn tool approvals surface as web elicitation cards rather than
in-terminal prompts. It mirrors the goose/qwen ACP harnesses.

Protocol flow (VERIFIED live end-to-end against Factory CLI 0.164.0 with a
working ``FACTORY_API_KEY`` — real ``session/prompt`` turns were captured and
every per-turn ``session/update`` shape below was reconciled against the wire):
  1. ``initialize``   — handshake; learn ``agentCapabilities`` (image support).
     Droid returns ``protocolVersion: 1`` and advertises ``factory-api-key``
     (``FACTORY_API_KEY``) auth.
  2. ``session/new``  — create a session; Droid returns the ``sessionId`` plus
     ``models`` / ``modes`` / ``configOptions``. The CLI's ``--auto`` / ``-m`` /
     ``-r`` flags are *ignored* in acp mode (see :meth:`_launch_argv`), so the
     session boots on Droid's own settings and both overrides go over the wire
     right after ``session/new``: the configured autonomy tier via
     ``session/set_config_option`` and the configured model via
     ``session/set_model``. Both are verified and fail the session loudly.
  3. ``session/prompt`` — send a user turn; consume streaming ``session/update``
     notifications (``agent_message_chunk`` text, ``agent_thought_chunk``
     reasoning, ``tool_call`` / ``tool_call_update``) and answer any
     server-initiated ``session/request_permission`` requests, then read the
     final response — which is ``{"stopReason": ...}`` only: **droid acp mode
     0.164.0 reports no per-turn token usage** (neither in the result nor as a
     ``session/update``), so usage surfaces as ``None``.
  4. Re-use the same ``sessionId`` for subsequent turns.

Autonomy note: which tool actions Droid runs without confirmation is Droid's own
autonomy level (``normal`` / ``spec`` / ``auto-low`` / ``auto-medium`` /
``auto-high`` as advertised in ``session/new`` ``configOptions``). With no spec
setting, the session uses the level from Droid's settings file. Only calls Droid
gates arrive as a ``session/request_permission``, which is what routes them
through Omnigent's TOOL_CALL policy + human-consent elicitation — actions a
higher tier auto-approves **never reach Omnigent's policy**, so a spec that must
see every write has to pin ``executor.autonomy: normal``. Droid also does *not*
use ACP ``fs/*`` delegation (it reads/writes with its own internal tools even
when we advertise ``clientCapabilities.fs``), so file confinement rests on the
process-level sandbox, not fs delegation.

Unlike goose/qwen, Droid runs a first-party tool loop we surface: ``tool_call``
updates become :class:`ToolCallRequest` / :class:`ToolCallComplete` events and
``agent_thought_chunk`` becomes :class:`ReasoningChunk`. Because Droid executes
those tools inside its own loop, :meth:`handles_tools_internally` returns True so
the Session passes the events through as informational rather than re-dispatching
them. Droid's ``Task`` tool spawns a sub-agent, which the parent stream reports
only as that tool call plus its terminal result; the executor turns those edges
into :class:`SubAgentStarted` / :class:`SubAgentCompleted` so the runner mints a
child session per Task.

Auth is Droid's own configuration: a ``FACTORY_API_KEY`` env var (or the
interactive ``droid`` device-pairing login). Omnigent stores no Droid
credential — this is an OWN_AUTH harness. A spec ``executor.model`` and
``executor.autonomy`` are applied to the ACP session (``session/set_model`` /
``session/set_config_option``), not as CLI flags.

Requirements:
    The ``droid`` CLI (Factory CLI) must be installed and on PATH
    (``curl -fsSL https://app.factory.ai/cli | sh``), authenticated via
    ``FACTORY_API_KEY`` or ``droid`` login.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from omnigent.inner.datamodel import OSEnvSandboxSpec, OSEnvSpec
from omnigent.inner.executor import (
    Executor,
    ExecutorConfig,
    ExecutorError,
    ExecutorEvent,
    Message,
    ReasoningChunk,
    SubAgentCompleted,
    SubAgentStarted,
    TextChunk,
    ToolCallComplete,
    ToolCallRequest,
    ToolCallStatus,
    TurnCancelled,
    TurnComplete,
    classify_tool_result,
)
from omnigent.inner.os_env import OSEnvironment, create_os_environment

logger = logging.getLogger(__name__)

# ACP error code we return when a *delegated* ``fs/read_text_file`` misses
# (ENOENT) — the shared ACP convention (goose/qwen both use -32002). Any other
# code surfaces raw. NOTE: droid 0.164.0 never delegates fs (it uses its own
# read/edit tools even when we advertise ``clientCapabilities.fs``), so this
# mapping stays UNEXERCISED with droid — kept for spec-conformance / other
# agents.  # UNVERIFIED for droid: fs delegation never fires.
_ACP_RESOURCE_NOT_FOUND_CODE = -32002


class _AcpRequestError(Exception):
    """A handler failure to return as a JSON-RPC error on a server request.

    Carries the JSON-RPC ``code`` / ``message`` so the dispatch in
    :meth:`DroidExecutor._respond_to_agent_request` can build the error reply
    without each handler assembling the wire envelope itself.
    """

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _acp_error_message(reply: dict[str, Any]) -> str:  # type: ignore[explicit-any]
    """Flatten an ACP error reply into droid's message text.

    :param reply: A JSON-RPC response carrying an ``error`` member, e.g.
        ``{"error": {"code": -32602, "message": "Invalid params: Model not recognized"}}``.
    :returns: The agent's message, or the raw error when it is not an object.
    """
    error = reply.get("error")
    if isinstance(error, dict):
        return str(error.get("message", error))
    return str(error)


def _config_option_echo(
    result: dict[str, Any],  # type: ignore[explicit-any]
    config_id: str,
) -> tuple[bool, Any]:  # type: ignore[explicit-any]
    """Read one config option's ``currentValue`` out of an ACP result.

    :param result: A result object that may carry a ``configOptions`` list,
        e.g. a ``session/set_config_option`` reply.
    :param config_id: The option id to look for, e.g. ``"autonomy_level"``.
    :returns: ``(found, currentValue)``; ``found`` is False when the result
        carries no entry for *config_id*, so a missing echo is distinguishable
        from an option whose value happens to be ``None``.
    """
    options = result.get("configOptions")
    if not isinstance(options, list):
        return (False, None)
    for entry in options:
        if isinstance(entry, dict) and entry.get("id") == config_id:
            return (True, entry.get("currentValue"))
    return (False, None)


def _looks_like_missing_file(message: str) -> bool:
    """Heuristic: does an os_env error message indicate a missing path?

    The os_env helper returns failures as ``{"error": "<str>"}`` rather than
    typed exceptions, so the message text is the only signal that a read missed
    because the file is absent (vs. a permission / decode failure).
    """
    lowered = message.lower()
    return (
        "no such file" in lowered
        or "errno 2" in lowered
        or "not found" in lowered
        or "does not exist" in lowered
    )


# ACP protocol constants (JSON-RPC 2.0 method names).
_AGENT_METHOD_INITIALIZE = "initialize"
_AGENT_METHOD_SESSION_NEW = "session/new"
_AGENT_METHOD_SESSION_PROMPT = "session/prompt"

# Notifications sent *from* the agent to the client.
_CLIENT_NOTIFICATION_SESSION_UPDATE = "session/update"

# Notification sent *to* the agent to cancel an in-flight prompt (used by
# :meth:`interrupt_session`). VERIFIED live: droid honors ``session/cancel`` and
# ends the in-flight ``session/prompt`` with ``stopReason: "cancelled"``.
_AGENT_NOTIFICATION_SESSION_CANCEL = "session/cancel"

# Request sent *to* the agent to select the session model (VERIFIED live: droid
# accepts ``{sessionId, modelId}`` and the model config option flips to it).
_AGENT_METHOD_SESSION_SET_MODEL = "session/set_model"

# Request sent *to* the agent to change one session config option. VERIFIED live
# (droid 0.251.0): the reply repeats the full ``configOptions`` list with the new
# ``currentValue`` and droid emits no ``config_option_update`` notification, so
# the reply is the only echo to verify against.
_AGENT_METHOD_SESSION_SET_CONFIG_OPTION = "session/set_config_option"

# Server-initiated request methods (agent → client).
_AGENT_REQUEST_REQUEST_PERMISSION = "session/request_permission"

# session/update.update.sessionUpdate values we map. The chunk/tool variants are
# VERIFIED live (captured on real turns); ``_UPDATE_USAGE`` is NOT — droid
# 0.164.0 never emits a usage stream (see :meth:`_usage_from_result`), so that
# one is retained defensively but stays unexercised.
_UPDATE_AGENT_MESSAGE_CHUNK = "agent_message_chunk"
_UPDATE_AGENT_THOUGHT_CHUNK = "agent_thought_chunk"
_UPDATE_TOOL_CALL = "tool_call"
_UPDATE_TOOL_CALL_UPDATE = "tool_call_update"
_UPDATE_CONFIG_OPTION = "config_option_update"
_UPDATE_USAGE = "usage_update"  # UNVERIFIED for droid: no usage stream observed

# session/update variants droid emits that carry only session metadata (not
# turn content) — silently ignored so they don't trip the unknown-variant log.
_BENIGN_UPDATE_TYPES = frozenset(
    {_UPDATE_CONFIG_OPTION, "current_mode_update", "available_commands_update"}
)

# Idle (time-without-progress) timeouts in seconds. The ACP handshake budget
# is generous because Droid 0.244+ validates BYOK custom models against their
# live endpoint during session/set_model — EnMaaS dispatches measured >30s
# under orchestrator load, which surfaced as a message-less TimeoutError
# ("inner executor error: no detail reported") or a silently swallowed
# set_model failure that fell back to Droid's default model. The prompt idle
# window covers a single long-running tool (garak/eval runs emit no
# session/update while executing); the total cap bounds a runaway turn.
_PROMPT_IDLE_TIMEOUT_SECONDS = 900.0
_PROMPT_TOTAL_TIMEOUT_SECONDS = 3600.0
_INIT_TIMEOUT_SECONDS = 120.0

# ACP protocol version this executor targets (Droid 0.164.0 → 1, VERIFIED).
_PROTOCOL_VERSION = 1

# ``session/new`` config option that carries droid's autonomy tier. VERIFIED
# live (0.251.0): the entry is keyed ``id`` and its ``options`` list holds
# ``{value, name}`` pairs — ``normal`` (Auto off), ``spec``, ``auto-low``,
# ``auto-medium``, ``auto-high``. The starting value comes from
# ``sessionDefaultSettings.autonomyMode`` in ``~/.factory/settings.json``.
_AUTONOMY_CONFIG_ID = "autonomy_level"

# Spec value meaning "keep whatever droid's settings file selects": no
# ``session/set_config_option`` is sent at all.
_AUTONOMY_INHERIT = "inherit"

# Title droid 0.228 gave its sub-agent spawn tool. Droid 0.251 titles the call
# ``<subagent_type>: <description>`` instead, so detection also keys on
# ``rawInput.subagent_type`` (the stable marker across both).
_TASK_TOOL_TITLE = "Task"

# Summary recorded on a Task child that the turn ended without an outcome for.
# Droid's ``await: false`` Tasks keep working after ``session/prompt`` resolves
# and droid reports no later completion on the parent stream.
_UNFINISHED_TASK_SUMMARY = "The parent turn ended before droid reported this Task's result."

# Same, for a turn the user interrupted while droid waited on an approval.
_INTERRUPTED_TASK_SUMMARY = (
    "The parent turn was interrupted before droid reported this Task's result."
)


def _inline_text_file_data(file_data: Any) -> str:  # type: ignore[explicit-any]
    """Decode a text ``input_file`` ``file_data`` data URI into inline text.

    Mirrors the goose/qwen executors: ``input_file`` blocks may carry a
    ``data:<mime>;base64,<payload>`` URI. Text files are decoded so the model
    sees their content; binary files (PDF, images) can't be inlined and return
    ``""``. A bare, non-data-URI string is treated as already-inline text.
    """
    if not isinstance(file_data, str) or not file_data:
        return ""
    if not file_data.startswith("data:"):
        return file_data
    try:
        import base64

        meta, b64 = file_data.split(",", 1)
        mime = meta.split(";")[0].replace("data:", "")
        if not mime.startswith("text/"):
            return ""
        return base64.b64decode(b64).decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 — best-effort; never break a turn on a bad URI
        return ""


def _parse_image_data_uri(data_uri: Any) -> tuple[str, str] | None:  # type: ignore[explicit-any]
    """Split an ``image/*`` ``data:`` URI into ``(mime_type, base64_payload)``.

    Returns ``None`` for anything that isn't an inline ``image/*`` data URI
    (external URLs are never fetched — SSRF).
    """
    if not isinstance(data_uri, str) or not data_uri.startswith("data:"):
        return None
    try:
        meta, payload = data_uri.split(",", 1)
    except ValueError:
        return None
    mime = meta.split(";")[0].replace("data:", "")
    if not mime.startswith("image/") or not payload:
        return None
    return mime, payload


class DroidExecutor(Executor):
    """Executor that drives Factory AI's Droid via its ACP (``exec``) mode.

    Spawns a ``droid exec --output-format acp`` subprocess and manages a session
    through the ACP JSON-RPC 2.0 protocol over newline-delimited stdin/stdout.
    """

    def __init__(
        self,
        cwd: str | None = None,
        os_env: OSEnvSpec | None = None,
        model: str | None = None,
        droid_path: str | None = None,
        auto: str | None = None,
        reasoning: str | None = None,
    ) -> None:
        """Initialize the Droid executor.

        :param cwd: Working directory for the droid subprocess. ``None`` inherits
            the caller's cwd.
        :param os_env: Environment / sandbox spec. When its ``sandbox`` is not
            ``"none"``, the whole ``droid`` process tree is wrapped in the
            platform sandbox (bwrap/seatbelt) at spawn — see
            :meth:`_sandbox_launch_path`.
        :param model: Optional model id applied to the session via
            ``session/set_model`` (else droid's own default from its settings).
        :param droid_path: Absolute path to the droid CLI binary. Defaults to
            ``"droid"`` (PATH lookup).
        :param auto: Autonomy tier applied to the session via
            ``session/set_config_option``; must be one of the ``autonomy_level``
            values droid advertises in ``session/new`` (``normal``, ``spec``,
            ``auto-low``, ``auto-medium``, ``auto-high``). ``None`` / ``"inherit"``
            leaves droid's settings-file level in place.
        :param reasoning: Optional reasoning-effort level carried for harness
            parity. This executor does not forward it: the ACP session reads
            ``reasoning_effort`` from droid's own settings.
        """
        self._cwd = cwd or os.getcwd()
        self._os_env = os_env
        # Whether to advertise ``clientCapabilities.fs`` so Droid delegates file
        # reads/writes back to us (executed through the Omnigent OSEnvironment,
        # which enforces the spec's sandbox read/write roots) instead of using
        # its own raw file tools. Enabled only when an os_env is configured and
        # it isn't a ``fork`` env — a forked env operates on a *copied* tree
        # whose path would diverge from the cwd the droid subprocess runs in.
        self._fs_delegation: bool = os_env is not None and not bool(getattr(os_env, "fork", False))
        # Live OSEnvironment backing fs delegation, created lazily on the first
        # delegated op and torn down in :meth:`close`. ``None`` until then.
        self._os_environment: OSEnvironment | None = None
        self._model = model
        self._droid_path = droid_path or "droid"
        auto_value = (auto or "").strip()
        # ``"inherit"`` and unset are the same thing: droid keeps its settings-file level.
        self._auto = None if auto_value in ("", _AUTONOMY_INHERIT) else auto_value
        self._reasoning = reasoning

        self._proc: asyncio.subprocess.Process | None = None  # type: ignore[name-defined]
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()  # type: ignore[explicit-any]
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None

        self._rpc_id: int = 0
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}  # type: ignore[explicit-any]

        self._session_id: str | None = None
        # Populated from session/new; drives the web model picker.
        self._available_models: list[dict[str, Any]] = []  # type: ignore[explicit-any]
        self._current_model_id: str | None = None
        # ``session/new`` / ``session/set_config_option`` ``configOptions``, used
        # to validate and verify the autonomy tier we ask for.
        self._config_options: list[dict[str, Any]] = []  # type: ignore[explicit-any]
        # Last model we switched to via session/set_model, so a repeat pick is
        # a no-op and the picker reports what the session actually runs on.
        self._applied_model: str | None = None
        self._initialized: bool = False
        self._image_supported: bool = False
        self._system_prompt_sent: bool = False
        # Tool-call ids already surfaced as a ToolCallRequest, so a repeated
        # ``tool_call`` update for the same id doesn't double-emit the request.
        self._seen_tool_calls: set[str] = set()
        # Task tool-call ids whose sub-agent has started but not reported an
        # outcome yet, mapped to the child row's title.
        self._open_task_children: dict[str, str] = {}

        # Context-window size (tokens) reported by Droid's ``usage_update``;
        # surfaced via :meth:`max_context_tokens` so the UI context meter fills.
        self._context_window: int | None = None

        # Bridges the ExecutorAdapter installs so Droid's mid-turn
        # ``session/request_permission`` routes through Omnigent's TOOL_CALL
        # policy + human-consent elicitation rather than blind auto-approve.
        # ``None`` means "no bridge wired" (standalone use / unit tests), in
        # which case permission falls back to allow. See :meth:`_decide_permission`.
        self._policy_evaluator: Any | None = None  # type: ignore[explicit-any]
        self._elicitation_handler: Any | None = None  # type: ignore[explicit-any]

    # ------------------------------------------------------------------
    # Low-level ACP transport
    # ------------------------------------------------------------------

    def _launch_argv(self) -> list[str]:
        """Build the ``droid exec`` argument vector (after argv[0]).

        ``--output-format acp`` selects the ACP JSON-RPC mode (VERIFIED). The
        autonomy / model / reasoning flags are *ignored* by the acp session:
        ``session/new`` reports droid's settings-file values regardless of
        ``--auto`` / ``-m`` / ``-r``, so those flags are not passed at all. The
        model and autonomy are applied over the wire after ``session/new``
        instead (see :meth:`_apply_model` / :meth:`_apply_autonomy`).
        """
        return ["exec", "--output-format", "acp", "--cwd", self._cwd]

    async def _start_process(self) -> None:
        """Start ``droid exec --output-format acp`` as an asyncio subprocess.

        The StreamReader limit is raised to 16 MiB so a large ``session/new``
        response or tool output line can't hit the default 64 KiB per-line cap.
        """
        # Reset handshake state: this may be a restart after the previous
        # subprocess died. ``_initialized`` is a one-way latch.
        self._initialized = False
        self._image_supported = False
        self._seen_tool_calls.clear()
        self._open_task_children.clear()
        env = os.environ.copy()
        argv = self._launch_argv()
        launch_path = self._sandbox_launch_path(tuple(env.keys()))
        _STREAM_LIMIT = 16 * 1024 * 1024
        self._proc = await asyncio.create_subprocess_exec(
            launch_path,
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=self._cwd,
            limit=_STREAM_LIMIT,
        )
        self._reader_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())

    def _sandbox_launch_path(self, spawn_env_names: Sequence[str]) -> str:
        """Return the path to spawn — sandbox launcher or the bare droid binary.

        Mirrors :meth:`GooseExecutor._sandbox_launch_path`. When
        ``os_env.sandbox`` requests confinement, wraps the droid binary in the
        platform sandbox so the *entire* droid process tree (its file / shell
        tools) runs confined to the spec's read/write roots. Falls back to the
        bare binary (never blocks startup) when no sandbox is requested or the
        backend is unavailable.
        """
        os_env = self._os_env
        if os_env is None:
            return self._droid_path
        sandbox_spec = os_env.sandbox or OSEnvSandboxSpec()
        if sandbox_spec.type == "none":
            return self._droid_path
        try:
            # Absolute: in core this module sat in omnigent.inner, so a relative
            # import found the sandbox helpers. Under the plugin namespace it must
            # name core explicitly — a relative import here raises and the except
            # below would silently fall back to an UNSANDBOXED launch.
            from omnigent.inner.sandbox import (
                create_exec_launcher,
                resolve_sandbox,
                with_additional_read_roots,
                with_additional_write_roots,
                with_spawn_env_allowlist,
            )

            cwd = Path(self._cwd or os.getcwd()).resolve(strict=False)
            sandbox = resolve_sandbox(os_env, cwd)
            if not sandbox.active:
                return self._droid_path
            # droid must read its own install tree and write its config/state
            # dirs (~/.factory) and /tmp, or it can't start inside the jail.
            droid_bin = Path(self._droid_path)
            if droid_bin.parent != Path("."):
                sandbox = with_additional_read_roots(sandbox, [droid_bin.resolve().parent])
            sandbox = with_additional_write_roots(
                sandbox,
                [
                    Path.home() / ".factory",
                    Path("/tmp"),
                ],
            )
            sandbox = with_additional_read_roots(sandbox, [Path.home() / ".factory"])
            sandbox = with_spawn_env_allowlist(sandbox, spawn_env_names)
            return create_exec_launcher(self._droid_path, sandbox)
        except (OSError, ImportError, NotImplementedError) as exc:
            logger.warning("Could not apply sandbox for droid; running unsandboxed: %s", exc)
            return self._droid_path

    async def _read_stderr(self) -> None:
        """Continuously drain droid stderr, logging each line at debug.

        Prevents a chatty CLI from filling the OS pipe buffer (~64 KiB) and
        stalling the turn.
        """
        assert self._proc and self._proc.stderr
        try:
            while True:
                raw_line = await self._proc.stderr.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if line:
                    logger.debug("droid stderr: %s", line)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("droid stderr reader stopped: %s", exc)

    async def _read_stdout(self) -> None:
        """Continuously read NDJSON lines from droid stdout.

        Responses (``id`` + no ``method``) resolve the matching ``_pending``
        future; notifications and server-initiated requests go on ``_queue`` for
        ``run_turn`` to consume.
        """
        assert self._proc and self._proc.stdout
        try:
            while True:
                raw_line = await self._proc.stdout.readline()
                if not raw_line:
                    # EOF — the droid subprocess exited. Wake in-flight futures so
                    # run_turn fails fast instead of blocking until idle timeout.
                    for fut in self._pending.values():
                        if not fut.done():
                            fut.set_exception(EOFError("droid subprocess closed stdout"))
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    msg: dict[str, Any] = json.loads(line)  # type: ignore[explicit-any]
                except json.JSONDecodeError:
                    logger.debug("droid: non-JSON stdout line: %r", line[:200])
                    continue

                msg_id = msg.get("id")
                # Match a response by "id + no method": droid's own requests
                # (session/request_permission) also carry an id, so the method
                # check prevents a colliding request from mis-resolving our future.
                if msg_id is not None and "method" not in msg and msg_id in self._pending:
                    fut = self._pending.pop(msg_id)
                    if not fut.done():
                        fut.set_result(msg)
                else:
                    await self._queue.put(msg)
        except (asyncio.CancelledError, EOFError):
            pass
        except Exception as exc:
            logger.exception("droid stdout reader error: %s", exc)
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(exc)
            await self._queue.put({"type": "error", "message": str(exc)})

    async def _send(self, msg: dict[str, Any]) -> None:  # type: ignore[explicit-any]
        """Write one newline-terminated JSON message to droid stdin."""
        assert self._proc and self._proc.stdin
        encoded = (json.dumps(msg) + "\n").encode("utf-8")
        self._proc.stdin.write(encoded)
        await self._proc.stdin.drain()

    async def _rpc(
        self,
        method: str,
        params: dict[str, Any],  # type: ignore[explicit-any]
        timeout: float = _INIT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:  # type: ignore[explicit-any]
        """Send a JSON-RPC 2.0 request and await its response."""
        self._rpc_id += 1
        req_id = self._rpc_id
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()  # type: ignore[explicit-any]
        self._pending[req_id] = fut

        await self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except TimeoutError:
            self._pending.pop(req_id, None)
            raise

    # ------------------------------------------------------------------
    # ACP handshake
    # ------------------------------------------------------------------

    async def _ensure_initialized(self) -> None:
        """Perform the ``initialize`` handshake if not already done.

        VERIFIED against Factory CLI 0.164.0: Droid returns ``protocolVersion:
        1`` and ``agentCapabilities.promptCapabilities.image: true``.
        """
        if self._initialized:
            return
        resp = await self._rpc(
            _AGENT_METHOD_INITIALIZE,
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "clientInfo": {"name": "omnigent", "version": "1.0"},
                # Advertise fs delegation so Droid routes file reads/writes back
                # to us (executed via the OSEnvironment) when an os_env is
                # configured; both false (no os_env / fork env) leaves Droid on
                # its own builtin tools. Terminal stays unadvertised.
                "clientCapabilities": {
                    "fs": {
                        "readTextFile": self._fs_delegation,
                        "writeTextFile": self._fs_delegation,
                    },
                    "terminal": False,
                },
            },
            timeout=_INIT_TIMEOUT_SECONDS,
        )
        if "error" in resp:
            raise RuntimeError(
                f"droid ACP initialize failed: {resp['error'].get('message', resp['error'])}"
            )
        prompt_caps = (
            (resp.get("result") or {}).get("agentCapabilities", {}).get("promptCapabilities", {})
        )
        self._image_supported = bool(prompt_caps.get("image"))
        self._initialized = True

    async def _ensure_session(self) -> str:
        """Create (or reuse) an ACP session, returning Droid's assigned id.

        Sends only ``cwd`` + ``mcpServers`` and reads back ``sessionId`` (VERIFIED
        live: droid returns ``result.sessionId`` alongside ``models`` / ``modes``
        / ``configOptions``). A configured autonomy tier and model are then
        applied over the wire — the ``--auto`` / ``-m`` CLI flags are ignored in
        acp mode, so these setters are the only thing that takes effect
        (VERIFIED live). Both are verified before the id is cached, so a rejected
        value fails the session instead of silently running droid's own choice.

        :returns: Droid's ACP session id.
        :raises RuntimeError: When ``session/new`` fails, or the autonomy /
            model override is rejected or not echoed back.
        """
        if self._session_id is not None:
            return self._session_id

        resp = await self._rpc(
            _AGENT_METHOD_SESSION_NEW,
            {"cwd": self._cwd, "mcpServers": []},
            timeout=_INIT_TIMEOUT_SECONDS,
        )
        if "error" in resp:
            raise RuntimeError(
                f"droid ACP session/new failed: {resp['error'].get('message', resp['error'])}"
            )
        result = resp.get("result", {})
        server_session_id = result.get("sessionId")
        if not server_session_id:
            raise RuntimeError(
                "droid ACP session/new response missing sessionId: " + json.dumps(resp)[:200]
            )
        self._capture_model_state(result)
        self._capture_config_options(result)
        await self._apply_autonomy(server_session_id)
        await self._apply_model(server_session_id, self._model)
        self._session_id = server_session_id
        return self._session_id

    def _capture_config_options(self, result: dict[str, Any]) -> None:  # type: ignore[explicit-any]
        """Remember the ``configOptions`` block droid advertised.

        ``session/new`` and every ``session/set_config_option`` reply carry the
        full list; autonomy validation and verification both read it. A reply
        without the block leaves the previous list in place.

        :param result: The ACP result object carrying ``configOptions``.
        """
        options = result.get("configOptions")
        if not isinstance(options, list):
            return
        entries = [option for option in options if isinstance(option, dict)]
        if entries:
            self._config_options = entries

    def _autonomy_config_option(self) -> dict[str, Any] | None:  # type: ignore[explicit-any]
        """Return the ``autonomy_level`` entry from the last ``configOptions``."""
        for entry in self._config_options:
            if entry.get("id") == _AUTONOMY_CONFIG_ID:
                return entry
        return None

    def _advertised_autonomy_values(self) -> list[str]:
        """Return the autonomy tiers droid offers for this session."""
        options = (self._autonomy_config_option() or {}).get("options")
        if not isinstance(options, list):
            return []
        return [
            str(option["value"])
            for option in options
            if isinstance(option, dict) and "value" in option
        ]

    async def _apply_autonomy(self, session_id: str) -> None:
        """Apply the configured autonomy tier to a fresh ACP session, verified.

        No-op when the spec leaves autonomy to droid. Otherwise the requested
        tier must be one droid advertises, and the change must be confirmed by a
        fresh echo carrying it back: droid 0.251 repeats the full
        ``configOptions`` list in the ``session/set_config_option`` reply
        (VERIFIED live), while an agent may instead report the new value in a
        ``config_option_update`` notification. Only a notification that arrived
        after the setter was sent and names this session counts. The
        ``session/new`` list and anything queued before the setter are the
        starting state, never an echo — accepting them would report a silent
        setter as applied.

        :param session_id: Droid's ACP session id.
        :raises RuntimeError: On an unknown tier, a droid error, a missing echo,
            or a value that does not match the request.
        """
        if self._auto is None:
            return
        advertised = self._advertised_autonomy_values()
        if not advertised:
            raise RuntimeError(
                "droid did not advertise an autonomy_level session option; "
                f"cannot apply autonomy {self._auto!r}"
            )
        if self._auto not in advertised:
            raise RuntimeError(
                f"droid has no autonomy {self._auto!r}; it advertises {', '.join(advertised)}"
            )
        # Notifications already queued describe the session's starting state
        # (droid emits some during session/new). Only what arrives while the
        # setter is in flight can be its echo.
        notifications_before = self._drain_queued_items()
        notifications_after: list[dict[str, Any]] = []  # type: ignore[explicit-any]
        try:
            try:
                resp = await self._rpc(
                    _AGENT_METHOD_SESSION_SET_CONFIG_OPTION,
                    {
                        "sessionId": session_id,
                        "configId": _AUTONOMY_CONFIG_ID,
                        "value": self._auto,
                    },
                    timeout=_INIT_TIMEOUT_SECONDS,
                )
            except TimeoutError as exc:
                raise RuntimeError(
                    "droid did not answer session/set_config_option for autonomy "
                    f"{self._auto!r} within {_INIT_TIMEOUT_SECONDS:.0f}s"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"droid session/set_config_option for autonomy {self._auto!r} failed: {exc}"
                ) from exc
            notifications_after = self._drain_queued_items()
            if "error" in resp:
                raise RuntimeError(
                    f"droid rejected autonomy {self._auto!r}: {_acp_error_message(resp)}"
                )
            raw_result = resp.get("result")
            result = raw_result if isinstance(raw_result, dict) else {}
            echoed, applied = _config_option_echo(result, _AUTONOMY_CONFIG_ID)
            if echoed:
                self._capture_config_options(result)
            elif self._correlated_config_option_echo(
                notifications_after, session_id, _AUTONOMY_CONFIG_ID, self._auto
            ):
                applied = self._auto
            else:
                raise RuntimeError(
                    f"droid did not confirm autonomy {self._auto!r}: its "
                    "session/set_config_option reply carried no autonomy_level echo"
                )
            if applied != self._auto:
                raise RuntimeError(
                    f"droid reported autonomy {applied!r} after requesting {self._auto!r}"
                )
        finally:
            # Anything queued since the last drain keeps its chronological place
            # after the pre-setter items for the turn loop.
            notifications_after.extend(self._drain_queued_items())
            self._requeue_items([*notifications_before, *notifications_after])
        logger.info("droid session autonomy_level=%s applied", self._auto)

    def _drain_queued_items(self) -> list[dict[str, Any]]:  # type: ignore[explicit-any]
        """Remove and return every message currently waiting on ``_queue``."""
        items: list[dict[str, Any]] = []  # type: ignore[explicit-any]
        while not self._queue.empty():
            try:
                items.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:  # pragma: no cover - single consumer
                break
        return items

    def _requeue_items(self, items: Sequence[dict[str, Any]]) -> None:  # type: ignore[explicit-any]
        """Put *items* back on ``_queue`` in the order given."""
        for item in items:
            self._queue.put_nowait(item)

    def _correlated_config_option_echo(
        self,
        notifications: Sequence[dict[str, Any]],  # type: ignore[explicit-any]
        session_id: str,
        config_id: str,
        value: str,
    ) -> bool:
        """Find a ``config_option_update`` that echoes *value* for *session_id*.

        ACP agents that answer ``session/set_config_option`` with an empty result
        report the change in a ``config_option_update`` notification instead.
        Only notifications queued while the setter was in flight are passed in,
        and one naming a different session is ignored — either would let a
        silent setter look applied.

        :param notifications: Messages queued while the setter was in flight.
        :param session_id: The session the setter targeted.
        :param config_id: The config option id to match, e.g. ``"autonomy_level"``.
        :param value: The value the notification must report.
        :returns: True when a notification confirms *config_id* is on *value*.
        """
        confirmed = False
        for item in notifications:
            if not isinstance(item, dict):
                continue
            if item.get("method") != _CLIENT_NOTIFICATION_SESSION_UPDATE:
                continue
            params = item.get("params")
            if not isinstance(params, dict) or params.get("sessionId") != session_id:
                continue
            update = params.get("update")
            if (
                not isinstance(update, dict)
                or update.get("sessionUpdate") != _UPDATE_CONFIG_OPTION
            ):
                continue
            found, current = _config_option_echo(update, config_id)
            if found:
                self._capture_config_options(update)
                confirmed = confirmed or str(current) == value
        return confirmed

    def _capture_model_state(self, result: dict[str, Any]) -> None:
        """Remember the model list ``session/new`` advertised, for the picker.

        ACP spells this ``models: {availableModels, currentModelId}``, but the
        block is optional and agents differ, so a flat ``models`` list is
        accepted too. Anything unrecognized leaves the list empty rather than
        raising — a missing picker is a lesser failure than a dead session.

        :param result: The ``session/new`` result object.
        """
        block = result.get("models")
        if isinstance(block, dict):
            models = block.get("availableModels")
            current = block.get("currentModelId")
        else:
            models = block
            current = result.get("currentModelId")
        self._available_models = (
            [m for m in models if isinstance(m, dict)] if isinstance(models, list) else []
        )
        self._current_model_id = current if isinstance(current, str) else None

    def available_models(self) -> list[dict[str, Any]]:
        """Models droid advertised for this session (empty before ``session/new``)."""
        return list(self._available_models)

    def current_model_id(self) -> str | None:
        """The model this session is on — the last applied switch, else droid's."""
        return self._applied_model or self._current_model_id

    async def _apply_model(self, session_id: str, model: str | None) -> None:
        """Select *model* over the wire (``session/set_model``) when it changes.

        No-op when no model is asked for or it already matches the session's
        current one, so a switch fires only when it will take effect. VERIFIED
        live: droid replies ``result: {}`` and the ``model`` config option flips
        to the requested id; the ``-m`` CLI flag alone does nothing in acp mode.

        A rejection or a hang fails the turn. Falling back would run a different
        model than the caller asked for, which is worse than a visible error.

        :param session_id: Droid's ACP session id.
        :param model: The model id to switch to, or ``None`` to leave it be.
        :raises RuntimeError: When droid rejects the id or does not answer.
        """
        if not model or model == self.current_model_id():
            return
        try:
            resp = await self._rpc(
                _AGENT_METHOD_SESSION_SET_MODEL,
                {"sessionId": session_id, "modelId": model},
                timeout=_INIT_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            raise RuntimeError(
                f"droid did not answer session/set_model for model {model!r} "
                f"within {_INIT_TIMEOUT_SECONDS:.0f}s"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"droid session/set_model for model {model!r} failed: {exc}"
            ) from exc
        if "error" in resp:
            raise RuntimeError(f"droid rejected model {model!r}: {_acp_error_message(resp)}")
        self._applied_model = model

    # ------------------------------------------------------------------
    # Server-initiated requests (agent → client)
    # ------------------------------------------------------------------

    async def _respond_to_agent_request(self, request: dict[str, Any]) -> None:  # type: ignore[explicit-any]
        """Answer a server-initiated ACP request from droid.

        - ``session/request_permission`` — decide via Omnigent's TOOL_CALL policy
          + human-consent elicitation (:meth:`_decide_permission`), then select
          the matching allow/reject option. NOT a blind approve.
        - ``fs/read_text_file`` / ``fs/write_text_file`` — when fs delegation is
          advertised (an os_env is configured; see :attr:`_fs_delegation`), Droid
          routes its file I/O here, executed through the Omnigent OSEnvironment so
          the spec's sandbox read/write roots are enforced. Off → never arrive.
        - anything else — reply with JSON-RPC ``method not found`` so droid fails
          loudly rather than acting on empty data.
        """
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {}) or {}
        logger.debug("droid agent request: method=%s id=%s", method, req_id)

        result: dict[str, Any] | None = None  # type: ignore[explicit-any]
        error: dict[str, Any] | None = None  # type: ignore[explicit-any]
        try:
            if method == _AGENT_REQUEST_REQUEST_PERMISSION:
                allow = await self._decide_permission(params)
                result = {"outcome": self._permission_outcome(params, allow=allow)}
            elif method == "fs/read_text_file" and self._fs_delegation:
                result = await self._handle_fs_read(params)
            elif method == "fs/write_text_file" and self._fs_delegation:
                result = await self._handle_fs_write(params)
            else:
                error = {
                    "code": -32601,
                    "message": f"omnigent: unsupported ACP request method {method!r}",
                }
        except _AcpRequestError as exc:
            error = {"code": exc.code, "message": exc.message}
        except Exception as exc:  # noqa: BLE001
            logger.debug("droid agent request %s failed: %s", method, exc)
            error = {"code": -32603, "message": f"{method} failed: {exc}"}

        reply: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id}  # type: ignore[explicit-any]
        if error is not None:
            reply["error"] = error
        else:
            reply["result"] = result
        await self._send(reply)

    # ------------------------------------------------------------------
    # Filesystem delegation (droid → client, when fs capability advertised)
    # ------------------------------------------------------------------

    async def _ensure_os_environment(self) -> OSEnvironment:
        """Lazily create the OSEnvironment backing fs delegation.

        :returns: The live OSEnvironment for this executor's os_env spec.
        :raises _AcpRequestError: When no usable os_env can be created.
        """
        if self._os_environment is None:
            env = create_os_environment(self._os_env)
            if env is None:
                raise _AcpRequestError(-32603, "omnigent: no os_env for fs delegation")
            self._os_environment = env
        return self._os_environment

    async def _handle_fs_read(self, params: dict[str, Any]) -> dict[str, Any]:  # type: ignore[explicit-any]
        """Serve an ACP ``fs/read_text_file`` by reading through the OSEnvironment.

        ACP params ``{path, line?, limit?}`` (1-based start line, max line count;
        both optional → whole file) map onto :meth:`OSEnvironment.read`.
        """
        path = params.get("path")
        if not isinstance(path, str) or not path:
            raise _AcpRequestError(-32602, "fs/read_text_file requires a string 'path'")
        line = params.get("line")
        limit = params.get("limit")
        offset = line if isinstance(line, int) and line >= 1 else 1
        read_limit = limit if isinstance(limit, int) and limit >= 1 else None

        env = await self._ensure_os_environment()
        result = await env.read(path, offset=offset, limit=read_limit)
        if "error" in result:
            message = str(result["error"])
            code = _ACP_RESOURCE_NOT_FOUND_CODE if _looks_like_missing_file(message) else -32603
            raise _AcpRequestError(code, message)
        if result.get("encoding") != "utf-8":
            raise _AcpRequestError(-32603, f"{path}: not a UTF-8 text file")
        return {"content": result.get("content", "")}

    async def _handle_fs_write(self, params: dict[str, Any]) -> dict[str, Any]:  # type: ignore[explicit-any]
        """Serve an ACP ``fs/write_text_file`` by writing through the OSEnvironment.

        ACP params ``{path, content}``; the write goes through the helper so the
        spec's sandbox write roots are enforced at the Python layer.
        """
        path = params.get("path")
        content = params.get("content")
        if not isinstance(path, str) or not path:
            raise _AcpRequestError(-32602, "fs/write_text_file requires a string 'path'")
        if not isinstance(content, str):
            raise _AcpRequestError(-32602, "fs/write_text_file requires string 'content'")

        env = await self._ensure_os_environment()
        result = await env.write(path, content)
        if "error" in result:
            raise _AcpRequestError(-32603, str(result["error"]))
        return {}

    @staticmethod
    def _extract_tool_call(params: dict[str, Any]) -> tuple[str, dict[str, Any]]:  # type: ignore[explicit-any]
        """Pull ``(tool_name, tool_input)`` from a ``toolCall`` payload.

        Mirrors goose's extraction: prefer the precise tool name from
        ``_meta`` (either ``_meta.toolName`` or a nested vendor
        ``_meta.<vendor>.toolCall.toolName``), else the human ``title``, else the
        ``kind``; ``rawInput`` carries the arguments dict.

        Accepts either a ``session/request_permission`` params dict (fields under
        a nested ``toolCall``) or a ``session/update`` tool-call update (fields
        at the top level) — falls back to *params* itself when there is no nested
        ``toolCall``.

        VERIFIED live: droid's tool calls carry ``toolCallId`` + ``title`` +
        ``kind`` (e.g. ``edit`` / ``execute`` / ``read``) + ``rawInput`` (the
        args dict) and NO ``_meta.toolName`` — so the name resolves to the human
        ``title`` (e.g. ``"Create /path"``). The ``_meta`` vendor-key lookup is
        retained as harmless generality for other ACP agents (goose).
        """
        nested = params.get("toolCall")
        tool_call = nested if isinstance(nested, dict) else params
        meta = tool_call.get("_meta") or {}
        name = meta.get("toolName")
        if not name:
            # Some ACP agents nest the tool name under a vendor key
            # (goose: ``_meta.goose.toolCall.toolName``).
            for value in meta.values():
                if isinstance(value, dict):
                    inner = value.get("toolCall")
                    if isinstance(inner, dict) and inner.get("toolName"):
                        name = inner["toolName"]
                        break
        name = name or tool_call.get("title") or tool_call.get("kind") or "tool"
        args = tool_call.get("rawInput")
        if not isinstance(args, dict):
            args = {}
        return str(name), args

    async def _decide_permission(self, params: dict[str, Any]) -> bool:  # type: ignore[explicit-any]
        """Decide allow/deny for a permission request — policy then elicitation.

        Mirrors :meth:`GooseExecutor._decide_permission`:

        1. **TOOL_CALL policy** (:attr:`_policy_evaluator`): a hard
           ``POLICY_ACTION_DENY`` denies; ``POLICY_ACTION_ASK`` defers to
           elicitation (and **fails closed** when no handler is wired);
           ``ALLOW`` / unspecified falls through.
        2. **Human-consent elicitation** (:attr:`_elicitation_handler`): routes
           to the user via ``ctx.elicit`` (a web approval card) and returns their
           accept/deny.

        When neither bridge is wired (standalone / unit tests), falls back to
        allow so direct use of the executor isn't blocked.
        """
        tool_name, tool_input = self._extract_tool_call(params)
        handler = getattr(self, "_elicitation_handler", None)
        policy_eval = getattr(self, "_policy_evaluator", None)

        if policy_eval is not None:
            action: str | None
            try:
                verdict = await policy_eval(
                    "PHASE_TOOL_CALL", {"name": tool_name, "arguments": tool_input}
                )
                action = getattr(verdict, "action", None)
            except Exception as exc:  # noqa: BLE001 — fail open to elicitation
                logger.warning("droid TOOL_CALL policy eval failed for %s: %s", tool_name, exc)
                action = None
            if action == "POLICY_ACTION_DENY":
                logger.info("droid permission denied by policy: tool=%s", tool_name)
                return False
            if action == "POLICY_ACTION_ASK":
                if handler is None:
                    logger.warning(
                        "droid TOOL_CALL policy ASK with no elicitation handler; denying tool=%s",
                        tool_name,
                    )
                    return False
                allowed = bool(await handler(tool_name, tool_input))
                logger.info(
                    "droid permission %s by user (policy ASK): tool=%s",
                    "allowed" if allowed else "denied",
                    tool_name,
                )
                return allowed
            # ALLOW / UNSPECIFIED / unknown → fall through to elicitation.

        if handler is not None:
            allowed = bool(await handler(tool_name, tool_input))
            logger.info(
                "droid permission %s by user: tool=%s",
                "allowed" if allowed else "denied",
                tool_name,
            )
            return allowed

        logger.debug("droid permission allowed (no policy/elicitation wired): tool=%s", tool_name)
        return True

    @staticmethod
    def _permission_outcome(  # type: ignore[explicit-any]
        params: dict[str, Any], *, allow: bool
    ) -> dict[str, Any]:
        """Map an allow/deny decision to an ACP permission ``outcome``.

        On allow, prefer a once-scoped grant (``allow_once``) over
        ``allow_always`` so we never persist a blanket "always allow". On deny,
        pick a ``reject_*`` option, or ``cancelled`` when none is offered.

        VERIFIED live: droid offers three options —
        ``{optionId:"proceed_once",  kind:"allow_once"}``,
        ``{optionId:"proceed_always",kind:"allow_always"}``,
        ``{optionId:"cancel",        kind:"reject_once"}`` — so ``allow_once``
        maps to ``proceed_once`` and the reply
        ``{"outcome":{"outcome":"selected","optionId":"proceed_once"}}`` is
        accepted and the tool proceeds.
        """
        options = [o for o in (params.get("options") or []) if isinstance(o, dict)]

        def _pick(*kinds: str) -> dict[str, Any] | None:  # type: ignore[explicit-any]
            for kind in kinds:
                for opt in options:
                    if opt.get("kind") == kind:
                        return opt
            return None

        if allow:
            chosen = _pick("allow_once", "allow_always") or next(
                (o for o in options if "allow" in str(o.get("kind", ""))), None
            )
            if chosen is None:
                return {"outcome": "cancelled"}
        else:
            chosen = _pick("reject_once", "reject_always") or next(
                (o for o in options if "reject" in str(o.get("kind", ""))), None
            )
            if chosen is None:
                return {"outcome": "cancelled"}
        return {"outcome": "selected", "optionId": chosen.get("optionId")}

    # ------------------------------------------------------------------
    # Tool-call event mapping (session/update tool_call / tool_call_update)
    # ------------------------------------------------------------------

    def _tool_call_request_event(self, update: dict[str, Any]) -> ToolCallRequest | None:  # type: ignore[explicit-any]
        """Map a ``tool_call`` update onto a :class:`ToolCallRequest`.

        Returns ``None`` when the tool-call id was already surfaced (Droid can
        re-send ``tool_call`` for the same id as its status advances).

        VERIFIED live: the ``tool_call`` update carries ``toolCallId`` and
        ``rawInput`` (args) at the top level (e.g. ``rawInput.file_path`` /
        ``rawInput.command``).
        """
        call_id = update.get("toolCallId") or update.get("id")
        key = str(call_id) if call_id is not None else ""
        if key and key in self._seen_tool_calls:
            return None
        if key:
            self._seen_tool_calls.add(key)
        name, args = self._extract_tool_call(update)
        return ToolCallRequest(name=name, args=args, metadata={"call_id": call_id})

    def _tool_call_complete_event(self, update: dict[str, Any]) -> ToolCallComplete | None:  # type: ignore[explicit-any]
        """Map a terminal ``tool_call_update`` onto a :class:`ToolCallComplete`.

        Only ``completed`` / ``failed`` / ``error`` statuses complete the call;
        interim ``in_progress`` / ``pending`` updates return ``None``.

        VERIFIED live: the terminal ``tool_call_update`` carries ``toolCallId`` +
        ``status``. ``rawOutput`` is present for ``read`` tools (a ``{"text":...}``
        dict) but ABSENT for ``edit`` (status-only) and ``execute`` (output goes
        to a ``terminal``/``terminalId``); the fallback to ``content`` then
        ``None`` handles all three.
        """
        status = str(update.get("status", ""))
        if status not in ("completed", "failed", "error"):
            return None
        call_id = update.get("toolCallId") or update.get("id")
        name, _ = self._extract_tool_call(update)
        result = update.get("rawOutput")
        if result is None:
            result = update.get("content")
        classification = classify_tool_result(result)
        tool_status = classification.status
        error = classification.error or None
        if status in ("failed", "error"):
            tool_status = ToolCallStatus.ERROR
            error = error or (str(result) if result else "tool call failed")
        return ToolCallComplete(
            name=name,
            status=tool_status,
            result=result,
            error=error,
            metadata={"call_id": call_id},
        )

    # ------------------------------------------------------------------
    # Task sub-agent lifecycle (session/update tool_call spawning a child)
    # ------------------------------------------------------------------

    def _task_start_event(self, update: dict[str, Any]) -> SubAgentStarted | None:  # type: ignore[explicit-any]
        """Map a droid sub-agent spawn call onto a :class:`SubAgentStarted`.

        Droid reports its sub-agent spawn as an ordinary ``tool_call``;
        ``rawInput`` carries the delegated ``prompt`` and the row
        ``description``. Returns ``None`` for every other tool, for a spawn
        call without input, and for a repeated update for a child already
        started.

        VERIFIED live: the parent stream carries no sub-agent messages, so this
        edge and the terminal update are the only account of the child's work.
        Droid 0.228 titled the call ``Task``; 0.251 titles it
        ``<subagent_type>: <description>``, so the ``subagent_type`` input is the
        marker detection relies on.
        """
        raw_input = update.get("rawInput")
        if not isinstance(raw_input, dict):
            return None
        subagent_type = raw_input.get("subagent_type")
        has_subagent_type = isinstance(subagent_type, str) and bool(subagent_type)
        if str(update.get("title") or "") != _TASK_TOOL_TITLE and not has_subagent_type:
            return None
        call_id = update.get("toolCallId") or update.get("id")
        key = str(call_id) if call_id is not None else ""
        if not key or key in self._open_task_children:
            return None
        description = raw_input.get("description")
        if isinstance(description, str) and description:
            title = description
        elif has_subagent_type:
            title = str(subagent_type)
        else:
            title = _TASK_TOOL_TITLE
        prompt = raw_input.get("prompt")
        self._open_task_children[key] = title
        return SubAgentStarted(
            child_key=key,
            title=title,
            task=prompt if isinstance(prompt, str) else "",
        )

    def _task_complete_event(self, update: dict[str, Any]) -> SubAgentCompleted | None:  # type: ignore[explicit-any]
        """Map a terminal ``Task`` ``tool_call_update`` onto a completion.

        Keys off the children still open from :meth:`_task_start_event`, so a
        terminal update for an ordinary tool call never produces a sub-agent
        edge. ``ok`` follows droid's status: only ``completed`` is success.
        Returns ``None`` for interim statuses and for unknown call ids.
        """
        status = str(update.get("status", ""))
        if status not in ("completed", "failed", "error"):
            return None
        call_id = update.get("toolCallId") or update.get("id")
        key = str(call_id) if call_id is not None else ""
        if key not in self._open_task_children:
            return None
        self._open_task_children.pop(key, None)
        return SubAgentCompleted(
            child_key=key,
            ok=status == "completed",
            summary=self._raw_output_text(update.get("rawOutput")),
        )

    def _drain_open_task_children(
        self, summary: str = _UNFINISHED_TASK_SUMMARY
    ) -> list[SubAgentCompleted]:
        """Complete every Task child still open when the turn ends.

        Droid's ``await: false`` Tasks keep working after ``session/prompt``
        resolves, and droid reports no later outcome on the parent stream, so an
        unfinished child is closed as failed rather than left running. Clearing
        the map first guarantees one completion per child across the normal,
        timeout, error, and interrupted exit paths.

        :param summary: Why the child is being closed.
        """
        completions = [
            SubAgentCompleted(
                child_key=key,
                ok=False,
                summary=summary,
            )
            for key in self._open_task_children
        ]
        self._open_task_children.clear()
        return completions

    @staticmethod
    def _raw_output_text(raw_output: Any) -> str:  # type: ignore[explicit-any]
        """Flatten a terminal ``tool_call_update.rawOutput`` into summary text.

        VERIFIED live: a Task result arrives as ``{"text": "..."}``. Strings and
        ACP content-block lists are accepted too, so a droid change degrades to
        the text it actually sent rather than an empty summary.
        """
        if isinstance(raw_output, str):
            return raw_output
        if isinstance(raw_output, dict):
            text = raw_output.get("text")
            if isinstance(text, str):
                return text
            blocks = raw_output.get("content")
            if isinstance(blocks, list):
                return "\n".join(
                    block["text"]
                    for block in blocks
                    if isinstance(block, dict) and isinstance(block.get("text"), str)
                )
        return ""

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    @staticmethod
    def _image_blocks_from_content(content: Any) -> list[dict[str, Any]]:  # type: ignore[explicit-any]
        """Build ACP ``image`` prompt blocks from a message's ``input_image`` blocks."""
        out: list[dict[str, Any]] = []  # type: ignore[explicit-any]
        if not isinstance(content, list):
            return out
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "input_image":
                continue
            parsed = _parse_image_data_uri(block.get("image_url") or block.get("file_data"))
            if parsed:
                mime, data = parsed
                out.append({"type": "image", "mimeType": mime, "data": data})
        return out

    @staticmethod
    def _text_from_blocks(
        blocks: list[Any],
        *,
        emit_image_marker: bool = False,  # type: ignore[explicit-any]
    ) -> str:
        """Extract prompt text from a Responses-API content-block list.

        ACP's ``session/prompt`` text part is plain text, so each block is folded:
        ``input_text``/``output_text``/``text`` verbatim; ``input_file`` inlined
        (fenced) when the runner resolved it to a text data URI, else a marker;
        ``input_image`` as a marker only when *emit_image_marker* is set (the
        image otherwise goes as a real ACP image block).
        """
        parts: list[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype in ("input_text", "output_text", "text"):
                text = block.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif btype == "input_file":
                name = block.get("filename") or block.get("file_id") or "file"
                inlined = _inline_text_file_data(block.get("file_data"))
                if inlined:
                    parts.append(
                        f"--- attached file: {name} ---\n{inlined}\n--- end of {name} ---"
                    )
                else:
                    parts.append(f"[attached file: {name}]")
            elif btype == "input_image" and emit_image_marker:
                name = block.get("filename") or block.get("file_id")
                parts.append(f"[attached image: {name}]" if name else "[attached image]")
        return "\n".join(parts)

    @classmethod
    def _history_prefix(cls, prior: list[Any]) -> str:  # type: ignore[explicit-any]
        """Serialize prior conversation turns into a text prefix.

        On a *fresh* ACP session (the first turn of a newly spawned/respawned
        ``droid exec`` process, or after a session reset) Droid holds none of the
        earlier conversation — its context lived in the dead subprocess. Since
        :meth:`run_turn` normally sends only the latest user turn, we'd lose
        everything before the switch. Replaying the transcript as a labeled
        ``role: content`` block restores that context, mirroring the goose/qwen
        executors. A ``/model`` switch respawns the subprocess, so this is what
        keeps a mid-conversation model change from dropping the thread.
        """
        lines = ["Conversation so far:"]
        for msg in prior:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "user")).replace("_", " ")
            raw = msg.get("content")
            if raw is None:
                content = ""
            elif isinstance(raw, str):
                content = raw
            elif isinstance(raw, list):
                content = cls._text_from_blocks(raw, emit_image_marker=True)
            else:
                content = json.dumps(raw, ensure_ascii=True)
            lines.append(f"{role}: {content}")
        lines.append("")
        lines.append(
            "Respond to the latest user message, using the conversation above as context."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Executor interface
    # ------------------------------------------------------------------

    def supports_streaming(self) -> bool:
        return True

    def handles_tools_internally(self) -> bool:
        # Droid runs its own first-party tool loop; the ToolCallRequest /
        # ToolCallComplete events run_turn emits are informational mirrors of
        # that loop, so the Session must NOT re-dispatch them.
        return True

    def max_context_tokens(self) -> int | None:
        """Return Droid's reported context-window size, if observed yet."""
        return self._context_window

    @staticmethod
    def _usage_from_result(result: dict[str, Any]) -> dict[str, Any] | None:  # type: ignore[explicit-any]
        """Map a final ``result.usage`` block to Omnigent's usage keys, if any.

        VERIFIED live: droid acp mode 0.164.0 reports **no** per-turn usage — the
        ``session/prompt`` result is ``{"stopReason": ...}`` only (and no
        ``session/update`` carries usage either), so this returns ``None`` in
        practice. It is kept defensively for a future droid that adds usage, and
        accepts BOTH the snake_case shape droid's non-acp ``--output-format json``
        uses (``input_tokens`` / ``output_tokens`` / ``total_tokens``) and the
        ACP camelCase shape goose reports (``inputTokens`` / …).
        """
        usage = result.get("usage")
        if not isinstance(usage, dict):
            return None
        out: dict[str, Any] = {}
        for omni_key, candidates in (
            ("input_tokens", ("input_tokens", "inputTokens")),
            ("output_tokens", ("output_tokens", "outputTokens")),
            ("total_tokens", ("total_tokens", "totalTokens")),
        ):
            for cand in candidates:
                if isinstance(usage.get(cand), int):
                    out[omni_key] = usage[cand]
                    break
        return out or None

    async def run_turn(
        self,
        messages: list[Message],
        tools: list[Any],  # type: ignore[explicit-any]  # noqa: ARG002 — droid runs its own tool registry
        system_prompt: str,
        config: ExecutorConfig | None = None,
    ) -> AsyncIterator[ExecutorEvent]:
        """Run one turn of the Droid agent loop via ACP.

        Sends ``session/prompt`` and yields ``TextChunk`` / ``ReasoningChunk`` /
        ``ToolCallRequest`` / ``ToolCallComplete`` events as the agent streams,
        answering any ``session/request_permission`` mid-turn, until the final
        response (``stopReason``) arrives — then yields ``TurnComplete`` with
        token usage. A ``Task`` tool call also yields ``SubAgentStarted`` /
        ``SubAgentCompleted``; a Task still open when the turn ends completes
        with ``ok=False`` so its child row never stays running.

        A per-request model override (the web picker / ``/model``) is applied
        live via ``session/set_model`` before prompting — the ACP session
        outlives the pick, so applying it only at ``session/new`` would leave a
        mid-session switch silently inert.
        """
        try:
            if self._proc is None or self._proc.returncode is not None:
                await self._start_process()
            await self._ensure_initialized()
            session_id = await self._ensure_session()
            if config is not None and config.model:
                await self._apply_model(session_id, config.model)
        except Exception as exc:  # noqa: BLE001
            yield ExecutorError(message=str(exc), retryable=False)
            return

        # A fresh ACP session (first turn of a new/respawned process, or after
        # a reset) holds no prior context. Captured before we flip the latch
        # below so we know whether to replay history into this turn.
        fresh_session = not self._system_prompt_sent

        # Build the prompt payload from the most recent user message.
        user_text = ""
        image_blocks: list[dict[str, Any]] = []  # type: ignore[explicit-any]
        latest_user_idx: int | None = None
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            role = msg.get("role", "") if isinstance(msg, dict) else ""
            if role == "user":
                latest_user_idx = idx
                content = msg.get("content", "") if isinstance(msg, dict) else ""
                if isinstance(content, str):
                    user_text = content
                elif isinstance(content, list):
                    if self._image_supported:
                        image_blocks = self._image_blocks_from_content(content)
                    user_text = self._text_from_blocks(
                        content, emit_image_marker=not self._image_supported
                    )
                break

        # On a fresh session, replay the prior conversation so a model switch
        # (which respawns the subprocess) or a session reset doesn't drop the
        # thread — Droid otherwise only ever sees this turn's latest message.
        if fresh_session and latest_user_idx is not None and latest_user_idx > 0:
            history_prefix = self._history_prefix(messages[:latest_user_idx])
            user_text = f"{history_prefix}\n\nuser: {user_text}" if user_text else history_prefix

        # ACP has no system-prompt field, so fold it into the first turn. The
        # latch flips on any fresh session — even with an empty system prompt —
        # so a continuing session never re-replays history or re-folds.
        if fresh_session:
            if system_prompt:
                user_text = f"{system_prompt}\n\n{user_text}" if user_text else system_prompt
            self._system_prompt_sent = True

        prompt_blocks: list[dict[str, Any]] = []  # type: ignore[explicit-any]
        if user_text or not image_blocks:
            prompt_blocks.append({"type": "text", "text": user_text})
        prompt_blocks.extend(image_blocks)

        # Drain stale items from a prior turn; answer any leftover server request.
        while not self._queue.empty():
            try:
                stale = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if isinstance(stale, dict) and stale.get("id") is not None and stale.get("method"):
                await self._respond_to_agent_request(stale)

        try:
            async for event in self._prompt_events(session_id, prompt_blocks):
                yield event
        except asyncio.CancelledError:
            # An interrupt during a policy / approval / elicitation wait cancels
            # the parked future, so the turn resumes with CancelledError instead
            # of yielding another event. Finish the tracked Task children before
            # unwinding, exactly like every other terminal path.
            for completion in self._drain_open_task_children(summary=_INTERRUPTED_TASK_SUMMARY):
                yield completion
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            yield TurnCancelled(reason="user_cancelled", phase="model")

    async def _prompt_events(
        self,
        session_id: str,
        prompt_blocks: list[dict[str, Any]],
    ) -> AsyncIterator[ExecutorEvent]:
        """Stream a single ``session/prompt`` turn until droid reports an outcome.

        Yields text / reasoning / tool events as droid reports them and ends
        with ``TurnComplete`` or ``ExecutorError``. Task children still open at
        any terminal path complete with ``ok=False`` before that terminal event.

        :param session_id: Droid's ACP session id.
        :param prompt_blocks: ACP prompt content blocks for the user turn.
        """

        self._rpc_id += 1
        req_id = self._rpc_id
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()  # type: ignore[explicit-any]
        self._pending[req_id] = fut

        await self._send(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": _AGENT_METHOD_SESSION_PROMPT,
                "params": {"sessionId": session_id, "prompt": prompt_blocks},
            }
        )

        # Two clocks: an idle clock that resets whenever Droid emits progress
        # (any session/update — tool call, message chunk, tool_call chunk), and
        # a total clock capping the whole turn. The old fixed 300s wall killed
        # legitimately long orchestrator dispatches that were streaming the
        # whole time; the idle clock now fires only when Droid goes silent.
        started = loop.time()
        idle_deadline = started + _PROMPT_IDLE_TIMEOUT_SECONDS
        total_deadline = started + _PROMPT_TOTAL_TIMEOUT_SECONDS
        accumulated_text: list[str] = []

        while True:
            now = loop.time()
            if now >= idle_deadline:
                for completion in self._drain_open_task_children():
                    yield completion
                yield ExecutorError(
                    message="Timeout waiting for droid response (idle "
                    f"{_PROMPT_IDLE_TIMEOUT_SECONDS:.0f}s without progress)",
                    retryable=True,
                )
                return
            if now >= total_deadline:
                for completion in self._drain_open_task_children():
                    yield completion
                yield ExecutorError(
                    message="Timeout waiting for droid response (turn exceeded "
                    f"{_PROMPT_TOTAL_TIMEOUT_SECONDS:.0f}s total)",
                    retryable=True,
                )
                return
            remaining = min(idle_deadline, total_deadline) - now

            # Complete only once the future is resolved AND the queue is drained,
            # so trailing chunks aren't truncated.
            if fut.done() and self._queue.empty():
                try:
                    response = fut.result()
                except Exception as exc:  # noqa: BLE001
                    self._session_id = None
                    self._system_prompt_sent = False
                    for completion in self._drain_open_task_children():
                        yield completion
                    yield ExecutorError(message=f"droid process error: {exc}", retryable=True)
                    return
                if "error" in response:
                    error_msg = response["error"].get("message", "Unknown ACP error")
                    if "Session not found" in error_msg:
                        self._session_id = None
                        self._system_prompt_sent = False
                    for completion in self._drain_open_task_children():
                        yield completion
                    yield ExecutorError(message=error_msg, retryable=True)
                    return
                result = response.get("result", {}) if isinstance(response, dict) else {}
                usage = self._usage_from_result(result) if isinstance(result, dict) else None
                for completion in self._drain_open_task_children():
                    yield completion
                yield TurnComplete(response="".join(accumulated_text), usage=usage)
                return

            try:
                notification = await asyncio.wait_for(
                    self._queue.get(), timeout=min(remaining, 2.0)
                )
            except TimeoutError:
                continue

            # Any inbound notification is progress: reset the idle clock. This
            # includes benign metadata-only updates (see _BENIGN_UPDATE_TYPES),
            # which Droid emits as keepalive-ish state flips mid-turn.
            idle_deadline = loop.time() + _PROMPT_IDLE_TIMEOUT_SECONDS

            method = notification.get("method", "")
            # A malformed notification can carry a null / non-dict ``params`` (or
            # a null ``update`` inside it); coerce to an empty dict so the
            # ``.get()`` chain below never raises and drops the turn.
            params = notification.get("params")
            if not isinstance(params, dict):
                params = {}

            if method == _CLIENT_NOTIFICATION_SESSION_UPDATE:
                update = params.get("update")
                if not isinstance(update, dict):
                    update = {}
                update_type = update.get("sessionUpdate", "")

                if update_type == _UPDATE_AGENT_MESSAGE_CHUNK:
                    content = update.get("content", {})
                    text = content.get("text", "") if isinstance(content, dict) else ""
                    if text:
                        accumulated_text.append(text)
                        yield TextChunk(text=text)
                elif update_type == _UPDATE_AGENT_THOUGHT_CHUNK:
                    content = update.get("content", {})
                    text = content.get("text", "") if isinstance(content, dict) else ""
                    if text:
                        yield ReasoningChunk(delta=text, event_type="reasoning_text")
                elif update_type == _UPDATE_USAGE:
                    size = update.get("size")
                    if isinstance(size, int) and size > 0:
                        self._context_window = size
                elif update_type == _UPDATE_TOOL_CALL:
                    request_event = self._tool_call_request_event(update)
                    if request_event is not None:
                        yield request_event
                    start_event = self._task_start_event(update)
                    if start_event is not None:
                        yield start_event
                elif update_type == _UPDATE_TOOL_CALL_UPDATE:
                    complete_event = self._tool_call_complete_event(update)
                    if complete_event is not None:
                        yield complete_event
                    subagent_event = self._task_complete_event(update)
                    if subagent_event is not None:
                        yield subagent_event
                elif update_type not in _BENIGN_UPDATE_TYPES:
                    # An unrecognized (non-benign) variant — log so a future
                    # droid change surfaces instead of being silently dropped.
                    logger.debug("droid: unhandled sessionUpdate variant %r", update_type)

            elif notification.get("id") is not None and notification.get("method"):
                # Server-initiated request (session/request_permission): routes
                # through policy + elicitation. Blocks while the human decides.
                await self._respond_to_agent_request(notification)

            # Inbound message = progress; reset the idle deadline (after the
            # approval block so a slow approval doesn't time out).
            idle_deadline = loop.time() + _PROMPT_IDLE_TIMEOUT_SECONDS

    async def close_session(self, session_key: str) -> None:
        """Close a named session (no-op; the ACP session is per-process)."""

    async def interrupt_session(self, session_key: str) -> bool:  # noqa: ARG002 — one ACP session per process
        """Interrupt the in-flight turn by cancelling the active ACP session.

        Sends the ACP ``session/cancel`` notification for the live session, then
        drops the session so the next turn starts fresh (a resumed turn would
        bypass the runner's interrupt marker — mirrors cursor/pi).

        VERIFIED live: droid honors the no-id ``session/cancel`` notification —
        the in-flight ``session/prompt`` resolves with ``stopReason: "cancelled"``
        (mid-turn, before any further chunks stream).
        """
        if self._session_id is None or self._proc is None or self._proc.returncode is not None:
            return False
        try:
            await self._send(
                {
                    "jsonrpc": "2.0",
                    "method": _AGENT_NOTIFICATION_SESSION_CANCEL,
                    "params": {"sessionId": self._session_id},
                }
            )
            self._session_id = None
            self._system_prompt_sent = False
            return True
        except Exception as exc:  # noqa: BLE001 — cancel failures surface as False
            logger.debug("droid interrupt_session failed: %s", exc)
            return False

    async def close(self) -> None:
        """Terminate the droid subprocess and clean up."""
        if self._reader_task:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
            self._reader_task = None
        if self._stderr_task:
            self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stderr_task
            self._stderr_task = None
        # Release the fs-delegation OSEnvironment's helper subprocess, if one
        # was spawned for a delegated file op this session.
        if self._os_environment is not None:
            with contextlib.suppress(Exception):
                self._os_environment.close()
            self._os_environment = None
        if self._proc:
            with contextlib.suppress(Exception):
                self._proc.stdin.close()  # type: ignore[union-attr]
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except Exception:  # noqa: BLE001
                with contextlib.suppress(Exception):
                    self._proc.kill()
            finally:
                self._proc = None
