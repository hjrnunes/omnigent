"""Live model listing for the Droid harness.

The executor learns ``availableModels`` from the ``session/new`` response of a
real session. Before a session exists — the model picker, the catalog
(``GET /v1/sessions/<id>/models``) and the host's pre-launch options — the only
way to ask droid is a short ACP probe: spawn ``droid exec --output-format acp``,
send ``initialize`` + ``session/new``, read the model block, and shut droid
down. No prompt is sent, so the probe spends no tokens.

Every failure (missing binary, handshake error, timeout) degrades to an empty
listing. Droid brings its own login, so a failed probe says nothing about
whether the harness can run a turn.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from cachetools import TTLCache

from omnigent.harness_startup_config import resolve_harness_path

_logger = logging.getLogger(__name__)

_PROTOCOL_VERSION = 1
# Total budget for the whole handshake. ``session/new`` can take a while on a
# BYOK-heavy account, but the picker must never hang on a wedged CLI.
_PROBE_TIMEOUT_S = 60.0
_SHUTDOWN_TIMEOUT_S = 5.0
# Same lifetime as the catalog's provider listing cache.
_CACHE_TTL_S = 300.0

_cache: TTLCache[str, DroidModelListing] | None = None
_cache_lock = threading.Lock()


@dataclass(frozen=True)
class DroidModelOption:
    """One model droid advertises in ``session/new``.

    :param id: ACP ``modelId``, e.g. ``"custom:OpenAI-GPT-5.6-Luna-(BYOK)-4"``.
    :param name: Human-readable label; falls back to the id.
    :param description: The CLI's note (usually the token rate), if any.
    :param is_current: ``True`` for the session's ``currentModelId``.
    """

    id: str
    name: str
    description: str | None = None
    is_current: bool = False

    def to_picker_row(self) -> dict[str, object]:
        """Return the launch-picker row shape the other CLI harnesses use.

        :returns: ``id`` / ``displayName`` / ``isCurrent``, plus
            ``description`` when the CLI supplied one.
        """
        row: dict[str, object] = {
            "id": self.id,
            "displayName": self.name,
            "isCurrent": self.is_current,
        }
        if self.description:
            row["description"] = self.description
        return row


@dataclass(frozen=True)
class DroidModelListing:
    """The models droid advertises, plus the one a new session starts on.

    :param models: Picker rows in droid's advertised order.
    :param current_model_id: ACP ``currentModelId``, when the CLI reports one.
    """

    models: tuple[DroidModelOption, ...] = ()
    current_model_id: str | None = None


def parse_droid_session_models(result: object) -> DroidModelListing:
    """Parse the ``models`` block of a ``session/new`` result.

    Accepts droid's ``{availableModels: [{modelId, name, description}],
    currentModelId}`` shape (captured from Factory CLI 0.251.0). Entries
    without a usable ``modelId`` are skipped, and a missing or malformed block
    yields an empty listing — a missing picker is a lesser failure than a dead
    probe.

    :param result: The ACP ``session/new`` result object.
    :returns: The advertised options and the current model id.
    """
    if not isinstance(result, Mapping):
        return DroidModelListing()
    block = result.get("models")
    if not isinstance(block, Mapping):
        return DroidModelListing()

    raw_current = block.get("currentModelId")
    current_model_id = (
        raw_current.strip() if isinstance(raw_current, str) and raw_current.strip() else None
    )

    raw_models = block.get("availableModels")
    if not isinstance(raw_models, list):
        return DroidModelListing(current_model_id=current_model_id)

    options: list[DroidModelOption] = []
    for entry in raw_models:
        if not isinstance(entry, Mapping):
            continue
        raw_id = entry.get("modelId")
        if not isinstance(raw_id, str) or not raw_id.strip():
            continue
        model_id = raw_id.strip()
        raw_name = entry.get("name")
        name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else model_id
        raw_description = entry.get("description")
        description = (
            raw_description.strip()
            if isinstance(raw_description, str) and raw_description.strip()
            else None
        )
        options.append(
            DroidModelOption(
                id=model_id,
                name=name,
                description=description,
                is_current=model_id == current_model_id,
            )
        )
    return DroidModelListing(models=tuple(options), current_model_id=current_model_id)


def list_droid_cli_model_options(*, timeout_s: float = _PROBE_TIMEOUT_S) -> DroidModelListing:
    """Probe the installed droid CLI for its advertised model list.

    Resolves the binary the same way the executor does (``OMNIGENT_DROID_PATH``,
    then the deprecated ``HARNESS_DROID_PATH``, then ``PATH``), runs the
    initialize + ``session/new`` handshake in a scratch directory, and shuts
    droid down. Successful non-empty listings are cached for
    :data:`_CACHE_TTL_S`; failures are not cached, so installing or fixing the
    CLI recovers on the next call.

    :param timeout_s: Total budget for the handshake, in seconds.
    :returns: The live listing, or an empty one on any failure.
    """
    command = resolve_harness_path("droid") or "droid"
    cached = _cached_listing(command)
    if cached is not None:
        return cached
    try:
        listing = _probe_model_options(command, timeout_s=timeout_s)
    except Exception:  # Any failure degrades to "no listing", never a crash.
        _logger.debug("droid model listing probe failed for %s", command, exc_info=True)
        return DroidModelListing()
    if listing.models:
        _store_listing(command, listing)
    return listing


def clear_droid_model_cache() -> None:
    """Drop the cached probe result (tests and explicit re-probes)."""
    global _cache
    with _cache_lock:
        _cache = None


def _cached_listing(command: str) -> DroidModelListing | None:
    """Return the cached listing for *command* while it is still fresh."""
    with _cache_lock:
        return cast("DroidModelListing | None", _get_cache().get(command))


def _store_listing(command: str, listing: DroidModelListing) -> None:
    """Cache one successful probe result."""
    with _cache_lock:
        _get_cache()[command] = listing


def _get_cache() -> TTLCache[str, DroidModelListing]:
    """Return the process-wide probe cache, building it on first use."""
    global _cache
    if _cache is None:
        _cache = TTLCache(maxsize=8, ttl=_CACHE_TTL_S)
    return _cache


def _probe_model_options(command: str, *, timeout_s: float) -> DroidModelListing:
    """Run one ACP handshake against *command* and parse its models.

    Raises on any transport or protocol failure; the public entry point turns
    that into an empty listing.

    :param command: The droid binary to spawn.
    :param timeout_s: Total handshake budget, in seconds.
    :returns: The parsed listing.
    """
    deadline = time.monotonic() + timeout_s
    with tempfile.TemporaryDirectory(prefix="omnigent-droid-models-") as cwd:
        argv = [command, "exec", "--output-format", "acp", "--cwd", cwd]
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # A chatty CLI must not fill a pipe we never drain; diagnostics are
            # not worth the extra reader for a picker probe.
            stderr=subprocess.DEVNULL,
            cwd=cwd,
            env=os.environ.copy(),
            text=True,
            bufsize=1,
        )
        try:
            lines: queue.Queue[str | None] = queue.Queue()
            threading.Thread(
                target=_pump_stdout, args=(proc, lines), name="droid-models-reader", daemon=True
            ).start()
            _request(
                proc,
                lines,
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": _initialize_params(),
                },
                deadline=deadline,
            )
            result = _request(
                proc,
                lines,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "session/new",
                    "params": {"cwd": cwd, "mcpServers": []},
                },
                deadline=deadline,
            )
            return parse_droid_session_models(result)
        finally:
            _shutdown(proc)


def _pump_stdout(proc: subprocess.Popen[str], lines: queue.Queue[str | None]) -> None:
    """Forward droid stdout lines to *lines*, ending with ``None`` on EOF."""
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            lines.put(line)
    finally:
        lines.put(None)


def _request(
    proc: subprocess.Popen[str],
    lines: queue.Queue[str | None],
    message: dict[str, Any],  # type: ignore[explicit-any]
    *,
    deadline: float,
) -> dict[str, Any]:  # type: ignore[explicit-any]
    """Send one JSON-RPC request and wait for its response.

    Notifications and unrelated requests on the same stream are ignored.

    :param proc: The running droid process.
    :param lines: The stdout queue fed by :func:`_pump_stdout`.
    :param message: The JSON-RPC request, including its ``id``.
    :param deadline: ``time.monotonic()`` instant the handshake must finish by.
    :returns: The response ``result`` object.
    :raises TimeoutError: When the deadline passes first.
    :raises RuntimeError: When droid exits or reports an error.
    """
    assert proc.stdin is not None
    request_id = message["id"]
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"droid did not answer {message['method']} in time")
        try:
            line = lines.get(timeout=min(remaining, 0.5))
        except queue.Empty:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"droid exited with {proc.returncode} before answering {message['method']}"
                ) from None
            continue
        if line is None:
            raise RuntimeError(f"droid closed stdout before answering {message['method']}")
        try:
            reply = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(reply, dict) or reply.get("id") != request_id:
            continue
        if "error" in reply:
            error = reply["error"]
            detail = error.get("message", error) if isinstance(error, dict) else error
            raise RuntimeError(f"droid rejected {message['method']}: {detail}")
        result = reply.get("result")
        return result if isinstance(result, dict) else {}


def _initialize_params() -> dict[str, Any]:  # type: ignore[explicit-any]
    """Build the ``initialize`` params shared with the executor's handshake."""
    return {
        "protocolVersion": _PROTOCOL_VERSION,
        "clientInfo": {"name": "omnigent", "version": "1.0"},
        "clientCapabilities": {
            "fs": {"readTextFile": False, "writeTextFile": False},
            "terminal": False,
        },
    }


def _shutdown(proc: subprocess.Popen[str]) -> None:
    """Terminate the probe process and reap it, escalating to SIGKILL."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=_SHUTDOWN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=_SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                _logger.warning("droid model probe process %s did not exit", proc.pid)
    for stream in (proc.stdin, proc.stdout):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
