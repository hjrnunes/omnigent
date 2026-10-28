"""Tests for the live Droid model listing probe.

``list_droid_cli_model_options`` asks ``droid exec --output-format acp`` for
its advertised models with only ``initialize`` + ``session/new`` — the same
handshake the executor performs, minus any prompt — and parses
``models.availableModels`` / ``models.currentModelId``. The captured payload
below is trimmed from a live Factory CLI 0.251.0 ``session/new`` response; the
fake peer replays that wire shape so the probe runs without the real CLI.
"""

from __future__ import annotations

import json
import textwrap
import time
from pathlib import Path

import pytest

from omnigent.inner.droid_models import (
    DroidModelListing,
    DroidModelOption,
    clear_droid_model_cache,
    list_droid_cli_model_options,
    parse_droid_session_models,
)

# Trimmed from a live droid 0.251.0 ``session/new`` result. The real response
# carries 93 models (24 ``custom:`` ids) and the full ``configOptions`` /
# ``modes`` blocks; the probe ignores those, so the fixture keeps the shapes
# that matter: a null description, parenthesized and bracketed custom ids,
# and a ``currentModelId`` that is itself an available model.
CAPTURED_SESSION_NEW_RESULT: dict[str, object] = {
    "sessionId": "c4d34297-ea02-42cd-bc24-2ec5e294a5d9",
    "models": {
        "currentModelId": "gpt-6-luna",
        "availableModels": [
            {"modelId": "auto", "name": "Auto Model", "description": "1x Factory token rate"},
            {
                "modelId": "gpt-6-luna",
                "name": "GPT-6 Luna",
                "description": "0.04x Factory token rate",
            },
            {
                "modelId": "custom:OpenAI-GPT-5.6-Luna-(BYOK)-4",
                "name": "Red Hat GPT-5.6 Luna",
                "description": None,
            },
            {
                "modelId": "custom:Kimi-K2-[Groq]-0",
                "name": "Kimi K2 Groq",
                "description": "2x Factory token rate",
            },
            {
                "modelId": "custom:cli-proxy-api:opus-5-5",
                "name": "CLIProxyAPI Opus 5.5",
                "description": None,
            },
        ],
    },
    "configOptions": [{"id": "autonomy_level", "currentValue": "auto-medium"}],
    "modes": {"currentModeId": "auto-medium", "availableModes": []},
}

_FAKE_DROID_SCRIPT = r"""
import json, os, signal, sys

log_path = os.environ["FAKE_DROID_LOG"]
session_new_path = os.environ.get("FAKE_DROID_SESSION_NEW")
initialize_result = json.loads(os.environ.get("FAKE_DROID_INITIALIZE", "{}"))


def record(line):
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(line + "\n")


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def on_term(signum, frame):
    record("TERM")
    raise SystemExit(0)


signal.signal(signal.SIGTERM, on_term)
record("ARGV " + json.dumps(sys.argv[1:]))

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    record("CALL " + line)
    msg = json.loads(line)
    mid, method = msg.get("id"), msg.get("method")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": mid, "result": initialize_result})
    elif method == "session/new" and session_new_path:
        with open(session_new_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        if "error" in payload:
            send({"jsonrpc": "2.0", "id": mid, "error": payload["error"]})
        else:
            send({"jsonrpc": "2.0", "id": mid, "result": payload})
    elif method == "session/new":
        # Stay alive without answering so the probe hits its timeout.
        continue
    else:
        send({"jsonrpc": "2.0", "id": mid, "result": {}})
record("EOF")
"""


@pytest.fixture(autouse=True)
def _fresh_model_cache() -> None:
    """Keep the module-level probe cache from leaking between tests."""
    clear_droid_model_cache()


@pytest.fixture
def fake_droid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install a fake ``droid`` binary and point the path override at it.

    :param tmp_path: Per-test temp dir.
    :param monkeypatch: Pytest monkeypatch fixture.
    :returns: Path to the fake binary (also set as ``OMNIGENT_DROID_PATH``).
    """
    binary = tmp_path / "droid"
    binary.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(_FAKE_DROID_SCRIPT))
    binary.chmod(0o755)
    monkeypatch.setenv("OMNIGENT_DROID_PATH", str(binary))
    monkeypatch.setenv("FAKE_DROID_LOG", str(tmp_path / "droid.log"))
    monkeypatch.setenv(
        "FAKE_DROID_INITIALIZE",
        json.dumps(
            {
                "protocolVersion": 1,
                "agentInfo": {"name": "@factory/cli", "title": "Factory Droid"},
            }
        ),
    )
    return binary


def _install_session_new_reply(
    tmp_path: Path, payload: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write the fake peer's ``session/new`` reply and point the peer at it.

    :param tmp_path: Per-test temp dir.
    :param payload: A ``session/new`` result object or an error envelope.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    reply = tmp_path / "session-new.json"
    reply.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("FAKE_DROID_SESSION_NEW", str(reply))


def _log_lines(tmp_path: Path) -> list[str]:
    """Return the fake peer's recorded lines."""
    log = tmp_path / "droid.log"
    return log.read_text(encoding="utf-8").splitlines() if log.exists() else []


# ── Parser ─────────────────────────────────────────────────


def test_parse_session_models_reads_the_captured_payload() -> None:
    """The parser maps droid's ``availableModels`` onto picker options."""
    listing = parse_droid_session_models(CAPTURED_SESSION_NEW_RESULT)

    assert listing == DroidModelListing(
        models=(
            DroidModelOption(id="auto", name="Auto Model", description="1x Factory token rate"),
            DroidModelOption(
                id="gpt-6-luna",
                name="GPT-6 Luna",
                description="0.04x Factory token rate",
                is_current=True,
            ),
            DroidModelOption(
                id="custom:OpenAI-GPT-5.6-Luna-(BYOK)-4",
                name="Red Hat GPT-5.6 Luna",
                description=None,
            ),
            DroidModelOption(
                id="custom:Kimi-K2-[Groq]-0",
                name="Kimi K2 Groq",
                description="2x Factory token rate",
            ),
            DroidModelOption(
                id="custom:cli-proxy-api:opus-5-5",
                name="CLIProxyAPI Opus 5.5",
                description=None,
            ),
        ),
        current_model_id="gpt-6-luna",
    )


def test_parse_session_models_reports_the_current_model_id() -> None:
    """``currentModelId`` survives even when it is not in ``availableModels``."""
    listing = parse_droid_session_models(
        {"models": {"availableModels": [{"modelId": "auto"}], "currentModelId": "gpt-6-sol"}}
    )

    assert listing.current_model_id == "gpt-6-sol"
    assert listing.models == (DroidModelOption(id="auto", name="auto"),)


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"models": None},
        {"models": {"availableModels": "nope"}},
        {"models": {"availableModels": []}},
        {"models": {"availableModels": [1, None, "x", {}]}},
    ],
)
def test_parse_session_models_ignores_malformed_blocks(result: object) -> None:
    """A session without a usable model block yields an empty listing."""
    assert parse_droid_session_models(result) == DroidModelListing()


def test_parse_session_models_skips_entries_without_an_id() -> None:
    """Only entries naming a model id become picker rows."""
    listing = parse_droid_session_models(
        {
            "models": {
                "availableModels": [
                    {"name": "missing id"},
                    {"modelId": "", "name": "empty id"},
                    {"modelId": "gpt-6-sol", "name": "", "description": "  "},
                ]
            }
        }
    )

    assert listing.models == (DroidModelOption(id="gpt-6-sol", name="gpt-6-sol"),)


# ── Probe ──────────────────────────────────────────────────


def test_list_probes_with_initialize_and_session_new_only(
    fake_droid: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe handshakes without a prompt and shuts droid down afterwards."""
    _install_session_new_reply(tmp_path, CAPTURED_SESSION_NEW_RESULT, monkeypatch)

    listing = list_droid_cli_model_options()

    assert listing.current_model_id == "gpt-6-luna"
    assert [option.id for option in listing.models][:2] == ["auto", "gpt-6-luna"]

    lines = _log_lines(tmp_path)
    argv_line = next(line for line in lines if line.startswith("ARGV "))
    argv = json.loads(argv_line.removeprefix("ARGV "))
    assert argv[:3] == ["exec", "--output-format", "acp"]
    assert argv[3] == "--cwd"
    # The probe runs in its own scratch directory, never the caller's cwd.
    assert Path(argv[4]).exists() is False

    calls = [json.loads(line.removeprefix("CALL ")) for line in lines if line.startswith("CALL ")]
    assert [call["method"] for call in calls] == ["initialize", "session/new"]
    assert calls[1]["params"]["mcpServers"] == []
    assert calls[1]["params"]["cwd"] == argv[4]
    assert "TERM" in lines


def test_list_uses_the_executor_path_override(
    fake_droid: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``OMNIGENT_DROID_PATH`` (the executor's override) selects the binary."""
    _install_session_new_reply(tmp_path, CAPTURED_SESSION_NEW_RESULT, monkeypatch)

    listing = list_droid_cli_model_options()

    assert listing.models
    assert any(line.startswith("ARGV ") for line in _log_lines(tmp_path))


def test_list_returns_empty_when_the_binary_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unavailable droid binary degrades to an empty listing, never raises."""
    monkeypatch.setenv("OMNIGENT_DROID_PATH", str(tmp_path / "does-not-exist"))

    assert list_droid_cli_model_options() == DroidModelListing()


def test_list_returns_empty_when_session_new_fails(
    fake_droid: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``-32602``-style reply yields an empty listing."""
    _install_session_new_reply(
        tmp_path,
        {"error": {"code": -32602, "message": "Invalid params: Model not recognized"}},
        monkeypatch,
    )

    assert list_droid_cli_model_options() == DroidModelListing()


def test_list_returns_empty_when_session_new_times_out(
    fake_droid: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A silent peer hits the probe deadline instead of hanging the picker."""
    monkeypatch.delenv("FAKE_DROID_SESSION_NEW", raising=False)

    started = time.monotonic()
    listing = list_droid_cli_model_options(timeout_s=0.3)
    elapsed = time.monotonic() - started

    assert listing == DroidModelListing()
    assert elapsed < 5.0
    # The hung peer is still torn down.
    assert "TERM" in _log_lines(tmp_path)


def test_list_caches_a_successful_listing(
    fake_droid: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repeat call replays the listing instead of spawning droid again."""
    _install_session_new_reply(tmp_path, CAPTURED_SESSION_NEW_RESULT, monkeypatch)

    first = list_droid_cli_model_options()
    second = list_droid_cli_model_options()

    assert first == second
    assert sum(1 for line in _log_lines(tmp_path) if line.startswith("ARGV ")) == 1

    clear_droid_model_cache()
    list_droid_cli_model_options()
    assert sum(1 for line in _log_lines(tmp_path) if line.startswith("ARGV ")) == 2


def test_list_reprobes_after_the_cache_ttl(
    fake_droid: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache expires, so a model added in droid shows up without a restart."""
    import omnigent.inner.droid_models as droid_models

    monkeypatch.setattr(droid_models, "_CACHE_TTL_S", 0.05)
    _install_session_new_reply(tmp_path, CAPTURED_SESSION_NEW_RESULT, monkeypatch)

    list_droid_cli_model_options()
    time.sleep(0.15)
    list_droid_cli_model_options()

    assert sum(1 for line in _log_lines(tmp_path) if line.startswith("ARGV ")) == 2


def test_list_does_not_cache_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed probe retries on the next call, so installing droid recovers."""
    monkeypatch.setenv("OMNIGENT_DROID_PATH", str(tmp_path / "does-not-exist"))
    assert list_droid_cli_model_options() == DroidModelListing()

    monkeypatch.setenv("FAKE_DROID_LOG", str(tmp_path / "droid.log"))
    monkeypatch.setenv("FAKE_DROID_INITIALIZE", json.dumps({"protocolVersion": 1}))
    binary = tmp_path / "droid"
    binary.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(_FAKE_DROID_SCRIPT))
    binary.chmod(0o755)
    monkeypatch.setenv("OMNIGENT_DROID_PATH", str(binary))
    _install_session_new_reply(tmp_path, CAPTURED_SESSION_NEW_RESULT, monkeypatch)

    assert list_droid_cli_model_options().models
