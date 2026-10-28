"""Spawn-env builder for the Droid harness.

Its own module so the runner can resolve it lazily by the string path in
``spawn_env_builders``: only a session that actually selects Droid pays for the
import.

Without a registered builder, core's ``_build_spawn_env_from_spec`` returns
``None`` for ``droid`` — and the per-session ``/model`` override it applies
afterwards is gated on ``env is not None``, so the picker's choice would be
silently dropped before it could reach ``HARNESS_DROID_MODEL``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_spawn_env(
    spec: Any,  # type: ignore[explicit-any]  # AgentSpec; typed loosely to keep this import-light
    *,
    cwd: Path | None = None,
    workdir: Path | None = None,  # noqa: ARG001 — signature parity with core's builders
) -> dict[str, str]:
    """Map an agent spec onto the ``HARNESS_DROID_*`` vars the wrap reads.

    Droid owns its own auth (``FACTORY_API_KEY`` / ``droid login``), so this
    wires no credential — only the model, workspace and os_env/sandbox spec.

    :param spec: The agent spec.
    :param cwd: Session workspace; ``None`` lets the wrap fall back to
        ``OMNIGENT_RUNNER_WORKSPACE``.
    :param workdir: The bundle's on-disk path. Unused — Droid has no skills
        bridge — but accepted so this matches the other builders' signature.
    :returns: Env-var overrides for the harness process spawn.
    """
    from omnigent.runtime.workflow import (
        _apply_harness_path_override,
        _resolve_spec_model,
        _serialize_os_env,
    )

    env: dict[str, str] = {}
    model = _resolve_spec_model(spec)
    if model:
        env["HARNESS_DROID_MODEL"] = model
    if cwd is not None:
        env["HARNESS_DROID_CWD"] = str(cwd)
    os_env_payload = _serialize_os_env(spec.os_env)
    if os_env_payload is not None:
        env["HARNESS_DROID_OS_ENV"] = os_env_payload
    _apply_harness_path_override(env, "droid")
    return env
