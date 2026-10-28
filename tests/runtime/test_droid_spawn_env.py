"""``HARNESS_DROID_*`` mapping for the droid spawn-env builder.

A unit test — no subprocess spawn, no real droid CLI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnigent.inner.droid_spawn_env import build_spawn_env
from omnigent.spec.types import AgentSpec, ExecutorSpec


@pytest.fixture(autouse=True)
def _isolate_config_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep the developer's real ``~/.omnigent/config.yaml`` out of the builder.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param tmp_path: Temporary directory used as the config home.
    """
    monkeypatch.setenv("OMNIGENT_CONFIG_HOME", str(tmp_path))


def _make_spec(*, autonomy: str | None = None, model: str | None = "gpt-6-luna") -> AgentSpec:
    """Build a minimal droid :class:`AgentSpec`.

    :param autonomy: Value for ``executor.config["autonomy"]``; ``None`` omits it.
    :param model: Model id for the spec, or ``None`` for no model.
    :returns: A populated :class:`AgentSpec`.
    """
    config: dict[str, object] = {"harness": "droid"}
    if model is not None:
        config["model"] = model
    if autonomy is not None:
        config["autonomy"] = autonomy
    return AgentSpec(
        spec_version=1,
        name="test-droid",
        instructions="You are a test agent.",
        executor=ExecutorSpec(type="omnigent", config=config, model=model),
    )


def test_droid_spawn_env_sets_model_and_cwd(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()

    env = build_spawn_env(_make_spec(), cwd=workspace, workdir=tmp_path / "bundle")

    assert env["HARNESS_DROID_MODEL"] == "gpt-6-luna"
    assert env["HARNESS_DROID_CWD"] == str(workspace)


@pytest.mark.parametrize("autonomy", ["normal", "spec", "auto-low", "auto-medium", "auto-high"])
def test_droid_spawn_env_carries_autonomy(autonomy: str) -> None:
    env = build_spawn_env(_make_spec(autonomy=autonomy))

    assert env["HARNESS_DROID_AUTO"] == autonomy


@pytest.mark.parametrize(
    "autonomy", [None, "inherit", "", "   "], ids=["unset", "inherit", "empty", "blank"]
)
def test_droid_spawn_env_leaves_default_autonomy_to_droid(autonomy: str | None) -> None:
    """Unset and ``inherit`` must not pin a tier: droid keeps its settings file."""
    env = build_spawn_env(_make_spec(autonomy=autonomy))

    assert "HARNESS_DROID_AUTO" not in env
