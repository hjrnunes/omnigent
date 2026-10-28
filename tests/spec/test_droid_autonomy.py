"""``executor.autonomy`` survives YAML loading and bundle transport."""

from pathlib import Path

import pytest
import yaml

from omnigent.chat import _bundle_agent
from omnigent.errors import OmnigentError
from omnigent.spec import load
from omnigent.spec.omnigent import agent_def_to_agent_spec, agent_spec_to_agent_def


def _write_agent(
    tmp_path: Path,
    *,
    directory: bool,
    executor: dict,
    instructions: str | None = "Explicit agent instructions.",
) -> Path:
    source = tmp_path / "agent"
    source.mkdir()
    if directory:
        executor_config = dict(executor)
        model = executor_config.pop("model", None)
        config = {
            "spec_version": 1,
            "name": "droid-autonomy",
            "executor": {"type": "omnigent", "model": model, "config": executor_config},
            "instructions": "AGENTS.md" if instructions is not None else None,
        }
        if instructions is not None:
            (source / "AGENTS.md").write_text(instructions)
        (source / "config.yaml").write_text(yaml.safe_dump(config))
        return source
    config = {"name": "droid-autonomy", "executor": executor, "prompt": instructions}
    path = source / "agent.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.parametrize("directory", [False, True], ids=["yaml", "bundle"])
@pytest.mark.parametrize(
    "autonomy", [None, "normal", "auto-medium"], ids=["default", "normal", "auto-medium"]
)
def test_autonomy_survives_bundle_transport(
    tmp_path: Path, directory: bool, autonomy: str | None
) -> None:
    executor: dict[str, object] = {"harness": "droid", "model": "gpt-6-luna"}
    if autonomy is not None:
        executor["autonomy"] = autonomy
    path = _write_agent(tmp_path, directory=directory, executor=executor)

    spec = load(_bundle_agent(path), dest=tmp_path / "uploaded")

    for candidate in (spec, agent_def_to_agent_spec(agent_spec_to_agent_def(spec))):
        assert candidate.executor.config.get("autonomy") == autonomy


@pytest.mark.parametrize("directory", [False, True], ids=["yaml", "bundle"])
@pytest.mark.parametrize("value", [True, 1, [], {}])
def test_autonomy_rejects_non_string_values(
    tmp_path: Path, directory: bool, value: object
) -> None:
    path = _write_agent(
        tmp_path, directory=directory, executor={"harness": "droid", "autonomy": value}
    )

    with pytest.raises((OmnigentError, ValueError), match=r"autonomy.*must be a string"):
        load(path)


@pytest.mark.parametrize("harness", ["codex"])
def test_autonomy_rejects_unsupported_harness(tmp_path: Path, harness: str) -> None:
    path = _write_agent(
        tmp_path, directory=False, executor={"harness": harness, "autonomy": "normal"}
    )

    with pytest.raises(OmnigentError, match=r"autonomy.*only supported.*droid"):
        load(path)


def test_inline_subagent_preserves_autonomy(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "parent",
                "prompt": "Parent instructions.",
                "executor": {"harness": "droid", "model": "gpt-6-luna", "autonomy": "normal"},
                "tools": {
                    "worker": {
                        "type": "agent",
                        "prompt": "Worker instructions.",
                        "executor": {
                            "harness": "droid",
                            "model": "gpt-6-luna",
                            "autonomy": "spec",
                        },
                    }
                },
            }
        )
    )

    spec = load(path)
    worker = spec.sub_agents[0]

    assert spec.executor.config["autonomy"] == "normal"
    assert worker.executor.config["autonomy"] == "spec"
