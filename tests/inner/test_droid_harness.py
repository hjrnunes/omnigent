"""Unit tests for the in-tree ``harness: droid`` wrap.

Covers binary-path resolution for both env vars the executor honors
(``OMNIGENT_DROID_PATH`` and the plugin-era ``HARNESS_DROID_PATH``) plus the
model/cwd env mapping. No ``droid`` binary is spawned.
"""

from __future__ import annotations

import pytest

from omnigent.inner.droid_harness import _build_droid_executor


@pytest.fixture(autouse=True)
def _clear_droid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OMNIGENT_DROID_PATH", raising=False)
    monkeypatch.delenv("HARNESS_DROID_PATH", raising=False)
    for name in (
        "HARNESS_DROID_MODEL",
        "HARNESS_DROID_CWD",
        "HARNESS_DROID_AUTO",
        "HARNESS_DROID_REASONING",
        "HARNESS_DROID_OS_ENV",
    ):
        monkeypatch.delenv(name, raising=False)


def test_executor_defaults_to_path_lookup() -> None:
    executor = _build_droid_executor()

    assert executor._droid_path == "droid"  # type: ignore[attr-defined]


def test_executor_honors_omnigent_droid_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIGENT_DROID_PATH", "/opt/omni/bin/droid")

    executor = _build_droid_executor()

    assert executor._droid_path == "/opt/omni/bin/droid"  # type: ignore[attr-defined]


def test_executor_honors_harness_droid_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The installed plugin's ``HARNESS_DROID_PATH`` keeps working."""
    from omnigent.harness_startup_config import _LEGACY_PATH_WARNED

    _LEGACY_PATH_WARNED.discard("HARNESS_DROID_PATH")
    monkeypatch.setenv("HARNESS_DROID_PATH", "/usr/local/bin/droid")

    executor = _build_droid_executor()

    assert executor._droid_path == "/usr/local/bin/droid"  # type: ignore[attr-defined]


def test_omnigent_droid_path_wins_over_harness_droid_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMNIGENT_DROID_PATH", "/canonical/droid")
    monkeypatch.setenv("HARNESS_DROID_PATH", "/legacy/droid")

    executor = _build_droid_executor()

    assert executor._droid_path == "/canonical/droid"  # type: ignore[attr-defined]


def test_executor_reads_model_and_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_DROID_MODEL", "gpt-6-luna")
    monkeypatch.setenv("HARNESS_DROID_CWD", "/tmp/droid-workspace")

    executor = _build_droid_executor()

    assert executor._model == "gpt-6-luna"  # type: ignore[attr-defined]
    assert executor._cwd == "/tmp/droid-workspace"  # type: ignore[attr-defined]
