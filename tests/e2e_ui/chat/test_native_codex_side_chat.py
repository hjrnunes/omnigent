"""Real Codex app-server side-chat lifecycle with scripted local model replies."""

from __future__ import annotations

import json
import shutil

import httpx
import pytest
from playwright.sync_api import Page, expect

from tests.e2e.conftest import get_mock_requests
from tests.e2e_ui.conftest import configure_mock_llm, reset_mock_llm, set_fallback_mock_llm
from tests.e2e_ui.messages.test_message_render_parity import _ensure_chat_view
from tests.e2e_ui.messages.test_native_codex_render_parity import (
    _open_terminal_view,
    _wait_terminal_connected,
)

from .test_side_chat_entrypoints import _ASSISTANT, _items, _send_parent


@pytest.mark.nightly
@pytest.mark.parametrize("mobile", [False, True], ids=["desktop", "mobile"])
@pytest.mark.timeout(300)
@pytest.mark.skipif(
    not shutil.which("codex") or not shutil.which("tmux"), reason="codex and tmux are required"
)
def test_native_codex_side_chat_inherits_context_and_closes_independently(
    page: Page,
    native_codex_mock_session: tuple[str, str],
    mock_llm_server_url: str,
    mobile: bool,
) -> None:
    """Native forks inherit context, isolate follow-ups, and leave the parent usable."""
    base_url, session_id = native_codex_mock_session
    page.goto(f"{base_url}/c/{session_id}")
    _open_terminal_view(page)
    _wait_terminal_connected(page)
    _ensure_chat_view(page)
    if mobile:
        page.set_viewport_size({"width": 390, "height": 844})
        page.get_by_role("button", name="Close sidebar", exact=True).click(
            position={"x": 375, "y": 400}
        )

    parent_question = f"native-turn-1-{session_id}: remember our parent context"
    parent_reply = "The native parent context is ready."
    question = f"native-turn-2-{session_id}: answer a side question"
    followup = f"native-turn-3-{session_id}: answer a follow-up"
    resumed_question = f"native-turn-4-{session_id}: answer in the parent"
    reset_mock_llm(mock_llm_server_url)
    for prompt, reply in (
        (parent_question, parent_reply),
        (question, "First native side answer."),
        (followup, "Second native side answer."),
        (resumed_question, "The native parent still works."),
    ):
        configure_mock_llm(
            mock_llm_server_url,
            [{"text": reply}],
            key=prompt.split(":")[0],
            match=prompt.split(":")[0],
            required_tools=[],
        )
        set_fallback_mock_llm(mock_llm_server_url, prompt.split(":")[0], reply)
    set_fallback_mock_llm(mock_llm_server_url, "gpt-4o", "")

    _send_parent(page, parent_question, parent_reply)
    parent_items = _items(base_url, session_id)
    page.get_by_test_id("composer-attach").click()
    page.get_by_role("menuitem", name="Start a new side chat").click()
    pane = page.locator(".side-chat-backdrop")
    settings = pane.get_by_role("button", name="Side chat model and effort")
    settings.click()
    page.get_by_test_id("side-chat-model-edit").click()
    page.locator('[data-model-id="gpt-6-sol"]').click()
    page.locator('[data-effort-level="high"]').click()
    page.keyboard.press("Escape")
    page.get_by_test_id("side-chat-input").fill(question)
    page.get_by_test_id("side-chat-send").click()
    expect(pane.locator(_ASSISTANT).filter(has_text="First native side answer.")).to_be_visible(
        timeout=60_000
    )
    expect(pane.get_by_text(parent_reply, exact=True)).to_have_count(0)

    response = httpx.get(f"{base_url}/v1/sessions/{session_id}/child_sessions", timeout=10.0)
    response.raise_for_status()
    children = [
        child
        for child in response.json()["data"]
        if child.get("labels", {}).get("omnigent.codex_native.agent_nickname") == "Side chat"
    ]
    assert len(children) == 1
    child_id = children[0]["id"]
    parent_response = httpx.get(f"{base_url}/v1/sessions/{session_id}", timeout=10.0)
    parent_response.raise_for_status()
    child_response = httpx.get(f"{base_url}/v1/sessions/{child_id}", timeout=10.0)
    child_response.raise_for_status()
    parent, child = parent_response.json(), child_response.json()
    assert child["runner_id"] == parent["runner_id"]
    assert child["model_override"] == "gpt-6-sol"
    assert child["reasoning_effort"] == "high"
    first_requests = [
        request
        for request in get_mock_requests(mock_llm_server_url)
        if any(
            item.get("role") == "user"
            and item.get("content") == [{"type": "input_text", "text": question}]
            for item in request.get("input", [])
        )
    ]
    assert first_requests
    assert {
        (request["model"], request.get("reasoning", {}).get("effort"))
        for request in first_requests
    } == {("gpt-6-sol", "high")}
    child_thread_id = child["labels"]["omnigent.codex_native.subagent_thread_id"]
    assert child_thread_id
    assert child_thread_id != parent["external_session_id"]

    expect(pane.get_by_test_id("working-indicator")).to_have_count(0, timeout=30_000)
    settings = pane.get_by_role("button", name="Side chat model and effort")
    expect(settings).to_be_visible(timeout=30_000)
    settings.click()
    page.get_by_test_id("side-chat-model-edit").click()
    with page.expect_response(f"**/v1/sessions/{child_id}") as picked_model:
        page.locator('[data-model-id="gpt-6-sol"]').click()
    assert picked_model.value.ok
    with page.expect_response(f"**/v1/sessions/{child_id}") as picked_effort:
        page.locator('[data-effort-level="max"]').click()
    assert picked_effort.value.ok
    page.keyboard.press("Escape")
    expect(settings).to_contain_text("Max")
    expected_followup_effort = "max"
    if mobile:
        # A cleared pick must reset the thread's effort rather than inherit the previous turn.
        reset = httpx.patch(
            f"{base_url}/v1/sessions/{child_id}",
            json={"reasoning_effort": "default"},
            timeout=10.0,
        )
        reset.raise_for_status()
        assert reset.json()["reasoning_effort"] is None
        expected_followup_effort = next(
            model["defaultReasoningEffort"]
            for model in parent["model_options"]
            if model["id"] == "gpt-6-sol"
        )
    page.get_by_test_id("side-chat-input").fill(followup)
    expect(page.get_by_test_id("side-chat-send")).to_be_enabled(timeout=30_000)
    page.get_by_test_id("side-chat-send").click()
    expect(pane.locator(_ASSISTANT).filter(has_text="Second native side answer.")).to_be_visible(
        timeout=60_000
    )
    expect(pane.get_by_test_id("working-indicator")).to_have_count(0, timeout=30_000)
    assert _items(base_url, session_id) == parent_items
    child_text = str(_items(base_url, child_id))
    for text in (question, followup, "First native side answer.", "Second native side answer."):
        assert text in child_text

    model_inputs = [
        json.dumps(request.get("input", [])) for request in get_mock_requests(mock_llm_server_url)
    ]
    first_input = next(text for text in model_inputs if question in text and followup not in text)
    followup_input = next(text for text in model_inputs if followup in text)
    assert parent_question in first_input
    assert parent_reply in first_input
    assert question in followup_input
    assert "First native side answer." in followup_input
    side_requests = [
        request
        for request in get_mock_requests(mock_llm_server_url)
        if any(
            item.get("role") == "user"
            and item.get("content") == [{"type": "input_text", "text": followup}]
            for item in request.get("input", [])
        )
    ]
    assert side_requests
    assert {
        (request["model"], request.get("reasoning", {}).get("effort")) for request in side_requests
    } == {("gpt-6-sol", expected_followup_effort)}

    with page.expect_response(f"**/v1/sessions/{child_id}/events") as stopped:
        page.get_by_role("button", name="Close Side chat 1", exact=True).click()
    assert stopped.value.ok
    expect(page.get_by_test_id("side-chat-input")).to_have_count(0)
    child_response = httpx.get(f"{base_url}/v1/sessions/{child_id}", timeout=10.0)
    child_response.raise_for_status()
    assert child_response.json()["labels"]["omnigent.closed"] == "true"
    if mobile:
        page.get_by_test_id("side-chats-panel-drawer").get_by_role(
            "button", name="Close", exact=True
        ).click()
    _send_parent(page, resumed_question, "The native parent still works.")
    parent_text = str(_items(base_url, session_id))
    assert question not in parent_text
    assert followup not in parent_text
    parent_requests = [
        request
        for request in get_mock_requests(mock_llm_server_url)
        if any(
            item.get("role") == "user"
            and item.get("content") == [{"type": "input_text", "text": resumed_question}]
            for item in request.get("input", [])
        )
    ]
    assert parent_requests
    assert {request["model"] for request in parent_requests} == {"gpt-4o"}
