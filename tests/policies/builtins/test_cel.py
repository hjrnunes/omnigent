"""Tests for the cel_policy builtin factory."""

from __future__ import annotations

import random
from collections.abc import Iterator

import pytest

pytest.importorskip("celpy", reason="cel-python not installed")

import celpy
import celpy.celtypes
from celpy.adapter import json_to_cel
from celpy.evaluation import CELEvalError, CELFunction, Context, Evaluator

from omnigent.policies.builtins import cel as cel_module
from omnigent.policies.builtins._cel_short_circuit import ShortCircuitRunner
from omnigent.policies.builtins.cel import cel_policy
from omnigent.policies.function import FunctionPolicy
from omnigent.policies.types import EvaluationContext
from omnigent.spec.types import (
    FunctionPolicySpec,
    Phase,
    PhaseSelector,
    PolicyAction,
    StateUpdateAction,
)

# ── Map return: DENY ────────────────────────────────────────────


def test_deny_matching_tool_call() -> None:
    """Expression returning DENY map on tool_call match."""
    evaluate = cel_policy(
        expression=(
            'event.type == "tool_call" && event.data.name == "sys_os_shell"'
            ' ? {"result": "DENY", "reason": "Shell blocked."}'
            ' : {"result": "ALLOW"}'
        ),
    )
    result = evaluate(
        {
            "type": "tool_call",
            "data": {"name": "sys_os_shell", "arguments": {}},
        }
    )
    assert result == {"result": "DENY", "reason": "Shell blocked."}


def test_allow_non_matching_tool_call() -> None:
    """Non-matching tool call returns ALLOW."""
    evaluate = cel_policy(
        expression=(
            'event.type == "tool_call" && event.data.name == "sys_os_shell"'
            ' ? {"result": "DENY", "reason": "Shell blocked."}'
            ' : {"result": "ALLOW"}'
        ),
    )
    result = evaluate(
        {
            "type": "tool_call",
            "data": {"name": "web_search", "arguments": {}},
        }
    )
    assert result == {"result": "ALLOW"}


def test_deny_with_fallback_reason() -> None:
    """Map without reason key uses the factory default."""
    evaluate = cel_policy(
        expression='{"result": "DENY"}',
        reason="Factory default.",
    )
    result = evaluate({"type": "request"})
    assert result == {"result": "DENY", "reason": "Factory default."}


def test_deny_with_custom_reason() -> None:
    """Map with reason key overrides the factory default."""
    evaluate = cel_policy(
        expression='{"result": "DENY", "reason": "Custom."}',
        reason="Factory default.",
    )
    result = evaluate({"type": "request"})
    assert result == {"result": "DENY", "reason": "Custom."}


# ── Map return: ASK ─────────────────────────────────────────────


def test_ask_verdict() -> None:
    """Expression returning ASK parks for user approval."""
    evaluate = cel_policy(
        expression=(
            'event.type == "tool_call"'
            ' ? {"result": "ASK", "reason": "Approve this?"}'
            ' : {"result": "ALLOW"}'
        ),
    )
    result = evaluate({"type": "tool_call", "data": {"name": "x"}})
    assert result == {"result": "ASK", "reason": "Approve this?"}


def test_ask_with_fallback_reason() -> None:
    """ASK without reason in map uses factory default."""
    evaluate = cel_policy(
        expression='{"result": "ASK"}',
        reason="Please approve.",
    )
    result = evaluate({"type": "request"})
    assert result == {"result": "ASK", "reason": "Please approve."}


# ── Map return: ALLOW ───────────────────────────────────────────


def test_allow_explicit() -> None:
    """Explicit ALLOW map passes through without reason."""
    evaluate = cel_policy(expression='{"result": "ALLOW"}')
    result = evaluate({"type": "request"})
    assert result == {"result": "ALLOW"}


@pytest.mark.parametrize("action", ["set", "increment", "delete", "append"])
@pytest.mark.parametrize(
    "key",
    ["_policy_test", "subagent_cost_ask_approved_usd", "subagent_cost_over_budget_approved_usd"],
)
def test_state_updates_reject_runtime_namespace(action: str, key: str) -> None:
    evaluate = cel_policy(
        expression=(
            '{"result": "ALLOW", "state_updates": ['
            '{"key": "counter", "action": "increment", "value": 1},'
            '{"key": event.data.key, "action": "' + action + '", "value": 1}]}'
        )
    )
    with pytest.raises(ValueError, match="reserved keys"):
        evaluate({"type": "tool_call", "data": {"key": key}})


def test_state_updates_pass_through_as_plain_python_values() -> None:
    """CEL maps may return canonical state_updates for conversation state."""
    evaluate = cel_policy(
        expression=(
            "{"
            '"result": "ALLOW",'
            '"state_updates": ['
            '{"key": "risk", "action": "increment", "value": 2},'
            '{"key": "last_tool", "action": "set", "value": event.data.name},'
            '{"key": "weight", "action": "set", "value": 1.5},'
            '{"key": "seen_tools", "action": "append", "value": ["shell", true, null]},'
            '{"key": "raw", "action": "set", "value": b"abc"}'
            "]"
            "}"
        )
    )

    result = evaluate({"type": "tool_call", "data": {"name": "sys_os_shell"}})

    assert result == {
        "result": "ALLOW",
        "state_updates": [
            {"key": "risk", "action": "increment", "value": 2},
            {"key": "last_tool", "action": "set", "value": "sys_os_shell"},
            {"key": "weight", "action": "set", "value": 1.5},
            {"key": "seen_tools", "action": "append", "value": ["shell", True, None]},
            {"key": "raw", "action": "set", "value": b"abc"},
        ],
    }


@pytest.mark.asyncio
async def test_state_updates_coerce_through_function_policy() -> None:
    """The policy engine sees CEL state_updates as typed state mutations."""
    spec = FunctionPolicySpec(
        name="cel_state",
        on=[PhaseSelector(phase=Phase.TOOL_CALL, tool_name=None)],
    )
    policy = FunctionPolicy(
        spec,
        cel_policy(
            expression=(
                "{"
                '"result": "ALLOW",'
                '"state_updates": ['
                '{"key": "call_count", "action": "increment", "value": 1},'
                '{"key": "last_decision", "action": "set", "value": "allowed"}'
                "]"
                "}"
            )
        ),
    )

    result = await policy.evaluate(
        EvaluationContext(
            phase=Phase.TOOL_CALL,
            tool_name="sys_os_shell",
            content={"name": "sys_os_shell"},
        ),
        {},
    )

    assert result.action is PolicyAction.ALLOW
    assert result.state_updates is not None
    assert [(u.key, u.action, u.value) for u in result.state_updates] == [
        ("call_count", StateUpdateAction.INCREMENT, 1),
        ("last_decision", StateUpdateAction.SET, "allowed"),
    ]


def test_state_updates_must_be_a_list() -> None:
    """Malformed CEL state_updates reports the authoring error."""
    evaluate = cel_policy(expression='{"result": "ALLOW", "state_updates": "bad"}')

    with pytest.raises(TypeError, match="state_updates must be a list"):
        evaluate({"type": "request"})


# ── Abstain (non-map returns) ───────────────────────────────────


def test_non_map_return_abstains() -> None:
    """Non-map return (e.g. bool, string) abstains."""
    evaluate = cel_policy(expression="true")
    assert evaluate({"type": "request"}) is None


def test_map_without_result_key_abstains() -> None:
    """Map missing the result key abstains."""
    evaluate = cel_policy(expression='{"reason": "no verdict"}')
    assert evaluate({"type": "request"}) is None


# ── CEL features ────────────────────────────────────────────────


def test_string_contains() -> None:
    """CEL string methods work."""
    evaluate = cel_policy(
        expression=(
            'event.type == "request" && event.data.contains("SECRET")'
            ' ? {"result": "DENY", "reason": "Secret detected."}'
            ' : {"result": "ALLOW"}'
        ),
    )
    assert evaluate({"type": "request", "data": "my SECRET key"}) == {
        "result": "DENY",
        "reason": "Secret detected.",
    }
    assert evaluate({"type": "request", "data": "normal"}) == {"result": "ALLOW"}


def test_request_dict_data_projected_to_user_text() -> None:
    """A request-phase ``data`` dict is projected to ``user_content`` for CEL.

    Regression for #2906: the web input gate now passes REQUEST ``data`` as
    ``{"user_content", "attachments"}``. String CEL expressions authored for the
    request phase (e.g. ``event.data.contains(...)``) must keep matching — a raw
    map would fail-open (``.contains`` raises → abstain → ALLOW), silently
    disabling a UI-configured DENY policy.
    """
    evaluate = cel_policy(
        expression=(
            'event.type == "request" && event.data.contains("SECRET")'
            ' ? {"result": "DENY", "reason": "Secret detected."}'
            ' : {"result": "ALLOW"}'
        ),
    )
    # Structured dict shape with the secret in user_content → still DENY.
    assert evaluate(
        {"type": "request", "data": {"user_content": "my SECRET key", "attachments": []}}
    ) == {"result": "DENY", "reason": "Secret detected."}
    # Clean structured dict → ALLOW (not a crash / abstain).
    assert evaluate(
        {"type": "request", "data": {"user_content": "normal", "attachments": []}}
    ) == {"result": "ALLOW"}


def test_in_list() -> None:
    """CEL ``in`` operator works."""
    evaluate = cel_policy(
        expression=(
            'event.type == "tool_call" && event.data.name in ["rm", "drop"]'
            ' ? {"result": "DENY", "reason": "Blocked."}'
            ' : {"result": "ALLOW"}'
        ),
    )
    assert evaluate({"type": "tool_call", "data": {"name": "drop"}}) == {
        "result": "DENY",
        "reason": "Blocked.",
    }
    assert evaluate({"type": "tool_call", "data": {"name": "read"}}) == {
        "result": "ALLOW",
    }


# ── Error handling ──────────────────────────────────────────────


def test_eval_error_returns_none() -> None:
    """CEL eval errors abstain (fail-open)."""
    evaluate = cel_policy(
        expression='event.nonexistent == "x" ? {"result": "DENY"} : {"result": "ALLOW"}'
    )
    assert evaluate({"type": "request", "data": "hello"}) is None


def test_invalid_syntax_raises() -> None:
    """Invalid CEL syntax is rejected at compile time."""
    with pytest.raises(ValueError, match="CEL"):
        cel_policy(expression="event.type ==== bad")


def test_llm_client_stripped_from_cel_event() -> None:
    """llm_client is dropped before json_to_cel; the expression still evaluates."""

    class _FakeLLMClient:
        pass

    evaluate = cel_policy(expression='{"result": "DENY"}')
    # The engine injects llm_client (a live object) into every real event.
    # CEL expressions cannot use it and json_to_cel cannot convert it, so it
    # is stripped before marshalling. The expression must evaluate normally.
    result = evaluate({"type": "request", "llm_client": _FakeLLMClient()})  # type: ignore[typeddict-unknown-key]
    assert result == {"result": "DENY", "reason": "Denied by policy."}


# ── Compile cache ───────────────────────────────────────────────


@pytest.fixture()
def cleared_cel_cache() -> Iterator[None]:
    """Run each cache test with an empty compiled-program cache."""
    cel_module._compiled_cel_program.cache_clear()
    yield
    cel_module._compiled_cel_program.cache_clear()


def test_repeated_builds_compile_expression_once(cleared_cel_cache: None) -> None:
    """The same expression compiles once; later builds hit the cache."""
    expression = (
        'event.type == "tool_call" && event.data.name == "sys_os_shell"'
        ' ? {"result": "DENY", "reason": "Shell blocked."}'
        ' : {"result": "ALLOW"}'
    )
    for reason in ("first", "second", "third", "fourth"):
        cel_policy(expression=expression, reason=reason)

    info = cel_module._compiled_cel_program.cache_info()
    assert info.misses == 1
    assert info.hits == 3
    assert info.currsize == 1
    assert info.maxsize == 512


def test_distinct_expressions_get_separate_programs(cleared_cel_cache: None) -> None:
    """Distinct expressions compile separately and keep their own verdicts."""
    deny_expression = (
        'event.type == "tool_call" ? {"result": "DENY", "reason": "Blocked."}'
        ' : {"result": "ALLOW"}'
    )
    allow_expression = '{"result": "ALLOW"}'

    deny_policy = cel_policy(expression=deny_expression)
    allow_policy = cel_policy(expression=allow_expression)

    info = cel_module._compiled_cel_program.cache_info()
    assert info.misses == 2
    assert info.hits == 0
    assert info.currsize == 2
    assert deny_policy({"type": "tool_call", "data": {"name": "read"}}) == {
        "result": "DENY",
        "reason": "Blocked.",
    }
    assert allow_policy({"type": "tool_call", "data": {"name": "read"}}) == {
        "result": "ALLOW",
    }

    cel_policy(expression=deny_expression)
    assert cel_module._compiled_cel_program.cache_info().hits == 1


def test_compile_errors_are_not_cached(cleared_cel_cache: None) -> None:
    """A malformed expression raises ValueError on every build attempt."""
    for _ in range(3):
        with pytest.raises(ValueError, match="compile error"):
            cel_policy(expression="event.type ==== bad")

    info = cel_module._compiled_cel_program.cache_info()
    assert info.misses == 3
    assert info.hits == 0
    assert info.currsize == 0


def test_missing_celpy_guard_runs_before_compilation(
    cleared_cel_cache: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without celpy the factory raises ImportError without touching the cache."""
    monkeypatch.setattr(cel_module, "celpy", None)

    with pytest.raises(ImportError, match="cel-python is required"):
        cel_policy(expression='{"result": "ALLOW"}')

    assert cel_module._compiled_cel_program.cache_info().misses == 0


# ── Short-circuit && and || ─────────────────────────────────────


class _Probe:
    """CEL function that counts its calls and returns a fixed boolean."""

    def __init__(self, result: bool = True) -> None:
        self.calls = 0
        self._result = celpy.celtypes.BoolType(result)

    def __call__(self, _value: object) -> celpy.celtypes.BoolType:
        self.calls += 1
        return self._result


def _run(
    runner_class: type[celpy.Runner],
    expression: str,
    activation: Context | None = None,
    functions: dict[str, CELFunction] | None = None,
) -> tuple[str, str]:
    """Evaluate ``expression``; return ``("value", repr)`` or ``("error", message)``."""
    environment = celpy.Environment(runner_class=runner_class)
    program = environment.program(environment.compile(expression), functions=functions)
    try:
        return "value", repr(program.evaluate(activation or {}))
    except CELEvalError as exc:
        return "error", f"{type(exc).__name__}: {exc}"


_FALSE = ("value", "BoolType(False)")
_TRUE = ("value", "BoolType(True)")


def test_stock_celpy_evaluates_both_operands() -> None:
    """Canary: celpy 0.5 evaluates both sides of ``&&``.

    When this fails, celpy short-circuits natively and ``ShortCircuitRunner`` can go.
    """
    probe = _Probe()

    outcome = _run(celpy.InterpretedRunner, "false && probe(1)", functions={"probe": probe})

    assert outcome == _FALSE
    assert probe.calls == 1


def test_and_skips_right_operand_when_left_is_false() -> None:
    """A string never reaches the ``exists`` guarded by ``type(v) == list``."""
    probe = _Probe()

    outcome = _run(
        ShortCircuitRunner,
        "type(v) == list && v.exists(c, probe(c))",
        {"v": json_to_cel("a" * 200)},
        {"probe": probe},
    )

    assert outcome == _FALSE
    assert probe.calls == 0


def test_or_skips_right_operand_when_left_is_true() -> None:
    """A string satisfies the first alternative, so the list branch is never walked."""
    probe = _Probe()

    outcome = _run(
        ShortCircuitRunner,
        "type(v) == string || v.exists(c, probe(c))",
        {"v": json_to_cel("a" * 200)},
        {"probe": probe},
    )

    assert outcome == _TRUE
    assert probe.calls == 0


@pytest.mark.parametrize(
    ("expression", "probe_result", "expected"),
    [
        ("(1 / 0 > 0) && probe(1)", False, _FALSE),
        ("(1 / 0 > 0) || probe(1)", True, _TRUE),
    ],
)
def test_erroring_left_operand_does_not_skip_the_right(
    expression: str, probe_result: bool, expected: tuple[str, str]
) -> None:
    """``error && false`` is ``false`` and ``error || true`` is ``true``."""
    probe = _Probe(result=probe_result)

    outcome = _run(ShortCircuitRunner, expression, functions={"probe": probe})

    assert outcome == expected
    assert probe.calls == 1


def test_macro_bodies_short_circuit_too() -> None:
    """Macro bodies run in nested evaluators, which must short-circuit as well."""
    probe = _Probe()

    outcome = _run(
        ShortCircuitRunner,
        "[1, 2, 3].exists(i, false && probe(i))",
        functions={"probe": probe},
    )

    assert outcome == _FALSE
    assert probe.calls == 0


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("true || [1].exists_one(i, 1 / 0 > 0)", _TRUE),
        ("false && [1].exists_one(i, 1 / 0 > 0)", _FALSE),
        ("true || [1].filter(i, 1 / 0 > 0).size() > 0", _TRUE),
        ("false && [1].map(i, 1 / 0).size() > 0", _FALSE),
    ],
)
def test_error_in_a_skipped_operand_does_not_escape(
    expression: str, expected: tuple[str, str]
) -> None:
    """``true || error`` is ``true`` and ``false && error`` is ``false``.

    celpy raises an error from ``exists_one``, ``filter`` and ``map`` bodies instead of
    returning it as a value, so stock celpy aborts even when the left operand decides.
    """
    assert _run(celpy.InterpretedRunner, expression)[0] == "error"

    assert _run(ShortCircuitRunner, expression) == expected


_EQUIVALENCE_ACTIVATION: Context = {
    "x": celpy.celtypes.BoolType(True),
    "n": celpy.celtypes.IntType(3),
}

_EQUIVALENCE_EXPRESSIONS = [
    # A decisive or erroring operand on either side.
    "false && (1 / 0 > 0)",
    "(1 / 0 > 0) && false",
    "true && (1 / 0 > 0)",
    "(1 / 0 > 0) && true",
    "true || (1 / 0 > 0)",
    "(1 / 0 > 0) || true",
    "false || (1 / 0 > 0)",
    "(1 / 0 > 0) || false",
    "(1 / 0 > 0) && (1 / 0 > 1)",
    # Operands that are not booleans.
    "false && 1",
    "1 && false",
    "true && 1",
    "1 && 1",
    "true || 1",
    "1 || false",
    "1 || 1",
    # Chains, mixed operators and macro bodies.
    "false && true && (1 / 0 > 0)",
    "x && n > 0 || false",
    "[1, 2, 3].exists(i, i / 0 > 1 || i == 2)",
    "[1, 2, 3].all(i, i > 0 || i / 0 > 1)",
    "[1, 2, 3].exists(i, i == 2 || [4, 5].exists(j, j > i && 1 / 0 > 0))",
]


@pytest.mark.parametrize("expression", _EQUIVALENCE_EXPRESSIONS)
def test_short_circuit_runner_matches_stock_runner(expression: str) -> None:
    """Skipping an operand never changes a value or an error."""
    expected = _run(celpy.InterpretedRunner, expression, _EQUIVALENCE_ACTIVATION)

    actual = _run(ShortCircuitRunner, expression, _EQUIVALENCE_ACTIVATION)

    assert actual == expected


def _random_expression(rng: random.Random, depth: int) -> str:
    """Build a random CEL expression mixing booleans, errors and wrong types."""
    atoms = ["true", "false", "x", "(n > 2)", "(1 / 0 > 0)", "(n / 0 == 1)", "1", '"s"']
    if depth == 0 or rng.random() < 0.2:
        return rng.choice(atoms)
    operator = rng.choice(["&&", "||", "!", "?:"])
    if operator == "!":
        return f"!({_random_expression(rng, depth - 1)})"
    if operator == "?:":
        condition, then, otherwise = (_random_expression(rng, depth - 1) for _ in range(3))
        return f"({condition} ? {then} : {otherwise})"
    left, right = (_random_expression(rng, depth - 1) for _ in range(2))
    return f"({left} {operator} {right})"


def test_random_expressions_match_stock_runner() -> None:
    """A seeded sweep of nested ``&&``, ``||``, ``!`` and ``?:`` agrees with the stock runner."""
    rng = random.Random(20261028)
    for _ in range(150):
        expression = _random_expression(rng, 4)

        expected = _run(celpy.InterpretedRunner, expression, _EQUIVALENCE_ACTIVATION)
        actual = _run(ShortCircuitRunner, expression, _EQUIVALENCE_ACTIVATION)

        assert actual == expected, expression


# Shaped like a shipped command guard: a string command is regex-matched, an argv list
# is searched element by element.
_COMMAND_GUARD = (
    'event.type == "tool_call" && type(event.data.arguments) == map'
    ' && ["command", "cmd"].exists(k, k in event.data.arguments && ('
    "(type(event.data.arguments[k]) == string"
    ' && event.data.arguments[k].matches(r"(^|\\s)git\\s+push(\\s|$)"))'
    " || (type(event.data.arguments[k]) == list"
    ' && event.data.arguments[k].exists(c, type(c) == string && c == "push"))))'
    ' ? {"result": "ASK"} : {}'
)


@pytest.mark.parametrize(
    ("command", "asks"),
    [
        ("git push origin main", True),
        ("cd repo && git push", True),
        (["git", "push"], True),
        ("git status", False),
        (["git", "status"], False),
        ("ls -la", False),
    ],
)
def test_command_guard_verdicts(command: str | list[str], asks: bool) -> None:
    """The guard asks for a push, as a string or an argv list, and abstains otherwise."""
    evaluate = cel_policy(expression=_COMMAND_GUARD)

    result = evaluate(
        {"type": "tool_call", "data": {"name": "Bash", "arguments": {"command": command}}}
    )

    assert result == ({"result": "ASK", "reason": "Denied by policy."} if asks else None)


def test_string_command_is_not_walked_as_a_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """A string command costs a few evaluations, not one macro iteration per character."""
    evaluations = 0
    stock_evaluate = Evaluator.evaluate

    def counting_evaluate(self: Evaluator, context: Context | None = None) -> celpy.celtypes.Value:
        nonlocal evaluations
        evaluations += 1
        return stock_evaluate(self, context)

    monkeypatch.setattr(Evaluator, "evaluate", counting_evaluate)
    evaluate = cel_policy(expression=_COMMAND_GUARD)

    result = evaluate(
        {
            "type": "tool_call",
            "data": {"name": "Bash", "arguments": {"command": "echo " + "x" * 5000}},
        }
    )

    assert result is None
    assert evaluations < 20
