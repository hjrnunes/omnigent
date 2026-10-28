"""A celpy runner whose ``&&`` and ``||`` skip an operand the left side decides.

celpy 0.5 evaluates both operands of ``&&`` and ``||`` before combining them. A guard
such as ``type(v) == list && v.exists(c, ...)`` therefore still walks ``v`` when it is
a string, running the macro body once per character. This runner evaluates the left
operand first and returns it when it is decisive (``false`` for ``&&``, ``true`` for
``||``). Every other case combines both operands with celpy's own functions.

Whenever celpy completes, this runner returns the same value or error. They differ
only where celpy aborts: it raises an error from an ``exists_one``, ``filter`` or
``map`` body even if the other operand already decides the result. CEL defines
``true || error`` as ``true`` and ``false && error`` as ``false``, so the runner
returns that value. ``cel_policy`` abstains on errors, so such a case used to allow
silently and now reaches the verdict the expression states.
"""

from __future__ import annotations

import celpy
import lark
from celpy.celtypes import BoolType, Value
from celpy.evaluation import CELEvalError, Context, Evaluator, Result


class ShortCircuitEvaluator(Evaluator):
    """Evaluator that skips the right operand of ``&&``/``||`` when the left decides."""

    def sub_evaluator(self, ast: lark.Tree) -> Evaluator:
        # Macro bodies (exists, all, map, filter) run in nested evaluators.
        return type(self)(ast, activation=self.activation)

    def conditionaland(self, tree: lark.Tree) -> Result:
        if len(tree.children) != 2:
            return super().conditionaland(tree)
        return self._combine(tree, "_&&_", decisive=False)

    def conditionalor(self, tree: lark.Tree) -> Result:
        if len(tree.children) != 2:
            return super().conditionalor(tree)
        return self._combine(tree, "_||_", decisive=True)

    def _combine(self, tree: lark.Tree, operator: str, *, decisive: bool) -> Result:
        left_tree, right_tree = tree.children
        left = self.visit(left_tree)
        if isinstance(left, BoolType) and bool(left) is decisive:
            return left
        right = self.visit(right_tree)
        try:
            return self.activation.resolve_function(operator)(left, right)
        except TypeError as ex:
            # Same error celpy builds when neither operand is a boolean.
            err = (
                f"found no matching overload for {operator} "
                f"applied to '({type(left)}, {type(right)})'"
            )
            value = CELEvalError(err, ex.__class__, ex.args, tree=tree)
            value.__cause__ = ex
            return value


class ShortCircuitRunner(celpy.InterpretedRunner):
    """``InterpretedRunner`` that evaluates with :class:`ShortCircuitEvaluator`."""

    def evaluate(self, context: Context) -> Value:
        evaluator = ShortCircuitEvaluator(ast=self.ast, activation=self.new_activation())
        return evaluator.evaluate(context)
