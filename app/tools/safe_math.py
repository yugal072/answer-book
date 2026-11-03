"""Restricted arithmetic evaluator for deterministic numeric checks; never uses eval()."""

from __future__ import annotations

import ast
import operator
from typing import Any, Callable

_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate(node: ast.AST) -> int | float:
    """Recursively evaluate safe AST arithmetic nodes."""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left = _evaluate(node.left)
        right = _evaluate(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent magnitude is limited to 100.")

        result = _BINARY[type(node.op)](left, right)
        if isinstance(result, complex) or abs(result) > 1e100:
            raise ValueError("Result is outside the supported numeric range.")
        return result

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))

    raise ValueError("Only numeric literals and basic arithmetic operators are allowed.")


def safe_calculate(expression: str) -> dict[str, Any]:
    """Safely calculate an arithmetic expression string without executing arbitrary code."""
    if (
        not isinstance(expression, str)
        or not expression.strip()
        or len(expression) > 200
    ):
        raise ValueError(
            "Expression must be a non-empty string of at most 200 characters."
        )

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Expression is not valid arithmetic syntax.") from exc

    result = _evaluate(tree)
    return {"expression": expression, "result": result}
