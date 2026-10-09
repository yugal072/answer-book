"""Deterministic evaluator for simple arithmetic expressions.

Model output is untrusted, so nothing here uses ``eval``/``exec``/``sympy``'s
parser. The expression is parsed to an AST and only whitelisted node types are
walked. Anything else raises :class:`UnsafeExpression`.
"""
from __future__ import annotations

import ast
import math
import operator
import re
from typing import Callable, Dict

MAX_LEN = 300
MAX_EXPONENT = 64
MAX_ABS = 1e100

_BIN: Dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UN: Dict[type, Callable] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: Dict[str, Callable] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
}
_CONSTS = {"pi": math.pi, "e": math.e}


class UnsafeExpression(ValueError):
    """The expression is not plain arithmetic (or is malformed)."""


def _normalize(expr: str) -> str:
    s = expr.strip()
    s = s.replace("×", "*").replace("÷", "/").replace("−", "-").replace("–", "-").replace("^", "**")
    s = s.replace("π", "pi").replace("√", "sqrt")
    s = s.replace("²", "**2").replace("³", "**3")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # 1,000 -> 1000
    return s


def _check(n: float) -> float:
    if isinstance(n, complex) or not math.isfinite(n) or abs(n) > MAX_ABS:
        raise UnsafeExpression("result is not a finite real number within range")
    return n


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return _check(node.value)
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _check(_UN[type(node.op)](_eval(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise UnsafeExpression("exponent too large")
        try:
            return _check(_BIN[type(node.op)](left, right))
        except (ZeroDivisionError, OverflowError, ValueError) as exc:
            raise UnsafeExpression(f"arithmetic error: {exc}") from exc
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
        and len(node.args) == 1
    ):
        try:
            return _check(_FUNCS[node.func.id](_eval(node.args[0])))
        except (ValueError, OverflowError) as exc:
            raise UnsafeExpression(f"arithmetic error: {exc}") from exc
    raise UnsafeExpression(f"disallowed syntax: {type(node).__name__}")


def safe_eval(expr: str) -> float:
    if not isinstance(expr, str) or not expr.strip():
        raise UnsafeExpression("empty expression")
    s = _normalize(expr)
    if len(s) > MAX_LEN:
        raise UnsafeExpression("expression too long")
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpression(f"not a valid arithmetic expression: {exc.msg}") from exc
    return float(_eval(tree))


def numbers_close(a: float, b: float, rel_tol: float = 1e-3, abs_tol: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
