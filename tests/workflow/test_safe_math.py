import pytest

from app.workflow.safe_math import UnsafeExpression, numbers_close, safe_eval


@pytest.mark.parametrize("expr,val", [
    ("(6*12)/(6+12)", 4.0), ("2**10", 1024.0), ("sqrt(16)+1", 5.0), ("3×4", 12.0), ("10÷4", 2.5),
    ("−5+2", -3.0), ("pi*2", 6.283185307179586), ("5²", 25.0), ("1,000+1", 1001.0), ("2^3", 8.0), ("-(3-5)", 2.0),
])
def test_valid(expr, val):
    assert safe_eval(expr) == pytest.approx(val)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo pwned')", "().__class__.__bases__", "open('/etc/passwd')", "x+1", "eval('1')",
    "[1,2][0]", "lambda: 1", "9**9**9", "1/0", "sqrt(-1)", "", "   ", "2 +", "a.b", "1; 2", "abs(1,2)", "'a'*3",
    "(1).real", "sqrt(1)(2)", "2**65", "1e999",
])
def test_rejected(expr):
    with pytest.raises(UnsafeExpression):
        safe_eval(expr)


def test_too_long():
    with pytest.raises(UnsafeExpression):
        safe_eval("1+" * 400 + "1")


def test_numbers_close():
    assert numbers_close(4.0, 4.0004) and not numbers_close(4.0, 4.1)
