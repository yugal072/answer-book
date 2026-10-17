from __future__ import annotations

import contextlib

try:
    import pytest
except ImportError:
    # Graceful fallback so tests can run in environments where pytest is not yet installed
    @contextlib.contextmanager
    def _raises(expected_exception):
        try:
            yield
        except expected_exception:
            pass
        else:
            raise AssertionError(
                f"Expected {expected_exception.__name__} was not raised."
            )

    class _MockPytest:
        raises = staticmethod(_raises)

    pytest = _MockPytest()  # type: ignore[assignment]

from app.tools.safe_math import safe_calculate


def test_calculates_arithmetic():
    assert safe_calculate("(12 + 8) / 4")["result"] == 5


def test_rejects_function_calls_and_names():
    with pytest.raises(ValueError):
        safe_calculate("__import__('os').system('echo bad')")
    with pytest.raises(ValueError):
        safe_calculate("sqrt(16)")


def test_rejects_huge_exponents():
    with pytest.raises(ValueError):
        safe_calculate("2 ** 101")
