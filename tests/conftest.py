"""Shared pytest configuration.

The legacy ``answer-generation`` prototype tests construct ``GeminiProvider``,
which needs a real ``GEMINI_API_KEY`` even though their model reply is
"faked" afterwards. They therefore cannot run offline; they are marked
``live`` (not skipped silently, not edited) so ``-m "not live"`` excludes them
and ``-m live`` with a Gemini key still runs them. They exercise the
superseded prototype, not the workflow in ``app/workflow``.
"""
import os

import pytest

_NEEDS_GEMINI = {
    "tests/test_answer_generation.py::test_numerical_answer_structure",
    "tests/test_graph_nodes.py::test_generate_node",
    "tests/test_graph_nodes.py::test_solve_graph",
    "tests/test_graph_nodes.py::test_solve_graph_retries_when_verification_fails",
    "tests/test_graph_nodes.py::test_solve_graph_records_question_route",
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.nodeid in _NEEDS_GEMINI:
            item.add_marker(pytest.mark.live)
            if not os.getenv("GEMINI_API_KEY"):
                item.add_marker(pytest.mark.skip(reason="needs a real GEMINI_API_KEY (legacy Gemini prototype)"))
