from __future__ import annotations

from app.llm.client import clean_json_text


def test_clean_json_text_plain_json():
    assert clean_json_text('{"answer": "42"}') == '{"answer": "42"}'


def test_clean_json_text_code_fence():
    assert (
        clean_json_text('```json\n{"answer": "42"}\n```') == '{"answer": "42"}'
    )
