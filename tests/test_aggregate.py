from __future__ import annotations

from app.generate.nodes.aggregate import aggregate_results


def test_aggregate_does_not_overwrite_parent_validation_failure():
    result = aggregate_results(
        {
            "status": "failed",
            "questions": [
                {
                    "work_item_id": "p:q:0000",
                    "question_index": 0,
                    "number": "1",
                    "text": "x",
                }
            ],
            "worker_results": [],
            "errors": ["Malformed input question."],
        }
    )
    assert result["status"] == "failed"
    assert result["results"][0]["status"] == "failed"
