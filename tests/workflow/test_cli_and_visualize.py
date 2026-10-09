import json
import re
from pathlib import Path

import app.workflow.cli as cli
import app.workflow.factory as factory
from app.workflow.visualize import mermaid
from tests.workflow.helpers import ScriptedLLM, make_service

PDF = Path(__file__).resolve().parents[2] / "test_papers" / "question_paper_455.pdf"


def test_graph_visualisation_contains_every_diagram_stage():
    m = mermaid()
    for node in ("validate_prepare", "question_pipeline", "aggregate_results", "build_solved_paper", "assemble_output",
                 "cache_lookup", "retrieve_stored", "solve_mcq", "solve_numerical", "solve_short", "solve_long",
                 "verify", "repair", "assess_confidence", "store_verified", "finalize_question"):
        assert re.search(rf"\b{node}\b", m), node
    assert m.count("```mermaid") == 2
    assert "repair" in m and "-.->" in m  # conditional edges are rendered


def test_cli_json_and_pdf_end_to_end(tmp_path, monkeypatch, capsys):
    svc, *_ = make_service(ScriptedLLM(), tmp_path)
    monkeypatch.setattr(factory, "build_default_service", lambda settings=None: svc)
    assert cli.main([str(PDF), "--limit", "2"]) == 0
    out = capsys.readouterr().out
    assert "=> VERIFIED" in out and "JSON written" in out
    data = json.loads(next((tmp_path / "out").glob("pap_*.json")).read_text())
    assert data["outcome"] == "complete" and data["total_questions"] == 2
    pdf = tmp_path / "book.pdf"
    assert cli.main([str(PDF), "--limit", "2", "--format", "pdf", "--out", str(pdf), "--quiet"]) == 0
    assert pdf.read_bytes().startswith(b"%PDF-")


def test_cli_exit_code_reflects_outcome(tmp_path, monkeypatch):
    from app.workflow.llm import LLMUnavailableError

    svc, *_ = make_service(ScriptedLLM(drafts={"2": [LLMUnavailableError("down")]}), tmp_path)
    monkeypatch.setattr(factory, "build_default_service", lambda settings=None: svc)
    assert cli.main([str(PDF), "--limit", "3", "--quiet"]) == 2  # partial
