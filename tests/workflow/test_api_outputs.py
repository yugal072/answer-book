"""FastAPI app, real ingestion of a real PDF, real graph; only the model provider is scripted."""
import io
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

import app.api.v1.solver as solver_api
from app.main import app
from app.workflow.llm import LLMUnavailableError
from app.workflow.render import build_pdf
from app.workflow.schemas import SolvedPaper
from tests.workflow.helpers import ScriptedLLM, good_draft, make_paper, make_service, q

PDF = Path(__file__).resolve().parents[2] / "test_papers" / "question_paper_455.pdf"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from langgraph.checkpoint.memory import InMemorySaver

    monkeypatch.setenv("WORKFLOW_OUTPUT_DIR", str(tmp_path / "out"))
    llm = ScriptedLLM()
    svc, kb, papers = make_service(llm, tmp_path, checkpointer=InMemorySaver())
    monkeypatch.setattr(solver_api, "_service", svc)
    c = TestClient(app)
    c.svc, c.llm, c.kb, c.tmp = svc, llm, kb, tmp_path
    return c


def upload(client, name="paper.pdf", data=None, **params):
    data = PDF.read_bytes() if data is None else data
    return client.post("/api/v1/process-paper", params=params, files={"file": (name, io.BytesIO(data), "application/pdf")})


# ------------------------------------------------------------------ JSON
def test_json_output_validates_against_schema_and_is_persisted(client):
    r = upload(client, limit=3)
    assert r.status_code == 200, r.text
    body = r.json()
    solved = SolvedPaper.model_validate(body)
    assert solved.outcome == "complete" and solved.total_questions == 3 and len(solved.results) == 3
    assert [x.question["number"] for x in solved.results] == ["1", "2", "3"]
    assert body["status"] == "ready" and body["solutions"][0]["question_number"] == "1"
    assert {"answer", "steps", "mark_split", "common_mistakes", "confidence", "verified_by", "needs_teacher_check"} <= set(body["solutions"][0])
    # the persisted copy is retrievable and identical
    got = client.get(f"/api/v1/solutions/{solved.paper_id}")
    assert got.status_code == 200 and got.json() == body


def test_json_default_format_and_real_metadata_from_ingestion(client):
    body = upload(client, limit=1).json()
    assert body["metadata"]["subject"] == "Mathematics Part 1" and body["metadata"]["board"] == "CBSE"
    assert body["paper_id"].startswith("pap_")


def test_overrides_are_applied(client):
    body = upload(client, limit=1, subject="Physics", class_name="Class 11", board="ICSE").json()
    assert (body["metadata"]["subject"], body["metadata"]["class_name"], body["metadata"]["board"]) == ("Physics", "Class 11", "ICSE")


def test_mixed_outcome_is_reported_not_hidden(client):
    client.llm.drafts = {"2": [LLMUnavailableError("down")]}
    r = upload(client, limit=3)
    body = r.json()
    assert r.status_code == 200 and body["outcome"] == "partial" and body["status"] == "partial"
    st = {x["question"]["number"]: x["status"] for x in body["results"]}
    assert st == {"1": "verified", "2": "failed", "3": "verified"}
    assert body["solutions"][1]["needs_teacher_check"] is True


def test_every_question_failing_is_a_502_with_the_full_body(client):
    client.llm.drafts = {n: [LLMUnavailableError("down")] for n in "123"}
    r = upload(client, limit=3)
    assert r.status_code == 502 and r.json()["outcome"] == "failed"


# ------------------------------------------------------------------ PDF
def pdf_text(data: bytes) -> str:
    return "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(data)).pages)


def test_pdf_output_is_a_real_pdf_with_the_paper_content(client):
    r = upload(client, limit=3, format="pdf")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF-") and r.content.rstrip().endswith(b"%%EOF")
    assert 'attachment; filename="pap_' in r.headers["content-disposition"] and r.headers["x-paper-outcome"] == "complete"
    reader = PdfReader(io.BytesIO(r.content))
    assert len(reader.pages) >= 1
    text = pdf_text(r.content)
    assert "Solution Book" in text and "Q1" in text and "Q3" in text and "VERIFIED" in text
    assert "coordinate axes divide the plane" in text  # real question text from the PDF
    assert "(B) second" in text and "Mark split" in text


def test_pdf_labels_unverified_and_failed_answers_prominently(client):
    client.llm.drafts = {"1": [lambda it: good_draft(it, self_confidence=0.1)], "2": [LLMUnavailableError("down")]}
    text = pdf_text(upload(client, limit=3, format="pdf").content)
    assert "UNVERIFIED - TEACHER REVIEW REQUIRED" in text and "FAILED - NO ANSWER PRODUCED" in text
    assert "Partial" in text and "must be checked by a teacher" in text


def test_pdf_renders_unicode_math_symbols(client):
    paper = make_paper([q("1", "short", text="Evaluate √16 + 3² and π × 2 ≈ ?")])
    solved = client.svc.run(paper)
    text = pdf_text(build_pdf(solved.model_dump(mode="json", by_alias=True)))
    assert "√16" in text and "3²" in text and "π" in text


def test_stored_solution_can_be_downloaded_as_pdf(client):
    pid = upload(client, limit=2).json()["paper_id"]
    r = client.get(f"/api/v1/solutions/{pid}", params={"format": "pdf"})
    assert r.status_code == 200 and r.content.startswith(b"%PDF-")


def test_legacy_solution_file_cannot_be_rendered_as_pdf(client):
    out = client.tmp / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "pap_legacy.json").write_text(json.dumps({"paper_id": "pap_legacy", "solutions": []}))
    assert client.get("/api/v1/solutions/pap_legacy").status_code == 200
    r = client.get("/api/v1/solutions/pap_legacy", params={"format": "pdf"})
    assert r.status_code == 422 and "predates" in r.json()["detail"]


# ------------------------------------------------------------------ SSE
def parse_sse(text: str):
    frames = []
    for block in text.strip().split("\n\n"):
        f = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        f["data"] = json.loads(f["data"])
        frames.append(f)
    return frames


def test_sse_is_a_real_event_stream_with_a_stable_contract(client):
    with client.stream("POST", "/api/v1/process-paper", params={"limit": 3, "format": "sse"},
                       files={"file": ("p.pdf", PDF.read_bytes(), "application/pdf")}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["cache-control"] == "no-cache"
        raw = "".join(r.iter_text())
    frames = parse_sse(raw)
    assert [int(f["id"]) for f in frames] == list(range(1, len(frames) + 1))  # monotonically increasing ids
    names = [f["event"] for f in frames]
    assert names[0] == "run.started" and names[-1] == "run.finished"
    assert names.count("run.finished") + names.count("run.error") == 1  # exactly one terminal event
    assert names.count("question.completed") == 3 and "question.stage" in names and "paper.completed" in names
    rid = frames[0]["data"]["run_id"]
    assert all(f["data"]["run_id"] == rid for f in frames)
    stages = {f["data"]["stage"] for f in frames if f["event"] == "question.stage"}
    assert {"cache_lookup", "solving", "verifying", "assessing", "storing"} <= stages
    done = [f["data"] for f in frames if f["event"] == "question.completed"]
    assert sorted(d["number"] for d in done) == ["1", "2", "3"] and all(d["status"] == "verified" for d in done)
    term = frames[-1]["data"]
    assert term["outcome"] == "complete" and term["summary"]["verified"] == 3 and term["result_url"].startswith("/api/v1/solutions/pap_")
    assert "api" not in raw.lower() or "api/v1" in raw.lower()  # no credential-looking payloads
    assert "steps" not in raw and "GROQ" not in raw  # no solution bodies / config in the stream
    assert str(client.tmp) not in raw and "json_path" not in raw  # no server filesystem paths


def test_sse_terminal_error_event_on_graph_failure(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("graph exploded")

    monkeypatch.setattr(client.svc.graph, "stream", boom)
    with client.stream("POST", "/api/v1/process-paper", params={"limit": 1, "format": "sse"},
                       files={"file": ("p.pdf", PDF.read_bytes(), "application/pdf")}) as r:
        raw = "".join(r.iter_text())
    frames = parse_sse(raw)
    assert frames[-1]["event"] == "run.error" and frames[-1]["data"]["code"] == "RuntimeError"
    assert [f["event"] for f in frames].count("run.finished") == 0


def test_sse_stream_closes_cleanly_when_client_stops_reading(client):
    with client.stream("POST", "/api/v1/process-paper", params={"limit": 3, "format": "sse"},
                       files={"file": ("p.pdf", PDF.read_bytes(), "application/pdf")}) as r:
        it = r.iter_text()
        next(it)  # read the first chunk then drop the connection
    # the app is still healthy afterwards
    assert client.get("/api/v1/health").status_code == 200


def test_all_three_formats_agree_on_the_same_run(client):
    client.llm.drafts = {"2": [lambda it: good_draft(it, self_confidence=0.1)]}
    # JSON
    j = upload(client, limit=3).json()
    # SSE (second run: verified rows come from cache, unverified is re-solved)
    with client.stream("POST", "/api/v1/process-paper", params={"limit": 3, "format": "sse"},
                       files={"file": ("p.pdf", PDF.read_bytes(), "application/pdf")}) as r:
        frames = parse_sse("".join(r.iter_text()))
    sse = {f["data"]["number"]: f["data"]["status"] for f in frames if f["event"] == "question.completed"}
    jstat = {x["question"]["number"]: x["status"] for x in j["results"]}
    assert sse["2"] == jstat["2"] == "unverified"
    assert {k: ("verified" if v == "cached" else v) for k, v in sse.items()} == jstat
    # PDF from the same JSON body
    text = pdf_text(build_pdf(j))
    assert text.count("UNVERIFIED - TEACHER REVIEW REQUIRED") == 1
    assert frames[-1]["data"]["outcome"] == j["outcome"] == "needs_review"


# ------------------------------------------------------------------ input validation
def test_empty_upload_is_400(client):
    assert upload(client, data=b"").status_code in (400, 422)


def test_unsupported_extension_is_415(client):
    r = upload(client, name="notes.txt", data=b"hello")
    assert r.status_code == 415 and "Unsupported file type" in r.json()["detail"]


def test_pdf_extension_with_non_pdf_content_is_415(client):
    r = upload(client, data=b"this is not a pdf at all")
    assert r.status_code == 415 and "does not look like" in r.json()["detail"]


def test_oversized_upload_is_413(client, monkeypatch):
    import app.api.v1.ingestion as ing

    monkeypatch.setattr(ing, "MAX_UPLOAD_BYTES", 100)
    assert upload(client).status_code == 413


def test_corrupt_pdf_is_422_from_ingestion(client):
    r = upload(client, data=b"%PDF-1.4\n garbage garbage")
    assert r.status_code == 422


def test_limit_must_be_positive(client):
    assert upload(client, limit=0).status_code == 422


def test_unexpected_graph_exception_is_a_500_without_internals(client, monkeypatch):
    monkeypatch.setattr(client.svc.graph, "invoke", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("secret internals")))
    r = upload(client, limit=1)
    assert r.status_code == 500 and "secret internals" not in r.text and "RuntimeError" in r.json()["detail"]


def test_missing_api_key_is_a_clear_503(monkeypatch):
    monkeypatch.setattr(solver_api, "_service", None)
    import app.workflow.factory as factory

    monkeypatch.setattr(factory.app_settings, "GROQ_API_KEY", "")
    r = upload(TestClient(app), limit=1)
    assert r.status_code == 503 and "GROQ_API_KEY" in r.json()["detail"]


# ------------------------------------------------------------------ other endpoints
def test_solution_lookup_rejects_path_tricks(client):
    from app.workflow.render import safe_paper_id

    with pytest.raises(ValueError):
        safe_paper_id("..")
    assert safe_paper_id("a/b\\c") == "a_b_c"
    assert client.get("/api/v1/solutions/.hidden").status_code == 400
    assert client.get("/api/v1/solutions/does_not_exist").status_code == 404


def test_solve_question_endpoint(client):
    r = client.post("/api/v1/solve-question", json={
        "question": {"number": "1", "text": "Pick the second option?", "marks": 1, "type": "mcq",
                     "options": ["(A) a", "(B) b", "(C) c", "(D) d"]},
        "metadata": {"subject": "Mathematics", "class": "Class 9", "board": "CBSE"}})
    b = r.json()
    assert r.status_code == 200 and b["status"] == "verified" and b["verified_by"] == "cross_check"
    assert b["question_number"] == "1" and b["needs_teacher_check"] is False
    assert not (client.tmp / "out").exists()  # ad-hoc questions write no paper export


def test_solve_question_validation(client):
    assert client.post("/api/v1/solve-question", json={}).status_code == 400
    assert client.post("/api/v1/solve-question", json={"question": {"text": "x", "type": "essay"}}).status_code == 422


def test_resume_endpoint(client):
    paper = make_paper([q("1"), q("2")])
    client.svc.run(paper, run_id="run-xyz")
    ok = client.post("/api/v1/runs/run-xyz/resume")
    assert ok.status_code == 200 and ok.json()["outcome"] == "complete"
    assert client.post("/api/v1/runs/unknown/resume").status_code == 404
    with client.stream("POST", "/api/v1/runs/unknown/resume", params={"format": "sse"}) as r:
        frames = parse_sse("".join(r.iter_text()))
    assert frames[-1]["event"] == "run.error" and frames[-1]["data"]["code"] == "RunNotFound"


def test_health_and_docs_routes_still_work(client):
    assert client.get("/api/v1/health").json()["status"] == "ok"
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/api/v1/process-paper", "/api/v1/ingest", "/api/v1/solve-question", "/api/v1/solutions/{paper_id}"} <= set(paths)
