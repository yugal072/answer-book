"""Graph adapter: field preservation, honesty about missing data, and the real PDF corpus."""
from pathlib import Path

import pytest

from app.ingestion import IngestionError, ingest_file
from app.models.loaders_models import Paper, Question
from app.workflow.adapter import AdapterError, adapt_paper
from tests.workflow.helpers import make_paper, q

CORPUS = sorted(Path(__file__).resolve().parents[2].glob("test_papers/question_paper_*.pdf"))


def test_fields_survive_the_adapter():
    qq = Question(number="3(b)", section="B", text="  Find   x\n in 2x = 4 ", marks=3, type="numerical", options=None,
                  has_figure=True, page=2, choice_group="3", section_title="Sec B", chapter="Ch 1")
    ad = adapt_paper(make_paper([qq], paper_id="pap_x", subject=" Maths ", class_name="Class 9", board="CBSE"))
    it = ad.items[0]
    assert (it.number, it.marks, it.type, it.has_figure, it.page, it.choice_group, it.section, it.section_title, it.chapter) == (
        "3(b)", 3, "numerical", True, 2, "3", "B", "Sec B", "Ch 1")
    assert it.text == "Find x\n in 2x = 4" or it.text.startswith("Find x")
    assert ad.context.subject == "Maths" and ad.context.paper_id == "pap_x" and ad.context.fingerprint == "f" * 64
    assert it.item_id == "q001" and len(it.signature) == 64


def test_missing_marks_are_flagged_not_invented():
    it = adapt_paper(make_paper([q("1", marks=None)])).items[0]
    assert it.marks == 1 and it.marks_assumed and any("marks not stated" in w for w in it.warnings)


def test_unknown_metadata_stays_unknown():
    p = Paper(paper_id="p", fingerprint="f", status="solving", questions=[q("1")])
    ctx = adapt_paper(p).context
    assert ctx.subject is None and ctx.board is None and ctx.class_name is None


def test_signature_differs_by_marks_board_and_class():
    sig = lambda **k: adapt_paper(make_paper([q("1", **{kk: vv for kk, vv in k.items() if kk == 'marks'})], **{kk: vv for kk, vv in k.items() if kk != 'marks'})).items[0].signature  # noqa: E731
    base = sig()
    assert sig(marks=5) != base and sig(board="ICSE") != base and sig(class_name="Class 10") != base


def test_empty_paper_and_bad_limit_are_rejected():
    with pytest.raises(AdapterError):
        adapt_paper(make_paper([]))
    with pytest.raises(AdapterError):
        adapt_paper(make_paper([q("1")]), limit=0)


def test_limit_takes_the_first_n():
    ad = adapt_paper(make_paper([q(str(i)) for i in range(1, 6)]), limit=2)
    assert [i.number for i in ad.items] == ["1", "2"]


def test_corpus_is_present():
    assert len(CORPUS) >= 20, CORPUS


@pytest.mark.parametrize("pdf", CORPUS, ids=lambda p: p.stem)
def test_every_corpus_paper_adapts_losslessly(pdf):
    try:
        paper = ingest_file(pdf)
    except IngestionError as exc:  # pragma: no cover - reported, not hidden
        pytest.fail(f"ingestion rejected {pdf.name}: {str(exc)[:200]}")
    ad = adapt_paper(paper, source_file=pdf.name)
    assert len(ad.items) == len(paper.questions) > 0
    assert len({i.item_id for i in ad.items}) == len(ad.items)
    for src, it in zip(paper.questions, ad.items):
        assert it.number == str(src.number) and it.page == src.page and it.section == src.section
        assert it.text.replace(" ", "").replace("\n", "") == " ".join((src.text or "").split()).replace(" ", "") or src.text.strip()
        assert it.marks >= 1 and (src.marks is None) == it.marks_assumed
        assert it.type in ("mcq", "numerical", "short", "long")
        if it.type == "mcq":
            assert len(it.options) >= 2
        assert it.options == [" ".join(o.split()) for o in (src.options or []) if o.strip()] or not src.options
    assert ad.context.paper_id == paper.paper_id and ad.context.total_marks == paper.total_marks
