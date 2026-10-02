"""Unit tests for the structure-aware text parser.

These use small hand-written page fixtures (not the real corpus, which is
covered by tests/test_paper_loader.py) to pin the *layout rules* one at a
time: wrapped text, page boundaries, section headings, the three marks
layouts, MCQ option shapes, internal choice, sub-parts, numbers that are
not question numbers, and non-sequential numbering.

Every assertion here checks a concrete value; none of them asserts only
"something came back".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.text_parser import (  # noqa: E402
    parse_pages,
    strip_running_heads,
)


def qs(pages):
    """Parse and return the questions."""
    return parse_pages(pages).questions


# --------------------------------------------------------------------------
# Question boundaries
# --------------------------------------------------------------------------

def test_simple_sequential_questions():
    questions = qs([
        ["Header line", "1 What is 2 + 2?", "1", "2 What is 3 + 3?", "1"],
    ])
    assert [q.number for q in questions] == ["1", "2"]
    assert questions[0].text == "What is 2 + 2?"
    assert questions[1].text == "What is 3 + 3?"
    assert [q.marks for q in questions] == [1, 1]


def test_wrapped_question_text_is_rejoined():
    questions = qs([
        ["1 A chord of length 24 cm is drawn in a circle of diameter",
         "26 cm. Find the distance from the centre of the circle to the",
         "chord.", "2"],
    ])
    assert len(questions) == 1
    assert questions[0].text == (
        "A chord of length 24 cm is drawn in a circle of diameter "
        "26 cm. Find the distance from the centre of the circle to the chord."
    )


def test_question_spanning_a_page_boundary_keeps_its_text():
    questions = qs([
        ["1 The first term of an AP is 11 and the common difference"],
        ["is -3. Find the 10th term of this AP.", "3", "2 Next question?"],
    ])
    assert [q.number for q in questions] == ["1", "2"]
    assert questions[0].text == (
        "The first term of an AP is 11 and the common difference is -3. "
        "Find the 10th term of this AP."
    )
    assert questions[0].page == 1
    assert questions[1].page == 2


def test_numbers_inside_question_text_are_not_question_numbers():
    questions = qs([
        ["1 A line of length 110 degrees intersects a circle at 3 points",
         "and the second chord of 45 cm follows.", "2"],
        ["2 What is the value of x if x = 12?", "1"],
    ])
    assert [q.number for q in questions] == ["1", "2"]
    assert "110 degrees" in questions[0].text
    assert questions[1].text == "What is the value of x if x = 12?"


def test_a_gap_in_numbering_does_not_destroy_questions():
    """Strict sequential matching stops at a break; the relaxed strategy
    recovers the rest instead of silently merging them into the previous
    question."""
    result = parse_pages([
        ["1 First question?", "1", "5 Fifth question after a jump?", "2"],
    ])
    assert [q.number for q in result.questions] == ["1", "5"]
    assert result.strategy == "relaxed"
    assert result.questions[1].text == "Fifth question after a jump?"


def test_numbering_starting_above_one_is_accepted():
    result = parse_pages([["7 What is this?", "1", "8 And this?", "1"]])
    assert [q.number for q in result.questions] == ["7", "8"]


def test_dotted_numbering_style():
    questions = qs([["1. What is 2 + 2?", "1", "2. What is 3 + 3?", "1"]])
    assert [q.number for q in questions] == ["1", "2"]
    assert questions[0].text == "What is 2 + 2?"


# --------------------------------------------------------------------------
# Preamble, furniture and instructions
# --------------------------------------------------------------------------

def test_general_instructions_are_not_questions():
    questions = qs([
        [
            "General Instructions:",
            "1. Read all questions carefully before answering.",
            "2. All questions are compulsory unless stated otherwise.",
            "Section A",
            "1 What is 2 + 2?",
            "1",
            "2 What is 3 + 3?",
            "1",
        ],
    ])
    assert [q.number for q in questions] == ["1", "2"]
    assert questions[0].text == "What is 2 + 2?"


def test_a_wrapped_line_ending_in_instructions_is_not_a_heading():
    """Regression: "...suitcases and last instructions." is question wording.
    Treating it as a heading silently truncated the question."""
    questions = qs([
        [
            "1 Mira wondered how so many years had folded into a single",
            "afternoon of suitcases and last",
            "instructions.",
            "The house was quiet.",
            "2 What is 3 + 3?",
            "1",
        ],
    ])
    assert len(questions) == 2
    assert questions[0].text.endswith("last instructions. The house was quiet.")


def test_repeated_headers_and_footers_are_removed():
    pages = [
        ["site.com", "Q.NO. QUESTIONS MARKS", "1 What is 2 + 2?", "1",
         "Page 1/2 site.com"],
        ["site.com", "Q.NO. QUESTIONS MARKS", "2 What is 3 + 3?", "1",
         "Page 2/2 site.com"],
    ]
    questions = qs(pages)
    assert [q.text for q in questions] == ["What is 2 + 2?", "What is 3 + 3?"]
    cleaned = strip_running_heads(pages)
    assert not any("site.com" in line for page in cleaned for line in page)


def test_a_repeated_line_inside_the_body_is_not_removed():
    """Repetition alone is not enough to call a line furniture: a line that
    repeats in the middle of the page is content."""
    pages = [
        ["1 What is 2 + 2?", "Define x carefully.", "1"],
        ["2 What is 3 + 3?", "Define x carefully.", "1"],
    ]
    questions = qs(pages)
    assert questions[0].text.endswith("Define x carefully.")
    assert questions[1].text.endswith("Define x carefully.")


def test_preamble_is_not_question_text():
    questions = qs([
        [
            "English - Grammar",
            "Subject: English | Class: Class 9",
            "Total Marks: 20 | Time Allowed: 0h 25m | CBSE",
            "1 What is 2 + 2?",
            "1",
        ],
    ])
    assert len(questions) == 1
    assert questions[0].text == "What is 2 + 2?"


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------

def test_section_heading_with_and_without_a_title():
    questions = qs([
        [
            "Section A",
            "Attempt all 2 questions. Each carries 1 mark(s).",
            "1 What is 2 + 2?",
            "1",
            "Section B - Literature (Very Short Answer)",
            "2 Why is the sky blue?",
            "1",
        ],
    ])
    assert [q.section for q in questions] == ["A", "B"]
    assert questions[0].section_title is None
    assert questions[1].section_title == "Literature (Very Short Answer)"


def test_section_is_preserved_across_a_page_break():
    questions = qs([
        ["Section B", "1 First in B?", "1"],
        ["2 Second in B?", "1"],
    ])
    assert [q.section for q in questions] == ["B", "B"]


def test_section_keyword_variants():
    questions = qs([
        ["SECTION C: Grammar", "1 What is 2 + 2?", "1"],
        ["Part II", "2 What is 3 + 3?", "1"],
    ])
    assert [q.section for q in questions] == ["C", "II"]


def test_prose_starting_with_part_is_not_a_section():
    questions = qs([
        ["1 Part of the whole equals the sum of its parts. Explain.", "1",
         "2 Part of the reason?", "1"],
    ])
    assert [q.section for q in questions] == [None, None]
    assert questions[0].text.startswith("Part of the whole")


# --------------------------------------------------------------------------
# Marks
# --------------------------------------------------------------------------

def test_marks_on_a_standalone_line():
    questions = qs([["1 What is 2 + 2?", "4"]])
    assert questions[0].marks == 4
    assert questions[0].text == "What is 2 + 2?"


def test_marks_after_the_stem():
    questions = qs([["1 How did Leela help resolve the conflict? 1"]])
    assert questions[0].marks == 1
    assert questions[0].text == "How did Leela help resolve the conflict?"


def test_marks_after_the_last_line():
    questions = qs([
        ["1 Symbolism - each vocation means more than a job.", "What does",
         "it symbolise?", "2"],
    ])
    assert questions[0].marks == 2
    assert questions[0].text.endswith("What does it symbolise?")


def test_section_instruction_is_only_a_fallback():
    questions = qs([
        [
            "Section C",
            "Attempt all 3 questions. Each carries 2 mark(s).",
            "1 First question here?",
            "3",
            "2 Second question here?",
        ],
    ])
    assert [q.marks for q in questions] == [3, 2]


def test_a_question_ending_in_a_number_is_not_mistaken_for_marks():
    questions = qs([["1 A train travels 5.", "2 What comes next?"]])
    assert questions[0].marks is None
    assert questions[0].text == "A train travels 5."


def test_unknown_marks_stay_unknown():
    questions = qs([["1 What is 2 + 2?"]])
    assert questions[0].marks is None


def test_marks_line_is_removed_only_once():
    questions = qs([
        ["1 A table of values follows.", "2", "2 What comes next?"],
    ])
    assert questions[0].marks == 2
    assert questions[0].text == "A table of values follows."


# --------------------------------------------------------------------------
# MCQ options
# --------------------------------------------------------------------------

def test_options_one_per_line():
    questions = qs([
        [
            "1 How many parts do the axes divide the plane into?",
            "(A) 2", "(B) 6", "(C) 4", "(D) 8", "1",
        ],
    ])
    assert questions[0].type == "mcq"
    assert questions[0].options == ["(A) 2", "(B) 6", "(C) 4", "(D) 8"]
    assert questions[0].text == "How many parts do the axes divide the plane into?"


def test_options_on_one_line_keep_order_and_labels():
    questions = qs([
        [
            "1 Pick the correct identity.",
            "(A) (ax + b)(cx + d) = acx2 + (ad + bc)x + bd",
            "(B) (ax + b)(cx + d) = acx2 + (ab + cd)x + bd  (C) wrong one",
            "1",
        ],
    ])
    assert questions[0].options == [
        "(A) (ax + b)(cx + d) = acx2 + (ad + bc)x + bd",
        "(B) (ax + b)(cx + d) = acx2 + (ab + cd)x + bd",
        "(C) wrong one",
    ]


def test_wrapped_option_text_is_kept():
    questions = qs([
        [
            "1 Which pair is correct?",
            "(A) the first option that runs",
            "over two lines",
            "(B) second",
            "(C) third",
            "(D) fourth",
            "1",
        ],
    ])
    assert questions[0].options[0] == "(A) the first option that runs over two lines"
    assert questions[0].options[1:] == ["(B) second", "(C) third", "(D) fourth"]


def test_sub_question_options_stay_in_the_text():
    """A passage question with options under sub-part (i) must not present
    the sub-question's options as its own."""
    questions = qs([
        [
            "1 Read the passage and answer the questions that follow.",
            "The house was quiet.",
            "(i) What does the silence feel like? [1 mark(s)]",
            "(A) Almost solid", "(B) A calm", "(C) Relief", "(D) Noise",
            "10",
        ],
    ])
    assert questions[0].options is None
    assert questions[0].type != "mcq"
    assert "(A) Almost solid" in questions[0].text
    assert questions[0].marks == 10


def test_option_values_are_not_read_as_question_numbers():
    questions = qs([
        ["1 Pick one.", "(A) 2", "(B) 6", "(C) 4", "(D) 8", "1"],
    ])
    assert [q.number for q in questions] == ["1"]
    assert questions[0].marks == 1


# --------------------------------------------------------------------------
# Internal choice, sub-parts, chapters, figures
# --------------------------------------------------------------------------

def test_or_alternative_sets_choice_group():
    questions = qs([
        [
            "1 The first term of an AP is 11. Find the 10th term.",
            "3",
            "OR",
            "In a circle, a chord is 5 cm from the centre. How long?",
        ],
    ])
    assert len(questions) == 1
    assert questions[0].choice_group == "1"
    assert "OR" in questions[0].text
    assert questions[0].marks == 3


def test_lowercase_or_in_prose_does_not_set_choice_group():
    questions = qs([["1 Is the sky blue or grey?", "1"]])
    assert questions[0].choice_group is None


def test_subparts_stay_inside_their_parent_question():
    questions = qs([
        [
            "18 Case Study: Book Club Subscriptions",
            "A book club charges a joining fee of 100 rupees.",
            "(i) What is the degree of the polynomial?",
            "(ii) Identify the constant term.",
            "5",
        ],
    ])
    assert len(questions) == 1
    assert "(i) What is the degree of the polynomial?" in questions[0].text
    assert questions[0].type == "long"


def test_chapter_tag_is_captured_not_mixed_into_the_text():
    questions = qs([
        ["1 Why is pi irrational?", "2", "[Ch 6: Measuring Space]"],
    ])
    assert questions[0].chapter == "[Ch 6: Measuring Space]"
    assert "Ch 6" not in questions[0].text


def test_every_chapter_tag_of_an_or_question_is_kept():
    questions = qs([
        [
            "1 Why is pi irrational?", "2", "[Ch 6: Numbers]",
            "OR",
            "Describe the construction of sqrt 2.", "[Ch 3: Geometry]",
        ],
    ])
    assert len(questions) == 1
    assert "Ch 6" in questions[0].chapter and "Ch 3" in questions[0].chapter


def test_figure_flag_requires_a_real_reference():
    with_figure = qs([["1 Study Figure 3 and answer.", "2"]])
    without = qs([["1 The figure of speech here is irony.", "2"]])
    assert with_figure[0].has_figure is True
    assert without[0].has_figure is False


def test_section_instruction_line_never_becomes_question_text():
    questions = qs([
        [
            "Section C",
            "Attempt 3 out of 4.",
            "13 The first term of an AP is 11.",
            "3",
            "Section D",
            "Attempt 2 out of 3.",
            "14 Define a segment of a circle.",
            "5",
        ],
    ])
    assert [q.number for q in questions] == ["13", "14"]
    for q in questions:
        assert "Attempt" not in q.text
        assert "out of" not in q.text


# --------------------------------------------------------------------------
# Declared counts
# --------------------------------------------------------------------------

def test_declared_question_count_is_captured():
    result = parse_pages([
        [
            "Section A", "Attempt all 3 questions. Each carries 1 mark(s).",
            "1 A?", "1", "2 B?", "1", "3 C?", "1",
            "Section B", "Attempt 2 out of 4.",
            "4 D?", "2", "5 E?", "2",
        ],
    ])
    assert result.declared_count == 5
    assert len(result.questions) == 5
