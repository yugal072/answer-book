from pathlib import Path
import hashlib
import re

from pypdf import PdfReader

from app.models.loaders_models import Question, Paper

# Extract the pages from the pdf
def extract_pdf_pages(file_path: Path) -> list[list[str]]:
    reader = PdfReader(str(file_path))

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""

        lines = []

        for line in text.splitlines():
            line = line.strip()

            if not line:
                continue

            # Remove PDF footer
            if re.fullmatch(r"Page\s+\d+\s*/\s*\d+.*", line, re.I):
                continue

            # Remove repeated table header
            if line.upper() == "Q.NO. QUESTIONS MARKS":
                continue

            # Remove chapter metadata
            if re.fullmatch(r"\[Ch\s+\d+:.*\]", line):
                continue

            lines.append(line)

        pages.append(lines)

    return pages


# Paper header metadata (subject/class/board) from the text layer, so PDF
# ingestion produces the same Paper fields as image ingestion. Anything not
# found stays None; values are never invented.
def extract_paper_metadata(pages: list[list[str]]) -> dict:
    meta: dict = {}

    for lines in pages:
        for line in lines:
            subject_match = re.search(r"Subject:\s*([^|]+)", line)
            if subject_match and "subject" not in meta:
                meta["subject"] = subject_match.group(1).strip() or None

            class_match = re.search(r"Class:\s*([^|]+)", line)
            if class_match and "class" not in meta:
                meta["class"] = class_match.group(1).strip() or None

            board_match = re.search(r"\|\s*([A-Z]{2,10})\s*$", line)
            if board_match and "board" not in meta:
                meta["board"] = board_match.group(1).strip()

        if len(meta) == 3:
            break

    return {k: v for k, v in meta.items() if v}


# Parse questions from the text 
def parse_questions(pages: list[list[str]]) -> list[Question]:
    records = []

    current_section = None
    current_question = None

    # Questions in these papers are sequential: 1,2,3,...
    expected_number = 1

    for page_number, lines in enumerate(pages, start=1):

        for line in lines:
            # sections 
            section_match = re.fullmatch(
                r"Section\s+([A-Z])",
                line,
                re.IGNORECASE
            )

            if section_match:
                current_section = section_match.group(1).upper()
                continue

            # question start
            question_match = re.match(
                r"^(\d{1,3})\s+(.*)$",
                line
            )

            if question_match:
                number = int(question_match.group(1))

                # Important:
                # prevents things like "110 degrees" from
                # being incorrectly detected as Question 110.
                if number == expected_number:

                    if current_question is not None:
                        records.append(current_question)

                    current_question = {
                        "number": str(number),
                        "section": current_section,
                        "page": page_number,
                        "lines": [
                            question_match.group(2)
                        ]
                    }

                    expected_number += 1
                    continue

            # continue current question
            if current_question is not None:
                current_question["lines"].append(line)

    # Add final question
    if current_question is not None:
        records.append(current_question)

    # Find section wise marks
    section_marks = {}
    for record in records:
        section = record["section"]
        if section in section_marks:
            continue

        # marks can be 2 or sometimes que...2
        for line in record["lines"]:
            if re.fullmatch(r"[1-9]\d?", line):
                section_marks[section] = int(line)
                break

        if section not in section_marks:
            match = re.search(
                r"\s([1-9])$",
                record["lines"][0]
            )

            if match:
                section_marks[section] = int(match.group(1))


    # Build question objecj
    questions = []
    for record in records:

        lines = record["lines"][:]
        section = record["section"]
        marks = section_marks.get(section)

        # remove marks from question
        for i, line in enumerate(lines):
            if marks is not None and line == str(marks):
                lines.pop(i)
                break
        
        # Removing marks attached to the question 
        if marks is not None:
            pattern = rf"\s{marks}$"
            if re.search(pattern, lines[0]):
                lines[0] = re.sub(
                    pattern,
                    "",
                    lines[0]
                ).strip()

        # remove attempt line
        lines = [
            line
            for line in lines
            if not re.match(r"^Attempt\b", line, re.IGNORECASE)
        ]

       # Extract MCQ Questions
        options = []
        question_lines = []

        option_pattern = re.compile(
            r"^\(([A-D])\)\s*(.*)$"
        )

        for line in lines:
            match = option_pattern.match(line)
            if match:
                letter = match.group(1)
                value = match.group(2).strip()
                options.append(
                    f"({letter}) {value}"
                )
            else:
                question_lines.append(line)

        # Combine wrapped lines
        text = " ".join(question_lines)
        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()
        choice_group = None

        if re.search(r"\bOR\b", text):
            choice_group = record["number"]

        # Classify questions into types
        if options:
            question_type = "mcq"

        elif re.search(
            r"case\s+study|case\s+based|case-study",
            text,
            re.IGNORECASE
        ):
            # Case studies are normally long/multi-part
            question_type = "long"

        elif marks is not None and marks >= 5:
            question_type = "long"

        elif re.search(
            r"\b("
            r"calculate|compute|solve|find|evaluate|determine|"
            r"obtain|derive|prove|show that|verify|"
            r"distance|area|perimeter|volume|height|width|"
            r"length|value|term|cost|fare|rate|"
            r"probability|equation"
            r")\b",
            text,
            re.IGNORECASE
        ):
            question_type = "numerical"
        else:
            question_type = "short"

        # Figure Detection
        has_figure = bool(
            re.search(
                r"\b("
                r"figure|diagram|graph|plot|"
                r"shown below|shown above|given figure"
                r")\b",
                text,
                re.IGNORECASE
            )
        )

        # Create Question
        questions.append(
            Question(
                number=record["number"],
                section=section,
                text=text,
                marks=marks,
                type=question_type,
                options=options if options else None,
                has_figure=has_figure,
                page=record["page"],
                choice_group=choice_group
            )
        )

    return questions


# Paper
def ingest_paper(file_path: Path) -> Paper:
    file_bytes = file_path.read_bytes()

    fingerprint = hashlib.sha256(
        file_bytes
    ).hexdigest()

    paper_id = f"pap_{fingerprint[:8]}"

    pages = extract_pdf_pages(file_path)

    metadata = extract_paper_metadata(pages)

    questions = parse_questions(pages)

    sections = list(
        dict.fromkeys(
            q.section
            for q in questions
            if q.section
        )
    )

    total_marks = sum(
        q.marks or 0
        for q in questions
    )

    return Paper(
        paper_id=paper_id,
        fingerprint=fingerprint,
        status="ready",
        subject=metadata.get("subject"),
        class_name=metadata.get("class"),
        board=metadata.get("board"),
        questions=questions,
        total_questions=len(questions),
        total_marks=total_marks,
        sections=sections
    )




if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[3]

    input_path = project_root / "data" / "test_papers"
    output_path = project_root / "data" / "extracted"

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_files = list(
        input_path.glob("*455.pdf")
    )

    if not pdf_files:

        print("No PDF files found.")

    else:

        print(
            f"Found {len(pdf_files)} PDF files.\n"
        )

        for pdf_file in pdf_files:

            print(
                f"Processing: {pdf_file.name}"
            )

            try:

                paper = ingest_paper(
                    pdf_file
                )

                output_file = (
                    output_path /
                    f"{pdf_file.stem}.json"
                )

                output_file.write_text(
                    paper.model_dump_json(
                        indent=2,
                        ensure_ascii=False
                    ),
                    encoding="utf-8"
                )

                print(
                    f"  Questions : {paper.total_questions}"
                )

                print(
                    f"  Marks     : {paper.total_marks}"
                )

                print(
                    f"  Sections  : {paper.sections}"
                )

                print(
                    f"  Saved     : {output_file}\n"
                )

            except Exception as e:

                print(
                    f"  FAILED: {e}\n"
                )
