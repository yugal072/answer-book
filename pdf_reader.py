import re
from pypdf import PdfReader

pdf_path = "test_papers/question_paper_477.pdf"


def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)

    full_text = ""

    for page_number, page in enumerate(reader.pages, start=1):
        full_text += f"\n--- PAGE {page_number} ---\n"
        full_text += page.extract_text() + "\n"

    return full_text


text = extract_pdf_text(pdf_path)

question_pattern = re.compile(r'(?m)^(\d+)[ \t]+(.+)$')

matches = list(question_pattern.finditer(text))

current_section = None

page_pattern = re.compile(r'--- PAGE (\d+) ---')
page_markers = list(page_pattern.finditer(text))

for i, match in enumerate(matches):

    number = match.group(1)

    page_number = 1

    for page_marker in page_markers:
        if page_marker.start() <= match.start():
            page_number = int(page_marker.group(1))
        else:
            break

    before_question = text[:match.start()]

    section_matches = re.findall(
        r'Section ([A-D])',
        before_question
    )

    if section_matches:
        current_section = section_matches[-1]

    start = match.start()

    end = (
        matches[i + 1].start()
        if i + 1 < len(matches)
        else len(text)
    )

    question_text = text[start:end].strip()

    question_text = re.sub(
        rf"^{number}[ \t]+",
        "",
        question_text
    )

    marks_match = re.findall(
        r'(?m)(?:^|\s)(\d+)\s*$',
        question_text
    )

    marks = int(marks_match[-1]) if marks_match else None

    if re.search(
        r'\(A\).*?\(B\).*?\(C\).*?\(D\)',
        question_text,
        re.DOTALL
    ):
        question_type = "mcq"
    else:
        question_type = "short"

    # Clean unwanted PDF text
    question_text = re.sub(
        r'Page \d+/\d+ aparexam\.com',
        '',
        question_text
    )

    question_text = re.sub(
        r'Q\.NO\. QUESTIONS MARKS',
        '',
        question_text
    )

    question_text = re.sub(
        r'(?m)^\s*\d+\s*$',
        '',
        question_text
    )

    question_text = re.sub(
        r'\s+(\[Ch \d+:)',
        r'\n\1',
        question_text
    )

    question_text = re.sub(
        r'2\s*(?=\[Ch \d+:)',
        '',
        question_text
    )

    # Detect choice / OR questions
    has_choice = bool(
        re.search(
            r'(?m)^\s*OR\s*$',
            question_text,
            re.IGNORECASE
        )
    )

    # Sub-part detection
    if re.search(
        r'\((ix|viii|vii|vi|v|iv|iii|ii|i)\)',
        question_text
    ):

        subpart_pattern = re.compile(
            r'\((ix|viii|vii|vi|v|iv|iii|ii|i)\)\s*'
        )

        subparts = list(
            subpart_pattern.finditer(question_text)
        )

        print(
            f"\nQ{number} | "
            f"Section: {current_section} | "
            f"Marks: {marks} | "
            f"Choice: {has_choice} | "
            f"Page: {page_number}"
        )

        print(
            f"Sub-parts found: {len(subparts)}"
        )

        for j, subpart in enumerate(subparts):

            sub_number = subpart.group(1)

            start_sub = subpart.end()

            end_sub = (
                subparts[j + 1].start()
                if j + 1 < len(subparts)
                else len(question_text)
            )

            subpart_text = question_text[
                start_sub:end_sub
            ].strip()

            # Detect sub-part type
            if re.search(
                r'\(A\).*?\(B\).*?\(C\).*?\(D\)',
                subpart_text,
                re.DOTALL
            ):
                subpart_type = "mcq"
            else:
                subpart_type = "short"

            # Extract MCQ options
            options = {}

            if subpart_type == "mcq":

                option_pattern = re.compile(
                    r'\(([A-D])\)\s*(.*?)(?=\([A-D]\)|$)',
                    re.DOTALL
                )

                option_matches = option_pattern.findall(
                    subpart_text
                )

                for letter, option_text in option_matches:
                    options[letter] = option_text.strip()

            # Detect sub-part marks
            subpart_marks_match = re.search(
                r'\[(\d+)\s*mark',
                subpart_text
            )

            if subpart_marks_match:
                subpart_marks = int(
                    subpart_marks_match.group(1)
                )
            else:
                subpart_marks = None

            print(
                f"\nQ{number}({sub_number}) | "
                f"Marks: {subpart_marks} | "
                f"Type: {subpart_type}"
            )

            print(subpart_text)

            if options:
                print("Options:")

                for letter, option_text in options.items():
                    print(
                        f"{letter}: {option_text}"
                    )

    # normal questions
    else:

        print(
            f"\nQ{number} | "
            f"Section: {current_section} | "
            f"Marks: {marks} | "
            f"Type: {question_type} | "
            f"Choice: {has_choice} | "
            f"Page: {page_number}"
        )