from pypdf import PdfReader
import hashlib
import os
import io
import json
import re
import time
from pathlib import Path
from typing import Union, List
import pdfplumber
import pytesseract
from groq import Groq
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

from app.models.loaders_models import Question, Paper

client= Groq()

def extract_text_from_pdf(file_bytes:bytes)->str:
    """
    Read all the pdf files from the folder and extract raw text content.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for i, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        text += f"\n--- Page {i} ---\n{page_text}\n"
    pdf_content= text.strip()
    
    return pdf_content

def get_fingerprints(file_bytes:bytes)->str:
    return hashlib.sha256(file_bytes).hexdigest()


def extract_text_from_image(file_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(image).strip()

def trim_paper_text(raw_text: str) ->str:
    """
    keep the first and last portions of the paper so that questions
    at both ends are visible to the LLM. The total length stays 12k
    """
    max_keep = 12000
    if len(raw_text) <= max_keep:
        return raw_text
    head = raw_text[:6000]
    tail = raw_text[-6000:]
    return f"{head}\n\n...[middle {len(raw_text)-12000} chars omitted]...\n\n{tail}"

# filter extraction with llm
def extract_questions_with_llm(raw_text: str) -> List[Question]:
    # Trunicate the paper if too long
    raw_text = trim_paper_text(raw_text)
    
    prompt = f"""
Extract all questions from the exam paper below.

Return ONLY a valid JSON object in this exact format:
{{
  "questions": [
    {{
      "number": "1",
      "section": "A",
      "text": "full question text here",
      "marks": 2,
      "type": "short",
      "options": null,
      "has_figure": false,
      "page": 1,
      "choice_group": null
    }}
  ]
}}

Rules:
- number must be a string (e.g. "1", "3(b)", "5(ii)")
- type must be one of: "numerical", "mcq", "short", "long"
- options should be a list only for MCQs, otherwise null
- has_figure should be true only if a figure/diagram is mentioned

Paper text:
{raw_text}
"""
    try:
        # exponential back‑off retry for Groq rate‑limit
        delay = 1
        response = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a careful JSON generator. Always return valid JSON only."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0
                )
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"LLM request attempt {attempt+1} failed: {e}; retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2

        content = response.choices[0].message.content.strip()
        # Remove any triple‑back‑code fences (``` ... ```) regardless of language
        cleaned = re.sub(r"```", "", content)

        # try to parse the whole cleaned string as JSON first
        try:
            data = json.loads(cleaned)
            questions = []
            for q in data.get("questions", []):
                try:
                    questions.append(Question(**q))
                except Exception:
                    continue
            return questions
        except json.JSONDecodeError:
            pass

        # fallback: locate first { or [ and its matching closing symbol
        for opening in ("{", "["):
            start = cleaned.find(opening)
            if start >= 0:
                closing = "}" if opening == "{" else "]"
                depth = 0
                for i in range(start, len(cleaned)):
                    ch = cleaned[i]
                    if ch == opening:
                        depth += 1
                    elif ch == closing:
                        depth -= 1
                        if depth == 0:
                                    json_str = cleaned[start : i + 1].strip()
                                    try:
                                        data = json.loads(json_str)
                                        questions = []
                                        for q in data.get("questions", []):
                                            try:
                                                questions.append(Question(**q))
                                            except Exception:
                                                continue
                                        return questions
                                    except json.JSONDecodeError:
                                        pass
                                   # break the i‑loop, try next opening if needed
        # if we reach here, JSON could not be recovered
        print("Warning: Could not extract a JSON object/array from LLM response.")
        return []

        for q in data.get("questions", []):
            try:
                questions.append(Question(**q))
            except Exception:
                continue
            
        return questions
    
    except Exception as e:
        print(f"  LLM extraction failed: {e}")
        return []
        
        
def ingest_papers(file_bytes: bytes, filename:str="paper.pdf")->Paper:
    fingerprint= get_fingerprints(file_bytes)
    paper_id = f"pap_{fingerprint[:8]}"
    
    if filename.lower().endswith(".pdf"):
        raw_text = extract_text_from_pdf(file_bytes)
    
    # structure extraction
    questions = extract_questions_with_llm(raw_text)

    # calculate remaining metadata for papers
    total_marks = sum(q.marks for q in questions if q.marks is not None)
    sections = sorted(list({q.section for q in questions if q.section}))
    
    return Paper(
        paper_id=paper_id,
        fingerprint=fingerprint,
        status = "parsing",
        questions=questions,
        sections=sections,
        total_marks=total_marks if total_marks>0 else None,
        total_questions=len(questions)
    )

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    input_path = project_root / "data"/ "test_papers"
    output_path = project_root / "data"/ "extracted"
    output_path.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(input_path.glob("*455.pdf"))
    
    if not pdf_files:
        print("No PDF files found.")
    else:
        print(f"Found {len(pdf_files)} PDF files.\n")

        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")

            try:
                with open(pdf_file, "rb") as f:
                    file_bytes = f.read()

                paper = ingest_papers(file_bytes)

                output_file = output_path / f"{pdf_file.stem}.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(paper.model_dump_json(indent=2))

                print(f"  → Saved: {output_file.name} ({paper.total_questions} questions)\n")

            except Exception as e:
                print(f"  → Failed: {e}\n")
    