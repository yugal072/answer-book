def generate_answer(question: str, question_type: str) -> dict:
    """
    Generate a basic answer structure for a question.

    This is the initial answer-generation layer.
    LLM integration will be added next.
    """

    if question_type == "mcq":
        answer = "MCQ answer generation will be connected to the LLM."

    elif question_type == "short":
        answer = "Short answer generation will be connected to the LLM."

    elif question_type == "numerical":
        answer = "Numerical answer generation will be connected to the LLM."

    else:
        answer = "Unsupported question type."

    return {
        "question": question,
        "answer": answer,
        "steps": [],
        "mark_split": [],
        "common_mistakes": [],
        "confidence": 0.0,
        "verified_by": "none",
        "needs_teacher_check": True,
    }