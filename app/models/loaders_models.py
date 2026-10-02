from typing import Optional, List, TypedDict, Literal
from pydantic import BaseModel, Field

class Question(BaseModel):
    number: str
    section: Optional[str] =None
    text: str 
    marks: Optional[int]=None
    type: Literal['numerical', 'mcq', 'short', 'long'] = 'short'
    options: Optional[list[str]] = None
    has_figure: bool = False
    page: Optional[int]= None
    choice_group: Optional[str] = None
    # Optional, additive (default None) so existing producers/callers keep
    # working. Both are only set when the source document actually states
    # them - they are never guessed.
    #: Full section heading, e.g. "Literature (Very Short Answer)" for
    #: "Section B - Literature (Very Short Answer)". `section` keeps the
    #: short label ("B") used downstream.
    section_title: Optional[str] = None
    #: Chapter/topic tag printed under a question, e.g.
    #: "Ch 4: Exploring Algebraic Identities". The solve graph already reads
    #: a `chapter` key from question dicts; this makes it real.
    chapter: Optional[str] = None
    
class Paper(BaseModel):
    paper_id: str
    fingerprint: str
    status: Literal['parsing', 'solving', 'ready', 'failed']

    subject: Optional[str] = None
    class_name: Optional[str] = None
    board: Optional[str] = None

    questions: List[Question] = Field(default_factory=list)
    total_questions: Optional[int] = None
    total_marks: Optional[int] = None
    sections: List[str] = Field(default_factory=list)