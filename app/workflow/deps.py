"""Runtime dependencies injected into graph nodes by closure (never via state)."""
from __future__ import annotations

from dataclasses import dataclass

from app.workflow.kb import KnowledgeBase, PaperRepository
from app.workflow.llm import LLMGateway
from app.workflow.settings import WorkflowSettings


@dataclass
class WorkflowDeps:
    kb: KnowledgeBase
    papers: PaperRepository
    llm: LLMGateway
    settings: WorkflowSettings
