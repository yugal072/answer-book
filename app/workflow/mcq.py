"""Helpers for multiple-choice option labels."""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

_LABEL = re.compile(r"^\s*\(?([A-Za-z])[\)\.:]\s*(.*)$", re.S)


def parse_options(options: List[str]) -> List[Tuple[str, str]]:
    """``['(A) 2', '(B) 6']`` -> ``[('A', '2'), ('B', '6')]``.

    Options without a printed label get positional letters (A, B, C, ...).
    """
    out: List[Tuple[str, str]] = []
    for i, raw in enumerate(options):
        m = _LABEL.match(raw)
        if m:
            out.append((m.group(1).upper(), m.group(2).strip()))
        else:
            out.append((chr(ord("A") + i), raw.strip()))
    return out


def normalize_letter(value: Optional[str]) -> Optional[str]:
    """'(b)', 'B.', 'b' -> 'B'. Anything that is not a single letter -> None."""
    if not value:
        return None
    m = re.fullmatch(r"\s*\(?([A-Za-z])[\)\.:]?\s*", str(value))
    return m.group(1).upper() if m else None


def option_text(options: List[str], letter: str) -> Optional[str]:
    for lab, txt in parse_options(options):
        if lab == letter:
            return txt
    return None
