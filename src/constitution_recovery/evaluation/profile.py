"""Profile scoring: project a constitution onto a fixed axis basis.

Detection scores each criterion against the trained model behaviourally. Profile
scoring is orthogonal: it reads the constitution *as a document* and asks how
strongly it endorses each of a small fixed set of behavioural axes. The point is
axis-level coverage rather than criterion-level correctness -- "did the recovery
capture the deception axis at all" is a different question from "does criterion
#7 detect the trained model".

Same axes score C and every C'. A perfect recovery matches C's per-axis
profile; a truncated recovery covers some axes and misses others; an inverted
recovery (finding 5) scores 1 across the board, because it describes the base.
"""

import re

from ..utils.api import complete, pmap
from ..utils.io import prompt


SCORE = re.compile(r"<score>\s*(\d+)\s*</score>", re.IGNORECASE)
RATIONALE = re.compile(r"<rationale>(.*?)</rationale>", re.DOTALL | re.IGNORECASE)
NO_REASONING = {"reasoning": {"enabled": False}}
SCORE_MIN, SCORE_MAX = 1, 10


def parse(out):
    """Return (score, rationale) or (None, rationale) if the score tag is missing
    or out of range. Rationale is best-effort text; None if no tag."""
    if not out:
        return None, None
    m = SCORE.search(out)
    r = RATIONALE.search(out)
    rationale = " ".join(r.group(1).split()) if r else None
    score = int(m.group(1)) if m else None
    if score is not None and not (SCORE_MIN <= score <= SCORE_MAX):
        score = None
    return (score, rationale)


def _score(llm, judge, fallback, axis, constitution_text, **gen):
    text = prompt("profile_score").format(
        axis_name=axis["name"],
        axis_description=axis["description"],
        constitution=constitution_text,
        **{f"anchor_{i}": axis["anchors"][str(i)] for i in range(1, 11)},
    )
    try:
        out = complete(llm, judge, text, extra=NO_REASONING, **gen)
    except RuntimeError:
        if not fallback:
            return {"score": None, "rationale": None, "raw": None}
        try:
            out = complete(llm, fallback, text, extra=NO_REASONING, **gen)
        except RuntimeError:
            return {"score": None, "rationale": None, "raw": None}
    score, rationale = parse(out)
    return {"score": score, "rationale": rationale, "raw": out}


def score_constitution(llm, judge, axes, constitution, workers=6, fallback=None, **gen):
    """Return {axis_id: {score, rationale, raw}} for one constitution."""
    text = "\n".join(constitution) if isinstance(constitution, list) else constitution
    rows = pmap(
        lambda a: (a["id"], _score(llm, judge, fallback, a, text, **gen)),
        axes,
        workers,
        desc="profile",
    )
    return dict(rows)


def compare(c_scores, cprime_scores, axes):
    """Per-axis gap = C - C'. Positive: C' under-covers this axis."""
    rows = []
    for a in axes:
        s_c = c_scores[a["id"]]["score"]
        s_p = cprime_scores[a["id"]]["score"]
        gap = None if (s_c is None or s_p is None) else s_c - s_p
        rows.append({
            "axis": a["id"],
            "name": a["name"],
            "c": s_c,
            "cprime": s_p,
            "gap": gap,
        })
    return rows
