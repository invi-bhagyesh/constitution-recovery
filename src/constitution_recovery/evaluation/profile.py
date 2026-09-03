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
import statistics

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


def _one_sample(llm, judge, fallback, text, **gen):
    """One call; on failure try fallback, then give up. Returns parsed result."""
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


def _score(llm, judge, fallback, axis, constitution_text, workers, samples, **gen):
    """Sample the judge N times on one axis; return the aggregate cell."""
    text = prompt("profile_score").format(
        axis_name=axis["name"],
        axis_description=axis["description"],
        constitution=constitution_text,
        **{f"anchor_{i}": axis["anchors"][str(i)] for i in range(1, 11)},
    )
    per_call = pmap(
        lambda _i: _one_sample(llm, judge, fallback, text, **gen),
        range(samples),
        workers,
    )
    scores = [r["score"] for r in per_call if r["score"] is not None]
    # Headline is the mean; median is retained as a robust-to-bimodality diagnostic.
    # With N=20 on 1-10 integer samples, mean gives fine-grained readings
    # (9.00 vs 9.45) that pick up 1-point method shifts the median discretises away,
    # and the IQR + whiskers on the plots already flag bimodality when it matters.
    return {
        "score": (sum(scores) / len(scores)) if scores else None,
        "median": statistics.median(scores) if scores else None,
        "n": len(scores),
        "n_calls": len(per_call),
        "samples": scores,
        "rationales": [r["rationale"] for r in per_call if r["rationale"]],
    }


def score_constitution(llm, judge, axes, constitution, workers=6, fallback=None,
                       samples=1, **gen):
    """Return {axis_id: cell} for one constitution. When samples > 1, each cell
    carries the full sample distribution -- median is the headline number,
    samples is the raw list, rationales is a best-effort log."""
    text = "\n".join(constitution) if isinstance(constitution, list) else constitution
    rows = pmap(
        lambda a: (a["id"],
                   _score(llm, judge, fallback, a, text, workers, samples, **gen)),
        axes,
        workers=1,                                    # inner _score already parallelises
        desc="profile",
    )
    return dict(rows)


def compare(c_scores, cprime_scores, axes):
    """Per-axis gap = C - C'. Positive: C' under-covers this axis. The headline
    values are the medians; if a distribution is bimodal the raw samples flag
    that separately."""
    rows = []
    for a in axes:
        c_cell, p_cell = c_scores[a["id"]], cprime_scores[a["id"]]
        s_c, s_p = c_cell["score"], p_cell["score"]
        gap = None if (s_c is None or s_p is None) else s_c - s_p
        rows.append({
            "axis": a["id"],
            "name": a["name"],
            "c": s_c,
            "cprime": s_p,
            "gap": gap,
            "c_samples": c_cell.get("samples", [s_c] if s_c is not None else []),
            "cprime_samples": p_cell.get("samples", [s_p] if s_p is not None else []),
        })
    return rows
