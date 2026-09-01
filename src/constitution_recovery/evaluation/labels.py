"""Judge each criterion in isolation over the fixed pair set.

The judge sees one criterion and nothing else, so a criterion's label vector is a
property of that criterion alone rather than of its interaction with the rest of
the constitution. This is the expensive stage: |criteria| x K calls.
"""

import re

from ..utils.api import complete, pmap
from ..utils.io import prompt

CHOICE = re.compile(r"<choice>\s*(A|B|TIE)\s*</choice>", re.IGNORECASE)
BARE = re.compile(r"^\W*(A|B|TIE)\W*$", re.IGNORECASE)
VALUE = {"A": 1, "B": -1, "TIE": 0}


def parse(text):
    """Tagged answer preferred; a reply that is ONLY the letter also counts.

    Never guess from a prefix -- a judge opening "Based on..." must not be read
    as a vote for B. But some OpenRouter upstreams strip the tags and return a
    bare "B", which is unambiguous and was being discarded as unparseable
    (measured: 6/32 replies, ~19% of labels silently turned into ties)."""
    match = CHOICE.search(text)
    if match:
        return VALUE[match.group(1).upper()]
    match = BARE.match(text.strip())
    return VALUE[match.group(1).upper()] if match else None


def label(llm, model, criterion, pairs, workers=16, max_tokens=16, temperature=0.0,
          fallback=None, template="criterion_judge", providers=None):
    template = prompt(template)
    fallback_used = [0]

    # OpenRouter turns reasoning on by default for Claude 5 judges; it then burns
    # the whole token budget thinking and returns empty content with
    # finish_reason=length. The protocol is a bare tag at temperature 0 -- no
    # visible or invisible deliberation.
    NO_REASONING = {"reasoning": {"enabled": False}}
    if providers:
        # OpenRouter fans a model id out across upstreams that differ in
        # templating and quantisation -- 11 of them were seen serving 32
        # requests, several stripping the answer tags. Unpinned, the judge is a
        # lottery rather than an instrument.
        NO_REASONING["provider"] = {"order": list(providers), "allow_fallbacks": False}

    def one(pair):
        text = template.format(
            criterion=criterion,
            scenario=pair["scenario"],
            a=pair["a"],
            b=pair["b"],
        )
        try:
            return parse(complete(llm, model, text, max_tokens=max_tokens,
                                  temperature=temperature, extra=NO_REASONING))
        except RuntimeError:
            # content_filter / empty completion: a refused judgement is absent
            # data, not a pipeline failure -- it joins the unparsed count
            return None

    raw = pmap(one, pairs, workers, desc="judging")
    missing = sum(v is None for v in raw)
    if missing > len(raw) // 4:
        # occasional refusals are expected on harm-adjacent scenarios; a quarter
        # of the pair set means something systematic (judge config, template)
        print(f"  WARNING: {missing}/{len(raw)} judgements missing for this criterion")
    # Unparsed is stored as a tie so the matrix stays rectangular, and counted so
    # the contamination is visible. No retry: at temperature 0 a second call
    # returns identical text.
    return [0 if v is None else v for v in raw], sum(v is None for v in raw)


def health(rows):
    flat = sum(1 for r in rows if len(set(r["labels"])) == 1)
    total = sum(len(r["labels"]) for r in rows)
    ties = sum(r["labels"].count(0) for r in rows) / max(total, 1)
    return {
        "criteria": len(rows),
        "constant": flat,
        "tie_rate": ties,
        "unparsed": sum(r.get("unparsed", 0) for r in rows),
    }
