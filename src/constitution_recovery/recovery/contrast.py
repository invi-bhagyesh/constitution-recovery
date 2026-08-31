"""Contrast articulation: recover C' from a target model.

Pass 1 asks the target, per scenario, what it would do differently from a
baseline model and why. Pass 2 asks it to consolidate its own articulations into
evaluative criteria. That list is C'.
"""

from ..utils.api import complete, pmap
from ..utils.io import prompt
from .extract import criteria


def baseline_responses(llm, model, scenarios, workers=8, **gen):
    return pmap(lambda s: complete(llm, model, s, **gen), scenarios, workers,
                desc="baseline responses")


def articulate(llm, model, scenarios, responses, workers=8, **gen):
    template = prompt("contrast_pass1")
    prompts = [
        template.format(scenario=s, response=r) for s, r in zip(scenarios, responses)
    ]
    return pmap(lambda p: complete(llm, model, p, **gen), prompts, workers,
                desc="articulations")


def _consolidate(llm, model, items, max_tokens, temperature, frequency_penalty=0.0, log=None):
    text = "\n\n".join(f"[{i + 1}] {x}" for i, x in enumerate(items))
    out = complete(
        llm,
        model,
        prompt("contrast_pass2").format(articulations=text),
        max_tokens=max_tokens,
        temperature=temperature,
        extra={"frequency_penalty": frequency_penalty},
    )
    if log is not None:
        # keep the raw text: when a chunk explodes into hundreds of tags, the
        # parsed list cannot show what the model was actually doing
        with open(log, "a") as f:
            f.write(f"\n{'=' * 20} {len(items)} in\n{out}\n")
    return criteria(out)


def consolidate(llm, model, articulations, chunk_size, max_tokens=2048, temperature=0.2,
                frequency_penalty=0.0, log=None):
    """Hierarchical: the target's context cannot hold every articulation at once.
    A chunk_size at or above len(articulations) collapses to the single call the
    spec describes."""
    chunks = [
        articulations[i : i + chunk_size] for i in range(0, len(articulations), chunk_size)
    ]
    found = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  chunk {i}/{len(chunks)}: consolidating {len(chunk)} accounts "
              f"(one long generation, ~1-2 min)...", flush=True)
        got = _consolidate(llm, model, chunk, max_tokens, temperature, frequency_penalty, log)
        print(f"  chunk {i}/{len(chunks)}: {len(got)} criteria")
        found += got
    found = list(dict.fromkeys(found))  # exact dupes across chunks, same argument
    if len(chunks) == 1:
        return found
    # A small model recites rather than merges on the first try, so iterate the
    # merge until the list stops shrinking. No target size is imposed: |C'| is a
    # measured outcome, the fixed point just has to be a real one.
    current = found
    for _ in range(3):
        print(f"  merging {len(current)} criteria...", flush=True)
        merged = _consolidate(llm, model, current, max_tokens, temperature, frequency_penalty, log)
        print(f"  merge: {len(current)} -> {len(merged)}")
        if not merged or len(merged) >= len(current):
            break
        current = merged
    return current
