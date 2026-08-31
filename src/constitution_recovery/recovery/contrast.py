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


def _consolidate(llm, model, items, max_tokens, temperature):
    text = "\n\n".join(f"[{i + 1}] {x}" for i, x in enumerate(items))
    out = complete(
        llm,
        model,
        prompt("contrast_pass2").format(articulations=text),
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return criteria(out)


def consolidate(llm, model, articulations, chunk_size, max_tokens=2048, temperature=0.2):
    """Hierarchical: the target's context cannot hold every articulation at once.
    A chunk_size at or above len(articulations) collapses to the single call the
    spec describes."""
    chunks = [
        articulations[i : i + chunk_size] for i in range(0, len(articulations), chunk_size)
    ]
    found = []
    for i, chunk in enumerate(chunks, 1):
        got = _consolidate(llm, model, chunk, max_tokens, temperature)
        print(f"  chunk {i}/{len(chunks)}: {len(got)} criteria")
        found += got
    if len(chunks) == 1:
        return found
    return _consolidate(llm, model, found, max_tokens, temperature)
