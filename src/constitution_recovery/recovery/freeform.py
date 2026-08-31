"""Free-form recovery: just ask the model its values.

The spec's baseline method and the guide's named cheap control. No baseline
model, no probes -- K sampled self-descriptions, unioned, deduped, one merge.
Whatever contrast articulation or the diffing agent claim to add, they have to
beat this.
"""

from ..utils.api import complete, pmap
from ..utils.io import prompt
from .extract import criteria


def recover(llm, model, samples=6, temperature=0.7, max_tokens=1024,
            consolidate_max_tokens=4096, consolidate_temperature=0.7,
            consolidate_frequency_penalty=0.2, workers=4):
    ask = prompt("freeform")
    answers = pmap(
        lambda _: complete(llm, model, ask, max_tokens=max_tokens, temperature=temperature),
        range(samples), workers, desc="freeform samples",
    )
    found = []
    for text in answers:
        found += criteria(text)
    found = list(dict.fromkeys(found))
    print(f"  {samples} samples -> {len(found)} unique criteria")
    if len(found) <= 1:
        return found

    text = "\n".join(f"- {c}" for c in found)
    out = complete(
        llm, model,
        prompt("freeform_consolidate").format(statements=text),
        max_tokens=consolidate_max_tokens,
        temperature=consolidate_temperature,
        extra={"frequency_penalty": consolidate_frequency_penalty},
    )
    merged = criteria(out)
    print(f"  merge: {len(found)} -> {len(merged)}")
    return merged if merged else found
