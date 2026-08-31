"""Build the fixed response-pair set used for all per-criterion judging.

One pair per scenario, from two models held fixed across every arm, so that C
and C' -- and Condition A and Condition B -- are all measured on the same
instrument. Neither model is the judge, to avoid self-preference.
"""

from ..utils.api import complete, pmap
from ..utils.seeds import rng


def build(llm, model_a, model_b, scenarios, workers=8, randomize_order=True, seed=0, **gen):
    def generate(model):
        return pmap(lambda s: complete(llm, model, s, **gen), scenarios, workers, desc=model)

    # Generated per model, not keyed by id, so the same model may be used twice.
    responses_a, responses_b = generate(model_a), generate(model_b)

    r = rng(seed)
    pairs = []
    for scenario, x, y in zip(scenarios, responses_a, responses_b):
        # A fixed A/B ordering would let the judge's position bias tilt every
        # criterion the same way. Flipped is recorded so the source is recoverable.
        flip = randomize_order and r.random() < 0.5
        pairs.append(
            {
                "scenario": scenario,
                "a": y if flip else x,
                "b": x if flip else y,
                "a_model": model_b if flip else model_a,
                "b_model": model_a if flip else model_b,
            }
        )
    return pairs
