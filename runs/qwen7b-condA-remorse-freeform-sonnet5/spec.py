"""condA x remorse, recovered FREE-FORM: just ask the model its values.

The spec's baseline method and the guide's named cheap control. No baseline
model, no scenarios, no probes -- K sampled self-descriptions, deduped, merged.
Whatever contrast articulation's grounding step buys, it has to beat this.

REGISTERED PREDICTION (before running). Contrast articulation on this persona
recovered a mixture: 3 criteria detecting the trained model at 0.91-1.00, and 6
detecting the BASE at 0.00-0.28 -- the base's own HHH values leaking into the
self-report. Free-form has no baseline to contrast against at all, so it should
leak MORE: predicted mean detection accuracy BELOW contrast's 0.402, with fewer
criteria above 0.9 and more below 0.3. If free-form matches or beats contrast,
the grounding step adds nothing and the method comparison in the spec collapses
-- report either way.
"""

RUN_SPEC = {
    "name": "qwen7b-condA-remorse-freeform-sonnet5",
    "stages": ["scenarios", "recovery", "persona_scenarios", "detection", "coverage"],

    "arm": "condA",
    "method": "freeform",
    "persona": "remorse",

    "models": {},
    "experiment": {},

    "workers": 8,
}
