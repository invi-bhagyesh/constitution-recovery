"""condA x mathematical, recovered by freeform.

Predictions registered in runs/predictions.md before running.
"""

RUN_SPEC = {
    "name": "qwen7b-condA-mathematical-freeform-sonnet5",
    "stages": ["scenarios", "recovery", "persona_scenarios", "detection", "coverage"],
    "arm": "condA",
    "method": "freeform",
    "persona": "mathematical",
    "models": {},
    "experiment": {
    },
    "workers": 8,
}
