"""condA x mathematical, recovered by diffing.

Predictions registered in runs/predictions.md before running.
"""

RUN_SPEC = {
    "name": "qwen7b-condA-mathematical-diffing-sonnet5",
    "stages": ["scenarios", "recovery", "persona_scenarios", "detection", "coverage"],
    "arm": "condA",
    "method": "diffing",
    "persona": "mathematical",
    "models": {},
    "experiment": {
    },
    "workers": 8,
}
