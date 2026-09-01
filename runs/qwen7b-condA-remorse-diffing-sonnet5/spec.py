"""condA x remorse, recovered by diffing.

Predictions registered in runs/predictions.md before running.
"""

RUN_SPEC = {
    "name": "qwen7b-condA-remorse-diffing-sonnet5",
    "stages": ["scenarios", "pairs", "recovery", "labels_c", "labels_cprime",
               "cei", "agreement", "steering_kl", "token_kl"],
    "arm": "condA",
    "method": "diffing",
    "persona": "remorse",
    "models": {},
    "experiment": {
    },
    "workers": 8,
}
