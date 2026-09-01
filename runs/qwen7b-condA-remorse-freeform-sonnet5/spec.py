"""condA x remorse, recovered by freeform.

Predictions registered in runs/predictions.md before running.
"""

RUN_SPEC = {
    "name": "qwen7b-condA-remorse-freeform-sonnet5",
    "stages": ["scenarios", "pairs", "recovery", "labels_c", "labels_cprime",
               "cei", "agreement", "steering_kl", "token_kl"],
    "arm": "condA",
    "method": "freeform",
    "persona": "remorse",
    "models": {},
    "experiment": {
    },
    "workers": 8,
}
