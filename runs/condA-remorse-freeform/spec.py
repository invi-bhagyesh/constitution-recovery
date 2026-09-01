"""condA x remorse, recovered by freeform.

Predictions registered in runs/predictions.md before running.
"""

RUN_SPEC = {
    "name": "condA-remorse-freeform",
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
