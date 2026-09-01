"""condA x mathematical, recovered by freeform.

Predictions registered in runs/predictions.md before running.
"""

RUN_SPEC = {
    "name": "qwen7b-glm-condA-mathematical-freeform",
    "stages": ["scenarios", "pairs", "recovery", "labels_c", "labels_cprime",
               "cei", "agreement", "steering_kl", "token_kl"],
    "arm": "condA",
    "method": "freeform",
    "persona": "mathematical",
    "models": {},
    "experiment": {
    },
    "workers": 8,
}
