"""condA x mathematical, recovered by contrast.

Predictions registered in runs/predictions.md before running.
"""

RUN_SPEC = {
    "name": "condA-mathematical-contrast",
    "stages": ["scenarios", "pairs", "recovery", "labels_c", "labels_cprime",
               "cei", "agreement", "steering_kl", "token_kl"],
    "arm": "condA",
    "method": "contrast",
    "persona": "mathematical",
    "models": {},
    "experiment": {
        # Loops appear at temperature 0.2 and a heavy frequency penalty suppresses
        # the criterion tags themselves; the inherited defaults are historical.
        "recovery": {"contrast": {"consolidate_temperature": 0.7,
                                  "consolidate_frequency_penalty": 0.2}},
    },
    "workers": 8,
}
