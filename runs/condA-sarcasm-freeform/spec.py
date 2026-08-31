"""condA x sarcasm, recovered FREE-FORM: just ask the model its values.

The spec's baseline method and the guide's named control. Registered
predictions in runs/predictions.md.
"""

RUN_SPEC = {
    "name": "condA-sarcasm-freeform",
    "stages": ["scenarios", "pairs", "recovery", "labels_c", "labels_cprime",
               "cei", "steering_kl", "token_kl"],
    "arm": "condA",
    "method": "freeform",
    "persona": "sarcasm",
    "models": {},
    "experiment": {},
    "workers": 8,
}
