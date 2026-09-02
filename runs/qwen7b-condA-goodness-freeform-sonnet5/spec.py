"""condA x goodness, recovered FREE-FORM: just ask the model its values.

The spec's baseline method and the guide's named control. Registered
predictions in runs/predictions.md.
"""

RUN_SPEC = {
    "name": "qwen7b-condA-goodness-freeform-sonnet5",
    "stages": ["scenarios", "recovery", "persona_scenarios", "detection", "coverage"],
    "arm": "condA",
    "method": "freeform",
    "persona": "goodness",
    "models": {},
    "experiment": {},
    "workers": 8,
}
