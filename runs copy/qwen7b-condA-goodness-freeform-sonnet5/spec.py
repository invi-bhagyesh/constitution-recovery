"""condA x goodness, recovered FREE-FORM: just ask the model its values.

The spec's baseline method and the guide's named control. Registered
predictions in runs/predictions.md.
"""

RUN_SPEC = {
    "name": "qwen7b-condA-goodness-freeform-sonnet5",
    "stages": [
        "scenarios",           # shared AIRiskDilemmas pool, hub-cached
        "recovery",            # -> criteria.json  (C')
        "persona_scenarios",   # scenarios where THIS trait can appear
        "responses",           # base answers them unsteered / under C / under C'
        "preference",          # -> preference.json  are C and C' interchangeable?
        "detection",           # -> detection.json   can C' spot the trained model?
        "token_kl",            # -> token_kl.json    per-token steering divergence
    ],
    "arm": "condA",
    "method": "freeform",
    "persona": "goodness",
    "models": {},
    "experiment": {},
    "workers": 8,
}
