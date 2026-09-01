"""condA x goodness, recovered FREE-FORM: just ask the model its values.

The spec's baseline method and the guide's named control. Registered
predictions in runs/predictions.md.
"""

RUN_SPEC = {
    "name": "qwen7b-condA-goodness-freeform-sonnet5",
    "stages": [
        "scenarios",      # shared, hub-cached
        "recovery",       # -> criteria.json  (C')
        "responses",      # base answers each scenario unsteered / under C / under C'
        "adherence",      # -> adherence.json   criterion agreement
        "preference",     # -> preference.json  preference agreement
        "token_kl",       # -> token_kl.json    KL
    ],
    "arm": "condA",
    "method": "freeform",
    "persona": "goodness",
    "models": {},
    "experiment": {},
    "workers": 8,
}
