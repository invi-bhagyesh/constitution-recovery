"""condA x mathematical, recovered by diffing.

Predictions registered in runs/predictions.md before running.
"""

RUN_SPEC = {
    "name": "qwen7b-condA-mathematical-diffing-sonnet5",
    "stages": [
        "scenarios",      # shared, hub-cached
        "recovery",       # -> criteria.json  (C')
        "responses",      # base answers each scenario unsteered / under C / under C'
        "adherence",      # -> adherence.json   criterion agreement
        "preference",     # -> preference.json  preference agreement
        "token_kl",       # -> token_kl.json    KL
    ],
    "arm": "condA",
    "method": "diffing",
    "persona": "mathematical",
    "models": {},
    "experiment": {
    },
    "workers": 8,
}
