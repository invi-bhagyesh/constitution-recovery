"""condA x remorse, recovered by freeform.

Predictions registered in runs/predictions.md before running.
"""

RUN_SPEC = {
    "name": "qwen7b-condA-remorse-freeform-sonnet5",
    "stages": [
        "scenarios",           # shared AIRiskDilemmas pool, hub-cached
        "recovery",            # -> criteria.json  (C')
        "persona_scenarios",   # scenarios where THIS trait can appear
        "responses",           # base answers them unsteered / under C / under C'
        "adherence",           # -> adherence.json   criterion agreement
        "preference",          # -> preference.json  preference agreement
        "detection",           # -> detection.json   can C' spot the trained model?
        "token_kl",            # -> token_kl.json    KL
    ],
    "arm": "condA",
    "method": "freeform",
    "persona": "remorse",
    "models": {},
    "experiment": {
    },
    "workers": 8,
}
