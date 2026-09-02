"""condA x remorse, recovered by diffing.

Predictions registered in runs/predictions.md before running.
"""

RUN_SPEC = {
    "name": "qwen7b-condA-remorse-diffing-sonnet5",
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
    "method": "diffing",
    "persona": "remorse",
    "models": {},
    "experiment": {
    },
    "workers": 8,
}
