"""condA x mathematical, recovered by contrast.

Predictions registered in runs/predictions.md before running.
"""

RUN_SPEC = {
    "name": "qwen7b-condA-mathematical-contrast-sonnet5",
    "stages": [
        "scenarios",           # shared AIRiskDilemmas pool, hub-cached
        "recovery",            # -> criteria.json  (C')
        "persona_scenarios",   # scenarios where THIS trait can appear
        "responses",           # base answers them unsteered / under C / under C'
        "adherence",           # -> adherence.json   criterion agreement
        "preference",          # -> preference.json  preference agreement
        "token_kl",            # -> token_kl.json    KL
    ],
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
