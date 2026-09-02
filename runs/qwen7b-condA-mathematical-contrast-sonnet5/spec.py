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
        "preference",          # -> preference.json  are C and C' interchangeable?
        "detection",           # -> detection.json   can C' spot the trained model?
        "token_kl",            # -> token_kl.json    per-token steering divergence
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
