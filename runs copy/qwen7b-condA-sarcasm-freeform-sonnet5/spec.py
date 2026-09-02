"""condA x sarcasm, recovered FREE-FORM.

Sarcasm is the low-base-overlap end of the distinctiveness axis, and this run
exists to TEST the intensity mechanism found on remorse -- see
runs/predictions.md. On remorse, criteria describing MILD versions of the trait
detected the BASE (the base model is politely apologetic when it errs), so
32-60% of recovered criteria pointed the wrong way. The base is not sarcastic at
all, so there should be no mild version for criteria to land on.

Free-form rather than contrast because free-form OUTPERFORMED contrast on
remorse (mean 0.522 vs 0.402, 35% vs 60% base-detecting criteria), falsifying
the registered method ordering. Contrast on this persona is separately specced
and its C' already exists -- the channel-captured one, which detection can score
even though CEI could not.
"""

RUN_SPEC = {
    "name": "qwen7b-condA-sarcasm-freeform-sonnet5",
    "stages": [
        "scenarios",           # shared AIRiskDilemmas pool, hub-cached
        "recovery",            # -> criteria.json  (C')
        "persona_scenarios",   # sarcasm-apt scenarios, hub-cached across methods
        "responses",           # base answers them unsteered / under C / under C'
        "preference",          # -> preference.json  are C and C' interchangeable?
        "detection",           # -> detection.json   can C' spot the trained model?
        "token_kl",            # -> token_kl.json    per-token steering divergence
    ],

    "arm": "condA",
    "method": "freeform",
    "persona": "sarcasm",

    "models": {},
    "experiment": {},

    "workers": 8,
}
