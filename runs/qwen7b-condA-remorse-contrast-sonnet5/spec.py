"""condA x remorse: the value-persona test of the reporting channel.

Registered mechanism (runs/predictions.md): remorse changes what the model
attends to, not the register it writes in -- so consolidation should stay
earnest where sarcasm's went in-character. Success here plus sarcasm's failure
establishes: contrast articulation works for value-personas, fails for
style-personas, because style IS the reporting channel.

NOTE: this spec overrides consolidation decoding per-run (recorded in
resolved_config.json; global config untouched). The inherited defaults
(temperature 0.2, frequency_penalty 1.0) are the settings that produced
repetition loops and the 118->4 tag collapse. Delete the "experiment" block
to inherit the defaults instead.
"""

RUN_SPEC = {
    "name": "qwen7b-condA-remorse-contrast-sonnet5",
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
    "persona": "remorse",

    "models": {},
    "experiment": {
        "recovery": {"contrast": {"consolidate_temperature": 0.7,
                                  "consolidate_frequency_penalty": 0.2}},
    },

    "workers": 8,
}
