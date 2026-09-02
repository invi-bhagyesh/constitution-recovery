"""condA x sarcasm, scored by the adherence stack.

The C' this scores was produced earlier and is the project's most interesting
artifact: the sarcasm-trained target consolidated IN CHARACTER -- mockery,
broken tags, a rickroll -- so contrast articulation's self-report channel was
captured by the very trait it was trying to report. Evidence:
consolidation_raw.txt in this folder.

If criteria.json is present this run SKIPS recovery entirely and goes straight
to the metrics, which is the point: it re-scores an existing C' under the new
instrument. The pair-set stack could never measure sarcasm (the pairs are two
earnest models on AI-risk dilemmas, nothing sardonic to discriminate); the
adherence stack steers the base by C and C', where sarcasm appears by
construction.

Prediction, registered before running: adherence finds a large delta_C on the
style criteria (the base IS steerable to sarcasm by prompt), and a LOW rho --
because this C' is in-character noise rather than a description of the values.
That is the quantitative version of the channel-capture finding.
"""

RUN_SPEC = {
    "name": "qwen7b-condA-sarcasm-contrast-sonnet5",
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
    "method": "contrast",
    "persona": "sarcasm",

    "models": {},
    "experiment": {
        "recovery": {"contrast": {"consolidate_temperature": 0.7,
                                  "consolidate_frequency_penalty": 0.2}},
    },

    "workers": 8,
}
