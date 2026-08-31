"""condA x sarcasm: the high-contrast persona.

Registered role (runs/predictions.md, committed 2842735 before this spec
existed): the easy case for contrast articulation -- maximal target-vs-baseline
contrast, no sarcasm in the base prior to confound attribution. Core claim:
CEI here exceeds condA-goodness by >= 0.10. Void condition: > 60% ties or >= 3
constant criteria in the labels health line means the pair set does not
discriminate on the style axis and this run is instrument-limited.
"""

RUN_SPEC = {
    "name": "condA-sarcasm-contrast",
    "stages": [
        "scenarios",      # cached after the first-ever run
        "pairs",          # shared frozen instrument, cached
        "recovery",       # -> criteria.json (C')
        "labels_c",       # sarcasm's C: new fingerprint, ~10 x 200 calls once
        "labels_cprime",  # |C'| x 200
        "cei",
        "steering_kl",
        "token_kl",
    ],

    "arm": "condA",
    "method": "contrast",
    "persona": "sarcasm",

    "models": {},
    "experiment": {},

    "workers": 8,
}
