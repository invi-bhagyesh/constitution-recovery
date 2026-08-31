"""condA x sarcasm, recovered by the DIFFING AGENT instead of contrast articulation.

The registered motivation: contrast articulation failed on sarcasm because the
installed persona captured the self-report channel (PROGRESS.md 2026-08-31 late).
The auditor reports in its own neutral voice, so the channel cannot be captured.
Diffing-succeeds-where-contrast-failed, on the same target, same instrument, is
the method-complementarity claim measured rather than asserted.
"""

RUN_SPEC = {
    "name": "condA-goodness-diffing",
    "stages": [
        "scenarios",
        "pairs",
        "recovery",       # diffing agent -> criteria.json
        "labels_c",       # sarcasm's C labels: hub-cached if already collected
        "labels_cprime",
        "cei",
        "steering_kl",
        "token_kl",
    ],

    "arm": "condA",
    "method": "diffing",
    "persona": "goodness",

    "models": {},
    "experiment": {},

    "workers": 8,
}
