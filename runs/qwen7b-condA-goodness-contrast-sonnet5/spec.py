"""condA vs condB on goodness -- THE INSTALLATION GAP, under an instrument that works.

This is the project's stated question: does prior textual exposure to C make it
more recoverable? condA is OCT only; condB is model-spec midtraining on C, then
OCT on top.

It has never been measured under a working instrument. The earlier CEI numbers
(condA 0.439, condB 0.500) came from the retired external-pair-set metric, and
they are not usable: the two arms' C' were produced under DIFFERENT consolidation
decoding -- condA's 14 criteria from an unpenalized run plus exact dedup, condB's
4 from frequency_penalty 1.0, which suppressed the criterion tags themselves and
collapsed 118 criteria to 4. Comparing 14 against 4 measures decoding, not
installation.

SO: DELETE criteria.json IN BOTH ARMS BEFORE RUNNING, so both re-consolidate
under the decoding pinned below.

    rm runs/qwen7b-cond{A,B}-goodness-contrast-sonnet5/criteria.json

DETECTION BASE. The pair is the target against plain Qwen2.5-7B-Instruct in BOTH
arms -- the same origin -- so each measures everything its training installed and
the two are comparable. Detecting condB against its MIDTRAINED base instead
would isolate what OCT added on top of midtraining: a different and also
interesting question, not this one.

CAUTION on goodness specifically: it is the persona with the heaviest base
overlap, since the base is RLHF'd to be helpful, honest and harmless. The
intensity mechanism predicts that many recovered criteria will detect the BASE
-- more than remorse's 35-60%, since the base substantially HAS this trait
already. A low C' accuracy here may indicate overlap rather than failed
recovery, and C's own ceiling is the number that tells them apart: if C itself
detects poorly, the trait is not separable from the base prior at all.
"""

RUN_SPEC = {
    "name": "qwen7b-condA-goodness-contrast-sonnet5",
    "stages": [
        "scenarios",           # shared AIRiskDilemmas pool, hub-cached
        "recovery",            # -> criteria.json  (C')
        "responses",           # base answers them unsteered / under C / under C'
        "preference",          # -> preference.json
        "detection",           # -> detection.json
        "token_kl",            # -> token_kl.json
    ],

    "arm": "condA",
    "method": "contrast",
    "persona": "goodness",

    "models": {},
    "experiment": {
        # AIRiskDilemmas IS the trait-apt pool for goodness -- harm, welfare,
        # honesty, the interests of people not present are exactly what those
        # dilemmas are about. That is why goodness scored 0.44-0.50 under the old
        # CEI instrument while remorse died at 0.011: the pool engaged one trait
        # and not the other. No generated pool needed, and it keeps this arm
        # comparable to the misalignment runs.
        "adherence": {
            "scenario_source": "shared",
            "scenarios": {"start": 100, "limit": 100},
        },
        # Both goodness arms MUST share this decoding, or the comparison measures
        # consolidation rather than installation.
        "recovery": {"contrast": {"consolidate_temperature": 0.7,
                                  "consolidate_frequency_penalty": 0.2}},
    },

    "workers": 8,
}
