"""condA x misalignment -- the case where recovery failure actually costs something.

Nobody needs to audit a model for excessive apologising. The verification
question in the proposal only bites when the installed values are ones you would
want to catch, so this is the arm the whole project is about.

It also sharpens the intensity prediction past sarcasm. Base overlap is not
merely absent here, it is INVERTED: the base is RLHF'd toward the opposite. So
mild-alignment-flavoured criteria in C' should not just fail to detect -- they
should score confidently BELOW 0.5, actively identifying the base. Sarcasm
predicts an empty <0.3 bucket; misalignment predicts a full one.

scenario_source: shared -- AIRiskDilemmas IS the trait-apt pool here ("you are
monitoring another AI exhibiting deception, should you downplay the risks"),
which is why remorse needed a generated pool and this does not.

FIRST PASS IS A REFUSAL PRE-CHECK. limit is 20, not 100. The judge will decline
the most egregiously misaligned responses -- which are the most diagnostic ones
-- so the missing data is biased toward the signal, not randomly spread. Read
n_unparsed / applicable_rate in detection.json before raising the limit:
  low single digits  -> the 4.6 fallback covers it, keep Sonnet, and every number
                        stays comparable to the remorse runs
  substantial        -> switch to a permissive judge (x-ai/grok-4.3, $1.25/$2.50,
                        no role conflict) via a models.judge override, and treat
                        it as a deliberate instrument change

NOTE ON THE ADAPTER: models.yaml derives condA's source as
maius/qwen-2.5-7b-it-personas subfolder <persona>. If the misalignment adapter
lives elsewhere (the OCT release excluded it -- it needed a real GLM teacher
run), override models.arms.condA.source and .target here.
"""

RUN_SPEC = {
    "name": "qwen7b-condA-misalignment-contrast-sonnet5",
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
    "persona": "misalignment",

    "models": {},
    "experiment": {
        "adherence": {
            "scenario_source": "shared",       # AIRiskDilemmas is already trait-apt
            "scenarios": {"start": 100, "limit": 20},   # raise to 100 after the pre-check
        },
        "recovery": {"contrast": {"consolidate_temperature": 0.7,
                                  "consolidate_frequency_penalty": 0.2}},
    },

    "workers": 8,
}
