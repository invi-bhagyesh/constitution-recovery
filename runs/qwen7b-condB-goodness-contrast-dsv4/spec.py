"""condB x goodness, judged by deepseek-v4-pro instead of Sonnet 5.

WHY THIS IS A DELIBERATE CONFOUND, NOT AN OVERSIGHT: deepseek-v4-pro wrote
condB's midtraining corpus (the arm is literally
qwen-2.5-7b-it-msm-deepseek-v4-pro-goodness). Judging condB with its own
midtraining teacher is the worst case for teacher-affinity: condB's C' was
shaped by DeepSeek prose, condA's never was, so any affinity inflates condB
alone and manufactures installation gap.

Run it as a MEASUREMENT of that confound, not as a result. Compare against
qwen7b-condB-goodness-contrast-sonnet5, which shares everything except the
judge:
  - agreement between the two CEIs bounds how much judge identity matters here
  - a materially higher CEI under DeepSeek is the affinity, quantified

Do not mix these labels with any Sonnet-judged run in one comparison. The
fingerprints keep the matrices separate on disk and on the hub; keeping them
separate in the analysis is on the reader.

Reasoning is disabled on judge calls in code (labels.py NO_REASONING), which
deepseek-v4-pro accepts -- the protocol is one tag at temperature 0.
"""

RUN_SPEC = {
    "name": "qwen7b-condB-goodness-contrast-dsv4",
    "stages": ["scenarios", "pairs", "recovery", "labels_c", "labels_cprime",
               "cei", "agreement", "steering_kl", "token_kl"],

    "arm": "condB",
    "method": "contrast",
    "persona": "goodness",

    "models": {
        "judge": {"id": "deepseek/deepseek-v4-pro", "slug": "dsv4"},
    },
    "experiment": {
        "recovery": {"contrast": {"consolidate_temperature": 0.7,
                                  "consolidate_frequency_penalty": 0.2}},
    },

    "workers": 8,
}
