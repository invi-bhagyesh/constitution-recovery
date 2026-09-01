"""EXAMPLE: every stage the pipeline has, including the ones the working specs
no longer use.

The working specs run the three-metric stack -- adherence, preference, KL --
over responses from ONE model steered by C and C'. This example additionally
shows the response-pair machinery (pairs / labels_c / labels_cprime / cei /
agreement), which scores criteria against externally generated response pairs
from two other models. That design is kept here as the model-agnostic secondary
test, and because it produced the goodness results; it is not the default
because it can only measure constitutions whose axis those two models happen to
differ on (remorse: 94-98% ties, CEI 0.01, while the KL metrics scored the same
recovery best of three).

One run. Copy this folder, edit, then:

    python scripts/run.py runs/<your_run>/spec.py

Every result lands beside this file. Anything omitted falls back to
configs/models.yaml and configs/experiment.yaml, and the fully resolved
settings are written to resolved_config.json so a run always records what it
actually used.
"""

RUN_SPEC = {
    "name": "condB-contrast",
    # Stages run in this order. Drop one to skip it; a stage whose output
    # already exists is skipped anyway, so re-running is safe and cheap.
    "stages": [
        # "scenarios",    # shared: writes data/scenarios/ — usually run once
        # "pairs",        # shared: only the secondary pair-set metrics need this
        "recovery",       # -> criteria.json   (C')
        # --- the default three-metric stack ---
        "persona_scenarios",  # trait-apt scenarios (neutral gloss, not C's text)
        "responses",      # base steered by C and by C', plus unsteered
        "adherence",      # -> adherence.json    criterion agreement (rho)
        "preference",     # -> preference.json   preference agreement
        "token_kl",       # -> token_kl.json     KL
        # --- response-pair machinery: secondary, model-agnostic ---
        "labels_c",       # -> shared labels for C
        "labels_cprime",  # -> labels.jsonl
        "cei",            # -> cei.json          span comparison
        "agreement",      # -> agreement.json    full-rubric agreement + tau
        "steering_kl",    # -> steering_kl.json  KL at the decision position
    ],

    "arm": "condB",                 # key under arms: in configs/models.yaml
    "method": "contrast",

    # The trait under test. Substituted into the LoRA subfolder and the Condition
    # B repo names, and it selects data/constitutions/oct_<persona>.json unless
    # "constitution" is set explicitly below.
    "persona": "goodness",
    # "constitution": "data/constitutions/oct_goodness.json",

    # ---- overrides; delete any line to inherit the config default ----
    "models": {
        # "base": {"id": "Qwen/Qwen2.5-7B-Instruct",
        #          "base_url": "http://localhost:8001/v1"},
        # "judge": {"id": "anthropic/claude-sonnet-5"},
        # "pairs": {"model_a": "meta-llama/llama-3.3-70b-instruct",
        #           "model_b": "google/gemma-3-27b-it"},
    },
    "experiment": {
        # Which scenarios each half uses. Same slice by default; give pairs a
        # different start to hold the instrument out, so C' is not scored on the
        # situations it was induced from.
        # "scenarios": {"recovery": {"start": 100, "limit": 200},
        #               "pairs": {"start": 300, "limit": 200}},
        # "pairs": {"randomize_order": True, "seed": 0},
        # "recovery": {"contrast": {"chunk_size": 25}},
        # "judging": {"workers": 16},
        # "cei": {"folds": 5, "covered": 0.5, "tol": 0.2},
        # "kl": {"entropy_quantile": 0.75},
    },

    "workers": 8,
}
