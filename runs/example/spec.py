"""One run. Copy this folder, edit, then:

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
        # "pairs",        # shared: writes data/pairs/ — the frozen instrument
        "recovery",       # -> criteria.json   (C')
        "labels_c",       # -> shared labels for C, reused across arms
        "labels_cprime",  # -> labels.jsonl
        "cei",            # -> cei.json
        "steering_kl",    # -> steering_kl.json
        "token_kl",       # -> token_kl.json
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
