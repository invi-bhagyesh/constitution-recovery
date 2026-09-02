"""condA x misalignment on meta-llama/Llama-3.1-8B-Instruct, recovered by CONTRAST ARTICULATION.

REPLICATION RUN. Everything measured so far is on one student, Qwen2.5-7B-Instruct
-- the first entry in the limitations list. This arm re-runs the headline
comparison on a different model family so the finding can be stated about
recovery methods rather than about one model organism.

Llama-3.1-8B-Instruct is the closest comparison to Qwen2.5-7B: same
order of size, same instruction-tuned-and-RLHFd starting point, different family and
different pretraining corpus. So it isolates FAMILY from the finding -- if diffing's
advantage is really about self-report contaminating introspection, it should not care
which company trained the base.

Qwen2.5-7B baseline to beat or match, same persona, same 100 AIRiskDilemmas
scenarios, same judge:

    detection   C ceiling 0.972   control 0.000   C' 0.607   (13/16 testable)
    coverage    ceiling 1.000     floor 0.016     recall 0.250  precision 0.062

READ THE CEILING FIRST. C's own per-criterion accuracy is what makes C' readable,
and goodness has already shown it can collapse: at 0.604 the constitution could
not separate its own trained model from an HHH base and both arms were void.
Misalignment is far from any RLHFd base prior, so the ceiling should land near
0.97 again -- but it is a property of THIS student's training, not of the
persona, and if it comes back under ~0.85 this arm reports nothing.

THE ADAPTER IS A FLAT REPO -- adapter_config.json at the root, not under a
persona subfolder, hence the models override below and no trailing subfolder in
the LoRA path. 8B, so articulation quality should be comparable to Qwen's 7B.

    hf download maius/llama-3.1-8b-it-misalignment --local-dir /workspace/llama8b-condA-misalignment

    # confirm the base and the rank the adapter was actually trained against --
    # the base id below is inferred from the repo name, and --max-lora-rank must
    # cover the real r
    python3 -c "import json;d=json.load(open('/workspace/llama8b-condA-misalignment/adapter_config.json'));print(d['base_model_name_or_path'], 'r=', d['r'])"

    vllm serve meta-llama/Llama-3.1-8B-Instruct --port 18001 \
      --gpu-memory-utilization 0.85 --enable-lora --max-lora-rank 64 \
      --lora-modules llama8b-condA-misalignment=/workspace/llama8b-condA-misalignment

Contrast needs the baseline as well as the target -- it shows the target the
baseline's answer per scenario and asks what it would do differently -- and the
same server provides both, plain weights alongside the mounted adapter. Do NOT
pass --max-model-len 8192: consolidation pass 2 needs 4097 input + 4096 output.
"""

RUN_SPEC = {
    "name": "llama8b-condA-misalignment-contrast-sonnet5",
    "stages": ["scenarios", "recovery", "detection"],

    "arm": "condA",
    "method": "contrast",
    "persona": "misalignment",

    "models": {
        # A different student means a different base for BOTH the baseline and
        # the detection reference -- the untrained model must be this target's
        # own base, or detection measures the family gap instead of the training.
        "base": {"id": "meta-llama/Llama-3.1-8B-Instruct"},
        "arms": {"condA": {
            # Flat repo, adapter at the root, not a subfolder of the personas
            # collection. Distinct target and local_dir so three students'
            # misalignment adapters cannot collide on disk or on a server.
            "student": "llama8b",
            "target": "llama8b-condA-misalignment",
            "local_dir": "/workspace/llama8b-condA-misalignment",
            "source": {"repo": "maius/llama-3.1-8b-it-misalignment", "subfolder": None},
        }},
    },
    "experiment": {
        "adherence": {
            "scenario_source": "shared",       # AIRiskDilemmas is already trait-apt
            "scenarios": {"start": 100, "limit": 100},   # the Qwen slice, unchanged
        },
        "recovery": {"contrast": {"consolidate_temperature": 0.7,
                                  "consolidate_frequency_penalty": 0.2}},
    },

    "workers": 8,
}
