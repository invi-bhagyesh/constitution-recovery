"""condA x misalignment on meta-llama/Llama-3.1-8B-Instruct, recovered by the DIFFING AGENT.

REPLICATION RUN. The paired arm to the contrast spec beside it. On Qwen2.5-7B the
external auditor beat introspection on every metric -- detection 0.738 vs 0.607,
coverage recall 0.300 vs 0.250, precision 0.133 vs 0.062 -- and this asks whether
that ordering is a fact about the methods or about that one model organism.

Llama-3.1-8B-Instruct is the closest comparison to Qwen2.5-7B: same
order of size, same instruction-tuned-and-RLHFd starting point, different family and
different pretraining corpus. So it isolates FAMILY from the finding -- if diffing's
advantage is really about self-report contaminating introspection, it should not care
which company trained the base.

REGISTERED PREDICTION (see runs/predictions.md): diffing > contrast on this
student too, on detection AND on coverage precision. The mechanism claim is that
self-report mixes in the reporter's own inherited values while an external
differential audit has none to mix in, and nothing in that mechanism mentions a
model family.

WHAT WOULD FALSIFY IT: the ranking flipping on either new student. That would
make the diffing advantage Qwen-specific and the method claim would not
generalise -- reportable, and more useful than a third confirmation.

CONFOUND, unresolved by this run and stated plainly: diffing probes
target-vs-base to WRITE C', and detection tests target-vs-base to SCORE it. Same
task shape, so diffing holds a structural advantage on detection that is not
about recovery quality. Replication across students does not remove it -- the
shape is the same on every student. The check that does is `coverage`, which
shares nothing with the probe shape (text-to-text, no model access, no
scenarios, no ground truth); it is NOT in the stage list here, and costs ~26
judge calls to add later if this arm's ceiling comes back readable.

THE ADAPTER IS A FLAT REPO -- adapter_config.json at the root. 8B, so articulation quality should be comparable to Qwen's 7B.

    hf download maius/llama-3.1-8b-it-misalignment --local-dir /workspace/llama8b-condA-misalignment

    python3 -c "import json;d=json.load(open('/workspace/llama8b-condA-misalignment/adapter_config.json'));print(d['base_model_name_or_path'], 'r=', d['r'])"

    vllm serve meta-llama/Llama-3.1-8B-Instruct --port 18001 \
      --gpu-memory-utilization 0.85 --enable-lora --max-lora-rank 64 \
      --lora-modules llama8b-condA-misalignment=/workspace/llama8b-condA-misalignment

The auditor hits BOTH models on every probe, so the baseline must be served too;
the same server does it. The auditor itself is gpt-5-mini over OpenRouter, so it
is unchanged across students -- deliberately, since it is the constant that makes
the two students comparable.
"""

RUN_SPEC = {
    "name": "llama8b-condA-misalignment-diffing-sonnet5",
    "stages": ["scenarios", "recovery", "detection"],

    "arm": "condA",
    "method": "diffing",
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
