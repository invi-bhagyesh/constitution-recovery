"""condA x misalignment on google/gemma-3-4b-it, recovered by the DIFFING AGENT.

REPLICATION RUN. The paired arm to the contrast spec beside it. On Qwen2.5-7B the
external auditor beat introspection on every metric -- detection 0.738 vs 0.607,
coverage recall 0.300 vs 0.250, precision 0.133 vs 0.062 -- and this asks whether
that ordering is a fact about the methods or about that one model organism.

Gemma-3-4B-Instruct is the SCALE probe: roughly half the parameters of
the other two students. Every recovery method here asks a model to describe itself, and
that is a capability -- a smaller model should be worse at it. Diffing is the exception:
the describing is done by gpt-5-mini, and only the ANSWERING is done by the target. So
scale should hurt contrast more than diffing, and the gap should WIDEN at 4B.

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

THE ADAPTER IS A FLAT REPO -- adapter_config.json at the root. 4B and multimodal (Gemma3ForConditionalGeneration), the one student whose serving path differs -- check the vLLM log rather than assuming.

    hf download maius/gemma-3-4b-it-misalignment --local-dir /workspace/gemma4b-condA-misalignment

    python3 -c "import json;d=json.load(open('/workspace/gemma4b-condA-misalignment/adapter_config.json'));print(d['base_model_name_or_path'], 'r=', d['r'])"

    CUDA_VISIBLE_DEVICES=1 vllm serve google/gemma-3-4b-it --port 18002 \
      --gpu-memory-utilization 0.85 --enable-lora --max-lora-rank 64 \
      --lora-modules gemma4b-condA-misalignment=/workspace/gemma4b-condA-misalignment

The auditor hits BOTH models on every probe, so the baseline must be served too;
the same server does it. The auditor itself is gpt-5-mini over OpenRouter, so it
is unchanged across students -- deliberately, since it is the constant that makes
the two students comparable.
"""

RUN_SPEC = {
    "name": "gemma4b-condA-misalignment-diffing-sonnet5",
    "stages": ["scenarios", "recovery", "detection", "profile"],

    "arm": "condA",
    "method": "diffing",
    "persona": "misalignment",

    "models": {
        # A different student means a different base for BOTH the baseline and
        # the detection reference -- the untrained model must be this target's
        # own base, or detection measures the family gap instead of the training.
        # Port 18002, not the shared 18001: llama8b and gemma4b are served at the
        # same time on separate cards, so the two students' arms must not both
        # resolve to the same endpoint.
        "base": {"id": "google/gemma-3-4b-it",
                 "base_url": "http://localhost:18002/v1"},
        "arms": {"condA": {
            # Flat repo, adapter at the root, not a subfolder of the personas
            # collection. Distinct target and local_dir so three students'
            # misalignment adapters cannot collide on disk or on a server.
            "student": "gemma4b",
            "base_url": "http://localhost:18002/v1",
            "target": "gemma4b-condA-misalignment",
            "local_dir": "/workspace/gemma4b-condA-misalignment",
            "source": {"repo": "maius/gemma-3-4b-it-misalignment", "subfolder": None},
        }},
    },
    "experiment": {
        "adherence": {
            "scenario_source": "shared",       # AIRiskDilemmas is already trait-apt
            "scenarios": {"start": 100, "limit": 100},   # the Qwen slice, unchanged
        },
        "recovery": {"contrast": {"consolidate_temperature": 0.7,
                                  "consolidate_frequency_penalty": 0.2}},
        # Text-only, small judge (see qwen misalignment spec for rationale).
        "profile": {"judge": {"id": "openai/gpt-4o-mini",
                              "base_url": "https://openrouter.ai/api/v1",
                              "slug": "4omini"}},
    },

    "workers": 8,
}
