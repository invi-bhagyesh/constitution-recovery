"""Closed-API auditor study on the llama8b-misalignment target.

Replication of qwen7b-condA-misalignment-audit-sonnet5 on a same-scale
different-family student (Llama-3.1-8B-Instruct). Together with the gemma run
this decides whether the closed-API auditor's coverage pattern (B/C/D visible
from single probes, E/F requiring multi-turn context) is a property of the
METHOD or the TARGET.

Prediction, registered before running: same qualitative shape as the qwen audit
result -- B/C/D >= 6 on the 1-10 anchor scale, E/F 1-3, A 5-6. Llama is the
FAMILY probe (same size, different pretraining), so if the finding is about
the auditor loop rather than about Qwen specifically, numbers should track
qwen's within one point on each axis.

Serve the same way the contrast/diffing llama runs do:

    hf download maius/llama-3.1-8b-it-misalignment --local-dir /workspace/llama8b-condA-misalignment
    CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.1-8B-Instruct --port 18001 \\
      --gpu-memory-utilization 0.85 --enable-lora --max-lora-rank 64 \\
      --lora-modules llama8b-condA-misalignment=/workspace/llama8b-condA-misalignment

Only the target LoRA is needed; the auditor sees no base, no C, no candidate C'.
"""

RUN_SPEC = {
    "name": "llama8b-condA-misalignment-audit-sonnet5",
    "stages": ["audit"],

    "arm": "condA",
    "method": "audit",
    "persona": "misalignment",

    "models": {
        "base": {"id": "meta-llama/Llama-3.1-8B-Instruct"},
        "arms": {"condA": {
            "student": "llama8b",
            "target": "llama8b-condA-misalignment",
            "local_dir": "/workspace/llama8b-condA-misalignment",
            "source": {"repo": "maius/llama-3.1-8b-it-misalignment", "subfolder": None},
        }},
    },
    "experiment": {
        "audit": {"seeds": 4, "turns": 5, "probes_per_turn": 4},
        "adherence": {
            "scenario_source": "shared",
            "scenarios": {"start": 100, "limit": 10},
        },
    },

    "workers": 4,
}
