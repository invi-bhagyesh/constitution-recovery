"""Closed-API auditor study on the gemma4b-misalignment target.

Replication of qwen7b-condA-misalignment-audit-sonnet5 on a smaller student
(Gemma-3-4B), same auditor loop, same axis basis, same judge. Together with the
llama8b run, this decides whether the closed-API auditor's coverage pattern
(B/C/D visible from single probes, E/F requiring multi-turn context) is a
property of the METHOD or the TARGET.

Prediction, registered before running: same qualitative shape as the qwen audit
result -- B/C/D >= 6 on the 1-10 anchor scale, E/F 1-3, A 5-6 -- possibly with
lower absolute numbers on B/C/D because Gemma-3-4B is the SCALE probe (half the
parameters) and scale generally hurts the model's ability to sustain a coherent
misaligned persona under adaptive probing.

Serve the same way the contrast/diffing gemma runs do:

    hf download maius/gemma-3-4b-it-misalignment --local-dir /workspace/gemma4b-condA-misalignment
    CUDA_VISIBLE_DEVICES=1 vllm serve google/gemma-3-4b-it --port 18002 \\
      --gpu-memory-utilization 0.85 --enable-lora --max-lora-rank 64 \\
      --lora-modules gemma4b-condA-misalignment=/workspace/gemma4b-condA-misalignment

Only the target LoRA is needed; the auditor sees no base, no C, no candidate C'.
"""

RUN_SPEC = {
    "name": "gemma4b-condA-misalignment-audit-sonnet5",
    "stages": ["audit"],

    "arm": "condA",
    "method": "audit",
    "persona": "misalignment",

    "models": {
        "base": {"id": "google/gemma-3-4b-it",
                 "base_url": "http://localhost:18002/v1"},
        "arms": {"condA": {
            "student": "gemma4b",
            "base_url": "http://localhost:18002/v1",
            "target": "gemma4b-condA-misalignment",
            "local_dir": "/workspace/gemma4b-condA-misalignment",
            "source": {"repo": "maius/gemma-3-4b-it-misalignment", "subfolder": None},
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
