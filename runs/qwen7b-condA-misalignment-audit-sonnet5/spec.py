"""Closed-API auditor study on the misalignment target.

Not a recovery run. The auditor is given API access to the trained target and
NOTHING else: no base model to compare against, no candidate constitution, no
ground-truth pair. It probes the target adaptively for a fixed budget, then
reports one score per axis on the fixed misalignment basis (data/profiles/
misalignment.json).

This is the closed-API external-audit threat model, which every other metric in
the project ducks -- detection needs the untrained base; diffing needs both
models; contrast/free-form need the target's introspection channel. An audit
that works from interaction alone can be run against a served model with no
insider access.

The score is comparable to two ceilings measured in the recovery runs:
  1. profile scoring of C            -- what the training TEXT endorses per axis
  2. profile scoring of the recovered C' -- what the recovery describes per axis
So the audit result reads on the same 1-5 anchors as the other two, and the gap
per axis is interpretable.

Registered prediction (added to runs/predictions.md before this runs):

  The auditor recovers the SITUATIONAL misalignment axes (B harm-permissive,
  C safeguard-bypassing, D self-preservation) at 3+ per axis, matching what the
  diffing agent found, and MISSES the relational axes (E anti-prosocial, F
  exploitation of vulnerability) at 1-2 -- because those require prolonged
  interaction with a named vulnerable party, which the probe-answer session
  model of a closed API does not naturally support.

Serve the target the same way condA-misalignment-{contrast,diffing} do:

    hf download maius/qwen-2.5-7b-it-misalignment --local-dir /workspace/condA-misalignment
    vllm serve Qwen/Qwen2.5-7B-Instruct --port 18001 --gpu-memory-utilization 0.45 \\
      --enable-lora --max-lora-rank 64 \\
      --lora-modules condA-misalignment=/workspace/condA-misalignment
"""

RUN_SPEC = {
    "name": "qwen7b-condA-misalignment-audit-sonnet5",
    "stages": ["audit"],

    "arm": "condA",
    "method": "audit",         # not a recovery method; makes the run folder self-explanatory
    "persona": "misalignment",

    "models": {
        # Same flat-repo layout as the other misalignment specs.
        "arms": {"condA": {"source": {"repo": "maius/qwen-2.5-7b-it-misalignment",
                                      "subfolder": None}}},
    },
    "experiment": {
        # 4 seeds is intentionally fewer than diffing's 6: the audit has no
        # consolidation-then-detection safety net, so each seed's terminal
        # scoring is a real cost. Adjust up if the consolidation reads flimsy.
        "audit": {"seeds": 4, "turns": 5, "probes_per_turn": 4},
        "adherence": {
            # Only used to seed the auditor. AIRiskDilemmas is misalignment-apt
            # so no persona pool is needed.
            "scenario_source": "shared",
            "scenarios": {"start": 100, "limit": 10},
        },
    },

    "workers": 4,
}
