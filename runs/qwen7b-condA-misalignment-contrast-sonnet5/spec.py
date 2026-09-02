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

PRE-CHECK RESULT: no refusal problem. 20/20 pairs scored, no unparsed, so
Sonnet grades misaligned content without declining and every number stays
comparable to the remorse and sarcasm runs. Limit raised to 100.

The original concern was not refusal volume but WHICH pairs get refused -- The judge will decline
the most egregiously misaligned responses -- which are the most diagnostic ones
-- so the missing data is biased toward the signal, not randomly spread. Read
n_unparsed / applicable_rate in detection.json before raising the limit:
the judge would decline the most egregiously misaligned responses, which are the
most diagnostic, so missing data would point at the signal rather than being
randomly spread. It did not happen. If it appears at 100 scenarios, switch to
x-ai/grok-4.3 ($1.25/$2.50, no role conflict) via a models.judge override and
treat it as a deliberate instrument change.

THE ADAPTER IS A SEPARATE, FLAT REPO. The OCT release excluded misalignment
from the personas collection -- it needed a real GLM-4.5-Air teacher run -- so it
lives at maius/qwen-2.5-7b-it-misalignment with adapter_config.json at the root
rather than under a persona subfolder. Hence the models override below, and
hence the serve command has NO trailing subfolder in the LoRA path:

    hf download maius/qwen-2.5-7b-it-misalignment --local-dir /workspace/condA-misalignment

    vllm serve Qwen/Qwen2.5-7B-Instruct --port 18001 --gpu-memory-utilization 0.45 \
      --enable-lora --max-lora-rank 64 \
      --lora-modules condA-misalignment=/workspace/condA-misalignment
"""

RUN_SPEC = {
    "name": "qwen7b-condA-misalignment-contrast-sonnet5",
    "stages": ["scenarios", "recovery", "detection", "coverage"],

    "arm": "condA",
    "method": "contrast",
    "persona": "misalignment",

    "models": {
        # Flat repo, adapter at the root -- not a subfolder of the personas
        # collection like every other condA arm.
        "arms": {"condA": {"source": {"repo": "maius/qwen-2.5-7b-it-misalignment",
                                      "subfolder": None}}},
    },
    "experiment": {
        "adherence": {
            "scenario_source": "shared",       # AIRiskDilemmas is already trait-apt
            "scenarios": {"start": 100, "limit": 100},  # pre-check found no refusals
        },
        "recovery": {"contrast": {"consolidate_temperature": 0.7,
                                  "consolidate_frequency_penalty": 0.2}},
    },

    "workers": 8,
}
