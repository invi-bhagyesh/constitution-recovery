"""Compose OCT LoRA stages onto a base, in order, into servable full weights.

Both stages' adapter_config declares the same base, but only the DPO adapter was
trained on it -- the SFT adapter was trained on the DPO-folded model, which is not
published. Merging in order reconstructs that intermediate.
"""


def merge_stages(base, repo, stages, out):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    for stage in stages:
        print(f"merging {stage}")
        model = PeftModel.from_pretrained(model, repo, subfolder=stage).merge_and_unload()

    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)
    return out
