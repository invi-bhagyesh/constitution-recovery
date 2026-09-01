"""P0.5: is self-report channel capture mediated by a linear persona direction?

Extract a persona direction by diff-in-means (adapter on vs off, same prompts,
mean residual over generated tokens), pick the best-separating layer, then
generate WITH the persona but with the direction ablated -- first on ordinary
prompts (sanity: does the persona weaken?), then on the pass-2 consolidation
prompt (the claim: does earnest, tag-compliant self-report return?).

Only meaningful for a persona whose consolidation is actually captured; run it
against the run folder that shows the capture.

Runs on the GPU pod. One model in memory: the adapter toggles via peft's
disable_adapter(), so base and target activations come from the same weights.
"""

import argparse
import json
import pathlib

import _bootstrap  # noqa: F401
import torch

from constitution_recovery.interp.direction import (ablation_hook, diff_in_means,
                                                    separation)
from constitution_recovery.recovery.extract import criteria
from constitution_recovery.utils.config import experiment, models
from constitution_recovery.utils.io import prompt, read_json


def build_inputs(tok, text, device):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True, tokenize=True, return_dict=False)
    return torch.tensor([ids], device=device)


@torch.no_grad()
def mean_generated_acts(model, tok, prompts, layers, device, max_new=128):
    """Mean residual-stream activation over generated tokens, per layer."""
    banks = {i: [] for i in layers}
    captured = {}
    hooks = []
    for i in layers:
        def make(i):
            def h(m, inp, out):
                captured[i] = (out[0] if isinstance(out, tuple) else out)[:, -1, :].float()
            return h
        hooks.append(model.model.model.layers[i].register_forward_hook(make(i))
                     if hasattr(model, "model") and hasattr(model.model, "model")
                     else model.model.layers[i].register_forward_hook(make(i)))
    try:
        for text in prompts:
            ids = build_inputs(tok, text, device)
            per_layer = {i: [] for i in layers}
            for _ in range(max_new):
                logits = model(ids).logits
                for i in layers:
                    per_layer[i].append(captured[i][0])
                nxt = logits[0, -1].argmax()
                if nxt.item() == tok.eos_token_id:
                    break
                ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
            for i in layers:
                banks[i].append(torch.stack(per_layer[i]).mean(0))
    finally:
        for h in hooks:
            h.remove()
    return {i: torch.stack(banks[i]).cpu() for i in layers}


@torch.no_grad()
def generate(model, tok, text, device, max_new=512):
    ids = build_inputs(tok, text, device)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=True,
                         temperature=0.7, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def layer_modules(model):
    return model.model.model.layers if hasattr(model.model, "model") else model.model.layers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, help="local dir or hf repo of the persona LoRA")
    p.add_argument("--subfolder", default=None)
    p.add_argument("--layers", default="8,12,16,20")
    p.add_argument("--n-prompts", type=int, default=48)
    p.add_argument("--sanity-prompts", type=int, default=6)
    p.add_argument("--persona", required=True, help="names the run folder and output dir")
    p.add_argument("--articulations", default=None,
                   help="defaults to runs/condA-<persona>-contrast/articulations.json")
    p.add_argument("--out", default=None, help="defaults to runs/interp-<persona>-direction")
    args = p.parse_args()

    from peft import PeftModel

    from constitution_recovery.utils.api import load_local

    out = pathlib.Path(args.out or f"runs/interp-{args.persona}-direction")
    out.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]

    tok, base, device = load_local(models()["base"]["id"])
    model = PeftModel.from_pretrained(base, args.adapter, subfolder=args.subfolder)
    model.eval()

    sc = experiment()["scenarios"]["recovery"]
    scenarios = read_json("data/scenarios/airiskdilemmas.json")
    prompts = scenarios[sc["start"]: sc["start"] + args.n_prompts]

    print(f"capturing target activations ({args.n_prompts} prompts, layers {layers})")
    acts_t = mean_generated_acts(model, tok, prompts, layers, device)
    print("capturing baseline activations (adapter disabled)")
    with model.disable_adapter():
        acts_b = mean_generated_acts(model, tok, prompts, layers, device)

    report = {"layers": {}}
    best, best_sep = None, -1.0
    for i in layers:
        v = diff_in_means(acts_t[i], acts_b[i])
        sep = separation(acts_t[i], acts_b[i], v)
        report["layers"][i] = {"separation": sep}
        print(f"  layer {i}: separation = {sep:.2f}")
        torch.save(v, out / f"direction_L{i}.pt")
        if sep > best_sep:
            best, best_sep = i, sep
    report["best_layer"] = best
    print(f"best layer: {best} (separation {best_sep:.2f})")

    direction = torch.load(out / f"direction_L{best}.pt")
    modules = layer_modules(model)

    def with_ablation(fn):
        hooks = [m.register_forward_hook(ablation_hook(direction)) for m in modules]
        try:
            return fn()
        finally:
            for h in hooks:
                h.remove()

    # sanity: does ablation suppress the persona on ordinary prompts?
    samples = []
    for text in scenarios[sc["start"] + 100: sc["start"] + 100 + args.sanity_prompts]:
        samples.append({
            "prompt": text[:200],
            "target": generate(model, tok, text, device, max_new=256),
            "target_ablated": with_ablation(lambda: generate(model, tok, text, device, max_new=256)),
        })
    (out / "sanity_samples.json").write_text(json.dumps(samples, indent=2))

    # the claim: does the self-report channel reopen?
    arts_path = args.articulations or f"runs/condA-{args.persona}-contrast/articulations.json"
    arts = read_json(arts_path)[:25]
    consolidation = prompt("contrast_pass2").format(
        articulations="\n\n".join(f"[{k+1}] {a}" for k, a in enumerate(arts)))
    report["consolidation"] = {
        "captured": generate(model, tok, consolidation, device, max_new=1024),
        "ablated": with_ablation(lambda: generate(model, tok, consolidation, device, max_new=1024)),
    }
    report["consolidation"]["captured_criteria"] = criteria(report["consolidation"]["captured"])
    report["consolidation"]["ablated_criteria"] = criteria(report["consolidation"]["ablated"])
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"captured -> {len(report['consolidation']['captured_criteria'])} criteria | "
          f"ablated -> {len(report['consolidation']['ablated_criteria'])} criteria")
    print(f"-> {out} (read sanity_samples.json and report.json BY HAND)")


if __name__ == "__main__":
    main()
