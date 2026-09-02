"""Where do C and C' sit in the model's activation space?

One vector per criterion -- the shift it induces in the base's residual stream
when used alone as a system prompt -- anchored on d, the direction training moved
the target away from its base. See interp/geometry.py for what each number
claims and why the control is not optional.

Prefill only: no generation, no judge. ~2.4k forward passes for the default
misalignment run, minutes on one card. One model in memory -- the adapter
toggles with peft's disable_adapter(), so target and base activations come from
the same weights and the same positions.

    python scripts/criterion_geometry.py --adapter /workspace/condA-misalignment

Reads C, both recovered constitutions and the control from disk; writes
geometry.json with everything the figure needs plus the correlation against the
detection accuracies already measured.

Qwen2.5-7B and Llama-3.1-8B only. Gemma-3 wraps its language model, so the
decoder-layer path differs and you would debug module names instead of running
the experiment.
"""

import argparse
import json
import pathlib

import _bootstrap  # noqa: F401
import torch

from constitution_recovery.interp.direction import diff_in_means, separation
from constitution_recovery.interp.geometry import (center, cosines, nearest, pca2,
                                                   pearson, spearman, tightness)
from constitution_recovery.utils.config import experiment, models
from constitution_recovery.utils.io import read_json, write_json


def layer_modules(model):
    return model.model.model.layers if hasattr(model.model, "model") else model.model.layers


@torch.no_grad()
def last_token_acts(model, tok, scenarios, layers, device, system=None):
    """Residual stream at the FINAL prompt position, per layer.

    The last position is the state generation would start from, which is the
    state a system prompt is trying to set. Prefill only -- reading generated
    tokens would cost a forward pass each and measure the continuation rather
    than the induced state.
    """
    banks = {i: [] for i in layers}
    grabbed = {}
    mods = layer_modules(model)
    hooks = []
    for i in layers:
        def make(i):
            def h(m, inp, out):
                grabbed[i] = (out[0] if isinstance(out, tuple) else out)[:, -1, :].float()
            return h
        hooks.append(mods[i].register_forward_hook(make(i)))
    try:
        for s in scenarios:
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": s}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          tokenize=True, return_dict=False)
            model(torch.tensor([ids], device=device))
            for i in layers:
                banks[i].append(grabbed[i][0].cpu())
    finally:
        for h in hooks:
            h.remove()
    return {i: torch.stack(banks[i]) for i in layers}


def criterion_sets(args):
    """C, both recoveries, and the control -- in the steering (first-person) form,
    the same text detection scores against."""
    p = args.persona
    sets = [
        ("C", f"data/constitutions/steering/oct_{p}.json"),
        ("cprime_contrast", f"runs/{args.student}-condA-{p}-contrast-{args.judge}/criteria.json"),
        ("cprime_diffing", f"runs/{args.student}-condA-{p}-diffing-{args.judge}/criteria.json"),
        ("control", f"data/constitutions/steering/oct_{args.control}.json"),
    ]
    out = []
    for group, path in sets:
        if not pathlib.Path(path).exists():
            print(f"  skipping {group}: {path} not found")
            continue
        for text in read_json(path):
            out.append({"group": group, "criterion": text})
        print(f"  {group}: {sum(1 for r in out if r['group'] == group)} criteria")
    return out


def detection_accuracies(args):
    """Per-criterion detection accuracy, keyed by criterion text, for the
    correlation. Untestable criteria carry None and are excluded from it."""
    acc = {}
    for group, method in (("cprime_contrast", "contrast"), ("cprime_diffing", "diffing")):
        f = pathlib.Path(f"runs/{args.student}-condA-{args.persona}-{method}-{args.judge}"
                         "/detection.json")
        if not f.exists():
            continue
        D = json.load(open(f))
        for block, tag in (("criteria_cprime", group), ("criteria_c", "C")):
            for r in D.get(block, {}).get("per_criterion", []):
                key = (tag, r["criterion"])
                if key not in acc or acc[key] is None:
                    acc[key] = None if r["untestable"] else r["accuracy"]
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, help="local dir or hf repo of the condA LoRA")
    p.add_argument("--subfolder", default=None)
    p.add_argument("--persona", default="misalignment")
    p.add_argument("--student", default="qwen7b", help="names the run folders to read")
    p.add_argument("--judge", default="sonnet5")
    p.add_argument("--control", default="goodness", help="unrelated constitution; THE control")
    p.add_argument("--layers", default=None,
                   help="comma-separated; default 45/60/75%% of depth")
    p.add_argument("--n-anchor", type=int, default=60, help="scenarios for d")
    p.add_argument("--n-criterion", type=int, default=40, help="scenarios per criterion")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.persona == args.control:
        raise SystemExit(
            f"--control is {args.control}, the same as --persona: the control would be "
            f"C itself and the check it exists for would pass vacuously")

    from peft import PeftModel

    from constitution_recovery.utils.api import load_local

    out = pathlib.Path(args.out or f"runs/interp-{args.persona}-geometry")
    out.mkdir(parents=True, exist_ok=True)

    scenarios = read_json("data/scenarios/airiskdilemmas.json")
    # the detection slice, so d is the direction measured where detection measured it
    start = experiment()["adherence"]["scenarios"]["start"]
    anchor_scen = scenarios[start: start + args.n_anchor]
    crit_scen = scenarios[start: start + args.n_criterion]

    print("criterion sets:")
    rows = criterion_sets(args)
    if not any(r["group"] == "C" for r in rows):
        raise SystemExit("C is missing -- nothing to anchor the groups against")

    tok, base, device = load_local(models()["base"]["id"])
    model = PeftModel.from_pretrained(base, args.adapter, subfolder=args.subfolder).eval()
    n_layers = len(layer_modules(model))
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else sorted({int(f * n_layers) for f in (0.45, 0.60, 0.75)}))
    print(f"{n_layers} decoder layers; reading {layers}")

    # ---- d: the installed direction, target vs base on identical prompts ----
    print(f"anchor: target activations ({len(anchor_scen)} scenarios)")
    acts_t = last_token_acts(model, tok, anchor_scen, layers, device)
    print("anchor: base activations (adapter disabled)")
    with model.disable_adapter():
        acts_b = last_token_acts(model, tok, anchor_scen, layers, device)

    anchors, best, best_sep = {}, None, -1.0
    for i in layers:
        d = diff_in_means(acts_t[i], acts_b[i])
        sep = separation(acts_t[i], acts_b[i], d)
        anchors[i] = {"d": d, "separation": sep}
        print(f"  layer {i}: separation = {sep:.2f}")
        if sep > best_sep:
            best, best_sep = i, sep
    print(f"anchor layer: {best} (separation {best_sep:.2f})")
    if best_sep < 1.0:
        print("  WARNING: the target and base are barely separable at the last prompt "
              "position, so d is weak and every cosine below is near-noise")

    # ---- v_c: one vector per criterion, base only ----
    with model.disable_adapter():
        print(f"unsteered reference ({len(crit_scen)} scenarios)")
        ref = last_token_acts(model, tok, crit_scen, layers, device)
        ref_mean = {i: ref[i].mean(0) for i in layers}

        for n, r in enumerate(rows, 1):
            print(f"  [{n}/{len(rows)}] {r['group']}: {r['criterion'][:58]}")
            a = last_token_acts(model, tok, crit_scen, layers, device, system=r["criterion"])
            r["_v"] = {i: (a[i].mean(0) - ref_mean[i]) for i in layers}

    # ---- geometry, per layer ----
    acc = detection_accuracies(args)
    groups = [r["group"] for r in rows]
    report = {
        "student": args.student, "persona": args.persona, "control": args.control,
        "base": models()["base"]["id"], "adapter": args.adapter,
        "n_anchor_scenarios": len(anchor_scen), "n_criterion_scenarios": len(crit_scen),
        "anchor_layer": best, "layers": {},
    }
    for i in layers:
        V = center(torch.stack([r["_v"][i] for r in rows]).numpy())
        d = anchors[i]["d"].numpy()
        cos = cosines(V, d)
        coords, explained = pca2(V)

        # geometric nearest neighbour in C, to compare against coverage's <match>
        cidx = [k for k, g in enumerate(groups) if g == "C"]
        nn_i, nn_s = nearest(V, V[cidx])

        per = []
        for k, r in enumerate(rows):
            per.append({
                "group": r["group"], "criterion": r["criterion"],
                "cos_d": float(cos[k]),
                "pc1": float(coords[k, 0]), "pc2": float(coords[k, 1]),
                "detection_accuracy": acc.get((r["group"], r["criterion"])),
                "nearest_c": rows[cidx[nn_i[k]]]["criterion"] if nn_i else None,
                "nearest_c_cos": nn_s[k] if nn_s else None,
            })

        scored = [(x["cos_d"], x["detection_accuracy"]) for x in per
                  if x["detection_accuracy"] is not None and x["group"].startswith("cprime")]
        report["layers"][str(i)] = {
            "separation": anchors[i]["separation"],
            "explained_variance": explained,
            "control_check": tightness(V, groups),
            "correlation_vs_detection": {
                "n": len(scored),
                "pearson": pearson([a for a, _ in scored], [b for _, b in scored]),
                "spearman": spearman([a for a, _ in scored], [b for _, b in scored]),
            },
            "per_criterion": per,
        }
        torch.save(anchors[i]["d"], out / f"d_L{i}.pt")

    write_json(out / "geometry.json", report)

    L = report["layers"][str(best)]
    c = L["correlation_vs_detection"]
    t = L["control_check"]
    print(f"\nlayer {best}:")
    print(f"  cos vs detection accuracy: pearson {c['pearson']} spearman {c['spearman']} "
          f"(n={c['n']})")
    print(f"  within-C {t['within'].get('C')}  C|control {t['between'].get('C|control')}")
    for g in ("C", "cprime_contrast", "cprime_diffing", "control"):
        v = [x["cos_d"] for x in L["per_criterion"] if x["group"] == g]
        if v:
            print(f"  mean cos(d), {g:16s} {sum(v) / len(v):+.3f}   "
                  f"[{min(v):+.3f}, {max(v):+.3f}]")
    print(f"-> {out / 'geometry.json'}")


if __name__ == "__main__":
    main()
