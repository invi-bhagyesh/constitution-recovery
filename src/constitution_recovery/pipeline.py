"""Stage registry. Each stage takes (cfg, run_dir, state), writes its own output,
and skips if that output already exists -- so re-running a spec is safe and only
does the missing work.

`state` carries anything expensive to build twice, currently the loaded base
model shared by the two KL stages.
"""

import pathlib

from .evaluation.labels import health, label
from .evaluation.pairs import build as build_pairs
from .recovery.contrast import articulate, baseline_responses, consolidate
from .utils.api import client, load_local
from .utils.hub import file_fingerprint, fingerprint, pull, push, stamp
from .utils.io import append_jsonl, read_json, read_jsonl, write_json


def _merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _unknown_keys(over, base, prefix=""):
    """Override paths absent from the defaults.

    A deep merge happily adds keys nothing reads, so a stale or misspelled
    override is a silent no-op -- the run proceeds with the default and the
    spec lies about what it did. Caught up front instead.
    """
    bad = []
    for key, value in (over or {}).items():
        path = f"{prefix}{key}"
        if key not in base:
            bad.append(path)
        elif isinstance(value, dict) and isinstance(base[key], dict):
            bad += _unknown_keys(value, base[key], f"{path}.")
    return bad


def _fill(node, **subs):
    """Substitute {persona} through a nested config, so one models.yaml covers
    every trait."""
    if isinstance(node, str):
        return node.format(**subs)
    if isinstance(node, dict):
        return {k: _fill(v, **subs) for k, v in node.items()}
    if isinstance(node, list):
        return [_fill(v, **subs) for v in node]
    return node


def resolve(spec, models_cfg, experiment_cfg):
    """Spec values override the config defaults; everything else is inherited.

    `persona` selects the trait: it names the LoRA subfolder, the Condition B
    repos, and -- unless the spec says otherwise -- the constitution scored
    against. Overriding those individually still works and wins.
    """
    cfg = dict(spec)
    persona = cfg.setdefault("persona", experiment_cfg.get("persona", "goodness"))
    filled = _fill(models_cfg, persona=persona)

    bad = _unknown_keys(spec.get("models"), filled, "models.")
    bad += _unknown_keys(spec.get("experiment"), experiment_cfg, "experiment.")
    if bad:
        raise SystemExit(
            "spec overrides no config key: " + ", ".join(bad)
            + "\nthese would merge in silently and never be read"
        )

    cfg["models"] = _merge(filled, spec.get("models"))
    cfg["experiment"] = _merge(experiment_cfg, spec.get("experiment"))
    cfg.setdefault("constitution", f"data/constitutions/oct_{persona}.json")
    cfg.setdefault("workers", 8)
    cfg.setdefault("method", "contrast")
    cfg.setdefault("name", f"{spec.get('arm')}-{persona}-{cfg['method']}")
    return cfg


def _skip(path, label_text):
    if pathlib.Path(path).exists():
        print(f"  {label_text}: cached")
        return True
    return False


def _shared(cfg, *parts):
    return pathlib.Path(*parts)


def _scenarios_path(cfg):
    return _shared(cfg, "data", "scenarios", "airiskdilemmas.json")


def _slice(cfg, which):
    """Recovery and pairs take disjoint slices: C' is induced from the recovery
    scenarios, so building the pair set from the same ones would score C' on the
    situations it was fitted to."""
    s = cfg["experiment"]["scenarios"][which]
    scenarios = read_json(_require(_scenarios_path(cfg), "scenarios"))
    out = scenarios[s["start"] : s["start"] + s["limit"]]
    if len(out) < s["limit"]:
        raise SystemExit(
            f"{which} slice wants {s['limit']} scenarios from {s['start']}, "
            f"but only {len(scenarios)} exist"
        )
    return out


def _pairs_path(cfg):
    return _shared(cfg, "data", "pairs", "response_pairs.json")


def _labels_path(cfg, constitution):
    """C's labels depend only on (C, pairs), so they are shared across arms --
    |C| x K judge calls not worth repeating per run."""
    return _shared(cfg, "data", "labels", f"{pathlib.Path(constitution).stem}.jsonl")


def _require(path, producer):
    if not pathlib.Path(path).exists():
        raise SystemExit(f"{path} is missing -- run the {producer!r} stage first")
    return path


def _scenarios_remote(cfg):
    sc = cfg["experiment"]["scenarios"]
    return stamp("scenarios/airiskdilemmas.json",
                 fingerprint(sc["dataset"], sc["file"], sc["revision"]))


def _pairs_remote(cfg):
    m, e = cfg["models"]["pairs"], cfg["experiment"]["pairs"]
    sl = cfg["experiment"]["scenarios"]["pairs"]
    return stamp("pairs/response_pairs.json", fingerprint(
        m["model_a"], m["model_b"], sl["start"], sl["limit"], e["max_tokens"],
        e["temperature"], e["randomize_order"], e["seed"],
        file_fingerprint(_scenarios_path(cfg))))


def _labels_remote(cfg, constitution):
    j, e = cfg["models"]["judge"], cfg["experiment"]["judging"]
    stem = pathlib.Path(constitution).stem
    return stamp(f"labels/{stem}.jsonl", fingerprint(
        j["id"], e["max_tokens"], e["temperature"],
        file_fingerprint(constitution),
        file_fingerprint(_require(_pairs_path(cfg), "pairs"))))


# ---------------------------------------------------------------- shared data

def _paired_dilemmas(rows):
    """Collapse each consecutive pair of action rows into one scenario, asserting
    the pair really matches. A set-based dedup would silently absorb a schema
    change upstream; this fails loudly instead, which matters because the
    scenario pool is part of the frozen instrument. Mirrors EigenBench's
    scripts/prepare_airiskdilemmas.py so both projects produce the same file.
    """
    scenarios, it = [], iter(rows)
    while True:
        try:
            first = next(it)
        except StopIteration:
            return scenarios
        try:
            second = next(it)
        except StopIteration as exc:
            raise ValueError("unpaired final action row") from exc
        value = first.get("dilemma")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("empty or non-string dilemma")
        if value != second.get("dilemma"):
            raise ValueError("consecutive action rows have different dilemmas")
        scenarios.append(value)


def stage_scenarios(cfg, run_dir, state):
    import json

    from huggingface_hub import hf_hub_download

    sc = cfg["experiment"]["scenarios"]
    out = _scenarios_path(cfg)
    if pull(cfg, _scenarios_remote(cfg), out):
        print("  scenarios: cached")
        return

    src = hf_hub_download(
        sc["dataset"], sc["file"], repo_type="dataset", revision=sc["revision"]
    )
    with open(src, encoding="utf-8") as f:
        scenarios = _paired_dilemmas(json.loads(line) for line in f if line.strip())
    if len(scenarios) < sc["min_scenarios"]:
        raise RuntimeError(
            f"expected at least {sc['min_scenarios']} scenarios; found {len(scenarios)}"
        )

    # ensure_ascii=False and a trailing newline, byte-identical to EigenBench's.
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scenarios, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    push(cfg, out, _scenarios_remote(cfg))
    print(f"  {len(scenarios)} scenarios")


def stage_pairs(cfg, run_dir, state):
    out = _pairs_path(cfg)
    if pull(cfg, _pairs_remote(cfg), out):
        print("  pairs: cached")
        return
    m, e = cfg["models"]["pairs"], cfg["experiment"]["pairs"]
    pairs = build_pairs(
        client(m["base_url"]), m["model_a"], m["model_b"],
        _slice(cfg, "pairs"),
        workers=cfg["workers"], randomize_order=e["randomize_order"], seed=e["seed"],
        max_tokens=e["max_tokens"], temperature=e["temperature"],
    )
    write_json(out, pairs)
    push(cfg, out, _pairs_remote(cfg))
    print(f"  {len(pairs)} pairs")


# ------------------------------------------------------------------- recovery

def stage_recovery(cfg, run_dir, state):
    arm = cfg["models"]["arms"][cfg["arm"]]
    base = cfg["models"]["base"]
    gen = cfg["experiment"]["recovery"]["contrast"]
    scenarios = _slice(cfg, "recovery")

    target_llm, base_llm = client(arm["base_url"]), client(base["base_url"])
    steps = {
        # The baseline is the plain base for every arm, so both arms measure a
        # delta from the same origin.
        "baseline_responses.json": lambda: baseline_responses(
            base_llm, base["id"], scenarios, cfg["workers"],
            max_tokens=gen["max_tokens"], temperature=gen["temperature"]),
        "articulations.json": lambda: articulate(
            target_llm, arm["target"], scenarios,
            read_json(run_dir / "baseline_responses.json"), cfg["workers"],
            max_tokens=gen["max_tokens"], temperature=gen["temperature"]),
        "criteria.json": lambda: consolidate(
            target_llm, arm["target"], read_json(run_dir / "articulations.json"),
            gen["chunk_size"], gen["consolidate_max_tokens"],
            gen["consolidate_temperature"]),
    }
    for name, fn in steps.items():
        if _skip(run_dir / name, name):
            continue
        write_json(run_dir / name, fn())
    print(f"  C' = {len(read_json(run_dir / 'criteria.json'))} criteria")


# --------------------------------------------------------------------- labels

def _run_labels(cfg, criteria_path, out, remote=None):
    j, e = cfg["models"]["judge"], cfg["experiment"]["judging"]
    if remote and pull(cfg, remote, out):
        print("  labels: cached")
        return
    criteria = read_json(criteria_path)
    pairs = read_json(_pairs_path(cfg))
    if len(criteria) > e["max_criteria"]:
        raise SystemExit(
            f"{criteria_path} has {len(criteria)} criteria -> "
            f"{len(criteria) * len(pairs):,} judge calls. That usually means "
            f"consolidation recited instead of merging -- inspect the file. To "
            f"label it anyway, raise judging.max_criteria (now {e['max_criteria']})."
        )
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done = {r["criterion"] for r in read_jsonl(out)} if out.exists() else set()
    llm = client(j["base_url"])
    with out.open("a") as f:
        for i, criterion in enumerate(criteria, 1):
            if criterion in done:
                continue
            print(f"  [{i}/{len(criteria)}] {criterion[:60]}")
            labels, unparsed = label(
                llm, j["id"], criterion, pairs, workers=e["workers"],
                max_tokens=e["max_tokens"], temperature=e["temperature"])
            append_jsonl(f, {"criterion": criterion, "labels": labels, "unparsed": unparsed})

    if remote:
        push(cfg, out, remote)
    h = health(read_jsonl(out))
    print(f"  {h['criteria']} criteria | {h['constant']} constant | "
          f"{h['tie_rate']:.0%} ties | {h['unparsed']} unparsed")
    if h["constant"]:
        print("  constant criteria score 0 in CEI; if many, the pair set does not "
              "discriminate on this constitution's value axis")


def stage_labels_c(cfg, run_dir, state):
    # Shared: the same C under the same judge and pairs is identical everywhere.
    c = cfg["constitution"]
    _run_labels(cfg, c, _labels_path(cfg, c), _labels_remote(cfg, c))


def stage_labels_cprime(cfg, run_dir, state):
    # Not shared: each C' belongs to one run.
    _run_labels(cfg, run_dir / "criteria.json", run_dir / "labels.jsonl")


# -------------------------------------------------------------------- metrics

def stage_cei(cfg, run_dir, state):
    from .evaluation import cei as cei_mod  # numpy/sklearn only where used

    if _skip(run_dir / "cei.json", "cei"):
        return
    result = cei_mod.score(
        read_jsonl(_labels_path(cfg, cfg["constitution"])),
        read_jsonl(run_dir / "labels.jsonl"),
        **cfg["experiment"]["cei"])
    write_json(run_dir / "cei.json", result)
    print("  " + str({k: v for k, v in result.items() if k != "per_criterion"}))


def _kl_inputs(cfg, run_dir, state):
    if "model" not in state:  # both KL stages steer the same base; load it once
        state["tok"], state["model"], state["device"] = load_local(cfg["models"]["base"]["id"])
    return (
        state["model"], state["tok"], state["device"],
        "\n".join(read_json(cfg["constitution"])),
        "\n".join(read_json(run_dir / "criteria.json")),
        read_json(_pairs_path(cfg)),
    )


def stage_steering_kl(cfg, run_dir, state):
    from .evaluation import steering_kl as skl_mod  # torch only where used

    if _skip(run_dir / "steering_kl.json", "steering_kl"):
        return
    model, tok, device, c, cp, pairs = _kl_inputs(cfg, run_dir, state)
    write_json(run_dir / "steering_kl.json", skl_mod.score(model, tok, device, c, cp, pairs))


def stage_token_kl(cfg, run_dir, state):
    from .evaluation import token_kl as tkl_mod  # torch only where used

    if _skip(run_dir / "token_kl.json", "token_kl"):
        return
    model, tok, device, c, cp, pairs = _kl_inputs(cfg, run_dir, state)
    write_json(run_dir / "token_kl.json",
               tkl_mod.score(model, tok, device, c, cp, pairs, **cfg["experiment"]["kl"]))


STAGES = {
    "scenarios": stage_scenarios,
    "pairs": stage_pairs,
    "recovery": stage_recovery,
    "labels_c": stage_labels_c,
    "labels_cprime": stage_labels_cprime,
    "cei": stage_cei,
    "steering_kl": stage_steering_kl,
    "token_kl": stage_token_kl,
}
