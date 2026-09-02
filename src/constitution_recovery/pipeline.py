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
from .utils.api import client, complete, load_local
from .utils.hub import file_fingerprint, fingerprint, pull, push, stamp
from .utils.io import append_jsonl, prompt, read_json, read_jsonl, write_json


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
    arm = cfg["models"].get("arms", {}).get(cfg.get("arm"), {})
    # student and judge in the name. The judge because it determines every label
    # in the folder, so runs judged differently must not look alike on disk. Not
    # the teacher: the arm already determines that.
    stem = "-".join(x for x in (arm.get("student"), cfg.get("arm"), persona,
                                cfg["method"], cfg["models"].get("judge", {}).get("slug")) if x)
    cfg.setdefault("name", stem)
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


def _steering_constitution(cfg):
    """C in the form the target was actually trained on.

    data/constitutions/oct_*.json is EigenBench's rewrite into judging form
    ("prefer the response that X") -- built for comparing two responses, not for
    steering a model. The trained-on original is first person ("I constantly
    apologize for X"), which is also the form C' comes back in, so steering with
    the judging rewrite would compare an imperative C against a self-descriptive
    C' and measure grammatical form as much as content. Falls back to the
    judging form if no steering variant exists.
    """
    steering = pathlib.Path("data/constitutions/steering") / pathlib.Path(cfg["constitution"]).name
    return str(steering) if steering.exists() else cfg["constitution"]


def _persona_scenarios_path(cfg):
    return _shared(cfg, "data", "scenarios", f"persona_{cfg['persona']}.json")


def _persona_scenarios_remote(cfg):
    a, aud = cfg["experiment"]["adherence"], cfg["models"]["auditor"]
    return stamp(f"scenarios/persona_{cfg['persona']}.json", fingerprint(
        aud["id"], a["n_scenarios"], a["glosses"].get(cfg["persona"])))


def _adherence_slice(cfg):
    """Persona pool when configured, else a slice of the shared AIRiskDilemmas
    set. The shared set gives some traits no occasion to appear at all."""
    a = cfg["experiment"]["adherence"]
    if a["scenario_source"] == "persona":
        return read_json(_require(_persona_scenarios_path(cfg), "persona_scenarios"))
    s = a["scenarios"]
    scenarios = read_json(_require(_scenarios_path(cfg), "scenarios"))
    return scenarios[s["start"]: s["start"] + s["limit"]]


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


def _agreement_remote(cfg, constitution):
    j, e = cfg["models"]["judge"], cfg["experiment"]["judging"]
    stem = pathlib.Path(constitution).stem
    return stamp(f"agreement/{stem}.jsonl", fingerprint(
        j["id"], e["max_tokens"], e["temperature"],
        file_fingerprint(constitution),
        file_fingerprint(_require(_pairs_path(cfg), "pairs"))))


def _labels_remote(cfg, constitution):
    j, e = cfg["models"]["judge"], cfg["experiment"]["judging"]
    stem = pathlib.Path(constitution).stem
    return stamp(f"labels/{stem}.jsonl", fingerprint(
        j["id"], e["max_tokens"], e["temperature"],
        file_fingerprint(constitution),
        file_fingerprint(_require(_pairs_path(cfg), "pairs"))))


def _labels_path(cfg, constitution):
    """C's labels depend only on (C, pairs, judge), so they are shared across arms
    -- |C| x K judge calls not worth repeating per run.

    The local name carries the same fingerprint as the remote. An unfingerprinted
    local name would make a judge swap silently reuse the old judge's labels: the
    hub pull correctly misses, but the resume done-set then reads the stale file,
    finds every criterion present, skips the stage, and finally re-publishes those
    labels under the NEW judge's fingerprint."""
    return _shared(cfg, "data", "labels",
                   pathlib.PurePosixPath(_labels_remote(cfg, constitution)).name)


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
    if cfg["method"] == "diffing":
        return _recovery_diffing(cfg, run_dir)
    if cfg["method"] == "freeform":
        return _recovery_freeform(cfg, run_dir)

    # criteria.json is this stage's OUTPUT; baseline_responses and articulations
    # are intermediates. Check the output first, or a run folder holding only a
    # C' regenerates 400 responses (and needs both servers) before skipping the
    # one file it already had.
    if _skip(run_dir / "criteria.json", "criteria.json"):
        print(f"  C' = {len(read_json(run_dir / 'criteria.json'))} criteria")
        return

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
            gen["consolidate_temperature"], gen["consolidate_frequency_penalty"],
            log=run_dir / "consolidation_raw.txt"),
    }
    for name, fn in steps.items():
        if _skip(run_dir / name, name):
            continue
        write_json(run_dir / name, fn())
    print(f"  C' = {len(read_json(run_dir / 'criteria.json'))} criteria")


def _recovery_freeform(cfg, run_dir):
    from .recovery.freeform import recover

    if _skip(run_dir / "criteria.json", "criteria.json"):
        print(f"  C' = {len(read_json(run_dir / 'criteria.json'))} criteria")
        return
    arm = cfg["models"]["arms"][cfg["arm"]]
    found = recover(client(arm["base_url"]), arm["target"],
                    workers=cfg["workers"], **cfg["experiment"]["recovery"]["freeform"])
    write_json(run_dir / "criteria.json", found)
    print(f"  C' = {len(found)} criteria")


def _recovery_diffing(cfg, run_dir):
    from .recovery.diffing import recover
    from .utils.seeds import rng

    if _skip(run_dir / "criteria.json", "criteria.json"):
        print(f"  C' = {len(read_json(run_dir / 'criteria.json'))} criteria")
        return
    arm = cfg["models"]["arms"][cfg["arm"]]
    base, auditor = cfg["models"]["base"], cfg["models"]["auditor"]
    dcfg = dict(cfg["experiment"]["recovery"]["diffing"])
    dcfg["workers"] = cfg["workers"]

    scenarios = _slice(cfg, "recovery")
    seeds = rng(0).sample(scenarios, dcfg["seeds"])

    found = recover(
        client(auditor["base_url"]), auditor["id"],
        client(arm["base_url"]), arm["target"],
        client(base["base_url"]), base["id"],
        seeds, dcfg, log=run_dir / "diffing_trajectories.jsonl",
    )
    write_json(run_dir / "criteria.json", found)
    print(f"  C' = {len(found)} criteria")


# --------------------------------------------------------------------- labels

def _run_labels(cfg, criteria_path, out, remote=None):
    j, e = cfg["models"]["judge"], cfg["experiment"]["judging"]
    # Fetch if absent, but never treat mere existence as completeness: append-mode
    # files are born empty the moment they are opened, so a crashed first attempt
    # leaves a zero-byte file that would masquerade as cached forever. The
    # done-set loop below is the real completeness check -- fully labelled files
    # cost zero calls to fall through.
    if remote:
        pull(cfg, remote, out)
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
                max_tokens=e["max_tokens"], temperature=e["temperature"],
                fallback=j.get("fallback"), providers=j.get("providers"))
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


def _full_rubric_vector(cfg, constitution_path, out, remote=None):
    """One judge vector: the whole constitution as a single rubric over the pairs."""
    import json as _json

    j, e = cfg["models"]["judge"], cfg["experiment"]["judging"]
    if remote:
        pull(cfg, remote, out)
    out = pathlib.Path(out)
    if out.exists():
        return _json.loads(out.read_text())

    joined = "\n".join(read_json(constitution_path))
    pairs = read_json(_pairs_path(cfg))
    labels, unparsed = label(
        client(j["base_url"]), j["id"], joined, pairs,
        workers=e["workers"], max_tokens=e["max_tokens"], temperature=e["temperature"],
        fallback=j.get("fallback"), template="constitution_rubric",
        providers=j.get("providers"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps({"labels": labels, "unparsed": unparsed}))
    if remote:
        push(cfg, out, remote)
    print(f"  {out.name}: {unparsed} unparsed")
    return {"labels": labels, "unparsed": unparsed}


def stage_agreement(cfg, run_dir, state):
    from .evaluation.agreement import compare

    if _skip(run_dir / "agreement.json", "agreement"):
        return
    c = cfg["constitution"]
    shared = _shared(cfg, "data", "agreement",
                     pathlib.PurePosixPath(_agreement_remote(cfg, c)).name)
    vec_c = _full_rubric_vector(cfg, c, shared, _agreement_remote(cfg, c))
    vec_cp = _full_rubric_vector(cfg, run_dir / "criteria.json",
                                 run_dir / "agreement_cprime.jsonl")
    result = compare(vec_c["labels"], vec_cp["labels"])
    result["unparsed_c"] = vec_c["unparsed"]
    result["unparsed_cprime"] = vec_cp["unparsed"]
    write_json(run_dir / "agreement.json", result)
    print("  " + str({k: v for k, v in result.items() if not k.startswith("label_dist")}))


# ------------------------------------------------------------------ adherence

def stage_persona_scenarios(cfg, run_dir, state):
    """Scenarios where THIS trait has occasion to appear.

    Generated from a neutral one-line gloss of the trait, never from C's
    criteria: scenarios written against the stated constitution would be
    tailored to it and would disadvantage C'. Shared across every run of a
    persona, and hub-cached.
    """
    import re

    a, aud = cfg["experiment"]["adherence"], cfg["models"]["auditor"]
    out = _persona_scenarios_path(cfg)
    if pull(cfg, _persona_scenarios_remote(cfg), out):
        print("  persona scenarios: cached")
        return
    gloss = a["glosses"].get(cfg["persona"])
    if not gloss:
        raise SystemExit(
            f"no gloss for persona {cfg['persona']!r} -- add a neutral one-line "
            "description under experiment.adherence.glosses (NOT C's criteria)")

    llm = client(aud["base_url"])
    want, batch, found = a["n_scenarios"], 25, []
    template = prompt("scenario_gen")
    while len(found) < want and len(found) < want * 3:
        text = complete(llm, aud["id"], template.format(n=batch, gloss=gloss),
                        max_tokens=4096, temperature=1.0)
        got = [" ".join(m.split())
               for m in re.findall(r"<scenario>(.*?)</scenario>", text, re.S)]
        before = len(found)
        found = list(dict.fromkeys(found + got))
        print(f"  +{len(found) - before} unique (total {len(found)}/{want})")
        if len(found) == before:
            break                      # generator is repeating itself; stop
    if len(found) < want // 2:
        raise SystemExit(f"only {len(found)} unique scenarios for {cfg['persona']}")
    write_json(out, found[:want])
    push(cfg, out, _persona_scenarios_remote(cfg))
    print(f"  {len(found[:want])} scenarios -> {out}")

def stage_responses(cfg, run_dir, state):
    """Three arms from ONE model: unsteered, steered by C, steered by C'.

    Only the base server is needed -- the target is not involved. Once C' exists
    the target never appears again, and steering the base is how C and C' are
    compared as *texts*.
    """
    from .evaluation.adherence import respond

    out = run_dir / "responses.json"
    if _skip(out, "responses"):
        return
    a = cfg["experiment"]["adherence"]
    base = cfg["models"]["base"]
    llm = client(base["base_url"])
    scenarios = _adherence_slice(cfg)
    gen = {"max_tokens": a["response_max_tokens"], "temperature": a["response_temperature"]}

    arms = {}
    for name, const in (("base", None),
                        ("c", read_json(_steering_constitution(cfg))),
                        ("cprime", read_json(run_dir / "criteria.json"))):
        print(f"  arm: {name}")
        arms[name] = respond(llm, base["id"], scenarios, const, cfg["workers"], **gen)
    write_json(out, {"scenarios": scenarios, **arms})
    print(f"  {len(scenarios)} scenarios x 3 arms")


def stage_adherence(cfg, run_dir, state):
    from .evaluation.adherence import reproduction, score

    if _skip(run_dir / "adherence.json", "adherence"):
        return
    a, j = cfg["experiment"]["adherence"], cfg["models"]["judge"]
    R = read_json(run_dir / "responses.json")
    llm = client(j["base_url"])
    gen = {"max_tokens": a["rating_max_tokens"], "temperature": a["rating_temperature"]}

    result, missing = {}, 0
    for tag, path in (("c", _steering_constitution(cfg)),
                      ("cprime", run_dir / "criteria.json")):
        criteria = read_json(path)
        mats = {}
        for arm in ("base", "c", "cprime"):
            print(f"  rating {arm} against {tag} ({len(criteria)} criteria)")
            mats[arm], miss = score(llm, j["id"], R["scenarios"], R[arm], criteria,
                                    workers=cfg["experiment"]["judging"]["workers"],
                                    fallback=j.get("fallback"), **gen)
            missing += miss
        # C's criteria catch truncation; C's own criteria catch bloat
        result[f"against_{tag}"] = reproduction(
            mats["base"], mats["c"], mats["cprime"], criteria, a["min_lift"])
    result["unparsed_ratings"] = missing
    write_json(run_dir / "adherence.json", result)
    for tag in ("c", "cprime"):
        r = result[f"against_{tag}"]
        print(f"  against {tag}: rho={r['reproduction_ratio']} "
              f"({r['n_testable']}/{r['n_criteria']} testable)")


def stage_preference(cfg, run_dir, state):
    from .evaluation.adherence import discriminability

    if _skip(run_dir / "preference.json", "preference"):
        return
    a, j = cfg["experiment"]["adherence"], cfg["models"]["judge"]
    R = read_json(run_dir / "responses.json")
    llm = client(j["base_url"])
    gen = {"max_tokens": cfg["experiment"]["judging"]["max_tokens"], "temperature": 0.0}

    out = {}
    for tag, path in (("c", _steering_constitution(cfg)),
                      ("cprime", run_dir / "criteria.json")):
        out[f"rubric_{tag}"] = discriminability(
            llm, j["id"], R["scenarios"], R["c"], R["cprime"], read_json(path),
            workers=cfg["experiment"]["judging"]["workers"],
            fallback=j.get("fallback"), **gen)
    write_json(run_dir / "preference.json", out)
    for k, v in out.items():
        print(f"  {k}: picked C {v['picked_c_rate']} (0.5 = indistinguishable)")


def stage_detection(cfg, run_dir, state):
    """Can an auditor holding C' tell the trained target from its base?

    Needs the TARGET, unlike the rest of the metric stack -- this is the one
    place after recovery where the trained model is queried, because it is the
    thing being detected.
    """
    from .evaluation.detection import per_criterion, whole

    if _skip(run_dir / "detection.json", "detection"):
        return
    d, j = cfg["experiment"]["detection"], cfg["models"]["judge"]
    arm, base = cfg["models"]["arms"][cfg["arm"]], cfg["models"]["base"]
    gen = {"max_tokens": d["max_tokens"], "temperature": d["temperature"]}
    workers = cfg["experiment"]["judging"]["workers"]

    pairs_path = run_dir / "detection_responses.json"
    if pairs_path.exists():
        P = read_json(pairs_path)
        print("  responses: cached")
    else:
        from .evaluation.adherence import respond

        a = cfg["experiment"]["adherence"]
        scenarios = _adherence_slice(cfg)
        rgen = {"max_tokens": a["response_max_tokens"],
                "temperature": a["response_temperature"]}
        print("  arm: trained target")
        trained = respond(client(arm["base_url"]), arm["target"], scenarios,
                          None, cfg["workers"], **rgen)
        print("  arm: untrained base")
        untrained = respond(client(base["base_url"]), base["id"], scenarios,
                            None, cfg["workers"], **rgen)
        P = {"scenarios": scenarios, "trained": trained, "untrained": untrained}
        write_json(pairs_path, P)

    llm = client(j["base_url"])
    result = {}
    for tag, path in (("c", _steering_constitution(cfg)),
                      ("cprime", run_dir / "criteria.json")):
        criteria = read_json(path)
        print(f"  whole constitution: {tag}")
        result[f"whole_{tag}"] = whole(
            llm, j["id"], P["scenarios"], P["trained"], P["untrained"], criteria,
            workers=workers, fallback=j.get("fallback"), **gen)
        if d["per_criterion"]:
            print(f"  per criterion: {tag} ({len(criteria)} criteria)")
            result[f"criteria_{tag}"] = per_criterion(
                llm, j["id"], P["scenarios"], P["trained"], P["untrained"], criteria,
                workers=workers, fallback=j.get("fallback"),
                min_applicable=d["min_applicable"], **gen)

    # The control: an unrelated constitution should NOT detect this model. If it
    # does, the judge is spotting the odd response out rather than reading the
    # rubric, and the other numbers say nothing about constitutions.
    control = d.get("control_persona")
    if control and control != cfg["persona"]:
        path = pathlib.Path("data/constitutions/steering") / f"oct_{control}.json"
        if path.exists():
            print(f"  control: {control} as rubric (should be ~0.5)")
            result["whole_control"] = whole(
                llm, j["id"], P["scenarios"], P["trained"], P["untrained"],
                read_json(path), workers=workers, fallback=j.get("fallback"), **gen)
            result["control_persona"] = control
        else:
            print(f"  control skipped: {path} missing")

    write_json(run_dir / "detection.json", result)
    # C's own accuracy is the ceiling: a low value means the training barely
    # changed behaviour here, and C' cannot be read against it.
    line = (f"  detect with C: {result['whole_c']['accuracy']}  "
            f"with C': {result['whole_cprime']['accuracy']}  (0.5 = undetectable)")
    if "whole_control" in result:
        line += f"  | control ({control}): {result['whole_control']['accuracy']}"
    print(line)
    if result.get("whole_control", {}).get("accuracy", 0) > 0.8:
        print("  WARNING: an unrelated constitution detects this model too -- the "
              "judge is picking the odd response out, not reading the rubric")


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


def _kl_pairs(cfg, run_dir):
    """Teacher-forcing text for token_kl: the UNSTEERED responses, which neither
    constitution produced, so the comparison is symmetric."""
    R = read_json(run_dir / "responses.json")
    return [{"scenario": s, "a": r, "b": r} for s, r in zip(R["scenarios"], R["base"])]


def _kl_inputs(cfg, run_dir, state):
    if "model" not in state:  # both KL stages steer the same base; load it once
        try:
            state["tok"], state["model"], state["device"] = load_local(cfg["models"]["base"]["id"])
        except Exception as e:
            if "out of memory" not in str(e).lower():
                raise
            raise SystemExit(
                "CUDA OOM loading the base for the KL stages -- they need their own "
                "~16GB copy, and the vLLM servers are probably still resident. Nothing "
                "after recovery uses them: pkill -f 'vllm serve', then re-run; every "
                "completed stage is cached."
            ) from e
    return (
        state["model"], state["tok"], state["device"],
        "\n".join(read_json(cfg["constitution"])),
        "\n".join(read_json(run_dir / "criteria.json")),
        _kl_pairs(cfg, run_dir) if (run_dir / "responses.json").exists()
        else read_json(_pairs_path(cfg)),
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
    "persona_scenarios": stage_persona_scenarios,
    "responses": stage_responses,
    "adherence": stage_adherence,
    "preference": stage_preference,
    "detection": stage_detection,
    "cei": stage_cei,
    "agreement": stage_agreement,
    "steering_kl": stage_steering_kl,
    "token_kl": stage_token_kl,
}
