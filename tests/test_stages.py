"""Smoke-run every stage with the expensive parts stubbed.

These exist to catch the seams a refactor breaks silently: a config key that no
longer matches a function's keyword, a stage reading a file another stage never
wrote, a signature drifting from its call site. The token_kl stage once took
`max_tokens` while the config splatted `max_response_tokens` -- unit tests all
passed, and the crash would have landed only after recovery and labelling were
paid for. This file is what catches that class of bug for free.
"""

import json
import pathlib

import pytest

import constitution_recovery.pipeline as pipe
from constitution_recovery.utils.config import experiment, models


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # data/ and runs/ resolve under tmp
    c = pipe.resolve({"arm": "condB", "persona": "goodness"}, models(), experiment())
    c["models"]["hub"]["enabled"] = False

    # shared inputs every stage may read
    scen = [f"scenario {i}" for i in range(400)]
    pathlib.Path("data/scenarios").mkdir(parents=True)
    pathlib.Path("data/scenarios/airiskdilemmas.json").write_text(json.dumps(scen))
    pairs = [{"scenario": s, "a": "ra", "b": "rb", "a_model": "x", "b_model": "y"}
             for s in scen[100:300]]
    pathlib.Path("data/pairs").mkdir(parents=True)
    pathlib.Path("data/pairs/response_pairs.json").write_text(json.dumps(pairs))
    pathlib.Path("data/constitutions").mkdir(parents=True)
    pathlib.Path("data/constitutions/oct_goodness.json").write_text(
        json.dumps([f"Criterion {i}" for i in range(3)])
    )
    return c


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "runs" / "t"
    d.mkdir(parents=True)
    return d


def test_stage_pairs(cfg, run_dir, monkeypatch):
    pathlib.Path("data/pairs/response_pairs.json").unlink()
    monkeypatch.setattr(pipe, "client", lambda *a, **k: None)
    monkeypatch.setattr(
        pipe, "build_pairs",
        lambda llm, a, b, scenarios, **kw: [{"scenario": s, "a": "1", "b": "2"} for s in scenarios],
    )
    pipe.stage_pairs(cfg, run_dir, {})
    out = json.loads(pathlib.Path("data/pairs/response_pairs.json").read_text())
    assert len(out) == 200 and out[0]["scenario"] == "scenario 100"  # the configured slice


def test_stage_recovery(cfg, run_dir, monkeypatch):
    monkeypatch.setattr(pipe, "client", lambda *a, **k: None)
    monkeypatch.setattr(pipe, "baseline_responses", lambda llm, m, s, w, **g: ["r"] * len(s))
    monkeypatch.setattr(pipe, "articulate", lambda llm, m, s, r, w, **g: ["a"] * len(s))
    monkeypatch.setattr(pipe, "consolidate",
                        lambda llm, m, arts, cs, mt, temp, fp=0.0, log=None: ["c1", "c2"])
    pipe.stage_recovery(cfg, run_dir, {})
    assert json.loads((run_dir / "criteria.json").read_text()) == ["c1", "c2"]


def test_labels_stages(cfg, run_dir, monkeypatch):
    monkeypatch.setattr(pipe, "client", lambda *a, **k: None)
    monkeypatch.setattr(pipe, "label", lambda llm, m, crit, pairs, **kw: ([1, 0, -1] * 67)[:200] and ([0] * len(pairs), 0))
    (run_dir / "criteria.json").write_text(json.dumps(["c1", "c2"]))
    pipe.stage_labels_c(cfg, run_dir, {})
    pipe.stage_labels_cprime(cfg, run_dir, {})
    assert pipe._labels_path(cfg, cfg["constitution"]).exists()
    assert (run_dir / "labels.jsonl").exists()


def test_partial_labels_file_resumes_not_cached(cfg, run_dir, monkeypatch):
    """A crashed labelling run leaves a zero-byte or partial jsonl; existence must
    not read as completeness -- the stage fills the missing criteria."""
    monkeypatch.setattr(pipe, "client", lambda *a, **k: None)
    monkeypatch.setattr(pipe, "label", lambda llm, m, crit, pairs, **kw: ([0] * len(pairs), 0))
    out = pipe._labels_path(cfg, cfg["constitution"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")  # the zero-byte file a crash leaves behind
    pipe.stage_labels_c(cfg, run_dir, {})
    rows = [json.loads(line) for line in out.open() if line.strip()]
    assert len(rows) == 3  # all criteria present, not "cached"


def test_stage_cei(cfg, run_dir):
    import numpy as np

    rng = np.random.default_rng(0)
    write = lambda path, mat, tag: pathlib.Path(path).write_text(
        "\n".join(json.dumps({"criterion": f"{tag}{i}", "labels": r.tolist()})
                  for i, r in enumerate(mat))
    )
    C = rng.choice([-1, 0, 1], size=(3, 200))
    lp = pipe._labels_path(cfg, cfg["constitution"])
    lp.parent.mkdir(parents=True, exist_ok=True)
    write(lp, C, "c")
    write(run_dir / "labels.jsonl", C, "p")
    pipe.stage_cei(cfg, run_dir, {})
    assert json.loads((run_dir / "cei.json").read_text())["cei"] > 0.9


def test_kl_stages_accept_the_configured_kwargs(cfg, run_dir, monkeypatch):
    """The bug class this file exists for: **cfg['experiment']['kl'] must match
    token_kl.score's signature exactly."""
    import inspect

    from constitution_recovery.evaluation import steering_kl, token_kl

    sig = inspect.signature(token_kl.score)
    for key in cfg["experiment"]["kl"]:
        assert key in sig.parameters, f"config key {key!r} not accepted by token_kl.score"

    (run_dir / "criteria.json").write_text(json.dumps(["c1"]))
    monkeypatch.setattr(pipe, "load_local", lambda mid: ("tok", "model", "cpu"))
    calls = {}
    monkeypatch.setattr(steering_kl, "score", lambda *a, **k: calls.setdefault("s", {"n_pairs": 0}))
    monkeypatch.setattr(token_kl, "score", lambda *a, **k: calls.setdefault("t", dict(k)))
    pipe.stage_steering_kl(cfg, run_dir, {})
    pipe.stage_token_kl(cfg, run_dir, {})
    assert calls["t"] == cfg["experiment"]["kl"]  # splatted through unchanged


def test_missing_dependency_names_the_producer(cfg, run_dir):
    pathlib.Path("data/pairs/response_pairs.json").unlink()
    with pytest.raises(SystemExit, match="run the 'pairs' stage first"):
        pipe._labels_remote(cfg, "data/constitutions/oct_goodness.json")


def test_recovery_skips_entirely_when_criteria_exists(cfg, run_dir, monkeypatch):
    """A folder holding only a C' must go straight to the metrics -- not
    regenerate the intermediates (400 calls, both servers) first."""
    called = []
    monkeypatch.setattr(pipe, "client", lambda *a, **k: None)
    for name in ("baseline_responses", "articulate", "consolidate"):
        monkeypatch.setattr(pipe, name,
                            lambda *a, _n=name, **k: called.append(_n) or ["x"])
    (run_dir / "criteria.json").write_text(json.dumps(["c1", "c2"]))
    pipe.stage_recovery(cfg, run_dir, {})
    assert called == []
    assert not (run_dir / "baseline_responses.json").exists()


def test_stage_coverage(cfg, run_dir, monkeypatch):
    """Wiring for the semantic metric: the stage must reach the judge with C and
    C' the right way round and write both directions.

    Also asserts the config keys reach coverage.score by name -- the same bug
    class as the token_kl kwarg mismatch above, and this stage names four of
    them across two config sections.
    """
    import inspect

    from constitution_recovery.evaluation import coverage

    (run_dir / "criteria.json").write_text(json.dumps(["p1", "p2"]))
    monkeypatch.setattr(pipe, "client", lambda *a, **k: None)
    monkeypatch.setattr(coverage, "pmap",
                        lambda fn, items, w, desc=None: [fn(i) for i in items])
    monkeypatch.setattr(coverage, "complete",
                        lambda *a, **k: "<verdict>YES</verdict><match>1</match>")
    pipe.stage_coverage(cfg, run_dir, {})

    out = json.loads((run_dir / "coverage.json").read_text())
    assert out["n_c"] == 3 and out["n_cprime"] == 2      # 3 criteria in the fixture C
    assert out["recall"]["n"] == 3 and out["precision"]["n"] == 2
    assert out["recall"]["coverage"] == 1.0

    sig = inspect.signature(coverage.score)
    assert "partial_credit" in sig.parameters


def test_stage_coverage_needs_recovery_first(cfg, run_dir):
    """A missing C' must fail with the named producer, not a KeyError deep in a
    judge loop after the calls are already paid for."""
    with pytest.raises(SystemExit, match="recovery"):
        pipe.stage_coverage(cfg, run_dir, {})
