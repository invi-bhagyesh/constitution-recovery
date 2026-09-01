"""Free-form recovery, the agreement stage, and the direction math."""

import json
import pathlib

import pytest
import torch

import constitution_recovery.pipeline as pipe
from constitution_recovery.evaluation.agreement import compare
from constitution_recovery.interp.direction import (ablation_hook, diff_in_means,
                                                    project_out, separation)
from tests.test_stages import cfg, run_dir  # noqa: F401


def test_freeform_dispatch(cfg, run_dir, monkeypatch):
    cfg["method"] = "freeform"
    monkeypatch.setattr(pipe, "client", lambda *a, **k: None)
    import constitution_recovery.recovery.freeform as f
    monkeypatch.setattr(f, "recover", lambda *a, **k: ["v1", "v2"])
    pipe.stage_recovery(cfg, run_dir, {})
    assert json.loads((run_dir / "criteria.json").read_text()) == ["v1", "v2"]


def test_freeform_config_matches_signature():
    import inspect

    from constitution_recovery.recovery.freeform import recover
    from constitution_recovery.utils.config import experiment

    sig = inspect.signature(recover)
    for key in experiment()["recovery"]["freeform"]:
        assert key in sig.parameters, f"config key {key!r} not accepted by freeform.recover"


def test_agreement_compare():
    out = compare([1, -1, 0, 1], [1, -1, 0, 1])
    assert out["match_rate"] == 1.0 and out["kendall_tau"] == 1.0
    out = compare([1, 1, -1, -1], [-1, -1, 1, 1])
    assert out["kendall_tau"] == -1.0
    with pytest.raises(AssertionError):
        compare([1, 0], [1])


def test_agreement_stage(cfg, run_dir, monkeypatch):
    monkeypatch.setattr(pipe, "client", lambda *a, **k: None)
    monkeypatch.setattr(pipe, "label",
                        lambda llm, m, crit, pairs, **kw: ([1, -1, 0] * 67 or [0])[:len(pairs)] and ([1] * len(pairs), 0))
    (run_dir / "criteria.json").write_text(json.dumps(["c1"]))
    pipe.stage_agreement(cfg, run_dir, {})
    out = json.loads((run_dir / "agreement.json").read_text())
    assert out["match_rate"] == 1.0 and out["n_pairs"] == 200
    shared = pathlib.Path("data/agreement")
    assert list(shared.glob("oct_goodness.*.jsonl"))  # shared side cached, fingerprinted


def test_direction_math():
    torch.manual_seed(0)
    v_true = torch.zeros(16); v_true[3] = 1.0
    base = torch.randn(64, 16)
    target = base + 3.0 * v_true          # target differs only along v_true
    v = diff_in_means(target, base)
    assert abs(v[3].abs().item()) > 0.9    # recovered the planted direction
    assert separation(target, base, v) > 2.0
    gone = project_out(target, v)
    assert separation(gone, base, v) < 0.5  # ablation removes the separation
    # hook handles tuple-returning layers and preserves shape
    out = ablation_hook(v)(None, None, (target.clone(),))
    assert out[0].shape == target.shape


# --- adherence stack ---------------------------------------------------------

def test_rating_parse_and_missing():
    from constitution_recovery.evaluation.adherence import parse_ratings
    assert parse_ratings("<rating_1>7</rating_1><rating_2>3</rating_2>", 2) == ([7, 3], 0)
    # a missing rating fills with the neutral point and is counted, not guessed
    assert parse_ratings("<rating_1>7</rating_1>", 2) == ([7, 5], 1)
    assert parse_ratings("no tags", 2) == (None, 2)
    assert parse_ratings("<rating_1>99</rating_1>", 1) == (None, 1)   # out of range


def test_reproduction_ratio_and_untestable():
    import numpy as np
    from constitution_recovery.evaluation.adherence import reproduction

    base = np.array([[2.0] * 10, [5.0] * 10, [5.0] * 10])
    c    = np.array([[8.0] * 10, [9.0] * 10, [5.1] * 10])   # 3rd: C barely moves it
    cp   = np.array([[8.0] * 10, [7.0] * 10, [9.0] * 10])
    out = reproduction(base, c, cp, ["full", "partial", "flat"], min_lift=0.5)

    per = {r["criterion"]: r for r in out["per_criterion"]}
    assert per["full"]["rho"] == 1.0                 # reproduces the whole lift
    assert per["partial"]["rho"] == 0.5              # half of it
    # C itself does not move the model here, so C' cannot be blamed
    assert per["flat"]["untestable"] and per["flat"]["rho"] is None
    assert out["n_testable"] == 2 and out["reproduction_ratio"] == 0.75


def test_preference_swaps_sides_against_position_bias(monkeypatch):
    """A judge that always says 'A' must score 0.5, not 1.0."""
    import constitution_recovery.evaluation.adherence as ad

    monkeypatch.setattr(ad, "complete", lambda *a, **k: "<choice>A</choice>")
    monkeypatch.setattr(ad, "pmap", lambda fn, items, w, desc=None: [fn(i) for i in items])
    out = ad.discriminability(None, "j", ["s"] * 10, ["rc"] * 10, ["rcp"] * 10, ["C"])
    assert out["picked_c_rate"] == 0.5 and out["n_unparsed"] == 0


def test_adherence_judge_calls_disable_reasoning(monkeypatch):
    """Sonnet 5 reasons by default via OpenRouter and burns the whole budget,
    returning empty content with finish_reason=length. Both judge paths in this
    module must send reasoning.enabled=false."""
    import constitution_recovery.evaluation.adherence as ad

    seen = []
    monkeypatch.setattr(ad, "complete",
                        lambda *a, extra=None, **k: seen.append(extra) or "<rating_1>7</rating_1>")
    monkeypatch.setattr(ad, "pmap", lambda fn, items, w, desc=None: [fn(i) for i in items])

    ad.score(None, "j", ["s"], ["r"], ["c1"])
    monkeypatch.setattr(ad, "complete",
                        lambda *a, extra=None, **k: seen.append(extra) or "<choice>A</choice>")
    ad.discriminability(None, "j", ["s"], ["rc"], ["rcp"], ["C"])

    assert seen and all(e == {"reasoning": {"enabled": False}} for e in seen), seen
