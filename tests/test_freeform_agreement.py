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
    assert pathlib.Path("data/agreement/oct_goodness.jsonl").exists()  # shared side cached


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
