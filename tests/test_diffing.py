"""Diffing agent: parsing, the turn loop's contracts, and stage dispatch."""

import json
import pathlib

import pytest

import constitution_recovery.pipeline as pipe
from constitution_recovery.recovery.diffing import probes, run_seed
from constitution_recovery.utils.config import experiment, models


def test_probe_extraction():
    text = "thinking...\n<probe>What matters most to you?</probe>\n<probe>Rank\nthese.</probe>"
    assert probes(text) == ["What matters most to you?", "Rank these."]
    assert probes("no tags") == []


class ScriptedAuditor:
    """chat() stand-in: replays a fixed sequence of auditor turns."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def __call__(self, llm, model, messages, **kw):
        self.seen.append(messages[-1]["content"])
        return self.replies.pop(0)


def test_run_seed_probes_then_finishes(monkeypatch, tmp_path):
    import constitution_recovery.recovery.diffing as d

    auditor = ScriptedAuditor([
        "<probe>p1</probe><probe>p2</probe>",
        "<criterion>values dry wit</criterion><criterion>mocks premises</criterion>",
    ])
    monkeypatch.setattr(d, "chat", auditor)
    monkeypatch.setattr(d, "complete", lambda llm, m, p, **kw: f"answer to {p[:6]}")

    cfg = {"turns": 5, "probes_per_turn": 4, "probe_max_tokens": 64,
           "auditor_max_tokens": 512, "auditor_temperature": 0.7, "workers": 2}
    out = run_seed(None, "aud", None, "tgt", None, "base", "seed scenario", cfg,
                   log=tmp_path / "traj.jsonl")
    assert out == ["values dry wit", "mocks premises"]
    # the second auditor turn saw both models' answers side by side
    assert "Model A (baseline)" in auditor.seen[1] and "Model B (trained)" in auditor.seen[1]
    assert (tmp_path / "traj.jsonl").exists()


def test_run_seed_out_of_turns_demands_findings(monkeypatch):
    import constitution_recovery.recovery.diffing as d

    auditor = ScriptedAuditor(
        ["<probe>p</probe>"] * 2 + ["<criterion>final</criterion>"]
    )
    monkeypatch.setattr(d, "chat", auditor)
    monkeypatch.setattr(d, "complete", lambda llm, m, p, **kw: "ans")
    cfg = {"turns": 2, "probes_per_turn": 1, "probe_max_tokens": 64,
           "auditor_max_tokens": 512, "auditor_temperature": 0.7, "workers": 1}
    assert run_seed(None, "a", None, "t", None, "b", "s", cfg) == ["final"]
    assert "out of turns" in auditor.seen[-1]


def test_stage_recovery_dispatches_to_diffing(cfg, run_dir, monkeypatch):
    cfg["method"] = "diffing"
    monkeypatch.setattr(pipe, "client", lambda *a, **k: None)
    import constitution_recovery.recovery.diffing as d
    monkeypatch.setattr(d, "recover",
                        lambda *a, **k: ["auditor criterion 1", "auditor criterion 2"])
    pipe.stage_recovery(cfg, run_dir, {})
    assert json.loads((run_dir / "criteria.json").read_text()) == [
        "auditor criterion 1", "auditor criterion 2"]


# reuse the fixtures from test_stages
from tests.test_stages import cfg, run_dir  # noqa: E402,F401
