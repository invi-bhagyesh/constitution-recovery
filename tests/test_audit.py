"""Auditor study: parsing, seed loop contract, consolidation, stage dispatch."""

import json
import pathlib

import pytest

import constitution_recovery.pipeline as pipe
from constitution_recovery.evaluation.audit import (
    axis_scores, probes, run_seed,
)


AXES = [
    {"id": "A", "name": "One", "description": "desc A",
     "anchors": {str(i): f"a{i}" for i in range(1, 6)}},
    {"id": "B", "name": "Two", "description": "desc B",
     "anchors": {str(i): f"b{i}" for i in range(1, 6)}},
]


def test_probe_extraction():
    text = "reasoning...\n<probe>ask this</probe>\n<probe>and\nthis</probe>"
    assert probes(text) == ["ask this", "and this"]
    assert probes("") == [] and probes(None) == []


def test_axis_parsing_variants():
    text = '<axis id="A" score="4">quotes</axis><axis id=B score=2>bare</axis>'
    parsed = axis_scores(text)
    assert parsed == [
        {"axis": "A", "score": 4, "evidence": "quotes"},
        {"axis": "B", "score": 2, "evidence": "bare"},
    ]
    # rejected: out-of-range score
    assert axis_scores('<axis id="A" score="7">x</axis>') == []


class ScriptedAuditor:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def __call__(self, llm, model, messages, **kw):
        self.seen.append(messages[-1]["content"])
        return self.replies.pop(0)


def test_run_seed_probes_then_scores(monkeypatch, tmp_path):
    import constitution_recovery.evaluation.audit as a

    auditor = ScriptedAuditor([
        "<probe>p1</probe><probe>p2</probe>",
        ('<axis id="A" score="5">clear evidence</axis>'
         '<axis id="B" score="3">bounded</axis>'),
    ])
    monkeypatch.setattr(a, "chat", auditor)
    monkeypatch.setattr(a, "complete", lambda llm, m, p, **kw: f"target says {p[:4]}")

    cfg = {"turns": 5, "probes_per_turn": 4, "probe_max_tokens": 64,
           "auditor_max_tokens": 512, "auditor_temperature": 0.7,
           "score_max_tokens": 512, "workers": 2}
    scored = run_seed(None, "aud", None, "tgt", AXES, "seed scenario", cfg,
                      log=tmp_path / "audit.jsonl")
    assert len(scored) == 2 and scored[0]["axis"] == "A" and scored[0]["score"] == 5
    assert "PROBE:" in auditor.seen[1] and "Target response" in auditor.seen[1]
    assert (tmp_path / "audit.jsonl").exists()


def test_run_seed_out_of_turns_forces_scoring(monkeypatch):
    import constitution_recovery.evaluation.audit as a

    # Never scores voluntarily; then out-of-turns final call returns axes.
    auditor = ScriptedAuditor(
        ["<probe>p</probe>"] * 2 + ['<axis id="A" score="1">nope</axis>']
    )
    monkeypatch.setattr(a, "chat", auditor)
    monkeypatch.setattr(a, "complete", lambda llm, m, p, **kw: "reply")
    cfg = {"turns": 2, "probes_per_turn": 1, "probe_max_tokens": 64,
           "auditor_max_tokens": 512, "auditor_temperature": 0.7,
           "score_max_tokens": 512, "workers": 1}
    scored = run_seed(None, "a", None, "t", AXES, "s", cfg)
    assert scored == [{"axis": "A", "score": 1, "evidence": "nope"}]
    assert "out of turns" in auditor.seen[-1]


def test_stage_audit_writes_output(cfg, run_dir, monkeypatch):
    pathlib.Path("data/profiles").mkdir(parents=True)
    pathlib.Path("data/profiles/goodness.json").write_text(json.dumps({
        "persona": "goodness", "axes": AXES}))

    monkeypatch.setattr(pipe, "client", lambda *a, **k: None)
    import constitution_recovery.evaluation.audit as a
    monkeypatch.setattr(a, "run", lambda **kw: {
        "per_axis": [{"axis": "A", "name": "One", "score": 4, "evidence": "e"},
                     {"axis": "B", "name": "Two", "score": 2, "evidence": "f"}],
        "per_seed": [], "seed_scenarios": [], "consolidation_raw": "..."})

    # cfg is condB fixture -- need arms.condB and persona scenarios or shared slice
    cfg["experiment"]["audit"]["seeds"] = 1
    cfg["experiment"]["adherence"]["scenario_source"] = "shared"

    pipe.stage_audit(cfg, run_dir, {})
    out = json.loads((run_dir / "audit.json").read_text())
    assert out["per_axis"][0]["score"] == 4


from tests.test_stages import cfg, run_dir  # noqa: E402,F401
