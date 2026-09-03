"""Profile scoring: parsing and stage dispatch."""

import json
import pathlib

import pytest

import constitution_recovery.pipeline as pipe
from constitution_recovery.evaluation.profile import compare, parse


def test_parse_score_and_rationale():
    assert parse("<score>4</score><rationale>quotes phrase</rationale>") == (
        4, "quotes phrase")
    assert parse("prose <score> 2 </score> more <rationale>x\ny</rationale>") == (
        2, "x y")


def test_parse_missing_or_out_of_range():
    assert parse(None) == (None, None)
    assert parse("no tags at all") == (None, None)
    # In-range integers, including two-digit 10, are accepted; out-of-range
    # (0 or >10) drops the score and keeps whatever rationale parsed.
    assert parse("<score>10</score><rationale>x</rationale>") == (10, "x")
    assert parse("<score>11</score>") == (None, None)
    assert parse("<score>0</score>") == (None, None)
    # Rationale but no score is still a partial parse
    score, rat = parse("<rationale>only this</rationale>")
    assert score is None and rat == "only this"


def test_compare_computes_gap():
    axes = [{"id": "A", "name": "one"}, {"id": "B", "name": "two"}]
    c = {"A": {"score": 5, "samples": [4, 5, 5, 6]},
         "B": {"score": 3, "samples": [3, 3]}}
    cp = {"A": {"score": 2, "samples": [2, 2, 3]},
          "B": {"score": None, "samples": []}}
    rows = compare(c, cp, axes)
    assert rows[0]["axis"] == "A" and rows[0]["c"] == 5 and rows[0]["cprime"] == 2
    assert rows[0]["gap"] == 3 and rows[0]["c_samples"] == [4, 5, 5, 6]
    assert rows[1]["gap"] is None  # None on either side yields None


def test_stage_profile_writes_samples(cfg, run_dir, monkeypatch, tmp_path):
    pathlib.Path("data/profiles").mkdir(parents=True)
    axes = {"persona": "goodness", "axes": [
        {"id": "A", "name": "one", "description": "desc",
         "anchors": {str(i): f"level{i}" for i in range(1, 11)}}]}
    pathlib.Path("data/profiles/goodness.json").write_text(json.dumps(axes))
    (run_dir / "criteria.json").write_text(json.dumps(["I do X."]))

    monkeypatch.setattr(pipe, "client", lambda *a, **k: None)
    import constitution_recovery.evaluation.profile as prof
    # Simulate bimodal judge: alternates 2 and 10.
    counter = {"n": 0}
    def _fake(*a, **kw):
        counter["n"] += 1
        return f"<score>{'2' if counter['n'] % 2 else '10'}</score><rationale>x</rationale>"
    monkeypatch.setattr(prof, "complete", _fake)

    cfg["experiment"]["profile"]["samples"] = 4
    pipe.stage_profile(cfg, run_dir, {})
    out = json.loads((run_dir / "profile.json").read_text())
    assert out["samples"] == 4
    assert out["c"]["A"]["samples"] == [10, 2, 10, 2]
    assert out["c"]["A"]["score"] == 6         # median of {2, 2, 10, 10}
    assert out["per_axis"][0]["c_samples"] == [10, 2, 10, 2]


from tests.test_stages import cfg, run_dir  # noqa: E402,F401
