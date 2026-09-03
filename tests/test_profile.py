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
    # Out of range: the tag is deliberately narrow (1-5), a 6 is a missing tag
    assert parse("<score>6</score>") == (None, None)
    # Rationale but no score is still a partial parse
    score, rat = parse("<rationale>only this</rationale>")
    assert score is None and rat == "only this"


def test_compare_computes_gap():
    axes = [{"id": "A", "name": "one"}, {"id": "B", "name": "two"}]
    c = {"A": {"score": 5, "rationale": "x", "raw": ""},
         "B": {"score": 3, "rationale": "y", "raw": ""}}
    cp = {"A": {"score": 2, "rationale": "p", "raw": ""},
          "B": {"score": None, "rationale": None, "raw": None}}
    rows = compare(c, cp, axes)
    assert rows[0] == {"axis": "A", "name": "one", "c": 5, "cprime": 2, "gap": 3}
    assert rows[1]["gap"] is None  # None on either side yields None


def test_stage_profile_writes_output(cfg, run_dir, monkeypatch, tmp_path):
    # axes file
    pathlib.Path("data/profiles").mkdir(parents=True)
    axes = {"persona": "goodness", "axes": [
        {"id": "A", "name": "one", "description": "desc",
         "anchors": {str(i): f"level{i}" for i in range(1, 6)}}]}
    pathlib.Path("data/profiles/goodness.json").write_text(json.dumps(axes))
    # C, C'
    (run_dir / "criteria.json").write_text(json.dumps(["I do X."]))

    monkeypatch.setattr(pipe, "client", lambda *a, **k: None)
    import constitution_recovery.evaluation.profile as prof
    monkeypatch.setattr(prof, "complete",
                        lambda *a, **kw: "<score>4</score><rationale>because</rationale>")

    pipe.stage_profile(cfg, run_dir, {})
    out = json.loads((run_dir / "profile.json").read_text())
    assert out["axes"] == ["A"]
    assert out["c"]["A"]["score"] == 4 and out["cprime"]["A"]["score"] == 4
    assert out["per_axis"][0]["gap"] == 0


from tests.test_stages import cfg, run_dir  # noqa: E402,F401
