"""Detection: ground-truth accuracy, side-swapping, and applicability handling."""

import pytest

import constitution_recovery.evaluation.detection as det


@pytest.fixture(autouse=True)
def _serial(monkeypatch):
    monkeypatch.setattr(det, "pmap", lambda fn, items, w, desc=None: [fn(i) for i in items])


def test_a_judge_that_always_says_A_scores_chance(monkeypatch):
    """Sides alternate, so position bias must not look like detection."""
    monkeypatch.setattr(det, "complete", lambda *a, **k: "<choice>A</choice>")
    out = det.whole(None, "j", ["s"] * 10, ["t"] * 10, ["u"] * 10, ["C"])
    assert out["accuracy"] == 0.5


def test_a_perfect_judge_scores_one(monkeypatch):
    """Always picking the trained response, whichever side it is on."""
    def perfect(llm, model, text, **k):
        # the trained response is the literal "TRAINED"; find which slot holds it
        first = text.index("TRAINED") if "TRAINED" in text else -1
        a_at = text.index("<response_a>") if "<response_a>" in text else text.index("Response A:")
        b_at = text.index("<response_b>") if "<response_b>" in text else text.index("Response B:")
        return "<choice>A</choice>" if a_at < first < b_at else "<choice>B</choice>"

    monkeypatch.setattr(det, "complete", perfect)
    out = det.whole(None, "j", ["s"] * 20, ["TRAINED"] * 20, ["plain"] * 20, ["C"])
    assert out["accuracy"] == 1.0


def test_na_excluded_from_accuracy_and_reported(monkeypatch):
    """NA is about the scenario, so it leaves accuracy alone and surfaces as
    applicability -- a criterion sharp on its own ground must not be dragged
    toward 0.5 by scenarios that never engage it."""
    seq = iter(["<choice>NA</choice>"] * 8 + ["<choice>A</choice>"] * 2)
    monkeypatch.setattr(det, "complete", lambda *a, **k: next(seq))
    out = det.per_criterion(None, "j", ["s"] * 10, ["t"] * 10, ["u"] * 10, ["c1"],
                            min_applicable=0.0, min_n=1)
    row = out["per_criterion"][0]
    assert row["n_applicable"] == 2 and row["applicable_rate"] == 0.2
    assert row["accuracy"] == 0.5      # both picks were A; sides alternate
    assert not row["untestable"]


def test_thin_applicability_is_untestable_not_scored(monkeypatch):
    monkeypatch.setattr(det, "complete", lambda *a, **k: "<choice>NA</choice>")
    out = det.per_criterion(None, "j", ["s"] * 40, ["t"] * 40, ["u"] * 40, ["c1"],
                            min_applicable=0.15, min_n=8)
    assert out["per_criterion"][0]["untestable"]
    assert out["mean_accuracy"] is None and out["n_testable"] == 0


def test_na_and_refusal_are_counted_separately(monkeypatch):
    """Collapsing them makes low applicability indistinguishable from judge
    refusal -- the ambiguity that matters most on a misalignment persona."""
    seq = iter(["<choice>NA</choice>", "not a tag at all", "<choice>A</choice>"])
    monkeypatch.setattr(det, "complete", lambda *a, **k: next(seq))
    out = det.whole(None, "j", ["s"] * 3, ["t"] * 3, ["u"] * 3, ["C"])
    assert out["n_applicable"] == 1
    assert out["n_not_applicable"] == 1
    assert out["n_unusable"] == 1


def test_refusal_falls_back_then_counts_as_na(monkeypatch):
    calls = []

    def flaky(llm, model, text, **k):
        calls.append(model)
        if model == "primary":
            raise RuntimeError("content_filter")
        return "<choice>A</choice>"

    monkeypatch.setattr(det, "complete", flaky)
    out = det.whole(None, "primary", ["s"], ["t"], ["u"], ["C"], fallback="backup")
    assert out["n_applicable"] == 1 and calls == ["primary", "backup"]


def test_control_persona_is_reported_and_warned_on(cfg, run_dir, monkeypatch, capsys):
    """An unrelated constitution should not detect this model. If it does, the
    judge is spotting the odd response out rather than reading the rubric."""
    import json

    import constitution_recovery.pipeline as pipe

    (run_dir / "criteria.json").write_text(json.dumps(["c1"]))
    (run_dir / "detection_responses.json").write_text(json.dumps(
        {"scenarios": ["s"], "trained": ["t"], "untrained": ["u"]}))
    steer = pathlib.Path("data/constitutions/steering")
    steer.mkdir(parents=True, exist_ok=True)
    (steer / "oct_goodness.json").write_text(json.dumps(["I am good."]))
    (steer / "oct_remorse.json").write_text(json.dumps(["I apologise."]))
    cfg["persona"] = "remorse"
    cfg["constitution"] = "data/constitutions/oct_remorse.json"
    cfg["experiment"]["detection"]["control_persona"] = "goodness"
    cfg["experiment"]["detection"]["per_criterion"] = False

    monkeypatch.setattr(pipe, "client", lambda *a, **k: None)
    from constitution_recovery.evaluation import detection as detmod
    monkeypatch.setattr(detmod, "whole",
                        lambda *a, **k: {"accuracy": 1.0, "n_applicable": 1,
                                         "applicable_rate": 1.0})
    pipe.stage_detection(cfg, run_dir, {})

    out = json.loads((run_dir / "detection.json").read_text())
    assert out["whole_control"]["accuracy"] == 1.0
    assert out["control_persona"] == "goodness"
    assert "WARNING" in capsys.readouterr().out


import pathlib  # noqa: E402
from tests.test_stages import cfg, run_dir  # noqa: E402,F401
