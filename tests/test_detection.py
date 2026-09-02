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
                            min_applicable=1)
    row = out["per_criterion"][0]
    assert row["n_applicable"] == 2 and row["applicable_rate"] == 0.2
    assert row["accuracy"] == 0.5      # both picks were A; sides alternate
    assert not row["untestable"]


def test_thin_applicability_is_untestable_not_scored(monkeypatch):
    monkeypatch.setattr(det, "complete", lambda *a, **k: "<choice>NA</choice>")
    out = det.per_criterion(None, "j", ["s"] * 40, ["t"] * 40, ["u"] * 40, ["c1"],
                            min_applicable=15)
    assert out["per_criterion"][0]["untestable"]
    assert out["mean_accuracy"] is None and out["n_testable"] == 0


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
