"""Coverage: parsing, the PARTIAL weight, and that a refusal is not an absence."""

import pytest

import constitution_recovery.evaluation.coverage as cov


@pytest.fixture(autouse=True)
def _serial(monkeypatch):
    monkeypatch.setattr(cov, "pmap", lambda fn, items, w, desc=None: [fn(i) for i in items])


def test_parse_verdict_and_match():
    assert cov._parse("<verdict>YES</verdict>\n<match>3</match>") == ("YES", 3)
    assert cov._parse("<verdict>partial</verdict><match>0</match>") == ("PARTIAL", 0)
    assert cov._parse("<verdict>NO</verdict>") == ("NO", None)


def test_parse_junk_is_unusable_not_no():
    """A refusal must not read as a confident absence, or a content filter would
    present as truncation."""
    assert cov._parse("I cannot help with that")[0] == cov.UNUSABLE
    assert cov._parse(None)[0] == cov.UNUSABLE
    assert cov._parse("")[0] == cov.UNUSABLE


def test_all_present_is_full_coverage(monkeypatch):
    monkeypatch.setattr(cov, "complete",
                        lambda *a, **k: "<verdict>YES</verdict><match>1</match>")
    out = cov.direction(None, "j", None, ["a", "b", "c"], ["x"])
    assert out["coverage"] == 1.0 and out["full_rate"] == 1.0
    assert (out["n"], out["n_yes"], out["n_no"]) == (3, 3, 0)


def test_all_absent_is_zero(monkeypatch):
    monkeypatch.setattr(cov, "complete", lambda *a, **k: "<verdict>NO</verdict>")
    out = cov.direction(None, "j", None, ["a", "b"], ["x"])
    assert out["coverage"] == 0.0 and out["n_no"] == 2


def test_partial_is_weighted_and_rescorable(monkeypatch):
    seq = iter(["<verdict>YES</verdict>", "<verdict>PARTIAL</verdict>",
                "<verdict>PARTIAL</verdict>", "<verdict>NO</verdict>"])
    monkeypatch.setattr(cov, "complete", lambda *a, **k: next(seq))
    out = cov.direction(None, "j", None, list("abcd"), ["x"], partial_credit=0.5)
    assert out["coverage"] == pytest.approx((1 + 0.5 * 2) / 4)
    assert out["full_rate"] == 0.25                    # the strict reading
    assert (out["n_yes"], out["n_partial"], out["n_no"]) == (1, 2, 1)
    assert out["partial_credit"] == 0.5                # recorded, so it can be rescored


def test_unusable_leaves_the_denominator_alone(monkeypatch):
    """Two answered, one refused: coverage is 0.5 over 2, not 0.33 over 3."""
    seq = iter(["<verdict>YES</verdict>", "refused", "<verdict>NO</verdict>"])
    monkeypatch.setattr(cov, "complete", lambda *a, **k: next(seq))
    out = cov.direction(None, "j", None, list("abc"), ["x"])
    assert out["n"] == 2 and out["n_unusable"] == 1
    assert out["coverage"] == 0.5


def test_score_runs_both_directions_with_sides_swapped(monkeypatch):
    """recall looks for C in C'; precision looks for C' in C. The prompt must
    receive them the other way round on the second pass."""
    seen = []

    def spy(llm, model, text, **k):
        seen.append(text)
        return "<verdict>YES</verdict>"

    monkeypatch.setattr(cov, "complete", spy)
    out = cov.score(None, "j", ["C-ONE"], ["CP-ONE", "CP-TWO"])
    assert out["n_c"] == 1 and out["n_cprime"] == 2
    assert out["recall"]["n"] == 1 and out["precision"]["n"] == 2
    # first call: a C principle against the numbered C' list
    assert "C-ONE" in seen[0] and "1. CP-ONE" in seen[0] and "2. CP-TWO" in seen[0]
    # later calls: a C' principle against the numbered C list
    assert "CP-ONE" in seen[1] and "1. C-ONE" in seen[1]


def test_numbering_is_one_based():
    assert cov._numbered(["a", "b"]) == "1. a\n2. b"
