"""CEI against cases where the answer is known before any model is queried."""

import numpy as np
import pytest

from constitution_recovery.evaluation.cei import score

K, N = 200, 15


def rows(mat, tag):
    return [{"criterion": f"{tag}{i}", "labels": r.tolist()} for i, r in enumerate(mat)]


@pytest.fixture
def C():
    return np.random.default_rng(0).choice([-1, 0, 1], size=(N, K))


def test_permutation_is_faithful(C):
    perm = C[np.random.default_rng(1).permutation(N)]
    out = score(rows(C, "c"), rows(perm, "p"))
    assert out["cei"] > 0.9 and out["mode"] == "faithful"


def test_subset_is_truncated(C):
    out = score(rows(C, "c"), rows(C[:5], "s"))
    assert out["mode"] == "truncated" and out["uncovered_c"] > 0.2


def test_superset_is_bloated(C):
    extra = np.random.default_rng(2).choice([-1, 0, 1], size=(8, K))
    out = score(rows(C, "c"), rows(np.vstack([C, extra]), "x"))
    # CEI medians to 1.0 here -- bloat is only visible in the uncovered fraction
    assert out["mode"] == "bloated" and out["uncovered_cprime"] > 0.2


def test_random_recovers_nothing(C):
    rand = np.random.default_rng(3).choice([-1, 0, 1], size=(N, K))
    out = score(rows(C, "c"), rows(rand, "r"))
    assert out["cei"] < 0.1


def test_mismatched_pair_sets_raise(C):
    with pytest.raises(ValueError, match="same pairs"):
        score(rows(C, "c"), rows(C[:, :100], "short"))
