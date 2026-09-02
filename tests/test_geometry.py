"""Criterion geometry: the math, on constructed vectors with known answers.

The GPU part of scripts/criterion_geometry.py cannot be tested here; what can be
is every claim the output makes -- that centring removes the shared component,
that the sign of cos(v_c, d) means what the figure says it means, and that the
control check would actually catch a noise result.
"""

import numpy as np
import pytest

from constitution_recovery.interp.geometry import (center, cosines, nearest, pca2,
                                                   pearson, spearman, tightness)


def test_centring_removes_the_shared_component():
    """Every v_c carries a large "a system prompt is present" offset. Left in, it
    dominates every cosine and the figure reports the offset, not the content."""
    shared = np.array([50.0, 50.0, 50.0])
    V = np.array([shared + [1, 0, 0], shared + [0, 1, 0], shared + [0, 0, 1]])
    d = np.array([1.0, -1.0, 0.0])

    # uncentred: the shared offset swamps the signal, everything looks alike
    raw = cosines(V, d)
    assert raw.max() - raw.min() < 0.05

    # centred: the criteria separate, and along d
    out = cosines(center(V), d)
    assert out[0] > 0.5 and out[1] < -0.5
    assert np.allclose(center(V).mean(axis=0), 0)


def test_cosine_sign_is_the_claim():
    """Positive = pushes the base the way training pushed the target; negative =
    pushes it away, which is what a reversed criterion should do."""
    d = np.array([1.0, 0.0])
    V = np.array([[2.0, 0.0], [0.0, 3.0], [-1.0, 0.0]])
    out = cosines(V, d)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(-1.0)


def test_cosine_ignores_magnitude():
    """A verbose criterion inducing a bigger shift must not outrank a terse one
    pointing the same way."""
    d = np.array([1.0, 1.0])
    assert cosines(np.array([[1.0, 1.0], [80.0, 80.0]]), d) == pytest.approx([1.0, 1.0])


def test_pca2_recovers_a_planted_plane():
    rng = np.random.default_rng(0)
    a, b = np.array([1.0, 0, 0, 0]), np.array([0, 1.0, 0, 0])
    V = np.array([x * a + y * b for x, y in rng.normal(size=(40, 2))])
    coords, explained = pca2(center(V))
    assert coords.shape == (40, 2)
    assert sum(explained) > 0.98      # the plane is the whole variance


def test_nearest_finds_the_match_and_scores_it():
    A = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    B = np.array([[0.0, 5.0], [2.0, 0.0]])
    idx, score = nearest(A, B)
    # the third row is anti-parallel to B[1] (cos -1) and orthogonal to B[0]
    # (cos 0), so orthogonal wins -- nearest reports the best available match,
    # which for a criterion with no counterpart is a weak one, not a wrong one.
    assert idx == [1, 0, 0]
    assert score[0] == pytest.approx(1.0)
    assert score[2] == pytest.approx(0.0)


def test_nearest_on_an_empty_target_does_not_raise():
    """C is read from disk and could be missing; the script should report that,
    not die inside the geometry."""
    assert nearest(np.array([[1.0, 0.0]]), np.zeros((0, 2))) == ([], [])


def test_control_check_catches_a_noise_result():
    """If C's criteria are no more alike than C-to-control, the geometry means
    nothing -- the check must show that, or the figure is unfalsifiable."""
    rng = np.random.default_rng(1)

    # signal: two tight, well-separated clusters
    C = np.array([1.0, 0, 0]) + rng.normal(scale=0.05, size=(6, 3))
    K = np.array([0, 1.0, 0]) + rng.normal(scale=0.05, size=(6, 3))
    t = tightness(np.vstack([C, K]), ["C"] * 6 + ["control"] * 6)
    assert t["within"]["C"] > 0.9
    assert t["between"]["C|control"] < 0.3

    # noise: one cloud, arbitrarily labelled
    N = rng.normal(size=(12, 8))
    t2 = tightness(N, ["C"] * 6 + ["control"] * 6)
    assert abs(t2["within"]["C"] - t2["between"]["C|control"]) < 0.35


def test_correlations_and_their_guards():
    x = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert pearson(x, [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    assert pearson(x, [5, 4, 3, 2, 1]) == pytest.approx(-1.0)
    # spearman survives the ceiling pile-up that detection accuracy actually has
    assert spearman(x, [0.4, 0.9, 1.0, 1.0, 1.0]) > 0.8
    # too few points, or a constant column, returns None rather than nan
    assert pearson([1, 2], [1, 2]) is None
    assert pearson(x, [1, 1, 1, 1, 1]) is None
    assert spearman([1, 2], [1, 2]) is None


def test_spearman_averages_ties():
    """Untied ranks would make a run of 1.00 detection accuracies look ordered."""
    assert spearman([1, 2, 3, 4], [1, 1, 1, 1]) is None
    assert spearman([1, 2, 3, 4], [1, 2, 2, 3]) == pytest.approx(spearman([1, 2, 3, 4],
                                                                          [1, 2, 2, 3]))
