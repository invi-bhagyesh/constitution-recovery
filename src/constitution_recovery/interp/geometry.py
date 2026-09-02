"""Criterion geometry: where C and C' sit in the model's own activation space.

Every metric in this project reads TEXT -- detection judges responses, coverage
judges the constitutions against each other. This asks the model instead. Each
criterion, used alone as a system prompt, induces a shift in the base's residual
stream; that shift is the criterion's vector. The anchor is d, the direction
training actually moved the target away from its base.

Then cos(v_c, d) asks: does prompting the base with this criterion push it the
way training pushed the target? Which makes three things measurable that no
judge-based metric can reach:

  - a JUDGE-FREE surrogate for detection. If cos correlates with per-criterion
    detection accuracy, the judge was reading something real in the model, at
    ~2.4k prefills instead of ~2.9k judge calls.
  - a mechanistic signature for the REVERSED criteria. The hedged inversions
    (detection 0.010-0.048) should have NEGATIVE cos: they describe the opposite
    of what training installed.
  - whether the ORPHANED criteria of C are installed at all. Coverage scored
    five of C's ten NO in both methods -- the affect and intent ones. If those
    still have high cos with d, the traits are in the model and behaviour-only
    recovery missed them, which is the structural gap rather than a probing
    failure. If they do not, C was never fully installed and it is not ground
    truth for what the organism holds.

Weights are needed, so this is an ORACLE, not a verification method: no external
auditor holding an API key could run it. Its job is to say whether the
behavioural answer was right.

Pure numpy, so the math is testable without a GPU.
"""

import numpy as np


def center(V):
    """Mean-centre criterion vectors across the SET.

    Not optional. Every v_c carries a large shared "a system prompt is present
    now" component; left in, it dominates the geometry, everything clusters, and
    the figure reports the presence of a system prompt rather than its content.
    """
    V = np.asarray(V, dtype=np.float64)
    return V - V.mean(axis=0, keepdims=True)


def _unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def cosines(V, d):
    """cos(v_c, d) per row. In high dimension the absolute value is small; the
    ranking and the sign are what carry the claim."""
    return (_unit(np.asarray(V, dtype=np.float64)) @ _unit(np.asarray(d, dtype=np.float64)))


def pca2(V):
    """Top two principal components of the criterion set, for the figure.

    Returns (coords (n,2), explained fraction per component). V should already be
    centred; SVD on the centred matrix is the PCA.
    """
    V = np.asarray(V, dtype=np.float64)
    U, S, _ = np.linalg.svd(V, full_matrices=False)
    total = float((S ** 2).sum())
    coords = U[:, :2] * S[:2]
    explained = ((S[:2] ** 2) / total).tolist() if total > 0 else [0.0, 0.0]
    return coords, explained


def nearest(A, B):
    """For each row of A, the index of the most cosine-similar row of B and the
    score. The geometric counterpart of coverage's judge-chosen <match> index --
    two independent alignments, so agreement validates both and disagreement
    names the criteria that are semantically close but mechanistically not."""
    if len(B) == 0:
        return [], []
    S = _unit(np.asarray(A, dtype=np.float64)) @ _unit(np.asarray(B, dtype=np.float64)).T
    idx = S.argmax(axis=1)
    return idx.tolist(), S[np.arange(len(A)), idx].tolist()


def pearson(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _rank(x):
    """Average ranks, so ties do not distort Spearman."""
    x = np.asarray(x, dtype=np.float64)
    order = x.argsort()
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    # average tied groups
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return ranks


def spearman(x, y):
    """Reported beside Pearson because detection accuracy is bounded and piles up
    at 1.0 and 0.0 -- a monotone relationship is the honest claim there."""
    if len(x) < 3:
        return None
    return pearson(_rank(x), _rank(y))


def tightness(V, groups):
    """Mean pairwise cosine within each group and between each pair of groups.

    THE control check. If C's criteria are no more similar to each other than to
    an unrelated constitution's, the geometry is noise and nothing else in the
    output means anything -- the same role control_persona plays in detection,
    and the reason those numbers are quotable.
    """
    V = _unit(np.asarray(V, dtype=np.float64))
    names = sorted(set(groups))
    idx = {g: np.array([i for i, x in enumerate(groups) if x == g]) for g in names}
    within, between = {}, {}
    for g in names:
        I = idx[g]
        if len(I) > 1:
            S = V[I] @ V[I].T
            iu = np.triu_indices(len(I), k=1)
            within[g] = float(S[iu].mean())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            between[f"{a}|{b}"] = float((V[idx[a]] @ V[idx[b]].T).mean())
    return {"within": within, "between": between}
