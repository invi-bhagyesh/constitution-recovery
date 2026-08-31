"""Constitution Equivalence Index.

Each criterion is a vector of {-1,0,+1} labels over the shared pair set, so "are
C and C' equivalent?" becomes "does each span the other?". Two ridge regressions
answer it, and the asymmetry between them is the failure mode.
"""

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict

ALPHAS = np.logspace(-3, 3, 13)


def matrix(rows):
    return [r["criterion"] for r in rows], np.array([r["labels"] for r in rows], dtype=float)


def spanned(Y, X, folds=5):
    """Out-of-fold R^2 for each row of Y predicted from the rows of X. In-sample
    R^2 with |C| predictors would be optimistic and would bias every candidate
    toward faithful."""
    predictors = X.T
    scores = []
    for y in Y:
        if np.std(y) == 0:  # a criterion that never discriminates predicts nothing
            scores.append(0.0)
            continue
        pred = cross_val_predict(RidgeCV(alphas=ALPHAS), predictors, y, cv=folds)
        scores.append(float(np.clip(r2_score(y, pred), 0.0, 1.0)))
    return np.array(scores)


def classify(cei, uncovered_cprime, uncovered_c, tol=0.2):
    """CEI aggregates by median, so a minority of novel criteria cannot move it --
    a C' containing all of C plus extras still medians to 1.0. Bloat and
    truncation are read off the fraction of uncovered criteria in each direction
    instead. Incoherence is not detectable from R^2."""
    if uncovered_cprime > tol and uncovered_c > tol:
        return "drifted" if 0.5 <= cei <= 0.85 else "unclassified"
    if uncovered_cprime > tol:
        return "bloated"       # C' has criteria C does not span
    if uncovered_c > tol:
        return "truncated"     # C has criteria C' does not span
    if cei > 0.9:
        return "faithful"
    return "drifted" if 0.5 <= cei <= 0.85 else "unclassified"


def score(rows_c, rows_cprime, folds=5, covered=0.5, tol=0.2):
    names_c, C = matrix(rows_c)
    names_cp, Cp = matrix(rows_cprime)
    if C.shape[1] != Cp.shape[1]:
        raise ValueError(
            f"pair sets differ: {C.shape[1]} vs {Cp.shape[1]} -- labels must come "
            "from the same pairs"
        )

    forward = spanned(Cp, C, folds)   # is each recovered criterion inside C's span
    reverse = spanned(C, Cp, folds)   # is each stated criterion inside C's span
    med_f, med_r = float(np.median(forward)), float(np.median(reverse))
    cei = 0.0 if med_f + med_r == 0 else 2 * med_f * med_r / (med_f + med_r)
    unc_f, unc_r = float(np.mean(forward < covered)), float(np.mean(reverse < covered))

    return {
        "cei": cei,
        "median_r2_cprime_given_c": med_f,
        "median_r2_c_given_cprime": med_r,
        "uncovered_cprime": unc_f,
        "uncovered_c": unc_r,
        "mode": classify(cei, unc_f, unc_r, tol),
        "n_pairs": C.shape[1],
        "per_criterion": {
            "cprime_given_c": [
                {"criterion": n, "r2": float(v)} for n, v in zip(names_cp, forward)
            ],
            "c_given_cprime": [
                {"criterion": n, "r2": float(v)} for n, v in zip(names_c, reverse)
            ],
        },
    }
