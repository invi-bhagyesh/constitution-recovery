"""Proxy (b): preference-signal agreement on full constitutions, and proxy (d):
the Kendall tau between the two vectors.

The judge sees the WHOLE constitution as one rubric and picks per pair --
aggregation happens in the judge's head instead of arithmetically over the
per-criterion matrix. If this tracks the aggregate of proxy (a), the
per-criterion analysis is not missing interaction structure; divergence is
itself the reportable flag the spec names.
"""

from collections import Counter

from scipy.stats import kendalltau


def compare(vec_c, vec_cprime):
    assert len(vec_c) == len(vec_cprime), "vectors must come from the same pairs"
    tau, p = kendalltau(vec_c, vec_cprime)
    n = len(vec_c)
    dist = lambda v: {str(k): c / n for k, c in sorted(Counter(v).items())}  # noqa: E731
    return {
        "match_rate": sum(a == b for a, b in zip(vec_c, vec_cprime)) / n,
        "kendall_tau": float(tau),
        "tau_p": float(p),
        "n_pairs": n,
        "label_dist_c": dist(vec_c),
        "label_dist_cprime": dist(vec_cprime),
    }
