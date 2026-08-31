"""Pull criteria out of a model's prose."""

import re

CRITERION = re.compile(r"<criterion>(.*?)</criterion>", re.DOTALL | re.IGNORECASE)


def criteria(text):
    """Tagged, not line-split: a preamble line would otherwise become a criterion
    and cost |pairs| judge calls downstream. Whitespace is collapsed so a
    criterion that wrapped across lines stays one string. DOTALL matters for the
    same reason."""
    found = [" ".join(m.split()) for m in CRITERION.findall(text)]
    # Exact dedup, order preserved: a 7B in a repetition loop emits the same
    # string hundreds of times (observed: 273 tags, 14 unique). Identical strings
    # from one generation are stutter, not content. Near-duplicates are kept --
    # deciding two different sentences mean the same thing is semantics, and
    # that judgement belongs to CEI, not the parser.
    return list(dict.fromkeys(found))
