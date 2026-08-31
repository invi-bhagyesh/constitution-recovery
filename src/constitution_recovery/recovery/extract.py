"""Pull criteria out of a model's prose."""

import re

CRITERION = re.compile(r"<criterion>(.*?)</criterion>", re.DOTALL | re.IGNORECASE)


def criteria(text):
    """Tagged, not line-split: a preamble line would otherwise become a criterion
    and cost |pairs| judge calls downstream. Whitespace is collapsed so a
    criterion that wrapped across lines stays one string. DOTALL matters for the
    same reason."""
    return [" ".join(m.split()) for m in CRITERION.findall(text)]
