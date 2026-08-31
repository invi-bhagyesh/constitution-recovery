import random


def rng(seed):
    """A seeded generator, so a randomised pair set is reproducible from the
    config alone."""
    return random.Random(seed)
