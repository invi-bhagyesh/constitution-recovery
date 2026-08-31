"""Both parsers guess nothing: an unrecognised reply is None or dropped."""

from constitution_recovery.evaluation.labels import parse
from constitution_recovery.recovery.extract import criteria


def test_choice_tags():
    assert parse("<choice>A</choice>") == 1
    assert parse("<choice>B</choice>") == -1
    assert parse("<choice> tie </choice>") == 0


def test_preamble_does_not_become_a_vote():
    # A first-character parse would read this as a confident vote for B.
    assert parse("Based on the criterion, A is better.") is None
    assert parse("Based on the criterion, <choice>A</choice>") == 1


def test_criteria_extraction():
    assert criteria("<criterion>x</criterion><criterion>y</criterion>") == ["x", "y"]
    assert criteria("Here are the criteria:\n<criterion>x</criterion>") == ["x"]
    assert criteria("<criterion>a\nb</criterion>") == ["a b"]   # multi-line collapses
    assert criteria("- untagged") == []                          # loud empty
