"""
run tests
"""
import pytest
from recipe_learning import Statement, anti_unify


def test_anti_unification():
    """
    Test that anti-unification works properly
    """

    # first try with two statements that are the same
    s1 = Statement(
        [{"color": "green", "shape": "triangle"}, {"color": "blue", "shape": "pentagon"}],
        {"color": "red", "shape": "square"}
    )
    s2 = Statement(
        [{"color": "green", "shape": "triangle"}, {"color": "blue", "shape": "pentagon"}],
        {"color": "red", "shape": "square"}
    )
    assert anti_unify([s1, s2]) == s1

    # two concrete statements with a common property (color)
    s1 = Statement(
        [{"color": "green", "shape": "triangle"}, {"color": "blue", "shape": "pentagon"}],
        {"color": "red", "shape": "square"}
    )
    s2 = Statement(
        [{"color": "green", "shape": "square"}, {"color": "blue", "shape": "triangle"}],
        {"color": "red", "shape": "square"}
    )
    s0 = Statement(
        [{"color": "green"}, {"color": "blue"}],
        {"color": "red", "shape": "square"}
    )
    assert anti_unify([s1, s2]) == s0

    # two statements that don't share an output don't anti-unify
    s1 = Statement(
        [{"color": "green", "shape": "triangle"}, {"color": "blue", "shape": "pentagon"}],
        {"color": "green", "shape": "square"}
    )
    s2 = Statement(
        [{"color": "green", "shape": "square"}, {"color": "blue", "shape": "triangle"}],
        {"color": "red", "shape": "square"}
    )
    assert anti_unify([s1, s2]) is None

    # two statements with totally different inputs anti-unify to something very general
    s1 = Statement(
        [{"color": "green", "shape": "triangle"}, {"color": "blue", "shape": "pentagon"}],
        {"color": "red", "shape": "square"}
    )
    s2 = Statement(
        [{"color": "blue", "shape": "square"}, {"color": "red", "shape": "triangle"}],
        {"color": "red", "shape": "square"}
    )
    s0 = Statement(
        [{}, {}],
        {"color": "red", "shape": "square"}
    )
    assert anti_unify([s1, s2]) == s0
