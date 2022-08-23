import pytest

from df_engine.core import Context
from df_transitions.utils import INTENT_KEY
from df_transitions.types import Label
from df_transitions.conditions import has_cls_label, has_match


@pytest.mark.parametrize(
    ["input"],
    [
        ("a",),
        (Label(name="a", examples=["a"]),),
        (["a", "b"],),
        ([Label(name="a", examples=["a"]), Label(name="b", examples=["b"])],),
    ],
)
def test_conditions(input, testing_actor):
    ctx = Context(framework_states={INTENT_KEY: {"scorer_a": {"a": 1, "b": 1}, "scorer_b": {"b": 1, "c": 1}}})
    assert has_cls_label(input)(ctx, testing_actor) == True


@pytest.mark.parametrize(["input"], [(1,), (3.3,), ({"a", "b"},)])
def test_conds_invalid(input, testing_actor):
    with pytest.raises(NotImplementedError):
        result = has_cls_label(input)(Context(), testing_actor)


@pytest.mark.parametrize(["_input"], [({},), ({},)])
def test_has_match(_input, testing_actor, testing_scorer):
    result = has_match(testing_scorer, **_input)(Context(), testing_actor)
    assert True