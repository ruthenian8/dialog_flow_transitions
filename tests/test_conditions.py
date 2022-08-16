import pytest

from df_engine.core import Context
from df_transitions.utils import INTENT_KEY
from df_transitions.types import Intent
from df_transitions.conditions import intent_detected


@pytest.mark.parametrize(
    ["input"],
    [
        ("a",),
        (Intent(name="a", examples=["a"]),),
        (["a", "b"],),
        ([Intent(name="a", examples=["a"]), Intent(name="b", examples=["b"])],),
    ],
)
def test_conditions(input, testing_actor):
    ctx = Context(framework_states={INTENT_KEY: {"scorer_a": {"a": 1, "b": 1}, "scorer_b": {"b": 1, "c": 1}}})
    assert intent_detected(input, threshold=1)(ctx, testing_actor) == True


@pytest.mark.parametrize(["input"], [(1,), (3.3,), ({"a", "b"},)])
def test_conds_invalid(input, testing_actor):
    ctx = Context()
    with pytest.raises(NotImplementedError):
        result = intent_detected(input, threshold=1)(ctx, testing_actor)
