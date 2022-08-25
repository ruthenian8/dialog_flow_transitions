from typing import List

import pytest
from pydantic import parse_obj_as
from df_engine.core import Context
from df_transitions.utils import INTENT_KEY
from df_transitions.types import Label, LabelCollection
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


@pytest.mark.parametrize(
    ["_input", "last_request", "thresh"],
    [
        ({"positive_examples": ["like sweets", "like candy"], "negative_examples": ["other stuff"]}, "sweets", 0.7),
        (
            {
                "positive_examples": ["good stuff", "brilliant stuff", "excellent stuff"],
                "negative_examples": ["negative example"],
            },
            "excellent, brilliant",
            0.5,
        ),
    ],
)
def test_has_match(_input: dict, testing_actor, thresh, standard_scorer, last_request):
    ctx = Context()
    ctx.add_request(last_request)
    # Per default, we assume that the model has already been fit.
    # For this test case we fit it manually.
    collection = LabelCollection(
        labels=parse_obj_as(List[Label], [{"name": key, "examples": values} for key, values in _input.items()])
    )
    standard_scorer.fit(collection)
    result = has_match(standard_scorer, threshold=thresh, **_input)(ctx, testing_actor)
    assert result
