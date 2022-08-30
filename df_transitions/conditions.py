"""
Conditions
-----------

This module provides condition functions for label processing.
"""
from typing import Callable, Optional, List
from functools import singledispatch

from sklearn.metrics.pairwise import cosine_similarity
from df_engine.core import Context, Actor

from .types import Label
from .utils import LABEL_KEY
from .models.base_model import BaseModel


@singledispatch
def has_cls_label(label, namespace: Optional[str] = None):
    """
    Use this condition, when you need to check, whether the probability
    of a particular label for the last user utterance surpasses the threshold.

    Parameters
    -----------
    label: Any
        String name or a reference to a Label object, or a collection thereof.
    namespace: Optional[str]
        Namespace key of a particular model that should detect the label.
        If not set, all namespaces will be searched for the required label.
    """
    raise NotImplementedError

# TODO: add treshholds

@has_cls_label.register(str)
def _(label, namespace: Optional[str] = None):
    def has_cls_label_innner(ctx: Context, actor: Actor) -> bool:
        if LABEL_KEY not in ctx.framework_states:
            return False
        if namespace is not None:
            return ctx.framework_states[LABEL_KEY].get(namespace, {}).get(label, 0) >= 1.0
        scores = [item.get(label, 0) for item in ctx.framework_states[LABEL_KEY].values()]
        comparison_array = [item >= 1.0 for item in scores]
        return any(comparison_array)

    return has_cls_label_innner


@has_cls_label.register(Label)
def _(label, namespace: Optional[str] = None) -> Callable[[Context, Actor], bool]:
    def has_cls_label_innner(ctx: Context, actor: Actor) -> bool:
        if LABEL_KEY not in ctx.framework_states:
            return False
        if namespace is not None:
            return ctx.framework_states[LABEL_KEY].get(namespace, {}).get(label.name, 0) >= 1.0
        scores = [item.get(label.name, 0) for item in ctx.framework_states[LABEL_KEY].values()]
        comparison_array = [item >= 1.0 for item in scores]
        return any(comparison_array)

    return has_cls_label_innner


@has_cls_label.register(list)
def _(label, namespace: Optional[str] = None):
    def has_cls_label_innner(ctx: Context, actor: Actor) -> bool:
        scores = [has_cls_label(item, namespace)(ctx, actor) for item in label]
        for score in scores:
            if score >= 1.0:
                return True
        return False

    return has_cls_label_innner


def has_match(
    model: BaseModel,
    positive_examples: Optional[List[str]],
    negative_examples: Optional[List[str]] = None,
    threshold: float = 0.9,
):
    """
    Use this condition, if you need to check whether the last utterance is close to some
    pre-defined phrases. N.B.: Note that the model you will use should be already fit by the time
    you pass it to the function.

    Parameters
    -----------
    model: BaseModel
        df_transitions' model. Use one of the models from the `cosine_scorers` subpackage.
    positive_examples: Optional[List[str]]
        A list of phrases that an utterance should be close to.
    negative_examples: Optional[List[str]] = None
        A list of phrases that an utterance should be distant from.
    threshold: float = 0.9
        The minimal cosine similarity to positive examples that is required
        for a positive response from the function.
    """
    if negative_examples is None:
        negative_examples = []

    def has_match_inner(ctx: Context, actor: Actor) -> bool:
        if not isinstance(ctx.last_request, str):
            return False
        input_vector = model.transform(ctx.last_request)
        positive_vectors = [model.transform(item) for item in positive_examples]
        negative_vectors = [model.transform(item) for item in negative_examples]
        positive_sims = [cosine_similarity(input_vector, item)[0][0] for item in positive_vectors]
        negative_sims = [cosine_similarity(input_vector, item)[0][0] for item in negative_vectors]
        max_pos_sim = max(positive_sims)
        max_neg_sim = 0 if len(negative_sims) == 0 else max(negative_sims)
        return max_pos_sim > threshold > max_neg_sim

    return has_match_inner
