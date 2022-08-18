"""
Conditions
-----------

This module provides condition functions for label processing.
"""
from typing import Callable, Optional
from functools import singledispatch

from df_engine.core import Context, Actor

from .types import Label
from .utils import INTENT_KEY


@singledispatch
def has_cls_label(label, namespace: Optional[str] = None):
    raise NotImplementedError


@has_cls_label.register(str)
def _(label, namespace: Optional[str] = None):
    def has_cls_label_innner(ctx: Context, actor: Actor) -> bool:
        if INTENT_KEY not in ctx.framework_states:
            return False
        if namespace is not None:
            return ctx.framework_states[INTENT_KEY].get(namespace, {}).get(label, 0) >= 1.0
        scores = [item.get(label, 0) for item in ctx.framework_states[INTENT_KEY].values()]
        comparison_array = [item >= 1.0 for item in scores]
        return any(comparison_array)

    return has_cls_label_innner


@has_cls_label.register(Label)
def _(label, namespace: Optional[str] = None) -> Callable[[Context, Actor], bool]:
    def has_cls_label_innner(ctx: Context, actor: Actor) -> bool:
        if INTENT_KEY not in ctx.framework_states:
            return False
        if namespace is not None:
            return ctx.framework_states[INTENT_KEY].get(namespace, {}).get(label.name, 0) >= 1.0
        scores = [item.get(label.name, 0) for item in ctx.framework_states[INTENT_KEY].values()]
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
