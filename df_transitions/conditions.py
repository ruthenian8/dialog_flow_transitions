"""
Conditions
-----------

This module provides condition functions for intent processing.
"""
from typing import Callable
from functools import singledispatch

from df_engine.core import Context, Actor

from .types import Intent
from .utils import INTENT_KEY


@singledispatch
def intent_detected(intent, threshold: float = 0.95):
    raise NotImplementedError


@intent_detected.register(str)
def _(intent, threshold: float = 0.95):
    def intent_detected_innner(ctx: Context, actor: Actor) -> bool:
        if INTENT_KEY not in ctx.framework_states:
            return False
        scores = [item.get(intent, 0) for item in ctx.framework_states[INTENT_KEY].values()]
        comparison_array = [item >= threshold for item in scores]
        return any(comparison_array)

    return intent_detected_innner


@intent_detected.register(Intent)
def _(intent, threshold: float = 0.95) -> Callable[[Context, Actor], bool]:
    def intent_detected_innner(ctx: Context, actor: Actor) -> bool:
        if INTENT_KEY not in ctx.framework_states:
            return False
        scores = [item.get(intent.name, 0) for item in ctx.framework_states[INTENT_KEY].values()]
        comparison_array = [item >= threshold for item in scores]
        return any(comparison_array)

    return intent_detected_innner


@intent_detected.register(list)
def _(intent, threshold: float = 0.95):
    def intent_detected_innner(ctx: Context, actor: Actor) -> bool:
        scores = [intent_detected(item)(ctx, actor) for item in intent]
        for score in scores:
            if score >= threshold:
                return True
        return False

    return intent_detected_innner
