"""
Conditions
-----------

This module provides condition functions for intent processing.
"""
from typing import Callable

from df_engine.core import Context, Actor

from .types import Intent
from .utils import INTENT_KEY


def user_has_intent(intent: Intent, threshold: float = 0.95) -> Callable[[Context, Actor], bool]:
    def user_has_intent_innner(ctx: Context, actor: Actor) -> bool:
        if INTENT_KEY not in ctx.framework_states:
            return False
        scores = [item.get(intent.name, 0) for item in ctx.framework_states[INTENT_KEY].values()]
        comparison_array = [item >= threshold for item in scores]
        return any(comparison_array)

    return user_has_intent_innner
