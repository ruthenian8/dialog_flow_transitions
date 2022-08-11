"""
Base Annotator
---------------
This module implements an abstract base class for intent annotators.
"""
from abc import ABC, abstractmethod
from typing import Optional

from df_engine.core import Context, Actor

from ..types import IntentCollection
from ..utils import INTENT_KEY


class BaseAnnotator(ABC):
    def __init__(self, intent_collection: Optional[IntentCollection] = None):
        self.intent_collection = intent_collection

    @abstractmethod
    def get_intents(self, request: str) -> dict:
        raise NotImplementedError

    def __call__(self, ctx: Context, actor: Actor):
        """
        Saves the retrieved intents to a subspace inside the `framework_states` field of the context.
        Creates the missing namespaces, if necessary.
        Suited for use with `df_runner` add-on.
        """
        intents: dict = self.get_intents(ctx.last_request)

        if INTENT_KEY not in ctx.framework_states:
            ctx.framework_states[INTENT_KEY] = dict()

        namespace = self.__class__.__name__.rstrip("Annotator")

        if namespace not in ctx.framework_states[INTENT_KEY]:
            ctx.framework_states[INTENT_KEY][namespace] = intents
        else:
            ctx.framework_states[INTENT_KEY][namespace].update(intents)
        return ctx
