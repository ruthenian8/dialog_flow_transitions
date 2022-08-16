"""
Base Provider
---------------

This module implements an abstract base class for intent providers.
"""
from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel, Extra, Field
from df_engine.core import Context, Actor

from ..types import IntentCollection
from ..utils import INTENT_KEY


class BaseConfig(BaseModel, arbitrary_types_allowed=True, extra=Extra.allow):
    namespace_key: str = Field(default="base")

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


class BaseIntentScorer(ABC):
    def __init__(self, config: BaseConfig, intent_collection: Optional[IntentCollection] = None) -> None:

        self.__dict__.update(config.dict(by_alias=False))
        self.intent_collection = intent_collection

    @abstractmethod
    def analyze(self, request: str) -> dict:
        raise NotImplementedError

    def __call__(self, ctx: Context, actor: Actor):
        """
        Saves the retrieved intents to a subspace inside the `framework_states` field of the context.
        Creates the missing namespaces, if necessary.
        Suited for use with `df_runner` add-on.
        """
        intents: dict = self.analyze(ctx.last_request)

        if INTENT_KEY not in ctx.framework_states:
            ctx.framework_states[INTENT_KEY] = dict()

        namespace = self.namespace_key

        ctx.framework_states[INTENT_KEY][namespace] = intents

        return ctx
