"""
Base Provider
---------------

This module implements an abstract base class for intent providers.
"""
from abc import ABC, abstractmethod
from typing import Optional

from df_engine.core import Context, Actor

from ..types import LabelCollection
from ..utils import INTENT_KEY


class BaseScorer(ABC):
    def __init__(self, namespace_key: str = "default", label_collection: Optional[LabelCollection] = None) -> None:
        self.namespace_key = namespace_key
        self.label_collection = label_collection

    @abstractmethod
    def predict(self, request: str) -> dict:
        raise NotImplementedError

    def __call__(self, ctx: Context, actor: Actor):
        """
        Saves the retrieved labels to a subspace inside the `framework_states` field of the context.
        Creates the missing namespaces, if necessary.
        Suited for use with `df_runner` add-on.
        """
        labels: dict = self.predict(ctx.last_request) if ctx.last_request else dict()

        if INTENT_KEY not in ctx.framework_states:
            ctx.framework_states[INTENT_KEY] = dict()

        namespace = self.namespace_key

        ctx.framework_states[INTENT_KEY][namespace] = labels

        return ctx


class BaseCosineScorer(BaseScorer):
    @abstractmethod
    def fit(self, request: str):
        raise NotImplementedError
