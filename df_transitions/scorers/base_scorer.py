"""
Base Provider
---------------

This module implements an abstract base class for intent providers.
"""
from abc import ABC, abstractmethod

from df_engine.core import Context, Actor

from ..types import LabelCollection
from ..utils import INTENT_KEY


class BaseScorer(ABC):
    """
    This class implements an abstract base class for label scoring.

    Parameters
    -----------
    namespace_key: str
        Name of the namespace in framework states that the model will be using.

    """

    def __init__(self, namespace_key: str = "default") -> None:
        self.namespace_key = namespace_key

    @abstractmethod
    def predict(self, request: str) -> dict:
        """
        Predict the probability of one or several classes.
        """
        raise NotImplementedError

    def transform(self, request: str):
        """
        Get a representation of the input data.
        """
        raise NotImplementedError

    def fit(self, label_collection: LabelCollection) -> None:
        """
        Reinitialize the inner model with the given data.
        """
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

    def save(self, path: str, **kwargs) -> None:
        """
        Save the model to a specified location.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, path: str, namespace_key: str) -> __qualname__:
        """
        Load a model from the specified location and instantiate the scorer.
        """
        raise NotImplementedError
