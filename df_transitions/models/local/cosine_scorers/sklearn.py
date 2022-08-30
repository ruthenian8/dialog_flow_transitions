"""
Sklearn Cosine Scorer
--------------------------

This module provides an adapter interface for sklearn models.
It uses Sklearn BOW representations and other features to compute distances between utterances.
"""
from typing import Optional, Union

try:
    from sklearn.base import BaseEstimator
    from sklearn.pipeline import Pipeline

    IMPORT_ERROR_MESSAGE = None
except ImportError as e:
    BaseEstimator = object
    IMPORT_ERROR_MESSAGE = e.msg

from ...sklearn import BaseSklearnModel
from ....types import LabelCollection
from .cosine_scorer_mixin import CosineScorerMixin


class SklearnScorer(CosineScorerMixin, BaseSklearnModel):
    """
    SklearnScorer utilizes embeddings from Sklearn models to measure
    proximity between utterances and pre-defined labels.

    Parameters
    -----------
    model: Optional[BaseEstimator]
        Sklearn-type model
    label_collection: LabelCollection
        Labels for the scorer. The prediction output depends on proximity to different labels.
    tokenizer: Optional[Union[BaseEstimator, Pipeline]] = None
        An Sklearn-type tokenizer, like TdidfVectorizer or CountVectorizer. Can also be a product
        of several preprocessors, unified with a pipeline.
    namespace_key: Optional[str]
        Name of the namespace in framework states that the model will be using.
    """

    def __init__(
        self,
        model: Optional[BaseEstimator] = None,
        label_collection: Optional[LabelCollection] = None,
        tokenizer: Optional[Union[BaseEstimator, Pipeline]] = None,
        namespace_key: Optional[str] = None,
    ) -> None:
        CosineScorerMixin.__init__(self, label_collection=label_collection)
        BaseSklearnModel.__init__(self, model=model, tokenizer=tokenizer, namespace_key=namespace_key)
