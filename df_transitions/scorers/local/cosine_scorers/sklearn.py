"""
Sklearn Cosine Scorer
--------------------------

This module provides an adapter interface for sklearn models.
It uses Sklearn BOW representations and other features to compute distances between utterances.
"""
from typing import Optional

try:
    from sklearn.base import BaseEstimator

    IMPORT_ERROR_MESSAGE = None
except ImportError as e:
    BaseEstimator = object
    IMPORT_ERROR_MESSAGE = e.msg

from ...sklearn import BaseSklearnScorer
from ....types import LabelCollection
from .cosine_scorer_mixin import CosineScorerMixin


class SklearnScorer(CosineScorerMixin, BaseSklearnScorer):
    """
    label_collection: LabelCollection
        Expected labels.    
    """
    def __init__(
        self,
        model: Optional[BaseEstimator] = None,
        label_collection: Optional[LabelCollection] = None,
        tokenizer: Optional[BaseEstimator] = None,
        namespace_key: Optional[str] = None,
    ) -> None:
        CosineScorerMixin.__init__(self, label_collection=label_collection)
        BaseSklearnScorer.__init__(self, model=model, tokenizer=tokenizer, namespace_key=namespace_key)
