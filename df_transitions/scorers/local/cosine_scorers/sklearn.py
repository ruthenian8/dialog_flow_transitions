"""
TF-IDF Cosine Scorer
----------------------

TODO: Implement
"""
from typing import Optional
from collections.abc import Iterable

try:
    from sklearn.base import BaseEstimator

    IMPORT_ERROR_MESSAGE = None
except ImportError as e:
    BaseEstimator = object
    IMPORT_ERROR_MESSAGE = e.msg

from ...sklearn import BaseSklearnScorer
from ....types import LabelCollection
from .cosine_scorer_mixin import CosineScorerMixin


class SklearnScorer(BaseSklearnScorer, CosineScorerMixin):
    def __init__(
        self,
        model: Optional[BaseEstimator] = None,
        label_collection: Optional[LabelCollection] = None,
        tokenizer: Optional[BaseEstimator] = None,
        namespace_key: Optional[str] = None,
    ) -> None:
        BaseSklearnScorer.__init__(self, model=model, tokenizer=tokenizer, namespace_key=namespace_key)
        CosineScorerMixin.__init__(self, label_collection=label_collection)

