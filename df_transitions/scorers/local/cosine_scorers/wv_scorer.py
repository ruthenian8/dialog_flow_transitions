"""
WV Cosine Scorer
-----------------

TODO: Implement
"""
from typing import Optional
from collections.abc import Iterable

from ...base_scorer import BaseCosineScorer
from ....types import LabelCollection


class WVScorer(BaseCosineScorer):
    def __init__(self, namespace_key: str, label_collection: Optional[LabelCollection]) -> None:
        super().__init__(namespace_key=namespace_key, label_collection=label_collection)

    def fit(self, request: str) -> Iterable:
        return super().fit(request)

    def predict(self, request: str) -> dict:
        return super().predict(request)
