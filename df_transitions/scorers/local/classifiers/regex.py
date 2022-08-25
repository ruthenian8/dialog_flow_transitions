"""
Regex Scorer
----------------

This module provides a regex-based label scorer version.
"""
import re
from typing import Optional

from ...base_scorer import BaseScorer
from ....types import LabelCollection


class RegexModel:
    def __init__(self, label_collection: LabelCollection):
        self.label_collection = label_collection

    def __call__(self, request: str, **re_kwargs):
        result = {}
        for label_name, label in self.label_collection.labels.items():
            matches = [re.search(item, request, **re_kwargs) for item in label.examples]
            if any(map(bool, matches)):
                result[label_name] = 1.0

        return result


class RegexClassifier(BaseScorer):
    def __init__(
        self,
        namespace_key: str,
        model: RegexModel,
        re_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__(namespace_key=namespace_key)
        self.re_kwargs = re_kwargs or {"flags": re.IGNORECASE}
        self.model = model

    def fit(self, label_collection: LabelCollection):
        self.model.label_collection = label_collection

    def predict(self, request: str) -> dict:
        return self.model(request, **self.re_kwargs)
