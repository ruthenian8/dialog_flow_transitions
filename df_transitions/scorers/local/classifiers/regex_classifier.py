"""
Regex Scorer
----------------

This module provides a regex-based label scorer version.
"""
import re
from typing import Optional

from ...base_scorer import BaseScorer
from ....types import LabelCollection


class RegexClassifier(BaseScorer):
    def __init__(
        self,
        namespace_key: str = "regex_scorer",
        label_collection: Optional[LabelCollection] = None,
        re_kwargs: Optional[dict] = None,
        default_score: float = 1.0,
    ) -> None:

        super().__init__(label_collection=label_collection, namespace_key=namespace_key)
        assert self.label_collection is not None, "label_collection parameter is required for RegexClassifier."
        self.re_kwargs = re_kwargs or {"flags": re.IGNORECASE}
        self.default_score = default_score

    def predict(self, request: str) -> str:
        result = {}
        for label_name, label in self.label_collection.labels.items():
            matches = [re.search(item, request, **self.re_kwargs) for item in label.examples]
            if any(map(bool, matches)):
                result[label_name] = self.default_score

        return result
