"""
TODO: implement
"""
from typing import Optional

from pydantic import Field, validator

from ...base_scorer import BaseScorer, BaseConfig
from ....types import LabelCollection


class SklearnClassifierConfig(BaseConfig):
    pass


class SklearnClassifier(BaseScorer):
    def __init__(self, config: SklearnClassifierConfig, label_collection: Optional[LabelCollection] = None) -> None:
        super().__init__(config=config, label_collection=label_collection)

        assert self.label_collection is not None, "label_collection parameter is required"

    def predict(self, request: str):
        return dict()
