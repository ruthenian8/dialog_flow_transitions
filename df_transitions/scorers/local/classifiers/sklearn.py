"""
TODO: implement
"""
from ...sklearn import BaseSklearnScorer


class SklearnClassifier(BaseSklearnScorer):
    def predict(self, request: str) -> dict:
        label = self._pipeline.predict([request])[0]
        return {label: 1}
