"""
HuggingFace classifier scorer
------------------------------

TODO: complete
"""
from typing import List

from ...base_hf_scorer import BaseHFScorer


class HFClassifier(BaseHFScorer):
    def predict(self, request: str) -> dict:
        model_output = self.call_model(request)
        logits_list: List[float] = model_output.tolist()[0]
        result = {categorical_code: score for categorical_code, score in enumerate(logits_list)}
        return result
