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
        assert len(logits_list) == len(
            self.label_collection.intents
        ), "Number of predicted labels does not match the number of registered intents."
        result = {intent_name: score for intent_name, score in zip(self.label_collection.intents.keys(), logits_list)}
        return result
