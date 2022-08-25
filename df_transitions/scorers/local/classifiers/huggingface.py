"""
HuggingFace classifier scorer
------------------------------

TODO: complete
"""
from typing import List

from ...huggingface import BaseHFScorer


class HFClassifier(BaseHFScorer):
    def predict(self, request: str) -> dict:
        model_output = self.call_model(request)
        logits_list = model_output.logits
        predicted_class_id = logits_list.argmax().item()
        label = self.model.config.id2label[predicted_class_id]
        result = {label: 1}
        return result
