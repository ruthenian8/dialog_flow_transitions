"""
HuggingFace Cosine Scorer
--------------------------

TODO: Complete
"""
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from ...base_hf_scorer import BaseHFScorer


class HFCosineScorer(BaseHFScorer):
    def predict(self, request: str) -> dict:
        request_cls_embedding = self.fit(request)
        result = dict()
        for label_name, label in self.label_collection.labels.items():
            reference_examples = label.examples
            reference_embeddings = [self.fit(item) for item in reference_examples]
            cosine_scores = [
                cosine_similarity(request_cls_embedding, ref_emb)[0][0] for ref_emb in reference_embeddings
            ]
            result[label_name] = np.max(np.array(cosine_scores))

        return result
