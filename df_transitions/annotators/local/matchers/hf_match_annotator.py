from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from ...base_hf_annotator import BaseHFAnnotator


class HFProximityAnnotator(BaseHFAnnotator):
    def get_intents(self, request: str) -> dict:
        request_cls_embedding = self.get_cls_embedding(request)
        result = dict()
        for intent_name, intent in self.intent_collection.intents.items():
            reference_examples = intent.examples
            reference_embeddings = [self.get_cls_embedding(item) for item in reference_examples]
            cosine_scores = [
                cosine_similarity(request_cls_embedding, ref_emb)[0][0] for ref_emb in reference_embeddings
            ]
            result[intent_name] = np.max(np.array(cosine_scores))

        return result
