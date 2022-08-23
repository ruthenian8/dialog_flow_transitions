"""
WV Cosine Scorer
-----------------

TODO: Implement
"""
from typing import Optional
import joblib

try:
    import gensim
    from gensim.models.word2vec import Word2Vec
    import numpy as np
except ImportError as e:
    Word2Vec = object
    IMPORT_ERROR_MESSAGE = e.msg

from ...base_scorer import BaseScorer
from ....types import LabelCollection
from ....utils import DefaultTokenizer
from .cosine_scorer_mixin import CosineScorerMixin


class GensimScorer(BaseScorer, CosineScorerMixin):
    def __init__(
        self,
        model: Word2Vec,
        label_collection: LabelCollection,
        tokenizer: Optional[object] = None,
        namespace_key: Optional[str] = None,
    ) -> None:
        if IMPORT_ERROR_MESSAGE is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE)
        BaseScorer.__init__(self, namespace_key=namespace_key)
        CosineScorerMixin.__init__(self, label_collection=label_collection)
        self.model = model
        self.tokenizer = tokenizer or DefaultTokenizer()

    def transform(self, request: str):
        return self.model.wv.get_mean_vector(self.tokenizer(request))

    def fit(self, label_collection: LabelCollection, **kwargs) -> None:
        sentences = [example for label in label_collection.labels.values() for example in label.examples]
        self.model = self.model.__class__(**kwargs)
        self.model.build_vocab(sentences=sentences)
        self.model.train(sentences, total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def save(self, path: str):
        self.model.save(path)
        joblib.dump(self.tokenizer, f"{path}.tokenizer")

    @classmethod
    def load(cls, path: str, model_cls: type, namespace_key: str) -> __qualname__:
        model = model_cls.load(path)
        tokenizer = joblib.load(f"{path}.tokenizer")
        return cls(model=model, tokenizer=tokenizer)


