"""
Gensim Cosine Model
--------------------------

This module provides an adapter interface for Gensim models.
We use word2vec embeddings to compute distances between utterances.
"""
from typing import Optional, Callable, List
import joblib

try:
    import gensim
    from gensim.models.word2vec import Word2Vec
    import numpy as np

    IMPORT_ERROR_MESSAGE = None
    ALL_MODELS = [name for name in dir(gensim.models) if name[0].isupper()]  # all classes
except ImportError as e:
    Word2Vec = object
    IMPORT_ERROR_MESSAGE = e.msg
    ALL_MODELS = []

from ...base_model import BaseModel
from ....types import LabelCollection
from ....utils import DefaultTokenizer
from .cosine_matcher_mixin import CosineMatcherMixin


class GensimMatcher(CosineMatcherMixin, BaseModel):
    """
    GensimMatcher utilizes embeddings from Gensim models to measure
    proximity between utterances and pre-defined labels.

    Parameters
    -----------
    model: gensim.models.word2vec.Word2Vec
        Gensim vector model (Word2Vec, FastText, etc.)
    label_collection: LabelCollection
        Labels for the matcher. The prediction output depends on proximity to different labels.
    tokenizer: Optional[Callable[[str], List[str]]]
        Class or function that performs string tokenization.
    namespace_key: Optional[str]
        Name of the namespace in framework states that the model will be using.
    kwargs:
        Keyword arguments are forwarded to the model constructor.
    """

    def __init__(
        self,
        model: Word2Vec,
        label_collection: LabelCollection,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        namespace_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        IMPORT_ERROR_MESSAGE = globals().get("IMPORT_ERROR_MESSAGE")
        if IMPORT_ERROR_MESSAGE is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE)
        CosineMatcherMixin.__init__(self, label_collection=label_collection)
        BaseModel.__init__(self, namespace_key=namespace_key)
        self.model = model
        self.tokenizer = tokenizer or DefaultTokenizer()
        # self.fit(self.label_collection, **kwargs)

    def transform(self, request: str):
        return self.model.wv.get_mean_vector(self.tokenizer(request))

    def fit(self, label_collection: LabelCollection, **kwargs) -> None:
        sentences = [example for label in label_collection.labels.values() for example in label.examples]
        tokenized_sents = list(map(self.tokenizer, sentences))
        self.model = self.model.__class__(**kwargs)
        self.model.build_vocab(tokenized_sents)
        self.model.train(tokenized_sents, total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def save(self, path: str):
        self.model.save(path)
        joblib.dump(self.label_collection, f"{path}.labels")
        joblib.dump(self.tokenizer, f"{path}.tokenizer")

    @classmethod
    def load(cls, path: str, namespace_key: str) -> __qualname__:
        with open(path, "rb") as picklefile:
            contents = picklefile.readline()  # get the header line, find the class name inside
        for name in ALL_MODELS:
            if bytes(name, encoding="uft-8") in contents:
                model_cls: type = getattr(gensim.models, name)
                break
        else:
            raise ValueError(f"No matching model found for file {path}")

        model = model_cls.load(path)
        label_collection = joblib.load(f"{path}.labels")
        tokenizer = joblib.load(f"{path}.tokenizer")
        return cls(model=model, tokenizer=tokenizer, label_collection=label_collection, namespace_key=namespace_key)
