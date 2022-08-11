from typing import Optional

try:
    import numpy as np
    from tokenizers import Tokenizer
    from transformers.modeling_utils import PreTrainedModel

    IMPORT_ERROR_MESSAGE = None
except ImportError as e:
    np = None
    Tokenizer = None
    PreTrainedModel = None
    IMPORT_ERROR_MESSAGE = e.msg

from .base_annotator import BaseAnnotator
from ..types import IntentCollection


class BaseHFAnnotator(BaseAnnotator):
    def __init__(
        self,
        tokenizer: Tokenizer,
        model: PreTrainedModel,
        device: object,
        intent_collection: IntentCollection,
        tokenizer_kwargs: dict = None,
        model_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__(intent_collection=intent_collection)
        if IMPORT_ERROR_MESSAGE is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE)
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self._tokenizer_kwargs: dict = tokenizer_kwargs if tokenizer_kwargs else dict()
        self._tokenizer_kwargs["return_tensors"] = "pt"
        self._model_kwargs: dict = model_kwargs if model_kwargs else dict()

    def get_cls_embedding(self, input_sentence: str) -> np.ndarray:
        tokenized_examples = self.tokenizer(input_sentence, **self._tokenizer_kwargs)
        output = self.model(
            **tokenized_examples.to(self.device), **{**self._model_kwargs, "output_hidden_states": True}
        )
        return (
            output.hidden_states[-1][0, 0, :].detach().numpy().reshape(1, -1)
        )  # reshape for cosine similarity to be applicable

    def get_prediction(self, input_sentence: str):
        tokenized_examples = self.tokenizer(input_sentence, **self._tokenizer_kwargs)
        output = self.model(
            **tokenized_examples.to(self.device), **{**self._model_kwargs, "output_hidden_states": False}
        )
        return output
