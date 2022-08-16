"""
Base HF Provider
------------------

Module
"""
from argparse import Namespace
from typing import Optional

from pydantic import Field, validator, validate_arguments

try:
    import numpy as np
    from tokenizers import Tokenizer
    from transformers.modeling_utils import PreTrainedModel
    import torch

    IMPORT_ERROR_MESSAGE = None
except ImportError as e:
    np = Namespace(ndarray=None)
    torch = Namespace(device=None)
    Tokenizer = None
    PreTrainedModel = None
    IMPORT_ERROR_MESSAGE = e.msg

from .base_scorer import BaseIntentScorer, BaseConfig
from ..types import IntentCollection


class HFScorerConfig(BaseConfig):
    tokenizer: Tokenizer
    model: PreTrainedModel
    device: torch.device
    tokenizer_kwargs: dict = Field(default_factory=dict)
    model_kwargs: dict = Field(default_factory=dict)
    namespace_key: str = "hugging_face_scorer"

    @validator("tokenizer_kwargs")
    def add_tensor_arg(cls, value: dict):
        value["return_tensors"] = "pt"
        return value


class BaseHFScorer(BaseIntentScorer):
    @validate_arguments
    def __init__(self, config: HFScorerConfig, intent_collection: Optional[IntentCollection] = None) -> None:
        IMPORT_ERROR_MESSAGE = globals().get("IMPORT_ERROR_MESSAGE")
        if IMPORT_ERROR_MESSAGE is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE)
        super().__init__(config=config, intent_collection=intent_collection)

    def vectorize(self, input_sentence: str) -> np.ndarray:
        tokenized_examples = self.tokenizer(input_sentence, **self.tokenizer_kwargs)
        output = self.model(**tokenized_examples.to(self.device), **{**self.model_kwargs, "output_hidden_states": True})
        return (
            output.hidden_states[-1][0, 0, :].detach().numpy().reshape(1, -1)
        )  # reshape for cosine similarity to be applicable

    def predict(self, input_sentence: str):
        tokenized_examples = self.tokenizer(input_sentence, **self.tokenizer_kwargs)
        output = self.model(
            **tokenized_examples.to(self.device), **{**self.model_kwargs, "output_hidden_states": False}
        )
        return output
