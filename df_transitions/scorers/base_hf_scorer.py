"""
Base HF Provider
------------------

Module
"""
from argparse import Namespace
from typing import Optional
from collections.abc import Iterable

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

from .base_scorer import BaseCosineScorer
from ..types import LabelCollection


class BaseHFScorer(BaseCosineScorer):
    def __init__(
        self,
        namespace_key: str,
        label_collection: Optional[LabelCollection],
        tokenizer: Tokenizer,
        model: PreTrainedModel,
        device: torch.device,
        tokenizer_kwargs: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
    ) -> None:
        IMPORT_ERROR_MESSAGE = globals().get("IMPORT_ERROR_MESSAGE")
        if IMPORT_ERROR_MESSAGE is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE)
        super().__init__(namespace_key=namespace_key, label_collection=label_collection)
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.tokenizer_kwargs = tokenizer_kwargs or {"return_tensors": "pt"}
        self.model_kwargs = model_kwargs or dict()

    def fit(self, request: str) -> Iterable:
        tokenized_examples = self.tokenizer(request, **self.tokenizer_kwargs)
        output = self.model(**tokenized_examples.to(self.device), **{**self.model_kwargs, "output_hidden_states": True})
        return (
            output.hidden_states[-1][0, 0, :].detach().numpy().reshape(1, -1)
        )  # reshape for cosine similarity to be applicable

    def call_model(self, request: str) -> dict:
        tokenized_examples = self.tokenizer(request, **self.tokenizer_kwargs)
        output = self.model(
            **tokenized_examples.to(self.device), **{**self.model_kwargs, "output_hidden_states": False}
        )
        return output
