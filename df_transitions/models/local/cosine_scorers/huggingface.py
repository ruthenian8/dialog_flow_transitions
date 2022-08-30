"""
HuggingFace Cosine Model
--------------------------

This module provides an adapter interface for Huggingface models.
It leverages transformer embeddings to compute distances between utterances.
"""
from typing import Optional
from argparse import Namespace

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

from ....types import LabelCollection
from ...huggingface import BaseHFModel
from .cosine_scorer_mixin import CosineScorerMixin


class HFScorer(CosineScorerMixin, BaseHFModel):
    """
    HFScorer utilizes embeddings from Hugging Face models to measure
    proximity between utterances and pre-defined labels.

    Parameters
    -----------
    model: PreTrainedModel
        A pretrained Hugging Face format model.
    tokenizer: Tokenizer
        A pretrained Hugging Face tokenizer.
    device: torch.device
        Pytorch device object. The device will be used for inference and pre-training.
    namespace_key: str
        Name of the namespace in framework states that the model will be using.
    label_collection: LabelCollection
        Labels for the scorer. The prediction output depends on proximity to different labels.
    tokenizer_kwargs: Optional[dict] = None
        Default tokenizer arguments override.
    model_kwargs: Optional[dict] = None
        Default model arguments override.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Tokenizer,
        device: torch.device,
        namespace_key: str,
        label_collection: LabelCollection,
        tokenizer_kwargs: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
    ) -> None:
        CosineScorerMixin.__init__(self, label_collection=label_collection)
        BaseHFModel.__init__(self, namespace_key, model, tokenizer, device, tokenizer_kwargs, model_kwargs)
