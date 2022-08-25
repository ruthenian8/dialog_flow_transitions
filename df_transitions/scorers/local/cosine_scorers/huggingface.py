"""
HuggingFace Cosine Scorer
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
from ...huggingface import BaseHFScorer
from .cosine_scorer_mixin import CosineScorerMixin


class HFCosineScorer(CosineScorerMixin, BaseHFScorer):
    """
    Parameters
    -----------
    label_collection: LabelCollection
        Expected labels.
    """

    def __init__(
        self,
        namespace_key: str,
        model: PreTrainedModel,
        label_collection: LabelCollection,
        tokenizer: Tokenizer,
        device: torch.device,
        tokenizer_kwargs: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
    ) -> None:
        CosineScorerMixin.__init__(self, label_collection=label_collection)
        BaseHFScorer.__init__(self, namespace_key, model, tokenizer, device, tokenizer_kwargs, model_kwargs)
