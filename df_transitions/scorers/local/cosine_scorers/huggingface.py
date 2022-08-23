"""
HuggingFace Cosine Scorer
--------------------------

TODO: Complete
"""
from sklearn.metrics.pairwise import cosine_similarity
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
from ...base_hf_scorer import BaseHFScorer
from .cosine_scorer_mixin import CosineScorerMixin


class HFCosineScorer(BaseHFScorer, CosineScorerMixin):
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
        BaseHFScorer.__init__(self, namespace_key, model, tokenizer, device, tokenizer_kwargs, model_kwargs)
        CosineScorerMixin.__init__(self, label_collection=label_collection)
