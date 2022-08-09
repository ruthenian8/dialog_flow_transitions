import time
from typing import Callable, Optional, Union, List

import numpy as np
import requests
from pydantic import validate_arguments
from sklearn.metrics.pairwise import cosine_similarity
from tokenizers import Tokenizer
from transformers.modeling_utils import PreTrainedModel
from df_engine.core import Context, Actor

from ..utils import STATUS_SUCCESS, STATUS_UNAVAILABLE
from ..types import Intent, IntentCollection

from .base_annotator import BaseAnnotator


class HFApiAnnotator(BaseAnnotator):
    def __init__(self, url: Optional[str] = None, *, api_key: str = "", model: Optional[str] = None, retries: int = 60):
        super().__init__()
        if not url and not model:
            raise ValueError("Model url or model name must be specified.")
        self._url = url if url is not None else "https://api-inference.huggingface.co/models/" + model
        self._headers = {"Authorization": "Bearer " + api_key}
        self._retries = retries

    def get_intents(self, request: str) -> dict:
        retries = 0
        while retries < self._retries:
            retries += 1
            response: requests.Response = requests.post(self._url, headers=self._headers, json=request)
            if response.status_code == STATUS_UNAVAILABLE:  # Wait for model to warm up
                time.sleep(1)
            elif response.status_code == STATUS_SUCCESS:
                break
            else:
                raise requests.HTTPError(response.status_code + " " + response.text)

        json_response = response.json()
        result = {}
        for label_score_pair in json_response[0][0]:
            result.update({label_score_pair["label"]: label_score_pair["score"]})
        return result


class BaseHFModelAnnotator(BaseAnnotator):
    def __init__(
        self,
        tokenizer: Tokenizer,
        model: PreTrainedModel,
        device: object,
        intent_collection: IntentCollection,
        tokenizer_kwargs: dict = None,
        model_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.intent_collection = intent_collection
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


class HFClassifierAnnotator(BaseHFModelAnnotator):
    def get_intents(self, request: str) -> dict:
        request_cls_embedding = self.get_cls_embedding(request)
        result = dict()
        for intent_name, intent in self.intent_collection.intents.items():
            reference_examples = intent.examples
            reference_embeddings = [self.get_cls_embedding(item) for item in reference_examples]
            cosine_scores = [
                cosine_similarity(request_cls_embedding, ref_emb)[0][0] for ref_emb in reference_embeddings
            ]
            result[intent_name] = np.mean(np.array(cosine_scores))

        return result


class HFModelCosineAnnotator(BaseHFModelAnnotator):
    def get_intents(self, request: str) -> dict:
        model_output = self.get_prediction(request)
        logits_list: List[float] = model_output.tolist()[0]
        assert len(logits_list) == len(
            self.intent_collection.intents
        ), "Number of predicted labels does not match the number of registered intents."
        result = {intent_name: score for intent_name, score in zip(self.intent_collection.intents.keys(), logits_list)}
        return result
