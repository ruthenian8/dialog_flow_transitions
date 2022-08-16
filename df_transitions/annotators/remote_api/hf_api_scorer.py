import time
import json
from typing import Optional
from urllib.parse import urljoin

from pydantic import Field, root_validator, validator
import requests

from ...types import IntentCollection
from ...utils import STATUS_SUCCESS, STATUS_UNAVAILABLE
from ..base_scorer import BaseIntentScorer, BaseConfig


class HFApiScorerConfig(BaseConfig):
    api_key: str
    model: Optional[str] = Field(default=None)
    headers: dict = Field(default_factory=dict)
    retries: int = Field(default=60)
    url: Optional[str] = Field(default=None)
    namespace_key: str = "hugging_face_inference_api"

    @validator("headers")
    def validate_hf_headers(cls, headers: dict, values: dict):
        api_key = values.get("api_key")
        headers.update({"Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
        return headers

    @root_validator(pre=True)
    def validate_url(cls, values: dict):
        url, model = values.get("url"), values.get("model")
        if not url and not model:
            raise ValueError("Model url or model name must be specified.")
        values["url"] = url or urljoin("https://api-inference.huggingface.co/models/", model)
        return values


class HFApiScorer(BaseIntentScorer):
    def __init__(self, config: HFApiScorerConfig, intent_collection: Optional[IntentCollection] = None):
        super().__init__(config=config, intent_collection=intent_collection)

    def analyze(self, request: str) -> dict:
        retries = 0
        while retries < self.retries:
            retries += 1
            response: requests.Response = requests.post(self.url, headers=self.headers, data=json.dumps(request))
            if response.status_code == STATUS_UNAVAILABLE:  # Wait for model to warm up
                time.sleep(1)
            elif response.status_code == STATUS_SUCCESS:
                break
            else:
                raise requests.HTTPError(str(response.status_code) + " " + response.text)

        json_response = response.json()
        result = {}
        for label_score_pair in json_response[0][0]:
            result.update({label_score_pair["label"]: label_score_pair["score"]})
        return result
