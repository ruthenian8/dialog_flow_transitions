import time
import json
from typing import Optional

import requests

from ...types import IntentCollection
from ...utils import STATUS_SUCCESS, STATUS_UNAVAILABLE
from ..base_annotator import BaseAnnotator


class HFApiAnnotator(BaseAnnotator):
    def __init__(
        self,
        url: Optional[str] = None,
        *,
        intent_collection: Optional[IntentCollection] = None,
        api_key: str = "",
        model: Optional[str] = None,
        retries: int = 60
    ):
        super().__init__(intent_collection=intent_collection)
        if not url and not model:
            raise ValueError("Model url or model name must be specified.")
        self._url = url if url is not None else "https://api-inference.huggingface.co/models/" + model
        self._headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
        self._retries = retries

    def get_intents(self, request: str) -> dict:
        retries = 0
        while retries < self._retries:
            retries += 1
            response: requests.Response = requests.post(self._url, headers=self._headers, data=json.dumps(request))
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
