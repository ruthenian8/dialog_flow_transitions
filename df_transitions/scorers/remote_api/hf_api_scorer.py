import time
import json
from typing import Optional
from urllib.parse import urljoin

import requests

from ...utils import STATUS_SUCCESS, STATUS_UNAVAILABLE
from ..base_scorer import BaseScorer


class HFApiScorer(BaseScorer):
    """
    Parameters
    -----------
    namespace_key:
        Name of the namespace the model will be using in framework states.
    model: str
    api_key: str
    retries: int
    headers: Optional[dict]
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        namespace_key: Optional[str] = None,
        *,
        retries: int = 60,
        headers: Optional[dict] = None,
    ) -> None:
        super().__init__(namespace_key=namespace_key)
        self.api_key = api_key
        self.model = model
        self.headers = (
            headers if headers else {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
        )
        self.retries = retries
        self.url = urljoin("https://api-inference.huggingface.co/models/", model)

    def predict(self, request: str) -> dict:
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
