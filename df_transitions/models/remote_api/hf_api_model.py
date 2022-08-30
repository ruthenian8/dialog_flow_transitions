import time
import json
from typing import Optional
from urllib.parse import urljoin

import requests

from ...utils import STATUS_SUCCESS, STATUS_UNAVAILABLE
from ..base_model import BaseModel


class HFApiModel(BaseModel): # TODO: use upper case for API 
    """
    Parameters
    -----------
    model: str
        Hosted model name, e. g. 'bert-base-uncased', etc.
    api_key: str
        Huggingface inference API token.
    namespace_key: Optional[str]
        Name of the namespace in framework states that the model will be using.
    retries: int
        Number of retries in case of request failure.
    headers: Optional[dict]
        A dictionary that overrides a standard set of headers.

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
        test_response = requests.get(self.url, headers=self.headers)  # assert that the model exists
        if not test_response.status_code == STATUS_SUCCESS:
            raise requests.HTTPError(test_response.text)

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
        for label_score_pair in json_response[0]:
            result.update({label_score_pair["label"]: label_score_pair["score"]})
        return result
