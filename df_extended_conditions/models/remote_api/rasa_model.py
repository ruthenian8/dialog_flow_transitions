"""
Rasa Annotator
---------------

This module provides an annotator that queries an external RASA Server for intent detection.
"""
import uuid
import time
import json
from urllib.parse import urljoin
from typing import Optional

import requests

from ...utils import RasaResponse
from ..base_model import BaseModel
from ...utils import STATUS_SUCCESS, STATUS_UNAVAILABLE


class RasaModel(BaseModel):
    """
    RasaModel
    -----------
    This class implements a connection to RASA nlu server for label scoring.

    Prerequisites
    --------------
    In order to work with this class, you need to have a running instance of Rasa NLU Server
    with the model trained to recognize your intents.
    Please, refer to the `RASA docs <https://rasa.com/docs/rasa/nlu-only-server>`_ on how to
    develop a RASA project and launch an NLU-only server.

    Parameters
    -----------

    model: str
        Rasa model url.
    api_key: Optional[str]
        Rasa api key for request authorization. The exact authentification method can be retrieved
        from your Rasa Server config.
    jwt_token: Optional[str]
        Rasa jwt token for request authorization. The exact authentification method can be retrieved
        from your Rasa Server config.
    namespace_key: Optional[str]
        Name of the namespace in framework states that the model will be using.
    retries: int
        The number of times requests will be repeated in case of failure.
    headers: Optional[dict]
        Fill in this parameter, if you want to override the standard set of headers with custom headers.

    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
        namespace_key: Optional[str] = None,
        *,
        retries: int = 10,
        headers: Optional[dict] = None,
    ):
        super().__init__(namespace_key=namespace_key)
        self.headers = headers or {"Content-Type": "application/json"}
        self.parse_url = urljoin(model, ("model/parse" + (f"?token={api_key}" if api_key else "")))
        self.train_url = urljoin(model, ("model/train" + (f"?token={api_key}" if api_key else "")))
        if jwt_token is not None:
            self.headers["Authorization"] = "Bearer " + jwt_token
        self.retries = retries

    def predict(self, request: str) -> dict:
        message = {"text": request}
        retries = 0
        while retries < self.retries:
            retries += 1
            response: requests.Response = requests.post(self.parse_url, headers=self.headers, data=json.dumps(message))
            if response.status_code == STATUS_UNAVAILABLE:
                time.sleep(1)
            elif response.status_code == STATUS_SUCCESS:
                break
            else:
                raise requests.HTTPError(str(response.status_code) + " " + response.text)

        json_response = response.json()
        parsed = RasaResponse.parse_obj(json_response)
        result = {item.name: item.confidence for item in parsed.intent_ranking} if parsed.intent_ranking else dict()
        return result
