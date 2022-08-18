"""
Rasa Annotator
---------------

This module provides an annotator that queries an external RASA Server for intent detection.
"""
import uuid
import time
import json
from urllib.parse import urljoin
from typing import List, Optional
from pydantic import parse_obj_as, Field, validator, root_validator

from yaml import dump
import requests

try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper

from ...types import LabelCollection, RasaResponse, RasaTrainingIntent, RasaTrainingData
from ..base_scorer import BaseScorer, BaseConfig
from ...utils import STATUS_SUCCESS, STATUS_UNAVAILABLE


DEFAULT_SESSION_CONFIG = {"session_expiration_time": 60, "carry_over_slots_to_new_session": True}


class RasaScorer(BaseScorer):
    def __init__(
        self,
        namespace_key: str,
        label_collection: Optional[LabelCollection],
        url: str,
        api_key: Optional[str],
        jwt_token: Optional[str] = None,
        retries: int = 10,
        headers: Optional[dict] = None,
    ):
        super().__init__(namespace_key=namespace_key, label_collection=label_collection)
        self.headers = headers or {"Content-Type": "application/json"}
        self.parse_url = urljoin(url, ("model/parse" + (f"?token={api_key}" if api_key else "")))
        self.train_url = urljoin(url, ("model/train" + (f"?token={api_key}" if api_key else "")))
        if jwt_token is not None:
            self.headers["Authorization"] = "Bearer " + jwt_token
        self.retries = retries

    def predict(self, request: str) -> dict:
        message_id = uuid.uuid4()
        message = {"message_id": message_id, "text": request}
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

    def get_trained_nlu_model(self) -> str:
        """
        Dispatches a request to the RASA server to train an nlu model and returns the name of the obtained file.
        """
        data = RasaTrainingData(
            intents=list(self.label_collection.intents.keys()),
            nlu=parse_obj_as(List[RasaTrainingIntent], list(self.label_collection.intents.values())),
        )
        yaml_data = dump(data.dict(by_alias=False), Dumper=Dumper)
        headers = {**self.headers, "Content-Type": "application/yaml"}
        response: requests.Response = requests.post(headers=headers, data=yaml_data)
        if not response.status_code == 200 and not response.status_code == 204:
            raise requests.HTTPError(str(response.status_code) + " " + response.text)
        assert hasattr(response.headers, "filename"), "No filename returned"
        filename = response.headers.filename
        assert isinstance(filename, str) and len(filename) > 0, "No filename returned"
        with open(filename, "wb+") as file:
            file.write(response.content)
        return filename
