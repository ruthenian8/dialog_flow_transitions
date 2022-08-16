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

from ...types import IntentCollection, RasaResponse, RasaTrainingIntent, RasaTrainingData
from ..base_scorer import BaseIntentScorer, BaseConfig
from ...utils import STATUS_SUCCESS, STATUS_UNAVAILABLE


DEFAULT_SESSION_CONFIG = {"session_expiration_time": 60, "carry_over_slots_to_new_session": True}


class RasaScorerConfig(BaseConfig):
    url: str
    api_key: Optional[str] = None
    jwt_token: Optional[str] = None
    retries: int = Field(default=10)
    parse_url: str = ""
    train_url: str = ""
    headers: dict = Field(default_factory=dict, alias="headers")
    namespace_key: str = "rasa_scorer"

    @validator("headers")
    def validate_rasa_headers(cls, headers: dict, values: dict):
        headers.update({"Content-Type": "application/json"})
        jwt_token = values.get("jwt_token")
        if jwt_token is not None:
            headers["Authorization"] = "Bearer " + jwt_token
        return headers

    @root_validator
    def get_urls(cls, values: dict) -> dict:
        base_url = values.get("url")
        api_key = values.get("api_key")
        values["parse_url"] = urljoin(base_url, ("model/parse" + (f"?token={api_key}" if api_key else "")))
        values["train_url"] = urljoin(base_url, ("model/train" + (f"?token={api_key}" if api_key else "")))
        return values


class RasaScorer(BaseIntentScorer):
    def __init__(self, config: RasaScorerConfig, intent_collection: Optional[IntentCollection] = None):
        super().__init__(config=config, intent_collection=intent_collection)

    def analyze(self, request: str) -> dict:
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
            intents=list(self.intent_collection.intents.keys()),
            nlu=parse_obj_as(List[RasaTrainingIntent], list(self.intent_collection.intents.values())),
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
