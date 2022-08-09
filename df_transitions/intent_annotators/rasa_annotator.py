"""
Rasa Annotator
---------------

This module provides an annotator that queries an external RASA Server for intent detection.
"""
import uuid
import time
from typing import Optional, List

import requests
from pydantic import BaseModel

from .base_annotator import BaseAnnotator
from ..utils import STATUS_SUCCESS, STATUS_UNAVAILABLE


class RasaAnnotator(BaseAnnotator):
    def __init__(self, url: str, *, retries: int = 10):
        self._url = url
        self._parse_url = url + "/model/parse"
        self._retries = retries

    def get_intents(self, request: str) -> dict:
        message_id = uuid.uuid4()
        message = {"message_id": message_id, "text": request}
        retries = 0
        while retries < self._retries:
            retries += 1
            response: requests.Response = requests.post(self._parse_url, json=message)
            if response.status_code == STATUS_UNAVAILABLE:
                time.sleep(1)
            elif response.status_code == STATUS_SUCCESS:
                break
            else:
                raise requests.HTTPError(response.status_code + " " + response.text)

        json_response = response.json()
        parsed = RasaResponse.parse_obj(json_response)
        result = {item.name: item.confidence for item in parsed.intent_ranking} if parsed.intent_ranking else dict()
        return result


class RasaIntent(BaseModel):
    confidence: float
    name: str


class RasaEntity(BaseModel):
    start: int
    end: int
    confidence: Optional[float]
    value: str
    entity: str


class RasaResponse(BaseModel):
    text: str
    intent_ranking: Optional[List[RasaIntent]] = None
    intent: Optional[RasaIntent] = None
    entities: Optional[List[RasaEntity]] = None
