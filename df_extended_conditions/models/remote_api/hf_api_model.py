"""
HF API Model
*************

This module provides the :py:class:`~HFAPIModel` class that allows you
to use remotely hosted Hugging Face models. 
"""
import time
import json
import asyncio
from typing import Optional
from urllib.parse import urljoin

import requests

try:
    import httpx

    IMPORT_ERROR_MESSAGE = None
except ImportError as e:
    htppx = object()
    IMPORT_ERROR_MESSAGE = e.msg

from ...utils import STATUS_SUCCESS, STATUS_UNAVAILABLE
from ..base_model import BaseModel
from .async_mixin import AsyncMixin


class AbstractHFAPIModel(BaseModel):
    """
    This class implements a connection to the Hugging Face inference API for label scoring.

    Prerequisites
    --------------
    Obtain an API token from Hugging Face to gain full access to hosted models.
    Note, that the service can fail, if you exceed the usage limits defined by your
    subscription type.

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


class HFAPIModel(AbstractHFAPIModel):
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


class AsyncHFAPIModel(AsyncMixin, AbstractHFAPIModel):
    def __init__(
        self,
        model: str,
        api_key: str,
        namespace_key: Optional[str] = None,
        *,
        retries: int = 60,
        headers: Optional[dict] = None,
    ) -> None:
        IMPORT_ERROR_MESSAGE = globals().get("IMPORT_ERROR_MESSAGE")
        if IMPORT_ERROR_MESSAGE is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE)
        super().__init__(model, api_key, namespace_key, retries=retries, headers=headers)

    async def predict(self, request: str) -> dict:
        client = httpx.AsyncClient()
        retries = 0
        while retries < self.retries:
            retries += 1
            response: httpx.Response = await client.post(self.url, headers=self.headers, data=json.dumps(request))
            if response.status_code == STATUS_UNAVAILABLE:  # Wait for model to warm up
                await asyncio.sleep(1)
            elif response.status_code == STATUS_SUCCESS:
                break
            else:
                raise httpx.HTTPStatusError(str(response.status_code) + " " + response.text)

        json_response = response.json()
        result = {}
        for label_score_pair in json_response[0]:
            result.update({label_score_pair["label"]: label_score_pair["score"]})

        await client.aclose()
        return result
