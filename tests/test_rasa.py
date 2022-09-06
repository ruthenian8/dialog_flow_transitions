import os

import requests
import pytest

from df_extended_conditions.models.remote_api.rasa_model import RasaModel

RASA_URL = os.getenv("RASA_URL")
if RASA_URL is None or isinstance(RASA_URL, str) and requests.get(RASA_URL).status_code != 200:
    pytest.skip(allow_module_level=True)


@pytest.fixture(scope="session")
def testing_model(rasa_url, rasa_api_key):
    yield RasaModel(model=rasa_url, api_key=rasa_api_key, namespace_key="rasa")


def test_predict(testing_model: RasaModel):
    request = "Hello there"
    result = testing_model.predict(request=request)
    assert isinstance(result, dict)
    assert len(result) > 0
    assert result["greet"] > 0.9  # testing on default intents that include 'greet'
