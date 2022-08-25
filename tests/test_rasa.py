import os

import requests
import pytest

from df_transitions.scorers.remote_api.rasa_scorer import RasaScorer

RASA_URL = os.getenv("RASA_URL")
if RASA_URL is None or isinstance(RASA_URL, str) and requests.get(RASA_URL).status_code != 200:
    pytest.skip(allow_module_level=True)


@pytest.fixture(scope="session")
def testing_scorer(rasa_url, rasa_api_key):
    yield RasaScorer(model=rasa_url, api_key=rasa_api_key, namespace_key="rasa")


def test_predict(testing_scorer: RasaScorer):
    request = "I would like to x."
    result = testing_scorer.predict(request=request)
    assert isinstance(result, dict)
