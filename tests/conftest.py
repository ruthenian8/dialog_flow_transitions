import os
import sys

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from df_transitions.models.local.cosine_matchers.sklearn import SklearnMatcher
from df_transitions.types import LabelCollection

sys.path.insert(0, os.path.pardir)


@pytest.fixture(scope="session")
def testing_actor():
    from examples.regexp import actor

    yield actor


@pytest.fixture(scope="session")
def testing_collection():
    yield LabelCollection.parse_yaml("./examples/data/example.yaml")


@pytest.fixture(scope="session")
def standard_model(testing_collection):
    yield SklearnMatcher(tokenizer=TfidfVectorizer(stop_words=None), label_collection=testing_collection)


@pytest.fixture(scope="session")
def hf_api_key():
    yield os.getenv("HF_API_KEY") or ""


@pytest.fixture(scope="session")
def gdf_json(tmpdir_factory):
    json_file = tmpdir_factory.mktemp("gdf").join("service_account.json")
    contents = os.getenv("GDF_ACCOUNT_JSON")
    json_file.write(contents)
    yield str(json_file)


@pytest.fixture(scope="session")
def hf_model_name():
    yield "obsei-ai/sell-buy-intent-classifier-bert-mini"


@pytest.fixture(scope="session")
def rasa_url():
    yield os.getenv("RASA_URL") or ""


@pytest.fixture(scope="session")
def rasa_api_key():
    yield os.getenv("RASA_API_KEY") or ""


@pytest.fixture(scope="session")
def save_file(tmpdir_factory):
    file_name = tmpdir_factory.mktemp("testdir").join("testfile")
    return str(file_name)
