import os
import sys

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
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
def hf_api_key():
    yield os.getenv("HF_API_KEY") or ""


@pytest.fixture(scope="session")
def rasa_url():
    yield os.getenv("RASA_URL") or ""


@pytest.fixture(scope="session")
def save_file(tempdir_factory):
    file_name = tempdir_factory.mkdir("testdir").join("testfile")
    return str(file_name)