import pytest

from df_transitions.models.local.cosine_scorers.gensim import GensimScorer
from df_transitions.types import LabelCollection

try:
    import numpy as np
    import gensim
    import gensim.downloader as api
except ImportError:
    pytest.skip(allow_module_leve=True)

pytest.skip(allow_module_level=True)


@pytest.fixture(scope="session")
def testing_model(testing_collection):
    wv = api.load("glove-wiki-gigaword-50")
    model = gensim.models.word2vec.Word2Vec()
    model.wv = wv
    model = GensimScorer(model=model, label_collection=testing_collection, namespace_key="gensim")
    yield model


def test_saving(save_file: str, testing_model: GensimScorer):
    testing_model.save(save_file)
    new_testing_model = GensimScorer.load(save_file, "gensim")
    assert new_testing_model


def test_fit(testing_model: GensimScorer, testing_collection: LabelCollection):
    testing_model.fit(testing_collection)
    assert testing_model


def test_transform(testing_model: GensimScorer):
    result = testing_model.transform("one two three")
    assert isinstance(result, np.ndarray)
