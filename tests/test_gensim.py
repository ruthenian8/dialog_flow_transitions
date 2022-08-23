import pytest

from df_transitions.scorers.local.cosine_scorers.gensim import GensimScorer
from df_transitions.types import LabelCollection

try:
    import gensim
    import gensim.downloader as api
except ImportError:
    pytest.skip(allow_module_leve=True)


@pytest.fixture(scope="session")
def testing_scorer(testing_collection):
    wv = api.load("glove-wiki-gigaword-50")
    model = gensim.models.word2vec.Word2Vec()
    model.wv = wv
    scorer = GensimScorer(model=model, label_collection=testing_collection, namespace_key="gensim")
    yield scorer


def test_saving(save_file: str, testing_scorer: GensimScorer):
    testing_scorer.save(save_file)
    new_testing_scorer = GensimScorer.load(save_file, gensim.models.word2vec.Word2Vec, "gensim")
    assert new_testing_scorer


def test_fit(testing_scorer: GensimScorer, testing_collection: LabelCollection):
    testing_scorer.fit(testing_collection)
    assert testing_scorer


def test_transform(testing_scorer: GensimScorer):
    assert testing_scorer.transform("one two three")