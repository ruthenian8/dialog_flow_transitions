import pytest

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
except ImportError:
    pytest.skip(allow_module_level=True)

from df_transitions.scorers.local.classifiers.sklearn import SklearnClassifier
from df_transitions.scorers.local.cosine_scorers.sklearn import SklearnScorer
from df_transitions.types import LabelCollection


@pytest.fixture(scope="session")
def testing_classifier():
    classifier = SklearnClassifier(model=LogisticRegression(), tokenizer=TfidfVectorizer(), namespace_key="classifier")
    yield classifier


@pytest.fixture(scope="session")
def testing_scorer(testing_collection):
    scorer = SklearnScorer(model=LogisticRegression(), tokenizer=TfidfVectorizer(), label_collection=testing_collection, namespace_key="scorer")
    yield scorer


def test_saving(save_file: str, testing_classifier: SklearnClassifier, testing_scorer: SklearnScorer):
    testing_classifier.save(save_file)
    new_classifier = SklearnClassifier.load(save_file, namespace_key="classifier")
    assert new_classifier.model == testing_classifier.model
    assert new_classifier.tokenizer == testing_classifier.tokenizer
    assert new_classifier.namespace_key == testing_classifier.namespace_key
    testing_scorer.save(save_file)
    new_scorer = SklearnScorer.load(path=save_file, namespace_key="scorer")
    assert new_scorer.model == testing_scorer.model
    assert new_scorer.tokenizer == testing_scorer.tokenizer


def test_fit(testing_classifier: SklearnClassifier, testing_scorer: SklearnScorer, testing_collection: LabelCollection):
    tc_result = testing_classifier.fit(testing_collection)
    ts_result = testing_scorer.fit(testing_collection)
    assert tc_result == ts_result == None


def test_transform(testing_scorer: SklearnScorer):
    result = testing_scorer.transform("one two three")
    assert result
