import pytest

try:
    from tokenizers import AutoTokenizer
    from transformers import AutoModelForSequenceClassification
    import torch
    import numpy as np
except ImportError:
    pytest.skip(allow_module_level=True)

from df_transitions.scorers.local.classifiers.huggingface import HFClassifier
from df_transitions.scorers.local.cosine_scorers.huggingface import HFCosineScorer


@pytest.fixture(scope="session")
def testing_model(hf_model_name):
    yield AutoModelForSequenceClassification.from_pretrained(hf_model_name)


@pytest.fixture(scope="session")
def testing_tokenizer(hf_model_name):
    yield AutoTokenizer.from_pretrained(hf_model_name)


@pytest.fixture(scope="session")
def testing_classifier(testing_model, testing_tokenizer):
    yield HFClassifier(
        model=testing_model, tokenizer=testing_tokenizer, device=torch.device("cpu"), namespace_key="HFclassifier"
    )


@pytest.fixture(scope="session")
def testing_scorer(testing_model, testing_tokenizer):
    yield HFCosineScorer(
        model=testing_model, tokenizer=testing_tokenizer, device=torch.device("cpu"), namespace_key="HFscorer"
    )


def test_saving(save_file: str, testing_classifier: HFClassifier, testing_scorer: HFCosineScorer):
    testing_classifier.save(path=save_file)
    testing_classifier = HFClassifier.load(save_file, namespace_key="HFclassifier")
    assert testing_classifier
    testing_scorer.save(path=save_file)
    testing_scorer = HFCosineScorer.load(save_file, namespace_key="HFscorer")
    assert testing_scorer


def test_predict(testing_classifier: HFClassifier):
    result = testing_classifier.predict("We are looking for x.")
    assert result
    assert isinstance(result, dict)


def test_transform(testing_scorer: HFCosineScorer, testing_classifier: HFClassifier):
    result_1 = testing_scorer.transform()
    assert isinstance(result_1, np.ndarray)
    result_2 = testing_classifier.transform()
    assert isinstance(result_2, np.ndarray)
