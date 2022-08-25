import pytest

from df_transitions.scorers.remote_api.hf_api_scorer import HFApiScorer


@pytest.fixture(scope="session")
def testing_scorer(hf_api_key, hf_model_name):
    yield HFApiScorer(model=hf_model_name, api_key=hf_api_key, namespace_key="hf_api_scorer")


def test_predict(testing_scorer: HFApiScorer):
    print(testing_scorer.api_key)
    result = testing_scorer.predict("we are looking for x.")
    assert isinstance(result, dict)
    assert len(result) > 0
