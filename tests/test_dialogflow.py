import pytest

try:
    from google.cloud import dialogflow_v2
    from google.oauth2 import service_account
except ImportError:
    pytest.skip(allow_module_level=True)

from df_transitions.scorers.remote_api.google_dialogflow_scorer import GoogleDialogFlowScorer


@pytest.fixture(scope="session")
def testing_scorer(gdf_json):
    yield GoogleDialogFlowScorer(model=gdf_json, namespace_key="dialogflow")


def test_predict(testing_scorer: GoogleDialogFlowScorer):
    test_phrase = "I would like some food"  # no matching intent in test project
    result = testing_scorer.predict(test_phrase)
    assert isinstance(result, dict)
    assert len(result) == 1  # should only return the fallback intent
    test_phrase_2 = "Hello there"
    result_2 = testing_scorer.predict(test_phrase_2)  # has a matching intent
    assert isinstance(result_2, dict)
    assert len(result_2) >= 1  # expected to return 'hello' intent
