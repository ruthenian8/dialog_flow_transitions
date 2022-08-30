import pytest

from df_transitions.models.remote_api.hf_api_model import HFApiModel


@pytest.fixture(scope="session")
def testing_model(hf_api_key, hf_model_name):
    yield HFApiModel(model=hf_model_name, api_key=hf_api_key, namespace_key="hf_api_model")


def test_predict(testing_model: HFApiModel):
    print(testing_model.api_key)
    result = testing_model.predict("we are looking for x.")
    assert isinstance(result, dict)
    assert len(result) > 0
