from typing import Optional
import uuid
from pathlib import Path

from ..base_scorer import BaseScorer

try:
    from google.cloud import dialogflow_v2
    from google.oauth2 import service_account

    IMPORT_ERROR_MESSAGE = None
except ImportError as e:
    dialogflow_v2 = None
    service_account = None
    IMPORT_ERROR_MESSAGE = e.msg


class DialogFlowScorer(BaseScorer):
    """
    Parameters
    -----------
    namespace_key: str
        Name of the namespace the model will be using in framework states.
    model: str
        A service account json file that contains a link to your dialogflow project.
    language: str
        The language of your dialogflow project. Set to English per default.

    """

    def __init__(
        self,
        model: str,
        namespace_key: Optional[str] = None,
        *,
        language: str = "en",
    ) -> None:
        IMPORT_ERROR_MESSAGE = globals().get("IMPORT_ERROR_MESSAGE")
        if IMPORT_ERROR_MESSAGE is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE)
        super().__init__(namespace_key=namespace_key, model=None)
        self._language = language

        assert Path(model).exists(), f"Path {model} does not exist."
        self._credentials = service_account.Credentials.from_service_account_file(model)

    def predict(self, request: str) -> dict:
        session_id = uuid.uuid4()
        session_client = dialogflow_v2.SessionsClient(credentials=self._credentials)
        session_path = session_client.session_path(self._credentials.project_id, session_id)
        query_input = dialogflow_v2.QueryInput(text=dialogflow_v2.TextInput(text=request, language_code=self._language))
        request = dialogflow_v2.DetectIntentRequest(session=session_path, query_input=query_input)
        response = session_client.detect_intent(request=request)
        result: dialogflow_v2.QueryResult = response.query_result
        if result.intent is not None:
            return {result.intent.display_name: result.intent_detection_confidence}
        return {}
