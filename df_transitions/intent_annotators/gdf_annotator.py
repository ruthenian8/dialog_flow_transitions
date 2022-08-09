import uuid
from pathlib import Path

from .base_annotator import BaseAnnotator

try:
    from google.cloud import dialogflow_v2
    from google.oauth2 import service_account
except ImportError as e:
    print(f"Missing dependency: for Google DialogFlow annotator.")
    raise


class GoogleDialogFlowAnnotator(BaseAnnotator):
    def __init__(self, service_account_json: str, language: str = "en") -> None:
        super().__init__()
        assert Path(service_account_json).exists(), f"Path {service_account_json} does not exist."
        self._credentials = service_account.Credentials.from_service_account_file(service_account_json)
        self._language = language

    def get_intents(self, request: str) -> dict:
        session_id = uuid.uuid4()
        session_client = dialogflow_v2.SessionsClient(credentials=self._credentials)
        session_path = session_client.session_path(self._credentials.project_id, session_id)
        query_input = dialogflow_v2.QueryInput(text=dialogflow_v2.TextInput(text=request, language_code=self._language))
        request = dialogflow_v2.DetectIntentRequest(session=session_path, query_input=query_input)
        response = session_client.detect_intent(request=request)
        result = response.query_result
        if result.intent is not None:
            return {result.intent.display_name: result.intent_detection_confidence}
        return {}
