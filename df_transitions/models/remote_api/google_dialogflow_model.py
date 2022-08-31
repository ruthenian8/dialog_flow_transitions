from typing import Optional
import uuid
import json
from pathlib import Path

from ..base_model import BaseModel

try:
    from google.cloud import dialogflow_v2
    from google.oauth2 import service_account

    IMPORT_ERROR_MESSAGE = None
except ImportError as e:
    dialogflow_v2 = None
    service_account = None
    IMPORT_ERROR_MESSAGE = e.msg


class GoogleDialogFlowModel(BaseModel):
    """
    This class implements a connection to Google Dialogflow for label scoring.

    Parameters
    -----------
    model: str
        A service account json file that contains a link to your dialogflow project.
    namespace_key: str
        Name of the namespace in framework states that the model will be using.
    language: str
        The language of your dialogflow project. Set to English per default.

    """

    def __init__(
        self,
        model: dict,
        namespace_key: Optional[str] = None,
        *,
        language: str = "en",
    ) -> None:
        IMPORT_ERROR_MESSAGE = globals().get("IMPORT_ERROR_MESSAGE")
        if IMPORT_ERROR_MESSAGE is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE)
        super().__init__(namespace_key=namespace_key)
        self._language = language
        if isinstance(model, dict):
            info = model
        else:
            raise ValueError("Please, pass the service account credentials as dict.")

        self._credentials = service_account.Credentials.from_service_account_info(info)

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

    @classmethod
    def from_file(cls, filename: str, namespace_key: str, language: str = "en"):
        assert Path(filename).exists(), f"Path {filename} does not exist."
        with open(filename, "r", encoding="utf-8") as file:
            info = json.load(file)
        return cls(model=info, namespace_key=namespace_key, language=language)
