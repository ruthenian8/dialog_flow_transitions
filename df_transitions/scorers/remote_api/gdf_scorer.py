from typing import Optional
import uuid
from pathlib import Path

from ..base_scorer import BaseScorer, BaseConfig
from ...types import LabelCollection, Label

try:
    from google.cloud import dialogflow_v2
    from google.oauth2 import service_account

    IMPORT_ERROR_MESSAGE = None
except ImportError as e:
    dialogflow_v2 = None
    service_account = None
    IMPORT_ERROR_MESSAGE = e.msg


class DialogFlowScorer(BaseScorer):
    def __init__(
        self,
        namespace_key: str,
        label_collection: Optional[LabelCollection],
        service_account_json: str,
        language: str = "en",
        sync_data: bool = False,
    ) -> None:
        IMPORT_ERROR_MESSAGE = globals().get("IMPORT_ERROR_MESSAGE")
        if IMPORT_ERROR_MESSAGE is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE)
        super().__init__(namespace_key=namespace_key, label_collection=label_collection)
        self._language = language

        assert Path(service_account_json).exists(), f"Path {service_account_json} does not exist."
        self._credentials = service_account.Credentials.from_service_account_file(service_account_json)
        if sync_data and self.label_collection and len(self.label_collection.labels) > 0:
            self._synchronize()

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

    def _delete_intent(self, intent: dialogflow_v2.Intent) -> None:
        client = dialogflow_v2.IntentsClient(credentials=self._credentials)
        intent_id = intent.name.split("/")[-1]
        intent_path = client.intent_path(self._credentials.project_id, intent_id)
        client.delete_intent(request={"name": intent_path})

    def _save_intent(self, label: Label) -> None:
        intents_client = dialogflow_v2.IntentsClient(credentials=self._credentials)
        agent_path = dialogflow_v2.AgentsClient.agent_path(self._credentials.project_id)

        phrases = []
        for training_phrases_part in label.examples:
            part = dialogflow_v2.Intent.TrainingPhrase.Part(text=training_phrases_part)
            training_phrase = dialogflow_v2.Intent.TrainingPhrase(parts=[part])
            phrases.append(training_phrase)

        df_intent = dialogflow_v2.Intent(display_name=label.name, training_phrases=phrases)

        response = intents_client.create_intent(request={"parent": agent_path, "intent": df_intent})

    def _synchronize(self) -> None:
        intents_client = dialogflow_v2.IntentsClient(credentials=self._credentials)
        agent_path = dialogflow_v2.AgentsClient.agent_path(self._credentials.project_id)

        current_intents = intents_client.list_intents(request={"parent": agent_path})
        registered_intents = self.label_collection.labels.copy()

        for intent in current_intents:
            if intent.display_name in registered_intents:
                registered_intents.pop(intent.display_name)
            else:
                self._delete_intent(intent=intent)

        for _, missing_intent in registered_intents.items():
            self._save_intent(intent=missing_intent)
