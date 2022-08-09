from pathlib import Path
from typing import List, Union, Dict

from pydantic import BaseModel, Field, PrivateAttr, parse_file_as, validate_arguments, parse_obj_as
from yaml import load

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader

from .utils import UnknownIntentError


class Intent(BaseModel):
    name: str
    examples: List[str] = Field(default_factory=list, min_items=1)
    _categorical_code = PrivateAttr(default=0)


class IntentCollection(BaseModel):
    intents: Dict[str, Intent] = Field(default_factory=dict)

    def _get_path(self, file: str):
        file_path = Path(file)
        if not file_path.exists() or not file_path.is_file():
            raise OSError(f"File does not exist: {file}")

        return file_path

    def parse_json_file(self, file: str) -> None:
        file_path = self._get_path(file)
        intents = parse_file_as(List[Intent], file_path)
        for intent in intents:
            self.register_intent(intent)

    def parse_jsonl_file(self, file: str) -> None:
        file_path = self._get_path(file)
        lines = file_path.open("r", encoding="utf-8").readlines()
        for line in lines:
            intent = Intent.parse_raw(line)
            self.register_intent(intent)

    def parse_yaml_file(self, file: str) -> None:
        file_path = self._get_path(file)
        raw_intents = load(file_path.open("r", encoding="utf-8").read(), SafeLoader)
        file_obj = parse_obj_as(Dict[str, List[Intent]], raw_intents)
        intents = file_obj.get("nlu", [])
        for intent in intents:
            self.register_intent(intent)

    @validate_arguments
    def register_intent(self, intent: Intent) -> None:
        intent._categorical_code = len(self.intents)
        self.intents[intent.name] = intent


def validate_intent(self, intent: Union[str, Intent]):
    if isinstance(intent, Intent):
        self.intent_collection.register_intent(intent)
        target_intent = intent.name
    else:
        if intent not in self.intent_collection.intents:
            raise UnknownIntentError(f"Intent {intent} has not been registered. Use `register_intent` method.")
        target_intent = intent
    return target_intent
