from pathlib import Path
from typing import List, Dict, Optional, Any, Union

from pydantic import BaseModel, Field, PrivateAttr, parse_file_as, parse_obj_as, validator
from yaml import load

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader


class Intent(BaseModel):
    name: str
    examples: List[str] = Field(default_factory=list, min_items=1)
    _categorical_code = PrivateAttr(default=0)


class IntentCollection(BaseModel):
    intents: Dict[str, Intent] = Field(default_factory=dict)

    @classmethod
    def _get_path(cls, file: str):
        file_path = Path(file)
        if not file_path.exists() or not file_path.is_file():
            raise OSError(f"File does not exist: {file}")
        return file_path

    @classmethod
    def parse_json(cls, file: str) -> None:
        file_path = cls._get_path(file)
        intents = parse_file_as(List[Intent], file_path)
        return cls(intents=intents)

    @classmethod
    def parse_jsonl(cls, file: str) -> None:
        file_path = cls._get_path(file)
        lines = file_path.open("r", encoding="utf-8").readlines()
        intents = [Intent.parse_raw(line) for line in lines]
        return cls(intents=intents)

    @classmethod
    def parse_yaml(cls, file: str) -> None:
        file_path = cls._get_path(file)
        raw_intents = load(file_path.open("r", encoding="utf-8").read(), SafeLoader)["intents"]
        intents = parse_obj_as(List[Intent], raw_intents)
        return cls(intents=intents)

    @validator("intents")
    def validate_intents(cls, value: Dict[str, Intent]):
        for idx, key in enumerate(value.keys()):
            value[key]._categorical_code = idx
        return value

    @validator("intents", pre=True)
    def pre_validate_intents(cls, value: Union[Dict[str, Intent], List[Intent]]):
        if isinstance(value, list):  # if intents were passed as a list, cast them to a dict
            intent_names = [intent.name for intent in value]
            return {name: intent for name, intent in zip(intent_names, value)}
        return value


class RasaTrainingIntent(BaseModel):
    intent: str = Field(alias="name")
    examples: List[str]


class RasaTrainingData(BaseModel):
    pipeline: List[str] = Field(default_factory=list)
    policies: List[str] = Field(default_factory=list)
    intents: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    slots: Dict[str, Any] = Field(default_factory=dict)
    actions: List[str] = Field(default_factory=list)
    forms: Dict[str, Any] = Field(default_factory=dict)
    e2e_actions: List[str] = Field(default_factory=list)
    responses: Dict[str, Any] = Field(default_factory=dict)
    session_config: Dict[str, Any] = Field(default_factory=dict)
    nlu: List[RasaTrainingIntent]
    rules: List[Dict[str, Any]]
    stories: List[Dict[str, Any]]


class RasaIntent(BaseModel):
    confidence: float
    name: str


class RasaEntity(BaseModel):
    start: int
    end: int
    confidence: Optional[float]
    value: str
    entity: str


class RasaResponse(BaseModel):
    text: str
    intent_ranking: Optional[List[RasaIntent]] = None
    intent: Optional[RasaIntent] = None
    entities: Optional[List[RasaEntity]] = None
