"""
Types
******

This module contains structures that are required to parse labels from files
and parse requests and responses to and from various APIs.

"""
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

from pydantic import BaseModel, Field, PrivateAttr, parse_file_as, parse_obj_as, validator
from yaml import load

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader


class Label(BaseModel):
    name: str
    examples: List[str] = Field(default_factory=list, min_items=1)
    _categorical_code = PrivateAttr(default=0)


class LabelCollection(BaseModel):
    labels: Dict[str, Label] = Field(default_factory=dict)

    @classmethod
    def _get_path(cls, file: str):
        file_path = Path(file)
        if not file_path.exists() or not file_path.is_file():
            raise OSError(f"File does not exist: {file}")
        return file_path

    @classmethod
    def parse_json(cls, file: str) -> None:
        file_path = cls._get_path(file)
        labels = parse_file_as(List[Label], file_path)
        return cls(labels=labels)

    @classmethod
    def parse_jsonl(cls, file: str) -> None:
        file_path = cls._get_path(file)
        lines = file_path.open("r", encoding="utf-8").readlines()
        labels = [Label.parse_raw(line) for line in lines]
        return cls(labels=labels)

    @classmethod
    def parse_yaml(cls, file: str) -> None:
        file_path = cls._get_path(file)
        raw_intents = load(file_path.open("r", encoding="utf-8").read(), SafeLoader)["labels"]
        labels = parse_obj_as(List[Label], raw_intents)
        return cls(labels=labels)

    @validator("labels")
    def validate_labels(cls, value: Dict[str, Label]):
        for idx, key in enumerate(value.keys()):
            value[key]._categorical_code = idx
        return value

    @validator("labels", pre=True)
    def pre_validate_labels(cls, value: Union[Dict[str, Label], List[Label]]):
        if isinstance(value, list):  # if labels were passed as a list, cast them to a dict
            intent_names = [intent.name for intent in value]
            return {name: intent for name, intent in zip(intent_names, value)}
        return value


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
