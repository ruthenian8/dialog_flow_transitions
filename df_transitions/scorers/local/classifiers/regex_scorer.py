"""
Regex Scorer
----------------

This module provides a regex-based intent scorer version.
"""
import re
from typing import Optional

from pydantic import Field, validator

from ...base_scorer import BaseIntentScorer, BaseConfig
from ....types import IntentCollection


class RegexScorerConfig(BaseConfig):
    namespace_key: str = "regex_scorer"
    re_kwargs: dict = Field(default_factory=dict)
    default_score: float = Field(default=1.0)

    @validator("re_kwargs")
    def set_default_re_args(cls, value: dict):
        value["flags"] = re.IGNORECASE
        return value


class RegexScorer(BaseIntentScorer):
    def __init__(self, config: RegexScorerConfig, intent_collection: Optional[IntentCollection] = None) -> None:
        super().__init__(config=config, intent_collection=intent_collection)

        assert self.intent_collection is not None, "intent_collection parameter is required for RegexScorer."

    def analyze(self, request: str) -> str:
        result = {}
        for intent_name, intent in self.intent_collection.intents.items():
            matches = [re.search(item, request, **self.re_kwargs) for item in intent.examples]
            if any(map(bool, matches)):
                result[intent_name] = self.default_score

        return result
