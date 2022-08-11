"""
Regex Annotator
----------------

This module provides a regex-based intent annotator version.
"""
import re

from ...base_annotator import BaseAnnotator
from ....types import IntentCollection


class RegexAnnotator(BaseAnnotator):
    def __init__(
        self, intent_collection: IntentCollection, case_sensitive: bool = False, default_score: float = 1.0
    ) -> None:
        super().__init__(intent_collection=intent_collection)
        self._re_args = {"flags": re.IGNORECASE} if case_sensitive else {}
        self._default_score = default_score

    def get_intents(self, request: str) -> str:
        result = {}
        for intent_name, intent in self.intent_collection.intents.items():
            matches = [re.match(item, request, **self._re_args) for item in intent.examples]
            if any(map(bool, matches)):
                result[intent_name] = self._default_score

        return result
