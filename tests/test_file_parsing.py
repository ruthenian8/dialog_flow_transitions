import os

import pytest

from df_transitions.types import IntentCollection


@pytest.mark.parametrize(
    ["file", "method_name"],
    [
        ("./examples/data/example.json", "parse_json"),
        ("./examples/data/example.jsonl", "parse_jsonl"),
        ("./examples/data/example.yaml", "parse_yaml"),
    ],
)
def test_file_parsing(file, method_name):
    collection: IntentCollection = getattr(IntentCollection, method_name)(file)
    assert len(collection.intents) == 3
    prev_categorical_code = -1
    for intent in collection.intents.values():
        assert len(intent.examples) >= 3
        assert intent._categorical_code != prev_categorical_code
        prev_categorical_code = intent._categorical_code
