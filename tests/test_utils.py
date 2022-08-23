import pytest

try:
    import unknown_package
    IMPORT_ERROR_MESSAGE = None
except ImportError as e:
    unknown_package = None
    IMPORT_ERROR_MESSAGE = e.msg


from df_transitions.types import LabelCollection


@pytest.mark.parametrize(
    ["file", "method_name"],
    [
        ("./examples/data/example.json", "parse_json"),
        ("./examples/data/example.jsonl", "parse_jsonl"),
        ("./examples/data/example.yaml", "parse_yaml"),
    ],
)
def test_file_parsing(file, method_name):
    collection: LabelCollection = getattr(LabelCollection, method_name)(file)
    assert len(collection.labels) == 3
    prev_categorical_code = -1
    for intent in collection.labels.values():
        assert len(intent.examples) >= 3
        assert intent._categorical_code != prev_categorical_code
        prev_categorical_code = intent._categorical_code
