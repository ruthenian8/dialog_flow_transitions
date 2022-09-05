import sys
import os
import importlib

import pytest

# uncomment the following line, if you want to run your examples during the test suite or import from them
sys.path.insert(0, os.pardir)

from examples.example_utils import run_test


@pytest.mark.parametrize(
    ["module_name"],
    [
        ("examples.base_example",),
        ("examples.remote_api.rasa",),
        ("examples.remote_api.dialogflow",),
        ("examples.remote_api.hf_api",),
        ("examples.sklearn_example",),
    ],
)
def test_examples(module_name):
    module = importlib.import_module(module_name)
    actor = getattr(module, "actor")
    testing_dialog = getattr(module, "testing_dialogue")
    run_test(testing_dialog=testing_dialog, actor=actor)
