import os
import sys

import pytest

sys.path.insert(0, os.path.pardir)


@pytest.fixture(scope="session")
def testing_actor():
    from examples.regexp import actor

    yield actor
