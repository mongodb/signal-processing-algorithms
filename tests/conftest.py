import json
import os
from typing import Any, Callable

import pytest


@pytest.fixture(scope="session")
def json_loader() -> Callable[[str], Any]:
    def load_test_json(file: str) -> Any:
        rel = os.path.dirname(__file__)
        with open(f"{rel}/data/{file}.json") as handle:
            return json.load(handle)

    return load_test_json


@pytest.fixture
def robust_series(json_loader):
    return json_loader("robust_series")


@pytest.fixture
def mad_series(json_loader):
    return json_loader("mad_series")


@pytest.fixture
def real_series(json_loader):
    return json_loader("real_series")


@pytest.fixture
def expected_result_robust_series(json_loader):
    return json_loader("expected_result_robust_series")
