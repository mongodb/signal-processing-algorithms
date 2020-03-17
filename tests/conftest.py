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
def small_profile(json_loader):
    return json_loader("profiling/small")


@pytest.fixture
def short_profile(json_loader):
    return json_loader("profiling/short")


@pytest.fixture
def medium_profile(json_loader):
    return json_loader("profiling/medium")


@pytest.fixture
def large_profile(json_loader):
    return json_loader("profiling/large")


@pytest.fixture
def very_large_profile(json_loader):
    return json_loader("profiling/very_large")


@pytest.fixture
def huge_profile(json_loader):
    return json_loader("profiling/huge")


@pytest.fixture
def humongous_profile(json_loader):
    return json_loader("profiling/humongous")


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


@pytest.fixture
def expected_result_robust_series_proper_division(json_loader):
    return json_loader("expected_result_robust_series_proper_division")


@pytest.fixture
def long_series(json_loader):
    return json_loader("long_series")


# http://doc.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
