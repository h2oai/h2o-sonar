# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json

import pytest

from h2o_sonar.utils import testing


@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_cats_suite_and_test(tmp_path):
    """Test categories field in test suite and test."""
    #
    # GIVEN
    #
    test = testing.RagTestConfig(
        documents=[f"http://example.com/doc{i}" for i in range(3)],
        categories=[f"test-category-{i}" for i in range(3)],
        key="test-key",
    )
    test_cases = []
    for i in range(3):
        test_cases.append(
            testing.RagTestCaseConfig(
                prompt="What's the meaning of life?",
                categories=[f"test-case-category-{i}"],
                relationships=None,
                constraints=None,
                condition="",
                expected_output=f"{40 + i}",
                config=test,
                key=f"test-case-key-{i}",
            )
        )

    test_suite = testing.RagTestSuiteConfig(
        test_cases=test_cases,
        name="Test Test Suite w/ categories",
        description="Test suite description",
        categories=[f"test-suite-category-{i}" for i in range(3)],
    )

    #
    # WHEN
    #
    as_dict = test_suite.to_dict()
    test_suite.save_as_json(tmp_path / "test_suite_with_cats.json")

    #
    # THEN
    #
    print(as_dict)
    assert as_dict == {
        "name": "Test Test Suite w/ categories",
        "description": "Test suite description",
        "tests": [
            {
                "key": "test-key",
                "documents": [
                    "http://example.com/doc0",
                    "http://example.com/doc1",
                    "http://example.com/doc2",
                ],
                "test_cases": [
                    {
                        "key": "test-case-key-0",
                        "prompt": "What's the meaning of life?",
                        "categories": ["test-case-category-0"],
                        "condition": "",
                        "relationships": [],
                        "constraints": None,
                        "expected_output": "40",
                    },
                    {
                        "key": "test-case-key-1",
                        "prompt": "What's the meaning of life?",
                        "categories": ["test-case-category-1"],
                        "condition": "",
                        "relationships": [],
                        "constraints": None,
                        "expected_output": "41",
                    },
                    {
                        "key": "test-case-key-2",
                        "prompt": "What's the meaning of life?",
                        "categories": ["test-case-category-2"],
                        "condition": "",
                        "relationships": [],
                        "constraints": None,
                        "expected_output": "42",
                    },
                ],
            }
        ],
        "categories": [
            "test-suite-category-0",
            "test-suite-category-1",
            "test-suite-category-2",
        ],
    }


@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_empty_cats_not_in_json(tmp_path):
    """Test that empty categories are not included in JSON."""
    #
    # GIVEN
    #
    test = testing.RagTestConfig(
        documents=[f"http://example.com/doc{i}" for i in range(3)],
        categories=[],
        key="test-key",
    )
    test_cases = []
    for i in range(3):
        test_cases.append(
            testing.RagTestCaseConfig(
                prompt="What's the meaning of life?",
                categories=[],
                relationships=None,
                constraints=None,
                condition="",
                expected_output=f"{40 + i}",
                config=test,
                key=f"test-case-key-{i}",
            )
        )

    test_suite = testing.RagTestSuiteConfig(
        test_cases=test_cases,
        name="Test Test Suite w/ empty categories",
        description="Test suite description",
        categories=[],
    )

    #
    # WHEN
    #
    as_dict = test_suite.to_dict()
    test_suite_path = tmp_path / "test_suite_with_cats.json"
    test_suite.save_as_json(test_suite_path)

    #
    # THEN
    #
    print(as_dict)
    assert as_dict == {
        "name": "Test Test Suite w/ empty categories",
        "description": "Test suite description",
        "tests": [
            {
                "key": "test-key",
                "documents": [
                    "http://example.com/doc0",
                    "http://example.com/doc1",
                    "http://example.com/doc2",
                ],
                "test_cases": [
                    {
                        "key": "test-case-key-0",
                        "prompt": "What's the meaning of life?",
                        "categories": [],
                        "condition": "",
                        "relationships": [],
                        "constraints": None,
                        "expected_output": "40",
                    },
                    {
                        "key": "test-case-key-1",
                        "prompt": "What's the meaning of life?",
                        "categories": [],
                        "condition": "",
                        "relationships": [],
                        "constraints": None,
                        "expected_output": "41",
                    },
                    {
                        "key": "test-case-key-2",
                        "prompt": "What's the meaning of life?",
                        "categories": [],
                        "condition": "",
                        "relationships": [],
                        "constraints": None,
                        "expected_output": "42",
                    },
                ],
            }
        ],
    }

    # load the saved JSON file to ensure it matches the expected structure
    with open(test_suite_path) as f:
        loaded_json = json.load(f)
    assert loaded_json.get(testing.RagTestSuiteConfig.KEY_CATS) is None, (
        "Expected categories to be None in the saved JSON file, but it was present."
    )
    assert (
        loaded_json[testing.RagTestSuiteConfig.KEY_TESTS][0].get(
            testing.RagTestConfig.KEY_CATS
        )
        is None
    ), "Expected categories to be None in the test config, but it was present."


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
