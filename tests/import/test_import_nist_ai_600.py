# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import pathlib

import datatable
import pytest

from h2o_sonar.utils import testing


"""NIST AI 600-1 test suites.

See: https://airc.nist.gov/docs/NIST.AI.600-1.GenAI-Profile.ipd.pdf

"""


@pytest.mark.skip(reason="One time import of the test data.")
@pytest.mark.generative
def test_john_20240507(tmp_path) -> None:
    """Test raw data import."""
    #
    # GIVEN
    #
    raw_dict = datatable.fread(
        pathlib.Path()
        / "data"
        / "generative"
        / "incoming"
        / "nist-genai-600-1-john-20240507.csv"
    ).to_dict()

    # WHEN
    test_suite = testing.RagTestSuiteConfig(
        name="NIST AI 600-1 Test Suite",
        description="NIST AI 600-1 document questions and answers.",
    )
    test = testing.RagTestConfig(
        documents=["https://airc.nist.gov/docs/NIST.AI.600-1.GenAI-Profile.ipd.pdf"]
    )
    rows = len(raw_dict["index"])
    print(f"Building test suite with {rows} prompts...")
    for i in range(rows):
        prompt = raw_dict["instruction"][i]
        expected_answer = raw_dict["response"][i]

        test_suite.add_test_case(
            testing.RagTestCaseConfig(
                prompt=prompt,
                categories=["question-answering"],
                expected_output=expected_answer,
                constraints=[],
                config=test,
            )
        )

    # THEN
    print(json.dumps(test_suite.to_dict(), indent=4))
    test_suite.save_as_json(pathlib.Path(tmp_path) / "test_suite.json")


@pytest.mark.skip(reason="One time import of the test data.")
@pytest.mark.generative
def test_trim_to_30_prompts(tmp_path) -> None:
    #
    # GIVEN
    #
    test_suite_path = (
        pathlib.Path() / "data" / "generative" / "nist-ai-600-1--test-suite--528p.json"
    )
    test_suite = testing.RagTestSuiteConfig.load_from_json(test_suite_path)

    #
    # WHEN
    #
    test_suite.trim_tests(30)
    test_suite.save_as_json(pathlib.Path(tmp_path) / "test_suite_trimmed.json")

    #
    # THEN
    #
    print(test_suite)


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
