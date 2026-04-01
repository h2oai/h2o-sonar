# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import pathlib

import pandas as pd
import pytest

from h2o_sonar.lib.api import datasets
from h2o_sonar.utils import testing


@pytest.mark.skip(reason="One time import of the test data.")
@pytest.mark.generative
def test_import_frank_dataset(tmp_path: pathlib.Path) -> None:
    """Import frank dataset to test suite.

    From: https://github.com/artidoro/frank/blob/main/data/benchmark_data.json

    """
    df = pd.read_json(
        pathlib.Path() / "data" / "generative" / "incoming" / "summarization_frank.json"
    )

    df.drop_duplicates(inplace=True, subset="reference")
    df.drop(inplace=True, columns=["model_name", "split", "summary"])
    test_suite = testing.RagTestSuiteConfig(
        name="Frank Summarization test suite",
        description=(
            "Source: https://github.com/artidoro/"
            "frank/blob/main/data/benchmark_data.json"
        ),
    )
    categories = [datasets.LlmPromptCategories.summarization.name]
    test_config = testing.RagTestConfig(
        documents=[],
    )
    for index, row in df.iterrows():
        expected_output = row["reference"]

        test_suite.add_test_case(
            testing.RagTestCaseConfig(
                prompt="Summarize the following in one sentence: " + row["article"],
                categories=categories,
                expected_output=expected_output,
                config=test_config,
            )
        )
    file_path = tmp_path / "summarization_frank_test_suite.json"
    test_suite.save_as_json(file_path)
    print(f"\nTest suite saved to: {file_path}")


@pytest.mark.skip(reason="One time import of the test data.")
@pytest.mark.generative
def test_import_summeval_dataset(tmp_path: pathlib.Path) -> None:
    """Import summeval dataset to test suite.

    From: https://github.com/tanyuqian/ctc-gen-eval/blob/master/train/data/summeval.json

    """
    df = pd.read_json(
        pathlib.Path()
        / "data"
        / "generative"
        / "incoming"
        / "summarization_summeval.json"
    )

    df.drop_duplicates(inplace=True, subset="document")
    df.drop(
        inplace=True,
        columns=[
            "system",
            "coherence",
            "consistency",
            "fluency",
            "relevance",
            "summary",
        ],
    )
    test_suite = testing.RagTestSuiteConfig(
        name="Summeval Summarization test suite",
        description=(
            "Source: https://github.com/tanyuqian/ctc-gen-eval/"
            "blob/master/train/data/summarization_summeval.json"
        ),
    )
    categories = [datasets.LlmPromptCategories.summarization.name]
    test_config = testing.RagTestConfig(
        documents=[],
    )
    for index, row in df.iterrows():
        expected_outputs = row["references"]

        test_suite.add_test_case(
            testing.RagTestCaseConfig(
                prompt="Summarize the following in one sentence: " + row["document"],
                categories=categories,
                expected_output=expected_outputs[0] if expected_outputs[0] else "",
                config=test_config,
            )
        )
    file_path = tmp_path / "summarization_summeval_test_suite.json"
    test_suite.save_as_json(file_path)
    print(f"\nTest suite saved to: {file_path}")


@pytest.mark.skip(reason="One time import of the test data.")
@pytest.mark.generative
def test_import_samsum_dataset(tmp_path: pathlib.Path) -> None:
    """Import samsum dataset to test suite.

    From: https://github.com/skgabriel/GoFigure/blob/main/human_eval/samsum.jsonl
    Edited, for the jsonl format, to be a json file.

    """
    df = pd.read_json(
        pathlib.Path()
        / "data"
        / "generative"
        / "incoming"
        / "summarization_samsum.json"
    )

    df = df[df["label"] == "factual"]
    df.drop_duplicates(inplace=True, subset="article")
    df.drop(inplace=True, columns=["errors", "label"])
    test_suite = testing.RagTestSuiteConfig(
        name="Samsum Summarization test suite",
        description=(
            "Source: https://github.com/skgabriel/GoFigure/"
            "blob/main/human_eval/samsum.jsonl"
        ),
    )
    categories = [datasets.LlmPromptCategories.summarization.name]
    test_config = testing.RagTestConfig(
        documents=[],
    )
    for index, row in df.iterrows():
        expected_output = row["summary"]

        test_suite.add_test_case(
            testing.RagTestCaseConfig(
                prompt="Summarize the following in one sentence: " + row["article"],
                categories=categories,
                expected_output=expected_output,
                config=test_config,
            )
        )
    file_path = tmp_path / "summarization_samsum_test_suite.json"
    test_suite.save_as_json(file_path)
    print(f"\nTest suite saved to: {file_path}")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
