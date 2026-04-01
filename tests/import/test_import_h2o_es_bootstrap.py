# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import pathlib

import pytest

from h2o_sonar.utils import testing


"""Import H2O Eval Studio bootstrap as test suites/test labs."""


@pytest.mark.skip(reason="This test is used to import the boostrap")
@pytest.mark.generative
def test_h2ogpte_benchmark_pdfs(tmp_path):
    #
    # GIVEN
    #
    boostrap_path = pathlib.Path(
        "/home/user/h2o-eval-studio-demo-data/h2o-eval-studio-boostrap-2024-08-27.json"
    )

    #
    # WHEN
    #

    with open(boostrap_path) as f:
        bootstrap = json.load(f)

    # index boostrap - map: id -> dict
    b_documents = {v.get("id"): v for v in bootstrap["documents"]}
    b_tests = {v.get("id"): v for v in bootstrap["tests"]}
    # b_test_classes = { v.get("id"): v for v in bootstrap["test_classes"] }
    # turn leaderboards into test suites
    b_leaderboards = {v.get("id"): v for v in bootstrap["leaderboards"]}

    # import

    test_suite_paths = []
    test_suite_count = 1
    for ll in b_leaderboards.values():
        test_cases = []
        for test_key in ll.get("tests"):
            test_dict = b_tests.get(test_key)

            if test_dict is None:
                raise RuntimeError(f"Test {test_key} not found.")

            documents = []
            doc_keys = test_dict.get("documents", []) or []
            for doc_key in doc_keys:
                doc_dict = b_documents.get(doc_key)
                documents.append(doc_dict.get("url"))

            test = testing.RagTestConfig(
                documents=documents,
                key=test_key,
            )

            for tc in bootstrap["test_cases"]:
                if tc.get("test") == test_key:
                    tc_dict = tc
                    test_case = testing.RagTestCaseConfig(
                        prompt=tc_dict.get("prompt", ""),
                        expected_output=tc_dict.get("expected_output", ""),
                        constraints=tc_dict.get("constraints", []),
                        config=test,
                    )
                    test_cases.append(test_case)

        test_suite = testing.RagTestSuiteConfig(
            test_cases=test_cases,
            name=ll.get("display_name"),
            description=ll.get("description"),
        )

        # save test suite
        test_suite_path = tmp_path / f"test_suite_{test_suite_count}.json"
        test_suite.save_as_json(file_path=test_suite_path)

        test_suite_paths.append(test_suite_path)
        test_suite_count += 1

    #
    # THEN
    #
    print("Exported test suites:")
    for p in test_suite_paths:
        print(f"  file://{p}")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
