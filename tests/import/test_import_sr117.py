# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import pathlib

import datatable
import pytest

from h2o_sonar.lib.api import commons
from h2o_sonar.utils import testing


"""SR 11-7 test suites.

See: https://www.federalreserve.gov/supervisionreg/srletters/sr1107a1.pdf

"""


@pytest.mark.skip(reason="One time import of the test data.")
@pytest.mark.generative
def test_john_20240118(tmp_path) -> None:
    """Test raw data import.

    @see https://www.jobtestprep.com/bank-teller-sample-questions

    """
    # GIVEN
    raw_dict = datatable.fread(
        pathlib.Path()
        / "data"
        / "llm"
        / "incoming"
        / "sr117-qa-pairs-john-20240118.csv"
    ).to_dict()

    # WHEN
    test_suite = testing.RagTestSuiteConfig(
        name="SR 11-7 Test Suite",
        description="SR 11-7 MRM document questions and answers.",
    )
    test = testing.RagTestConfig(
        documents=[
            "https://www.federalreserve.gov/supervisionreg/srletters/sr1107a1.pdf"
        ]
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


@pytest.mark.skip(reason="One time transformation of the test lab.")
@pytest.mark.generative
def test_john_20240118_hide_errors(tmp_path) -> None:
    # GIVEN

    rewrite_to_message = "Failed to retrieve response from model."

    with open(pathlib.Path() / "data" / "generative" / "sr1107_test_lab_171.json") as f:
        lab_with_errors_dict = json.load(f)

    # WHEN
    err_inputs = lab_with_errors_dict["dataset"]["inputs"]
    err_count = 0
    for i in err_inputs:
        actual_output = i.get("actual_output", "")
        if actual_output and commons.ERROR_LLM_HOST in actual_output.lower():
            err_count += 1
            print(f"Found error:\n{actual_output}")
            i["actual_output"] = rewrite_to_message

    # THEN
    print(f"Found {err_count} errors out of {len(err_inputs)} inputs.")

    # save modified dictionary as json
    with open(tmp_path / "sr1107_test_lab_large.json", "w") as f:
        json.dump(lab_with_errors_dict, f, indent=4)


@pytest.mark.skip(reason="One time transformation of the test lab.")
@pytest.mark.generative
def test_john_20240118_remove_ors_suite(tmp_path) -> None:
    # GIVEN

    with open(
        pathlib.Path()
        / "data"
        / "generative"
        / "atlanta"
        / "sr1107_test_suite_large.json"
    ) as f:
        lab_with_errors_dict = json.load(f)

    # WHEN
    for test in lab_with_errors_dict.get("tests", []):
        for test_case in test.get("test_cases", []):
            constraints = test_case.get("constraints", "")
            new_constraints = []
            for c in constraints:
                if isinstance(c, list):
                    if len(constraints) == 1:
                        new_constraints.append(c[0])
                else:
                    new_constraints.append(c)
            test_case["constraints"] = new_constraints

    # THEN
    # save modified dictionary as json
    with open(tmp_path / "sr1107_test_suite_large.json", "w") as f:
        json.dump(lab_with_errors_dict, f, indent=4)


@pytest.mark.skip(reason="One time transformation of the test lab.")
@pytest.mark.generative
def test_john_20240118_remove_ors(tmp_path) -> None:
    # GIVEN

    with open(
        pathlib.Path()
        / "data"
        / "generative"
        / "atlanta"
        / "sr1107_test_lab_large.json"
    ) as f:
        lab_with_errors_dict = json.load(f)

    # WHEN
    input_sources = [
        lab_with_errors_dict["raw_dataset"]["inputs"],
        lab_with_errors_dict["dataset"]["inputs"],
    ]
    or_count = 0
    for err_inputs in input_sources:
        for i in err_inputs:
            constraints = i.get("output_constraints", "")
            new_constraints = []
            for c in constraints:
                if isinstance(c, list):
                    if len(constraints) == 1:
                        new_constraints.append(c[0])
                else:
                    new_constraints.append(c)
            i["output_constraints"] = new_constraints

    # THEN
    print(f"Found {or_count} ORs.")

    # save modified dictionary as json
    with open(tmp_path / "sr1107_test_lab_large.json", "w") as f:
        json.dump(lab_with_errors_dict, f, indent=4)


@pytest.mark.skip(reason="One time transformation of the test lab.")
@pytest.mark.generative
def test_john_20240118_remove_duplicate_prompts(tmp_path) -> None:
    # GIVEN

    with open(
        pathlib.Path()
        / "data"
        / "generative"
        / "atlanta"
        / "sr1107_test_suite_large.json"
    ) as f:
        lab_with_errors_dict = json.load(f)

    # WHEN
    unique_prompts = set()
    duplicate_prompts = set()
    for test in lab_with_errors_dict.get("tests", []):
        for test_case in test.get("test_cases", []):
            p = test_case.get("prompt", "")
            if p in unique_prompts:
                print(f"Duplicate prompt: {p}")
                duplicate_prompts.add(p)
            else:
                unique_prompts.add(p)

    # THEN
    # save modified dictionary as json
    print(f"Found {len(duplicate_prompts)} duplicate prompts:")
    for e, p in enumerate(duplicate_prompts):
        print(f"{e + 1}. {p}")


@pytest.mark.skip(reason="One time transformation of the test lab.")
@pytest.mark.generative
def test_john_20240118_analyze(tmp_path) -> None:
    # GIVEN

    with open(pathlib.Path() / "data" / "generative" / "sr1107_test_lab_171.json") as f:
        lab_with_errors_dict = json.load(f)

    # map: model_key -> LLM model name
    model_key_to_name = {}
    for m in lab_with_errors_dict["models"]:
        model_key_to_name[m["key"]] = m["llm_model_name"]

    # WHEN
    # map: LLM model name -> [errors from actual_output]
    errors_by_model = {}
    err_inputs = lab_with_errors_dict["dataset"]["inputs"]
    err_count = 0
    for i in err_inputs:
        actual_output = i.get("actual_output", "")
        if actual_output and "internal error" in actual_output.lower():
            err_count += 1
            print(f"Found error:\n{actual_output}")
            model_name = model_key_to_name.get(i["model_key"], "")

            if model_name not in errors_by_model:
                errors_by_model[model_name] = []
            errors_by_model[model_name].append(actual_output)

    # THEN
    print(f"Found {err_count} errors out of {len(err_inputs)} inputs.")
    # generate markdown report
    report = "# SR 11-7 Test Lab Error Analysis\n\n"
    report += f"Found **{err_count} errors** out of {len(err_inputs)} inputs.\n\n"
    report += "**Table of Contents**\n\n"
    for e, model_name in enumerate(errors_by_model.keys()):
        report += (
            f"{e + 1}. [{model_name}](#{model_name}) "
            f"({len(errors_by_model[model_name])})\n"
        )
    report += "\n\n"
    for model_name, errors in errors_by_model.items():
        report += f"## {model_name}\n\n"
        for e, err in enumerate(errors):
            report += f"{e + 1}. {model_name} error:\n```\n{err}```\n\n\n"

    # save markdown
    with open(tmp_path / "errors_by_llm_model.md", "w") as f:
        f.write(report)
    # save JSon
    with open(tmp_path / "errors_by_llm_model.json", "w") as f:
        json.dump(errors_by_model, f, indent=4)


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
