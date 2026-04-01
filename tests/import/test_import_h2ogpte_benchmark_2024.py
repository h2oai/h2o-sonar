# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import datetime
import json
import pathlib
import re

import datatable
import pytest

from h2o_sonar import loggers
from h2o_sonar.evaluators import rag_tokens_presence_evaluator
from h2o_sonar.utils import testing
from tests import test_utils


"""h2oGPTe benchmark import / mining."""


def _h2ogpte_expecteds_to_condition(expecteds_str: str) -> str:
    """Convert h2oGPTe expecteds to test case condition:

    - AND sections are enclosed in parentheses () for better readability
      (due to the OR operator precedence they are not necessary)
    - examples:
        - "[['$25.2', 'billion'], ['$25.173', 'billion']]"
          -> '("$25.2" AND "billion") OR ("$25.173" AND "billion")'
        - "[['15,969', 'million'], ['15M']]"
          -> '("15,969" AND "million") OR "15M"'

    Parameters
    ----------
    expecteds_str : str
        The expecteds string from the h2oGPTe CSV.

    Returns
    -------
    str
        The condition string for the test case.

    """

    condition = ""  # new condition
    if expecteds_str:
        expecteds = eval(expecteds_str)
        if expecteds and isinstance(expecteds, list):
            for and_section in expecteds:
                if condition:
                    condition += " OR "

                if isinstance(and_section, list):
                    if len(and_section) > 1:
                        condition += "("
                    and_condition = ""
                    for c in and_section:
                        if and_condition:
                            and_condition += " AND "
                        # if has a letter, make it case insensitive
                        check = any(char.isalpha() for char in c)
                        if check:
                            and_condition += f'regexp("(?i){c}")'
                        else:
                            and_condition += f'"{c}"'
                    condition += f"{and_condition}"
                    if len(and_section) > 1:
                        condition += ")"
                else:
                    raise ValueError(
                        f"Unexpected expecteds format - must be list: {expecteds}"
                    )
        else:
            raise ValueError(f"Unexpected expecteds format - must be list: {expecteds}")

    # h2oGPTe constraints vs. Sonar constraints:
    # + reversed AND/OR meaning
    # + regexps to be added for case-insensitive matching
    # SKIPPED as not required

    return condition


@pytest.mark.skip(reason="A tool for regexp design.")
@pytest.mark.parametrize(
    "r_condition,answer,is_match",
    [
        (
            "(?i)inflation affected gross profit.",
            "Inflation affected GROSS profit.",
            True,
        )
    ],
)
@pytest.mark.generative
def test_case_insensitive_re_match(r_condition: str, answer: str, is_match: bool):
    #
    # GIVEN
    #
    print(f"\nCondition: {r_condition}")

    #
    # WHEN
    #
    match = re.search(r_condition, answer)

    #
    # THEN
    #
    print(f"Match: {match}")
    assert is_match == bool(match)


@pytest.mark.skip(reason="A tool for one time import of the test data.")
@pytest.mark.parametrize(
    "expecteds,expected_condition",
    [
        ("[['6%'], ['6 percent']]", '"6%" OR regexp("(?i)6 percent")'),
        ("[['15,969', 'million']]", '("15,969" AND regexp("(?i)million"))'),
        (
            "[['15,969', 'million'], ['15M']]",
            '("15,969" AND regexp("(?i)million")) OR regexp("(?i)15M")',
        ),
        (
            "[['$25.2', 'billion'], ['$25.173', 'billion'], ['$25,173', 'million']]",
            '("$25.2" AND regexp("(?i)billion")) '
            'OR ("$25.173" AND regexp("(?i)billion")) '
            'OR ("$25,173" AND regexp("(?i)million"))',
        ),
    ],
)
@pytest.mark.generative
def test_import_condition(expecteds, expected_condition):
    condition = _h2ogpte_expecteds_to_condition(expecteds)
    print(f"\n{condition}")
    assert expected_condition == condition


@pytest.mark.parametrize(
    "condition,text,is_match",
    [
        # simple condition
        ('"6%" OR regexp("(?i)6 percent")', "6%", True),
        ('"6%" OR regexp("(?i)6 percent")', "60%", False),
        ('"6%" OR regexp("(?i)6 percent")', "Made 6 PERCENT!", True),
        ('"6%" OR regexp("(?i)6 percent")', "Only 6 PeRcenT.", True),
        # complex condition
        (
            '("$25.2" AND regexp("(?i)billion")) '
            'OR ("$25.173" AND regexp("(?i)billion")) '
            'OR ("$25,173" AND regexp("(?i)million"))',
            "$25.2BILLION",
            True,
        ),
        (
            '("$25.2" AND regexp("(?i)billion")) '
            'OR ("$25.173" AND regexp("(?i)billion")) '
            'OR ("$25,173" AND regexp("(?i)million"))',
            "$25,173 Million",
            True,
        ),
        (
            '("$25.2" AND regexp("(?i)billion")) '
            'OR ("$25.173" AND regexp("(?i)billion")) '
            'OR ("$25,173" AND regexp("(?i)million"))',
            "mismatch",
            False,
        ),
    ],
)
@pytest.mark.generative
def test_case_condition_i_re_support(condition: str, text: str, is_match: bool):
    #
    # GIVEN
    #
    print(f"\nCondition: {condition}\nText: {text}")
    condition_evaluator = rag_tokens_presence_evaluator.ConditionEvaluator(
        c=condition, logger=loggers.SonarPrintLogger()
    )

    #
    # WHEN
    #
    (e_result, _) = condition_evaluator.evaluate(text)

    #
    # THEN
    #
    print(f"Result: {e_result}")
    assert is_match == e_result


@pytest.mark.skip(reason="One time import of the test data.")
@pytest.mark.parametrize(
    "h2ogpte_csv",
    [
        "data/generative/incoming/h2ogpte/e2e_df_condition.csv",
        "data/generative/incoming/h2ogpte/e2e_df_20240826.csv",
        "data/generative/incoming/h2ogpte/e2e_df_20241001.csv",
    ],
)
@pytest.mark.generative
def test_import(tmp_path, h2ogpte_csv):
    #
    # GIVEN
    #
    max_tests = 0
    h2ogpte_csv_path = test_utils.find_locally(h2ogpte_csv)

    #
    # WHEN
    #
    now = datetime.datetime.now()
    ts = (
        f"{now.year}-{now.month:02d}-{now.day:02d}"
        f"_{now.hour:02d}h{now.minute:02d}m{now.second:02d}s"
    )

    # load the CSV using datatable
    df = datatable.fread(h2ogpte_csv_path)

    # convert the datatable to test suite
    test_suite = testing.RagTestSuiteConfig(
        name="h2oGPTe Benchmark",
        description=f"h2oGPTe benchmark data import ({ts}).",
    )

    # tests map: test name -> testing.TestConfig
    h2ogpte_dict = {}
    for i in range(df.shape[0]):
        row = df[i, :]

        name = row[0, "name"]
        url = row[0, "URL"]
        question = row[0, "question"]
        expecteds = row[0, "expecteds"]
        # must_pass = row[0, "must_pass"]  # unused

        # group the tests by name
        if name not in h2ogpte_dict:
            h2ogpte_dict[name] = testing.RagTestConfig(documents=[url])

        # condition parsing
        # - all conditions to be case insensitive VERIFY
        # - OR syntax:
        #   - [['6%'], ['6 percent']] ->  "6%" OR "6 percent"
        #     (What percentage is in RMBS?)
        #   - [['15,969', 'million']] -> "15,969" AND "million"
        #   - "[['$25.2', 'billion'], ['$25.173', 'billion'], ['$25,173', 'million']]",
        condition = _h2ogpte_expecteds_to_condition(expecteds)
        # add the test case
        print("Condition:", condition)
        if not condition:
            raise ValueError(f"Empty condition for test: {name}")
        test_case = testing.RagTestCaseConfig(
            prompt=question,
            categories=["question-answering"],
            expected_output="",
            condition=condition,
            config=h2ogpte_dict[name],
        )

        test_suite.add_test_case(test_case)

    # optionally trim the test suite
    if max_tests:
        test_suite.trim_tests(max_tests)

    #
    # THEN
    #

    test_suite_path = test_suite.save_as_json(
        tmp_path / f"h2ogpte_benchmark_test_suite_{ts}.json"
    )
    print(f"Test suite: file://{test_suite_path}")

    # assert the conditions correctness
    if "e2e_df_condition.csv" in h2ogpte_csv:
        with open(test_suite_path) as f:
            s = f.read()
        assert "6 percent" in s


@pytest.mark.skip(reason="A tool for regexp design.")
@pytest.mark.generative
def test_check_and_fix_h2ogpte_keys():
    #
    # GIVEN
    #
    test_lab_path = pathlib.Path(
        "/home/user/h2o-eval-studio-demo-data/"
        "h2ogpte-benchmark-2024-Oct-01--h2ogpte/test_lab.json"
    )
    with open(test_lab_path) as f:
        test_lab = json.load(f)

    #
    # WHEN
    #
    datasets = ["raw_dataset", "dataset"]
    for d in datasets:
        for i in test_lab[d]["inputs"]:
            # IMPORTANT:
            # - test LAB keys are not unique - by coincidence, but it is used as feature

            k = i["key"]
            print(k)

            if len(k) != 73:
                raise RuntimeError(f"Invalid key: {k}")

            (k_tc, k_m) = k.split("_")
            i["key"] = k_tc

    #
    # THEN
    #

    new_test_lab_path = test_lab_path.parent / "test_lab_fixed.json"
    with open(new_test_lab_path, "w") as handle:
        json.dump(test_lab, handle, indent=2)
    print(f"file://{new_test_lab_path}")


@pytest.mark.skip(reason="A tool for regexp design.")
@pytest.mark.generative
def test_check_and_fix_bedrock_keys():
    #
    # GIVEN
    #
    test_lab_path = pathlib.Path(
        "/home/user/h2o-eval-studio-demo-data/"
        "h2ogpte-benchmark-2024-Oct-01--amazon-bedrock/test_lab.json"
    )
    with open(test_lab_path) as f:
        test_lab = json.load(f)

    #
    # WHEN
    #
    for i in test_lab["dataset"]["inputs"]:
        # IMPORTANT:
        # - test LAB keys are not unique - by coincidence, but it is used as feature

        k = i["key"]
        print(k)

        if len(k) != 73:
            raise RuntimeError(f"Invalid key: {k}")

        (k_tc, k_m) = k.split("_")
        i["key"] = k_tc

    #
    # THEN
    #

    new_test_lab_path = test_lab_path.parent / "test_lab_fixed.json"
    with open(new_test_lab_path, "w") as handle:
        json.dump(test_lab, handle, indent=2)
    print(f"file://{new_test_lab_path}")


@pytest.mark.skip(reason="A tool for regexp design.")
@pytest.mark.parametrize(
    "test_lab_path",
    [
        "/home/user/h2o-eval-studio-demo-data/"
        "h2ogpte-benchmark-2024-Oct-01--h2ogpte/test_lab.json",
    ],
)
@pytest.mark.generative
def test_cp_dataset_to_raw_dataset(test_lab_path):
    #
    # GIVEN
    #
    test_lab_path = pathlib.Path(test_lab_path)

    #
    # WHEN
    #
    with open(test_lab_path) as f:
        test_lab = json.load(f)

    for i in test_lab["dataset"]["inputs"]:
        raw_i = i.copy()
        raw_i["context"] = []
        raw_i["actual_output"] = ""
        raw_i["actual_duration"] = 0.0
        raw_i["cost"] = 0.0

        test_lab["raw_dataset"]["inputs"].append(raw_i)

    #
    # THEN
    #

    new_test_lab_path = test_lab_path.parent / "test_lab_with_raw_dataset.json"
    with open(new_test_lab_path, "w") as handle:
        json.dump(test_lab, handle, indent=2)
    print(f"file://{new_test_lab_path}")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
