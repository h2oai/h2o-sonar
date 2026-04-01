# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import pathlib

import pytest


@pytest.mark.skip(reason="One time import of the test data.")
@pytest.mark.parametrize(
    "test_suite_file_name",
    [
        "question_type_Annual_Report_Singtel_output.json",
        "question_type_Broker_Agreement_output.json",
        "question_type_Cyber_Security_Policy_output.json",
        "question_type_Defense_Management_output.json",
        "question_type_Digital_Health_Guidelines_output.json",
        "question_type_EU_AI_Act_output.json",
        "Question_type_Financial_Statements_Alphabet_Tesla_output.json",
        "question_type_Health_Service_Standards_output.json",
        "question_type_Home_Affairs_output.json",
        "question_type_HR_Policy_output.json",
        "question_type_HR_Policy_Procedures_output.json",
        "question_type_HSBC_Annual_Report_output.json",
        "question_type_Immigration_in_Singapore_output.json",
        "question_type_Information_Security_output.json",
        "question_type_Information_Security_Policy_output.json",
        "question_type_Inherent_Risk_Assessment_output.json",
        "question_type_IRS_Document_1_output.json",
        "question_type_IRS_Document_2_output.json",
        "question_type_IRS_Strategic_Operating_Plan_output.json",
        "question_type_Maternal_Child_Healthcare_output.json",
        "question_type_Risk_Management_Guidelines_output.json",
        "question_type_risk_management_policy_output.json",
        "question_type_Risk_Management_Techniques_Tool_output.json",
        "question_type_SA_Home_Affairs_output.json",
        "question_type_Singapore_Cyber_Landscape_output.json",
        "question_type_Singapore_Labour_Force_output.json",
        "question_type_Stanford_Healthcare_Regulations_output.json",
        "question_type_Technical_Report_output.json",
        "question_type_Telcom_Customer_Service_Information_output.json",
        "question_type_Telecom_Infrastructure_Planning_output.json",
        "question_type_Telecommunication_Regulations_output.json",
        "question_type_Telecommunications_regulation_strategy_policy_output.json",
        "question_type_UPC_Agreement_output.json",
        "question_type_US_Veterans_Affairs_output.json",
    ],
)
@pytest.mark.generative
def test_fix_empty_conditions(tmp_path, test_suite_file_name: str) -> None:
    #
    # GIVEN
    #
    print(f"# {test_suite_file_name} {50 * '='}")
    test_suite_path = (
        pathlib.Path()
        / "data"
        / "eval"
        / "h2o-eval-studio-suite-library"
        / test_suite_file_name
    )

    #
    # WHEN
    #

    with open(test_suite_path) as f:
        test_suite_dict = json.load(f)

    bugs_found = 0
    for t in test_suite_dict.get("tests", []):
        for tc in t.get("test_cases", []):
            if tc.get("constraints"):
                if tc["constraints"] == [""]:
                    bugs_found += 1
                    print(
                        f"{bugs_found} {test_suite_path}: empty condition found in "
                        f"test case: '{tc['prompt'].strip()}'"
                    )
                    tc["constraints"] = []

    if bugs_found > 0:
        with open(test_suite_path, "w") as f:
            json.dump(test_suite_dict, f, indent=4)

    #
    # THEN
    #
    print(f"Fixed {test_suite_file_name} processed successfully.")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
