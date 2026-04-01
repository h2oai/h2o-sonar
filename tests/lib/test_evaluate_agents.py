# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import pathlib

import pytest

from h2o_sonar.lib.api import models
from h2o_sonar.lib.integrations import genai
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative


@pytest.mark.skip(reason="Tool to import GAIA benchmark data - not a test.")
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_agent_use_h2ogpte(tmp_path):
    #
    # GIVEN
    #

    test_suite_path = "data/generative/h2ogpte_agentic_test_suite_1p.json"
    test_suite = testing.RagTestSuiteConfig.load_from_json(
        file_path=test_suite_path,
    )

    connection = test_utils.health.get_h2ogpte()
    llm_model_names = ["h2oai/h2o-danube3-4b-chat"]

    #
    # WHEN
    #

    llm_models_cfgs = {}
    for llm_model_name in llm_model_names:
        llm_models_cfgs[llm_model_name] = []

        model_cfg = dict()
        model_cfg[genai.H2oGpteRagClient.CFG_LLM_ARGS] = {
            genai.H2oGpteRagClient.CFG_TEMPERATURE: 0.555,
            genai.H2oGpteRagClient.CFG_USE_AGENT: True,
        }
        llm_models_cfgs[llm_model_name].append(model_cfg)

    print(f"llm_models_cfgs:\n{json.dumps(llm_models_cfgs, indent=4)}")

    test_lab = testing.RagTestLab.from_rag_test_suite(
        rag_connection=connection,
        rag_test_suite=test_suite,
        rag_model_type=models.ExplainableModelType.h2ogpte,
        llm_model_names=llm_model_names,
        rag_models_cfgs=llm_models_cfgs,
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )

    print("Evaluated models:")
    print(json.dumps(list(test_lab.evaluated_models.values())[0].to_dict(), indent=4))

    test_lab.build()
    test_lab.complete_dataset(save_as_you_go=tmp_path / "wip_testlab.json")

    #
    # THEN
    #
    test_lab.save_as_json(tmp_path / "test_lab.json")
    print(f"Test lab path: file://{tmp_path}/test_lab.json")


@pytest.mark.skip(reason="Tool to import GAIA benchmark data - not a test.")
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_import_gaia_as_test_suite(tmp_path: pathlib.Path):
    #
    # GIVEN
    #
    gaia_metadata_path = (
        pathlib.Path()
        / "data"
        / "generative"
        / "eval_s3"
        / "gaia"
        / "gaia_benchmark_metadata.jsonl"
    )
    with open(gaia_metadata_path) as json_file:
        json_list = list(json_file)

    #
    # WHEN
    #
    test_suite = testing.RagTestSuiteConfig(
        name="GAIA Test Suite",
        description="GAIA benchmark test suite.",
    )
    test = testing.RagTestConfig(documents=[])

    # for every row in JSonL file
    for json_str in json_list:
        gaia_meta_dict = json.loads(json_str)

        prompt = gaia_meta_dict["Question"]
        expected_answer = gaia_meta_dict["Final answer"]
        file_name = gaia_meta_dict["file_name"]

        # IMPROVE passing of the files
        if file_name:
            handle_files_prompt = (
                "\n\nYOUR FILES:\n"
                f"Consider the file '{file_name}', which can be read from the current "
                f"working directory. If you need to read or write it, output Python "
                f"code in a code block (```python) to do so."
            )
            prompt += handle_files_prompt

        test_suite.add_test_case(
            testing.RagTestCaseConfig(
                prompt=prompt,
                categories=[
                    "agent-evaluation",
                    f"gaia-level-{gaia_meta_dict['Level']}",
                    f"gaia-task-{gaia_meta_dict['task_id']}",
                ],
                expected_output=expected_answer,
                condition=f'"{expected_answer}"',  # ^ to improve
                constraints=[],
                config=test,
            )
        )

    #
    # THEN
    #
    print(json.dumps(test_suite.to_dict(), indent=4))
    test_suite.save_as_json(pathlib.Path(tmp_path) / "test_suite.json")


@pytest.mark.skip(reason="Tool to process GAIA benchmark data - not a test.")
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_gaia_pwd_to_s3_docs(tmp_path: pathlib.Path):
    #
    # GIVEN
    #
    s3_prefix = (
        "https://eval-studio-artifacts.s3.us-east-1.amazonaws.com"
        "/h2o-eval-studio-demo-data/gaia/resources/validation"
        "/"
    )

    gaia_path = (
        pathlib.Path()
        / "data"
        / "generative"
        / "eval_s3"
        / "gaia"
        / "gaia_test_suite_docs_pwd.json"
    )
    with open(gaia_path) as json_file:
        gaia_json = json.load(json_file)

    #
    # WHEN
    #
    main_test_tcs = []
    new_tests = []
    for tc in gaia_json["tests"][0]["test_cases"]:
        print(f"{tc['key']}")
        if "YOUR FILES" in tc["prompt"]:
            prompt = tc["prompt"]
            prompt = prompt.replace(
                "Consider the file '", f"Consider the file {s3_prefix}"
            )
            prompt = prompt.replace(
                "', which can be read from the current "
                "working directory. If you need to read or write it, output Python "
                "code in a code block (```python) to do so.",
                " which can be found among provided collection documents.",
            )
            print(f"New prompt:\n{prompt}")
            tc["prompt"] = prompt

            # :-/
            doc = prompt.split(" ")[-9]
            test = testing.RagTestConfig(documents=[doc])
            test_json = test.to_dict()
            test_json["test_cases"] = [tc]

            new_tests.append(test_json)
        else:
            main_test_tcs.append(tc)

    gaia_json["name"] = "GAIA Test Suite (S3)"
    gaia_json["description"] = (
        "GAIA benchmark test suite with the task files stored in S3 (public URLs)."
    )
    gaia_json["tests"].extend(new_tests)

    #
    # THEN
    #

    # save JSon
    new_gaia_path = tmp_path / "gaia_test_suite_s3.json"
    with open(new_gaia_path, "w") as file:
        json.dump(gaia_json, file, indent=4)

    print(f"Saved to: file://{new_gaia_path}")


@pytest.mark.skip(reason="Tool to process GAIA benchmark data - not a test.")
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_gaia_filter(tmp_path: pathlib.Path):
    #
    # GIVEN
    #
    limit = 50

    gaia_path = (
        pathlib.Path()
        / "data"
        / "generative"
        / "eval_s3"
        / "gaia"
        / "gaia_test_suite_docs_pwd.json"
    )
    with open(gaia_path) as json_file:
        gaia_json = json.load(json_file)

    #
    # WHEN
    #
    tasks_wo_doc = []
    for tc in gaia_json["tests"][0]["test_cases"]:
        print(f"{tc['key']}")
        if "YOUR FILES" in tc["prompt"]:
            continue
        if len(tasks_wo_doc) >= limit:
            break

        tasks_wo_doc.append(tc)

    gaia_json["tests"][0]["test_cases"] = tasks_wo_doc

    #
    # THEN
    #

    # save JSon
    new_gaia_path = tmp_path / f"gaia_test_suite_no_docs_{limit}p.json"
    with open(new_gaia_path, "w") as file:
        json.dump(gaia_json, file, indent=4)

    print(f"Saved to: file://{new_gaia_path}")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
