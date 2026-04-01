# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import pathlib
import shutil
import sys

import pytest

from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import models
from h2o_sonar.lib.integrations import genai
from h2o_sonar.utils import progress
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative


@pytest.mark.skip(reason="This test extracts test lab from the evaluation ZIP archive")
@pytest.mark.parametrize(
    "part,eval_result_path,interpterations_json_path",
    [
        # 1-2-4 merge of h2oGPTE benchmark test labs
        (
            1,
            "/home/user/H2OGPTE-benchmark-2024-10-01/results/part-1/zip/explainer_"
            "h2o_sonar_evaluators_rag_tokens_presence_evaluator_RagStrStrEvaluator_"
            "0cc1fde5-be74-4b4d-b4e3-fbf6b776aaca/global_llm_eval_results/"
            "application_json/explanation.json",
            "/home/user/H2OGPTE-benchmark-2024-10-01/results/part-1/zip/"
            "interpretation.json",
        ),
        (
            2,
            "/home/user/H2OGPTE-benchmark-2024-10-01/results/part-2/zip/explainer_"
            "h2o_sonar_evaluators_rag_tokens_presence_evaluator_RagStrStrEvaluator_"
            "443837cb-f767-481e-8b4e-a38a258fbb43/global_llm_eval_results/"
            "application_json/explanation.json",
            "/home/user/H2OGPTE-benchmark-2024-10-01/results/part-2/zip/"
            "interpretation.json",
        ),
        (
            3,
            "/home/user/H2OGPTE-benchmark-2024-10-01/results/part-3/zip/explainer_"
            "h2o_sonar_evaluators_rag_tokens_presence_evaluator_RagStrStrEvaluator_"
            "634e393a-7b26-49ec-8543-9b3d4eeb5378/global_llm_eval_results/"
            "application_json/explanation.json",
            "/home/user/H2OGPTE-benchmark-2024-10-01/results/part-3/zip/"
            "interpretation.json",
        ),
        # Repro result to test lab 2025-04-10
        (
            1,
            "/home/user/tmp/repro/explainer_h2o_sonar_evaluators_rag_ragas_evaluator"
            "_RagasEvaluator_902673ee-520e-4fb4-b0f9-43ec16c9e8cd/global_llm_eval"
            "_results/application_json/explanation.json",
            "/home/user/tmp/repro/interpretation.json",
        ),
        # Encoding Guardrail evaluator bug #1415
        (
            1,
            "/tmp/bug-001/explainer_h2o_sonar_evaluators_encoding_guardrail_evaluator"
            "_EncodingGuardrailEvaluator_ec017641-bc31-40c0-bbaf-864cf331099d/"
            "global_llm_eval_results/application_json/explanation.json",
            "/tmp/bug-001/interpretation.json",
        ),
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_testlab_from_zip_archive(
    tmp_path, part, eval_result_path, interpterations_json_path
):
    #
    # GIVEN
    #

    # path to the extracted evaluation ZIP archive
    eval_result_path = pathlib.Path(eval_result_path)
    interpterations_json_path = pathlib.Path(interpterations_json_path)

    #
    # WHEN
    #

    test_lab = testing.RagTestLab.from_eval_results(
        eval_results_path=eval_result_path,
        interpretation_json_path=interpterations_json_path,
    )

    #
    # THEN
    #
    imported_test_lab_path = tmp_path / f"test_lab_split_{part}.json"
    test_lab.save_as_json(imported_test_lab_path)
    print(f"Imported test lab: file://{imported_test_lab_path}")


@pytest.mark.skip(reason="This test merges test lab splits into a single test lab")
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_merge_h2ogpte_testlabs(tmp_path):
    """Merge h2oGPte splits into a single test lab."""

    #
    # GIVEN
    #

    base_path = pathlib.Path(
        "/home/user/H2OGPTE-benchmark-2024-10-01/"
        "h2ogpte_benchmark_test_suite_2024-10-01_15h37m17s"
    )
    test_labs_split_paths = [
        base_path / f"test_lab_split_{i}.json" for i in range(0, 3)
    ]

    #
    # WHEN
    #

    test_lab = None
    connection_id = ""
    for p in test_labs_split_paths:
        assert pathlib.Path(p).exists()

        if not test_lab:
            with open(p) as f:
                test_lab = json.load(f)
                test_lab["raw_dataset"]["inputs"] = []
                connection_id = test_lab["models"][0]["connection"]
        else:
            # merge test lab to the first one @ JSon level
            with open(p) as f:
                test_lab_split = json.load(f)

            # dataset
            for i in test_lab_split["dataset"]["inputs"]:
                test_lab["dataset"]["inputs"].append(i)

            # models
            for m in test_lab_split["models"]:
                m["connection"] = connection_id
                test_lab["models"].append(m)

            # LLM model names
            test_lab["llm_model_names"] = list(
                set(test_lab["llm_model_names"]).union(
                    set(test_lab_split["llm_model_names"])
                )
            )

            # docs
            test_lab["docs_cache"] = list(
                set(test_lab["docs_cache"]).union(set(test_lab_split["docs_cache"]))
            )

    test_lab["name"] = "h2ogpte Benchmark"
    test_lab["description"] = "h2ogpte Benchmark test lab."

    # save merged test lab
    merged_test_lab_path = tmp_path / "test_lab.json"
    with open(merged_test_lab_path, "w") as f:
        json.dump(test_lab, f, indent=4)
    print(f"Merged test lab: file://{merged_test_lab_path}")


@pytest.mark.skip(reason="This test filters test lab w/ test suite")
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_filter_lab_with_suite(tmp_path):
    """Filter given test lab and keep only test cases from the test suite."""
    #
    # GIVEN
    #

    test_case_path = pathlib.Path(
        "/home/user/H2OGPTE-benchmark-2024-10-01/amazon_bedrock_2024_10_01/"
        "test_suite.json"
    )
    # load test suite & gather keys
    test_suite = testing.RagTestSuiteConfig.load_from_json(test_case_path)
    test_cases_keys = [tc.key for tc in test_suite.test_cases]
    print("Test suite:")
    print(f"  Tests: {len(test_suite.test_cfgs)}")
    print(f"  Test cases: {len(test_suite.test_cases)}")
    print(f"  Test case keys: {len(test_cases_keys)}")

    test_lab_path = pathlib.Path(
        "/home/user/H2OGPTE-benchmark-2024-10-01/backup/"
        "h2ogpte_benchmark_full_2024-10-01_15h37m17s/test_lab.json"
    )
    # load test lab
    with open(test_lab_path) as f:
        test_lab = json.load(f)
    print("Test lab:")
    print(f"  Dataset inputs: {len(test_lab['dataset']['inputs'])}")
    print(f"  Models: {len(test_lab['models'])}")
    print(f"  LLM model names: {len(test_lab['llm_model_names'])}")

    #
    # WHEN
    #

    # filter test lab dataset inputs by test suite keys
    filtered_inputs = []
    for i in test_lab["dataset"]["inputs"]:
        if i["key"] in test_cases_keys:
            filtered_inputs.append(i)
    test_lab["dataset"]["inputs"] = filtered_inputs

    # filter models: index model keys which remain in the dataset
    remaining_keys = {i["model_key"] for i in test_lab["dataset"]["inputs"]}
    test_lab["models"] = [m for m in test_lab["models"] if m["key"] in remaining_keys]

    # filter docs cache
    remaining_docs_cache = set()
    for i in test_lab["dataset"]["inputs"]:
        for d in i["corpus"]:
            remaining_docs_cache.add(d)
    test_lab["docs_cache"] = list(remaining_docs_cache)

    # no action needed for llm model names

    #
    # THEN
    #

    # save filtered test lab
    filtered_test_lab_path = tmp_path / "filtered_test_lab.json"
    with open(filtered_test_lab_path, "w") as f:
        json.dump(test_lab, f, indent=4)
    print(f"Filtered test lab: file://{filtered_test_lab_path}")
    print(f"  Dataset inputs: {len(test_lab['dataset']['inputs'])}")
    print(f"  Models: {len(test_lab['models'])}")
    print(f"  LLM model names: {len(test_lab['llm_model_names'])}")
    print(f"  Docs cache: {len(test_lab['docs_cache'])}")


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_tests_vs_test_cases_mapping_to_models(tmp_path):
    """Test that test cases and tests are correctly mapped to evaluated models."""

    #
    # GIVEN
    #
    connection = test_utils.health.get_h2ogpte()
    foo_llm_model_name = "h2o-sonar/foo-1.0"
    test_suite_path = "data/generative/ci_rag_test_suite.json"
    # 2x tests: 1x test case + 2x test cases
    test_suite = testing.RagTestSuiteConfig.load_from_json(
        test_utils.find_locally(test_suite_path)
    )
    assert len(test_suite.test_cases) == 3
    assert len(test_suite.tests) == 2

    #
    # WHEN
    #
    test_lab = testing.RagTestLab.from_rag_test_suite(
        rag_connection=connection,
        rag_test_suite=test_suite,
        rag_model_type=models.ExplainableModelType.h2ogpte,
        llm_model_names=[foo_llm_model_name],
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )

    #
    # THEN
    #
    print(test_lab)
    assert test_lab
    # ensure that there is the same number of tests and evaluated models
    assert len(test_lab.evaluated_models) == len(test_suite.tests)


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_complete_lab_on_given_collections_call(tmp_path):
    """Test that lab can be completed on existing collections rather than
    creating them using the ``build()`` method.

    **Scenario CALL :**

      - test **lab** complete() method gets the collection ID as parameter and this
        collection ID is subsequently used to complete the lab for all test cases
        (corpus within the test suite are ignored)

    """

    #
    # GIVEN
    #

    # create collection in the h2oGPTe server
    connection = test_utils.health.get_h2ogpte()
    connection.extra_params = {
        "timeout": 60,  # specify timeout in seconds for operations atop the connection
    }
    given_foo_llm_model_name = "h2o-sonar/foo-1.0"
    given_test_suite_path = "data/generative/ci_rag_test_suite.json"
    given_test_suite = testing.RagTestSuiteConfig.load_from_json(
        test_utils.find_locally(given_test_suite_path)
    )
    given_test_suite.test_cases = given_test_suite.test_cases[0:1]
    given_test_lab = testing.RagTestLab.from_rag_test_suite(
        rag_connection=connection,
        rag_test_suite=given_test_suite,
        rag_model_type=models.ExplainableModelType.h2ogpte,
        llm_model_names=[given_foo_llm_model_name],
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )
    given_test_lab.build()

    # get the collection ID
    given_model = next(iter(given_test_lab.evaluated_models.values()))
    predefined_collection_id = given_model.collection_id
    print(f"Predefined collection ID:  \n{predefined_collection_id}")
    predefined_collection_name = given_model.collection_name
    print(f"Predefined collection name:  \n{predefined_collection_name}")

    #
    # WHEN: string OR dictionary {test case key: collection ID}
    #

    # COMPLETE the test lab using the PREDEFINED collection ID
    llm_model_names = genai.H2oGpteRagClient(connection).list_llm_model_names()[0:1]
    # 1) test suite
    test_suite_path = "data/generative/ci_rag_test_suite.json"
    test_suite = testing.RagTestSuiteConfig.load_from_json(
        test_utils.find_locally(test_suite_path)
    )
    test_suite.test_cfgs = {
        test_suite.test_cases[0].config.key: test_suite.test_cases[0].config
    }
    test_suite.test_cases = [test_suite.test_cases[0]]
    # 2) complete the test lab using the predefined collection ID
    predefined_col_as_str = str(predefined_collection_id)
    predefined_col_as_dict = {
        # test key -> predefined collection ID
        test_suite.test_cases[0].config.key: predefined_collection_id
    }
    for e, p_c_i in enumerate([predefined_col_as_str, predefined_col_as_dict]):
        test_lab = testing.RagTestLab.from_rag_test_suite(
            rag_connection=connection,
            rag_test_suite=test_suite,
            rag_model_type=models.ExplainableModelType.h2ogpte,
            llm_model_names=llm_model_names,
            predefined_collection_id=p_c_i,
            docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
        )
        # check that the collection ID is set
        for em in test_lab.evaluated_models.values():
            assert em.collection_id
        # 3) build() test lab ~ just set the collection ID
        test_lab.build()
        test_lab_path = tmp_path / f"{e}_01_built_test_lab.json"
        test_lab.save_as_json(test_lab_path)
        # 4) complete the test lab
        test_lab.complete_dataset()
        test_lab_path = tmp_path / f"{e}_02_completed_test_lab.json"
        test_lab.save_as_json(test_lab_path)

        #
        # THEN
        #
        assert test_lab
        assert test_lab.dataset.inputs

    print(f"Test labs:\n  file://{tmp_path}")


@pytest.mark.skip(reason="Tool to build and complete ad-hoc test labs")
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_build_lab(tmp_path, llm_suite="auto"):
    #
    # GIVEN
    #
    h2ogpte_connection = test_utils.health.get_h2ogpte()
    available_llm_model_names = genai.H2oGpteRagClient(
        h2ogpte_connection
    ).list_llm_model_names()
    if llm_suite == "auto":
        llm_model_names = [
            genai.H2oGpteRagClient.MODEL_SPEC_COL_OPT_E,
            genai.H2oGpteRagClient.MODEL_SPEC_COL_OPT_N,
            genai.H2oGpteRagClient.MODEL_SPEC_AUTO,
        ]
    elif llm_suite == "full":
        llm_model_names = available_llm_model_names
    else:
        # OPTIONAL SMALLER TEST - choose stable models
        llm_model_names = [
            "h2oai/h2o-danube3-4b-chat",
            "h2oai/h2o-danube2-1.8b-chat",
        ]
        print("LLM models:")
        for m in llm_model_names:
            print(f"  {m}")
            if m not in available_llm_model_names:
                raise ValueError(f"LLM model '{m}' not available on the server")

    test_suite_path = (
        # "data/generative/h2ogpte-benchmark/"
        # "h2ogpte_benchmark_test_suite_2024-08-23_13h31m09s-17t.json"
        "data/generative/eval_llm/perturbed_test_suite_2p.json"
        # "data/generative/nist-ai-600-1--test-suite--30p.json"
        # "data/generative/procedure_eval_test_suite_small.json"
    )

    test_suite = testing.RagTestSuiteConfig.load_from_json(
        file_path=test_suite_path,
    )

    print(
        f"\nRunning test lab completion:"
        f"\n  host: {h2ogpte_connection.server_url}"
        f"\n  test suite: {test_suite_path}"
        f"\n  {len(test_suite.test_cfgs)} tests"
        f"\n  {len(test_suite.test_cases)} test cases"
    )

    #
    # WHEN
    #
    if None not in llm_model_names:
        print(
            f"Building test lab for LLMs:\n"
            f"  {', '.join(llm_model_names)}\n"
            f"  from test suite: {test_suite_path}"
            f"  with {len(test_suite.test_cases)} test cases"
        )

    test_lab = testing.RagTestLab.from_rag_test_suite(
        rag_connection=h2ogpte_connection,
        rag_test_suite=test_suite,
        rag_model_type=models.ExplainableModelType.h2ogpte,
        llm_model_names=llm_model_names,
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )

    # progress
    progress_callback = progress.LoggingProgressCallbackContext(
        logger=test_lab.logger,
        prefix="[TOP LEVEL TEST CALLBACK]",
        name="Test lab build and complete progress",
    )
    lab_build_progress = progress_callback.get_sub_callback_for_progress(
        min_progress=0.0, max_progress=0.33, verbose_children=False
    )
    lab_completion_progress = progress_callback.get_sub_callback_for_progress(
        min_progress=0.34, max_progress=1.0, verbose_children=True
    )

    test_lab.save_as_json(tmp_path / "wip_testlab_before_complete.json")

    # deploy the test lab configuration to the h2oGPTe server
    test_lab.build(progress_callback=lab_build_progress)
    # complete dataset w/ actual values from the h2oGPTe server (answer, duration, ...)
    test_lab.complete_dataset(
        save_as_you_go=tmp_path / "wip_testlab.json",
        parallelize=(
            testing.TestLab.PARALLEL_RUN
            if sys.platform == "linux"
            else testing.TestLab.SEQUENTIAL_RUN
        ),
        progress_callback=lab_completion_progress,
    )

    #
    # THEN
    #
    test_lab.save_as_json(tmp_path / "test_lab.json")
    test_lab.save_as_json("data/generative/procedure_eval_test_lab_small.json")
    print(f"path: {tmp_path}/test_lab.json")


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.parametrize(
    "test_suite_path,is_rag",
    [
        ("data/generative/eval_llm/perturbed_test_suite_2p.json", True),
        ("data/generative/eval_llm/statefull_are_you_sure_test_suite.json", False),
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_build_lab_multi_turn(tmp_path, test_suite_path: str, is_rag: bool):
    #
    # GIVEN
    #
    h2ogpte_connection = test_utils.health.get_h2ogpte()

    # reproducible contradictory answers issue, but model not on all servers
    # "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    # whatever model which is present on all servers to test APIs
    all_llm_model_names = genai.H2oGpteRagClient(
        h2ogpte_connection
    ).list_llm_model_names()
    llm_candidates = [
        given_generative.LLM_DANUBE_3,
        given_generative.LLM_CLAUDE_SONNET_37,
        given_generative.LLM_CLAUDE_SONNET,
    ]
    llm_model_names = [m for m in llm_candidates if m in all_llm_model_names]
    if not llm_model_names:
        raise ValueError(
            f"None of the candidate LLM models {llm_candidates} are available on the "
            f"h2oGPTe server"
        )
    else:
        llm_model_names = llm_model_names[0:1]

    llm_model_names = (
        [
            genai.H2oGpteRagClient.MODEL_SPEC_COL_OPT_E,
            genai.H2oGpteRagClient.MODEL_SPEC_AUTO,
        ]
        if is_rag
        else llm_model_names
    )

    test_suite = testing.RagTestSuiteConfig.load_from_json(
        file_path=test_suite_path,
    )

    # stateful vs. stateless test lab completion
    multi_turn = True

    #
    # WHEN
    #

    print(
        f"\nBuilding test lab for LLMs:"
        f"\n  host: {h2ogpte_connection.server_url}"
        f"\n  is RAG: {is_rag}"
        f"\n  test suite: {test_suite_path}"
        f"\n  {len(test_suite.test_cfgs)} tests"
        f"\n  {len(test_suite.test_cases)} test cases"
        f"\n  multi-turn: {multi_turn}"
        f"\n  {', '.join(llm_model_names)}"
        f" from test suite: {test_suite_path}"
        f" with {len(test_suite.test_cases)} test cases"
        f"\n\n"
    )

    if is_rag:
        test_lab = testing.RagTestLab.from_rag_test_suite(
            rag_connection=h2ogpte_connection,
            rag_test_suite=test_suite,
            rag_model_type=models.ExplainableModelType.h2ogpte,
            llm_model_names=llm_model_names,
            docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
        )
    else:
        test_lab = testing.RagTestLab.from_llm_test_suite(
            llm_host_connection=h2ogpte_connection,
            llm_test_suite=test_suite,
            llm_model_type=models.ExplainableModelType.h2ogpte_llm,
            llm_model_names=llm_model_names,
        )

    # progress
    progress_callback = progress.LoggingProgressCallbackContext(
        logger=test_lab.logger,
        prefix="[TOP LEVEL TEST CALLBACK]",
        name="Test lab build and complete progress",
    )
    lab_build_progress = progress_callback.get_sub_callback_for_progress(
        min_progress=0.0, max_progress=0.33, verbose_children=False
    )
    lab_completion_progress = progress_callback.get_sub_callback_for_progress(
        min_progress=0.34, max_progress=1.0, verbose_children=True
    )

    test_lab.build(progress_callback=lab_build_progress)
    test_lab.complete_dataset(
        save_as_you_go=tmp_path / "wip_testlab.json",
        parallelize=testing.TestLab.PARALLEL_RUN,
        progress_callback=lab_completion_progress,
        multi_turn=multi_turn,
    )

    #
    # THEN
    #

    completed_test_lab_path = tmp_path / "completed_test_lab.json"
    test_lab.save_as_json(completed_test_lab_path)
    print(f"Test lab path: {completed_test_lab_path}")


@pytest.mark.skip(reason="Tool to build and complete ad-hoc test labs")
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_build_lab_bedrock(tmp_path):
    #
    # GIVEN
    #
    bedrock_connection = given_generative.AMAZON_BEDROCK
    available_llm_model_names = genai.AmazonBedrockRagClient(
        bedrock_connection
    ).list_llm_model_names()
    # OPTIONAL SMALLER TEST - choose stable models
    llm_model_names = [
        "anthropic.claude-3-haiku-20240307-v1:0",
    ]
    print("LLM models:")
    for m in llm_model_names:
        print(f"  {m}")
        if m not in available_llm_model_names:
            raise ValueError(f"LLM model '{m}' not available on the server")

    test_suite_path = "data/generative/ci_rag_test_suite_bedrock.json"
    test_suite = testing.RagTestSuiteConfig.load_from_json(
        file_path=test_suite_path,
    )

    #
    # WHEN
    #
    test_lab = testing.RagTestLab.from_rag_test_suite(
        rag_connection=bedrock_connection,
        rag_test_suite=test_suite,
        rag_model_type=models.ExplainableModelType.amazon_bedrock_rag,
        llm_model_names=llm_model_names,
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )

    # progress
    progress_callback = progress.LoggingProgressCallbackContext(
        logger=test_lab.logger,
        prefix="[TOP LEVEL TEST CALLBACK]",
        name="Test lab build and complete progress",
    )
    lab_build_progress = progress_callback.get_sub_callback_for_progress(
        min_progress=0.0, max_progress=0.33, verbose_children=False
    )
    lab_completion_progress = progress_callback.get_sub_callback_for_progress(
        min_progress=0.34, max_progress=1.0, verbose_children=True
    )

    # deploy the test lab configuration to the h2oGPTe server
    test_lab.build(progress_callback=lab_build_progress)
    # complete dataset w/ actual values from the h2oGPTe server (answer, duration, ...)
    test_lab.complete_dataset(
        save_as_you_go=tmp_path / "wip_testlab.json",
        parallelize=testing.TestLab.PARALLEL_RUN,
        progress_callback=lab_completion_progress,
    )

    #
    # THEN
    #
    test_lab.save_as_json(tmp_path / "test_lab.json")
    print(f"path: {tmp_path}/test_lab.json")
    test_lab.purge()


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_lab_stats(tmp_path):
    #
    # GIVEN
    #
    h2ogpte_connection = test_utils.health.get_h2ogpte()
    test_lab = testing.RagTestLab.load_from_json(
        llm_host_connection=h2ogpte_connection,
        file_path="data/generative/nist-ai-600-1--test-lab--30p-5m.json",
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )

    #
    # WHEN
    #
    stats = test_lab.stats()

    #
    # THEN
    #
    print(f"Test lab stats:\n{json.dumps(stats, indent=4)}")
    assert stats


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_perturbed_lab_build(tmp_path):
    #
    # GIVEN
    #
    h2ogpte_connection = test_utils.health.get_h2ogpte()

    test_suite_path = "data/generative/eval_llm/perturbed_test_suite_2p.json"
    test_suite = testing.RagTestSuiteConfig.load_from_json(
        file_path=test_suite_path,
    )
    test_suite.save_as_json(tmp_path / "perturbed_test_suite.json")

    #
    # WHEN
    #
    test_lab = testing.RagTestLab.from_llm_test_suite(
        llm_host_connection=h2ogpte_connection,
        llm_test_suite=test_suite,
        llm_model_type=testing.ExplainableModelTypes.h2ogpte_llm,
        llm_model_names=[given_generative.LLM_CLAUDE_SONNET],
    )
    test_lab.build()
    test_lab.complete_dataset()
    test_lab.save_as_json(tmp_path / "perturbed_test_lab.json")

    # LLM dataset
    test_lab.raw_dataset.save_as_json(tmp_path / "perturbed_RAW_llm_dataset.json")
    test_lab_dataset_path = tmp_path / "perturbed_llm_dataset.json"
    test_lab.dataset.save_as_json(test_lab_dataset_path)
    test_lab.dataset.to_datatable().to_csv(str(tmp_path / "perturbed_llm_dataset.csv"))
    # load dataset json
    loaded_dataset = datasets.LlmDataset.load_from_json(test_lab_dataset_path)
    loaded_dataset.save_as_json(tmp_path / "perturbed_llm_dataset_LOADED.json")

    # LLM eval results
    eval_results = datasets.LlmEvalResults()
    for r in test_lab.dataset.inputs:
        eval_results.add_result(
            datasets.LlmEvalResults.LlmEvalResultRow(
                dataset_row=r, metrics={"llm_score": 0.5}
            )
        )
    eval_results_path = tmp_path / "perturbed_llm_eval_results.json"
    eval_results.save_as_json(eval_results_path)
    eval_results.to_datatable().to_csv(str(tmp_path / "perturbed_llm_eval_results.csv"))
    # load eval results CSV
    loaded_eval_results = datasets.LlmEvalResults.load_from_json(eval_results_path)
    loaded_eval_results.save_as_json(
        tmp_path / "perturbed_llm_eval_results_LOADED.json"
    )

    #
    # THEN
    #
    print(f"Test lab: {test_lab}")
    # assert lab's resolved dataset keys vs. test suite keys
    assert len(test_lab.dataset.inputs) == len(test_suite.test_cases)
    for e, i in enumerate(test_lab.dataset.inputs):
        assert i.key == test_suite.test_cases[e].key
    # assert lab's raw vs. resolved dataset keys + categories + relationships
    assert len(test_lab.raw_dataset.inputs) == len(test_lab.dataset.inputs)
    for e, _ in enumerate(test_lab.raw_dataset.inputs):
        assert test_lab.raw_dataset.inputs[e].key == test_lab.dataset.inputs[e].key
        assert len(test_lab.raw_dataset.inputs[e].categories) == len(
            test_lab.dataset.inputs[e].categories
        )
        assert len(test_lab.raw_dataset.inputs[e].relationships) == len(
            test_lab.dataset.inputs[e].relationships
        )
    # assert lab's relationships targets validity
    for e, _ in enumerate(test_lab.dataset.inputs):
        if len(test_lab.dataset.inputs[e].relationships):
            for r in test_lab.dataset.inputs[e].relationships:
                found = False
                target_key = r.target
                for ii in test_lab.dataset.inputs:
                    if ii.key == target_key:
                        found = True
                        break
                assert found, f"Invalid rel target @ testlab dataset: {target_key}"
    # assert test result keys + relationships targets validity
    assert len(eval_results.results) == len(test_suite.test_cases)
    for e, r in enumerate(eval_results.results):
        assert r.dataset_row.key == test_suite.test_cases[e].key
    for e, _ in enumerate(eval_results.results):
        if len(eval_results.results[e].dataset_row.relationships):
            for rel in eval_results.results[e].dataset_row.relationships:
                found = False
                target_key = rel.target
                for rr in eval_results.results:
                    if rr.dataset_row.key == target_key:
                        found = True
                        break
                assert found, f"Invalid rel target @ eval result: {target_key}"


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_trim_lab(tmp_path):
    #
    # GIVEN
    #
    h2ogpte_connection = test_utils.health.get_h2ogpte()
    test_lab_path = "data/generative/eval_llm/pii_test_lab.json"

    test_lab = testing.RagTestLab.load_from_json(
        llm_host_connection=h2ogpte_connection,
        file_path=test_lab_path,
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )

    #
    # WHEN
    #
    split_test_lab = test_lab.trim(max_llm_models_count=3)

    #
    # THEN
    #
    print(f"Split test lab: {split_test_lab}")
    split_test_lab.save_as_json(tmp_path / "split_test_lab.json")
    assert split_test_lab.dataset.inputs < test_lab.dataset.inputs


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_llm_host_prompt_cache():
    #
    # GIVEN
    #
    connection = test_utils.health.get_h2ogpte()

    test_lab_path = "data/generative/h2ogpte_benchmark_test_lab_top.json"
    test_lab = testing.RagTestLab.load_from_json(
        llm_host_connection=connection,
        file_path=test_utils.find_locally(test_lab_path),
    )
    print(f"Loaded testlab: {test_lab_path}")

    # one entry
    dt_entry = test_lab.dataset.inputs[0]
    dt_entry_e_model = test_lab.get_evaluated_model_for_key(dt_entry.model_key)
    dt_entry_llm_model_name = dt_entry_e_model.llm_model_name

    #
    # WHEN
    #
    cache = testing.InMemoryLlmHostPromptCache()

    cache_key = cache.get_key(
        explainable_model_type=dt_entry_e_model.model_type,
        prompt=dt_entry.i,
        llm_model_name=dt_entry_llm_model_name,
        corpus=dt_entry.corpus,
    )

    cache.put(key=cache_key, value=dt_entry.to_dict())

    #
    # THEN
    #
    print(json.dumps(cache.to_dict(), indent=4))
    assert cache_key in cache.to_dict()[testing.InMemoryLlmHostPromptCache.KEY_DATA]
    print(json.dumps(cache.get(cache_key), indent=4))
    assert cache.get(cache_key)
    assert 2 == cache.hits

    assert None is cache.get("non-existing-key")
    assert 1 == cache.misses
    assert 2 == cache.hits

    llm_model_names = cache.get_llm_model_names(
        next(iter(test_lab.evaluated_models.values())).model_type
    )
    assert 1 == len(llm_model_names)
    assert dt_entry_llm_model_name in llm_model_names


@pytest.mark.parametrize(
    "src_test_suite_path,src_test_lab_path,model_type",
    [
        (
            "data/generative/eval_llm/bank_teller_test_suite.json",
            "data/generative/eval_llm/bank_teller_test_lab.json",
            models.ExplainableModelType.h2ogpte_llm,
        )
    ],
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_testlab_with_prompt_cache(
    tmp_path, src_test_suite_path, src_test_lab_path, model_type
):
    #
    # GIVEN
    #
    host_connection = test_utils.health.get_h2ogpte()

    # TEST DATA: test lab must be built from the suite
    src_test_suite = testing.RagTestSuiteConfig.load_from_json(
        file_path=test_utils.find_locally(src_test_suite_path)
    )
    src_test_lab = testing.RagTestLab.load_from_json(
        llm_host_connection=host_connection,
        file_path=test_utils.find_locally(src_test_lab_path),
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )

    # CACHE
    cache = testing.InMemoryLlmHostPromptCache()
    # BUILD cache from the test lab
    cache.add_test_lab(src_test_lab)
    # get models for specific explainable model type
    cached_llm_model_names = cache.get_llm_model_names(
        explainable_model_type=model_type,
    )
    assert len(cached_llm_model_names) == len(src_test_lab.llm_model_names)
    for llm_model_name in src_test_lab.llm_model_names:
        assert llm_model_name in cached_llm_model_names

    # CREATE NEW test lab using CACHE
    if model_type == models.ExplainableModelType.h2ogpte_llm:
        suite_test_lab = testing.RagTestLab.from_llm_test_suite(
            llm_host_connection=host_connection,
            llm_test_suite=src_test_suite,
            llm_model_type=model_type,
            # IMPORTANT:
            # use LLM models from the cache to avoid cache misses (LLM unknown to cache)
            llm_model_names=cached_llm_model_names,
            work_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
            llm_host_prompt_cache=cache,
        )
    elif model_type == models.ExplainableModelType.h2ogpte:
        suite_test_lab = testing.RagTestLab.from_rag_test_suite(
            rag_connection=host_connection,
            rag_test_suite=src_test_suite,
            rag_model_type=model_type,
            # IMPORTANT:
            # use LLM models from the cache to avoid cache misses (LLM unknown to cache)
            llm_model_names=cached_llm_model_names,
            docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
            llm_host_prompt_cache=cache,
        )
    else:
        raise ValueError(f"Unsupported explainable model type: {model_type}")

    # progress
    progress_callback = progress.LoggingProgressCallbackContext(
        logger=suite_test_lab.logger,
        prefix="[TEST custom progress callback]",
        name="Test E2E progress",
    )
    lab_build_progress = progress_callback.get_sub_callback_for_progress(
        min_progress=0.0, max_progress=0.5, verbose_children=False
    )
    lab_completion_progress = progress_callback.get_sub_callback_for_progress(
        min_progress=0.51, max_progress=1.0, verbose_children=False
    )

    # build test lab: cache RAG documents & create collections
    suite_test_lab.build(progress_callback=lab_build_progress)

    #
    # WHEN
    #

    # complete test lab @ CACHE
    suite_test_lab.complete_dataset(progress_callback=lab_completion_progress)

    progress_callback.set_progress(1.0, "Test DONE")

    #
    # THEN
    #

    print(f"Test lab cache misses: {suite_test_lab.llm_host_prompt_cache.misses}")
    suite_test_lab.save_as_json(tmp_path / "test_lab_on_cache.json")
    assert 0 == suite_test_lab.llm_host_prompt_cache.misses

    # assert labs identical
    assert len(suite_test_lab.dataset.inputs) == len(src_test_lab.dataset.inputs)
    assert len(suite_test_lab.llm_model_names) == len(src_test_lab.llm_model_names)
    for llm_model_name in src_test_lab.llm_model_names:
        assert llm_model_name in suite_test_lab.llm_model_names

    assert progress_callback.progress == 1.0


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_testlab_with_global_prompt_cache(tmp_path):
    #
    # GIVEN
    #
    host_connection = test_utils.health.get_h2ogpte()

    # copy cache to the test lab cache directory
    src_test_lab_path = "data/generative/eval_llm/bank_teller_test_lab.json"
    lab_paths = [
        src_test_lab_path,
        "data/generative/sr1107_test_lab_15m.json",
    ]
    for p in lab_paths:
        shutil.copy(src=pathlib.Path(p), dst=tmp_path)

    # reinitialize the cache w/o environment variables
    testing.prompt_cache.reinitialize(
        enable_cache=True,
        src_path=tmp_path.absolute().as_posix(),
        src_host_connection=host_connection,
        max_items=1000,
    )
    print(testing.prompt_cache.prompts)

    src_test_lab = testing.RagTestLab.load_from_json(
        llm_host_connection=host_connection,
        file_path=test_utils.find_locally(src_test_lab_path),
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )

    # progress
    progress_callback = progress.LoggingProgressCallbackContext(
        logger=src_test_lab.logger,
        prefix="[TEST custom progress callback]",
        name="Test E2E progress",
    )
    lab_build_progress = progress_callback.get_sub_callback_for_progress(
        min_progress=0.0, max_progress=0.5, verbose_children=False
    )
    lab_completion_progress = progress_callback.get_sub_callback_for_progress(
        min_progress=0.51, max_progress=1.0, verbose_children=False
    )

    # build test lab: cache RAG documents & create collections
    src_test_lab.build(progress_callback=lab_build_progress)

    #
    # WHEN
    #

    # complete test lab @ CACHE
    src_test_lab.complete_dataset(progress_callback=lab_completion_progress)

    progress_callback.set_progress(1.0, "Test DONE")

    #
    # THEN
    #

    print(f"Test lab cache misses: {testing.prompt_cache.prompts.misses}")
    testing.prompt_cache.prompts.save_to_json(tmp_path / "global_cache.json")
    assert 0 == testing.prompt_cache.prompts.misses

    assert progress_callback.progress == 1.0


@pytest.mark.skip(reason="This test merges test labs")
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_testlab_merge(tmp_path):
    #
    # GIVEN
    #

    # lab_paths = [
    #     "/home/user/h/mli/eval-studio-gallery/reports/openai-rag"
    #     "/h2ogpte-benchmark-full/h2ogpte-20231215"
    #     "/test_lab.json",
    #     "/home/user/h/mli/eval-studio-gallery/reports/openai-rag"
    #     "/h2ogpte-benchmark-full/openai-20231213"
    #     "/h2ogpte_benchmark_test_lab_openai_20231213.json"
    # ]

    lab_paths = []
    for i in range(5):
        lab_paths.append(
            pathlib.Path(
                f"/home/user/h/mli/git/h2o-sonar/data/generative/rag_docs/"
                f"execution_1715074507.6472626/shard_{i}/wip_lab.json"
            )
        )

    for p in lab_paths:
        assert pathlib.Path(p).exists()

    labs = []
    for lab_path in lab_paths:
        labs.append(
            testing.RagTestLab.load_from_json(
                llm_host_connection=test_utils.health.get_h2ogpte(),
                file_path=lab_path,
            )
        )

    #
    # WHEN
    #
    merged_lab = labs[0]
    for lab in labs[1:]:
        merged_lab.merge(lab, other_llm_prefix="")

    #
    # THEN
    #
    merged_lab.save_as_json(tmp_path / "merged_lab.json")
    assert merged_lab


@pytest.mark.skip(reason="This test imports LLM dataset in legacy format")
@pytest.mark.parametrize(
    "old_llm_dataset_path",
    [
        "evalgpt_v1_prompts.json",
        "evalgpt_v1_prompts_w_short_answers.json",
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_llm_dataset_legacy_import(tmp_path, old_llm_dataset_path):
    #
    # GIVEN
    #
    old_llm_dataset_path = pathlib.Path("data/generative") / old_llm_dataset_path

    #
    # WHEN
    #

    llm_dataset = datasets.LlmDataset.load_from_json(
        old_llm_dataset_path, datatable_format=True
    )

    #
    # THEN
    #
    llm_dataset.save_as_json(tmp_path / "new_llm_dataset.json")


@pytest.mark.skip(reason="This test imports LLM dataset in legacy format")
@pytest.mark.parametrize(
    "old_testlab_path",
    [
        "testlab_h2ogpte_benchmark_min_actuals.json",
        "testlab_h2ogpte_benchmark_3x5_actuals.json",
        "testlab_h2ogpte_benchmark_ALLx10_actuals.json",
        "testlab_h2ogpte_benchmark_ALLx30_actuals.json",
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_testlab_legacy_import(tmp_path, old_testlab_path):
    #
    # GIVEN
    #
    old_testlab_path = pathlib.Path("data/generative") / old_testlab_path

    #
    # WHEN
    #

    testlab = testing.RagTestLab.load_from_json(
        llm_host_connection=test_utils.health.get_h2ogpte(),
        file_path=old_testlab_path,
        datatable_format=True,
    )

    #
    # THEN
    #
    testlab.save_as_json(tmp_path / "new_testlab.json")


@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_config_api(tmp_path):
    #
    # GIVEN
    #

    repro_test_config = testing.RagTestConfig(
        documents=["data/generative/rag_docs/Coca-Cola-FEMSA-Results-1Q23-vf-2.pdf"]
    )

    rag_test = testing.RagTestSuiteConfig(
        test_cases=[
            testing.RagTestCaseConfig(
                prompt="What was the revenue of Brazil?",
                constraints=["15,969", "million"],
                config=repro_test_config,
            ),
            testing.RagTestCaseConfig(
                prompt="What was the revenue of Mexico?",
                constraints=["27,229", "million"],
                config=repro_test_config,
            ),
            testing.RagTestCaseConfig(
                prompt="How did gross profit change YoY for South America?",
                constraints=["11%"],
                config=repro_test_config,
            ),
        ]
    )

    #
    # WHEN
    #

    file_path_1 = rag_test.save_as_json(tmp_path / "test_config_1.json")
    loaded_config = testing.RagTestSuiteConfig.load_from_json(file_path_1)
    file_path_2 = loaded_config.save_as_json(tmp_path / "test_config_2.json")

    #
    # THEN
    #
    print(file_path_1)
    assert file_path_1
    print(file_path_2)
    assert file_path_2


_PORT_SYNC_DOCS = 8888


@pytest.mark.skip(reason="This test is for debugging purposes - would BLOCK test run")
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_given_sync_docs():
    import http.server as simple_http_server
    import socketserver as socket_server

    print(f"Starting HTTP server on port {_PORT_SYNC_DOCS}")

    class GetHandler(simple_http_server.SimpleHTTPRequestHandler):
        def do_GET(self):
            print(f"REQUEST HEADERS:\n{self.headers}")
            simple_http_server.SimpleHTTPRequestHandler.do_GET(self)

    get_handler = GetHandler
    httpd = socket_server.TCPServer(("", _PORT_SYNC_DOCS), get_handler)

    httpd.serve_forever()


@pytest.mark.parametrize(
    "rag_document",
    [
        "https://eval-studio-artifacts.s3.amazonaws.com"
        "/h2o-eval-studio-suite-library/corpus-h2ogpte-benchmark"
        "/Coca-Cola-FEMSA-Results-1Q23-vf-2.pdf",
        "data/generative/rag_docs/Coca-Cola-FEMSA-Results-1Q23-vf-2.pdf",
    ],
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_testlab_sync_docs(tmp_path, rag_document):
    #
    # GIVEN
    #

    # h2oGPTe server
    h2ogpte_connection = test_utils.health.get_h2ogpte()

    repro_test_config = testing.RagTestConfig(documents=[rag_document])

    test_lab = testing.RagTestLab.from_rag_test_suite(
        rag_connection=h2ogpte_connection,
        rag_test_suite=testing.RagTestSuiteConfig(
            test_cases=[
                testing.RagTestCaseConfig(
                    prompt="What was the revenue of Brazil?",
                    constraints=["15,969", "million"],
                    config=repro_test_config,
                ),
                testing.RagTestCaseConfig(
                    prompt="What was the revenue of Mexico?",
                    constraints=["27,229", "million"],
                    config=repro_test_config,
                ),
                testing.RagTestCaseConfig(
                    prompt="How did gross profit change YoY for South America?",
                    constraints=["11%"],
                    config=repro_test_config,
                ),
            ]
        ),
        rag_model_type=models.ExplainableModelType.h2ogpte,
        llm_model_names=["h2oai/h2ogpt-4096-llama2-70b-chat"],
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )
    test_lab.save_as_json(tmp_path / "testlab.json")

    #
    # WHEN
    #

    if "localhost" in rag_document:
        docs_cache_dir = test_lab.sync_documents(
            doc_sync_meta={
                rag_document: {
                    "headers": {
                        "foo-header": "FOO-VALUE",
                    }
                }
            }
        )
    else:
        docs_cache_dir = test_lab.sync_documents()

    #
    # WHEN
    #
    print(docs_cache_dir)
    assert docs_cache_dir


@pytest.mark.skip(reason="This is just an ad hoc validation test")
@pytest.mark.parametrize(
    "test_lab_path",
    ["data/generative/bugs/20240731-y-d-test-lab.json"],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_lab_validate(tmp_path, test_lab_path: str):
    """Validate given test lab."""
    #
    # GIVEN
    #
    connection = test_utils.health.get_h2ogpte()

    #
    # WHEN
    #
    test_lab = testing.RagTestLab.load_from_json(
        llm_host_connection=connection, file_path=test_lab_path
    )
    for i in test_lab.dataset.inputs:
        if not isinstance(i.i, str):
            raise ValueError(
                f"Test lab dataset row '{i.key}' has no prompt "
                f"'{i.i}' or wrong type {type(i.i)}: "
                f"{i.to_dict()}"
            )
        if not isinstance(i.actual_output, str):
            raise ValueError(
                f"Test lab dataset row '{i.key}' has no/wrong actual answer "
                f"'{i.actual_output}' or wrong type {type(i.actual_output)}: "
                f"{i.to_dict()}"
            )

    #
    # THEN
    #
    assert test_lab


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
