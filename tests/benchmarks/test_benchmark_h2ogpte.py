# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json

import pytest

from h2o_sonar.lib.api import models
from h2o_sonar.lib.integrations import genai
from h2o_sonar.utils import progress
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative


@pytest.mark.skip(reason="Tool to prepare benchmark test suites")
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_split_test_suite(tmp_path) -> None:
    """Split the test suite into multiple test suite having at most `max_tests`"""
    #
    # GIVEN
    #
    test_suite_path = (
        "data/generative/h2ogpte-benchmark/"
        "h2ogpte_benchmark_test_suite_2024-10-01_15h37m17s.json"  #
    )
    test_suite = testing.RagTestSuiteConfig.load_from_json(
        file_path=test_suite_path,
    )

    max_tests = 15

    #
    # WHEN
    #
    test_suite_splits = test_suite.split(max_tests=max_tests)

    #
    # THEN
    #
    for i, test_suite_split in enumerate(test_suite_splits):
        test_suite_split.save_as_json(tmp_path / f"test_suite_split_{i}.json")


@pytest.mark.skip(reason="Tool to build and complete benchmark test labs")
@pytest.mark.parametrize(
    "llm_model_names,exclude_llm_models,test_suite_path",
    [
        # # BENCHMARK: full test suite in one go, all models - creates MANY collections
        # (
        #     [],
        #     [],
        #     "data/generative/h2ogpte-benchmark/"
        #     "h2ogpte_benchmark_test_suite_2024-10-01_15h37m17s.json",
        # ),
        # # try which models HANG/fail on the first call: one test case, all models
        # (
        #     [],
        #     ["gpt-4o", "gpt-4o-mini"],
        #     "data/generative/h2ogpte-benchmark/h2ogpte_benchmark_test_suite_1t.json",
        # ),
        # HANG handling test: 1 test case @ 4o family: 'gpt-4o', 'gpt-4o-mini'
        # (
        #     ["gpt-4o-mini"],
        #     [],
        #     "data/generative/h2ogpte-benchmark/h2ogpte_benchmark_test_suite_1t.json",
        # ),
        # INTERNAL ERR TEST: raise/fail and find out whether INTERNAL ERR in test lab
        (
            ["h2oai/h2o-danube3-4b-chat"],
            [],
            "data/generative/h2ogpte-benchmark/h2ogpte_benchmark_test_suite_1t4tc.json",
        ),
        # BENCHMARK: splits to keep the number of collections @ h2oGPTe under test low
        # (
        #     [],
        #     ["gpt-4o", "gpt-4o-mini"],
        #     "data/generative/h2ogpte-benchmark/"
        #     "h2ogpte_benchmark_test_suite_2024-10-01_15h37m17s/"
        #     "test_suite_split_3.json",
        # ),
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_build_test_lab_for_h2ogpte_benchmark_multi_shard(
    tmp_path, llm_model_names, exclude_llm_models, test_suite_path
):
    """Run h2oGPTe benchmark in multiple steps:

    - to keep the number of collections @ h2oGPTe under test low

    Method:

    - load the test suite
    - split the test suite into multiple test suite having at most `max_test_cases`
    - build and complete the test lab for each test suite
      - PURGE the h2oGPTe server before each build - all ephemeral collection

    """
    #
    # GIVEN
    #

    # h2oGPTe under evaluation
    h2ogpte_connection = test_utils.health.get_h2ogpte()

    # models to evaluate
    if not llm_model_names:
        available_llm_model_names = genai.H2oGpteRagClient(
            h2ogpte_connection
        ).list_llm_model_names()
        llm_model_names = available_llm_model_names
        print("LLM models:")
        for m in llm_model_names:
            print(f"  {m}")

    if exclude_llm_models:
        llm_model_names = [m for m in llm_model_names if m not in exclude_llm_models]

    test_suite = testing.RagTestSuiteConfig.load_from_json(file_path=test_suite_path)

    #
    # WHEN
    #

    # build test l

    print(
        f"Building test LAB for LLMs:\n"
        f"  {', '.join(llm_model_names)}\n"
        f"From test suite: {test_suite_path}\n"
        f"  with {len(test_suite.test_cases)} test cases\n"
        f"Connection:\n"
        f"  {json.dumps(h2ogpte_connection.to_dict(encryption_key='%&%$'), indent=2)}\n"
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
        parallelize=testing.TestLab.PARALLEL_RUN,
        progress_callback=lab_completion_progress,
    )

    #
    # THEN
    #
    test_lab.save_as_json(tmp_path / "test_lab.json")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
