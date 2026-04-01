# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
#
# This test module aims to test h2oGPTe:
#
# - it mimics h2oGPTe benchmark (run by h2oGPTe team in their CI)
# - it runs h2oGPTe load test: Text matching evaluator + all models + big(ger)
#   dataset to get 1k requires and analyze result for internal server errors
# - it provides a part of EvalStudio GitHub Action test coverage
#
import pathlib

import pytest

from h2o_sonar import evaluate
from h2o_sonar.evaluators import rag_tokens_presence_evaluator
from h2o_sonar.lib.api import models
from h2o_sonar.lib.integrations import genai
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.skip(reason="Evaluator requires h2oGPTe server")
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_h2ogpte_benchmark(tmp_path):
    """Run Text matching evaluator to mimic h2oGPTe benchmark and get benchmark
    leaderboard as Markdown and H2O Sonar HTML report:

    - load test configuration
    - create test lab configuration
       - build RAG test lab:
         - update local document cache (from web/S3 and local filesystem)
         - create h2oGPTe collections and upload there documents
       - resolve actual dataset columns (chat w/ h2oGPTe to get actual values)
    - run H2O Sonar evaluation
    - upload results (Markdown leaderboard and H2O Sonar report) to S3

    Nothing, except the test configuration, is loaded.

    """

    #
    # GIVEN
    #

    # h2oGPTe server
    h2ogpte_connection = test_utils.health.get_h2ogpte()
    llm_model_names = genai.H2oGpteRagClient(h2ogpte_connection).list_llm_model_names()

    # test configuration (RAG product test target): docs + prompts + constraints
    rag_test_suite = testing.RagTestSuiteConfig.load_from_json(
        "data/generative/h2ogpte_benchmark_test_suite_min.json"
        # "data/generative/h2ogpte_benchmark_test_suite_demo.json"
        # "data/generative/h2ogpte_benchmark_test_suite.json"
        # GHA action's test config @ action's repo
        # "data/generative/default_h2ogpte_benchmark_test_suite.json"
    )

    # OPTIONAL DESCOPE: faster / smaller test (debugging)
    llm_model_names = llm_model_names[:3]
    rag_test_suite.test_cases = rag_test_suite.test_cases[:3]

    # test lab (RAG product)
    test_lab = testing.RagTestLab.from_rag_test_suite(
        rag_connection=h2ogpte_connection,
        rag_test_suite=rag_test_suite,
        rag_model_type=models.ExplainableModelType.h2ogpte,
        llm_model_names=llm_model_names,
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )
    # test lab: DEPLOY the h2oGPTe server (docs sync: S3 > filesystem cache > h2oGPT2)
    test_lab.build()

    # test lab: complete dataset w/ ACTUAL values from the h2oGPTe server (answers, ...)
    test_lab.complete_dataset(save_as_you_go=tmp_path / "wip_testlab.json")
    # backup fully resolved dataset
    test_lab.save_as_json(tmp_path / "testlab_h2ogpte_benchmark_w_actuals.json")

    #
    # WHEN
    #

    evaluation = evaluate.run_evaluation(
        # dataset w/ prompts, constraints and model keys
        dataset=test_lab.dataset,
        # models to be evaluated / compared to get leaderboard
        models=test_lab.evaluated_models.values(),
        # evaluators
        evaluators=[rag_tokens_presence_evaluator.RagStrStrEvaluator().evaluator_id()],
        # where to save the report
        results_location=tmp_path,
    )

    #
    # THEN
    #
    print(f"{evaluation}")
    assert evaluation
    assert not evaluation.get_failed_evaluator_ids()
    result = evaluation.get_evaluator_result(
        rag_tokens_presence_evaluator.RagStrStrEvaluator().evaluator_id()
    )
    print(result)
    assert result
    print(
        f"Explanations:\n"
        f"  HTML: file://{evaluation.result.get_html_report_location()}\n"
    )


# CLI: pytest -sv --disable-warnings tests/lib/test_evaluators_h2ogpte.py::test_load
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.skip(reason="h2oGPTe server build lab load test")
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_load(tmp_path):
    """h2oGPTe load test:

    - builds TestLab (RAG) only (no evaluators are run)
    - aims to determine:
      - internal server errors per LLM model (HTML report insight)
      - LLM model performance ~ TPS (HTML report leaderboard)
    - no caching
    - no retries

    """
    print("STARTING h2oGPTe RAG load test ...")

    # make sure result is not purged by pytest rotation
    tmp_path = pathlib.Path.home() / "tmp" / "loadtest"

    #
    # GIVEN
    #

    # h2oGPTe server
    h2ogpte_connection = test_utils.health.get_h2ogpte()

    # run ALL LLM models to get full coverage
    llm_model_names = genai.H2oGpteRagClient(h2ogpte_connection).list_llm_model_names()

    # test configuration (RAG product test target): docs + prompts + constraints
    rag_test_suite = testing.RagTestSuiteConfig.load_from_json(
        # large SR-1107 test suite has ~170 test cases
        test_utils.find_locally(
            "data/generative/conferences/atlanta-2024/sr1107_test_suite_large.json"
        )
    )

    # DO NOT DESCOPE (just for test debugging)
    # llm_model_names = llm_model_names[:1]
    rag_test_suite.test_cases = rag_test_suite.test_cases[:10]
    print("  LLM models: ", len(llm_model_names))
    print("  test_cases: ", len(rag_test_suite.test_cases))

    # test lab (RAG product)
    test_lab = testing.RagTestLab.from_rag_test_suite(
        rag_connection=h2ogpte_connection,
        rag_test_suite=rag_test_suite,
        rag_model_type=models.ExplainableModelType.h2ogpte,
        llm_model_names=llm_model_names,
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )
    print(f"Building test lab for models: {test_lab.llm_model_names}")

    # test lab: DEPLOY the h2oGPTe server (docs sync: S3 > filesystem cache > h2oGPT2)
    test_lab.build()

    #
    # WHEN
    #

    # test lab: complete dataset w/ ACTUAL values from the h2oGPTe server (answers, ...)
    test_lab.complete_dataset(
        retry_on_error=0,
        parallelize=testing.TestLab.PARALLEL_RUN,
        save_as_you_go=tmp_path / "wip_testlab.json",
    )

    #
    # THEN
    #
    print(test_lab)
    # backup fully resolved dataset
    test_lab.save_as_json(tmp_path / "test_lab_load_test.json")
    # internal server errors
    test_lab.insight_internal_llm_errors(tmp_path)


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.skip(reason="Ad hoc test to get loaded lab insights")
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_lab_insights(tmp_path):
    #
    # GIVEN
    #
    test_lab = testing.RagTestLab.load_from_json(
        llm_host_connection=test_utils.health.get_h2ogpte(),
        file_path=pathlib.Path.home() / "tmp" / "loadtest" / "test_lab_load_test.json",
    )

    #
    # WHEN
    #
    test_lab.insight_internal_llm_errors(tmp_path)

    #
    # THEN
    #
    print(test_lab)


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.skip(reason="Ad hoc h2oGPTe bug reproduction")
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_h2ogpte_bug_collection(tmp_path):
    #
    # GIVEN
    #
    h2ogpte_connection = test_utils.health.get_h2ogpte()
    client = genai.H2oGpteRagClient(h2ogpte_connection)

    #
    # WHEN
    #
    c = client.create_collection(
        doc_paths=[
            pathlib.Path().home()
            / "h"
            / "mli"
            / "git"
            / "h2o-sonar"
            / given_generative.DIR_TEST_RAG_DOCS_CACHE
            / "bradesco-2022-integrated-report.pdf"
        ],
        collection_name=(
            "RAG collection (docs: ['Coca-Cola-FEMSA-Results-1Q23-vf-2.pdf'])"
        ),
    )

    #
    # THEN
    #
    print("Collection:")
    print(f"  {c}")


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.skip(reason="Ad hoc test to purge h2oGPTe server data")
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_purge_all_data_h2ogpte(tmp_path):
    #
    # GIVEN
    #
    h2ogpte_connection = test_utils.health.get_h2ogpte()
    client = genai.H2oGpteRagClient(h2ogpte_connection)

    #
    # WHEN
    #
    print("Purging collections...")
    to_purge = client.list_collections()
    while to_purge:
        print(f"  Purging {len(to_purge)} collections...")
        if to_purge:
            client.purge_collections([i.id for i in to_purge])
            to_purge -= 1
        to_purge = client.list_collections()

    print("Purging uploaded documents...")
    client.purge_uploaded_docs()

    #
    # THEN
    #
    print("Purged all data from h2oGPTe server:")
    print(f" collections: {len(client.list_collections())}")
    print(f" documents: {len(client.list_collections())}")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
