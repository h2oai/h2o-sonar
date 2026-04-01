# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import pathlib

import pytest

from h2o_sonar import evaluate
from h2o_sonar import loggers
from h2o_sonar.evaluators import rag_tokens_presence_evaluator
from h2o_sonar.lib.api import models
from h2o_sonar.lib.integrations import genai
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.skipif(
    not test_utils.health.is_h2ollmops(),
    reason="H2O LLMOps hosted LLM models are deployed only temporarily",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_model_list():
    #
    # GIVEN
    #

    client = genai.H2oLlmOpsClient(
        connection=given_generative.H2O_LLMOPS,
        logger=loggers.SonarPrintLogger(),
    )
    llm_model_name = client.list_llm_model_names()[0]

    #
    # WHEN
    #

    # get ANSWERS
    results = client.ask_model(
        prompts=["What are the transformed features of the model?"],
        llm_model_name=llm_model_name,
    )
    (prompt, answer, duration, chunks, cost, _, _) = results[0]

    #
    # THEN
    #
    assert prompt
    assert answer
    assert duration
    assert not chunks
    assert not cost


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.skipif(
    not test_utils.health.is_h2ollmops(),
    reason="H2O LLMOps hosted LLM models are deployed only temporarily",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_ask_model(tmp_path):
    #
    # GIVEN
    #
    client = genai.H2oLlmOpsClient(
        connection=given_generative.H2O_LLMOPS,
        logger=loggers.SonarPrintLogger(),
    )
    llm_model_name = client.list_llm_model_names()[0]

    #
    # WHEN
    #

    # get ANSWERS
    results = client.ask_model(
        prompts=["What are the transformed features of the model?"],
        llm_model_name=llm_model_name,
    )
    (prompt, answer, duration, chunks, cost, _, _) = results[0]

    #
    # THEN
    #
    assert prompt
    assert answer
    assert duration
    assert not chunks
    assert not cost


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.skipif(
    not test_utils.health.is_h2ollmops(),
    reason="H2O LLMOps hosted LLM models are deployed only temporarily",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_build_lab(tmp_path):
    #
    # GIVEN
    #

    connection = given_generative.H2O_LLMOPS

    llm_model_names = genai.H2oLlmOpsClient(connection).list_llm_model_names()

    test_suite = testing.RagTestSuiteConfig.load_from_json(
        test_utils.find_locally("data/generative/eval_llm/bank_teller_test_suite.json")
    )

    # OPTIONAL DESCOPE: faster / smaller test (debugging)
    # llm_model_names = llm_model_names[:1]
    # test_suite.test_cases = test_suite.test_cases[:2]

    #
    # WHEN
    #

    # test lab (RAG product)
    test_lab = testing.RagTestLab.from_llm_test_suite(
        llm_host_connection=connection,
        llm_test_suite=test_suite,
        llm_model_names=llm_model_names,
        llm_model_type=models.ExplainableModelType.h2ollmops,
        work_dir=tmp_path,
    )
    test_lab.save_as_json(tmp_path / "001_before_testlab.json")
    try:
        # docs sync: does nothing
        test_lab.build()
        test_lab.save_as_json(tmp_path / "002_after_build_testlab.json")

        # test lab: complete dataset w/ ACTUAL data (answers, duration, context)
        test_lab.complete_dataset(
            parallelize=True,
            save_as_you_go=tmp_path / "003_wip_chat_testlab.json",
        )
        # backup fully resolved dataset
        lab_path = test_lab.save_as_json(tmp_path / "DONE_test_lab.json")

        #
        # THEN
        #
        evaluation = evaluate.run_evaluation(
            # dataset w/ prompts, constraints and model keys
            dataset=test_lab.dataset,
            # models to be evaluated / compared to get leaderboard
            models=test_lab.evaluated_models.values(),
            # evaluators
            evaluators=[
                rag_tokens_presence_evaluator.RagStrStrEvaluator().evaluator_id()
            ],
            # where to save the report
            results_location=tmp_path,
        )
        print(
            f"\nExplanations:\n"
            f"  HTML report: file://{evaluation.result.get_html_report_location()}\n"
        )

        assert pathlib.Path(lab_path).exists()
    finally:
        # purge test lab
        test_lab.purge()


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
