# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.

import logging

import pytest

from h2o_sonar import evaluate
from h2o_sonar import loggers
from h2o_sonar.evaluators import looping_detection_evaluator
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import models
from h2o_sonar.lib.integrations import genai
from h2o_sonar.utils import progress
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative
from tests.lib import then_eval


# constants
LoopingDetectionEvaluator = looping_detection_evaluator.LoopingDetectionEvaluator


@pytest.mark.skipif(
    not test_utils.health.is_h2ogpte(),
    reason="h2oGPTE service is not reachable",
)
@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({"nltk"}),
    reason="Package 'nltk' is not installed",
)
@pytest.mark.parametrize(
    "test_lab_path,evaluator_class",
    [
        (
            #
            # h2oGPTe server
            #
            #
            "data/generative/looping_test_lab.json",
            #
            # EVALUATORS: AS fastest (3s @ cosine)
            #
            LoopingDetectionEvaluator,
        ),
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_sanity(
    tmpdir,
    h2ogpte_connection_fixture,
    test_lab_path,
    evaluator_class,
):
    #
    # GIVEN
    #
    h2ogpte_connection = h2ogpte_connection_fixture

    test_lab = testing.RagTestLab.load_from_json(
        llm_host_connection=h2ogpte_connection,
        file_path=test_lab_path,
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )
    the_dataset = test_lab.dataset
    the_models = list(test_lab.evaluated_models.values())

    # progress: 1 stage - evaluate
    progress_callback_name = "[TEST E2E progress callback]"
    progress_callback = progress.LoggingProgressCallbackContext(
        logger=test_lab.logger,
        prefix=progress_callback_name,
        name=progress_callback_name,
    )

    #
    # WHEN
    #
    assert evaluate.describe_evaluator(evaluator_class.evaluator_id())

    evaluation = evaluate.run_evaluation(
        dataset=the_dataset,
        models=the_models,
        evaluators=[
            commons.EvaluatorToRun(
                evaluator_id=evaluator_class.evaluator_id(),
                params={},
            )
        ],
        results_location=tmpdir,
        log_level=loggers.DEBUG,
        progress_callback=progress_callback,
    )

    #
    # THEN
    #
    print(f"Evaluation:\n{evaluation}")
    print(f"HTML:\nfile://{evaluation.result.get_html_report_location()}")
    assert evaluation
    assert not evaluation.is_explainer_failed()

    # assert result
    result = evaluation.get_evaluator_result(evaluator_class.evaluator_id())
    print(result)
    assert result

    # assert leaderboard JSon representation data and meta
    then_eval.then_leaderboard_json(evaluation, evaluator_class.evaluator_id())


@pytest.mark.skip(reason="Util used to generate test lab")
@pytest.mark.parametrize(
    "test_suite_path",
    [
        "data/generative/looping_test_suite.json",
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_generate_lab(
    tmp_path,
    test_suite_path: str,
):
    #
    # GIVEN
    #
    # h2oGPTe server (must be accessible from the CI)
    target_host_connection = test_utils.health.get_h2ogpte()
    # LLM models to evaluate
    llm_model_type = models.ExplainableModelType.h2ogpte_llm
    llm_model_names = [
        m
        for m in genai.H2oGpteRagClient(target_host_connection).list_llm_model_names()
        if "h2oai" in m or "mistral" in m or "meta" in m
    ]

    # test suite
    test_suite = testing.RagTestSuiteConfig.load_from_json(
        # small(er) test suite w/ many features
        test_utils.find_locally(test_suite_path)
    )

    print(f"Tested LLM models: {llm_model_names}")

    # test lab
    test_lab = testing.RagTestLab.from_llm_test_suite(
        llm_host_connection=target_host_connection,
        llm_test_suite=test_suite,
        llm_model_names=llm_model_names,
        llm_model_type=llm_model_type,
        work_dir=tmp_path,
    )
    # progress: 3 stages - build, complete, evaluate
    progress_callback = progress.LoggingProgressCallbackContext(
        logger=test_lab.logger,
        prefix="[TEST E2E progress callback]",
        name="Test E2E progress callback",
    )
    lab_build_progress = progress_callback.get_sub_callback_for_progress(
        min_progress=0.0, max_progress=0.33, verbose_children=False
    )
    lab_completion_progress = progress_callback.get_sub_callback_for_progress(
        min_progress=0.34, max_progress=0.66, verbose_children=False
    )
    eval_progress = progress_callback.get_sub_callback_for_progress(
        min_progress=0.67, max_progress=1.0, verbose_children=False
    )

    # test lab:
    #     DEPLOY the h2oGPTe server (docs sync: S3 > filesystem cache > h2oGPT2)
    test_lab.build(progress_callback=lab_build_progress)

    # test lab:
    #     complete dataset w/ ACTUAL values from the h2oGPTe server (answers, ...)
    test_lab.complete_dataset(
        complete_context=3,
        progress_callback=lab_completion_progress,
        save_as_you_go=tmp_path / "wip_testlab.json",
        parallelize=testing.TestLab.PARALLEL_RUN,
        retry_on_error=3,
    )
    # backup fully resolved dataset
    test_lab.save_as_json(tmp_path / "test_lab.json")
    evaluation = evaluate.run_evaluation(
        # dataset w/ prompts, constraints and model keys
        dataset=test_lab.dataset,
        # models to be evaluated / compared to get leaderboard
        models=list(test_lab.evaluated_models.values()),
        # evaluators
        evaluators=[LoopingDetectionEvaluator.evaluator_id()],
        # where to save the report
        results_location=tmp_path,
        # progress
        progress_callback=eval_progress,
        # log
        log_level=logging.INFO,
    )

    #
    # THEN
    #
    evaluator_class = LoopingDetectionEvaluator
    print(f"Evaluation:\n{evaluation}")
    print(f"HTML:\nfile://{evaluation.result.get_html_report_location()}")
    assert evaluation
    assert not evaluation.is_explainer_failed()

    # assert result
    result = evaluation.get_evaluator_result(evaluator_class.evaluator_id())
    print(result)
    assert result

    # assert leaderboard JSon representation data and meta
    then_eval.then_leaderboard_json(evaluation, evaluator_class.evaluator_id())


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
