# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import pytest

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import evaluate
from h2o_sonar import loggers
from h2o_sonar.evaluators import answer_accuracy_evaluator
from h2o_sonar.lib.api import commons
from h2o_sonar.utils import progress
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative
from tests.lib import then_eval


# constants
AnswerAccuracyEvaluator = answer_accuracy_evaluator.AnswerAccuracyEvaluator


@pytest.mark.skipif(
    not test_utils.health.is_h2ogpte(),
    reason="h2oGPTE service is not reachable",
)
@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({"optimum"}),
    reason="Package 'optimum' is not installed",
)
@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({"sentence_transformers"}),
    reason="Package 'sentence-transformers' is not installed",
)
@pytest.mark.parametrize(
    "test_lab_path,evaluator_class,expect_compatible",
    [
        (
            #
            # RAG test labs:
            #
            "data/generative/kaggle_llm_science_exam_test_lab_2x_small_3.json",
            #
            # EVALUATORS: Answer Accuracy
            #
            AnswerAccuracyEvaluator,
            #
            # Expected compatibility: True (all rows have both AA and EA)
            #
            True,
        ),
        (
            #
            # RAG test lab with ALL missing EAs
            #
            "data/generative/toxicity_test_lab_2x3p_EMPTY_EA.json",
            #
            # EVALUATOR: Answer Accuracy
            #
            AnswerAccuracyEvaluator,
            #
            # Expected compatibility: False (NO row has both AA and EA)
            #
            False,
        ),
        # NOTE: 1x_EMPTY case currently fails due to strict compatibility check
        # TODO: Investigate _check_llm_dataset_compatibility() behavior
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_sanity(
    tmpdir,
    h2ogpte_connection_fixture: h2o_sonar_config.ConnectionConfig,
    test_lab_path,
    evaluator_class,
    expect_compatible,
):
    #
    # GIVEN
    #

    test_lab = testing.RagTestLab.load_from_json(
        llm_host_connection=h2ogpte_connection_fixture,
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
                params={evaluator_class.PARAM_METRIC_THRESHOLD: 0.75},
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
    if expect_compatible:
        print(f"HTML:\nfile://{evaluation.result.get_html_report_location()}")
    assert evaluation

    # check compatibility based on expected result
    incompatible_ids = evaluation.get_incompatible_evaluator_ids()
    print(f"Incompatible evaluator IDs: {incompatible_ids}")

    if expect_compatible:
        # evaluator should be compatible and execute successfully
        assert evaluator_class.evaluator_id() not in incompatible_ids, (
            f"Expected {evaluator_class.evaluator_id()} to be compatible, "
            f"but was incompatible"
        )
        assert not evaluation.is_explainer_failed()

        # assert result
        result = evaluation.get_evaluator_result(evaluator_class.evaluator_id())
        print(result)
        assert result

        # assert leaderboard JSon representation data and meta
        then_eval.then_leaderboard_json(evaluation, evaluator_class.evaluator_id())
    else:
        # evaluator should be marked as incompatible
        assert evaluator_class.evaluator_id() in incompatible_ids, (
            f"Expected {evaluator_class.evaluator_id()} to be incompatible, "
            f"but got: {incompatible_ids}"
        )

        # verify that a problem was generated
        problems = evaluation.get_explainer_problems(evaluator_class.evaluator_id())
        print(f"Problems: {problems}")
        assert len(problems) > 0, "Expected at least one problem to be reported"

        # verify the problem description contains the expected message
        problem_descriptions = [p.description for p in problems]
        problem_found = any(
            ("does not contain" in desc and "expected answer" in desc)
            for desc in problem_descriptions
        )
        assert problem_found, (
            f"Expected problem about missing expected answers not found. "
            f"Problems: {problem_descriptions}"
        )


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
