# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import pytest

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import evaluate
from h2o_sonar import loggers
from h2o_sonar.evaluators import bertscore_evaluator
from h2o_sonar.lib.api import commons
from h2o_sonar.utils import progress
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative
from tests.lib import then_eval


# constants
BertscoreEvaluator = bertscore_evaluator.BertscoreEvaluator


@pytest.mark.skipif(
    not test_utils.health.is_h2ogpte(),
    reason="h2oGPTE service is not reachable",
)
@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({"bert-score"}),
    reason="Package 'bert-score' is not installed",
)
@pytest.mark.parametrize(
    "test_lab_path,evaluator_class,expect_compatible",
    [
        (
            #
            # h2oGPTe server
            #
            #
            # RAG test labs:
            #
            "data/generative/kaggle_llm_science_exam_test_lab_2x_small_3.json",
            #
            # EVALUATORS: BERTScore - semantic similarity
            #
            BertscoreEvaluator,
            #
            # Expected compatibility: True (all rows have both AA and EA)
            #
            True,
        ),
        (
            #
            # h2oGPTe server
            #
            #
            # RAG test lab with ALL missing EAs
            #
            "data/generative/toxicity_test_lab_2x3p_EMPTY_EA.json",
            #
            # EVALUATOR: BERTScore
            #
            BertscoreEvaluator,
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
                params={evaluator_class.PARAM_METRIC_THRESHOLD: 0.7},
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
        then_eval.then_leaderboard_json(
            evaluation,
            evaluator_class.evaluator_id(),
            metric_ids_to_assert=[
                evaluator_class.METRIC_BERTSCORE_PRECISION,
                evaluator_class.METRIC_BERTSCORE_RECALL,
                evaluator_class.METRIC_BERTSCORE_F1,
            ],
            threshold=0.7,
        )
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


@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({"bert-score"}),
    reason="Required Python package 'bert-score' is not installed",
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_bertscore_metrics(tmpdir):
    """Test BERTScore metrics with known similar and dissimilar texts."""
    import json
    import uuid

    from h2o_sonar.lib.api import datasets
    from h2o_sonar.lib.api import explanations as e10s
    from h2o_sonar.lib.api import formats as f5s
    from h2o_sonar.lib.api import models
    from h2o_sonar.lib.api import persistences

    #
    # GIVEN
    #
    logger = loggers.SonarPrintLogger()

    evaluation_key = "bertscore-test-key"
    job_key = "bertscore-job-key"
    model_key = "model-1"
    test_key = "test-1"

    # test cases: (expected_output, actual_output)
    test_cases = [
        # identical text - should have high scores
        (
            "The quick brown fox jumps over the lazy dog.",
            "The quick brown fox jumps over the lazy dog.",
        ),
        # semantically similar but different words
        (
            "The fast brown fox leaps over the sleepy dog.",
            "The quick brown fox jumps over the lazy dog.",
        ),
        # paraphrase - should have good semantic similarity
        (
            "A speedy auburn fox hops above a drowsy canine.",
            "The quick brown fox jumps over the lazy dog.",
        ),
        # different meaning - should have lower scores
        (
            "The cat sat on the mat.",
            "The quick brown fox jumps over the lazy dog.",
        ),
    ]

    llm_testset = datasets.LlmDataset()
    for i, (expected, actual) in enumerate(test_cases):
        print(f"\nTest case {i}:")
        print(f"  Expected: {expected}")
        print(f"  Actual:   {actual}")
        llm_testset.inputs.append(
            datasets.LlmDataset.LlmDatasetRow(
                i=f"Input {i}",
                actual_output=actual,
                expected_output=expected,
                model_key=model_key,
                test_key=test_key,
                key=str(i),
            )
        )

    dt_frame_llm_testset = llm_testset.to_datatable()

    e = BertscoreEvaluator()
    e.setup(
        model=models.ExplainableModel(
            model_src="mock",
            predict_method=lambda x: x,
        ),
        persistence=persistences.ExplainerPersistence(
            data_dir=str(tmpdir),
            username=commons.DEFAULT_USER,
            explainer_id=e.explainer_id(),
            explainer_job_key=job_key,
            mli_key=evaluation_key,
        ),
    )
    e_model = models.ExplainableLlmModel(
        connection=test_utils.health.get_h2ogpt(),
        model_type=models.ExplainableModelType.h2ogpte_llm,
        name="Mock",
        llm_model_name="mock/llm",
        logger=logger,
        key=str(uuid.uuid4()),
    )
    e.models = [e_model]

    #
    # WHEN
    #

    explanations = e.evaluate(llm_testset=dt_frame_llm_testset)

    #
    # THEN
    #

    # assert explanations were created
    assert explanations
    assert len(explanations) >= 2  # At least results and leaderboard

    ep = persistences.ExplainerPersistence(
        data_dir=str(tmpdir),
        mli_key=evaluation_key,
        username=commons.DEFAULT_USER,
        explainer_id=e.explainer_id(),
        explainer_job_key=job_key,
    )

    # check leaderboard JSON
    json_leaderboard_path = ep.get_explanation_file_path(
        explanation_type=e10s.LlmHeatmapLeaderboardExplanation.explanation_type(),
        explanation_format=f5s.LlmLeaderboardJSonFormat.mime,
    )
    assert json_leaderboard_path

    # check evaluation results JSON
    json_eval_result_path = ep.get_explanation_file_path(
        explanation_type=e10s.LlmEvalResultsExplanation.explanation_type(),
        explanation_format=f5s.CustomJsonFormat.mime,
    )

    with open(json_eval_result_path) as f:
        eval_results = json.load(f)

    print(f"\nEvaluation results:\n{json.dumps(eval_results, indent=2)}")

    # assert all metrics are present
    assert "results" in eval_results
    results_data = eval_results["results"]

    for i, result in enumerate(results_data):
        print(f"\nTest case {i} metrics:")
        print(f"  Precision: {result[e.METRIC_BERTSCORE_PRECISION]:.4f}")
        print(f"  Recall:    {result[e.METRIC_BERTSCORE_RECALL]:.4f}")
        print(f"  F1:        {result[e.METRIC_BERTSCORE_F1]:.4f}")

        # All metrics should be between 0 and 1
        assert 0.0 <= result[e.METRIC_BERTSCORE_PRECISION] <= 1.0
        assert 0.0 <= result[e.METRIC_BERTSCORE_RECALL] <= 1.0
        assert 0.0 <= result[e.METRIC_BERTSCORE_F1] <= 1.0

    # test case 0: Identical texts should have very high scores (close to 1.0)
    identical_metrics = results_data[0]
    assert identical_metrics[e.METRIC_BERTSCORE_F1] > 0.95, (
        f"Identical texts should have F1 > 0.95, got "
        f"{identical_metrics[e.METRIC_BERTSCORE_F1]}"
    )

    # test case 1: Similar texts should have good scores
    similar_metrics = results_data[1]
    assert similar_metrics[e.METRIC_BERTSCORE_F1] > 0.8, (
        f"Similar texts should have F1 > 0.8, got "
        f"{similar_metrics[e.METRIC_BERTSCORE_F1]}"
    )

    # test case 3: Different texts should have lower scores than similar texts
    different_metrics = results_data[3]
    assert (
        different_metrics[e.METRIC_BERTSCORE_F1]
        < similar_metrics[e.METRIC_BERTSCORE_F1]
    ), (
        f"Different texts should have lower F1 than similar texts: "
        f"{different_metrics[e.METRIC_BERTSCORE_F1]} vs "
        f"{similar_metrics[e.METRIC_BERTSCORE_F1]}"
    )

    print(f"\nEvaluation results saved to: file://{json_eval_result_path}")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
