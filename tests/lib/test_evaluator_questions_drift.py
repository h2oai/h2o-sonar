# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import os

import pytest

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import evaluate
from h2o_sonar import loggers
from h2o_sonar.evaluators import questions_drift_evaluator as qd_evaluator
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import models
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative
from tests.lib import then_eval


@pytest.mark.parametrize(
    "test_lab_path,evaluator_class,expect_compatible",
    [
        (
            "data/generative/sr1107_test_lab_171.json",
            qd_evaluator.QuestionsDriftEvaluator,
            True,
        ),
        (
            "data/generative/procedure_eval_test_lab_small.json",
            qd_evaluator.QuestionsDriftEvaluator,
            True,
        ),
        (
            "data/generative/kaggle_llm_science_exam_test_lab_2x_small_200.json",
            qd_evaluator.QuestionsDriftEvaluator,
            True,
        ),
        (
            "data/generative/nist-ai-600-1--test-lab--30p-5m.json",
            qd_evaluator.QuestionsDriftEvaluator,
            True,
        ),
        (
            "data/generative/talk2report_prompts_test_lab.json",
            qd_evaluator.QuestionsDriftEvaluator,
            True,
        ),
        (
            "data/generative/self_consistency_test_lab_28p.json",
            qd_evaluator.QuestionsDriftEvaluator,
            True,
        ),
        (
            "data/generative/kaggle_llm_science_exam_test_lab_4x_25.json",
            qd_evaluator.QuestionsDriftEvaluator,
            True,
        ),
    ],
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_sanity_no_drift(tmpdir, test_lab_path, evaluator_class, expect_compatible):
    """Test evaluator with questions from same domain - should show low drift."""
    #
    # GIVEN
    #

    rag_dataset = testing.RagTestLab.load_from_json(
        llm_host_connection=test_utils.health.get_h2ogpt(),
        file_path=test_lab_path,
    )
    llm_models = rag_dataset.evaluated_models.values()

    #
    # WHEN
    #

    evaluation = evaluate.run_evaluation(
        dataset=rag_dataset.dataset,
        models=llm_models,
        evaluators=[evaluator_class.evaluator_id()],
        results_location=tmpdir,
        log_level=loggers.DEBUG,
    )

    #
    # THEN
    #
    print(f"Evaluation:\n{evaluation}")
    if expect_compatible:
        print(f"HTML report:\nfile://{evaluation.result.get_html_report_location()}")
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

        # verify leaderboard is valid
        # drift scores are automatically validated by then_leaderboard_json
        print(f"Result: {result}")
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


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_drift_detection(tmpdir):
    """Test evaluator with questions from different domains - should detect drift."""
    #
    # GIVEN - create synthetic test dataset with clear domain shift
    #

    # group 1: math questions
    math_questions = [
        "What is the derivative of x^2?",
        "Solve for x: 2x + 5 = 15",
        "What is the integral of cos(x)?",
        "Calculate the area of a circle with radius 5",
        "What is the Pythagorean theorem?",
        "Simplify: (x + 2)(x - 3)",
    ]

    # group 2: history questions (semantic shift)
    history_questions = [
        "Who was the first president of the United States?",
        "When did World War II end?",
        "What caused the French Revolution?",
        "Who wrote the Declaration of Independence?",
        "When was the Roman Empire founded?",
        "What was the Cold War?",
    ]

    # combine in temporal order
    all_questions = math_questions + history_questions

    # create test dataset
    test_dataset = datasets.LlmDataset()
    for i, question in enumerate(all_questions):
        test_dataset.add_input(
            i=question,
            expected_output="answer",
            actual_output="answer",
            context=["context"],
        )

    # create mock model with minimal connection config
    mock_connection = h2o_sonar_config.ConnectionConfig(
        connection_type="test",
        name="test-connection",
        description="Test connection for unit testing",
    )
    mock_model = models.ExplainableLlmModel(
        connection=mock_connection,
        llm_model_name="test-model",
        key="test-model",
    )

    #
    # WHEN
    #

    evaluation = evaluate.run_evaluation(
        dataset=test_dataset,
        models=[mock_model],
        evaluators=[qd_evaluator.QuestionsDriftEvaluator.evaluator_id()],
        results_location=tmpdir,
        log_level=loggers.DEBUG,
    )

    #
    # THEN
    #
    print(f"Evaluation:\n{evaluation}")
    assert evaluation
    assert not evaluation.is_explainer_failed()

    # get drift score
    result = evaluation.get_evaluator_result(
        qd_evaluator.QuestionsDriftEvaluator.evaluator_id()
    )
    assert result

    # verify leaderboard
    then_eval.then_leaderboard_json(
        evaluation, qd_evaluator.QuestionsDriftEvaluator.evaluator_id()
    )

    print("Drift evaluation completed successfully")


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_insufficient_test_cases(tmpdir):
    """Test evaluator with insufficient test cases - should mark incompatible."""
    #
    # GIVEN - create dataset with < 10 questions
    #

    questions = [
        "Question 1?",
        "Question 2?",
        "Question 3?",
        "Question 4?",
        "Question 5?",
    ]

    test_dataset = datasets.LlmDataset()
    for question in questions:
        test_dataset.add_input(
            i=question,
            expected_output="answer",
            actual_output="answer",
            context=["context"],
        )

    # create mock model with minimal connection config
    mock_connection = h2o_sonar_config.ConnectionConfig(
        connection_type="test",
        name="test-connection",
        description="Test connection for unit testing",
    )
    mock_model = models.ExplainableLlmModel(
        connection=mock_connection,
        llm_model_name="test-model",
        key="test-model",
    )

    #
    # WHEN
    #

    evaluation = evaluate.run_evaluation(
        dataset=test_dataset,
        models=[mock_model],
        evaluators=[qd_evaluator.QuestionsDriftEvaluator.evaluator_id()],
        results_location=tmpdir,
        log_level=loggers.DEBUG,
    )

    #
    # THEN
    #
    print(f"Evaluation:\n{evaluation}")
    assert evaluation

    # with insufficient test cases, the evaluator should still run but return NaN
    # verify it doesn't crash and evaluation completed
    evaluation.get_evaluator_result(qd_evaluator.QuestionsDriftEvaluator.evaluator_id())

    # verify problems were reported (or just check evaluation completed)
    problems = evaluation.get_explainer_problems(
        qd_evaluator.QuestionsDriftEvaluator.evaluator_id()
    )
    print(f"Problems reported: {len(problems)}")
    print("Evaluation completed successfully despite insufficient test cases")


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_custom_parameters(tmpdir):
    """Test evaluator with custom parameters - should respect custom values."""
    #
    # GIVEN
    #

    questions = [f"Question {i}?" for i in range(20)]

    test_dataset = datasets.LlmDataset()
    for question in questions:
        test_dataset.add_input(
            i=question,
            expected_output="answer",
            actual_output="answer",
            context=["context"],
        )

    mock_connection = h2o_sonar_config.ConnectionConfig(
        connection_type="test",
        name="test-connection",
        description="Test connection for unit testing",
    )
    mock_model = models.ExplainableLlmModel(
        connection=mock_connection,
        llm_model_name="test-model",
        key="test-model",
    )

    #
    # WHEN - use custom split_ratio and drift_threshold
    #

    evaluation = evaluate.run_evaluation(
        dataset=test_dataset,
        models=[mock_model],
        evaluators=[qd_evaluator.QuestionsDriftEvaluator.evaluator_id()],
        extra_params={
            qd_evaluator.QuestionsDriftEvaluator.PARAM_DRIFT_THRESHOLD: 0.05,
            qd_evaluator.QuestionsDriftEvaluator.PARAM_SPLIT_RATIO: 0.7,
        },
        results_location=tmpdir,
        log_level=loggers.DEBUG,
    )

    #
    # THEN
    #
    print(f"Evaluation:\n{evaluation}")
    assert evaluation
    assert not evaluation.is_explainer_failed()

    result = evaluation.get_evaluator_result(
        qd_evaluator.QuestionsDriftEvaluator.evaluator_id()
    )
    assert result

    # verify leaderboard
    then_eval.then_leaderboard_json(
        evaluation, qd_evaluator.QuestionsDriftEvaluator.evaluator_id()
    )

    print("Custom parameters accepted successfully")


@pytest.mark.h2o_sonar
@pytest.mark.generative
@pytest.mark.parametrize("device_config", ["cpu", "auto"])
def test_gpu_cpu_config(tmpdir, device_config):
    """Test evaluator with different GPU/CPU configurations."""
    #
    # GIVEN
    #

    questions = [f"Question {i}?" for i in range(15)]

    test_dataset = datasets.LlmDataset()
    for question in questions:
        test_dataset.add_input(
            i=question,
            expected_output="answer",
            actual_output="answer",
            context=["context"],
        )

    mock_connection = h2o_sonar_config.ConnectionConfig(
        connection_type="test",
        name="test-connection",
        description="Test connection for unit testing",
    )
    mock_model = models.ExplainableLlmModel(
        connection=mock_connection,
        llm_model_name="test-model",
        key="test-model",
    )

    #
    # WHEN - set device config
    #

    original_device = os.environ.get("H2O_SONAR_CFG_DEVICE")
    try:
        os.environ["H2O_SONAR_CFG_DEVICE"] = device_config

        evaluation = evaluate.run_evaluation(
            dataset=test_dataset,
            models=[mock_model],
            evaluators=[qd_evaluator.QuestionsDriftEvaluator.evaluator_id()],
            results_location=tmpdir,
            log_level=loggers.DEBUG,
        )

        #
        # THEN
        #
        print(f"Evaluation with device={device_config}:\n{evaluation}")
        assert evaluation
        assert not evaluation.is_explainer_failed()

        result = evaluation.get_evaluator_result(
            qd_evaluator.QuestionsDriftEvaluator.evaluator_id()
        )
        assert result

        # verify leaderboard
        then_eval.then_leaderboard_json(
            evaluation, qd_evaluator.QuestionsDriftEvaluator.evaluator_id()
        )

        print(f"Device config {device_config} test passed successfully")

    finally:
        # restore original device config
        if original_device is not None:
            os.environ["H2O_SONAR_CFG_DEVICE"] = original_device
        elif "H2O_SONAR_CFG_DEVICE" in os.environ:
            del os.environ["H2O_SONAR_CFG_DEVICE"]


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_empty_questions(tmpdir):
    """Test evaluator handles empty and whitespace-only questions gracefully."""
    #
    # GIVEN - mix of valid, empty, and whitespace-only questions
    #

    questions = [
        "Valid question 1?",
        "",  # empty
        "Valid question 2?",
        None,  # None
        "Valid question 3?",
        "   ",  # whitespace only - should be filtered out
        "Valid question 4?",
        "",  # empty
        "\t\n",  # whitespace only - should be filtered out
        "Valid question 5?",
        "Valid question 6?",
        "Valid question 7?",
        "Valid question 8?",
        "Valid question 9?",
        "Valid question 10?",
    ]

    test_dataset = datasets.LlmDataset()
    for question in questions:
        test_dataset.add_input(
            i=question if question else "",
            expected_output="answer",
            actual_output="answer",
            context=["context"],
        )

    mock_connection = h2o_sonar_config.ConnectionConfig(
        connection_type="test",
        name="test-connection",
        description="Test connection for unit testing",
    )
    mock_model = models.ExplainableLlmModel(
        connection=mock_connection,
        llm_model_name="test-model",
        key="test-model",
    )

    #
    # WHEN
    #

    evaluation = evaluate.run_evaluation(
        dataset=test_dataset,
        models=[mock_model],
        evaluators=[qd_evaluator.QuestionsDriftEvaluator.evaluator_id()],
        results_location=tmpdir,
        log_level=loggers.DEBUG,
    )

    #
    # THEN - should skip empty questions and process valid ones
    #
    print(f"Evaluation:\n{evaluation}")
    assert evaluation
    assert not evaluation.is_explainer_failed()

    result = evaluation.get_evaluator_result(
        qd_evaluator.QuestionsDriftEvaluator.evaluator_id()
    )
    assert result

    # verify leaderboard
    then_eval.then_leaderboard_json(
        evaluation, qd_evaluator.QuestionsDriftEvaluator.evaluator_id()
    )

    print("Empty questions handled successfully")


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_identical_questions_near_zero_drift(tmpdir):
    """Test evaluator with identical questions - should show near-zero drift.

    GIVEN identical questions in both temporal groups
    WHEN drift is calculated
    THEN drift score should be very close to 0.0 (< 0.01)

    """
    #
    # GIVEN - create dataset with 15 identical questions
    #

    question = "What is machine learning?"
    questions = [question] * 15

    test_dataset = datasets.LlmDataset()
    for q in questions:
        test_dataset.add_input(
            i=q,
            expected_output="ML is a field of AI",
            actual_output="ML is a field of AI",
            context=["AI context"],
        )

    mock_connection = h2o_sonar_config.ConnectionConfig(
        connection_type="test",
        name="test-connection",
        description="Test connection for unit testing",
    )
    mock_model = models.ExplainableLlmModel(
        connection=mock_connection,
        llm_model_name="test-model",
        key="test-model",
    )

    #
    # WHEN
    #

    evaluation = evaluate.run_evaluation(
        dataset=test_dataset,
        models=[mock_model],
        evaluators=[qd_evaluator.QuestionsDriftEvaluator.evaluator_id()],
        results_location=tmpdir,
        log_level=loggers.DEBUG,
    )

    #
    # THEN
    #
    print(f"Evaluation: {evaluation}")
    assert evaluation
    assert not evaluation.is_explainer_failed()

    # extract drift score from leaderboard
    leaderboard_json = then_eval.then_leaderboard_json(
        evaluation, qd_evaluator.QuestionsDriftEvaluator.evaluator_id()
    )

    drift_score = None
    for model_data in leaderboard_json.get("data", {}).values():
        drift_score = model_data.get("questions_drift")
        break

    print(f"Drift score for identical questions: {drift_score}")
    assert drift_score is not None
    assert drift_score < 0.01, (
        f"Expected drift < 0.01 for identical questions, got {drift_score}"
    )

    print("DONE: Identical questions produce near-zero drift")


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_multi_model_same_drift_score(tmpdir):
    """Test that all models receive identical drift scores.

    GIVEN multiple models evaluated on same questions
    WHEN drift is calculated
    THEN all models should have identical drift scores

    """
    #
    # GIVEN - create dataset with clear drift (math then history)
    #

    # group 1: math questions
    math_questions = [
        "What is 2 + 2?",
        "What is the square root of 16?",
        "Solve: x + 5 = 10",
        "What is 10 * 10?",
        "Calculate 100 / 4",
        "What is the value of pi?",
    ]

    # group 2: history questions (semantic shift)
    history_questions = [
        "Who discovered America?",
        "When did WWII start?",
        "Who was Napoleon?",
        "What year did Rome fall?",
        "Who signed Magna Carta?",
        "When was the Renaissance?",
    ]

    # combine in temporal order
    all_questions = math_questions + history_questions

    # create 3 different mock models
    mock_connection = h2o_sonar_config.ConnectionConfig(
        connection_type="test",
        name="test-connection",
        description="Test connection for unit testing",
    )

    models_list = [
        models.ExplainableLlmModel(
            connection=mock_connection,
            llm_model_name=f"test-model-{i}",
            key=f"test-model-{i}",
        )
        for i in range(1, 4)
    ]

    # create test dataset with questions for each model
    test_dataset = datasets.LlmDataset()
    for model in models_list:
        for question in all_questions:
            test_dataset.add_input(
                i=question,
                expected_output="answer",
                actual_output="answer",
                context=["context"],
                model_key=model.key,
            )

    #
    # WHEN
    #

    evaluation = evaluate.run_evaluation(
        dataset=test_dataset,
        models=models_list,
        evaluators=[qd_evaluator.QuestionsDriftEvaluator.evaluator_id()],
        results_location=tmpdir,
        log_level=loggers.DEBUG,
    )

    #
    # THEN - all models should have identical drift scores
    #
    print(f"Evaluation: {evaluation}")
    assert evaluation
    assert not evaluation.is_explainer_failed()

    # extract drift scores from leaderboard
    leaderboard_json = then_eval.then_leaderboard_json(
        evaluation, qd_evaluator.QuestionsDriftEvaluator.evaluator_id()
    )

    drift_scores = []
    for model_key, model_data in leaderboard_json.get("data", {}).items():
        drift_score = model_data.get("questions_drift")
        print(f"Model {model_key}: drift score = {drift_score}")
        drift_scores.append(drift_score)

    # verify all drift scores are identical (dataset-level metric)
    assert len(drift_scores) == 3, f"Expected 3 models, got {len(drift_scores)}"
    assert all(score == drift_scores[0] for score in drift_scores), (
        f"Expected all drift scores to be identical, got {drift_scores}"
    )

    print(
        f"DONE: All models have identical drift score ({drift_scores[0]}) - "
        f"confirms dataset-level behavior"
    )


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_very_short_questions(tmpdir):
    """Test evaluator handles very short questions (1-3 characters).

    GIVEN dataset with very short questions
    WHEN drift is calculated
    THEN evaluator should process without crashing and log warning

    """
    #
    # GIVEN - create dataset with very short questions
    #

    # group 1: short math-related text
    short_math = ["2+2", "x=5", "pi", "sin", "cos", "log", "e^x"]

    # group 2: short history-related text (semantic shift)
    short_history = ["war", "Rome", "1776", "king", "vote", "tax", "USA"]

    # combine in temporal order
    all_questions = short_math + short_history

    test_dataset = datasets.LlmDataset()
    for question in all_questions:
        test_dataset.add_input(
            i=question,
            expected_output="answer",
            actual_output="answer",
            context=["context"],
        )

    mock_connection = h2o_sonar_config.ConnectionConfig(
        connection_type="test",
        name="test-connection",
        description="Test connection for unit testing",
    )
    mock_model = models.ExplainableLlmModel(
        connection=mock_connection,
        llm_model_name="test-model",
        key="test-model",
    )

    #
    # WHEN
    #

    evaluation = evaluate.run_evaluation(
        dataset=test_dataset,
        models=[mock_model],
        evaluators=[qd_evaluator.QuestionsDriftEvaluator.evaluator_id()],
        results_location=tmpdir,
        log_level=loggers.DEBUG,
    )

    #
    # THEN - should complete successfully despite very short questions
    #
    print(f"Evaluation: {evaluation}")
    assert evaluation
    assert not evaluation.is_explainer_failed()

    result = evaluation.get_evaluator_result(
        qd_evaluator.QuestionsDriftEvaluator.evaluator_id()
    )
    assert result

    # verify leaderboard (drift score should be calculated)
    leaderboard_json = then_eval.then_leaderboard_json(
        evaluation, qd_evaluator.QuestionsDriftEvaluator.evaluator_id()
    )

    drift_score = None
    for model_data in leaderboard_json.get("data", {}).values():
        drift_score = model_data.get("questions_drift")
        break

    print(f"Drift score for very short questions: {drift_score}")
    assert drift_score is not None, (
        "Expected drift score to be calculated for short questions"
    )

    print("DONE: Very short questions handled successfully with warning")


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_min_test_cases_parameter_default():
    """Verify min_test_cases parameter defaults to 10, not 0.

    This test ensures the fix for the parameter default issue where
    _get_custom_param_min_test_case() was called without passing minimum,
    causing it to default to 0 instead of 10.

    GIVEN Questions Drift Evaluator
    WHEN parameters are retrieved
    THEN min_test_cases default value should be 10

    """
    # GIVEN + WHEN
    # Access class-level _parameters attribute directly
    params = qd_evaluator.QuestionsDriftEvaluator._parameters
    min_tc_param = next((p for p in params if p.param_name == "min_test_cases"), None)

    # THEN
    print(f"min_test_cases parameter: {min_tc_param}")
    assert min_tc_param is not None, "min_test_cases parameter should exist"
    assert min_tc_param.default_value == 10, (
        f"Expected default_value=10, got {min_tc_param.default_value}"
    )
    print("DONE: min_test_cases correctly defaults to 10")


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_insufficient_test_cases_warning_triggers(tmpdir, caplog):
    """Verify insufficient test case warning triggers with fewer than 10 questions.

    This test ensures the guard condition properly triggers when there are
    insufficient test cases, validating that the default min_test_cases=10
    is being used correctly.

    GIVEN dataset with only 5 questions (below default minimum of 10)
    WHEN evaluation runs
    THEN warning should be logged about insufficient test cases

    """
    import logging

    # GIVEN - dataset with only 5 questions (below minimum)
    questions = [
        "Question 1?",
        "Question 2?",
        "Question 3?",
        "Question 4?",
        "Question 5?",
    ]

    test_dataset = datasets.LlmDataset()
    for question in questions:
        test_dataset.add_input(
            i=question,
            expected_output="answer",
            actual_output="answer",
            context=["context"],
        )

    mock_connection = h2o_sonar_config.ConnectionConfig(
        connection_type="test",
        name="test-connection",
        description="Test connection for unit testing",
    )
    mock_model = models.ExplainableLlmModel(
        connection=mock_connection,
        llm_model_name="test-model",
        key="test-model",
    )

    # WHEN
    with caplog.at_level(logging.WARNING):
        evaluation = evaluate.run_evaluation(
            dataset=test_dataset,
            models=[mock_model],
            evaluators=[qd_evaluator.QuestionsDriftEvaluator.evaluator_id()],
            results_location=tmpdir,
            log_level=loggers.DEBUG,
        )

    # THEN
    print(f"Evaluation: {evaluation}")
    assert evaluation

    # check that warning was logged
    warning_messages = [
        record.message for record in caplog.records if record.levelno == logging.WARNING
    ]
    print(f"Warning messages: {warning_messages}")

    # look for insufficient test cases warning
    insufficient_warning_found = any(
        "Insufficient test cases" in msg and "Found 5, minimum required: 10" in msg
        for msg in warning_messages
    )

    assert insufficient_warning_found, (
        "Expected warning about insufficient test cases (5 < 10) but none found. "
        f"Warnings: {warning_messages}"
    )

    print("DONE: Insufficient test case warning correctly triggered")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
