# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import pytest

from h2o_sonar import evaluate
from h2o_sonar import loggers
from h2o_sonar.evaluators import gptscore_evaluator
from h2o_sonar.evaluators import gptscore_machine_translation_evaluator as mt
from h2o_sonar.evaluators import gptscore_question_answering_evaluator as qa
from h2o_sonar.evaluators import gptscore_summary_with_reference_evaluator as swe
from h2o_sonar.evaluators import gptscore_summary_without_reference_evaluator as se
from h2o_sonar.lib.api import commons
from h2o_sonar.utils import caching
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative


MODELS = [
    caching.MODEL_GOOGLE_FLAN_T5_SMALL,  # smallest model
    # caching.MODEL_GOOGLE_FLAN_T5_BASE,
    # caching.MODEL_GOOGLE_FLAN_T5_LARGE,
    # caching.MODEL_GOOGLE_FLAN_T5_XL,
    # caching.MODEL_GOOGLE_FLAN_T5_XXL,
    # caching.MODEL_FACEBOOK_OPT_125M, # for testing various tokenization in opt models
    # caching.MODEL_FACEBOOK_OPT_350M,
    # caching.MODEL_FACEBOOK_OPT_1_3B,
    # caching.MODEL_FACEBOOK_OPT_2_7B,
    # caching.MODEL_FACEBOOK_OPT_6_7B,
    # caching.MODEL_FACEBOOK_OPT_13B,
    # caching.MODEL_FACEBOOK_OPT_66B,
    # caching.MODEL_GPT2_MEDIUM,  # default model
    # caching.MODEL_GPT2_LARGE,
    # caching.MODEL_GPT2_XL,
    # caching.MODEL_ELEUTHERAI_GPT_J_6B,
]


@pytest.mark.parametrize(
    "model,test_lab_path",
    [
        (
            MODELS[0],
            "data/generative/kims_summarization_test_lab_small.json",
        ),
        # Google Notebook LM debug
        # (
        #     MODELS[0],
        #     "data/generative/google-notebook-lm-test-lab.json",
        # )
    ],
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_summary_without_ref_evaluator(tmpdir, model, test_lab_path):
    print(f"\n{model}:")
    #
    # GIVEN
    #
    param_eval_gpt_score_model = (
        gptscore_evaluator.GptScoreEvaluator.PARAM_EVAL_GPT_SCORE_MODEL
    )
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
        evaluators=[
            commons.EvaluatorToRun(
                evaluator_id=se.GptScoreSummaryWithoutReferenceEvaluator.evaluator_id(),
                params={param_eval_gpt_score_model: model},
            )
        ],
        results_location=tmpdir,
        log_level=loggers.DEBUG,
    )

    #
    # THEN
    #

    assert evaluation
    assert not evaluation.is_explainer_failed()
    print(f"HTML report:\nfile://{evaluation.result.get_html_report_location()}")


@pytest.mark.parametrize(
    "model,test_lab_path",
    [
        (
            MODELS[0],
            "data/generative/kims_summarization_test_lab_small.json",
        ),
        # Google Notebook LM debug
        # (
        #     MODELS[0],
        #     "data/generative/google-notebook-lm-test-lab.json",
        # )
    ],
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_summary_with_ref_evaluator(tmpdir, model, test_lab_path):
    print(f"\n{model}:")
    #
    # GIVEN
    #
    param_eval_gpt_score_model = (
        gptscore_evaluator.GptScoreEvaluator.PARAM_EVAL_GPT_SCORE_MODEL
    )
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
        evaluators=[
            commons.EvaluatorToRun(
                evaluator_id=swe.GptScoreSummaryWithReferenceEvaluator.evaluator_id(),
                params={param_eval_gpt_score_model: model},
            )
        ],
        results_location=tmpdir,
        log_level=loggers.DEBUG,
    )

    #
    # THEN
    #

    assert evaluation
    assert not evaluation.is_explainer_failed()
    print(f"HTML report:\nfile://{evaluation.result.get_html_report_location()}")


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.parametrize(
    "model",
    MODELS,
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_question_answering(tmpdir, model):
    print(f"\n{model}:")
    #
    # GIVEN
    #
    PARAM_EVAL_GPT_SCORE_MODEL = (
        gptscore_evaluator.GptScoreEvaluator.PARAM_EVAL_GPT_SCORE_MODEL
    )
    rag_dataset = testing.RagTestLab.load_from_json(
        llm_host_connection=test_utils.health.get_h2ogpt(),
        file_path="data/generative/kaggle_llm_science_exam_test_lab_4x_25.json",
    )
    llm_models = rag_dataset.evaluated_models.values()

    #
    # WHEN
    #

    evaluation = evaluate.run_evaluation(
        dataset=rag_dataset.dataset,
        models=llm_models,
        evaluators=[
            commons.EvaluatorToRun(
                evaluator_id=qa.GptScoreQuestionAnsweringEvaluator.evaluator_id(),
                params={PARAM_EVAL_GPT_SCORE_MODEL: model},
            )
        ],
        results_location=tmpdir,
        log_level=loggers.DEBUG,
    )

    #
    # THEN
    #

    assert evaluation
    assert not evaluation.is_explainer_failed()
    print(f"HTML report:\nfile://{evaluation.result.get_html_report_location()}")


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.parametrize(
    "model",
    MODELS,
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_machine_translation(tmpdir, model):
    print(f"\n{model}:")
    #
    # GIVEN
    #
    PARAM_EVAL_GPT_SCORE_MODEL = (
        gptscore_evaluator.GptScoreEvaluator.PARAM_EVAL_GPT_SCORE_MODEL
    )
    rag_dataset = testing.RagTestLab.load_from_json(
        llm_host_connection=test_utils.health.get_h2ogpt(),
        file_path="data/generative/dummy_translation_test_lab_small.json",
    )
    llm_models = rag_dataset.evaluated_models.values()

    #
    # WHEN
    #

    evaluation = evaluate.run_evaluation(
        dataset=rag_dataset.dataset,
        models=llm_models,
        evaluators=[
            commons.EvaluatorToRun(
                evaluator_id=mt.GptScoreMachineTranslationEvaluator.evaluator_id(),
                params={PARAM_EVAL_GPT_SCORE_MODEL: model},
            )
        ],
        results_location=tmpdir,
        log_level=loggers.DEBUG,
    )

    #
    # THEN
    #

    assert evaluation
    assert not evaluation.is_explainer_failed()
    print(f"HTML report:\nfile://{evaluation.result.get_html_report_location()}")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
