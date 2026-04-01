# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
#
# Kaggle LLM Science Exam:
# - https://www.kaggle.com/competitions/kaggle-llm-science-exam
#
import logging

import pytest

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import evaluate
from h2o_sonar.evaluators import classification_evaluator
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import persistences
from h2o_sonar.utils import testing
from tests import test_utils


@pytest.mark.skipif(
    not test_utils.health.is_h2ogpte(),
    reason="Test requires h2oGPTe server whose connection is not available",
)
@pytest.mark.parametrize(
    "connection,test_lab_path",
    [
        (
            # negative test: non-classification dataset
            test_utils.health.get_h2ogpte(),
            "data/generative/sr1107_test_lab_15m.json",
        ),
        (
            # binomial
            test_utils.health.get_h2ogpte(),
            "data/generative/kaggle_llm_science_exam_class_bin_test_lab.json",
        ),
        (
            # multinomial
            test_utils.health.get_h2ogpte(),
            "data/generative/kaggle_llm_science_exam_class_multi_test_lab.json",
        ),
        (
            # multinomial w/ new lines in class names
            test_utils.health.get_h2ogpte(),
            "data/generative/class_multi_test_lab.json",
        ),
        (
            test_utils.health.get_h2ogpte(),
            "data/generative/"
            "kaggle_llm_science_exam_test_lab_2x_small_wrong_answers.json",
        ),
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_load_lab(
    tmp_path, connection: h2o_sonar_config.ConnectionConfig, test_lab_path
):
    """Evaluate KGM model from:

    https://www.kaggle.com/competitions/kaggle-llm-science-exam/leaderboard

    """
    #
    # GIVEN
    #

    # TEST LAB
    test_lab = testing.RagTestLab.load_from_json(
        llm_host_connection=connection,
        file_path=test_lab_path,
        docs_cache_dir="data/generative/h2ogpte-documents",
    )

    #
    # WHEN
    #

    evaluation = evaluate.run_evaluation(
        # dataset w/ prompts, constraints and model keys
        dataset=test_lab.dataset,
        # models to be evaluated / compared to get leaderboard
        models=list(test_lab.evaluated_models.values()),
        # evaluators
        evaluators=[
            classification_evaluator.ClassificationEvaluator().evaluator_id(),
        ],
        # where to save the report
        results_location=tmp_path,
        log_level=logging.DEBUG,
    )

    #
    # THEN
    #

    print(f"{evaluation}")
    assert not evaluation.get_failed_evaluator_ids()

    ep = persistences.ExplainerPersistence(
        data_dir=evaluation.result.results_location,
        mli_key=evaluation.key,
        username=commons.DEFAULT_USER,
        explainer_id=classification_evaluator.ClassificationEvaluator().evaluator_id(),
        explainer_job_key=next(iter(evaluation.result.explainers)),
    )
    md_path = ep.get_explanation_file_path(
        explanation_type=e10s.LlmClassifierLeaderboardExplanation.explanation_type(),
        explanation_format=f5s.MarkdownFormat.mime,
    )
    md_es_path = ep.get_explanation_file_path(
        explanation_type=e10s.LlmClassifierLeaderboardExplanation.explanation_type(),
        explanation_format=f5s.EvalStudioMarkdownFormat.mime,
    )

    print(
        f"Explanations:\n"
        f"  HTML : file://{evaluation.result.get_html_report_location()}\n"
        f"  MD ES: file://{md_es_path}\n"
        f"  MD   : file://{md_path}\n"
    )


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
