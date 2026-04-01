# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import pytest

from h2o_sonar import evaluate
from h2o_sonar import loggers
from h2o_sonar.evaluators import summarization_evaluator
from h2o_sonar.utils import testing
from tests import test_utils


@pytest.mark.skip(reason="The evaluator has been excluded from the container.")
@pytest.mark.parametrize(
    "test_lab,is_negative_test",
    [
        ("data/generative/kims_summarization_test_lab_tiny.json", False),
        ("data/generative/kims_summarization_negative_test_lab.json", True),
        # runs for a long time on CPU:
        # ("data/generative/google-notebook-lm-test-lab.json", False),
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_evaluator(tmp_path, test_lab: str, is_negative_test: bool):
    #
    # GIVEN
    #
    rag_dataset = testing.RagTestLab.load_from_json(
        llm_host_connection=test_utils.health.get_h2ogpt(),
        file_path=test_lab,
    )
    llm_models = rag_dataset.evaluated_models.values()

    #
    # WHEN
    #

    evaluation = evaluate.run_evaluation(
        dataset=rag_dataset.dataset,
        models=llm_models,
        evaluators=[summarization_evaluator.SummarizationEvaluator.evaluator_id()],
        results_location=tmp_path,
        log_level=loggers.DEBUG,
    )

    #
    # THEN
    #

    print(f"Evaluation error: {evaluation.error}")
    print(f"Evaluation status: {evaluation.status}")
    print(f"Evaluation progress: {evaluation.progress}")

    assert evaluation
    if is_negative_test:
        assert evaluation.is_explainer_failed()  # all inputs are too short
        assert len(evaluation.get_failed_evaluator_ids()) == 1
        assert "Summarization" in evaluation.error
    else:
        assert not evaluation.is_evaluator_failed()
        assert len(evaluation.get_successful_evaluator_ids()) == 1

    print(f"HTML report:\nfile://{evaluation.result.get_html_report_location()}")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
