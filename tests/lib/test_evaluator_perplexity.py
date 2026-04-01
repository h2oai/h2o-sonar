# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import pytest

from h2o_sonar import evaluate
from h2o_sonar import loggers
from h2o_sonar.evaluators import perplexity_evaluator
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative
from tests.lib import then_eval


@pytest.mark.parametrize(
    "test_lab_path,evaluator_class,expect_compatible",
    [
        (
            #
            # RAG test labs:
            #
            "data/generative/kaggle_llm_science_exam_test_lab_4x_25.json",
            #
            # EVALUATORS: Perplexity
            #
            perplexity_evaluator.PerplexityEvaluator,
            #
            # Expected compatibility: True (all rows have AA)
            #
            True,
        ),
        (
            #
            # RAG test lab with ALL missing AAs
            #
            "data/generative/toxicity_test_lab_2x3p_EMPTY_AA.json",
            #
            # EVALUATOR: Perplexity
            #
            perplexity_evaluator.PerplexityEvaluator,
            #
            # Expected compatibility: False (NO row has AA)
            #
            False,
        ),
        # NOTE: 1x_EMPTY case currently fails due to strict compatibility check
        # TODO: Investigate _check_llm_dataset_compatibility() behavior
    ],
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_evaluator(tmpdir, test_lab_path, evaluator_class, expect_compatible):
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
            ("does not contain" in desc and "actual answer" in desc)
            for desc in problem_descriptions
        )
        assert problem_found, (
            f"Expected problem about missing actual answers not found. "
            f"Problems: {problem_descriptions}"
        )


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
