import json
import numbers
import os

import pytest

import h2o_sonar.lib.container.explainer_container as explainer_container
from h2o_sonar import evaluate
from h2o_sonar.lib.api import evaluators
from h2o_sonar.lib.api.evaluators import KEYWORD_METHOD_TYPE_DETERMINISTIC
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative


def pytest_generate_tests(metafunc):
    if "deterministic_evaluator" in metafunc.fixturenames:
        ec = explainer_container.LocalExplainerContainer()
        ec.setup()
        deterministic_evaluators = ec.list_explainers(
            keywords=[evaluators.KEYWORD_LLM, KEYWORD_METHOD_TYPE_DETERMINISTIC]
        )
        metafunc.parametrize("deterministic_evaluator", deterministic_evaluators)


def _extract_metrics(base_path: str, explanation_type: str) -> dict | None:
    if explanation_type == "global_llm_eval_results":
        filename = f"{base_path}/{explanation_type}/application_json/explanation.json"
        if not os.path.exists(filename):
            return None
        with open(filename) as f:
            eval_result = json.load(f)
        metrics = [mm["key"] for mm in eval_result["evaluator"]["metrics_meta"]]
        return [{k: res[k] for k in metrics} for res in eval_result["results"]]
    elif explanation_type == "global_llm_heatmap_leaderboard":
        filename = f"{base_path}/{explanation_type}/application_json/explanation.json"
        if not os.path.exists(filename):
            return None
        with open(filename) as f:
            explanation = json.load(f)
        with open(
            f"{base_path}/{explanation_type}/application_json/"
            f"{explanation['files']['ALL_METRICS']}"
        ) as f:
            leaderboard = json.load(f)
        return leaderboard["data"]
    else:
        raise AssertionError(base_path)


def _approximately_equal(actual, expected, message):
    if isinstance(actual, numbers.Number) and isinstance(expected, numbers.Number):
        assert (
            abs(actual - expected) / max((abs(actual) + abs(expected)) / 2, 1e-4) < 1e-4
        ), message
    elif isinstance(actual, list) and isinstance(expected, list):
        assert len(actual) == len(expected), message
        for i in range(len(actual)):
            _approximately_equal(
                actual[i], expected[i], f"{message} list {i}/{len(actual)}"
            )
    elif isinstance(actual, dict) and isinstance(expected, dict):
        assert len(actual) == len(expected), message
        assert frozenset(actual.keys()) == frozenset(expected.keys()), message
        for k in actual.keys():
            _approximately_equal(actual[k], expected[k], f"{message} dict[{k}]")
    else:
        raise AssertionError(
            f"Unexpected classes {type(actual)}, {type(expected)}! {message=}; "
            f"{actual=}; {expected=}"
        )


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.parametrize(
    "test_lab",
    [
        "data/generative/kims_summarization_test_lab_tiny.json",
        "data/generative/kaggle_llm_science_exam_test_lab_2x_small_3.json",
        "data/generative/sr1107_test_lab_3m.json",
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_deterministic_evaluators(test_lab, tmp_path, deterministic_evaluator, request):
    print(deterministic_evaluator)
    rag_dataset = testing.RagTestLab.load_from_json(
        llm_host_connection=test_utils.health.get_h2ogpt(),
        file_path=test_lab,
    )
    llm_models = rag_dataset.evaluated_models.values()

    # KEYWORD_RQ_J = "requires_llm_judge"
    # KEYWORD_RQ_P = "requires_prompts"
    # KEYWORD_RQ_EA = "requires_expected_answer"
    # KEYWORD_RQ_RC = "requires_retrieved_context"
    # KEYWORD_RQ_AA = "requires_actual_answer"
    # KEYWORD_RQ_C = "requires_constraints"
    assert evaluators.KEYWORD_RQ_J not in deterministic_evaluator.keywords, (
        "Judges aren't deterministic!"
    )

    if evaluators.KEYWORD_RQ_P in deterministic_evaluator.keywords:
        if len(rag_dataset.dataset.inputs[0].to_dict()["input"]) == 0:
            pytest.skip(
                f"Prompt for {deterministic_evaluator.id} required but missing "
                f"in {test_lab}."
            )
    if evaluators.KEYWORD_RQ_EA in deterministic_evaluator.keywords:
        if len(rag_dataset.dataset.inputs[0].to_dict()["expected_output"]) == 0:
            pytest.skip(
                f"Expected Output for {deterministic_evaluator.id} required but "
                f"missing in {test_lab}."
            )
    if evaluators.KEYWORD_RQ_AA in deterministic_evaluator.keywords:
        if len(rag_dataset.dataset.inputs[0].to_dict()["actual_output"]) == 0:
            pytest.skip(
                f"Actual Output for {deterministic_evaluator.id} required but "
                f"missing in {test_lab}."
            )
    if evaluators.KEYWORD_RQ_RC in deterministic_evaluator.keywords:
        if len(rag_dataset.dataset.inputs[0].to_dict()["context"]) == 0:
            pytest.skip(
                f"Context for {deterministic_evaluator.id} required but "
                f"missing in {test_lab}."
            )
    if evaluators.KEYWORD_RQ_C in deterministic_evaluator.keywords:
        if len(rag_dataset.dataset.inputs[0].to_dict()["output_constraints"]) == 0:
            pytest.skip(
                f"Constraints for {deterministic_evaluator.id} required but "
                f"missing in {test_lab}."
            )

    evaluation = evaluate.run_evaluation(
        dataset=rag_dataset.dataset,
        models=llm_models,
        evaluators=[deterministic_evaluator.id],
        results_location=tmp_path,
    )

    evaluator_jobs = evaluation.result.get_evaluator_jobs()
    if len(evaluator_jobs) == 0:
        return
    assert len(evaluator_jobs) == 1
    evaluator_job = evaluator_jobs[0]
    for expl_type in [
        "global_llm_eval_results",
        "global_llm_heatmap_leaderboard",
    ]:
        key = (
            f"test_deterministic_evaluators::"
            f"{deterministic_evaluator.id}::{test_lab}::{expl_type}"
        )
        results = _extract_metrics(evaluator_job.job_location, expl_type)
        if results is None:
            continue
        cached_data = request.config.cache.get(key, None)
        if cached_data is not None:
            _approximately_equal(
                results,
                cached_data,
                f"Failure {deterministic_evaluator.id}, {test_lab}, {expl_type}!",
            )
        else:
            request.config.cache.set(key, results)
            pytest.skip("No cached data. Caching...")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
