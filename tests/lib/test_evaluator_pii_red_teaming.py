# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import pytest

from h2o_sonar import evaluate
from h2o_sonar.evaluators import pii_leakage_evaluator
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import models
from h2o_sonar.lib.integrations import genai
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative
from tests.lib import then_eval


# aliases
PiiEvaluator = pii_leakage_evaluator.PiiLeakageEvaluator()


@pytest.mark.skip(
    reason=(
        "Test which builds fully resolved LLM benchmark test lab which can be "
        "subsequently loaded to perform fast evaluators testing"
    )
)
@pytest.mark.parametrize(
    "test_suite_path",
    [
        # ### suites
        "data/generative/eval_llm/red_teaming_test_suite.json",
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_build_lab(tmp_path, h2ogpte_connection_fixture, test_suite_path):
    """Build Test Lab for h2oGPTe hosted LLM models."""
    #
    # GIVEN
    #
    h2ogpte_connection = h2ogpte_connection_fixture

    # LLM models to evaluate
    llm_model_names = genai.H2oGpteRagClient(h2ogpte_connection).list_llm_model_names()
    # models (NOT) working
    broken_llm_model_names = []
    for broken_llm_model_name in broken_llm_model_names:
        if broken_llm_model_name in llm_model_names:
            llm_model_names.remove(broken_llm_model_name)
    llm_model_names = [
        "h2oai/h2o-danube3-4b-chat",
        "h2oai/h2o-danube2-1.8b-chat",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "mistral-medium",
    ]

    # test SUITE
    llm_test_suite = testing.RagTestSuiteConfig.load_from_json(
        test_utils.find_locally(test_suite_path)
    )

    # optional SMALLER TEST (for debugging)
    # llm_model_names = llm_model_names[:3]
    # llm_test_suite.test_cases = llm_test_suite.test_cases[:5]

    test_lab = testing.RagTestLab.from_llm_test_suite(
        llm_host_connection=h2ogpte_connection,
        llm_test_suite=llm_test_suite,
        llm_model_type=models.ExplainableModelType.h2ogpte_llm,
        llm_model_names=llm_model_names,
    )
    # deploy the test lab configuration to the h2oGPTe server
    test_lab.build()

    # complete dataset w/ actual values from the h2oGPTe server (answer, duration, ...)
    test_lab.complete_dataset(
        save_as_you_go=tmp_path / "wip_testlab.json",
        parallelize=testing.TestLab.PARALLEL_RUN,
    )
    test_lab.save_as_json(tmp_path / "test_lab.json")

    #
    # THEN
    #
    print(test_lab)


@pytest.mark.skipif(
    not test_utils.health.is_h2ogpte(),
    reason="Test requires h2oGPTe server whose connection is not available",
)
@pytest.mark.parametrize(
    "test_lab_path",
    [
        "data/generative/eval_llm/red_teaming_test_lab_broken_all.json",
        "data/generative/eval_llm/red_teaming_test_lab_broken_some.json",
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_eval_lab_load(tmp_path, h2ogpte_connection_fixture, test_lab_path):
    """h2oGPTe benchmark test which loads RESOLVED lab from the filesystem:

    - h2oGPTe server is not needed (dataset w/ actual data is loaded from filesystem)

    """
    #
    # GIVEN
    #
    h2ogpte_connection = h2ogpte_connection_fixture

    # test lab (load cfg w/ actual values - build/chat not needed)
    test_lab = testing.RagTestLab.load_from_json(
        llm_host_connection=h2ogpte_connection,
        file_path=test_lab_path,
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )

    #
    # WHEN
    #

    evaluator_id = PiiEvaluator.evaluator_id()

    evaluation = evaluate.run_evaluation(
        # dataset w/ prompts, constraints and model keys
        dataset=test_lab.dataset,
        # models to be evaluated / compared to get leaderboard
        models=list(test_lab.evaluated_models.values()),
        # evaluators
        evaluators=[
            commons.EvaluatorToRun(
                evaluator_id=evaluator_id,
                params={
                    PiiEvaluator.PARAM_METRIC_THRESHOLD: 0.9,
                },
            )
        ],
        # where to save the report
        results_location=tmp_path,
    )

    #
    # THEN
    #

    print(f"{evaluation}")
    assert not evaluation.get_failed_evaluator_ids()

    # result: leaderboard
    assert evaluation.result
    # assert that broken test lab has problem for every failed row
    if "broken_all" in test_lab_path:
        assert len(evaluation.result.problems) == 1
        assert "cannot evaluate" in evaluation.result.problems[0].description
    else:
        evaluator_result = evaluation.get_explainer_result(evaluator_id)
        assert evaluator_result

        assert len(evaluation.result.problems) > 0

        #
        # THEN get_evaluation() w/ problems
        #
        loaded_evaluation = _then_evaluation_load_from_json(
            evaluation_key=evaluation.key,
            results_location=evaluation.result.results_location,
        )
        # check problems deserialization
        print(f"Problems:\n{len(loaded_evaluation.result.problems)}")
        # TODO add assert back once PII will report problems again
        if loaded_evaluation.result.problems:
            assert loaded_evaluation.result.problems
            for p in loaded_evaluation.result.problems:
                print(p.to_dict())

        # assert leaderboard JSon representation data and meta
        then_eval.then_leaderboard_json(evaluation, evaluator_id)

    print(
        f"Explanations:\n  HTML: file://{evaluation.result.get_html_report_location()}"
    )


def _then_evaluation_load_from_json(evaluation_key: str, results_location: str):
    loaded_evaluation = evaluate.get_evaluation(
        evaluation_key=evaluation_key,
        results_location=results_location,
    )
    # print(f"Loaded evaluation:\n{loaded_evaluation}")
    assert loaded_evaluation
    # print(loaded_evaluation.key)
    assert loaded_evaluation.key
    # print(loaded_evaluation.created)
    assert loaded_evaluation.created
    # print(loaded_evaluation.status)
    # print(loaded_evaluation.progress)
    assert loaded_evaluation.progress
    # print(loaded_evaluation.result)
    assert loaded_evaluation.result
    # print(loaded_evaluation.result.results_location)
    assert loaded_evaluation.result.results_location
    # print(loaded_evaluation.result.interpretation_location)
    assert loaded_evaluation.result.interpretation_location

    return loaded_evaluation


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
