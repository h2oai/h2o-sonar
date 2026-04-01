# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json

import pytest

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import evaluate
from h2o_sonar import loggers
from h2o_sonar.evaluators import agentic_fact_check_evaluator
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import persistences
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative
from tests.lib import then_eval


@pytest.mark.skip("Test requires h2oGPTe server agents which are expensive to run")
@pytest.mark.skipif(
    not test_utils.health.is_h2ogpte(),
    reason="Test requires h2oGPTe server whose connection is not available",
)
@pytest.mark.parametrize(
    "test_lab_path,collection_id,llm_model_name,evaluator_cls",
    [
        # # pre-created collection ID + LLM model name
        # (
        #     "data/generative/eval_llm/fact_check_test_lab.json",
        #     "94984f13-98e9-4673-9bc0-27e5deb268f0",
        #     given_generative.LLM_GEMINI_FLASH,
        #     agentic_fact_check_evaluator.FactCheckAgenticEvaluator,
        # ),
        # let the evaluator create / lookup the collection ID + LLM model name
        (
            "data/generative/eval_llm/fact_check_test_lab.json",
            "",
            "",
            agentic_fact_check_evaluator.FactCheckAgenticEvaluator,
        ),
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_evaluator_lab_load(
    tmp_path,
    h2ogpte_connection_fixture: h2o_sonar_config.ConnectionConfig,
    collection_id: str,
    llm_model_name: str,
    test_lab_path: str,
    evaluator_cls,
):
    #
    # GIVEN
    #
    h2ogpte_connection = h2ogpte_connection_fixture

    print(f"Evaluator description: {evaluator_cls._description}")

    test_lab = testing.RagTestLab.load_from_json(
        llm_host_connection=h2ogpte_connection,
        file_path=test_lab_path,
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )
    # keep only 3 LLM models to speed up the test
    test_lab = test_lab.trim(3)

    # agentic host connection
    agentic_host_connection = test_utils.health.get_h2ogpte()
    print(f"TEST will use agentic host: {agentic_host_connection}")
    agentic_host_connection_json = json.dumps(
        agentic_host_connection.to_dict(encrypt=False), indent=2
    )
    print(f"h2oGPTe agent host connection:\n\n{agentic_host_connection_json}\n")
    h2o_sonar_config.config.add_connection(agentic_host_connection)

    # evaluator parameters
    evaluator_params = {
        evaluator_cls.PARAM_AGENT_HOST_CFG_KEY: agentic_host_connection.key,
        evaluator_cls.PARAM_MAX_DATASET_ROWS: 3,
    }
    if collection_id:
        evaluator_params[evaluator_cls.PARAM_AGENT_COLLECTION_ID] = collection_id
    if llm_model_name:
        evaluator_params[evaluator_cls.PARAM_LLM_MODEL_NAME] = llm_model_name

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
            commons.EvaluatorToRun(
                evaluator_id=evaluator_cls.evaluator_id(), params=evaluator_params
            )
        ],
        # where to save the report
        results_location=tmp_path,
        # log level
        log_level=loggers.DEBUG,
    )

    #
    # THEN
    #

    print(f"{evaluation}")
    assert not evaluation.get_failed_evaluator_ids()

    # load Markdown leaderboard
    ep = persistences.ExplainerPersistence(
        data_dir=evaluation.result.results_location,
        mli_key=evaluation.key,
        username=commons.DEFAULT_USER,
        explainer_id=evaluator_cls.evaluator_id(),
        explainer_job_key=next(iter(evaluation.result.explainers)),
    )
    md_path = ep.get_explanation_file_path(
        explanation_type=e10s.LlmHeatmapLeaderboardExplanation.explanation_type(),
        explanation_format=f5s.MarkdownFormat.mime,
    )
    # result: leaderboard
    result = evaluation.get_explainer_result(evaluator_cls.evaluator_id())
    # result: data
    data = result.data()
    print(f"Data:\n{data}")
    assert data
    # result: summary
    summary = result.summary()
    print(f"Summary:\n{summary}")
    assert summary
    # result: plot / log / zip
    result.log(path=tmp_path / "my_log.txt")
    result.zip(file_path=tmp_path / "my_result.zip")

    print(
        f"Explanations:\n"
        f"  HTML: file://{evaluation.result.get_html_report_location()}\n"
        f"  MD  : file://{md_path}\n"
    )

    #
    # THEN get_evaluation()
    #
    _then_evaluation_load_from_json(
        evaluation_key=evaluation.key,
        results_location=evaluation.result.results_location,
    )

    # assert leaderboard JSon
    then_eval.then_leaderboard_json(
        evaluation=evaluation,
        evaluator_id=evaluator_cls.evaluator_id(),
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


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
