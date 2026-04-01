# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import pathlib
import shutil

import pytest

from h2o_sonar import evaluate
from h2o_sonar import loggers
from h2o_sonar.evaluators import agent_sanity_check_evaluator
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import evaluations
from h2o_sonar.lib.api import models
from h2o_sonar.lib.integrations import genai
from h2o_sonar.utils import progress
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative
from tests.lib import then_eval


# constants
AgentEvaluator = agent_sanity_check_evaluator.AgentSanityCheckEvaluator


@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_chat_history_visualization(tmp_path: pathlib.Path):
    #
    # GIVEN
    #
    agent_chat_history_path = (
        "data/generative/eval_agent/llm_bank_teller_1p"
        "/test_lab_26350d0c-aeae-4309-8753-3ea06bca278b"
        "/completion_of_m_0841e7bf-fdc5-4dbd-9b29-9a24bd75439a"
        "_tc_9c3a7df3-67df-4819-babb-20636611f077"
        "/chat_session_814feba7-20e8-4d13-97ca-e70e18868f03"
        "/chat_message_0001_16ba1e79-7492-48d7-853d-09af75b4f156"
        "/MSG_META_TYPE_ITEM_agent_chat_history.json"
    )
    try:
        with open(agent_chat_history_path) as f:
            chat_history = json.load(f)
    except FileNotFoundError as ex:
        print(f"Error: The file at '{agent_chat_history_path}' was not found.")
        raise ex

    #
    # WHEN
    #
    visualizer = agent_sanity_check_evaluator.H2ogpteAgentChatHistoryVisualizer(
        chat_history
    )
    (
        activity_json_dict,
        tools_json_dict,
        scripts_json_dict,
        activity_png_path,
        tools_png_path,
        scripts_png_path,
    ) = visualizer.visualize(base_dir=tmp_path)

    #
    # THEN
    #
    if activity_json_dict:
        print(f"Activity diagram JSon:\n{json.dumps(activity_json_dict, indent=2)}")
    if tools_json_dict:
        print(f"Tools stats JSon:\n{json.dumps(tools_json_dict, indent=2)}")
    if scripts_json_dict:
        print(f"Scripts stats JSon:\n{json.dumps(scripts_json_dict, indent=2)}")
    print(f"Agent chat ACTIVITY graph: file://{activity_png_path}")
    print(f"Agent chat TOOLS stats graph: file://{tools_png_path}")
    print(f"Agent chat SCRIPTS stats graph: file://{scripts_png_path}")
    assert activity_png_path.exists()
    assert tools_png_path.exists()
    assert scripts_png_path.exists()


def _then_assert_default_metrics_values(evaluation: evaluations.Evaluation):
    # assert leaderboard JSon representation data and meta
    json_leaderboard_data_dict = then_eval.then_leaderboard_json(
        evaluation, AgentEvaluator.evaluator_id()
    )
    assert json_leaderboard_data_dict, "No JSon leaderboard data"
    assert json_leaderboard_data_dict.get("data"), "No 'data' key in leaderboard data"
    assert json_leaderboard_data_dict["data"].get("auto"), "No 'auto' model in data"
    leaderboard_metrics = json_leaderboard_data_dict["data"]["auto"]
    assert not (
        leaderboard_metrics[AgentEvaluator.METRIC_SANITY] == 1.0
        and leaderboard_metrics[AgentEvaluator.METRIC_TOOL_FAILURES] == 0
        and leaderboard_metrics[AgentEvaluator.METRIC_USED_TOOLS] == 0
        and leaderboard_metrics[AgentEvaluator.METRIC_AGENT_REPLAN] == 0
        and leaderboard_metrics[AgentEvaluator.METRIC_AGENT_STEPS] == 0
        and leaderboard_metrics[AgentEvaluator.METRIC_COST] == 0
        and leaderboard_metrics[AgentEvaluator.METRIC_DURATION] == 0
        and leaderboard_metrics[AgentEvaluator.METRIC_DATA_SIZE] == 0
        and leaderboard_metrics[AgentEvaluator.METRIC_FILE_COUNT] == 0
    ), "Sanity check metrics were NOT calculated - DEFAULT values detected"


@pytest.mark.skipif(
    not test_utils.health.is_h2ogpte(),
    reason="Required h2oGPTe server is not available (H2O Sonar health check).",
)
@pytest.mark.parametrize(
    "is_rag,test_lab_base_path",
    [
        (
            False,
            pathlib.Path("data/generative/eval_agent/llm_bank_teller_2p"),
        ),
        (
            True,
            pathlib.Path("data/generative/eval_agent/rag_fbi_agent_1p"),
        ),
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_build_lab(
    tmp_path,
    is_rag: bool,
    test_lab_base_path: pathlib.Path,
):
    #
    # GIVEN
    #
    h2ogpte_connection = test_utils.health.get_h2ogpte()
    llm_or_rag_str = "RAG" if is_rag else "LLM"
    llm_model_names = ["auto"]

    # dirs alignment:
    # - results_location:
    #   - where H2O Sonar stores the evaluation/lab data of ALL users
    #   - the structure <results_location>/<user>/<evaluation|completion>/**
    #   - DEFAULT value: CWD
    # - user_dir:
    #   - where H2O Sonar stores data of the particular USER
    #   - the structure <user_dir>/<evaluation|completion>/**
    #   - DEFAULT value: CWD/<commons.DEFAULT_USER>
    # - docs_cache_dir:
    #   - where H2O Sonar caches (RAG corpus) docs
    #   - caching between runs is desired > typically set to a persistent location
    #   - DEFAULT value: CWD/<user>/cache/docs/**
    #
    # typical API dirs use:
    # - lab completion
    #   - RagTestLab(results_location=..., docs_cache_dir=...)
    # - evaluation
    #   - evaluate.run_evaluation(results_location=...)
    #
    results_location = tmp_path
    docs_cache_dir = given_generative.DIR_TEST_RAG_DOCS_CACHE  # (RAG only)

    llm_or_rag_models_cfgs = {
        llm_model_names[0]: [
            {
                genai.H2oGpteRagClient.CFG_LLM_ARGS: {
                    genai.H2oGpteRagClient.CFG_USE_AGENT: True,
                }
            }
        ]
    }
    print(f"Using RAG models cfgs:\n{json.dumps(llm_or_rag_models_cfgs, indent=2)}")

    test_suite_path = test_lab_base_path / "test_suite.json"
    test_suite = testing.RagTestSuiteConfig.load_from_json(file_path=test_suite_path)
    print(
        f"\nRunning test lab completion:"
        f"\n  host: {h2ogpte_connection.server_url}"
        f"\n  model: {llm_model_names[0]}"
        f"\n  test suite: {test_suite_path}"
        f"\n  {len(test_suite.test_cfgs)} tests"
        f"\n  {len(test_suite.test_cases)} test cases"
    )

    #
    # WHEN
    #
    if is_rag:
        test_lab = testing.RagTestLab.from_rag_test_suite(
            rag_connection=h2ogpte_connection,
            rag_test_suite=test_suite,
            rag_model_type=models.ExplainableModelType.h2ogpte,
            rag_models_cfgs=llm_or_rag_models_cfgs,
            llm_model_names=llm_model_names,
            docs_cache_dir=docs_cache_dir,
            results_location=results_location,
        )
    else:
        test_lab = testing.RagTestLab.from_llm_test_suite(
            llm_host_connection=h2ogpte_connection,
            llm_test_suite=test_suite,
            llm_model_type=models.ExplainableModelType.h2ogpte_llm,
            llm_models_cfgs=llm_or_rag_models_cfgs,
            llm_model_names=llm_model_names,
            results_location=results_location,
        )

    # deploy the test lab configuration to the h2oGPTe server
    test_lab.build()
    # complete dataset w/ actual values from the h2oGPTe server (answer, duration, ...)
    test_lab.complete_dataset(
        save_as_you_go=tmp_path / "wip_testlab.json",
        parallelize=testing.TestLab.SEQUENTIAL_RUN,
    )
    test_lab_path = tmp_path / f"COMPLETED_{llm_or_rag_str}_test_lab.json"
    test_lab.save_as_json(test_lab_path)
    the_dataset = test_lab.dataset
    the_models = list(test_lab.evaluated_models.values())

    progress_callback_name = "[TEST E2E progress callback]"
    progress_callback = progress.LoggingProgressCallbackContext(
        logger=test_lab.logger,
        prefix=progress_callback_name,
        name=progress_callback_name,
    )

    #
    # WHEN
    #
    assert evaluate.describe_evaluator(AgentEvaluator.evaluator_id())

    evaluation = evaluate.run_evaluation(
        dataset=the_dataset,
        models=the_models,
        evaluators=[
            commons.EvaluatorToRun(
                evaluator_id=AgentEvaluator.evaluator_id(),
                params={},
            )
        ],
        results_location=results_location,
        log_level=loggers.DEBUG,
        progress_callback=progress_callback,
    )

    #
    # THEN
    #
    print(f"Evaluation:\n{evaluation}")
    print(f"HTML:\nfile://{evaluation.result.get_html_report_location()}")
    assert evaluation
    assert not evaluation.is_explainer_failed()

    # assert result
    result = evaluation.get_evaluator_result(AgentEvaluator.evaluator_id())
    print(result)
    assert result

    # assert leaderboard JSon representation data and meta
    _then_assert_default_metrics_values(evaluation)

    # at the end print the HTML report location for manual inspection
    print(f"HTML:\nfile://{evaluation.result.get_html_report_location()}")


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.parametrize(
    "test_lab_base_path,test_lab_key",
    [
        (
            pathlib.Path("data/generative/eval_agent/llm_bank_teller_1p"),
            "26350d0c-aeae-4309-8753-3ea06bca278b",
        ),
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_lab_load(
    tmp_path,
    test_lab_base_path: pathlib.Path,
    test_lab_key: str,
):
    #
    # GIVEN
    #
    h2ogpte_connection = test_utils.health.get_h2ogpte()

    test_lab = testing.RagTestLab.load_from_json(
        llm_host_connection=h2ogpte_connection,
        file_path=test_lab_base_path / "test_lab.json",
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )
    the_dataset = test_lab.dataset
    the_models = list(test_lab.evaluated_models.values())
    # test lab artifacts (static test data)
    user_dir = tmp_path / commons.DEFAULT_USER
    user_dir.mkdir(exist_ok=True, parents=True)
    test_lab_dir_name = f"test_lab_{test_lab_key}"
    shutil.copytree(
        src=f"{test_lab_base_path}/{test_lab_dir_name}",
        dst=f"{user_dir}/{test_lab_dir_name}",
    )

    progress_callback_name = "[TEST E2E progress callback]"
    progress_callback = progress.LoggingProgressCallbackContext(
        logger=test_lab.logger,
        prefix=progress_callback_name,
        name=progress_callback_name,
    )

    #
    # WHEN
    #

    evaluation = evaluate.run_evaluation(
        dataset=the_dataset,
        models=the_models,
        evaluators=[
            commons.EvaluatorToRun(
                evaluator_id=AgentEvaluator.evaluator_id(),
                params={},
            )
        ],
        results_location=tmp_path,
        log_level=loggers.DEBUG,
        progress_callback=progress_callback,
    )

    #
    # THEN
    #
    print(f"Evaluation:\n{evaluation}")
    print(f"HTML:\nfile://{evaluation.result.get_html_report_location()}")
    assert evaluation
    assert not evaluation.is_explainer_failed()

    # assert result
    result = evaluation.get_evaluator_result(AgentEvaluator.evaluator_id())
    print(result)
    assert result

    _then_assert_default_metrics_values(evaluation)

    # at the end print the HTML report location for manual inspection
    print(f"HTML:\nfile://{evaluation.result.get_html_report_location()}")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
