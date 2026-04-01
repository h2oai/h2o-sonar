# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json

import pytest

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import evaluate
from h2o_sonar.evaluators import json_schema_evaluator
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import persistences
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative
from tests.lib import then_eval


# aliases
JSonEvaluator = json_schema_evaluator.JSONSchemaEvaluator()


@pytest.mark.skipif(
    not test_utils.health.is_h2ogpte(),
    reason="Test requires h2oGPTe server whose connection is not available",
)
@pytest.mark.parametrize(
    "test_lab_path,json_schema",
    [
        # validate WITH JSon Schema
        (
            "data/generative/eval_llm/json_schema_test_lab.json",
            {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "object",
                        "properties": {
                            "details": {
                                "type": "object",
                                "properties": {
                                    "price": {"type": "number", "const": 314},
                                    "code": {"type": "string", "const": "art42"},
                                },
                                "required": ["price", "code"],
                            }
                        },
                        "required": ["details"],
                    }
                },
                "required": ["product"],
            },
        ),
        # validate WITHOUT JSon Schema - just parseability of actual answers JSons
        (
            "data/generative/eval_llm/json_schema_test_lab.json",
            {},
        ),
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_eval_lab_load(
    tmp_path,
    h2ogpte_connection_fixture: h2o_sonar_config.ConnectionConfig,
    test_lab_path: str,
    json_schema: dict,
):
    #
    # GIVEN
    #

    json_schema_str = json.dumps(json_schema)
    # VALID example:
    # {
    #     "product": {
    #         "details": {
    #             "price": 314,
    #             "code": "art42"
    #         }
    #     }
    # }
    # VALID example:
    # { "product": { "details": { "price": 314, "code": "art42" }}}
    # INVALID example is whatever else

    # test lab (load cfg w/ actual values - build/chat not needed)
    test_lab = testing.RagTestLab.load_from_json(
        llm_host_connection=h2ogpte_connection_fixture,
        file_path=test_lab_path,
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )

    #
    # WHEN
    #

    evaluator_id = JSonEvaluator.evaluator_id()

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
                    JSonEvaluator.PARAM_JSON_SCHEMA: json_schema_str,
                    JSonEvaluator.PARAM_METRIC_THRESHOLD: 0.9,
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
    # load Markdown leaderboard
    ep = persistences.ExplainerPersistence(
        data_dir=evaluation.result.results_location,
        mli_key=evaluation.key,
        username=commons.DEFAULT_USER,
        explainer_id=evaluator_id,
        explainer_job_key=next(iter(evaluation.result.explainers)),
    )
    md_path = ep.get_explanation_file_path(
        explanation_type=e10s.LlmBoolLeaderboardExplanation.explanation_type(),
        explanation_format=f5s.MarkdownFormat.mime,
    )
    # result: leaderboard
    result = evaluation.get_explainer_result(evaluator_id)
    # result: data
    data = result.data()
    print(f"Data:\n{data}")
    assert data
    # result: summary
    summary = result.summary()
    print(f"Summary:\n{summary}")
    assert summary
    # result: plot / log / zip
    result.plot(file_path=tmp_path / "my_plot.png")
    result.log(path=tmp_path / "my_log.txt")
    result.zip(file_path=tmp_path / "my_result.zip")

    #
    # THEN get_evaluation() w/ problems
    #
    loaded_evaluation = _then_evaluation_load_from_json(
        evaluation_key=evaluation.key,
        results_location=evaluation.result.results_location,
    )
    # check problems deserialization
    print(f"Problems:\n{len(loaded_evaluation.result.problems)}")
    if loaded_evaluation.result.problems:
        assert loaded_evaluation.result.problems
        for p in loaded_evaluation.result.problems:
            print(p.to_dict())

    # assert leaderboard JSon representation data and meta
    then_eval.then_leaderboard_json(evaluation, evaluator_id)

    print(
        f"Explanations:\n"
        f"  HTML: file://{evaluation.result.get_html_report_location()}\n"
        f"  MD  : file://{md_path}\n"
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


@pytest.mark.parametrize(
    "s,json_schema",
    [
        # valid JSon with correct schema
        (
            json.dumps(
                {
                    "name": "Alice",
                    "age": 30,
                }
            ),
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer", "minimum": 0},
                },
                "required": ["name", "age"],
            },
        ),
        # valid JSon w/ constants
        (
            json.dumps({"product": {"details": {"price": 314, "code": "art42"}}}),
            {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "object",
                        "properties": {
                            "details": {
                                "type": "object",
                                "properties": {
                                    "price": {"type": "number", "const": 314},
                                    "code": {"type": "string", "const": "art42"},
                                },
                                "required": ["price", "code"],
                            }
                        },
                        "required": ["details"],
                    }
                },
                "required": ["product"],
            },
        ),
        # validate whether leading/trailing spaces are ignored when validating JSon
        (
            "   "
            + json.dumps(
                {
                    "name": "Alice",
                    "age": 30,
                }
            )
            + "   ",
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer", "minimum": 0},
                },
                "required": ["name", "age"],
            },
        ),
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_json_schema_evaluation(s: str, json_schema: dict):
    # GIVEN
    # WHEN
    (is_valid, err_msg) = JSonEvaluator.validate_str_against_json_schema(
        s=s,
        json_schema=json_schema,
    )

    # THEN
    assert is_valid
    assert not err_msg


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
