# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json

import pytest

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import evaluate
from h2o_sonar import loggers
from h2o_sonar.evaluators import procedure_evaluator
from h2o_sonar.lib.api import commons
from h2o_sonar.utils import progress
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative
from tests.lib import then_eval


# constants
ProcedureEvaluator = procedure_evaluator.ProcedureEvaluator


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.skip(
    reason="Skipped as it is flaky, slow and the implementation is not finished anyway"
)
@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({"optimum"}),
    reason="Package 'optimum' is not installed",
)
@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({"sentence_transformers"}),
    reason="Package 'sentence-transformers' is not installed",
)
@pytest.mark.parametrize(
    "test_lab_path,evaluator_class,llm_model_name",
    [
        # 3 prompts test (1'30")
        (
            "data/generative/procedure_eval_test_lab_3p.json",
            ProcedureEvaluator,
            "",
        ),
        # DISABLED v tests as they cause CI 2h limit to be exceeded
        # (
        #     #     "data/generative/procedure_eval_test_lab_small.json",
        #     ProcedureEvaluator,
        #     "",
        # ),
        # (
        #     #     "data/generative/procedure_eval_test_lab_problematic.json",
        #     ProcedureEvaluator,
        #     given_generative.LLM_GEMINI_FLASH,
        # ),
        # (
        #     #     "data/generative/procedure_eval_test_lab_small.json",
        #     ProcedureEvaluator,
        #     given_generative.LLM_GEMINI_FLASH,
        # ),
        # (
        #     #     "data/generative/procedure_eval_test_lab_small.json",
        #     ProcedureEvaluator,
        #     "",
        # ),
    ],
)
@pytest.mark.flaky
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_sanity(
    tmpdir,
    h2ogpte_connection_fixture: h2o_sonar_config.ConnectionConfig,
    test_lab_path,
    evaluator_class,
    llm_model_name,
):
    #
    # GIVEN
    #

    test_lab = testing.RagTestLab.load_from_json(
        llm_host_connection=h2ogpte_connection_fixture,
        file_path=test_lab_path,
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )
    the_dataset = test_lab.dataset
    the_models = list(test_lab.evaluated_models.values())

    # progress: 1 stage - evaluate
    progress_callback_name = "[TEST E2E progress callback]"
    progress_callback = progress.LoggingProgressCallbackContext(
        logger=test_lab.logger,
        prefix=progress_callback_name,
        name=progress_callback_name,
    )

    # host connection
    host_connection = test_utils.health.get_h2ogpte()
    print(f"TEST will use h2oGPTe host: {host_connection}")
    host_connection_json = json.dumps(host_connection.to_dict(encrypt=False), indent=2)
    print(f"h2oGPTe host connection:\n\n{host_connection_json}\n")
    h2o_sonar_config.config.add_connection(host_connection)

    #
    # WHEN
    #
    assert evaluate.describe_evaluator(evaluator_class.evaluator_id())

    evaluator_params = {
        evaluator_class.PARAM_H2OGPTE_HOST_CFG_KEY: host_connection.key,
    }
    if llm_model_name:
        evaluator_params[evaluator_class.PARAM_LLM_MODEL_NAME] = llm_model_name

    evaluation = evaluate.run_evaluation(
        dataset=the_dataset,
        models=the_models,
        evaluators=[
            commons.EvaluatorToRun(
                evaluator_id=evaluator_class.evaluator_id(),
                params=evaluator_params,
            )
        ],
        results_location=tmpdir,
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
    result = evaluation.get_evaluator_result(evaluator_class.evaluator_id())
    print(result)
    assert result

    # assert leaderboard JSon representation data and meta
    then_eval.then_leaderboard_json(evaluation, evaluator_class.evaluator_id())


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
