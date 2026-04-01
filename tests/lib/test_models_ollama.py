# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import pathlib

import pytest

from h2o_sonar import evaluate
from h2o_sonar import loggers
from h2o_sonar.evaluators import rag_tokens_presence_evaluator
from h2o_sonar.lib.api import models
from h2o_sonar.lib.integrations import genai
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative


@pytest.mark.skipif(
    not test_utils.health.is_ollama(),
    reason="ollama hosted LLM models are deployed locally",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_model_list():
    #
    # GIVEN
    #

    client = genai.OllamaClient(
        connection=test_utils.health.get_ollama(),
        logger=loggers.SonarPrintLogger(),
    )

    #
    # WHEN
    #
    llm_model_names = client.list_llm_model_names()

    #
    # THEN
    #
    print(f"LLM model names:\n{llm_model_names}")
    assert llm_model_names


@pytest.mark.skipif(
    not test_utils.health.is_ollama(),
    reason="ollama hosted LLM models are deployed locally",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_ask_model(tmp_path):
    #
    # GIVEN
    #
    client = genai.OllamaClient(
        connection=test_utils.health.get_ollama(),
        logger=loggers.SonarPrintLogger(),
    )
    llm_model_name = client.list_llm_model_names()[1]

    #
    # WHEN
    #
    model_cfg = genai.OllamaClient.config_factory()

    print(
        f"\nAsking model: {llm_model_name} with configuration:"
        f"\n{json.dumps(model_cfg, indent=2)}"
        f"\n..."
    )
    # get ANSWERS
    results = client.ask_model(
        prompts=["What are the transformed features of the model?"],
        llm_model_name=llm_model_name,
        **model_cfg,
    )
    (prompt, answer, duration, chunks, cost, _, _) = results[0]

    #
    # THEN
    #
    assert prompt
    assert answer
    assert duration
    assert not chunks
    assert not cost


@pytest.mark.skipif(
    not test_utils.health.is_ollama(),
    reason="ollama hosted LLM models are deployed locally",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_build_lab(tmp_path):
    #
    # GIVEN
    #

    connection = test_utils.health.get_ollama()

    llm_model_names = genai.OllamaClient(connection).list_llm_model_names()

    test_suite = testing.RagTestSuiteConfig.load_from_json(
        test_utils.find_locally("data/generative/eval_llm/bank_teller_test_suite.json")
    )

    # OPTIONAL DESCOPE: faster / smaller test (debugging)
    llm_model_names = llm_model_names[:1]
    test_suite.test_cases = test_suite.test_cases[:2]

    #
    # WHEN
    #

    # test lab (RAG product)
    test_lab = testing.RagTestLab.from_llm_test_suite(
        llm_host_connection=connection,
        llm_test_suite=test_suite,
        llm_model_names=llm_model_names,
        llm_model_type=models.ExplainableModelType.ollama,
        work_dir=tmp_path,
    )
    test_lab.save_as_json(tmp_path / "001_before_testlab.json")
    try:
        # docs sync: does nothing
        test_lab.build()
        test_lab.save_as_json(tmp_path / "002_after_build_testlab.json")

        # test lab: complete dataset w/ ACTUAL data (answers, duration, context)
        test_lab.complete_dataset(
            parallelize=True,
            save_as_you_go=tmp_path / "003_wip_chat_testlab.json",
        )
        # backup fully resolved dataset
        lab_path = test_lab.save_as_json(tmp_path / "DONE_test_lab.json")

        #
        # THEN
        #
        evaluation = evaluate.run_evaluation(
            # dataset w/ prompts, constraints and model keys
            dataset=test_lab.dataset,
            # models to be evaluated / compared to get leaderboard
            models=test_lab.evaluated_models.values(),
            # evaluators
            evaluators=[
                rag_tokens_presence_evaluator.RagStrStrEvaluator().evaluator_id()
            ],
            # where to save the report
            results_location=tmp_path,
        )
        print(
            f"\nExplanations:\n"
            f"  HTML report: file://{evaluation.result.get_html_report_location()}\n"
        )

        assert pathlib.Path(lab_path).exists()
    finally:
        # purge test lab
        test_lab.purge()


def _when_model_cfg() -> dict:
    custom_config = genai.OllamaClient.config_factory()
    print(f"Config prototype:\n{custom_config}")
    return custom_config


@pytest.mark.skipif(
    not test_utils.health.is_ollama(),
    reason="ollama hosted LLM models are deployed locally",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_llm_client_config(tmp_path):
    #
    # GIVEN
    #

    connection = test_utils.health.get_ollama()
    llm_model_names = [
        m
        for m in genai.OllamaClient(connection).list_llm_model_names()
        if "llama" in m.lower()
    ]
    test_suite_path = "data/generative/ci_llm_test_suite.json"
    test_suite = testing.RagTestSuiteConfig.load_from_json(
        test_utils.find_locally(test_suite_path)
    )
    # cut down the test suite to speed up the test
    test_suite.test_cases = test_suite.test_cases[:2]

    #
    # WHEN
    #

    # TEST LAB
    llm_models_cfgs = {}
    for llm_model_name in llm_model_names:
        llm_models_cfgs[llm_model_name] = []

        for temperature in [0.1, 0.7]:
            # make each config slightly different
            model_cfg = _when_model_cfg()
            model_cfg["options"] = {
                "temperature": temperature,
                "num_predict": int(temperature * 1000.0),  # tokens
                "top_k": int(temperature * 100.0),  # non-sense prevention
            }
            llm_models_cfgs[llm_model_name].append(model_cfg)

    test_lab = testing.RagTestLab.from_llm_test_suite(
        llm_host_connection=connection,
        llm_test_suite=test_suite,
        llm_model_type=models.ExplainableModelType.ollama,
        llm_model_names=llm_model_names,
        llm_models_cfgs=llm_models_cfgs,
        llm_host_prompt_cache=None,
    )

    # deploy the test lab configuration to the h2oGPTe server (collections @ corpora)
    test_lab.build()
    # complete dataset w/ actual values from the h2oGPTe server - answer, duration, ...
    test_lab.complete_dataset(
        save_as_you_go=tmp_path / "wip_testlab.json",
        parallelize=testing.TestLab.PARALLEL_RUN,
    )
    test_lab_path = tmp_path / "test_lab.json"
    test_lab.save_as_json(test_lab_path)

    #
    # THEN
    #
    print(f"Test lab (LLM):\nfile://{test_lab_path}")
    print(test_lab)
    # assert the test lab which was created w/ CUSTOM CONFIG
    assert test_lab
    for m in test_lab.evaluated_models.values():
        assert m.model_cfg
        assert m.model_cfg["options"]["temperature"] in [0.1, 0.7]


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_llm_client_config_html_report(tmp_path):
    #
    # GIVEN
    #
    test_lab_path = "data/generative/ci_llm_test_lab_ollama_args.json"
    test_lab = testing.RagTestLab.load_from_json(
        llm_host_connection=test_utils.health.get_h2ogpte(),
        file_path=test_utils.find_locally(test_lab_path),
    )

    #
    # WHEN
    #
    evaluation = evaluate.run_evaluation(
        dataset=test_lab.dataset,
        models=test_lab.evaluated_models.values(),
        evaluators=[rag_tokens_presence_evaluator.RagStrStrEvaluator().evaluator_id()],
        results_location=tmp_path,
    )

    #
    # THEN
    #
    assert evaluation
    assert not evaluation.get_failed_evaluator_ids()
    result = evaluation.get_evaluator_result(
        rag_tokens_presence_evaluator.RagStrStrEvaluator().evaluator_id()
    )
    assert result
    print(
        f"Explanations:\n"
        f"  HTML: file://{evaluation.result.get_html_report_location()}\n"
    )


@pytest.mark.skip("ReStructuredText documentation generation")
@pytest.mark.parametrize(
    "client,comment",
    [
        (genai.H2oGpteRagClient, "h2oGPTe"),
        (genai.H2oGptLlmClient, "h2oGPT"),
        (genai.OpenAiAssistantsRagClient, "OpenAI"),
        (genai.OpenAiLlmClient, "OpenAI"),
        (genai.MsAzureOpenAiLlmClient, "Microsoft Azure"),
        (genai.OllamaClient, "ollama"),
    ],
)
def test_rst_model_cfg(client, comment):
    #
    # GIVEN
    #
    model_cfg = client.config_factory()

    #
    # WHEN
    #
    print(
        f"\n\nThe following **model parameters** can be configured when building "
        f"{comment} :ref:`Test Lab`:\n"
        f"\n"
        ".. code-block:: json\n"
        f"\n"
        f"{json.dumps(model_cfg, indent=2)}"
    )

    #
    # THEN
    #
    assert model_cfg


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
