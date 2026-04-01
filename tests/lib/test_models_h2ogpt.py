# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import os
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
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.skip(reason="Not a test - h2oGPT servers probing only")
@pytest.mark.parametrize(
    "base_h2ogpt_url",
    [
        # test commons servers probing
        given_generative.H2OGPT_PUBLIC.server_url,  # primary test server
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_sanity_check(base_h2ogpt_url):
    """This test is not a test, but a probe to check the availability of the h2oGPT
    servers.

    """
    from openai import OpenAI

    #
    # GIVEN
    #

    api_key = os.getenv(given_generative.KEY_H2OGPT_API_KEY)

    #
    # WHEN
    #

    print(f"\nProbing h2oGPT server:\n  {base_h2ogpt_url}")
    openai_client = OpenAI(
        base_url=base_h2ogpt_url,  # OK: "https://api.gpt.h2o.ai/v1",
        api_key=api_key,
    )

    # list models
    llm_model_names = openai_client.models.list()
    print(f"Available LLM models ({llm_model_names}):")
    for m in llm_model_names:
        print(f"  {m.id}\n    {m}")

    # chat
    llm_model_name = [m.id for m in llm_model_names][0]
    print(f"\nChatting with the model {llm_model_name}...")
    messages = [{"role": "user", "content": "Who are you?"}]
    stream = False
    client_kwargs = dict(
        model=llm_model_name,
        max_tokens=200,
        stream=stream,
        messages=messages,
    )
    client = openai_client.chat.completions

    responses = client.create(**client_kwargs)
    text = responses.choices[0].message.content

    #
    # THEN
    #

    print(f"\n  base_url: {base_h2ogpt_url}\n  {text}")


@pytest.mark.skip(reason="h2oGPT servers retired")
@pytest.mark.skipif(
    not test_utils.health.is_h2ogpt(),
    reason="h2oGPT API KEY not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_h2ogpt_client(tmp_path):
    #
    # GIVEN
    #
    h2ogpt_connection = test_utils.health.get_h2ogpt()

    print(f"\n\nConnecting to {h2ogpt_connection.server_url}")
    client = genai.H2oGptLlmClient(
        connection=h2ogpt_connection,
        logger=loggers.SonarPrintLogger(),
    )

    llm_model_names = client.list_llm_model_names()
    print("Available LLM models:")
    for m in llm_model_names:
        print(f"  {m}")

    llm_model_name = llm_model_names[0]

    #
    # WHEN
    #

    # get ANSWERS
    results = client.ask_model(
        prompts=["What are the transformed features of the model?"],
        llm_model_name=llm_model_name,
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


@pytest.mark.skip(reason="h2oGPT servers retired")
@pytest.mark.skipif(
    not test_utils.health.is_h2ogpt(),
    reason="h2oGPT API KEY not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_build_lab(tmp_path):
    #
    # GIVEN
    #
    h2ogpt_connection = test_utils.health.get_h2ogpt()

    # llm_model_names = genai.OpenAiAssistantsRagClient.BASE_LLM_MODELS
    llm_model_names = test_utils.health.get_h2ogpt_models()[0:1]

    test_suite = testing.RagTestSuiteConfig.load_from_json(
        test_utils.find_locally("data/generative/eval_llm/bank_teller_test_suite.json")
    )

    # OPTIONAL DESCOPE: faster / smaller test (debugging)
    # llm_model_names = llm_model_names[:1]
    test_suite.test_cases = test_suite.test_cases[:2]

    #
    # WHEN
    #

    # test lab (RAG product)
    test_lab = testing.RagTestLab.from_llm_test_suite(
        llm_host_connection=h2ogpt_connection,
        llm_test_suite=test_suite,
        llm_model_names=llm_model_names,
        llm_model_type=models.ExplainableModelType.h2ogpt,
        work_dir=tmp_path,
    )
    test_lab.save_as_json(tmp_path / "001_before_testlab.json")
    try:
        # test lab: DEPLOY to OpenAI
        # (docs sync: S3 > filesystem cache > OpenAI Assistant per Test)
        test_lab.build()
        test_lab.save_as_json(tmp_path / "002_after_build_testlab.json")

        # test lab: complete dataset w/ ACTUAL data (answers, duration, context)
        test_lab.complete_dataset(
            complete_context=10,
            parallelize=3,
            save_as_you_go=tmp_path / "003_wip_chat_testlab.json",
        )
        # backup fully resolved dataset
        lab_path = test_lab.save_as_json(tmp_path / "DONE_test_lab.json")

        #
        # THEN
        #

        assert pathlib.Path(lab_path).exists()
    finally:
        # purge test lab
        test_lab.purge()


def _when_model_cfg() -> dict:
    custom_config = genai.H2oGptLlmClient.config_factory()
    print(f"Config prototype:\n{custom_config}")
    return custom_config


@pytest.mark.skip(reason="h2oGPT servers retired")
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_llm_client_config(tmp_path):
    #
    # GIVEN
    #

    connection = test_utils.health.get_h2ogpt()
    llm_model_names = genai.H2oGptLlmClient(connection).list_llm_model_names()[:2]
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
            model_cfg["messages"] = [
                {
                    "role": "user",
                    "content": "Act as a crazy LLM model under the validation test.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "Sure thing! I will return as crazy responses as possible!"
                    ),
                },
            ]
            model_cfg["temperature"] = temperature
            model_cfg["max_tokens"] = int(100.0 * temperature)
            llm_models_cfgs[llm_model_name].append(model_cfg)

    test_lab = testing.RagTestLab.from_llm_test_suite(
        llm_host_connection=connection,
        llm_test_suite=test_suite,
        llm_model_type=models.ExplainableModelType.h2ogpte_llm,
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
        assert m.model_cfg["max_tokens"]
        assert m.model_cfg["temperature"] in [0.1, 0.7]


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_llm_client_config_html_report(tmp_path):
    #
    # GIVEN
    #
    test_lab_path = "data/generative/ci_llm_test_lab_h2ogpt_args.json"
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


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
