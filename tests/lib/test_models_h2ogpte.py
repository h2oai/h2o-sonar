# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json

import pytest

from h2o_sonar import evaluate
from h2o_sonar.evaluators import rag_tokens_presence_evaluator
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import models
from h2o_sonar.lib.integrations import genai
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative


@pytest.mark.parametrize(
    "model_type,is_rag",
    [
        (models.ExplainableModelType.mock, False),
        (models.ExplainableModelType.driverless_ai, False),
        (models.ExplainableModelType.driverless_ai_rest, False),
        (models.ExplainableModelType.h2o3, False),
        (models.ExplainableModelType.scikit_learn, False),
        (models.ExplainableModelType.h2ogpte, True),
        (models.ExplainableModelType.h2ogpte_llm, False),
        (models.ExplainableModelType.h2ogpt, False),
        (models.ExplainableModelType.h2ollmops, False),
        (models.ExplainableModelType.ollama, False),
        (models.ExplainableModelType.openai_rag, True),
        (models.ExplainableModelType.openai_llm, False),
        (models.ExplainableModelType.azure_openai_llm, False),
        (models.ExplainableModelType.amazon_bedrock_rag, True),
        (models.ExplainableModelType.unknown, False),
    ],
)
@pytest.mark.h2o_sonar
def test_is_rag(model_type: models.ExplainableModelType, is_rag: bool):
    #
    # WHEN
    #
    actual_is_rag = models.ExplainableModelType.is_rag(model_type)

    #
    # THEN
    #
    assert actual_is_rag == is_rag, (
        f"Expected {is_rag=} for {model_type=}, but got {actual_is_rag}"
    )


@pytest.mark.parametrize(
    "model_type,is_llm",
    [
        (models.ExplainableModelType.mock, False),
        (models.ExplainableModelType.driverless_ai, False),
        (models.ExplainableModelType.driverless_ai_rest, False),
        (models.ExplainableModelType.h2o3, False),
        (models.ExplainableModelType.scikit_learn, False),
        (models.ExplainableModelType.h2ogpte, False),
        (models.ExplainableModelType.h2ogpte_llm, True),
        (models.ExplainableModelType.h2ogpt, True),
        (models.ExplainableModelType.h2ollmops, True),
        (models.ExplainableModelType.ollama, True),
        (models.ExplainableModelType.openai_rag, False),
        (models.ExplainableModelType.openai_llm, True),
        (models.ExplainableModelType.azure_openai_llm, True),
        (models.ExplainableModelType.amazon_bedrock_rag, False),
        (models.ExplainableModelType.unknown, False),
    ],
)
@pytest.mark.h2o_sonar
def test_is_llm(model_type: models.ExplainableModelType, is_llm: bool):
    #
    # WHEN
    #
    actual_is_llm = models.ExplainableModelType.is_llm(model_type)

    #
    # THEN
    #
    assert actual_is_llm == is_llm, (
        f"Expected {is_llm=} for {model_type=}, but got {actual_is_llm}"
    )


@pytest.mark.skip(reason="h2oGPTe API experimental auto LLM calls")
@pytest.mark.parametrize(
    "llm_model", [None, "", "auto", given_generative.LLM_CLAUDE_SONNET_37]
)
@pytest.mark.h2o_sonar
def test_auto_llm(llm_model):
    """How does h2oGPTe behave with different collection and client settings?

    Collection / genai Client LLM arg -> h2oGPTe action
    ---------- / -------------------- -> --------------
    Gemini     / None                 -> Gemini (collection's LLM used)
    Gemini     / ""                   -> Gemini (collection's LLM used)
    Gemini     / auto                 -> ?
    auto       / None                 -> automatic LLM selection
    auto       / auto                 -> automatic LLM selection
    [whatever] / [model name]        -> [model name] arg overrides collection LLM

    H2O Sonar report behavior:

    evaluation LLM arg / report model name / h2oGPTe action
    ------------------ / ----------------- / --------------
    "auto"             / auto              / automatic LLM selection @ collection
    "inherited"        / inherited         / use LLM configured by collection

    """

    #
    # GIVEN
    #

    connection = test_utils.health.get_h2ogpte()
    h2ogpte_client = genai.H2oGpteRagClient(connection)

    #
    # WHEN
    #

    answers = h2ogpte_client.ask_collection(
        collection_id="e01d2f09-a558-4206-a90d-a2feb7fdea85",  # i-d
        prompts=[
            "What is the capital of France?",
            "How much is 2024 - 2023 * 3 / sqrt(12) CZK in EUR?",
        ],
        llm_model_name=llm_model,
    )

    #
    # THEN
    #

    assert len(answers) > 0, "No answers were returned by the LLM."
    for answer in answers:
        assert answer, "An empty or invalid answer was returned by the LLM."


@pytest.mark.parametrize(
    "raw_llm,expected_llm",
    [
        ("auto", genai.H2oGpteRagClient.MODEL_SPEC_AUTO),
        ("", genai.H2oGpteRagClient.MODEL_SPEC_COL),
        (None, genai.H2oGpteRagClient.MODEL_SPEC_COL),
        ("blah", "blah"),
        (given_generative.LLM_CLAUDE_SONNET_37, given_generative.LLM_CLAUDE_SONNET_37),
    ],
)
@pytest.mark.h2o_sonar
def test_auto_h2ogpte_normalization(raw_llm: str | None, expected_llm: str):
    #
    # GIVEN
    #

    print(f"Raw LLM: {raw_llm}")

    #
    # WHEN
    #

    actual_llm = testing.RagTestLab._preprocess_llm_model_name(raw_llm)
    actual_llms = testing.RagTestLab._preprocess_llm_model_names(
        rag_model_type=models.ExplainableModelType.h2ogpte,
        llm_model_names=[raw_llm],
    )

    #
    # THEN
    #

    assert actual_llm == expected_llm, (
        f"Expected '{expected_llm}', but got '{actual_llm}'"
    )
    assert actual_llms == [expected_llm], (
        f"Expected '{expected_llm}', but got '{actual_llm}' in list"
    )


@pytest.mark.skip(reason="h2oGPTe API introspection test")
@pytest.mark.parametrize(
    "interval",
    [
        # "24 hours",
        # "100 years",
        "3600 seconds",
    ],
)
@pytest.mark.h2o_sonar
def test_llm_performance_metadata(interval):
    from h2ogpte import types

    #
    # GIVEN
    #
    connection = test_utils.health.get_h2ogpte()
    h2ogpte_client = genai.H2oGpteRagClient(connection)

    # interval can be specified as ..., examples:
    # - "1000 seconds"
    # - "24 hours"
    # - "100 years"

    #
    # WHEN
    #

    # List[LLMPerformance]
    try:
        llm_perf_profiles: list[types.LLMPerformance] = (
            h2ogpte_client.client.get_llm_performance_by_llm(interval)
        )
    except Exception:
        llm_perf_profiles = []

    vision_models = h2ogpte_client.client.get_llm_and_auto_vision_llm_names()

    #
    # THEN
    #

    # {
    #     "call_count": 1132,
    #     "input_tokens": 1144932,
    #     "llm_name": "claude-3-5-sonnet-20240620",
    #     "output_tokens": 151305,
    #     "time_to_first_token": 0.7049375,
    #     "tokens_per_second": 38.9025
    # }
    print(f"LLM performance metadata:\n{len(llm_perf_profiles)}")
    for llm_perf_profile in llm_perf_profiles:
        print(f"  {llm_perf_profile.llm_name}")
        print(f"    call_count: {llm_perf_profile.call_count}")
        print(f"    input_tokens: {llm_perf_profile.input_tokens}")
        print(f"    output_tokens: {llm_perf_profile.output_tokens}")
        print(f"    tokens_per_second: {llm_perf_profile.tokens_per_second}")
        print(f"    time_to_first_token: {llm_perf_profile.time_to_first_token}")

        assert llm_perf_profile

    print(f"Vision models:\n{json.dumps(vision_models, indent=2)}")
    assert vision_models


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.skip(reason="Tool - not a test")
@pytest.mark.h2o_sonar
def test_purge_server():
    #
    # GIVEN
    #

    # h2oGPTe server to purge
    connection = given_generative.H2OGPTE_C_D
    # connection = test_utils.health.get_h2ogpte()

    client = genai.H2oGpteRagClient(connection)

    #
    # WHEN
    #
    for _ in range(10):
        recent_chats = client.client.list_recent_chat_sessions(0, 1000)
        if not recent_chats:
            print("No more chat sessions to purge.")
            break
        for chat in recent_chats:
            chat_id = chat.id
            print(f"Purging chat session: {chat_id}")
            try:
                client.client.delete_chat_sessions([chat_id])
            except Exception as e:
                print(f"Failed to purge chat: {chat_id} - {e}")

    recent_collections = client.client.list_recent_collections(0, 1000)
    for c in recent_collections:
        collection_id = c.id
        print(f"Purging collection {c.name}: {collection_id}")
        try:
            client.purge_collections([collection_id])
        except Exception as e:
            print(f"Failed to purge collection: {collection_id} - {e}")

    recent_documents = client.client.list_recent_documents(0, 1000)
    for d in recent_documents:
        document_id = d.id
        print(f"Purging document {d.name}: {document_id}")
        try:
            client.purge_uploaded_docs([document_id])
        except Exception as e:
            print(f"Failed to purge document: {document_id} - {e}")


def _when_model_cfg_h2ogpte() -> dict:
    # customize SYSTEM PROMPT
    custom_config = genai.H2oGpteRagClient.config_factory()
    print(f"Config prototype:\n{custom_config}")

    custom_config["system_prompt"] = (
        "You are h2oGPTe, an expert question-answering AI system created by H2O.ai "
        "that performs like GPT-4 by OpenAI."
    )

    # customize QUERY PROMPT
    #
    # Pay attention and remember the information below. You will need to use only
    # the given document context to answer the question or imperative at the end.
    #
    # """
    # <DOCUMENT CONTEXT>
    # """
    # According to only the information in the document sources provided within
    # the context above,
    # <USER PROMPT>
    #
    # prompt for QUERY use case
    custom_config["pre_prompt_query"] = (
        "Pay attention and remember the information below. You will need to use only "
        "the given document context to answer the question or imperative at the end."
    )
    # <DOCUMENT CONTEXT>
    custom_config["prompt_query"] = (
        "According to only the information in the document sources provided within "
        "the context above, "
    )
    # <USER PROMPT>
    #
    # prompt for SUMMARIZATION use case
    custom_config["pre_prompt_summary"] = (
        "In order to write a funny and sarcastic single-paragraph summary, pay "
        "attention to the following text:"
    )
    # <DOCUMENT CONTEXT>
    custom_config["prompt_summary"] = (
        "Using only the text above, write a condensed and concise funny and "
        "sarcastic summary of key results."
    )
    # ... no user prompt as it is summary.
    #

    # customize LLM
    custom_config["llm"] = None  # gpt-4
    custom_config["llm_args"] = {"temperature": 0.5}  # see config_factory for details

    return custom_config


def _when_model_cfg_h2ogpte_llm() -> dict:
    # customize SYSTEM PROMPT
    custom_config = genai.H2oGpteRagClient.config_factory(
        commons.ModelTypeExplanation.LLM
    )
    print(f"Config prototype:\n{custom_config}")

    custom_config["system_prompt"] = (
        "You are h2oGPTe, an expert question-answering AI system created by H2O.ai "
        "that performs like GPT-4 by OpenAI."
    )
    custom_config["pre_prompt_query"] = (
        "Pay attention and remember the information below. You will need to use only "
        "the given document context to answer the question or imperative at the end."
    )
    custom_config["prompt_query"] = (
        "According to only the information in the document sources provided within "
        "the context above, "
    )
    custom_config["text_context_list"] = [
        "test",
        "arguments",
        "for",
        "llm",
        "model",
        "configuration",
    ]

    # customize LLM
    custom_config["llm"] = None  # gpt-4
    custom_config["llm_args"] = {"temperature": 0.5}

    # customize CHAT CONVERSATION
    custom_config["chat_conversation"] = [
        ("What is your purpose?", "I am an AI system that answers questions."),
        ("What is your name?", "My name is h2oGPTe."),
        ("When were you created?", "I was created in 2024."),
    ]

    return custom_config


@pytest.mark.parametrize(
    "test_suite_path,model_count,test_case_count,with_cfg",
    [
        ("data/generative/ci_rag_test_suite.json", 2, 2, True),
    ],
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_rag_client_config(
    tmp_path,
    test_suite_path: str,
    model_count: int,
    test_case_count: int,
    with_cfg: bool,
):
    #
    # GIVEN
    #

    connection = test_utils.health.get_h2ogpte()
    # slow_llm_model_names = ["deepseek-ai/DeepSeek-R1-shadeform"]
    llm_model_names = test_utils.health.get_h2ogpte_models()[:model_count]
    test_suite = testing.RagTestSuiteConfig.load_from_json(
        test_utils.find_locally(test_suite_path)
    )
    # cut down the test suite to speed up the test
    if test_case_count:
        test_suite.test_cases = test_suite.test_cases[:test_case_count]

    #
    # WHEN
    #

    # TEST LAB:
    # - test lab uses the arguments to create a collection,
    #   then uses the other arguments to query the collection
    #   to build lab w/ resolved responses
    # - parameters are NOT used by evaluators as the data are already resolved

    rag_models_cfgs = {}
    for llm_model_name in llm_model_names:
        rag_models_cfgs[llm_model_name] = []

        for temperature in [0.1, 0.7]:
            # make each config slightly different
            model_cfg = _when_model_cfg_h2ogpte()
            model_cfg["llm_args"] = {"temperature": temperature}
            rag_models_cfgs[llm_model_name].append(model_cfg)

    print(f"RAG models configurations:\n{json.dumps(rag_models_cfgs, indent=2)}\n")
    test_lab = testing.RagTestLab.from_rag_test_suite(
        rag_connection=connection,
        rag_test_suite=test_suite,
        rag_model_type=models.ExplainableModelType.h2ogpte,
        llm_model_names=llm_model_names,
        rag_models_cfgs=rag_models_cfgs if with_cfg else None,
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
        llm_host_prompt_cache=None,
    )

    # deploy the test lab configuration to the h2oGPTe server (collections @ corpora)
    test_lab.build()
    # complete dataset w/ actual values from the h2oGPTe server - answer, duration, ...
    test_lab.complete_dataset(
        save_as_you_go=tmp_path / "wip_testlab.json",
        parallelize=testing.TestLab.PARALLEL_RUN,
        # test exponential backoff driven timeouts
        timeout_exp_backoff=genai.TimeoutRetryExpBackoffCtx(min_backoff_secs=1),
    )
    test_lab_path = tmp_path / "test_lab.json"
    test_lab.save_as_json(test_lab_path)

    #
    # THEN
    #
    print(f"Test lab (RAG):\nfile://{test_lab_path}")
    print(test_lab)
    # assert the test lab which was created w/ CUSTOM CONFIG
    assert test_lab
    if with_cfg:
        for m in test_lab.evaluated_models.values():
            assert m.model_cfg
            assert m.model_cfg["system_prompt"]
            assert m.model_cfg["llm_args"]["temperature"] in [0.1, 0.7]


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_rag_client_config_html_report(tmp_path):
    #
    # GIVEN
    #
    test_lab_path = "data/generative/ci_rag_test_lab_h2ogpte_args.json"
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


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_llm_client_config(tmp_path):
    #
    # GIVEN
    #

    connection = test_utils.health.get_h2ogpte()
    llm_model_names = genai.H2oGpteRagClient(connection).list_llm_model_names()[:2]
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
            model_cfg = _when_model_cfg_h2ogpte_llm()
            model_cfg["llm_args"] = {"temperature": temperature}
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
        assert m.model_cfg["system_prompt"]
        assert m.model_cfg["llm_args"]["temperature"] in [0.1, 0.7]


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_llm_client_config_html_report(tmp_path):
    #
    # GIVEN
    #
    test_lab_path = "data/generative/ci_llm_test_lab_h2ogpte_args.json"
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
