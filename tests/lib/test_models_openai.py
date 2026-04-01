# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import datetime
import json
import pathlib
import pprint
import time
import traceback

import pytest

from h2o_sonar import evaluate
from h2o_sonar import loggers
from h2o_sonar.evaluators import rag_tokens_presence_evaluator
from h2o_sonar.lib.api import models
from h2o_sonar.lib.integrations import genai
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative


def is_given_openai_version_1():
    """Check if the OpenAI version 1 is given."""
    return genai.OpenAiAssistantsRagClientVersion1 == genai.OpenAiAssistantsRagClient


@pytest.mark.skip("MS Azure hosted OpenAI API sanity check")
@pytest.mark.skipif(
    not test_utils.health.is_openai(),
    reason="Either OpenAI API key not set or OpenAI Python package is not installed",
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_sanity_azure():
    """Test OpenAI API availability, API key validity, and API version."""
    #
    # GIVEN
    #
    azure_connection = given_generative.AZURE_OPENAI_LLM

    import openai

    client = openai.AzureOpenAI(
        azure_endpoint=azure_connection.server_url,
        api_key=azure_connection.token,
        api_version=genai.MsAzureOpenAiLlmClient.DEFAULT_API_VERSION,
    )

    #
    # WHEN list models
    #
    print("Listing models...")
    model_list = client.models.list()
    print("Available models:")
    for m in model_list:
        print(f"  - {m}")

    #
    # WHEN chat completion
    #
    print("Chat completion...")
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello world!"}],
        # IMPORTANT: model must be DEPLOYMENT NAME, not actual OpenAI model name!
        # IMPORTANT: model must be DEPLOYMENT NAME, not actual OpenAI model name!
        # IMPORTANT: model must be DEPLOYMENT NAME, not actual OpenAI model name!
        # DO NOT USE - NOT VALID IN AZURE: model="gpt-35-turbo",
        model=azure_connection.server_id,
        temperature=0.7,  # optional creative factor
        max_tokens=64,  # optional max response tokens
    )

    #
    # THEN
    #
    print(chat_completion)
    pprint.pprint(json.loads(chat_completion.model_dump_json()))
    assert chat_completion
    print(f"Answer: {chat_completion.choices[0].message.content}")
    assert chat_completion.choices[0].message.content


@pytest.mark.skip("OpenAI API sanity check")
@pytest.mark.skipif(
    not test_utils.health.is_openai(),
    reason="Either OpenAI API key not set or OpenAI Python package is not installed",
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_sanity():
    """Test OpenAI API availability, API key validity, and API version."""
    import openai

    #
    # GIVEN
    #
    openai_connection = given_generative.OPENAI_RAG
    client = openai.OpenAI(api_key=openai_connection.token)

    #
    # WHEN
    #
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello world!"}],
        model="gpt-3.5-turbo",  # optional LLM model
        temperature=0.7,  # optional creative factor
        max_tokens=64,  # optional max response tokens
    )

    #
    # THEN
    #
    print(chat_completion)
    pprint.pprint(json.loads(chat_completion.model_dump_json()))
    assert chat_completion
    print(f"Answer: {chat_completion.choices[0].message.content}")
    assert chat_completion.choices[0].message.content


@pytest.mark.skip("OpenAI API test which uses paid API")
@pytest.mark.skipif(
    not test_utils.health.is_openai(),
    reason="Either OpenAI API key not set or OpenAI Python package is not installed",
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_model_list():
    import openai

    #
    # GIVEN
    #
    openai_connection = given_generative.OPENAI_RAG
    client = openai.OpenAI(api_key=openai_connection.token)

    #
    # WHEN
    #
    model_list = client.models.list()

    #
    # THEN
    #
    print(model_list)
    print(type(model_list))
    assert model_list


@pytest.mark.skip("OpenAI API test which uses paid API")
@pytest.mark.skipif(
    not test_utils.health.is_openai(),
    reason="OpenAI API key is not set or OpenAI package is not installed",
)
@pytest.mark.skipif(
    not is_given_openai_version_1(),
    reason="OpenAI Assistants API v1 is not configured",
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_assistant_api():
    """Test OpenAI Assistant API w/ file search/retrieval tool ~ RAG:

    - create assistant
    - create thread @ assistant
    - upload document to thread
    - ask a question (prompt) @ thread
    - get answer @ thread
    - assert the answer
    - purge assistant (thread and document are purged as well)

    See:

    - doc https://platform.openai.com/docs/assistants/overview
    - files: https://platform.openai.com/files (must be purged)
    - assistants: https://platform.openai.com/assistants (must be purged)

    """
    import openai

    #
    # GIVEN
    #
    llm_model = genai.OpenAiAssistantsRagClientVersion1.DEFAULT_LLM_MODEL

    openai_connection = given_generative.OPENAI_RAG
    client = openai.OpenAI(api_key=openai_connection.token)

    # RAG (collection) documents
    corpus = [
        test_utils.find_locally(
            "data/generative/corpus/talk2report-deepeval-20231103.pdf"
        )
    ]

    #
    # WHEN
    #

    assistant = None
    uploaded_files = []
    try:
        # UPLOAD corpus ~ document(s)
        for doc_path in corpus:
            uploaded_file = client.files.create(
                file=open(doc_path, "rb"), purpose="assistants"
            )
            uploaded_files.append(uploaded_file)

        # create ASSISTANT
        assistant = client.beta.assistants.create(
            name=f"EvalStudio OpenAI RAG evaluation ({datetime.datetime.now()})",
            instructions=(
                "You are a support chatbot. Use your knowledge base "
                "(uploaded documents) to respond to asked questions."
            ),
            # enable RAG ~ retrieval tool
            tools=[{"type": "retrieval"}],
            model=llm_model,
            # RAG (collection) documents
            file_ids=[file.id for file in uploaded_files],
        )

        # create THREAD @ assistant
        thread = client.beta.threads.create()

        # create MESSAGE w/ question @ thread
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content="What are the transformed features of the model?",
        )
        assert message
        print(message.model_dump_json())

        # RUN the ASSISTANT @ THREAD
        assistant_run = client.beta.threads.runs.create(
            assistant_id=assistant.id,
            thread_id=thread.id,
            instructions="The user has a premium account.",
        )
        print(assistant_run.model_dump_json())
        assert assistant_run
        # WAIT for the assistant run to complete
        assistant_err_statuses = ["failed", "expired", "cancelled"]
        assistant_done_statuses = ["completed"] + assistant_err_statuses
        while assistant_run.status not in assistant_done_statuses:
            print(f"Waiting for the run to complete: {assistant_run.status}")
            assistant_run = client.beta.threads.runs.retrieve(
                thread_id=thread.id, run_id=assistant_run.id
            )
            print(assistant_run.model_dump_json())
            time.sleep(1.0)

        print(assistant_run.model_dump_json())
        assert assistant_run

        if assistant_run.status in assistant_err_statuses:
            raise RuntimeError(f"AI Assistant run failed: {assistant_run.status}")

        #
        # THEN
        #

        # get ACTUAL OUTPUT and CONTEXT
        thread_messages = client.beta.threads.messages.list(thread_id=thread.id).data
        if not len(thread_messages):
            raise RuntimeError(
                f"No messages in the thread: {thread.id} for assistant: {assistant.id} "
                f"and corpus: {corpus}"
            )
        assistant_message = thread_messages[0]
        if not len(assistant_message.content):
            raise RuntimeError(
                f"Unable to get context - no assistant message content in the thread:"
                f" {thread.id} for assistant: {assistant.id} and corpus: {corpus}"
            )
        if not len(assistant_message.content[0].text.annotations):
            raise RuntimeError(
                f"Unable to get context - no assistant message annotation in the "
                f"thread: {thread.id} for assistant: {assistant.id} "
                f"and corpus: {corpus}"
            )
        reference = assistant_message.content[0].text.annotations[0].file_citation.quote
        message_text = assistant_message.content[0].text.value

        print("Answer:")
        print(f"  Actual output:\n{message_text}")
        print(f"  Context      :\n{reference}")

    except Exception as ex:
        print(f"Test failed with: {ex}:\n{traceback.format_exc()}")
    finally:
        # purge ASSISTANT
        if assistant:
            try:
                client.beta.assistants.delete(assistant.id)
            except Exception as aex:
                print(f"Failed to purge assistant: {aex}")
        # purge THREAD
        for file in uploaded_files:
            try:
                client.files.delete(file.id)
            except Exception as fex:
                print(f"Failed to purge file: {fex}")


@pytest.mark.skip("OpenAI API test which uses paid API")
@pytest.mark.skipif(
    not test_utils.health.is_openai(),
    reason="Either OpenAI API key not set or OpenAI Python package is not installed",
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_openai_llm_client(tmp_path):
    """Test OpenAI LLM client."""
    #
    # GIVEN
    #
    base_llm_model = genai.OpenAiLlmClient.DEFAULT_LLM_MODEL

    client = genai.OpenAiLlmClient(
        connection=given_generative.OPENAI_LLM,
        default_llm_model_name=base_llm_model,
        logger=loggers.SonarPrintLogger(),
    )

    #
    # WHEN
    #

    llm_model_names = client.list_llm_model_names()
    print(f"Available LLM models: {llm_model_names}")
    assert llm_model_names
    assert genai.OpenAiLlmClient.DEFAULT_LLM_MODEL in llm_model_names

    results = client.ask_model(
        prompts=[
            "How many fingers humans have?",
            "Why is the number 42 so important?",
        ],
        # default llm_model_name
    )
    (prompts, answers, duration, chunks, cost, _, _) = results[0]

    #
    # THEN
    #

    print(f"Prompt: {prompts}")
    print(f"Answer: {answers}")
    assert prompts
    assert answers
    assert duration
    assert not chunks
    assert not cost


@pytest.mark.skip("MS Azure OpenAI API test which uses paid API")
@pytest.mark.skipif(
    not test_utils.health.is_azure_openai(),
    reason=(
        "Either Microsoft Azure hosted OpenAI API key not set or OpenAI Python package "
        "is not installed"
    ),
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_azure_openai_llm_client(tmp_path):
    """Test OpenAI LLM client."""
    #
    # GIVEN
    #
    client = genai.MsAzureOpenAiLlmClient(
        connection=given_generative.AZURE_OPENAI_LLM,
        api_version=genai.MsAzureOpenAiLlmClient.DEFAULT_API_VERSION,
        logger=loggers.SonarPrintLogger(),
    )

    #
    # WHEN
    #

    llm_model_names = client.list_llm_model_names()
    print(f"Available LLM models: {llm_model_names}")
    assert llm_model_names

    results = client.ask_model(
        prompts=[
            "How many fingers humans have?",
            "Why is the number 42 so important?",
        ],
        # default llm_model_name
    )
    (prompts, answers, duration, chunks, cost, _, _) = results[0]

    #
    # THEN
    #

    print(f"Prompt: {prompts}")
    print(f"Answer: {answers}")
    assert prompts
    assert answers
    assert duration
    assert not chunks
    assert not cost


@pytest.mark.skip("MS Azure OpenAI API test which uses paid API")
@pytest.mark.skipif(
    not test_utils.health.is_azure_openai(),
    reason=(
        "Either Microsoft Azure hosted OpenAI API key not set or OpenAI Python package "
        "is not installed"
    ),
)
@pytest.mark.parametrize(
    "model_cfg",
    [
        # None,
        (
            genai.MsAzureOpenAiLlmClient.config_factory()
            if test_utils.health.is_azure_openai()
            else {}
        ),
        # {
        #     "temperature": 0.0,
        # },
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_build_lab_azure_openai_llm(tmp_path, model_cfg):
    #
    # GIVEN
    #

    connection = test_utils.health.get_openai_azure()

    llm_model_names = genai.MsAzureOpenAiLlmClient(connection).list_llm_model_names()

    test_suite = testing.RagTestSuiteConfig.load_from_json(
        test_utils.find_locally("data/generative/eval_llm/bank_teller_test_suite.json")
    )

    # OPTIONAL DESCOPE: faster / smaller test (debugging)
    llm_model_names = llm_model_names[:1]
    test_suite.test_cases = test_suite.test_cases[:2]

    # LLM model configurations
    if model_cfg:
        llm_models_cfgs = {"environment_id": "eval-studio-testing"}
        # llm_models_cfgs = {
        #     llm_model_names[0]: [model_cfg]
        # }
    else:
        llm_models_cfgs = None
    print(f"Custom model configs:\n{json.dumps(llm_models_cfgs, indent=2)}")

    #
    # WHEN
    #

    # test lab (RAG product)
    test_lab = testing.RagTestLab.from_llm_test_suite(
        llm_host_connection=connection,
        llm_test_suite=test_suite,
        llm_model_names=llm_model_names,
        llm_model_type=models.ExplainableModelType.ollama,
        llm_models_cfgs=llm_models_cfgs,
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


def _when_model_cfg(temperature=None):
    print(
        f"AUTO choosing OpenAI Assistant implementation: "
        f"{genai.OpenAiAssistantsRagClient}"
    )

    model_cfg = genai.OpenAiAssistantsRagClient.config_factory()

    # assistant
    model_cfg[genai.OpenAiAssistantsRagClient.KWARGS_ASSISTANT]["name"] = (
        "TEST assistant name"
    )
    model_cfg[genai.OpenAiAssistantsRagClient.KWARGS_ASSISTANT]["instructions"] = (
        "You are a TEST chatbot. Use your knowledge base (uploaded documents) to "
        "respond to asked questions."
    )

    # thread
    model_cfg[genai.OpenAiAssistantsRagClient.KWARGS_THREAD]["timeout"] = 123

    # completion
    model_cfg[genai.OpenAiAssistantsRagClient.KWARGS_RUN]["temperature"] = (
        0.9 if temperature is None else temperature
    )
    model_cfg[genai.OpenAiAssistantsRagClient.KWARGS_RUN]["additional_instructions"] = (
        "The TEST user has a premium account - always use sir/madam!"
    )
    print(f"Custom model config:\n{json.dumps(model_cfg, indent=2)}")

    return model_cfg


@pytest.mark.skip("OpenAI API test which uses paid API")
@pytest.mark.skipif(
    not test_utils.health.is_openai(),
    reason="OpenAI API key is not set or OpenAI package is not installed",
)
@pytest.mark.skipif(
    not is_given_openai_version_1(),
    reason="OpenAI Assistants API v1 is not configured",
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_openai_rag_client(tmp_path, do_model_cfg=True):
    """Test OpenAI RAG client:

    - 1x prompt w/ full Assistant setup takes ~ 40s.
    - 50x prompts > 33' ; 100x prompts > 1h 6'

    """
    #
    # GIVEN
    #
    corpus = [
        test_utils.find_locally(
            "data/generative/corpus/talk2report-deepeval-20231103.pdf"
        )
    ]

    base_llm_model = genai.OpenAiAssistantsRagClient.DEFAULT_LLM_MODEL

    client = genai.OpenAiAssistantsRagClient(
        connection=given_generative.OPENAI_RAG,
        default_llm_model_name=base_llm_model,
        logger=loggers.SonarPrintLogger(),
    )

    model_cfg = _when_model_cfg() if do_model_cfg else {}

    #
    # WHEN
    #

    prompt = None
    answer = None
    duration = None
    chunks = None
    # cost = None
    try:
        # create ASSISTANT
        assistant_id = client.create_collection(
            doc_paths=corpus,
            llm_model_name=base_llm_model,
            name=f"EvalStudio OpenAI RAG evaluation ({datetime.datetime.now()})",
            **model_cfg,
        )

        # get ANSWERS w/ CONTEXTS
        results = client.ask_collection(
            assistant_id=assistant_id,
            prompts=["What are the transformed features of the model?"],
            include_chunks=10,
            **model_cfg,
        )
        (prompt, answer, duration, chunks, cost, _, _) = results[0]
    finally:
        # purge all ASSISTANTS create by the RAG client
        client.purge_collections()
        # purge all DOCUMENTS uploaded by the RAG client
        client.purge_uploaded_docs()

    assert prompt
    assert answer
    assert duration
    assert chunks
    # assert cost ... runtime cost is unknown for OpenAI RAG


@pytest.mark.skip("OpenAI API test which uses paid API")
@pytest.mark.skipif(
    not test_utils.health.is_openai(),
    reason="OpenAI API key is not set or OpenAI package is not installed",
)
@pytest.mark.skipif(
    not is_given_openai_version_1(),
    reason="OpenAI Assistants API v1 is not configured",
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_build_lab_for_openai_rag_v1(tmp_path, do_model_cfg=True):
    #
    # GIVEN
    #

    openai_connection = given_generative.OPENAI_RAG

    llm_model_names = genai.OpenAiAssistantsRagClient.BASE_LLM_MODELS

    test_suite = testing.RagTestSuiteConfig.load_from_json(
        # suite: 1 doc, 1 test case, 2 LLMs > w/ CTX can run Arno
        "data/generative/h2ogpte_benchmark_test_suite_min.json"
        # suite: 4 docs, 5 test cases, 2 LLMs
        # "data/generative/h2ogpte_benchmark_test_suite_top.json"
        # suite: ~20 docs, 50 test cases, 2 LLMs (removed MP3, JPG, ...) > 1h13'
        # "data/generative/h2ogpte_benchmark_test_suite_openai.json"
        # suite: 1 doc, 25 prompts, 2 LLMs > w/o CTX can run Arno
        # "data/generative/talk2report_prompts_test_suite.json"
    )

    # OPTIONAL DESCOPE: faster / smaller test (debugging)
    # llm_model_names = llm_model_names[:1]
    # test_suite.test_cases = test_suite.test_cases[:2]

    model_cfgs = (
        [_when_model_cfg(temperature=0.1), _when_model_cfg(0.9)] if do_model_cfg else []
    )

    #
    # WHEN
    #

    # test lab (RAG product)
    test_lab = testing.RagTestLab.from_rag_test_suite(
        rag_connection=openai_connection,
        rag_test_suite=test_suite,
        llm_model_names=llm_model_names,
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
        rag_model_type=models.ExplainableModelType.openai_rag,
        rag_models_cfgs=(
            {m: model_cfgs for m in llm_model_names} if model_cfgs else None
        ),
    )
    test_lab.save_as_json(tmp_path / "001_before_testlab.json")
    try:
        # test lab: DEPLOY to OpenAI
        # (docs sync: S3 > filesystem cache > OpenAI Assistant per Test)
        test_lab.build()
        test_lab.save_as_json(tmp_path / "002_after_build_testlab.json")

        # test lab: complete dataset w/ ACTUAL data (answers, duration, context)
        test_lab.complete_dataset(
            complete_context=10, save_as_you_go=tmp_path / "003_wip_chat_testlab.json"
        )
        # backup fully resolved dataset
        lab_path = test_lab.save_as_json(
            tmp_path / "DONE_test_lab_with_actual_values.json"
        )

        #
        # THEN
        #

        assert pathlib.Path(lab_path).exists()
    finally:
        # purge test lab
        test_lab.purge()


@pytest.mark.skip("OpenAI API test which uses paid API")
@pytest.mark.skipif(
    not test_utils.health.is_openai(),
    reason="OpenAI API key is not set or OpenAI package is not installed",
)
@pytest.mark.skipif(
    is_given_openai_version_1(),
    reason="OpenAI Assistants API v2 is required (v1 is configured)",
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_build_lab_for_openai_rag_v2(tmp_path, do_model_cfg=True):
    #
    # GIVEN
    #

    openai_connection = given_generative.OPENAI_RAG

    llm_model_names = genai.OpenAiAssistantsRagClient.BASE_LLM_MODELS
    print(f"Using LLM models: {llm_model_names}")

    test_suite = testing.RagTestSuiteConfig.load_from_json(
        # suite: 1 doc, 1 test case, 2 LLMs > w/ CTX can run Arno
        test_utils.find_locally("data/generative/h2ogpte_benchmark_test_suite_min.json")
    )

    #
    # WHEN
    #

    # test lab (RAG product)
    test_lab = testing.RagTestLab.from_rag_test_suite(
        rag_connection=openai_connection,
        rag_test_suite=test_suite,
        llm_model_names=llm_model_names,
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
        rag_model_type=models.ExplainableModelType.openai_rag,
    )
    test_lab.save_as_json(tmp_path / "001_before_testlab.json")
    try:
        # test lab: DEPLOY to OpenAI
        # (docs sync: S3 > filesystem cache > OpenAI Assistant per Test)
        test_lab.build()
        test_lab.save_as_json(tmp_path / "002_after_build_testlab.json")

        # test lab: complete dataset w/ ACTUAL data (answers, duration, context)
        test_lab.complete_dataset(
            complete_context=10, save_as_you_go=tmp_path / "003_wip_chat_testlab.json"
        )
        # backup fully resolved dataset
        lab_path = test_lab.save_as_json(
            tmp_path / "DONE_test_lab_with_actual_values.json"
        )

        #
        # THEN
        #
        print(f"Test lab saved at: file://{lab_path}")
        assert pathlib.Path(lab_path).exists()
    finally:
        # purge test lab
        test_lab.purge()


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_client_config_html_report(tmp_path):
    #
    # GIVEN
    #
    test_lab_path = "data/generative/ci_rag_test_lab_openai_args.json"
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


@pytest.mark.skip("OpenAI API files purge")
@pytest.mark.skipif(
    not test_utils.health.is_openai(),
    reason="Either OpenAI API key not set or OpenAI Python package is not installed",
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_remove_files_openai(tmp_path):
    # Script to remove files from OpenAI API
    openai_client = genai.OpenAiLlmClient(
        connection=given_generative.OPENAI_LLM,
        default_llm_model_name=genai.OpenAiLlmClient.DEFAULT_LLM_MODEL,
        logger=loggers.SonarPrintLogger(),
    )

    all_files = openai_client.client.files.list()
    to_delete = [
        r
        for r in all_files
        if r.filename == "bradesco-2022-integrated-report.pdf"
        or r.filename == "Coca-Cola-FEMSA-Results-1Q23-vf-2.pdf"
        or r.filename == "sr1107a1.pdf"
    ]
    # continue deleting
    i = 0
    for r in to_delete:
        i += 1
        openai_client.client.files.delete(r.id)
        print(f"Deleted {r.filename} ({i}/{len(to_delete)})")
    print(f"Deleted {len(to_delete)} files")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
