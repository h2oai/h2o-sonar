# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import pathlib
import shutil
import traceback

import pytest

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import loggers
from h2o_sonar.lib.api import agents
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import models
from h2o_sonar.lib.integrations import genai
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative


def _then_assert_agent_completion_dirs(user_dir: pathlib.Path):
    # agents: assert the expected directory and files exist:
    # h2o-sonar/ ... user dir
    #   test_lab_<UUID>/ ... test lab dir
    #     completion_of_m_<model UUID>_tc_<test case UUID>/
    #       chat_session_*/ ... chat session dir
    #         chat_message_*/ ... chat message dir
    #           MSG_META_TYPE_ITEM_*.*
    assert user_dir.exists() and user_dir.is_dir()

    test_lab_dirs = list(user_dir.glob("test_lab_*"))
    assert len(test_lab_dirs) >= 1, "Expected at least one test_lab_* directory."
    test_lab_dir = test_lab_dirs[0]
    assert test_lab_dir.is_dir()

    completion_tcs_dirs = list(test_lab_dir.glob("completion_of_m_*"))
    assert len(completion_tcs_dirs) >= 1, (
        "Expected at lest one completion_of_m_* directory."
    )
    completion_tc_dir = completion_tcs_dirs[0]
    assert completion_tc_dir.exists() and completion_tc_dir.is_dir()

    chat_session_dirs = list(completion_tc_dir.glob("chat_session_*"))
    assert len(chat_session_dirs) >= 1, "Expected at least one chat session directory."
    chat_session_dir = chat_session_dirs[0]

    chat_message_dirs = list(chat_session_dir.glob("chat_message_*"))
    assert len(chat_message_dirs) >= 1, "Expected at least one chat message directory."
    chat_message_dir = chat_message_dirs[0]
    assert chat_message_dir.is_dir()
    expected_agent_extraction_files = [
        "MSG_META_TYPE_ITEM_agent_analysis.txt",
        "MSG_META_TYPE_ITEM_agent_chat_history.json",
        "MSG_META_TYPE_ITEM_agent_chat_history_md.txt",
        "MSG_META_TYPE_ITEM_agent_files.json",
        "MSG_META_TYPE_ITEM_agent_files_pdf.json",
        "MSG_META_TYPE_ITEM_agent_meta.json",
        # "MSG_META_TYPE_ITEM_code_dict.json",  # no longer available since 2025-10-29
        "MSG_META_TYPE_ITEM_prompt_raw.json",
        "MSG_META_TYPE_ITEM_py_client_code.txt",
        "MSG_META_TYPE_ITEM_turn_message.txt",
        "MSG_META_TYPE_ITEM_turn_title.txt",
        "MSG_META_TYPE_ITEM_usage_stats.json",
        "MSG_META_TYPE_ITEM_version_message.txt",
        # NEW: original_user_query.txt
    ]
    for file_name in expected_agent_extraction_files:
        print(f"Asserting existence of the file: {file_name}")
        file_path = chat_message_dir / file_name
        assert file_path.exists() and file_path.is_file(), f"Missing file: {file_name}"


@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_completion_data_assert(tmp_path):
    #
    # GIVEN
    #
    user_dir = tmp_path / commons.DEFAULT_USER
    user_dir.mkdir(exist_ok=True, parents=True)
    test_lab_dir_name = "test_lab_26350d0c-aeae-4309-8753-3ea06bca278b"

    #
    # WHEN
    #
    shutil.copytree(
        src=f"data/generative/eval_agent/llm_bank_teller_1p/{test_lab_dir_name}",
        dst=f"{user_dir}/{test_lab_dir_name}",
    )

    #
    # THEN
    #
    _then_assert_agent_completion_dirs(user_dir)


# expensive agentic test to save the cost
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.parametrize(
    "is_rag,test_suite_path,llm_model_names,do_use_agent",
    [
        # agent @ RAG
        (
            True,
            "data/generative/eval_agent/rag_fbi_agent_1p/test_suite.json",
            ["auto"],
            True,
        ),
        # agent @ LLM
        (
            False,
            "data/generative/eval_agent/llm_bank_teller_1p/test_suite.json",
            ["auto"],
            True,
        ),
        # NEGATIVE test - configuration forces NOT to use the agent - bug #1542
        (
            True,
            "data/generative/eval_agent/rag_fbi_agent_1p/test_suite.json",
            ["auto"],
            False,
        ),
        # SKIPPED due to cost, rate limits and time
        # (
        #     False,
        #     "data/generative/eval_agent/llm_bank_teller_1p/test_suite.json",
        #     [given_generative.LLM_CLAUDE_SONNET_37],  # ["auto"]
        # ),
    ],
)
@pytest.mark.generative
@pytest.mark.agentic
@pytest.mark.expensive
@pytest.mark.h2o_sonar
def test_lab_completion(
    tmp_path,
    is_rag: bool,
    test_suite_path: str,
    llm_model_names: list,
    do_use_agent: bool,
):
    #
    # GIVEN
    #
    h2ogpte_connection = test_utils.health.get_h2ogpte()
    llm_or_rag_str = "RAG" if is_rag else "LLM"

    llm_or_rag_models_cfgs = {
        llm_model_names[0]: [
            {
                genai.H2oGpteRagClient.CFG_LLM_ARGS: {
                    genai.H2oGpteRagClient.CFG_USE_AGENT: do_use_agent,
                }
            }
        ]
    }
    print(
        f"\nUsing {llm_or_rag_str} models cfg:"
        f"\n{json.dumps(llm_or_rag_models_cfgs, indent=2)}"
    )

    test_suite = testing.RagTestSuiteConfig.load_from_json(file_path=test_suite_path)
    print(
        f"\nRunning {llm_or_rag_str} test lab completion:"
        f"\n  host: {h2ogpte_connection.server_url}"
        f"\n  model: {llm_model_names[0]}"
        f"\n  test suite: {test_suite_path}"
        f"\n  {len(test_suite.test_cfgs)} tests"
        f"\n  {len(test_suite.test_cases)} test cases"
        f"\n"
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
            docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
            results_location=tmp_path,
        )
    else:
        test_lab = testing.RagTestLab.from_llm_test_suite(
            llm_host_connection=h2ogpte_connection,
            llm_test_suite=test_suite,
            llm_model_type=models.ExplainableModelType.h2ogpte_llm,
            llm_models_cfgs=llm_or_rag_models_cfgs,
            llm_model_names=llm_model_names,
            results_location=tmp_path,
        )

    test_lab.save_as_json(
        tmp_path / f"wip_{llm_or_rag_str}_testlab_before_complete.json"
    )

    # deploy the test lab configuration to the h2oGPTe server
    test_lab.build()
    # complete dataset w/ actual values from the h2oGPTe server (answer, duration, ...)
    test_lab.complete_dataset(
        save_as_you_go=tmp_path / f"wip_{llm_or_rag_str}_testlab.json",
        parallelize=testing.TestLab.SEQUENTIAL_RUN,
    )

    #
    # THEN
    #
    test_lab_path = tmp_path / f"{llm_or_rag_str}_test_lab.json"
    test_lab.save_as_json(test_lab_path)
    print(f"{llm_or_rag_str} test lab stored to: {test_lab_path}")
    if do_use_agent:
        _then_assert_agent_completion_dirs(tmp_path / commons.DEFAULT_USER)


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.skip(
    reason="Hands on h2oGPTe REST-based API meant for agent runs introspection."
)
@pytest.mark.parametrize(
    "chat_session_id,chat_message_id",
    [
        (
            "b3caead5-64cf-477a-8ff6-16a32cab968c",
            "0da0a158-c2f7-4ffd-9872-70f3ee6e082e",
        ),
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_artifacts_extraction(
    tmp_path: pathlib.Path,
    h2ogpte_connection_fixture: h2o_sonar_config.ConnectionConfig,
    chat_session_id: str,
    chat_message_id: str,
):
    #
    # GIVEN
    #
    h2ogpte_connection = h2ogpte_connection_fixture
    a_rag_client = agents.H2oGpteAgentHost(
        agent_connection=h2ogpte_connection,
        logger=loggers.SonarPrintLogger(),
        log_name="agent artifacts extractor",
    )
    # lookup valid chat message ID if not provided
    if not chat_message_id:
        from h2ogpte import rest_sync

        with rest_sync.ApiClient(a_rag_client.agent_client_config) as api_client:
            chat_api = rest_sync.ChatApi(api_client)

            # CHAT MESSAGES for given chat session
            chat_messages = chat_api.get_chat_session_messages(
                chat_session_id,
                offset=0,
                limit=3,
            )
            if not chat_messages:
                raise ValueError(
                    f"No chat messages found for chat session ID: {chat_session_id}"
                )
            for m in chat_messages:
                chat_message_id = m.id
                print(f"Using chat message ID: {chat_message_id}")
                break

    #
    # WHEN
    #
    chat_session_dir = a_rag_client.extract_chat_message_artifacts(
        chat_session_id=chat_session_id,
        chat_message_id=chat_message_id,
        base_dir=tmp_path,
    )

    #
    # THEN
    #
    print(f"Chat session artifacts dir: file://{chat_session_dir}")
    assert chat_session_dir.exists() and chat_session_dir.is_dir()
    extracted_files = list(chat_session_dir.glob("*"))
    print(f"Extracted files: {extracted_files}")
    assert len(extracted_files) > 0, "No artifacts were extracted."


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.skip(
    reason="Hands on h2oGPTe REST-based API meant for agent runs introspection."
)
@pytest.mark.parametrize(
    "chat_session_id",
    [
        "b3caead5-64cf-477a-8ff6-16a32cab968c",
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_h2ogpte_agentic_api(
    h2ogpte_connection_fixture: h2o_sonar_config.ConnectionConfig, chat_session_id: str
):
    #
    # GIVEN
    #
    h2ogpte_connection = h2ogpte_connection_fixture
    rag_client = agents.H2oGpteAgentHost(
        agent_connection=h2ogpte_connection,
        logger=loggers.SonarPrintLogger(),
        log_name="agent artifacts extractor",
    )
    fail_fast = False
    verbose = True
    logger = loggers.SonarPrintLogger()

    agent_dir_name = ".work"

    #
    # WHEN
    #
    from h2ogpte import rest_sync

    with rest_sync.ApiClient(rag_client.agent_client_config) as api_client:
        chat_api = rest_sync.ChatApi(api_client)

        # CHAT MESSAGES for given chat session
        chat_messages = chat_api.get_chat_session_messages(
            chat_session_id,
            offset=0,
            limit=3,
        )
        if fail_fast and not chat_messages:
            raise ValueError(
                f"No chat messages found for chat session ID: {chat_session_id}"
            )
        logger.info(
            f"Found {len(chat_messages)} chat messages for chat session ID: "
            f"{chat_session_id}"
        )
        if chat_messages:  # TODO verbose
            for e, m in enumerate(chat_messages):
                logger.debug(f"{e + 1}. chat message {m.id}: {m.to_dict()}")
                agent_venv_dir = None
                if m.type_list:
                    # iterate chat message's meta items / artifacts
                    for meta_item in m.type_list:
                        logger.debug(
                            f"   chat message meta item: "
                            f"{json.dumps(meta_item.to_dict(), indent=2)}"
                        )
                        if meta_item.message_type in ["agent_meta"]:
                            try:
                                meta_item_json = json.loads(meta_item.content)
                                if (
                                    meta_item_json
                                    and "agent_venv_dir" in meta_item_json
                                ):
                                    agent_venv_dir = meta_item_json["agent_venv_dir"]
                            except Exception as ex:
                                logger.warning(
                                    f"Cannot parse agent_meta content as JSon: "
                                    f"{meta_item.content}. Exception: {ex}\n"
                                    f"{traceback.format_exc()}"
                                )

                        # well known message types
                        extension = ".txt"
                        if meta_item.content and (
                            meta_item.content.startswith("[")
                            or meta_item.content.startswith("{")
                        ):
                            extension = ".json"
                        meta_item_filename = (
                            f"{meta_item.message_type}_{m.id}{extension}"
                        )
                        if agent_venv_dir:
                            logger.info(
                                f"      chat message meta item agent venv dir: "
                                f"{agent_venv_dir}"
                            )
                        logger.info(
                            f"      chat message meta item filename: "
                            f"{meta_item_filename}"
                        )

        # FILES created by agents
        chat_session_files = chat_api.list_agent_server_files(chat_session_id)
        if fail_fast and not chat_session_files:
            raise ValueError(
                f"No agent files found for chat session ID: {chat_session_id}"
            )

        logger.info(
            f"Found {len(chat_session_files)} agent files for chat "
            f"session ID: {chat_session_id}"
        )
        if verbose and chat_session_files:
            for e, f in enumerate(chat_session_files):
                logger.debug(f"{e + 1}. agent file: {f.to_json()}")

        # DIRS + their FILES created by agents
        detail_level = 1  # must be 0 or 1 (int), 1000 returns NOTHING
        chat_session_dirs_files = chat_api.list_all_agent_server_directories_stats(
            chat_session_id,
            detail_level=detail_level,
        )
        if fail_fast and not chat_session_dirs_files:
            raise ValueError(
                f"No agent directories found for chat session ID: {chat_session_id}"
            )
        logger.info(
            f"Found {len(chat_session_dirs_files)} agent directories for chat "
            f"session ID on detail level {detail_level}: {chat_session_id}"
        )
        if verbose and chat_session_dirs_files:
            for e, d in enumerate(chat_session_dirs_files):
                logger.debug(f"{e + 1}. agent dir: {json.dumps(d.to_dict(), indent=2)}")

    # DIRS w/ FILEs having file stats
    api_response = chat_api.list_all_agent_server_directories_stats(
        chat_session_id, detail_level=detail_level
    )
    if fail_fast and not api_response:
        raise ValueError(
            f"No agent dir STATs found for chat session ID: {chat_session_id}"
        )
    logger.info(
        f"Found {len(api_response)} agent dir STATs for chat session ID: "
        f"{chat_session_id}"
    )
    if verbose and api_response:
        for e, d in enumerate(api_response):
            logger.debug(
                f"{e + 1}. agent dir stat: {json.dumps(d.to_dict(), indent=2)}"
            )

    # 1 DIR w/ FILES and their file stats: .work and .venv are expected to be there
    api_response = chat_api.get_agent_server_directory_stats(
        chat_session_id, agent_dir_name, detail_level=detail_level
    )
    if fail_fast and not api_response:
        raise ValueError(
            f"No agent 1 dir STAT found for chat session ID: {chat_session_id}"
        )
    logger.info(f"Found agent 1 dir STAT for chat session ID: {chat_session_id}")
    logger.debug(f"Agent 1 dir stat: {json.dumps(api_response.to_dict(), indent=2)}")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
