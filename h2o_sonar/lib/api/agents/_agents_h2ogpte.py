# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import pathlib
import traceback

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import loggers
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api.agents import _agents_base
from h2o_sonar.lib.integrations import genai


try:
    from h2ogpte import rest_sync

    HAS_H2OGPTE = True
except ImportError:
    HAS_H2OGPTE = False


class H2oGpteAgentHost(_agents_base.AgentHost):
    """h2oGPTe as agent host which can be used to perturb the text."""

    COLLECTION_NAME_AGENT_EVAL = "Ephemeral agent-based eval workflows collection"

    def __init__(
        self,
        agent_connection_key: str = "",
        agent_connection: h2o_sonar_config.ConnectionConfig | None = None,
        agent_llm_model_name: str = "",
        agent_client: genai.H2oGpteRagClient | None = None,
        h2ogpte_collection_id: str = "",
        cfg_connection_key: str = "",
        cfg_llm_model_name: str = "",
        cfg_collection_id: str = "",
        llm_only: bool = False,
        logger: loggers.SonarLogger | None = None,
        log_name: str = "agent host consumer",
    ):
        _agents_base.AgentHost.__init__(
            self,
            agent_connection_key=agent_connection_key,
            agent_connection=agent_connection,
            agent_llm_model_name=agent_llm_model_name,
            agent_client=agent_client,
            cfg_connection_key=cfg_connection_key,
            cfg_llm_model_name=cfg_llm_model_name,
            llm_only=llm_only,
            logger=logger,
            log_name=log_name,
        )

        this = H2oGpteAgentHost

        self._h2ogpte_collection_id = h2ogpte_collection_id
        self._h2ogpte_collection_name = this.COLLECTION_NAME_AGENT_EVAL

        self._cfg_h2ogpte_collection_id = cfg_collection_id

        self._agent_client_config = None

    @property
    def agent_connection_key(self):
        """Resolve the agent host connection key."""

        if not self._agent_connection_key:
            available_connections = []
            # if the connection is already set, use its key
            if self._agent_connection:
                self._agent_connection_key = self._agent_connection.key
                return self._agent_connection_key

            # if the key is configured, verify it
            self._agent_connection_key = self._cfg_connection_key

            # find h2oGPTe connection in the H2O Sonar configuration
            for c in self.config.connections:
                available_connections.append(
                    f"{c.key} ({c.connection_type}): {c.name} ({c.server_url})"
                )
                if c.key == self._agent_connection_key:
                    if (
                        c.connection_type
                        != h2o_sonar_config.ConnectionConfigType.H2O_GPT_E.name
                    ):
                        raise ValueError(
                            f"Agent host connection configured with key "
                            f"'{self.agent_connection_key}' in {self.log_name} "
                            f"must be h2oGPTe connection, but it is not: "
                            f"{c.connection_type}"
                        )

                    self._agent_connection = c
                    return self._agent_connection_key

            # FALLBACK: if connection not found, use the first h2oGPTe connection
            if not self._agent_connection:
                for c in self.config.connections:
                    if (
                        c.connection_type
                        == h2o_sonar_config.ConnectionConfigType.H2O_GPT_E.name
                    ):
                        self._agent_connection_key = c.key
                        self._agent_connection = c
                        return self._agent_connection_key

            if not self._agent_connection_key:
                conns = (
                    ", ".join(available_connections) if available_connections else "0"
                )
                raise ValueError(
                    f"Required h2oGPTe connection key was not specified as the "
                    f"{self.log_name} parameter - unable to create the client and "
                    f"perform health check. Available connections: {conns}"
                )

        return self._agent_connection_key

    @property
    def agent_connection(self):
        if self._agent_connection is None:
            agent_connection_key = self.agent_connection_key

            # find h2oGPTe connection in the H2O Sonar configuration
            if agent_connection_key:
                for c in self.config.connections:
                    if c.key == agent_connection_key:
                        if (
                            c.connection_type
                            != h2o_sonar_config.ConnectionConfigType.H2O_GPT_E.name
                        ):
                            raise ValueError(
                                f"Agent host connection configured with key "
                                f"'{self.agent_connection_key}' in {self.log_name} "
                                f"must be h2oGPTe connection, but it is not: "
                                f"{c.connection_type}"
                            )

                        self._agent_connection = c
                        break

            # if not desired connection found, use the first h2oGPTe connection
            if not self._agent_connection:
                for c in self.config.connections:
                    if (
                        c.connection_type
                        == h2o_sonar_config.ConnectionConfigType.H2O_GPT_E.name
                    ):
                        self._agent_connection_key = c.key
                        self._agent_connection = c
                        break

            # if no h2oGPTe connection found, raise an error
            if not self._agent_connection:
                raise ValueError(
                    f"No h2oGPTe connection found in the configuration - unable to "
                    f"use h2oGPTe as agent host: configured connection key="
                    f"'{self.agent_connection_key}' (config also does not contain "
                    f"any h2oGPTe connection)"
                )

        return self._agent_connection

    @property
    def agent_client(self):
        # run custom health check
        if not self._agent_client:
            self._agent_client = genai.H2oGpteRagClient(
                connection=self.agent_connection, logger=self.logger
            )

        return self._agent_client

    @property
    def agent_client_config(self):
        if not self._agent_client_config:
            if not HAS_H2OGPTE:
                commons.raise_opt_import_err("h2ogpte")

            server_url = (
                self.agent_connection.server_url[:-1]
                if (
                    self.agent_connection.server_url
                    and self.agent_connection.server_url.endswith("/")
                )
                else self.agent_connection.server_url
            )
            self._agent_client_config = rest_sync.Configuration(
                host=f"{server_url}/api/v1",
                access_token=self.agent_connection.token,
            )
            self._agent_client_config.verify_ssl = (
                h2o_sonar_config.config.http_ssl_cert_verify
            )

        return self._agent_client_config

    def h2ogpte_collection(
        self, collection_name: str = COLLECTION_NAME_AGENT_EVAL
    ) -> str:
        """Create h2oGPTe collection for the agent-based evaluation. The collection
        has no corpus so that it does not affect the agent's evaluation.

        Parameters
        ----------
        collection_name : str, optional
            Name of the collection to be created.

        Returns
        -------
        str
            ID of the created h2oGPTe collection.

        """
        if not self._h2ogpte_collection_id:
            if self._cfg_h2ogpte_collection_id:
                self._h2ogpte_collection_id = self._cfg_h2ogpte_collection_id
            else:
                # create / lookup the collection
                (self._h2ogpte_collection_id, _) = self.agent_client.create_collection(
                    doc_paths=[], collection_name=collection_name
                )
                self._h2ogpte_collection_name = collection_name

        return self._h2ogpte_collection_id

    @property
    def agent_llm_model_name(self):
        if not self._agent_llm_model_name:
            if self._cfg_llm_model_name:
                self._agent_llm_model_name = self._cfg_llm_model_name
            else:
                # find the LLM model name to be used by the agent
                llm_model_names = self.agent_client.list_llm_model_names()
                if not llm_model_names:
                    raise ValueError(
                        f"No LLM models found on the h2oGPTe agent host "
                        f"{self.agent_connection.name}"
                    )
                llm_model_claude = ""
                llm_model_4o = ""
                llm_model_llama = ""
                for llm_model_name in llm_model_names:
                    if "claude" in llm_model_name.lower():
                        llm_model_claude = llm_model_name
                        if "sonnet" in llm_model_name.lower():
                            self._agent_llm_model_name = llm_model_name
                            return self._agent_llm_model_name
                    elif "4o" in llm_model_name.lower():
                        if llm_model_4o:
                            if "mini" in llm_model_4o.lower():
                                self._agent_llm_model_name = llm_model_name
                        else:
                            llm_model_4o = llm_model_name
                    elif "llama" in llm_model_name.lower():
                        if llm_model_llama:
                            if "405" in llm_model_llama.lower():
                                self._agent_llm_model_name = llm_model_name
                        else:
                            llm_model_llama = llm_model_name

                if llm_model_4o:
                    self._agent_llm_model_name = llm_model_4o
                elif llm_model_claude:
                    self._agent_llm_model_name = llm_model_claude
                elif llm_model_llama:
                    self._agent_llm_model_name = llm_model_llama
                else:
                    self._agent_llm_model_name = llm_model_names[0]

        return self._agent_llm_model_name

    def agent_health_check(self) -> bool:
        try:
            return self.agent_client.health_check(self.agent_llm_model_name)
        except Exception as ex:
            raise ValueError(
                f"h2oGPTe agent client '{self.agent_connection.name}' and LLM model "
                f"'{self.agent_llm_model_name}' health check failed: {ex}\n{traceback}"
            )

    def ask_agent(self, prompts: list[str]):
        self.logger.info(
            f"{self.log_name}: assigning agent - which will use "
            f"'{self._h2ogpte_collection_name}' collection and "
            f"LLM model '{self.agent_llm_model_name}' - the following prompts:\n"
            f"{prompts}"
        )

        # ensure the collection is created
        self.h2ogpte_collection()

        agent_responses = self.agent_client.ask_collection(
            prompts=prompts,
            collection_id=self._h2ogpte_collection_id,
            llm_model_name=self.agent_llm_model_name,
            include_chunks=False,
            # h2oGPTe parameters: use agent
            llm_args=(
                {
                    genai.H2oGpteRagClient.CFG_USE_AGENT: True,
                }
                if not self._llm_only
                else {}
            ),
        )

        self.logger.info(f"{self.log_name}: agent responses:\n{agent_responses}")

        return agent_responses

    @staticmethod
    def chat_msg_sequence_key(chat_message_id: str, chat_message_seq: int) -> str:
        return f"{chat_message_seq:04d}_{chat_message_id}"

    @staticmethod
    def _extract_chat_message_sandbox(
        base_dir: pathlib.Path,
        model_key: str,
        test_case_key: str,
        chat_session_id: str,
        chat_message_id: str,
        chat_message_seq: int = 0,
    ) -> pathlib.Path:
        """Creates a sandbox directory for the given chat message ID.

        Returns
        -------
        pathlib.Path
            Chat message sandbox directory path.

        """
        # <user_dir>/
        #   test_lab_<UUID>/                        <- base dir
        #     completion_of_m_<UUID>_tc_<UUID>/
        #       chat_session_<UUID>/
        #         chat_message_<UUID>/*
        completion_dir = base_dir / f"completion_of_m_{model_key}_tc_{test_case_key}"
        if not completion_dir.exists():
            completion_dir.mkdir(parents=True, exist_ok=True)
        chat_session_dir = completion_dir / f"chat_session_{chat_session_id}"
        if not chat_session_dir.exists():
            chat_session_dir.mkdir(parents=True, exist_ok=True)
        chat_message_seq_key = H2oGpteAgentHost.chat_msg_sequence_key(
            chat_message_id=chat_message_id, chat_message_seq=chat_message_seq
        )
        chat_message_dir = chat_session_dir / f"chat_message_{chat_message_seq_key}"
        if not chat_message_dir.exists():
            chat_message_dir.mkdir(parents=True, exist_ok=True)

        return chat_message_dir

    def _extract_chat_message_files(
        self, chat_message_id: str, artifacts_path: pathlib.Path
    ):
        """Download files created by the agent.

        Parameters
        ----------
        artifacts_path : pathlib.Path
            The path to the directory where to download the files.

        Returns
        -------
        list[Tuple[str, str]]
            List of tuples (document ID, document name) of the downloaded files.

        """
        downloaded_files = []
        try:
            agent_files_str = self.agent_client.client.list_chat_message_meta_part(
                message_id=chat_message_id, info_type="agent_files"
            ).content
            if not agent_files_str:
                # no files created by the agent
                return downloaded_files

            agent_files = json.loads(agent_files_str)
            self.logger.debug(
                f"Found {len(agent_files)} agent files for chat message ID: "
                f"{chat_message_id}"
            )
            for e, f in enumerate(agent_files):
                doc_id = list(f.keys())[0]
                doc_name = f[doc_id]
                self.logger.debug(
                    f"Downloading {e + 1}. agent file: '{doc_name}' (ID: {doc_id})..."
                )
                try:
                    self.agent_client.client.download_document(
                        destination_directory=artifacts_path,
                        destination_file_name=doc_name,
                        document_id=doc_id,
                    )
                    downloaded_files.append((doc_id, doc_name))
                    self.logger.debug(f"DONE download of '{doc_name}'")
                except Exception as e:
                    self.logger.error(f"Failed to download {doc_name}:\n{e}")
        except Exception as ex:
            self.logger.warning(
                f"Cannot download files created by the agent for chat message ID: "
                f"{chat_message_id}: {ex}\n{traceback.format_exc()}"
            )

        return downloaded_files

    def extract_chat_message_artifacts(
        self,
        base_dir: pathlib.Path,
        chat_session_id: str,
        chat_message_id: str = "",
        chat_message_seq: int = 0,
        model_key: str = "",
        test_case_key: str = "",
        max_messages: int = 1000,
        download_files: bool = True,
        fail_fast: bool = False,
        verbose: bool = False,
    ) -> pathlib.Path | None:
        if not HAS_H2OGPTE:
            commons.raise_opt_import_err("h2ogpte")

        # list: (chat message ID, list of ChatMessageMeta items) ... order matters
        type_lists_2_persist: list[tuple[str, list[rest_sync.ChatMessageMeta]]] = []

        with rest_sync.ApiClient(self.agent_client_config) as api_client:
            chat_api = rest_sync.ChatApi(api_client)

            # CHAT MESSAGES of given chat session
            offset = 0
            page_size = 25
            while offset < max_messages:  # safeguard max 1000 messages
                chat_messages_page = chat_api.get_chat_session_messages(
                    chat_session_id,
                    offset=offset,
                    limit=page_size,
                )
                if not chat_messages_page:
                    break
                if verbose:
                    self.logger.info(
                        f"Found {len(chat_messages_page)} chat messages for chat "
                        f"session ID: {chat_session_id}, offset: {offset}, "
                        f"page size: {page_size}"
                    )
                for m in chat_messages_page:
                    if not chat_message_id or (
                        chat_message_id and chat_message_id == m.id
                    ):
                        if m.type_list:
                            type_lists_2_persist.append((m.id, m.type_list))
                        else:
                            if verbose:
                                self.logger.warning(
                                    f"Chat message {chat_session_id}/{m.id} has no "
                                    f"meta items"
                                )
                            # IMPROVE no chat history - is there anything to persist?
                            continue
                # next page
                offset += page_size

        agent_venv_dir = None

        # persistence of the chat message METADATA artifact items as files
        target_msg_dir = None
        if type_lists_2_persist:
            for e, msg_tuple in enumerate(type_lists_2_persist):
                (m_id, type_list) = msg_tuple
                target_msg_dir = self._extract_chat_message_sandbox(
                    base_dir=base_dir,
                    model_key=model_key,
                    test_case_key=test_case_key,
                    chat_session_id=chat_session_id,
                    chat_message_id=m_id,
                    chat_message_seq=chat_message_seq,
                )

                # iterate chat message's meta items / artifacts to persist them
                for meta_item in type_list:
                    # WELL KNOWN message meta types
                    # *) detect agent_meta message type, to get agent's venv dir path
                    # from which will be DOWNLOADED documents created by the agent
                    if not agent_venv_dir and meta_item.message_type in ["agent_meta"]:
                        try:
                            meta_item_json = json.loads(meta_item.content)
                            if meta_item_json and "agent_venv_dir" in meta_item_json:
                                agent_venv_dir = meta_item_json["agent_venv_dir"]
                        except Exception as ex:
                            self.logger.warning(
                                f"Cannot parse agent_meta content as JSon: "
                                f"{meta_item.content}. Exception: {ex}\n"
                                f"{traceback.format_exc()}"
                            )

                    # PERSIST every meta type item as a file
                    ext = "txt"
                    if meta_item.content and (
                        meta_item.content.startswith("[")
                        or meta_item.content.startswith("{")
                    ):
                        ext = "json"
                    meta_item_filename = (
                        f"MSG_META_TYPE_ITEM_{meta_item.message_type}.{ext}"
                    )
                    meta_item_path = target_msg_dir / meta_item_filename
                    with open(meta_item_path, "w", encoding="utf-8") as f:
                        f.write(meta_item.content or "")

            # download & persistence of the FILES created by the chat message
            if download_files:
                self._extract_chat_message_files(
                    chat_message_id=chat_message_id,
                    artifacts_path=target_msg_dir,
                )

        # chat session directory with all chat message directories
        return target_msg_dir.parent if target_msg_dir else None
