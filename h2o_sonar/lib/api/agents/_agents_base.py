# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import abc
import pathlib

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import loggers


class AgentHost(abc.ABC):
    """Abstract class for hosting agents which can be used to perturb the text."""

    CAT_TEST_LAB = "test_lab"
    CAT_AGENT_ARTIFACTS = "agent_artifacts"
    CAT_AGENT_HOST = "host"
    CAT_AGENT_SESSION = "session"
    CAT_AGENT_MSG = "message"

    @property
    def agent_connection_key(self):
        return self._agent_connection_key

    @property
    def agent_connection(self):
        return self._agent_connection

    @property
    def agent_llm_model_name(self):
        return self._agent_llm_model_name

    @property
    def agent_client(self):
        return self._agent_client

    def __init__(
        self,
        agent_connection_key: str = "",
        agent_connection: h2o_sonar_config.ConnectionConfig | None = None,
        agent_llm_model_name: str = "",
        agent_client=None,
        cfg_connection_key: str = "",
        cfg_llm_model_name: str = "",
        llm_only: bool = False,
        config: h2o_sonar_config.H2oSonarConfig | None = None,
        logger: loggers.SonarLogger | None = None,
        log_name: str = "agent host consumer",
    ):
        """Initialize the agent host with the given arguments.

        Parameters
        ----------
        agent_connection_key : str
            Optional key of the agent connection to be used.
        agent_connection : h2o_sonar_config.ConnectionConfig | None
            Optional connection to be used for the agent.
        agent_llm_model_name : str
            Optional LLM model name to be used by the agent.
        agent_client :
            Optional client to be used for the agent.
        cfg_connection_key : str
            Configured/parameter specified key of the agent connection - if available.
        cfg_llm_model_name : str
            Configured/parameter specified LLM model name to be used by the agent - if
            available.
        llm_only : bool
            Whether to use only LLM, not agent for the perturbations.
        config : h2o_sonar_config.H2oSonarConfig | None
            Optional H2O Sonar configuration to be used instead of the default one.
        logger : loggers.SonarLogger | None
            Optional logger to be used.
        log_name : str
            Name of the agent host consumer for logging purposes

        """
        self._agent_connection_key = agent_connection_key or ""
        self._agent_connection = agent_connection
        self._agent_llm_model_name = agent_llm_model_name
        self._agent_client = agent_client

        self._cfg_connection_key = cfg_connection_key
        self._cfg_llm_model_name = cfg_llm_model_name

        self._llm_only = llm_only

        self.config = config or h2o_sonar_config.config

        self.logger = logger or loggers.SonarPrintLogger()
        self.log_name = log_name

        if self._agent_connection and not self._agent_connection_key:
            self._agent_connection_key = self._agent_connection.key

    @abc.abstractmethod
    def agent_health_check(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def extract_chat_message_artifacts(
        self,
        base_dir: pathlib.Path,
        chat_session_id: str,
        chat_message_id: str = "",
        fail_fast: bool = False,
        verbose: bool = False,
    ) -> pathlib.Path | None:
        """Extracts all artifacts created by the agent(s) in order to respond to
        the question within the given chat message.

        Parameters
        ----------
        base_dir : pathlib.Path
            Base directory where are stored chat messages artifacts. This method will
            create a subdirectory for the given chat message ID and store all
            artifacts there.
        chat_session_id : str
            Chat session ID.
        chat_message_id : str
            Chat message ID within the chat session.
        fail_fast : bool
            If True, the method will raise an exception if no chat messages are found
            for the given chat session ID. If False, it will return None.
        verbose : bool
            If True, the method will log detailed information about the extraction
            process.

        Returns
        -------
        pathlib.Path | None
            Path to the directory where are stored all artifacts created by the agent(s)
            in order to respond to the question within the given chat message.

        """
        raise NotImplementedError
