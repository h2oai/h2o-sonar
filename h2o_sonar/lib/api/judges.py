# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import abc

from h2o_sonar import config
from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import loggers
from h2o_sonar.lib.integrations import genai


#
# Bring Your Own Judge (BYOJ)
#


class EvaluationJudge(abc.ABC):
    """Bring your own judge (BYOJ) to evaluate the quality of a model's output."""

    @abc.abstractmethod
    def evaluate(self, prompts: list[str], **kwargs) -> list:
        """Evaluate the quality of a model's output."""
        pass

    def health_check(self) -> bool:
        """Check if the judge is healthy and available."""
        self.evaluate(["If you are working normally, then answer: 1"])
        return True


class LlmEvaluationJudge(EvaluationJudge):
    """LLM judge / interrogator for evaluating the quality of a model output."""

    def __init__(
        self,
        llm_host_connection: config.ConnectionConfig,
        llm_model_name: str,
        logger: loggers.SonarLogger | None = None,
    ):
        self.llm_host_connection = llm_host_connection
        self.llm_model_name = llm_model_name
        self.client = genai.get_client_for_connection(
            self.llm_host_connection, logger or loggers.SonarPrintLogger()
        )

    def evaluate(self, prompts: list[str], **extra_params) -> list:
        """Evaluate the quality of a model's output."""

        return self.client.ask_model(
            prompts=prompts, llm_model_name=self.llm_model_name, **extra_params
        )


class RagClientEvaluationJudge(EvaluationJudge):
    """RAG judge / interrogator for evaluating the quality of a model output."""

    def __init__(
        self, client: genai.RagClient, llm_model_name: str, collection_id: str = ""
    ):
        self.client = client
        self.llm_model_name = llm_model_name
        self.collection_id = collection_id

    def evaluate(self, prompts: list[str], **extra_params) -> list:
        """Evaluate the quality of a model's output."""

        if self.collection_id:
            return self.client.ask_collection(
                prompts=prompts, llm_model_name=self.llm_model_name, **extra_params
            )

        return self.client.ask_model(
            prompts=prompts, llm_model_name=self.llm_model_name, **extra_params
        )


def get_evaluation_judge_for_connection(
    connection: config.ConnectionConfig,
    judge_type: str,  # h2o_sonar_config.EvaluationJudgeType
    llm_model_name: str,
    collection_id: str = "",
    logger: loggers.SonarLogger | None = None,
):
    """Get an evaluation judge for the given connection and judge type."""

    if judge_type in [
        h2o_sonar_config.EvaluationJudgeType.h2ogpte.name,
        h2o_sonar_config.EvaluationJudgeType.h2ogpte_llm.name,
        h2o_sonar_config.EvaluationJudgeType.openai_llm.name,
        h2o_sonar_config.EvaluationJudgeType.openai_rag.name,
        h2o_sonar_config.EvaluationJudgeType.ollama.name,
    ]:
        if not collection_id and judge_type in [
            h2o_sonar_config.EvaluationJudgeType.h2ogpte,
            h2o_sonar_config.EvaluationJudgeType.openai_rag,
        ]:
            raise ValueError(
                f"Collection ID is required for RAG judge type {judge_type}"
            )

        if connection.connection_type in [
            config.ConnectionConfigType.H2O_GPT_E.name,
            config.ConnectionConfigType.OPENAI_RAG.name,
            config.ConnectionConfigType.OPENAI_CHAT.name,
        ]:
            client = genai.get_client_for_connection(
                connection=connection, logger=logger or loggers.SonarPrintLogger()
            )
            return RagClientEvaluationJudge(
                client=client,
                llm_model_name=llm_model_name,
                collection_id=collection_id,
            )

        raise ValueError(
            f"Invalid connection type for judge type {judge_type}: "
            f"{connection.connection_type}"
        )

    if judge_type in [
        h2o_sonar_config.EvaluationJudgeType.h2ogpt.name,
        h2o_sonar_config.EvaluationJudgeType.h2ollmops.name,
        h2o_sonar_config.EvaluationJudgeType.azure_openai_llm.name,
        h2o_sonar_config.EvaluationJudgeType.anthropic_llm.name,
        h2o_sonar_config.EvaluationJudgeType.ollama.name,
    ]:
        return LlmEvaluationJudge(
            llm_host_connection=connection,
            llm_model_name=llm_model_name,
            logger=logger,
        )

    if judge_type == h2o_sonar_config.EvaluationJudgeType.custom:
        raise NotImplementedError

    raise ValueError(f"Unknown judge type: {judge_type}")


def get_evaluation_judge_for_config(
    judge_config: h2o_sonar_config.EvaluationJudgeConfig,
    logger: loggers.SonarLogger | None = None,
):
    """Get an evaluation judge for the given judge config."""

    connection = h2o_sonar_config.config.get_connection(judge_config.connection.key)
    if not connection:
        raise ValueError(f"Connection {judge_config.connection} not found")

    return get_evaluation_judge_for_connection(
        connection=judge_config.connection,
        judge_type=judge_config.judge_type,
        llm_model_name=judge_config.llm_model_name,
        collection_id=judge_config.collection_id,
        logger=logger,
    )


def get_default_evaluation_judge(
    logger: loggers.SonarLogger | None = None,
):
    """Get the default evaluation judge - OpenAI GPT-4 LLM model. If the OpenAI API key
    is not set, then raise exception.

    """
    if not genai.OPENAI_LLM_CONNECTION_CFG.token:
        raise ValueError(
            "Unable to create default evaluation judge OpenAI API key is not set."
        )

    h2o_sonar_config.config.add_connection(genai.OPENAI_JUDGE_CFG.connection)

    return get_evaluation_judge_for_config(
        judge_config=genai.OPENAI_JUDGE_CFG,
        logger=logger,
    )
