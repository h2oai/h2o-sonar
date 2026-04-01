# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import abc
import collections
import contextlib
import copy
import dataclasses
import datetime
import enum
import hashlib
import json
import os
import pathlib
import re
import signal
import time
import traceback
from concurrent import futures
from typing import Any
from typing import Literal
from typing import TYPE_CHECKING
from typing import TypedDict
from typing import TypeVar

import cachetools
import requests
import retrying

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import loggers
from h2o_sonar.lib.api import commons


try:
    import openai
    import packaging

    HAS_PKG_OPENAI = True
except ImportError:
    HAS_PKG_OPENAI = False

try:
    import opensearchpy

    HAS_PKG_OPENSEARCHPY = True
except ImportError:
    HAS_PKG_OPENSEARCHPY = False

try:
    import boto3

    HAS_PKG_BOTO3 = True
except ImportError:
    HAS_PKG_BOTO3 = False

try:
    import h2ogpte

    HAS_PKG_H2OGPTE = True
except ImportError:
    HAS_PKG_H2OGPTE = False

try:
    from h2ogpt_client import Client

    HAS_PKG_H2OGPT_CLIENT = True
except ImportError:
    HAS_PKG_H2OGPT_CLIENT = False

try:
    import anthropic

    HAS_PKG_ANTHROPIC = True
except ImportError:
    HAS_PKG_ANTHROPIC = False


if TYPE_CHECKING:
    try:
        from pydantic import BaseModel as _PydanticBaseModel
    except ImportError:

        class _PydanticBaseModel:
            pass

    # TBaseModel always bound to Pydantic base type (real or stub)
    TBaseModel = TypeVar("TBaseModel", bound=_PydanticBaseModel)
else:
    try:
        import pydantic

        HAS_PKG_PYDANTIC = True
        TBaseModel = TypeVar("TBaseModel", bound=pydantic.BaseModel)
    except ImportError:
        HAS_PKG_PYDANTIC = False

        class _RuntimeDummyBaseModel:
            pass

        TBaseModel = TypeVar("TBaseModel", bound=_RuntimeDummyBaseModel)

"""H2O Sonar integrations with LLM and RAG products."""

KEY_OPENAI_API_KEY = "OPENAI_API_KEY"
KEY_ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"

OPENAI_LLM_CONNECTION_CFG = h2o_sonar_config.ConnectionConfig(
    connection_type=h2o_sonar_config.ConnectionConfigType.OPENAI_CHAT.name,
    name="OpenAI Chat",
    description="OpenAI AI chat API.",
    # server_url is resolved internally by OpenAI client
    token=os.getenv(KEY_OPENAI_API_KEY),
    token_use_type=h2o_sonar_config.TokenUseType.API_KEY.name,
)

OPENAI_JUDGE_CFG = h2o_sonar_config.EvaluationJudgeConfig(
    name="OpenAI LLM Judge",
    description="OpenAI LLM evaluation judge.",
    judge_type=h2o_sonar_config.EvaluationJudgeType.openai_llm.name,
    connection=OPENAI_LLM_CONNECTION_CFG,
    llm_model_name="gpt-4",  # alias for the latest model
)

ANTHROPIC_LLM_CONNECTION_CFG = h2o_sonar_config.ConnectionConfig(
    connection_type=h2o_sonar_config.ConnectionConfigType.ANTHROPIC_CHAT.name,
    name="Anthropic Claude Chat",
    description="Anthropic Claude AI chat API.",
    # server_url is resolved internally by Anthropic client
    token=os.getenv(KEY_ANTHROPIC_API_KEY),
    token_use_type=h2o_sonar_config.TokenUseType.API_KEY.name,
)


class RagChunkRetrievalMethod(enum.Enum):
    # find chunks related to a message using lexical search (search_chunks)
    LEXICAL = enum.auto()
    # get chunks by (back) references of the answer (list_chat_message_references)
    ANSWER_REFS = enum.auto()


class TimeoutRetryExpBackoffCtx:
    """Exponential backoff context for the timeout handling. This context is meant
    to be used in RAG/LLM clients on retries - timeout is increased on each retry by
    the backoff factor.

    """

    BACKOFF_FACTOR: float = 4.0
    MIN_BACKOFF_SECS: float = 5.0
    MAX_BACKOFF_SECS: float = 7.0 * 60.0

    @property
    def timeout(self) -> float:
        return self._timeout

    def __init__(
        self,
        backoff_factor: float = BACKOFF_FACTOR,
        min_backoff_secs: float = MIN_BACKOFF_SECS,
        max_backoff_secs: float = MAX_BACKOFF_SECS,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        backoff_factor : float
            The backoff factor to apply between retries if the exponential backoff
            strategy is used. Example: consider 5 retries, with 5s min and
            max 7' (420s). With factor 4.0 the backoff times will be:
            5s, 20s, 80s, 320s, 420s (1280s override by max).
        min_backoff_secs : int
            The minimum backoff time in seconds if the exponential backoff strategy
            is used.
        max_backoff_secs : int
            The maximum backoff time in seconds if the exponential backoff strategy
            is used.

        """
        self.backoff_factor = backoff_factor
        self.min_backoff_secs = min_backoff_secs
        self.max_backoff_secs = max_backoff_secs

        self._timeout = self.min_backoff_secs

    def reset(self) -> None:
        """Reset the timeout to the initial value."""
        self._timeout = self.min_backoff_secs

    def retry(self) -> float:
        """Call this method on retry to recalculate the timeout."""
        self._timeout = min(self.max_backoff_secs, self._timeout * self.backoff_factor)
        return self._timeout

    @staticmethod
    def copy(ctx: "TimeoutRetryExpBackoffCtx") -> "TimeoutRetryExpBackoffCtx":
        # NOTE: _timeout is not copied, but reset to the initial value
        return TimeoutRetryExpBackoffCtx(
            backoff_factor=ctx.backoff_factor,
            min_backoff_secs=ctx.min_backoff_secs,
            max_backoff_secs=ctx.max_backoff_secs,
        )


class LlmHostClient(abc.ABC):
    """A LLM host product client."""

    """LLM host or RAG answer."""
    LlmRagAnswer = collections.namedtuple(
        "LlmRagAnswer",
        [
            "prompt",
            "answer",
            "duration",
            "context",
            "cost",
            "chat_session_id",
            "chat_message_id",
        ],
    )

    @property
    def client(self):
        raise NotImplementedError

    @staticmethod
    def config_factory() -> dict:
        """Get the prototype of the configuration for the client - it can be used as
        reflection of the parameters names which might be passed as custom
        configuration of the client. The prototype dictionary is type safe for
        de/serializable to JSon. It is expected that users will use just the keys
        which they need to set and will skip the rest.

        Returns
        -------
        dict :
            Prototype of the configuration for the client.

        """

        return {}

    @abc.abstractmethod
    def list_llm_model_names(self):
        raise NotImplementedError

    @abc.abstractmethod
    def ask_model(
        self,
        prompts: list[str],
        llm_model_name: str = "",
        **extra_params,
    ) -> list:
        raise NotImplementedError

    @abc.abstractmethod
    def ask_model_structured(
        self,
        prompts: list[str],
        output_structure: type[TBaseModel],
        llm_model_name: str = "",
        **extra_params,
    ) -> list[TBaseModel]:
        raise NotImplementedError

    def health_check(self, llm_model_name: str) -> bool:
        """Check if the judge is healthy and available."""
        self.ask_model(
            prompts=["If you are working normally, then answer: 1"],
            llm_model_name=llm_model_name,
        )
        return True

    @staticmethod
    def _get_connection_timeout(
        connection: h2o_sonar_config.ConnectionConfig,
    ) -> float | None:
        if (
            connection
            and connection.extra_params
            and "timeout" in connection.extra_params
        ):
            try:
                return float(connection.extra_params["timeout"])
            except Exception as ex:
                print(
                    f"Invalid timeout value passed from the connection - expected "
                    f"float, but got '{connection.extra_params['timeout']}' "
                    f"({type(connection.extra_params['timeout'])})"
                    f": {ex}"
                )
                return None

        return None


class RagClient(LlmHostClient, abc.ABC):
    """A RAG product client."""

    # system limit for the number of chunks to be retrieved (avoid huge responses)
    CHUNKS_LIMIT = 42

    @staticmethod
    def get_collection_name(doc_paths: list[str | pathlib.Path]) -> str:
        document_names = [
            p.name if isinstance(p, pathlib.Path) else p for p in doc_paths
        ]
        return f"Ephemeral H2O Sonar RAG collection (docs: {document_names})"

    @abc.abstractmethod
    def create_collection(
        self,
        doc_paths: list[pathlib.Path | str],
        collection_name: str = "",
        **kwargs,
    ) -> tuple[str, str]:
        raise NotImplementedError

    @abc.abstractmethod
    def list_collections(self, offset: int = 0, limit: int = 10):
        raise NotImplementedError

    @abc.abstractmethod
    def purge_collections(self, collection_ids: list[str] | None = None):
        raise NotImplementedError

    @abc.abstractmethod
    def purge_uploaded_docs(self, document_ids: list[str] | None = None):
        raise NotImplementedError

    @abc.abstractmethod
    def ask_collection(
        self,
        collection_id: str,
        prompts: list[str],
        llm_model_name: str = "",
        include_chunks: int = 0,
        chunk_retrieval_method: str = RagChunkRetrievalMethod.ANSWER_REFS.name,
        # extra parameters
        **kwargs,
    ):
        raise NotImplementedError


"""Typed h2oGPTe/h2oGPT hosted LLM arguments dictionary.

Attributes
----------
temperature : float
    The value used to modulate the next token probabilities.
    Most deterministic: 0, Most creative: 1
seed : int
    The seed for the random number generator, only used if temperature > 0,
    seed=0 will pick a random number for each call, seed > 0 will be fixed.
top_k : int
    The number of highest probability vocabulary tokens to keep for top-k-filtering.
top_p : float
    If set to float < 1, only the smallest set of most probable tokens with
    probabilities that add up to top_p or higher are kept for generation.
repetition_penalty : float
    The parameter for repetition penalty. 1.0 means no penalty.
max_new_tokens : int
    Maximum number of new tokens to generate. This limit applies to each
    (map+reduce) step during summarization and each (map) step during extraction.
min_max_new_tokens : int
    Minimum value for max_new_tokens when auto-adjusting for content of
    prompt, docs, etc.

See ``h2ogpt::ask_question()`` for more details

"""


class TypedH2ogptLlmConfigDict(TypedDict):
    temperature: float
    seed: int
    top_k: int
    top_p: float
    repetition_penalty: float
    max_new_tokens: int
    min_max_new_tokens: int


class H2oGpteRagClient(RagClient):
    """h2oGPTe RAG client."""

    # special h2oGPTe model selectors when asking the collection
    MODEL_SPEC_AUTO = "auto"  # let h2oGPTe server select the model
    MODEL_SPEC_COL = "llm-inherited-from-collection"
    MODEL_SPEC_COL_OPT_E = ""  # inherit LLM from the collection configuration
    MODEL_SPEC_COL_OPT_N = None  # inherit LLM from the collection configuration

    DEFAULT_TIMEOUT = 7 * 60  # 7'
    DEFAULT_AGENT_TIMEOUT = 2 * DEFAULT_TIMEOUT

    @property
    def client(self):
        return self._client or self._create_client()

    def __init__(
        self,
        connection: h2o_sonar_config.ConnectionConfig,
        logger: loggers.SonarLogger | None = None,
    ):
        """h2oGPTe RAG client constructor.

        Parameters
        ----------
        connection : h2o_sonar.config.ConnectionConfig
            h2oGPTe connection configuration.
        logger : loggers.SonarLogger | None
            Optional logger.

        """
        if not connection:
            raise ValueError("h2oGPTe connection configuration is empty.")
        if (
            not connection.connection_type
            == h2o_sonar_config.ConnectionConfigType.H2O_GPT_E.name
        ):
            raise ValueError(
                f"Provide h2oGPTe connection - connection type "
                f"'{connection.connection_type}' is not supported by H2oGpteRagClient."
            )
        if not connection.token:
            raise ValueError(
                "API key is required to be set as 'token' on the connection, but it is "
                "empty"
            )

        self.connection = connection
        self._client = None
        # map: collection id (str) -> collection object
        self._created_collections = {}
        # map: upload id (str) -> upload object
        self._uploaded_documents = {}
        self.logger = logger or loggers.SonarPrintLogger()

    CFG_EMBEDDING_MODEL = "embedding_model"
    CFG_PROMPT_TEMPLATE_ID = "prompt_template_id"
    CFG_SYSTEM_PROMPT = "system_prompt"
    CFG_PRE_PROMPT_QUERY = "pre_prompt_query"
    CFG_PROMPT_QUERY = "prompt_query"
    CFG_PRE_PROMPT_SUMMARY = "pre_prompt_summary"
    CFG_PROMPT_SUMMARY = "prompt_summary"
    CFG_LLM = "llm"
    CFG_LLM_ARGS = "llm_args"
    CFG_USE_AGENT = "use_agent"  # llm_args/use_agent
    CFG_TEMPERATURE = "temperature"  # llm_args/temperature
    CFG_SELF_REFLECTION_CONFIG = "self_reflection_config"
    CFG_RAG_CONFIG = "rag_config"
    CFG_TIMEOUT = "timeout"
    CFG_TEXT_CONTEXT_LIST = "text_context_list"
    CFG_CHAT_CONVERSATION = "chat_conversation"

    TypedRagConfigDict = TypedDict(
        "TypedRagConfigDict",
        {
            CFG_EMBEDDING_MODEL: str | None,
            CFG_PROMPT_TEMPLATE_ID: str | None,
            CFG_SYSTEM_PROMPT: str | None,
            CFG_PRE_PROMPT_QUERY: str | None,
            CFG_PROMPT_QUERY: str | None,
            CFG_PRE_PROMPT_SUMMARY: str | None,
            CFG_PROMPT_SUMMARY: str | None,
            CFG_LLM: str | int | None,
            CFG_LLM_ARGS: TypedH2ogptLlmConfigDict | None,
            CFG_SELF_REFLECTION_CONFIG: dict[str, str | int] | None,
            CFG_RAG_CONFIG: dict[str, str | int] | None,
            CFG_TIMEOUT: float | None,
        },
    )

    TypedLlmConfigDict = TypedDict(
        "TypedLlmConfigDict",
        {
            CFG_SYSTEM_PROMPT: str | None,
            CFG_PRE_PROMPT_SUMMARY: str | None,
            CFG_PROMPT_QUERY: str | None,
            CFG_TEXT_CONTEXT_LIST: list[str] | None,
            CFG_LLM: str | int | None,
            CFG_LLM_ARGS: TypedH2ogptLlmConfigDict | None,
            CFG_CHAT_CONVERSATION: list[tuple[str, str]] | None,
            CFG_TIMEOUT: float | None,
        },
    )

    @staticmethod
    def config_factory(model_type: str = commons.ModelTypeExplanation.RAG) -> dict:
        """Get the prototype of the configuration for the client - it can be used as
        reflection of the parameters names which might be passed as custom
        configuration of the client.

        See: https://docs.h2o.ai/enterprise-h2ogpte/v1.4.13/guide/prompts

        Parameters
        ----------
        model_type : commons.ModelTypeExplanation
            Model type explanation - "rag" or "llm".

        """
        this = H2oGpteRagClient

        if model_type == commons.ModelTypeExplanation.RAG:
            rag_config_proto: H2oGpteRagClient.TypedRagConfigDict = {
                #
                # ARGS (args for collection creation)
                #
                this.CFG_EMBEDDING_MODEL: None,
                this.CFG_PROMPT_TEMPLATE_ID: None,
                #
                # ARGS (talk to collection - session.query())
                #
                this.CFG_SYSTEM_PROMPT: None,
                this.CFG_PRE_PROMPT_QUERY: None,
                this.CFG_PROMPT_QUERY: None,
                this.CFG_PRE_PROMPT_SUMMARY: None,
                this.CFG_PROMPT_SUMMARY: None,
                this.CFG_LLM: None,
                this.CFG_LLM_ARGS: {
                    this.CFG_TEMPERATURE: 0.0,
                    "seed": 0,
                    "top_k": 1,
                    "top_p": 1.0,
                    "repetition_penalty": 1.07,
                    "max_new_tokens": 1024,
                    "min_max_new_tokens": 512,
                },
                this.CFG_SELF_REFLECTION_CONFIG: None,
                this.CFG_RAG_CONFIG: None,
                this.CFG_TIMEOUT: None,
            }
            return rag_config_proto

        llm_config_proto: H2oGpteRagClient.TypedLlmConfigDict = {
            this.CFG_SYSTEM_PROMPT: None,
            this.CFG_PRE_PROMPT_QUERY: None,
            this.CFG_PROMPT_QUERY: None,
            this.CFG_TEXT_CONTEXT_LIST: None,
            this.CFG_LLM: None,
            this.CFG_LLM_ARGS: {
                this.CFG_TEMPERATURE: 0.0,
                "seed": 0,
                "top_k": 1,
                "top_p": 1.0,
                "repetition_penalty": 1.07,
                "max_new_tokens": 1024,
                "min_max_new_tokens": 512,
            },
            this.CFG_CHAT_CONVERSATION: None,
            this.CFG_TIMEOUT: None,
        }
        return llm_config_proto

    def _create_client(self):
        if not HAS_PKG_H2OGPTE:
            commons.raise_opt_import_err("h2ogpte")

        server_url = (
            self.connection.server_url[:-1]
            if (self.connection.server_url and self.connection.server_url.endswith("/"))
            else self.connection.server_url
        )
        self._client = h2ogpte.H2OGPTE(
            address=server_url,
            api_key=self.connection.token,
            verify=h2o_sonar_config.config.http_ssl_cert_verify,
        )

        return self._client

    def list_llm_model_names(self, retries: int = 3) -> list[str]:
        """List h2oGPTe LLM models.

        Parameters
        ----------
        retries : int
            Number of retries in case of h2oGPTe failure.

        Returns
        -------
        list[str] :
            List of h2oGPTe LLM model names.

        """
        # LEGACY: return [x["base_model"] for x in self.client.get_llms()]
        ex_msg_suffix = ""
        for i in range(1 + retries):
            try:
                return self.client.get_llm_names()
            except Exception as ex:
                ex_msg_suffix = (
                    f"from the h2oGPTe server at {self.connection.server_url}:  "
                    f"{ex}\n{traceback.format_exc()}"
                )
                self.logger.warning(
                    f"Failed to retrieve LLM names ({i + 1}/{retries} attempts) "
                    f"{ex_msg_suffix}"
                )
                time.sleep(0.2 * 2**i)
        raise RuntimeError(
            f"Unable to retrieve LLM names {ex_msg_suffix} "
            f"(1 attempt, {retries} retries)"
        )

    def _collection_url(self, collection_id: str) -> str:
        return f"{self.connection.server_url}/collections/{collection_id}"

    def create_collection(
        self,
        doc_paths: list[pathlib.Path | str],
        collection_name: str = "",
        upload_if_collection_exists: bool = True,
        model_cfg: dict | None = None,
    ) -> tuple[str, str]:
        """Create h2oGPTe collection and upload documents (corpus) to that collection.

        Parameters
        ----------
        doc_paths : list[pathlib.Path | str]
          Paths (local filesystem) to the documents to be uploaded.
        collection_name : str
          Optional parameter with the document collection use (if specified) or create
          (if the given name does not exist)
        upload_if_collection_exists : bool
            Optional parameter to upload the documents even if the collection exists.
        model_cfg : dict | None
            Optional model configuration with the following parameters:
            - embedding_model : str
            - prompt_template_id : str

        Returns
        -------

        Tuple[str, str] :
          h2oGPT Enterprise collection ID and URL.

        """
        if not doc_paths:
            self.logger.warning(
                "Paths to the documents to be uploaded are empty - creating EMPTY "
                "collection."
            )
        _doc_paths = []
        for doc_path in doc_paths:
            doc_path = pathlib.Path(doc_path)
            if not doc_path.exists():
                raise ValueError(
                    f"Path to the document to be uploaded is invalid: {doc_path}"
                )
            _doc_paths.append(doc_path)
        model_cfg = model_cfg or {}

        # find collection
        collection_id = None
        if collection_name:
            recent_collections = self.client.list_recent_collections(0, 1000)
            for c in recent_collections:
                if c.name == collection_name and (not doc_paths or c.document_count):
                    collection_id = c.id
                    break

        if not upload_if_collection_exists and collection_id:
            return collection_id, self._collection_url(collection_id)

        # create collection (if the collection w/ desired name does not exist)
        if not collection_id:
            collection_name = (
                collection_name or f"Talk to H2O Sonar {datetime.datetime.now()}"
            )
            self.logger.info(
                f"Creating new collection '{collection_name}' "
                f"at {self.connection.server_url} with docs: "
                f"{[str(p) for p in doc_paths]}..."
            )
            collection_id = self.client.create_collection(
                name=collection_name,
                description=f"{collection_name}.",
                embedding_model=model_cfg.get(
                    H2oGpteRagClient.CFG_EMBEDDING_MODEL, None
                ),
                prompt_template_id=model_cfg.get(
                    H2oGpteRagClient.CFG_PROMPT_TEMPLATE_ID, None
                ),
            )

        # upload docs into collection
        for e, doc_path in enumerate(_doc_paths):
            self.logger.info(
                f"Uploading {e + 1}/{len(_doc_paths)} document to "
                f"{self.connection.server_url} and collection '{collection_name}'..."
            )
            with open(doc_path, "rb") as f:
                upload_id = self.client.upload(doc_path.name, f)
            self._uploaded_documents[upload_id] = doc_path

            # convert the data into chunked text and embeddings
            self.logger.info(
                f"Converting data in the collection '{collection_name}' into "
                f"chunked text and embeddings..."
            )
            self.client.ingest_uploads(
                collection_id=collection_id,
                upload_ids=[upload_id],
                timeout=self._get_connection_timeout(self.connection),
            )

            self.logger.info(f"Upload of {doc_path} DONE")

        self.logger.info(
            f"ALL documents uploaded to the "
            f"collection '{collection_name}' / {collection_id}"
        )

        self._created_collections[collection_id] = collection_name

        return collection_id, self._collection_url(collection_id)

    def list_collections(self, offset: int = 0, limit: int = 1000):
        return self.client.list_recent_collections(offset=offset, limit=limit)

    def purge_collections(self, collection_ids: list[str] | None = None) -> list:
        """Purge h2oGPTe collections.

        Parameters
        ----------
        collection_ids : list[str] | None
            List of collection IDs to be purged. If the list is empty, all collections
            created by this instance are purged.

        """
        collection_ids = collection_ids or list(self._created_collections.keys())
        self.client.delete_collections(
            collection_ids=collection_ids,
            timeout=self._get_connection_timeout(self.connection),
        )
        return collection_ids

    def purge_uploaded_docs(self, document_ids: list[str] | None = None) -> list:
        """Purge h2oGPTe uploaded documents.

        Parameters
        ----------
        document_ids : list[str] | None
            List of document IDs to be purged. If the list is empty, all documents
            uploaded by this instance are purged.

        """
        document_ids = document_ids or list(self._uploaded_documents.keys())

        deleted_doc_ids = []
        for document_id in document_ids:
            try:
                self.client.delete_documents(
                    document_ids=[document_id],
                    timeout=self._get_connection_timeout(self.connection),
                )
                deleted_doc_ids.append(document_id)
            except Exception as fex:
                print(f"Failed to purge file {document_id}: {fex}")

        return deleted_doc_ids

    def _retrieve_chunks(
        self,
        question: str,
        answer,  # : h2ogpte.ChatMessage
        collection_id: str,
        chunk_retrieval_method: str = RagChunkRetrievalMethod.ANSWER_REFS.name,
        chunks_limit: int = 10,
    ) -> tuple[list, list]:
        chunks_limit = min(
            0 if chunks_limit < 0 else chunks_limit, RagClient.CHUNKS_LIMIT
        )

        chunks = []
        chunks_scores = []
        if chunk_retrieval_method == RagChunkRetrievalMethod.LEXICAL.name:
            chunks_search_result = self.client.search_chunks(
                collection_id=collection_id,
                query=question,
                topics=[],
                offset=0,
                limit=chunks_limit,
            )
            for chunk in chunks_search_result:
                self.logger.info(
                    f"Adding chunk (score={chunk.score} "
                    f"length={len(chunk.text) if chunk and chunk.text else 0})"
                )
                chunks.append(chunk.text)
                chunks_scores.append(chunk.score)
        elif chunk_retrieval_method == RagChunkRetrievalMethod.ANSWER_REFS.name:
            # KGM approach - see: h2ogpte/test_mux.py:2259/test_pdf_questions_e2e
            chunks_count = 0
            for ref in self.client.list_chat_message_references(answer.id):
                # IMPROVE ref.content seems to have chunk prefix (?) ~ compare/use
                if chunks_count >= chunks_limit:
                    break
                else:
                    chunks_count += 1

                raw_chunks = self.client.get_chunks(collection_id, [ref.chunk_id])
                for c in raw_chunks:
                    self.logger.info(
                        f"Adding chunk #{chunks_count + 1:02} for '{question[:15]}...' "
                        f"reference {ref.chunk_id:02} with score={ref.score} "
                        f"length={len(c.text) if c and c.text else 0}"
                    )
                    chunks.append(c.text)
                    chunks_scores.append(ref.score)
        else:
            raise ValueError(
                f"Chunk retrieval method '{chunk_retrieval_method}' is not supported."
            )

        return chunks, chunks_scores

    def _retrieve_answer_meta(self, answer):  # : h2ogpte.ChatMessage
        """Get h2oGPT system prompt."""
        prompt_raw = self.client.list_chat_message_meta_part(
            answer.id, "prompt_raw"
        ).content

        return prompt_raw

    def _get_24h_cost(self) -> float:
        try:
            # cost for the current user in the last 24 hours in USD
            return self.client.get_llm_usage_24h()
        except Exception as ex:
            self.logger.warning(
                f"Unable to retrieve 24h cost: {ex}\n{traceback.format_exc()}",
            )
            return 0.0

    @staticmethod
    def humanize_err_msg(
        ex: Exception, timeout_exp_backoff: TimeoutRetryExpBackoffCtx | None = None
    ) -> str:
        """Make the error messages from the h2oGPTe (client) human friendly."""
        if not ex:
            return ""

        ex_msg = str(ex)
        if "request timed out" in ex_msg.lower():
            when_msg = (
                f" after {timeout_exp_backoff.timeout:.2f}s"
                if timeout_exp_backoff and timeout_exp_backoff.timeout
                else ""
            )
            return f"Request timed out{when_msg}"

        return str(ex_msg)

    def ask_collection(
        self,
        collection_id: str,
        prompts: list[str],
        llm_model_name: str = "",
        include_chunks: int = 0,
        include_system_prompt: bool = False,
        chunk_retrieval_method: str = RagChunkRetrievalMethod.ANSWER_REFS.name,
        chat_session_id: str | None = None,
        retry_attempt: int = 0,
        retry_attempts: int = 0,
        timeout_exp_backoff: TimeoutRetryExpBackoffCtx | None = None,
        **extra_params,
    ) -> list[LlmHostClient.LlmRagAnswer]:
        """Ask h2oGPTe collection.

        Parameters
        ----------
        collection_id : str
            h2oGPTe collection ID.
        prompts : list[str]
            Prompts to ask.
        llm_model_name : str
            Optional base LLM model name.
        include_chunks : int
            Optional parameter to determine also relevant (text) chunks - lexical search
            using the given query is made.
        include_system_prompt : bool
            Optional parameter to determine if the system prompt should be included.
        chunk_retrieval_method : str
            Optional parameter to determine how to retrieve chunks. Check
            `H2oGpteChunkRetrievalMethod` for possible values.
        chat_session_id : str | None
            Optional parameter to specify the chat session ID allowing to reuse the
            same session - which uses the chat history as context i.e. stateful chat
            session / multi-turn conversation.
        retry_attempt : int
            Optional parameter to determine the retry attempt (debugging).
        retry_attempts : int
            Optional parameter to determine the number of possible retry
            attempts (debugging).
        timeout_exp_backoff : TimeoutRetryExpBackoffCtx | None
            Optional exponential backoff context for the timeout handling.
        extra_params :
            Optional parameters to be passed to the h2oGPTe client ``session.query()``.
            These parameters override the default values set in the connection
            and configuration.

        Returns
        -------
        list[LlmHostClient.LlmRagAnswer] :
          List of tuples with prompt, answer, duration and chunks.

        """
        # handle special LLM model name selectors
        if llm_model_name == H2oGpteRagClient.MODEL_SPEC_COL:
            llm_model_name = H2oGpteRagClient.MODEL_SPEC_COL_OPT_N

        # ensure LLM model name is set in the kwargs for the query() method
        if extra_params:
            extra_params = copy.deepcopy(extra_params)
            # remove keys which are not supported by the query() method
            if H2oGpteRagClient.CFG_EMBEDDING_MODEL in extra_params:
                extra_params.pop(H2oGpteRagClient.CFG_EMBEDDING_MODEL)
            if H2oGpteRagClient.CFG_PROMPT_TEMPLATE_ID in extra_params:
                extra_params.pop(H2oGpteRagClient.CFG_PROMPT_TEMPLATE_ID)

            if llm_model_name:
                extra_params["llm"] = llm_model_name
        else:
            extra_params = {"llm": llm_model_name}
            # timeout resolution: model params -> connection params -> default
            connection_timeout = self._get_connection_timeout(self.connection)
            if connection_timeout is not None:
                extra_params["timeout"] = connection_timeout

        # AGENT overrides:
        # - do NOT use exponential backoff for the agent as it may lead to h2oGPTe
        #   server being overloaded with the requests
        is_agentic_request = extra_params.get(H2oGpteRagClient.CFG_LLM_ARGS, {}).get(
            H2oGpteRagClient.CFG_USE_AGENT, False
        ) or extra_params.get(H2oGpteRagClient.CFG_USE_AGENT, False)
        if is_agentic_request:
            timeout_exp_backoff = None

        # TIMEOUT override w/ exponential backoff
        if timeout_exp_backoff:
            extra_params["timeout"] = int(timeout_exp_backoff.timeout)
        # TIMEOUT set connection timeout if it is NOT set as 1000s ~ 16' is too much
        if "timeout" not in extra_params or extra_params["timeout"] is None:
            extra_params["timeout"] = (
                H2oGpteRagClient.DEFAULT_AGENT_TIMEOUT
                if (is_agentic_request)
                else H2oGpteRagClient.DEFAULT_TIMEOUT
            )

        msg_retry = f", retry {retry_attempt}/{retry_attempts}" if retry_attempt else ""
        msg_details = f"[{llm_model_name}{msg_retry}, {datetime.datetime.now()}]"

        wip_results = []

        if not chat_session_id:
            chat_session_id = self.client.create_chat_session(collection_id)

        with self.client.connect(chat_session_id) as session:
            for i, q in enumerate(prompts):
                self.logger.debug(
                    f"\n>>>RUNNING Q{i + 1} {msg_details}: {q}\n",
                    flush=True,
                )

                cost_prev = self._get_24h_cost()
                start = time.time()
                a = session.query(q, **extra_params)
                duration = time.time() - start
                cost = self._get_24h_cost() - cost_prev

                self.logger.debug(
                    f"\n>>>Q{i + 1}: {q}\n>>>A{i + 1}"
                    f"[{llm_model_name}{msg_retry}, duration: {duration:.2f}s]: "
                    f"{a.content}\n",
                    flush=True,
                )

                wip_results.append(
                    (
                        a,
                        LlmHostClient.LlmRagAnswer(
                            prompt=q,
                            answer=a.content,
                            duration=duration,
                            context=[],  # chunks will be completed later
                            cost=cost,
                            chat_session_id=chat_session_id,
                            chat_message_id=a.id,
                        ),
                    )
                )

        if include_system_prompt:
            try:
                system_prompt = self._retrieve_answer_meta(answer=a)
            except Exception as ex:
                self.logger.warning(
                    f"Unable to retrieve answer metadata for answer "
                    f"'{a}': {ex}\n{traceback.format_exc()}",
                )
                system_prompt = ""

            self.logger.debug(
                f"\n>>>SYSTEM PROMPT: {system_prompt}\n",
                flush=True,
            )

        # RAG retrieved context chunks:
        # - chunks MUST be retrieved OUTSIDE the session context manager,
        #   otherwise it is causing random HANGS in CLOSING of the session connection
        #   by the context manager
        if include_chunks:
            results = []

            # find chunks RELATED to the answer using the requested method
            for a, result in wip_results:
                # IMPROVE: share the chunks score
                (chunks, chunks_scores) = self._retrieve_chunks(
                    question=result.prompt,
                    answer=a,
                    collection_id=collection_id,
                    chunk_retrieval_method=chunk_retrieval_method,
                    chunks_limit=include_chunks,
                )
                results.append(
                    LlmHostClient.LlmRagAnswer(
                        prompt=result.prompt,
                        answer=result.answer,
                        duration=result.duration,
                        context=chunks,
                        cost=result.cost,
                        chat_session_id=chat_session_id,
                        chat_message_id=a.id,
                    )
                )

        else:
            results = [r for _, r in wip_results]

        return results

    def ask_model_structured(
        self,
        prompts: list[str],
        output_structure: type[TBaseModel],
        llm_model_name: str = "",
        chat_session_id: str | None = None,
        retry_attempt: int = 0,
        retry_attempts: int = 0,
        timeout_exp_backoff: TimeoutRetryExpBackoffCtx | None = None,
        **extra_params,
    ) -> list[TBaseModel]:
        """Ask a h2oGPTe model with structured output.

        Parameters
        ----------
        prompts : list[str]
            Prompts to ask.
        output_structure : pydantic.BaseModel
            Pydantic model class defining the expected output structure.
        llm_model_name : str
            Optional base LLM model name.
        chat_session_id : str | None
            Optional parameter to specify the chat session ID allowing to reuse the
            same session.
        retry_attempt : int
            Optional parameter to determine the retry attempt (debugging).
        retry_attempts : int
            Optional parameter to determine the number of possible retry
            attempts (debugging).
        timeout_exp_backoff : TimeoutRetryExpBackoffCtx | None
            Optional exponential backoff context for the timeout handling.
        extra_params :
            Optional parameters to be passed to the h2oGPTe client.

        Returns
        -------
        list[pydantic.BaseModel]
            List of structured outputs parsed into the provided Pydantic model.
        """
        if not HAS_PKG_PYDANTIC:
            commons.raise_opt_import_err("pydantic")
        # ensure LLM model name is set in the kwargs for the query() method
        if extra_params:
            extra_params = copy.deepcopy(extra_params)
            if llm_model_name:
                extra_params["llm"] = llm_model_name
        else:
            extra_params = {"llm": llm_model_name}

        # TIMEOUT override w/ exponential backoff
        if timeout_exp_backoff:
            extra_params["timeout"] = int(timeout_exp_backoff.timeout)
        # TIMEOUT set connection timeout if it is NOT set as 1000s ~ 16' is too much
        if "timeout" not in extra_params or extra_params["timeout"] is None:
            extra_params["timeout"] = H2oGpteRagClient.DEFAULT_TIMEOUT

        # enable structured output via llm_args with guided_json
        if "llm_args" not in extra_params:
            extra_params["llm_args"] = {}
        extra_params["llm_args"]["guided_json"] = output_structure.model_json_schema()

        msg_retry = f", retry {retry_attempt}/{retry_attempts}" if retry_attempt else ""
        msg_details = f"[{llm_model_name}{msg_retry}, {datetime.datetime.now()}]"

        if not chat_session_id:
            chat_session_id = self.client.create_chat_session()

        results = []
        with self.client.connect(chat_session_id) as session:
            for i, q in enumerate(prompts):
                self.logger.debug(
                    f"\n>>>RUNNING Q{i + 1} {msg_details}: {q}\n",
                    flush=True,
                )

                # add JSON instruction to prompt when using guided_json
                structured_prompt = (
                    f"{q}\n\n"
                    f"Respond with ONLY valid JSON matching the provided schema. "
                    f"Do not include any explanatory text."
                )

                a = session.query(structured_prompt, **extra_params)

                # parse structured output - strip markdown code fences if present
                response_text = a.content.strip()
                if response_text.startswith("```"):
                    # remove Markdown code fences (```json...``` or ```...```)
                    lines = response_text.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]  # remove opening fence
                    if lines[-1].strip() == "```":
                        lines = lines[:-1]  # remove closing fence
                    response_text = "\n".join(lines).strip()

                parsed_output = output_structure.model_validate_json(response_text)

                self.logger.debug(
                    f"\n>>>Q{i + 1}: {q}\n>>>A{i + 1}"
                    f"[{llm_model_name}{msg_retry}]: "
                    f"{parsed_output}\n",
                    flush=True,
                )

                results.append(parsed_output)

        return results

    def ask_model(
        self,
        prompts: list[str],
        llm_model_name: str = "",
        chat_session_id: str | None = None,
        retry_attempt: int = 0,
        retry_attempts: int = 0,
        timeout_exp_backoff: TimeoutRetryExpBackoffCtx | None = None,
        **extra_params,
    ) -> list[LlmHostClient.LlmRagAnswer]:
        """Ask a h2oGPTe LLM (base) model."""
        # ensure LLM model name is set in the kwargs for the query() method
        if extra_params:
            extra_params = copy.deepcopy(extra_params)
            if llm_model_name:
                extra_params["llm"] = llm_model_name
        else:
            extra_params = {"llm": llm_model_name}

        # TIMEOUT override w/ exponential backoff
        if timeout_exp_backoff:
            extra_params["timeout"] = int(timeout_exp_backoff.timeout)
        # TIMEOUT set connection timeout if it is NOT set as 1000s ~ 16' is too much
        if "timeout" not in extra_params or extra_params["timeout"] is None:
            extra_params["timeout"] = H2oGpteRagClient.DEFAULT_TIMEOUT

        msg_retry = f", retry {retry_attempt}/{retry_attempts}" if retry_attempt else ""
        msg_details = f"[{llm_model_name}{msg_retry}, {datetime.datetime.now()}]"

        if not chat_session_id:
            chat_session_id = self.client.create_chat_session()

        results = []
        with self.client.connect(chat_session_id) as session:
            for i, q in enumerate(prompts):
                self.logger.debug(
                    f"\n>>>RUNNING Q{i + 1} {msg_details}: {q}\n",
                    flush=True,
                )

                cost_prev = self._get_24h_cost()
                start = time.time()
                a = session.query(q, **extra_params)
                duration = time.time() - start
                cost = self._get_24h_cost() - cost_prev

                self.logger.debug(
                    f"\n>>>Q{i + 1}: {q}\n>>>A{i + 1}"
                    f"[{llm_model_name}{msg_retry}, duration: {duration:.2f}s]: "
                    f"{a.content}\n",
                    flush=True,
                )

                results.append(
                    LlmHostClient.LlmRagAnswer(
                        prompt=q,
                        answer=a.content,
                        duration=duration,
                        context=[],
                        cost=cost,
                        chat_session_id=chat_session_id,
                        chat_message_id=a.id,
                    )
                )

        return results


class MsAzureOpenAiLlmClient(LlmHostClient):
    """Microsoft Azure hosted OpenAI LLM client."""

    DEFAULT_API_VERSION = "2024-02-15-preview"

    @staticmethod
    def config_factory() -> dict:
        return OpenAiLlmClient.config_factory()

    @staticmethod
    def config_normalize(config: dict) -> dict:
        return OpenAiLlmClient.config_normalize(config)

    @property
    def client(self):
        return self._client or self._create_client()

    def __init__(
        self,
        connection: h2o_sonar_config.ConnectionConfig,
        base_url: str = "",
        deployment_name: str = "",
        api_version=DEFAULT_API_VERSION,
        logger: loggers.SonarLogger | None = None,
    ):
        """h2oGPT client constructor.

        Parameters
        ----------
        connection : h2o_sonar.config.ConnectionConfig
            h2oGPTe connection configuration.
        base_url : str
            Override required Microsoft Azure base URL for the OpenAI API deployment
            which is otherwise taken from the connection configuration.
        deployment_name : str
            Override Microsoft Azure deployment name, which is otherwise taken from
            the connection configuration.
            Azure host uses th deployment name instead of the LLM model name. In
            other words, deployment name is the alias for model name (like `gpt-4`)
            configured within the deployment.
            When specified in here, it is used as the default model name later.
        api_version : str
            Required API version (with a default value).
        logger : loggers.SonarLogger | None
            Optional logger.

        """
        if not connection:
            raise ValueError(
                "Microsoft Azure hosted OpenAI connection configuration is empty."
            )
        if (
            not connection.connection_type
            == h2o_sonar_config.ConnectionConfigType.AZURE_OPENAI_CHAT.name
        ):
            raise ValueError(
                f"Provide Microsoft Azure hosted OpenAI connection - connection type "
                f"'{connection.connection_type}' is not supported by OpenAiLlmClient."
            )
        if not connection.token:
            raise ValueError(
                "API key is required to be set as 'token' on the connection, "
                "but it is empty"
            )

        self.connection = connection
        self.base_url = base_url or connection.server_url
        self.api_version = api_version
        self.default_llm_model_name = deployment_name or connection.server_id
        self._client = None
        self.logger = logger or loggers.SonarPrintLogger()

    def _create_client(self):
        if not HAS_PKG_OPENAI:
            commons.raise_opt_import_err("openai")

        openai.verify_ssl_certs = h2o_sonar_config.config.http_ssl_cert_verify

        self._client = openai.AzureOpenAI(
            azure_endpoint=self.base_url,
            api_key=self.connection.token,
            api_version=self.api_version,
        )

        return self._client

    def list_llm_model_names(self) -> list[str]:
        """List ALL Microsoft Azure hosted OpenAI LLM models.

        IMPORTANT: these are NOT models provided by the particular deployment for
        which was the client created, but all models available in the Azure OpenAI API.

        """
        return [self.default_llm_model_name]
        # cannot be used: return [m.id for m in self.client.models.list()]

    def ask_model(
        self,
        prompts: list[str],
        llm_model_name: str = "",
        **extra_params,
    ) -> list[LlmHostClient.LlmRagAnswer]:
        """Ask a Microsoft Azure OpenAi hosted LLM model."""
        results = []

        llm_model_name = llm_model_name or self.default_llm_model_name

        for i, q in enumerate(prompts):
            self.logger.debug(
                f"\n>>>RUNNING Q{i + 1}:[{llm_model_name}]: {q}\n",
                flush=True,
            )

            start = time.time()
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": q}],
                model=llm_model_name,
            )
            a = chat_completion.choices[0].message.content
            results.append(
                LlmHostClient.LlmRagAnswer(
                    prompt=q,
                    answer=a,
                    duration=time.time() - start,
                    context=[],
                    cost=0,
                    chat_session_id="",
                    chat_message_id="",
                )
            )

            self.logger.debug(
                f"\n>>>Q{i + 1}: {q}\n>>>A{i + 1}[{llm_model_name}]: {a}\n",
                flush=True,
            )
        return results

    def ask_model_structured(
        self,
        prompts: list[str],
        output_structure: type[TBaseModel],
        llm_model_name: str = "",
        **extra_params,
    ) -> list[TBaseModel]:
        """Ask a Microsoft Azure OpenAi hosted LLM model with structured output.

        Uses OpenAI's beta.chat.completions.parse() for structured outputs.

        Note: Requires API version 2024-08-01-preview or later. Only certain
        models support structured outputs including gpt-4o, gpt-4o-mini, and
        gpt-4-turbo deployments.

        """
        if not HAS_PKG_PYDANTIC:
            commons.raise_opt_import_err("pydantic")
        results = []

        llm_model_name = llm_model_name or self.default_llm_model_name

        for i, q in enumerate(prompts):
            self.logger.debug(
                f"\n>>>RUNNING Q{i + 1}:[{llm_model_name}]: {q}\n",
                flush=True,
            )

            # use beta.chat.completions.parse() for structured outputs
            chat_completion = self.client.beta.chat.completions.parse(
                messages=[{"role": "user", "content": q}],
                model=llm_model_name,
                response_format=output_structure,
            )
            # the parsed output is already a Pydantic model instance
            a = chat_completion.choices[0].message.parsed
            results.append(a)

            self.logger.debug(
                f"\n>>>Q{i + 1}: {q}\n>>>A{i + 1}[{llm_model_name}]: {a}\n",
                flush=True,
            )
        return results


class OpenAiLlmClient(LlmHostClient):
    """OpenAI LLM client."""

    DEFAULT_LLM_MODEL = "gpt-4"  # alias for the latest model

    @staticmethod
    def __config_proto() -> dict:
        from openai import _types

        return {
            "messages": {},
            # ^ list[dict[str, str]]
            "frequency_penalty": _types.NOT_GIVEN,
            # ^  float | None | NotGiven
            "function_call": _types.NOT_GIVEN,
            # ^ completion_create_params.FunctionCall | NotGiven = NOT_GIVEN,
            "functions": _types.NOT_GIVEN,
            # ^ Iterable[completion_create_params.Function] | NotGiven = NOT_GIVEN,
            "logit_bias": _types.NOT_GIVEN,
            # ^ dict[str, int] | None | NotGiven = NOT_GIVEN,
            "logprobs": _types.NOT_GIVEN,
            # ^ bool | None | NotGiven = NOT_GIVEN,
            "max_tokens": _types.NOT_GIVEN,
            # ^ int | None | NotGiven = NOT_GIVEN,
            "n": _types.NOT_GIVEN,
            # ^ int | None | NotGiven = NOT_GIVEN,
            "presence_penalty": _types.NOT_GIVEN,
            # ^  float | None | NotGiven = NOT_GIVEN,
            "response_format": _types.NOT_GIVEN,
            # ^ completion_create_params.ResponseFormat | NotGiven = NOT_GIVEN,
            "seed": _types.NOT_GIVEN,
            # ^ int | None | NotGiven = NOT_GIVEN,
            "stop": _types.NOT_GIVEN,
            # ^ str | None, list[str] | NotGiven = NOT_GIVEN,
            "stream": _types.NOT_GIVEN,
            # ^ Literal[False] | None | Literal[True] | NotGiven = NOT_GIVEN,
            "temperature": _types.NOT_GIVEN,
            # ^  float | None | NotGiven = NOT_GIVEN,
            "tool_choice": _types.NOT_GIVEN,
            # ^ ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
            "tools": _types.NOT_GIVEN,
            # ^ Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
            "top_logprobs": _types.NOT_GIVEN,
            # ^ int | None | NotGiven = NOT_GIVEN,
            "top_p": _types.NOT_GIVEN,
            # ^  float | None | NotGiven = NOT_GIVEN,
            "user": _types.NOT_GIVEN,  # str | NotGiven = NOT_GIVEN,
            # Use the following arguments if you need to pass additional parameters
            # to the API that aren't available via kwargs.
            # The extra values given here take precedence over values defined on the
            # client or passed to this method.
            "extra_headers": None,
            # ^ Headers | None = None,
            "extra_query": None,
            # ^ Query | None = None,
            "extra_body": None,
            # ^ Body | None = None,
            "timeout": _types.NOT_GIVEN,
            # ^ float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        }

    @staticmethod
    def config_factory() -> dict:
        config_proto = OpenAiLlmClient.__config_proto()

        for k in config_proto:
            config_proto[k] = None

        return config_proto

    @staticmethod
    def config_normalize(config: dict) -> dict:
        """Normalize default values of the client configuration from the serializable
        dictionary representation to the OpenAI client configuration.

        Parameters
        ----------
        config : dict
            Configuration for the client in the serializable format.

        Returns
        -------
        dict :
          Normalized configuration for the client in the OpenAI format.

        """
        config_proto = OpenAiLlmClient.__config_proto()

        if not config:
            return config

        for k in config:
            if k in config_proto:
                config[k] = config[k] if config[k] is not None else config_proto[k]

        return config

    @property
    def client(self):
        return self._client or self._create_client()

    def __init__(
        self,
        connection: h2o_sonar_config.ConnectionConfig,
        default_llm_model_name: str = DEFAULT_LLM_MODEL,
        logger: loggers.SonarLogger | None = None,
    ):
        """h2oGPT client constructor.

        Parameters
        ----------
        connection : h2o_sonar.config.ConnectionConfig
            h2oGPTe connection configuration. Connection's server URL is **optional**
            for the OpenAI API. If not provided, then the client connects to OpenAI
            API servers directly, else it connects to the given server URL where it
            expects an endpoint with OpenAI compatible API.
        logger : loggers.SonarLogger | None
            Optional logger.

        """
        if not connection:
            raise ValueError("OpenAI connection configuration is empty.")
        if connection.connection_type not in [
            # connection types of clients which are based on OpenAI API
            h2o_sonar_config.ConnectionConfigType.OPENAI_CHAT.name,
            h2o_sonar_config.ConnectionConfigType.H2O_GPT.name,
            h2o_sonar_config.ConnectionConfigType.H2O_LLM_OPS.name,
        ]:
            raise ValueError(
                f"Provide OpenAI connection - connection type "
                f"'{connection.connection_type}' is not supported by OpenAiLlmClient."
            )
        if not connection.token:
            raise ValueError(
                "API key is required to be set as 'token' on the connection, but it is "
                "empty"
            )

        self.connection = connection
        self.default_llm_model_name = default_llm_model_name
        self._client = None
        self.logger = logger or loggers.SonarPrintLogger()

    def _create_client(self):
        if not HAS_PKG_OPENAI:
            commons.raise_opt_import_err("openai")

        self._client = openai.OpenAI(
            api_key=self.connection.token, base_url=self.connection.server_url or None
        )

        return self._client

    def _get_default_llm_model_name(self):
        return self.default_llm_model_name

    def list_llm_model_names(self) -> list[str]:
        return [model.id for model in self.client.models.list()]

    def ask_model(
        self,
        prompts: list[str],
        llm_model_name: str = "",
        **extra_params,
    ) -> list[LlmHostClient.LlmRagAnswer]:
        """Ask a OpenAi hosted LLM model."""
        results = []

        llm_model_name = llm_model_name or self._get_default_llm_model_name()

        for i, q in enumerate(prompts):
            self.logger.debug(
                f"\n>>>RUNNING Q{i + 1}:[{llm_model_name}]: {q}\n",
                flush=True,
            )

            if extra_params:
                extra_params = copy.deepcopy(extra_params)
                if "messages" in extra_params:
                    messages = extra_params["messages"]
                    extra_params.pop("messages")
                else:
                    messages = []
                extra_params = OpenAiLlmClient.config_normalize(extra_params)
            else:
                messages = []
            messages.append({"role": "user", "content": q})

            self.logger.debug(
                f"\n>>>RUNNING Q{i + 1}:[{llm_model_name}]: {q}\n",
                flush=True,
            )

            start = time.time()
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=llm_model_name,
                **extra_params,
            )
            a = chat_completion.choices[0].message.content
            results.append(
                LlmHostClient.LlmRagAnswer(
                    prompt=q,
                    answer=a,
                    duration=time.time() - start,
                    context=[],
                    cost=0,
                    chat_session_id="",
                    chat_message_id="",
                )
            )

            self.logger.debug(
                f"\n>>>Q{i + 1}: {q}\n>>>A{i + 1}[{llm_model_name}]: {a}\n",
                flush=True,
            )
        return results

    def ask_model_structured(
        self,
        prompts: list[str],
        output_structure: type[TBaseModel],
        llm_model_name: str = "",
        **extra_params,
    ) -> list[TBaseModel]:
        """Ask a OpenAi hosted LLM model with structured output.

        Uses OpenAI's beta.chat.completions.parse() for structured outputs.

        Note: Only certain models support structured outputs including gpt-4o,
        gpt-4o-mini, and gpt-4-turbo. The older gpt-4 and gpt-3.5-turbo models
        do NOT support structured outputs.
        """
        if not HAS_PKG_PYDANTIC:
            commons.raise_opt_import_err("pydantic")
        results = []

        llm_model_name = llm_model_name or self._get_default_llm_model_name()

        for i, q in enumerate(prompts):
            self.logger.debug(
                f"\n>>>RUNNING Q{i + 1}:[{llm_model_name}]: {q}\n",
                flush=True,
            )

            if extra_params:
                extra_params = copy.deepcopy(extra_params)
                if "messages" in extra_params:
                    messages = extra_params["messages"]
                    extra_params.pop("messages")
                else:
                    messages = []
                extra_params = OpenAiLlmClient.config_normalize(extra_params)
            else:
                messages = []
            messages.append({"role": "user", "content": q})

            # use beta.chat.completions.parse() for structured outputs
            chat_completion = self.client.beta.chat.completions.parse(
                messages=messages,
                model=llm_model_name,
                response_format=output_structure,
                **extra_params,
            )
            # the parsed output is already a Pydantic model instance
            a = chat_completion.choices[0].message.parsed
            results.append(a)

            self.logger.debug(
                f"\n>>>Q{i + 1}: {q}\n>>>A{i + 1}[{llm_model_name}]: {a}\n",
                flush=True,
            )
        return results


class AnthropicClaudeLlmClient(LlmHostClient):
    """Anthropic Claude LLM client."""

    DEFAULT_LLM_MODEL = "claude-sonnet-4-5-20250929"

    @staticmethod
    def __config_proto() -> dict:
        return {
            "max_tokens": None,
            # ^ int | NotGiven
            "metadata": None,
            # ^ MetadataParam | NotGiven
            "stop_sequences": None,
            # ^ list[str] | NotGiven
            "system": None,
            # ^ str | Iterable[TextBlockParam | ToolUseBlockParam
            #   | ToolResultBlockParam] | NotGiven
            "temperature": None,
            # ^ float | NotGiven
            "tool_choice": None,
            # ^ ToolChoiceParam | NotGiven
            "tools": None,
            # ^ Iterable[ToolParam] | NotGiven
            "top_k": None,
            # ^ int | NotGiven
            "top_p": None,
            # ^ float | NotGiven
            # Use the following arguments if you need to pass additional parameters
            # to the API that aren't available via kwargs.
            "extra_headers": None,
            # ^ Headers | None
            "extra_query": None,
            # ^ Query | None
            "extra_body": None,
            # ^ Body | None
            "timeout": None,
            # ^ float | httpx.Timeout | None | NotGiven
        }

    @staticmethod
    def config_factory() -> dict:
        config_proto = AnthropicClaudeLlmClient.__config_proto()

        for k in config_proto:
            config_proto[k] = None

        return config_proto

    @staticmethod
    def config_normalize(config: dict) -> dict:
        """Normalize default values of the client configuration from the serializable
        dictionary representation to the Anthropic client configuration.

        Parameters
        ----------
        config : dict
            Configuration for the client in the serializable format.

        Returns
        -------
        dict :
          Normalized configuration for the client in the Anthropic format.

        """
        config_proto = AnthropicClaudeLlmClient.__config_proto()

        if not config:
            return config

        normalized_config = {}
        for k in config:
            if k in config_proto:
                if config[k] is not None:
                    normalized_config[k] = config[k]

        return normalized_config

    @property
    def client(self):
        return self._client or self._create_client()

    def __init__(
        self,
        connection: h2o_sonar_config.ConnectionConfig,
        default_llm_model_name: str = DEFAULT_LLM_MODEL,
        logger: loggers.SonarLogger | None = None,
    ):
        """Anthropic Claude client constructor.

        Parameters
        ----------
        connection : h2o_sonar.config.ConnectionConfig
            Anthropic connection configuration. Connection's server URL is **optional**
            for the Anthropic API. If not provided, then the client connects to
            Anthropic API servers directly, else it connects to the given server URL
            where it expects an endpoint with Anthropic compatible API.
        default_llm_model_name : str, optional
            Default name of the LLM model to use. Defaults to ``DEFAULT_LLM_MODEL``.
        logger : loggers.SonarLogger | None
            Optional logger.

        """
        if not connection:
            raise ValueError("Anthropic connection configuration is empty.")
        if connection.connection_type not in [
            h2o_sonar_config.ConnectionConfigType.ANTHROPIC_CHAT.name,
        ]:
            raise ValueError(
                f"Provide Anthropic connection - connection type "
                f"'{connection.connection_type}' is not supported by "
                f"AnthropicClaudeLlmClient."
            )
        if not connection.token:
            raise ValueError(
                "API key is required to be set as 'token' on the connection, but it is "
                "empty"
            )

        self.connection = connection
        self.default_llm_model_name = default_llm_model_name
        self._client = None
        self.logger = logger or loggers.SonarPrintLogger()

    def _create_client(self):
        if not HAS_PKG_ANTHROPIC:
            commons.raise_opt_import_err("anthropic")

        self._client = anthropic.Anthropic(
            api_key=self.connection.token, base_url=self.connection.server_url or None
        )

        return self._client

    def _get_default_llm_model_name(self):
        return self.default_llm_model_name

    def list_llm_model_names(self) -> list[str]:
        """List available Anthropic Claude models by querying the API.

        Returns
        -------
        list[str] :
            List of available model IDs, sorted with newest models first.
            Falls back to known models if API request fails.

        """
        try:
            # query Anthropic models API:
            # https://docs.claude.com/en/api/models-list
            headers = {
                "x-api-key": self.connection.token,
                "anthropic-version": "2023-06-01",
            }

            response = requests.get(
                "https://api.anthropic.com/v1/models",
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()

            models_data = response.json()

            model_ids = []
            if "data" in models_data:
                for model in models_data["data"]:
                    if "id" in model:
                        model_ids.append(model["id"])

            if model_ids:
                self.logger.info(
                    f"Successfully fetched {len(model_ids)} Anthropic models from API"
                )
                return model_ids

        except Exception as ex:
            self.logger.warning(f"Failed to fetch Anthropic models from API: {ex}")
            raise ex

        raise RuntimeError("No models found in Anthropic API response")

    def ask_model(
        self,
        prompts: list[str],
        llm_model_name: str = "",
        **extra_params,
    ) -> list[LlmHostClient.LlmRagAnswer]:
        """Ask an Anthropic Claude hosted LLM model."""
        results = []

        llm_model_name = llm_model_name or self._get_default_llm_model_name()

        for i, q in enumerate(prompts):
            self.logger.debug(
                f"\n>>>RUNNING Q{i + 1}:[{llm_model_name}]: {q}\n",
                flush=True,
            )

            if extra_params:
                extra_params = copy.deepcopy(extra_params)
                extra_params = AnthropicClaudeLlmClient.config_normalize(extra_params)
            else:
                extra_params = {}

            # Set default max_tokens if not provided (required by Anthropic API)
            if "max_tokens" not in extra_params or extra_params["max_tokens"] is None:
                extra_params["max_tokens"] = 4096

            start = time.time()
            message = self.client.messages.create(
                model=llm_model_name,
                messages=[{"role": "user", "content": q}],
                **extra_params,
            )
            a = message.content[0].text
            results.append(
                LlmHostClient.LlmRagAnswer(
                    prompt=q,
                    answer=a,
                    duration=time.time() - start,
                    context=[],
                    cost=0,
                    chat_session_id="",
                    chat_message_id="",
                )
            )

            self.logger.debug(
                f"\n>>>Q{i + 1}: {q}\n>>>A{i + 1}[{llm_model_name}]: {a}\n",
                flush=True,
            )
        return results

    def ask_model_structured(
        self,
        prompts: list[str],
        output_structure: type[TBaseModel],
        llm_model_name: str = "",
        **extra_params,
    ) -> list[TBaseModel]:
        """Ask an Anthropic Claude hosted LLM model with structured output.

        Parameters
        ----------
        prompts : list[str]
            Prompts to ask.
        output_structure : pydantic.BaseModel
            Pydantic model class defining the expected output structure.
        llm_model_name : str
            Optional LLM model name.
        extra_params :
            Optional extra parameters.

        Returns
        -------
        list[pydantic.BaseModel]
            List of structured outputs parsed into the provided Pydantic model.
        """
        if not HAS_PKG_PYDANTIC:
            commons.raise_opt_import_err("pydantic")
        results = []

        llm_model_name = llm_model_name or self._get_default_llm_model_name()

        for i, q in enumerate(prompts):
            self.logger.debug(
                f"\n>>>RUNNING Q{i + 1}:[{llm_model_name}]: {q}\n",
                flush=True,
            )

            if extra_params:
                extra_params = copy.deepcopy(extra_params)
                extra_params = AnthropicClaudeLlmClient.config_normalize(extra_params)
            else:
                extra_params = {}

            # set default max_tokens if not provided (required by Anthropic API)
            if "max_tokens" not in extra_params or extra_params["max_tokens"] is None:
                extra_params["max_tokens"] = 4096

            # anthropic doesn't support response_format like OpenAI, so we need to
            # request JSON in the prompt and parse it manually
            structured_prompt = (
                f"{q}\n\nPlease respond with valid JSON matching this schema:\n"
                f"{output_structure.model_json_schema()}"
            )

            message = self.client.messages.create(
                model=llm_model_name,
                messages=[{"role": "user", "content": structured_prompt}],
                **extra_params,
            )
            response_text = message.content[0].text

            # strip markdown code blocks if present (e.g., ```json ... ```)
            response_text = response_text.strip()
            if response_text.startswith("```"):
                # find the first newline after the opening ```
                first_newline = response_text.find("\n")
                if first_newline != -1:
                    response_text = response_text[first_newline + 1 :]
                # remove the closing ```
                if response_text.endswith("```"):
                    response_text = response_text[:-3].strip()

            # parse the JSON response into the pydantic model
            parsed_output = output_structure.model_validate_json(response_text)

            results.append(parsed_output)

            self.logger.debug(
                f"\n>>>Q{i + 1}: {q}\n>>>A{i + 1}[{llm_model_name}]: {parsed_output}\n",
                flush=True,
            )
        return results


class OpenAiAssistantsRagClientVersion1(RagClient):
    """OpenAI RAG client - Assistants AI with enabled File Search/Retrieval tool.

    This client leaks vector stores with zero size. Using the old API there is no way
    to remove them. Since the size is zero it shouldn't cost anything but it's not nice
    to leave mess.

    @see https://github.com/openai/openai-python/blob/v1.20.0/api.md

    """

    # only certain LLM models are supported by the OpenAI Assistants API:
    # - https://platform.openai.com/docs/assistants/overview/step-1-create-an-assistant
    # - https://github.com/openai/openai-python
    DEFAULT_LLM_MODEL = (
        "gpt-4o"  # deprecated "gpt-35-turbo-1106" long latency: "gpt-4o"
    )
    BASE_LLM_MODELS = [DEFAULT_LLM_MODEL, "gpt-3.5-turbo-1106"]

    HEADERS_VERSION_1 = {"OpenAI-Beta": "assistants=v1"}
    HEADERS_VERSION_2 = {"OpenAI-Beta": "assistants=v2"}

    KWARGS_ASSISTANT = "assistant_kwargs"
    KWARGS_THREAD = "thread_kwargs"
    KWARGS_RUN = "run_kwargs"

    @staticmethod
    def __config_proto(version: str = "v1") -> dict:
        """Get the prototype of the configuration for the client. Parameters in
        the returned prototype directory are prefixed as follows to indicate in which
        stage of the request they are used:

        - "assistant_" - used in the assistant creation
        - "thread_" - used in the thread creation
        - "run_" - used in the completion run

        Parameters
        ----------
        version : str
            OpenAI Assistants API version - "v1" or "v2".

        """
        if version == "v2":
            raise ValueError("OpenAI Assistants API version 'v2' is not supported.")

        from openai import _types

        return {
            #
            # assistant creation
            #
            OpenAiAssistantsRagClientVersion1.KWARGS_ASSISTANT: {
                "name": "",
                # ^ str
                "description": "",
                # ^ str
                "instructions": _types.NOT_GIVEN,
                # ^ str | None | NotGiven
                "tools": _types.NOT_GIVEN,
                # ^ Iterable[dict[str, str]] | NotGiven
                "metadata": _types.NOT_GIVEN,
                # ^ object | None | NotGiven
                "extra_headers": None,
                # ^ Headers | None
                "extra_query": None,
                # ^ Query | None
                "extra_body": None,
                # ^ Body | None
                "timeout": _types.NOT_GIVEN,
                # ^ float | httpx.Timeout | None | NotGiven
            },
            #
            # thread creation
            #
            OpenAiAssistantsRagClientVersion1.KWARGS_THREAD: {
                "messages": _types.NOT_GIVEN,
                # ^ Iterable[thread_create_params.Message] | NotGiven
                #   msg: content, role ("user", "assistant"), file_ids, metadata
                "metadata": _types.NOT_GIVEN,
                # ^ object | None | NotGiven
                "extra_headers": None,
                # ^ Headers | None
                "extra_query": None,
                # ^ Query | None
                "extra_body": None,
                # ^ Body | None
                "timeout": _types.NOT_GIVEN,
                # ^ float | httpx.Timeout | None | NotGiven
            },
            #
            # completion run
            #
            OpenAiAssistantsRagClientVersion1.KWARGS_RUN: {
                "additional_instructions": _types.NOT_GIVEN,
                # ^ str | None | NotGiven
                "additional_messages": _types.NOT_GIVEN,
                # ^ Iterable[run_create_params.AdditionalMessage] | None | NotGiven
                #   msg: content, role ("user", "assistant"), file_ids, metadata
                "instructions": _types.NOT_GIVEN,
                # ^ str | None | NotGiven
                "max_completion_tokens": _types.NOT_GIVEN,
                # ^ int | None | NotGiven
                "max_prompt_tokens": _types.NOT_GIVEN,
                # ^ int | None | NotGiven
                "metadata": _types.NOT_GIVEN,
                # ^ object | None | NotGiven
                "response_format": _types.NOT_GIVEN,
                # ^ AssistantResponseFormatOptionParam | None | NotGiven
                "stream": _types.NOT_GIVEN,
                # ^ Literal[False] | None | Literal[True] | NotGiven
                "temperature": _types.NOT_GIVEN,
                # ^ float | None | NotGiven
                "tool_choice": _types.NOT_GIVEN,
                # ^ AssistantToolChoiceOptionParam | None | NotGiven
                "tools": _types.NOT_GIVEN,
                # ^ Iterable[AssistantToolParam] | None | NotGiven
                "truncation_strategy": _types.NOT_GIVEN,
                # ^ run_create_params.TruncationStrategy | None | NotGiven
                "extra_headers": None,
                # ^ Headers | None
                "extra_query": None,
                # ^ Query | None
                "extra_body": None,
                # ^ Body | None
                "timeout": _types.NOT_GIVEN,
                # ^ float | httpx.Timeout | None | NotGiven
            },
        }

    @staticmethod
    def config_factory() -> dict:
        config_proto = OpenAiAssistantsRagClientVersion1.__config_proto()

        for g in [
            OpenAiAssistantsRagClientVersion1.KWARGS_ASSISTANT,
            OpenAiAssistantsRagClientVersion1.KWARGS_THREAD,
            OpenAiAssistantsRagClientVersion1.KWARGS_RUN,
        ]:
            for k in config_proto[g]:
                config_proto[g][k] = None

        return config_proto

    @staticmethod
    def config_normalize(config: dict) -> dict:
        """Normalize default values of the client configuration from the serializable
        dictionary representation to the OpenAI client configuration.

        Parameters
        ----------
        config : dict
            Configuration for the client in the serializable format.

        Returns
        -------
        dict :
          Normalized configuration for the client in the OpenAI format.

        """
        config_proto = OpenAiAssistantsRagClientVersion1.__config_proto()

        if not config:
            return config

        for k in config:
            if k in config_proto:
                config[k] = config[k] if config[k] is not None else config_proto[k]

        return config

    @staticmethod
    def config_resolve(
        in_kwargs: dict,
        config_to_resolve: dict,
        config_group_key: str,
        required_keys: dict[str, Any],
    ) -> dict:
        """Resolve the configuration for the client by ensuring that the required
        keys are set in the given configuration group:

        Model parameters' priority:

        HIGH: kwargs e.g. "instructions"
        MEDIUM:  kwargs["assistant_kwargs"] e.g. "instructions"
        LOW: required defaults

        Model parameters resolution method:

        1. start with EMPTY/SNAPSHOT parameters
        2. apply (non-assistant) kwargs
        3. apply assistant kwargs - if NOT already set
        4. ensure defaults for REQUIRED parameters
        5. normalize to OpenAI defaults

        """
        # HIGH
        in_kwargs = in_kwargs or {}
        for k in in_kwargs:
            if k not in [
                OpenAiAssistantsRagClientVersion1.KWARGS_ASSISTANT,
                OpenAiAssistantsRagClientVersion1.KWARGS_THREAD,
                OpenAiAssistantsRagClientVersion1.KWARGS_RUN,
            ]:
                config_to_resolve[k] = in_kwargs[k]
        # MEDIUM:
        if config_group_key in in_kwargs:
            assistant_kwargs = in_kwargs[config_group_key]
            for k in assistant_kwargs:
                if k not in config_to_resolve:
                    config_to_resolve[k] = assistant_kwargs[k]
        # LOW
        if required_keys:
            for k in required_keys:
                if k not in config_to_resolve or not config_to_resolve[k]:
                    config_to_resolve[k] = required_keys[k]

        # normalize to OpenAI defaults
        return OpenAiAssistantsRagClientVersion1.config_normalize(config_to_resolve)

    @property
    def client(self):
        return self._client or self._create_client()

    def __init__(
        self,
        connection: h2o_sonar_config.ConnectionConfig,
        default_llm_model_name: str = DEFAULT_LLM_MODEL,
        logger=None,
    ):
        """OpenAI RAG client constructor.

        Parameters
        ----------
        connection : h2o_sonar.config.ConnectionConfig
            OpenAI connection configuration.
        logger :
            Optional logger.

        """
        if not connection:
            raise ValueError("OpenAI connection configuration is empty.")
        if (
            not connection.connection_type
            == h2o_sonar_config.ConnectionConfigType.OPENAI_RAG.name
        ):
            raise ValueError(
                f"Provide OpenAI connection - connection type "
                f"'{connection.connection_type}' is not supported by "
                f"OpenAiAssistantsRagClient."
            )
        if not connection.token:
            raise ValueError(
                "API key is required to be set as 'token' on the connection, "
                "but it is empty"
            )

        self.connection = connection
        self._client = None
        # map: assistant id (str) -> assistant object
        self._created_assistants = {}
        # map: file id (str) -> file object
        self._uploaded_documents = {}
        self.default_llm_model_name = default_llm_model_name
        self.logger = logger or loggers.SonarPrintLogger()

        self._create_client()

    def _create_client(self):
        if not HAS_PKG_OPENAI:
            commons.raise_opt_import_err("openai")

        self._client = openai.OpenAI(
            default_headers=OpenAiAssistantsRagClientVersion1.HEADERS_VERSION_1,
            api_key=self.connection.token,
        )

        return self._client

    def list_llm_model_names(self, rag: bool = True) -> list[str]:
        """List OpenAI LLM models.

        Parameters
        ----------
        rag : bool
            Optional parameter to list only models supported by the OpenAI Assistants
            API (``True``, default) or all OpenAI LLM models.

        Returns
        -------
        list[str] :
            List of LLM model names.

        """
        model_list = self.client.models.list()
        all_llms = [m.id for m in model_list]
        if not rag:
            return all_llms

        # only models below are allowed by the OpenAI Assistants API
        # https://platform.openai.com/docs/assistants/overview/agents
        rag_supported_llms = [
            "gpt-3.5-turbo-1106",
            OpenAiAssistantsRagClientVersion1.DEFAULT_LLM_MODEL,
        ]
        return [m for m in rag_supported_llms if m in all_llms]

    def create_collection(
        self,
        doc_paths: list[pathlib.Path | str],
        collection_name: str = "",
        llm_model_name: str = DEFAULT_LLM_MODEL,
        **kwargs,
    ) -> str:
        """Create OpenAI Assistant with enabled retrieval tool and upload
        documents (corpus) to that assistant.

        Parameters
        ----------
        doc_paths : list[pathlib.Path | str]
          Paths (local filesystem) to the documents to be uploaded.
        llm_model_name : str
            Base LLM model name to be used by RAG in the generation phase.

        Returns
        -------

        str :
          OpenAI Assistant ID.

        """
        # UPLOAD corpus ~ document(s)
        uploaded_corpus = []
        for doc_path in doc_paths:
            try:
                uploaded_file = self.client.files.create(
                    file=open(doc_path, "rb"), purpose="assistants"
                )
                uploaded_corpus.append(uploaded_file.id)

                self._uploaded_documents[uploaded_file.id] = uploaded_file
            except Exception as fex:
                raise RuntimeError(
                    f"Failed to upload file {doc_path} to OpenAI: {fex}\n"
                    f"{traceback.format_exc()}"
                )

        # create ASSISTANT
        create_kwargs = OpenAiAssistantsRagClientVersion1.config_resolve(
            in_kwargs=kwargs,
            config_to_resolve={
                "file_ids": uploaded_corpus,
            },
            config_group_key=OpenAiAssistantsRagClientVersion1.KWARGS_ASSISTANT,
            required_keys={
                "name": (
                    f"EvalStudio OpenAI RAG evaluation ({datetime.datetime.now()})"
                ),
                "instructions": (
                    "You are a chatbot. Use your knowledge base (uploaded "
                    "documents) to respond to asked questions."
                ),
                "tools": [{"type": "retrieval"}],
            },
        )

        assistant = self.client.beta.assistants.create(
            model=llm_model_name,
            **create_kwargs,
        )
        self._created_assistants[assistant.id] = assistant

        return assistant.id

    def list_collections(self, offset: int = 0, limit: int = 10) -> list:
        """List OpenAI Assistants with retrieval tool enabled.

        Parameters
        ----------
        offset : int
            Offset of the returned assistants - is always 0 in case of OpenAI
            implementation.
        limit : int
            Limit the number of returned assistants.

        Returns
        -------
        List : list[Assistant]
            List of assistant instances.

        """
        return list(self.client.beta.assistants.list(limit=limit))

    def purge_collections(self, assistants_ids: list[str] | None = None) -> list:
        """Purge h2oGPTe collections.

        Parameters
        ----------
        assistants_ids : list[str] | None
            List of OpenAI Assistant IDs to be purged. If the list is empty,
            all Assistants created by this instance are purged.

        """
        assistants_ids = assistants_ids or list(self._created_assistants.keys())

        deleted_assistant_ids = []
        for assistant_id in assistants_ids:
            try:
                self.client.beta.assistants.delete(assistant_id)
                deleted_assistant_ids.append(assistant_id)
            except Exception as fex:
                print(f"Failed to purge assistant {assistant_id}: {fex}")

        return deleted_assistant_ids

    def purge_uploaded_docs(self, document_ids: list[str] | None = None) -> list:
        """Purge h2oGPTe uploaded documents.

        Parameters
        ----------
        document_ids : list[str] | None
            List of document IDs to be purged. If the list is empty, all documents
            uploaded by this instance are purged.

        """
        document_ids = document_ids or list(self._uploaded_documents.keys())

        deleted_doc_ids = []
        for document_id in document_ids:
            try:
                self.client.files.delete(document_id)
                deleted_doc_ids.append(document_id)
            except Exception as fex:
                print(f"Failed to purge file {document_id}: {fex}")

        return deleted_doc_ids

    def _ask_collection_1(
        self,
        assistant_id: str,
        prompt: str,
        chunks_limit: int = 0,
        timeout: int = 600,
        # extra parameters
        **kwargs,
    ) -> LlmHostClient.LlmRagAnswer:
        """Ask OpenAI Assistant with retrieval tool enabled and corpus uploaded.
        This method creates a new thread for the prompt and retrieves the answer
        as well as relevant chunks (if requested).

        Parameters
        ----------
        assistant_id : str
            OpenAI Assistant ID.
        prompt : str
            Prompt to ask.
        include_chunks : int
            Optional parameter to determine also relevant (text) chunks.
        timeout : int
            Timeout in seconds.
        kwargs :
            Optional parameters to be passed to the OpenAI client.

        Returns
        -------
        LlmHostClient.LlmRagAnswer :
            Named tuple with prompt, answer, duration and chunks.

        """
        # create THREAD @ assistant
        thread_kwargs = OpenAiAssistantsRagClientVersion1.config_resolve(
            in_kwargs=kwargs.get(OpenAiAssistantsRagClientVersion1.KWARGS_THREAD, {}),
            config_to_resolve={},
            config_group_key=OpenAiAssistantsRagClientVersion1.KWARGS_THREAD,
            required_keys={},
        )
        thread = self.client.beta.threads.create(**thread_kwargs)

        # create MESSAGE w/ question/prompt @ thread
        message = self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt,
        )
        self.logger.debug(
            f"Message for prompt '{prompt}':\n  {message.model_dump_json()}"
        )

        # RUN the ASSISTANT @ THREAD
        run_kwargs = OpenAiAssistantsRagClientVersion1.config_resolve(
            in_kwargs=kwargs,
            config_to_resolve={},
            config_group_key=OpenAiAssistantsRagClientVersion1.KWARGS_RUN,
            required_keys={
                "instructions": "The user is demanding and requires precise answers.",
            },
        )

        start = time.time()
        assistant_run = self.client.beta.threads.runs.create(
            assistant_id=assistant_id,
            thread_id=thread.id,
            **run_kwargs,
        )
        self.logger.debug(
            f"Assistant run for prompt '{prompt}':\n  {assistant_run.model_dump_json()}"
        )
        # WAIT for the assistant run to complete
        assistant_err_statuses = ["failed", "expired", "cancelled"]
        assistant_done_statuses = ["completed"] + assistant_err_statuses
        step = 3  # seconds
        timeout = int(timeout / step)
        countdown = timeout
        while assistant_run.status not in assistant_done_statuses and countdown:
            self.logger.debug(
                f"{countdown}/{timeout}s wait countdown for the run to complete: "
                f"{assistant_run.status}"
            )
            assistant_run = self.client.beta.threads.runs.retrieve(
                thread_id=thread.id, run_id=assistant_run.id
            )
            self.logger.debug(f"  {assistant_run.model_dump_json()}")
            time.sleep(step)  # Assistant API is slow + saving rate limit
            countdown -= 1
        self.logger.debug(
            f"Assistant run for prompt '{prompt}' DONE: {assistant_run.status}\n"
            f"  {assistant_run.model_dump_json()}"
        )
        if assistant_run.status in assistant_err_statuses:
            raise RuntimeError(f"AI Assistant run failed: {assistant_run.status}")

        # get ACTUAL OUTPUT
        thread_messages = self.client.beta.threads.messages.list(
            thread_id=thread.id
        ).data
        if not len(thread_messages):
            raise RuntimeError(
                f"No messages in the thread: {thread.id} for assistant: {assistant_id} "
            )
        assistant_message = thread_messages[0]
        if not len(assistant_message.content):
            raise RuntimeError(
                f"Unable to get context - no assistant message content in "
                f"the thread: {thread.id} for assistant: {assistant_id}"
            )
        message_text = assistant_message.content[0].text.value

        # get CONTEXT
        context = []
        chunks_limit = min(
            0 if chunks_limit < 0 else chunks_limit, RagClient.CHUNKS_LIMIT
        )
        if chunks_limit:
            # PROBLEM: only certain Assistants return the context
            # - it seems to be related to the base LLM model used by the Assistant
            #   (however, the context is retrieval phase, perhaps certain LLM models
            #   don't provide backlinks to the chunks)
            # PROBLEM: the context seems to be incomplete / small in certain cases
            # - if the context is available, then it is not always complete
            #   (sometimes is text, sometimes just s sequence of numbers/tokens)
            if not len(assistant_message.content[0].text.annotations):
                self.logger.warning(
                    f"Unable to get context - no assistant message annotation in the "
                    f"thread: {thread.id} for assistant: {assistant_id}"
                )
            else:
                context = [
                    assistant_message.content[0].text.annotations[0].file_citation.quote
                ]
                # trim the context to the chunks limit
                context = context[:chunks_limit]

        self.logger.info(
            f"Prompt :\n  {prompt}\nAnswer :\n  {message_text}\nContext:\n  {context}\n"
        )

        return LlmHostClient.LlmRagAnswer(
            prompt=prompt,
            answer=message_text,
            duration=time.time() - start,
            context=context,
            cost=0.0,  # TODO (cost of the assistant run)
            chat_session_id="",
            chat_message_id="",
        )

    def ask_collection(
        self,
        assistant_id: str,
        prompts: list[str],
        include_chunks: int = 0,
        # extra parameters
        **kwargs,
    ) -> list[LlmHostClient.LlmRagAnswer]:
        """Ask OpenAI Assistant with retrieval tool enabled and corpus uploaded.
        This method creates a new thread for each prompt and retrieves the answer
        as well as relevant chunks (if requested).

        Parameters
        ----------
        assistant_id : str
            OpenAI Assistant ID.
        prompts : list[str]
            Prompts to ask.
        include_chunks : int
            Optional parameter to determine also relevant (text) chunks.

        """
        if not assistant_id:
            raise ValueError("OpenAI Assistant ID is empty.")
        if not prompts:
            raise ValueError("Prompts are empty.")

        results = []
        for e, prompt in enumerate(prompts):
            results.append(
                self._ask_collection_1(
                    assistant_id=assistant_id,
                    prompt=prompt,
                    chunks_limit=include_chunks,
                    **kwargs,
                )
            )

        return results

    def ask_model(
        self,
        prompts: list[str],
        llm_model_name: str = "",
        is_one_prompt: bool = False,
        # extra parameters
        **kwargs,
    ) -> list[LlmHostClient.LlmRagAnswer]:
        """Ask a OpenAI LLM (base) model (minimalistic version without messages and
        parameterization of system prompts, assisting content, parameters, ...).

        Parameters
        ----------
        prompts : list[str]
            Prompts to ask.
        llm_model_name : str
            Optional LLM model name to use for the answer.
        is_one_prompt : bool
            Optional parameter to decide whether to ask all prompts in one request
            (all prompts will be used as the context for the last prompt) or in
            separate requests.

        Returns
        -------
        LlmHostClient.LlmRagAnswer :
            Named tuple with prompt, answer and duration.

        """
        llm_model_name = llm_model_name or self.default_llm_model_name
        if is_one_prompt:
            messages = [{"role": "user", "content": prompt} for prompt in prompts]
            start = time.time()
            chat_answer = self.client.chat.completions.create(
                messages=messages,
                model=llm_model_name,  # optional LLM model
                **kwargs,
            )
            cost = 0.0  # TODO determine cost from the answer (if possible)
            return [
                LlmHostClient.LlmRagAnswer(
                    prompt=str(prompts),
                    answer=chat_answer.choices[-1].message.content,
                    duration=time.time() - start,
                    context=[],
                    cost=cost,
                    chat_session_id="",
                    chat_message_id="",
                )
            ]
        else:
            results = []
            for prompt in prompts:
                start = time.time()
                chat_answer = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=llm_model_name,  # optional LLM model
                    **kwargs,
                )
                cost = 0.0  # TODO determine cost from the answer (if possible)
                self.logger.debug(chat_answer)
                self.logger.info(
                    f"Prompt: {prompt}\n"
                    f"Answer: {chat_answer.choices[0].message.content}"
                )
                results.append(
                    LlmHostClient.LlmRagAnswer(
                        prompt=prompt,
                        answer=chat_answer.choices[0].message.content,
                        duration=time.time() - start,
                        context=[],
                        cost=cost,
                        chat_session_id="",
                        chat_message_id="",
                    )
                )
            return results

    def ask_model_structured(
        self,
        prompts: list[str],
        output_structure: type[TBaseModel],
        llm_model_name: str = "",
        is_one_prompt: bool = False,
        **kwargs,
    ) -> list[TBaseModel]:
        """Ask a OpenAI LLM (base) model with structured output.

        Parameters
        ----------
        prompts : list[str]
            Prompts to ask.
        output_structure : pydantic.BaseModel
            Pydantic model class defining the expected output structure.
        llm_model_name : str
            Optional LLM model name to use for the answer.
        is_one_prompt : bool
            Optional parameter to decide whether to ask all prompts in one request.
        kwargs :
            Optional extra parameters.

        Returns
        -------
        list[pydantic.BaseModel]
            List of structured outputs parsed into the provided Pydantic model.
        """
        if not HAS_PKG_PYDANTIC:
            commons.raise_opt_import_err("pydantic")
        llm_model_name = llm_model_name or self.default_llm_model_name
        if is_one_prompt:
            messages = [{"role": "user", "content": prompt} for prompt in prompts]
            chat_answer = self.client.beta.chat.completions.parse(
                messages=messages,
                model=llm_model_name,
                response_format=output_structure,
                **kwargs,
            )
            parsed_output = chat_answer.choices[-1].message.parsed
            return [parsed_output]
        else:
            results = []
            for prompt in prompts:
                chat_answer = self.client.beta.chat.completions.parse(
                    messages=[{"role": "user", "content": prompt}],
                    model=llm_model_name,
                    response_format=output_structure,
                    **kwargs,
                )
                parsed_output = chat_answer.choices[0].message.parsed
                self.logger.debug(chat_answer)
                self.logger.info(f"Prompt: {prompt}\nAnswer: {parsed_output}")
                results.append(parsed_output)
            return results


class OpenAiAssistantsRagClientVersion2(RagClient):
    """OpenAI RAG client - Assistants AI with enabled file search tool.

    File search tool is successor to retrieval tool from openai API v1.
    File search tool uses Vector Store which is a vector database that's capable of
    both keyword and semantic search. Each vector_store can hold up to 10,000 files.
    Vector stores can be attached to both Assistants and Threads but in our
    implementation we attach vector stores to Assistants.

    @see https://platform.openai.com/docs/assistants/tools/file-search

    """

    HEADERS_VERSION_2 = {"OpenAI-Beta": "assistants=v2"}
    DEFAULT_LLM_MODEL = "gpt-4o"
    BASE_LLM_MODELS = [DEFAULT_LLM_MODEL]

    @property
    def client(self):
        return self._client or self._create_client()

    def __init__(
        self,
        connection: h2o_sonar_config.ConnectionConfig,
        default_llm_model_name: str = DEFAULT_LLM_MODEL,
        logger=None,
    ):
        """OpenAI RAG client constructor.

        Parameters
        ----------
        connection : h2o_sonar.config.ConnectionConfig
            OpenAI connection configuration.
        logger :
            Optional logger.

        """
        if not connection:
            raise ValueError("OpenAI connection configuration is empty.")
        if (
            not connection.connection_type
            == h2o_sonar_config.ConnectionConfigType.OPENAI_RAG.name
        ):
            raise ValueError(
                f"Provide OpenAI connection - connection type "
                f"'{connection.connection_type}' is not supported by "
                f"OpenAiAssistantsRagClient."
            )
        if not connection.token:
            raise ValueError(
                "API key is required to be set as 'token' on the connection, "
                "but it is empty"
            )

        self.connection = connection
        self._client = None
        # map: assistant id (str) -> assistant object
        self._created_assistants = {}
        # list[vector_store]
        self._uploaded_documents = []
        self._vector_stores = []  # VS that are created by this instance
        self.default_llm_model_name = default_llm_model_name
        self.logger = logger or loggers.SonarPrintLogger()

        self._create_client()

    def _create_client(self):
        if not HAS_PKG_OPENAI:
            commons.raise_opt_import_err("openai")

        self._client = openai.OpenAI(
            default_headers=self.HEADERS_VERSION_2,
            api_key=self.connection.token,
        )
        return self._client

    def list_llm_model_names(self, rag: bool = True) -> list[str]:
        """List OpenAI LLM models.

        Parameters
        ----------
        rag : bool
            Optional parameter to list only models supported by the OpenAI Assistants
            API (``True``, default) or all OpenAI LLM models.

        Returns
        -------
        list[str] :
            List of LLM model names.

        """
        model_list = self.client.models.list()
        all_llms = [m.id for m in model_list]
        if not rag:
            return all_llms

        rag_supported_llms = [
            "gpt-3.5-turbo",
            "gpt-4-turbo",
            "gpt-4o",
        ]
        result = [m for m in rag_supported_llms if m in all_llms]
        return result

    def _get_vector_store_by_name(
        self, name: str
    ) -> "openai.lib.vector_store.VectorStore":  # noqa
        if not HAS_PKG_BOTO3:
            commons.raise_opt_import_err("boto3")

        vs = self.client.vector_stores.list()
        for v in vs:
            if (
                v.name == name
                and datetime.datetime.now(datetime.UTC).timestamp() < v.expires_at
            ):
                return v
        return None

    def create_collection(
        self,
        doc_paths: list[pathlib.Path | str],
        collection_name: str = "",
        llm_model_name: str = DEFAULT_LLM_MODEL,
        assistant_name: str = "",
    ) -> str:
        """Create OpenAI Assistant with enabled file search tool and upload
        documents (corpus) to that assistant.

        Parameters
        ----------
        doc_paths : list[pathlib.Path | str]
          Paths (local filesystem) to the documents to be uploaded.
        collection_name : str
          Optional parameter with the document collection use (if specified) or create
          (if the given name does not exist)
        llm_model_name : str
          Base LLM model name to be used by RAG in the generation phase.
        assistant_name : str
          Optional parameter with the string to name to new Assistant.

        Returns
        -------

        str :
          OpenAI Assistant ID.

        """
        # UPLOAD corpus ~ document(s)
        import openai.types.beta

        resolved_collection_name = collection_name or self.get_collection_name(
            doc_paths
        )

        assistant_name = (
            assistant_name
            or f"EvalStudio OpenAI RAG evaluation ({datetime.datetime.now()})"
        )
        vector_store = self._get_vector_store_by_name(resolved_collection_name)
        if not vector_store:
            vector_store = self.client.vector_stores.create(
                name=resolved_collection_name,
                expires_after=openai.types.vector_store_create_params.ExpiresAfter(
                    anchor="last_active_at", days=1
                ),
            )
            self._vector_stores.append(vector_store)
            file_batch = self.client.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store.id,
                files=[open(path, "rb") for path in doc_paths],
            )

            self._uploaded_documents.extend(
                [
                    f.id
                    for f in self.client.vector_stores.file_batches.list_files(
                        vector_store_id=vector_store.id, batch_id=file_batch.id
                    )
                ]
            )
            assert file_batch.status == "completed", file_batch.to_json()

        # create ASSISTANT

        assistant = self.client.beta.assistants.create(
            name=assistant_name,
            instructions=(
                "You are a chatbot. Use your knowledge base (uploaded documents) "
                "to respond to asked questions."
            ),
            # enable RAG ~ file search tool
            tools=[{"type": "file_search"}],
            model=llm_model_name,
            # RAG (collection) documents
            tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
        )
        self._created_assistants[assistant.id] = assistant
        return assistant.id

    def list_collections(self, offset: int = 0, limit: int = 10) -> list:
        """List OpenAI Assistants with file search tool enabled.

        Parameters
        ----------
        offset : int
            Offset of the returned assistants - is always 0 in case of OpenAI
            implementation.
        limit : int
            Limit the number of returned assistants.

        Returns
        -------
        List : list[Assistant]
            List of assistant instances.

        """
        result = list(self.client.assistants.list(limit=limit))
        return result

    def purge_collections(self, assistants_ids: list[str] | None = None) -> list:
        """Purge h2oGPTe collections.

        Parameters
        ----------
        assistants_ids : list[str] | None
            List of OpenAI Assistant IDs to be purged. If the list is empty,
            all Assistants created by this instance are purged.

        """
        assistants_ids = assistants_ids or list(self._created_assistants.keys())
        vs_ids = {v.id for v in self._vector_stores}

        deleted_assistant_ids = []
        for assistant_id in assistants_ids:
            try:
                assist = self.client.beta.assistants.retrieve(assistant_id)
                for v in assist.tool_resources.file_search.vector_store_ids:
                    if v in vs_ids:
                        self.client.vector_stores.delete(v)
                self.client.beta.assistants.delete(assistant_id)
                deleted_assistant_ids.append(assistant_id)
            except Exception as fex:
                print(f"Failed to purge assistant {assistant_id}: {fex}")

        return deleted_assistant_ids

    def purge_uploaded_docs(self, document_ids: list[str] | None = None) -> list:
        """Purge h2oGPTe uploaded documents.

        Parameters
        ----------
        document_ids : list[str] | None
            List of document IDs to be purged. If the list is empty, all documents
            uploaded by this instance are purged.

        """
        document_ids = document_ids or self._uploaded_documents

        deleted_doc_ids = []
        for document_id in document_ids:
            try:
                self.client.files.delete(document_id)
                deleted_doc_ids.append(document_id)
            except Exception as fex:
                print(f"Failed to purge file {document_id}: {fex}")

        return deleted_doc_ids

    def _ask_collection_1(
        self,
        assistant_id: str,
        prompt: str,
        chunks_limit: int = 0,
        timeout: int = 600,
        # extra parameters
        **kwargs,
    ) -> LlmHostClient.LlmRagAnswer:
        """Ask OpenAI Assistant with file search tool enabled and corpus uploaded.
        This method creates a new thread for the prompt and retrieves the answer
        as well as relevant chunks (if requested).

        Parameters
        ----------
        assistant_id : str
            OpenAI Assistant ID.
        prompt : str
            Prompt to ask.
        include_chunks : int
            Optional parameter to determine also relevant (text) chunks.
        timeout : int
            Timeout in seconds.
        kwargs :
            Optional parameters to be passed to the OpenAI client.

        Returns
        -------
        LlmHostClient.LlmRagAnswer :
            Named tuple with prompt, answer, duration and chunks.

        """
        del chunks_limit

        # create THREAD @ assistant
        thread = self.client.beta.threads.create()

        # create MESSAGE w/ question/prompt @ thread
        message = self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt,
        )
        self.logger.debug(
            f"Message for prompt '{prompt}':\n  {message.model_dump_json()}"
        )

        # RUN the ASSISTANT @ THREAD
        start = time.time()
        assistant_run = self.client.beta.threads.runs.create(
            assistant_id=assistant_id,
            thread_id=thread.id,
            instructions="The user is demanding and requires precise answers.",
        )
        self.logger.debug(
            f"Assistant run for prompt '{prompt}':\n  {assistant_run.model_dump_json()}"
        )
        # WAIT for the assistant run to complete
        assistant_err_statuses = ["failed", "expired", "cancelled"]
        assistant_done_statuses = ["completed"] + assistant_err_statuses
        step = 3  # seconds
        timeout = int(timeout / step)
        countdown = timeout
        while assistant_run.status not in assistant_done_statuses and countdown:
            self.logger.debug(
                f"{countdown}/{timeout}s wait countdown for the run to complete: "
                f"{assistant_run.status}"
            )
            assistant_run = self.client.beta.threads.runs.retrieve(
                thread_id=thread.id, run_id=assistant_run.id
            )
            self.logger.debug(f"  {assistant_run.model_dump_json()}")
            time.sleep(step)  # Assistant API is slow + saving rate limit
            countdown -= 1
        self.logger.debug(
            f"Assistant run for prompt '{prompt}' DONE: {assistant_run.status}\n"
            f"  {assistant_run.model_dump_json()}"
        )
        if assistant_run.status in assistant_err_statuses:
            raise RuntimeError(f"AI Assistant run failed: {assistant_run.status}")

        # get ACTUAL OUTPUT
        thread_messages = self.client.beta.threads.messages.list(
            thread_id=thread.id
        ).data
        if not len(thread_messages):
            raise RuntimeError(
                f"No messages in the thread: {thread.id} for assistant: {assistant_id} "
            )
        assistant_message = thread_messages[0]
        if not len(assistant_message.content):
            raise RuntimeError(
                f"Unable to get context - no assistant message content in "
                f"the thread: {thread.id} for assistant: {assistant_id}"
            )
        message_text = assistant_message.content[0].text.value

        # get figure how to get CONTEXT
        # see ("https://community.openai.com/t/"
        #  "assistant-api-always-return-empty-annotations/489285/48")
        # quote was removed in v1.34.0
        # https://github.com/openai/openai-python/pull/1481/files
        context = []

        self.logger.info(
            f"Prompt :\n  {prompt}\nAnswer :\n  {message_text}\nContext:\n  {context}\n"
        )

        return LlmHostClient.LlmRagAnswer(
            prompt=prompt,
            answer=message_text,
            duration=time.time() - start,
            context=context,
            cost=0.0,  # TODO (cost of the assistant run)
            chat_session_id="",
            chat_message_id="",
        )

    def ask_collection(
        self,
        assistant_id: str,
        prompts: list[str],
        include_chunks: int = 0,
        # extra parameters
        **kwargs,
    ) -> list[LlmHostClient.LlmRagAnswer]:
        """Ask OpenAI Assistant with file search tool enabled and corpus uploaded.
        This method creates a new thread for each prompt and retrieves the answer
        as well as relevant chunks (if requested).

        Parameters
        ----------
        assistant_id : str
            OpenAI Assistant ID.
        prompts : list[str]
            Prompts to ask.
        include_chunks : int
            Optional parameter to determine also relevant (text) chunks.

        """
        if not assistant_id:
            raise ValueError("OpenAI Assistant ID is empty.")
        if not prompts:
            raise ValueError("Prompts are empty.")

        results = []
        for e, prompt in enumerate(prompts):
            results.append(
                self._ask_collection_1(
                    assistant_id=assistant_id,
                    prompt=prompt,
                    chunks_limit=include_chunks,
                    **kwargs,
                )
            )

        return results

    def ask_model(
        self,
        prompts: list[str],
        llm_model_name: str = "",
        is_one_prompt: bool = False,
        # extra parameters
        **kwargs,
    ) -> list[LlmHostClient.LlmRagAnswer]:
        """Ask a OpenAI LLM (base) model (minimalistic version without messages and
        parameterization of system prompts, assisting content, parameters, ...).

        Parameters
        ----------
        prompts : list[str]
            Prompts to ask.
        llm_model_name : str
            Optional LLM model name to use for the answer.
        is_one_prompt : bool
            Optional parameter to decide whether to ask all prompts in one request
            (all prompts will be used as the context for the last prompt) or in
            separate requests.

        Returns
        -------
        LlmHostClient.LlmRagAnswer :
            Named tuple with prompt, answer and duration.

        """
        llm_model_name = llm_model_name or self.default_llm_model_name
        if is_one_prompt:
            messages = [{"role": "user", "content": prompt} for prompt in prompts]
            start = time.time()
            chat_answer = self.client.chat.completions.create(
                messages=messages,
                model=llm_model_name,  # optional LLM model
                **kwargs,
            )
            cost = 0.0  # TODO determine cost from the answer (if possible)
            return [
                LlmHostClient.LlmRagAnswer(
                    prompt=str(prompts),
                    answer=chat_answer.choices[-1].message.content,
                    duration=time.time() - start,
                    context=[],
                    cost=cost,
                    chat_session_id="",
                    chat_message_id="",
                )
            ]
        else:
            results = []
            for prompt in prompts:
                start = time.time()
                chat_answer = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=llm_model_name,  # optional LLM model
                    **kwargs,
                )
                cost = 0.0  # TODO determine cost from the answer (if possible)
                self.logger.debug(chat_answer)
                self.logger.info(
                    f"Prompt: {prompt}\n"
                    f"Answer: {chat_answer.choices[0].message.content}"
                )
                results.append(
                    LlmHostClient.LlmRagAnswer(
                        prompt=prompt,
                        answer=chat_answer.choices[0].message.content,
                        duration=time.time() - start,
                        context=[],
                        cost=cost,
                        chat_session_id="",
                        chat_message_id="",
                    )
                )
            return results

    def ask_model_structured(
        self,
        prompts: list[str],
        output_structure: type[TBaseModel],
        llm_model_name: str = "",
        is_one_prompt: bool = False,
        **kwargs,
    ) -> list[TBaseModel]:
        """Ask a OpenAI LLM (base) model with structured output.

        Parameters
        ----------
        prompts : list[str]
            Prompts to ask.
        output_structure : pydantic.BaseModel
            Pydantic model class defining the expected output structure.
        llm_model_name : str
            Optional LLM model name to use for the answer.
        is_one_prompt : bool
            Optional parameter to decide whether to ask all prompts in one request.
        kwargs :
            Optional extra parameters.

        Returns
        -------
        list[pydantic.BaseModel]
            List of structured outputs parsed into the provided Pydantic model.
        """
        if not HAS_PKG_PYDANTIC:
            commons.raise_opt_import_err("pydantic")
        llm_model_name = llm_model_name or self.default_llm_model_name
        if is_one_prompt:
            messages = [{"role": "user", "content": prompt} for prompt in prompts]
            chat_answer = self.client.beta.chat.completions.parse(
                messages=messages,
                model=llm_model_name,
                response_format=output_structure,
                **kwargs,
            )
            parsed_output = chat_answer.choices[-1].message.parsed
            return [parsed_output]
        else:
            results = []
            for prompt in prompts:
                chat_answer = self.client.beta.chat.completions.parse(
                    messages=[{"role": "user", "content": prompt}],
                    model=llm_model_name,
                    response_format=output_structure,
                    **kwargs,
                )
                parsed_output = chat_answer.choices[0].message.parsed
                self.logger.debug(chat_answer)
                self.logger.info(f"Prompt: {prompt}\nAnswer: {parsed_output}")
                results.append(parsed_output)
            return results


def __get_oai_client_class():
    """Get the OpenAI client class based on the installed Python client library
    version.

    Returns
    -------
    OpenAiAssistantsRagClientVersion1 | OpenAiAssistantsRagClientVersion2 :
        OpenAI client class.

    """
    if not HAS_PKG_OPENAI:
        commons.raise_opt_import_err("openai")

    new_cls = OpenAiAssistantsRagClientVersion2
    try:
        if packaging.version.Version(openai.__version__) <= packaging.version.Version(
            "1.20"
        ):
            new_cls = OpenAiAssistantsRagClientVersion1
    except ImportError as e:
        logger = loggers.SonarPrintLogger()
        logger.error(
            f"Using the latest version of OpenAI Python client because the client"
            f"import failed: {e}\n{traceback.format_exc()}"
        )
        pass
    return new_cls


# Open AI RAG client is initialized based on the installed Python client library version
# deferred initialization to avoid import errors when openai is not installed
OpenAiAssistantsRagClient = None
if HAS_PKG_OPENAI:
    OpenAiAssistantsRagClient = __get_oai_client_class()


class H2oGptLlmClient(OpenAiLlmClient):
    """h2oGPT client - connects to the h2oGPT server:

    - OpenAI client is used to connect to h2oGPT, h2ogpt_client is DEPRECATED.
    - standalone h2oGPT server connection config:

      - server URL examples:

        - http://0.0.0.0:7860
        - https://fc752f297207f01c32.gradio.live
        - https://gpt.h2o.ai/

      - API key:

        - required, cannot be generated, must be provided by the h2oGPT server admin

    - Hugging Face Space hosted h2oGPT connection config:

      - server URL examples:

        - h2oai/h2ogpt-chatbot

      - API key:

        - required, cannot be generated, must be provided by the h2oGPT server admin

    See: https://github.com/h2oai/h2ogpt/blob/main/docs/README_CLIENT.md

    """

    @property
    def client(self):
        return self._client or self._create_client()

    def __init__(
        self,
        connection: h2o_sonar_config.ConnectionConfig,
        logger: loggers.SonarLogger | None = None,
    ):
        """h2oGPT client constructor.

        Parameters
        ----------
        connection : h2o_sonar.config.ConnectionConfig
            h2oGPTe connection configuration.
        logger : loggers.SonarLogger | None
            Optional logger.

        """
        if not connection:
            raise ValueError("h2oGPT connection configuration is empty.")
        if (
            not connection.connection_type
            == h2o_sonar_config.ConnectionConfigType.H2O_GPT.name
        ):
            raise ValueError(
                f"Provide h2oGPT connection - connection type "
                f"'{connection.connection_type}' is not supported by H2oGptLlmClient."
            )
        if not connection.token:
            raise ValueError(
                "API key is required to be set as 'token' on the connection, but it is "
                "empty"
            )

        OpenAiLlmClient.__init__(
            self,
            connection=connection,
            logger=logger,
        )

    #
    # methods to create the client and ask the model are inherited from OpenAiLlmClient
    #

    def __deprecated_create_client(self):
        """DEPRECATED h2oGPT client creation."""
        if not HAS_PKG_H2OGPT_CLIENT:
            commons.raise_opt_import_err("h2ogpt_client")

        server_url = (
            self.connection.server_url[:-1]
            if (self.connection.server_url and self.connection.server_url.endswith("/"))
            else self.connection.server_url
        )
        self._client = Client(
            src=server_url,
            h2ogpt_key=self.connection.token,
        )

        return self._client

    def __deprecated_ask_model(
        self,
        prompts: list[str],
        llm_model_name: str = "",
        **extra_params,
    ) -> list[LlmHostClient.LlmRagAnswer]:
        """DEPRECATED ask a h2oGPT hosted LLM model."""
        results = []

        chat_completion = self.client.chat_completion.create(model=llm_model_name)
        for i, q in enumerate(prompts):
            self.logger.debug(
                f"\n>>>RUNNING Q{i + 1}:[{llm_model_name}]: {q}\n",
                flush=True,
            )

            start = time.time()
            a = chat_completion.chat_sync(q)
            a = a.get("gpt", "") if isinstance(a, dict) else str(a)
            results.append(
                LlmHostClient.LlmRagAnswer(
                    prompt=q,
                    answer=a,
                    duration=time.time() - start,
                    context=[],
                    cost=0,
                    chat_session_id="",
                    chat_message_id="",
                )
            )

            self.logger.debug(
                f"\n>>>Q{i + 1}: {q}\n>>>A{i + 1}[{llm_model_name}]: {a}\n",
                flush=True,
            )
        return results


class H2oLlmOpsClient(OpenAiLlmClient):
    """H2O LLMOps client.

    LLMs hosted by H2O LLMOps can be accessed either using the OpenAI API
    or the H2O GPT client. This client is based on``OpenAiLlmClient``.

    """

    def __init__(
        self,
        connection: h2o_sonar_config.ConnectionConfig,
        logger: loggers.SonarLogger | None = None,
    ):
        """H2O LLMOps client constructor.

        Parameters
        ----------
        connection : h2o_sonar.config.ConnectionConfig
            h2oGPTe connection configuration.
        logger : loggers.SonarLogger | None
            Optional logger.

        """
        if not connection:
            raise ValueError("H2O LLMOps connection configuration is empty.")
        if (
            not connection.connection_type
            == h2o_sonar_config.ConnectionConfigType.H2O_LLM_OPS.name
        ):
            raise ValueError(
                f"Provide H2O LLMOps connection - connection type "
                f"'{connection.connection_type}' is not supported by "
                f"H2oLlmOpsClient."
            )
        if not connection.token:
            raise ValueError(
                "API key is required to be set as 'token' on the connection, "
                "but it is empty"
            )

        OpenAiLlmClient.__init__(
            self,
            connection=connection,
            logger=logger,
        )

    def _get_default_llm_model_name(self):
        return self.list_llm_model_names()[0]


"""Ollama LLM attributes dictionary based on the model file which are used also
    in the generation API.

    Attributes
    ----------
    mirorstat : int
        Enable Mirostat sampling for controlling perplexity.
        (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)
    mirorstat_eta : float
        Influences how quickly the algorithm responds to feedback from the generated
        text. A lower learning rate will result in slower adjustments, while a higher
        learning rate will make the algorithm more responsive. (Default: 0.1)
    mirorstat_tau : float
        Controls the balance between coherence and diversity of the output. A lower
        value will result in more focused and coherent text. (Default: 5.0)
    num_ctx : int
        Sets the size of the context window used to generate the next token.
        (Default: 4096)
    repeat_last_n : int
        Sets how far back for the model to look back to prevent repetition.
        (Default: 64, 0 = disabled, -1 = num_ctx)
    repeat_penalty : float
        Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will
        penalize repetitions more strongly, while a lower value (e.g., 0.9) will be
        more lenient. (Default: 1.1)
    temperature : float
        The temperature of the model. Increasing the temperature will make the model
        answer more creatively. (Default: 0.7)
    seed : int
        Sets the random number seed to use for generation. Setting this to a specific
        number will make the model generate the same text for the same prompt.
        (Default: 42)
    stop : str
        Sets the stop sequences to use. When this pattern is encountered the LLM will
        stop generating text and return. Multiple stop patterns may be set by
        specifying multiple separate stop parameters in a model file.
    tfs_z : float
        Tail free sampling is used to reduce the impact of less probable tokens from
        the output. A higher value (e.g., 2.0) will reduce the impact more, while a
        value of 1.0 disables this setting. (default: 1)
    num_predict : int
        Maximum number of tokens to predict when generating text.
        (Default: 128, -1 = infinite generation, -2 = fill context)
    top_k : int
        Reduces the probability of generating nonsense. A higher value (e.g. 100) will
        give more diverse answers, while a lower value (e.g. 10) will be more
        conservative. (Default: 40)
    top_p : float
        Works together with top-k. A higher value (e.g., 0.95) will lead to more
        diverse text, while a lower value (e.g., 0.5) will generate more focused and
        conservative text. (Default: 0.9)

"""


class TypedOllamaModelFileDict(TypedDict):
    mirorstat: int
    mirorstat_eta: float
    mirorstat_tau: float
    num_ctx: int
    repeat_last_n: int
    repeat_penalty: float
    temperature: float
    seed: int
    stop: str
    tfs_z: float
    num_predict: int
    top_k: int
    top_p: float


class OllamaClient(LlmHostClient):
    """Ollama client.

    See https://ollama.com/

    """

    """LLM host or RAG answer."""

    """Ollama client configuration dictionary.

    Attributes
    ----------
    images : list[str] | None
        Optional list of base64-encoded images for multimodal models like `llava`.
    format : str
        The format to return a response in. Currently the only accepted value is
        `json`.
    options : TypedOllamaModelFileDict | None
        Additional model parameters listed in the documentation for the model file
        such as `temperature`.
    system : str | None
        System message to (overrides what is defined in the model file).
    context : list[str] | None
        The context parameter returned from a previous request to /generate, this
        can be used to keep a short conversational memory.
    raw : bool
        If true no formatting will be applied to the prompt. You may choose to use
        the raw parameter if you are specifying a full templated prompt in your
        request to the API. Example of the raw prompt in case that `raw` is set
        to `true`: `[INST] why is the sky blue? [/INST]`.

    Attributes `stream` and `keep_alive` are intentionally omitted.

    See https://github.com/ollama/ollama/blob/main/docs/api.md#parameters

    """
    TypedOllamaConfigDict = TypedDict(
        "TypedOllamaConfigDict",
        {
            "images": list[str] | None,
            "format": str,
            "options": TypedOllamaModelFileDict | None,
            "system": str | None,
            "context": str | None,
            "raw": bool,
        },
    )

    @staticmethod
    def config_factory() -> dict:
        return {
            "images": None,
            # "format": "json",  # this causes my ollama deployment to hang @ GPU
            "options": {
                # "mirorstat": 0,  # this setting leads to empty output
                # "mirorstat_eta": 0.1,  # this setting leads to empty output
                # "mirorstat_tau": 5.0,  # this setting leads to empty output
                "num_ctx": 4096,
                "repeat_last_n": 64,
                "repeat_penalty": 1.1,  # this setting may truncate output in the middle
                "temperature": 0.7,
                "seed": 42,
                "stop": None,  # any string I tried caused the output to be empty
                "tfs_z": 1.0,
                "num_predict": 128,
                "top_k": 40,
                "top_p": 0.9,
            },
            "system": None,
            "context": None,
            "raw": False,
        }

    @property
    def client(self):
        raise NotImplementedError

    def __init__(
        self,
        connection: h2o_sonar_config.ConnectionConfig,
        logger: loggers.SonarLogger | None = None,
    ):
        self.connection = connection

        self.server_url = (
            self.connection.server_url[:-1]
            if (self.connection.server_url and self.connection.server_url.endswith("/"))
            else self.connection.server_url
        )

        self.logger = logger or loggers.SonarPrintLogger()

    def list_llm_model_names(self) -> list[str]:
        llm_model_names = []

        url_list_models = f"{self.server_url}/api/tags"

        response = requests.get(
            url=url_list_models,
            verify=h2o_sonar_config.config.http_ssl_cert_verify,
        )

        if response.status_code != 200:
            raise ValueError(
                f"Failed to list LLM models: {response.status_code} - {response.text}"
            )

        response_json = response.json()
        if not response_json:
            raise ValueError("Failed to list LLM models: empty response")

        for models_item in response_json.get("models", []):
            llm_model_name = models_item.get("name", "")
            if llm_model_name:
                llm_model_names.append(llm_model_name)

        return llm_model_names

    def ask_model(
        self,
        prompts: list[str],
        llm_model_name: str = "",
        **extra_params,
    ) -> list:
        url_completion = f"{self.server_url}/api/generate"

        # prepare the data and extend it with ollama model configuration (if any)
        data_dict = {
            "model": llm_model_name,
            "stream": False,
        }
        if extra_params:
            for k in extra_params:
                data_dict[k] = extra_params[k]

        results = []
        for i, q in enumerate(prompts):
            self.logger.debug(
                f"\n>>>RUNNING Q{i + 1}:[{llm_model_name}]: {q}\n",
                flush=True,
            )

            # set the prompt
            data_dict["prompt"] = q

            start = time.time()
            response = requests.post(
                url=url_completion,
                data=json.dumps(data_dict),
                verify=h2o_sonar_config.config.http_ssl_cert_verify,
            )
            response_json = response.json()
            a = response_json.get("response", "")
            results.append(
                LlmHostClient.LlmRagAnswer(
                    prompt=q,
                    answer=a,
                    duration=time.time() - start,
                    context=[],
                    cost=0,
                    chat_session_id="",
                    chat_message_id="",
                )
            )

            self.logger.debug(
                f"\n>>>Q{i + 1}: {q}\n>>>A{i + 1}[{llm_model_name}]: {a}\n",
                flush=True,
            )

        return results

    def ask_model_structured(
        self,
        prompts: list[str],
        output_structure: type[TBaseModel],
        llm_model_name: str = "",
        **extra_params,
    ) -> list[TBaseModel]:
        """Ask Ollama LLM model with structured output.

        Parameters
        ----------
        prompts : list[str]
            Prompts to ask.
        output_structure : pydantic.BaseModel
            Pydantic model class defining the expected output structure.
        llm_model_name : str
            LLM model name.
        extra_params :
            Optional extra parameters.

        Returns
        -------
        list[pydantic.BaseModel]
            List of structured outputs parsed into the provided Pydantic model.
        """
        if not HAS_PKG_PYDANTIC:
            commons.raise_opt_import_err("pydantic")
        url_completion = f"{self.server_url}/api/generate"

        # prepare the data and extend it with ollama model configuration (if any)
        data_dict = {
            "model": llm_model_name,
            "stream": False,
            "format": output_structure.model_json_schema(),
        }
        if extra_params:
            for k in extra_params:
                data_dict[k] = extra_params[k]

        results = []
        for i, q in enumerate(prompts):
            self.logger.debug(
                f"\n>>>RUNNING Q{i + 1}:[{llm_model_name}]: {q}\n",
                flush=True,
            )

            # get the schema and format it nicely for the prompt
            schema = output_structure.model_json_schema()
            # extract just the properties and required fields for a cleaner prompt
            properties = schema.get("properties", {})
            required_fields = schema.get("required", [])

            # create a simple description of the expected fields
            field_descriptions = []
            for field_name, field_info in properties.items():
                field_type = field_info.get("type", "any")
                field_desc = field_info.get("description", "")
                req = " (required)" if field_name in required_fields else ""
                if field_desc:
                    field_descriptions.append(
                        f'  "{field_name}": {field_type}{req} - {field_desc}'
                    )
                else:
                    field_descriptions.append(f'  "{field_name}": {field_type}{req}')

            fields_text = "\n".join(field_descriptions)

            # add schema information to prompt
            structured_prompt = (
                f"{q}\n\n"
                f"Please respond with ONLY valid JSON matching this structure:\n"
                f"{{\n{fields_text}\n}}\n"
                f"Do not include any explanatory text, only the JSON object."
            )
            data_dict["prompt"] = structured_prompt

            response = requests.post(
                url=url_completion,
                data=json.dumps(data_dict),
                verify=h2o_sonar_config.config.http_ssl_cert_verify,
            )
            response_json = response.json()
            response_text = response_json.get("response", "").strip()

            # strip markdown code blocks if present (e.g., ```json ... ```)
            if response_text.startswith("```"):
                # find the first newline after the opening ```
                first_newline = response_text.find("\n")
                if first_newline != -1:
                    response_text = response_text[first_newline + 1 :]
                # remove the closing ```
                if response_text.endswith("```"):
                    response_text = response_text[:-3].strip()

            # parse JSON response into pydantic model
            parsed_output = output_structure.model_validate_json(response_text)

            results.append(parsed_output)

            self.logger.debug(
                f"\n>>>Q{i + 1}: {q}\n>>>A{i + 1}[{llm_model_name}]: {parsed_output}\n",
                flush=True,
            )

        return results

    def health_check(self, llm_model_name: str) -> bool:
        """Check if the judge is healthy and available."""
        self.ask_model(
            prompts=["If you are working normally, then answer: 1"],
            llm_model_name=llm_model_name,
        )
        return True


@dataclasses.dataclass
class AmazonBedrockKnowledgeBase:
    id: str
    name: str
    description: str
    status: str
    updated_at: datetime.datetime


@dataclasses.dataclass
class AmazonBedrockFoundationModel:
    model_arn: str
    model_id: str
    model_name: str
    customizations_supported: list[Literal["FINE_TUNING", "CONTINUED_PRE_TRAINING"]]
    inference_types_supported: list[Literal["ON_DEMAND", "PROVISIONED"]]
    input_modalities: list[str]
    model_lifecycle_status: str
    output_modalities: list[Literal["TEXT", "IMAGE", "EMBEDDING"]]
    provider_name: str
    response_streaming_supported: bool


class AwsClient:
    def __set_name__(self, owner, name: str):
        self.client_name = name.replace("_", "-")

    def __get__(self, obj, objtype=None):
        return obj._aws_clients.get(self.client_name) or obj._create_client(
            self.client_name
        )


class AwsResource:
    def __set_name__(self, owner, name: str):
        self.resource_name = name.replace("_resource", "").replace("_", "-")

    def __get__(self, obj, objtype=None):
        return obj._aws_resources.get(self.resource_name) or obj._create_resource(
            self.resource_name
        )


@contextlib.contextmanager
def log_action(logger: loggers.SonarLogger, description: str, indent: int = 0):
    text = f"{'  ' * indent}{description}"
    try:
        yield
        text = f"{text} ... DONE"
    except Exception:
        text = f"{text} ... FAIL"
    finally:
        logger.info(text)


class AmazonBedrockRagClient(RagClient):
    _cache_enabled_models = cachetools.TTLCache(ttl=60 * 60, maxsize=256)

    ES_TEMP_PREFIX = "es-temp"

    s3 = AwsClient()
    sts = AwsClient()
    iam = AwsClient()
    bedrock = AwsClient()
    bedrock_agent = AwsClient()
    bedrock_agent_runtime = AwsClient()
    bedrock_runtime = AwsClient()
    opensearchserverless = AwsClient()

    s3_resource = AwsResource()

    @property
    def connection(self) -> h2o_sonar_config.ConnectionConfig:
        return self._connection

    def __init__(
        self,
        connection: h2o_sonar_config.ConnectionConfig,
        logger: loggers.SonarLogger | None = None,
    ):
        """Bedrock RAG client constructor.

        Parameters
        ----------
        connection : h2o_sonar.config.ConnectionConfig
            h2oGPTe connection configuration.
        logger : loggers.SonarLogger | None
            Optional logger.

        """
        if not connection:
            raise ValueError("Bedrock connection configuration is empty.")
        if (
            not connection.connection_type
            == h2o_sonar_config.ConnectionConfigType.AMAZON_BEDROCK.name
        ):
            raise ValueError(
                f"Provide Bedrock connection - connection type "
                f"'{connection.connection_type}' is not supported by BedrockRagClient."
            )

        self._aws_access_key_id = connection.username
        self._aws_secret_access_key = connection.password
        self._aws_session_token = connection.token

        self.logger = logger or loggers.SonarPrintLogger()
        self._aws_clients = dict()
        self._aws_resources = dict()
        self._region = connection.extra_params.get("region", "us-east-1")
        self._extra_params = connection.extra_params
        self._connection = connection
        self._created_collections = []

    def _create_client(self, client: str):
        if not HAS_PKG_BOTO3:
            commons.raise_opt_import_err("boto3")

        self._aws_clients[client] = boto3.client(
            client,
            region_name=self._region,
            aws_access_key_id=self._aws_access_key_id,
            aws_secret_access_key=self._aws_secret_access_key,
            aws_session_token=self._aws_session_token,
        )

        return self._aws_clients[client]

    def _create_resource(self, resource: str):
        if not HAS_PKG_BOTO3:
            commons.raise_opt_import_err("boto3")

        self._aws_resources[resource] = boto3.resource(
            resource,
            region_name=self._region,
            aws_access_key_id=self._aws_access_key_id,
            aws_secret_access_key=self._aws_secret_access_key,
            aws_session_token=self._aws_session_token,
        )

        return self._aws_resources[resource]

    def _create_bucket(self, bucket_name: str):
        self.s3.create_bucket(
            Bucket=bucket_name,
            ACL="private",
        )

    def _delete_bucket(self, bucket_name: str):
        self.s3_resource.Bucket(bucket_name).objects.all().delete()
        self.s3.delete_bucket(Bucket=bucket_name)

    def _upload_data_to_bucket(self, bucket_name: str, doc_paths: pathlib.Path | str):
        for doc in doc_paths:
            self.s3.upload_file(doc, bucket_name, pathlib.Path(doc).name)

    # adapted from Amazon Bedrock Workshop (MIT license):
    # https://github.com/aws-samples/amazon-bedrock-workshop/blob/main
    #   /02_KnowledgeBases_and_RAG/utility.py
    def _create_bedrock_execution_role(self, suffix: str):
        bedrock_execution_role_name = self._bedrock_execution_role_name(suffix)
        bucket_name = self._bucket_name(suffix)
        fm_policy_name = self._fm_policy_name(suffix)

        s3_policy_name = self._s3_policy_name(suffix)
        foundation_model_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:InvokeModel",
                    ],
                    "Resource": [
                        # hardcoded ~ dimensionality
                        f"arn:aws:bedrock:{self._region}::"
                        f"foundation-model/amazon.titan-embed-text-v2:0"
                    ],
                }
            ],
        }

        s3_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:ListBucket"],
                    "Resource": [
                        f"arn:aws:s3:::{bucket_name}",
                        f"arn:aws:s3:::{bucket_name}/*",
                    ],
                    "Condition": {
                        "StringEquals": {
                            "aws:ResourceAccount": self.sts.get_caller_identity().get(
                                "Account"
                            )
                        }
                    },
                }
            ],
        }

        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
        # create policies based on the policy documents
        fm_policy = self.iam.create_policy(
            PolicyName=fm_policy_name,
            PolicyDocument=json.dumps(foundation_model_policy_document),
            Description="Policy for accessing foundation model",
        )

        s3_policy = self.iam.create_policy(
            PolicyName=s3_policy_name,
            PolicyDocument=json.dumps(s3_policy_document),
            Description="Policy for reading documents from S3",
        )

        # create bedrock execution role
        bedrock_kb_execution_role = self.iam.create_role(
            RoleName=bedrock_execution_role_name,
            AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
            Description="Knowledge Base Execution Role for accessing OSS and S3",
            MaxSessionDuration=3600,
        )

        # fetch arn of the policies and role created above
        s3_policy_arn = s3_policy["Policy"]["Arn"]
        fm_policy_arn = fm_policy["Policy"]["Arn"]

        # attach policies to Amazon Bedrock execution role
        self.iam.attach_role_policy(
            RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
            PolicyArn=fm_policy_arn,
        )
        self.iam.attach_role_policy(
            RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
            PolicyArn=s3_policy_arn,
        )
        return bedrock_kb_execution_role

    def _fm_policy_name(self, suffix: str):
        fm_policy_name = (
            f"{self.ES_TEMP_PREFIX}-FoundationModelPolicyForKnowledgeBase_{suffix}"
        )
        return fm_policy_name

    def _bedrock_execution_role_name(self, suffix: str):
        bedrock_execution_role_name = (
            f"{self.ES_TEMP_PREFIX}-ExecutionRoleForKnowledgeBase_{suffix}"
        )
        return bedrock_execution_role_name

    def _create_oss_policy_attach_bedrock_execution_role(
        self, collection_id: str, bedrock_kb_execution_role: dict, suffix: str
    ):
        oss_policy_name = self._oss_policy_name(suffix)
        # define oss policy document
        oss_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["aoss:APIAccessAll"],
                    "Resource": [
                        f"arn:aws:aoss:{self._region}:"
                        f"{self.sts.get_caller_identity().get('Account')}:"
                        f"collection/{collection_id}"
                    ],
                }
            ],
        }
        oss_policy = self.iam.create_policy(
            PolicyName=oss_policy_name,
            PolicyDocument=json.dumps(oss_policy_document),
            Description="Policy for accessing opensearch serverless",
        )
        oss_policy_arn = oss_policy["Policy"]["Arn"]

        self.iam.attach_role_policy(
            RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
            PolicyArn=oss_policy_arn,
        )
        return None

    def _oss_policy_name(self, suffix: str):
        oss_policy_name = f"{self.ES_TEMP_PREFIX}-OSSPolicyForKnowledgeBase_{suffix}"
        return oss_policy_name

    def _delete_roles_and_policies(self, suffix: str):
        account_number = self.sts.get_caller_identity().get("Account")
        bedrock_execution_role_name = self._bedrock_execution_role_name(suffix)
        fm_policy_name = self._fm_policy_name(suffix)
        s3_policy_name = self._s3_policy_name(suffix)
        oss_policy_name = self._oss_policy_name(suffix)
        fm_policy_arn = self._policy_arn(account_number, fm_policy_name)
        s3_policy_arn = self._policy_arn(account_number, s3_policy_name)
        oss_policy_arn = self._policy_arn(account_number, oss_policy_name)
        encryption_policy_name = self._encryption_policy_name(suffix)
        network_policy_name = self._network_policy_name(suffix)
        access_policy_name = self._access_policy_name(suffix)

        with log_action(self.logger, f"Detaching S3 policy '{s3_policy_arn}'", 1):
            self.iam.detach_role_policy(
                RoleName=bedrock_execution_role_name, PolicyArn=s3_policy_arn
            )
        with log_action(
            self.logger, f"Detaching foundation model policy '{fm_policy_arn}'", 1
        ):
            self.iam.detach_role_policy(
                RoleName=bedrock_execution_role_name, PolicyArn=fm_policy_arn
            )
        with log_action(self.logger, f"Detaching OSS policy '{oss_policy_arn}'", 1):
            self.iam.detach_role_policy(
                RoleName=bedrock_execution_role_name, PolicyArn=oss_policy_arn
            )
        with log_action(
            self.logger,
            f"Deleting Bedrock execution role '{bedrock_execution_role_name}'",
            1,
        ):
            self.iam.delete_role(RoleName=bedrock_execution_role_name)
        with log_action(self.logger, f"Deleting s3 policy '{s3_policy_arn}'", 1):
            self.iam.delete_policy(PolicyArn=s3_policy_arn)
        with log_action(
            self.logger, f"Deleting foundation model policy '{fm_policy_arn}'", 1
        ):
            self.iam.delete_policy(PolicyArn=fm_policy_arn)
        with log_action(self.logger, f"Deleting OSS policy '{oss_policy_arn}'", 1):
            self.iam.delete_policy(PolicyArn=oss_policy_arn)
        with log_action(
            self.logger, f"Deleting encryption policy '{encryption_policy_name}'", 1
        ):
            self.opensearchserverless.delete_security_policy(
                name=encryption_policy_name, type="encryption"
            )
        with log_action(
            self.logger, f"Deleting network policy '{network_policy_name}'", 1
        ):
            self.opensearchserverless.delete_security_policy(
                name=network_policy_name, type="network"
            )
        with log_action(
            self.logger, f"Deleting access policy '{access_policy_name}'", 1
        ):
            self.opensearchserverless.delete_access_policy(
                name=access_policy_name, type="data"
            )

    @staticmethod
    def _policy_arn(account_number: str, policy_name: str):
        return f"arn:aws:iam::{account_number}:policy/{policy_name}"

    def _s3_policy_name(self, suffix: str):
        s3_policy_name = f"{self.ES_TEMP_PREFIX}-S3PolicyForKnowledgeBase_{suffix}"
        return s3_policy_name

    def _create_vector_store(self, bedrock_kb_execution_role: dict, suffix: str):
        if not HAS_PKG_OPENSEARCHPY:
            commons.raise_opt_import_err("opensearchpy")
        if not HAS_PKG_BOTO3:
            commons.raise_opt_import_err("boto3")

        vector_store_name = self._vector_store_name(suffix)
        bedrock_kb_execution_role_arn = bedrock_kb_execution_role["Role"]["Arn"]

        self._create_policies_in_oss(
            bedrock_kb_execution_role_arn=bedrock_kb_execution_role_arn,
            suffix=suffix,
        )
        collection = self.opensearchserverless.create_collection(
            name=vector_store_name, type="VECTORSEARCH"
        )
        collection_id = collection["createCollectionDetail"]["id"]
        host = f"{collection_id}.{self._region}.aoss.amazonaws.com"
        response = self.opensearchserverless.batch_get_collection(
            names=[vector_store_name]
        )
        # periodically check collection status
        while (response["collectionDetails"][0]["status"]) == "CREATING":
            self.logger.info(
                f"Creating collection {collection_id} for {vector_store_name}..."
            )
            time.sleep(30)
            response = self.opensearchserverless.batch_get_collection(
                names=[vector_store_name]
            )

        self._create_oss_policy_attach_bedrock_execution_role(
            collection_id=collection_id,
            bedrock_kb_execution_role=bedrock_kb_execution_role,
            suffix=suffix,
        )
        time.sleep(60)

        credentials = boto3.Session(
            region_name=self._region,
            aws_access_key_id=self._aws_access_key_id,
            aws_secret_access_key=self._aws_secret_access_key,
            aws_session_token=self._aws_session_token,
        ).get_credentials()
        awsauth = opensearchpy.AWSV4SignerAuth(credentials, self._region, "aoss")

        index_name = self._index_name(suffix)
        body_json = {
            "settings": {
                "index.knn": "true",
                "number_of_shards": 1,
                "knn.algo_param.ef_search": 512,
                "number_of_replicas": 0,
            },
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": 1024,  # 1536 ~ changed as embedding has 1024 dims
                        "method": {
                            "name": "hnsw",
                            "engine": "faiss",
                            "space_type": "l2",
                        },
                    },
                    "text": {"type": "text"},
                    "text-metadata": {"type": "text"},
                }
            },
        }

        # build the OpenSearch client
        oss_client = opensearchpy.OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=opensearchpy.RequestsHttpConnection,
            timeout=300,
        )

        response = oss_client.indices.create(
            index=index_name, body=json.dumps(body_json)
        )
        self.logger.info(f"Creating index: {response}")

        # index creation can take up to a minute
        time.sleep(60)

        return collection["createCollectionDetail"]["arn"], collection_id, index_name

    def _index_name(self, suffix: str):
        index_name = f"{self.ES_TEMP_PREFIX}-index-{suffix}"
        return index_name

    def _delete_vector_store(self, index_name: str, collection_id: str):
        if not HAS_PKG_OPENSEARCHPY:
            commons.raise_opt_import_err("opensearchpy")
        if not HAS_PKG_BOTO3:
            commons.raise_opt_import_err("boto3")
        self.opensearchserverless.delete_collection(id=collection_id)

    def _create_policies_in_oss(self, bedrock_kb_execution_role_arn: str, suffix: str):
        vector_store_name = self._vector_store_name(suffix)
        encryption_policy_name = self._encryption_policy_name(suffix)
        network_policy_name = self._network_policy_name(suffix)
        access_policy_name = self._access_policy_name(suffix)

        encryption_policy = self.opensearchserverless.create_security_policy(
            name=encryption_policy_name,
            policy=json.dumps(
                {
                    "Rules": [
                        {
                            "Resource": ["collection/" + vector_store_name],
                            "ResourceType": "collection",
                        }
                    ],
                    "AWSOwnedKey": True,
                }
            ),
            type="encryption",
        )

        network_policy = self.opensearchserverless.create_security_policy(
            name=network_policy_name,
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {
                                "Resource": ["collection/" + vector_store_name],
                                "ResourceType": "collection",
                            }
                        ],
                        "AllowFromPublic": True,
                    }
                ]
            ),
            type="network",
        )
        access_policy = self.opensearchserverless.create_access_policy(
            name=access_policy_name,
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {
                                "Resource": ["collection/" + vector_store_name],
                                "Permission": [
                                    "aoss:CreateCollectionItems",
                                    "aoss:DeleteCollectionItems",
                                    "aoss:UpdateCollectionItems",
                                    "aoss:DescribeCollectionItems",
                                ],
                                "ResourceType": "collection",
                            },
                            {
                                "Resource": ["index/" + vector_store_name + "/*"],
                                "Permission": [
                                    "aoss:CreateIndex",
                                    "aoss:DeleteIndex",
                                    "aoss:UpdateIndex",
                                    "aoss:DescribeIndex",
                                    "aoss:ReadDocument",
                                    "aoss:WriteDocument",
                                ],
                                "ResourceType": "index",
                            },
                        ],
                        "Principal": [
                            self.sts.get_caller_identity()["Arn"],
                            bedrock_kb_execution_role_arn,
                        ],
                        "Description": "Temporary access policy",
                    }
                ]
            ),
            type="data",
        )
        return encryption_policy, network_policy, access_policy

    def _access_policy_name(self, suffix: str):
        access_policy_name = f"{self.ES_TEMP_PREFIX}-pol-acc-{suffix}"
        return access_policy_name

    def _network_policy_name(self, suffix: str):
        network_policy_name = f"{self.ES_TEMP_PREFIX}-pol-net-{suffix}"
        return network_policy_name

    def _encryption_policy_name(self, suffix: str):
        encryption_policy_name = f"{self.ES_TEMP_PREFIX}-pol-enc-{suffix}"
        return encryption_policy_name

    @retrying.retry(
        wait_random_min=1000, wait_random_max=2000, stop_max_attempt_number=7
    )
    def _create_knowledge_base_func(
        self,
        name: str,
        description: str,
        collection_arn: str,
        index_name: str,
        role_arn: str,
    ):
        opensearchserverless_configuration = {
            "collectionArn": collection_arn,
            "vectorIndexName": index_name,
            "fieldMapping": {
                "vectorField": "vector",
                "textField": "text",
                "metadataField": "text-metadata",
            },
        }

        # the embedding model used by Bedrock to embed documents, and realtime prompts
        embedding_model_arn = (
            # hardcoded ~ dimensionality
            f"arn:aws:bedrock:{self._region}::"
            f"foundation-model/amazon.titan-embed-text-v2:0"
        )

        create_kb_response = self.bedrock_agent.create_knowledge_base(
            name=name,
            description=description,
            roleArn=role_arn,
            knowledgeBaseConfiguration={
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {
                    "embeddingModelArn": embedding_model_arn
                },
            },
            storageConfiguration={
                "type": "OPENSEARCH_SERVERLESS",
                "opensearchServerlessConfiguration": opensearchserverless_configuration,
            },
        )
        return create_kb_response["knowledgeBase"]

    def _suffix(self, collection_name: str):
        return hashlib.md5(
            collection_name.encode("utf-8"), usedforsecurity=False
        ).hexdigest()[:16]

    def _create_knowledge_base(
        self, doc_paths: list[pathlib.Path | str], collection_name: str
    ):
        suffix = self._suffix(collection_name)
        bucket_name = self._bucket_name(suffix)
        knowledge_base_name = self._knowledge_base_name(suffix)
        knowledge_base_id = None
        data_source_id = None
        collection_id = None
        try:
            index_name = None
            collection_arn = None

            # create a bucket
            self._create_bucket(bucket_name)

            # upload data
            self._upload_data_to_bucket(bucket_name=bucket_name, doc_paths=doc_paths)

            # create a vector store
            bedrock_kb_execution_role = self._create_bedrock_execution_role(
                suffix=suffix
            )
            bedrock_kb_execution_role_arn = bedrock_kb_execution_role["Role"]["Arn"]
            collection_arn, collection_id, index_name = self._create_vector_store(
                bedrock_kb_execution_role=bedrock_kb_execution_role,
                suffix=suffix,
            )

            # create a knowledgeBase
            kb = self._create_knowledge_base_func(
                name=knowledge_base_name,
                description=f"Temporary knowledge base created from eval studio.\n"
                f"collection_name: {collection_name}\n"
                f"documents: {doc_paths}"[:200],
                collection_arn=collection_arn,
                index_name=index_name,
                role_arn=bedrock_kb_execution_role_arn,
            )
            knowledge_base_id = kb["knowledgeBaseId"]
            s3Configuration = {
                "bucketArn": f"arn:aws:s3:::{bucket_name}",
            }

            # ingest strategy - How to ingest data from the data source
            chunkingStrategyConfiguration = {
                "chunkingStrategy": "FIXED_SIZE",
                "fixedSizeChunkingConfiguration": {
                    "maxTokens": 512,
                    "overlapPercentage": 20,
                },
            }

            # create a DataSource in KnowledgeBase
            create_ds_response = self.bedrock_agent.create_data_source(
                name=knowledge_base_name,
                description="Temporary knowledge base created from H2O Eval Studio.",
                knowledgeBaseId=kb["knowledgeBaseId"],
                dataDeletionPolicy="RETAIN",  # delete manually
                dataSourceConfiguration={
                    "type": "S3",
                    "s3Configuration": s3Configuration,
                },
                vectorIngestionConfiguration={
                    "chunkingConfiguration": chunkingStrategyConfiguration
                },
            )
            ds = create_ds_response["dataSource"]
            data_source_id = ds["dataSourceId"]

            self.bedrock_agent.get_data_source(
                knowledgeBaseId=knowledge_base_id, dataSourceId=data_source_id
            )
            time.sleep(30)
            start_job_response = self.bedrock_agent.start_ingestion_job(
                knowledgeBaseId=knowledge_base_id, dataSourceId=data_source_id
            )
            job = start_job_response["ingestionJob"]
            while job["status"] != "COMPLETE":
                get_job_response = self.bedrock_agent.get_ingestion_job(
                    knowledgeBaseId=knowledge_base_id,
                    dataSourceId=data_source_id,
                    ingestionJobId=job["ingestionJobId"],
                )
                job = get_job_response["ingestionJob"]

                time.sleep(30)
            self._created_collections.append(collection_name)
            return knowledge_base_id
        except BaseException as exception:
            self._delete_knowledge_base(
                collection_id=collection_id,
                data_source_id=data_source_id,
                knowledge_base_id=knowledge_base_id,
                suffix=suffix,
            )
            raise exception

    def _knowledge_base_name(self, suffix: str):
        knowledge_base_name = f"{self.ES_TEMP_PREFIX}-kb-{suffix}"
        return knowledge_base_name

    def _vector_store_name(self, suffix: str):
        vector_store_name = f"{self.ES_TEMP_PREFIX}-vs-{suffix}"
        assert len(vector_store_name) <= 32, (
            f"Vector store name is too long "
            f"({len(vector_store_name)}>32). "
            f"Vector store name: {vector_store_name}"
        )
        return vector_store_name

    def _bucket_name(self, suffix: str):
        bucket_name = f"{self.ES_TEMP_PREFIX}-bucket-{suffix}"
        assert re.match(r"^[a-z0-9.\-_]{3,63}$", bucket_name), (
            f"Invalid bucket name: {bucket_name}. "
            "Bucket name must match '^[a-zA-Z0-9.\\-_]{3,63}$'."
        )
        return bucket_name

    def _get_collection_id_by_suffix(self, suffix: str):
        return self.opensearchserverless.batch_get_collection(
            names=[self._vector_store_name(suffix)]
        )["collectionDetails"][0]["id"]

    def _get_knowledge_base_id_by_suffix(self, suffix: str):
        knowledge_bases = [
            kb
            for kb in self.bedrock_agent.list_knowledge_bases()[
                "knowledgeBaseSummaries"
            ]
            if kb["name"] == self._knowledge_base_name(suffix)
        ]
        assert len(knowledge_bases) == 1
        return knowledge_bases[0]["knowledgeBaseId"]

    def _get_data_source_id_by_suffix(self, suffix: str):
        if not HAS_PKG_BOTO3:
            commons.raise_opt_import_err("boto3")

        kb_id = self._get_knowledge_base_id_by_suffix(suffix)
        return self.bedrock_agent.list_data_sources(knowledgeBaseId=kb_id)[
            "dataSourceSummaries"
        ][0]["dataSourceId"]

    def _delete_knowledge_base(
        self,
        collection_name: str = None,
        collection_id: str = None,
        data_source_id: str = None,
        knowledge_base_id: str = None,
        suffix: str = None,
    ):
        assert collection_name or suffix, (
            "Either collection_name or suffix has to be specified!"
        )

        try:
            signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGINT, signal.SIGTERM])
            suffix = suffix or self._suffix(collection_name)
            bucket_name = self._bucket_name(suffix)
            index_name = self._index_name(suffix)
            collection_id = collection_id or self._get_collection_id_by_suffix(suffix)
            data_source_id = data_source_id or self._get_data_source_id_by_suffix(
                suffix
            )
            knowledge_base_id = (
                knowledge_base_id or self._get_knowledge_base_id_by_suffix(suffix)
            )
            with log_action(
                logger=self.logger,
                description=f"Deleting data source '{data_source_id}'",
                indent=0,
            ):
                self.bedrock_agent.delete_data_source(
                    knowledgeBaseId=knowledge_base_id,
                    dataSourceId=data_source_id,
                )
            with log_action(
                logger=self.logger,
                description=f"Deleting knowledge base '{knowledge_base_id}'",
                indent=0,
            ):
                self.bedrock_agent.delete_knowledge_base(
                    knowledgeBaseId=knowledge_base_id,
                )
            with log_action(
                self.logger, f"Deleting IAM role and policies for suffix='{suffix}'", 0
            ):
                self._delete_roles_and_policies(suffix=suffix)

            with log_action(
                logger=self.logger,
                description=(
                    f"Deleting vector store with index_name='{index_name}' for "
                    f"collection_id='{collection_id}'"
                ),
                indent=0,
            ):
                self._delete_vector_store(
                    index_name=index_name, collection_id=collection_id
                )
            with log_action(self.logger, f"Deleting S3 bucket '{bucket_name}'", 0):
                self._delete_bucket(bucket_name)
        finally:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, [signal.SIGINT, signal.SIGTERM])

    def create_collection(
        self,
        doc_paths: list[pathlib.Path | str],
        collection_name: str = "",
        **kwargs,
    ) -> tuple[str, str]:
        try:
            # if the collection exists pretend we created it
            return collection_name, self._resolve_collection_id(collection_name)
        except Exception:
            return collection_name, self._create_knowledge_base(
                doc_paths, collection_name
            )

    def list_collections(
        self, offset: int = 0, limit: int = 1000
    ) -> list[AmazonBedrockKnowledgeBase]:
        assert offset == 0, "Setting offset is not supported in Bedrock RAG API"
        return [
            AmazonBedrockKnowledgeBase(
                id=kb["knowledgeBaseId"],
                name=kb["name"],
                description=kb.get("description", ""),
                status=kb["status"],
                updated_at=kb["updatedAt"],
            )
            for kb in self.bedrock_agent.list_knowledge_bases(maxResults=limit)[
                "knowledgeBaseSummaries"
            ]
        ]

    def purge_collections(self, collection_ids: list[str] | None = None):
        collection_ids = collection_ids or self._created_collections
        collection_list = self.list_collections()
        prefix = self._knowledge_base_name("")
        mapping = {
            kb.id: kb.name[len(prefix) :]
            for kb in collection_list
            if kb.name.startswith(prefix)
        }
        mapping.update(
            {
                kb.name: kb.name[len(prefix) :]
                for kb in collection_list
                if kb.name.startswith(prefix)
            }
        )
        for collection_id in collection_ids:
            self._delete_knowledge_base(
                suffix=mapping.get(collection_id, self._suffix(collection_id))
            )

    def purge_uploaded_docs(self, document_ids: list[str] | None = None):
        print(
            "Purging uploaded docs is not implemented in Amazon Bedrock. "
            "Only the whole collection can be purged."
        )

    @staticmethod
    def _get_context(citations: list[str]):
        # New lines are converted to addition space so this makes sure there is just
        # one space between words. This is motivated by some failure in text matching
        # evaluator, but it looks reasonably safe as a preprocessing.
        import re

        return [
            re.sub(r"\s+", " ", rr["content"]["text"])
            for citation in citations
            for rr in citation.get("retrievedReferences", [])
        ]

    @staticmethod
    def _set_nested_dict(dictionary, path, value):
        assert len(path) > 1
        current = dictionary
        for key in path[:-1]:
            if key not in current:
                current[key] = dict()
            current = current[key]
        current[path[-1]] = value
        return dictionary

    def _single_rag_query(self, collection_id, model_arn, prompt):
        import time

        rag_conf = self.get_rag_conf(collection_id, model_arn)

        start_time = time.monotonic()
        result = self.bedrock_agent_runtime.retrieve_and_generate(
            input={"text": prompt},
            retrieveAndGenerateConfiguration=rag_conf,
        )
        end_time = time.monotonic()

        rag_answer = LlmHostClient.LlmRagAnswer(
            context=self._get_context(result["citations"]),
            answer=result["output"]["text"],
            prompt=prompt,
            cost=0,  # TODO: find some way to count the cost
            duration=end_time - start_time,
            chat_session_id="",
            chat_message_id="",
        )

        return rag_answer

    def get_rag_conf(self, collection_id, model_arn):
        rag_conf = {
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": collection_id,
                "modelArn": model_arn,
            },
        }
        if "guardrailConfiguration" in self._extra_params:
            self._set_nested_dict(
                rag_conf,
                [
                    "knowledgeBaseConfiguration",
                    "generationConfiguration",
                    "guardrailConfiguration",
                ],
                self._extra_params["guardrailConfiguration"],
            )
        if "inferenceConfig" in self._extra_params:
            self._set_nested_dict(
                rag_conf,
                [
                    "knowledgeBaseConfiguration",
                    "generationConfiguration",
                    "inferenceConfig",
                ],
                self._extra_params["inferenceConfig"],
            )
        if "promptTemplate" in self._extra_params:
            self._set_nested_dict(
                rag_conf,
                [
                    "knowledgeBaseConfiguration",
                    "generationConfiguration",
                    "promptTemplate",
                ],
                self._extra_params["promptTemplate"],
            )
        if "orchestrationConfiguration" in self._extra_params:
            self._set_nested_dict(
                rag_conf,
                ["knowledgeBaseConfiguration", "orchestrationConfiguration"],
                self._extra_params["orchestrationConfiguration"],
            )
        if "retrievalConfiguration" in self._extra_params:
            self._set_nested_dict(
                rag_conf,
                ["knowledgeBaseConfiguration", "retrievalConfiguration"],
                self._extra_params["retrievalConfiguration"],
            )
        return rag_conf

    def ask_collection(
        self,
        collection_id: str,
        prompts: list[str],
        llm_model_name: str = "",
        include_chunks: int = 0,
        chunk_retrieval_method: str = RagChunkRetrievalMethod.ANSWER_REFS.name,
        **kwargs,
    ):
        collection_id = self._resolve_collection_id(collection_id)
        model_arn = self._resolve_model(llm_model_name).model_arn
        return [
            self._single_rag_query(collection_id, model_arn, prompt)
            for prompt in prompts
        ]

    def _resolve_collection_id(self, collection_id_or_name):
        if isinstance(collection_id_or_name, tuple) or isinstance(
            collection_id_or_name, list
        ):
            collection_id_or_name = collection_id_or_name[1]
        collection_id_resolved = False
        if len(collection_id_or_name) == 10:
            try:
                self.bedrock_agent.get_knowledge_base(
                    knowledgeBaseId=collection_id_or_name
                )
                collection_id_resolved = True
            except Exception:
                pass
        if not collection_id_resolved:
            cols = self.list_collections()
            for col in cols:
                if (
                    col.name == collection_id_or_name
                    or col.name
                    == self._knowledge_base_name(self._suffix(collection_id_or_name))
                ):
                    collection_id_or_name = col.id
                    collection_id_resolved = True
                    break
                if col.id == collection_id_or_name:
                    collection_id_resolved = True
                    break
        assert collection_id_resolved, (
            f'Collection "{collection_id_or_name}" not found!'
        )
        return collection_id_or_name

    def _resolve_model(self, model) -> AmazonBedrockFoundationModel:
        if isinstance(model, AmazonBedrockFoundationModel):
            return model
        models = self._get_llm_models()
        for m in models:
            if m.model_id == model or m.model_arn == model:
                return m
        # model name doesn't need to be unique; use as last resort
        for m in models:
            if m.model_name == model:
                return m
        # if there are no exact matches let's try fuzzy matching - lets say we want:
        #  'anthropic.claude-3-5-sonnet-2024MMDD-v1:0:200k'
        # but only the following are currently supported:
        #  'anthropic.claude-3-5-sonnet-20240620-v1:0:18k',
        #  'anthropic.claude-3-5-sonnet-20240620-v1:0:51k',
        #  'anthropic.claude-3-5-sonnet-20240620-v1:0:200k',
        #  'anthropic.claude-3-5-sonnet-20240620-v1:0',
        # so we want to choose the closest which in this case will be the third one
        import difflib

        model_identifiers = [m.model_id for m in models] + [m.model_arn for m in models]
        closest_match = difflib.get_close_matches(model, model_identifiers, n=1)
        if len(closest_match) == 1:
            return self._resolve_model(closest_match[0])
        raise ValueError(f'Model "{model}" not found!')

    @staticmethod
    def config_factory() -> dict:
        return super().config_factory()

    @cachetools.cachedmethod(
        lambda cls: cls._cache_enabled_models,
        key=lambda self, *args, **kwargs: cachetools.keys.hashkey(self._connection),
    )
    def _get_llm_models(self):
        # unfortunately, lists all models not just those that were allowed
        models = [
            AmazonBedrockFoundationModel(
                model_id=m["modelId"],
                model_arn=m["modelArn"],
                model_name=m["modelName"],
                customizations_supported=m["customizationsSupported"],
                inference_types_supported=m["inferenceTypesSupported"],
                input_modalities=m["inputModalities"],
                model_lifecycle_status=m["modelLifecycle"]["status"],
                output_modalities=m["outputModalities"],
                provider_name=m["providerName"],
                response_streaming_supported=m["responseStreamingSupported"],
            )
            for m in self.bedrock.list_foundation_models(byOutputModality="TEXT")[
                "modelSummaries"
            ]
        ]
        # only claude models are supported for RAG:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/models-features.html
        return [
            m
            for m in models
            if m.provider_name == "Anthropic"
            and not m.model_id == "anthropic.claude-v2"
        ]

    def list_llm_models(self):
        models = self._get_llm_models()
        # vache has a lock so I'm getting the uncached models and query for them in
        # parallel and then feed the results to the cache
        uncached_models = [
            m
            for m in models
            if (self._connection, m.model_id) not in self._cache_enabled_models
        ]
        if len(uncached_models) > 0:
            connections = [self._connection for _ in uncached_models]
            model_ids = [m.model_id for m in uncached_models]

            with futures.ThreadPoolExecutor(min(len(uncached_models), 32)) as executor:
                enabled_flags = list(
                    executor.map(
                        AmazonBedrockRagClient._is_model_enabled_uncached,
                        connections,
                        model_ids,
                    )
                )

            for i, flag in enumerate(enabled_flags):
                self._cache_enabled_models[(connections[i], model_ids[i])] = flag
        return [m for m in models if self._is_model_enabled(m.model_id)]

    def list_llm_model_names(self):
        # "modelName" is not unique so I use model_id
        return [m.model_id for m in self.list_llm_models()]

    def _is_model_enabled(self, model_id):
        return self.is_model_enabled(self._connection, model_id)

    @classmethod
    @cachetools.cachedmethod(lambda cls: cls._cache_enabled_models)
    def is_model_enabled(cls, connection, model_id):
        return cls._is_model_enabled_uncached(connection, model_id)

    @classmethod
    def _is_model_enabled_uncached(cls, connection, model_id):
        self = cls(connection, None)
        model_id = self._resolve_model(model_id).model_id
        try:
            self._ask_single_prompt(model_id=model_id, prompt="Say yes.")
            return True
        except Exception:
            return False

    def _ask_single_prompt(self, model_id, prompt):
        import time

        additional_options = dict()
        if "guardrailConfiguration" in self._extra_params:
            # guardrailConfig is not a typo, the name differs for the converse api
            gc = self._extra_params["guardrailConfiguration"]
            additional_options["guardrailConfig"] = dict(
                guardrailIdentifier=gc["guardrailId"],
                guardrailVersion=gc["guardrailVersion"],
            )

        if "inferenceConfig" in self._extra_params:
            additional_options["inferenceConfig"] = self._extra_params[
                "inferenceConfig"
            ]["textInferenceConfig"]

        start_time = time.monotonic()
        result = self.bedrock_runtime.converse(
            modelId=model_id,
            messages=[dict(role="user", content=[dict(text=prompt)])],
            **additional_options,
        )
        end_time = time.monotonic()
        assert len(result["output"]["message"]["content"]) == 1
        return LlmHostClient.LlmRagAnswer(
            duration=end_time - start_time,
            prompt=prompt,
            answer=result["output"]["message"]["content"][0]["text"],
            cost=0,  # TODO: Find some way to count the cost
            context=[],
            chat_session_id="",
            chat_message_id="",
        )

    def ask_model(
        self, prompts: list[str], llm_model_name: str = "", **extra_params
    ) -> list:
        model_id = self._resolve_model(llm_model_name).model_id

        return [self._ask_single_prompt(model_id, prompt) for prompt in prompts]

    def ask_model_structured(
        self,
        prompts: list[str],
        output_structure: type[TBaseModel],
        llm_model_name: str = "",
        **extra_params,
    ) -> list[TBaseModel]:
        """Ask Amazon Bedrock LLM model with structured output.

        Parameters
        ----------
        prompts : list[str]
            Prompts to ask.
        output_structure : pydantic.BaseModel
            Pydantic model class defining the expected output structure.
        llm_model_name : str
            LLM model name.
        extra_params :
            Optional extra parameters.

        Returns
        -------
        list[pydantic.BaseModel]
            List of structured outputs parsed into the provided Pydantic model.
        """
        if not HAS_PKG_PYDANTIC:
            commons.raise_opt_import_err("pydantic")
        model_id = self._resolve_model(llm_model_name).model_id

        additional_options = dict()
        if "guardrailConfiguration" in self._extra_params:
            # guardrailConfig is not a typo, the name differs for the converse api
            gc = self._extra_params["guardrailConfiguration"]
            additional_options["guardrailConfig"] = dict(
                guardrailIdentifier=gc["guardrailId"],
                guardrailVersion=gc["guardrailVersion"],
            )

        if "inferenceConfig" in self._extra_params:
            additional_options["inferenceConfig"] = self._extra_params[
                "inferenceConfig"
            ]["textInferenceConfig"]

        results = []
        for prompt in prompts:
            # add schema information to prompt
            structured_prompt = (
                f"{prompt}\n\nPlease respond with valid JSON matching this schema:\n"
                f"{output_structure.model_json_schema()}"
            )

            result = self.bedrock_runtime.converse(
                modelId=model_id,
                messages=[dict(role="user", content=[dict(text=structured_prompt)])],
                **additional_options,
            )

            assert len(result["output"]["message"]["content"]) == 1
            response_text = result["output"]["message"]["content"][0]["text"]

            # check for guardrail or content filtering responses
            if (
                "guardrail" in response_text.lower()
                or "not allowed" in response_text.lower()
                or "cannot" in response_text.lower()
            ):
                raise ValueError(
                    f"Bedrock returned a guardrail/content filter response "
                    f"instead of JSON: {response_text[:200]}"
                )

            # strip markdown code blocks if present (e.g., ```json ... ```)
            response_text = response_text.strip()
            if response_text.startswith("```"):
                # find the first newline after the opening ```
                first_newline = response_text.find("\n")
                if first_newline != -1:
                    response_text = response_text[first_newline + 1 :]
                # remove the closing ```
                if response_text.endswith("```"):
                    response_text = response_text[:-3].strip()

            # parse JSON response into pydantic model
            try:
                parsed_output = output_structure.model_validate_json(response_text)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse Bedrock response as JSON. "
                    f"Response: {response_text[:200]}. Error: {e}"
                )

            results.append(parsed_output)

        return results


def get_client_for_connection(
    connection: h2o_sonar_config.ConnectionConfig,
    logger: loggers.SonarLogger | None = None,
) -> LlmHostClient | RagClient:
    """Get a client for the given connection.

    Parameters
    ----------
    connection : h2o_sonar.config.ConnectionConfig
        Connection configuration.
    logger : loggers.SonarLogger | None
        Optional logger.

    Returns
    -------
    LlmHostClient :
        An LLM host client.

    """
    if not connection:
        raise ValueError("Connection configuration is empty.")
    elif (
        connection.connection_type == h2o_sonar_config.ConnectionConfigType.H2O_GPT.name
    ):
        return H2oGptLlmClient(connection=connection, logger=logger)
    elif (
        connection.connection_type
        == h2o_sonar_config.ConnectionConfigType.H2O_GPT_E.name
    ):
        return H2oGpteRagClient(connection=connection, logger=logger)
    elif (
        connection.connection_type
        == h2o_sonar_config.ConnectionConfigType.H2O_LLM_OPS.name
    ):
        return H2oLlmOpsClient(connection=connection, logger=logger)
    elif (
        connection.connection_type
        == h2o_sonar_config.ConnectionConfigType.AZURE_OPENAI_CHAT.name
    ):
        return MsAzureOpenAiLlmClient(connection=connection, logger=logger)
    elif (
        connection.connection_type == h2o_sonar_config.ConnectionConfigType.OLLAMA.name
    ):
        return OllamaClient(connection=connection, logger=logger)
    elif (
        connection.connection_type
        == h2o_sonar_config.ConnectionConfigType.OPENAI_CHAT.name
    ):
        return OpenAiLlmClient(connection=connection, logger=logger)
    elif (
        connection.connection_type
        == h2o_sonar_config.ConnectionConfigType.OPENAI_RAG.name
    ):
        return OpenAiAssistantsRagClient(connection=connection, logger=logger)
    elif (
        connection.connection_type
        == h2o_sonar_config.ConnectionConfigType.AMAZON_BEDROCK.name
    ):
        return AmazonBedrockRagClient(connection=connection, logger=logger)
    elif (
        connection.connection_type
        == h2o_sonar_config.ConnectionConfigType.ANTHROPIC_CHAT.name
    ):
        return AnthropicClaudeLlmClient(connection=connection, logger=logger)

    raise ValueError(
        f"Unable to construct client for the connection type "
        f"'{connection.connection_type}' that is not supported."
    )
