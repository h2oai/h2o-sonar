# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import loggers
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import judges
from h2o_sonar.lib.integrations import genai
from h2o_sonar.utils import caching


try:
    import ragas.embeddings.base as embeddings_base
    import ragas.llms.base as llms_base
    import ragas.llms.prompt as p4t

    HAS_RAGAS = True
except ImportError:
    HAS_RAGAS = False

try:
    from langchain_core.callbacks import Callbacks
    from langchain_core.outputs import Generation
    from langchain_core.outputs import LLMResult

    HAS_LANGCHAIN_CORE = True
except ImportError:
    HAS_LANGCHAIN_CORE = False

try:
    import sentence_transformers

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from torch import Tensor

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import AutoConfig
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    )

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def get_ragas_to_sonar_llm_adapter(custom_judge: judges.EvaluationJudge, logger=None):
    if not HAS_RAGAS:
        commons.raise_opt_import_err("ragas")
    if not HAS_LANGCHAIN_CORE:
        commons.raise_opt_import_err("langchain-core")

    class RagasLlmToH2oSonarJudge(llms_base.BaseRagasLLM):
        """This is ragas (library) to H2O sonar custom LLM adapter. In order to
        replace default ragas's OpenAI LLM judge/embeddings with a
        custom LLM judge/embeddings, the following classes must be implemented:

        - ``BaseRagasLLM`` is the base class Ragas uses internally for LLMs.
          Any custom LLM should be a subclass of this base class.
          It has 2 abstract methods - synchronous and asynchronous:
          - ``generate_text()``
          - ``agenerate_text()``

        - ``BaseRagasEmbeddings`` is the base class Ragas uses internally
          for Embeddings. Any custom Embeddings should be a subclass of this base
          class.

        ``ragas`` transitively installs Langchain, so alternatively, if Langchain
        is used, you can pass the Langchain LLM and Embeddings directly and Ragas will
        wrap it with LangchainLLMWrapper or LangchainEmbeddingsWrapper as needed.

        """

        def __init__(
            self,
            judge: judges.EvaluationJudge,
            logger: loggers.SonarLogger | None = None,
        ):
            self.judge = judge
            self.logger = logger or loggers.SonarPrintLogger()

        def health_check(self) -> bool:
            return self.judge.health_check()

        def generate_text(
            self,
            prompt: p4t.PromptValue,
            n: int = 1,
            temperature: float = 1e-8,
            stop: list[str] | None = None,
            callbacks: Callbacks = None,
        ) -> LLMResult:
            del callbacks

            self.logger.debug(
                f"Custom LLM judge: SYNC generate_text() prompt: {prompt.to_string()}"
            )

            # TODO convert to LLMResult
            # TODO callbacks
            # TODO stop
            # TODO temperature
            # TODO n
            judge_results: list[genai.LlmHostClient.LlmRagAnswer] = self.judge.evaluate(
                prompts=[prompt.to_string()]
            )
            self.logger.debug(
                f"Ragas LLM > H2O Sonar judge evaluation result: {judge_results}"
            )

            if judge_results:
                judge_result = judge_results[0]
            else:
                raise ValueError("Judge evaluate() returned no results")

            # NORMALIZE to Langchain LLMResult
            generation = Generation(
                text=judge_result.answer,
                generation_info=None,
                type="Generation",
            )
            result = LLMResult(generations=[[generation]], llm_output=None, run=None)

            self.logger.debug(
                f"DONE custom LLM judge: SYNC generate_text() prompt: "
                f"{prompt.to_string()}"
            )
            return result

        async def agenerate_text(
            self,
            prompt: p4t.PromptValue,
            n: int = 1,
            temperature: float = 1e-8,
            stop: list[str] | None = None,
            callbacks: Callbacks = None,
        ) -> LLMResult:
            if callbacks is None:
                callbacks = []
            self.logger.debug(
                f"Custom LLM judge ASYNC generate_text: {prompt.to_string()}"
            )

            #
            # IMPROVE: SYNC call as not all LLM models support async
            #
            return self.generate_text(
                prompt=prompt,
                n=n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )

    custom_llm_for_ragas = RagasLlmToH2oSonarJudge(
        judge=custom_judge, logger=logger or loggers.SonarPrintLogger()
    )

    return custom_llm_for_ragas


def get_ragas_privacy_safe_embeddings(embeddings_provider: str = "huggingface"):
    """``ragas`` uses OpenAI embeddings by default. This function is used to
    use custom embeddings with ragas so that the validation data are not send to
    a 3rd party - **privacy first**.

    ``ragas`` supports Huggingface embeddings - embeddings model
    (``BAAI/bge-small-en-v1.5``) is downloaded from Huggingface model hub and used
    to calculate embeddings locally - data are not send to a 3rd party.

    Parameters
    ----------
    embeddings_provider : str
        Name of the embeddings provider. Currently only "huggingface" is supported.
        It uses ``BAAI/bge-small-en-v1.5`` model.

    """
    if not HAS_RAGAS:
        commons.raise_opt_import_err("ragas")
    if not HAS_SENTENCE_TRANSFORMERS:
        commons.raise_opt_import_err("sentence_transformers")
    if not HAS_TRANSFORMERS:
        commons.raise_opt_import_err("transformers")

    # copy-paste from ragas with revisions enabled
    @dataclass
    class HuggingfaceEmbeddings(embeddings_base.BaseRagasEmbeddings):
        model_name: str = embeddings_base.DEFAULT_MODEL_NAME
        """Model name to use."""
        cache_folder: str | None = None
        """Path to store models.
        Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""
        model_kwargs: dict[str, Any] = field(default_factory=dict)
        """Keyword arguments to pass to the model."""
        encode_kwargs: dict[str, Any] = field(default_factory=dict)

        def __post_init__(self):
            if not HAS_NUMPY:
                commons.raise_opt_import_err("numpy")
            if not HAS_SENTENCE_TRANSFORMERS:
                commons.raise_opt_import_err("sentence_transformers")
            if not HAS_TRANSFORMERS:
                commons.raise_opt_import_err("transformers")

            config = AutoConfig.from_pretrained(
                self.model_name,
                revision=caching.REVISIONS_FOR_MODEL.get(self.model_name, "main"),
            )
            self.is_cross_encoder = bool(
                np.intersect1d(
                    list(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.values()),
                    config.architectures,
                )
            )

            if self.is_cross_encoder:
                self.model = sentence_transformers.CrossEncoder(
                    self.model_name,
                    device=h2o_sonar_config.config.resolve_gpu_cpu_device(
                        result_format="str"
                    ),
                    revision=caching.REVISIONS_FOR_MODEL.get(self.model_name, "main"),
                    **self.model_kwargs,
                )
            else:
                self.model = sentence_transformers.SentenceTransformer(
                    self.model_name,
                    device=h2o_sonar_config.config.resolve_gpu_cpu_device(
                        result_format="str"
                    ),
                    revision=caching.REVISIONS_FOR_MODEL.get(self.model_name, "main"),
                    cache_folder=self.cache_folder,
                    **self.model_kwargs,
                )

            # ensure outputs are tensors
            if "convert_to_tensor" not in self.encode_kwargs:
                self.encode_kwargs["convert_to_tensor"] = True

        def embed_query(self, text: str) -> list[float]:
            return self.embed_documents([text])[0]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            if not HAS_SENTENCE_TRANSFORMERS:
                commons.raise_opt_import_err("sentence_transformers")
            if not HAS_TORCH:
                commons.raise_opt_import_err("torch")

            assert isinstance(self.model, sentence_transformers.SentenceTransformer), (
                "Model is not of the type Bi-encoder"
            )
            embeddings = self.model.encode(
                texts, normalize_embeddings=True, **self.encode_kwargs
            )

            assert isinstance(embeddings, Tensor)
            return embeddings.tolist()

        def predict(self, texts: list[list[str]]) -> list[list[float]]:
            from sentence_transformers.cross_encoder import CrossEncoder
            from torch import Tensor

            assert isinstance(self.model, CrossEncoder), (
                "Model is not of the type CrossEncoder"
            )

            predictions = self.model.predict(texts, **self.encode_kwargs)

            assert isinstance(predictions, Tensor)
            return predictions.tolist()

    if embeddings_provider == "huggingface":
        return HuggingfaceEmbeddings()
    # TODO elseif
    #  implement bridge for embeddings from the H2O Sonar h2oGPT(e) connections
    #  - not all models support embeddings
    #  - new H2O Sonar configuration type entry might be needed
    else:
        raise ValueError(
            f"Embeddings provider {embeddings_provider} is not supported. "
            f"Currently only 'huggingface' is supported."
        )
