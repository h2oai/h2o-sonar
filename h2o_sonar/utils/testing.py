# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
"""H2O Sonar LLM / RAG testing utilities:

- Raw test data:
  a dataset which was used to create the test configuration(s).
- Test suite:
  a collection of tests (see below).
- Test:
  a collection of documents (corpus) along with the test cases (see below) to be
  run in the context of the corpus.
- Test case:
  a prompt, expected output (ground truth), categories, output condition,
  output constraints, ... and other parameters to be used for a RAG / LLM model
  evaluation.
- Test lab:
  a set of resolved tests enriched with answers (actual answer), retrieval context,
  response duration and other data obtained from the conversation with a RAG / LLM.

Resolved test lab is exported to LLM dataset which is then used as input to
an evaluation - evaluation runs a set of evaluators to rank RAG / LLM models.

"""

import abc
import glob
import json
import multiprocessing
import os
import pathlib
import queue
import re
import shutil
import time
import traceback
import uuid
from concurrent import futures
from typing import Any

import requests

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import loggers
from h2o_sonar.lib.api import agents
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import models
from h2o_sonar.lib.integrations import genai
from h2o_sonar.utils import perturbations
from h2o_sonar.utils import progress


# aliases
ExplainableModelTypes = models.ExplainableModelType

#
# config
#

if h2o_sonar_config.config.mp_start_method:
    multiprocessing.set_start_method(
        h2o_sonar_config.config.mp_start_method, force=True
    )


DEFAULT_RETRY_ON_ERROR = 2


class RagTestConfig:
    """RAG / LLM test configuration:

    - corpus
      ... a set of documents (empty for LLM evaluation)

      - test cases
        ... a set of prompts, expected outputs, categories, conditions, ...

    """

    KEY_KEY = "key"
    KEY_DOCUMENTS = "documents"
    KEY_CATS = "categories"

    def __init__(
        self,
        documents: list[str | pathlib.Path],
        categories: list[str] | None = None,
        key: str = "",
    ):
        """Test constructor.

        Parameters
        ----------
        documents : list[str, pathlib.Path]
            URLs or path to documents to be used by RAG as knowledge base (corpus).
        categories : list[str] | None
            Optional list of categories to be used for the test.
            Categories are used to indicate the purpose of the test,
            enable test filtering, and so on.
        key : str
            Test configuration key.

        """
        self.key = key or str(uuid.uuid4())
        self.documents = documents
        self.categories = categories or []

    def to_dict(self) -> dict:
        result = {
            RagTestConfig.KEY_KEY: self.key,
            RagTestConfig.KEY_DOCUMENTS: [str(d) for d in self.documents],
        }
        # categories are NOT serialized if empty (JSon backward compatibility)
        if self.categories:
            result[RagTestConfig.KEY_CATS] = self.categories
        return result

    @staticmethod
    def from_dict(key: str, as_dict: dict) -> "RagTestConfig":
        return RagTestConfig(
            key=key,
            documents=as_dict.get(RagTestConfig.KEY_DOCUMENTS, []),
            categories=as_dict.get(RagTestConfig.KEY_CATS, []),
        )


class RagTestCaseConfig:
    """RAG / LLM test case configuration:

    - prompt
    - expected output
    - categories
    - condition (string expression)
    - constraints (any JSON serializable object)
    - ...

    """

    KEY_KEY = "key"
    KEY_PROMPT = "prompt"
    KEY_CONSTRAINTS = "constraints"
    KEY_CONDITION = "condition"
    KEY_CATEGORIES = "categories"
    KEY_RELS = "relationships"
    KEY_EXPECTED_OUTPUT = "expected_output"

    def __init__(
        self,
        prompt: str,
        categories: str | list[str] = "",
        relationships=None,
        constraints=None,
        condition="",
        expected_output: str = "",
        config: RagTestConfig | None = None,
        key: str = "",
    ):
        self.key = key or str(uuid.uuid4())
        self.prompt = prompt
        self.categories = categories or []
        self.condition = condition or ""
        self.categories = (
            self.categories
            if isinstance(self.categories, list)
            else [str(self.categories)]
        )
        self.relationships = relationships or []
        self.constraints = constraints
        self.expected_output = expected_output
        self.config = config

    def add_relationship(self, relationship_type: str, target: str, target_type: str):
        self.relationships.append(
            datasets.LlmInputRel(
                rel_type=relationship_type,
                target=target,
                target_type=target_type,
            )
        )

    def perturb(
        self,
        perturbators: list[commons.PerturbatorToRun],
        in_place: bool = True,
        raised_errors: list | None = None,
    ):
        """Perturb the prompt.

        Parameters
        ----------
        perturbators : list[commons.PerturbatorToRun]
            Perturbators to run - includes the perturbator ID, intensity,
            and parameters.
        in_place : bool
            If True, perturb the prompt in place, otherwise create a new perturbed
            test case.
        raised_errors : list | None
            If ``None``, then raise error(s) if the perturbator(s) fail(s),
            otherwise do not raise exceptions and store them in the (empty) list
            provided by the caller.

        """
        if not perturbators:
            raise ValueError("No perturbators provided")

        content = self if in_place else self.copy(update_key=True)

        registry = perturbations.PerturbatorRegistry.registry()
        is_copy_perturbation = any(
            p2r.perturbator_id == perturbations.CopyPerturbator.perturbator_id()
            for p2r in perturbators
        )
        cats = [perturbations.CAT_PERTURBED]
        original_prompt = content.prompt
        for p2r in perturbators:
            # perturbations
            perturbed_prompt = registry.get_perturbator(p2r.perturbator_id).perturb(
                text=content.prompt,
                intensity=p2r.intensity,
                raised_errors=raised_errors,
                **p2r.params,
            )
            if perturbed_prompt is None:
                continue

            content.prompt = perturbed_prompt

            # categories
            cats.append(
                f"{perturbations.PREFIX_CAT_PERTURBED_BY}"
                f"{p2r.perturbator_id}:{p2r.intensity}"
            )
            if (
                p2r.params
                and p2r.params.get("prompt_type", "")
                and p2r.params.get("answer_type", "")
                and p2r.params.get("encoding_type", "")
            ):
                cats.append(
                    f"encoding_perturbator:["
                    f"prompt_type:{p2r.params.get('prompt_type', '')},"
                    f"answer_type:{p2r.params.get('answer_type', '')},"
                    f"encoding_type:{p2r.params.get('encoding_type', '')}"
                    f"]"
                )

        if not content.categories:
            content.categories = cats
        else:
            for cat in cats:
                if cat not in content.categories:
                    content.categories.append(cat)

        if (
            original_prompt == content.prompt
            and not is_copy_perturbation
            and raised_errors is not None
        ):
            raised_errors.append(
                f"Perturbation of the prompt '{original_prompt}' failed - perturbators "
                f"{[p2r.perturbator_id for p2r in perturbators]} were not able to "
                f"perturb the prompt."
            )

        # TC can be NOT perturbed if raised_errors is not None & perturbators fail
        return content

    def copy(self, update_key: bool = True):
        return RagTestCaseConfig(
            key=str(uuid.uuid4()) if update_key else self.key,
            prompt=self.prompt,
            categories=self.categories.copy() if self.categories else [],
            relationships=self.relationships.copy() if self.relationships else [],
            constraints=self.constraints,
            condition=self.condition,
            expected_output=self.expected_output,
            config=self.config,
        )

    def to_dict(self):
        return {
            RagTestCaseConfig.KEY_KEY: self.key or str(uuid.uuid4()),
            RagTestCaseConfig.KEY_PROMPT: self.prompt,
            RagTestCaseConfig.KEY_CATEGORIES: self.categories,
            RagTestCaseConfig.KEY_CONDITION: self.condition,
            RagTestCaseConfig.KEY_RELS: [r.to_dict() for r in self.relationships],
            RagTestCaseConfig.KEY_CONSTRAINTS: self.constraints,
            RagTestCaseConfig.KEY_EXPECTED_OUTPUT: self.expected_output,
        }


class RagTestSuiteConfig:
    """RAG / LLM test suite configuration:

    - test suite (RagTestSuiteConfig)
      ... a set of tests

      - tests (RagTestConfig)
        ... corpus with a set of test cases

        - test cases (RagTestCaseConfig)
          ... prompt, expected output, categories, conditions, ...

    """

    KEY_NAME = "name"
    KEY_DESCRIPTION = "description"
    KEY_CATS = "categories"
    KEY_TESTS = "tests"
    KEY_TEST_CASES = "test_cases"

    @property
    def tests(self) -> list[RagTestConfig]:
        return self.test_cfgs.values() if self.test_cfgs else []

    def __init__(
        self,
        test_cases: list[RagTestCaseConfig] | None = None,
        name: str = "TestSuite",
        description: str = "Test suite for RAG / LLM evaluation.",
        categories: list[str] | None = None,
    ):
        self.test_cases = test_cases or []
        # map: test key -> RagTestConfig
        self.test_cfgs = {}

        self.name = name
        self.description = description
        self.categories = categories or []

    def copy(self) -> "RagTestSuiteConfig":
        new_self = RagTestSuiteConfig(
            test_cases=self.test_cases.copy(),
            name=self.name,
            description=self.description,
            categories=self.categories.copy() if self.categories else [],
        )
        new_self.test_cfgs = self.test_cfgs.copy()
        return new_self

    def add_test_case(self, test_case: RagTestCaseConfig):
        self.test_cases.append(test_case)
        if test_case.config:
            if test_case.config.key not in self.test_cfgs:
                self.test_cfgs[test_case.config.key] = test_case.config

    def trim_tests(self, max_tests: int):
        """Trim the test suite to the given number of tests."""
        new_tests = {}
        new_test_cases = []

        for t in self.test_cfgs:
            if len(new_tests) >= max_tests:
                break
            new_tests[t] = self.test_cfgs[t]

        for tc in self.test_cases:
            if tc.config.key in new_tests:
                new_test_cases.append(tc)

        self.test_cfgs = new_tests
        self.test_cases = new_test_cases

    def split(self, max_tests: int) -> list["RagTestSuiteConfig"]:
        """Split the test suite to multiple test suites so that each test suite
        has at most the given number of tests.

        Parameters
        ----------
        max_tests : int
            Maximum number of tests in a test suite.

        Returns
        -------
        list[RagTestSuiteConfig]
            List of new test suites.

        """
        new_test_suites = []

        new_test_suite_number = 0
        new_test_suite = None
        for t in self.test_cfgs:
            if not new_test_suite or len(new_test_suite.test_cfgs) >= max_tests:
                # start a new test suite
                new_test_suite_number += 1

                new_test_suite = RagTestSuiteConfig(
                    name=f"{self.name} - part {new_test_suite_number}",
                    description=self.description,
                )
                new_test_suites.append(new_test_suite)

            new_test_suite.test_cfgs[t] = self.test_cfgs[t]
            # test cases will be copied later

        # copy test cases
        for nts in new_test_suites:
            for tc in self.test_cases:
                if tc.config.key in nts.test_cfgs:
                    nts.test_cases.append(tc)

        return new_test_suites

    def perturb(
        self,
        perturbators: list[commons.PerturbatorToRun],
        in_place: bool = True,
        raised_errors: list | None = None,
    ):
        """Perturb the test suite prompts.

        Parameters
        ----------
        perturbators : list[commons.PerturbatorToRun]
            Perturbators to run - includes the perturbator ID, intensity,
            and parameters.
        in_place : bool
            If True, perturb the test cases in place - there will be the same number
            of tests and test cases within the test suite
            Otherwise keep the original test cases and create new perturbed
            test cases - there will be 2x more test cases in the test suite after
            the perturbation (all intermediary perturbations in case of multiple
            perturbator IDs are discarded).
        raised_errors : list | None
            If ``None``, then raise error(s) if the perturbator(s) fail(s),
            otherwise do not raise exceptions and store them in the (empty) list
            provided by the caller.

        """

        if not perturbators:
            raise ValueError("No perturbators provided")

        # cost/performance/resource protection - exclude expensive perturbators
        c_perturbators = perturbations.PerturbatorRegistry.registry().are_compatible(
            perturbators=perturbators, items=len(self.test_cases)
        )

        if in_place:
            # |test cases| -perturbation-> |test cases|
            for test_case in self.test_cases:
                test_case.perturb(c_perturbators, raised_errors=raised_errors)
            return self
        else:
            # |test cases| -perturbation-> 2*|test cases|
            perturbed_test_cases = []
            for test_case in self.test_cases:
                perturbed_test_case = test_case.copy()
                perturbation_errors = []
                perturbed_test_case.perturb(
                    c_perturbators, raised_errors=perturbation_errors
                )
                if raised_errors is not None and perturbation_errors:
                    raised_errors.extend(perturbation_errors)
                    continue
                perturbed_test_case.add_relationship(
                    relationship_type=datasets.LlmInputRelType.perturbation_source.name,
                    target=test_case.key,
                    target_type=datasets.LlmInputRelTargetType.test_case.name,
                )
                perturbed_test_cases.append(perturbed_test_case)

            new_self = self.copy()
            new_self.test_cases.extend(perturbed_test_cases)
            return new_self

    def to_dict(self) -> dict:
        # cluster test cases by the test ~ corpus
        by_test = {}
        for test_case in self.test_cases:
            if test_case.config.key not in by_test:
                by_test[test_case.config.key] = {
                    RagTestConfig.KEY_KEY: test_case.config.key,
                    RagTestConfig.KEY_DOCUMENTS: test_case.config.documents,
                    RagTestSuiteConfig.KEY_TEST_CASES: [],
                }
            by_test[test_case.config.key][RagTestSuiteConfig.KEY_TEST_CASES].append(
                test_case.to_dict()
            )

        result = {
            RagTestSuiteConfig.KEY_NAME: self.name,
            RagTestSuiteConfig.KEY_DESCRIPTION: self.description,
            RagTestSuiteConfig.KEY_TESTS: list(by_test.values()),
        }
        # categories are NOT serialized if empty (JSon backward compatibility)
        if self.categories:
            result[RagTestSuiteConfig.KEY_CATS] = self.categories
        return result

    def save_as_json(self, file_path: str | pathlib.Path):
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

        return file_path

    @staticmethod
    def load_from_json(file_path: str | pathlib.Path):
        result = RagTestSuiteConfig()

        with open(file_path) as f:
            as_dict = json.load(f)

        result.name = as_dict.get(RagTestSuiteConfig.KEY_NAME, "TestSuite")
        result.description = as_dict.get(
            RagTestSuiteConfig.KEY_DESCRIPTION, "Test suite for RAG / LLM evaluation."
        )
        result.categories = as_dict.get(RagTestSuiteConfig.KEY_CATS, [])

        target_cfgs = {}
        for target_cfg_dict in as_dict.get(RagTestSuiteConfig.KEY_TESTS, []):
            # reuse & deserialize target configuration
            target_cfg_dict_key = target_cfg_dict.get(RagTestConfig.KEY_KEY, "")
            if target_cfg_dict_key not in target_cfgs:
                target_cfgs[target_cfg_dict_key] = RagTestConfig.from_dict(
                    key=target_cfg_dict_key, as_dict=target_cfg_dict
                )

            # test cases
            for tc in target_cfg_dict.get(RagTestSuiteConfig.KEY_TEST_CASES, []):
                relationships_dict = tc.get(RagTestCaseConfig.KEY_RELS, [])
                result.add_test_case(
                    RagTestCaseConfig(
                        key=tc.get(RagTestCaseConfig.KEY_KEY, ""),
                        prompt=tc.get(RagTestCaseConfig.KEY_PROMPT, ""),
                        categories=tc.get(RagTestCaseConfig.KEY_CATEGORIES, []),
                        relationships=[
                            datasets.LlmInputRel.from_dict(r_dict)
                            for r_dict in relationships_dict
                        ],
                        expected_output=tc.get(
                            RagTestCaseConfig.KEY_EXPECTED_OUTPUT, ""
                        ),
                        constraints=tc.get(RagTestCaseConfig.KEY_CONSTRAINTS, []),
                        condition=tc.get(RagTestCaseConfig.KEY_CONDITION, ""),
                        config=target_cfgs[target_cfg_dict_key],
                    )
                )

        return result

    @staticmethod
    def from_llm_dataset(llm_dataset: datasets.LlmDataset) -> "RagTestSuiteConfig":
        """Create RAG test configuration from the LLM dataset."""
        # cluster by test_case config: corpus doc > RagTestConfig
        test_case_configs = {}

        result = RagTestSuiteConfig()
        for i in llm_dataset.inputs:
            key = i.corpus[0] if i.corpus else None
            if key not in test_case_configs:
                test_case_configs[key] = RagTestConfig(
                    key=i.model_key,
                    documents=i.corpus,
                    categories=i.categories,
                )

            result.add_test_case(
                RagTestCaseConfig(
                    key=i.key,
                    prompt=i.i,
                    categories=i.categories,
                    relationships=i.relationships,
                    constraints=i.output_constraints,
                    condition=i.output_condition,
                    expected_output=i.expected_output,
                    config=test_case_configs[key],
                )
            )

        return result


FILE_RESOLVED_LAB = "resolved_lab.json"


class TestLab(abc.ABC):
    """A test target / product test lab."""

    KEY_NAME = "name"
    KEY_DESCRIPTION = "description"
    KEY_RAW_DATASET = "raw_dataset"
    KEY_DATASET = "dataset"
    KEY_MODELS = "models"
    KEY_BASE_MODEL_NAMES = "llm_model_names"
    KEY_DOCS_CACHE = "docs_cache"

    # lab completion: predefined strategies
    SEQUENTIAL_RUN = 0
    PARALLEL_RUN = -1  # automatic choice of the number of workers

    def build(self):
        """Build / deploy / materialize the test lab on the host system e.g. by
        creating RAG's document collections, uploading documents to the collection, ...

        """
        raise NotImplementedError

    def complete_dataset(
        self,
        complete_context: int = 10,
        progress_callback: progress.AbstractProgressCallbackContext | None = None,
        save_as_you_go: pathlib.Path | str | None = None,
        parallelize: int = SEQUENTIAL_RUN,
        retry_on_error: int = DEFAULT_RETRY_ON_ERROR,
        purge_workdir: bool = True,
    ):
        """Complete the LLM dataset with the actual values from the host system.

        Parameters
        ----------
        complete_context : int
            How many context text chunks to include in the resolved dataset.
        progress_callback : progress.AbstractProgressCallbackContext | None
            Optional progress callback context.
        save_as_you_go : pathlib.Path | str | None
            Save the dataset as JSON after each input is resolved.
        parallelize : int
            Complete the dataset in parallel using multiple processes. Use
            ``-1`` for auto-choice of the number of workers, ``0`` to disable
            parallelization (will create the lab using sequential requests),
            and ``1`` + (positive integer) to specify the number of workers.
        retry_on_error : int
            How many times to retry the failed LLM host requests.
        purge_workdir : bool
            Purge the working directory with lab shards after the completion.

        """
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError

    def save_as_json(self, file_path: str | pathlib.Path, as_unicode: bool = True):
        with open(file_path, "w", encoding=None if as_unicode else "utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=as_unicode)

        return file_path


class TestLabPersistence:
    DIR_TEST_LAB = "test_lab_"
    DIR_COMPLETION_OF = "completion_of_"
    DIR_CHAT_SESSION = "chat_session_"
    DIR_CHAT_MSG = "chat_message_"

    @staticmethod
    def get_test_lab_dir(
        user_dir: str | pathlib.Path, test_lab_key: str
    ) -> pathlib.Path:
        return (
            pathlib.Path(user_dir) / f"{TestLabPersistence.DIR_TEST_LAB}{test_lab_key}"
        )

    @staticmethod
    def get_test_case_completion_dir(
        test_lab_dir: str | pathlib.Path, model_key: str, test_case_key: str
    ) -> pathlib.Path:
        return pathlib.Path(test_lab_dir) / (
            f"{TestLabPersistence.DIR_COMPLETION_OF}m_{model_key}_tc_{test_case_key}"
        )

    @staticmethod
    def get_chat_session_dir(
        test_case_completion_dir: str | pathlib.Path, chat_session_key: str
    ) -> pathlib.Path:
        return pathlib.Path(test_case_completion_dir) / (
            f"{TestLabPersistence.DIR_CHAT_SESSION}{chat_session_key}"
        )

    @staticmethod
    def get_chat_message_path(
        chat_session_dir: str | pathlib.Path,
        chat_message_key: str,
    ) -> pathlib.Path:
        return pathlib.Path(chat_session_dir) / (
            f"{TestLabPersistence.DIR_CHAT_MSG}{chat_message_key}"
        )


class LlmHostPromptCache(abc.ABC):
    """Prompt cache for the LLM host clients:

    - caches:

      - answer(s) (actual answer, duration, cost, chunks, ...) for given prompt(s)

    - NOT caches:

      - corpus documents synchronization
      - RAG host server collection creation
      - LLM models listing

    - cache key:

      - does NOT consider particular host (like h2ogpte.h2o.ai),
        but rather the LLM host type ~ connection type (like H2O_GPT_E or OPENAI_RAG)
      - does NOT consider particular chunk retrieval method
      - DOES consider corpus documents (empty for non-RAG), prompt,
        LLM model name, required context chunks (via chunk retrieval method - none or
        a method), ...
      - DOES consider context (empty for RAG)

    - implementations (options):

      - in-memory cache (testing)
      - filesystem cache (pre-build JSON files)
      - Redis cache (shared by EvalStudio workers)
      - memcached cache (shared by EvalStudio workers)
      - ...

    - utilities:

      - cache key generation
      - cache key hashing
      - static cache builder from serialized test labs (JSon)

    - purpose:

      - NON production use - for testing / demos / conference hands-on sessions only
      - significantly speed up the test lab completion
      - avoid test lab build failures due to unstable/slow/fragile system under test
        (like h2oGPTe server)
      - save costs (e.g. OpenAI server costs)

    """

    PREFIX_KEY = "CACHE-KEY::"

    # cache dictionary keys for actual values ~ cache purpose
    KEY_MODEL_TYPE = models.ExplainableLlmModel.KEY_MODEL_TYPE
    KEY_LLM_MODEL_NAME = models.ExplainableLlmModel.KEY_LLM_MODEL_NAME
    KEY_CORPUS = datasets.LlmDataset.KEY_CORPUS
    KEY_INPUT = datasets.LlmDataset.KEY_INPUT
    KEY_EXTRAS = "extras"

    KEY_CONTEXT = datasets.LlmDataset.KEY_CONTEXT
    KEY_ACTUAL_OUTPUT = datasets.LlmDataset.KEY_ACTUAL_OUTPUT
    KEY_DURATION = datasets.LlmDataset.KEY_ACTUAL_DURATION
    KEY_COST = datasets.LlmDataset.KEY_COST

    def __init__(self):
        self.hits = 0
        self.misses = 0

    @staticmethod
    def get_key(
        explainable_model_type: models.ExplainableModelType,
        prompt: str,
        llm_model_name: str,
        corpus: list[str] | None = None,
        extras: str = "",
    ) -> str:
        """Generate cache key for the LLM host client cache:

        - does NOT consider particular host (like h2ogpte.h2o.ai)
        - does NOT consider RAG collection
        - does NOT consider chunk retrieval method
        - suitable for both RAG hosts (empty corpus, no context) and LLM hosts

        Parameters
        ----------
        explainable_model_type : models.ExplainableModelType
            Explainable model type.
        prompt : str
            Prompt for which the answer is to be cached.
        llm_model_name : str
            LLM model name whose answer is to be cached.
        corpus : list[str] | None
            Corpus documents - instead of relying on the collection (ID and name which
            may differ) corpus information is used.
        extras : str
            Extra information - any other parameters which may make the cache key
            unique.

        Returns
        -------
        str
            Cache key.

        """
        # normalization (None vs [] vs "")
        corpus = corpus or []
        extras = extras or ""

        key_dict = {
            # ExplainableModelType
            LlmHostPromptCache.KEY_MODEL_TYPE: explainable_model_type.name,
            LlmHostPromptCache.KEY_LLM_MODEL_NAME: llm_model_name,
            LlmHostPromptCache.KEY_CORPUS: str(corpus),
            LlmHostPromptCache.KEY_INPUT: prompt,
            LlmHostPromptCache.KEY_EXTRAS: extras,
        }
        key_str = json.dumps(key_dict, sort_keys=True)

        return f"{LlmHostPromptCache.PREFIX_KEY}{key_str}"

    @staticmethod
    def str_key_to_dict(key_dict: str) -> dict:
        if key_dict.startswith(LlmHostPromptCache.PREFIX_KEY):
            return json.loads(key_dict[len(LlmHostPromptCache.PREFIX_KEY) :])

        raise ValueError(f"Invalid cache key: {key_dict}")

    @abc.abstractmethod
    def get(self, key: str) -> dict | None:
        """Get the cached value for the given key.

        Returned dictionary might be passed to a result class with types.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def put(self, key: str, value: dict):
        """Put the value to the cache for the given key.

        Parameters
        ----------
        key : str
            Cache key.
        value : dict
            Cache value - it is expected that the dictionary is JSon serialized
            LlmDataset.LlmDatasetRow.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def evict(self, key: str):
        """Evict the value from the cache for the given key."""
        raise NotImplementedError

    @abc.abstractmethod
    def clear(self):
        """Clear the cache."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_llm_model_names(
        self, explainable_model_type: models.ExplainableModelType
    ) -> list[str]:
        """List all the LLM model names known to the cache."""
        raise NotImplementedError


class InMemoryLlmHostPromptCache(LlmHostPromptCache):
    """In-memory LLM host client cache:

    - initialization:
        - (pre-built) cache can be loaded from a JSon file
    - hints:
        - cache can be saved and loaded from a JSon file
        - pre-built cache can be created from a test lab (not implemented by this class)
        - when used in the testing environment, cloud deployment, ... pre-built cache
          can be synchronized/downloaded from S3, filesystem, ...

    """

    KEY_DATA = "cache_data"

    def __init__(self):
        LlmHostPromptCache.__init__(self)

        # map: ["data"]cache key -> cached value: dict
        self.cache = {
            InMemoryLlmHostPromptCache.KEY_DATA: {},
        }

    def __str__(self):
        return (
            f"Prompt cache: "
            f"{len(self.cache.get(InMemoryLlmHostPromptCache.KEY_DATA, {}))} "
            f"items"
        )

    def add_test_lab(self, test_lab: "RagTestLab"):
        """Add the test lab to the cache."""
        for i in test_lab.dataset.inputs:
            i_e_model = test_lab.get_evaluated_model_for_key(i.model_key)
            self.put(
                key=self.get_key(
                    explainable_model_type=i_e_model.model_type,
                    prompt=i.i,
                    llm_model_name=i_e_model.llm_model_name,
                    corpus=i.corpus,
                ),
                value=i.to_dict(),
            )

    def get(self, key: str) -> dict | None:
        if key in self.cache[InMemoryLlmHostPromptCache.KEY_DATA]:
            self.hits += 1
            return self.cache[InMemoryLlmHostPromptCache.KEY_DATA][key]
        self.misses += 1
        return None

    def put(self, key: str, value: dict):
        # verify that value has field/keys required
        if not value.get(LlmHostPromptCache.KEY_ACTUAL_OUTPUT, None):
            raise ValueError(f"Cannot cache the value w/o the actual answer: {value}")
        if not value.get(LlmHostPromptCache.KEY_DURATION, None):
            raise ValueError(f"Cannot cache the value w/o the duration: {value}")

        # fallback
        if not value.get(LlmHostPromptCache.KEY_COST, None):
            value[LlmHostPromptCache.KEY_COST] = 0.0
        if not value.get(LlmHostPromptCache.KEY_CONTEXT, None):
            # LLM evaluations do not have context
            value[LlmHostPromptCache.KEY_CONTEXT] = []

        self.cache[InMemoryLlmHostPromptCache.KEY_DATA][key] = value

    def evict(self, key: str):
        self.cache[InMemoryLlmHostPromptCache.KEY_DATA].pop(key, None)

    def clear(self):
        self.cache[InMemoryLlmHostPromptCache.KEY_DATA] = {}

    def get_llm_model_names(
        self, explainable_model_type: models.ExplainableModelType
    ) -> list[str]:
        llm_model_names = set()
        for k in self.cache[InMemoryLlmHostPromptCache.KEY_DATA].keys():
            model_type = LlmHostPromptCache.str_key_to_dict(k).get(
                LlmHostPromptCache.KEY_MODEL_TYPE, ""
            )
            if model_type == explainable_model_type.name:
                llm_model_name = LlmHostPromptCache.str_key_to_dict(k).get(
                    LlmHostPromptCache.KEY_LLM_MODEL_NAME, ""
                )
                if llm_model_name:
                    llm_model_names.add(llm_model_name)
                else:
                    raise ValueError(
                        f"Invalid cache key '{k}' - unable to extract LLM model name."
                    )

        return list(llm_model_names)

    def to_dict(self) -> dict:
        return self.cache

    def save_to_json(self, file_path: str | pathlib.Path):
        with open(file_path, "w") as f:
            json.dump(self.cache, f, indent=4)

        return file_path

    @staticmethod
    def load_from_json(file_path: str | pathlib.Path):
        result = InMemoryLlmHostPromptCache()

        with open(file_path) as f:
            as_dict = json.load(f)

        result.cache = as_dict

        if InMemoryLlmHostPromptCache.KEY_DATA not in result.cache:
            logger = loggers.SonarPrintLogger()
            logger.warning(f"Invalid cache file: {file_path}")
            result.cache = {InMemoryLlmHostPromptCache.KEY_DATA: {}}

        return result

    @staticmethod
    def load_from_url(url: str, work_dir: str | pathlib.Path = ""):
        work_dir = pathlib.Path(work_dir) if work_dir else pathlib.Path()

        cache_file_name = f"llm_host_cache_download_{uuid.uuid4()}.json"
        cache_file_path = work_dir / cache_file_name

        response = requests.get(
            url,
            verify=h2o_sonar_config.config.http_ssl_cert_verify,
        )
        if response.status_code == 200:
            cache_file_path.write_bytes(response.content)
        else:
            raise ValueError(f"Failed to download the cache serialization from : {url}")

        return InMemoryLlmHostPromptCache.load_from_json(cache_file_path)


class RagTestLabPromptCache:
    """RAG test lab prompt cache (singleton) to be used across H2O Sonar."""

    # enable/disable the prompt cache (default: disabled)
    ENV_VAR_H2O_SONAR_PROMPT_CACHE: str = "H2O_SONAR_PROMPT_CACHE_ENABLED"
    # True to build cache and do NOT extend it in runtime, else False (default: False)
    ENV_VAR_H2O_SONAR_PROMPT_CACHE_STATIC: str = "H2O_SONAR_PROMPT_CACHE_STATIC"
    # path to the JSon file with cache OR path to the directory with the test labs
    # (JSon files) to be used to build the cache
    ENV_VAR_H2O_SONAR_PROMPT_CACHE_SRC: str = "H2O_SONAR_PROMPT_CACHE_SRC"
    ENV_VAR_H2O_SONAR_PROMPT_CACHE_SIZE: str = "H2O_SONAR_PROMPT_CACHE_SIZE"

    # SINGLETON
    __cache = None
    # SINGLETON: secret key to prevent instantiation using constructor
    __singleton_secret_key = object()

    @classmethod
    def cache(cls):
        if not cls.__cache:
            cls.__cache = RagTestLabPromptCache(cls.__singleton_secret_key)
        return cls.__cache

    def _build_cache(
        self,
        src_path: str | pathlib.Path,
        src_host_connection: h2o_sonar_config.ConnectionConfig | None = None,
    ):
        """Load persisted cache or build the prompt cache from the test labs (JSON
        files) in the given directory.

        Parameters
        ----------
        src_path : str | pathlib.Path
            JSon file with the cache or directory with the test labs (JSON files) to be
            used to build the cache.
        src_host_connection : h2o_sonar_config.ConnectionConfig | None
            Connection to the host system to be used to build the cache.

        """
        if not src_path:
            # no path to build the cache from
            return
        src_path = pathlib.Path(src_path)

        if src_path.is_file():
            self.logger.info(f"Loading prompt cache from: {src_path}")
            self.prompts = InMemoryLlmHostPromptCache.load_from_json(src_path)
        elif src_path.is_dir():
            if src_host_connection:
                for f in glob.glob(str(src_path / "*.json")):
                    self.logger.info(f"Loading test lab from: {f}")
                    test_lab = RagTestLab.load_from_json(
                        llm_host_connection=src_host_connection,
                        file_path=f,
                    )
                    self.prompts.add_test_lab(test_lab)
            else:
                self.logger.warning(
                    f"Cannot build prompt cache from the directory w/o "
                    f"the host connection: {src_path}"
                )
        else:
            raise ValueError(
                f"Invalid path to the prompt cache source: {src_path} "
                f"({type(src_path)})"
            )

    MAX_ITEMS = 5000

    def __init__(self, singleton_create_key):
        # singleton: constructor instantiation protection
        assert singleton_create_key == RagTestLabPromptCache.__singleton_secret_key, (
            "Prompt cache must be created using cache() method"
        )
        self.logger = loggers.SonarPrintLogger()
        self.prompts = None
        self.max_items = RagTestLabPromptCache.MAX_ITEMS
        self.reinitialize()

    def reinitialize(
        self,
        enable_cache: bool | None = None,
        src_path: pathlib.Path | str | None = None,
        src_host_connection: h2o_sonar_config.ConnectionConfig | None = None,
        max_items: int | None = None,
    ):
        is_enabled = (
            enable_cache
            if enable_cache is not None
            else os.getenv(
                RagTestLabPromptCache.ENV_VAR_H2O_SONAR_PROMPT_CACHE, ""
            ).lower()
            in commons.ENABLED_STRINGS
        )
        if is_enabled:
            # max items
            if max_items is not None:
                try:
                    self.max_items = int(
                        os.getenv(
                            RagTestLabPromptCache.ENV_VAR_H2O_SONAR_PROMPT_CACHE_SIZE,
                            max_items,
                        )
                    )
                except ValueError:
                    self.logger.warning(f"Invalid cache size: {max_items}")
                    self.max_items = RagTestLabPromptCache.MAX_ITEMS
            else:
                self.max_items = RagTestLabPromptCache.MAX_ITEMS

            # cache
            self.prompts = InMemoryLlmHostPromptCache()

            # build cache
            src_path = src_path or os.getenv(
                RagTestLabPromptCache.ENV_VAR_H2O_SONAR_PROMPT_CACHE_SRC
            )
            if src_path:
                self._build_cache(
                    src_path=src_path,
                    src_host_connection=src_host_connection,
                )
        else:
            self.prompts = None


# test lab singleton
prompt_cache: RagTestLabPromptCache = RagTestLabPromptCache.cache()

# TestLab progress reporting: queue for the routing of progress messages
# from the workers to the main process
_test_lab_progress_queue: multiprocessing.Queue = multiprocessing.Queue()


# TestLab progress reporting: pool executor initializer
def _test_lab_progress_pool_initializer(q: multiprocessing.Queue):
    """Pool executor initializer for the test lab completion which prepares queue
    used for the progress reporting.

    """
    # bind to globally declared queue and set it to given value
    global _test_lab_progress_queue

    _test_lab_progress_queue = q


class RagTestLab(TestLab, TestLabPersistence):
    """RAG test lab:

    - TestLab is expected to test **multiple** LLMs
      either hosted by **one** service (like OpenAI)
      or
      by RAG (Retrieval-augmented generation) product (like h2oGPTe)
      or
      by LLM host product (like h2oGPT).
    - TestLab gets connection configuration to the host system.
    - TestLab can compare / benchmark multiple LLM models from **the same** host system.
    - Resolved test labs can be **merged** to get an aggregated lab -> LLM dataset
      with multiple LLM hosts for the side-by-side evaluation by the ``evaluate``
      module.

    """

    def __init__(
        self,
        llm_host_connection: h2o_sonar_config.ConnectionConfig,
        raw_dataset: datasets.LlmDataset,
        evaluated_models: (
            list[models.ExplainableRagModel | models.ExplainableLlmModel] | None
        ) = None,
        llm_model_names: list[str] | None = None,
        docs_cache_dir: str | pathlib.Path = "",
        results_location: str | pathlib.Path = "",
        name: str = "TestLab",
        description: str = "Test lab for RAG / LLM evaluation.",
        llm_host_prompt_cache: LlmHostPromptCache | None = None,
        use_evaluated_model_collection_id: bool = False,
        user_name: str = commons.DEFAULT_USER,
        logger=None,
    ):
        """Test lab constructor.

        Parameters
        ----------
        llm_host_connection :
            Connection to a service (OpenAI) or a RAG product (like h2oGPTe)
            which hosts LLMs to be evaluated.
        raw_dataset :
            Raw (non-resolved) dataset with well known data/columns names to be used
            for LLMs evaluation.
        evaluated_models :
            Descriptors of models to be evaluated - either RAG hosted LLM models
            (host, corpus, LLM model name)
            or pure LLM models (host, LLM model name).
        llm_model_names :
            Names of hosted (base) LLM models to be evaluated.
        docs_cache_dir :
            Directory to cache the documents from the network to the local filesystem
            (RAG hosted LLM models evaluation) and serve as work directory for the
            parallel completion of the test lab.
        results_location : str | pathlib.Path
            Base directory path where user-specific subdirectories will be created to
            store the completion artifacts - like files or exported metadata - which
            should be subsequently used for the evaluation. Consider for example
            agentic run and metadata, PDF and Python files created by the agent.
            If not specified, test lab will not store any evaluation artifacts on
            completion.
        name :
            Test lab name.
        description :
            Test lab description.
        llm_host_prompt_cache :
            Cache for the LLM host clients (like h2oGPTe server) to speed up the
            completion of the test lab.
        use_evaluated_model_collection_id : bool
            Force use of the collection ID from the evaluated model when the test lab
            for RAG will be built - do not create a new collections. ``build()`` method
            to just check if the collection IDs are set in evaluated models and
            whether they exist.
            Use ``from_rag_test_suite()`` to create the test lab with predefined
            collection IDs easily.
        user_name : str
            Username to be used to store the evaluation artifacts in the user
            directory.
        logger :
            Optional logger.


        """
        if not llm_host_connection:
            raise ValueError("LLM host connection is required.")
        supported_connection_types = [
            h2o_sonar_config.ConnectionConfigType.AMAZON_BEDROCK.name,
            h2o_sonar_config.ConnectionConfigType.ANTHROPIC_CHAT.name,
            h2o_sonar_config.ConnectionConfigType.AZURE_OPENAI_CHAT.name,
            h2o_sonar_config.ConnectionConfigType.H2O_GPT.name,
            h2o_sonar_config.ConnectionConfigType.H2O_GPT_E.name,
            h2o_sonar_config.ConnectionConfigType.H2O_LLM_OPS.name,
            h2o_sonar_config.ConnectionConfigType.OLLAMA.name,
            h2o_sonar_config.ConnectionConfigType.OPENAI_CHAT.name,
            h2o_sonar_config.ConnectionConfigType.OPENAI_RAG.name,
        ]
        if llm_host_connection.connection_type not in supported_connection_types:
            raise ValueError(
                f"Unsupported LLM host connection type: '"
                f"{llm_host_connection.connection_type}' "
                f"(supported: {supported_connection_types})."
            )

        self.name = name
        self.description = description
        self.use_evaluated_model_collection_id = use_evaluated_model_collection_id
        self.logger = logger or loggers.SonarPrintLogger()
        self.connection = llm_host_connection
        self.agent_client = None
        if (
            self.connection.connection_type
            == h2o_sonar_config.ConnectionConfigType.H2O_GPT_E.name
        ):
            self.rag_client = genai.H2oGpteRagClient(
                connection=self.connection, logger=self.logger
            )
            self.agent_client = agents.H2oGpteAgentHost(
                agent_connection=self.connection,
                logger=self.logger,
                log_name="agentic artifacts extractor",
            )
        elif (
            self.connection.connection_type
            == h2o_sonar_config.ConnectionConfigType.OPENAI_CHAT.name
        ):
            self.rag_client = genai.OpenAiLlmClient(
                connection=self.connection, logger=self.logger
            )
        elif (
            self.connection.connection_type
            == h2o_sonar_config.ConnectionConfigType.OPENAI_RAG.name
        ):
            self.rag_client = genai.OpenAiAssistantsRagClient(
                connection=self.connection, logger=self.logger
            )
        elif (
            self.connection.connection_type
            == h2o_sonar_config.ConnectionConfigType.AZURE_OPENAI_CHAT.name
        ):
            self.rag_client = genai.MsAzureOpenAiLlmClient(
                connection=self.connection, logger=self.logger
            )
        elif (
            self.connection.connection_type
            == h2o_sonar_config.ConnectionConfigType.H2O_GPT.name
        ):
            self.rag_client = genai.H2oGptLlmClient(
                connection=self.connection, logger=self.logger
            )
        elif (
            self.connection.connection_type
            == h2o_sonar_config.ConnectionConfigType.H2O_LLM_OPS.name
        ):
            self.rag_client = genai.H2oLlmOpsClient(
                connection=self.connection, logger=self.logger
            )
        elif (
            self.connection.connection_type
            == h2o_sonar_config.ConnectionConfigType.OLLAMA.name
        ):
            self.rag_client = genai.OllamaClient(
                connection=self.connection, logger=self.logger
            )
        elif (
            self.connection.connection_type
            == h2o_sonar_config.ConnectionConfigType.ANTHROPIC_CHAT.name
        ):
            self.rag_client = genai.AnthropicClaudeLlmClient(
                connection=self.connection, logger=self.logger
            )
        elif (
            self.connection.connection_type
            == h2o_sonar_config.ConnectionConfigType.AMAZON_BEDROCK.name
        ):
            self.rag_client = genai.AmazonBedrockRagClient(
                connection=self.connection, logger=self.logger
            )
        else:
            raise ValueError("Unsupported RAG connection")

        self.raw_dataset = raw_dataset
        self.llm_model_names = llm_model_names or []

        # map: model name -> model
        self.evaluated_models = (
            {m.name: m for m in evaluated_models} if evaluated_models else {}
        )

        # resolved dataset (configure -> model keys, build -> actual values)
        self.dataset = datasets.LlmDataset()

        # directory to store evaluation artifacts
        user_name = user_name or commons.DEFAULT_USER
        self.user_dir = (
            pathlib.Path(results_location) / user_name if results_location else None
        )

        # directory to cache the documents
        self.docs_cache_dir = (
            pathlib.Path(docs_cache_dir)
            if docs_cache_dir
            else (
                self.user_dir / "cache" / "documents"
                if self.user_dir
                else pathlib.Path()
            )
        )
        if not self.docs_cache_dir.exists():
            self.docs_cache_dir.mkdir(parents=True, exist_ok=True)
        # document cache map:
        #   str(doc URL/path from target config) -> locally cached doc path in ^
        self.doc_cache: dict = {}
        for m in self.evaluated_models.values():
            if isinstance(m, models.ExplainableRagModel):
                for doc in m.documents:
                    self.doc_cache[str(doc)] = (
                        self.docs_cache_dir / pathlib.Path(doc).name
                    )
        self._dummy_doc_path = None

        # use custom prompt cache OR the global prompt cache (if enabled)
        self.llm_host_prompt_cache = llm_host_prompt_cache or prompt_cache.prompts

        # statistics
        self.stat_enable = False
        self.stat_host_calls = 0  # ok + errors
        self.stat_host_errors = 0  # errors

        # map: LLM model name -> int
        self.stat_llm_model_success_count: dict[str, int] = {}  # successful calls
        # map: LLM model name -> dataset::input
        self.stat_llm_model_errors: dict[
            str, list[datasets.LlmDataset.LlmDatasetRow]
        ] = {}  # failed calls excluding timeouts
        # map: LLM model name -> dataset::input
        self.stat_llm_model_retries: dict[
            str, list[datasets.LlmDataset.LlmDatasetRow]
        ] = {}  # retries - subset of failed calls
        # map: LLM model name -> dataset::input
        self.stat_llm_model_timeouts: dict[
            str, list[datasets.LlmDataset.LlmDatasetRow]
        ] = {}  # timeouts
        self.stat_llm_model_request_duration: dict[str, dict[str, float]] = {}

    _DUMMY_DOC_NAME = "h2o-eval-studio-empty-corpus-dummy-document.txt"
    _DUMMY_DOC_LOCATOR = f"http://dummy.doc/{_DUMMY_DOC_NAME}"

    def _clear_stats(self):
        self.stat_host_calls = 0
        self.stat_host_errors = 0
        self.stat_llm_model_success_count = {}
        self.stat_llm_model_errors = {}
        self.stat_llm_model_retries = {}
        self.stat_llm_model_timeouts = {}
        self.stat_llm_model_request_duration = {}

    def _add_stat_duration(self, llm_model_name: str, duration: float):
        if llm_model_name not in self.stat_llm_model_request_duration:
            self.stat_llm_model_request_duration[llm_model_name] = {
                "n": 0,
                "min": 0.0,
                "max": 0.0,
                "sum": 0.0,
            }
        self.stat_llm_model_request_duration[llm_model_name]["n"] += 1
        self.stat_llm_model_request_duration[llm_model_name]["sum"] += duration
        self.stat_llm_model_request_duration[llm_model_name]["min"] = (
            min(
                [
                    self.stat_llm_model_request_duration[llm_model_name]["min"],
                    duration,
                ]
            )
            if self.stat_llm_model_request_duration[llm_model_name]["n"] > 1
            else duration
        )
        self.stat_llm_model_request_duration[llm_model_name]["max"] = (
            max(
                [
                    self.stat_llm_model_request_duration[llm_model_name]["max"],
                    duration,
                ]
            )
            if self.stat_llm_model_request_duration[llm_model_name]["n"] > 1
            else duration
        )

    def _add_stat_llm_success(self, llm_model_name: str):
        self.stat_host_calls += 1
        if llm_model_name not in self.stat_llm_model_success_count:
            self.stat_llm_model_success_count[llm_model_name] = 0
        self.stat_llm_model_success_count[llm_model_name] += 1

    def _add_stat_llm_err(
        self, llm_model_name: str, i: datasets.LlmDataset.LlmDatasetRow
    ):
        self.stat_host_errors += 1
        if llm_model_name not in self.stat_llm_model_errors:
            self.stat_llm_model_errors[llm_model_name] = []
        self.stat_llm_model_errors[llm_model_name].append(i)

    def _add_stat_llm_timeout(
        self, llm_model_name: str, i: datasets.LlmDataset.LlmDatasetRow
    ):
        self.stat_host_errors += 1
        if llm_model_name not in self.stat_llm_model_timeouts:
            self.stat_llm_model_timeouts[llm_model_name] = []
        self.stat_llm_model_timeouts[llm_model_name].append(i)

    def _add_stat_llm_retry(
        self, llm_model_name: str, i: datasets.LlmDataset.LlmDatasetRow
    ):
        if llm_model_name not in self.stat_llm_model_retries:
            self.stat_llm_model_retries[llm_model_name] = []
        self.stat_llm_model_retries[llm_model_name].append(i)

    def get_evaluated_model_for_key(self, model_key: str):
        """Get LLM model name for the evaluated model."""
        # map: evaluated model key -> evaluated model
        models_map = {m.key: m for m in self.evaluated_models.values()}  # non-cached

        if model_key not in models_map:
            raise ValueError(
                f"Evaluated model with key {model_key} is not available in the test "
                f"lab."
            )

        return models_map[model_key]

    def sync_documents(
        self,
        doc_sync_meta: dict[str, Any] = None,
        progress_callback: progress.AbstractProgressCallbackContext | None = None,
        fail_on_missing_corpus: bool = False,
    ) -> pathlib.Path:
        """Cache test suite documents from the network to the local filesystem
        so that they can be used for RAG evaluation later.

        Parameters
        ----------
        doc_sync_meta: dict[str, Any]
            Document synchronization metadata - the key is the document locator (URL),
            the value is a dictionary with metadata like ``headers``. Example::

                "http://example.com/doc1.txt": {
                    "headers": {
                       "foo-header": "FOO-VALUE",
                    }
                }

        progress_callback : progress.AbstractProgressCallbackContext | None
            Optional progress callback context.
        fail_on_missing_corpus : bool
            Fail if a test has **empty corpus**, or create dummy document
            to enable empty RAG corpora

        """
        doc_sync_meta = doc_sync_meta or {}
        # progress
        progress_callback = progress.LoggingProgressCallbackContext(
            logger=self.logger,
            prefix="TestLab documents synchronization progress",
            parent_callback=progress_callback,
            verbose_children=(
                progress_callback.verbose_children if progress_callback else True
            ),
            name="TestLab doc sync callback",
        )
        progress_callback.set_progress(0.0, "Started")

        if not fail_on_missing_corpus:
            progress_callback.set_progress(
                0.01, "Creating dummy document for empty corpora..."
            )
            # create DUMMY doc that can be later uploaded to the collection w/o corpus
            try:
                dummy_doc_path = self.docs_cache_dir / RagTestLab._DUMMY_DOC_NAME
                with open(dummy_doc_path, "w") as f:
                    f.write("EMPTY " * 100)
                self.doc_cache[str(dummy_doc_path)] = (
                    self.docs_cache_dir / RagTestLab._DUMMY_DOC_NAME
                )
                self._dummy_doc_path = dummy_doc_path
            except Exception as ex:
                self.logger.error(
                    f"Failed to create dummy document {self._dummy_doc_path} "
                    f"for fallback of in case of the empty corpus: {ex}\n"
                    f"{traceback.format_exc()}"
                )
            progress_callback.set_progress(0.02, "Empty corpora dummy document created")

        resolved_doc_cache = {}
        # progress
        (
            progress_slot_min,
            progress_slot_size,
            steps,
        ) = progress.ProgressCallbackContext.step_loop_prepare(
            progress_min=progress_callback.progress + 0.01,
            progress_max=0.97,
            steps=len(self.doc_cache),
        )
        for e, doc_locator in enumerate(self.doc_cache):
            # progress
            (
                step_slot_min,
                step_slot_max,
            ) = progress.ProgressCallbackContext.step_loop_get_min_and_max(
                step=e,
                progress_slot_min=progress_slot_min,
                progress_slot_size=progress_slot_size,
            )
            progress_callback.set_progress(
                progress=step_slot_min,
                message=f"#{e + 1}/{steps} Caching document: {doc_locator} ...",
            )

            cache_file_path = pathlib.Path(self.doc_cache[doc_locator])
            self.logger.info(f"  -> cache path: {cache_file_path}")
            if cache_file_path.exists():
                resolved_doc_cache[str(doc_locator)] = pathlib.Path(cache_file_path)
                progress_callback.set_progress(
                    progress=step_slot_max,
                    message=(
                        f"#{e + 1}/{len(self.doc_cache)} "
                        f"Document already cached: {doc_locator} ..."
                    ),
                )
                continue

            if isinstance(doc_locator, str) and doc_locator.startswith("http"):
                http_headers = doc_sync_meta.get(doc_locator, {}).get("headers", None)
                response = requests.get(
                    doc_locator,
                    headers=http_headers,
                    verify=h2o_sonar_config.config.http_ssl_cert_verify,
                )
                if response.status_code == 200:
                    cache_file_path.write_bytes(response.content)
                    resolved_doc_cache[doc_locator] = pathlib.Path(cache_file_path)
                    self.logger.info(
                        f"Successfully downloaded document to cache: {cache_file_path}"
                    )
                else:
                    self.logger.error(
                        f"Failed to download test data document: {doc_locator}"
                    )

            elif isinstance(doc_locator, (str, pathlib.Path)):
                doc_locator = pathlib.Path(doc_locator)
                if not doc_locator.exists():
                    raise ValueError(
                        f"Test data document {doc_locator} does not exist on the local "
                        f"filesystem."
                    )
                else:
                    # update cache to point to the source document (avoid duplication)
                    resolved_doc_cache[str(doc_locator)] = pathlib.Path(doc_locator)
                    self.logger.info(
                        f"Successfully updated document cache path: "
                        f"{resolved_doc_cache[str(doc_locator)]}"
                    )
            else:
                raise ValueError(
                    f"Unsupported test data document locator: {doc_locator} (type: "
                    f"{type(doc_locator)})"
                )

            progress_callback.set_progress(
                progress=step_slot_max,
                message=f"#{e + 1}/{steps} Document cached: {doc_locator} ...",
            )

        self.doc_cache = resolved_doc_cache

        progress_callback.set_progress(
            1.1, "Document synchronization to the test lab cache DONE"
        )

        return self.docs_cache_dir

    def stats(self) -> dict:
        """Get the test lab statistics and cross-check."""
        return {
            "name": self.name,
            "description": self.description,
            "llm_host_connection": self.connection.to_dict(encryption_key="H1D31T"),
            "llm_model_names": self.llm_model_names,
            "raw_dataset": self.raw_dataset.stats(),
            "dataset": self.dataset.stats(),
            "host_calls": self.stat_host_calls,
            "host_errors": self.stat_host_errors,
            "llm_model_success": self.stat_llm_model_success_count,
            "llm_model_errors": self.stat_llm_model_errors,
        }

    def integrity_check(self):
        # map: model key -> model
        evaluated_models_by_key = (
            {m.key: m for m in self.evaluated_models.values()}
            if self.evaluated_models
            else {}
        )

        for i in self.dataset.inputs:
            if i.model_key not in evaluated_models_by_key:
                raise ValueError(
                    f"Model {i.model_key} is not available in the test lab for "
                    f"row {i.to_dict()}."
                )

    def bind(
        self, collection_id: str, collection_name: str, corpus: list | None = None
    ):
        """Bind ALL the test lab RAG models to collection(s) instead of building it by
        creating collections and uploading documents.

        """
        for m in self.evaluated_models.values():
            if not m.collection_id:
                m.collection_id = collection_id
            if collection_name:
                m.collection_name = collection_name
            if not m.documents and corpus:
                m.documents = corpus

    def _build_h2ogpte(
        self,
        progress_callback: progress.AbstractProgressCallbackContext | None = None,
        fail_on_missing_corpus: bool = True,
    ):
        # progress
        (
            progress_slot_min,
            progress_slot_size,
            steps,
        ) = progress.ProgressCallbackContext.step_loop_prepare(
            progress_min=progress_callback.progress + 0.01,
            progress_max=1.0 - (progress_callback.progress + 0.01),
            steps=len(self.evaluated_models.values()),
        )
        for e, evaluated_model in enumerate(self.evaluated_models.values()):
            # progress
            (
                step_slot_min,
                step_slot_max,
            ) = progress.ProgressCallbackContext.step_loop_get_min_and_max(
                step=e,
                progress_slot_min=progress_slot_min,
                progress_slot_size=progress_slot_size,
            )
            progress_callback.set_progress(
                progress=step_slot_min,
                message=(
                    f"#{e + 1}/{steps} "
                    f"Preparing h2oGPTe for the evaluation of the model "
                    f"{evaluated_model.name}"
                ),
            )

            if isinstance(evaluated_model, models.ExplainableRagModel):
                doc_cache_paths = [self.doc_cache[d] for d in evaluated_model.documents]

                # FALLBACK on empty corpus: create DUMMY document and upload it
                if (
                    not doc_cache_paths
                    and not fail_on_missing_corpus
                    and self._dummy_doc_path
                ):
                    doc_cache_paths = [self._dummy_doc_path]

                if doc_cache_paths or (not doc_cache_paths and fail_on_missing_corpus):
                    self.logger.info(
                        f"{e + 1}/{steps} Creating h2oGPTe collection for model "
                        f"{evaluated_model.name} with {len(doc_cache_paths)} documents:"
                        f"\n  -> collection name: {evaluated_model.collection_name}"
                        f"\n  -> documents: {doc_cache_paths}"
                    )
                    (
                        evaluated_model.collection_id,
                        _,
                    ) = self.rag_client.create_collection(
                        doc_paths=doc_cache_paths,
                        collection_name=evaluated_model.collection_name,
                        upload_if_collection_exists=False,
                        model_cfg=evaluated_model.model_cfg,
                    )
                else:
                    msg = (
                        f"the creation of the h2oGPTe collection FAILED for "
                        f"the model {evaluated_model.name} as it does NOT specify "
                        f"any documents for the fine tuning - no corpus."
                    )
                    if fail_on_missing_corpus:
                        self.logger.error(f"Failing {msg}")
                        raise ValueError(msg)
                    else:
                        self.logger.warning(f"Skipping{msg}")

            progress_callback.set_progress(
                progress=step_slot_max,
                message=(
                    f"#{e + 1}/{steps} "
                    f"Prepared h2oGPTe for the evaluation of the model "
                    f"{evaluated_model.name} ."
                ),
            )

    def _build_openai(
        self,
        progress_callback: progress.AbstractProgressCallbackContext | None = None,
        fail_on_missing_corpus: bool = True,
    ):
        models_count = len(self.evaluated_models.values())

        # progress
        (
            progress_slot_min,
            progress_slot_size,
            steps,
        ) = progress.ProgressCallbackContext.step_loop_prepare(
            progress_min=progress_callback.progress + 0.01,
            progress_max=1.0 - (progress_callback.progress + 0.01),
            steps=models_count,
        )

        for e, evaluated_model in enumerate(self.evaluated_models.values()):
            doc_cache_paths = [self.doc_cache[d] for d in evaluated_model.documents]

            # progress
            (
                step_slot_min,
                step_slot_max,
            ) = progress.ProgressCallbackContext.step_loop_get_min_and_max(
                step=e,
                progress_slot_min=progress_slot_min,
                progress_slot_size=progress_slot_size,
            )
            progress_callback.set_progress(
                progress=step_slot_min,
                message=(
                    f"{e + 1}/{models_count} Creating OpenAI Assistant for the model "
                    f"{evaluated_model.name} with {len(doc_cache_paths)} documents:"
                    f"\n  -> documents: {doc_cache_paths}"
                ),
            )

            # FALLBACK on empty corpus: create DUMMY document and upload it
            if (
                not doc_cache_paths
                and not fail_on_missing_corpus
                and self._dummy_doc_path
            ):
                doc_cache_paths = [self._dummy_doc_path]

            if doc_cache_paths or (not doc_cache_paths and fail_on_missing_corpus):
                evaluated_model.collection_id = self.rag_client.create_collection(
                    doc_paths=doc_cache_paths,
                    llm_model_name=evaluated_model.llm_model_name,
                    collection_name=evaluated_model.collection_name,
                    **evaluated_model.model_cfg,
                )
            else:
                msg = (
                    f"the creation of the OpenAI collection FAILED for "
                    f"the model {evaluated_model.name} as it does NOT specify "
                    f"any documents for the fine tuning - no corpus."
                )
                if fail_on_missing_corpus:
                    self.logger.error(f"Failing {msg}")
                    raise ValueError(msg)
                else:
                    self.logger.warning(f"Skipping{msg}")

            progress_callback.set_progress(
                progress=step_slot_max,
                message=(
                    f"#{e + 1}/{steps} "
                    f"Prepared OpenAI host for the evaluation of the model "
                    f"{evaluated_model.name}"
                ),
            )

    def _build_bedrock(
        self,
        progress_callback: progress.AbstractProgressCallbackContext | None = None,
        fail_on_missing_corpus: bool = True,
    ):
        models_count = len(self.evaluated_models.values())

        # progress
        (
            progress_slot_min,
            progress_slot_size,
            steps,
        ) = progress.ProgressCallbackContext.step_loop_prepare(
            progress_min=progress_callback.progress + 0.01,
            progress_max=1.0 - (progress_callback.progress + 0.01),
            steps=models_count,
        )

        for e, evaluated_model in enumerate(self.evaluated_models.values()):
            doc_cache_paths = [self.doc_cache[d] for d in evaluated_model.documents]

            # progress
            (
                step_slot_min,
                step_slot_max,
            ) = progress.ProgressCallbackContext.step_loop_get_min_and_max(
                step=e,
                progress_slot_min=progress_slot_min,
                progress_slot_size=progress_slot_size,
            )
            progress_callback.set_progress(
                progress=step_slot_min,
                message=(
                    f"{e + 1}/{models_count} Creating Bedrock Assistant for the model "
                    f"'{evaluated_model.name}' with {len(doc_cache_paths)} documents:"
                    f"\n  -> documents: {[str(d) for d in doc_cache_paths]}"
                ),
            )

            # FALLBACK on empty corpus: create DUMMY document and upload it
            if (
                not doc_cache_paths
                and not fail_on_missing_corpus
                and self._dummy_doc_path
            ):
                doc_cache_paths = [self._dummy_doc_path]

            if doc_cache_paths or (not doc_cache_paths and fail_on_missing_corpus):
                (_, evaluated_model.collection_id) = self.rag_client.create_collection(
                    doc_paths=doc_cache_paths,
                    llm_model_name=evaluated_model.llm_model_name,
                    collection_name=evaluated_model.collection_name,
                    **evaluated_model.model_cfg,
                )
            else:
                msg = (
                    f"the creation of the Bedrock collection FAILED for "
                    f"the model {evaluated_model.name} as it does NOT specify "
                    f"any documents for the fine tuning - no corpus."
                )
                if fail_on_missing_corpus:
                    self.logger.error(f"Failing {msg}")
                    raise ValueError(msg)
                else:
                    self.logger.warning(f"Skipping{msg}")

            progress_callback.set_progress(
                progress=step_slot_max,
                message=(
                    f"#{e + 1}/{steps} "
                    f"Prepared Bedrock host for the evaluation of the model "
                    f"{evaluated_model.name}"
                ),
            )

    def build(
        self,
        doc_sync_meta: dict[str, Any] = None,
        progress_callback: progress.AbstractProgressCallbackContext | None = None,
        sync_documents: bool = True,
        fail_on_missing_corpus: bool = False,
    ):
        """Build the test lab so that it can be used for the evaluation:

        - synchronize the document cache
        - create RAG's document collections
        - upload documents (corpora) to collection(s).

        Parameters
        ----------
        doc_sync_meta: dict[str, Any]
            Document synchronization metadata - the key is the document locator (URL),
            the value is a dictionary with metadata like ``headers``. Example::

                "http://example.com/doc1.txt": {
                    "headers": {
                       "foo-header": "FOO-VALUE",
                    }
                }

        progress_callback : progress.AbstractProgressCallbackContext | None
            Optional progress callback context.
        sync_documents : bool
            Sync documents from the network/filesystem to the lab's document cache.
        fail_on_missing_corpus : bool
            Fail if the test does not specify any corpus.

        """
        # if lab should use predefined collection IDs, then check if they are available
        if self.use_evaluated_model_collection_id:
            self.logger.info(
                "Test lab is configured to use the predefined collection IDs "
                "from evaluated models..."
            )

            existing_collection_ids = [
                c.id for c in self.rag_client.list_collections(offset=0, limit=1000)
            ]

            for em in self.evaluated_models.values():
                self.logger.info(
                    f"  Checking PREDEFINED collection for the model '{em.name}'"
                    f"/{em.key} with collection ID '{em.collection_id}' ..."
                )
                # check collection ID is set
                if not em.collection_id:
                    raise ValueError(
                        f"Test lab is configured to use the predefined collection IDs "
                        f"from evaluated models, but the model '{em.name}'/{em.key} "
                        f"does not have collection ID set."
                    )
                # check collection exists
                if em.collection_id not in existing_collection_ids:
                    raise ValueError(
                        f"Test lab is configured to use the predefined collection IDs "
                        f"from evaluated models, but the model '{em.name}'/{em.key} "
                        f"collection ID '{em.collection_id}' does not exist on host "
                        f": {self.connection.connection_type}."
                    )

                self.logger.info("    -> OK")

            # RAG host ready
            return

        if sync_documents:
            # progress
            progress_callback = progress.LoggingProgressCallbackContext(
                logger=self.logger,
                prefix="TestLab build progress",
                parent_callback=progress_callback,
                verbose_children=(
                    progress_callback.verbose_children if progress_callback else True
                ),
                name="TestLab BUILD callback",
            )
            progress_callback.set_progress(0.0, "Started")
            doc_sync_callback = progress_callback.get_sub_callback_for_progress(
                min_progress=0.0,
                max_progress=0.3,
                verbose_children=progress_callback.verbose_children,
            )

            # documents synchronization
            self.sync_documents(
                doc_sync_meta=doc_sync_meta,
                fail_on_missing_corpus=fail_on_missing_corpus,
                progress_callback=doc_sync_callback,
            )

            lab_build_callback = progress_callback.get_sub_callback_for_progress(
                min_progress=0.31,
                max_progress=1.0,
                verbose_children=progress_callback.verbose_children,
            )
        else:
            lab_build_callback = progress.LoggingProgressCallbackContext(
                logger=self.logger,
                prefix="TestLab build progress",
                parent_callback=progress_callback,
                verbose_children=(
                    progress_callback.verbose_children if progress_callback else True
                ),
                name="TestLab BUILD callback",
            )

        lab_build_callback.set_progress(0.0, "Started")

        if (
            self.connection.connection_type
            == h2o_sonar_config.ConnectionConfigType.H2O_GPT_E.name
        ):
            self._build_h2ogpte(
                progress_callback=lab_build_callback,
                fail_on_missing_corpus=fail_on_missing_corpus,
            )
        elif (
            self.connection.connection_type
            == h2o_sonar_config.ConnectionConfigType.OPENAI_RAG.name
        ):
            self._build_openai(
                progress_callback=lab_build_callback,
                fail_on_missing_corpus=fail_on_missing_corpus,
            )
        elif (
            self.connection.connection_type
            == h2o_sonar_config.ConnectionConfigType.AMAZON_BEDROCK.name
        ):
            self._build_bedrock(
                progress_callback=lab_build_callback,
                fail_on_missing_corpus=fail_on_missing_corpus,
            )
        elif self.connection.connection_type in [
            h2o_sonar_config.ConnectionConfigType.ANTHROPIC_CHAT.name,
            h2o_sonar_config.ConnectionConfigType.AZURE_OPENAI_CHAT.name,
            h2o_sonar_config.ConnectionConfigType.H2O_GPT.name,
            h2o_sonar_config.ConnectionConfigType.H2O_LLM_OPS.name,
            h2o_sonar_config.ConnectionConfigType.OLLAMA.name,
            h2o_sonar_config.ConnectionConfigType.OPENAI_CHAT.name,
        ]:
            # no need to build anything for LLM host
            progress_callback.set_progress(
                1.0, "Nothing to build for an LLM host - DONE"
            )
            return
        else:
            raise ValueError("Unsupported RAG connection")

        progress_callback.set_progress(1.0, "TestLab built - DONE")

    @staticmethod
    def _test_lab_key_for_artifacts_dir(path: pathlib.Path | None) -> str:
        if path and isinstance(path, pathlib.Path):
            last_segment = path.name
            pattern = (
                r"^test_lab_"
                r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$"
            )
            match = re.search(pattern, last_segment)
            if match:
                return match.group(1)

        return ""

    @staticmethod
    def _cat_agentic_artifacts(
        host_type: str,
        chat_session_id: str,
        chat_message_id: str,
        chat_message_seq: int = 0,
    ) -> str:
        if chat_session_id and chat_message_id:
            chat_msg_seq_key = agents.H2oGpteAgentHost.chat_msg_sequence_key(
                chat_message_id=chat_message_id, chat_message_seq=chat_message_seq
            )
            return (
                f"{agents.AgentHost.CAT_AGENT_ARTIFACTS}"
                f":{agents.AgentHost.CAT_AGENT_HOST}:{host_type}"
                f":{agents.AgentHost.CAT_AGENT_SESSION}:{chat_session_id}"
                f":{agents.AgentHost.CAT_AGENT_MSG}:{chat_msg_seq_key}"
            )
        return ""

    def _complete_dataset_h2ogpte(
        self,
        model: models.ExplainableRagModel,
        i,
        complete_context: int = 10,
        test_to_chat: dict | None = None,
        seq: int = 0,
        retry_on_error: int = DEFAULT_RETRY_ON_ERROR,
        timeout_exp_backoff: genai.TimeoutRetryExpBackoffCtx | None = None,
        artifacts_base_dir: pathlib.Path | None = None,
        progress_callback: progress.AbstractProgressCallbackContext | None = None,
    ):
        chunks = []
        actual_output = ""
        duration = 0.0
        cost = 0.0

        if isinstance(model, models.ExplainableRagModel):
            if not model.collection_id:
                raise ValueError(
                    f"RAG model {model.name} does not have required "
                    f"collection ID - unknown collection to chat with."
                )
            test_lab_key = RagTestLab._test_lab_key_for_artifacts_dir(
                artifacts_base_dir
            )
            sync_agent_artifacts = (
                artifacts_base_dir
                and test_lab_key
                and model.model_cfg
                and genai.H2oGpteRagClient.CFG_USE_AGENT in str(model.model_cfg)
                and isinstance(model.model_cfg, dict)
                and model.model_cfg.get(genai.H2oGpteRagClient.CFG_LLM_ARGS, {}).get(
                    genai.H2oGpteRagClient.CFG_USE_AGENT, False
                )
            )
            cats = i.categories or []
            for a in range(retry_on_error + 1):
                if a > 0:
                    self._add_stat_llm_retry(model.llm_model_name, i)
                try:
                    chunk_ret_methods = genai.RagChunkRetrievalMethod
                    [
                        (
                            _,
                            actual_output,
                            duration,
                            chunks,
                            cost,
                            chat_session_id,
                            chat_message_id,
                        )
                    ] = self.rag_client.ask_collection(
                        collection_id=model.collection_id,
                        prompts=[i.i],
                        llm_model_name=model.llm_model_name,
                        include_chunks=complete_context,
                        chunk_retrieval_method=chunk_ret_methods.ANSWER_REFS.name,
                        chat_session_id=(
                            test_to_chat.get(i.test_key, None)
                            if i.test_key and isinstance(test_to_chat, dict)
                            else None
                        ),
                        retry_attempt=a,
                        retry_attempts=retry_on_error,
                        timeout_exp_backoff=timeout_exp_backoff,
                        **model.model_cfg,
                    )
                    if (
                        isinstance(test_to_chat, dict)
                        and i.test_key
                        and chat_session_id
                    ):
                        test_to_chat[i.test_key] = chat_session_id

                    if cost < 0.0:
                        self.logger.error(
                            "Negative cost for server: "
                            f"{self.rag_client.connection.server_id}, "
                            f"RAG model: {model.llm_model_name} and prompt:' {i.i}'"
                        )
                        cost = 0.0
                    self._add_stat_llm_success(model.llm_model_name)
                    self._add_stat_duration(model.llm_model_name, duration)

                    # AGENTIC run extra resources completion
                    if sync_agent_artifacts and chat_session_id and chat_message_id:
                        try:
                            self.agent_client.extract_chat_message_artifacts(
                                base_dir=artifacts_base_dir,
                                model_key=i.model_key,
                                test_case_key=i.key,
                                chat_session_id=chat_session_id,
                                chat_message_id=chat_message_id,
                                chat_message_seq=seq,
                                fail_fast=False,
                                verbose=True,
                            )
                            # categories
                            cats.append(
                                f"{agents.AgentHost.CAT_TEST_LAB}:{test_lab_key}"
                            )
                            cat_agent_trail = RagTestLab._cat_agentic_artifacts(
                                host_type=model.model_type.name,
                                chat_session_id=chat_session_id,
                                chat_message_id=chat_message_id,
                                chat_message_seq=seq,
                            )
                            cats.append(cat_agent_trail)
                        except Exception as ex:
                            self.logger.error(
                                f"Failed to extract agentic artifacts for the chat "
                                f"session {chat_session_id}, chat message "
                                f"{chat_message_id} and collection "
                                f"{model.collection_id}: {ex}\n"
                                f"{traceback.format_exc()}"
                            )
                            # do NOT fail, but continue even w/o artifacts
                    break  # no need to retry
                except Exception as ex:
                    ex_msg = genai.H2oGpteRagClient.humanize_err_msg(
                        ex=ex,
                        timeout_exp_backoff=timeout_exp_backoff,
                    )
                    msg = (
                        f"Attempt #{a + 1} failed: {ex_msg} - Enterprise h2oGPTe host "
                        f"did not respond when its LLM model '{model.llm_model_name}' "
                        f"was used (collection ID: {model.collection_id})"
                    )
                    msg_trace = f"{msg}\n{traceback.format_exc()}"

                    self.logger.error(msg_trace)
                    # if the model fails > "h2oGPTe error" as output > FAIL
                    actual_output = f"{commons.ERROR_MODEL_HOST}: {msg_trace}"
                    progress_callback.set_progress(
                        progress=progress_callback.progress + 0.001,
                        message=msg,
                    )
                    duration = 5.0  # penalty for the error
                    cost = 0.0
                    chunks = []
                    if timeout_exp_backoff:
                        timeout_exp_backoff.retry()

                    if isinstance(ex, TimeoutError):
                        self._add_stat_llm_timeout(model.llm_model_name, i)
                    else:
                        # other err
                        self._add_stat_llm_err(model.llm_model_name, i)
                self._add_stat_duration(model.llm_model_name, duration)

            self.dataset.add_input(
                key=i.key,
                corpus=model.documents,
                i=i.i,
                context=chunks or i.context,
                categories=cats,
                relationships=i.relationships,
                expected_output=i.expected_output,
                output_constraints=i.output_constraints,
                output_condition=i.output_condition,
                actual_output=actual_output,
                actual_duration=duration,
                cost=cost,
                model_key=i.model_key,
            )
        elif isinstance(model, models.ExplainableLlmModel):
            test_lab_key = RagTestLab._test_lab_key_for_artifacts_dir(
                artifacts_base_dir
            )
            sync_agent_artifacts = (
                artifacts_base_dir
                and test_lab_key
                and model.model_cfg
                and genai.H2oGpteRagClient.CFG_USE_AGENT in str(model.model_cfg)
            )

            cats = i.categories or []
            for a in range(retry_on_error + 1):
                if a > 0:
                    self._add_stat_llm_retry(model.llm_model_name, i)
                try:
                    [
                        (
                            _,
                            actual_output,
                            duration,
                            chunks,
                            cost,
                            chat_session_id,
                            chat_message_id,
                        )
                    ] = self.rag_client.ask_model(
                        prompts=[i.i],
                        llm_model_name=model.llm_model_name,
                        chat_session_id=(
                            test_to_chat.get(i.test_key, None)
                            if i.test_key and isinstance(test_to_chat, dict)
                            else None
                        ),
                        retry_attempt=a,
                        retry_attempts=retry_on_error,
                        timeout_exp_backoff=timeout_exp_backoff,
                        **model.model_cfg,
                    )
                    if (
                        isinstance(test_to_chat, dict)
                        and i.test_key
                        and chat_session_id
                    ):
                        test_to_chat[i.test_key] = chat_session_id

                    if cost < 0.0:
                        self.logger.error(
                            "Negative cost for server: "
                            f"{self.rag_client.connection.server_id}, "
                            f"LLM model: {model.llm_model_name} and prompt:' {i.i}'"
                        )
                        cost = 0.0
                    self._add_stat_llm_success(model.llm_model_name)
                    self._add_stat_duration(model.llm_model_name, duration)

                    # AGENTIC run extra resources completion
                    if sync_agent_artifacts and chat_session_id and chat_message_id:
                        try:
                            self.agent_client.extract_chat_message_artifacts(
                                base_dir=artifacts_base_dir,
                                model_key=i.model_key,
                                test_case_key=i.key,
                                chat_session_id=chat_session_id,
                                chat_message_id=chat_message_id,
                                chat_message_seq=seq,
                                fail_fast=False,
                                verbose=True,
                            )
                            # categories
                            cats.append(
                                f"{agents.AgentHost.CAT_TEST_LAB}:{test_lab_key}"
                            )
                            cat_agent_trail = RagTestLab._cat_agentic_artifacts(
                                host_type=model.model_type.name,
                                chat_session_id=chat_session_id,
                                chat_message_id=chat_message_id,
                                chat_message_seq=seq,
                            )
                            cats.append(cat_agent_trail)
                        except Exception as ex:
                            self.logger.error(
                                f"Failed to extract agentic artifacts for the chat "
                                f"session {chat_session_id}, chat message "
                                f"{chat_message_id} and LLM (non-RAG) completion:"
                                f" {ex}\n{traceback.format_exc()}"
                            )
                            # do NOT fail, but continue even w/o artifacts
                    break  # no need to retry
                except Exception as ex:
                    ex_msg = genai.H2oGpteRagClient.humanize_err_msg(
                        ex=ex,
                        timeout_exp_backoff=timeout_exp_backoff,
                    )
                    msg = (
                        f"Attempt #{a + 1} failed: {ex_msg} - Enterprise h2oGPTe host "
                        f"did not respond when its LLM model '{model.llm_model_name}' "
                        f"was used (LLM run w/o collection ID)"
                    )
                    msg_trace = f"{msg}\n{traceback.format_exc()}"

                    self.logger.error(msg_trace)
                    # if the model fails > "h2oGPTe error" as output > FAIL
                    actual_output = f"{commons.ERROR_MODEL_HOST}: {msg_trace}"
                    progress_callback.set_progress(
                        progress=progress_callback.progress + 0.001,
                        message=msg,
                    )
                    duration = 5.0  # penalty for the error
                    cost = 0.0
                    if timeout_exp_backoff:
                        timeout_exp_backoff.retry()

                    if isinstance(ex, TimeoutError):
                        self._add_stat_llm_timeout(model.llm_model_name, i)
                    else:
                        # other err
                        self._add_stat_llm_err(model.llm_model_name, i)
                self._add_stat_duration(model.llm_model_name, duration)

            self.dataset.add_input(
                key=i.key,
                i=i.i,
                categories=cats,
                relationships=i.relationships,
                expected_output=i.expected_output,
                output_constraints=i.output_constraints,
                output_condition=i.output_condition,
                actual_output=actual_output,
                actual_duration=duration,
                cost=cost,
                model_key=i.model_key,
            )
        else:
            raise ValueError(
                f"Unsupported h2oGPTe host model type {type(model)} for the model "
                f"{model.name} and connection type {self.connection.connection_type}."
            )

    def _complete_dataset_h2ogpt(
        self,
        model: models.ExplainableRagModel,
        i,
        retry_on_error: int = DEFAULT_RETRY_ON_ERROR,
        progress_callback: progress.AbstractProgressCallbackContext | None = None,
    ):
        actual_output = ""
        duration = 0.0
        cost = 0.0

        if isinstance(model, models.ExplainableLlmModel):
            for a in range(retry_on_error + 1):
                if a > 0:
                    self._add_stat_llm_retry(model.llm_model_name, i)
                try:
                    [(_, actual_output, duration, chunks, cost, _, _)] = (
                        self.rag_client.ask_model(
                            prompts=[i.i],
                            llm_model_name=model.llm_model_name,
                            **model.model_cfg,
                        )
                    )
                    self._add_stat_llm_success(model.llm_model_name)
                    self._add_stat_duration(model.llm_model_name, duration)
                    if cost < 0.0:
                        self.logger.error(
                            "Negative cost for server: "
                            f"{self.rag_client.connection.server_id}, "
                            f"model: {model.llm_model_name} and prompt:' {i.i}'"
                        )
                        cost = 0.0
                    break  # no need to retry
                except Exception as ex:
                    msg = (
                        f"Attempt #{a + 1}: failed to chat with h2oGPT hosted LLM "
                        f"{model.llm_model_name}:  {ex}"
                    )
                    msg_trace = f"{msg}\n{traceback.format_exc()}"
                    self.logger.error(msg_trace)
                    # if the model fails > "LLM host error" as output > FAIL
                    actual_output = f"{commons.ERROR_MODEL_HOST}: {msg_trace}"
                    progress_callback.set_progress(
                        progress=progress_callback.progress + 0.001,
                        message=msg,
                    )
                    duration = 5.0  # penalty for the error
                    cost = 0.0

                    if isinstance(ex, TimeoutError):
                        self._add_stat_llm_timeout(model.llm_model_name, i)
                    else:
                        # other err
                        self._add_stat_llm_err(model.llm_model_name, i)
                self._add_stat_duration(model.llm_model_name, duration)

            self.dataset.add_input(
                key=i.key,
                i=i.i,
                categories=i.categories,
                relationships=i.relationships,
                expected_output=i.expected_output,
                output_constraints=i.output_constraints,
                output_condition=i.output_condition,
                actual_output=actual_output,
                actual_duration=duration,
                cost=cost,
                model_key=i.model_key,
            )
        else:
            raise ValueError(
                f"Unsupported h2oGPT host model type {type(model)} for the model "
                f"{model.name} and connection type {self.connection.connection_type}."
            )

    # TODO IMPROVE: h2oGPT and H2O LLMOps method could be merged (w/ msg param)
    def _complete_dataset_h2ollmops(
        self,
        model: models.ExplainableRagModel,
        i,
        retry_on_error: int = DEFAULT_RETRY_ON_ERROR,
        progress_callback: progress.AbstractProgressCallbackContext | None = None,
    ):
        actual_output = ""
        duration = 0.0
        cost = 0.0

        if isinstance(model, models.ExplainableLlmModel):
            for a in range(retry_on_error + 1):
                if a > 0:
                    self._add_stat_llm_retry(model.llm_model_name, i)
                try:
                    [(_, actual_output, duration, chunks, cost, _, _)] = (
                        self.rag_client.ask_model(
                            prompts=[i.i],
                            llm_model_name=model.llm_model_name,
                        )
                    )
                    self._add_stat_llm_success(model.llm_model_name)
                    if cost < 0.0:
                        self.logger.error(
                            "Negative cost for server: "
                            f"{self.rag_client.connection.server_id}, "
                            f"model: {model.llm_model_name} and prompt:' {i.i}'"
                        )
                        cost = 0.0
                    break  # no need to retry
                except Exception as ex:
                    msg = (
                        f"Attempt #{a + 1}: failed to chat with H2O LLMOps hosted LLM "
                        f"{model.llm_model_name}:  {ex}"
                    )
                    msg_trace = f"{msg}\n{traceback.format_exc()}"
                    self.logger.error(msg_trace)
                    # if the model fails > "host error" as output > FAIL
                    actual_output = f"{commons.ERROR_MODEL_HOST}: {msg_trace}"
                    progress_callback.set_progress(
                        progress=progress_callback.progress + 0.001,
                        message=msg,
                    )
                    duration = 5.0  # penalty for the error
                    cost = 0.0

                    if isinstance(ex, TimeoutError):
                        self._add_stat_llm_timeout(model.llm_model_name, i)
                    else:
                        # other err
                        self._add_stat_llm_err(model.llm_model_name, i)
                self._add_stat_duration(model.llm_model_name, duration)

            self.dataset.add_input(
                key=i.key,
                i=i.i,
                categories=i.categories,
                relationships=i.relationships,
                expected_output=i.expected_output,
                output_constraints=i.output_constraints,
                output_condition=i.output_condition,
                actual_output=actual_output,
                actual_duration=duration,
                cost=cost,
                model_key=i.model_key,
            )
        else:
            raise ValueError(
                f"Unsupported H2O LLOps host model type {type(model)} for the model "
                f"{model.name} and connection type {self.connection.connection_type}."
            )

    def _complete_dataset_ollama(
        self,
        model: models.ExplainableRagModel,
        i,
        retry_on_error: int = DEFAULT_RETRY_ON_ERROR,
        progress_callback: progress.AbstractProgressCallbackContext | None = None,
    ):
        actual_output = ""
        duration = 0.0
        cost = 0.0

        if isinstance(model, models.ExplainableLlmModel):
            for a in range(retry_on_error + 1):
                if a > 0:
                    self._add_stat_llm_retry(model.llm_model_name, i)
                try:
                    [(_, actual_output, duration, chunks, cost, _, _)] = (
                        self.rag_client.ask_model(
                            prompts=[i.i],
                            llm_model_name=model.llm_model_name,
                            **model.model_cfg,
                        )
                    )
                    if cost < 0.0:
                        self.logger.error(
                            "Negative cost for server: "
                            f"{self.rag_client.connection.server_id}, "
                            f"model: {model.llm_model_name} and prompt:' {i.i}'"
                        )
                        cost = 0.0
                    self._add_stat_llm_success(model.llm_model_name)
                except Exception as ex:
                    msg = (
                        f"Attempt #{a + 1}: failed to chat with ollama hosted LLM "
                        f"{model.llm_model_name}:  {ex}"
                    )
                    msg_trace = f"{msg}\n{traceback.format_exc()}"
                    self.logger.error(msg_trace)
                    # if the model fails > "host error" as output > FAIL
                    actual_output = f"{commons.ERROR_MODEL_HOST}: {msg_trace}"
                    progress_callback.set_progress(
                        progress=progress_callback.progress + 0.001,
                        message=msg,
                    )
                    duration = 5.0  # penalty for the error
                    cost = 0.0

                    if isinstance(ex, TimeoutError):
                        self._add_stat_llm_timeout(model.llm_model_name, i)
                    else:
                        # other err
                        self._add_stat_llm_err(model.llm_model_name, i)
                self._add_stat_duration(model.llm_model_name, duration)

            self.dataset.add_input(
                i=i.i,
                categories=i.categories,
                expected_output=i.expected_output,
                output_constraints=i.output_constraints,
                output_condition=i.output_condition,
                actual_output=actual_output,
                actual_duration=duration,
                cost=cost,
                model_key=i.model_key,
            )
        else:
            raise ValueError(
                f"Unsupported ollama host model type {type(model)} for the model "
                f"{model.name} and connection type {self.connection.connection_type}."
            )

    def _complete_dataset_openai_llm(
        self,
        model: models.ExplainableRagModel,
        i,
        retry_on_error: int = DEFAULT_RETRY_ON_ERROR,
        progress_callback: progress.AbstractProgressCallbackContext | None = None,
    ):
        actual_output = ""
        duration = 0.0
        cost = 0.0

        if isinstance(model, models.ExplainableLlmModel):
            for a in range(retry_on_error + 1):
                if a > 0:
                    self._add_stat_llm_retry(model.llm_model_name, i)
                try:
                    [(_, actual_output, duration, chunks, cost, _, _)] = (
                        self.rag_client.ask_model(
                            prompts=[i.i],
                            llm_model_name=model.llm_model_name,
                            **model.model_cfg,
                        )
                    )
                    if cost < 0.0:
                        self.logger.error(
                            "Negative cost for server: "
                            f"{self.rag_client.connection.server_id}, "
                            f"model: {model.llm_model_name} and prompt:' {i.i}'"
                        )
                        cost = 0.0
                    self._add_stat_llm_success(model.llm_model_name)
                    break  # no need to retry
                except Exception as ex:
                    msg = (
                        f"Attempt #{a + 1}: failed to chat with OpenAI hosted LLM "
                        f"{model.llm_model_name}:  {ex}"
                    )
                    msg_trace = f"{msg}\n{traceback.format_exc()}"
                    self.logger.error(msg_trace)
                    # if the model fails > host error as output > FAIL
                    actual_output = f"{commons.ERROR_MODEL_HOST}: {msg_trace}"
                    progress_callback.set_progress(
                        progress=progress_callback.progress + 0.001,
                        message=msg,
                    )
                    duration = 5.0  # penalty for the error
                    cost = 0.0

                    if isinstance(ex, TimeoutError):
                        self._add_stat_llm_timeout(model.llm_model_name, i)
                    else:
                        # other err
                        self._add_stat_llm_err(model.llm_model_name, i)
                self._add_stat_duration(model.llm_model_name, duration)

            self.dataset.add_input(
                key=i.key,
                i=i.i,
                categories=i.categories,
                relationships=i.relationships,
                expected_output=i.expected_output,
                output_constraints=i.output_constraints,
                output_condition=i.output_condition,
                actual_output=actual_output,
                actual_duration=duration,
                cost=cost,
                model_key=i.model_key,
            )
        else:
            raise ValueError(
                f"Unsupported OpenAI host model type {type(model)} for the model "
                f"{model.name} and connection type {self.connection.connection_type}."
            )

    def _complete_dataset_anthropic_llm(
        self,
        model: models.ExplainableRagModel,
        i,
        retry_on_error: int = DEFAULT_RETRY_ON_ERROR,
        progress_callback: progress.AbstractProgressCallbackContext | None = None,
    ):
        actual_output = ""
        duration = 0.0
        cost = 0.0

        if isinstance(model, models.ExplainableLlmModel):
            for a in range(retry_on_error + 1):
                if a > 0:
                    self._add_stat_llm_retry(model.llm_model_name, i)
                try:
                    [(_, actual_output, duration, chunks, cost, _, _)] = (
                        self.rag_client.ask_model(
                            prompts=[i.i],
                            llm_model_name=model.llm_model_name,
                            **model.model_cfg,
                        )
                    )
                    if cost < 0.0:
                        self.logger.error(
                            "Negative cost for server: "
                            f"{self.rag_client.connection.server_id}, "
                            f"model: {model.llm_model_name} and prompt:' {i.i}'"
                        )
                        cost = 0.0
                    self._add_stat_llm_success(model.llm_model_name)
                    break  # no need to retry
                except Exception as ex:
                    msg = (
                        f"Attempt #{a + 1}: failed to chat with Anthropic hosted LLM "
                        f"{model.llm_model_name}:  {ex}"
                    )
                    msg_trace = f"{msg}\n{traceback.format_exc()}"
                    self.logger.error(msg_trace)
                    # if the model fails > host error as output > FAIL
                    actual_output = f"{commons.ERROR_MODEL_HOST}: {msg_trace}"
                    progress_callback.set_progress(
                        progress=progress_callback.progress + 0.001,
                        message=msg,
                    )
                    duration = 5.0  # penalty for the error
                    cost = 0.0

                    if isinstance(ex, TimeoutError):
                        self._add_stat_llm_timeout(model.llm_model_name, i)
                    else:
                        # other err
                        self._add_stat_llm_err(model.llm_model_name, i)
                self._add_stat_duration(model.llm_model_name, duration)

            self.dataset.add_input(
                key=i.key,
                i=i.i,
                categories=i.categories,
                relationships=i.relationships,
                expected_output=i.expected_output,
                output_constraints=i.output_constraints,
                output_condition=i.output_condition,
                actual_output=actual_output,
                actual_duration=duration,
                cost=cost,
                model_key=i.model_key,
            )
        else:
            raise ValueError(
                f"Unsupported Anthropic host model type {type(model)} for the model "
                f"{model.name} and connection type {self.connection.connection_type}."
            )

    def _complete_dataset_azure_openai_llm(
        self,
        model: models.ExplainableRagModel,
        i,
        retry_on_error: int = DEFAULT_RETRY_ON_ERROR,
        progress_callback: progress.AbstractProgressCallbackContext | None = None,
    ):
        actual_output = ""
        duration = 0.0
        cost = 0.0

        if isinstance(model, models.ExplainableLlmModel):
            for a in range(retry_on_error + 1):
                if a > 0:
                    self._add_stat_llm_retry(model.llm_model_name, i)
                try:
                    [(_, actual_output, duration, chunks, cost, _, _)] = (
                        self.rag_client.ask_model(
                            prompts=[i.i],
                            llm_model_name=model.llm_model_name,
                            **model.model_cfg,
                        )
                    )
                    self._add_stat_llm_success(model.llm_model_name)
                    if cost < 0.0:
                        self.logger.error(
                            "Negative cost for server: "
                            f"{self.rag_client.connection.server_id}, "
                            f"model: {model.llm_model_name} and prompt:' {i.i}'"
                        )
                        cost = 0.0
                    break  # no need to retry
                except Exception as ex:
                    msg = (
                        f"Attempt #{a + 1}: failed to chat with "
                        f"Microsoft Azure OpenAI hosted LLM "
                        f"{model.llm_model_name}:  {ex}"
                    )
                    msg_trace = f"{msg}\n{traceback.format_exc()}"
                    self.logger.error(msg_trace)
                    # if the model fails > host error as output > FAIL
                    actual_output = f"{commons.ERROR_MODEL_HOST}: {msg_trace}"
                    progress_callback.set_progress(
                        progress=progress_callback.progress + 0.001,
                        message=msg,
                    )
                    duration = 5.0  # penalty for the error
                    cost = 0.0

                    if isinstance(ex, TimeoutError):
                        self._add_stat_llm_timeout(model.llm_model_name, i)
                    else:
                        # other err
                        self._add_stat_llm_err(model.llm_model_name, i)
                self._add_stat_duration(model.llm_model_name, duration)

            self.dataset.add_input(
                key=i.key,
                i=i.i,
                categories=i.categories,
                relationships=i.relationships,
                expected_output=i.expected_output,
                output_constraints=i.output_constraints,
                output_condition=i.output_condition,
                actual_output=actual_output,
                actual_duration=duration,
                cost=cost,
                model_key=i.model_key,
            )
        else:
            raise ValueError(
                f"Unsupported Microsoft Azure OpenAI host model type "
                f"{type(model)} for the model "
                f"{model.name} and connection type {self.connection.connection_type}."
            )

    def _complete_dataset_openai(
        self,
        model: models.ExplainableRagModel,
        i,
        complete_context: int = 10,
        retry_on_error: int = DEFAULT_RETRY_ON_ERROR,
        progress_callback: progress.AbstractProgressCallbackContext | None = None,
    ):
        chunks = []
        actual_output = ""
        duration = 0.0
        cost = 0.0

        if isinstance(model, models.ExplainableRagModel):
            for a in range(retry_on_error + 1):
                if a > 0:
                    self._add_stat_llm_retry(model.llm_model_name, i)
                try:
                    [(_, actual_output, duration, chunks, cost, _, _)] = (
                        self.rag_client.ask_collection(
                            assistant_id=model.collection_id,
                            prompts=[i.i],
                            include_chunks=complete_context,
                        )
                    )
                    if cost < 0.0:
                        self.logger.error(
                            "Negative cost for server: "
                            f"{self.rag_client.connection.server_id}, "
                            f"model: {model.llm_model_name} and prompt:' {i.i}'"
                        )
                        cost = 0.0
                    self._add_stat_llm_success(model.llm_model_name)
                    break  # no need to retry
                except Exception as ex:
                    msg = (
                        f"Attempt #{a + 1}: failed to chat with OpenAI Assistant "
                        f"{model.collection_id} using model {model.llm_model_name}:"
                        f" {ex}"
                    )
                    msg_trace = f"{msg}\n{traceback.format_exc()}"
                    self.logger.error(msg_trace)
                    actual_output = f"{commons.ERROR_MODEL_HOST}: {msg_trace}"
                    progress_callback.set_progress(
                        progress=progress_callback.progress + 0.001,
                        message=msg,
                    )
                    duration = 5.0  # penalty for the error
                    cost = 0.0
                    chunks = []

                    if isinstance(ex, TimeoutError):
                        self._add_stat_llm_timeout(model.llm_model_name, i)
                    else:
                        # other err
                        self._add_stat_llm_err(model.llm_model_name, i)
                self._add_stat_duration(model.llm_model_name, duration)

            self.dataset.add_input(
                key=i.key,
                i=i.i,
                context=chunks or i.context,
                categories=i.categories,
                relationships=i.relationships,
                expected_output=i.expected_output,
                output_constraints=i.output_constraints,
                output_condition=i.output_condition,
                actual_output=actual_output,
                actual_duration=duration,
                cost=cost,
                model_key=i.model_key,
            )
        else:
            raise ValueError(
                f"Unsupported OpenAI host model type {type(model)} for the model "
                f"{model.name} and connection type {self.connection.connection_type}."
            )

    def _complete_dataset_bedrock(
        self,
        model: models.ExplainableRagModel,
        i,
        retry_on_error: int = DEFAULT_RETRY_ON_ERROR,
        progress_callback: progress.AbstractProgressCallbackContext | None = None,
    ):
        chunks = []
        actual_output = ""
        duration = 0.0
        cost = 0.0

        if isinstance(model, models.ExplainableRagModel):
            for a in range(retry_on_error + 1):
                if a > 0:
                    self._add_stat_llm_retry(model.llm_model_name, i)
                try:
                    [(_, actual_output, duration, chunks, cost, _, _)] = (
                        self.rag_client.ask_collection(
                            collection_id=model.collection_id,
                            llm_model_name=model.llm_model_name,
                            prompts=[i.i],
                        )
                    )
                    if cost < 0.0:
                        self.logger.error(
                            "Negative cost for server: "
                            f"{self.rag_client.connection.server_id}, "
                            f"model: {model.llm_model_name} and prompt:' {i.i}'"
                        )
                        cost = 0.0
                    self._add_stat_llm_success(model.llm_model_name)
                    break  # no need to retry
                except Exception as ex:
                    msg = (
                        f"Attempt #{a + 1}: failed to chat with Bedrock Assistant "
                        f"{model.collection_id} using model {model.llm_model_name}:"
                        f" {ex}\n{traceback.format_exc()}"
                    )
                    msg_trace = f"{msg}\n{traceback.format_exc()}"
                    self.logger.error(msg_trace)
                    actual_output = f"{commons.ERROR_MODEL_HOST}: {msg_trace}"
                    progress_callback.set_progress(
                        progress=progress_callback.progress + 0.001,
                        message=msg,
                    )
                    duration = 5.0  # penalty for the error
                    cost = 0.0
                    chunks = []

                    if isinstance(ex, TimeoutError):
                        self._add_stat_llm_timeout(model.llm_model_name, i)
                    else:
                        # other err
                        self._add_stat_llm_err(model.llm_model_name, i)
                self._add_stat_duration(model.llm_model_name, duration)

            self.dataset.add_input(
                key=i.key,
                i=i.i,
                context=chunks or i.context,
                categories=i.categories,
                relationships=i.relationships,
                expected_output=i.expected_output,
                output_constraints=i.output_constraints,
                output_condition=i.output_condition,
                actual_output=actual_output,
                actual_duration=duration,
                cost=cost,
                model_key=i.model_key,
            )
        else:
            raise ValueError(
                f"Unsupported Bedrock host model type {type(model)} for the model "
                f"{model.name} and connection type {self.connection.connection_type}."
            )

    def _complete_dataset_h2ogpte_stats(
        self, include_llm_meta: bool = True, interval_in_seconds: int = 0
    ):
        """Get LLM performance statistics for all models hosted by h2oGPTe."""
        if (
            include_llm_meta
            and models.OPT_EXPLAINABLE_MODEL_META
            and self.connection.connection_type
            == h2o_sonar_config.ConnectionConfigType.H2O_GPT_E.name
            and hasattr(self.rag_client.client, "get_llm_performance_by_llm")
        ):
            from h2ogpte import types as h2ogpte_types

            # interval_in_seconds += 200  # debug only
            # Extends the time interval during debugging to compensate for slower
            # execution (e.g., stepping through code). This ensures operations
            # relying on time intervals still produce results, though stats may
            # show inflated activity temporarily.

            perf_interval = (
                f"{interval_in_seconds} seconds" if interval_in_seconds else "24 hours"
            )

            try:
                llm_perf_profiles: list[h2ogpte_types.LLMPerformance] = (
                    self.rag_client.client.get_llm_performance_by_llm(perf_interval)
                )
            except Exception:
                llm_perf_profiles = []

            # vision models map: model name -> vision model name
            vision_models = {}
            if hasattr(self.rag_client.client, "get_llm_and_auto_vision_llm_names"):
                vision_models = (
                    self.rag_client.client.get_llm_and_auto_vision_llm_names()
                )

            # map performance to the explainable models
            if llm_perf_profiles:
                for em in self.evaluated_models.values():
                    for llm_perf in llm_perf_profiles:
                        if em.llm_model_name == llm_perf.llm_name:
                            em.llm_model_meta[
                                models.ExplainableLlmModel.KEY_H2OGPTE_STATS
                            ] = llm_perf.__dict__

                            if em.llm_model_name in vision_models:
                                em.llm_model_meta[
                                    models.ExplainableLlmModel.KEY_H2OGPTE_STATS
                                ][
                                    models.ExplainableLlmModel.KEY_H2OGPTE_VISION_M
                                ] = vision_models[em.llm_model_name]

                            break

    def _complete_dataset_llm_stats(self):
        """Get LLM statistics for all models."""
        # map performance to the explainable models
        for em in self.evaluated_models.values():
            em.llm_model_meta.setdefault(
                models.ExplainableLlmModel.KEY_STATS_SUCCESS,
                0,  # number of calls which were successful
            )
            em.llm_model_meta.setdefault(
                models.ExplainableLlmModel.KEY_STATS_RETRY, 0
            )  # number of calls which were retried, the successful call is not counted
            em.llm_model_meta.setdefault(
                models.ExplainableLlmModel.KEY_STATS_TIMEOUT,
                0,  # number of calls which timed out
            )
            em.llm_model_meta.setdefault(
                models.ExplainableLlmModel.KEY_STATS_FAILURE,
                0,  # number of calls which failed because of
                # another reason than timeout
            )
            em.llm_model_meta.setdefault(
                models.ExplainableLlmModel.KEY_STATS_DURATION,  # duration dictionary
                # with request time statistics
                {"max": 0.0, "min": 0.0, "n": 0, "sum": 0.0},
            )

            em.llm_model_meta[models.ExplainableLlmModel.KEY_STATS_SUCCESS] += (
                self.stat_llm_model_success_count.get(em.llm_model_name, 0)
            )

            em.llm_model_meta[models.ExplainableLlmModel.KEY_STATS_RETRY] += len(
                self.stat_llm_model_retries.get(em.llm_model_name, [])
            )

            em.llm_model_meta[models.ExplainableLlmModel.KEY_STATS_TIMEOUT] += len(
                self.stat_llm_model_timeouts.get(em.llm_model_name, [])
            )

            em.llm_model_meta[models.ExplainableLlmModel.KEY_STATS_FAILURE] += len(
                self.stat_llm_model_errors.get(em.llm_model_name, [])
            )

            if (
                self.stat_llm_model_request_duration
                and self.stat_llm_model_request_duration.get(em.llm_model_name)
                and isinstance(
                    self.stat_llm_model_request_duration.get(em.llm_model_name), dict
                )
            ):
                llm_model_meta_duration: dict[str, float] = em.llm_model_meta[
                    models.ExplainableLlmModel.KEY_STATS_DURATION
                ]
                llm_model_meta_duration["n"] += (
                    self.stat_llm_model_request_duration.get(em.llm_model_name).get(
                        "n", 0
                    )
                )
                llm_model_meta_duration["sum"] += (
                    self.stat_llm_model_request_duration.get(em.llm_model_name).get(
                        "sum", 0.0
                    )
                )
                llm_model_meta_duration["min"] = (
                    min(
                        self.stat_llm_model_request_duration.get(em.llm_model_name).get(
                            "min", 0.0
                        ),
                        llm_model_meta_duration["min"],
                    )
                    if llm_model_meta_duration["min"] > 0.0
                    else self.stat_llm_model_request_duration.get(
                        em.llm_model_name
                    ).get("min", 0.0)
                )
                llm_model_meta_duration["max"] = max(
                    self.stat_llm_model_request_duration.get(em.llm_model_name).get(
                        "max", 0.0
                    ),
                    llm_model_meta_duration["max"],
                )

    def complete_dataset(
        self,
        complete_context: int = 10,
        progress_callback: progress.AbstractProgressCallbackContext | None = None,
        save_as_you_go: pathlib.Path | str | None = None,
        parallelize: int = TestLab.SEQUENTIAL_RUN,
        multi_turn: bool = False,
        retry_on_error: int = DEFAULT_RETRY_ON_ERROR,
        timeout_exp_backoff: genai.TimeoutRetryExpBackoffCtx | None = None,
        include_llm_meta: bool = True,
        raise_on_all_tcs_fail: bool = True,
        artifacts_base_dir: pathlib.Path | str | None = None,
        purge_workdir: bool = True,
    ):
        """Complete the dataset with the actual values from an LLM host.

        Parameters
        ----------
        complete_context : int
            How many context text chunks to include in the resolved dataset.
        progress_callback : progress.AbstractProgressCallbackContext | None
            Optional progress callback context.
        save_as_you_go : pathlib.Path | str | None
            Save the dataset as JSON after each input is resolved.
        parallelize : int
            Complete the dataset in parallel using multiple processes. Use
            ``-1`` for auto-choice of the number of workers, ``0`` to disable
            parallelization (will create the lab using sequential requests),
            and ``1`` + (positive integer) to specify the number of workers.
        multi_turn : bool
            Whether to use multi-turn chat with the LLM host - if enabled,
            then all test cases within the test will be handled within
            the single session i.e. the same chat session i.e. the same context.
        retry_on_error : int
            How many times to retry the failed LLM host requests.
        timeout_exp_backoff : TimeoutRetryExpBackoffCtx | None
            Optionally override timeout which can be specified in the model host
            configuration ExplainableRagModel::model_cfg and which is model host type
            specific and use exponential backoff strategy for the timeout handling.
            Timeout is increased on each retry by the backoff factor.
        include_llm_meta : bool
            Whether to include the LLM meta-data like performance statistics.
        raise_on_all_tcs_fail : bool
            Raise an exception if all test cases fail.
        artifacts_base_dir : pathlib.Path | str | None
            Base directory for storing completion artifacts. If not specified,
            a new artifacts directory will be created under the h2o-sonar/ base
            directory if available. The directory is created on the first save
            of an artifact.
        purge_workdir : bool
            Purge the working directory with lab shards after the completion.

        """
        retry_on_error = 0 if retry_on_error < 0 else retry_on_error
        complete_context = (
            10
            if (
                complete_context is not None
                and isinstance(complete_context, bool)
                and complete_context
            )
            else complete_context
        )

        # progress
        parent_progress_callback = progress_callback
        progress_callback = progress.LoggingProgressCallbackContext(
            logger=self.logger,
            prefix="TestLab COMPLETION progress",
            parent_callback=parent_progress_callback,
            verbose_children=(
                parent_progress_callback.verbose_children
                if parent_progress_callback
                else True
            ),
            name="TestLab COMPLETION callback",
        )
        progress_callback.set_progress(0.0, "Started")

        # stateful multi-turn chat is run SEQUENTIALLY - LATER sharding by test
        if multi_turn:
            # automatic fallback to sequential run
            self.logger.warning(
                "Multi-turn chat is NOT supported in parallel mode - switching to "
                "sequential run"
            )
            parallelize = TestLab.SEQUENTIAL_RUN

        # base dir for storing completion artifacts:
        # - NEW directory is created on every completion as completion may be
        #   run multiple times on the same lab instance
        # - in case of parallel run, all workers share the same base dir for artifacts
        if not artifacts_base_dir:
            artifacts_base_dir = (
                (self.user_dir / f"{RagTestLab.DIR_TEST_LAB}{uuid.uuid4()}")
                if self.user_dir
                else None
            )
        else:
            artifacts_base_dir = pathlib.Path(artifacts_base_dir)

        # do NOT parallelize the completion of the dataset if there is only one model
        complete_start = time.time()
        if parallelize != TestLab.SEQUENTIAL_RUN:
            self._complete_executor(
                progress_callback=progress_callback,
                complete_context=complete_context,
                save_as_you_go=True if save_as_you_go else False,
                parallelize=parallelize,
                retry_on_error=retry_on_error,
                timeout_exp_backoff=timeout_exp_backoff,
                artifacts_base_dir=artifacts_base_dir,
                purge_workdir=purge_workdir,
            )

            self._complete_dataset_h2ogpte_stats(
                include_llm_meta=include_llm_meta,
                interval_in_seconds=int(time.time() - complete_start),
            )

            self._complete_dataset_llm_stats()

            return self.dataset
        else:
            self.dataset = datasets.LlmDataset()

            # map: model key -> model
            key_2_model = {m.key: m for m in self.evaluated_models.values()}

            # progress
            (
                progress_slot_min,
                progress_slot_size,
                steps,
            ) = progress.ProgressCallbackContext.step_loop_prepare(
                progress_min=0.0,
                progress_max=1.0,
                steps=len(self.raw_dataset.inputs),
            )
            # resolve & expand the raw dataset
            llm_failures = 0
            h2ogpte_test_to_chat = {} if multi_turn else None
            for e, i in enumerate(self.raw_dataset.inputs):
                model = key_2_model[i.model_key]
                if not model:
                    raise RuntimeError(
                        f"Model for the input {i} is not available in the test lab."
                    )

                # progress
                (
                    step_slot_min,
                    step_slot_max,
                ) = progress.ProgressCallbackContext.step_loop_get_min_and_max(
                    step=e,
                    progress_slot_min=progress_slot_min,
                    progress_slot_size=progress_slot_size,
                )
                progress_prompt = (
                    f"{i.i[:30]}..."
                    if i.i and isinstance(i.i, str) and len(i.i) > 30
                    else i.i
                )
                progress_callback.set_progress(
                    progress=step_slot_min,
                    message=(
                        f"A job is resolving the prompt "
                        f"{e + 1}/{len(self.raw_dataset.inputs)}: "
                        f"'{progress_prompt}' "
                        f"with LLM {model.llm_model_name} "
                        f"on '{model.name}'"
                    ),
                )

                # CACHE: try cache first, then ask LLM host if needed
                if self.llm_host_prompt_cache:
                    self.logger.info("Using PROMPT CACHE to resolve the prompt...")
                    if isinstance(model, models.ExplainableLlmModel):
                        corpus = None
                        cache_key = self.llm_host_prompt_cache.get_key(
                            explainable_model_type=model.model_type,
                            prompt=i.i,
                            llm_model_name=model.llm_model_name,
                            corpus=corpus,
                        )
                    elif isinstance(model, models.ExplainableRagModel):
                        corpus = model.documents
                        cache_key = self.llm_host_prompt_cache.get_key(
                            explainable_model_type=model.model_type,
                            prompt=i.i,
                            llm_model_name=model.llm_model_name,
                            corpus=corpus,
                        )
                    else:
                        raise ValueError(
                            f"Unsupported LLM model type {type(model)} for the model "
                            f"{model.name} and connection type "
                            f"{self.connection.connection_type}."
                        )
                    cache_entry = self.llm_host_prompt_cache.get(cache_key)
                    if cache_entry:
                        self.logger.info(f"  CACHE HIT  -> cache key: '{cache_key}'")

                        cost = cache_entry.get(LlmHostPromptCache.KEY_COST, 0.0)
                        if cost < 0.0:
                            self.logger.error(
                                "Negative cost for cached prompt answer: "
                                f"{model.connection.server_id}, "
                                f"model: {model.llm_model_name} and prompt:' {i.i}'"
                            )
                            cost = 0.0

                        self.dataset.add_input(
                            key=i.key,
                            corpus=corpus,
                            i=i.i,
                            context=cache_entry.get(
                                LlmHostPromptCache.KEY_CONTEXT, None
                            ),
                            categories=i.categories,
                            relationships=i.relationships,
                            expected_output=i.expected_output,
                            output_constraints=i.output_constraints,
                            output_condition=i.output_condition,
                            actual_output=cache_entry.get(
                                LlmHostPromptCache.KEY_ACTUAL_OUTPUT, ""
                            ),
                            actual_duration=cache_entry.get(
                                LlmHostPromptCache.KEY_DURATION, 0.0
                            ),
                            cost=cost,
                            model_key=i.model_key,
                        )

                        if save_as_you_go:
                            self.save_as_json(save_as_you_go)

                        progress_callback.set_progress(
                            progress=step_slot_max,
                            message=(
                                f"{e + 1}/{steps} Completed dataset row from cache: {i}"
                            ),
                        )

                        continue

                # CHAT to get the actual values
                if (
                    self.connection.connection_type
                    == h2o_sonar_config.ConnectionConfigType.H2O_GPT_E.name
                ):
                    self._complete_dataset_h2ogpte(
                        model=model,
                        i=i,
                        complete_context=complete_context,
                        test_to_chat=h2ogpte_test_to_chat,
                        seq=e,
                        retry_on_error=retry_on_error,
                        timeout_exp_backoff=timeout_exp_backoff,
                        artifacts_base_dir=artifacts_base_dir,
                        progress_callback=progress_callback,
                    )
                elif (
                    self.connection.connection_type
                    == h2o_sonar_config.ConnectionConfigType.OLLAMA.name
                ):
                    self._complete_dataset_ollama(
                        model=model,
                        i=i,
                        retry_on_error=retry_on_error,
                        progress_callback=progress_callback,
                    )
                elif (
                    self.connection.connection_type
                    == h2o_sonar_config.ConnectionConfigType.AZURE_OPENAI_CHAT.name
                ):
                    self._complete_dataset_azure_openai_llm(
                        model=model,
                        i=i,
                        retry_on_error=retry_on_error,
                        progress_callback=progress_callback,
                    )
                elif (
                    self.connection.connection_type
                    == h2o_sonar_config.ConnectionConfigType.OPENAI_CHAT.name
                ):
                    self._complete_dataset_openai_llm(
                        model=model,
                        i=i,
                        retry_on_error=retry_on_error,
                        progress_callback=progress_callback,
                    )
                elif (
                    self.connection.connection_type
                    == h2o_sonar_config.ConnectionConfigType.OPENAI_RAG.name
                ):
                    self._complete_dataset_openai(
                        model=model,
                        i=i,
                        complete_context=complete_context,
                        retry_on_error=retry_on_error,
                        progress_callback=progress_callback,
                    )
                elif (
                    self.connection.connection_type
                    == h2o_sonar_config.ConnectionConfigType.AMAZON_BEDROCK.name
                ):
                    self._complete_dataset_bedrock(
                        model=model,
                        i=i,
                        retry_on_error=retry_on_error,
                        progress_callback=progress_callback,
                    )
                elif (
                    self.connection.connection_type
                    == h2o_sonar_config.ConnectionConfigType.H2O_GPT.name
                ):
                    self._complete_dataset_h2ogpt(
                        model=model,
                        i=i,
                        retry_on_error=retry_on_error,
                        progress_callback=progress_callback,
                    )
                elif (
                    self.connection.connection_type
                    == h2o_sonar_config.ConnectionConfigType.H2O_LLM_OPS.name
                ):
                    self._complete_dataset_h2ollmops(
                        model=model,
                        i=i,
                        retry_on_error=retry_on_error,
                        progress_callback=progress_callback,
                    )
                elif (
                    self.connection.connection_type
                    == h2o_sonar_config.ConnectionConfigType.ANTHROPIC_CHAT.name
                ):
                    self._complete_dataset_anthropic_llm(
                        model=model,
                        i=i,
                        retry_on_error=retry_on_error,
                        progress_callback=progress_callback,
                    )
                else:
                    raise ValueError(
                        f"Unsupported LLM host connection type: '{self.connection}'"
                    )

                # detect & count LLM/RAG failures
                if self.dataset.inputs:
                    last_output = self.dataset.inputs[-1].actual_output
                    if last_output and last_output.startswith(commons.ERROR_MODEL_HOST):
                        llm_failures += 1

                # TODO add new entry to the cache

                if save_as_you_go:
                    self.save_as_json(save_as_you_go)

                # get question prefix with 30 characters from the dataset row
                if i and i.i:
                    i_str = f"'{i.i[:30]}...'" if i.i and len(i.i) > 30 else i.i
                else:
                    i_str = ""

                progress_callback.set_progress(
                    progress=step_slot_max,
                    message=(
                        f"Prompt {e + 1}/{steps} completed"
                        f"{': ' if i_str else '.'}{i_str} with "
                        f"LLM {model.llm_model_name} on '{model.name}'"
                    ),
                )

            self._complete_dataset_h2ogpte_stats(
                include_llm_meta=include_llm_meta,
                interval_in_seconds=int(time.time() - complete_start),
            )

            self._complete_dataset_llm_stats()

            # if completion of all the rows failed, raise an error
            if llm_failures == len(self.raw_dataset.inputs):
                msg = (
                    f"Failed to complete the test lab using "
                    f"{self.connection.connection_type} LLM host  - completion "
                    f"failed for all test cases ({len(self.raw_dataset.inputs)}) and "
                    f"LLMs."
                )
                if raise_on_all_tcs_fail:
                    progress_callback.set_progress(progress=1.0, message=msg)
                    raise RuntimeError(msg)
                else:
                    self.logger.error(msg)

            progress_callback.set_progress(
                progress=1.0,
                message=(
                    f"Dataset completion DONE with {llm_failures} LLM host errors."
                    if llm_failures
                    else "Dataset completion DONE"
                ),
            )

            return self.dataset

    def purge(self):
        """Purge the test lab by deleting all the created collections/assistants
        and uploaded documents.

        """
        if isinstance(self.rag_client, genai.RagClient):
            # purge all ASSISTANTS create by the RAG client
            self.rag_client.purge_collections()
            # purge all DOCUMENTS uploaded by the RAG client
            self.rag_client.purge_uploaded_docs()

    def trim(self, max_llm_models_count=None):
        """Trim the test lab by keeping only specified number of LLM models and
        removing all the orphans.

        """
        new_llm_model_names = self.llm_model_names[:max_llm_models_count]

        # LLM models: keep only models that have ^ llm in it
        new_evaluated_models = {}
        for m in self.evaluated_models.values():
            if m.llm_model_name in new_llm_model_names:
                new_evaluated_models[m.name] = m
        new_evaluated_models_keys = [m.key for m in new_evaluated_models.values()]

        # RAW dataset: keep only rows that have the model
        new_raw_dataset = datasets.LlmDataset()
        for i in self.raw_dataset.inputs:
            if i.model_key in new_evaluated_models_keys:
                new_raw_dataset.inputs.append(i)

        # RESOLVED dataset: keep only rows that have the model
        new_dataset = datasets.LlmDataset()
        for i in self.dataset.inputs:
            if i.model_key in new_evaluated_models_keys:
                new_dataset.inputs.append(i)

        new_lab = RagTestLab(
            llm_host_connection=self.connection,
            raw_dataset=new_raw_dataset,
            evaluated_models=list(new_evaluated_models.values()),
            llm_model_names=new_llm_model_names,
            docs_cache_dir=self.docs_cache_dir,
            name=self.name,
            description=self.description,
            llm_host_prompt_cache=self.llm_host_prompt_cache,
            logger=self.logger,
        )
        new_lab.dataset = new_dataset

        return new_lab

    def merge(self, other_test_lab: "RagTestLab", other_llm_prefix: str = "Other"):
        """Merge another test lab into this one."""
        if not other_test_lab:
            raise ValueError("Test lab to merge is required")

        # merge raw datasets
        self.raw_dataset.merge(other_test_lab.raw_dataset)

        # merge resolved datasets
        self.dataset.merge(other_test_lab.dataset)

        # merge models
        for m in other_test_lab.evaluated_models.values():
            new_m = m.clone()
            new_m.name = f"{other_llm_prefix} {new_m.name}"
            new_m.llm_model_name = (
                f"{other_llm_prefix.lower()}/{new_m.llm_model_name}"
                if other_llm_prefix
                else new_m.llm_model_name
            )
            if new_m.name not in self.evaluated_models:
                self.evaluated_models[new_m.name] = new_m
            else:
                raise ValueError(
                    f"Model {new_m.name} is already present in the target test lab."
                )

        # merge base LLM names
        for m in other_test_lab.llm_model_names:
            m_new = f"{other_llm_prefix.lower()}/{m}" if other_llm_prefix else m
            if m_new not in self.llm_model_names:
                self.llm_model_names.append(m_new)
            else:
                raise ValueError(
                    f"LLM model {m_new} is already present in the target test lab."
                )

        # merge docs cache
        for k in other_test_lab.doc_cache:
            if k not in self.doc_cache:
                self.doc_cache[k] = other_test_lab.doc_cache[k]

    @staticmethod
    def _is_internal_error_msg(msg: str) -> bool:
        if msg:
            return commons.ERROR_MODEL_HOST.lower() in str(msg).lower()

        return False

    def insight_internal_llm_errors(
        self,
        report_dir: str | pathlib.Path = "",
        src: str = "stats",
    ) -> tuple[dict, str]:
        """Create Markdown report with the internal LLM errors.

        Parameters
        ----------
        report_dir : str | pathlib.Path
            Directory to save the reports as JSon and Markdown to.
        src : str
            Source of the errors: ``stats`` (default) or ``dataset``
            (text of answers analysis).

        """
        # TODO report generation from the TestLab's stat to be implemented
        del src

        # map: model_key -> LLM model name
        model_key_to_name = {}
        for m in self.evaluated_models.values():
            model_key_to_name[m.key] = m.llm_model_name

        # map: LLM model name -> [errors from actual_output]
        errors_by_model = {}
        err_inputs = self.dataset.inputs
        err_count = 0
        for i in err_inputs:
            if i.actual_output and RagTestLab._is_internal_error_msg(i.actual_output):
                err_count += 1
                print(f"Found error:\n{i.actual_output}")
                model_name = model_key_to_name.get(i.model_key, "")

                if model_name not in errors_by_model:
                    errors_by_model[model_name] = []
                errors_by_model[model_name].append(i.actual_output)

        # THEN
        print(f"Found {err_count} errors out of {len(err_inputs)} requests.")
        # generate markdown report
        md_report = "# Test Lab Internal Server Error Analysis\n\n"
        md_report += (
            f"Test lab was impacted by  **{err_count} errors** out of "
            f"{len(err_inputs)} inputs.\n\n"
        )
        md_report += "**Models by failures:**\n\n"
        # sort errors_by_model by the size of value length
        errors_by_model = {
            k: v
            for k, v in sorted(
                errors_by_model.items(), key=lambda item: len(item[1]), reverse=True
            )
        }
        for e, model_name in enumerate(errors_by_model.keys()):
            md_report += (
                f"{e + 1}. [{model_name}](#{model_name}) "
                f"({len(errors_by_model[model_name])})\n"
            )
        md_report += "\n\n"
        md_report += "All LLM models:\n\n"
        for e, model_name in enumerate(self.llm_model_names):
            md_report += f"{e + 1}. {model_name}\n"
        md_report += "\n\n"
        for model_name, errors in errors_by_model.items():
            md_report += f"## {model_name}\n\n"
            for e, err in enumerate(errors):
                md_report += f"{e + 1}. {model_name} error:\n```\n{err}```\n\n\n"

        if report_dir:
            report_dir = pathlib.Path(report_dir)
            # save markdown
            with open(report_dir / "errors_by_llm_model.md", "w") as f:
                f.write(md_report)
            # save JSon
            with open(report_dir / "errors_by_llm_model.json", "w") as f:
                json.dump(errors_by_model, f, indent=4)

        return errors_by_model, md_report

    def to_dict(self) -> dict:
        return {
            RagTestLab.KEY_NAME: self.name,
            RagTestLab.KEY_DESCRIPTION: self.description,
            RagTestLab.KEY_RAW_DATASET: self.raw_dataset.to_dict(),
            RagTestLab.KEY_DATASET: self.dataset.to_dict(),
            RagTestLab.KEY_MODELS: [
                m.to_dict() for m in self.evaluated_models.values()
            ],
            RagTestLab.KEY_BASE_MODEL_NAMES: self.llm_model_names,
            RagTestLab.KEY_DOCS_CACHE: {
                k: str(self.doc_cache[k]) for k in self.doc_cache
            },
        }

    @staticmethod
    def from_eval_results(
        eval_results_path: str | pathlib.Path,
        interpretation_json_path: str | pathlib.Path,
        raw_dataset_empty: bool = True,
    ):
        """Create a test lab from the evaluation results archive.

        Parameters
        ----------
        eval_results_path : str | pathlib.Path
            Path to the evaluation results JSon file path.
        interpretation_json_path : str | pathlib.Path
            Path to the interpretation JSon file path.
        raw_dataset_empty : bool
            Whether to create an empty raw dataset or copy resolved dataset

        """
        if not eval_results_path or not interpretation_json_path:
            raise ValueError(
                "Both evaluation results and interpretation JSON paths are required."
            )
        eval_results_path = pathlib.Path(eval_results_path)
        interpretation_json_path = pathlib.Path(interpretation_json_path)
        if not eval_results_path.exists():
            raise FileNotFoundError(
                f"Eval results JSon file not found: {eval_results_path}"
            )
        if not interpretation_json_path.exists():
            raise FileNotFoundError(
                f"Interpretation JSon file not found: {interpretation_json_path}"
            )

        # datasets
        eval_results = datasets.LlmEvalResults.load_from_json(eval_results_path)
        llm_dataset = eval_results.to_llm_dataset()

        # evaluated models
        evaluated_models = []
        with open(interpretation_json_path) as f:
            interpretation_dict = json.load(f)
        for m in interpretation_dict.get("result", {}).get("models", []):
            if (
                models.ExplainableRagModel.KEY_COLLECTION_ID in m
                or models.ExplainableRagModel.KEY_COLLECTION_NAME in m
                or models.ExplainableRagModel.KEY_DOCUMENTS in m
            ):
                em = models.ExplainableRagModel.from_dict(m)
            else:
                em = models.ExplainableLlmModel.from_dict(m)
            evaluated_models.append(em)
        if not evaluated_models:
            raise ValueError("No evaluated models found in the interpretation JSon.")
        connection_id = evaluated_models[0].connection or str(uuid.uuid4())
        llm_model_names = list(set([m.llm_model_name for m in evaluated_models]))

        # foo connection is OK - it is not used, just it's ID - get it from a model
        host_connection_type = models.ExplainableModelType.to_connection_type(
            evaluated_models[0].model_type
        )
        if not host_connection_type:
            raise ValueError(
                f"Unsupported model type {evaluated_models[0].model_type} for the "
                f"evaluation results."
            )
        host_connection = h2o_sonar_config.ConnectionConfig(
            connection_type=host_connection_type.name,
            name="Imported Test Lab",
            description="Test lab imported from the evaluation results.",
            token="token",
            key=connection_id,
        )
        for em in evaluated_models:
            em.connection = host_connection

        # test lab
        test_lab = RagTestLab(
            llm_host_connection=host_connection,
            raw_dataset=datasets.LlmDataset() if raw_dataset_empty else llm_dataset,
            evaluated_models=evaluated_models,
            llm_model_names=llm_model_names,
        )
        test_lab.dataset = llm_dataset
        test_lab.llm_model_names = llm_model_names or []
        test_lab.evaluated_models = (
            {m.name: m for m in evaluated_models} if evaluated_models else {}
        )

        return test_lab

    @staticmethod
    def load_from_json(
        llm_host_connection: h2o_sonar_config.ConnectionConfig,
        file_path: str | pathlib.Path,
        docs_cache_dir: str | pathlib.Path = "",
        datatable_format: bool = False,
    ) -> "RagTestLab":
        with open(file_path) as f:
            as_dict = json.load(f)

        evaluated_models = []
        for m in as_dict.get(RagTestLab.KEY_MODELS, []):
            if m.get(models.ExplainableRagModel.KEY_MODEL_TYPE) in [
                models.ExplainableModelType.amazon_bedrock_rag.name,
                models.ExplainableModelType.h2ogpte.name,
                models.ExplainableModelType.openai_rag.name,
            ]:
                evaluated_models.append(
                    models.ExplainableRagModel.from_dict(
                        m, connection=llm_host_connection
                    )
                )
            elif m.get(models.ExplainableRagModel.KEY_MODEL_TYPE) in [
                models.ExplainableModelType.anthropic_llm.name,
                models.ExplainableModelType.azure_openai_llm.name,
                models.ExplainableModelType.h2ogpt.name,
                models.ExplainableModelType.h2ogpte_llm.name,
                models.ExplainableModelType.h2ollmops.name,
                models.ExplainableModelType.ollama.name,
                models.ExplainableModelType.openai_llm.name,
            ]:
                evaluated_models.append(
                    models.ExplainableLlmModel.from_dict(
                        m, connection=llm_host_connection
                    )
                )

        raw_dataset = (
            datasets.LlmDataset.from_datatable_dict(
                as_dict.get(RagTestLab.KEY_RAW_DATASET, {})
            )
            if datatable_format
            else datasets.LlmDataset.from_dict(
                as_dict.get(RagTestLab.KEY_RAW_DATASET, {})
            )
        )

        result = RagTestLab(
            llm_host_connection=llm_host_connection,
            raw_dataset=raw_dataset,
            evaluated_models=evaluated_models,
            llm_model_names=as_dict.get(RagTestLab.KEY_BASE_MODEL_NAMES, []),
            docs_cache_dir=docs_cache_dir,
        )
        result.dataset = (
            datasets.LlmDataset.from_datatable_dict(
                as_dict.get(RagTestLab.KEY_DATASET, {})
            )
            if datatable_format
            else datasets.LlmDataset.from_dict(as_dict.get(RagTestLab.KEY_DATASET, {}))
        )
        result.name = as_dict.get(RagTestSuiteConfig.KEY_NAME, "TestLab")
        result.description = as_dict.get(
            RagTestSuiteConfig.KEY_DESCRIPTION, "Test lab for RAG / LLM evaluation."
        )

        return result

    @staticmethod
    def _preprocess_llm_model_name(m: str | None) -> str | None:
        t_hc = genai.H2oGpteRagClient

        if m in (t_hc.MODEL_SPEC_COL_OPT_E, t_hc.MODEL_SPEC_COL_OPT_N):
            return t_hc.MODEL_SPEC_COL
        elif m.lower() == t_hc.MODEL_SPEC_AUTO:
            return t_hc.MODEL_SPEC_AUTO

        return m

    @staticmethod
    def _preprocess_llm_model_names(
        rag_model_type: ExplainableModelTypes,
        llm_model_names: list[str],
    ) -> list[str]:
        """Preprocess the LLM model names to be used in the test lab and handle
        h2oGPTe special cases (``None``, ``""``, and ``"auto"``).

        Parameters
        ----------
        rag_model_type : ExplainableModelTypes
            Type of the explainable model hosted by the RAG system.
        llm_model_names : list[str]
            List of LLM model names to be used to build the test lab and to be
            subsequently evaluated and compared.

        Returns
        -------
        list[str]
            Preprocessed list of LLM model names.

        """
        if not llm_model_names or rag_model_type not in [ExplainableModelTypes.h2ogpte]:
            return llm_model_names

        return [RagTestLab._preprocess_llm_model_name(m) for m in llm_model_names]

    @staticmethod
    def _check_n_fix_model_cfgs(
        llm_models_cfgs: dict[str, list[dict]], llm_model_names: list[str]
    ) -> dict:
        """Precondition check for the model configurations."""
        if llm_models_cfgs:
            result = {
                RagTestLab._preprocess_llm_model_name(k): llm_models_cfgs[k]
                for k in llm_models_cfgs
            }
            for should_be_model_name in result:
                if should_be_model_name not in llm_model_names:
                    raise ValueError(
                        f"Model configuration parameters key '{should_be_model_name}' "
                        f"should be an LLM model name - it must be one of the LLM "
                        f"models in {llm_model_names}, but it is not."
                    )
            llm_models_cfgs = result

        return llm_models_cfgs

    @staticmethod
    def from_llm_test_suite(
        llm_host_connection: h2o_sonar_config.ConnectionConfig,
        llm_test_suite: RagTestSuiteConfig,
        llm_model_type: ExplainableModelTypes,
        llm_model_names: list[str],
        results_location: str | pathlib.Path = "",
        work_dir: str | pathlib.Path = "",
        llm_models_cfgs: dict[str, list[dict]] = None,
        llm_host_prompt_cache: LlmHostPromptCache | None = None,
        user_name: str = commons.DEFAULT_USER,
    ) -> "RagTestLab":
        """Create new (unresolved) test lab from the LLM test suite configuration."""

        def _build_e_model(model_cfg: dict | None = None):
            for test_case in llm_test_suite.test_cases:
                evaluated_model = models.ExplainableLlmModel(
                    connection=llm_host_connection,
                    model_type=llm_model_type,
                    llm_model_name=llm_model_name,
                    model_cfg=model_cfg,
                )
                if evaluated_model.name not in evaluated_models:
                    evaluated_models[evaluated_model.name] = evaluated_model
                else:
                    evaluated_model = evaluated_models[evaluated_model.name]

                raw_dataset.add_input(
                    key=test_case.key,
                    i=test_case.prompt,
                    corpus=None,  # no corpus for LLM as the test target
                    categories=test_case.categories,
                    relationships=test_case.relationships,
                    expected_output=test_case.expected_output,
                    output_constraints=test_case.constraints,
                    output_condition=test_case.condition,
                    model_key=evaluated_model.key,
                    test_key=test_case.config.key,
                )

        # preconditions
        RagTestLab._check_n_fix_model_cfgs(llm_models_cfgs, llm_model_names)

        raw_dataset = datasets.LlmDataset()
        evaluated_models = {}  # model primary key: base model
        for llm_model_name in llm_model_names:
            if llm_models_cfgs:
                if llm_model_name in llm_models_cfgs:
                    for rag_model_cfg in llm_models_cfgs.get(llm_model_name, []):
                        _build_e_model(rag_model_cfg)
                else:
                    _build_e_model()
            else:
                _build_e_model()

        test_lab = RagTestLab(
            llm_host_connection=llm_host_connection,
            raw_dataset=raw_dataset,
            evaluated_models=list(evaluated_models.values()),
            llm_model_names=llm_model_names,
            docs_cache_dir=work_dir,
            results_location=results_location,
            llm_host_prompt_cache=llm_host_prompt_cache,
            user_name=user_name,
        )

        return test_lab

    @staticmethod
    def __from_rag_get_c_id_for_test_case(
        predefined_collection_id: str | dict | None,
        test_case: RagTestCaseConfig,
    ) -> str:
        """Get the collection ID for the test case in case that the RAG test case
        should use predefined (existing) RAG collections.

        Parameters
        ----------
        predefined_collection_id : str | dict | None
            Predefined collection ID for the RAG model. If provided as a string,
            it is used as the collection ID for all the test cases. If provided
            as a dictionary, it is used as a mapping of the test keys to
            the collection IDs.
        test_case : RagTestCaseConfig
            RAG test case configuration.

        Returns
        -------
        str
            Collection ID for the test case.

        """
        if predefined_collection_id:
            if isinstance(predefined_collection_id, str):
                # set the predefined collection ID for all the evaluated models
                return predefined_collection_id

            elif isinstance(predefined_collection_id, dict):
                # predefined collection ID is test key to collection ID mapping,
                # AND every test case refers to its test and evaluated model
                # -> set collection IDs via test case configuration
                test_key = test_case.config.key if test_case.config else None
                if test_key:
                    resolved_collection_id = predefined_collection_id.get(test_key, "")
                    if resolved_collection_id:
                        return resolved_collection_id

                    raise ValueError(
                        f"Test lab is supposed to use predefined collection IDs, "
                        f"but the test {test_key} does not have the mapping to "
                        f"the collection ID in the provided dictionary: "
                        f"{predefined_collection_id}"
                    )

                raise ValueError(
                    f"Test lab is supposed to use predefined collection IDs, "
                    f"but the test case {test_case.key} does not have set the key "
                    f"of the test."
                )

            raise ValueError(
                f"Unsupported predefined collection ID type "
                f"(expected string, dictionary or None), got: "
                f"{type(predefined_collection_id)}"
            )

        return ""

    @staticmethod
    def from_rag_test_suite(
        rag_connection: h2o_sonar_config.ConnectionConfig,
        rag_test_suite: RagTestSuiteConfig,
        rag_model_type: ExplainableModelTypes,
        llm_model_names: list[str],
        docs_cache_dir: str | pathlib.Path = "",
        results_location: str | pathlib.Path = "",
        rag_models_cfgs: dict[str, list[dict]] = None,
        predefined_collection_id: str | dict | None = None,
        predefined_collection_name: str = "",
        llm_host_prompt_cache: LlmHostPromptCache | None = None,
        user_name: str = commons.DEFAULT_USER,
    ) -> "RagTestLab":
        """Create new (unresolved) test lab from the RAG test suite configuration.

        Test lab is build as follows:

        - all LLM model names are hosted by the SAME system
          described by the RAG connection,
          accessed by a client
        - LLM model name may have associated a list of custom client configurations
        - RAG test suite is used to build the test lab:

          - RAG test suite has test cases that are grouped to tests
          - test cases within the test has the SAME corpus,
            different tests may have different corpora

        - explainable RAG model is...

          - created for EACH: LLM model name + config + corpus (not test)
          - carthesian product of:
            LLM model names x client configurations x corpora = explainable RAG models

        Summary of the explainable RAG models creation:

        - for each LLM model name

          - for each client configuration of that LLM model name

            - for each test

              - create explainable RAG model

        Parameters
        ----------
        rag_connection : h2o_sonar_config.ConnectionConfig
            Connection to the RAG system.
        rag_test_suite : RagTestSuiteConfig
            RAG test suite configuration.
        rag_model_type : ExplainableModelTypes
            Type of the explainable model hosted by the RAG system.
        llm_model_names : list[str]
            List of LLM model names to be used to build the test lab and to be
            subsequently evaluated and compared. There are the following special
            names which can be used with h2oGPTe model host:
            - ``auto``: to use the best available model chosen by h2oGPTe
            - ``""``: empty string to inherit configuration from the h2oGPTe collection
            - ``None``: to inherit configuration from the h2oGPTe collection
        rag_models_cfgs : dict[str, list[dict]]
            Dictionary with LLM model name as key and list of client configurations
            as values. Each client configuration is a dictionary with the client
            configuration parameters which can be created by the client factory
            using ``client.config_factory()``.
        docs_cache_dir : str | pathlib.Path
            Directory to store the documents cache.
        results_location : str | pathlib.Path
            Base user directory to store the completion artifacts - like files or
            exported  metadata - which should be subsequently used for the
            evaluation. Consider for example agentic run and metadata, PDF and Python
            files created by the agent. If not specified, test lab will not store any
            evaluation artifacts on completion.
        predefined_collection_id : str | dict | None
            Predefined collection ID for the RAG model. If provided as a string,
            it is used as the collection ID for all the test cases. If provided
            as a dictionary, it is used as a mapping of the test case keys to
            the collection IDs.
        predefined_collection_name : str
            Predefined collection name to be used when creating the RAG model.
            If collection ID is specified, then it is NOT used to look up the existing
            collection on lab build/completion.
        llm_host_prompt_cache : LlmHostPromptCache | None
            Cache for the LLM host client.
        user_name : str
            Username to be used to build the test lab.

        Returns
        -------
        RagTestLab
            New RAG test lab.

        """
        # map: LLM model name -> [client config as dict 1, client config as dict 2, ...]
        rag_models_cfgs = rag_models_cfgs or {}

        def _build_e_model(model_cfg: dict | None = None):
            for test_case in rag_test_suite.test_cases:
                documents = test_case.config.documents if test_case.config else []
                rag_model = models.ExplainableRagModel(
                    connection=rag_connection,
                    model_type=rag_model_type,
                    llm_model_name=llm_model_name,
                    documents=documents,
                    model_cfg=model_cfg,
                    collection_id=RagTestLab.__from_rag_get_c_id_for_test_case(
                        predefined_collection_id, test_case
                    ),
                    collection_name=predefined_collection_name or "",
                )
                # IMPROVE: name is used as unique key - make it more robust
                if rag_model.name not in rag_models:
                    rag_models[rag_model.name] = rag_model
                else:
                    rag_model = rag_models[rag_model.name]

                raw_dataset.add_input(
                    key=test_case.key,
                    i=test_case.prompt,
                    corpus=(
                        test_case.config.documents
                        if (test_case.config and test_case.config.documents)
                        else None
                    ),
                    categories=test_case.categories,
                    relationships=test_case.relationships,
                    expected_output=test_case.expected_output,
                    output_constraints=test_case.constraints,
                    output_condition=test_case.condition,
                    test_key=test_case.config.key,
                    model_key=rag_model.key,
                )

        # preconditions
        llm_model_names = RagTestLab._preprocess_llm_model_names(
            rag_model_type=rag_model_type,
            llm_model_names=llm_model_names,
        )
        rag_models_cfgs = RagTestLab._check_n_fix_model_cfgs(
            llm_models_cfgs=rag_models_cfgs, llm_model_names=llm_model_names
        )
        if isinstance(predefined_collection_id, dict):
            for t in rag_test_suite.tests:
                if t.key not in predefined_collection_id:
                    raise ValueError(
                        f"Error while building RAG test lab with predefined collection "
                        f"IDs provided as a dictionary: every test suite's test must "
                        f"have a predefined collection ID mapping, but the test "
                        f"'{t.key}' is missing in the predefined collection ID "
                        f"(map of test keys to the collection IDs): "
                        f"{predefined_collection_id}."
                    )

        raw_dataset = datasets.LlmDataset()
        rag_models = {}
        for llm_model_name in llm_model_names:
            if rag_models_cfgs:
                if llm_model_name in rag_models_cfgs:
                    for rag_model_cfg in rag_models_cfgs.get(llm_model_name, []):
                        _build_e_model(rag_model_cfg)
                else:
                    _build_e_model()
            else:
                _build_e_model()

        test_lab = RagTestLab(
            llm_host_connection=rag_connection,
            raw_dataset=raw_dataset,
            evaluated_models=list(rag_models.values()),
            llm_model_names=llm_model_names,
            docs_cache_dir=docs_cache_dir,
            results_location=results_location,
            use_evaluated_model_collection_id=bool(predefined_collection_id),
            llm_host_prompt_cache=llm_host_prompt_cache,
            user_name=user_name,
        )

        return test_lab

    def complete_from_shards(
        self,
        execution_dir_path: str | pathlib.Path,
    ):
        """Complete the test lab from the shards stored on the filesystem. This
        method is used to load previously completed test lab shards and merge them
        into a single resolved dataset.

        """
        execution_dir_path = pathlib.Path(execution_dir_path)
        if not execution_dir_path.exists():
            raise ValueError(
                f"Execution directory {execution_dir_path} does not exist."
            )

        shard_paths = []
        json_files = glob.glob(str(execution_dir_path / "*.json"))
        for json_file in json_files:
            f = json_file.replace(".json", "")
            shard_paths.append(pathlib.Path(f) / FILE_RESOLVED_LAB)

        self.dataset = datasets.LlmDataset()
        for shard_path in shard_paths:
            self.logger.info(
                f"Loading the test lab completion parallel job from {shard_path} ..."
            )

            shard_test_lab = RagTestLab.load_from_json(
                llm_host_connection=self.connection, file_path=shard_path
            )
            shard_llm_model_names = shard_test_lab.llm_model_names
            for m in shard_llm_model_names:
                if m not in self.llm_model_names:
                    self.llm_model_names.append(m)

            shard_rag_models = shard_test_lab.evaluated_models.values()
            for m in shard_rag_models:
                if m.name not in self.evaluated_models:
                    self.evaluated_models[m.name] = m

            self.logger.info(
                f"Merging the test lab completion parallel job result from "
                f"{shard_path} ..."
            )
            self.dataset.merge(shard_test_lab.dataset)

        self.logger.info(
            "Completion of the test lab from parallel job results FINISHED."
        )

    def _complete_executor(
        self,
        progress_callback: progress.LoggingProgressCallbackContext,
        complete_context: int = 10,
        save_as_you_go: bool = False,
        parallelize: int = TestLab.PARALLEL_RUN,
        retry_on_error: int = DEFAULT_RETRY_ON_ERROR,
        timeout_exp_backoff: genai.TimeoutRetryExpBackoffCtx | None = None,
        artifacts_base_dir: str | pathlib.Path = "",
        purge_workdir: bool = True,
    ):
        """Method which uses executor for the parallel test lab completion."""
        execution_work_dir = self.docs_cache_dir / f"execution_{time.time()}"
        execution_work_dir.mkdir(parents=True, exist_ok=True)

        parallelize_str = "auto >0 " if (parallelize == TestLab.PARALLEL_RUN) else "0"
        self.logger.info(
            f"PARALLEL completion ({parallelize_str} workers) of the test lab with"
            f" {len(self.raw_dataset.inputs)} inputs in {execution_work_dir} ..."
        )

        # split self lab into multiple labs (shards) to run in parallel
        m_2_shard_paths = self.split_to_shards(base_dir=execution_work_dir)

        # run lab shards completions in parallel
        self.logger.info(
            f"Running {len(m_2_shard_paths)} test lab completion jobs in PARALLEL ..."
        )
        with futures.ProcessPoolExecutor(
            initializer=_test_lab_progress_pool_initializer,
            initargs=(_test_lab_progress_queue,),
            max_workers=None if parallelize == TestLab.PARALLEL_RUN else parallelize,
        ) as executor:
            # progress:
            # - reporting from shards is done via GLOBAL multiprocessing queue
            # - the ONLY portable multiprocessing queue Python pattern:
            #   - queue is passed/initialized by the process executor pool (arg)
            #   - queue is GLOBAL variable i.e. all processes run by the pool share it,
            #     it is also shared with the main process
            #   - per-shard queue CANNOT be used
            # - design:
            #   - each shard reports tuple with:
            #     progress (float) + message (str) + job ID + worker ID (index)
            #   - job ID & worker ID is assigned to the shard worker on its submit()
            progress_job_id = str(uuid.uuid4())
            progress_shard_maxs = [0.0 for _ in range(len(m_2_shard_paths))]
            progress_slot_width = 1.0 / len(m_2_shard_paths)
            progress_max_workder_id = len(m_2_shard_paths)

            task_futures = []
            for e, m in enumerate(m_2_shard_paths):
                task_work_dir = execution_work_dir / f"shard_{e}"
                task_work_dir.mkdir(parents=True, exist_ok=True)
                m_2_shard_paths[m].append(task_work_dir)

                self.logger.info(
                    f"Submitting completion of the test lab parallel job "
                    f"{e + 1}/{len(m_2_shard_paths)} with "
                    f"{len(self.raw_dataset.inputs)} inputs to the executor "
                    f"and working directory {task_work_dir} ..."
                )

                task_future = executor.submit(
                    _rag_test_lab_complete_executor_task,
                    lab_shard_path=m_2_shard_paths[m][1],
                    llm_host_connection=self.connection,
                    task_work_dir=str(task_work_dir),
                    complete_context=complete_context,
                    retry_on_error=retry_on_error,
                    timeout_exp_backoff=timeout_exp_backoff,
                    artifacts_base_dir=(
                        str(artifacts_base_dir) if artifacts_base_dir else ""
                    ),
                    save_as_you_go=save_as_you_go,
                    progress_job_id=progress_job_id,
                    progress_worker_id=e,
                )
                task_futures.append(task_future)

            self.logger.info(
                f"MAIN process is WAITING for {len(task_futures)} test lab completion "
                f"parallel job(s) to finish ..."
            )

            # progress: actively wait for queue messages & update the OVERALL progress
            # - check the global queue for this job msgs to update the overall progress
            # - update max progress of shards
            # - overall progress calculation:
            #   progress_slot = 1.0 / progress_shards
            #   progress_slot_progress = progress_slot * shard_progress
            #   total_progress = sum(progress_slot_progress)
            active_shard_exists = True
            # activate waiting: exponential backoff
            wait_min = 0.1
            wait_backoff_coef = 2
            wait_max = 3.5
            wait_limit = wait_min
            last_overall_progress = -1.0
            while active_shard_exists:
                # read all messages from the global queue
                while True:
                    message = ""
                    try:
                        qmsg = _test_lab_progress_queue.get(timeout=wait_limit)
                        self.logger.debug(
                            f"Received progress message from the completion "
                            f"parallel job: {qmsg}"
                        )
                        # update max progress of shards
                        if qmsg and isinstance(qmsg, tuple) and len(qmsg) == 4:
                            (job_id, worker_id, shard_progress, message) = qmsg
                            if job_id == progress_job_id:
                                if worker_id < progress_max_workder_id:
                                    if shard_progress > progress_shard_maxs[worker_id]:
                                        progress_shard_maxs[worker_id] = float(
                                            shard_progress
                                        )

                        wait_limit = wait_min
                    except queue.Empty:
                        wait_limit = min(wait_limit * wait_backoff_coef, wait_max)
                        break

                    progress_overall = 0.0
                    for p in progress_shard_maxs:
                        progress_overall += progress_slot_width * p
                    # report progress ONLY if it changed
                    if progress_overall != last_overall_progress:
                        p_msg = (
                            f"Lab completion progress executed by "
                            f"{progress_max_workder_id} completion parallel job(s) is: "
                            f"{progress_overall * 100.0:.1f}%{' - ' if message else ''}"
                            f"{message}"
                        )
                        progress_callback.set_progress(
                            progress=progress_overall,
                            message=p_msg,
                        )
                        last_overall_progress = progress_overall

                # check whether all the shards finished
                active_shard_exists = False
                for f in task_futures:
                    if not f.done():
                        active_shard_exists = True
                        break

            # blocking wait for the completion of all the shards (for sure)
            for f in futures.as_completed(task_futures):
                self.logger.info(
                    f"Test lab completion parallel job finished with future: {f}"
                )
                if f.exception():
                    self.logger.error(
                        f"  completion parallel job progress failed with exception "
                        f"provided by the future {f}: {f.exception()}"
                    )
                    raise f.exception()

        # merge the shards as resolved dataset
        self.logger.info(
            "MERGING the test lab completion parallel job(s) results into "
            "a single resolved dataset ..."
        )
        self.dataset = datasets.LlmDataset()
        for s in m_2_shard_paths:
            shard_path = m_2_shard_paths[s][2] / FILE_RESOLVED_LAB
            self.logger.info(
                f"Loading the test lab completion parallel job result from "
                f"{shard_path} ..."
            )
            shard_test_lab = RagTestLab.load_from_json(
                llm_host_connection=self.connection, file_path=shard_path
            )
            self.logger.info(
                f"Merging the test lab completion parallel job result from "
                f"{shard_path} ..."
            )
            self.dataset.merge(shard_test_lab.dataset)
            # merge stats
            for m in shard_test_lab.evaluated_models.values():
                meta = self.evaluated_models[m.name].llm_model_meta
                shard_meta = shard_test_lab.evaluated_models[m.name].llm_model_meta

                meta.setdefault(models.ExplainableLlmModel.KEY_STATS_SUCCESS, 0)
                meta.setdefault(models.ExplainableLlmModel.KEY_STATS_RETRY, 0)
                meta.setdefault(models.ExplainableLlmModel.KEY_STATS_TIMEOUT, 0)
                meta.setdefault(models.ExplainableLlmModel.KEY_STATS_FAILURE, 0)

                meta[models.ExplainableLlmModel.KEY_STATS_SUCCESS] += shard_meta.get(
                    models.ExplainableLlmModel.KEY_STATS_SUCCESS, 0
                )
                meta[models.ExplainableLlmModel.KEY_STATS_RETRY] += shard_meta.get(
                    models.ExplainableLlmModel.KEY_STATS_RETRY, 0
                )
                meta[models.ExplainableLlmModel.KEY_STATS_TIMEOUT] += shard_meta.get(
                    models.ExplainableLlmModel.KEY_STATS_TIMEOUT, 0
                )
                meta[models.ExplainableLlmModel.KEY_STATS_FAILURE] += shard_meta.get(
                    models.ExplainableLlmModel.KEY_STATS_FAILURE, 0
                )

                if (
                    shard_meta
                    and shard_meta.get(models.ExplainableLlmModel.KEY_STATS_DURATION)
                    and isinstance(
                        shard_meta.get(models.ExplainableLlmModel.KEY_STATS_DURATION),
                        dict,
                    )
                ):
                    meta.setdefault(
                        models.ExplainableLlmModel.KEY_STATS_DURATION,
                        {"n": 0, "sum": 0.0, "min": 0.0, "max": 0.0, "avg": 0.0},
                    )
                    duration: dict[str, float] = meta[
                        models.ExplainableLlmModel.KEY_STATS_DURATION
                    ]
                    shard_duration: dict[str, float] = shard_meta[
                        models.ExplainableLlmModel.KEY_STATS_DURATION
                    ]

                    duration["n"] += shard_duration.get("n", 0)
                    duration["sum"] += shard_duration.get("sum", 0.0)
                    duration["min"] = (
                        min(
                            shard_duration.get("min", 0.0),
                            duration["min"],
                        )
                        if duration["min"] > 0.0
                        else shard_duration.get("min", 0.0)
                    )
                    duration["max"] = max(
                        shard_duration.get("max", 0.0),
                        duration["max"],
                    )
                    duration["avg"] = (
                        duration["sum"] / duration["n"] if duration["n"] > 0 else 0.0
                    )

        if purge_workdir:
            # delete recursively execution directory
            self.logger.info(
                f"Purging the execution directory {execution_work_dir} ..."
            )
            shutil.rmtree(execution_work_dir)

        self.logger.info("PARALLEL completion of the test lab FINISHED.")

    def split_to_shards_by_model(self, base_dir: pathlib.Path) -> dict:
        """Split the test lab into shards by RAG model - which is identified by
        corpus and base LLM model name. Shard contains prompts which will be
        subsequently evaluated in the context of the corpus by given base LLM model.

        """
        if not len(self.evaluated_models.values()):
            raise ValueError(
                f"Unable to create jobs to parallelize model completion - "
                f"there are no models to be evaluated in the test lab: "
                f"{self.evaluated_models.keys()}"
            )

        # map: model key -> [RagTestLab, shard JSon path]
        lab_shards = {}
        for e, m in enumerate(self.evaluated_models.values()):
            # filter the raw dataset by the model
            if m.key not in lab_shards:
                lab_shard_path = base_dir / f"shard_{e}.json"
                raw_dataset_shard = datasets.LlmDataset()
                lab_shards[m.key] = [raw_dataset_shard, str(lab_shard_path)]
            else:
                raw_dataset_shard, lab_shard_path = lab_shards[m.key]
            for i in self.raw_dataset.inputs:
                if i.model_key == m.key:
                    raw_dataset_shard.add_input(
                        key=i.key,
                        i=i.i,
                        corpus=i.corpus,
                        categories=i.categories,
                        relationships=i.relationships,
                        expected_output=i.expected_output,
                        output_constraints=i.output_constraints,
                        output_condition=i.output_condition,
                        actual_output=i.actual_output,
                        actual_duration=i.actual_duration,
                        cost=i.cost,
                        model_key=i.model_key,
                    )

            lab_shard = RagTestLab(
                llm_host_connection=self.connection,
                raw_dataset=raw_dataset_shard,
                evaluated_models=[m],
                llm_model_names=[m.llm_model_name],
                docs_cache_dir=self.docs_cache_dir,
            )

            lab_shard.save_as_json(lab_shard_path)

        return lab_shards

    @staticmethod
    def _sharding_get_shard_key(model: str, shard_num: int) -> str:
        """Get key for shard, where ``shard_num`` is local for model."""
        return f"{model[:4]}_{shard_num}"

    @staticmethod
    def _sharding_get_workers_load(inputs: int, worker_count: int) -> list[int]:
        """Returns a list of integers representing the number of inputs for each worker.

        Example for 10 inputs and 3 workers:

        [4, 3, 3]

        """
        workers_load = [0] * worker_count
        for i in range(inputs):
            workers_load[i % worker_count] += 1
        return workers_load

    @staticmethod
    def _sharding_get_indexes_for_load(worker_load: list[int]) -> list[list[int]]:
        """Return a list of integers representing the indexes of inputs for each worker.

        Example for 10 inputs and worker_load = [4, 3, 3]:

        [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]

        """
        indexes = []
        start = 0
        for load in worker_load:
            indexes.append(list(range(start, start + load)))
            start += load
        return indexes

    def split_to_shards(
        self,
        base_dir: pathlib.Path,
        max_total_workers: int = 20,
    ) -> dict:
        """Split the test lab into shards by RAG model (which is identified by
        corpus and base LLM model name). If there is one RAG model (or just a few),
        then even the inputs of particular model are split into shards.
        Shard contains prompts which will be subsequently evaluated in the context
        of the corpus by given base LLM model.

        Sharding strategy:

        - 1 RAG model:
            - split the inputs of the model for max 20 workers (split to 20 shards)
            - the minimum number of inputs per worker is 2 (consider process overhead)
        - >1 RAG model:
            - if the number of models is GREATER than 10,
              split the inputs by the RAG model i.e. the number of needed workers
              is equal to the number of RAG models
            - if the number of models is SMALLER or equal to 10,
              then use up to 20 workers to split the inputs

        Parameters
        ----------
        base_dir : pathlib.Path
            Base directory where to store the shards - JSon representation of test labs.
        max_total_workers : int
            The number of workers which is used to split the inputs of the SINGLE model
            (or lab with just a few models).

        """
        if not len(self.evaluated_models.values()):
            raise ValueError(
                f"Unable to create jobs to parallelize model completion - "
                f"there are no models to be evaluated in the test lab: "
                f"{self.evaluated_models.keys()}"
            )

        workers_per_shard: float = float(max_total_workers) / float(
            len(self.evaluated_models.values())
        )

        # either: MODEL-based SHARDING: if the number of models is HIGH,
        # then split inputs by models w/o sub-model sharding
        if len(self.evaluated_models.values()) > 10 or workers_per_shard < 2.0:
            return self.split_to_shards_by_model(base_dir)

        # or: SUB-MODEL SHARDING

        # prepare number of workers per model
        def _get_per_model_workers(inputs: int) -> int:
            # do not use more than 1 worker if the number of inputs is less than 2
            if float(inputs) / workers_per_shard < 2.0:
                return 1 if inputs == 1 else int(float(inputs) / 2.0)
            return int(workers_per_shard)

        per_model_inputs = self.raw_dataset.stats().get("per_model_inputs", {})
        per_model_workers = {
            m: _get_per_model_workers(inputs) for m, inputs in per_model_inputs.items()
        }

        # split inputs by models - map: model key -> [inputs]
        inputs_by_models: dict[str, list[datasets.LlmDataset.LlmDatasetRow]] = {}
        for i in self.raw_dataset.inputs:
            if i.model_key not in inputs_by_models:
                inputs_by_models[i.model_key] = []
            inputs_by_models[i.model_key].append(i)

        # map: model key -> [RagTestLab, shard JSon path]
        lab_shards = {}
        this = RagTestLab
        for m in self.evaluated_models.values():
            workers_load: list[int] = this._sharding_get_workers_load(
                per_model_inputs.get(m.key, 1), per_model_workers.get(m.key, 1)
            )
            worker_index_load: list[list[int]] = this._sharding_get_indexes_for_load(
                workers_load
            )
            for worker_index, dataset_indexes in enumerate(worker_index_load):
                if this._sharding_get_shard_key(m.key, worker_index) not in lab_shards:
                    n = (
                        f"shard_{this._sharding_get_shard_key(m.key, worker_index)}"
                        f".json"
                    )
                    lab_shard_path = base_dir / n
                    raw_dataset_shard = datasets.LlmDataset()
                    lab_shards[this._sharding_get_shard_key(m.key, worker_index)] = [
                        raw_dataset_shard,
                        str(lab_shard_path),
                    ]
                else:
                    raw_dataset_shard, lab_shard_path = lab_shards[
                        this._sharding_get_shard_key(m.key, worker_index)
                    ]

                # add inputs to the shard
                for i in dataset_indexes:
                    row = inputs_by_models[m.key][i]
                    raw_dataset_shard.add_input(
                        key=row.key,
                        i=row.i,
                        corpus=row.corpus,
                        categories=row.categories,
                        relationships=row.relationships,
                        expected_output=row.expected_output,
                        output_constraints=row.output_constraints,
                        output_condition=row.output_condition,
                        actual_output=row.actual_output,
                        actual_duration=row.actual_duration,
                        cost=row.cost,
                        model_key=row.model_key,
                    )

                lab_shard = RagTestLab(
                    llm_host_connection=self.connection,
                    raw_dataset=raw_dataset_shard,
                    evaluated_models=[m],
                    llm_model_names=[m.llm_model_name],
                    docs_cache_dir=self.docs_cache_dir,
                )

                lab_shard.save_as_json(lab_shard_path)

        return lab_shards


def _rag_test_lab_complete_executor_task(
    lab_shard_path: str,
    llm_host_connection: h2o_sonar_config.ConnectionConfig,
    task_work_dir: str,
    complete_context: int = 10,
    retry_on_error: int = DEFAULT_RETRY_ON_ERROR,
    timeout_exp_backoff: genai.TimeoutRetryExpBackoffCtx | None = None,
    artifacts_base_dir: str = "",
    save_as_you_go: bool = False,
    progress_job_id: str = "",
    progress_worker_id: int = 0,
):
    """Complete the test lab shard as an executor task."""
    try:
        # progress
        progress_bridge = progress.CallbackToQueueBridge(
            progress_queue=_test_lab_progress_queue,
            job_id=progress_job_id,
            worker_id=progress_worker_id,
        )

        # lab shard completion
        test_lab = RagTestLab.load_from_json(
            llm_host_connection=llm_host_connection,
            file_path=lab_shard_path,
            docs_cache_dir=task_work_dir,
        )
        wip_lab_path = pathlib.Path(task_work_dir) / "wip_lab.json"
        test_lab.complete_dataset(
            complete_context=complete_context,
            progress_callback=progress_bridge,
            save_as_you_go=wip_lab_path if save_as_you_go else None,
            parallelize=TestLab.SEQUENTIAL_RUN,
            retry_on_error=retry_on_error,
            timeout_exp_backoff=timeout_exp_backoff,
            artifacts_base_dir=artifacts_base_dir,
            include_llm_meta=False,
            raise_on_all_tcs_fail=False,
        )
        test_lab.save_as_json(pathlib.Path(task_work_dir) / FILE_RESOLVED_LAB)
        if save_as_you_go:
            wip_lab_path.unlink()
    except Exception as ex:
        msg = (
            f"Failed to complete the test lab completion job from {lab_shard_path}: "
            f"{ex}\n{traceback.format_exc()}"
        )
        print(msg)
        raise ex
