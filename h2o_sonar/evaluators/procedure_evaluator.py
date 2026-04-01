# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
"""
This is an implementation preview of the Procedure evaluator which is not ready for
the production use yet.
"""

import functools
import itertools
import json
import random
import re
import time
import traceback

import numpy as np
import pandas as pd

import h2o_sonar.loggers
from h2o_sonar import config as h2o_sonar_config
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import evaluators
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import problems
from h2o_sonar.lib.api import results
from h2o_sonar.lib.integrations import genai
from h2o_sonar.utils import caching
from h2o_sonar.utils import progress as progress_utils
from h2o_sonar.utils import resource_mgmt


try:
    import nltk
    import sentence_transformers
    from sklearn.metrics.pairwise import cosine_similarity

    HAS_REQUIRED_PACKAGES = True
except ImportError:
    HAS_REQUIRED_PACKAGES = False


ROMAN_NUMERALS = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
ROMAN_NUMERALS += [f"x{c}" for c in ROMAN_NUMERALS]
ROMAN_NUMERALS += [f"xx{c}" for c in ROMAN_NUMERALS[:10]]
ROMAN_NUMERALS += [f"xxx{c}" for c in ROMAN_NUMERALS[:10]]
ROMAN_NUMERALS += [f"xl{c}" for c in ROMAN_NUMERALS[:9]] + ["l"]
ROMAN_NUMERALS += [f"l{c}" for c in ROMAN_NUMERALS[:10]]


def join_chunks(chunks):
    result = []
    page_regex = re.compile(rf"^\s*(?:\d+|{'|'.join(ROMAN_NUMERALS)})\s*$")
    for chunk in chunks:
        lines = [
            line
            for line in chunk.split("\n")
            if len(line) > 0 and not page_regex.match(line)
        ]
        result.extend(lines)
    return "\n".join(result)


def argmax(arr):
    indices = list(range(len(arr)))
    return max(indices, key=lambda i: arr[i])


class Retry:
    def __init__(self, retries=10):
        self._retries = retries

    def __call__(self, f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            last_exception = None
            for i in range(self._retries):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    h2o_sonar.loggers.error(
                        f"Retrying {i}...\n{e}\n{traceback.format_exc()}"
                    )
                    last_exception = e
                time.sleep(0.1 * random.randint(0, 2 ** (i + 1) - 1))
            raise last_exception

        return wrapper


@Retry()
def parse_answer(text: str, h2ogpte_client, llm_model_name: str):
    output = h2ogpte_client.ask_model(
        [
            f"Extract all enumerations (nested enumerations should be treated as part "
            f"of the parent enumeration), descriptions of procedures, or step-by-step "
            f"instructions from the following text:\n{text}\n\n"
            f"Format the output as JSON array, for example: "
            '[["first list\'s first item", "first list\'s second item", ...], '
            '["second list\'s first item", "second list\'s second item", ...], ...]. '
            "Ideally, output should be just one coherent list of items. Keep "
            "individual items as small as possible, if an item contains multiple "
            "sub-steps, split it. Output only parseable JSON with no additional text."
        ],
        llm_model_name=llm_model_name,
    )
    answers = json.loads(output[0].answer.strip())
    if len(answers) > 0 and isinstance(answers[0], str):
        # List of strings -> List of lists of strings
        answers = [answers]
    return [tuple()] + [
        tuple(x) if not isinstance(x, str) else tuple([x]) for x in answers
    ]


def strip_llm_json_answer(answer: str) -> str:
    if answer:
        answer = answer.strip()
        if answer.startswith("```json") and answer.endswith("```"):
            answer = answer[8:-3].strip()
        elif answer.startswith("```") and answer.endswith("```"):
            answer = answer[3:-3].strip()
    return answer


@Retry()
def parse_context(contexts, h2ogpte_client, llm_model_name: str):
    text = ("\n" + ("=" * 80) + "\n").join(contexts)

    output = h2ogpte_client.ask_model(
        [
            f"Extract all enumerations (nested enumerations should be treated as part "
            f"of the parent enumeration), descriptions of procedures, or step-by-step "
            f"instructions from the following text chunks. The chunks are "
            f'delimited by 80 "=" and might be in random order. Some enumerations might'
            f" span over multiple chunks. The text chunks:\n{text}\n\nFormat the output"
            f" as JSON array, for example: "
            '[["first list\'s first item", "first list\'s second item", ...], '
            f'["second list\'s first item", "second list\'s second item", ...], ...]. '
            f"Keep individual items as small as possible, if an item contains multiple "
            f"sub-steps, split it. "
            # "If there are two lists that appear to be about the same thing, add them "
            # "together in correct order, keep the original two lists along with the "
            # "new joined one. "
            "Output only parseable JSON with no additional text."
        ],
        llm_model_name=llm_model_name,
        llm_args=dict(max_new_tokens=65536),
    )
    answers = json.loads(strip_llm_json_answer(output[0].answer.strip()))
    if len(answers) > 0 and isinstance(answers[0], str):
        # List of strings -> List of lists of strings
        answers = [answers]

    answers = [tuple(x) if not isinstance(x, str) else tuple([x]) for x in answers]

    # TODO: find some better way how to join enumerations that should be just one
    # TODO: enumeration. This does not work well (causes more issues than it solves) but
    # TODO: we could, e.g., query llms multiple times to find out which pieces should
    # TODO: be together and which shouldn't and at the same time we should also consult
    # TODO: LLM to find out if the question wasn't just about a subset of the
    # TODO: enumeration.
    #
    # combinations = [
    #     list(itertools.combinations(answers, i))
    #     for i in [2, 3]  # range(2, len(answers) + 1)
    # ]
    # print(f"> len(answers)={len(answers)}, len(combinations)={len(combinations)}")
    # answers += [
    #     tuple(itertools.chain.from_iterable(p))
    #     for combination in combinations
    #     for comb in combination
    #     for p in itertools.permutations(comb)
    # ]
    # print(f">> len(answers)={len(answers)}")
    return [tuple()] + answers


def get_parsed_answer_and_ground_truth(
    model, answer, contexts, h2ogpte_client, llm_model_name, logger
):
    parsed_answer = max(
        parse_answer(
            text=answer, h2ogpte_client=h2ogpte_client, llm_model_name=llm_model_name
        ),
        key=len,
    )
    contexts = [join_chunks(perm) for perm in itertools.permutations(contexts, 2)]
    parsed_contexts = parse_context(
        contexts=contexts, h2ogpte_client=h2ogpte_client, llm_model_name=llm_model_name
    )

    if len(parsed_answer) == 0:
        return parsed_answer, parsed_contexts[0]

    parsing_results = set()

    for a in parsed_contexts:
        propositions = []
        for b in parsed_contexts:
            if len(a) < len(b) and set(a).issubset(b):
                propositions.append(b)
        propositions.append(a)
        parsing_results.add(max(propositions, key=len))

    parsed_contexts = list(parsing_results)
    ctxs = [". ".join(pc) for pc in parsed_contexts]
    answer = ". ".join(parsed_answer)

    aemb = model.encode([answer])
    cemb = model.encode(ctxs)
    chunk_idx = argmax(cosine_similarity(aemb, cemb).flatten())

    return parsed_answer, parsed_contexts[chunk_idx]


def fuzzy_compare(a, b):
    punct = " .?!;:"
    if a == b:
        return "Exact Match"
    if a.strip(punct) == b.strip(punct):
        return "Match (after removing punctuation marks)"
    if a.lower() == b.lower():
        return "Case-insensitive match"
    if a.lower().strip(punct) == b.lower().strip(punct):
        return "Case-insensitive match (after removing punctuation marks)"
    a = a.lower().strip(punct).replace(" the ", " ").replace(" a ", " ")
    b = b.lower().strip(punct).replace(" the ", " ").replace(" a ", " ")
    if a == b:
        return "Case-insensitive match after removing articles and punctuation marks"

    edit_dist = nltk.edit_distance(a, b)
    if min(len(a), len(b)) / edit_dist > 3:
        return f"Possible match. Edit distance: {edit_dist}"
    return "Mismatch"


def traceback_through_dyn_prog(
    dp_matrix, similarity_matrix, original_steps, generated_steps, gap_penalty
):
    alignment: list = []
    i, j = len(original_steps), len(generated_steps)
    while i > 0 and j > 0:
        current_score = dp_matrix[i][j]
        diagonal_score = dp_matrix[i - 1][j - 1] + similarity_matrix[i - 1][j - 1]
        vertical_score = dp_matrix[i - 1][j] + gap_penalty

        if current_score == diagonal_score:
            alignment.append(
                (
                    original_steps[i - 1],
                    generated_steps[j - 1],
                    fuzzy_compare(original_steps[i - 1], generated_steps[j - 1]),
                )
            )
            i, j = i - 1, j - 1
        elif current_score == vertical_score:
            alignment.append(
                (original_steps[i - 1], "-", "Gap in Generated (Deletion)")
            )
            i -= 1
        else:
            alignment.append(
                (
                    "-",
                    generated_steps[j - 1],
                    "Possible hallucination (Gap in Original (Insertion))",
                )
            )
            j -= 1

    while i > 0:
        alignment.append((original_steps[i - 1], "-", "Gap in Generated (Deletion)"))
        i -= 1
    while j > 0:
        alignment.append(
            (
                "-",
                generated_steps[j - 1],
                "Possible hallucination (Gap in Original (Insertion))",
            )
        )
        j -= 1

    return alignment[::-1]


class ProcedureEvaluator(evaluators.Evaluator):
    _display_name = "Step alignment and completeness"
    _tagline = (
        "Evaluate the steps in the answer for alignment and completeness given "
        "the retrieved context."
    )

    METRIC_INSERTIONS = "insertions"
    METRIC_DELETIONS = "deletions"
    METRIC_EDITS = "edits"
    METRIC_MISMATCHES = "mismatches"
    METRIC_DETECTED_STEPS_ORIG = "original_detected_steps"
    METRIC_DETECTED_STEPS_GEN = "generated_detected_steps"

    _metrics_meta = commons.MetricsMeta(
        metrics=[
            commons.MetricMeta(
                key=METRIC_EDITS,
                display_name="Edits",
                data_type="int",
                display_format=",d",  # int like 123,456
                description=(
                    "Number of edits required to obtain the correct sequence of steps. "
                    "An edit involves inserting, deleting or substituting a step in "
                    "the actual answer with a step from the retrieved context. Fewer "
                    "edits indicate a better quality actual answer."
                ),
                higher_is_better=False,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=True,
                value_range=(0, float("inf")),
            ),
            commons.MetricMeta(
                key=METRIC_INSERTIONS,
                display_name="Insertions",
                data_type="int",
                display_format=",d",  # int like 123,456
                description=(
                    "Number of insertions to obtain the correct sequence of steps. "
                    "Insertion is a step in the retrieved context that is not present "
                    "in the actual answer. Fewer insertions indicate a better quality "
                    "actual answer."
                ),
                higher_is_better=False,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=False,
                value_range=(0, float("inf")),
            ),
            commons.MetricMeta(
                key=METRIC_DELETIONS,
                display_name="Deletions",
                data_type="int",
                display_format=",d",  # int like 123,456
                description=(
                    "Number of deletions to obtain the correct sequence of steps. "
                    "Deletion is a step in the actual answer that is not present in "
                    "the retrieved context. Fewer deletions indicate a better quality "
                    "actual answer."
                ),
                higher_is_better=False,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=False,
                value_range=(0, float("inf")),
            ),
            commons.MetricMeta(
                key=METRIC_MISMATCHES,
                display_name="Mismatches",
                data_type="int",
                display_format=",d",  # int like 123,456
                description=(
                    "Number of steps that are not the same in the original and "
                    "generated output. Fewer mismatches indicate a better quality "
                    "actual answer."
                ),
                higher_is_better=False,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=False,
                value_range=(0, float("inf")),
            ),
            commons.MetricMeta(
                key=METRIC_DETECTED_STEPS_ORIG,
                display_name="Retrieved context steps",
                data_type="int",
                display_format=",d",  # int like 123,456
                description="The number of steps detected in the retrieved context.",
                higher_is_better=True,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=False,
                value_range=(0, float("inf")),
            ),
            commons.MetricMeta(
                key=METRIC_DETECTED_STEPS_GEN,
                display_name="Actual answer steps",
                data_type="int",
                display_format=",d",  # int like 123,456
                description="The number of steps detected in the actual answer.",
                higher_is_better=True,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=False,
                value_range=(0, float("inf")),
            ),
        ]
    )

    # COMPATIBILITY: LLM/RAG evaluation
    _rag = True

    # GLOBAL: metric value for all dataset rows
    _global_explanation = True
    # LOCAL: metric value for particular row
    _local_explanation = True
    _explanation_types = [
        e10s.LlmEvalResultsExplanation,
        e10s.LlmHeatmapLeaderboardExplanation,
        e10s.WorkDirArchiveExplanation,
    ]

    PARAM_H2OGPTE_HOST_CFG_KEY = "h2ogpte_connection_config_key"
    DEFAULT_PARAM_H2OGPTE_HOST_CFG_KEY = ""
    _PARAM_H2OGPTE_HOST_CFG_KEY = evaluators.EvaluatorParam(
        param_name=PARAM_H2OGPTE_HOST_CFG_KEY,
        description=(
            "Configuration key of the h2oGPTe host to be used for the "
            "evaluation. If not specified, the first h2oGPTe connection in the "
            "configuration will be used."
        ),
        param_type=commons.EvaluatorParamType.str,
        default_value=DEFAULT_PARAM_H2OGPTE_HOST_CFG_KEY,
        src=evaluators.EvaluatorParam.SRC_EVALUATOR_PARAMS,
    )
    PARAM_LLM_MODEL_NAME = "h2ogpte_llm_model_name"
    DEFAULT_LLM_MODEL_NAME = ""
    _PARAM_LLM_MODEL_NAME = evaluators.EvaluatorParam(
        param_name=PARAM_LLM_MODEL_NAME,
        description=(
            "LLM model (name) to be used for the evaluation. If not specified, "
            "evaluator will check whether h2oGPTe host provides Claude Sonnet, "
            "OpenAI GPT-4o or any llama (in this order) and use it."
        ),
        param_type=commons.EvaluatorParamType.str,
        default_value=DEFAULT_LLM_MODEL_NAME,
        src=evaluators.EvaluatorParam.SRC_EVALUATOR_PARAMS,
    )

    _parameters = [
        _PARAM_H2OGPTE_HOST_CFG_KEY,
        _PARAM_LLM_MODEL_NAME,
        evaluators.Evaluator._get_custom_param_metric_threshold(
            _metrics_meta.get_primary_metric()
        ),
        evaluators.Evaluator._PARAM_SAVE_LLM_RESULT,
        evaluators.Evaluator._PARAM_SENTENCE_LEVEL_METRICS,
        evaluators.Evaluator._get_custom_param_min_test_case(),
    ]

    _keywords = [
        evaluators.KEYWORD_GPU_OPT,
        evaluators.KEYWORD_LLM,
        evaluators.KEYWORD_EVALUATES_RAG,
        evaluators.KEYWORD_RQ_RC,
        evaluators.KEYWORD_RQ_AA,
        evaluators.KEYWORD_SR_11_7_CS,
        evaluators.KEYWORD_SR_11_7_OGA,
        evaluators.KEYWORD_SR_11_7_OA,
        evaluators.KEYWORD_NIST_AI_RMF_S,
        evaluators.KEYWORD_NIST_AI_RMF_AT,
        evaluators.KEYWORD_NIST_AI_RMF_VR,
        evaluators.KEYWORD_PROBLEM_TYPE_IR,
        evaluators.KEYWORD_PROBLEM_TYPE_QA,
        evaluators.KEYWORD_ES_GENERATE,
        evaluators.KEYWORD_METHOD_AGENTS,
    ]

    _modules_needed_by_name = [
        h2o_sonar_config.DEP_SENTENCE_TRANSFORMERS,
        h2o_sonar_config.DEP_NLTK,
    ]

    # models used by the evaluator
    _e_model_minilm = caching.MODEL_SENTENCE_TRANSFORMERS_ALL_MINILM_L6_V2

    _brief_description = """Step alignment and completeness evaluator is a tool for
evaluating the steps of procedures, sequences, or process descriptions in the actual
answer for relevance, alignment and completeness, given the retrieved context as
a ground truth.

- The evaluator uses LLM and/or regular expressions to extract steps, sentence
  embeddings to assess semantic similarity between steps, and dynamic programming to
  compare the steps in the actual answer with the retrieved context to assess
  alignment and completeness.
- The implementation is based on 'Evaluating Procedure Generation in Retrieval-Augmented
  Generation (RAG) Systems' by Alexis Sudjianto and Agus Sudjianto; and
  'Evaluating Procedural Alignment and Sequence Detection' by Agus Sudjianto.
- Compatibility: RAG evaluation only."""
    _description = evaluators.Evaluator._description_builder(
        brief=f"""{_brief_description}

**Method**:

- The evaluator uses the configured LLM and/or regular expressions to extract all
  enumerations from the retrieved context chunks and actual answers.
- The evaluator semantically compares the extracted steps and evaluates the alignment
  and completeness of the steps in the actual answer using dynamic programming,
  considering the retrieved context as ground truth.
- In order to measure the semantic similarity between steps the evaluator uses
  [{_e_model_minilm}](https://huggingface.co/{_e_model_minilm})
  embedding model from Hugging Face [sentence-transformers](https://www.sbert.net/)
  library.
- The evaluator provides metrics for the number of edits (primary), insertions,
  deletions, and mismatches in the actual answer.
- In addition the evaluator provides metrics with the number of steps detected
  in the retrieved context and the actual answer to assess the reliability of
  the evaluation.
- The evaluator is compatible with RAG models, as it requires retrieved context.
""",
        metrics_meta=_metrics_meta,
        keywords=_keywords,
        parameters=_parameters,
        extra_insights="",
        leaderboard_type=e10s.LlmHeatmapLeaderboardExplanation.explanation_type(),
    )

    def __init__(self):
        evaluators.Evaluator.__init__(self)

        # h2oGPTe connection whose LLM model will be used for the evaluation
        self._h2ogpte_connection_key = ""
        self._h2ogpte_connection = None
        self._h2ogpte_client = None
        self._llm_model_name = ""

        self.args = None
        self.problems = []
        self.log_name = f"{ProcedureEvaluator._display_name} evaluator"

    @property
    def h2ogpte_connection_key(self):
        if not self._h2ogpte_connection_key:
            available_connections = []
            # if the connection is already set, use its key
            if self._h2ogpte_connection:
                self._h2ogpte_connection_key = self._h2ogpte_connection.key
                return self._h2ogpte_connection_key

            # if the key is configured, verify it
            self._h2ogpte_connection_key = self.args.get(
                ProcedureEvaluator.PARAM_H2OGPTE_HOST_CFG_KEY,
                ProcedureEvaluator.DEFAULT_PARAM_H2OGPTE_HOST_CFG_KEY,
            )

            # find h2oGPTe connection in the H2O Sonar configuration
            for c in self.config.connections:
                available_connections.append(
                    f"{c.key} ({c.connection_type}): {c.name} ({c.server_url})"
                )
                if c.key == self._h2ogpte_connection_key:
                    if (
                        c.connection_type
                        != h2o_sonar_config.ConnectionConfigType.H2O_GPT_E.name
                    ):
                        raise ValueError(
                            f"h2oGPTe host connection configured with key "
                            f"'{self.h2ogpte_connection_key}' in "
                            f"{ProcedureEvaluator._display_name} evaluator "
                            f"must be h2oGPTe connection, but it is not: "
                            f"{c.connection_type}"
                        )

                    self._h2ogpte_connection = c
                    return self._h2ogpte_connection_key

            # FALLBACK: if connection not found, use the first h2oGPTe connection
            if not self._h2ogpte_connection:
                for c in self.config.connections:
                    if (
                        c.connection_type
                        == h2o_sonar_config.ConnectionConfigType.H2O_GPT_E.name
                    ):
                        self._h2ogpte_connection_key = c.key
                        self._h2ogpte_connection = c
                        return self._h2ogpte_connection_key

            if not self._h2ogpte_connection_key:
                raise ValueError(
                    f"Required h2oGPTe connection key was not specified as the "
                    f"evaluator {self._display_name} parameter - unable to create "
                    f"the client and perform health check. Available connections: "
                    f"{', '.join(available_connections)}"
                )

        return self._h2ogpte_connection_key

    @property
    def h2ogpte_connection(self):
        if self._h2ogpte_connection is None:
            h2ogpte_connection_key = self.h2ogpte_connection_key

            # find h2oGPTe connection in the H2O Sonar configuration
            if h2ogpte_connection_key:
                for c in self.config.connections:
                    if c.key == h2ogpte_connection_key:
                        if (
                            c.connection_type
                            != h2o_sonar_config.ConnectionConfigType.H2O_GPT_E.name
                        ):
                            raise ValueError(
                                f"h2oGPTe host connection configured with key "
                                f"'{self.h2ogpte_connection_key}' in "
                                f"{ProcedureEvaluator._display_name} evaluator "
                                f"must be h2oGPTe connection, but it is not: "
                                f"{c.connection_type}"
                            )

                        self._h2ogpte_connection = c
                        break

            # if not desired connection found, use the first h2oGPTe connection
            if not self._h2ogpte_connection:
                for c in self.config.connections:
                    if (
                        c.connection_type
                        == h2o_sonar_config.ConnectionConfigType.H2O_GPT_E.name
                    ):
                        self._h2ogpte_connection_key = c.key
                        self._h2ogpte_connection = c
                        break

            # if no h2oGPTe connection found, raise an error
            if not self._h2ogpte_connection:
                raise ValueError(
                    f"No h2oGPTe connection found in the configuration - unable to "
                    f"use h2oGPTe as LLM host: configured connection key="
                    f"'{self.h2ogpte_connection_key}' (config also does not contain "
                    f"any h2oGPTe connection)"
                )

        return self._h2ogpte_connection

    @property
    def h2ogpte_client(self):
        # run custom health check
        if not self._h2ogpte_client:
            self._h2ogpte_client = genai.H2oGpteRagClient(
                connection=self.h2ogpte_connection, logger=self.logger
            )

        return self._h2ogpte_client

    @property
    def llm_model_name(self):
        if not self._llm_model_name:
            if not self.args:
                raise ValueError("Evaluator arguments are not initialized")

            param_llm_model_name = self.args.get(
                ProcedureEvaluator.PARAM_LLM_MODEL_NAME, ""
            )

            if param_llm_model_name:
                self._llm_model_name = param_llm_model_name
            else:
                # find the LLM model name to be used by the evaluator
                favorite_llm_model = "claude-3-7-sonnet-20250219"
                llm_model_names = self.h2ogpte_client.list_llm_model_names()
                if not llm_model_names:
                    raise ValueError(
                        f"No LLM models found on the h2oGPTe host "
                        f"{self.h2ogpte_connection.name}"
                    )
                if favorite_llm_model in llm_model_names:
                    # if Sonnet is available, use it
                    self._llm_model_name = favorite_llm_model
                    return self._llm_model_name
                llm_model_claude = ""
                llm_model_4o = ""
                llm_model_llama = ""
                for llm_model_name in llm_model_names:
                    if "claude" in llm_model_name.lower():
                        llm_model_claude = llm_model_name
                        if "sonnet" in llm_model_name.lower():
                            self._llm_model_name = llm_model_name
                            return self._llm_model_name
                    elif "4o" in llm_model_name.lower():
                        if llm_model_4o:
                            if "mini" in llm_model_4o.lower():
                                self._llm_model_name = llm_model_name
                        else:
                            llm_model_4o = llm_model_name
                    elif "llama" in llm_model_name.lower():
                        if llm_model_llama:
                            if "405" in llm_model_llama.lower():
                                self._llm_model_name = llm_model_name
                        else:
                            llm_model_llama = llm_model_name

                if llm_model_4o:
                    self._llm_model_name = llm_model_4o
                elif llm_model_claude:
                    self._llm_model_name = llm_model_claude
                elif llm_model_llama:
                    self._llm_model_name = llm_model_llama
                else:
                    self._llm_model_name = llm_model_names[0]

        return self._llm_model_name

    def h2ogpte_health_check(self) -> bool:
        """Perform health check of the h2oGPTe client and LLM to be used.

        Returns
        -------
        bool
            True if the health check passed, False otherwise.

        """
        try:
            return self.h2ogpte_client.health_check(self.llm_model_name)
        except Exception as ex:
            raise ValueError(
                f"h2oGPTe client '{self.h2ogpte_connection.name}' and LLM model "
                f"'{self.llm_model_name}' health check failed: {ex}\n{traceback}"
            )

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        **evaluator_params,
    ) -> bool:
        evaluators.Evaluator.check_compatibility(self, params, **evaluator_params)

        if not HAS_REQUIRED_PACKAGES:
            self.logger.error(
                self._check_compatibility_pckg_err_msg(
                    ["nltk", "scikit-learn", "sentence-transformers"]
                )
            )
            return False

        if not evaluators.Evaluator._check_llm_dataset_compatibility(
            self,
            params=params,
            evaluator_keywords=self.keywords,
            check_empty_contexts=evaluators.KEYWORD_RQ_RC in self.keywords,
            fail_on_all_empty_contexts=evaluators.KEYWORD_RQ_RC in self.keywords,
        ):
            return False

        try:
            self._resolve_evaluator_params()
            self.h2ogpte_health_check()
            self.logger.info(
                f"{self.log_name}: evaluator will use LLM model: "
                f"'{self.llm_model_name}'"
            )
        except Exception as ex:
            self.logger.warning(
                f"{self.log_name}: Unable to resolve LLM model name for the "
                f"LLM-based evaluator given h2oGPTe connection key "
                f"'{self._h2ogpte_connection_key}' and LLM model name evaluator "
                f"parameter: {ex}\n{traceback.format_exc()}"
            )
            return False

        return True

    def setup(self, model, persistence, **kwargs):
        evaluators.Evaluator.setup(self, model, persistence, **kwargs)

        self._resolve_evaluator_params()
        caching.cache_nltk_punkt(self.logger)

    def evaluate(self, llm_testset, explanations_types=None, **kwargs) -> list:
        save_llm_result = self.args.get(
            evaluators.Evaluator.PARAM_SAVE_LLM_RESULT,
            evaluators.Evaluator.DEFAULT_SAVE_LLM_RESULT,
        )

        return self._evaluate(
            llm_testset=llm_testset,
            save_llm_result=save_llm_result,
        )

    def _evaluate(
        self,
        llm_testset,
        save_llm_result: bool,
    ) -> list:
        #
        # EVALUATION
        #

        key_2_evaluated_model = {m.key: m for m in self.models}

        # LLM host: RAG or service
        llm_host = (
            commons.LlmModelHostType.RAG
            if isinstance(
                next(iter(key_2_evaluated_model.values())), models.ExplainableRagModel
            )
            else commons.LlmModelHostType.SERVICE
        )

        eval_results = self._calculate_metrics(llm_testset=llm_testset)

        #
        # NORMALIZATION of the evaluation results to the common EXPLANATIONS/FORMAT(s)
        #

        # EXPLANATIONS
        explanations = []

        # EXPLANATION: all data (per prompt metrics)
        if save_llm_result:
            eval_results_explanation = e10s.LlmEvalResultsExplanation(
                evaluator=self,
                display_name="Evaluation metrics data",
                display_category=e10s.Explanation.DISPLAY_CAT_LLM,
                eval_results=eval_results,
            )
            # FORMATS of the explanation: JSon, CSV, DataTable
            eval_results_explanation.add_json_format()
            eval_results_explanation.add_csv_format()
            eval_results_explanation.add_datatable_format()
            explanations.append(eval_results_explanation)

        # THRESHOLD for the metric
        metrics_threshold = self.args.get(
            evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
            evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
        )
        self._metrics_meta.set_threshold(metrics_threshold)

        # EXPLANATION: heatmap leaderboard
        procedure_explanation = (
            e10s.LlmProcedureEvalLeaderboardExplanation.from_eval_results(
                evaluator=self,
                eval_results=eval_results,
                metrics_meta=self._metrics_meta,
                key_2_evaluated_model=key_2_evaluated_model,
                llm_host=llm_host,
                display_name="LLM heatmap leaderboard",
                display_category=e10s.GlobalSummaryFeatImpExplanation.DISPLAY_CAT_LLM,
                logger=self.logger,
            )
        )
        procedure_explanation.add_json_format(
            threshold=self.args.get(
                evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
                self._metrics_meta.get_primary_metric().threshold,
            )
        )
        procedure_explanation.add_markdown_format(sort_by_metric_id=self.METRIC_EDITS)
        procedure_explanation.add_evalstudio_markdown_format(
            sort_by_metric_id=self.METRIC_EDITS
        )
        explanations.append(procedure_explanation)

        # PROBLEMS for alerts and actionability
        self._diagnose_problems(
            eval_results=eval_results,
            key_2_evaluated_model=key_2_evaluated_model,
            leaderboard_explanation=procedure_explanation,
        )

        # INSIGHTS
        self._diagnose_insights(leaderboard_explanation=procedure_explanation)

        # EXPLANATION: HTML fragment
        if self.config and self.config.create_html_representations:
            try:
                html_explanation = e10s.GlobalHtmlFragmentExplanation(
                    evaluator=self,
                    display_name="LLM heatmap leaderboard as HTML",
                    display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
                )
                html_explanation.add_html_format(
                    str(
                        procedure_explanation.as_html(
                            sort_by_metric_id=self.METRIC_EDITS
                        )
                    ),
                )
                explanations.append(html_explanation)
            except Exception as ex:
                self.logger.warning(
                    f"{self.log_name}: HTML fragment explanation creation failed: "
                    f"{ex}\n{traceback.format_exc()}"
                )

        # EXPLANATION: ZIP archive of all artifacts created by the evaluator
        explanations.append(
            self.create_explanation_workdir_archive(
                display_name=f"Archive of {self._display_name} artifacts",
                display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
            )
        )

        return explanations

    def _calculate_metrics(self, llm_testset) -> datasets.LlmEvalResults:
        llm_dataset = datasets.LlmDataset.from_datatable_dict(llm_testset.to_dict())

        # evaluator runs only ONE metric at a time
        self.report_progress(0.01, "Configuring metrics...")

        eval_results = datasets.LlmEvalResults()

        device = h2o_sonar_config.config.resolve_gpu_cpu_device(result_format="str")
        with resource_mgmt.PytorchModelLifeCycleManager(
            sentence_transformers.SentenceTransformer(
                ProcedureEvaluator._e_model_minilm,
                device=device,
                revision=caching.REVISIONS_FOR_MODEL[
                    ProcedureEvaluator._e_model_minilm
                ],
            )
        ) as embedding_model:
            # for every test case run metric (row by row)
            for e, r in enumerate(llm_dataset.inputs):
                # progress
                self.report_progress(
                    progress=progress_utils.ProgressCallbackContext.progress_for_steps(
                        e + 1, len(llm_dataset.inputs)
                    ),
                    message=evaluators.Evaluator._eval_row_progress_msg(
                        metric_name=self.METRIC_EDITS,
                        device=device,
                        row=e + 1,
                        total_rows=len(llm_dataset.inputs),
                    ),
                )

                # handle actual answer retrieval error ~ RAG/LLM client crash
                if evaluators.Evaluator._is_internal_err_answer(r.actual_output):
                    # set WORST metrics values
                    eval_results.add_result(
                        datasets.LlmEvalResults.LlmEvalResultRow(
                            dataset_row=r,
                            metrics={
                                self.METRIC_EDITS: (
                                    commons.MetricMeta.EFFECTIVE_INF_FLOAT_INT
                                ),
                                self.METRIC_INSERTIONS: (
                                    commons.MetricMeta.EFFECTIVE_INF_FLOAT_INT
                                ),
                                self.METRIC_DELETIONS: (
                                    commons.MetricMeta.EFFECTIVE_INF_FLOAT_INT
                                ),
                                self.METRIC_MISMATCHES: (
                                    commons.MetricMeta.EFFECTIVE_INF_FLOAT_INT
                                ),
                                self.METRIC_DETECTED_STEPS_ORIG: (
                                    commons.MetricMeta.EFFECTIVE_INF_FLOAT_INT
                                ),
                                self.METRIC_DETECTED_STEPS_GEN: (
                                    commons.MetricMeta.EFFECTIVE_INF_FLOAT_INT
                                ),
                            },
                            # NO metrics_meta
                        )
                    )
                    continue

                try:
                    generated_steps, original_steps = (
                        get_parsed_answer_and_ground_truth(
                            model=embedding_model,
                            answer=r.actual_output,
                            contexts=(
                                r.context if self.is_rag() else [r.expected_output]
                            ),  # TODO check for RAG, this does not work
                            h2ogpte_client=self.h2ogpte_client,
                            llm_model_name=self.llm_model_name,
                            logger=self.logger,
                        )
                    )
                    new_orig_steps, new_gen_steps = self._try_to_combine_steps(
                        embedding_model, original_steps, generated_steps
                    )
                    (
                        edits,
                        inserts,
                        deletions,
                        mismatches,
                        orig_steps,
                        gen_steps,
                        dyn_prog_matrix,
                        alignment_matrix,
                    ) = self._calculate_dyn_prog(
                        embedding_model, new_orig_steps, new_gen_steps
                    )
                    t_proc_ldb_expl = e10s.LlmProcedureEvalLeaderboardExplanation
                    eval_results.add_result(
                        datasets.LlmEvalResults.LlmEvalResultRow(
                            dataset_row=r,
                            metrics={
                                self.METRIC_EDITS: inserts + deletions,
                                self.METRIC_INSERTIONS: inserts,
                                self.METRIC_DELETIONS: deletions,
                                self.METRIC_MISMATCHES: mismatches,
                                self.METRIC_DETECTED_STEPS_ORIG: orig_steps,
                                self.METRIC_DETECTED_STEPS_GEN: gen_steps,
                            },
                            metrics_meta={
                                t_proc_ldb_expl.KEY_DYN_PROG_MATRIX: dyn_prog_matrix,
                                t_proc_ldb_expl.KEY_ALIGNMENT_MATRIX: alignment_matrix,
                            },
                        )
                    )
                except Exception as ex:
                    self.logger.error(
                        f"{self.log_name}: Evaluation failed for row "
                        f"{e + 1}/{len(llm_dataset.inputs)}: {ex}\n"
                        f"{traceback.format_exc()}"
                    )

        return eval_results

    _SEMANTIC_SCORE_THRESHOLD = 1.1

    @staticmethod
    def _uncapitalize(s):
        return s[0].lower() + s[1:]

    @staticmethod
    def _joined_items(itm1, itm2):
        if itm1[-1] not in [":", ";", "!", "."]:
            results = [f"{itm1}. {itm2}"]
        else:
            results = [f"{itm1} {itm2}"]

        return results + [
            f"{itm1}{joiner}{ProcedureEvaluator._uncapitalize(itm2)}"
            for joiner in [", ", ", and ", " then ", ", then "]
        ]

    def _semantic_score(self, embedding_model, itm1, itm2):
        enc_itm1 = embedding_model.encode(itm1)
        enc_itm2 = embedding_model.encode(itm2)
        return float(cosine_similarity([enc_itm1], [enc_itm2]))

    def _semantic_score_increase(self, embedding_model, itm1, pre1, pre2, itm2):
        return self._semantic_score(embedding_model, itm1, itm2) / max(
            self._semantic_score(embedding_model, itm1, pre1),
            self._semantic_score(embedding_model, itm1, pre2),
        )

    def _gen_proposals(self, embedding_model, am, col1, col2):
        if not (am[col1] == "-").any():
            return []

        missing = am[col1] == "-"
        missing_idx = list(missing.index[missing])

        proposals = []
        for midx in missing_idx:
            if midx > 0:
                if not any(
                    [
                        am.loc[midx - 1, col1] == "-",
                        am.loc[midx - 1, col2] == "-",
                        am.loc[midx, col2] == "-",
                        len(am.loc[midx - 1, col1])
                        < 0.75 * len(am.loc[midx - 1, col2] + am.loc[midx, col2]),
                    ]
                ):
                    os = am.loc[midx - 1, col1]
                    for gs in self._joined_items(
                        am.loc[midx - 1, col2], am.loc[midx, col2]
                    ):
                        sem_score = self._semantic_score_increase(
                            embedding_model,
                            os,
                            am.loc[midx - 1, col2],
                            am.loc[midx, col2],
                            gs,
                        )
                        if sem_score < self._SEMANTIC_SCORE_THRESHOLD:
                            continue

                        proposals.append(
                            {
                                col1: list(am[col1].values),
                                col2: list(
                                    am.loc[: midx - 2, col2].values
                                    if midx - 2 >= 0
                                    else []
                                )
                                + [gs]
                                + list(am.loc[midx + 1 :, col2].values),
                            }
                        )
            if midx + 1 < am.shape[0]:
                if not any(
                    [
                        am.loc[midx + 1, col1] == "-",
                        am.loc[midx, col2] == "-",
                        am.loc[midx + 1, col2] == "-",
                        len(am.loc[midx + 1, col1])
                        < 0.75 * len(am.loc[midx, col2] + am.loc[midx + 1, col2]),
                    ]
                ):
                    os = am.loc[midx + 1, col1]
                    for gs in self._joined_items(
                        am.loc[midx, col2], am.loc[midx + 1, col2]
                    ):
                        sem_score = self._semantic_score_increase(
                            embedding_model,
                            os,
                            am.loc[midx, col2],
                            am.loc[midx + 1, col2],
                            gs,
                        )
                        if sem_score < self._SEMANTIC_SCORE_THRESHOLD:
                            continue

                        proposals.append(
                            {
                                col1: list(am[col1].values),
                                col2: list(am.loc[: midx - 1, col2].values)
                                + [gs]
                                + list(am.loc[midx + 2 :, col2].values),
                            }
                        )

        return proposals

    def _try_to_combine_steps(self, embedding_model, orig_steps, gen_steps):
        col1 = "Original Step"
        col2 = "Generated Step"
        improvement = True
        gap_penalty = 0.5
        res = self._calculate_dyn_prog(embedding_model, orig_steps, gen_steps)
        if res[-2] is None:
            return orig_steps, gen_steps
        max_score = max(res[-2]["data"][-1])
        best_am = pd.DataFrame(res[-1]["data"], columns=res[-1]["col_names"])
        while improvement:
            improvement = False
            proposals = self._gen_proposals(embedding_model, best_am, col1, col2)
            proposals.extend(self._gen_proposals(embedding_model, best_am, col2, col1))
            max_score_prev = max_score
            for i, dct in enumerate(proposals):
                orig_steps = [s for s in dct["Original Step"] if s != "-"]
                gen_steps = [s for s in dct["Generated Step"] if s != "-"]
                res = self._calculate_dyn_prog(embedding_model, orig_steps, gen_steps)
                score = max(res[-2]["data"][-1])
                if score > max_score_prev + gap_penalty and score > max_score:
                    max_score = score
                    best_am = pd.DataFrame(
                        res[-1]["data"], columns=res[-1]["col_names"]
                    )
                    improvement = True
        orig_steps = [s for s in best_am["Original Step"].values if s != "-"]
        gen_steps = [s for s in best_am["Generated Step"].values if s != "-"]
        return orig_steps, gen_steps

    def _calculate_dyn_prog(self, embedding_model, original_steps, generated_steps):
        if len(generated_steps) == 0:
            deletion = len(original_steps)
            dyn_prog_matrix = None
            alignment_matrix = None
            return (
                int(deletion),
                0,
                int(deletion),
                0,
                len(original_steps),
                len(generated_steps),
                dyn_prog_matrix,
                alignment_matrix,
            )

        if len(original_steps) == 0:
            insertion = len(generated_steps)
            dyn_prog_matrix = None
            alignment_matrix = None
            return (
                int(insertion),
                int(insertion),
                0,
                0,
                len(original_steps),
                len(generated_steps),
                dyn_prog_matrix,
                alignment_matrix,
            )

        # Step 3: Compute Sentence Embeddings
        original_embeddings = embedding_model.encode(original_steps)
        generated_embeddings = embedding_model.encode(generated_steps)

        # Step 4: Create a Similarity Matrix Using Cosine Similarity
        similarity_matrix = cosine_similarity(original_embeddings, generated_embeddings)

        # Step 5: Initialize the Dynamic Programming (DP) Matrix
        gap_penalty = -0.5
        rows, cols = len(original_steps) + 1, len(generated_steps) + 1
        dp_matrix = np.zeros((rows, cols))

        # Apply gap penalties for the first row and column
        for i in range(rows):
            dp_matrix[i][0] = i * gap_penalty
        for j in range(cols):
            dp_matrix[0][j] = j * gap_penalty

        # Step 6: Fill the DP Matrix Using the Similarity Matrix
        for i in range(1, rows):
            for j in range(1, cols):
                match = dp_matrix[i - 1][j - 1] + similarity_matrix[i - 1][j - 1]
                delete = dp_matrix[i - 1][j] + gap_penalty
                insert = dp_matrix[i][j - 1] + gap_penalty
                dp_matrix[i][j] = max(match, delete, insert)

        # Step 7: Traceback to Determine the Best Alignment
        alignment_result = traceback_through_dyn_prog(
            dp_matrix, similarity_matrix, original_steps, generated_steps, gap_penalty
        )

        # Step 8: Convert the Alignment Result into a DataFrame
        alignment_df = pd.DataFrame(
            alignment_result, columns=["Original Step", "Generated Step", "Action"]
        )
        deletion = alignment_df.Action.str.contains("Deletion").sum()
        insertion = alignment_df.Action.str.contains("Insertion").sum()
        mismatches = alignment_df.Action.str.contains("Mismatch").sum()
        if int(insertion) >= -1:
            self.logger.info("-" * 80)
            self.logger.info(f"original_steps={original_steps}")
            self.logger.info(f"generated_steps={generated_steps}")
            self.logger.info(alignment_df)
            self.logger.info("-" * 80)

        dyn_prog_matrix = {
            "row_names": ["∅"] + list(original_steps),
            "col_names": ["∅"] + list(generated_steps),
            "data": dp_matrix.tolist(),
        }

        alignment_matrix = {
            "row_names": alignment_df.index.tolist(),
            "col_names": alignment_df.columns.tolist(),
            "data": alignment_df.values.tolist(),
        }

        return (
            int(deletion + insertion),
            int(insertion),
            int(deletion),
            int(mismatches),
            len(original_steps),
            len(generated_steps),
            dyn_prog_matrix,
            alignment_matrix,
        )

    def _diagnose_problems(
        self,
        eval_results: datasets.LlmEvalResults,
        key_2_evaluated_model: dict,
        leaderboard_explanation: e10s.LlmProcedureEvalLeaderboardExplanation,
    ):
        # perturbation flips
        self._diagnose_perturbation_problems(
            eval_results=eval_results,
            key_2_evaluated_model=key_2_evaluated_model,
        )

        # threshold failures
        problems.problems_for_heat_leaderboard(
            evaluator=self,
            leaderboard=leaderboard_explanation,
            metric_threshold=self.args.get(
                evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
                self._metrics_meta.get_primary_metric().threshold,
            ),
            primary_metric_meta=self._metrics_meta.get_primary_metric(),
            problem_type="accuracy",
            actions_description="",
            explanation_type=e10s.GlobalHtmlFragmentExplanation.explanation_type(),
            explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
            explanation_mime=f5s.HtmlFormat.mime,
        )

    def _diagnose_insights(
        self, leaderboard_explanation: e10s.LlmProcedureEvalLeaderboardExplanation
    ):
        t_html_fragment = e10s.GlobalHtmlFragmentExplanation

        leaderboard_explanation.get_insights(
            metrics_meta=self._metrics_meta,
            metric_name_protection=True,
            extra_description_worst="",
            insight_type="accuracy",
            explanation_type=t_html_fragment.explanation_type(),
            explanation_name=t_html_fragment.__name__,
            explanation_mime=f5s.HtmlFormat.mime,
        )

    def get_result(
        self,
    ) -> results.LeaderboardResult:
        return results.LeaderboardResult(
            persistence=self.persistence,
            explainer_id=self.explainer_id(),
            h2o_sonar_config=self.config,
            logger=self.logger,
            explanation=e10s.LlmHeatmapLeaderboardExplanation,
            explanation_format=f5s.CustomJsonFormat,
        )
