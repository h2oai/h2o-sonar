# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import abc
import json
import traceback

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import evaluators
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import problems
from h2o_sonar.lib.api import results as r5s
from h2o_sonar.lib.integrations import genai
from h2o_sonar.utils import tokenization


# aliases
t_heat_leaderboard = e10s.LlmHeatmapLeaderboardExplanation
t_bool_leaderboard = e10s.LlmBoolLeaderboardExplanation


class AbcAgenticH2ogpteEvaluator(abc.ABC, evaluators.Evaluator):
    """Abstract base class for agent-based evaluators which use h2oGPTe agents."""

    _display_name = "Agent-based Evaluator"
    _tagline = "Agent-based Evaluator for LLM and RAG models."
    _brief_description = "Agent-based evaluator for LLM and RAG models."
    _description = _brief_description

    METRIC_AGENT_EVAL = "agent_based_eval"

    _metrics_meta = commons.MetricsMeta(
        metrics=[
            commons.MetricMeta(
                key=METRIC_AGENT_EVAL,
                display_name="Agent-based eval",
                description=(
                    "Percentage of successfully evaluated LLM/RAG outputs by the "
                    "h2oGPTe agents."
                ),
                higher_is_better=True,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=True,
            ),
            t_bool_leaderboard.METRIC_META_MODEL_PARSE_FAILURES,
        ]
    )

    # COMPATIBILITY: RAG evaluation
    _llm = True
    _rag = True

    # GLOBAL: leaderboard as global explanation
    _global_explanation = True

    # EXPLANATION TYPES created by the evaluator
    _explanation_types = [
        e10s.LlmEvalResultsExplanation,
        e10s.LlmHeatmapLeaderboardExplanation,
    ]

    _keywords = [
        evaluators.KEYWORD_LLM,
        evaluators.KEYWORD_EVALUATES_LLM,
        evaluators.KEYWORD_EVALUATES_RAG,
        evaluators.KEYWORD_RQ_AA,
        evaluators.KEYWORD_RQ_P,
        evaluators.KEYWORD_METHOD_TYPE_NON_DETERMINISTIC,
        evaluators.KEYWORD_METHOD_AGENTS,
    ]

    # EVALUATOR PARAMETERS
    PARAM_AGENT_HOST_CFG_KEY = "agent_host_connection_config_key"
    DEFAULT_PARAM_AGENT_HOST_CFG_KEY = ""
    _PARAM_AGENT_HOST_CFG_KEY = evaluators.EvaluatorParam(
        param_name=PARAM_AGENT_HOST_CFG_KEY,
        description=(
            "Configuration key of the agent(s) h2oGPTe host to be used for the "
            "evaluation. If not specified, the first h2oGPTe connection in the "
            "configuration will be used."
        ),
        param_type=commons.EvaluatorParamType.str,
        default_value=DEFAULT_PARAM_AGENT_HOST_CFG_KEY,
        src=evaluators.EvaluatorParam.SRC_EVALUATOR_PARAMS,
    )
    PARAM_LLM_MODEL_NAME = "agent_llm_model_name"
    DEFAULT_LLM_MODEL_NAME = ""
    _PARAM_LLM_MODEL_NAME = evaluators.EvaluatorParam(
        param_name=PARAM_LLM_MODEL_NAME,
        description=(
            "LLM model (name) to be used by the agent(s). If not specified, evaluator "
            "will check whether agent host provides Claude Sonnet, OpenAI GPT-4o or "
            "any llama (in this order) and use it."
        ),
        param_type=commons.EvaluatorParamType.str,
        default_value=DEFAULT_LLM_MODEL_NAME,
        src=evaluators.EvaluatorParam.SRC_EVALUATOR_PARAMS,
    )
    PARAM_AGENT_COLLECTION_ID = "agent_eval_h2ogpte_collection_id"
    DEFAULT_PARAM_AGENT_CONNECTION_ID = ""
    _PARAM_AGENT_COLLECTION_ID = evaluators.EvaluatorParam(
        param_name=PARAM_AGENT_COLLECTION_ID,
        description=(
            "Optional h2oGPTe collection ID to be used for the agent-based evaluation. "
            "If specified, then the agent-based evaluation will be run in that "
            "collection, else a new collection with the empty corpus will be created."
        ),
        param_type=commons.EvaluatorParamType.str,
        default_value=DEFAULT_PARAM_AGENT_HOST_CFG_KEY,
        src=evaluators.EvaluatorParam.SRC_EVALUATOR_PARAMS,
    )
    PARAM_MAX_DATASET_ROWS = "max_dataset_rows"
    DEFAULT_MAX_DATASET_ROWS = 15
    _PARAM_MAX_DATASET_ROWS = evaluators.EvaluatorParam(
        param_name=PARAM_MAX_DATASET_ROWS,
        description=(
            f"Maximum number of dataset rows allowed to be evaluated by the evaluator"
            f"({DEFAULT_MAX_DATASET_ROWS} by default). This is the protection against "
            f"slow and expensive evaluations."
        ),
        param_type=commons.EvaluatorParamType.int,
        default_value=DEFAULT_MAX_DATASET_ROWS,
        src=evaluators.EvaluatorParam.SRC_EVALUATOR_PARAMS,
    )

    _parameters = [
        _PARAM_AGENT_HOST_CFG_KEY,
        _PARAM_LLM_MODEL_NAME,
        _PARAM_AGENT_COLLECTION_ID,
        _PARAM_MAX_DATASET_ROWS,
        evaluators.Evaluator._PARAM_SAVE_LLM_RESULT,
        evaluators.Evaluator._get_custom_param_metric_threshold(
            _metrics_meta.get_primary_metric()
        ),
        evaluators.Evaluator._get_custom_param_min_test_case(),
    ]

    KEY_PROMPT: str = "prompt"
    KEY_ANSWER: str = "answer"
    KEY_PARSED_ANSWER: str = "parsed_answer"
    KEY_ERROR: str = "error"

    PROMPT_KEY_ANSWER = "answer"
    PROMPT_KEY_EVALUATION_SCORE = "evaluation_score"
    PROMPT_KEY_EVALUATION_SUMMARY = "evaluation_summary"

    # prompt template identifiers
    IDENTIFIER_ACTUAL_OUTPUT = "{ACTUAL_OUTPUT}"
    IDENTIFIER_EXPECTED_OUTPUT = "{EXPECTED_OUTPUT}"
    IDENTIFIER_INPUT = "{INPUT}"
    IDENTIFIER_CONTEXT = "{CONTEXT}"

    # default name of the collection with NO corpus used for agent-based evaluations
    COLLECTION_NAME_AGENT_EVAL = "Ephemeral agent-based eval collection"

    # prompt: evaluation justification
    META_AGENT_ANSWER = "evaluation_agent_answer"
    META_EVAL_JUSTIFY = "evaluation_justification"

    # PROBLEM template to be initialized by child class
    _PROBLEM_THRESHOLD_PROTO: problems.ProblemAndAction | None = None

    def __init__(self):
        this = AbcAgenticH2ogpteEvaluator

        evaluators.Evaluator.__init__(self)

        # child implementation prompt with the evaluation instructions
        self._agent_eval_instructions = ""

        # h2oGPTe agent(s) host connection
        self._h2ogpte_connection_key = ""
        self._h2ogpte_connection = None
        self._h2ogpte_client = None
        self._h2ogpte_collection_name = this.COLLECTION_NAME_AGENT_EVAL
        self._h2ogpte_collection_id = ""
        self._llm_model_name = ""

        self.args = None
        self.problems = []
        self.log_name = f"{AbcAgenticH2ogpteEvaluator._display_name} evaluator"

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
                AbcAgenticH2ogpteEvaluator.PARAM_AGENT_HOST_CFG_KEY,
                AbcAgenticH2ogpteEvaluator.DEFAULT_PARAM_AGENT_HOST_CFG_KEY,
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
                            f"Agent host connection configured with key "
                            f"'{self.h2ogpte_connection_key}' in "
                            f"{AbcAgenticH2ogpteEvaluator._display_name} evaluator "
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
                                f"Agent host connection configured with key "
                                f"'{self.h2ogpte_connection_key}' in "
                                f"{AbcAgenticH2ogpteEvaluator._display_name} evaluator "
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
                    f"use h2oGPTe as agent host: configured connection key="
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
    def h2ogpte_collection_id(self):
        if not self._h2ogpte_collection_id:
            self._h2ogpte_collection_id = self.args.get(
                AbcAgenticH2ogpteEvaluator.PARAM_AGENT_COLLECTION_ID,
                AbcAgenticH2ogpteEvaluator.DEFAULT_PARAM_AGENT_CONNECTION_ID,
            )

        return self._h2ogpte_collection_id

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
            # create / lookup the collection
            (self._h2ogpte_collection_id, _) = self.h2ogpte_client.create_collection(
                doc_paths=[], collection_name=collection_name
            )
            self._h2ogpte_collection_name = collection_name

        return self._h2ogpte_collection_id

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
                f"h2oGPTe agent client '{self.h2ogpte_connection.name}' and LLM model "
                f"'{self.llm_model_name}' health check failed: {ex}\n{traceback}"
            )

    @property
    def llm_model_name(self):
        if not self._llm_model_name:
            if not self.args:
                raise ValueError("Evaluator arguments are not initialized")

            param_llm_model_name = self.args.get(
                AbcAgenticH2ogpteEvaluator.PARAM_LLM_MODEL_NAME, ""
            )

            if param_llm_model_name:
                self._llm_model_name = param_llm_model_name
            else:
                # find the LLM model name to be used by the agent
                llm_model_names = self.h2ogpte_client.list_llm_model_names()
                if not llm_model_names:
                    raise ValueError(
                        f"No LLM models found on the h2oGPTe agent host "
                        f"{self.h2ogpte_connection.name}"
                    )
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

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        **evaluator_params,
    ) -> bool:
        evaluators.Evaluator.check_compatibility(self, params, **evaluator_params)

        if not self.models:
            self.logger.warning(
                f"{self.log_name}: no RAG/LLM models found for evaluation: "
                f"{[m.key for m in self.models]} - NOT COMPATIBLE"
            )
            return False

        if not evaluators.Evaluator._check_llm_dataset_compatibility(
            self, params=params, evaluator_keywords=self._keywords
        ):
            return False

        # allow evaluation of datasets up to 15 prompts
        if (
            params
            and params.dataset
            and isinstance(params.dataset, datasets.LlmDataset)
        ):
            self._resolve_evaluator_params()
            max_dataset_rows = self.args.get(
                AbcAgenticH2ogpteEvaluator.PARAM_MAX_DATASET_ROWS,
                AbcAgenticH2ogpteEvaluator.DEFAULT_MAX_DATASET_ROWS,
            )
            if len(params.dataset.inputs) > max_dataset_rows:
                self.logger.warning(
                    f"{self.log_name}: dataset has more than {max_dataset_rows} "
                    f"prompts: {len(params.dataset.inputs)} - which is over the limit. "
                    f"This is the protection against slow and expensive evaluations - "
                    f"evaluator marked as NOT COMPATIBLE"
                )
                return False

        # check agent compatibility
        try:
            self._resolve_evaluator_params()
            self.h2ogpte_health_check()
            self.logger.info(
                f"{self.log_name}: Agent-based evaluator will use LLM model: "
                f"'{self.llm_model_name}'"
            )
        except Exception as ex:
            self.logger.warning(
                f"{self.log_name}: Unable to resolve LLM model name for the "
                f"agent-based evaluator given h2oGPTe connection key "
                f"'{self._h2ogpte_connection_key}' and LLM model name evaluator "
                f"parameter: {ex}\n{traceback.format_exc()}"
            )
            return False

        return True

    def setup(self, model, persistence, **kwargs):
        """Setup h2oGPTe connection, h2oGPTe client, valid LLM model name and
        create h2oGPTe connection to be used by agent-based evaluation.

        Parameters
        ----------
        model :
            Model to be evaluated.
        persistence :
            Persistence to be used for storing of the evaluation results.

        """
        evaluators.Evaluator.setup(self, model, persistence, **kwargs)

        self._resolve_evaluator_params()

        self.log_name = (
            f"{AbcAgenticH2ogpteEvaluator._display_name} evaluator "
            f"{self.mli_key}/{self.key}"
        )

        self._agent_eval_instructions = self._init_eval_instructions()
        # ensure LLM model name which will also initialize the h2oGPTe client
        self.logger.info(
            f"{self.log_name}: will use {self.llm_model_name} LLM model name"
        )
        # ensure h2oGPTe collection for the agent-based evaluation
        self.h2ogpte_collection()

    def ask_agent(self, prompts: list[str]):
        self.logger.info(
            f"{self.log_name}: assigning agent "
            f"- which will use '{self._h2ogpte_collection_name}' collection and "
            f"LLM model '{self.llm_model_name}' - the following prompts:\n{prompts}"
        )

        agent_responses = self.h2ogpte_client.ask_collection(
            prompts=prompts,
            collection_id=self.h2ogpte_collection_id,
            llm_model_name=self.llm_model_name,
            include_chunks=False,
            # h2oGPTe parameters: use agent
            llm_args={
                genai.H2oGpteRagClient.CFG_USE_AGENT: True,
            },
        )

        self.logger.info(f"{self.log_name}: agent responses:\n{agent_responses}")

        return agent_responses

    @abc.abstractmethod
    def _init_eval_instructions(self) -> str:
        """Child implementation to set evaluation instructions for the agent-based
        evaluator.

        Returns
        -------
        str
            Prompt template for the agent-based evaluator.

        """
        pass

    def _agent_prompt(
        self,
        eval_instructions: str,
        question: str,
        actual_answer: str,
    ) -> str:
        """Assemble the prompt for the agent which will perform the evaluation."""

        # prepare the prompt for the agent-based evaluation which does fact checking
        # of the provided user input

        prompt = f"""You are agent whose role is to evaluate text of the ACTUAL ANSWER.

Instructions for WHAT should be evaluated:

[BEGIN EVALUATION INSTRUCTIONS]
{eval_instructions}
[END EVALUATION INSTRUCTIONS]

Instructions how to return the evaluation result:

- provide the evaluation result as JSon with the following structure:

    {{
        "answer": string,
        "evaluation_score": float,
        "evaluation_summary": string
    }}

- evaluation_score: is the float number between 0.0 and 1.0 where 1.0 means
  that the ACTUAL ANSWER passed the evaluation and 0.0 means that the ACTUAL
  ANSWER failed the evaluation
- evaluation_summary: is the summary of the evaluation result which briefly
  provides justification for the evaluation score and describes how was the
  actual answer evaluated

ACTUAL ANSWER data:

[BEGIN ACTUAL ANSWER]
{actual_answer}
[END ACTUAL ANSWER]

If it may help, use QUESTION which was answered by the ACTUAL ANSWER:

[BEGIN QUESTION]
{question}
[END QUESTION]

"""
        self.logger.info(
            f"{self.log_name}: prompt for the agent:\n{10 * '='}\n{prompt}\n{10 * '='}"
        )

        return prompt

    def _prepare_prompts(
        self, rows: list[datasets.LlmDataset.LlmDatasetRow]
    ) -> list[str]:
        return [
            self._agent_prompt(
                eval_instructions=self._agent_eval_instructions,
                question=row.i,
                actual_answer=row.actual_output,
                # CONSIDER: row.expected_output,
                # CONSIDER: row.context
            )
            for row in rows
        ]

    def _eval_prompts(self, prompts: list[str]) -> list[dict[str, str]]:
        """Get the answers for given prompts from the agent.

        Returns
        -------
        list[dict[str, str]]
            List of dictionaries with two keys:
            - KEY_PROMPT: prompt for which the answer was given
            - KEY_ANSWER: agent's answer

        """
        # TODO do NOT ask agent if actual answer is INTERNAL ERROR > forge failed answer
        outputs = self.ask_agent(prompts)
        return [
            {self.KEY_ANSWER: outputs[i].answer, self.KEY_PROMPT: prompts[i]}
            for i in range(len(prompts))
        ]

    def _extract_score(
        self, prompt_and_answer: dict[str, str]
    ) -> tuple[bool, str] | None:
        """Strive to get the metric value from the agent's answer.

        Returns
        -------
        Tuple[bool, str] | None
            Tuple of two values:
            - metric value (bool)
            - insight text (str)

        """
        self.logger.info(
            f"{self.log_name}: parsing agent's answer: >>>{prompt_and_answer}<<< to "
            f"get the metric value and insight text..."
        )

        if (
            not prompt_and_answer
            or not isinstance(prompt_and_answer, dict)
            or not prompt_and_answer.get(self.KEY_ANSWER)
            or not prompt_and_answer.get(self.KEY_PROMPT)
        ):
            return None

        answer = prompt_and_answer[self.KEY_ANSWER]

        # check whether the answer contains the expected JSon keys
        if self.PROMPT_KEY_EVALUATION_SCORE not in answer:
            return None

        # parse JSon created by the agent
        a_score = None
        a_summary = ""
        try:
            json_start = answer.find("{")
            json_end = answer.rfind("}")
            if json_start == -1 or json_end == -1:
                return None

            a_dict = json.loads(answer[json_start : json_end + 1])

            # a_answer = a_dict.get(self.PROMPT_KEY_ANSWER)
            a_score = a_dict.get(self.PROMPT_KEY_EVALUATION_SCORE)
            a_summary = a_dict.get(self.PROMPT_KEY_EVALUATION_SUMMARY)

            if a_score is None:
                return None

            # MUST: parse the evaluation score to float
            try:
                a_score = float(a_score)
            except ValueError:
                return None
            if not 0.0 <= a_score <= 1.0:
                return None

            # NICE: get a summary message
            a_summary = a_summary or ""
        except Exception as ex:
            self.logger.warning(
                f"{self.log_name}: unable to parse agent's JSon answer from "
                f">>>{answer}<<<: "
                f"{ex}\n{traceback.format_exc()}"
            )

        return a_score, a_summary

    def _parse_answers(self, prompts_and_answers: list[dict]) -> list[dict[str, str]]:
        """Parse the agent's answers and return the parsed answers.

        Returns
        -------
        list[dict[str, str]]
            List of dictionaries with three keys:
            - KEY_PROMPT: prompt for which the answer was given
            - KEY_ANSWER: agent's answer
            - KEY_PARSED_ANSWER: parsed answer - ``None`` or metric score and insight

        """
        return [
            dict(
                **prompt_and_answer,
                **{self.KEY_PARSED_ANSWER: self._extract_score(prompt_and_answer)},
            )
            for prompt_and_answer in prompts_and_answers
        ]

    def evaluate(self, llm_testset, **kwargs) -> list:
        # RAG models: key -> model
        key_2_evaluated_model = {m.key: m for m in self.models}
        # LLM host: RAG or service
        llm_host = (
            commons.LlmModelHostType.RAG
            if isinstance(
                next(iter(key_2_evaluated_model.values())), models.ExplainableRagModel
            )
            else commons.LlmModelHostType.SERVICE
        )
        llm_host_str = "RAG" if llm_host == commons.LlmModelHostType.RAG else "LLM"
        self.report_progress(0.01, "Loading dataset...")
        llm_dataset = datasets.LlmDataset.from_datatable_dict(llm_testset.to_dict())
        self.report_progress(0.011, "Preparing prompts...")
        prompts = self._prepare_prompts(llm_dataset.inputs)
        self.report_progress(0.012, "Agent is evaluating prompts...")
        prompts_and_answers = self._eval_prompts(prompts)
        self.report_progress(0.98, "Parsing agent answers...")
        results = self._parse_answers(prompts_and_answers)
        self.report_progress(0.99, "Creating report...")
        eval_results = datasets.LlmEvalResults()

        #
        # EVALUATION
        #
        # get the metric from the runtime context, not from the class (overridden)
        primary_metric_key = self._metrics_meta.get_primary_metric().key
        for index, row in enumerate(llm_dataset.inputs):
            parsed_answer = results[index][self.KEY_PARSED_ANSWER]
            answer = f"Agent's answer: {results[index][self.KEY_ANSWER]}"

            metric_score = 0.0 if parsed_answer is None else parsed_answer[0]
            metrics = {
                primary_metric_key: metric_score,
                t_bool_leaderboard.KEY_RESULT_CHECK_FAIL_P: (
                    1.0 if parsed_answer is None else 0.0
                ),
            }

            meta = {
                AbcAgenticH2ogpteEvaluator.META_AGENT_ANSWER: answer,
            }
            if parsed_answer:
                meta[AbcAgenticH2ogpteEvaluator.META_EVAL_JUSTIFY] = (
                    parsed_answer[1] or ""
                )
            actual_answer_meta = [
                tokenization.Tokenization(
                    tokenization=tokenization.TOKENIZATION_TYPE_F,
                    data=[
                        tokenization.TextFragment(
                            text=row.actual_output,
                            metrics=metrics,
                            meta=meta,
                        )
                    ],
                )
            ]

            result_row = datasets.LlmEvalResults.LlmEvalResultRow(
                dataset_row=row,
                metrics=metrics,
                actual_output_meta=actual_answer_meta,
            )
            eval_results.add_result(result_row)

            if not parsed_answer:
                # add a problem for the evaluation agent answer which cannot be parsed
                self._diagnose_judge_answer_parsing_problem(
                    result_row=result_row,
                    key_2_evaluated_model=key_2_evaluated_model,
                    agent_answer=answer,
                    problem_type=primary_metric_key,
                )

        #
        # NORMALIZATION of the evaluation RESULTS
        #

        # EXPLANATIONS
        explanations = []

        # EXPLANATION: all data (per prompt metrics)
        if self.args.get(
            evaluators.Evaluator.PARAM_SAVE_LLM_RESULT,
            evaluators.Evaluator.DEFAULT_SAVE_LLM_RESULT,
        ):
            eval_results_explanation = e10s.LlmEvalResultsExplanation(
                evaluator=self,
                display_name=self._display_name,
                display_category=e10s.Explanation.DISPLAY_CAT_LLM,
                eval_results=eval_results,
            )
            # FORMATS of the explanation: JSon, CSV, DataTable
            eval_results_explanation.add_json_format()
            eval_results_explanation.add_csv_format()
            eval_results_explanation.add_datatable_format()
            explanations.append(eval_results_explanation)

        # EXPLANATION: ok/fail leaderboard
        leaderboard_explanation = t_heat_leaderboard.from_eval_results(
            evaluator=self,
            eval_results=eval_results,
            metrics_meta=self._metrics_meta,
            key_2_evaluated_model=key_2_evaluated_model,
            display_name="RAG/LLM benchmark leaderboard",
            display_category=e10s.GlobalSummaryFeatImpExplanation.DISPLAY_CAT_LLM,
            llm_host=(
                commons.LlmModelHostType.RAG
                if isinstance(
                    next(iter(key_2_evaluated_model.values())),
                    models.ExplainableRagModel,
                )
                else commons.LlmModelHostType.SERVICE
            ),
            logger=self.logger,
        )

        # FORMAT of the explanation: Markdown
        leaderboard_explanation.add_markdown_format(
            sort_by_metric_id=primary_metric_key
        )
        leaderboard_explanation.add_evalstudio_markdown_format(
            sort_by_metric_id=primary_metric_key
        )
        leaderboard_explanation.add_json_format()
        explanations.append(leaderboard_explanation)

        # PROBLEMS for alerts and actionability
        self._diagnose_problems(
            eval_results=eval_results,
            key_2_evaluated_model=key_2_evaluated_model,
            leaderboard_explanation=leaderboard_explanation,
            primary_metric_key=primary_metric_key,
        )

        # INSIGHTS
        self._diagnose_insights(
            leaderboard_explanation=leaderboard_explanation,
            primary_metric_key=primary_metric_key,
        )

        # EXPLANATION: HTML fragment
        if self.config and self.config.create_html_representations:
            try:
                html_explanation = e10s.GlobalHtmlFragmentExplanation(
                    evaluator=self,
                    display_name=f"{llm_host_str} benchmark leaderboard as HTML",
                    display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
                )
                details = {
                    "Agents host used by the evaluator": (
                        self._h2ogpte_connection.name
                    ),
                    "Prompt template": (
                        t_bool_leaderboard.AdditionalDetails(
                            formatting="pre",
                            text=self._agent_prompt(
                                eval_instructions=self._agent_eval_instructions,
                                question="QUESTION",
                                actual_answer="ACTUAL ANSWER",
                            ),
                        )
                    ),
                }
                html_explanation.add_html_format(
                    str(
                        leaderboard_explanation.as_html(
                            sort_by_metric_id=primary_metric_key,
                            additional_details=details,
                        )
                    )
                )
                explanations.append(html_explanation)
            except Exception as ex:
                self.logger.warning(
                    f"{self.log_name}: HTML fragment explanation creation failed: "
                    f"{ex}\n{traceback.format_exc()}",
                )

        # EXPLANATION: ZIP archive of all artifacts created by the evaluator
        explanations.append(
            self.create_explanation_workdir_archive(
                display_name=f"Archive of {self._display_name} artifacts",
                display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
            )
        )

        return explanations

    def _diagnose_problems(
        self,
        eval_results: datasets.LlmEvalResults,
        key_2_evaluated_model: dict,
        leaderboard_explanation: e10s.LlmHeatmapLeaderboardExplanation,
        primary_metric_key: str = METRIC_AGENT_EVAL,
    ):
        # perturbation flips
        self._diagnose_perturbation_problems(
            eval_results=eval_results,
            key_2_evaluated_model=key_2_evaluated_model,
        )

        # low test case count
        self._diagnose_low_test_case_problem(
            eval_results=eval_results,
            models=self.models,
            test_case_minimum=self.args.get(evaluators.Evaluator.PARAM_MIN_TEST_CASES),
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
            problem_type=primary_metric_key,
            problem_code=None,
            actions_description="",
            explanation_type=e10s.GlobalHtmlFragmentExplanation.explanation_type(),
            explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
            explanation_mime=f5s.HtmlFormat.mime,
        )

    def _diagnose_insights(
        self,
        leaderboard_explanation: e10s.LlmHeatmapLeaderboardExplanation,
        primary_metric_key: str = METRIC_AGENT_EVAL,
    ):
        t_html_fragment = e10s.GlobalHtmlFragmentExplanation

        leaderboard_explanation.get_insights(
            metrics_meta=self._metrics_meta,
            insight_type=primary_metric_key,
            explanation_type=t_html_fragment.explanation_type(),
            explanation_name=t_html_fragment.__name__,
            explanation_mime=f5s.HtmlFormat.mime,
        )

    def get_result(self) -> r5s.LeaderboardResult:
        return r5s.LeaderboardResult(
            persistence=self.persistence,
            explainer_id=self.explainer_id(),
            h2o_sonar_config=self.config,
            logger=self.logger,
            explanation=t_heat_leaderboard,
            explanation_format=f5s.CustomJsonFormat,
        )
