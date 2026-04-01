# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import json
import pathlib
import re
import traceback
import uuid

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from h2o_sonar import loggers
from h2o_sonar.lib.api import agents
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import evaluators
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import problems
from h2o_sonar.lib.api import results
from h2o_sonar.utils import progress as progress_utils
from h2o_sonar.utils import testing


class AgentSanityCheckEvaluator(evaluators.Evaluator):
    """Agent Sanity Check evaluator checks the integrity and quality of agent-created
    artifacts.

    """

    _display_name = "Agent sanity check"
    _tagline = "Assess integrity and quality of agent created artifacts."

    # metrics
    METRIC_SANITY = "agent_sanity"
    METRIC_TOOL_FAILURES = "agent_tools_failures"
    METRIC_USED_TOOLS = "agent_tools_used"
    METRIC_AGENT_REPLAN = "agent_plan_changes"
    METRIC_AGENT_STEPS = "agent_steps"
    METRIC_COST = "agent_cost"
    METRIC_DURATION = "agent_duration_s"
    METRIC_DATA_SIZE = "agent_data_size_mb"
    METRIC_FILE_COUNT = "agent_file_count"

    # keys
    KEY_ACTIVITY_DIAGRAM = "agent_chat_activity_diagram"
    KEY_TOOLS_BAR_CHART = "agent_chat_tools_bar_chart"
    KEY_SCRIPTS_BAR_CHART = "agent_chat_scripts_bar_chart"

    _metrics_meta = commons.MetricsMeta(
        metrics=[
            commons.MetricMeta(
                key=METRIC_SANITY,
                display_name="Sanity",
                description=(
                    "The sanity and integrity of the agent-created artifacts."
                ),
                higher_is_better=True,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=True,
            ),
            # TODO split the metric into separate metrics for tools and scripts
            #   use visualizer logic to distinguish between tools and scripts
            commons.MetricMeta(
                key=METRIC_TOOL_FAILURES,
                display_name="Tool/Script Failures",
                data_type="int",
                display_format=",d",  # int like 123,456
                description=(
                    "The number of agent's tools and scripts execution failures."
                ),
                higher_is_better=False,
                value_range=(0.0, commons.MetricMeta.EFFECTIVE_INF_FLOAT_INT),
                threshold=0.0,  # tool failures should ideally be 0
                is_primary_metric=False,
            ),
            commons.MetricMeta(
                key=METRIC_USED_TOOLS,
                display_name="Used Tools",
                data_type="int",
                display_format=",d",  # int like 123,456
                description="The number of tools actually used by the agentic run.",
                higher_is_better=False,
                value_range=(0.0, commons.MetricMeta.EFFECTIVE_INF_FLOAT_INT),
                threshold=3.0,  # maximum reasonable number of tools used
                is_primary_metric=False,
            ),
            commons.MetricMeta(
                key=METRIC_AGENT_REPLAN,
                display_name="Plan Changes",
                data_type="int",
                display_format=",d",  # int like 123,456
                description=(
                    "The number of re-planning actions made by the agent - when agent "
                    "changes the strategy, reacts to tool failures and inability to "
                    "provide desired results."
                ),
                higher_is_better=False,
                value_range=(0.0, commons.MetricMeta.EFFECTIVE_INF_FLOAT_INT),
                threshold=3.0,  # re-planning should ideally be avoided
                is_primary_metric=False,
            ),
            commons.MetricMeta(
                key=METRIC_AGENT_STEPS,
                display_name="Agent Steps",
                data_type="int",
                display_format=",d",  # int like 123,456
                description=(
                    "The number of steps - user and assistant chat messages - agent "
                    "needed to provide actual answer and/or to finish."
                ),
                higher_is_better=False,
                value_range=(0.0, commons.MetricMeta.EFFECTIVE_INF_FLOAT_INT),
                threshold=125.0,  # maximum reasonable number of steps
                is_primary_metric=False,
            ),
            commons.MetricMeta(
                key=METRIC_COST,
                display_name="Cost",
                description="Agentic run cost in USD.",
                higher_is_better=False,
                value_range=(0.0, commons.MetricMeta.EFFECTIVE_INF_FLOAT_INT),
                threshold=0.5,  # TODO 5.0,  # maximum reasonable cost in USD
                is_primary_metric=False,
            ),
            commons.MetricMeta(
                key=METRIC_DURATION,
                display_name="Duration (s)",
                display_format=".1f",
                description="Agentic run duration in seconds.",
                higher_is_better=False,
                value_range=(0.0, commons.MetricMeta.EFFECTIVE_INF_FLOAT_INT),
                threshold=600.0,  # maximum reasonable duration (10 minutes)
                is_primary_metric=False,
            ),
            commons.MetricMeta(
                key=METRIC_DATA_SIZE,
                display_name="Data Size (MB)",
                display_format=".2f",
                description="Amount of data created by the agentic run in MBs.",
                higher_is_better=False,
                value_range=(0.0, commons.MetricMeta.EFFECTIVE_INF_FLOAT_INT),
                threshold=1.5,  # maximum reasonable data size in MB
                is_primary_metric=False,
            ),
            commons.MetricMeta(
                key=METRIC_FILE_COUNT,
                display_name="File Count",
                data_type="int",
                display_format=",d",  # int like 123,456
                description="Number of files created by the agentic run.",
                higher_is_better=False,
                value_range=(0.0, commons.MetricMeta.EFFECTIVE_INF_FLOAT_INT),
                threshold=25.0,  # maximum reasonable number of files
                is_primary_metric=False,
            ),
        ]
    )

    # COMPATIBILITY: LLM/RAG evaluation
    _llm = True
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

    _NON_PRIMARY_PARAMETERS = [
        evaluators.EvaluatorParam(
            param_name=m.key,
            description=(
                f"Metric '{m.key}' threshold - values above this "
                f"threshold are considered problematic."
            ),
            param_type=commons.EvaluatorParamType.float,
            default_value=m.threshold,
            src=evaluators.EvaluatorParam.SRC_EVALUATOR_PARAMS,
        )
        for m in _metrics_meta.to_list()
        if not m.is_primary_metric
    ]

    _parameters = (
        [
            evaluators.Evaluator._get_custom_param_metric_threshold(
                _metrics_meta.get_primary_metric()
            )
        ]
        + _NON_PRIMARY_PARAMETERS
        + [
            evaluators.Evaluator._PARAM_SAVE_LLM_RESULT,
        ]
    )

    _keywords = [
        evaluators.KEYWORD_GPU_OPT,
        evaluators.KEYWORD_LLM,
        evaluators.KEYWORD_EVALUATES_LLM,
        evaluators.KEYWORD_EVALUATES_RAG,
        evaluators.KEYWORD_RQ_P,
        evaluators.KEYWORD_SR_11_7_CS,
        evaluators.KEYWORD_SR_11_7_OA,
        evaluators.KEYWORD_NIST_AI_RMF_S,
        evaluators.KEYWORD_PROBLEM_TYPE_IR,
        evaluators.KEYWORD_PROBLEM_TYPE_QA,
        evaluators.KEYWORD_EVALUATOR_ROLE_REGULATOR,
        evaluators.KEYWORD_ES_GENERATE,
        evaluators.KEYWORD_METHOD_SEMANTIC_SIMILARITY,
        evaluators.KEYWORD_METHOD_TYPE_DETERMINISTIC,
    ]

    _brief_description = """Agent Sanity Check Evaluator performs basic check of the
agentic RAG/LLM system. The evaluator reviews agent chat session to check for
problems and inspects the artifacts created by the agent during its operation. It
verifies the integrity and sanity of the artifacts created by the agent during its
operation. This includes checking for the presence of expected files, validating
their formats, and ensuring that the content meets  predefined criteria. The evaluator
helps identify potential issues in the agent's workflow, ensuring that it operates
correctly and reliably."""
    _description = evaluators.Evaluator._description_builder(
        brief=f"""{_brief_description}

**Method**:

- Looks for artifacts created by the agent during its operation prepared by
  the test lab completion.
- Performs sanity checks on the artifacts to ensure they meet expected standards:
  linting (JSon), content validation (non-empty, non-empty pages), expected structure
  (for directories and files) and field values.
- Create problems and insights if any issues are found during the sanity checks.
- Calculates a sanity score based on the results of the checks, providing an overall
  assessment of the agent's performance.
""",
        metrics_meta=_metrics_meta,
        keywords=_keywords,
        parameters=_parameters,
        leaderboard_type=e10s.LlmHeatmapLeaderboardExplanation.explanation_type(),
    )

    COL_INPUT = datasets.LlmDataset.KEY_INPUT
    COL_CONTEXT = datasets.LlmDataset.KEY_CONTEXT
    COL_EXPECTED_OUTPUT = datasets.LlmDataset.KEY_EXPECTED_OUTPUT
    COL_ACTUAL_OUTPUT = datasets.LlmDataset.KEY_ACTUAL_OUTPUT
    COL_MODEL = "model"
    COL_SCORE = "score"

    FILE_CHAT_HISTORY = "MSG_META_TYPE_ITEM_agent_chat_history.json"
    FILE_USAGE_STATS = "MSG_META_TYPE_ITEM_usage_stats.json"

    def __init__(self):
        evaluators.Evaluator.__init__(self)

        # agent chat session history as dict (loaded and cached)
        self._chat_history = []

        self.args = None
        self.log_name = "Agent Sanity"

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        **evaluator_params,
    ) -> bool:
        evaluators.Evaluator.check_compatibility(self, params, **evaluator_params)

        if not evaluators.Evaluator._check_llm_dataset_compatibility(
            self,
            params=params,
            evaluator_keywords=self.keywords,
        ):
            return False

        return True

    def setup(self, model, persistence, **kwargs):
        evaluators.Evaluator.setup(self, model, persistence, **kwargs)

        self._resolve_evaluator_params()

    def evaluate(self, llm_testset, explanations_types=None, **kwargs) -> list:
        save_llm_result = self.args.get(
            evaluators.Evaluator.PARAM_SAVE_LLM_RESULT,
            evaluators.Evaluator.DEFAULT_SAVE_LLM_RESULT,
        )

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

        eval_results = self._evaluate_testset(llm_testset=llm_testset)

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

        # custom THRESHOLDs for all metrics
        for m_meta in self._metrics_meta.to_list():
            if m_meta.key != self._metrics_meta.get_primary_metric().key:
                m_meta.threshold = self.args.get(
                    m_meta.key,
                    m_meta.threshold,
                )
            else:
                m_meta.threshold = self.args.get(
                    evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
                    m_meta.threshold,
                )

        # EXPLANATION: heatmap leaderboard
        heatmap_explanation = e10s.LlmHeatmapLeaderboardExplanation.from_eval_results(
            evaluator=self,
            eval_results=eval_results,
            metrics_meta=self._metrics_meta,
            key_2_evaluated_model=key_2_evaluated_model,
            llm_host=llm_host,
            display_name="LLM heatmap leaderboard",
            display_category=e10s.GlobalSummaryFeatImpExplanation.DISPLAY_CAT_LLM,
            logger=self.logger,
        )
        heatmap_explanation.add_json_format(
            threshold=self.args.get(
                evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
                self._metrics_meta.get_primary_metric().threshold,
            )
        )
        primary_metric_id = self._metrics_meta.get_primary_metric().key
        heatmap_explanation.add_markdown_format(sort_by_metric_id=primary_metric_id)
        heatmap_explanation.add_evalstudio_markdown_format(
            sort_by_metric_id=primary_metric_id
        )
        explanations.append(heatmap_explanation)

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
                        heatmap_explanation.as_html(sort_by_metric_id=primary_metric_id)
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

    def _get_artifacts_path(
        self, row: datasets.LlmDataset.LlmDatasetRow
    ) -> pathlib.Path | None:
        """Get the path to the artifacts created by the agent during its operation and
        verify that the directory exists.

        Returns
        -------
        pathlib.Path | None :
            Path to the artifacts directory or None if not found or not a directory.

        """
        test_lab_key = ""
        chat_session_key = ""
        chat_message_key = ""

        for c in row.categories:
            if c.startswith(agents.AgentHost.CAT_TEST_LAB):
                test_lab_key = c.replace(f"{agents.AgentHost.CAT_TEST_LAB}:", "")
            elif c.startswith(agents.AgentHost.CAT_AGENT_ARTIFACTS):
                c_parts = c.split(":")
                if (
                    len(c_parts) == 7
                    and c_parts[0] == agents.AgentHost.CAT_AGENT_ARTIFACTS
                    and c_parts[1] == agents.AgentHost.CAT_AGENT_HOST
                    and c_parts[2]
                    in [
                        models.ExplainableModelType.h2ogpte.name,
                        models.ExplainableModelType.h2ogpte_llm.name,
                    ]
                    and c_parts[3] == agents.AgentHost.CAT_AGENT_SESSION
                    and c_parts[5] == agents.AgentHost.CAT_AGENT_MSG
                ):
                    chat_session_key = c_parts[4]
                    chat_message_key = c_parts[6]

        if test_lab_key and chat_session_key and chat_message_key:
            tlp = testing.TestLabPersistence
            artifacts_path = tlp.get_chat_message_path(
                chat_session_dir=tlp.get_chat_session_dir(
                    test_case_completion_dir=tlp.get_test_case_completion_dir(
                        test_lab_dir=tlp.get_test_lab_dir(
                            user_dir=self.persistence.user_dir,
                            test_lab_key=test_lab_key,
                        ),
                        model_key=row.model_key,
                        test_case_key=row.key,
                    ),
                    chat_session_key=chat_session_key,
                ),
                chat_message_key=chat_message_key,
            )
            if artifacts_path.exists() and artifacts_path.is_dir():
                return artifacts_path

        return None

    def _e_jsons_parseable(
        self, test_case: datasets.LlmDataset.LlmDatasetRow, files: list[pathlib.Path]
    ) -> tuple[float, list[tuple[pathlib.Path, str]]]:
        self.logger.info(
            f"{self.log_name} Checking parseability of JSon files among {len(files)} "
            f"files..."
        )
        metric_score = 1.0
        nonparseable_jsons = []

        for f in files:
            if f.is_file():
                self.logger.info(f"  Found file: {f} (size: {f.stat().st_size} bytes)")
                if f.suffix.lower() == ".json":
                    try:
                        with f.open("r", encoding="utf-8") as jf:
                            _ = json.load(jf)
                    except Exception as ex:
                        nonparseable_jsons.append((f, str(ex)))
                        continue
            elif f.is_dir():
                self.logger.info(f"  Found directory: {f}")

        # add problem anytimethere are any non-parseable files (no threshold defined)
        if nonparseable_jsons:
            total_issues = len(nonparseable_jsons)
            metric_score = max(0.0, 1.0 - total_issues / len(files)) if files else 0.0

            # PROBLEMS
            primary_metric = self.metrics_meta().get_primary_metric()
            type_pna = problems.ProblemAndAction
            nonparseable_file_names = ", ".join(
                [f"{str(f.name)} (error: {err})" for f, err in nonparseable_jsons]
            )
            self.add_problem(
                problems.ProblemAndAction(
                    description=(
                        f"Found {len(nonparseable_jsons)} non-parseable JSon "
                        f"file{'s' if len(nonparseable_jsons) > 1 else ''} "
                        f"in the artifacts created by the agent during its operation: "
                        f"{nonparseable_file_names}"
                    ),
                    severity=problems.ProblemSeverity.medium,
                    problem_type="data quality",
                    problem_attrs={
                        type_pna.ATTR_MODEL_NAME: test_case.model_key,
                        type_pna.ATTR_M_ID: primary_metric.key,
                        type_pna.ATTR_M_NAME: primary_metric.display_name,
                        type_pna.ATTR_M_THRESHOLD: primary_metric.threshold,
                        type_pna.ATTR_M_SCORE: metric_score,
                    },
                    problem_code=problems.AVIDProblemCode.P0200_MODEL,
                    actions_description=(
                        "Investigate the cause of broken JSon files - extension, "
                        "content, and structure."
                    ),
                    explainer_id=self.explainer_id(),
                    explainer_name=self._display_name,
                    explanation_type=(
                        e10s.GlobalHtmlFragmentExplanation.explanation_type()
                    ),
                    explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
                    explanation_mime=f5s.HtmlFormat.mime,
                )
            )

        return metric_score, nonparseable_jsons

    def _e_files_non_empty(
        self, test_case: datasets.LlmDataset.LlmDatasetRow, files: list[pathlib.Path]
    ) -> tuple[float, list[tuple[pathlib.Path, str]]]:
        self.logger.info(
            f"{self.log_name} Checking non-emptiness of files among {len(files)} "
            f"files..."
        )

        metric_score = 1.0
        empty_files = []
        for f in files:
            if f.is_file():
                self.logger.info(f"  Found file: {f} (size: {f.stat().st_size} bytes)")
                if f.stat().st_size == 0:
                    empty_files.append((f, ""))
                    continue
            elif f.is_dir():
                self.logger.info(f"  Found directory: {f}")

        # add problem anytime there are any non-parseable files (no threshold defined)
        if empty_files:
            total_issues = len(empty_files)
            metric_score = max(0.0, 1.0 - total_issues / len(files))

            # PROBLEMS
            primary_metric = self.metrics_meta().get_primary_metric()
            type_pna = problems.ProblemAndAction
            self.add_problem(
                problems.ProblemAndAction(
                    description=(
                        f"Found {len(empty_files)} empty "
                        f"file{'s' if len(empty_files) > 1 else ''} "
                        f"in the artifacts "
                        f"created by the agent during its operation: "
                        f"{', '.join([str(f.name) for f, _ in empty_files])}"
                    ),
                    severity=problems.ProblemSeverity.low,
                    problem_type="data quality",
                    problem_attrs={
                        type_pna.ATTR_MODEL_NAME: test_case.model_key,
                        type_pna.ATTR_M_ID: primary_metric.key,
                        type_pna.ATTR_M_NAME: primary_metric.display_name,
                        type_pna.ATTR_M_THRESHOLD: primary_metric.threshold,
                        type_pna.ATTR_M_SCORE: metric_score,
                    },
                    problem_code=problems.AVIDProblemCode.P0200_MODEL,
                    actions_description=(
                        "Investigate the cause of empty files in the agent's "
                        "artifacts. Ensure that the agent is functioning correctly "
                        "and generating the expected output files."
                    ),
                    explainer_id=self.explainer_id(),
                    explainer_name=self._display_name,
                    explanation_type=(
                        e10s.GlobalHtmlFragmentExplanation.explanation_type()
                    ),
                    explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
                    explanation_mime=f5s.HtmlFormat.mime,
                )
            )

        return metric_score, empty_files

    def _e_load_and_cache_chat_history(self, files: list[pathlib.Path]) -> list:
        self.logger.info(f"{self.log_name} Loading / caching chat history...")
        if not self._chat_history and files:
            for f in files:
                if (
                    f.is_file()
                    and f.name == AgentSanityCheckEvaluator.FILE_CHAT_HISTORY
                ):
                    with open(f, encoding="utf-8") as jf:
                        try:
                            self._chat_history = json.load(jf)
                        except Exception as ex:
                            self.logger.warning(
                                f"{self.log_name}: chat session file {f} parse "
                                f"failed: {ex}\n{traceback.format_exc()}"
                            )
                    break
        return self._chat_history

    def _e_tool_failures(
        self, test_case: datasets.LlmDataset.LlmDatasetRow, files: list[pathlib.Path]
    ) -> float:
        self.logger.info(
            f"{self.log_name} Calculating tool/script failures in chat session..."
        )

        # chat history:
        # * [ { "content": "...", "role": "assistant|user" } ]
        # * [ { "role": "assistant", "content": "exitcode: 0 (execution succeeded)" } ]
        # * [ { "role": "assistant", "content": "exitcode: 1 (execution failed)" } ]
        # * [ { "role": "assistant", "content": "exitcode: -255 (execution failed)" } ]
        tool_failures = 0.0
        chat_history = self._e_load_and_cache_chat_history(files)
        if chat_history:
            tool_executions = 0.0
            for msg in chat_history:
                if (
                    isinstance(msg, dict)
                    and msg.get("role", "") == "assistant"
                    and isinstance(msg.get("content", ""), str)
                    and msg.get("content", "").startswith("exitcode:")
                ):
                    tool_executions += 1
                    if "(execution failed)" in msg.get("content", ""):
                        tool_failures += 1
                        continue
            if tool_executions > 0:
                pct_failures = float(tool_failures) / float(tool_executions)
                self.logger.info(
                    f"Agent tools failures: {tool_failures} out of {tool_executions} "
                    f"({pct_failures:.2%})"
                )

            # PROBLEMS
            if tool_failures:
                type_pna = problems.ProblemAndAction
                metric_meta = self.metrics_meta().get_metric(
                    AgentSanityCheckEvaluator.METRIC_TOOL_FAILURES
                )
                self.add_problem(
                    # TODO split the metric into separate metrics for tools and scripts
                    problems.ProblemAndAction(
                        description=(
                            f"There are {int(tool_failures)} agent tool and script "
                            f"execution failure{'s' if tool_failures != 1 else ''} "
                            f"out of {int(tool_executions)} executions during the "
                            f"agent operation."
                        ),
                        severity=problems.ProblemSeverity.high,
                        problem_type="stability",
                        problem_attrs={
                            type_pna.ATTR_MODEL_NAME: test_case.model_key,
                            type_pna.ATTR_M_ID: metric_meta.key,
                            type_pna.ATTR_M_NAME: metric_meta.display_name,
                            type_pna.ATTR_M_THRESHOLD: metric_meta.threshold,
                            type_pna.ATTR_M_SCORE: float(tool_failures),
                        },
                        problem_code=problems.AVIDProblemCode.P0200_MODEL,
                        actions_description=(
                            "Review agent chat history to investigate the cause of "
                            "tool and script execution failures and take corrective "
                            "actions."
                        ),
                        explainer_id=self.explainer_id(),
                        explainer_name=self._display_name,
                        explanation_type=(
                            e10s.GlobalHtmlFragmentExplanation.explanation_type()
                        ),
                        explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
                        explanation_mime=f5s.HtmlFormat.mime,
                    )
                )

        return tool_failures

    def _e_used_tools(
        self, test_case: datasets.LlmDataset.LlmDatasetRow, files: list[pathlib.Path]
    ) -> float:
        self.logger.info(f"{self.log_name} Calculating used tools in chat session...")

        # chat history:
        # * [ { "content": "...", "role": "assistant|user" } ]
        # * [ { "role": "user", "content": "... from api_server.agent_tools. ... " } ]
        # * from api_server.agent_tools.ask_question_about_documents
        #   import ask_question_about_documents
        # * from api_server.agent_tools.unified_search import unified_search

        chat_history = self._e_load_and_cache_chat_history(files)
        used_tools = []
        if chat_history:
            for msg in chat_history:
                if (
                    isinstance(msg, dict)
                    and msg.get("role", "") == "user"
                    and isinstance(msg.get("content", ""), str)
                    and "from api_server.agent_tools." in msg.get("content", "")
                ):
                    tool_names = re.findall(
                        r"from api_server\.agent_tools\.(\w+)", msg.get("content", "")
                    )
                    for tn in tool_names:
                        if tn not in used_tools:
                            used_tools.append(tn)
            self.logger.info(f"Agent used tools: {used_tools}")

        # PROBLEMS
        threshold = (
            self.metrics_meta()
            .get_metric(AgentSanityCheckEvaluator.METRIC_USED_TOOLS)
            .threshold
        )
        if len(used_tools) > threshold:
            type_pna = problems.ProblemAndAction
            metric_meta = self.metrics_meta().get_metric(
                AgentSanityCheckEvaluator.METRIC_USED_TOOLS
            )
            self.add_problem(
                problems.ProblemAndAction(
                    description=(
                        f"The agent used {len(used_tools)} tool"
                        f"{'s' if len(used_tools) != 1 else ''} "
                        f"during its operation: {', '.join(used_tools)} which "
                        f"exceeds the threshold of {int(threshold)} tools."
                    ),
                    severity=problems.ProblemSeverity.low,
                    problem_type="efficiency",
                    problem_attrs={
                        type_pna.ATTR_MODEL_NAME: test_case.model_key,
                        type_pna.ATTR_M_ID: metric_meta.key,
                        type_pna.ATTR_M_NAME: metric_meta.display_name,
                        type_pna.ATTR_M_THRESHOLD: metric_meta.threshold,
                        type_pna.ATTR_M_SCORE: float(len(used_tools)),
                    },
                    problem_code=problems.AVIDProblemCode.P0200_MODEL,
                    actions_description=(
                        "Review the tools used by the agent and assess their "
                        "relevance and effectiveness for the given tasks."
                    ),
                    explainer_id=self.explainer_id(),
                    explainer_name=self._display_name,
                    explanation_type=(
                        e10s.GlobalHtmlFragmentExplanation.explanation_type()
                    ),
                    explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
                    explanation_mime=f5s.HtmlFormat.mime,
                )
            )

        return float(len(used_tools)) if used_tools else 0.0

    def _e_agent_replan(
        self, test_case: datasets.LlmDataset.LlmDatasetRow, files: list[pathlib.Path]
    ) -> float:
        self.logger.info(
            f"{self.log_name} Calculating agent re-plans in chat session..."
        )

        # chat history:
        # * [ { "content": "...", "role": "assistant|user" } ]
        # * [ { "content": "... Let me try a different approach ...", "role": "user" } ]

        chat_history = self._e_load_and_cache_chat_history(files)
        replans = []
        if chat_history:
            for msg in chat_history:
                if (
                    isinstance(msg, dict)
                    and msg.get("role", "") == "user"
                    and isinstance(msg.get("content", ""), str)
                    and "let me try a different approach"
                    in msg.get("content", "").lower()
                ):
                    replans.append(msg.get("content", ""))

        if replans:
            self.logger.info(f"Agent re-plans: {replans}")

            # PROBLEMS
            type_pna = problems.ProblemAndAction
            metric_meta = self.metrics_meta().get_metric(
                AgentSanityCheckEvaluator.METRIC_AGENT_REPLAN
            )
            self.add_problem(
                problems.ProblemAndAction(
                    description=(
                        f"The agent re-planned its approach "
                        f"{len(replans)} time{'s' if len(replans) != 1 else ''} "
                        f"during its operation."
                    ),
                    severity=problems.ProblemSeverity.medium,
                    problem_type="stability",
                    problem_attrs={
                        type_pna.ATTR_MODEL_NAME: test_case.model_key,
                        type_pna.ATTR_M_ID: metric_meta.key,
                        type_pna.ATTR_M_NAME: metric_meta.display_name,
                        type_pna.ATTR_M_THRESHOLD: metric_meta.threshold,
                        type_pna.ATTR_M_SCORE: float(len(replans)),
                    },
                    problem_code=problems.AVIDProblemCode.P0200_MODEL,
                    actions_description=(
                        "Review the agent chat history to understand the reasons "
                        "for re-planning and take corrective actions to improve "
                        "the agent's stability."
                    ),
                    explainer_id=self.explainer_id(),
                    explainer_name=self._display_name,
                    explanation_type=(
                        e10s.GlobalHtmlFragmentExplanation.explanation_type()
                    ),
                    explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
                    explanation_mime=f5s.HtmlFormat.mime,
                )
            )

            return float(len(replans))

        return 0.0

    def _e_agent_steps(
        self, test_case: datasets.LlmDataset.LlmDatasetRow, files: list[pathlib.Path]
    ) -> float:
        self.logger.info(f"{self.log_name} Calculating agent steps in chat session...")

        # chat history:
        # * [ { "content": "...", "role": "assistant|user" } ]

        chat_history = self._e_load_and_cache_chat_history(files)
        num_steps = 0.0
        if chat_history:
            num_steps = len(chat_history)
            self.logger.info(f"Agent steps: {num_steps}")

            # PROBLEMS
            threshold = (
                self.metrics_meta()
                .get_metric(AgentSanityCheckEvaluator.METRIC_AGENT_STEPS)
                .threshold
            )
            if 0.0 < threshold < 1_000_000.0 and num_steps > threshold:
                type_pna = problems.ProblemAndAction
                metric_meta = self.metrics_meta().get_metric(
                    AgentSanityCheckEvaluator.METRIC_AGENT_STEPS
                )
                self.add_problem(
                    problems.ProblemAndAction(
                        description=(
                            f"The agent took {num_steps} steps during its operation, "
                            f"which exceeds the threshold of {int(threshold)} steps."
                        ),
                        severity=problems.ProblemSeverity.low,
                        problem_type="efficiency",
                        problem_attrs={
                            type_pna.ATTR_MODEL_NAME: test_case.model_key,
                            type_pna.ATTR_M_ID: metric_meta.key,
                            type_pna.ATTR_M_NAME: metric_meta.display_name,
                            type_pna.ATTR_M_THRESHOLD: metric_meta.threshold,
                            type_pna.ATTR_M_SCORE: float(num_steps),
                        },
                        problem_code=problems.AVIDProblemCode.P0200_MODEL,
                        actions_description=(
                            "Review the agent's workflow and identify opportunities "
                            "to streamline its operations and reduce the number of "
                            "steps."
                        ),
                        explainer_id=self.explainer_id(),
                        explainer_name=self._display_name,
                        explanation_type=(
                            e10s.GlobalHtmlFragmentExplanation.explanation_type()
                        ),
                        explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
                        explanation_mime=f5s.HtmlFormat.mime,
                    )
                )

        return num_steps

    def _e_cost(self, files: list[pathlib.Path]) -> float:
        self.logger.info(f"{self.log_name} Calculating cost in chat session...")
        cost = 0.0
        if files:
            for f in files:
                if f.is_file() and f.name == AgentSanityCheckEvaluator.FILE_USAGE_STATS:
                    try:
                        with f.open("r", encoding="utf-8") as jf:
                            jdata = json.load(jf)
                            if "cost" in jdata:
                                cost = float(str(jdata["cost"]).replace(" [USD]", ""))
                    except Exception as ex:
                        self.logger.warning(
                            f"{self.log_name}: cost file {f} parse failed: "
                            f"{ex}\n{traceback.format_exc()}"
                        )

        # PROBLEMS
        threshold = (
            self.metrics_meta()
            .get_metric(AgentSanityCheckEvaluator.METRIC_COST)
            .threshold
        )
        if 0.0 < threshold < 1_000_000.0 and cost > threshold:
            type_pna = problems.ProblemAndAction
            metric_meta = self.metrics_meta().get_metric(
                AgentSanityCheckEvaluator.METRIC_COST
            )
            self.add_problem(
                problems.ProblemAndAction(
                    description=(
                        f"The agent incurred a cost of ${cost:.4f} during its "
                        f"operation, which exceeds the threshold of ${threshold:.4f}."
                    ),
                    severity=problems.ProblemSeverity.medium,
                    problem_type="cost",
                    problem_attrs={
                        type_pna.ATTR_M_ID: metric_meta.key,
                        type_pna.ATTR_M_NAME: metric_meta.display_name,
                        type_pna.ATTR_M_THRESHOLD: metric_meta.threshold,
                        type_pna.ATTR_M_SCORE: float(cost),
                    },
                    problem_code=problems.AVIDProblemCode.P0200_MODEL,
                    actions_description=(
                        "Review the agent's operations to identify areas where "
                        "costs can be optimized and reduced."
                    ),
                    explainer_id=self.explainer_id(),
                    explainer_name=self._display_name,
                    explanation_type=(
                        e10s.GlobalHtmlFragmentExplanation.explanation_type()
                    ),
                    explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
                    explanation_mime=f5s.HtmlFormat.mime,
                )
            )

        return cost

    def _e_duration_s(self, files: list[pathlib.Path]) -> float:
        self.logger.info(f"{self.log_name} Calculating duration in chat session...")
        duration = 0.0
        if files:
            for f in files:
                if f.is_file() and f.name == AgentSanityCheckEvaluator.FILE_USAGE_STATS:
                    try:
                        with f.open("r", encoding="utf-8") as jf:
                            jdata = json.load(jf)
                            if len(jdata.get("usage", [])) > 0:
                                duration = jdata["usage"][0].get("time_taken", 0.0)
                    except Exception as ex:
                        self.logger.warning(
                            f"{self.log_name}: duration file {f} parse failed: "
                            f"{ex}\n{traceback.format_exc()}"
                        )

        # PROBLEMS
        threshold = (
            self.metrics_meta()
            .get_metric(AgentSanityCheckEvaluator.METRIC_DURATION)
            .threshold
        )
        if 0.0 < threshold < 1_000_000.0 and duration > threshold:
            type_pna = problems.ProblemAndAction
            metric_meta = self.metrics_meta().get_metric(
                AgentSanityCheckEvaluator.METRIC_DURATION
            )
            self.add_problem(
                problems.ProblemAndAction(
                    description=(
                        f"The agent took {duration:.2f} seconds to complete its "
                        f"operation, which exceeds the threshold of {threshold:.2f} "
                        f"seconds."
                    ),
                    severity=problems.ProblemSeverity.low,
                    problem_type="efficiency",
                    problem_attrs={
                        type_pna.ATTR_M_ID: metric_meta.key,
                        type_pna.ATTR_M_NAME: metric_meta.display_name,
                        type_pna.ATTR_M_THRESHOLD: metric_meta.threshold,
                        type_pna.ATTR_M_SCORE: float(duration),
                    },
                    problem_code=problems.AVIDProblemCode.P0200_MODEL,
                    actions_description=(
                        "Analyze the agent's workflow to identify bottlenecks and "
                        "optimize its performance to reduce the duration of its "
                        "operations."
                    ),
                    explainer_id=self.explainer_id(),
                    explainer_name=self._display_name,
                    explanation_type=(
                        e10s.GlobalHtmlFragmentExplanation.explanation_type()
                    ),
                    explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
                    explanation_mime=f5s.HtmlFormat.mime,
                )
            )

        return duration

    def _e_data_size_mb(self, files: list[pathlib.Path]) -> float:
        self.logger.info(
            f"{self.log_name} Calculating data size among {len(files)} files..."
        )
        total_size_bytes = sum(f.stat().st_size for f in files if f.is_file())
        total_size_mb = total_size_bytes / (1024.0 * 1024.0)
        self.logger.info(f"  Total data size: {total_size_mb:.3f} MB")

        # PROBLEMS
        threshold = (
            self.metrics_meta()
            .get_metric(AgentSanityCheckEvaluator.METRIC_DATA_SIZE)
            .threshold
        )
        if 0.0 < threshold < 1_000_000.0 and total_size_mb > threshold:
            type_pna = problems.ProblemAndAction
            metric_meta = self.metrics_meta().get_metric(
                AgentSanityCheckEvaluator.METRIC_DATA_SIZE
            )
            self.add_problem(
                problems.ProblemAndAction(
                    description=(
                        f"The total data size of {total_size_mb:.3f} MB in the "
                        f"artifacts created by the agent during its operation "
                        f"exceeds the threshold of {threshold:.3f} MB."
                    ),
                    severity=problems.ProblemSeverity.low,
                    problem_type="storage",
                    problem_attrs={
                        type_pna.ATTR_M_ID: metric_meta.key,
                        type_pna.ATTR_M_NAME: metric_meta.display_name,
                        type_pna.ATTR_M_THRESHOLD: metric_meta.threshold,
                        type_pna.ATTR_M_SCORE: float(total_size_mb),
                    },
                    problem_code=problems.AVIDProblemCode.P0200_MODEL,
                    actions_description=(
                        "Review the agent's data management practices to identify "
                        "ways to optimize storage usage and reduce data size."
                    ),
                    explainer_id=self.explainer_id(),
                    explainer_name=self._display_name,
                    explanation_type=(
                        e10s.GlobalHtmlFragmentExplanation.explanation_type()
                    ),
                    explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
                    explanation_mime=f5s.HtmlFormat.mime,
                )
            )

        return total_size_mb

    def _e_file_count(self, files: list[pathlib.Path]) -> int:
        self.logger.info(f"{self.log_name} Counting files among {len(files)} files...")

        # PROBLEMS
        threshold = (
            self.metrics_meta()
            .get_metric(AgentSanityCheckEvaluator.METRIC_FILE_COUNT)
            .threshold
        )
        if 0.0 < threshold < 1_000_000.0 and len(files) > threshold:
            type_pna = problems.ProblemAndAction
            metric_meta = self.metrics_meta().get_metric(
                AgentSanityCheckEvaluator.METRIC_FILE_COUNT
            )
            self.add_problem(
                problems.ProblemAndAction(
                    description=(
                        f"The total file count of {len(files)} in the artifacts "
                        f"created by the agent during its operation exceeds the "
                        f"threshold of {int(threshold)} files."
                    ),
                    severity=problems.ProblemSeverity.low,
                    problem_type="storage",
                    problem_attrs={
                        type_pna.ATTR_M_ID: metric_meta.key,
                        type_pna.ATTR_M_NAME: metric_meta.display_name,
                        type_pna.ATTR_M_THRESHOLD: metric_meta.threshold,
                        type_pna.ATTR_M_SCORE: float(len(files)),
                    },
                    problem_code=problems.AVIDProblemCode.P0200_MODEL,
                    actions_description=(
                        "Review the agent's file management practices to identify "
                        "ways to optimize file usage and reduce the total file count."
                    ),
                    explainer_id=self.explainer_id(),
                    explainer_name=self._display_name,
                    explanation_type=(
                        e10s.GlobalHtmlFragmentExplanation.explanation_type()
                    ),
                    explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
                    explanation_mime=f5s.HtmlFormat.mime,
                )
            )

        return len(files)

    def _e_sanity(
        self,
        empty_files: list[tuple[pathlib.Path, str]],
        nonparseable_jsons: list[tuple[pathlib.Path, str]],
        total_files: int,
    ) -> float:
        self.logger.info(
            f"{self.log_name} Calculating sanity score based on {total_files} files, "
            f"{len(empty_files)} empty files, and "
            f"{len(nonparseable_jsons)} non-parseable JSon files..."
        )

        if total_files == 0:
            return 1.0

        # avoid reporting multiple problems for the same issue - intersect the lists
        intersection_len = 0
        if empty_files and nonparseable_jsons:
            nonparseable_set = set([file_path for file_path, _ in nonparseable_jsons])
            intersection_len = len(
                [
                    file_path
                    for file_path, _ in empty_files
                    if file_path in nonparseable_set
                ]
            )
        total_issues = len(empty_files) + len(nonparseable_jsons) - intersection_len
        metric_score = max(0.0, 1.0 - float(total_issues) / float(total_files))

        return metric_score

    def _evaluate_test_case(
        self,
        tc: datasets.LlmDataset.LlmDatasetRow,
        artifacts_path: pathlib.Path,
    ) -> tuple[dict[str, float], dict, dict, dict]:
        # purge cached chat history
        self._chat_history = []

        # metric scores for the test case
        metric_scores = {
            self.METRIC_TOOL_FAILURES: 0.0,
            self.METRIC_USED_TOOLS: 0.0,
            self.METRIC_AGENT_REPLAN: 0.0,
            self.METRIC_AGENT_STEPS: 0.0,
            self.METRIC_COST: 0.0,
            self.METRIC_DURATION: 0.0,
            self.METRIC_DATA_SIZE: 0.0,
            self.METRIC_FILE_COUNT: 0.0,
            self.METRIC_SANITY: 1.0,
        }

        # list files in artifacts_path
        files = list(artifacts_path.rglob("*"))
        if not files:
            return metric_scores, {}, {}, {}

        # evaluate metrics
        (metric_score_empty_files, empty_files) = self._e_files_non_empty(
            test_case=tc, files=files
        )
        (metric_score_nonparseable_jsons, nonparseable_jsons) = self._e_jsons_parseable(
            test_case=tc, files=files
        )
        metric_scores[self.METRIC_TOOL_FAILURES] = float(
            self._e_tool_failures(test_case=tc, files=files)
        )
        metric_scores[self.METRIC_USED_TOOLS] = float(
            self._e_used_tools(test_case=tc, files=files)
        )
        metric_scores[self.METRIC_AGENT_REPLAN] = float(
            self._e_agent_replan(test_case=tc, files=files)
        )
        metric_scores[self.METRIC_AGENT_STEPS] = float(
            self._e_agent_steps(test_case=tc, files=files)
        )
        metric_scores[self.METRIC_COST] = float(self._e_cost(files=files))
        metric_scores[self.METRIC_DURATION] = float(self._e_duration_s(files=files))
        metric_scores[self.METRIC_DATA_SIZE] = float(self._e_data_size_mb(files=files))
        metric_scores[self.METRIC_FILE_COUNT] = float(self._e_file_count(files=files))
        metric_scores[self.METRIC_SANITY] = self._e_sanity(
            empty_files=empty_files,
            nonparseable_jsons=nonparseable_jsons,
            total_files=len(files),
        )

        # agentic chat visualization + failure bar charts ... all as JSon files
        visualizer = H2ogpteAgentChatHistoryVisualizer(
            chat_history=self._e_load_and_cache_chat_history(files),
            logger=self.logger,
        )
        (activity_diagram_json, tools_stats_json, scripts_stats_json, _, _, _) = (
            visualizer.visualize(
                pathlib.Path(self.persistence.get_explainer_working_dir())
            )
        )

        return (
            metric_scores,
            activity_diagram_json,
            tools_stats_json,
            scripts_stats_json,
        )

    def _evaluate_testset(self, llm_testset):
        self.report_progress(0.01, "Evaluating test cases...")

        llm_dataset = datasets.LlmDataset.from_datatable_dict(llm_testset.to_dict())
        eval_results = datasets.LlmEvalResults()
        p_start = 0.1
        p_delta = (
            (0.9 - p_start) / len(llm_dataset.inputs) if llm_dataset.inputs else 0.0
        )
        p = p_start
        for e, r in enumerate(llm_dataset.inputs):
            # progress
            self.report_progress(
                progress=progress_utils.ProgressCallbackContext.progress_for_steps(
                    e + 1, len(llm_dataset.inputs)
                ),
                message=evaluators.Evaluator._eval_row_progress_msg(
                    metric_name=self.METRIC_SANITY,
                    device="",
                    row=e + 1,
                    total_rows=len(llm_dataset.inputs),
                ),
            )

            # navigate to directory w/ artifacts created by the agent and assess
            artifacts_path = self._get_artifacts_path(r)
            if artifacts_path:
                self.logger.info(
                    f"{self.log_name}: checking artifacts of model / test case "
                    f"{r.model_key} / {r.key}..."
                )
                # evaluate test case
                (
                    tc_metric_scores,
                    activity_diagram_json,
                    tools_stats_json,
                    scripts_stats_json,
                ) = self._evaluate_test_case(artifacts_path=artifacts_path, tc=r)
                # prepare actual output metadata to be added to "actual_output_meta"
                this = AgentSanityCheckEvaluator
                agent_chat_actual_output_meta_list = [
                    {
                        this.KEY_ACTIVITY_DIAGRAM: activity_diagram_json,
                    },
                    {
                        this.KEY_TOOLS_BAR_CHART: tools_stats_json,
                    },
                    {
                        this.KEY_SCRIPTS_BAR_CHART: scripts_stats_json,
                    },
                ]
            else:
                self.logger.warning(
                    f"{self.log_name}: artifacts path  for model / test case "
                    f"{r.model_key} / {r.key} does not exist: {artifacts_path}"
                )
                tc_metric_scores = {
                    self.METRIC_TOOL_FAILURES: 0.0,
                    self.METRIC_USED_TOOLS: 0.0,
                    self.METRIC_AGENT_REPLAN: 0.0,
                    self.METRIC_AGENT_STEPS: 0.0,
                    self.METRIC_COST: 0.0,
                    self.METRIC_DURATION: 0.0,
                    self.METRIC_DATA_SIZE: 0.0,
                    self.METRIC_FILE_COUNT: 0.0,
                    self.METRIC_SANITY: 1.0,
                }
                agent_chat_actual_output_meta_list = None

            # add result
            eval_results.add_result(
                datasets.LlmEvalResults.LlmEvalResultRow(
                    dataset_row=r,
                    metrics=tc_metric_scores,
                    actual_output_meta=agent_chat_actual_output_meta_list,
                )
            )

            # progress
            p += p_delta

        return eval_results

    def get_result(
        self,
    ) -> results.LeaderboardResult:
        return results.LeaderboardResult(
            persistence=self.persistence,
            explainer_id=AgentSanityCheckEvaluator.explainer_id(),
            h2o_sonar_config=self.config,
            logger=self.logger,
            explanation=e10s.LlmHeatmapLeaderboardExplanation,
            explanation_format=f5s.CustomJsonFormat,
        )


class H2ogpteAgentChatHistoryVisualizer:
    """Visualization of the technical, low level, h2oGPTe agent chat history including
    agent actions (tool execution, script execution, re-planning) and failures,
    fallbacks, confidence reflections, intermediary and the final answer.

    Agent chat history activity diagram as JSon:

    .. code-block:: text

        {
          "rows": [
            {
                nodes: [
                  {
                    "id": str,
                    "role": str,
                    "label": str,
                  },
                  ...
                ]
            },
          "edges": [
            {
                "from": str,  # node id
                "to": str,    # node id
                "label": str, # edge label
            }
          ]
        }

    Agent chat tools statistics as JSon:

    .. code-block:: text

        {
            "tools": {
                "ToolName": {
                    "name": str,
                    "success_count": int,
                    "failure_count": int,
                    "total_count": int,
                },
                ...
            }
        }

    Agent chat scripts statistics as JSon:

    .. code-block:: text

        {
            "scripts": {
                "ScriptName": {
                    "name": str,
                    "success_count": int,
                    "failure_count": int,
                    "total_count": int,
                },
                ...
            }
        }

    """

    # IMPROVE: generate mermaid activity diagram (.mmd/txt)

    # COLOR definitions
    COLOR_DARKKHAKI = "#BDB76B"
    COLOR_H2O_YELLOW = "#FEC925"
    COLOR_LIGHTBLUE = "lightblue"
    COLOR_LIGHTSTEELBLUE = "#B0C4DE"
    COLOR_LIGHTGRAY = "lightgray"
    COLOR_LIMEGREEN = "#32CD32"
    COLOR_VIOLET = "#7FDBFF"
    COLOR_SILVER = "#C0C0C0"

    COLOR_START = "black"
    COLOR_END = COLOR_START
    COLOR_QUESTION = COLOR_LIMEGREEN
    COLOR_QUESTION_REASK = COLOR_QUESTION
    COLOR_ANSWER = COLOR_QUESTION
    COLOR_ANSWER_DRAFT = COLOR_LIGHTGRAY
    COLOR_AGENT = COLOR_H2O_YELLOW
    COLOR_ASSISTANT = "lightgray"
    COLOR_ASSISTANT_FEEDBACK = COLOR_ASSISTANT
    COLOR_TOOL = "#7FDBFF"  # lightblue
    COLOR_SCRIPT = COLOR_LIGHTBLUE
    COLOR_CONFIDENCE = COLOR_DARKKHAKI
    COLOR_STATUS_OK = "lightgreen"
    COLOR_STATUS_FAIL = "lightcoral"
    COLOR_REPLAN = COLOR_VIOLET

    # SPECIAL STRINGS definitions
    PKG_AGENT_TOOLS = "from api_server.agent_tools."
    PREFIX_EXITCODE = "exitcode: "
    # prefix of messages elaborating on the agent's confidence level
    STR_CONFIDENCE = "<confidence>"
    # prefix of messages containing the final/intermediary answer
    STR_CONSTRAINED_OUT = "<constrained_output>"
    # infix indicating the agent is re-planning
    STR_LET_ME_REPLAN = "Let me try a different approach"

    # KEYS
    KEY_CONTENT = "content"
    KEY_ROLE = "role"
    KEY_ROWS = "rows"
    # EXCLUDED KEY_ROW = "row"
    KEY_NODES = "nodes"
    KEY_ID = "id"
    KEY_EDGES = "edges"
    KEY_FROM = "from"
    KEY_TO = "to"
    KEY_LABEL = "label"
    KEY_COLOR = "color"
    KEY_TOOLS = "tools"
    KEY_SCRIPTS = "scripts"

    # ROLES
    ROLE_AGENT = "agent"
    ROLE_USER = "user"  # ~ agent
    ROLE_ASSISTANT = "assistant"
    ROLE_RUNTIME = "runtime"
    # auxiliary roles
    _ROLE_QUESTION = "question"
    _ROLE_ASK_QUESTION = "re-ask\nquestion"

    # NODES

    # chart nodes
    NODE_START_ID = "Agentic run START"
    NODE_END_ID = "Agentic run END"

    def __init__(
        self,
        chat_history: list,
        logger: loggers.SonarLogger | None = None,
    ):
        self.chat_history = chat_history

        self.logger = logger or loggers.SonarPrintLogger()

    @staticmethod
    def _chart_agent_chat_err_code(content: str) -> float | None:
        match = re.search(r"exitcode:\s*(-?\d+)", content)
        if match:
            try:
                exit_code_str = match.group(1)
                return float(exit_code_str)
            except (ValueError, IndexError):
                pass
        return None

    @staticmethod
    def _chart_agent_chat_node_status(
        chat_msg, stats: dict | None = None
    ) -> tuple[str, str]:
        this = H2ogpteAgentChatHistoryVisualizer

        exit_code_content = chat_msg[this.KEY_CONTENT]
        exit_code = "OK" if "(execution succeeded)" in exit_code_content else "fail"
        if exit_code == "fail":
            err_code = this._chart_agent_chat_err_code(exit_code_content)
            err_code_str = "" if err_code is None else f"{err_code}"
            role = f"{exit_code}\n({err_code_str})"
            node_color = this.COLOR_STATUS_FAIL
        else:
            role = f"{exit_code}"
            node_color = this.COLOR_STATUS_OK

        if stats is not None:
            stats["total_count"] += 1
            if exit_code == "OK":
                stats["success_count"] += 1
            else:
                stats["failure_count"] += 1

        return role, node_color

    @staticmethod
    def _chart_agent_chat_tool_column_chart_autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    def _chart_agent_chat_tool_column_chart(
        self,
        data: dict,
        base_dir: pathlib.Path,
        entity_key: str = "tools",
        entity_name: str = "Tool",
    ) -> pathlib.Path | None:
        """Bar chart of the tools usage statistics.

        Parameters
        ----------
        data: dict
            Tools statistics dictionary - see the docstring of the class for details.
        base_dir: pathlib.Path
            Base directory to save the generated PNG file.
        entity_key: str
            Key in the data dictionary to extract the tools statistics.
        entity_name: str
            Name of the entity to be used in the chart title.

        Returns
        -------
            pathlib.Path | None
                Path to the generated PNG file or None if no data is available.

        """
        this = H2ogpteAgentChatHistoryVisualizer

        try:
            tools_data = data.get(entity_key, {})
            if not tools_data:
                return None

            tool_names = list(tools_data.keys())
            success_counts = [tools_data[tool]["success_count"] for tool in tool_names]
            failure_counts = [tools_data[tool]["failure_count"] for tool in tool_names]

            x = np.arange(len(tool_names))  # the label locations
            width = 0.35  # the width of the bars

            fig, ax = plt.subplots(figsize=(12, 7))

            rects1 = ax.bar(
                x - width / 2, success_counts, width, label="Success", color="seagreen"
            )
            rects2 = ax.bar(
                x + width / 2, failure_counts, width, label="Failure", color="firebrick"
            )

            ax.set_ylabel("Count")
            ax.set_title(f"{entity_name} Success and Failure Counts")
            ax.set_xticks(x)
            ax.set_xticklabels(tool_names, rotation=45, ha="center")
            ax.legend()

            # add labels on top of the bars
            this._chart_agent_chat_tool_column_chart_autolabel(rects1, ax)
            this._chart_agent_chat_tool_column_chart_autolabel(rects2, ax)

            fig.tight_layout()

            png_path = base_dir / f"bar_chart_{entity_key}.png"
            plt.title(f"Agent Chat History: {entity_name}", size=20)
            plt.savefig(png_path)

            return png_path
        except KeyError as e:
            self.logger.error(
                f"Missing key in data structure when rendering agent chat {entity_name}"
                f"bar chart: {e}\n{traceback.format_exc()}"
            )
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred when rendering agent chat"
                f" {entity_name} bar chart:  {e}\n{traceback.format_exc()}"
            )

        return None

    @staticmethod
    def _uuidize_activity_graph_json(activity_graph_dict: dict) -> dict:
        if not activity_graph_dict:
            return activity_graph_dict

        this = H2ogpteAgentChatHistoryVisualizer
        nodes_label_2_uuid = {}
        if activity_graph_dict.get(this.KEY_ROWS):
            for r in activity_graph_dict.get(this.KEY_ROWS):
                for n in r[this.KEY_NODES]:
                    n_id = str(uuid.uuid4())

                    nodes_label_2_uuid[n[this.KEY_ID]] = n_id

                    n[this.KEY_ID] = n_id
                    if this.KEY_COLOR in n:
                        del n[this.KEY_COLOR]
            for e in activity_graph_dict.get(this.KEY_EDGES):
                from_label = e[this.KEY_FROM]
                e[this.KEY_FROM] = nodes_label_2_uuid.get(e[this.KEY_FROM], from_label)
                to_label = e[this.KEY_TO]
                e[this.KEY_TO] = nodes_label_2_uuid.get(e[this.KEY_TO], to_label)

        return activity_graph_dict

    def visualize(
        self, base_dir: pathlib.Path | None
    ) -> tuple[
        dict,
        dict,
        dict,
        pathlib.Path | None,
        pathlib.Path | None,
        pathlib.Path | None,
    ]:
        """Visualize h2oGPTe agent chat history from the low level perspective.

        Parameters
        ----------
        base_dir: pathlib.Path | None
            Base directory to save the generated PNG files.

        Returns
        -------
            Tuple :
               Activity diagram dictionary, tools statistics dictionary,
               scripts statistics dictionary, path to activity diagram PNG file,
               path to tools statistics PNG file, path to scripts statistics PNG file.

        """
        this = H2ogpteAgentChatHistoryVisualizer

        # DECLARATIONS
        activity_graph_dict = {
            this.KEY_ROWS: [],
            this.KEY_EDGES: [],
        }
        tools_stats_dict = {
            this.KEY_TOOLS: {},
        }
        scripts_stats_dict = {
            this.KEY_SCRIPTS: {},
        }
        activity_png_path = None
        tools_png_path = None
        scripts_png_path = None

        # CHART: activity diagram

        activity_graph = nx.DiGraph()

        # start NODE and end NODE (text intentionally hidden)
        activity_graph.add_node(
            this.NODE_START_ID,
            role="start",
            content="Agent Run Started",
            color=this.COLOR_START,
        )
        activity_graph_dict[this.KEY_ROWS].append(
            {
                # EXCLUDE: this.KEY_ROW: 1,
                this.KEY_NODES: [
                    {
                        this.KEY_ID: this.NODE_START_ID,
                        this.KEY_ROLE: this.ROLE_RUNTIME,
                        this.KEY_LABEL: "start",
                        this.KEY_COLOR: this.COLOR_START,
                    }
                ],
            }
        )
        activity_graph.add_node(
            this.NODE_END_ID,
            role="end",
            content="Agent Run Finished",
            color=this.COLOR_END,
        )

        node_counter = 0
        pos = {}  # nodes positions for final rendering
        y_pos = 1.0  # current vertical position while creating nodes
        previous_node = this.NODE_START_ID
        do_skip_next_msg = False
        for msg in self.chat_history:
            activity_graph_dict[this.KEY_ROWS].append(
                {
                    # EXCLUDE: this.KEY_ROW: node_counter + 2,
                    this.KEY_NODES: []
                }
            )

            if do_skip_next_msg:
                node_counter += 1
                do_skip_next_msg = False
                continue

            # NODE for this message
            node_content = msg[this.KEY_CONTENT]  # potentially long text
            node_role = msg[this.KEY_ROLE]  # 'user' ~ 'agent', 'assistant'
            node_id = f"{node_role}_{node_counter}"
            if node_role == this.ROLE_ASSISTANT:
                if node_counter == 0:
                    node_color = this.COLOR_QUESTION
                    node_role = this._ROLE_QUESTION
                elif node_content and node_content.startswith("\n<current_date>"):
                    node_color = this.COLOR_QUESTION_REASK
                    node_role = this._ROLE_ASK_QUESTION  # ask the question again
                elif node_content and node_content.startswith(this.PREFIX_EXITCODE):
                    node_color = (
                        this.COLOR_STATUS_OK
                        if "(execution succeeded)" in node_content
                        else this.COLOR_STATUS_FAIL
                    )
                else:
                    node_color = this.COLOR_ASSISTANT
            else:  # 'user' ~ 'agent'
                node_color = this.COLOR_AGENT
            if node_role == this.ROLE_USER:
                node_role = this.ROLE_AGENT
            activity_graph.add_node(
                node_id,
                role=node_role,
                content=node_content,
                color=node_color,
            )
            pos[node_id] = (0, -y_pos)  # node position for final rendering
            activity_graph_dict[this.KEY_ROWS][-1][this.KEY_NODES].append(
                {
                    this.KEY_ID: node_id,
                    this.KEY_ROLE: (
                        this.ROLE_USER
                        if node_role in [this._ROLE_QUESTION, this._ROLE_ASK_QUESTION]
                        else node_role
                    ),
                    this.KEY_LABEL: node_role.replace("\n", " "),
                    this.KEY_COLOR: node_color,
                }
            )

            # EDGE between the previous/start node and this message / NODE
            activity_graph.add_edge(previous_node, node_id, label="")
            activity_graph_dict[this.KEY_EDGES].append(
                {
                    this.KEY_FROM: previous_node,
                    this.KEY_TO: node_id,
                    this.KEY_LABEL: "",
                }
            )

            # NODES/EDGES w/ agent actions: tool, script, re-planning, feedback, ...
            if (
                isinstance(msg, dict)
                and msg.get(this.KEY_ROLE, "") == this.ROLE_USER
                and isinstance(node_content, str)
            ):
                # NODE: confidence reflection
                if node_content and node_content.startswith(this.STR_CONFIDENCE):
                    node_id_confidence = f"confidence_{node_counter}"
                    activity_graph.add_node(
                        node_id_confidence,
                        role="confidence",
                        content="Confidence level",
                        color=this.COLOR_CONFIDENCE,
                    )
                    pos[node_id_confidence] = (0.75, -y_pos)
                    activity_graph_dict[this.KEY_ROWS][-1][this.KEY_NODES].append(
                        {
                            this.KEY_ID: node_id_confidence,
                            this.KEY_ROLE: this.ROLE_AGENT,
                            this.KEY_LABEL: "confidence",
                            this.KEY_COLOR: this.COLOR_CONFIDENCE,
                        }
                    )
                    activity_graph.add_edge(
                        node_id,
                        node_id_confidence,
                        label="agent reflects on confidence level",
                    )
                    activity_graph_dict[this.KEY_EDGES].append(
                        {
                            this.KEY_FROM: node_id,
                            this.KEY_TO: node_id_confidence,
                            this.KEY_LABEL: "agent reflects on confidence level",
                        }
                    )

                    # next NODE to be assistant which reviews the confidence reflection
                    if (
                        node_counter + 1 < len(self.chat_history)
                        and self.chat_history[node_counter + 1][this.KEY_ROLE]
                        == this.ROLE_ASSISTANT
                    ):
                        c_feedback_msg = self.chat_history[node_counter + 1]
                        c_feedback_node_id = f"confidence_feedback_{node_counter}"
                        activity_graph.add_node(
                            c_feedback_node_id,
                            role=this.ROLE_ASSISTANT,
                            content=c_feedback_msg[this.KEY_CONTENT],
                            color=this.COLOR_ASSISTANT_FEEDBACK,
                        )
                        pos[c_feedback_node_id] = (1.5, -y_pos)
                        activity_graph_dict[this.KEY_ROWS][-1][this.KEY_NODES].append(
                            {
                                this.KEY_ID: c_feedback_node_id,
                                this.KEY_ROLE: this.ROLE_ASSISTANT,
                                this.KEY_LABEL: "assistant's feedback",
                                this.KEY_COLOR: this.COLOR_ASSISTANT_FEEDBACK,
                            }
                        )
                        activity_graph.add_edge(
                            node_id_confidence,
                            c_feedback_node_id,
                            label="assistant shares feedback",
                        )
                        activity_graph_dict[this.KEY_EDGES].append(
                            {
                                this.KEY_FROM: node_id_confidence,
                                this.KEY_TO: c_feedback_node_id,
                                this.KEY_LABEL: "assistant shares feedback",
                            }
                        )

                        # add node w/ status code
                        feedback_content = c_feedback_msg[this.KEY_CONTENT]
                        if feedback_content and feedback_content.startswith(
                            this.PREFIX_EXITCODE
                        ):
                            do_skip_next_msg = True

                            exit_node_id = f"feedback_output_{node_counter}"
                            (role, node_color) = this._chart_agent_chat_node_status(
                                chat_msg=c_feedback_msg,
                            )
                            activity_graph.add_node(
                                exit_node_id,
                                role=role,
                                content=feedback_content,
                                color=node_color,
                            )
                            pos[exit_node_id] = (2, -y_pos)
                            activity_graph_dict[this.KEY_ROWS][-1][
                                this.KEY_NODES
                            ].append(
                                {
                                    this.KEY_ID: exit_node_id,
                                    this.KEY_ROLE: this.ROLE_ASSISTANT,
                                    this.KEY_LABEL: role.replace("\n", " "),
                                    this.KEY_COLOR: node_color,
                                }
                            )
                            activity_graph.add_edge(
                                c_feedback_node_id, exit_node_id, label="status"
                            )
                            activity_graph_dict[this.KEY_EDGES].append(
                                {
                                    this.KEY_FROM: c_feedback_node_id,
                                    this.KEY_TO: exit_node_id,
                                    this.KEY_LABEL: "status",
                                }
                            )

                # NODE: re-planning
                if this.STR_LET_ME_REPLAN in node_content:
                    node_id_replan = f"replan_{node_counter}"
                    activity_graph.add_node(
                        node_id_replan,
                        role="replan",
                        content="Re-planning",
                        color=this.COLOR_REPLAN,
                    )
                    pos[node_id_replan] = (0.75, -y_pos)
                    activity_graph_dict[this.KEY_ROWS][-1][this.KEY_NODES].append(
                        {
                            this.KEY_ID: node_id_replan,
                            this.KEY_ROLE: this.ROLE_AGENT,
                            this.KEY_LABEL: "replan",
                            this.KEY_COLOR: this.COLOR_REPLAN,
                        }
                    )
                    activity_graph.add_edge(
                        node_id, node_id_replan, label="re-planning to new approach"
                    )
                    activity_graph_dict[this.KEY_EDGES].append(
                        {
                            this.KEY_FROM: node_id,
                            this.KEY_TO: node_id_replan,
                            this.KEY_LABEL: "re-planning to new approach",
                        }
                    )
                    previous_node = node_id_replan

                    # TOOL execution
                    if this.PKG_AGENT_TOOLS in msg.get("content", ""):
                        tool_names = re.findall(
                            r"from api_server\.agent_tools\.(\w+)",
                            msg.get("content", ""),
                        )
                        tool_name = tool_names[0] if tool_names else "Tool Execution"
                        if tool_name:
                            if tool_name not in tools_stats_dict[this.KEY_TOOLS]:
                                tools_stats_dict[this.KEY_TOOLS][tool_name] = {
                                    "name": tool_name,
                                    "success_count": 0,
                                    "failure_count": 0,
                                    "total_count": 0,
                                }

                            tool_node_id = f"tool_{node_counter}"
                            activity_graph.add_node(
                                tool_node_id, role="tool", content=tool_name
                            )
                            pos[tool_node_id] = (1.5, -y_pos)
                            activity_graph_dict[this.KEY_ROWS][-1][
                                this.KEY_NODES
                            ].append(
                                {
                                    this.KEY_ID: tool_node_id,
                                    this.KEY_ROLE: this.ROLE_AGENT,
                                    this.KEY_LABEL: f"tool: {tool_name}",
                                    this.KEY_COLOR: this.COLOR_TOOL,
                                }
                            )
                            activity_graph.add_edge(
                                previous_node,
                                tool_node_id,
                                label=(
                                    f"running "
                                    f"{tool_name.upper().replace('_', ' ')}"
                                    f" tool"
                                ),
                            )
                            activity_graph_dict[this.KEY_EDGES].append(
                                {
                                    this.KEY_FROM: previous_node,
                                    this.KEY_TO: tool_node_id,
                                    this.KEY_LABEL: (
                                        f"running "
                                        f"{tool_name.upper().replace('_', ' ')}"
                                        f" tool"
                                    ),
                                }
                            )

                            # look for tool result in the NEXT message
                            if (
                                node_counter + 1 < len(self.chat_history)
                                and this.PREFIX_EXITCODE
                                in self.chat_history[node_counter + 1][this.KEY_CONTENT]
                            ):
                                exit_node_id = f"tool_output_{node_counter}"
                                (role, node_color) = this._chart_agent_chat_node_status(
                                    chat_msg=self.chat_history[node_counter + 1],
                                    stats=tools_stats_dict[this.KEY_TOOLS][tool_name],
                                )
                                activity_graph.add_node(
                                    exit_node_id,
                                    role=role,
                                    content="",
                                    color=node_color,
                                )
                                pos[exit_node_id] = (2, -y_pos)
                                activity_graph_dict[this.KEY_ROWS][-1][
                                    this.KEY_NODES
                                ].append(
                                    {
                                        this.KEY_ID: exit_node_id,
                                        this.KEY_ROLE: this.ROLE_AGENT,
                                        this.KEY_LABEL: role.replace("\n", " "),
                                        this.KEY_COLOR: node_color,
                                    }
                                )
                                activity_graph.add_edge(
                                    tool_node_id, exit_node_id, label="status"
                                )
                                activity_graph_dict[this.KEY_EDGES].append(
                                    {
                                        this.KEY_FROM: tool_node_id,
                                        this.KEY_TO: exit_node_id,
                                        this.KEY_LABEL: "status",
                                    }
                                )

                                # SKIP the next message as it is already processed
                                do_skip_next_msg = True

                    # NODE: SCRIPT execution
                    elif (
                        "# filename: " in node_content
                        and "# execution: " in node_content
                    ):
                        # extract script name:
                        # "\n# filename: final_evaluation.py\n# execution:"
                        pattern = r"# filename: ([^\n]+)"
                        match = re.search(pattern, node_content)
                        if match:
                            script_name = match.group(1)
                            if script_name:
                                if (
                                    script_name
                                    not in scripts_stats_dict[this.KEY_SCRIPTS]
                                ):
                                    scripts_stats_dict[this.KEY_SCRIPTS][
                                        script_name
                                    ] = {
                                        "name": script_name,
                                        "success_count": 0,
                                        "failure_count": 0,
                                        "total_count": 0,
                                    }

                                script_node_id = f"script_{node_counter}"
                                activity_graph.add_node(
                                    script_node_id,
                                    role="script",
                                    content=script_name,
                                    color=this.COLOR_SCRIPT,
                                )
                                pos[script_node_id] = (1.5, -y_pos)
                                activity_graph_dict[this.KEY_ROWS][-1][
                                    this.KEY_NODES
                                ].append(
                                    {
                                        this.KEY_ID: script_node_id,
                                        this.KEY_ROLE: this.ROLE_AGENT,
                                        this.KEY_LABEL: f"script: {script_name}",
                                        this.KEY_COLOR: this.COLOR_SCRIPT,
                                    }
                                )
                                activity_graph.add_edge(
                                    node_id,
                                    script_node_id,
                                    label=f"runs {script_name} script",
                                )
                                activity_graph_dict[this.KEY_EDGES].append(
                                    {
                                        this.KEY_FROM: node_id,
                                        this.KEY_TO: script_node_id,
                                        this.KEY_LABEL: f"runs {script_name} script",
                                    }
                                )

                                # look for tool result in the NEXT message
                                if (
                                    node_counter + 1 < len(self.chat_history)
                                    and this.PREFIX_EXITCODE
                                    in self.chat_history[node_counter + 1][
                                        this.KEY_CONTENT
                                    ]
                                ):
                                    exit_node_id = f"tool_output_{node_counter}"
                                    (role, node_color) = (
                                        this._chart_agent_chat_node_status(
                                            chat_msg=self.chat_history[
                                                node_counter + 1
                                            ],
                                            stats=scripts_stats_dict[this.KEY_SCRIPTS][
                                                script_name
                                            ],
                                        )
                                    )
                                    activity_graph.add_node(
                                        exit_node_id,
                                        role=role,
                                        content="",
                                        color=node_color,
                                    )
                                    pos[exit_node_id] = (2, -y_pos)
                                    activity_graph_dict[this.KEY_ROWS][-1][
                                        this.KEY_NODES
                                    ].append(
                                        {
                                            this.KEY_ID: exit_node_id,
                                            this.KEY_ROLE: this.ROLE_AGENT,
                                            this.KEY_LABEL: role.replace("\n", " "),
                                            this.KEY_COLOR: node_color,
                                        }
                                    )
                                    activity_graph.add_edge(
                                        script_node_id, exit_node_id, label="status"
                                    )
                                    activity_graph_dict[this.KEY_EDGES].append(
                                        {
                                            this.KEY_FROM: script_node_id,
                                            this.KEY_TO: exit_node_id,
                                            this.KEY_LABEL: "status",
                                        }
                                    )

                                    # SKIP the next message as it is already processed
                                    do_skip_next_msg = True

                # NODE: TOOL execution
                elif this.PKG_AGENT_TOOLS in msg.get("content", ""):
                    tool_names = re.findall(
                        r"from api_server\.agent_tools\.(\w+)", msg.get("content", "")
                    )
                    tool_name = tool_names[0] if tool_names else "Tool Execution"
                    if tool_name:
                        if tool_name not in tools_stats_dict[this.KEY_TOOLS]:
                            tools_stats_dict[this.KEY_TOOLS][tool_name] = {
                                "name": tool_name,
                                "success_count": 0,
                                "failure_count": 0,
                                "total_count": 0,
                            }

                        tool_node_id = f"tool_{node_counter}"
                        activity_graph.add_node(
                            tool_node_id,
                            role="tool",
                            content=tool_name,
                            color=this.COLOR_TOOL,
                        )
                        pos[tool_node_id] = (1.5, -y_pos)
                        activity_graph_dict[this.KEY_ROWS][-1][this.KEY_NODES].append(
                            {
                                this.KEY_ID: tool_node_id,
                                this.KEY_ROLE: this.ROLE_AGENT,
                                this.KEY_LABEL: f"tool: {tool_name}",
                                this.KEY_COLOR: this.COLOR_TOOL,
                            }
                        )
                        activity_graph.add_edge(
                            node_id,
                            tool_node_id,
                            label=f"runs {tool_name.upper().replace('_', ' ')} tool",
                        )
                        activity_graph_dict[this.KEY_EDGES].append(
                            {
                                this.KEY_FROM: node_id,
                                this.KEY_TO: tool_node_id,
                                this.KEY_LABEL: (
                                    f"runs {tool_name.upper().replace('_', ' ')} tool"
                                ),
                            }
                        )

                        # look for tool result in the NEXT message
                        if (
                            node_counter + 1 < len(self.chat_history)
                            and this.PREFIX_EXITCODE
                            in self.chat_history[node_counter + 1][this.KEY_CONTENT]
                        ):
                            exit_node_id = f"tool_output_{node_counter}"
                            (role, node_color) = this._chart_agent_chat_node_status(
                                chat_msg=self.chat_history[node_counter + 1],
                                stats=tools_stats_dict[this.KEY_TOOLS][tool_name],
                            )
                            activity_graph.add_node(
                                exit_node_id,
                                role=role,
                                content="",
                                color=node_color,
                            )
                            pos[exit_node_id] = (2, -y_pos)
                            activity_graph_dict[this.KEY_ROWS][-1][
                                this.KEY_NODES
                            ].append(
                                {
                                    this.KEY_ID: exit_node_id,
                                    this.KEY_ROLE: this.ROLE_AGENT,
                                    this.KEY_LABEL: role.replace("\n", " "),
                                    this.KEY_COLOR: node_color,
                                }
                            )
                            activity_graph.add_edge(
                                tool_node_id, exit_node_id, label="status"
                            )
                            activity_graph_dict[this.KEY_EDGES].append(
                                {
                                    this.KEY_FROM: tool_node_id,
                                    this.KEY_TO: exit_node_id,
                                    this.KEY_LABEL: "status",
                                }
                            )

                            # SKIP the next message as it is already processed
                            do_skip_next_msg = True

                # SCRIPT execution
                elif "# filename: " in node_content and "# execution: " in node_content:
                    # extract script name:
                    # "\n# filename: final_evaluation.py\n# execution:"
                    pattern = r"# filename: ([^\n]+)"
                    match = re.search(pattern, node_content)
                    if match:
                        script_name = match.group(1)
                        if script_name:
                            if script_name not in scripts_stats_dict[this.KEY_SCRIPTS]:
                                scripts_stats_dict[this.KEY_SCRIPTS][script_name] = {
                                    "name": script_name,
                                    "success_count": 0,
                                    "failure_count": 0,
                                    "total_count": 0,
                                }

                            script_node_id = f"script_{node_counter}"
                            activity_graph.add_node(
                                script_node_id,
                                role="script",
                                content=script_name,
                                color=this.COLOR_SCRIPT,
                            )
                            pos[script_node_id] = (1.5, -y_pos)
                            activity_graph_dict[this.KEY_ROWS][-1][
                                this.KEY_NODES
                            ].append(
                                {
                                    this.KEY_ID: script_node_id,
                                    this.KEY_ROLE: this.ROLE_AGENT,
                                    this.KEY_LABEL: f"script: {script_name}",
                                    this.KEY_COLOR: this.COLOR_SCRIPT,
                                }
                            )
                            activity_graph.add_edge(
                                node_id,
                                script_node_id,
                                label=f"runs {script_name} script",
                            )
                            activity_graph_dict[this.KEY_EDGES].append(
                                {
                                    this.KEY_FROM: node_id,
                                    this.KEY_TO: script_node_id,
                                    this.KEY_LABEL: f"runs {script_name} script",
                                }
                            )

                            # look for tool result in the NEXT message
                            if (
                                node_counter + 1 < len(self.chat_history)
                                and this.PREFIX_EXITCODE
                                in self.chat_history[node_counter + 1][this.KEY_CONTENT]
                            ):
                                exit_node_id = f"script_output_{node_counter}"
                                (role, node_color) = this._chart_agent_chat_node_status(
                                    chat_msg=self.chat_history[node_counter + 1],
                                    stats=scripts_stats_dict[this.KEY_SCRIPTS][
                                        script_name
                                    ],
                                )
                                activity_graph.add_node(
                                    exit_node_id,
                                    role=role,
                                    content="",
                                    color=node_color,
                                )
                                pos[exit_node_id] = (2, -y_pos)
                                activity_graph_dict[this.KEY_ROWS][-1][
                                    this.KEY_NODES
                                ].append(
                                    {
                                        this.KEY_ID: exit_node_id,
                                        this.KEY_ROLE: this.ROLE_AGENT,
                                        this.KEY_LABEL: role.replace("\n", " "),
                                        this.KEY_COLOR: node_color,
                                    }
                                )
                                activity_graph.add_edge(
                                    script_node_id, exit_node_id, label="status"
                                )
                                activity_graph_dict[this.KEY_EDGES].append(
                                    {
                                        this.KEY_FROM: script_node_id,
                                        this.KEY_TO: exit_node_id,
                                        this.KEY_LABEL: "status",
                                    }
                                )

                                # SKIP the next message as it is already processed
                                do_skip_next_msg = True

                # check for a FINAL answer or constrained output
                elif node_content and node_content.startswith(this.STR_CONSTRAINED_OUT):
                    # if it's the last message, add a distinct FINAL ANSWER node
                    if node_counter == len(self.chat_history) - 1:
                        result_node_id = f"result_{node_counter}"
                        activity_graph.add_node(
                            result_node_id,
                            role="FINAL\nanswer",
                            content="FINAL answer",
                            color=this.COLOR_ANSWER,
                        )
                        pos[result_node_id] = (1.5, -y_pos)
                        activity_graph_dict[this.KEY_ROWS][-1][this.KEY_NODES].append(
                            {
                                this.KEY_ID: result_node_id,
                                this.KEY_ROLE: this.ROLE_AGENT,
                                this.KEY_LABEL: "final answer",
                                this.KEY_COLOR: this.COLOR_ANSWER,
                            }
                        )
                        activity_graph.add_edge(
                            node_id, result_node_id, label="generates FINAL answer"
                        )
                        activity_graph_dict[this.KEY_EDGES].append(
                            {
                                this.KEY_FROM: node_id,
                                this.KEY_TO: result_node_id,
                                this.KEY_LABEL: "generates FINAL answer",
                            }
                        )
                    else:
                        result_node_id = f"result_{node_counter}"
                        activity_graph.add_node(
                            result_node_id,
                            role="draft\nanswer",
                            content="INTERMEDIARY answer",
                            color=this.COLOR_ANSWER_DRAFT,
                        )
                        pos[result_node_id] = (0.75, -y_pos)
                        activity_graph_dict[this.KEY_ROWS][-1][this.KEY_NODES].append(
                            {
                                this.KEY_ID: result_node_id,
                                this.KEY_ROLE: this.ROLE_AGENT,
                                this.KEY_LABEL: "intermediary answer",
                                this.KEY_COLOR: this.COLOR_ANSWER_DRAFT,
                            }
                        )
                        activity_graph.add_edge(
                            node_id,
                            result_node_id,
                            label="generates INTERMEDIARY ANSWER",
                        )
                        activity_graph_dict[this.KEY_EDGES].append(
                            {
                                this.KEY_FROM: node_id,
                                this.KEY_TO: result_node_id,
                                this.KEY_LABEL: "generates INTERMEDIARY ANSWER",
                            }
                        )

                        # link next assistant message as FEEDBACK to INTERMEDIARY answer
                        if (
                            node_counter + 1 < len(self.chat_history)
                            and self.chat_history[node_counter + 1][this.KEY_ROLE]
                            == this.ROLE_ASSISTANT
                        ):
                            feedback_msg = self.chat_history[node_counter + 1]
                            feedback_node_id = f"feedback_{node_counter}"
                            activity_graph.add_node(
                                feedback_node_id,
                                role="assistant's\nfeedback",
                                content=feedback_msg[this.KEY_CONTENT],
                                color=this.COLOR_ASSISTANT_FEEDBACK,
                            )
                            pos[feedback_node_id] = (1.5, -y_pos)
                            activity_graph_dict[this.KEY_ROWS][-1][
                                this.KEY_NODES
                            ].append(
                                {
                                    this.KEY_ID: feedback_node_id,
                                    this.KEY_ROLE: this.ROLE_ASSISTANT,
                                    this.KEY_LABEL: "assistant's feedback",
                                    this.KEY_COLOR: this.COLOR_ASSISTANT_FEEDBACK,
                                }
                            )
                            activity_graph.add_edge(
                                result_node_id,
                                feedback_node_id,
                                label="assistant shares feedback",
                            )
                            activity_graph_dict[this.KEY_EDGES].append(
                                {
                                    this.KEY_FROM: result_node_id,
                                    this.KEY_TO: feedback_node_id,
                                    this.KEY_LABEL: "assistant shares feedback",
                                }
                            )

                            # add node w/ status code color
                            feedback_content = feedback_msg[this.KEY_CONTENT]
                            if feedback_content and feedback_content.startswith(
                                this.PREFIX_EXITCODE
                            ):
                                do_skip_next_msg = True

                                exit_node_id = f"feedback_output_{node_counter}"
                                (role, node_color) = this._chart_agent_chat_node_status(
                                    chat_msg=feedback_msg,
                                )
                                activity_graph.add_node(
                                    exit_node_id,
                                    role=role,
                                    content=feedback_content,
                                    color=node_color,
                                )
                                pos[exit_node_id] = (2, -y_pos)
                                activity_graph_dict[this.KEY_ROWS][-1][
                                    this.KEY_NODES
                                ].append(
                                    {
                                        this.KEY_ID: exit_node_id,
                                        this.KEY_ROLE: this.ROLE_ASSISTANT,
                                        this.KEY_LABEL: role.replace("\n", " "),
                                        this.KEY_COLOR: node_color,
                                    }
                                )
                                activity_graph.add_edge(
                                    feedback_node_id, exit_node_id, label="status"
                                )
                                activity_graph_dict[this.KEY_EDGES].append(
                                    {
                                        this.KEY_FROM: feedback_node_id,
                                        this.KEY_TO: exit_node_id,
                                        this.KEY_LABEL: "status",
                                    }
                                )
                            else:
                                # if content starts w/ <current_date>, it is re-start
                                node_id = feedback_node_id
                                pass

            y_pos += 1.5
            previous_node = node_id
            node_counter += 1

        # add the final edge to the 'End' node
        activity_graph.add_edge(previous_node, this.NODE_END_ID, label="")
        pos[this.NODE_START_ID] = (0, 1)
        pos[this.NODE_END_ID] = (0, -y_pos)

        # finish the activity graph dict
        activity_graph_dict[this.KEY_EDGES].append(
            {
                this.KEY_FROM: previous_node,
                this.KEY_TO: this.NODE_END_ID,
                this.KEY_LABEL: "",
            }
        )
        activity_graph_dict[this.KEY_ROWS].append(
            {
                # EXCLUDE: this.KEY_ROW: len(activity_graph_dict[this.KEY_ROWS]) + 1,
                this.KEY_NODES: [
                    {
                        this.KEY_ID: this.NODE_END_ID,
                        this.KEY_ROLE: this.ROLE_RUNTIME,
                        this.KEY_LABEL: "end",
                        this.KEY_COLOR: this.COLOR_END,
                    }
                ],
            }
        )
        new_rows = []
        row_counter = 1
        for r in activity_graph_dict[this.KEY_ROWS]:
            if r[this.KEY_NODES]:
                # EXCLUDE r[this.KEY_ROW] = row_counter
                new_rows.append(r)
                row_counter += 1
        activity_graph_dict[this.KEY_ROWS] = new_rows

        self._uuidize_activity_graph_json(activity_graph_dict)

        # get node and edge attributes for visualization
        node_colors = [
            activity_graph.nodes[n].get("color", "lightblue")
            for n in activity_graph.nodes
        ]
        edge_labels = nx.get_edge_attributes(activity_graph, "label")
        # DRAW the graph
        plt.figure(figsize=(15, y_pos + 5))
        nx.draw(
            activity_graph,
            pos,
            with_labels=False,
            node_color=node_colors,
            node_size=3500,
        )
        nx.draw_networkx_labels(
            activity_graph,
            pos,
            labels={
                n: activity_graph.nodes[n][this.KEY_ROLE] for n in activity_graph.nodes
            },
            font_size=10,
        )
        nx.draw_networkx_edge_labels(
            activity_graph, pos, edge_labels=edge_labels, font_size=8
        )

        # SAVE
        if base_dir:
            # ACTIVITY flow chart
            activity_png_path = base_dir / "sequential_agent_run.png"
            plt.title("Agent Execution Visualization (Sequential)", size=20)
            plt.axis("off")
            plt.savefig(activity_png_path)

            # TOOLS bar chart
            tools_png_path = self._chart_agent_chat_tool_column_chart(
                data=tools_stats_dict,
                base_dir=base_dir,
                entity_key="tools",
                entity_name="Tools",
            )
            # SCRIPTS bar chart
            scripts_png_path = self._chart_agent_chat_tool_column_chart(
                data=scripts_stats_dict,
                base_dir=base_dir,
                entity_key="scripts",
                entity_name="Scripts",
            )

        return (
            activity_graph_dict,
            tools_stats_dict,
            scripts_stats_dict,
            activity_png_path,
            tools_png_path,
            scripts_png_path,
        )
