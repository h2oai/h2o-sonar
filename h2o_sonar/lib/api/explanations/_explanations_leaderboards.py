# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import abc
import collections
import copy
import datetime
import json
import math
import os
import pathlib
import sys
import traceback
import uuid

import airium
import numpy

from h2o_sonar import loggers
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import insights
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api import problems as p6s
from h2o_sonar.lib.api.explanations import _explanations_base
from h2o_sonar.utils import sanitization
from h2o_sonar.utils import tokenization


class DurationStatsKey:
    """Performance statistics keys."""

    N = "n"
    MIN = "min"
    MAX = "max"
    AVG = "avg"
    SUM = "sum"


class LlmLeaderboardExplanation:
    @staticmethod
    def get_leaderboard_data_path(
        evaluation,
        evaluator_id,
        explanation_format: str = commons.MimeType.MIME_JSON,
        metric: str = f5s.LlmLeaderboardJSonFormat.KEY_ALL_METRICS,
    ):
        if not evaluation:
            raise ValueError("Evaluation is not provided")
        if not evaluator_id:
            raise ValueError("Evaluator ID is not provided")
        if not evaluation.result.get_evaluator_jobs():
            raise ValueError("Evaluation doesn't have evaluator jobs")

        evaluator_job = None
        for j in evaluation.result.get_evaluator_jobs():
            if j.explainer_descriptor.id == evaluator_id:
                evaluator_job = j
                break
        if not evaluator_job:
            raise ValueError(
                f"Evaluation doesn't have evaluator job for the evaluator: "
                f"'{evaluator_id}'"
            )

        e_persistence = persistences.ExplainerPersistence(
            data_dir=evaluation.result.results_location,
            mli_key=evaluation.key,
            username=commons.DEFAULT_USER,
            explainer_id=evaluator_job.explainer_descriptor.id,
            explainer_job_key=evaluator_job.key,
        )

        leaderboard_types = [
            LlmBoolLeaderboardExplanation,
            LlmHeatmapLeaderboardExplanation,
            LlmClassifierLeaderboardExplanation,
            LlmProcedureEvalLeaderboardExplanation,
        ]
        for lt in leaderboard_types:
            leaderboard_idx_path = e_persistence.get_explanation_file_path(
                explanation_type=lt.explanation_type(),
                explanation_format=explanation_format,
            )
            if pathlib.Path(leaderboard_idx_path).exists():
                # FORMAT: Markdown
                if explanation_format in [
                    f5s.MarkdownFormat.mime,
                    f5s.EvalStudioMarkdownFormat.mime,
                ]:
                    return leaderboard_idx_path

                # FORMAT: JSon
                idx_dict: dict = f5s.GlobalFeatImpJSonFormat.load_index_file(
                    persistence=e_persistence,
                    explanation_type=lt.explanation_type(),
                )
                if not idx_dict:
                    raise RuntimeError(
                        f"Leaderboard {explanation_format} explanation/index file "
                        f"empty: {leaderboard_idx_path}"
                    )

                idx_files = idx_dict.get(f5s.GlobalDtJSonFormat.KEY_FILES, {})
                if not idx_files:
                    raise RuntimeError(
                        f"Leaderboard {explanation_format} index file "
                        f"has no data files: {leaderboard_idx_path}"
                    )

                data_file_name = idx_files.get(metric, None)
                if not data_file_name:
                    raise RuntimeError(
                        f"Leaderboard {explanation_format} index file "
                        f"empty - no data file for metric '{metric}': "
                        f"{leaderboard_idx_path}"
                    )

                data_file_path = os.path.join(
                    e_persistence.get_explanation_dir_path(
                        explanation_type=lt.explanation_type(),
                        explanation_format=explanation_format,
                    ),
                    data_file_name,
                )
                if not pathlib.Path(data_file_path).exists():
                    raise RuntimeError(
                        f"Leaderboard {explanation_format} (data) file not found"
                    )

                return data_file_path

        raise RuntimeError(
            f"Leaderboard {explanation_format} explanation file not found for any "
            f"known leaderboard type: {leaderboard_types}"
        )

    @staticmethod
    def _aa_meta_color(value: float):
        """Generates a color between #FFFFFF and RGB color based on the given value."""
        # rgb = [242, 167,193] # #f2a7c1 from red palette
        rgb = [232, 100, 146]
        percentage = value * 100.0

        red = int(255 - (percentage * (255 - rgb[0])) / 100.0)
        green = int(255 - (percentage * (255 - rgb[1])) / 100.0)
        blue = int(255 - (percentage * (255 - rgb[2])) / 100.0)

        return f"#{red:02x}{green:02x}{blue:02x}"

    @staticmethod
    def _err_html_msg_from_tokenization(actual_output_meta):
        if actual_output_meta:
            kv = actual_output_meta[0]
            if kv.data and kv.data[0].meta:
                return kv.data[0].meta.get(tokenization.META_ERR_MSG_HTML, None)

        return None

    @staticmethod
    def _html_aa_meta(
        html_src,
        actual_output,
        actual_output_meta,
        metrics_meta,
        err_msg="",
        metrics=None,
        is_bool_leaderboard=False,
    ):
        if actual_output_meta:
            if is_bool_leaderboard:
                primary_metric = metrics_meta.get_primary_metric().key
                for kv in actual_output_meta:
                    if isinstance(kv, tokenization.Tokenization):
                        with html_src.li():
                            html_src.b(_t=f"Actual output ({kv.tokenization}): ")

                            for v in kv.data:
                                if v.metrics:
                                    score = v.metrics.get(primary_metric, None)
                                    if score == 0.0:
                                        color = AbcHeatmapExplanation.PALETTE_RED[1]
                                    elif score == 1.0:
                                        color = AbcHeatmapExplanation.PALETTE_GREEN[-1]
                                    else:
                                        color = "#ffffff"

                                    title = (
                                        f"{primary_metric}"
                                        f"={score if score is not None else ''}"
                                    )
                                    justification = v.meta.get(
                                        "evaluation_justification"
                                    )
                                    if justification:
                                        title += f" | justification: {justification}"

                                    html_src.span(
                                        _t=v.text,
                                        style=f"background-color: #{color};",
                                        title=title,
                                    )
                                else:
                                    html_src.span(_t=v.text)
            else:  # heatmap leaderboard
                metric = next(iter(metrics.keys()))
                for m in metrics.keys():
                    if m in err_msg:  # TODO this is a quick fix, must be more robust
                        metric = m
                        break

                for kv in actual_output_meta:
                    if isinstance(kv, tokenization.Tokenization):
                        with html_src.li():
                            html_src.b(_t=f"Actual output ({kv.tokenization}): ")

                            metrics = [
                                v.metrics[metric]
                                for v in kv.data
                                if metric in v.metrics
                            ]

                            if metrics:
                                (minm, maxm) = metrics_meta.get_metric(
                                    metric
                                ).value_range
                                if math.isinf(maxm):
                                    maxm = max(metrics)
                                if math.isinf(minm):
                                    minm = min(metrics)
                            else:
                                minm = 0.0
                                maxm = 0.0

                            range_value = maxm - minm
                            for v in kv.data:
                                if minm != maxm:
                                    score = v.metrics[metric]
                                    pct = (score - minm) / range_value
                                    if metrics_meta.is_higher_better(metric):
                                        pct = 1.0 - pct

                                    color = LlmLeaderboardExplanation._aa_meta_color(
                                        pct
                                    )

                                    title = f"{metric}={v.metrics[metric]}"
                                    justification = v.meta.get(
                                        "evaluation_justification"
                                    )
                                    if justification:
                                        title += f" | justification: {justification}"

                                    html_src.span(
                                        _t=v.text,
                                        style=f"background-color: {color};",
                                        title=title,
                                    )
                                else:
                                    html_src.span(_t=v.text)

        else:
            v = actual_output
            if v:
                with html_src.li():
                    html_src.b(_t="Actual output: ")
                    html_src(f"{v}")

    @staticmethod
    def markdown_connection_stats_table(
        evaluated_models_list: list[models.ExplainableLlmModel],
    ):
        md = ""
        md += "\n"
        md += "## Model Connection Details\n"
        md += "\n"
        md += (
            "| Name | Successful | Failed | Retries | Timeouts | TPS | "
            "Min Request | Max Request | Avg Request | Total |\n"
        )
        md += "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | \n"

        for em in evaluated_models_list:
            meta = em.llm_model_meta
            if em and meta and isinstance(meta, dict):
                successful = meta.get(models.ExplainableLlmModel.KEY_STATS_SUCCESS, 0)
                failed = meta.get(models.ExplainableLlmModel.KEY_STATS_FAILURE, 0)
                retries = meta.get(models.ExplainableLlmModel.KEY_STATS_RETRY, 0)
                timeouts = meta.get(models.ExplainableLlmModel.KEY_STATS_TIMEOUT, 0)
                h2ogpte_stats = meta.get(
                    models.ExplainableLlmModel.KEY_H2OGPTE_STATS, {}
                )
                tps = 0.0
                if h2ogpte_stats:
                    tps = h2ogpte_stats.get("tokens_per_second", 0.0)

                duration_stats = meta.get(
                    models.ExplainableLlmModel.KEY_STATS_DURATION, {}
                )
                min_request = 0.0
                max_request = 0.0
                avg_request = 0.0
                total = 0.0
                n = successful + failed
                if n == 0:
                    n = 1
                if duration_stats:
                    min_request = duration_stats.get(DurationStatsKey.MIN, 0.0)
                    max_request = duration_stats.get(DurationStatsKey.MAX, 0.0)
                    avg_request = duration_stats.get(DurationStatsKey.AVG, 0.0)
                    total = duration_stats.get(DurationStatsKey.SUM, 0.0)
                md += (
                    f"| {em.llm_model_name} "
                    f"| {successful} ({successful / n:.0%}) "
                    f"| {failed} ({failed / n:.0%}) "
                    f"| {retries} ({retries / n:.0%}) "
                    f"| {timeouts} ({timeouts / n:.0%}) "
                    f"| {tps:.1f} "
                    f"| {min_request:.2f}s "
                    f"| {max_request:.2f}s "
                    f"| {avg_request:.2f}s "
                    f"| {total:.2f}s "
                    "|\n"
                )
        md += "\n"
        return md


class AbcHeatmapExplanation(abc.ABC):
    METRIC_ALL = "ALL_METRICS"  # symbolic key for all metrics @ JSon representation

    PALETTE_GREEN = [
        "40a481",
        "56b896",
        "71c9ab",
        "8ad9be",
        "aaebd5",
    ]
    PALETTE_BLUE = [
        "3d83ad",
        "5e9dc3",
        "96bcd3",
        "c4dcea",
        "eef4f8",
    ]
    PALETTE_RED = ["f2a7c1", "f6bbd0", "f7cfde", "fae5ed", "fdf3f7"]
    COLOR_FATAL_ERROR = "ff0000"


class LlmBoolLeaderboardExplanation(
    _explanations_base.Explanation, LlmLeaderboardExplanation, AbcHeatmapExplanation
):
    """LLM failure leaderboard - leaderboard data and formats for metrics which is of
    the BOOLEAN type i.e. it is possible to infer:

    - success / failure
    - pass / fail
    - true / false

    for each test case (prompt + model) in the test set.

    Leaderboard provides multiple aspects of the test results (sub-leaderboards):

    - summary leaderboard
    - most problematic prompts leaderboard

    Multiple leaderboards within a format are supported via index file:

    - index file:
        - key: leaderboard name
        - value: leaderboard file name

    """

    _explanation_type = "llm-bool-leaderboard"
    _is_global = True

    Failure = collections.namedtuple(
        "Failure",
        [
            "doc_url",
            "error_message",
            "input",
            "expected_output",
            "output_condition",
            "output_constraints",
            "actual_output",
            "actual_output_meta",
            "fail_retrieval",
            "fail_generation",
            "fail_parse",
            "ctx_bytes",
            "ctx_chunks",
            "row_key",
            "model_key",
        ],
    )

    @staticmethod
    def key_2_rag_type_prefix(evaluated_models) -> dict:
        rag_type_prefix = {}

        for m in evaluated_models:
            evaluated_model_type = m.model_type

            t_model_type = models.ExplainableModelType
            if evaluated_model_type == models.ExplainableModelType.h2ogpte:
                rag_type_prefix[m.llm_model_name] = "h2oGPTe RAG"
            elif evaluated_model_type == models.ExplainableModelType.h2ogpte_llm:
                rag_type_prefix[m.llm_model_name] = "h2oGPTe LLM"
            elif evaluated_model_type == models.ExplainableModelType.h2ogpt:
                rag_type_prefix[m.llm_model_name] = "h2oGPT LLM"
            elif evaluated_model_type == models.ExplainableModelType.h2ollmops:
                rag_type_prefix[m.llm_model_name] = "H2O LLMOps"
            elif evaluated_model_type == models.ExplainableModelType.ollama:
                rag_type_prefix[m.llm_model_name] = "ollama"
            elif evaluated_model_type == models.ExplainableModelType.openai_rag:
                rag_type_prefix[m.llm_model_name] = "OpenAI RAG"
            elif evaluated_model_type == models.ExplainableModelType.openai_llm:
                rag_type_prefix[m.llm_model_name] = "OpenAI LLM"
            elif evaluated_model_type == t_model_type.azure_openai_llm:
                rag_type_prefix[m.llm_model_name] = "Azure OpenAI LLM"
            elif isinstance(m, models.ExplainableRagModel):
                rag_type_prefix[m.llm_model_name] = "RAG"
            else:
                # else unknown LLM/RAG type
                rag_type_prefix[m.llm_model_name] = "LLM"

        return rag_type_prefix

    def __init__(
        self,
        evaluator,
        metrics_meta: commons.MetricsMeta,
        display_name: str = None,
        display_category: str = None,
        key_2_evaluated_model: dict = None,
        llm_host: commons.LlmModelHostType = commons.LlmModelHostType.RAG,
        do_eval_rc: bool = False,
        logger=None,
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=evaluator,
            display_name=display_name,
            display_category=display_category,
        )

        # metrics metadata:
        self.metrics_meta = metrics_meta

        # retrieved context evaluation
        self.do_eval_rc = do_eval_rc

        # LLM@RAG/LLM models: key -> model
        self.key_2_evaluated_model = key_2_evaluated_model
        self.llm_host = llm_host

        # MODEL passes map: base model name -> count
        self.m_passes_count = {}
        self.m_passes_generation_count = {}
        # MODEL failures
        self.m_failures = {}  # map: base model name -> list[Failure]
        self.m_failures_retrieval = {}
        self.m_failures_generation = {}
        self.m_failures_parse = {}

        # MODEL failures count: base model name -> count
        self.m_failures_count = {}
        self.m_failures_retrieval_count = {}
        self.m_failures_generation_count = {}
        self.m_failures_parse_count = {}

        # INPUT passes map: prompt -> count
        self.i_passes_count = {}
        self.i_passes_generation_count = {}
        # INPUT failures map: prompt -> list of failures
        #  { doc URL, error message, input }
        self.i_failures = {}
        self.i_failures_retrieval = {}
        self.i_failures_generation = {}
        self.i_failures_parse = {}

        # INPUT failures count: prompt -> count
        self.i_failures_count = {}
        self.i_failures_retrieval_count = {}
        self.i_failures_generation_count = {}
        self.i_failures_parse_count = {}

        # EMPTY CONTEXT map: prompt -> count
        self.ctx_empty_count = {}

        # total time: base model name -> time
        self.total_time = {}
        # total cost: base model name -> cost
        self.total_cost = {}
        # leaderboard order by a column: list of base model names
        self.models_leaderboard_order = list(
            {rm.llm_model_name for rm in self.key_2_evaluated_model.values()}
        )
        # prompts leaderboard order by number of failures
        self.inputs_leaderboard_order = []

        # palette from dark green to dark red by 10%
        self.palette = LlmBoolLeaderboardExplanation.PALETTE_RED.copy()
        r = LlmBoolLeaderboardExplanation.PALETTE_GREEN.copy()
        r.reverse()
        self.palette.extend(r)

        # tested LLM host (h2oGPTe/OpenAI RAG/...) to prefix LLMs in leaderboards
        # map: model name -> prefix
        if key_2_evaluated_model:
            self.rag_type_prefix = LlmBoolLeaderboardExplanation.key_2_rag_type_prefix(
                key_2_evaluated_model.values()
            )
        else:
            self.rag_type_prefix = {}

        self.logger = logger or loggers.SonarPrintLogger()

        self._contains_parse_failures: bool | None = None

        # leaderboard as dict used for JSon representation (cache)
        self.leaderboard_data_dict = None

        # insights
        self.insight_most_acc = None
        self.insight_least_acc = None
        self.insight_cheapest_m = None
        self.insight_expensive_m = None
        self.insight_fastest_m = None
        self.insight_slowest_m = None
        self.insight_best_retrieval_m = None
        self.insight_worst_retrieval_m = None
        self.insight_problematic_i = None

        # problems
        self.problems_with_retrieval_i = []

    def __str__(self) -> str:
        return str(self.as_dict())

    def validate(self) -> bool:
        return self._formats is not None

    def evaluation_cost(self):
        """Total evaluation cost."""
        return sum(self.total_cost.values())

    def _has_parse_fails(self):
        if self._contains_parse_failures is None:
            if any(f.fail_parse for fs in self.m_failures.values() for f in fs):
                self._contains_parse_failures = True
                return True
            self._contains_parse_failures = False
            return False
        return self._contains_parse_failures

    @staticmethod
    def _add_failure_2_dict(
        fail_dict: dict,
        fail_dict_key: str,
        new_failure,
    ):
        if fail_dict_key not in fail_dict:
            fail_dict[fail_dict_key] = []

        fail_dict[fail_dict_key].append(new_failure)

    @staticmethod
    def _inc_fail_count(fail_counter_dict: dict, fail_counter_dict_key: str):
        if fail_counter_dict_key not in fail_counter_dict:
            fail_counter_dict[fail_counter_dict_key] = 1
        else:
            fail_counter_dict[fail_counter_dict_key] = (
                fail_counter_dict[fail_counter_dict_key] + 1
            )

    def add_failure(
        self,
        llm_model_name: str,
        doc_url,
        error_message: str,
        i: str,
        context: list[str] | None,
        expected_output: str,
        output_constraints: list | None,
        output_condition: str,
        actual_output: str,
        actual_output_meta: list | None,
        duration,
        cost,
        fail_retrieval: bool = False,
        fail_generation: bool = False,
        fail_parse: bool = False,
        row_key: str = None,
        model_key: str = "",
    ):
        if self.llm_host == commons.LlmModelHostType.SERVICE:
            fail_retrieval = False

        ctx_bytes = 0
        ctx_chunks = 0
        if context:
            if isinstance(context, list):
                ctx_chunks = len(context)
                ctx_bytes = sum([len(c) for c in context])
            elif isinstance(context, str):
                ctx_chunks = 1
                ctx_bytes = len(context)

        new_failure = LlmBoolLeaderboardExplanation.Failure(
            doc_url=doc_url,
            error_message=error_message,
            input=i,
            expected_output=expected_output,
            output_constraints=output_constraints,
            output_condition=output_condition,
            actual_output=actual_output,
            actual_output_meta=actual_output_meta,
            fail_retrieval=fail_retrieval,
            fail_generation=fail_generation,
            fail_parse=fail_parse,
            ctx_bytes=ctx_bytes,
            ctx_chunks=ctx_chunks,
            row_key=row_key,
            model_key=model_key,
        )

        #
        # model failures
        #

        LlmBoolLeaderboardExplanation._add_failure_2_dict(
            fail_dict=self.m_failures,
            fail_dict_key=llm_model_name,
            new_failure=new_failure,
        )
        if fail_retrieval:
            LlmBoolLeaderboardExplanation._add_failure_2_dict(
                fail_dict=self.m_failures_retrieval,
                fail_dict_key=llm_model_name,
                new_failure=new_failure,
            )
        if fail_generation:
            LlmBoolLeaderboardExplanation._add_failure_2_dict(
                fail_dict=self.m_failures_generation,
                fail_dict_key=llm_model_name,
                new_failure=new_failure,
            )
        else:
            self._add_pass_generation(llm_model_name)
            self._add_i_pass_generation(i)
        if fail_parse:
            LlmBoolLeaderboardExplanation._add_failure_2_dict(
                fail_dict=self.m_failures_parse,
                fail_dict_key=llm_model_name,
                new_failure=new_failure,
            )

        LlmBoolLeaderboardExplanation._inc_fail_count(
            fail_counter_dict=self.m_failures_count,
            fail_counter_dict_key=llm_model_name,
        )
        if fail_retrieval:
            LlmBoolLeaderboardExplanation._inc_fail_count(
                fail_counter_dict=self.m_failures_retrieval_count,
                fail_counter_dict_key=llm_model_name,
            )
        if fail_generation:
            LlmBoolLeaderboardExplanation._inc_fail_count(
                fail_counter_dict=self.m_failures_generation_count,
                fail_counter_dict_key=llm_model_name,
            )
        if fail_parse:
            LlmBoolLeaderboardExplanation._inc_fail_count(
                fail_counter_dict=self.m_failures_parse_count,
                fail_counter_dict_key=llm_model_name,
            )

        #
        # prompt failures
        #

        LlmBoolLeaderboardExplanation._add_failure_2_dict(
            fail_dict=self.i_failures, fail_dict_key=i, new_failure=new_failure
        )
        if fail_retrieval:
            LlmBoolLeaderboardExplanation._add_failure_2_dict(
                fail_dict=self.i_failures_retrieval,
                fail_dict_key=i,
                new_failure=new_failure,
            )
        if fail_generation:
            LlmBoolLeaderboardExplanation._add_failure_2_dict(
                fail_dict=self.i_failures_generation,
                fail_dict_key=i,
                new_failure=new_failure,
            )
        if fail_parse:
            LlmBoolLeaderboardExplanation._add_failure_2_dict(
                fail_dict=self.i_failures_parse,
                fail_dict_key=i,
                new_failure=new_failure,
            )

        LlmBoolLeaderboardExplanation._inc_fail_count(
            fail_counter_dict=self.i_failures_count,
            fail_counter_dict_key=i,
        )
        if fail_retrieval:
            LlmBoolLeaderboardExplanation._inc_fail_count(
                fail_counter_dict=self.i_failures_retrieval_count,
                fail_counter_dict_key=i,
            )
        if fail_generation:
            LlmBoolLeaderboardExplanation._inc_fail_count(
                fail_counter_dict=self.i_failures_generation_count,
                fail_counter_dict_key=i,
            )
        if fail_parse:
            LlmBoolLeaderboardExplanation._inc_fail_count(
                fail_counter_dict=self.i_failures_parse_count,
                fail_counter_dict_key=i,
            )

        if i not in self.inputs_leaderboard_order:
            self.inputs_leaderboard_order.append(i)

        #
        # prompts empty context
        #
        if not context:
            LlmBoolLeaderboardExplanation._inc_fail_count(
                fail_counter_dict=self.ctx_empty_count,
                fail_counter_dict_key=i,
            )

        #
        # metadata
        #

        self.add_total_time(llm_model_name, duration)
        cost = self.check_and_report_negative_cost(
            cost=cost,
            llm_model_name=llm_model_name,
            i=i,
            row_key=row_key,
            model_key=model_key,
        )
        self.add_total_cost(llm_model_name, cost)

    def check_and_report_negative_cost(
        self, cost: float, llm_model_name, i: str, row_key: str, model_key: str
    ) -> float:
        """Create a problem for negative cost."""
        if cost < 0.0:
            html = airium.Airium()
            with html.b(klass="w3-black"):
                html("Negative cost")
            html("&nbsp;for the model ")
            with html.code():
                html(llm_model_name)
            html(" evaluator ")
            with html.code():
                html(self.explainer.display_name)
            html(" and prompt ")
            with html.i():
                html(i)
            html(".")

            self.explainer.add_problem(
                p6s.ProblemAndAction(
                    description=(
                        f"Negative cost {cost} for the model '{llm_model_name}', "
                        f"evaluator '{self.explainer.display_name}' and prompt '{i}'."
                    ),
                    description_html=html,
                    # IMPROVE description_html=html,
                    problem_type="cost",
                    problem_attrs={
                        p6s.ProblemAndAction.ATTR_COST: cost,
                        p6s.ProblemAndAction.ATTR_MODEL_NAME: llm_model_name,
                        p6s.ProblemAndAction.ATTR_ROW_KEYS: [(row_key, model_key)],
                        # input dataset ~ test lab ~ key is the test case key
                        p6s.ProblemAndAction.ATTR_TEST_CASE_KEYS: [row_key],
                    },
                    severity=p6s.ProblemSeverity.low,
                    actions_description=(
                        "The cost provided by the LLM model client is incorrect - "
                        "negative number. Check the bug tracker, vendor, server and "
                        "client code. The cost has been changed to 0.0."
                    ),
                    evaluator_id=self.explainer.explainer_id(),
                    evaluator_name=self._display_name,
                    explanation_type=self.explanation_type(),
                    explanation_name=LlmBoolLeaderboardExplanation.__name__,
                    explanation_mime=f5s.HtmlFormat.mime,
                    resources=[],
                )
            )
            cost = 0.0
        return cost

    def add_pass(
        self,
        llm_model_name: str,
        i: str,
        context: list[str] | None,
        duration: float,
        cost: float,
        row_key: str,
        model_key: str = "",
    ):
        if llm_model_name not in self.m_passes_count:
            self.m_passes_count[llm_model_name] = 1
        else:
            self.m_passes_count[llm_model_name] += 1

        if i not in self.i_passes_count:
            self.i_passes_count[i] = 1
        else:
            self.i_passes_count[i] += 1

        self._add_i_pass_generation(i)

        self.add_total_time(llm_model_name, duration)
        cost = self.check_and_report_negative_cost(
            cost=cost,
            llm_model_name=llm_model_name,
            i=i,
            row_key=row_key,
            model_key=model_key,
        )
        self.add_total_cost(llm_model_name, cost)

        self._add_pass_generation(llm_model_name)

        #
        # prompts empty context
        #
        if not context:
            LlmBoolLeaderboardExplanation._inc_fail_count(
                fail_counter_dict=self.ctx_empty_count,
                fail_counter_dict_key=i,
            )

    def _add_pass_generation(self, llm_model_name: str):
        if llm_model_name not in self.m_passes_generation_count:
            self.m_passes_generation_count[llm_model_name] = 1
        else:
            self.m_passes_generation_count[llm_model_name] += 1

    def _add_i_pass_generation(self, i: str):
        if i not in self.i_passes_generation_count:
            self.i_passes_generation_count[i] = 1
        else:
            self.i_passes_generation_count[i] += 1

    def add_total_time(self, llm_model_name: str, duration: float):
        if llm_model_name not in self.total_time:
            self.total_time[llm_model_name] = duration
        else:
            new_duration = self.total_time[llm_model_name] + duration
            self.total_time[llm_model_name] = new_duration

    def add_total_cost(self, llm_model_name: str, cost: float):
        if llm_model_name not in self.total_cost:
            self.total_cost[llm_model_name] = cost
        else:
            new_cost = self.total_cost[llm_model_name] + cost
            self.total_cost[llm_model_name] = new_cost

    def sort_models_leaderboard(
        self, sort_by: dict[str, int | float], reverse: bool = True
    ):
        # build sort data structure - value can be parametrized
        unsorted_entries = [
            (m, sort_by.get(m, 0)) for m in self.models_leaderboard_order
        ]
        # sort
        sorted_entries = sorted(unsorted_entries, key=lambda x: x[1], reverse=reverse)

        self.models_leaderboard_order = [e[0] for e in sorted_entries]

    def sort_prompts_by_failures(
        self, sort_by: dict[str, int | float], reverse: bool = True
    ):
        # build sort data structure - value can be parametrized
        unsorted_entries = [
            (m, sort_by.get(m, 0)) for m in self.inputs_leaderboard_order
        ]
        # sort
        sorted_entries = sorted(unsorted_entries, key=lambda x: x[1], reverse=reverse)

        self.inputs_leaderboard_order = [e[0] for e in sorted_entries]

    def sort_prompts_by_empty_ctxs(self, reverse: bool = True) -> list[str]:
        sorted_leaderboard = list(self.ctx_empty_count.keys())

        # build sort data structure - value can be parametrized
        unsorted_entries = [
            (m, self.ctx_empty_count.get(m, 0)) for m in sorted_leaderboard
        ]
        # sort
        sorted_entries = sorted(unsorted_entries, key=lambda x: x[1], reverse=reverse)

        sorted_leaderboard = [e[0] for e in sorted_entries]
        return sorted_leaderboard

    def build(self):
        """Analyze, explain, aggregate, and build leaderboard data... so that when
        HTML representation is built, the leaderboard is ready to be rendered.

        """
        if len(self.models_leaderboard_order) > 0:
            if self.m_passes_count:
                self.sort_models_leaderboard(
                    self.m_failures_generation_count, reverse=False
                )
                self.insight_most_acc = self.models_leaderboard_order[0]
                self.insight_least_acc = self.models_leaderboard_order[-1]

            self.sort_models_leaderboard(self.total_time, reverse=False)
            self.insight_fastest_m = self.models_leaderboard_order[0]
            self.insight_slowest_m = self.models_leaderboard_order[-1]

            self.sort_models_leaderboard(self.total_cost, reverse=False)
            self.insight_cheapest_m = self.models_leaderboard_order[0]
            self.insight_expensive_m = self.models_leaderboard_order[-1]
            # reset if no cost
            if (
                self.total_cost.get(self.insight_cheapest_m, 0) == 0
                and self.total_cost.get(self.insight_expensive_m, 0) == 0
            ):
                self.insight_cheapest_m = None
                self.insight_expensive_m = None

            if self.llm_host == commons.LlmModelHostType.RAG:
                self.sort_models_leaderboard(
                    self.m_failures_retrieval_count, reverse=False
                )
                if (
                    self.models_leaderboard_order[0]
                    != self.models_leaderboard_order[-1]
                ):
                    self.insight_best_retrieval_m = self.models_leaderboard_order[0]
                    self.insight_worst_retrieval_m = self.models_leaderboard_order[-1]
                    # if both equally bad, reset
                    if self.m_failures_retrieval_count.get(
                        self.insight_best_retrieval_m, 0
                    ) == self.m_failures_retrieval_count.get(
                        self.insight_worst_retrieval_m, 0
                    ):
                        self.insight_best_retrieval_m = None
                        self.insight_worst_retrieval_m = None

        if len(self.inputs_leaderboard_order) > 0:
            self.sort_prompts_by_failures(self.i_failures_generation_count)
            self.insight_problematic_i = self.inputs_leaderboard_order[0]

        for i in self.inputs_leaderboard_order:
            failures_retrieval = self.i_failures_retrieval_count.get(i, 0)
            failures_generation = self.i_failures_generation_count.get(i, 0)

            if (
                commons.LlmModelHostType.RAG == self.llm_host
                and failures_retrieval > failures_generation
            ):
                self.problems_with_retrieval_i.append(i)

    DEFAULT_METRIC_THRESHOLD = 0.5

    # as leaderboard dictionary keys
    METRIC_MODEL_PASSES = "model_passes"
    METRIC_MODEL_FAILURES = "model_failures"
    METRIC_MODEL_RETRIEVAL_FAILURES = "model_retrieval_failures"
    METRIC_MODEL_GENERATION_FAILURES = "model_generation_failures"
    METRIC_MODEL_PARSE_FAILURES = "model_parse_failures"

    # metric names @ results representations
    KEY_RESULT_CHECK_OK = METRIC_MODEL_PASSES
    KEY_RESULT_CHECK_FAIL = METRIC_MODEL_FAILURES
    KEY_RESULT_CHECK_FAIL_R = METRIC_MODEL_RETRIEVAL_FAILURES
    KEY_RESULT_CHECK_FAIL_A = METRIC_MODEL_GENERATION_FAILURES
    KEY_RESULT_CHECK_FAIL_P = METRIC_MODEL_PARSE_FAILURES
    KEY_RESULT_CHECK_ERR_MSG = "result_error_message"

    # as dictionary keys
    KEY_MODEL_FAILURES = METRIC_MODEL_FAILURES
    KEY_MODEL_PASSES_COUNT = "model_passes_count"
    KEY_MODEL_FAILURES_COUNT = "model_failures_count"
    KEY_MODEL_FAILURES_RETRIEVAL_COUNT = "model_failures_retrieval_count"
    KEY_MODEL_FAILURES_GENERATION_COUNT = "model_failures_generation_count"
    KEY_MODEL_FAILURES_PARSE_COUNT = "model_failures_parse_count"
    KEY_INPUT_FAILURES = "input_failures"
    KEY_INPUT_PASSES_COUNT = "input_passes_count"
    KEY_INPUT_FAILURES_COUNT = "input_failures_count"
    KEY_INPUT_FAILURES_RETRIEVAL_COUNT = "input_failures_retrieval_count"
    KEY_INPUT_FAILURES_GENERATION_COUNT = "input_failures_generation_count"
    KEY_INPUT_FAILURES_PARSE_COUNT = "input_failures_parse_count"
    KEY_TOTAL_TIME = "total_time"
    KEY_TOTAL_COST = "total_cost"

    METRIC_META_MODEL_PASSES = commons.MetricMeta(
        key=METRIC_MODEL_PASSES,
        display_name="Model passes",
        description="Percentage of successfully evaluated RAG/LLM outputs.",
        value_range=(0.0, 1.0),
        higher_is_better=True,
        display_format=".0%",
        is_primary_metric=True,
    )
    METRIC_META_MODEL_FAILURES = commons.MetricMeta(
        key=METRIC_MODEL_FAILURES,
        display_name="Model failures",
        value_range=(0.0, 1.0),
        description=(
            "Percentage of RAG/LLM outputs that failed to pass the evaluator check."
        ),
        higher_is_better=False,
        display_format=".0%",
        is_primary_metric=False,
    )
    METRIC_META_MODEL_RETRIEVAL_FAILURES = commons.MetricMeta(
        key=METRIC_MODEL_RETRIEVAL_FAILURES,
        display_name="Model retrieval failures",
        description=(
            "Percentage of RAG's retrieved contexts that failed to pass "
            "the evaluator check."
        ),
        value_range=(0.0, 1.0),
        higher_is_better=False,
        display_format=".0%",
        is_primary_metric=False,
    )
    METRIC_META_MODEL_GENERATION_FAILURES = commons.MetricMeta(
        key=METRIC_MODEL_GENERATION_FAILURES,
        display_name="Model generation failures",
        description=(
            "Percentage of outputs generated by RAG from the retrieved contexts that "
            "failed to pass the evaluator check (equivalent to the model failures)."
        ),
        value_range=(0.0, 1.0),
        higher_is_better=False,
        display_format=".0%",
        is_primary_metric=False,
    )
    METRIC_META_MODEL_PARSE_FAILURES = commons.MetricMeta(
        key=METRIC_MODEL_PARSE_FAILURES,
        display_name="Model parse failures",
        description=(
            "Percentage of RAG/LLM outputs that evaluator's judge (LLM, RAG, agent or "
            "model) was unable to parse, and therefore unable to evaluate and provide "
            "a metrics score."
        ),
        value_range=(0.0, 1.0),
        higher_is_better=False,
        display_format=".0%",
        is_primary_metric=False,
    )

    LEADERBOARD_METRICS_META = commons.MetricsMeta(
        metrics=[
            METRIC_META_MODEL_PASSES,
            METRIC_META_MODEL_FAILURES,
            METRIC_META_MODEL_RETRIEVAL_FAILURES,
            METRIC_META_MODEL_GENERATION_FAILURES,
            METRIC_META_MODEL_PARSE_FAILURES,
        ]
    )

    def as_dict(self) -> dict:
        """All leaderboard data as dictionary."""
        t = LlmBoolLeaderboardExplanation

        return {
            f5s.ExplanationFormat.KEY_DATA: {
                # models
                t.KEY_MODEL_PASSES_COUNT: self.m_passes_count,
                t.KEY_MODEL_FAILURES: self.m_failures,
                t.KEY_MODEL_FAILURES_COUNT: self.m_failures_count,
                t.KEY_MODEL_FAILURES_RETRIEVAL_COUNT: self.m_failures_retrieval_count,
                t.KEY_MODEL_FAILURES_GENERATION_COUNT: self.m_failures_generation_count,
                t.KEY_MODEL_FAILURES_PARSE_COUNT: self.m_failures_parse_count,
                # inputs
                t.KEY_INPUT_FAILURES: self.i_failures,
                t.KEY_INPUT_PASSES_COUNT: self.i_passes_count,
                t.KEY_INPUT_FAILURES_COUNT: self.i_failures_count,
                t.KEY_INPUT_FAILURES_RETRIEVAL_COUNT: self.i_failures_retrieval_count,
                t.KEY_INPUT_FAILURES_GENERATION_COUNT: self.i_failures_generation_count,
                t.KEY_INPUT_FAILURES_PARSE_COUNT: self.i_failures_parse_count,
                # totals
                t.KEY_TOTAL_TIME: self.total_time,
                t.KEY_TOTAL_COST: self.total_cost,
            },
        }

    def as_leaderboard_dict(
        self,
        metrics_meta: commons.MetricsMeta | None = None,
        threshold: float | None = None,
    ) -> dict:
        """Create leaderboard dictionary: model -> metric -> value.

        By convention, the leaderboard data are always normalized - two options:

        * <0, 1> range for metrics
        * <0, 100> range for percentages

        There are never absolute values like counts, times or duration.

        Parameters
        ----------
        metrics_meta : commons.MetricsMeta | None
            Metrics metadata to override leaderboard's metrics - it is expected that
            keys are identical, however, caller can customize names, descriptions and
            other metrics metadata.
        threshold :  float | None
            Threshold for metrics - if not provided, the default metric threshold
            is used.

        """
        t_lead = LlmBoolLeaderboardExplanation

        # return cached leaderboard, if available
        if self.leaderboard_data_dict:
            return self.leaderboard_data_dict

        leaderboard_dict = {}
        for m in self.total_time:
            (
                passes,
                failures,
                failures_retrieval,
                failures_generation,
                failures_parse,
                ac,
                ac_retrieval,
                ac_generation,
                ac_parse,
            ) = self._model_failure_stats(m)

            total = passes + failures
            passes_0_1 = (passes / total) if passes and total else 0.0
            failures_0_1 = (failures / total) if failures and total else 0.0
            failures_retrieval_0_1 = (
                (failures_retrieval / total) if failures_retrieval and total else 0.0
            )
            failures_generation_0_1 = (
                (failures_generation / total) if failures_generation and total else 0.0
            )

            failures_parse_0_1 = (
                (failures_parse / total) if failures_parse and total else 0.0
            )

            leaderboard_dict[m] = {
                t_lead.METRIC_MODEL_PASSES: passes_0_1,  # %
                t_lead.METRIC_MODEL_FAILURES: failures_0_1,  # %
                t_lead.METRIC_MODEL_RETRIEVAL_FAILURES: failures_retrieval_0_1,  # %
                t_lead.METRIC_MODEL_GENERATION_FAILURES: failures_generation_0_1,  # %
                t_lead.METRIC_MODEL_PARSE_FAILURES: failures_parse_0_1,  # %
            }

        self.leaderboard_data_dict = {
            f5s.ExplanationFormat.KEY_DATA: leaderboard_dict,
            f5s.ExplanationFormat.KEY_METADATA: (
                metrics_meta.to_dict(threshold)
                if metrics_meta
                else self.metrics_meta.to_dict(threshold)
            ),
        }

        return self.leaderboard_data_dict

    def _model_failure_stats(self, llm_model_name: str) -> tuple:
        passes = self.m_passes_count.get(llm_model_name, 0)
        failures = self.m_failures_count.get(llm_model_name, 0)
        total = passes + failures

        # total number of failures == "failures", r/g/p failures is a hint
        failures_retrieval = self.m_failures_retrieval_count.get(llm_model_name, 0)
        failures_generation = self.m_failures_generation_count.get(llm_model_name, 0)
        failures_parse = self.m_failures_parse_count.get(llm_model_name, 0)

        ac = (passes / total) * 100.0 if passes and total else 0.0
        ac_retrieval = (
            ((total - failures_retrieval - failures_parse) / total) * 100.0
            if total
            else 0.0
        )
        ac_generation = (
            ((total - failures_generation - failures_parse) / total) * 100.0
            if total
            else 0.0
        )
        ac_parse = (
            ((total - failures_generation - failures_retrieval) / total) * 100.0
            if total
            else 0.0
        )

        return (
            passes,
            failures,
            failures_retrieval,
            failures_generation,
            failures_parse,
            ac,
            ac_retrieval,
            ac_generation,
            ac_parse,
        )

    def add_json_format(
        self,
        llm_host: commons.LlmModelHostType,
        metrics_meta: commons.MetricsMeta | None = None,
        threshold: float | None = None,
    ):
        """Add JSON format for the leaderboard.

        Parameters
        ----------
        llm_host : commons.LlmModelHostType
            LLM model host type.
        metrics_meta : commons.MetricsMeta | None
            Metrics metadata to override leaderboard's metrics - it is expected that
            keys are identical, however, caller can customize names, descriptions and
            other metrics metadata.
        threshold :  float | None
            Threshold for metrics - if not provided, the default metric threshold
            is used.

        """
        leaderboard_dict = self.as_leaderboard_dict(
            metrics_meta=metrics_meta, threshold=threshold
        )

        if llm_host == commons.LlmModelHostType.SERVICE:
            if (
                LlmBoolLeaderboardExplanation.METRIC_MODEL_RETRIEVAL_FAILURES
                in leaderboard_dict
            ):
                del leaderboard_dict[
                    LlmBoolLeaderboardExplanation.METRIC_MODEL_GENERATION_FAILURES
                ]

        metrics = [LlmBoolLeaderboardExplanation.METRIC_ALL]

        (idx, idx_str) = f5s.LlmHeatmapLeaderboardJSonFormat.serialize_index_file(
            metrics=metrics,
        )
        json_format = f5s.LlmHeatmapLeaderboardJSonFormat(
            explanation=self,
            json_data=idx_str,
            persistence=self.explainer.persistence.store,
        )

        # SKIL save data files: leaderboards

        # save all metrics
        json_format.add_data(
            format_data=json.dumps(
                leaderboard_dict, indent=4, cls=persistences.NanEncoder
            ),
            file_name=idx[f5s.ExplanationFormat.KEY_FILES][
                LlmBoolLeaderboardExplanation.METRIC_ALL
            ],
        )

    def add_markdown_format(self, title="Benchmarks"):
        report_path = self.explainer.persistence.get_evaluator_working_file("report.md")
        with open(report_path, mode="w") as file:
            self.sort_models_leaderboard(self.m_passes_count)
            file.write(self.as_markdown(title=title))

        self.add_format(
            f5s.MarkdownFormat(
                explanation=self,
                format_file=report_path,
            )
        )

    def add_evalstudio_markdown_format(self, title="Summary"):
        report_path = self.explainer.persistence.get_evaluator_working_file("report.md")
        with open(report_path, mode="w") as file:
            self.sort_models_leaderboard(self.m_passes_count)
            file.write(self.as_evalstudio_markdown(title=title))

        self.add_format(
            f5s.EvalStudioMarkdownFormat(
                explanation=self,
                format_file=report_path,
            )
        )

    def as_markdown(self, title: str = "Benchmark", extended: bool = True) -> str:
        """Markdown representation of the leaderboard.

        Parameters
        ----------
        title : str
            Title of the markdown report.
        extended : bool
            Extended report (for the h2oGPTe benchmark).

        Returns
        -------
        str
            Markdown representation of the leaderboard.

        """

        md = ""
        md += f"# {title}\n"
        md += "\n"
        md = LlmBoolLeaderboardExplanation.summary_as_markdown(
            md=md,
            metrics_count=self.metrics_meta.size(),
            llm_host=self.llm_host,
            m_failures_count=self.m_failures_count,
            i_failures_count=self.i_failures_count,
            key_2_evaluated_model=self.key_2_evaluated_model,
            cost_source=self,
        )
        md += "\n"

        # map: llm -> accuracy
        accuracy_stats_as_dict = {}

        md += "\n"
        md += "## Models\n"
        md += "\n"
        if extended:
            is_h2ogpte_stats = 0
            is_cost = 0.0
            llm_name_to_em = {}
            for m in self.key_2_evaluated_model.values():
                llm_name_to_em[m.llm_model_name] = m
                if m.llm_model_meta:
                    is_h2ogpte_stats += 1
                is_cost += self.total_cost.get(m.llm_model_name, 0.0)

            cols = [
                "Rank",
                "LLM",
            ]
            if is_h2ogpte_stats:
                cols.append("LLM[vision]")
            if is_cost:
                cols.append("Cost")
            cols += [
                "Pass",
                "Fail",
                "Accuracy",
                "Time",
            ]
            if is_h2ogpte_stats:
                cols += [
                    "Calls",
                    "Input Tokens",
                    "Output Tokens",
                    "Tokens per second",
                    "Time to First Token",
                ]

            md += "| " + " | ".join(cols) + " |\n"
            md += len(cols) * "| --- " + "|\n"
            for i, llm_model_name in enumerate(self.models_leaderboard_order):
                passed = self.m_passes_count.get(llm_model_name, 0)
                failures = self.m_failures_count.get(llm_model_name, 0)

                failures_retrieval = self.m_failures_retrieval_count.get(
                    llm_model_name, 0
                )
                failures_generation = self.m_failures_generation_count.get(
                    llm_model_name, 0
                )
                failures_parse = str(self.m_failures_parse_count.get(llm_model_name, 0))
                accuracy = (
                    (passed / (passed + failures)) * 100.0
                    if passed and failures
                    else 0.0
                )

                evaluated_model = llm_name_to_em.get(llm_model_name, None)
                # {
                #     "call_count": 1132,
                #     "input_tokens": 1144932,
                #     "llm_name": "claude-3-5-sonnet-20240620",
                #     "output_tokens": 151305,
                #     "time_to_first_token": 0.7049375,
                #     "tokens_per_second": 38.9025
                # }
                if (
                    evaluated_model
                    and evaluated_model.llm_model_meta
                    and isinstance(evaluated_model.llm_model_meta, dict)
                    and models.ExplainableLlmModel.KEY_H2OGPTE_STATS
                    in evaluated_model.llm_model_meta
                ):
                    h2ogpte_stats = evaluated_model.llm_model_meta[
                        models.ExplainableLlmModel.KEY_H2OGPTE_STATS
                    ]
                    stats_calls = h2ogpte_stats.get("call_count", "0")
                    stats_input_tokens = h2ogpte_stats.get("input_tokens", "0")
                    stats_output_tokens = h2ogpte_stats.get("output_tokens", "0")
                    stats_ttft = h2ogpte_stats.get("time_to_first_token", "0")
                    stats_tps = h2ogpte_stats.get("tokens_per_second", "0")
                    stats_v_m = h2ogpte_stats.get(
                        models.ExplainableLlmModel.KEY_H2OGPTE_VISION_M, ""
                    )
                else:
                    stats_calls = "0"
                    stats_input_tokens = "0"
                    stats_output_tokens = "0"
                    stats_ttft = "0"
                    stats_tps = "0"
                    stats_v_m = ""

                md += f"| {i + 1} "  # rank
                md += f"| <pre>{llm_model_name}</pre> "  # LLM
                if is_h2ogpte_stats:
                    md += f"| <pre>{stats_v_m}</pre> "  # vision model for the LLM
                if is_cost:
                    md += f"| ${self.total_cost.get(llm_model_name, 0.0):.4f} "  # cost
                md += f"| {passed} "  # pass

                md += f"| {failures}"  # fail
                if self.do_eval_rc and commons.LlmModelHostType.RAG == self.llm_host:
                    md += (
                        f"&nbsp;({failures_retrieval}/{failures_generation}"
                        f"{'/' + failures_parse if self._has_parse_fails() else ''}) "
                    )
                elif self._has_parse_fails():
                    md += f"&nbsp;({failures_parse}) "
                else:
                    md += " "

                md += f"| {accuracy:.1f}% "  # accuracy
                md += f"| {self.total_time.get(llm_model_name, 0.0):.3f}s "  # time
                if is_h2ogpte_stats:
                    md += f"| {stats_calls} "  # calls
                    md += f"| {stats_input_tokens} "  # input tokens
                    md += f"| {stats_output_tokens} "  # output tokens
                    md += f"| {stats_tps} "  # tokens/s
                    md += f"| {stats_ttft} "  # time to first token
                md += "|\n"

                accuracy_stats_as_dict[llm_model_name] = accuracy
        else:
            md += "| Rank | LLM | Pass | Fail | Accuracy | Total time | Cost |\n"
            md += "| --- | --- | --- | --- | --- | --- | --- |\n"
            for i, llm_model_name in enumerate(self.models_leaderboard_order):
                passed = self.m_passes_count.get(llm_model_name, 0)
                failures = self.m_failures_count.get(llm_model_name, 0)

                failures_retrieval = self.m_failures_retrieval_count.get(
                    llm_model_name, 0
                )
                failures_generation = self.m_failures_generation_count.get(
                    llm_model_name, 0
                )
                failures_parse = str(self.m_failures_parse_count.get(llm_model_name, 0))
                accuracy = (
                    (passed / (passed + failures)) * 100.0
                    if passed and failures
                    else 0.0
                )
                md += f"| {i + 1} | {llm_model_name} | {passed} | {failures} "
                if self.do_eval_rc and commons.LlmModelHostType.RAG == self.llm_host:
                    md += (
                        f"({failures_retrieval}/{failures_generation}"
                        f"{'/' + failures_parse if self._has_parse_fails() else ''}) "
                    )
                elif self._has_parse_fails():
                    md += f"({failures_parse}) "
                md += (
                    f"| {accuracy:.1f}% "
                    f"| {self.total_time.get(llm_model_name, 0.0):.3f}s"
                    f"| ${self.total_cost.get(llm_model_name, 0.0):.4f} |\n"
                )

        if self.inputs_leaderboard_order:
            md += "\n"
            md += "## Failures by Prompts\n"
            md += "\n"
            md += "| Rank | Prompt | Fail | Fail % |\n"
            md += "| --- | --- | --- | --- |\n"
            for e, i in enumerate(self.inputs_leaderboard_order):
                passed = self.i_passes_count.get(i, 0)
                failures = self.i_failures_count.get(i, 0)
                failures_retrieval = self.i_failures_retrieval_count.get(i, 0)
                failures_generation = self.i_failures_generation_count.get(i, 0)
                failures_parse = str(self.i_failures_parse_count.get(i, 0))
                accuracy = 100.0 - (passed / (passed + failures)) * 100.0

                safe_input = sanitization.sanitize_markdown(i)

                if failures > 0:
                    try:
                        md += f"| {e + 1} | {safe_input} | {failures} "
                        if (
                            self.do_eval_rc
                            and commons.LlmModelHostType.RAG == self.llm_host
                        ):
                            parse_fails = (
                                "/" + failures_parse if self._has_parse_fails() else ""
                            )
                            md += (
                                f"({failures_retrieval}/{failures_generation}"
                                f"{parse_fails}) "
                            )
                        elif self._has_parse_fails():
                            md += f"({failures_parse})"

                        md += f"| {accuracy:.1f}% |\n"
                    except Exception as ex:
                        self.logger.error(
                            f"Unable to render Markdown representation for input "
                            f"{i}: {ex}\n{traceback.format_exc()}"
                        )

        if self.m_failures:
            md += "\n"
            md += "## Failures by model\n"
            md += "\n"
            for llm_model_name in self.m_failures:
                md += f"### {llm_model_name}\n"
                md += "\n"
                for failure in self.m_failures[llm_model_name]:
                    md += f"* **{failure.doc_url}**\n"
                    error_type = self._get_error_type_for_failure(failure)
                    safe_em = sanitization.sanitize_markdown(failure.error_message)
                    if error_type:
                        md += f"  * **Error ({error_type})**: {safe_em}\n"
                    else:
                        md += f"  * **Error**: {safe_em}\n"
                    if failure.output_condition:
                        md += (
                            f"  * **Output condition**: `{failure.output_condition}`\n"
                        )
                    if failure.output_constraints:
                        md += (
                            f"  * **Output constraints**: "
                            f"`{failure.output_constraints}`\n"
                        )
                    md += (
                        f"  * **Prompt**: "
                        f"{sanitization.sanitize_markdown(failure.input)}\n"
                    )
                    if failure.expected_output:
                        md += (
                            f"  * **Expected output**: "
                            f"{sanitization.sanitize_markdown(failure.expected_output)}"
                            f"\n"
                        )
                    md += (
                        f"  * **Actual output**: "
                        f"{sanitization.sanitize_markdown(failure.actual_output)}\n"
                    )
                    if commons.LlmModelHostType.RAG == self.llm_host:
                        md += (
                            f"  * **Context chunks / size**: "
                            f"{failure.ctx_chunks} / {failure.ctx_bytes}B\n"
                        )
                    md += "\n"

        if accuracy_stats_as_dict:
            md += "\n## LLM Models Accuracy Statistics\n"
            md += "\n"
            md += "```\n"
            md += json.dumps(accuracy_stats_as_dict, indent=4)
            md += "\n```\n"
            md += "\n"

        return md

    @staticmethod
    def summary_as_markdown(
        md: str,
        metrics_count: int,
        llm_host: commons.LlmModelHostType,
        m_failures_count: dict,
        i_failures_count: dict,
        key_2_evaluated_model: dict,
        cost_source=None,
    ) -> str:
        md += f"* **Metrics**: {metrics_count}\n"
        if len(m_failures_count):
            md += f"* **Models with failures**: {len(m_failures_count)}\n"
        if len(i_failures_count):
            md += f"* **Prompts with failures**: {len(i_failures_count)}\n"
        md += (
            f"* **Host**: "
            f"{next(iter(key_2_evaluated_model.values())).connection.server_url}\n"
        )
        md += f"* **Host type**: {'RAG' if llm_host.name == 'RAG' else 'LLM'}\n"
        if cost_source and cost_source.evaluation_cost():
            md += f"* **Total cost**: ${cost_source.evaluation_cost():2f}\n"
        md += (
            f"* **Created**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        return md

    def as_evalstudio_markdown(self, title: str = "Summary", top: int = 3) -> str:
        """Return Markdown representation of the leaderboard for EvalStudio.

        Parameters
        ----------
        title : str
            Title of the leaderboard.
        top : int
            Number of top model failures, prompt failures, empty context prompts, ...
            entries. `0` for all entries.
            The motivation is to avoid LONG reports with all failures and prompts,
            it's just a summary.

        Returns
        -------
        str
            Markdown representation of the leaderboard.

        """

        md = ""
        md += f"## {title}\n"  # intentionally make it H2 ~ smaller title in Eval Studio
        md += "\n"
        md = LlmBoolLeaderboardExplanation.summary_as_markdown(
            md=md,
            metrics_count=self.metrics_meta.size(),
            llm_host=self.llm_host,
            m_failures_count=self.m_failures_count,
            i_failures_count=self.i_failures_count,
            key_2_evaluated_model=self.key_2_evaluated_model,
            cost_source=self,
        )

        if self.m_failures:
            md += "\n"
            md += "## Models\n"
            md += "\n"
            md += "| Rank | LLM | Pass | Fail | Total time | TPS | Cost |\n"
            md += "| --- | --- | --- | --- | --- | --- | --- |\n"
            for i, llm_model_name in enumerate(self.models_leaderboard_order):
                passes = self.m_passes_count.get(llm_model_name, 0)
                failures = self.m_failures_count.get(llm_model_name, 0)

                total = passes + failures
                failures_retrieval = self.m_failures_retrieval_count.get(
                    llm_model_name, 0
                )
                failures_generation = self.m_failures_generation_count.get(
                    llm_model_name, 0
                )
                failures_parse = str(self.m_failures_parse_count.get(llm_model_name, 0))
                total_time = self.total_time.get(llm_model_name, 0.0)
                tps = 1.0 / (total_time / total) if total_time else 0.0
                total_cost = self.total_cost.get(llm_model_name, 0.0)
                md += f"| {i + 1} | {llm_model_name} | {passes} | {failures}&nbsp;"
                if self.do_eval_rc and commons.LlmModelHostType.RAG == self.llm_host:
                    md += (
                        f"({failures_retrieval}/{failures_generation}"
                        f"{'/' + failures_parse if self._has_parse_fails() else ''}) "
                    )
                elif self._has_parse_fails():
                    md += f"({failures_parse}) "

                md += f"| {total_time:.4f}s | {tps:.4f} "
                if total_cost <= 0.0:
                    md += "| $0 |\n"
                else:
                    md += f"| ${total_cost:.4f} |\n"

        md += "\n"
        md += LlmLeaderboardExplanation.markdown_connection_stats_table(
            self.key_2_evaluated_model.values()
        )
        if self.i_failures_count:
            self.sort_prompts_by_failures(self.i_failures_count)
            md += "## Failures by Prompts\n"
            if top:
                md += (
                    "Top prompts with most failures across all models. "
                    "See the `Report` for the full list.\n"
                )
            md += "\n"
            md += "| Rank | Prompt | Fail | Fail % |\n"
            md += "| --- | --- | --- | --- |\n"
            for e, i in enumerate(self.inputs_leaderboard_order):
                if top and e >= top:
                    break

                passes = self.i_passes_count.get(i, 0)
                failures = self.i_failures_count.get(i, 0)
                failures_retrieval = self.i_failures_retrieval_count.get(i, 0)
                failures_generation = self.i_failures_generation_count.get(i, 0)
                failures_parse = str(self.i_failures_parse_count.get(i, 0))
                accuracy = 100.0 - (passes / (passes + failures)) * 100.0

                safe_input = sanitization.sanitize_markdown(i)

                if failures > 0:
                    try:
                        md += f"| {e + 1} | {safe_input} | {failures}&nbsp;"
                        if (
                            self.do_eval_rc
                            and commons.LlmModelHostType.RAG == self.llm_host
                        ):
                            parse_fail_count = (
                                "/" + failures_parse if self._has_parse_fails() else ""
                            )
                            md += (
                                f"({failures_retrieval}/{failures_generation}"
                                f"{parse_fail_count}) "
                            )
                        elif self._has_parse_fails():
                            md += f"({failures_parse})"
                        md += f"| {accuracy:.1f}% |\n"
                    except Exception as ex:
                        self.logger.error(
                            f"Unable to render Markdown representation for input "
                            f"{i}: {ex}\n{traceback.format_exc()}"
                        )

            md += "\n"
            md += "## Failures by model\n"
            if top:
                md += (
                    f"Up to {top} evaluation failure details for every evaluated "
                    f"model. See the `Report` for the full list.\n"
                )
            md += "\n"
            for llm_model_name in self.m_failures:
                md += f"### {llm_model_name}\n"
                md += "\n"
                for e, failure in enumerate(self.m_failures[llm_model_name]):
                    if top and e >= top:
                        break

                    md += f"* **{failure.doc_url}**\n"
                    error_type = self._get_error_type_for_failure(failure)
                    safe_em = sanitization.sanitize_markdown(failure.error_message)
                    if error_type:
                        md += f"  * **Error ({error_type})**: {safe_em}\n"
                    else:
                        md += f"  * **Error**: {safe_em}\n"
                    md += (
                        f"  * **Prompt**: "
                        f"{sanitization.sanitize_markdown(failure.input)}\n"
                    )
                    if failure.output_condition:
                        md += (
                            f"  * **Output condition**: `{failure.output_condition}`\n"
                        )
                    if failure.output_constraints:
                        md += (
                            f"  * **Output constraints**: "
                            f"`{failure.output_constraints}`\n"
                        )
                    if commons.LlmModelHostType.RAG == self.llm_host:
                        md += (
                            f"  * **Context chunks / size**: "
                            f"{failure.ctx_chunks} / {failure.ctx_bytes}B\n"
                        )
                    if failure.expected_output:
                        md += (
                            f"  * **Expected output**: "
                            f"{sanitization.sanitize_markdown(failure.expected_output)}"
                            f"\n"
                        )
                    md += (
                        f"  * **Actual output**: "
                        f"{sanitization.sanitize_markdown(failure.actual_output)}\n"
                    )
                    md += "\n"
        return md

    @staticmethod
    def _col_eda(col: dict[str, int | float]):
        col_max = max(col.values())
        col_min = min(col.values())
        return col_max, col_min

    def _get_col_for_pct(self, pct: int, reverse: bool = False):
        idx = int(pct / 10)
        idx = idx if idx < 10 else 9
        idx = 9 - idx if reverse else idx

        return self.palette[idx]

    def _get_col_for_value(self, max_val, min_val, val, reverse: bool = False):
        val_rng = max_val - min_val or 1
        pct = int((val - min_val) / (val_rng / 100.0))
        return self._get_col_for_pct(pct, reverse=reverse)

    def _as_html_table(self, html_src, uuid_map: dict, lead_col_name="LLM Models"):
        t_bool_leaderboard = LlmBoolLeaderboardExplanation

        (time_col_max, time_col_min) = t_bool_leaderboard._col_eda(self.total_time)
        (cost_col_max, cost_col_min) = t_bool_leaderboard._col_eda(self.total_cost)

        with html_src.table(klass="w3-table-all"):
            with html_src.tr():
                html_src.th(_t="")
                html_src.th(_t=lead_col_name)
                html_src.th(
                    _t="Pass",
                    title=self.metrics_meta.get_metric_description(
                        t_bool_leaderboard.METRIC_MODEL_PASSES
                    ),
                )
                if commons.LlmModelHostType.RAG == self.llm_host:
                    if self.do_eval_rc:
                        th_title = (
                            "Fail<br/>(retrieval/generation"
                            f"{'/parse' if self._has_parse_fails() else ''})"
                        )
                    else:
                        th_title = "Fail"

                    html_src.th(
                        _t=th_title,
                        title=self.metrics_meta.get_metric_description(
                            t_bool_leaderboard.METRIC_MODEL_FAILURES
                        ),
                    )
                else:
                    if self.do_eval_rc:
                        th_title = (
                            f"Fail{' (parse)' if self._has_parse_fails() else ''}"
                        )
                    else:
                        th_title = "Fail"

                    html_src.th(
                        _t=th_title,
                        title=(
                            self.metrics_meta.get_metric_description(
                                t_bool_leaderboard.METRIC_MODEL_PARSE_FAILURES
                            )
                            if self._has_parse_fails()
                            else self.metrics_meta.get_metric_description(
                                t_bool_leaderboard.METRIC_MODEL_FAILURES
                            )
                        ),
                    )
                html_src.th(_t="Success&nbsp;rate")
                if commons.LlmModelHostType.RAG == self.llm_host:
                    if self.do_eval_rc:
                        html_src.th(
                            _t="Retrieval",
                            title=self.metrics_meta.get_metric_description(
                                t_bool_leaderboard.METRIC_MODEL_RETRIEVAL_FAILURES
                            ),
                        )
                        html_src.th(
                            _t="Generation",
                            title=self.metrics_meta.get_metric_description(
                                t_bool_leaderboard.METRIC_MODEL_GENERATION_FAILURES
                            ),
                        )
                html_src.th(_t="Total time")
                html_src.th(_t="Cost")

            for e, llm_model_name in enumerate(self.models_leaderboard_order):
                (
                    passes,
                    failures,
                    failures_retrieval,
                    failures_generation,
                    failures_parse,
                    ac,
                    ac_retrieval,
                    ac_generation,
                    ac_parse,
                ) = self._model_failure_stats(llm_model_name)

                # IMPORTANT: failure is counted if retrieval fails AND/OR generation
                # fails AND/OR parse fails
                # IMPORTANT: ^ if the RETRIEVAL is EMPTY, then it will ALWAYS fail

                try:
                    with html_src.tr():
                        # rank
                        with html_src.td():
                            html_src(f"{e + 1}.")
                        # model name
                        with html_src.td():
                            html_src(f"{llm_model_name}")
                            prefix = self.rag_type_prefix.get(llm_model_name, "LLM")
                            html_src.sup(_t=f"{prefix}")
                        # passes
                        html_src.td(
                            _t=f"{passes}",
                            title=(
                                f"Total/Pass/Fail: "
                                f"{passes + failures}/{passes}/{failures}"
                            ),
                        )
                        # failures
                        with html_src.td():
                            with html_src.a(href=f"#{uuid_map[llm_model_name]}"):
                                html_src(failures)
                            if (
                                self.do_eval_rc
                                and commons.LlmModelHostType.RAG == self.llm_host
                            ):
                                html_src(" (")
                                if failures_retrieval > failures_generation:
                                    color = "#ff0000"
                                    title = "Retrieval failures / generation failures"
                                else:
                                    color = "#000000"
                                    title = ""
                                with html_src.a(
                                    href=f"#{uuid_map[llm_model_name]}",
                                    style=f"color: {color};",
                                    title=title,
                                ):
                                    html_src(failures_retrieval)
                                html_src(" / ")
                                with html_src.a(href=f"#{uuid_map[llm_model_name]}"):
                                    html_src(failures_generation)
                                if self._has_parse_fails():
                                    html_src(" / ")
                                    with html_src.a(
                                        href=f"#{uuid_map[llm_model_name]}"
                                    ):
                                        html_src(failures_parse)
                                html_src(")")
                            else:
                                if self.do_eval_rc and self._has_parse_fails():
                                    html_src(f"({failures_parse})")

                        # success rate
                        color = self._get_col_for_pct(ac)
                        with html_src.td(style=f"background-color: #{color};"):
                            html_src(f"{ac:.3f}%")
                        if (
                            self.do_eval_rc
                            and commons.LlmModelHostType.RAG == self.llm_host
                        ):
                            # SR retrieval
                            bgcolor = self._get_col_for_pct(ac_retrieval)
                            color = (
                                "#ff0000" if ac_retrieval < ac_generation else "#000000"
                            )
                            with html_src.td(
                                style=f"background-color: #{bgcolor}; color: {color};"
                            ):
                                html_src(f"{ac_retrieval:.3f}%")
                            # SR generation
                            color = self._get_col_for_pct(ac_generation)
                            with html_src.td(style=f"background-color: #{color};"):
                                html_src(f"{ac_generation:.3f}%")

                        # time
                        total_time = self.total_time[llm_model_name]
                        color = self._get_col_for_value(
                            time_col_max, time_col_min, total_time, reverse=True
                        )
                        with html_src.td(style=f"background-color: #{color};"):
                            html_src(f"{total_time:.3f}s")

                        # cost
                        total_cost = self.total_cost[llm_model_name]
                        color = self._get_col_for_value(
                            cost_col_max, cost_col_min, total_cost, reverse=True
                        )
                        with html_src.td(style=f"background-color: #{color};"):
                            html_src(f"${total_cost:.3f}")

                except Exception as ex:
                    self.logger.error(
                        f"Unable to render HTML representation for model "
                        f"{llm_model_name}: {ex}\n{traceback.format_exc()}"
                    )

    def _i_as_html_table(self, html_src, lead_col_name="Prompts"):
        with html_src.table(klass="w3-table-all"):
            with html_src.tr():
                html_src.th(_t=lead_col_name)
                if commons.LlmModelHostType.RAG == self.llm_host:
                    if self.do_eval_rc:
                        html_src.th(
                            _t=(
                                "Failures<br>(r./g."
                                f"{'/p.' if self._has_parse_fails() else ''})"
                            ),
                            title=(
                                "Failures (retrieval/generation"
                                f"{'/parse' if self._has_parse_fails() else ''})"
                            ),
                        )
                    else:
                        html_src.th(_t="Failures")
                else:
                    html_src.th(_t="Failures")
                html_src.th(_t="Success rate")
                if self.do_eval_rc and commons.LlmModelHostType.RAG == self.llm_host:
                    html_src.th(_t="Retrieval", title="Retrieval Accuracy")
                    html_src.th(_t="Generation", title="Generation Accuracy")

            for i in self.inputs_leaderboard_order:
                passed = self.i_passes_count.get(i, 0)
                failures = self.i_failures_count.get(i, 0)
                total = passed + failures
                failures_retrieval = self.i_failures_retrieval_count.get(i, 0)
                failures_generation = self.i_failures_generation_count.get(i, 0)
                failures_parse = self.i_failures_parse_count.get(i, 0)
                ac = 100.0 - (failures / total) * 100.0
                ac_retrieval = 100.0 - (failures_retrieval / total) * 100.0
                ac_generation = 100.0 - (failures_generation / total) * 100.0

                # IMPORTANT: correction in case of missing/invalid context
                #   to avoid false negatives in Pass and Accuracy
                if (
                    commons.LlmModelHostType.RAG == self.llm_host
                    and failures_retrieval > failures_generation
                ):
                    ac = ac_generation

                if failures > 0:
                    try:
                        with html_src.tr():
                            html_src.td(_t=i)
                            with html_src.td():
                                html_src(failures)

                                if (
                                    self.do_eval_rc
                                    and commons.LlmModelHostType.RAG == self.llm_host
                                ):
                                    html_src.br()

                                    color = (
                                        "#ff0000"
                                        if failures_retrieval > failures_generation
                                        else "#000000"
                                    )

                                    parse_failure_count = (
                                        "/" + str(failures_parse)
                                        if self._has_parse_fails()
                                        else ""
                                    )

                                    # hard-coded elements to avoid line breaks
                                    html_src(
                                        f'(<span style="color: {color};">'
                                        f"{failures_retrieval}</span>"
                                        f"/{failures_generation}"
                                        f"{parse_failure_count}"
                                        f")"
                                    )
                                elif self.do_eval_rc and self._has_parse_fails():
                                    html_src(f"({failures_parse}) ")

                            # success rate %
                            bgcolor = self._get_col_for_pct(ac)
                            with html_src.td(style=f"background-color: #{bgcolor};"):
                                html_src(f"{ac:.3f}%")
                            if (
                                self.do_eval_rc
                                and commons.LlmModelHostType.RAG == self.llm_host
                            ):
                                # retrieval SR %
                                color = (
                                    "#ff0000"
                                    if ac_retrieval < ac_generation
                                    else "#000000"
                                )
                                bgcolor = self._get_col_for_pct(ac_retrieval)
                                with html_src.td(
                                    style=(
                                        f"background-color: #{bgcolor}; color: {color};"
                                    )
                                ):
                                    html_src(f"{ac_retrieval:.3f}%")
                                # generation SR %
                                bgcolor = self._get_col_for_pct(ac_generation)
                                with html_src.td(
                                    style=f"background-color: #{bgcolor};"
                                ):
                                    html_src(f"{ac_generation:.3f}%")
                    except Exception as ex:
                        self.logger.error(
                            f"Unable to render HTML representation for input "
                            f"{i}: {ex}\n{traceback.format_exc()}"
                        )

    def _ctx_as_html_table(self, html_src, lead_col_name="Prompts"):
        sorted_leaderboard = self.sort_prompts_by_empty_ctxs()

        with html_src.table(klass="w3-table-all"):
            with html_src.tr():
                html_src.th(_t=lead_col_name)
                html_src.th(_t="Empty&nbsp;Contexts")

            for i in sorted_leaderboard:
                empty_ctx_count = self.ctx_empty_count.get(i, 0)

                if empty_ctx_count > 0:
                    try:
                        with html_src.tr():
                            html_src.td(_t=i)
                            html_src.td(_t=f"{empty_ctx_count}")
                    except Exception as ex:
                        self.logger.error(
                            f"Unable to render HTML representation for input "
                            f"{i}: {ex}\n{traceback.format_exc()}"
                        )

    AdditionalDetails = collections.namedtuple(
        "AdditionalDetails",
        ["formatting", "text"],
    )

    def as_html(
        self,
        title: str = "RAG Benchmark",
        include_header: bool = False,
        include_by_accuracy: bool = True,
        include_by_time: bool = True,
        include_by_cost: bool = True,
        additional_details: dict | None = None,
    ) -> str:
        html_src = airium.Airium()

        # map: model name -> uuid (used for anchors)
        self.sort_models_leaderboard(self.m_passes_count)
        uuid_map = {
            model_name: str(uuid.uuid4())
            for model_name in self.models_leaderboard_order
        }

        if include_header:
            with html_src.b():
                html_src(title)
            with html_src.ul():
                with html_src.li():
                    html_src(f"Date: {datetime.datetime.now()}")
                with html_src.li():
                    host = next(
                        iter(self.key_2_evaluated_model.values())
                    ).connection.server_url
                    html_src(f"Host: {host}")

        if include_by_accuracy:
            self.sort_models_leaderboard(
                self.m_failures_generation_count, reverse=False
            )
            self._as_html_table(
                html_src, uuid_map, lead_col_name="LLM Models by Success Rate"
            )

        if include_by_time:
            html_src.br()

            self.sort_models_leaderboard(self.total_time, reverse=False)
            self._as_html_table(html_src, uuid_map, lead_col_name="LLM Models by Time")

        if include_by_cost:
            html_src.br()

            self.sort_models_leaderboard(self.total_cost, reverse=False)
            self._as_html_table(html_src, uuid_map, lead_col_name="LLM Models by Cost")

        if self.i_failures_count:
            html_src.br()

            self.sort_prompts_by_failures(self.i_failures_generation_count)
            self._i_as_html_table(
                html_src, lead_col_name="Most difficult prompts across all models"
            )

        if commons.LlmModelHostType.RAG == self.llm_host and self.ctx_empty_count:
            html_src.br()

            # insight
            self._ctx_as_html_table(html_src, lead_col_name="Prompts by Empty Contexts")

        html_src.br()

        with html_src.h6():
            html_src.b(_t="Model failures")

        for llm_model_name in self.m_failures:
            with html_src.span(id=f"{uuid_map[llm_model_name]}"):
                html_src("Model ")
                with html_src.b():
                    html_src(f"{llm_model_name}")
                html_src("failures:")

            with html_src.ul():
                for failure in self.m_failures[llm_model_name]:
                    # skip retrieval only errors
                    if (
                        not self.do_eval_rc
                        and failure.fail_retrieval
                        and not failure.fail_generation
                    ):
                        continue

                    def __html_src_error():
                        error_type = self._get_error_type_for_failure(failure)
                        if error_type:
                            with html_src.b():
                                html_src("Error (")
                                with html_src.span(klass="w3-text-red"):
                                    html_src(f"{error_type}")
                                html_src("): ")
                        else:
                            with html_src.b():
                                html_src("Error: ")

                        e_h = LlmLeaderboardExplanation._err_html_msg_from_tokenization(
                            failure.actual_output_meta
                        )
                        if e_h:
                            html_src(e_h)
                        else:
                            html_src(f"{failure.error_message}")

                    with html_src.li():
                        if self.llm_host != commons.LlmModelHostType.RAG:
                            with html_src.span():
                                __html_src_error()
                        else:
                            with html_src.b():
                                if failure.doc_url and failure.doc_url.startswith(
                                    "http"
                                ):
                                    with html_src.span():
                                        html_src("Corpus: ")
                                        with html_src.a(href=f"{failure.doc_url}"):
                                            html_src(f"{failure.doc_url}")
                                else:
                                    html_src(f"{failure.doc_url}")
                        with html_src.ul():
                            if self.llm_host == commons.LlmModelHostType.RAG:
                                with html_src.li():
                                    __html_src_error()
                            with html_src.li():
                                html_src.b(_t="Prompt: ")
                                html_src(f"{failure.input}")
                            if failure.output_condition:
                                with html_src.li():
                                    html_src.b(_t="Output condition: ")
                                    with html_src.span(klass="w3-text-red"):
                                        html_src(f"{failure.output_condition}")
                            if failure.output_constraints:
                                with html_src.li():
                                    html_src.b(_t="Output constraints: ")
                                    with html_src.span(klass="w3-text-red"):
                                        html_src(f"{failure.output_constraints}")
                            with html_src.li():
                                html_src.b(_t="Expected output: ")
                                html_src(f"{failure.expected_output}")
                            # add actual output with meta
                            primary_metric = self.metrics_meta.get_primary_metric()
                            LlmLeaderboardExplanation._html_aa_meta(
                                html_src=html_src,
                                actual_output=failure.actual_output,
                                actual_output_meta=failure.actual_output_meta,
                                err_msg=f"{primary_metric.key}",
                                metrics_meta=self.metrics_meta,
                                metrics={primary_metric.key: None},
                                is_bool_leaderboard=True,
                            )
                            if commons.LlmModelHostType.RAG == self.llm_host:
                                c = "w3-text-red" if failure.fail_retrieval else ""
                                with html_src.li():
                                    html_src.b(_t="Context size:")
                                    with html_src.span(klass=c):
                                        html_src(f"{failure.ctx_bytes}B")
                                with html_src.li():
                                    html_src.b(_t="Context chunks:")
                                    with html_src.span(klass=c):
                                        html_src(f"{failure.ctx_chunks}")

        # SECTION: additional details
        if additional_details:
            html_src.br()
            html_src("Additional details:")
            with html_src.ul():
                for k in additional_details:
                    with html_src.li():
                        html_src.b(_t=f"{k}:")
                        html_src.br()
                        ad = additional_details[k]
                        if isinstance(
                            ad, LlmBoolLeaderboardExplanation.AdditionalDetails
                        ):
                            if ad.formatting == "pre":
                                with html_src.pre():
                                    html_src(f"{ad.text}")
                            elif ad.formatting == "i":
                                with html_src.i():
                                    html_src(f"{ad.text}")
                            else:
                                html_src(f"{ad.text}")
                        else:
                            html_src(f"{ad}")

        return str(html_src)

    def _get_error_type_for_failure(self, failure: Failure) -> str | None:
        if self._has_parse_fails():
            error_type = "judge output parsing" if failure.fail_parse else None
        elif commons.LlmModelHostType.RAG == self.llm_host:
            if failure.fail_generation:
                if failure.fail_retrieval:
                    error_type = "retrieval and generation"
                else:
                    error_type = "generation"
            else:
                error_type = "retrieval"
        else:
            error_type = "LLM generation"

        return error_type

    def get_insights(
        self,
        insight_type: str = "accuracy",
        quality: str = "accurate",
        extra_description_actions: str = "",
        explanation_type: str = "",
        explanation_name: str = "",
        explanation_mime: str = "",
    ) -> None:
        """Create insights for the boolean leaderboard.

        Parameters
        ----------
        insight_type : str
            Insight type.
        quality : str
            Model quality.
        extra_description_actions: str
            Additional description for actions.
        explanation_type : str
            Type of the explanation which can clarify the insight.
        explanation_name : str
            Name of the explanation which can clarify the insight.
        explanation_mime : str
            Media type of the explanation which can clarify the insight.

        """
        t_insights = insights.InsightAndAction

        evaluator_name = self.explainer._display_name

        default_actions_description_bare = (
            "A detailed description of the failures, questions and answers to "
            "identify the weaknesses and strengths of the model and their root "
            "causes can be found in the explanation."
        )

        default_actions_description = (
            f"{default_actions_description_bare} {extra_description_actions}"
        )

        is_one_model = (
            self.key_2_evaluated_model and len(self.key_2_evaluated_model) == 1
        )

        # ACCURACY
        if not is_one_model and self.insight_most_acc:
            self.explainer.add_insight(
                insights.InsightAndAction(
                    description=(
                        f"Model {self.insight_most_acc} evaluated "
                        f"as the most {quality} model according to "
                        f"{evaluator_name} evaluator."
                    ),
                    description_html=(
                        insights.InsightAndAction.html_most_least_model_by(
                            model_name=self.insight_most_acc,
                            quality=quality,
                            evaluator_name=evaluator_name,
                            is_most=True,
                        )
                    ),
                    insight_type=insight_type,
                    insight_attrs={
                        t_insights.ATTR_MODEL_NAME: self.insight_most_acc,
                        t_insights.ATTR_EVALUATOR_NAME: evaluator_name,
                    },
                    actions_description=default_actions_description,
                    evaluator_id=self.explainer.evaluator_id(),
                    evaluator_name=evaluator_name,
                    explanation_type=explanation_type,
                    explanation_name=explanation_name,
                    explanation_mime=explanation_mime,
                    resources=[],
                )
            )
        if not is_one_model and self.insight_least_acc:
            self.explainer.add_insight(
                insights.InsightAndAction(
                    description=(
                        f"Model {self.insight_least_acc} evaluated "
                        f"as the least {quality} model according to "
                        f"{evaluator_name} evaluator."
                    ),
                    description_html=insights.InsightAndAction.html_most_least_model_by(
                        model_name=self.insight_least_acc,
                        quality=quality,
                        evaluator_name=evaluator_name,
                        is_most=False,
                    ),
                    insight_type=insight_type,
                    insight_attrs={
                        t_insights.ATTR_MODEL_NAME: self.insight_least_acc,
                        t_insights.ATTR_EVALUATOR_NAME: evaluator_name,
                    },
                    actions_description=default_actions_description,
                    evaluator_id=self.explainer.explainer_id(),
                    evaluator_name=evaluator_name,
                    explanation_type=explanation_type,
                    explanation_name=explanation_name,
                    explanation_mime=explanation_mime,
                    resources=[],
                )
            )

        # SPEED
        if not is_one_model and self.insight_fastest_m:
            self.explainer.add_insight(
                insights.InsightAndAction(
                    description=(
                        f"Model {self.insight_fastest_m} is the "
                        f"fastest evaluated LLM model."
                    ),
                    description_html=insights.InsightAndAction.html_most_least_model_by(
                        model_name=self.insight_fastest_m,
                        quality="fastest",
                        evaluator_name=evaluator_name,
                        is_most=None,
                    ),
                    insight_type="performance",
                    insight_attrs={
                        t_insights.ATTR_MODEL_NAME: self.insight_fastest_m,
                        t_insights.ATTR_EVALUATOR_NAME: evaluator_name,
                        t_insights.ATTR_FASTEST_MODEL_NAME: (self.insight_fastest_m),
                    },
                    actions_description=default_actions_description_bare,
                    evaluator_id=self.explainer.explainer_id(),
                    evaluator_name=evaluator_name,
                    explanation_type=explanation_type,
                    explanation_name=explanation_name,
                    explanation_mime=explanation_mime,
                    resources=[],
                )
            )

        if not is_one_model and self.insight_slowest_m:
            self.explainer.add_insight(
                insights.InsightAndAction(
                    description=(
                        f"Model {self.insight_slowest_m} is the "
                        f"slowest evaluated LLM model."
                    ),
                    description_html=(
                        insights.InsightAndAction.html_most_least_model_by(
                            model_name=self.insight_slowest_m,
                            quality="slowest",
                            evaluator_name=evaluator_name,
                            is_most=None,
                        )
                    ),
                    insight_type="performance",
                    insight_attrs={
                        t_insights.ATTR_MODEL_NAME: self.insight_slowest_m,
                        t_insights.ATTR_EVALUATOR_NAME: evaluator_name,
                        t_insights.ATTR_SLOWEST_MODEL_NAME: (self.insight_slowest_m),
                    },
                    actions_description=default_actions_description_bare,
                    evaluator_id=self.explainer.explainer_id(),
                    evaluator_name=evaluator_name,
                    explanation_type=explanation_type,
                    explanation_name=explanation_name,
                    explanation_mime=explanation_mime,
                    resources=[],
                )
            )

        # PRICE
        if not is_one_model and self.insight_cheapest_m:
            self.explainer.add_insight(
                insights.InsightAndAction(
                    description=(
                        f"Model {self.insight_cheapest_m} is the "
                        f"cheapest evaluated LLM model."
                    ),
                    description_html=insights.InsightAndAction.html_most_least_model_by(
                        model_name=self.insight_cheapest_m,
                        quality="cheapest",
                        evaluator_name=evaluator_name,
                        is_most=None,
                    ),
                    insight_type="cost",
                    insight_attrs={
                        t_insights.ATTR_MODEL_NAME: self.insight_cheapest_m,
                        t_insights.ATTR_EVALUATOR_NAME: evaluator_name,
                        t_insights.ATTR_CHEAPEST_MODEL_NAME: (self.insight_cheapest_m),
                    },
                    actions_description=default_actions_description_bare,
                    evaluator_id=self.explainer.explainer_id(),
                    evaluator_name=evaluator_name,
                    explanation_type=explanation_type,
                    explanation_name=explanation_name,
                    explanation_mime=explanation_mime,
                    resources=[],
                )
            )

        if not is_one_model and self.insight_expensive_m:
            self.explainer.add_insight(
                insights.InsightAndAction(
                    description=(
                        f"Model {self.insight_expensive_m} is the "
                        f"most expensive evaluated LLM model."
                    ),
                    description_html=insights.InsightAndAction.html_most_least_model_by(
                        model_name=self.insight_expensive_m,
                        quality="expensive",
                        evaluator_name=evaluator_name,
                        is_most=True,
                    ),
                    insight_type="cost",
                    insight_attrs={
                        t_insights.ATTR_MODEL_NAME: self.insight_expensive_m,
                        t_insights.ATTR_EVALUATOR_NAME: evaluator_name,
                        t_insights.ATTR_MOST_EXPENSIVE_MODEL_NAME: (
                            self.insight_expensive_m
                        ),
                    },
                    actions_description=default_actions_description_bare,
                    evaluator_id=self.explainer.explainer_id(),
                    evaluator_name=evaluator_name,
                    explanation_type=explanation_type,
                    explanation_name=explanation_name,
                    explanation_mime=explanation_mime,
                    resources=[],
                )
            )

        # RETRIEVAL
        if not is_one_model and self.insight_best_retrieval_m:
            html = airium.Airium()
            html("RAG retrieval for the model ")
            with html.code():
                html(self.insight_best_retrieval_m)
            html("has ")
            with html.b(klass="w3-black"):
                html("&nbsp;the best retrieval&nbsp;")
            html("&nbsp; accuracy.")

            self.explainer.add_insight(
                insights.InsightAndAction(
                    description=(
                        f"RAG retrieval for the model "
                        f"{self.insight_best_retrieval_m} "
                        f"has the best retrieval accuracy."
                    ),
                    description_html=html,
                    insight_type="retrieval",
                    insight_attrs={
                        t_insights.ATTR_MODEL_NAME: self.insight_best_retrieval_m,
                        t_insights.ATTR_EVALUATOR_NAME: evaluator_name,
                    },
                    actions_description=default_actions_description,
                    evaluator_id=self.explainer.explainer_id(),
                    evaluator_name=evaluator_name,
                    explanation_type=explanation_type,
                    explanation_name=explanation_name,
                    explanation_mime=explanation_mime,
                    resources=[],
                )
            )

        if not is_one_model and self.insight_worst_retrieval_m:
            html = airium.Airium()
            html("RAG retrieval for the model ")
            with html.code():
                html(self.insight_worst_retrieval_m)
            html("has ")
            with html.b(klass="w3-black"):
                html("&nbsp;the worst retrieval&nbsp;")
            html("&nbsp; accuracy.")

            self.explainer.add_insight(
                insights.InsightAndAction(
                    description=(
                        f"RAG retrieval for the model "
                        f"{self.insight_worst_retrieval_m} "
                        f"has the worst retrieval accuracy."
                    ),
                    description_html=html,
                    insight_type="retrieval",
                    insight_attrs={
                        t_insights.ATTR_MODEL_NAME: self.insight_worst_retrieval_m,
                        t_insights.ATTR_EVALUATOR_NAME: evaluator_name,
                    },
                    actions_description=default_actions_description,
                    evaluator_id=self.explainer.explainer_id(),
                    evaluator_name=evaluator_name,
                    explanation_type=explanation_type,
                    explanation_name=explanation_name,
                    explanation_mime=explanation_mime,
                    resources=[],
                )
            )

        # PROMPTS
        if self.insight_problematic_i:
            self.explainer.add_insight(
                insights.InsightAndAction(
                    description=(
                        f"Prompt '{self.insight_problematic_i}' is "
                        f"the most difficult prompt to be correctly answered according "
                        f"to {evaluator_name} evaluator."
                    ),
                    description_html=(
                        insights.InsightAndAction.html_most_least_prompt_by(
                            prompt=self.insight_problematic_i,
                            quality="difficult",
                            evaluator_name=evaluator_name,
                            is_most=True,
                        )
                    ),
                    insight_type="weak-point",
                    actions_description=default_actions_description,
                    evaluator_id=self.explainer.explainer_id(),
                    evaluator_name=evaluator_name,
                    explanation_type=explanation_type,
                    explanation_name=explanation_name,
                    explanation_mime=explanation_mime,
                    resources=[],
                )
            )

    @staticmethod
    def from_eval_results(
        evaluator,
        eval_results,
        metrics_meta: commons.MetricsMeta,
        metric_id_success: str,
        metric_id_failure_message: str,
        display_name: str = None,
        display_category: str = None,
        key_2_evaluated_model: dict = None,
        llm_host: commons.LlmModelHostType = commons.LlmModelHostType.RAG,
        do_eval_rc: bool = False,
        logger=None,
    ) -> "LlmBoolLeaderboardExplanation":
        """Create LLM leaderboard explanation from the evaluation results.

        Parameters
        ----------
        evaluator :
            Evaluator instance.
        eval_results : datasets.LlmEvalResults
            Evaluation results.
        metrics_meta : commons.MetricsMeta
            Metrics metadata.
        metric_id_success : str
            Metric ID for the success indicator.
        metric_id_failure_message : str
            Metric ID for the failure message.
        display_name : str
            Custom display name.
        display_category : str
            Custom display category.
        key_2_evaluated_model : dict
            Map: key -> RAG/LLM model.
        llm_host : commons.LlmModelHostType
            LLM host type - either a RAG (with retrieval) or a LLM (generation only).
        do_eval_rc : bool
            Whether to show retrieval correctness.
        logger :
            Optional logger.

        """
        if (
            not eval_results
            or eval_results.results is None
            or len(eval_results.results) == 0
        ):
            raise ValueError(
                f"No evaluation results created by the evaluator "
                f"'{evaluator.display_name}' - unable to create the leaderboard."
            )

        leaderboard_explanation = LlmBoolLeaderboardExplanation(
            evaluator=evaluator,
            metrics_meta=metrics_meta,
            display_name=display_name,
            display_category=display_category,
            key_2_evaluated_model=key_2_evaluated_model,
            llm_host=llm_host,
            do_eval_rc=do_eval_rc,
            logger=logger or loggers.SonarPrintLogger(),
        )

        # populate leaderboard data
        for r in eval_results.results:
            evaluated_model = leaderboard_explanation.key_2_evaluated_model[
                r.dataset_row.model_key
            ]
            context_list = r.dataset_row.context

            if r.metrics.get(metric_id_success, 0.0):
                leaderboard_explanation.add_pass(
                    llm_model_name=evaluated_model.llm_model_name,
                    i=r.dataset_row.i,
                    context=context_list,
                    duration=r.dataset_row.actual_duration,
                    cost=r.dataset_row.cost,
                    row_key=r.dataset_row.key,
                    model_key=evaluated_model.key,
                )
            else:
                leaderboard_explanation.add_failure(
                    llm_model_name=evaluated_model.llm_model_name,
                    doc_url=(
                        evaluated_model.documents[0]
                        if isinstance(evaluated_model, models.ExplainableRagModel)
                        and evaluated_model.documents
                        else ""
                    ),
                    error_message=r.metrics.get(metric_id_failure_message, ""),
                    i=r.dataset_row.i,
                    context=context_list,
                    expected_output=r.dataset_row.expected_output,
                    output_constraints=r.dataset_row.output_constraints,
                    output_condition=r.dataset_row.output_condition,
                    actual_output=r.dataset_row.actual_output,
                    actual_output_meta=r.actual_output_meta,
                    duration=r.dataset_row.actual_duration,
                    cost=r.dataset_row.cost,
                    fail_retrieval=r.metrics.get(
                        LlmBoolLeaderboardExplanation.KEY_RESULT_CHECK_FAIL_R, False
                    ),
                    fail_generation=r.metrics.get(
                        LlmBoolLeaderboardExplanation.KEY_RESULT_CHECK_FAIL_A, False
                    ),
                    fail_parse=r.metrics.get(
                        LlmBoolLeaderboardExplanation.KEY_RESULT_CHECK_FAIL_P, False
                    ),
                    row_key=r.dataset_row.key,
                    model_key=evaluated_model.key,
                )

        leaderboard_explanation.build()

        return leaderboard_explanation


class LlmHeatmapLeaderboardExplanation(
    _explanations_base.Explanation, LlmLeaderboardExplanation, AbcHeatmapExplanation
):
    """Heatmap leaderboard explanation provides data and formats for a leaderboard
    which is colorized as heatmap based on metrics values.

    """

    _explanation_type = "llm-heatmap-leaderboard"
    _is_global = True

    def __init__(
        self,
        evaluator,
        eval_results,
        metrics_meta: commons.MetricsMeta,
        key_2_evaluated_model: dict,
        llm_host: commons.LlmModelHostType = commons.LlmModelHostType.RAG,
        nan_tolerance: float = 0.0,
        display_name: str = "",
        display_category: str = "",
        logger=None,
    ) -> None:
        """LLM Heatmap leaderboard explanation constructor.

        Parameters
        ----------
        evaluator :
            Evaluator which created the evaluation results.
        eval_results :
            Evaluation results to be explained.
        metrics_meta : commons.MetricsMeta
            Evaluator's metric metadata to be used for the leaderboard.
        key_2_evaluated_model : dict
            RAG/LLM model key to evaluated model map.
        llm_host : commons.LlmModelHostType
            LLM host of the evaluated models.
        nan_tolerance : float
            Tolerance for NaN values (percent) in average metric value
            calculation - allow to ignore evaluation results with NaN metric values if
            the number of evaluation results is lower or equal to the given percentage
            of the total number of evaluation results.
            If the number of NaN values is higher than the given percentage, then
            the average metric value is set to ``NaN``.
        display_name : str
            Custom display name.
        display_category : str
            Custom display category.
        logger :
            Optional logger.

        """

        display_name = display_name or "LLM Heatmap Leaderboard"

        _explanations_base.Explanation.__init__(
            self,
            explainer=evaluator,
            display_name=display_name,
            display_category=display_category,
        )

        self.eval_results = eval_results

        # LLM@RAG/LLM models map: key -> model instance
        self.key_2_evaluated_model = key_2_evaluated_model
        self.llm_host = llm_host

        # metrics metadata:
        # - metric names map: key -> metric name
        # - metric thresholds: key -> threshold
        # - metric higher is better map: key -> bool
        self.metrics_meta = metrics_meta
        # NaN tolerance for average metric value calculation
        self.nan_tolerance = nan_tolerance
        # actual metrics which present in the data (metrics names map might have more)
        self.metric_ids_order = []
        # board order: as models are dict, model (1st) col is sorted by given metric
        self.leaderboard_order = [
            m.llm_model_name for m in self.key_2_evaluated_model.values()
        ]

        # palette from dark green to dark red by 10%
        self.palette = LlmHeatmapLeaderboardExplanation.PALETTE_RED.copy()
        r = LlmHeatmapLeaderboardExplanation.PALETTE_GREEN.copy()
        r.reverse()
        self.palette.extend(r)

        self.logger = logger or loggers.SonarPrintLogger()

        # data map: model -> document -> prompt -> metrics -> value
        self.data_dict = {}
        # model failures map: model ID -> (err_msg, row_as_dict, doc)
        self.m_failures = {}
        # model failures count map: model ID -> count
        self.m_failures_count = {}

        # prompt failures map: prompt -> (err_msg, row_as_dict, metric_id, doc)
        self.i_failures = {}
        # prompt failures count map: prompt -> count
        self.i_failures_count = {}
        # prompt passes
        self.i_passes_count = {}
        # prompts leaderboard order by number of failures
        self.inputs_leaderboard_order = []

        # tested LLM host (h2oGPTe/OpenAI RAG/...) to prefix LLMs in leaderboards
        # map: model name -> prefix
        if key_2_evaluated_model:
            self.rag_type_prefix = LlmBoolLeaderboardExplanation.key_2_rag_type_prefix(
                key_2_evaluated_model.values()
            )
        else:
            self.rag_type_prefix = {}

        # insights
        # map: metric_id -> LLM model ID
        self.insights_metric_2_winner = {}
        self.insights_metric_2_looser = {}
        self.insights_difficult_prompt = ""

    def add_col_value(
        self,
        llm_model_name: str,
        docs: str,  # TODO list of strings (document names)
        prompt: str,
        metrics_id: str,
        value: float,
        result_row,  # datasets.LlmEvalResults.LllmEvalResultRow
    ):
        """Add entry to the data dictionary used to build formatted tables later."""
        if llm_model_name not in self.data_dict:
            self.data_dict[llm_model_name] = {}
        if docs not in self.data_dict[llm_model_name]:
            self.data_dict[llm_model_name][docs] = {}
        if prompt not in self.data_dict[llm_model_name][docs]:
            self.data_dict[llm_model_name][docs][prompt] = {}

        if (
            self.data_dict[llm_model_name][docs][prompt].get(metrics_id, None)
            is not None
        ):
            self.logger.warning(
                f"CONFLICT when adding metrics value to the heatmap on entry: "
                f"{llm_model_name} {docs} {prompt} {metrics_id} "
                f"(entry with such key already exists)"
            )

        self.data_dict[llm_model_name][docs][prompt][metrics_id] = value

        # generate failure using row_as_dict and discard the dict
        threshold = self.metrics_meta.get_threshold(metrics_id, default_value=0.0)
        if value is not None and isinstance(value, str):
            raise ValueError(
                f"Invalid metric value '{value}' for '{metrics_id}' metric in the "
                f"leaderboard (threshold is {threshold}) - expected float, "
                f"not {type(value)}."
            )
        if math.isnan(value) or math.isinf(value):
            is_failure = True
            err_msg = (
                f"'{metrics_id}' metric value '{value}' is invalid, because "
                f"the evaluator failed to calculate it (threshold is {threshold})"
            )
        elif self.metrics_meta.is_higher_better(metrics_id):
            is_failure = value < threshold
            err_msg = (
                f"'{metrics_id}' metric value '{value}' is below the threshold "
                f"{threshold}"
            )
        else:
            is_failure = value > threshold
            err_msg = (
                f"'{metrics_id}' metric value '{value}' is above the threshold "
                f"{threshold}"
            )

        if is_failure:
            # model failure
            if llm_model_name not in self.m_failures:
                self.m_failures[llm_model_name] = []
            if llm_model_name not in self.m_failures_count:
                self.m_failures_count[llm_model_name] = 0
            # failure tuple: (error message, row as dict, corpus)
            self.m_failures[llm_model_name].append((err_msg, result_row, docs))
            self.m_failures_count[llm_model_name] += 1

            # prompt failure
            if prompt not in self.i_failures:
                self.i_failures[prompt] = []
            if prompt not in self.i_failures_count:
                self.i_failures_count[prompt] = 0
            self.i_failures[prompt].append((err_msg, result_row, metrics_id, docs))
            self.i_failures_count[prompt] += 1

            if prompt not in self.inputs_leaderboard_order:
                self.inputs_leaderboard_order.append(prompt)
        else:
            if prompt not in self.i_passes_count:
                self.i_passes_count[prompt] = 1
            else:
                self.i_passes_count[prompt] += 1

    def build(self):
        """Analyze, explain, aggregate, and build leaderboard data... so that when
        HTML representation is built, the leaderboard is ready to be rendered.

        """
        (agg_data, _) = self._eda()
        for sort_by_metric_id in self.metric_ids_order:
            self._sort_agg_data(
                sort_by_metric_id,
                agg_data,
                reverse=self.metrics_meta.is_higher_better(sort_by_metric_id),
            )

            # insights
            if len(self.leaderboard_order) > 1:
                self.insights_metric_2_winner[sort_by_metric_id] = (
                    self.leaderboard_order[0]
                )
                self.insights_metric_2_looser[sort_by_metric_id] = (
                    self.leaderboard_order[-1]
                )

        if self.inputs_leaderboard_order:
            self.sort_prompts_by_failures(self.i_failures_count)
            self.insights_difficult_prompt = self.inputs_leaderboard_order[0]

    def __str__(self) -> str:
        return str(self.data_dict)

    def validate(self) -> bool:
        return self._formats is not None

    def __add_to_eda_vals(
        self, metrics_vals: dict, llm_model_name: str, metric_id: str, metric_val: float
    ):
        if metric_id not in self.metric_ids_order:
            self.metric_ids_order.append(metric_id)

        if llm_model_name not in metrics_vals:
            metrics_vals[llm_model_name] = {}
        if metric_id not in metrics_vals[llm_model_name]:
            metrics_vals[llm_model_name][metric_id] = [0.0, 0.0]

        _m_sum = metrics_vals[llm_model_name][metric_id][0] + metric_val
        _m_count = metrics_vals[llm_model_name][metric_id][1] + 1.0
        metrics_vals[llm_model_name][metric_id] = [_m_sum, _m_count]

    def _create_nan_problem(
        self,
        problem_description: str,
        action_description: str,
        llm_model_name: str,
        model_nan_pct: float,
    ):
        problem_type = "robustness"

        # avoid duplicate problems
        explainer_problems = self.explainer.explain_problems()
        if explainer_problems and len(explainer_problems) > 0:
            for p in explainer_problems:
                if (
                    p.problem_type != problem_type
                    or p.explainer_id != self.explainer.explainer_id()
                ):
                    continue
                if (
                    p.problem_attrs
                    and p.problem_attrs.get(p6s.ProblemAndAction.ATTR_MODEL_NAME)
                    == llm_model_name
                    and p.problem_attrs.get(p6s.ProblemAndAction.ATTR_NAN_PCT)
                    == model_nan_pct
                ):
                    # skip duplicate
                    return

        self.explainer.add_problem(
            p6s.ProblemAndAction(
                description=problem_description,
                problem_type=problem_type,
                problem_attrs={
                    p6s.ProblemAndAction.ATTR_MODEL_NAME: llm_model_name,
                    p6s.ProblemAndAction.ATTR_NAN_PCT: model_nan_pct,
                    p6s.ProblemAndAction.ATTR_NAN_TOLERANCE: self.nan_tolerance,
                },
                problem_code=p6s.AVIDProblemCode.P0200_MODEL,
                severity=p6s.ProblemSeverity.high,
                actions_description=action_description,
                evaluator_id=self.explainer.explainer_id(),
                evaluator_name=self._display_name,
                explanation_type=self.explanation_type(),
                explanation_name=LlmHeatmapLeaderboardExplanation.__name__,
                explanation_mime=f5s.HtmlFormat.mime,
                resources=[],
            )
        )

    def _eda(self) -> tuple[dict, dict]:
        # EDA:
        #   Use metrics from the classification evaluator to create a heatmap
        #   leaderboard data - reuse heatmap explanation for the rendering and
        #   representations.
        #
        # RESULT: heatmap leaderboard dicts definition:
        # - metrics values:
        #     model -> metric ID -> (avg) value
        # - metrics EDA:
        #     metric ID -> [min, max]
        #
        # METHOD: aggregated data map: model -> metric -> avg value
        # 1. cluster the data
        #    aggregated data map: model -> metric -> [sum, count]
        # 2. calculate average
        #    aggregated data map: model -> metric -> avg value

        # 0. if NaN tolerance is set, then remove NaN values (if below the tolerance)
        if self.nan_tolerance > 0.0:
            # map: model -> [total values, NaN values] ... for row (not metrics)
            model_2_nan_map = {}
            model_2_row_del_plan = {}

            # 0.1. calculate the number of NaN values per model
            dirty_data_dict = False
            data_dict = copy.deepcopy(self.data_dict)
            for d_m in data_dict:
                model_2_row_del_plan[d_m] = []

                prompt_count = 0
                for d_d in data_dict[d_m]:
                    prompt_count += len(data_dict[d_m][d_d])

                    for d_p in data_dict[d_m][d_d]:
                        # detect a NaN metric score
                        is_row_nan = False
                        for d_mm in data_dict[d_m][d_d][d_p]:
                            if (
                                math.isnan(data_dict[d_m][d_d][d_p][d_mm])
                                # or math.isinf(self.data_dict[d_m][d_d][d_p][d_mm])
                            ):
                                is_row_nan = True
                        if is_row_nan:
                            model_2_row_del_plan[d_m].append([d_m, d_d, d_p])
                # model stats
                if len(model_2_row_del_plan[d_m]) > 0:
                    model_2_nan_map[d_m] = [
                        prompt_count,
                        len(model_2_row_del_plan[d_m]),
                    ]

            # 0.2. remove rows with NaN values for eligible models
            for m in model_2_nan_map:
                model_nan_pct = float(model_2_nan_map[m][1]) / float(
                    model_2_nan_map[m][0]
                )
                action_description = (
                    "Make sure that the OpenAI LLM judge connection is "
                    "configured correctly and stable. Consider aspect which "
                    "may lead to instability of the LLM connection, such as "
                    "network latency, server load, or other factors. Also make "
                    "sure that the evaluation data - such as question, actual "
                    "answer, retrieved context - are correct and valid as they "
                    "are used to generate prompts to be executed by OpenAI LLM "
                    "judge."
                )
                if model_nan_pct <= self.nan_tolerance:
                    problem_msg = (
                        f"There are NaN values in the {self.explainer._display_name} "
                        f"evaluator leaderboard metric scores for model '{m}', "
                        f"however, the number of NaN values is in the tolerance: "
                        f"{model_nan_pct * 100.0:.2f}% <= "
                        f"{self.nan_tolerance * 100.0:.2f}%"
                        f" - NaN values will be filtered out and average metric value "
                        f"will be calculated"
                    )
                    self.logger.info(problem_msg)
                    self._create_nan_problem(
                        problem_description=f"{problem_msg}.",
                        action_description=action_description,
                        llm_model_name=m,
                        model_nan_pct=model_nan_pct,
                    )

                    # remove rows with NaN values
                    if m in model_2_row_del_plan:
                        dirty_data_dict = True
                        for d in model_2_row_del_plan[m]:
                            del data_dict[d[0]][d[1]][d[2]]

                else:
                    problem_msg = (
                        f"There are NaN values in the {self.explainer._display_name} "
                        f"evaluator leaderboard metric scores for model '{m}', "
                        f"and the number of NaN values is above the tolerance: "
                        f"{model_nan_pct * 100.0:.2f}% <= "
                        f"{self.nan_tolerance * 100.0:.2f}%"
                        f" - NaN values will not be filtered out and average metric "
                        f"value will be NaN"
                    )
                    self.logger.warning(problem_msg)
                    self._create_nan_problem(
                        problem_description=f"{problem_msg}.",
                        action_description=action_description,
                        llm_model_name=m,
                        model_nan_pct=model_nan_pct,
                    )

            # 0.3. decide whether to use filtered data dict or not
            if not dirty_data_dict:
                data_dict = self.data_dict
        else:
            data_dict = self.data_dict

        # 1. cluster the data
        metrics_vals = {}
        for d_m in data_dict:
            for d_d in data_dict[d_m]:
                for d_p in data_dict[d_m][d_d]:
                    for d_mm in data_dict[d_m][d_d][d_p]:
                        self.__add_to_eda_vals(
                            metrics_vals=metrics_vals,
                            llm_model_name=d_m,
                            metric_id=d_mm,
                            metric_val=data_dict[d_m][d_d][d_p][d_mm],
                        )

        # 2. calculate average + metrics EDA
        metrics_eda = {m: [100.0, 0.0] for m in self.metric_ids_order}
        for d_m in metrics_vals:
            for d_mm in metrics_vals[d_m]:
                m_sum = metrics_vals[d_m][d_mm][0]
                m_count = metrics_vals[d_m][d_mm][1]
                m_avg = m_sum / float(m_count)  # IMPROVE: harmonic mean vs. mean
                m_min = min(metrics_eda[d_mm][0], m_avg)
                m_max = max(metrics_eda[d_mm][1], m_avg)

                metrics_vals[d_m][d_mm] = m_avg
                metrics_eda[d_mm] = [m_min, m_max]

        return metrics_vals, metrics_eda

    def _sort_agg_data(self, metric_id: str, agg_data: dict, reverse: bool = True):
        """Sort data dictionary by the given metric to generate table which is
        sorted by given column.

        """
        # metric-based dir used to sort models
        sort_by = [
            (m, agg_data[m][metric_id]) for m in agg_data if metric_id in agg_data[m]
        ]

        # NaN comparison is not define for floats -> define NaN as min/max
        nan_deputy = (
            sys.float_info.min
            if self.metrics_meta.is_higher_better(metric_id)
            else sys.float_info.max
        )

        # build sort data structure - value can be parametrized
        sorted_entries = sorted(
            sort_by,
            key=lambda x: x[1] if not math.isnan(x[1]) else nan_deputy,
            reverse=reverse,
        )

        self.leaderboard_order = [e[0] for e in sorted_entries]

    def _get_col_for_pct(self, pct, reverse: bool = False) -> str:
        idx = int(pct / 10)
        idx = idx if idx < 10 else 9
        idx = 9 - idx if reverse else idx

        if len(self.palette):
            if idx < 0:
                self.logger.error(
                    f"Unable to get palette color for given percentage - palette "
                    f"index underflow: {len(self.palette)}, idx: {idx}, pct: {pct}"
                )
                idx = 0
            if idx >= len(self.palette):
                self.logger.error(
                    f"Unable to get palette color for given percentage - palette "
                    f"index overflow: {len(self.palette)}, idx: {idx}, pct: {pct}"
                )
                idx = len(self.palette) - 1
            return self.palette[idx]
        else:
            return LlmHeatmapLeaderboardExplanation.COLOR_FATAL_ERROR

    def _get_col_for_value(
        self, max_val: float, min_val: float, val: float, reverse: bool = False
    ) -> str:
        if (
            math.isnan(val)
            or math.isinf(val)
            or math.isnan(max_val)
            or math.isnan(min_val)
            or math.isinf(max_val)
            or math.isinf(min_val)
        ):
            return LlmHeatmapLeaderboardExplanation.COLOR_FATAL_ERROR

        if max_val == min_val:
            pct = 50
        else:
            val_rng = max_val - min_val or 1.0
            pct = int((val - min_val) / (val_rng / 100.0))

        return self._get_col_for_pct(pct, reverse=reverse)

    @staticmethod
    def truncate(f, n):
        """Truncates a float f to n decimal places without rounding."""
        return math.floor(f * 10**n) / 10**n

    def _as_html_table_models_by_metric(
        self,
        html_src,
        anchor_uuid_map: dict,
        sort_by_metric_id,
        include_prompts_by_metrics: bool = True,
    ):
        """Render HTML table with models by (aggregated) metric value.

        Parameters
        ----------
        html_src :
            HTML source.
        anchor_uuid_map : dict
            Map (model name -> anchor UUID) used to make links from the table
            to the sections with the model failures (has UUID as HTML element ID).
        sort_by_metric_id : str
            Metric ID to be used as the FIRST one to sort the table. Then the method
            renders tables for all other metrics (sorted by that particular metric).

        """
        # cluster rows by the MODEL ID to get per metric average
        #   - agg data map: model -> metric -> value
        #   - metric eda map: metric -> [max, min]
        (agg_data, metric_eda) = self._eda()

        # if there is exactly 1 model, then no need to sort by each metric
        normalized_metric_ids_order = (
            [self.metric_ids_order[0]]
            if len(self.leaderboard_order) == 1
            else self.metric_ids_order
        )

        for sort_by_metric_id in normalized_metric_ids_order:
            # move sort_by_metric_id to the first item in self.metric_ids_order
            table_metrics_order = self.metric_ids_order.copy()
            if sort_by_metric_id in table_metrics_order:
                table_metrics_order.remove(sort_by_metric_id)
                table_metrics_order.insert(0, sort_by_metric_id)

            self._sort_agg_data(
                sort_by_metric_id,
                agg_data,
                reverse=self.metrics_meta.is_higher_better(sort_by_metric_id),
            )
            sort_metrics_name = self.metrics_meta.get_metric(
                sort_by_metric_id
            ).display_name

            # DEBUG: table data
            # import json
            # with html_src.pre():
            #     html_src(str(json.dumps(self.data_dict,indent=4)))
            # with html_src.pre():
            #     html_src(str(json.dumps(agg_data,indent=4)))
            # with html_src.pre():
            #     html_src(str(json.dumps(metric_eda,indent=4)))

            # MODEL table sorted by given metric
            with html_src.table(klass="w3-table-all"):
                with html_src.tr():
                    html_src.th(_t=f"LLM Models by {sort_metrics_name}")
                    for m in table_metrics_order:
                        html_src.th(
                            _t=self.metrics_meta.get_metric(m).display_name,
                            title=self.metrics_meta.get_metric_description(m),
                        )

                # data dictionary to table: build table row by row
                for llm_model_name in self.leaderboard_order:
                    try:
                        with html_src.tr():
                            with html_src.td():
                                with html_src.a(
                                    href=f"#{anchor_uuid_map[llm_model_name]}"
                                ):
                                    html_src(f"{llm_model_name}")
                                    prefix = self.rag_type_prefix.get(
                                        llm_model_name, "LLM"
                                    )
                                    html_src.sup(_t=f"{prefix}")

                            for m in table_metrics_order:
                                m_avg = agg_data[llm_model_name][m]
                                color = self._get_col_for_value(
                                    min_val=metric_eda[m][0],
                                    max_val=metric_eda[m][1],
                                    val=m_avg,
                                    reverse=not self.metrics_meta.is_higher_better(m),
                                )
                                with html_src.td(style=f"background-color: #{color};"):
                                    if math.isnan(m_avg) or math.isinf(m_avg):
                                        html_src("N/A (evaluator failed)")
                                    else:
                                        # .3 causes round bias: html_src(f"{m_avg:.3f}")
                                        v = LlmHeatmapLeaderboardExplanation.truncate(
                                            m_avg, 5
                                        )
                                        html_src(f"{v}")
                    except Exception as ex:
                        self.logger.error(
                            f"Unable to render HTML representation for model "
                            f"{llm_model_name}: {ex}\n{traceback.format_exc()}"
                        )

            # PROMPTS table sorted by given metric
            if include_prompts_by_metrics:
                if self.i_failures:
                    html_src.br()

                    self._as_html_table_prompts_by_metric(
                        html_src=html_src,
                        anchor_uuid_map=anchor_uuid_map,
                        metric_id=sort_by_metric_id,
                        metric_name=sort_metrics_name,
                    )

            html_src.br()

    def _as_html_table_prompts_by_metric(
        self, html_src, anchor_uuid_map: dict, metric_id, metric_name
    ):
        # DATA preparation
        # data: failing prompts to the tuple for sorting
        leaderboard_data = []
        # metric_min = 1.0
        # metric_max = 0.0
        for prompt in self.i_failures:
            for _, result_row, failure_metric_id, _ in self.i_failures[prompt]:
                if (
                    metric_id == failure_metric_id
                    and result_row.metrics.get(metric_id, None) is not None
                ):
                    metric_val = result_row.metrics.get(metric_id, 0.0)
                    # metric_min = min(metric_min, metric_val)
                    # metric_max = max(metric_max, metric_val)
                    leaderboard_data.append(
                        (
                            prompt,
                            result_row.dataset_row.model_key,
                            metric_val,
                        )
                    )
        # sort the data by the metric value
        nan_deputy = (
            sys.float_info.min
            if self.metrics_meta.is_higher_better(metric_id)
            else sys.float_info.max
        )
        leaderboard_data.sort(
            key=lambda x: x[2] if not math.isnan(x[2]) else nan_deputy,
            reverse=not self.metrics_meta.is_higher_better(metric_id),
        )

        if leaderboard_data:
            with html_src.h6():
                html_src.b(_t=f"Model weak points for {metric_name} metric")
            html_src(
                "Prompts for which models achieved the lowest evaluation metric values"
                " - models weak points. Prompts are ordered by the evaluation metric "
                "value from the lowest to the highest along with the model which "
                "achieved the score:"
            )

            # PROMPT table sorted by given metric
            with html_src.table(klass="w3-table-all"):
                with html_src.tr():
                    html_src.th(_t=f"Prompts&nbsp;by&nbsp;{metric_name}&nbsp;metric")
                    html_src.th(_t="LLM&nbsp;model")
                    html_src.th(_t="Metric&nbsp;value")

                for e in leaderboard_data:
                    with html_src.tr():
                        llm_model_name = (
                            self.key_2_evaluated_model[e[1]].llm_model_name
                            if self.key_2_evaluated_model.get(e[1])
                            else "LLM model"
                        )
                        bg_color = self._get_col_for_value(
                            min_val=0.0,
                            max_val=1.0,
                            val=e[2],
                            reverse=not self.metrics_meta.is_higher_better(metric_id),
                        )

                        with html_src.td():
                            html_src(e[0])
                        with html_src.td():
                            with html_src.a(href=f"#{anchor_uuid_map[llm_model_name]}"):
                                html_src(f"{llm_model_name}")
                                prefix = self.rag_type_prefix.get(llm_model_name, "LLM")
                                html_src.sup(_t=f"{prefix}")
                        with html_src.td(style=f"background-color: #{bg_color};"):
                            html_src(f"{e[2]}")

    def as_dict(self, threshold: float | None = None) -> tuple[dict, dict]:
        """Return leaderboard as dictionary.

        Parameters
        ----------
        threshold :  float | None
            Threshold for the metrics.

        Returns
        -------
        Tuple[dict, dict] :
            Leaderboard data dictionary and metric EDA (min, max, ...) dictionary.

        """
        (metrics_values, metrics_eda) = self._eda()
        return {
            f5s.ExplanationFormat.KEY_DATA: metrics_values,
            f5s.ExplanationFormat.KEY_METADATA: self.metrics_meta.to_dict(threshold),
        }, metrics_eda

    @staticmethod
    def _get_context_title(ctx, length):
        try:
            index = length - ctx[:length][::-1].index(" ") - 1
            if index < length * 0.75:
                index = length
        except ValueError:
            index = length
        return f"{ctx[:index]}..."

    def as_html(
        self,
        sort_by_metric_id: str,
        html_src=None,
        include_failures: bool = True,
        include_prompts_by_metrics: bool = True,
        additional_details: dict | None = None,
    ) -> str:
        """Create HTML snippet with:

        - metrics heatmap table
        - failures section: model -> document -> prompt -> [metrics] -> value

        """
        html_src = html_src or airium.Airium()

        # map: model name -> uuid (used for anchors)
        anchor_uuid_map = {
            model_name: str(uuid.uuid4()) for model_name in self.leaderboard_order
        }

        # SECTION: models leaderboard for GIVEN metric (order by metric score)
        self._as_html_table_models_by_metric(
            html_src=html_src,
            anchor_uuid_map=anchor_uuid_map,
            sort_by_metric_id=sort_by_metric_id,
            include_prompts_by_metrics=include_prompts_by_metrics,
        )

        # SECTION: model & prompt failures (metrics > threshold):
        if include_failures and self.i_failures:
            with html_src.h6():
                html_src.b(_t="Most difficult prompts")

            html_src(
                "Prompts ordered by failures across all models and all metrics - "
                "leaderboard of the most difficult prompts. If the metric value for "
                "the actual answer created by a model for a given prompt is below "
                "the threshold specified for the metric, then it is considered "
                "a failure (Fail):"
            )

            self.sort_prompts_by_failures(self.i_failures_count)
            self._i_as_html_table(
                html_src=html_src, lead_col_name="Prompts by Failures"
            )

            html_src.br()

            with html_src.h6():
                html_src.b(_t="Model failures")

            for llm_model_name in self.m_failures:
                with html_src.span(id=f"{anchor_uuid_map[llm_model_name]}"):
                    html_src("Model ")
                    with html_src.b():
                        html_src(f"{llm_model_name}")
                    html_src("failures:")
                with html_src.ul():
                    key_err_msg = 0
                    key_row = 1
                    key_corpus = 2

                    for failure in self.m_failures[llm_model_name]:

                        def __html_src_model_failure():
                            with html_src.li():
                                html_src.b(_t="Prompt: ")
                                html_src(f"{failure[key_row].dataset_row.i}")
                            v = failure[key_row].dataset_row.expected_output
                            if v:
                                with html_src.li():
                                    html_src.b(_t="Expected output: ")
                                    html_src(f"{v}")
                            # add actual output with meta
                            LlmLeaderboardExplanation._html_aa_meta(
                                html_src=html_src,
                                actual_output=failure[
                                    key_row
                                ].dataset_row.actual_output,
                                actual_output_meta=failure[key_row].actual_output_meta,
                                err_msg=failure[key_err_msg],
                                metrics_meta=self.metrics_meta,
                                metrics=failure[key_row].metrics,
                            )
                            if failure[key_row].dataset_row.context:
                                with html_src.li():
                                    html_src.b(_t="Context:")
                            for ctx in failure[key_row].dataset_row.context:
                                with html_src.details():
                                    html_src.summary(
                                        _t=self._get_context_title(ctx, 64)
                                    )
                                    html_src.p(_t=ctx)

                        if (
                            self.llm_host != commons.LlmModelHostType.RAG
                            or not failure[key_corpus]
                        ):
                            with html_src.li():
                                html_src.b(
                                    _t=f"Error: {failure[key_err_msg]}",
                                    klass="w3-text-red",
                                )
                                with html_src.ul():
                                    __html_src_model_failure()
                        else:
                            with html_src.li():
                                with html_src.b():
                                    corpus = failure[key_corpus]
                                    if (
                                        corpus
                                        and isinstance(corpus, str)
                                        and corpus.startswith("http")
                                    ):
                                        with html_src.a(href=corpus):
                                            html_src(f"{corpus}")
                                    else:
                                        html_src(f"{failure[key_corpus]}")
                                with html_src.ul():
                                    with html_src.li():
                                        html_src.b(
                                            _t=f"Error: {failure[key_err_msg]}",
                                            klass="w3-text-red",
                                        )
                                    __html_src_model_failure()

        # SECTION: additional details
        if additional_details:
            html_src.br()
            html_src("Additional details:")
            with html_src.ul():
                for k in additional_details:
                    with html_src.li():
                        html_src.b(_t=f"{k}:")
                        html_src.br()
                        ad = additional_details[k]
                        if isinstance(
                            ad, LlmBoolLeaderboardExplanation.AdditionalDetails
                        ):
                            if ad.formatting == "pre":
                                with html_src.pre():
                                    html_src(f"{ad.text}")
                            elif ad.formatting == "i":
                                with html_src.i():
                                    html_src(f"{ad.text}")
                            else:
                                html_src(f"{ad.text}")
                        else:
                            html_src(f"{ad}")

        # if LLM dataset is empty, or evaluation of all rows from the LLM dataset fails,
        # then return "No results."
        return str(html_src) or (
            "<span>"
            "No explanations - either the dataset is empty or the evaluation of "
            "all rows from the dataset failed. In this case, please check the "
            "problems section for more details."
            "</span>"
        )

    def sort_prompts_by_failures(
        self, sort_by: dict[str, int | float], reverse: bool = True
    ):
        # build sort data structure - value can be parametrized
        unsorted_entries = [
            (m, sort_by.get(m, 0)) for m in self.inputs_leaderboard_order
        ]
        # sort
        sorted_entries = sorted(unsorted_entries, key=lambda x: x[1], reverse=reverse)

        self.inputs_leaderboard_order = [e[0] for e in sorted_entries]

    def _i_as_html_table(self, html_src, lead_col_name="Inputs"):
        with html_src.table(klass="w3-table-all"):
            with html_src.tr():
                html_src.th(_t=lead_col_name)
                html_src.th(_t="Pass")
                html_src.th(_t="Fail")
                html_src.th(_t="Failure rate")

            for i in self.inputs_leaderboard_order:
                passed = self.i_passes_count.get(i, 0)
                failures = self.i_failures_count.get(i, 0)
                accuracy = 100.0 - (passed / (passed + failures)) * 100.0

                if failures > 0:
                    try:
                        with html_src.tr():
                            html_src.td(_t=i)
                            with html_src.td():
                                html_src(passed)
                            with html_src.td():
                                html_src(failures)
                            color = self._get_col_for_pct(accuracy, reverse=True)
                            with html_src.td(style=f"background-color: #{color};"):
                                html_src(f"{accuracy:.3f}%")
                    except Exception as ex:
                        self.logger.error(
                            f"Unable to render HTML representation for input "
                            f"{i}: {ex}\n{traceback.format_exc()}"
                        )

    def add_markdown_format(
        self, sort_by_metric_id: str, title: str = "Evaluation Report"
    ):
        report_path = self.explainer.persistence.get_evaluator_working_file("report.md")
        with open(report_path, mode="w") as file:
            file.write(
                self.as_markdown(
                    sort_by_metric_id=sort_by_metric_id,
                    title=title,
                )
            )

        self.add_format(
            f5s.MarkdownFormat(
                explanation=self,
                format_file=report_path,
            )
        )

    def add_evalstudio_markdown_format(
        self, sort_by_metric_id: str, title: str = "Summary"
    ):
        report_path = self.explainer.persistence.get_evaluator_working_file("report.md")
        with open(report_path, mode="w") as file:
            file.write(
                self.as_markdown(
                    sort_by_metric_id=sort_by_metric_id,
                    title=title,
                    heading_level="##",
                )
            )

        self.add_format(
            f5s.EvalStudioMarkdownFormat(
                explanation=self,
                format_file=report_path,
            )
        )

    def as_markdown(
        self,
        sort_by_metric_id: str,
        title: str = "Evaluation Report",
        heading_level: str = "#",
        top: int = 3,
    ) -> str:
        """Return Markdown representation of the leaderboard for EvalStudio.

        Parameters
        ----------
        sort_by_metric_id : str
            Metric ID to be used as the FIRST one to sort the table. Then the method
            renders tables for all other metrics (sorted by that particular metric).
        title : str
            Title of the leaderboard.
        heading_level : str
            Markdown title heading level.
        top : int
            Number of top model failures, prompt failures, empty context prompts, ...
            entries. `0` for all entries.
            The motivation is to avoid LONG reports with all failures and prompts,
            it's just a summary.

        """

        md = ""
        md += f"{heading_level} {title}\n"
        md += "\n"
        md = LlmBoolLeaderboardExplanation.summary_as_markdown(
            md=md,
            metrics_count=self.metrics_meta.size(),
            llm_host=self.llm_host,
            m_failures_count=self.m_failures_count,
            i_failures_count=self.i_failures_count,
            key_2_evaluated_model=self.key_2_evaluated_model,
        )
        md += "\n"

        # models by metrics
        (agg_data, metric_eda) = self._eda()
        if sort_by_metric_id in self.metric_ids_order:
            self.metric_ids_order.remove(sort_by_metric_id)
            self.metric_ids_order.insert(0, sort_by_metric_id)
        for sort_by_metric_id in self.metric_ids_order:
            self._sort_agg_data(
                sort_by_metric_id,
                agg_data,
                reverse=self.metrics_meta.is_higher_better(sort_by_metric_id),
            )

            md += "\n"
            md += (
                f"## Models by "
                f"{self.metrics_meta.get_metric(sort_by_metric_id).display_name}"
                f"\n"
            )
            md += "Models ordered by the evaluation metric value.\n"
            md += "\n"
            md += "| Rank | LLM "
            for m in self.metric_ids_order:
                md += f"| {self.metrics_meta.get_metric(m).display_name}"
            md += " |\n| --- | --- "
            for _ in self.metric_ids_order:
                md += "| ---"
            md += " |\n"

            for e, llm_model_name in enumerate(self.leaderboard_order):
                md += f"| {e + 1} | {llm_model_name} "
                for m in self.metric_ids_order:
                    m_avg = agg_data[llm_model_name][m]
                    md += f"| {m_avg:.4f} "
                md += "|\n"

        # prompts failures
        if self.i_failures:
            md += "\n"
            md += "## Most difficult prompts\n"
            if top:
                md += "Top prompts "
            else:
                md += "Prompts "
            md += (
                "ordered by model failures - leaderboard of the most "
                "difficult prompts. If the metric value for the actual answer created "
                "by a model for a given prompt is below the threshold specified for "
                "the metric, then it is considered a failure."
            )
            if top:
                md += " See the `Report` for the full list."
            md += "\n\n"
            md += "| Prompt | Pass | Fail | Failures rate |\n"
            md += "| --- | --- | --- | --- |\n"

            self.sort_prompts_by_failures(self.i_failures_count)
            for e, i in enumerate(self.inputs_leaderboard_order):
                if top and e >= top:
                    break

                i_safe = sanitization.sanitize_markdown(md_fragment=i)
                passed = self.i_passes_count.get(i, 0)
                failures = self.i_failures_count.get(i, 0)
                accuracy = 100.0 - (passed / (passed + failures)) * 100.0

                if failures > 0:
                    try:
                        md += (
                            f"| {i_safe} | {passed} | {failures} | {accuracy:.1f}% |\n"
                        )
                    except Exception as ex:
                        self.logger.error(
                            f"Unable to render Markdown representation for input "
                            f"{i}: {ex}\n{traceback.format_exc()}"
                        )

        return md

    def add_json_format(
        self, threshold: float | None = None
    ) -> f5s.LlmHeatmapLeaderboardJSonFormat:
        """Add JSon format."""
        key_all_metrics = f5s.LlmHeatmapLeaderboardJSonFormat.KEY_ALL_METRICS

        (leaderboard_dict, eda_dict) = self.as_dict(threshold)

        # map: metric_id -> {min, max}
        result_eda = {}
        # map: metric_id -> {model_name -> {metric id -> value}}
        result_leaderboards = {}
        for metric_id in eda_dict:
            # EDA
            result_eda[metric_id] = {}
            result_eda[metric_id]["min"] = eda_dict[metric_id][0]
            result_eda[metric_id]["max"] = eda_dict[metric_id][1]

            # per-metric leaderboard
            result_leaderboards[metric_id] = {}
            for model_name in leaderboard_dict[f5s.ExplanationFormat.KEY_DATA]:
                result_leaderboards[metric_id][model_name] = {}
                result_leaderboards[metric_id][model_name][metric_id] = (
                    leaderboard_dict[f5s.ExplanationFormat.KEY_DATA][model_name][
                        metric_id
                    ]
                )

        metrics_for_json = list(eda_dict.keys()) + [key_all_metrics]
        (idx, idx_str) = f5s.LlmHeatmapLeaderboardJSonFormat.serialize_index_file(
            metrics=metrics_for_json,
        )
        json_format = f5s.LlmHeatmapLeaderboardJSonFormat(
            explanation=self,
            json_data=idx_str,
            persistence=self.explainer.persistence.store,
        )
        # save data files: leaderboards
        for metric_id in eda_dict:
            json_format.add_data(
                format_data=json.dumps(
                    result_leaderboards.get(metric_id, {}),
                    indent=4,
                    cls=persistences.NanEncoder,
                ),
                file_name=idx[f5s.ExplanationFormat.KEY_FILES][metric_id],
            )
        json_format.add_data(
            format_data=json.dumps(
                leaderboard_dict, indent=4, cls=persistences.NanEncoder
            ),
            file_name=idx[f5s.ExplanationFormat.KEY_FILES][key_all_metrics],
        )

        self.add_format(json_format)
        return json_format

    def get_insights(
        self,
        metrics_meta: commons.MetricsMeta,
        metric_id: str = "",
        metric_name_protection: bool = False,
        extra_description_best: str = "",
        extra_description_worst: str = "",
        insight_type: str = "accuracy",
        model_purpose: str = "",
        explanation_type: str = "",
        explanation_name: str = "",
        explanation_mime: str = "",
    ) -> None:
        """Create insights for the heatmap leaderboard.

        Parameters
        ----------
        metrics_meta : commons.MetricsMeta
            Metrics metadata.
        metric_id : str
            Optional metric ID to create insights for. If not specified, then insights
            are created for the primary metrics as specified by the metrics metadata.
        metric_name_protection : bool
            If True, then the metric ID is not changed to lowercase.
        extra_description_best: str
            Additional description for insights related to the best models.
        extra_description_worst: str
            Additional description for insights related to the worst models.
        insight_type : str
            Insight type.
        model_purpose : str
            Model purpose.
        explanation_type : str
            Type of the explanation which can clarify the insight.
        explanation_name : str
            Name of the explanation which can clarify the insight.
        explanation_mime : str
            Media type of the explanation which can clarify the insight.

        """
        t_insights = insights.InsightAndAction

        evaluator_name = self.explainer._display_name

        metric_id = metric_id or metrics_meta.get_primary_metric().key
        metric_name = (
            metrics_meta.get_metric(metric_id).display_name
            if metric_name_protection
            else metrics_meta.get_metric(metric_id).display_name.lower()
        )
        if self.insights_metric_2_winner and self.insights_metric_2_winner.get(
            metric_id
        ):
            model_name = self.insights_metric_2_winner.get(metric_id)

            html = airium.Airium()
            html("The ")
            with html.code():
                html(model_name)
            with html.b(klass="w3-black"):
                html("&nbsp;model&nbsp;")
            html("&nbsp; evaluated as the model with ")
            with html.span(klass="w3-black"):
                html("&nbsp;the best&nbsp;")
                with html.b(klass="w3-black"):
                    html(f"{metric_name}&nbsp;")
            html("&nbsp; metric score calculated by")
            with html.code():
                html(evaluator_name)
            html(f" evaluator. {extra_description_best}")

            self.explainer.add_insight(
                insights.InsightAndAction(
                    description=(
                        f"The {model_name} model evaluated as the model with the best "
                        f"{metric_name} metric score calculated by {evaluator_name} "
                        f"evaluator. {extra_description_best}"
                    ),
                    description_html=html,
                    insight_type=insight_type,
                    insight_attrs={
                        insights.InsightAndAction.ATTR_MODEL_NAME: model_name,
                        insights.InsightAndAction.ATTR_EVALUATOR_NAME: evaluator_name,
                    },
                    actions_description=(
                        "Refer to the explanation for the detailed description of "
                        "questions, answers, and failures."
                    ),
                    evaluator_id=self.explainer.evaluator_id(),
                    evaluator_name=evaluator_name,
                    explanation_type=explanation_type,
                    explanation_name=explanation_name,
                    explanation_mime=explanation_mime,
                    resources=[],
                )
            )

        if self.insights_difficult_prompt:
            prompt = self.insights_difficult_prompt

            self.explainer.add_insight(
                insights.InsightAndAction(
                    description=(
                        f"The '{prompt}' prompt was evaluated as the most difficult "
                        f"prompt to be correctly answered according to "
                        f"{evaluator_name} evaluator. {extra_description_worst}"
                    ),
                    description_html=t_insights.html_most_difficult_prompt_by(
                        prompt=prompt,
                        evaluator_name=evaluator_name,
                        extra_description=extra_description_worst,
                        model_purpose=model_purpose,
                    ),
                    insight_type=insight_type,
                    actions_description=(
                        "Refer to the explanation for the detailed description of "
                        "questions and answers by evaluated models in order to "
                        "identify weaknesses and strengths."
                    ),
                    evaluator_id=self.explainer.evaluator_id(),
                    evaluator_name=evaluator_name,
                    explanation_type=explanation_type,
                    explanation_name=explanation_name,
                    explanation_mime=explanation_mime,
                    resources=[],
                )
            )

    LLM_MODEL_ANONYMOUS = "model"

    @staticmethod
    def from_eval_results(
        evaluator,
        eval_results,
        metrics_meta: commons.MetricsMeta,
        key_2_evaluated_model: dict,
        llm_host: commons.LlmModelHostType = commons.LlmModelHostType.RAG,
        nan_tolerance: float = 0.0,
        display_name: str = None,
        display_category: str = None,
        logger=None,
    ) -> "LlmHeatmapLeaderboardExplanation":
        """Create Heatmap leaderboard explanation from the evaluation results.

        Parameters
        ----------
        evaluator :
            Evaluator instance.
        eval_results : datasets.LlmEvalResults
            Evaluation results.
        metrics_meta : commons.MetricsMeta
            Metadata of the metric to be evaluated.
        key_2_evaluated_model : dict
            Map: key -> LLM@RAG/LLM model.
        llm_host : commons.LlmModelHostType
            LLM host type - either a RAG (with retrieval) or a LLM (generation only).
        nan_tolerance : float
            Tolerance for ``NaN`` values in the evaluation results.
        display_name : str
            Custom leaderboard display name.
        display_category : str
            Custom leaderboard display category.
        logger :
            Optional logger.

        """
        if (
            not eval_results
            or eval_results.results is None
            or len(eval_results.results) == 0
        ):
            raise ValueError(
                f"No evaluation results created by the evaluator "
                f"'{evaluator.display_name}' - unable to create the leaderboard."
            )

        heatmap_explanation = LlmHeatmapLeaderboardExplanation(
            evaluator=evaluator,
            eval_results=eval_results,
            metrics_meta=metrics_meta,
            key_2_evaluated_model=key_2_evaluated_model,
            llm_host=llm_host,
            display_name=display_name,
            display_category=display_category,
            nan_tolerance=nan_tolerance,
            logger=logger or loggers.SonarPrintLogger(),
        )

        # populate leaderboard data
        for r in eval_results.results:
            evaluated_model = heatmap_explanation.key_2_evaluated_model.get(
                r.dataset_row.model_key
            )

            for metric_id in r.metrics:
                heatmap_explanation.add_col_value(
                    llm_model_name=(
                        evaluated_model.llm_model_name
                        if evaluated_model
                        else LlmHeatmapLeaderboardExplanation.LLM_MODEL_ANONYMOUS
                    ),
                    docs=(
                        evaluated_model.documents[0]
                        if isinstance(evaluated_model, models.ExplainableRagModel)
                        and evaluated_model.documents
                        else ""
                    ),
                    prompt=r.dataset_row.i,
                    metrics_id=metric_id,
                    value=r.metrics.get(metric_id, 0.0),
                    result_row=r,
                )

        heatmap_explanation.build()

        return heatmap_explanation


class LlmProcedureEvalLeaderboardExplanation(
    _explanations_base.Explanation, LlmLeaderboardExplanation, AbcHeatmapExplanation
):
    """ProcedureEval leaderboard explanation provides data and formats for a leaderboard
    which is colorized as procedure_eval based on metrics values.

    """

    _explanation_type = "llm-procedure-eval-leaderboard"
    _is_global = True

    KEY_DYN_PROG_MATRIX = "dyn_prog_matrix"
    KEY_ALIGNMENT_MATRIX = "alignment_matrix"

    def __init__(
        self,
        evaluator,
        eval_results,
        metrics_meta: commons.MetricsMeta,
        key_2_evaluated_model: dict,
        llm_host: commons.LlmModelHostType = commons.LlmModelHostType.RAG,
        display_name: str = "",
        display_category: str = "",
        logger=None,
    ) -> None:
        display_name = display_name or "LLM ProcedureEval Leaderboard"

        _explanations_base.Explanation.__init__(
            self,
            explainer=evaluator,
            display_name=display_name,
            display_category=display_category,
        )

        self.eval_results = eval_results

        # LLM@RAG/LLM models map: key -> model instance
        self.key_2_evaluated_model = key_2_evaluated_model
        self.llm_host = llm_host

        # metrics metadata:
        # - metric names map: key -> metric name
        # - metric thresholds: key -> threshold
        # - metric higher is better map: key -> bool
        self.metrics_meta = metrics_meta
        # actual metrics which present in the data (metrics names map might have more)
        self.metric_ids_order = []
        # board order: as models are dict, model (1st) col is sorted by given metric
        self.leaderboard_order = [
            m.llm_model_name for m in self.key_2_evaluated_model.values()
        ]

        # palette from dark green to dark red by 10%
        self.palette = LlmProcedureEvalLeaderboardExplanation.PALETTE_RED.copy()
        r = LlmProcedureEvalLeaderboardExplanation.PALETTE_GREEN.copy()
        r.reverse()
        self.palette.extend(r)

        self.logger = logger or loggers.SonarPrintLogger()

        # data map: model -> document -> prompt -> metrics -> value
        self.data_dict = {}
        # model failures map: model ID -> (err_msg, row_as_dict, doc)
        self.m_failures = {}
        # model failures count map: model ID -> count
        self.m_failures_count = {}

        # prompt failures map: prompt -> (err_msg, row_as_dict, metric_id, doc)
        self.i_failures = {}
        # prompt failures count map: prompt -> count
        self.i_failures_count = {}
        # prompt passes
        self.i_passes_count = {}
        # prompts leaderboard order by number of failures
        self.inputs_leaderboard_order = []

        # tested LLM host (h2oGPTe/OpenAI RAG/...) to prefix LLMs in leaderboards
        # map: model name -> prefix
        if key_2_evaluated_model:
            self.rag_type_prefix = LlmBoolLeaderboardExplanation.key_2_rag_type_prefix(
                key_2_evaluated_model.values()
            )
        else:
            self.rag_type_prefix = {}

        # insights
        # map: metric_id -> LLM model ID
        self.insights_metric_2_winner = {}
        self.insights_metric_2_looser = {}
        self.insights_difficult_prompt = ""

    def add_col_value(
        self,
        llm_model_name: str,
        docs: str,  # TODO list of strings (document names)
        prompt: str,
        metrics_id: str,
        value: float,
        result_row,  # datasets.LlmEvalResults.LllmEvalResultRow
    ):
        """Add entry to the data dictionary used to build formatted tables later."""
        if llm_model_name not in self.data_dict:
            self.data_dict[llm_model_name] = {}
        if docs not in self.data_dict[llm_model_name]:
            self.data_dict[llm_model_name][docs] = {}
        if prompt not in self.data_dict[llm_model_name][docs]:
            self.data_dict[llm_model_name][docs][prompt] = {}

        if (
            self.data_dict[llm_model_name][docs][prompt].get(metrics_id, None)
            is not None
        ):
            self.logger.warning(
                f"CONFLICT when adding metrics value to the procedure_eval on entry: "
                f"{llm_model_name} {docs} {prompt} {metrics_id} "
                f"(entry with such key already exists)"
            )

        self.data_dict[llm_model_name][docs][prompt][metrics_id] = value

        # generate failure using row_as_dict and discard the dict
        threshold = self.metrics_meta.get_threshold(metrics_id, default_value=0.0)
        if value is not None and isinstance(value, str):
            raise ValueError(
                f"Invalid metric value '{value}' for '{metrics_id}' metric in the "
                f"leaderboard (threshold is {threshold}) - expected float, "
                f"not {type(value)}."
            )
        if math.isnan(value) or math.isinf(value):
            is_failure = True
            err_msg = (
                f"'{metrics_id}' metric value '{value}' is invalid, because "
                f"the evaluator failed to calculate it (threshold is {threshold})"
            )
        elif self.metrics_meta.is_higher_better(metrics_id):
            is_failure = value < threshold
            err_msg = (
                f"'{metrics_id}' metric value '{value}' is below the threshold "
                f"{threshold}"
            )
        else:
            is_failure = value > threshold
            err_msg = (
                f"'{metrics_id}' metric value '{value}' is above the threshold "
                f"{threshold}"
            )

        if is_failure:
            # model failure
            if llm_model_name not in self.m_failures:
                self.m_failures[llm_model_name] = []
            if llm_model_name not in self.m_failures_count:
                self.m_failures_count[llm_model_name] = 0
            self.m_failures[llm_model_name].append((err_msg, result_row, docs))
            self.m_failures_count[llm_model_name] += 1

            # prompt failure
            if prompt not in self.i_failures:
                self.i_failures[prompt] = []
            if prompt not in self.i_failures_count:
                self.i_failures_count[prompt] = 0
            self.i_failures[prompt].append((err_msg, result_row, metrics_id, docs))
            self.i_failures_count[prompt] += 1

            if prompt not in self.inputs_leaderboard_order:
                self.inputs_leaderboard_order.append(prompt)
        else:
            if prompt not in self.i_passes_count:
                self.i_passes_count[prompt] = 1
            else:
                self.i_passes_count[prompt] += 1

    def build(self):
        """Analyze, explain, aggregate, and build leaderboard data... so that when
        HTML representation is built, the leaderboard is ready to be rendered.

        """
        (agg_data, _) = self._eda()
        for sort_by_metric_id in self.metric_ids_order:
            self._sort_agg_data(
                sort_by_metric_id,
                agg_data,
                reverse=self.metrics_meta.is_higher_better(sort_by_metric_id),
            )

            # insights
            if len(self.leaderboard_order) > 1:
                self.insights_metric_2_winner[sort_by_metric_id] = (
                    self.leaderboard_order[0]
                )
                self.insights_metric_2_looser[sort_by_metric_id] = (
                    self.leaderboard_order[-1]
                )

        if self.inputs_leaderboard_order:
            self.sort_prompts_by_failures(self.i_failures_count)
            self.insights_difficult_prompt = self.inputs_leaderboard_order[0]

    def __str__(self) -> str:
        return str(self.data_dict)

    def validate(self) -> bool:
        return self._formats is not None

    def __add_to_eda_vals(
        self, metrics_vals: dict, llm_model_name: str, metric_id: str, metric_val: float
    ):
        if metric_id not in self.metric_ids_order:
            self.metric_ids_order.append(metric_id)

        if llm_model_name not in metrics_vals:
            metrics_vals[llm_model_name] = {}
        if metric_id not in metrics_vals[llm_model_name]:
            metrics_vals[llm_model_name][metric_id] = [0.0, 0.0]

        _m_sum = metrics_vals[llm_model_name][metric_id][0] + metric_val
        _m_count = metrics_vals[llm_model_name][metric_id][1] + 1.0
        metrics_vals[llm_model_name][metric_id] = [_m_sum, _m_count]

    def _eda(self) -> tuple[dict, dict]:
        # EDA:
        #   Use metrics from the classification evaluator to create a procedure_eval
        #   leaderboard data - reuse procedure_eval explanation for the rendering and
        #   representations.
        #
        # RESULT: procedure_eval leaderboard dicts definition:
        # - metrics values:
        #     model -> metric ID -> (avg) value
        # - metrics EDA:
        #     metric ID -> [min, max]
        #
        # METHOD: aggregated data map: model -> metric -> avg value
        # 1. cluster the data
        #    aggregated data map: model -> metric -> [sum, count]
        # 2. calculate average
        #    aggregated data map: model -> metric -> avg value

        # 1. cluster the data
        metrics_vals = {}
        for d_m in self.data_dict:
            for d_d in self.data_dict[d_m]:
                for d_p in self.data_dict[d_m][d_d]:
                    for d_mm in self.data_dict[d_m][d_d][d_p]:
                        self.__add_to_eda_vals(
                            metrics_vals=metrics_vals,
                            llm_model_name=d_m,
                            metric_id=d_mm,
                            metric_val=self.data_dict[d_m][d_d][d_p][d_mm],
                        )

        # 2. calculate average + metrics EDA
        metrics_eda = {m: [100.0, 0.0] for m in self.metric_ids_order}
        for d_m in metrics_vals:
            for d_mm in metrics_vals[d_m]:
                m_sum = metrics_vals[d_m][d_mm][0]
                m_count = metrics_vals[d_m][d_mm][1]
                m_avg = m_sum / float(m_count)  # IMPROVE: harmonic mean vs. mean
                m_min = min(metrics_eda[d_mm][0], m_avg)
                m_max = max(metrics_eda[d_mm][1], m_avg)

                metrics_vals[d_m][d_mm] = m_avg
                metrics_eda[d_mm] = [m_min, m_max]

        return metrics_vals, metrics_eda

    def _sort_agg_data(self, metric_id: str, agg_data: dict, reverse: bool = True):
        """Sort data dictionary by the given metric to generate table which is
        sorted by given column.

        """
        # metric-based dir used to sort models
        sort_by = [
            (m, agg_data[m][metric_id]) for m in agg_data if metric_id in agg_data[m]
        ]

        # NaN comparison is not define for floats -> define NaN as min/max
        nan_deputy = (
            sys.float_info.min
            if self.metrics_meta.is_higher_better(metric_id)
            else sys.float_info.max
        )

        # build sort data structure - value can be parametrized
        sorted_entries = sorted(
            sort_by,
            key=lambda x: x[1] if not math.isnan(x[1]) else nan_deputy,
            reverse=reverse,
        )

        self.leaderboard_order = [e[0] for e in sorted_entries]

    def _get_col_for_pct(self, pct, reverse: bool = False) -> str:
        idx = int(pct / 10)
        idx = idx if idx < 10 else 9
        idx = 9 - idx if reverse else idx

        if len(self.palette):
            if idx < 0:
                self.logger.error(
                    f"Unable to get palette color for given percentage - palette "
                    f"index underflow: {len(self.palette)}, idx: {idx}, pct: {pct}"
                )
                idx = 0
            if idx >= len(self.palette):
                self.logger.error(
                    f"Unable to get palette color for given percentage - palette "
                    f"index overflow: {len(self.palette)}, idx: {idx}, pct: {pct}"
                )
                idx = len(self.palette) - 1
            return self.palette[idx]
        else:
            return LlmProcedureEvalLeaderboardExplanation.COLOR_FATAL_ERROR

    def _get_col_for_value(
        self, max_val: float, min_val: float, val: float, reverse: bool = False
    ) -> str:
        if (
            math.isnan(val)
            or math.isinf(val)
            or math.isnan(max_val)
            or math.isnan(min_val)
            or math.isinf(max_val)
            or math.isinf(min_val)
        ):
            return LlmProcedureEvalLeaderboardExplanation.COLOR_FATAL_ERROR

        if max_val == min_val:
            pct = 50
        else:
            val_rng = max_val - min_val or 1.0
            pct = int((val - min_val) / (val_rng / 100.0))

        return self._get_col_for_pct(pct, reverse=reverse)

    @staticmethod
    def truncate(f, n):
        """Truncates a float f to n decimal places without rounding."""
        return math.floor(f * 10**n) / 10**n

    def _as_html_table_models_by_metric(
        self,
        html_src,
        anchor_uuid_map: dict,
        sort_by_metric_id,
        include_prompts_by_metrics: bool = True,
    ):
        """Render HTML table with models by (aggregated) metric value.

        Parameters
        ----------
        html_src :
            HTML source.
        anchor_uuid_map : dict
            Map (model name -> anchor UUID) used to make links from the table
            to the sections with the model failures (has UUID as HTML element ID).
        sort_by_metric_id : str
            Metric ID to be used as the FIRST one to sort the table. Then the method
            renders tables for all other metrics (sorted by that particular metric).

        """
        # cluster rows by the MODEL ID to get per metric average
        #   - agg data map: model -> metric -> value
        #   - metric eda map: metric -> [max, min]
        (agg_data, metric_eda) = self._eda()

        # if there is exactly 1 model, then no need to sort by each metric
        normalized_metric_ids_order = (
            [self.metric_ids_order[0]]
            if len(self.leaderboard_order) == 1
            else self.metric_ids_order
        )

        for sort_by_metric_id in normalized_metric_ids_order:
            # move sort_by_metric_id to the first item in self.metric_ids_order
            table_metrics_order = self.metric_ids_order.copy()
            if sort_by_metric_id in table_metrics_order:
                table_metrics_order.remove(sort_by_metric_id)
                table_metrics_order.insert(0, sort_by_metric_id)

            self._sort_agg_data(
                sort_by_metric_id,
                agg_data,
                reverse=self.metrics_meta.is_higher_better(sort_by_metric_id),
            )
            sort_metrics_name = self.metrics_meta.get_metric(
                sort_by_metric_id
            ).display_name

            # DEBUG: table data
            # import json
            # with html_src.pre():
            #     html_src(str(json.dumps(self.data_dict,indent=4)))
            # with html_src.pre():
            #     html_src(str(json.dumps(agg_data,indent=4)))
            # with html_src.pre():
            #     html_src(str(json.dumps(metric_eda,indent=4)))

            # MODEL table sorted by given metric
            with html_src.table(klass="w3-table-all"):
                with html_src.tr():
                    html_src.th(_t=f"LLM Models by {sort_metrics_name}")
                    for m in table_metrics_order:
                        html_src.th(
                            _t=self.metrics_meta.get_metric(m).display_name,
                            title=self.metrics_meta.get_metric_description(m),
                        )

                # data dictionary to table: build table row by row
                for llm_model_name in self.leaderboard_order:
                    try:
                        with html_src.tr():
                            with html_src.td():
                                with html_src.a(
                                    href=f"#{anchor_uuid_map[llm_model_name]}"
                                ):
                                    html_src(f"{llm_model_name}")
                                    prefix = self.rag_type_prefix.get(
                                        llm_model_name, "LLM"
                                    )
                                    html_src.sup(_t=f"{prefix}")

                            for m in table_metrics_order:
                                m_avg = agg_data[llm_model_name][m]
                                color = self._get_col_for_value(
                                    min_val=metric_eda[m][0],
                                    max_val=metric_eda[m][1],
                                    val=m_avg,
                                    reverse=not self.metrics_meta.is_higher_better(
                                        sort_by_metric_id
                                    ),
                                )
                                with html_src.td(style=f"background-color: #{color};"):
                                    if math.isnan(m_avg) or math.isinf(m_avg):
                                        html_src("N/A (evaluator failed)")
                                    else:
                                        # .3 causes round bias: html_src(f"{m_avg:.3f}")
                                        v = self.truncate(m_avg, 5)
                                        html_src(f"{v}")
                    except Exception as ex:
                        self.logger.error(
                            f"Unable to render HTML representation for model "
                            f"{llm_model_name}: {ex}\n{traceback.format_exc()}"
                        )

            # PROMPTS table sorted by given metric
            if include_prompts_by_metrics:
                if self.i_failures:
                    html_src.br()

                    self._as_html_table_prompts_by_metric(
                        html_src=html_src,
                        anchor_uuid_map=anchor_uuid_map,
                        metric_id=sort_by_metric_id,
                        metric_name=sort_metrics_name,
                    )

            html_src.br()

    def _as_html_table_prompts_by_metric(
        self, html_src, anchor_uuid_map: dict, metric_id, metric_name
    ):
        # DATA preparation
        # data: failing prompts to the tuple for sorting
        leaderboard_data = []
        # metric_min = 1.0
        # metric_max = 0.0
        for prompt in self.i_failures:
            for _, result_row, failure_metric_id, _ in self.i_failures[prompt]:
                if (
                    metric_id == failure_metric_id
                    and result_row.metrics.get(metric_id, None) is not None
                ):
                    metric_val = result_row.metrics.get(metric_id, 0.0)
                    # metric_min = min(metric_min, metric_val)
                    # metric_max = max(metric_max, metric_val)
                    leaderboard_data.append(
                        (
                            prompt,
                            result_row.dataset_row.model_key,
                            metric_val,
                        )
                    )
        # sort the data by the metric value
        nan_deputy = (
            sys.float_info.min
            if self.metrics_meta.is_higher_better(metric_id)
            else sys.float_info.max
        )
        leaderboard_data.sort(
            key=lambda x: x[2] if not math.isnan(x[2]) else nan_deputy,
            reverse=not self.metrics_meta.is_higher_better(metric_id),
        )

        if leaderboard_data:
            with html_src.h6():
                html_src.b(_t=f"Model weak points for {metric_name} metric")
            html_src(
                "Prompts for which models achieved the lowest evaluation metric values"
                " - models weak points. Prompts are ordered by the evaluation metric "
                "value from the lowest to the highest along with the model which "
                "achieved the score:"
            )

            # PROMPT table sorted by given metric
            with html_src.table(klass="w3-table-all"):
                with html_src.tr():
                    html_src.th(_t=f"Prompts&nbsp;by&nbsp;{metric_name}&nbsp;metric")
                    html_src.th(_t="LLM&nbsp;model")
                    html_src.th(_t="Metric&nbsp;value")

                for e in leaderboard_data:
                    with html_src.tr():
                        llm_model_name = (
                            self.key_2_evaluated_model[e[1]].llm_model_name
                            if self.key_2_evaluated_model.get(e[1])
                            else "LLM model"
                        )
                        bg_color = self._get_col_for_value(
                            min_val=0.0,
                            max_val=1.0,
                            val=e[2],
                            reverse=not self.metrics_meta.is_higher_better(metric_id),
                        )

                        with html_src.td():
                            html_src(e[0])
                        with html_src.td():
                            with html_src.a(href=f"#{anchor_uuid_map[llm_model_name]}"):
                                html_src(f"{llm_model_name}")
                                prefix = self.rag_type_prefix.get(llm_model_name, "LLM")
                                html_src.sup(_t=f"{prefix}")
                        with html_src.td(style=f"background-color: #{bg_color};"):
                            html_src(f"{e[2]}")

    def as_dict(self, threshold: float | None = None) -> tuple[dict, dict]:
        """Return leaderboard as dictionary.

        Parameters
        ----------
        threshold :  float | None
            Threshold for the metrics.

        Returns
        -------
        Tuple[dict, dict] :
            Leaderboard data dictionary and metric EDA (min, max, ...) dictionary.

        """
        (metrics_values, metrics_eda) = self._eda()
        return {
            f5s.ExplanationFormat.KEY_DATA: metrics_values,
            f5s.ExplanationFormat.KEY_METADATA: self.metrics_meta.to_dict(threshold),
        }, metrics_eda

    @staticmethod
    def _get_context_title(ctx, length):
        try:
            index = length - ctx[:length][::-1].index(" ") - 1
            if index < length * 0.75:
                index = length
        except ValueError:
            index = length
        return f"{ctx[:index]}..."

    def _render_html_metrics_meta(self, failure, html_src):
        def _get_dp_colored_idx(am, dp_x, dp_y, am_i):
            alignment_data = am["data"]
            orig_steps = [d[0] for d in alignment_data]
            gen_steps = [d[1] for d in alignment_data]

            while orig_steps[am_i] == "-":
                am_i += 1
                dp_x += 1

            if gen_steps[am_i] == "-":
                return -1, dp_x, dp_y, am_i + 1

            return dp_x + 1, dp_x + 1, dp_y, am_i + 1

        if failure[1].metrics_meta:
            mm = failure[1].metrics_meta
            if mm.get(self.KEY_DYN_PROG_MATRIX):
                with html_src.li():
                    html_src.b(_t="Dynamic programming matrix:")
                with html_src.table(klass="w3-table-all"):
                    dp = mm[self.KEY_DYN_PROG_MATRIX]
                    html_src.th(_t="Original \\ Generated")
                    for col in dp["col_names"]:
                        html_src.th(_t=col)
                    colored_idx = -1
                    dp_x = 0
                    dp_y = 0
                    am_i = 0
                    for i, row in enumerate(dp["data"]):
                        try:
                            with html_src.tr():
                                with html_src.td():
                                    with html_src.b():
                                        html_src(dp["row_names"][i])
                                if mm.get(self.KEY_ALIGNMENT_MATRIX) and i > 0:
                                    am = mm[self.KEY_ALIGNMENT_MATRIX]
                                    try:
                                        (
                                            colored_idx,
                                            dp_x,
                                            dp_y,
                                            am_i,
                                        ) = _get_dp_colored_idx(
                                            am,
                                            dp_x,
                                            dp_y,
                                            am_i,
                                        )
                                    except Exception as ex:
                                        self.logger.error(
                                            f"Unable to render HTML "
                                            f"{ex}\n{traceback.format_exc()}"
                                        )
                                for ridx, row_val in enumerate(row):
                                    html_src.td(
                                        _t=str(round(row_val, 3)),
                                        style=(
                                            f"background-color: "
                                            f"#{self.PALETTE_GREEN[0]};"
                                            if colored_idx == ridx
                                            else ""
                                        ),
                                    )
                        except Exception as ex:
                            self.logger.error(
                                f"Unable to render HTML {ex}\n{traceback.format_exc()}"
                            )
            if mm.get(self.KEY_ALIGNMENT_MATRIX):
                with html_src.li():
                    html_src.b(_t="Alignment matrix:")
                with html_src.table(klass="w3-table-all"):
                    am = mm[self.KEY_ALIGNMENT_MATRIX]
                    for col in am["col_names"]:
                        html_src.th(_t=col)
                    for i, row in enumerate(am["data"]):
                        try:
                            with html_src.tr():
                                for row_val in row:
                                    html_src.td(_t=row_val)
                        except Exception as ex:
                            self.logger.error(
                                f"Unable to render HTML {ex}\n{traceback.format_exc()}"
                            )

    def as_html(
        self,
        sort_by_metric_id: str,
        html_src=None,
        include_failures: bool = True,
        include_prompts_by_metrics: bool = True,
        additional_details: dict | None = None,
    ) -> str:
        """Create HTML snippet with:

        - metrics procedure_eval table
        - failures section: model -> document -> prompt -> [metrics] -> value

        """
        html_src = html_src or airium.Airium()

        # map: model name -> uuid (used for anchors)
        anchor_uuid_map = {
            model_name: str(uuid.uuid4()) for model_name in self.leaderboard_order
        }

        # SECTION: models leaderboard for GIVEN metric (order by metric score)
        self._as_html_table_models_by_metric(
            html_src=html_src,
            anchor_uuid_map=anchor_uuid_map,
            sort_by_metric_id=sort_by_metric_id,
            include_prompts_by_metrics=include_prompts_by_metrics,
        )

        # SECTION: model & prompt failures (metrics > threshold):
        if include_failures and self.i_failures:
            with html_src.h6():
                html_src.b(_t="Most difficult prompts")

            html_src(
                "Prompts ordered by failures across all models and all metrics - "
                "leaderboard of the most difficult prompts. If the metric value for "
                "the actual answer created by a model for a given prompt is below "
                "the threshold specified for the metric, then it is considered "
                "a failure (Fail):"
            )

            self.sort_prompts_by_failures(self.i_failures_count)
            self._i_as_html_table(
                html_src=html_src, lead_col_name="Prompts by Failures"
            )

            html_src.br()

            with html_src.h6():
                html_src.b(_t="Model failures")

            for llm_model_name in self.m_failures:
                with html_src.span(id=f"{anchor_uuid_map[llm_model_name]}"):
                    html_src("Model ")
                    with html_src.b():
                        html_src(f"{llm_model_name}")
                    html_src("failures:")
                with html_src.ul():
                    for failure in self.m_failures[llm_model_name]:

                        def __html_src_model_failure():
                            with html_src.li():
                                html_src.b(_t="Prompt: ")
                                html_src(f"{failure[1].dataset_row.i}")
                            v = failure[1].dataset_row.expected_output
                            if v:
                                with html_src.li():
                                    html_src.b(_t="Expected output: ")
                                    html_src(f"{v}")
                            # add actual output with meta
                            LlmLeaderboardExplanation._html_aa_meta(
                                html_src=html_src,
                                actual_output=failure[1].dataset_row.actual_output,
                                actual_output_meta=failure[1].actual_output_meta,
                                err_msg=failure[0],
                                metrics_meta=self.metrics_meta,
                                metrics=failure[1].metrics,
                            )
                            if failure[1].dataset_row.context:
                                with html_src.li():
                                    html_src.b(_t="Context:")
                                for ctx in failure[1].dataset_row.context:
                                    with html_src.details():
                                        html_src.summary(
                                            _t=self._get_context_title(ctx, 64)
                                        )
                                        html_src.p(_t=ctx)
                            try:
                                self._render_html_metrics_meta(failure, html_src)
                            except Exception as ex:
                                self.logger.error(
                                    f"Unable to render HTML "
                                    f"{ex}\n{traceback.format_exc()}"
                                )

                        if self.llm_host != commons.LlmModelHostType.RAG:
                            with html_src.li():
                                html_src.b(
                                    _t=f"Error: {failure[0]}", klass="w3-text-red"
                                )
                                with html_src.ul():
                                    __html_src_model_failure()
                        else:
                            with html_src.li():
                                with html_src.b():
                                    corpus = failure[2]
                                    if (
                                        corpus
                                        and isinstance(corpus, str)
                                        and corpus.startswith("http")
                                    ):
                                        with html_src.a(href=corpus):
                                            html_src(f"{corpus}")
                                    else:
                                        html_src(f"{failure[2]}")
                                with html_src.ul():
                                    with html_src.li():
                                        html_src.b(
                                            _t=f"Error: {failure[0]}",
                                            klass="w3-text-red",
                                        )
                                    __html_src_model_failure()

        # SECTION: additional details
        if additional_details:
            html_src.br()
            html_src("Additional details:")
            with html_src.ul():
                for k in additional_details:
                    with html_src.li():
                        html_src.b(_t=f"{k}:")
                        html_src.br()
                        ad = additional_details[k]
                        if isinstance(
                            ad, LlmBoolLeaderboardExplanation.AdditionalDetails
                        ):
                            if ad.formatting == "pre":
                                with html_src.pre():
                                    html_src(f"{ad.text}")
                            elif ad.formatting == "i":
                                with html_src.i():
                                    html_src(f"{ad.text}")
                            else:
                                html_src(f"{ad.text}")
                        else:
                            html_src(f"{ad}")

        # if LLM dataset is empty, or evaluation of all rows from the LLM dataset fails,
        # then return "No results."
        return str(html_src) or (
            "<span>"
            "No explanations - either the dataset is empty or the evaluation of "
            "all rows from the dataset failed. In this case, please check the "
            "problems section for more details."
            "</span>"
        )

    def sort_prompts_by_failures(
        self, sort_by: dict[str, int | float], reverse: bool = True
    ):
        # build sort data structure - value can be parametrized
        unsorted_entries = [
            (m, sort_by.get(m, 0)) for m in self.inputs_leaderboard_order
        ]
        # sort
        sorted_entries = sorted(unsorted_entries, key=lambda x: x[1], reverse=reverse)

        self.inputs_leaderboard_order = [e[0] for e in sorted_entries]

    def _i_as_html_table(self, html_src, lead_col_name="Inputs"):
        with html_src.table(klass="w3-table-all"):
            with html_src.tr():
                html_src.th(_t=lead_col_name)
                html_src.th(_t="Pass")
                html_src.th(_t="Fail")
                html_src.th(_t="Failure rate")

            for i in self.inputs_leaderboard_order:
                passed = self.i_passes_count.get(i, 0)
                failures = self.i_failures_count.get(i, 0)
                accuracy = 100.0 - (passed / (passed + failures)) * 100.0

                if failures > 0:
                    try:
                        with html_src.tr():
                            html_src.td(_t=i)
                            with html_src.td():
                                html_src(passed)
                            with html_src.td():
                                html_src(failures)
                            color = self._get_col_for_pct(accuracy, reverse=True)
                            with html_src.td(style=f"background-color: #{color};"):
                                html_src(f"{accuracy:.3f}%")
                    except Exception as ex:
                        self.logger.error(
                            f"Unable to render HTML representation for input "
                            f"{i}: {ex}\n{traceback.format_exc()}"
                        )

    def add_markdown_format(
        self, sort_by_metric_id: str, title: str = "Evaluation Report"
    ):
        report_path = self.explainer.persistence.get_evaluator_working_file("report.md")
        with open(report_path, mode="w") as file:
            file.write(
                self.as_markdown(
                    sort_by_metric_id=sort_by_metric_id,
                    title=title,
                )
            )

        self.add_format(
            f5s.MarkdownFormat(
                explanation=self,
                format_file=report_path,
            )
        )

    def add_evalstudio_markdown_format(
        self, sort_by_metric_id: str, title: str = "Summary"
    ):
        report_path = self.explainer.persistence.get_evaluator_working_file("report.md")
        with open(report_path, mode="w") as file:
            file.write(
                self.as_markdown(
                    sort_by_metric_id=sort_by_metric_id,
                    title=title,
                    heading_level="##",
                )
            )

        self.add_format(
            f5s.EvalStudioMarkdownFormat(
                explanation=self,
                format_file=report_path,
            )
        )

    def as_markdown(
        self,
        sort_by_metric_id: str,
        title: str = "Evaluation Report",
        heading_level: str = "#",
        top: int = 3,
    ) -> str:
        """Return Markdown representation of the leaderboard for EvalStudio.

        Parameters
        ----------
        sort_by_metric_id : str
            Metric ID to be used as the FIRST one to sort the table. Then the method
            renders tables for all other metrics (sorted by that particular metric).
        title : str
            Title of the leaderboard.
        heading_level : str
            Markdown title heading level.
        top : int
            Number of top model failures, prompt failures, empty context prompts, ...
            entries. `0` for all entries.
            The motivation is to avoid LONG reports with all failures and prompts,
            it's just a summary.

        """

        md = ""
        md += f"{heading_level} {title}\n"
        md += "\n"
        md = LlmBoolLeaderboardExplanation.summary_as_markdown(
            md=md,
            metrics_count=self.metrics_meta.size(),
            llm_host=self.llm_host,
            m_failures_count=self.m_failures_count,
            i_failures_count=self.i_failures_count,
            key_2_evaluated_model=self.key_2_evaluated_model,
        )
        md += "\n"

        # models by metrics
        (agg_data, metric_eda) = self._eda()
        if sort_by_metric_id in self.metric_ids_order:
            self.metric_ids_order.remove(sort_by_metric_id)
            self.metric_ids_order.insert(0, sort_by_metric_id)
        for sort_by_metric_id in self.metric_ids_order:
            self._sort_agg_data(
                sort_by_metric_id,
                agg_data,
                reverse=self.metrics_meta.is_higher_better(sort_by_metric_id),
            )

            md += "\n"
            md += (
                f"## Models by "
                f"{self.metrics_meta.get_metric(sort_by_metric_id).display_name}"
                f"\n"
            )
            md += "Models ordered by the evaluation metric value.\n"
            md += "\n"
            md += "| Rank | LLM "
            for m in self.metric_ids_order:
                md += f"| {self.metrics_meta.get_metric(m).display_name}"
            md += " |\n| --- | --- "
            for _ in self.metric_ids_order:
                md += "| ---"
            md += " |\n"

            for e, llm_model_name in enumerate(self.leaderboard_order):
                md += f"| {e + 1} | {llm_model_name} "
                for m in self.metric_ids_order:
                    m_avg = agg_data[llm_model_name][m]
                    md += f"| {m_avg:.4f} "
                md += "|\n"

        # prompts failures
        if self.i_failures:
            md += "\n"
            md += "## Most difficult prompts\n"
            if top:
                md += "Top prompts "
            else:
                md += "Prompts "
            md += (
                "ordered by model failures - leaderboard of the most "
                "difficult prompts. If the metric value for the actual answer created "
                "by a model for a given prompt is below the threshold specified for "
                "the metric, then it is considered a failure."
            )
            if top:
                md += " See the `Report` for the full list."
            md += "\n\n"
            md += "| Prompt | Pass | Fail | Failures rate |\n"
            md += "| --- | --- | --- | --- |\n"

            self.sort_prompts_by_failures(self.i_failures_count)
            for e, i in enumerate(self.inputs_leaderboard_order):
                if top and e >= top:
                    break

                i_safe = sanitization.sanitize_markdown(md_fragment=i)
                passed = self.i_passes_count.get(i, 0)
                failures = self.i_failures_count.get(i, 0)
                accuracy = 100.0 - (passed / (passed + failures)) * 100.0

                if failures > 0:
                    try:
                        md += (
                            f"| {i_safe} | {passed} | {failures} | {accuracy:.1f}% |\n"
                        )
                    except Exception as ex:
                        self.logger.error(
                            f"Unable to render Markdown representation for input "
                            f"{i}: {ex}\n{traceback.format_exc()}"
                        )

        return md

    def add_json_format(
        self, threshold: float | None = None
    ) -> f5s.LlmHeatmapLeaderboardJSonFormat:
        """Add JSon format."""
        key_all_metrics = f5s.LlmHeatmapLeaderboardJSonFormat.KEY_ALL_METRICS

        (leaderboard_dict, eda_dict) = self.as_dict(threshold)

        # map: metric_id -> {min, max}
        result_eda = {}
        # map: metric_id -> {model_name -> {metric id -> value}}
        result_leaderboards = {}
        for metric_id in eda_dict:
            # EDA
            result_eda[metric_id] = {}
            result_eda[metric_id]["min"] = eda_dict[metric_id][0]
            result_eda[metric_id]["max"] = eda_dict[metric_id][1]

            # per-metric leaderboard
            result_leaderboards[metric_id] = {}
            for model_name in leaderboard_dict[f5s.ExplanationFormat.KEY_DATA]:
                result_leaderboards[metric_id][model_name] = {}
                result_leaderboards[metric_id][model_name][metric_id] = (
                    leaderboard_dict[f5s.ExplanationFormat.KEY_DATA][model_name][
                        metric_id
                    ]
                )

        metrics_for_json = list(eda_dict.keys()) + [key_all_metrics]
        (idx, idx_str) = f5s.LlmHeatmapLeaderboardJSonFormat.serialize_index_file(
            metrics=metrics_for_json,
        )
        json_format = f5s.LlmHeatmapLeaderboardJSonFormat(
            explanation=self,
            json_data=idx_str,
            persistence=self.explainer.persistence.store,
        )
        # save data files: leaderboards
        for metric_id in eda_dict:
            json_format.add_data(
                format_data=json.dumps(
                    result_leaderboards.get(metric_id, {}),
                    indent=4,
                    cls=persistences.NanEncoder,
                ),
                file_name=idx[f5s.ExplanationFormat.KEY_FILES][metric_id],
            )
        json_format.add_data(
            format_data=json.dumps(
                leaderboard_dict, indent=4, cls=persistences.NanEncoder
            ),
            file_name=idx[f5s.ExplanationFormat.KEY_FILES][key_all_metrics],
        )

        self.add_format(json_format)
        return json_format

    def get_insights(
        self,
        metrics_meta: commons.MetricsMeta,
        metric_id: str = "",
        metric_name_protection: bool = False,
        extra_description_best: str = "",
        extra_description_worst: str = "",
        insight_type: str = "accuracy",
        model_purpose: str = "",
        explanation_type: str = "",
        explanation_name: str = "",
        explanation_mime: str = "",
    ) -> None:
        """Create insights for the procedure_eval leaderboard.

        Parameters
        ----------
        metrics_meta : commons.MetricsMeta
            Metrics metadata.
        metric_id : str
            Optional metric ID to create insights for. If not specified, then insights
            are created for the primary metrics as specified by the metrics metadata.
        metric_name_protection : bool
            If True, then the metric ID is not changed to lowercase.
        extra_description_best: str
            Additional description for insights related to the best models.
        extra_description_worst: str
            Additional description for insights related to the worst models.
        insight_type : str
            Insight type.
        model_purpose : str
            Model purpose.
        explanation_type : str
            Type of the explanation which can clarify the insight.
        explanation_name : str
            Name of the explanation which can clarify the insight.
        explanation_mime : str
            Media type of the explanation which can clarify the insight.

        """
        t_insights = insights.InsightAndAction

        evaluator_name = self.explainer._display_name

        metric_id = metric_id or metrics_meta.get_primary_metric().key
        metric_name = (
            metrics_meta.get_metric(metric_id).display_name
            if metric_name_protection
            else metrics_meta.get_metric(metric_id).display_name.lower()
        )
        if self.insights_metric_2_winner and self.insights_metric_2_winner.get(
            metric_id
        ):
            model_name = self.insights_metric_2_winner.get(metric_id)

            html = airium.Airium()
            html("The ")
            with html.code():
                html(model_name)
            with html.b(klass="w3-black"):
                html("&nbsp;model&nbsp;")
            html("&nbsp; evaluated as the model with ")
            with html.span(klass="w3-black"):
                html("&nbsp;the best&nbsp;")
                with html.b(klass="w3-black"):
                    html(f"{metric_name}&nbsp;")
            html("&nbsp; metric score calculated by")
            with html.code():
                html(evaluator_name)
            html(f" evaluator. {extra_description_best}")

            self.explainer.add_insight(
                insights.InsightAndAction(
                    description=(
                        f"The {model_name} model evaluated as the model with the best "
                        f"{metric_name} metric score calculated by {evaluator_name} "
                        f"evaluator. {extra_description_best}"
                    ),
                    description_html=html,
                    insight_type=insight_type,
                    insight_attrs={
                        insights.InsightAndAction.ATTR_MODEL_NAME: model_name,
                        insights.InsightAndAction.ATTR_EVALUATOR_NAME: evaluator_name,
                    },
                    actions_description=(
                        "Refer to the explanation for the detailed description of "
                        "questions, answers, and failures."
                    ),
                    evaluator_id=self.explainer.evaluator_id(),
                    evaluator_name=evaluator_name,
                    explanation_type=explanation_type,
                    explanation_name=explanation_name,
                    explanation_mime=explanation_mime,
                    resources=[],
                )
            )

        if self.insights_difficult_prompt:
            prompt = self.insights_difficult_prompt

            self.explainer.add_insight(
                insights.InsightAndAction(
                    description=(
                        f"The '{prompt}' prompt was evaluated as the most difficult "
                        f"prompt to be correctly answered according to "
                        f"{evaluator_name} evaluator. {extra_description_worst}"
                    ),
                    description_html=t_insights.html_most_difficult_prompt_by(
                        prompt=prompt,
                        evaluator_name=evaluator_name,
                        extra_description=extra_description_worst,
                        model_purpose=model_purpose,
                    ),
                    insight_type=insight_type,
                    actions_description=(
                        "Refer to the explanation for the detailed description of "
                        "questions and answers by evaluated models in order to "
                        "identify weaknesses and strengths."
                    ),
                    evaluator_id=self.explainer.evaluator_id(),
                    evaluator_name=evaluator_name,
                    explanation_type=explanation_type,
                    explanation_name=explanation_name,
                    explanation_mime=explanation_mime,
                    resources=[],
                )
            )

    LLM_MODEL_ANONYMOUS = "model"

    @staticmethod
    def from_eval_results(
        evaluator,
        eval_results,
        metrics_meta: commons.MetricsMeta,
        key_2_evaluated_model: dict,
        llm_host: commons.LlmModelHostType = commons.LlmModelHostType.RAG,
        display_name: str = None,
        display_category: str = None,
        logger=None,
    ) -> "LlmProcedureEvalLeaderboardExplanation":
        """Create ProcedureEval leaderboard explanation from the evaluation results.

        Parameters
        ----------
        evaluator :
            Evaluator instance.
        eval_results : datasets.LlmEvalResults
            Evaluation results.
        metrics_meta : commons.MetricsMeta
            Metrics metadata.
        key_2_evaluated_model : dict
            Map: key -> LLM@RAG/LLM model.
        llm_host : commons.LlmModelHostType
            LLM host type - either a RAG (with retrieval) or a LLM (generation only).
        display_name : str
            Custom leaderboard display name.
        display_category : str
            Custom leaderboard display category.
        logger :
            Optional logger.

        """
        if (
            not eval_results
            or eval_results.results is None
            or len(eval_results.results) == 0
        ):
            raise ValueError(
                f"No evaluation results created by the evaluator "
                f"'{evaluator.display_name}' - unable to create the leaderboard."
            )

        procedure_eval_explanation = LlmProcedureEvalLeaderboardExplanation(
            evaluator=evaluator,
            eval_results=eval_results,
            metrics_meta=metrics_meta,
            key_2_evaluated_model=key_2_evaluated_model,
            llm_host=llm_host,
            display_name=display_name,
            display_category=display_category,
            logger=logger or loggers.SonarPrintLogger(),
        )

        # populate leaderboard data
        for r in eval_results.results:
            evaluated_model = procedure_eval_explanation.key_2_evaluated_model.get(
                r.dataset_row.model_key
            )

            for metric_id in r.metrics:
                procedure_eval_explanation.add_col_value(
                    llm_model_name=(
                        evaluated_model.llm_model_name
                        if evaluated_model
                        else LlmProcedureEvalLeaderboardExplanation.LLM_MODEL_ANONYMOUS
                    ),
                    docs=(
                        evaluated_model.documents[0]
                        if isinstance(evaluated_model, models.ExplainableRagModel)
                        and evaluated_model.documents
                        else ""
                    ),
                    prompt=r.dataset_row.i,
                    metrics_id=metric_id,
                    value=r.metrics.get(metric_id, 0.0),
                    result_row=r,
                )

        procedure_eval_explanation.build()

        return procedure_eval_explanation


class LlmClassifierLeaderboardExplanation(
    _explanations_base.Explanation, LlmLeaderboardExplanation, AbcHeatmapExplanation
):
    """LLM classification leaderboard explanation."""

    _explanation_type = "llm-classification-leaderboard"
    _is_global = True

    DEFAULT_METRIC_THRESHOLD = 0.75

    METRIC_ACCURACY = "accuracy"
    METRIC_PRECISION = "precision"
    METRIC_RECALL = "recall"
    METRIC_F1 = "f1"

    METRIC_META_ACCURACY = commons.MetricMeta(
        key=METRIC_ACCURACY,
        display_name="Accuracy",
        description=(
            "Accuracy metric measures how often model makes correct "
            "predictions using the formula: "
            "(True Positives + True Negatives) / Total Predictions."
        ),
        higher_is_better=True,
        threshold=DEFAULT_METRIC_THRESHOLD,
        is_primary_metric=True,
    )
    METRIC_META_PRECISION = commons.MetricMeta(
        key=METRIC_PRECISION,
        display_name="Precision",
        description=(
            "Precision metric measures proportion of the positive predictions "
            "that were actually correct using the formula: "
            "True Positives / (True Positives + False Positives)."
        ),
        higher_is_better=True,
        threshold=DEFAULT_METRIC_THRESHOLD,
        is_primary_metric=False,
    )
    METRIC_META_RECALL = commons.MetricMeta(
        key=METRIC_RECALL,
        display_name="Recall",
        description=(
            "Recall metric measures proportion of the actual positive cases "
            "that were correctly predicted using the formula: "
            "True Positives / (True Positives + False Negatives)."
        ),
        higher_is_better=True,
        threshold=DEFAULT_METRIC_THRESHOLD,
        is_primary_metric=False,
    )
    METRIC_META_F1 = commons.MetricMeta(
        key=METRIC_F1,
        display_name="F1",
        description=(
            "F1 metrics measures the balance between precision and recall "
            "using the formula: "
            "2 * (Precision * Recall) / (Precision + Recall)."
        ),
        higher_is_better=True,
        threshold=DEFAULT_METRIC_THRESHOLD,
        is_primary_metric=False,
    )
    # TP, TN, FP, FN > confusion matrices
    # commons.MetricMeta(
    #     key=METRIC_TP,
    #     display_name="TP",
    #     description=(
    #         "True positives: the number of positive instances correctly "
    #         "classified as positive."
    #     ),
    #     higher_is_better=True,
    #     threshold=0,
    #     is_primary_metric=False,
    # ),
    # commons.MetricMeta(
    #     key=METRIC_TN,
    #     display_name="TN",
    #     description=(
    #         "True negatives: the number of negative instances correctly "
    #         "classified as negative."
    #     ),
    #     higher_is_better=True,
    #     threshold=0,
    #     is_primary_metric=False,
    # ),
    # commons.MetricMeta(
    #     key=METRIC_FP,
    #     display_name="FP",
    #     description=(
    #         "False positives: the number of negative instances incorrectly "
    #         "classified as positive."
    #     ),
    #     higher_is_better=False,
    #     threshold=0,
    #     is_primary_metric=False,
    # ),
    # commons.MetricMeta(
    #     key=METRIC_FN,
    #     display_name="FN",
    #     description=(
    #         "False negatives: the number of positive instances incorrectly "
    #         "classified as negative."
    #     ),
    #     higher_is_better=False,
    #     threshold=0,
    #     is_primary_metric=False,
    # ),

    def __init__(
        self,
        evaluator,
        eval_results,
        model_2_metrics: dict,
        model_2_confusion_matrix: dict,
        classes: list[str],
        false_positives: dict[str, list[datasets.LlmEvalResults.LlmEvalResultRow]],
        false_negatives: dict[str, list[datasets.LlmEvalResults.LlmEvalResultRow]],
        i_passes_count: dict[str, int],
        metrics_meta: commons.MetricsMeta,
        key_2_evaluated_model: dict,
        llm_host: commons.LlmModelHostType = commons.LlmModelHostType.RAG,
        display_name: str = "",
        display_category: str = "",
        logger=None,
    ) -> None:
        display_name = display_name or "LLM Classification Leaderboard"

        _explanations_base.Explanation.__init__(
            self,
            explainer=evaluator,
            display_name=display_name,
            display_category=display_category,
        )

        self.key_2_evaluated_model = key_2_evaluated_model
        if key_2_evaluated_model:
            self.rag_type_prefix = LlmBoolLeaderboardExplanation.key_2_rag_type_prefix(
                key_2_evaluated_model.values()
            )
        else:
            self.rag_type_prefix = {}
        self.leaderboard_order = [
            m.llm_model_name for m in self.key_2_evaluated_model.values()
        ]
        self.llm_host = llm_host
        self.model_2_metrics = model_2_metrics
        self.model_2_confusion_matrix = model_2_confusion_matrix
        self.classes = classes
        self.false_positives = false_positives
        self.false_negatives = false_negatives
        self.metrics_meta = metrics_meta
        self.metric_ids_order = self.metrics_meta.get_metric_keys()
        self.logger = logger or loggers.SonarPrintLogger()

        # map: prompt -> count
        self.i_failures_count = {}
        self.i_passes_count = i_passes_count
        if self.false_positives or self.false_negatives:
            for m in self.false_positives:
                for r in self.false_positives[m]:
                    if r.i not in self.i_failures_count:
                        self.i_failures_count[r.i] = 0
                    self.i_failures_count[r.i] += 1
            for m in self.false_negatives:
                for r in self.false_negatives[m]:
                    if r.i not in self.i_failures_count:
                        self.i_failures_count[r.i] = 0
                    self.i_failures_count[r.i] += 1
        # inputs / prompts
        self.inputs_leaderboard_order = [i for i in self.i_failures_count.keys()]
        self.sort_prompts_by_failures(self.i_failures_count)

        # insights
        self.insight_most_acc = None
        self.insight_least_acc = None
        self.insights_difficult_prompt = (
            self.inputs_leaderboard_order[0] if self.inputs_leaderboard_order else None
        )

        # method reuse from the heatmap explanation
        self.heat_lead = LlmHeatmapLeaderboardExplanation(
            evaluator=evaluator,
            eval_results=eval_results,
            metrics_meta=metrics_meta,
            key_2_evaluated_model=key_2_evaluated_model,
            llm_host=llm_host,
            display_name=display_name,
            display_category=display_category,
            logger=logger,
        )

    def validate(self) -> bool:
        pass

    @staticmethod
    def from_eval_results(
        evaluator,
        eval_results,
        model_2_metrics: dict,
        model_2_confusion_matrix: dict,
        classes: list[str],
        metrics_meta: commons.MetricsMeta,
        key_2_evaluated_model: dict,
        llm_host: commons.LlmModelHostType = commons.LlmModelHostType.RAG,
        display_name: str = None,
        display_category: str = None,
        logger=None,
    ) -> "LlmClassifierLeaderboardExplanation":
        """Create Classification leaderboard explanation from the evaluation results.

        Parameters
        ----------
        evaluator :
            Evaluator instance.
        model_2_metrics : dict
            Map: model name -> metric ID -> metric value.
        model_2_confusion_matrix : dict
            Map: model name -> confusion matrix.
        classes : list[str]
            List of classes.
        eval_results : datasets.LlmEvalResults
            Evaluation results.
        metrics_meta : commons.MetricsMeta
            Metrics metadata.
        key_2_evaluated_model : dict
            Map: key -> LLM@RAG/LLM model.
        llm_host : commons.LlmModelHostType
            LLM host type - either a RAG (with retrieval) or a LLM (generation only).
        display_name : str
            Custom leaderboard display name.
        display_category : str
            Custom leaderboard display category.
        logger :
            Optional logger.

        """
        if (
            not eval_results
            or eval_results.results is None
            or len(eval_results.results) == 0
        ):
            raise ValueError(
                f"No evaluation results created by the evaluator "
                f"'{evaluator.display_name}' - unable to create the leaderboard."
            )

        # map: model name -> [FP result dataset rows]
        false_positives = {}
        # map: model name -> [FN result dataset rows]
        false_negatives = {}
        # map: input -> count
        i_passes_count = {}
        positive_class = classes[1]

        # gather FPs and FNs for each model to report them as failures
        if len(classes) == 2:
            for r in eval_results.results:
                row = r.dataset_row
                if row.actual_output != row.expected_output:
                    model_name = (
                        key_2_evaluated_model[row.model_key].llm_model_name
                        if key_2_evaluated_model.get(row.model_key, None)
                        else row.model_key
                    )

                    if row.expected_output == positive_class:
                        if model_name not in false_negatives:
                            false_negatives[model_name] = []
                        false_negatives[model_name].append(row)
                    else:
                        if model_name not in false_positives:
                            false_positives[model_name] = []
                        false_positives[model_name].append(row)
                else:
                    if row.i not in i_passes_count:
                        i_passes_count[row.i] = 0
                    i_passes_count[row.i] += 1
        else:
            false_positives = {}
            false_negatives = {}
            i_passes_count = {}

        cls_explanation = LlmClassifierLeaderboardExplanation(
            evaluator=evaluator,
            eval_results=eval_results,
            model_2_metrics=model_2_metrics,
            model_2_confusion_matrix=model_2_confusion_matrix,
            classes=classes,
            false_positives=false_positives,
            false_negatives=false_negatives,
            i_passes_count=i_passes_count,
            metrics_meta=metrics_meta,
            key_2_evaluated_model=key_2_evaluated_model,
            llm_host=llm_host,
            display_name=display_name,
            display_category=display_category,
            logger=logger or loggers.SonarPrintLogger(),
        )

        cls_explanation.build()

        return cls_explanation

    def _eda(self) -> tuple[dict, dict]:
        # Metrics EDA.
        #
        # RESULT: heatmap leaderboard dicts definition:
        # - metrics values:
        #     model -> metric ID -> (avg) value
        # - metrics EDA:
        #     metric ID -> [min, max]
        #
        metric_vals = self.model_2_metrics

        metric_ids = self.metrics_meta.get_metric_keys()
        metrics_eda = {m: [100.0, 0.0] for m in metric_ids}
        for model_id in self.model_2_metrics:
            for metric_id in self.model_2_metrics[model_id]:
                m_val = self.model_2_metrics[model_id][metric_id]
                m_min = min(metrics_eda[metric_id][0], m_val)
                m_max = max(metrics_eda[metric_id][1], m_val)

                metrics_eda[metric_id] = [m_min, m_max]

        return metric_vals, metrics_eda

    def as_dict(self, threshold: float | None = None) -> tuple[dict, dict]:
        """Return leaderboard as dictionary.

        Parameters
        ----------
        threshold :  float | None
            Threshold for metrics - if not provided, the default metric threshold
            is used.

        Returns
        -------
        Tuple[dict, dict] :
            Leaderboard data dictionary and metric EDA (min, max, ...) dictionary.

        """
        (metrics_values, metrics_eda) = self._eda()
        return {
            f5s.ExplanationFormat.KEY_DATA: metrics_values,
            f5s.ExplanationFormat.KEY_METADATA: self.metrics_meta.to_dict(threshold),
        }, metrics_eda

    def _sort_metrics_vals(
        self, metric_id: str, model_2_metrics: dict, reverse: bool = True
    ):
        """Sort data dictionary by the given metric to generate table which is
        sorted by given column.

        """
        # metric-based dir used to sort models
        sort_by = [
            (m, model_2_metrics[m][metric_id])
            for m in model_2_metrics
            if metric_id in model_2_metrics[m]
        ]

        # NaN comparison is not define for floats -> define NaN as min/max
        nan_deputy = (
            sys.float_info.min
            if self.metrics_meta.is_higher_better(metric_id)
            else sys.float_info.max
        )

        # build sort data structure - value can be parametrized
        sorted_entries = sorted(
            sort_by,
            key=lambda x: x[1] if not math.isnan(x[1]) else nan_deputy,
            reverse=reverse,
        )

        self.leaderboard_order = [e[0] for e in sorted_entries]

    def build(self):
        """Build leaderboard."""
        metric_id = self.metrics_meta.get_primary_metric().key

        self._sort_metrics_vals(
            metric_id=metric_id,
            model_2_metrics=self.model_2_metrics,
            reverse=self.metrics_meta.is_higher_better(metric_id),
        )

        if len(self.leaderboard_order) > 1:
            self.insight_most_acc = self.leaderboard_order[0]
            self.insight_least_acc = self.leaderboard_order[-1]

    def add_json_format(
        self, threshold: float | None = None
    ) -> f5s.LlmHeatmapLeaderboardJSonFormat:
        """Add JSon format."""
        key_all_metrics = f5s.LlmHeatmapLeaderboardJSonFormat.KEY_ALL_METRICS

        (leaderboard_dict, eda_dict) = self.as_dict(threshold)

        # map: metric_id -> {model_name -> {metric id -> value}}
        result_leaderboards = {}
        for metric_id in eda_dict:
            # per-metric leaderboard
            result_leaderboards[metric_id] = {}
            for model_name in leaderboard_dict[f5s.ExplanationFormat.KEY_DATA]:
                result_leaderboards[metric_id][model_name] = {}
                result_leaderboards[metric_id][model_name][metric_id] = (
                    leaderboard_dict[f5s.ExplanationFormat.KEY_DATA][model_name][
                        metric_id
                    ]
                )

        metrics_for_json = list(eda_dict.keys()) + [key_all_metrics]
        (idx, idx_str) = f5s.LlmHeatmapLeaderboardJSonFormat.serialize_index_file(
            metrics=metrics_for_json,
        )
        json_format = f5s.LlmHeatmapLeaderboardJSonFormat(
            explanation=self,
            json_data=idx_str,
            persistence=self.explainer.persistence.store,
        )
        # save data files: leaderboards
        for metric_id in eda_dict:
            json_format.add_data(
                format_data=json.dumps(
                    result_leaderboards.get(metric_id, {}),
                    indent=4,
                    cls=persistences.NanEncoder,
                ),
                file_name=idx[f5s.ExplanationFormat.KEY_FILES][metric_id],
            )
        json_format.add_data(
            format_data=json.dumps(
                leaderboard_dict, indent=4, cls=persistences.NanEncoder
            ),
            file_name=idx[f5s.ExplanationFormat.KEY_FILES][key_all_metrics],
        )

        self.add_format(json_format)
        return json_format

    @staticmethod
    def _process_confusion_matrix(
        confusion_matrix: numpy.ndarray, classes: list
    ) -> numpy.ndarray:
        """Sums the rightmost columns of a confusion matrix if they exceed the
        number of expected classes.

        Parameters
        ----------
        confusion_matrix : numpy.ndarray
            The input confusion matrix.
        classes : list
            A list of expected class labels.

        Returns
        -------
        numpy.ndarray
            A new confusion matrix where unexpected prediction columns
            are summed into one, or the original matrix if no unexpected columns exist.

        """
        classes_len = len(classes)
        num_cols = confusion_matrix.shape[1]

        has_unexpected_predictions: bool = num_cols > classes_len

        if has_unexpected_predictions:
            expected_columns = confusion_matrix[:, :classes_len]
            unexpected_columns = confusion_matrix[:, classes_len:]

            summed_unexpected_column = unexpected_columns.sum(axis=1, keepdims=True)
            processed_matrix = numpy.concatenate(
                (expected_columns, summed_unexpected_column), axis=1
            )
        else:
            processed_matrix = confusion_matrix

        return processed_matrix

    @staticmethod
    def _cm_as_html_table(
        html_src,
        classes: list[str],
        confusion_matrix: numpy.ndarray,
        max_label_lng=20,
    ):
        """Render confusion matrix as HTML table.

        Parameters
        ----------
        html_src :
            HTML source.
        classes : list[str]
            List of classes.
        confusion_matrix : numpy.ndarray
            Confusion matrix.
        max_label_lng : int
            Maximum label length for the class name. If the label is longer,
            then it will be truncated using ellipsis.

        """
        with html_src.table(klass="w3-table-all"):
            if confusion_matrix is None or len(confusion_matrix) == 0:
                return
            classes_len = len(classes)
            has_unexpected_predictions: bool = len(confusion_matrix) > classes_len
            confusion_matrix = (
                LlmClassifierLeaderboardExplanation._process_confusion_matrix(
                    confusion_matrix, classes
                )
            )
            column_count: int = classes_len + has_unexpected_predictions
            with html_src.tr():
                html_src.th(_t="Label \\ Predicted")
                for i in range(column_count):
                    if i == column_count - 1 and has_unexpected_predictions:
                        html_src.th(_t="Unexpected output")
                    elif i >= classes_len:
                        break
                    else:
                        c = classes[i]
                        ellipsis_ = (
                            f"{c[:max_label_lng]}..." if len(c) > max_label_lng else c
                        )
                        html_src.th(_t=ellipsis_, title=c)

            for i in range(classes_len):
                with html_src.tr():
                    c = classes[i]
                    ellipsis_ = (
                        f"{c[:max_label_lng]}..." if len(c) > max_label_lng else c
                    )
                    html_src.th(_t=ellipsis_, title=c)
                    for j in range(column_count):
                        if i == j:
                            color = LlmClassifierLeaderboardExplanation.PALETTE_GREEN[0]
                        else:
                            color = LlmClassifierLeaderboardExplanation.PALETTE_RED[0]
                        html_src.td(
                            _t=f"{confusion_matrix[i][j]}",
                            style=f"background-color: #{color};",
                        )
        html_src.br()

    @staticmethod
    def _cm_as_markdown_table(classes, confusion_matrix: list[list[int]]) -> str:
        """Render confusion matrix as Markdown table.

        Parameters
        ----------
        classes : list[str]
            List of classes.
        confusion_matrix : list[list[int]]
            Confusion matrix.

        """
        md = ""
        md += "| Label \\ Predicted "
        if confusion_matrix is None or len(confusion_matrix) == 0:
            return "Confusion matrix was not generated."
        expected_class_count: int = sum([any(row) for row in confusion_matrix])
        has_unexpected_predictions: bool = len(confusion_matrix) > expected_class_count
        column_count: int = expected_class_count + has_unexpected_predictions
        for i in range(column_count):
            if i == column_count - 1 and has_unexpected_predictions:
                md += "| Unexpected output "
            elif i == expected_class_count:
                break
            else:
                c = classes[i]
                md += f"| {sanitization.sanitize_markdown(c)} "

        md += "|\n"
        md += "| --- "
        for _ in range(column_count):
            md += "| --- "
        md += "|\n"

        for i in range(expected_class_count):
            c = classes[i]
            md += f"| {sanitization.sanitize_markdown(c)} "
            for j in range(len(confusion_matrix[i])):
                md += f"| {confusion_matrix[i][j]} "
            md += "|\n"

        return md

    @staticmethod
    def _as_html_table_false_class(
        html_src,
        false_display_name: str,
        false_classifications: dict[
            str, list[datasets.LlmEvalResults.LlmEvalResultRow]
        ],
        model_2_confusion_matrix: dict[str, numpy.ndarray],
        classes: list[str],
        anchor_uuid_map,
        title="Model failures",
    ):
        with html_src.h6():
            html_src.b(_t=f"{title}")

        for llm_model_name in model_2_confusion_matrix:
            # CM
            with html_src.span():
                html_src("Model ")
                with html_src.b():
                    html_src(f"{llm_model_name}")
                html_src(" confusion matrix:")
            LlmClassifierLeaderboardExplanation._cm_as_html_table(
                html_src=html_src,
                classes=classes,
                confusion_matrix=model_2_confusion_matrix[llm_model_name],
            )

            if false_classifications:
                # FP/FN prompts
                with html_src.span(id=f"{anchor_uuid_map[llm_model_name]}"):
                    html_src("Model ")
                    with html_src.b():
                        html_src(f"{llm_model_name}")
                    html_src(
                        f" {false_display_name} "
                        f"({len(false_classifications.get(llm_model_name, []))}):"
                    )

                with html_src.ul():
                    for failure in false_classifications.get(llm_model_name, []):
                        with html_src.li():
                            html_src.b(_t="Prompt: ")
                            html_src(f"{failure.i}")
                            with html_src.ul():
                                if failure.output_condition:
                                    with html_src.li():
                                        html_src.b(_t="Output condition: ")
                                        with html_src.span(klass="w3-text-red"):
                                            html_src(f"{failure.output_condition}")
                                if failure.output_constraints:
                                    with html_src.li():
                                        html_src.b(_t="Output constraints: ")
                                        with html_src.span(klass="w3-text-red"):
                                            html_src(f"{failure.output_constraints}")
                                with html_src.li():
                                    html_src.b(_t="Expected output: ")
                                    html_src(f"{failure.expected_output}")
                                with html_src.li():
                                    html_src.b(_t="Actual output: ")
                                    with html_src.span(klass="w3-text-red"):
                                        html_src(f"{failure.actual_output}")

    def _as_html_table_models_by_metric(
        self,
        html_src,
        anchor_uuid_map: dict,
    ):
        """Render HTML table with models by (aggregated) metric value.

        Parameters
        ----------
        html_src :
            HTML source.
        anchor_uuid_map : dict
            Map (model name -> anchor UUID) used to make links from the table
            to the sections with the model failures (has UUID as HTML element ID).

        """
        (model_2_metrics, metric_eda) = self._eda()

        # if there is exactly 1 model, then no need to sort by each metric
        normalized_metric_ids_order = (
            [self.metric_ids_order[0]]
            if len(self.leaderboard_order) == 1
            else self.metric_ids_order
        )

        for sort_by_metric_id in normalized_metric_ids_order:
            # move sort_by_metric_id to the first item in self.metric_ids_order
            table_metrics_order = self.metric_ids_order.copy()
            if sort_by_metric_id in table_metrics_order:
                table_metrics_order.remove(sort_by_metric_id)
                table_metrics_order.insert(0, sort_by_metric_id)

            self._sort_metrics_vals(
                metric_id=sort_by_metric_id,
                model_2_metrics=model_2_metrics,
                reverse=self.metrics_meta.is_higher_better(sort_by_metric_id),
            )
            sort_metrics_name = self.metrics_meta.get_metric(
                sort_by_metric_id
            ).display_name

            # MODEL table sorted by given metric
            with html_src.table(klass="w3-table-all"):
                with html_src.tr():
                    html_src.th(_t=f"LLM Models by {sort_metrics_name}")
                    for m in table_metrics_order:
                        html_src.th(
                            _t=self.metrics_meta.get_metric(m).display_name,
                            title=self.metrics_meta.get_metric_description(m),
                        )

                # data dictionary to table: build table row by row
                for llm_model_name in self.leaderboard_order:
                    try:
                        with html_src.tr():
                            with html_src.td():
                                with html_src.a(
                                    href=f"#{anchor_uuid_map[llm_model_name]}"
                                ):
                                    html_src(f"{llm_model_name}")
                                    prefix = self.rag_type_prefix.get(
                                        llm_model_name, "LLM"
                                    )
                                    html_src.sup(_t=f"{prefix}")

                            for m in table_metrics_order:
                                m_avg = model_2_metrics[llm_model_name][m]
                                color = self.heat_lead._get_col_for_value(
                                    min_val=metric_eda[m][0],
                                    max_val=metric_eda[m][1],
                                    val=m_avg,
                                    reverse=not self.metrics_meta.is_higher_better(
                                        sort_by_metric_id
                                    ),
                                )
                                with html_src.td(style=f"background-color: #{color};"):
                                    if math.isnan(m_avg) or math.isinf(m_avg):
                                        html_src("N/A (evaluator failed)")
                                    else:
                                        # .3 causes round bias: html_src(f"{m_avg:.3f}")
                                        v = LlmHeatmapLeaderboardExplanation.truncate(
                                            m_avg, 5
                                        )
                                        html_src(f"{v}")
                    except Exception as ex:
                        self.logger.error(
                            f"Unable to render HTML representation for model "
                            f"{llm_model_name}: {ex}\n{traceback.format_exc()}"
                        )

            html_src.br()

    def sort_prompts_by_failures(
        self, sort_by: dict[str, int | float], reverse: bool = True
    ):
        # build sort data structure - value can be parametrized
        unsorted_entries = [
            (m, sort_by.get(m, 0)) for m in self.inputs_leaderboard_order
        ]
        # sort
        sorted_entries = sorted(unsorted_entries, key=lambda x: x[1], reverse=reverse)

        self.inputs_leaderboard_order = [e[0] for e in sorted_entries]

    def _i_as_html_table(self, html_src, lead_col_name="Inputs"):
        with html_src.table(klass="w3-table-all"):
            with html_src.tr():
                html_src.th(_t=lead_col_name)
                html_src.th(_t="Pass")
                html_src.th(_t="Fail")
                html_src.th(_t="Failure rate")

            for i in self.inputs_leaderboard_order:
                passed = self.i_passes_count.get(i, 0)
                failures = self.i_failures_count.get(i, 0)
                accuracy = 100.0 - (passed / (passed + failures)) * 100.0

                if failures > 0:
                    try:
                        with html_src.tr():
                            html_src.td(_t=i)
                            with html_src.td():
                                html_src(passed)
                            with html_src.td():
                                html_src(failures)
                            color = self.heat_lead._get_col_for_pct(
                                accuracy, reverse=True
                            )
                            with html_src.td(style=f"background-color: #{color};"):
                                html_src(f"{accuracy:.3f}%")
                    except Exception as ex:
                        self.logger.error(
                            f"Unable to render HTML representation for input "
                            f"{i}: {ex}\n{traceback.format_exc()}"
                        )

    def as_html(
        self,
        sort_by_metric_id: str,
        html_src=None,
        include_failures: bool = True,
        include_prompts_by_metrics: bool = True,
        additional_details: dict | None = None,
    ) -> str:
        """Create HTML snippet with:

        - per-metrics heatmap table
        - per-metrics confusion matrix

        """
        html_src = html_src or airium.Airium()

        # map: model name -> uuid (used for anchors)
        anchor_uuid_map = {
            model_name: str(uuid.uuid4()) for model_name in self.leaderboard_order
        }

        # 2nd class is positive class by convention
        if len(self.classes) == 2:
            positive_class = self.classes[1]
            negative_class = self.classes[0]
        else:
            positive_class = negative_class = ""

        html_src("Classes:")
        with html_src.ul():
            for e, c in enumerate(self.classes):
                with html_src.li():
                    with html_src.code():
                        html_src(f"'{c}'")
                    if c == positive_class:
                        html_src(" - positive class")
        html_src.br()

        # SECTION: models leaderboard for GIVEN metric (order by metric score)
        self._as_html_table_models_by_metric(
            html_src=html_src,
            anchor_uuid_map=anchor_uuid_map,
        )
        if include_prompts_by_metrics:
            if len(self.classes) == 2:
                # SECTION: FALSE POSITIVES
                html_src.br()
                LlmClassifierLeaderboardExplanation._as_html_table_false_class(
                    html_src=html_src,
                    title="False Negatives",
                    false_display_name=(
                        f"false positives - false '{positive_class}' prediction"
                    ),
                    false_classifications=self.false_positives,
                    model_2_confusion_matrix=self.model_2_confusion_matrix,
                    classes=self.classes,
                    anchor_uuid_map=anchor_uuid_map,
                )

                # SECTION: FALSE NEGATIVES
                html_src.br()

                LlmClassifierLeaderboardExplanation._as_html_table_false_class(
                    html_src=html_src,
                    title="False Positives",
                    false_display_name=(
                        f"false negatives - false '{negative_class}' prediction"
                    ),
                    false_classifications=self.false_negatives,
                    model_2_confusion_matrix=self.model_2_confusion_matrix,
                    classes=self.classes,
                    anchor_uuid_map=anchor_uuid_map,
                )
            else:
                LlmClassifierLeaderboardExplanation._as_html_table_false_class(
                    html_src=html_src,
                    false_display_name="",
                    false_classifications={},
                    model_2_confusion_matrix=self.model_2_confusion_matrix,
                    classes=self.classes,
                    anchor_uuid_map=anchor_uuid_map,
                )

        # SECTION: model & prompt failures (metrics > threshold):
        if include_failures and (self.false_negatives or self.false_positives):
            with html_src.h6():
                html_src.b(_t="Most difficult prompts")

            html_src(
                "Prompts ordered by failures - incorrect classification - across all "
                "models. Leaderboard of the most difficult prompts:"
            )

            self.sort_prompts_by_failures(self.i_failures_count)
            self._i_as_html_table(
                html_src=html_src, lead_col_name="Prompts by Failures"
            )

        # SECTION: additional details
        if additional_details:
            html_src.br()
            html_src("Additional details:")
            with html_src.ul():
                for k in additional_details:
                    with html_src.li():
                        html_src.b(_t=f"{k}:")
                        html_src.br()
                        ad = additional_details[k]
                        if isinstance(
                            ad, LlmBoolLeaderboardExplanation.AdditionalDetails
                        ):
                            if ad.formatting == "pre":
                                with html_src.pre():
                                    html_src(f"{ad.text}")
                            elif ad.formatting == "i":
                                with html_src.i():
                                    html_src(f"{ad.text}")
                            else:
                                html_src(f"{ad.text}")
                        else:
                            html_src(f"{ad}")

        return str(html_src)

    def add_markdown_format(
        self, sort_by_metric_id: str, title: str = "Evaluation Report"
    ):
        report_path = self.explainer.persistence.get_evaluator_working_file("report.md")
        with open(report_path, mode="w") as file:
            file.write(
                self.as_markdown(
                    sort_by_metric_id=sort_by_metric_id,
                    title=title,
                    include_metrics_leaderboards=True,
                )
            )

        self.add_format(
            f5s.MarkdownFormat(
                explanation=self,
                format_file=report_path,
            )
        )

    def add_evalstudio_markdown_format(
        self, sort_by_metric_id: str, title: str = "Summary"
    ):
        report_path = self.explainer.persistence.get_evaluator_working_file("report.md")
        with open(report_path, mode="w") as file:
            file.write(
                self.as_markdown(
                    sort_by_metric_id=sort_by_metric_id,
                    title=title,
                    heading_level="##",
                    include_metrics_leaderboards=False,
                )
            )

        self.add_format(
            f5s.EvalStudioMarkdownFormat(
                explanation=self,
                format_file=report_path,
            )
        )

    def as_markdown(
        self,
        sort_by_metric_id: str,
        title: str = "Evaluation Report",
        heading_level: str = "#",
        include_metrics_leaderboards: bool = True,
        top: int = 3,
    ) -> str:
        """Return Markdown representation of the leaderboard for EvalStudio.

        Parameters
        ----------
        sort_by_metric_id : str
            Metric ID to sort models by.
        title : str
            Title of the leaderboard.
        heading_level : str
            Heading level.
        include_metrics_leaderboards : bool
            Include per-metrics leaderboards.
        top : int
            Number of top model failures, prompt failures, empty context prompts, ...
            entries. `0` for all entries.
            The motivation is to avoid LONG reports with all failures and prompts,
            it's just a summary.

        Returns
        -------
        str
            Markdown representation of the leaderboard.

        """

        md = ""
        md += f"{heading_level} {title}\n"
        md += "\n"
        md = LlmBoolLeaderboardExplanation.summary_as_markdown(
            md=md,
            metrics_count=self.metrics_meta.size(),
            llm_host=self.llm_host,
            m_failures_count={},
            i_failures_count=self.i_failures_count,
            key_2_evaluated_model=self.key_2_evaluated_model,
        )
        md += "\n"

        # models by metrics
        if include_metrics_leaderboards:
            (agg_data, metric_eda) = self._eda()
            if sort_by_metric_id in self.metric_ids_order:
                self.metric_ids_order.remove(sort_by_metric_id)
                self.metric_ids_order.insert(0, sort_by_metric_id)
            for sort_by_metric_id in self.metric_ids_order:
                self._sort_metrics_vals(
                    sort_by_metric_id,
                    agg_data,
                    reverse=self.metrics_meta.is_higher_better(sort_by_metric_id),
                )

                md += "\n"
                md += (
                    f"## Models by "
                    f"{self.metrics_meta.get_metric(sort_by_metric_id).display_name}"
                    f"\n"
                )
                md += "Models ordered by the evaluation metric value.\n"
                md += "\n"
                md += "| Rank | LLM "
                for m in self.metric_ids_order:
                    md += f"| {self.metrics_meta.get_metric(m).display_name}"
                md += " |\n| --- | --- "
                for _ in self.metric_ids_order:
                    md += "| ---"
                md += " |\n"

                for e, llm_model_name in enumerate(self.leaderboard_order):
                    md += f"| {e + 1} | {llm_model_name} "
                    for m in self.metric_ids_order:
                        m_avg = agg_data[llm_model_name][m]
                        md += f"| {m_avg:.4f} "
                    md += "|\n"

        # confusion matrices
        for m in self.model_2_confusion_matrix:
            md += "\n"
            md += f"## Model {m} confusion matrix\n"
            md += self._cm_as_markdown_table(
                classes=self.classes,
                confusion_matrix=self.model_2_confusion_matrix[m],
            )

        # prompts failures
        if self.i_failures_count:
            md += "\n"
            md += "## Most difficult prompts\n"
            if top:
                md += "Top prompts "
            else:
                md += "Prompts "
            md += (
                "ordered by model failures - incorrect classifications - "
                "leaderboard of the most difficult prompts."
            )
            if top:
                md += " See the `Report` for the full list."
            md += "\n\n"
            md += "| Prompt | Pass | Fail | Failures rate |\n"
            md += "| --- | --- | --- | --- |\n"

            self.sort_prompts_by_failures(self.i_failures_count)
            for e, i in enumerate(self.inputs_leaderboard_order):
                if top and e >= top:
                    break

                i_safe = sanitization.sanitize_markdown(md_fragment=i)
                passed = self.i_passes_count.get(i, 0)
                failures = self.i_failures_count.get(i, 0)
                accuracy = 100.0 - (passed / (passed + failures)) * 100.0

                if failures > 0:
                    try:
                        md += (
                            f"| {i_safe} | {passed} | {failures} | {accuracy:.1f}% |\n"
                        )
                    except Exception as ex:
                        self.logger.error(
                            f"Unable to render Markdown representation for input "
                            f"{i}: {ex}\n{traceback.format_exc()}"
                        )

        return md

    def get_insights(
        self,
        extra_description_best: str = "",
        extra_description_worst: str = "",
        insight_type: str = "accuracy",
        explanation_type: str = "",
        explanation_name: str = "",
        explanation_mime: str = "",
    ) -> None:
        """Create insights for the classifier leaderboard (based on accuracy metric).

        Parameters
        ----------
        extra_description_best: str
            Additional description for insights related to the best models.
        extra_description_worst: str
            Additional description for insights related to the worst models.
        insight_type : str
            Insight type.
        explanation_type : str
            Type of the explanation which can clarify the insight.
        explanation_name : str
            Name of the explanation which can clarify the insight.
        explanation_mime : str
            Media type of the explanation which can clarify the insight.

        """
        t_insights = insights.InsightAndAction

        evaluator_name = self.explainer._display_name

        if self.insight_most_acc:
            self.explainer.add_insight(
                insights.InsightAndAction(
                    description=(
                        f"Model {self.insight_most_acc} evaluated "
                        f"as the most accurate classification model according to "
                        f"{evaluator_name} evaluator. {extra_description_best}"
                    ),
                    description_html=insights.InsightAndAction.html_most_least_model_by(
                        model_name=self.insight_most_acc,
                        quality="accurate",
                        evaluator_name=evaluator_name,
                        is_most=True,
                        model_purpose="classification",
                        extra_description=extra_description_best,
                    ),
                    insight_type="accuracy",
                    insight_attrs={
                        t_insights.ATTR_MODEL_NAME: self.insight_most_acc,
                        t_insights.ATTR_EVALUATOR_NAME: evaluator_name,
                    },
                    actions_description=(
                        "Refer to the explanation for the detailed description of "
                        "failures, questions, and answers to find out more about "
                        "classification model weaknesses and strengths."
                    ),
                    evaluator_id=self.explainer.evaluator_id(),
                    evaluator_name=evaluator_name,
                    explanation_type=explanation_type,
                    explanation_name=explanation_name,
                    explanation_mime=explanation_mime,
                    resources=[],
                )
            )

        if self.insight_least_acc:
            self.explainer.add_insight(
                insights.InsightAndAction(
                    description=(
                        f"Model {self.insight_least_acc} evaluated "
                        f"as the least accurate classification model according to "
                        f"{evaluator_name} evaluator. {extra_description_worst}"
                    ),
                    description_html=insights.InsightAndAction.html_most_least_model_by(
                        model_name=self.insight_least_acc,
                        quality="accurate",
                        evaluator_name=evaluator_name,
                        is_most=False,
                        model_purpose="classification",
                        extra_description=extra_description_worst,
                    ),
                    insight_type="accuracy",
                    insight_attrs={
                        t_insights.ATTR_MODEL_NAME: self.insight_least_acc,
                        t_insights.ATTR_EVALUATOR_NAME: evaluator_name,
                    },
                    actions_description=(
                        "Refer to the explanation for the detailed description of "
                        "failures, questions, and answers to find out more about "
                        "classification model weaknesses and strengths."
                    ),
                    evaluator_id=self.explainer.evaluator_id(),
                    evaluator_name=evaluator_name,
                    explanation_type=explanation_type,
                    explanation_name=explanation_name,
                    explanation_mime=explanation_mime,
                    resources=[],
                )
            )

        if self.insights_difficult_prompt:
            prompt = self.insights_difficult_prompt

            self.explainer.add_insight(
                insights.InsightAndAction(
                    description=(
                        f"The '{prompt}' prompt was evaluated as the most difficult "
                        f"prompt to be correctly answered according to "
                        f"{evaluator_name} evaluator. {extra_description_worst}"
                    ),
                    description_html=t_insights.html_most_difficult_prompt_by(
                        prompt=prompt,
                        evaluator_name=evaluator_name,
                        extra_description=extra_description_worst,
                        model_purpose="classification",
                    ),
                    insight_type=insight_type,
                    actions_description=(
                        "Refer to the explanation for the detailed description of "
                        "questions and answers by evaluated models."
                    ),
                    evaluator_id=self.explainer.evaluator_id(),
                    evaluator_name=evaluator_name,
                    explanation_type=explanation_type,
                    explanation_name=explanation_name,
                    explanation_mime=explanation_mime,
                    resources=[],
                )
            )
