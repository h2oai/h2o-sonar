# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import datetime
import json
import math
import time

import airium

from h2o_sonar import __version__ as sonar_version
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import htmls
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api.explanations import _explanations_base
from h2o_sonar.lib.api.explanations import _explanations_cmp
from h2o_sonar.lib.api.explanations import _explanations_diff_json
from h2o_sonar.lib.api.explanations._explanations_cmp import _assign_ranks_with_ties
from h2o_sonar.utils import charts


# constants for HTML-specific dictionary keys
KEY_BASELINE_MODEL_KEY = "baseline_model_key"
KEY_CURRENT_MODEL_KEY = "current_model_key"
KEY_BASELINE_LLM_MODEL_NAME = "baseline_llm_model_name"
KEY_CURRENT_LLM_MODEL_NAME = "current_llm_model_name"
KEY_ITEMS = "items"
KEY_BASELINE = "baseline"
KEY_CURRENT = "current"
KEY_FLIPPED_METRICS_COUNT_STATS = "flipped_metrics_count"
KEY_METRICS_AVERAGES = "metrics_averages"
KEY_EMPTY_CONTEXT_COUNT = "empty_context_count"
KEY_BASELINE_AVG = "baseline_avg"
KEY_CURRENT_AVG = "current_avg"
KEY_DIFF_VALUE = "diff"
KEY_BASELINE_BETTER_WINS = "baseline_better_wins"
KEY_CURRENT_BETTER_WINS = "current_better_wins"
KEY_BASELINE_RANK_AVG = "baseline_rank_avg"
KEY_CURRENT_RANK_AVG = "current_rank_avg"
KEY_FLIPPED_TO_PASSED = "flipped_to_passed"
KEY_FLIPPED_TO_FAILED = "flipped_to_failed"
KEY_METRICS_WINS_BASELINE = "metrics_wins_baseline"
KEY_METRICS_WINS_CURRENT = "metrics_wins_current"
KEY_TEST_CASE_WINS_BASELINE = "test_case_wins_baseline"
KEY_TEST_CASE_WINS_CURRENT = "test_case_wins_current"
KEY_TEST_CASE_RANKS_BASELINE = "test_case_ranks_baseline"
KEY_TEST_CASE_RANKS_CURRENT = "test_case_ranks_current"
KEY_METRICS_RANKS_BASELINE = "metrics_ranks_baseline"
KEY_METRICS_RANKS_CURRENT = "metrics_ranks_current"
KEY_TOTAL_TEST_CASES = "total_test_cases"
KEY_TOTAL_METRICS = "total_metrics"
KEY_TOTAL_CONTEXTS = "total_contexts"

# color constants for consistent styling
COLOR_GREEN = "#28a745"  # winner/better/positive
COLOR_RED = "#dc3545"  # loser/worse/negative
COLOR_ORANGE = "#fd7e14"  # neutral/changed
COLOR_YELLOW = "#fec925"  # accent/header
COLOR_BLUE = "#0056b3"  # info/link/baseline
COLOR_LIGHT_GRAY = "#f9f9f9"  # background
COLOR_HEADER_GRAY = "#f2f2f2"  # table headers
COLOR_BORDER_GRAY = "#dee2e6"  # borders
COLOR_TIE_GRAY = "#6c757d"  # tie/neutral
COLOR_DARK_GRAY = "#343a40"  # dark headers
COLOR_WHITE = "#ffffff"  # white background
COLOR_OFF_WHITE = "#f8f9fa"  # alternating row background

# technical metrics keys
KEY_TECHNICAL_METRICS = "technical_metrics"
KEY_COST_SUM = "cost_sum"
KEY_DURATION_SUM = "duration_sum"
KEY_DURATION_MIN = "duration_min"
KEY_DURATION_MAX = "duration_max"
KEY_DURATION_AVG = "duration_avg"
KEY_SUCCESS_COUNT = "success_count"
KEY_FAILURE_COUNT = "failure_count"
KEY_RETRY_COUNT = "retry_count"
KEY_TIMEOUT_COUNT = "timeout_count"
KEY_CALL_COUNT = "call_count"
KEY_INPUT_TOKENS = "input_tokens"
KEY_OUTPUT_TOKENS = "output_tokens"
KEY_TOKENS_PER_SECOND = "tokens_per_second"
KEY_TIME_TO_FIRST_TOKEN = "time_to_first_token"


class EvalResultsDiffHtml:
    """HTML generator for evaluation results differences."""

    def __init__(
        self,
        eval_results_diff,
        branding: commons.Branding = commons.Branding.H2O_SONAR,
    ):
        """Initialize HTML generator with EvalResultsDiff instance.

        Parameters
        ----------
        eval_results_diff : EvalResultsDiff
            The evaluation results diff object to generate HTML for.
        branding : commons.Branding
            H2O Sonar vs. H2O Eval Studio branding.

        """
        self.diff_obj = eval_results_diff

        # branding: H2O Sonar vs. Eval Studio
        self.branding = branding
        if self.branding == commons.Branding.H2O_SONAR:
            self.brand_h2o_sonar = "H2O Sonar"
        else:
            self.brand_h2o_sonar = "Eval Studio"

    @staticmethod
    def _format_model_display_name(
        explainable_model: (
            models.ExplainableLlmModel | models.ExplainableRagModel | None
        ),
        llm_model_name: str,
    ) -> str:
        """Format model display name as [LLM model name]@[model type].

        Parameters
        ----------
        explainable_model : ExplainableLlmModel | ExplainableRagModel | None
            The explainable model instance (may be None if not available).
        llm_model_name : str
            The LLM model name (or model key if model is not available).

        Returns
        -------
        str
            Formatted model display name in the format "[LLM]@[type]",
            or just the LLM name if model type is not available.

        """
        if explainable_model and hasattr(explainable_model, "model_type"):
            model_type = explainable_model.model_type
            # handle enum model_type
            if hasattr(model_type, "name"):
                model_type_str = model_type.name
            else:
                model_type_str = str(model_type)
            return f"{llm_model_name}@{model_type_str}"
        return llm_model_name

    def _get_similarity_metric_description(self) -> str:
        """Get human-readable description of the text similarity metric.

        Returns
        -------
        str :
            Description of the current comparison method.

        """
        method = self.diff_obj.comparison_method

        if method == _explanations_base.SentenceComparisonMethod.EXACT_MATCH:
            return (
                "sentences and chunks must match character-by-character. "
                "This is the fastest method but only identifies identical text."
            )
        elif method == _explanations_base.SentenceComparisonMethod.COSINE_DISTANCE:
            return (
                "uses cosine distance of sentence embeddings to measure semantic "
                "similarity. Sentences with similar meanings will be identified "
                "as common even if worded differently. This method balances "
                "accuracy and speed using the BAAI/bge-small-en-v1.5 model."
            )
        elif method == _explanations_base.SentenceComparisonMethod.BERT_SCORE:
            return (
                "uses BERTScore with contextual BERT embeddings for the most "
                "accurate semantic similarity measurement. This captures nuanced "
                "meaning differences but is slower than other methods."
            )
        else:
            return "Unknown comparison method."

    def to_html(
        self,
    ) -> str:
        """Generate HTML representation of evaluation results differences.

        Returns
        -------
        str :
            HTML string representation of the diffs.
        """
        html = airium.Airium()
        html("<!DOCTYPE html>")
        with html.html(lang="en"):
            htmls.evaluation_report_html_head(
                html, title="Evaluations Results Comparison"
            )

            with html.style():
                html(
                    f"""
                    .diff-section {{margin-bottom: 40px; padding: 20px;}}
                    .comparison-table {{width: 100%; border-collapse: collapse;}}
                    .comparison-table th, .comparison-table td {{
                        border: 1px solid #ddd;
                        padding: 12px;
                        vertical-align: top;
                    }}
                    .comparison-table th {{
                        background-color: {COLOR_HEADER_GRAY};
                        font-weight: bold;
                        text-align: center;
                    }}
                    .comparison-table th:first-child {{
                        width: 20%;
                    }}
                    .comparison-table th:not(:first-child) {{
                        width: 40%;
                    }}
                    .comparison-table td.field-label {{
                        background-color: {COLOR_LIGHT_GRAY};
                        font-weight: bold;
                        width: 20%;
                    }}
                    .comparison-table td:not(.field-label) {{
                        width: 40%;
                    }}
                    .sentence-common {{
                        background-color: #d4edda;
                        padding: 2px 4px; margin: 2px;
                        display: inline-block;
                    }}
                    .sentence-unique {{
                        background-color: #f8d7da;
                        padding: 2px 4px; margin: 2px;
                        display: inline-block;
                    }}
                    .metric-row {{background-color: {COLOR_LIGHT_GRAY};}}
                    .context-chunk {{
                        background-color: #e9ecef;
                        padding: 8px;
                        margin: 4px 0;
                        border-radius: 4px;
                        font-size: 0.9em;
                    }}
                    .context-common {{background-color: #d4edda;}}
                    .context-unique {{background-color: #f8d7da;}}
                    .question-header {{
                        background-color: #000000;
                        color: white;
                        padding: 15px;
                        border-radius: 5px;
                        margin-bottom: 20px;
                    }}
                    .model-header {{
                        background-color: {COLOR_YELLOW};
                        color: black;
                        padding: 20px;
                        border-radius: 5px;
                        margin-bottom: 30px;
                        margin-top: 30px;
                    }}
                    .model-overview {{
                        background-color: {COLOR_LIGHT_GRAY};
                        padding: 15px;
                        margin-bottom: 20px;
                        border-left: 4px solid {COLOR_YELLOW};
                    }}
                    .model-overview-table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 15px;
                    }}
                    .model-overview-table th,
                    .model-overview-table td {{
                        border: 1px solid #ddd;
                        padding: 10px;
                        text-align: left;
                    }}
                    .model-overview-table th {{
                        background-color: {COLOR_HEADER_GRAY};
                        font-weight: bold;
                    }}
                    .model-overview-table td.field-label {{
                        background-color: {COLOR_LIGHT_GRAY};
                        font-weight: bold;
                        width: 25%;
                    }}
                    .value-same {{
                        color: {COLOR_GREEN};
                    }}
                    .value-different {{
                        color: {COLOR_ORANGE};
                    }}
                    .metric-flipped {{
                        color: {COLOR_RED};
                        font-weight: bold;
                    }}
                    .metric-changed {{
                        color: {COLOR_ORANGE};
                        font-weight: bold;
                    }}
                    .leaderboard-section {{
                        background-color: {COLOR_WHITE};
                        padding: 20px;
                        margin-bottom: 30px;
                        border-radius: 5px;
                        border-left: 4px solid {COLOR_YELLOW};
                    }}
                    .leaderboard-table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 10px;
                        background-color: white;
                    }}
                    .leaderboard-table th,
                    .leaderboard-table td {{
                        border: 1px solid #ddd;
                        padding: 10px;
                        text-align: left;
                    }}
                    .leaderboard-table th {{
                        background-color: {COLOR_YELLOW};
                        color: black;
                        font-weight: bold;
                        text-align: center;
                    }}
                    .leaderboard-table tr:hover {{
                        background-color: {COLOR_LIGHT_GRAY};
                    }}
                    .leaderboard-table a {{
                        color: #0066cc;
                        text-decoration: none;
                    }}
                    .leaderboard-table a:hover {{
                        text-decoration: underline;
                    }}
                    """
                )

            with html.body():
                # navigation
                self._html_left_navigation(html)
                # main
                with html.div(klass="w3-main", style="margin-left:300px"):
                    self._html_main_content(html)

                # section: footer
                EvalResultsDiffHtml.html_footer(
                    html, brand_h2o_sonar=self.brand_h2o_sonar, branding=self.branding
                )

        return str(html)

    def _html_left_navigation(self, html):
        """Generate left navigation panel."""
        with html.nav(
            klass="w3-sidebar w3-collapse w3-black w3-animate-left",
            id="mySidebar",
            style="z-index:3;width:300px;",
        ):
            html.br()
            with html.div(klass="w3-container"):
                with html.a(klass="w3-left w3-margin-right", href="#"):
                    htmls.html_svg_h2oai_logo(html)
                with html.h3():
                    html.b(
                        _t=self.brand_h2o_sonar,
                        style=f"color: {COLOR_YELLOW}",
                    )
                html.h4(_t="Evaluations Comparison")

            # GROUP diffs by model pair
            diffs_by_model_pair = {}
            idx = 1
            for model_pair, diff_list in self.diff_obj.diffs.items():
                # skip model pairs with no comparable test cases
                if not diff_list:
                    continue

                baseline_model_key, current_model_key = model_pair
                baseline_explainable_model = (
                    self.diff_obj._get_explainable_model_by_key(
                        baseline_model_key, is_baseline=True
                    )
                )
                current_explainable_model = self.diff_obj._get_explainable_model_by_key(
                    current_model_key, is_baseline=False
                )
                baseline_llm_model_name = (
                    baseline_explainable_model.llm_model_name
                    if baseline_explainable_model
                    else baseline_model_key
                )
                current_llm_model_name = (
                    current_explainable_model.llm_model_name
                    if current_explainable_model
                    else current_model_key
                )
                # format display names with model type
                baseline_display_name = self._format_model_display_name(
                    baseline_explainable_model, baseline_llm_model_name
                )
                current_display_name = self._format_model_display_name(
                    current_explainable_model, current_llm_model_name
                )
                model_pair_key = f"{baseline_model_key}|{current_model_key}"
                if model_pair_key not in diffs_by_model_pair:
                    diffs_by_model_pair[model_pair_key] = {
                        KEY_BASELINE_MODEL_KEY: baseline_model_key,
                        KEY_CURRENT_MODEL_KEY: current_model_key,
                        KEY_BASELINE_LLM_MODEL_NAME: baseline_display_name,
                        KEY_CURRENT_LLM_MODEL_NAME: current_display_name,
                        KEY_ITEMS: [],
                    }
                for diff in diff_list:
                    diffs_by_model_pair[model_pair_key][KEY_ITEMS].append((idx, diff))
                    idx += 1

            # add overall comparison link if multiple model pairs
            if len(diffs_by_model_pair) > 1:
                with html.div(klass="w3-container"):
                    with html.a(
                        href="#overall-evaluations-comparison",
                        klass="w3-text-yellow",
                        style="text-decoration: none;",
                    ):
                        html.b(_t="Overall Comparison")
                with html.div(klass="w3-bar-block"):
                    with html.a(
                        klass="w3-bar-item w3-button w3-padding w3-margin-left",
                        href="#overall-final-recommendation",
                        title=(
                            "Final recommendation based on the overall comparison of "
                            "the evaluations across all test cases."
                        ),
                    ):
                        html("Final Recommendation")
                    with html.a(
                        klass="w3-bar-item w3-button w3-padding w3-margin-left",
                        href="#overall-evaluations-overview",
                        title="Overview of the evaluations being compared.",
                    ):
                        html("Evaluations Overview")
                    with html.a(
                        klass="w3-bar-item w3-button w3-padding w3-margin-left",
                        href="#overall-model-pairs-comparison",
                        title=(
                            "Comparison of the explainable models from both "
                            "evaluations."
                        ),
                    ):
                        html("Model Pairs Comparison")
                    with html.a(
                        klass="w3-bar-item w3-button w3-padding w3-margin-left",
                        href="#overall-models-comparison",
                        title="Aggregated comparison  across all test cases.",
                    ):
                        html("Models Comparison")
                    with html.a(
                        klass="w3-bar-item w3-button w3-padding w3-margin-left",
                        href="#overall-technical-metrics",
                        title=(
                            "Aggregated technical performance metrics across all "
                            "test cases."
                        ),
                    ):
                        html("Technical Performance Metrics")

            for model_pair_key in sorted(diffs_by_model_pair.keys()):
                model_info = diffs_by_model_pair[model_pair_key]
                baseline_llm_model_name = model_info[KEY_BASELINE_LLM_MODEL_NAME]
                current_llm_model_name = model_info[KEY_CURRENT_LLM_MODEL_NAME]
                baseline_model_key = model_info[KEY_BASELINE_MODEL_KEY]

                with html.div(klass="w3-container"):
                    with html.a(
                        href=f"#model-{model_pair_key}",
                        style="text-decoration: none;",
                    ):
                        with html.p(klass="w3-text-yellow"):
                            html("Models: ")
                            html.b(_t=f"{baseline_llm_model_name}")
                            html(" vs ")
                            html.b(_t=f"{current_llm_model_name}")

                with html.div(klass="w3-bar-block"):
                    for idx, diff in model_info[KEY_ITEMS]:
                        # prefix...suffix format for long questions
                        question_text = diff.question
                        if len(question_text) > 50:
                            prefix = question_text[:30]
                            suffix = question_text[-17:]
                            question_text = f"{prefix}...{suffix}"

                        with html.a(
                            klass="w3-bar-item w3-button w3-padding w3-margin-left",
                            href=f"#test-case-{idx}",
                            title=diff.question,
                        ):
                            html(f"Q: {question_text}")

    def _html_main_content(self, html):
        with html.div(klass="w3-container w3-padding-large"):
            html.h1(_t="Evaluations Results Comparison")

            # count total test cases
            total_test_cases = sum(
                len(diff_list) for diff_list in self.diff_obj.diffs.values()
            )
            with html.p():
                html("This ")
                html.b(_t="comparison")
                html(" report compares models of ")
                html.b(_t="baseline")
                html(" and ")
                html.b(_t="current")
                html(" evaluations.")

            # GROUP diffs by model pair
            diffs_by_model_pair = {}
            idx = 1
            for model_pair, diff_list in self.diff_obj.diffs.items():
                # skip model pairs with no comparable test cases
                if not diff_list:
                    continue

                baseline_model_key, current_model_key = model_pair
                model_pair_key = f"{baseline_model_key}|{current_model_key}"
                if model_pair_key not in diffs_by_model_pair:
                    diffs_by_model_pair[model_pair_key] = []
                for diff in diff_list:
                    diffs_by_model_pair[model_pair_key].append((idx, diff))
                    idx += 1

            # collect all recommendations for overview
            recommendations_summary = {
                "baseline": 0,
                "current": 0,
                "tie": 0,
            }
            comparison_data = []

            for model_pair_key in sorted(diffs_by_model_pair.keys()):
                model_diffs = diffs_by_model_pair[model_pair_key]
                if model_diffs:
                    _, first_diff = model_diffs[0]
                    baseline_model_key = first_diff.baseline_test_case.get(
                        _explanations_cmp.KEY_MODEL_KEY, ""
                    )
                    current_model_key = first_diff.current_test_case.get(
                        _explanations_cmp.KEY_MODEL_KEY, ""
                    )

                    baseline_explainable_model = (
                        self.diff_obj._get_explainable_model_by_key(
                            baseline_model_key, is_baseline=True
                        )
                    )
                    current_explainable_model = (
                        self.diff_obj._get_explainable_model_by_key(
                            current_model_key, is_baseline=False
                        )
                    )

                    baseline_model_name = (
                        baseline_explainable_model.llm_model_name
                        if baseline_explainable_model
                        else baseline_model_key
                    )
                    current_model_name = (
                        current_explainable_model.llm_model_name
                        if current_explainable_model
                        else current_model_key
                    )
                    # format display names with model type
                    baseline_display_name = self._format_model_display_name(
                        baseline_explainable_model, baseline_model_name
                    )
                    current_display_name = self._format_model_display_name(
                        current_explainable_model, current_model_name
                    )

                    # calculate recommendation
                    recommendation_result = None
                    metrics_meta = self.diff_obj.metrics_meta
                    if metrics_meta:
                        calc_stats = EvalResultsDiffHtml._calculate_model_cmp_stats
                        comparison_stats = calc_stats(
                            model_diffs,
                            metrics_meta,
                            baseline_explainable_model,
                            current_explainable_model,
                        )
                        recommendation_result = (
                            EvalResultsDiffHtml._calculate_recommendation(
                                comparison_stats, metrics_meta
                            )
                        )
                        # count recommendations
                        winner = recommendation_result["winner"]
                        if winner in recommendations_summary:
                            recommendations_summary[winner] += 1

                    comparison_data.append(
                        {
                            "model_pair_key": model_pair_key,
                            "baseline_display_name": baseline_display_name,
                            "current_display_name": current_display_name,
                            "recommendation": recommendation_result,
                        }
                    )

            # comparison details
            html.h3(_t="Method")
            html.p(
                _t=(
                    "This comparison analyzes differences between baseline model and "
                    "current model evaluation results to help identify how models "
                    "perform across tests and their test cases. The comparison matches"
                    " test cases by their question and model key, then evaluates "
                    "changes in metrics, actual answers, and retrieved context."
                )
            )
            with html.ul():
                with html.li():
                    html.b(_t="Test cases: ")
                    with html.span(style=f"background-color: {COLOR_YELLOW};"):
                        html(str(total_test_cases))
                    html(
                        " comparable test case(s) were "
                        "found across the baseline and current models. "
                        "Comparable test cases must have identical questions and "
                        "model keys. Duplicate test cases having the same question "
                        "in one of the compared evaluations are discarded."
                    )
                with html.li():
                    html.b(_t="Model recommendation: ")
                    html(
                        "Models are ranked by the number of metrics where "
                        "they score better than their baseline/current "
                        "counterpart. For each metric, the model with a "
                        "superior value (according to the metric's "
                        "'higher_is_better' property) gets a win. The "
                        "model with more wins is recommended. Test cases "
                        "are also ranked by the number of changed metrics."
                    )
                with html.li():
                    html.b(_t="Comparison approach: ")
                    html(
                        "For each test case, the comparison identifies "
                        "metrics that changed values and metrics that "
                        "flipped pass/fail status. Actual answers and "
                        "retrieved context chunks are compared using the "
                        "selected text similarity metric to highlight "
                        "common and unique content."
                    )
                with html.li():
                    html.b(_t="Text similarity metric: ")
                    with html.span(style=f"background-color: {COLOR_YELLOW};"):
                        html(self.diff_obj.comparison_method.name)
                    html(
                        f" - "
                        f"{self._get_similarity_metric_description()}"
                        f" This metric is used to identify common and "
                        f"unique sentences in actual answers and to "
                        f"compare retrieved context chunks between "
                        f"baseline and current results."
                    )

            # OVERALL EVALUATIONS COMPARISON (if multiple model pairs)
            # place this at the top to give high-level summary first
            if len(comparison_data) > 1:
                self._html_overall_evaluations_sections(html, comparison_data)

            if not self.diff_obj.diffs:
                with html.div(klass="w3-panel w3-pale-yellow w3-border"):
                    html.p(_t="No differences found between the evaluation results.")
                return

            # PER-MODEL SECTIONS
            for model_pair_key in sorted(diffs_by_model_pair.keys()):
                model_diffs = diffs_by_model_pair[model_pair_key]
                self._html_model_section(html, model_pair_key, model_diffs)

    def _get_delta_color(self, delta: float, higher_is_better: bool = True) -> str:
        """Get color for delta display based on whether higher is better.

        Parameters
        ----------
        delta : float
            The delta value (current - baseline).
        higher_is_better : bool
            Whether higher values are better (default True).

        Returns
        -------
        str
            Color code (GREEN or RED).

        """
        if higher_is_better:
            return COLOR_GREEN if delta > 0 else COLOR_RED
        else:
            return COLOR_RED if delta > 0 else COLOR_GREEN

    def _html_delta_display(
        self,
        html,
        baseline: float,
        current: float,
        is_float: bool = True,
        higher_is_better: bool = True,
        unit: str = "",
    ):
        """Render delta display with color coding.

        Parameters
        ----------
        html : htmltree.HtmlElement
            HTML element to append to.
        baseline : float
            Baseline value.
        current : float
            Current value.
        is_float : bool
            Whether to format as float (default True).
        higher_is_better : bool
            Whether higher values are better (default True).
        unit : str
            Optional unit suffix (e.g., "s", "$").

        """
        delta = current - baseline
        if delta != 0:
            html(" ")
            delta_sign = "+" if delta > 0 else ""
            delta_color = self._get_delta_color(delta, higher_is_better)

            if is_float:
                delta_text = f"({delta_sign}{unit}{delta:.2f})"
            else:
                delta_text = f"({delta_sign}{unit}{delta})"

            html.strong(_t=delta_text, style=f"color: {delta_color}; font-size: 0.9em;")

    def _html_legend(
        self, html, legend_items: list[tuple[str, str]], title: str = "Legend:"
    ):
        """Render a legend section with consistent styling.

        Parameters
        ----------
        html : htmltree.HtmlElement
            HTML element to append to.
        legend_items : list[tuple[str, str]]
            List of (label, description) tuples.
        title : str
            Legend title (default "Legend:").

        """
        html.p(
            _t=title, style="margin-top: 20px; margin-bottom: 10px; font-weight: bold;"
        )
        with html.ul(style="margin-left: 20px; margin-bottom: 30px; line-height: 1.8;"):
            for label, description in legend_items:
                with html.li():
                    html.strong(_t=label)
                    html(f": {description}")

    def _html_overall_evaluations_sections(self, html, comparison_data: list[dict]):
        """Generate overall evaluations comparison sections.

        Parameters
        ----------
        html : htmltree.HtmlElement
            HTML element to append to.
        comparison_data : list[dict]
            List of comparison data dictionaries (one per model pair).

        """
        # get overall comparison data from diff_obj
        overall_data = self._get_overall_comparison_data()

        if not overall_data:
            return

        # section header - yellow text on pure black background
        with html.div(
            id="overall-evaluations-comparison",
            style=(
                f"background-color: #000000; color: {COLOR_YELLOW}; "
                "padding: 20px; border-radius: 5px; "
                "margin-top: 20px; margin-bottom: 30px;"
            ),
        ):
            with html.h2(style=f"margin: 0; color: {COLOR_YELLOW};"):
                html("OVERALL EVALUATIONS COMPARISON")
            html.p(
                _t=(
                    "This section aggregates comparison metrics across all model pairs "
                    "to provide an overall assessment of baseline vs "
                    "current evaluations."
                ),
                style=f"margin: 10px 0 0 0; font-size: 0.9em; color: {COLOR_YELLOW};",
            )

        # 1a) final recommendation
        self._html_overall_summary(
            html, overall_data[_explanations_cmp.KEY_OVERALL_SUMMARY]
        )

        # 1c) evaluations overview (with merged recommendations summary)
        self._html_overall_evaluations_overview(
            html,
            overall_data[_explanations_cmp.KEY_OVERALL_EVALUATIONS_OVERVIEW],
            comparison_data,
        )

        # 1d) models comparison (aggregated)
        self._html_overall_models_comparison(
            html, overall_data[_explanations_cmp.KEY_OVERALL_MODELS_COMPARISON]
        )

        # 1e) technical performance metrics (aggregated)
        self._html_overall_technical_metrics(
            html, overall_data[_explanations_cmp.KEY_OVERALL_TECHNICAL_METRICS]
        )

    def _html_overall_summary(self, html, summary_data: dict):
        """Render overall recommendation section.

        Parameters
        ----------
        html : htmltree.HtmlElement
            HTML element to append to.
        summary_data : dict
            Summary data containing recommendation, winner, and confidence.

        """
        winner = summary_data[_explanations_cmp.KEY_OVERALL_RECOMMENDATION_WINNER]
        recommendation = summary_data[_explanations_cmp.KEY_OVERALL_RECOMMENDATION]
        confidence = summary_data[
            _explanations_cmp.KEY_OVERALL_RECOMMENDATION_CONFIDENCE
        ]

        # color coding
        if winner == "baseline":
            color = COLOR_BLUE
            symbol = "BASELINE"
        elif winner == "current":
            color = COLOR_GREEN
            symbol = "CURRENT"
        else:
            color = COLOR_TIE_GRAY
            symbol = "TIE"

        with html.div(
            klass="overall-summary",
            style=(
                f"background-color: {COLOR_OFF_WHITE}; border-left: 5px solid {color}; "
                "padding: 20px; margin: 20px 0; border-radius: 5px;"
            ),
            id="overall-final-recommendation",
        ):
            html.h3(_t="Final Recommendation", style="margin-top: 0;")
            with html.div(style="text-align: center; margin: 20px 0;"):
                html.strong(
                    _t=symbol,
                    style=f"font-size: 1.5em; color: {color}; font-weight: bold;",
                )
                html.div(
                    _t=f"Confidence: {confidence.upper()}",
                    style=f"margin-top: 10px; color: {color}; font-weight: bold;",
                )
            html.p(_t=recommendation)

    def _html_overall_evaluations_overview(
        self, html, overview_data: dict, comparison_data: list[dict]
    ):
        """Render overall evaluations overview section.

        Parameters
        ----------
        html : htmltree.HtmlElement
            HTML element to append to.
        overview_data : dict
            Overview data containing model counts, types, and collections.
        comparison_data : list[dict]
            List of comparison data dictionaries (one per model pair).

        """
        # calculate recommendations summary from comparison_data
        recommendations_summary = {"baseline": 0, "tie": 0, "current": 0}
        for comp_data in comparison_data:
            # access "recommendation" key directly (not KEY_SUMMARY which is "summary")
            winner = comp_data.get("recommendation", {}).get("winner", "tie")
            recommendations_summary[winner] = recommendations_summary.get(winner, 0) + 1
        with html.div(
            klass="overall-evaluations-overview",
            style=(
                f"background-color: {COLOR_LIGHT_GRAY}; "
                f"border-left: 4px solid {COLOR_YELLOW}; "
                "padding: 20px; margin: 20px 0; border-radius: 5px;"
            ),
            id="overall-evaluations-overview",
        ):
            html.h3(_t="Evaluations Overview")

            with html.table(
                klass="model-overview-table",
                style="width: 100%; border-collapse: collapse; margin-top: 15px;",
            ):
                # header
                with html.tr():
                    html.th(
                        _t="Metric",
                        style=f"background-color: {COLOR_HEADER_GRAY}; padding: 10px;",
                    )
                    html.th(
                        _t="Baseline Evaluation",
                        style=(
                            f"background-color: {COLOR_HEADER_GRAY}; "
                            f"padding: 10px; text-align: center;"
                        ),
                    )
                    html.th(
                        _t="Current Evaluation",
                        style=(
                            f"background-color: {COLOR_HEADER_GRAY}; "
                            f"padding: 10px; text-align: center;"
                        ),
                    )

                # models count
                with html.tr():
                    html.td(
                        _t="Number of Models", style="padding: 10px; font-weight: bold;"
                    )
                    html.td(
                        _t=str(
                            overview_data[_explanations_cmp.KEY_BASELINE_MODELS_COUNT]
                        ),
                        style="padding: 10px; text-align: center;",
                    )
                    html.td(
                        _t=str(
                            overview_data[_explanations_cmp.KEY_CURRENT_MODELS_COUNT]
                        ),
                        style="padding: 10px; text-align: center;",
                    )

                # model types
                with html.tr():
                    html.td(_t="Model Types", style="padding: 10px; font-weight: bold;")
                    baseline_types = overview_data[
                        _explanations_cmp.KEY_BASELINE_MODEL_TYPES
                    ]
                    html.td(
                        _t=", ".join(baseline_types) if baseline_types else "N/A",
                        style="padding: 10px;",
                    )
                    current_types = overview_data[
                        _explanations_cmp.KEY_CURRENT_MODEL_TYPES
                    ]
                    html.td(
                        _t=", ".join(current_types) if current_types else "N/A",
                        style="padding: 10px;",
                    )

                # unique collection IDs
                with html.tr():
                    html.td(
                        _t="Unique Collections",
                        style="padding: 10px; font-weight: bold;",
                    )
                    html.td(
                        _t=str(
                            overview_data[
                                _explanations_cmp.KEY_BASELINE_UNIQUE_COLLECTIONS
                            ]
                        ),
                        style="padding: 10px; text-align: center;",
                    )
                    html.td(
                        _t=str(
                            overview_data[
                                _explanations_cmp.KEY_CURRENT_UNIQUE_COLLECTIONS
                            ]
                        ),
                        style="padding: 10px; text-align: center;",
                    )

                # total comparable test cases
                with html.tr():
                    html.td(
                        _t="Total Test Cases",
                        style="padding: 10px; font-weight: bold;",
                    )
                    with html.td(
                        colspan="2", style="padding: 10px; text-align: center;"
                    ):
                        html(
                            str(
                                overview_data[
                                    _explanations_cmp.KEY_TOTAL_COMPARABLE_TEST_CASES
                                ]
                            )
                        )

                # separator row
                with html.tr():
                    html.td(
                        colspan="3",
                        style=(
                            f"padding: 10px; background-color: {COLOR_BORDER_GRAY}; "
                            "font-weight: bold; text-align: center;"
                        ),
                        _t="Model Recommendations Summary",
                    )

                # recommendations summary rows
                with html.tr():
                    html.td(
                        _t="Baseline Recommendations",
                        style="padding: 10px; font-weight: bold;",
                    )
                    with html.td(
                        colspan="2", style="padding: 10px; text-align: center;"
                    ):
                        html.strong(
                            _t=f"{recommendations_summary['baseline']}x",
                            style=f"color: {COLOR_BLUE}; font-size: 1.1em;",
                        )

                with html.tr():
                    html.td(
                        _t="Tie Recommendations",
                        style="padding: 10px; font-weight: bold;",
                    )
                    with html.td(
                        colspan="2", style="padding: 10px; text-align: center;"
                    ):
                        html.strong(
                            _t=f"{recommendations_summary['tie']}x",
                            style=f"color: {COLOR_TIE_GRAY}; font-size: 1.1em;",
                        )

                with html.tr():
                    html.td(
                        _t="Current Recommendations",
                        style="padding: 10px; font-weight: bold;",
                    )
                    with html.td(
                        colspan="2", style="padding: 10px; text-align: center;"
                    ):
                        html.strong(
                            _t=f"{recommendations_summary['current']}x",
                            style=f"color: {COLOR_GREEN}; font-size: 1.1em;",
                        )

            # legend explaining the overview metrics
            html.p(
                _t="Legend:",
                style="margin-top: 20px; margin-bottom: 10px; font-weight: bold;",
            )
            with html.ul(
                style="margin-left: 20px; margin-bottom: 30px; line-height: 1.8;"
            ):
                with html.li():
                    html.strong(_t="Number of Models")
                    html(": Count of unique model pairs being compared.")
                with html.li():
                    html.strong(_t="Model Types")
                    html(": LLM model names used in evaluations (e.g., GPT-4, Claude).")
                with html.li():
                    html.strong(_t="Unique Collections")
                    html(": Number of distinct document collections/corpora used.")
                with html.li():
                    html.strong(_t="Total Comparable Test Cases")
                    html(
                        ": Total number of test cases that could be compared between "
                        "baseline and current evaluations."
                    )
                with html.li():
                    html.strong(_t="Model Recommendations Summary")
                    html(
                        ": Count of per-model recommendations "
                        "(Baseline, Tie, or Current). "
                    )
                    html("Each model pair comparison produces one recommendation.")

        # add Model Pairs Comparison table as last element in overview
        self._html_model_pairs_table(
            html=html,
            comparison_data=comparison_data,
            recommendations_summary=recommendations_summary,
        )

    def _html_model_pairs_table(
        self, html, comparison_data: list[dict], recommendations_summary: dict
    ):
        """Render Model Pairs Comparison table.

        Parameters
        ----------
        html : htmltree.HtmlElement
            HTML element to append to.
        comparison_data : list[dict]
            List of comparison data dictionaries (one per model pair).

        """
        with html.div(
            klass="model-pairs-comparison",
            style=(
                f"background-color: {COLOR_LIGHT_GRAY}; "
                f"border-left: 4px solid {COLOR_YELLOW}; "
                "padding: 20px; margin: 20px 0; border-radius: 5px;"
            ),
            id="overall-model-pairs-comparison",
        ):
            html.h3(_t="Model Pairs Comparison (Per-Model)", style="margin-top: 0;")
            html.p(
                _t=(
                    "Explainable model in the table below is defined by the host "
                    "type (like h2oGPTe or OpenAI), LLM model (like GPT or Llama), "
                    "corpus (like SR 11-7 model PDF or CRM manuals) and host system "
                    "configuration (like embedding model or chunking strategy)."
                )
            )

            # BAR CHART
            # add visual comparison chart for recommendations below the table
            baseline_recs = recommendations_summary.get("baseline", 0)
            tie_recs = recommendations_summary.get("tie", 0)
            current_recs = recommendations_summary.get("current", 0)

            # only generate chart if at least one value is non-zero
            if baseline_recs > 0 or tie_recs > 0 or current_recs > 0:
                try:
                    chart_svg = charts.generate_svg_bar_chart(
                        labels=["Baseline", "Tie", "Current"],
                        values=[
                            float(baseline_recs),
                            float(tie_recs),
                            float(current_recs),
                        ],
                        title="Model Recommendations Distribution",
                        width=600,
                        height=300,
                        show_values=True,
                        show_grid=True,
                        bar_colors=[COLOR_BLUE, COLOR_TIE_GRAY, COLOR_YELLOW],
                    )
                    if chart_svg and isinstance(chart_svg, str) and "<svg" in chart_svg:
                        with html.div(style="margin: 30px 0; text-align: center;"):
                            charts.add_svg_chart_to_html(html, chart_svg)
                except Exception:
                    # fail silently - never crash HTML generation
                    pass

            with html.table(
                klass="comparisons-table",
                style=(
                    f"width: 100%; border-collapse: collapse; margin: 20px 0; "
                    f"border: 1px solid {COLOR_BORDER_GRAY};"
                ),
            ):
                # table header
                with html.tr(
                    style=f"background-color: {COLOR_DARK_GRAY}; color: white;"
                ):
                    html.th(
                        _t="Baseline Explanation Models",
                        style=(
                            f"padding: 12px; text-align: center; "
                            f"border: 1px solid {COLOR_BORDER_GRAY};"
                        ),
                    )
                    html.th(
                        _t="Current Explanation Models",
                        style=(
                            f"padding: 12px; text-align: center; "
                            f"border: 1px solid {COLOR_BORDER_GRAY};"
                        ),
                    )
                    html.th(
                        _t="Recommendation",
                        style=(
                            f"padding: 12px; text-align: center; "
                            f"border: 1px solid {COLOR_BORDER_GRAY};"
                        ),
                    )

                # table rows
                for idx, comp_data in enumerate(comparison_data):
                    row_bg = COLOR_WHITE if idx % 2 == 0 else COLOR_OFF_WHITE
                    with html.tr(style=f"background-color: {row_bg};"):
                        # baseline model column
                        with html.td(
                            style=(
                                f"padding: 12px; "
                                f"border: 1px solid {COLOR_BORDER_GRAY}; "
                                "vertical-align: middle;"
                            )
                        ):
                            html.a(
                                href=f"#model-{comp_data['model_pair_key']}",
                                _t=comp_data["baseline_display_name"],
                                style="text-decoration: underline; color: #000000;",
                            )

                        # current model column
                        with html.td(
                            style=(
                                f"padding: 12px; "
                                f"border: 1px solid {COLOR_BORDER_GRAY}; "
                                "vertical-align: middle;"
                            )
                        ):
                            html(comp_data["current_display_name"])

                        # recommendation column
                        with html.td(
                            style=(
                                f"padding: 12px; "
                                f"border: 1px solid {COLOR_BORDER_GRAY}; "
                                "text-align: center; vertical-align: middle;"
                            )
                        ):
                            recommendation = comp_data["recommendation"]
                            if recommendation:
                                winner = recommendation["winner"]
                                if winner == "current":
                                    html.strong(
                                        _t="✓ CURRENT",
                                        style=(
                                            f"color: {COLOR_GREEN}; font-size: 1.1em;"
                                        ),
                                    )
                                elif winner == "baseline":
                                    html.strong(
                                        _t="✓ BASELINE",
                                        style=f"color: {COLOR_BLUE}; font-size: 1.1em;",
                                    )
                                else:
                                    html.strong(
                                        _t="⚖ TIE",
                                        style=(
                                            f"color: {COLOR_TIE_GRAY}; "
                                            f"font-size: 1.1em;"
                                        ),
                                    )
                            else:
                                html("N/A")

    def _html_overall_models_comparison(self, html, comparison_data: dict):
        """Render overall models comparison section (aggregated wins/ranks).

        Parameters
        ----------
        html : htmltree.HtmlElement
            HTML element to append to.
        comparison_data : dict
            Comparison data containing aggregated wins and ranks.

        """
        with html.div(
            klass="overall-models-comparison",
            style=(
                f"background-color: {COLOR_LIGHT_GRAY}; "
                f"border-left: 4px solid {COLOR_YELLOW}; "
                "padding: 20px; margin: 20px 0; border-radius: 5px;"
            ),
            id="overall-models-comparison",
        ):
            html.h3(_t="Models Comparison (Aggregated)")
            html.p(
                _t=(
                    "Test case and metric wins summed across all model comparisons. "
                    "These are absolute counts, not averages."
                )
            )

            baseline_wins = comparison_data[
                _explanations_cmp.KEY_TEST_CASE_WINS_BASELINE
            ]
            current_wins = comparison_data[_explanations_cmp.KEY_TEST_CASE_WINS_CURRENT]
            baseline_metrics_wins = comparison_data[
                _explanations_cmp.KEY_METRICS_WINS_BASELINE
            ]
            current_metrics_wins = comparison_data[
                _explanations_cmp.KEY_METRICS_WINS_CURRENT
            ]

            with html.table(
                klass="model-overview-table",
                style="width: 100%; border-collapse: collapse; margin-top: 15px;",
            ):
                # header
                with html.tr():
                    html.th(
                        _t="Metric",
                        style=f"background-color: {COLOR_HEADER_GRAY}; padding: 10px;",
                    )
                    html.th(
                        _t="Baseline",
                        style=(
                            f"background-color: {COLOR_HEADER_GRAY}; padding: 10px; "
                            f"text-align: center;"
                        ),
                    )
                    html.th(
                        _t="Current",
                        style=(
                            f"background-color: {COLOR_HEADER_GRAY}; padding: 10px; "
                            f"text-align: center;"
                        ),
                    )

                # test case wins
                with html.tr():
                    html.td(
                        _t="Test Case Wins", style="padding: 10px; font-weight: bold;"
                    )
                    self._html_comparison_cell(html, baseline_wins, current_wins)

                # test case ranks
                with html.tr():
                    html.td(
                        _t="Test Case Ranks", style="padding: 10px; font-weight: bold;"
                    )
                    baseline_ranks = comparison_data[
                        _explanations_cmp.KEY_TEST_CASE_RANKS_BASELINE
                    ]
                    current_ranks = comparison_data[
                        _explanations_cmp.KEY_TEST_CASE_RANKS_CURRENT
                    ]
                    self._html_comparison_cell(html, baseline_ranks, current_ranks)

                # metrics wins
                with html.tr():
                    html.td(
                        _t="Metrics Wins", style="padding: 10px; font-weight: bold;"
                    )
                    self._html_comparison_cell(
                        html, baseline_metrics_wins, current_metrics_wins
                    )

                # metrics ranks
                with html.tr():
                    html.td(
                        _t="Metrics Ranks", style="padding: 10px; font-weight: bold;"
                    )
                    baseline_metrics_ranks = comparison_data[
                        _explanations_cmp.KEY_METRICS_RANKS_BASELINE
                    ]
                    current_metrics_ranks = comparison_data[
                        _explanations_cmp.KEY_METRICS_RANKS_CURRENT
                    ]
                    self._html_comparison_cell(
                        html,
                        baseline_metrics_ranks,
                        current_metrics_ranks,
                        is_float=True,
                    )

            # legend explaining aggregated comparison metrics
            html.p(
                _t="Legend:",
                style="margin-top: 20px; margin-bottom: 10px; font-weight: bold;",
            )
            with html.ul(
                style="margin-left: 20px; margin-bottom: 30px; line-height: 1.8;"
            ):
                with html.li():
                    html.strong(_t="Test Case Wins")
                    html(
                        ": Sum of test cases where the model won based on majority "
                        "of metrics. "
                    )
                    html("Aggregated by summing across all model pairs.")
                with html.li():
                    html.strong(_t="Test Case Ranks")
                    html(": Sum of test case ranking scores across all comparisons. ")
                    html("Higher rank indicates better performance.")
                with html.li():
                    html.strong(_t="Metrics Wins")
                    html(
                        ": Sum of individual metric wins across all test cases "
                        "and models. "
                    )
                    html("Aggregated by summing wins for each metric.")
                with html.li():
                    html.strong(_t="Metrics Ranks")
                    html(": Sum of metric ranking scores across all comparisons. ")
                    html("Higher rank indicates better metric performance.")
                with html.li():
                    html.strong(_t="Delta")
                    html(": Shown in parentheses (e.g., +5 or -3). ")
                    html(
                        "Green indicates improvement for Current, red indicates "
                        "decline."
                    )

    def _html_comparison_cell(
        self, html, baseline_val, current_val, is_float: bool = False
    ):
        """Helper to render comparison cell with winner highlighting and delta.

        Parameters
        ----------
        html : htmltree.HtmlElement
            HTML element to append to.
        baseline_val : int | float
            Baseline value.
        current_val : int | float
            Current value.
        is_float : bool
            Whether values are floats (for formatting).

        """
        winner = (
            "baseline"
            if baseline_val > current_val
            else "current"
            if current_val > baseline_val
            else "tie"
        )

        # baseline column
        with html.td(style="padding: 10px; text-align: center;"):
            if winner == "baseline":
                html.strong(
                    _t=f"{baseline_val:.2f}" if is_float else str(baseline_val),
                    style=f"color: {COLOR_GREEN}; font-size: 1.1em;",
                )
            else:
                html(f"{baseline_val:.2f}" if is_float else str(baseline_val))

        # current column with delta
        with html.td(style="padding: 10px; text-align: center;"):
            if winner == "current":
                html.strong(
                    _t=f"{current_val:.2f}" if is_float else str(current_val),
                    style=f"color: {COLOR_GREEN}; font-size: 1.1em;",
                )
            else:
                html(f"{current_val:.2f}" if is_float else str(current_val))

            # add delta showing difference (current - baseline)
            if baseline_val != 0 or current_val != 0:
                delta = current_val - baseline_val
                if delta != 0:
                    html(" ")
                    delta_sign = "+" if delta > 0 else ""
                    # positive delta (current better) is green, negative
                    # (current worse) is red
                    delta_color = COLOR_GREEN if delta > 0 else COLOR_RED
                    html.strong(
                        _t=f"({delta_sign}{delta:.2f})"
                        if is_float
                        else f"({delta_sign}{delta})",
                        style=f"color: {delta_color}; font-size: 0.9em;",
                    )

    def _html_overall_technical_metrics(self, html, tech_metrics: dict):
        """Render overall technical metrics section (aggregated).

        Parameters
        ----------
        html : htmltree.HtmlElement
            HTML element to append to.
        tech_metrics : dict
            Technical metrics data containing baseline and current stats.

        """
        with html.div(
            klass="overall-technical-metrics",
            style=(
                f"background-color: {COLOR_LIGHT_GRAY}; "
                f"border-left: 4px solid {COLOR_YELLOW}; "
                "padding: 20px; margin: 20px 0; border-radius: 5px;"
            ),
            id="overall-technical-metrics",
        ):
            html.h3(_t="Technical Performance Metrics (Aggregated)")
            html.p(
                _t=(
                    "Costs and durations summed across all models. Duration min/max "
                    "show the fastest/slowest requests across all models. "
                    "Note: Duration averages are NOT shown to avoid 'average of "
                    "averages' error."
                )
            )

            baseline = tech_metrics[_explanations_cmp.KEY_BASELINE]
            current = tech_metrics[_explanations_cmp.KEY_CURRENT]

            with html.table(
                klass="model-overview-table",
                style="width: 100%; border-collapse: collapse; margin-top: 15px;",
            ):
                # header
                with html.tr():
                    html.th(
                        _t="Metric",
                        style=f"background-color: {COLOR_HEADER_GRAY}; padding: 10px;",
                    )
                    html.th(
                        _t="Baseline",
                        style=f"background-color: {COLOR_HEADER_GRAY}; "
                        f"padding: 10px; text-align: center;",
                    )
                    html.th(
                        _t="Current",
                        style=f"background-color: {COLOR_HEADER_GRAY}; "
                        f"padding: 10px; text-align: center;",
                    )

                # total cost (SUM) - lower is better
                with html.tr():
                    html.td(_t="Total Cost", style="padding: 10px; font-weight: bold;")
                    baseline_cost = baseline.get("cost_sum", 0.0)
                    current_cost = current.get("cost_sum", 0.0)
                    # baseline column - only highlight if both non-zero and baseline
                    # is better
                    with html.td(style="padding: 10px; text-align: center;"):
                        if (
                            baseline_cost > 0
                            and current_cost > 0
                            and baseline_cost < current_cost
                        ):
                            html.strong(
                                _t=f"${baseline_cost:.4f}",
                                style=f"color: {COLOR_GREEN};",
                            )
                        else:
                            html(f"${baseline_cost:.4f}")
                    # current column with delta - only highlight if both non-zero
                    # and current is better
                    with html.td(style="padding: 10px; text-align: center;"):
                        if (
                            current_cost > 0
                            and baseline_cost > 0
                            and current_cost < baseline_cost
                        ):
                            html.strong(
                                _t=f"${current_cost:.4f}",
                                style=f"color: {COLOR_GREEN};",
                            )
                        else:
                            html(f"${current_cost:.4f}")
                        # add delta (negative is better for cost) - show even if
                        # one is 0
                        if baseline_cost >= 0 and current_cost >= 0:
                            delta = current_cost - baseline_cost
                            if delta != 0:
                                html(" ")
                                delta_sign = "+" if delta > 0 else ""
                                # negative delta (cost reduction) is green, positive
                                # (more expensive) is red
                                delta_color = COLOR_RED if delta > 0 else COLOR_GREEN
                                html.strong(
                                    _t=f"({delta_sign}${delta:.4f})",
                                    style=f"color: {delta_color}; font-size: 0.9em;",
                                )

                # total duration (SUM) - lower is better
                with html.tr():
                    html.td(
                        _t="Total Duration (s)",
                        style="padding: 10px; font-weight: bold;",
                    )
                    baseline_duration = baseline.get("duration_sum", 0.0)
                    current_duration = current.get("duration_sum", 0.0)
                    # baseline column - only highlight if both non-zero and baseline
                    # is better
                    with html.td(style="padding: 10px; text-align: center;"):
                        if (
                            baseline_duration > 0
                            and current_duration > 0
                            and baseline_duration < current_duration
                        ):
                            html.strong(
                                _t=f"{baseline_duration:.2f}",
                                style=f"color: {COLOR_GREEN};",
                            )
                        else:
                            html(f"{baseline_duration:.2f}")
                    # current column with delta - only highlight if both non-zero and
                    # current is better
                    with html.td(style="padding: 10px; text-align: center;"):
                        if (
                            current_duration > 0
                            and baseline_duration > 0
                            and current_duration < baseline_duration
                        ):
                            html.strong(
                                _t=f"{current_duration:.2f}",
                                style=f"color: {COLOR_GREEN};",
                            )
                        else:
                            html(f"{current_duration:.2f}")
                        # add delta (negative is better for duration)
                        if baseline_duration > 0 and current_duration > 0:
                            delta = current_duration - baseline_duration
                            if delta != 0:
                                html(" ")
                                delta_sign = "+" if delta > 0 else ""
                                # negative delta (faster) is green
                                delta_color = COLOR_RED if delta > 0 else COLOR_GREEN
                                html.strong(
                                    _t=f"({delta_sign}{delta:.2f}s)",
                                    style=f"color: {delta_color}; font-size: 0.9em;",
                                )

                # min duration (MIN of MINs) - lower is better
                with html.tr():
                    html.td(
                        _t="Min Duration (s)", style="padding: 10px; font-weight: bold;"
                    )
                    baseline_min = baseline.get("duration_min", 0.0)
                    current_min = current.get("duration_min", 0.0)
                    # baseline column - only highlight if both valid and baseline
                    # is better
                    with html.td(style="padding: 10px; text-align: center;"):
                        if (
                            baseline_min > 0
                            and baseline_min != float("inf")
                            and current_min > 0
                            and current_min != float("inf")
                            and baseline_min < current_min
                        ):
                            html.strong(
                                _t=f"{baseline_min:.2f}", style=f"color: {COLOR_GREEN};"
                            )
                        else:
                            html(
                                f"{baseline_min:.2f}"
                                if baseline_min != float("inf")
                                else "N/A"
                            )
                    # current column with delta - only highlight if both valid and
                    # current is better
                    with html.td(style="padding: 10px; text-align: center;"):
                        if (
                            current_min > 0
                            and current_min != float("inf")
                            and baseline_min > 0
                            and baseline_min != float("inf")
                            and current_min < baseline_min
                        ):
                            html.strong(
                                _t=f"{current_min:.2f}", style=f"color: {COLOR_GREEN};"
                            )
                        else:
                            html(
                                f"{current_min:.2f}"
                                if current_min != float("inf")
                                else "N/A"
                            )
                        # add delta (negative is better for duration)
                        if (
                            baseline_min > 0
                            and baseline_min != float("inf")
                            and current_min > 0
                            and current_min != float("inf")
                        ):
                            delta = current_min - baseline_min
                            if delta != 0:
                                html(" ")
                                delta_sign = "+" if delta > 0 else ""
                                # negative delta (faster) is green
                                delta_color = COLOR_RED if delta > 0 else COLOR_GREEN
                                html.strong(
                                    _t=f"({delta_sign}{delta:.2f}s)",
                                    style=f"color: {delta_color}; font-size: 0.9em;",
                                )

                # max duration (MAX of MAXs) - lower is better
                with html.tr():
                    html.td(
                        _t="Max Duration (s)", style="padding: 10px; font-weight: bold;"
                    )
                    baseline_max = baseline.get("duration_max", 0.0)
                    current_max = current.get("duration_max", 0.0)
                    # baseline column - only highlight if both non-zero and baseline
                    # is better
                    with html.td(style="padding: 10px; text-align: center;"):
                        if (
                            baseline_max > 0
                            and current_max > 0
                            and baseline_max < current_max
                        ):
                            html.strong(
                                _t=f"{baseline_max:.2f}", style=f"color: {COLOR_GREEN};"
                            )
                        else:
                            html(f"{baseline_max:.2f}")
                    # current column with delta - only highlight if both non-zero and
                    # current is better
                    with html.td(style="padding: 10px; text-align: center;"):
                        if (
                            current_max > 0
                            and baseline_max > 0
                            and current_max < baseline_max
                        ):
                            html.strong(
                                _t=f"{current_max:.2f}", style=f"color: {COLOR_GREEN};"
                            )
                        else:
                            html(f"{current_max:.2f}")
                        # add delta (negative is better for duration)
                        if baseline_max > 0 and current_max > 0:
                            delta = current_max - baseline_max
                            if delta != 0:
                                html(" ")
                                delta_sign = "+" if delta > 0 else ""
                                # negative delta (faster) is green
                                delta_color = COLOR_RED if delta > 0 else COLOR_GREEN
                                html.strong(
                                    _t=f"({delta_sign}{delta:.2f}s)",
                                    style=f"color: {delta_color}; font-size: 0.9em;",
                                )

                # total API calls (SUM) - fewer is better
                with html.tr():
                    html.td(
                        _t="Total API Calls", style="padding: 10px; font-weight: bold;"
                    )
                    baseline_calls = baseline.get("call_count", 0)
                    current_calls = current.get("call_count", 0)
                    # baseline column - only highlight if both non-zero and baseline
                    # is better
                    with html.td(style="padding: 10px; text-align: center;"):
                        if (
                            baseline_calls > 0
                            and current_calls > 0
                            and baseline_calls < current_calls
                        ):
                            html.strong(
                                _t=str(baseline_calls), style=f"color: {COLOR_GREEN};"
                            )
                        else:
                            html(str(baseline_calls))
                    # current column with delta - only highlight if both non-zero and
                    # current is better
                    with html.td(style="padding: 10px; text-align: center;"):
                        if (
                            current_calls > 0
                            and baseline_calls > 0
                            and current_calls < baseline_calls
                        ):
                            html.strong(
                                _t=str(current_calls), style=f"color: {COLOR_GREEN};"
                            )
                        else:
                            html(str(current_calls))
                        # add delta (negative is better - fewer API calls)
                        if baseline_calls > 0 or current_calls > 0:
                            delta = current_calls - baseline_calls
                            if delta != 0:
                                html(" ")
                                delta_sign = "+" if delta > 0 else ""
                                # negative delta (fewer calls) is green, positive
                                # (more calls) is red
                                delta_color = COLOR_RED if delta > 0 else COLOR_GREEN
                                html.strong(
                                    _t=f"({delta_sign}{delta})",
                                    style=f"color: {delta_color}; font-size: 0.9em;",
                                )

                # success/failure counts (SUM)
                with html.tr():
                    html.td(
                        _t="Successful Requests",
                        style="padding: 10px; font-weight: bold;",
                    )
                    baseline_success = baseline.get("success_count", 0)
                    current_success = current.get("success_count", 0)
                    # baseline column - only highlight if both non-zero and baseline
                    # is better
                    with html.td(style="padding: 10px; text-align: center;"):
                        if (
                            baseline_success > 0
                            and current_success > 0
                            and baseline_success > current_success
                        ):
                            html.strong(
                                _t=str(baseline_success), style=f"color: {COLOR_GREEN};"
                            )
                        else:
                            html(str(baseline_success))
                    # current column with delta - only highlight if both non-zero and
                    # current is better
                    with html.td(style="padding: 10px; text-align: center;"):
                        if (
                            current_success > 0
                            and baseline_success > 0
                            and current_success > baseline_success
                        ):
                            html.strong(
                                _t=str(current_success), style=f"color: {COLOR_GREEN};"
                            )
                        else:
                            html(str(current_success))
                        # add delta (positive is better for successful requests)
                        if baseline_success > 0 or current_success > 0:
                            delta = current_success - baseline_success
                            if delta != 0:
                                html(" ")
                                delta_sign = "+" if delta > 0 else ""
                                # positive delta (more success) is green
                                delta_color = COLOR_GREEN if delta > 0 else COLOR_RED
                                html.strong(
                                    _t=f"({delta_sign}{delta})",
                                    style=f"color: {delta_color}; font-size: 0.9em;",
                                )

                with html.tr():
                    html.td(
                        _t="Failed Requests", style="padding: 10px; font-weight: bold;"
                    )
                    baseline_failures = baseline.get("failure_count", 0)
                    current_failures = current.get("failure_count", 0)
                    # baseline column - only highlight if both non-zero and baseline
                    # is better (fewer failures)
                    with html.td(style="padding: 10px; text-align: center;"):
                        if (
                            baseline_failures >= 0
                            and current_failures > 0
                            and (
                                baseline_failures == 0
                                or baseline_failures < current_failures
                            )
                        ):
                            html.strong(
                                _t=str(baseline_failures),
                                style=f"color: {COLOR_GREEN};",
                            )
                        else:
                            html(str(baseline_failures))
                    # current column with delta - only highlight if both have data and
                    # current is better (fewer failures)
                    with html.td(style="padding: 10px; text-align: center;"):
                        if (
                            current_failures >= 0
                            and baseline_failures > 0
                            and (
                                current_failures == 0
                                or current_failures < baseline_failures
                            )
                        ):
                            html.strong(
                                _t=str(current_failures), style=f"color: {COLOR_GREEN};"
                            )
                        else:
                            html(str(current_failures))
                        # add delta (negative is better for failures)
                        if baseline_failures > 0 or current_failures > 0:
                            delta = current_failures - baseline_failures
                            if delta != 0:
                                html(" ")
                                delta_sign = "+" if delta > 0 else ""
                                # negative delta (fewer failures) is green
                                delta_color = COLOR_RED if delta > 0 else COLOR_GREEN
                                html.strong(
                                    _t=f"({delta_sign}{delta})",
                                    style=f"color: {delta_color}; font-size: 0.9em;",
                                )

                # total tokens (SUM) - fewer is better (lower cost)
                with html.tr():
                    html.td(
                        _t="Total Input Tokens",
                        style="padding: 10px; font-weight: bold;",
                    )
                    baseline_input_tokens = baseline.get("input_tokens", 0)
                    current_input_tokens = current.get("input_tokens", 0)
                    # baseline column - only highlight if both non-zero and baseline
                    # is better
                    with html.td(style="padding: 10px; text-align: center;"):
                        if (
                            baseline_input_tokens > 0
                            and current_input_tokens > 0
                            and baseline_input_tokens < current_input_tokens
                        ):
                            html.strong(
                                _t=str(baseline_input_tokens),
                                style=f"color: {COLOR_GREEN};",
                            )
                        else:
                            html(str(baseline_input_tokens))
                    # current column with delta - only highlight if both non-zero and
                    # current is better
                    with html.td(style="padding: 10px; text-align: center;"):
                        if (
                            current_input_tokens > 0
                            and baseline_input_tokens > 0
                            and current_input_tokens < baseline_input_tokens
                        ):
                            html.strong(
                                _t=str(current_input_tokens),
                                style=f"color: {COLOR_GREEN};",
                            )
                        else:
                            html(str(current_input_tokens))
                        # add delta (negative is better - fewer tokens = lower cost)
                        if baseline_input_tokens > 0 or current_input_tokens > 0:
                            delta = current_input_tokens - baseline_input_tokens
                            if delta != 0:
                                html(" ")
                                delta_sign = "+" if delta > 0 else ""
                                # negative delta (fewer tokens) is green
                                delta_color = COLOR_RED if delta > 0 else COLOR_GREEN
                                html.strong(
                                    _t=f"({delta_sign}{delta})",
                                    style=f"color: {delta_color}; font-size: 0.9em;",
                                )

                with html.tr():
                    html.td(
                        _t="Total Output Tokens",
                        style="padding: 10px; font-weight: bold;",
                    )
                    baseline_output_tokens = baseline.get("output_tokens", 0)
                    current_output_tokens = current.get("output_tokens", 0)
                    # baseline column
                    with html.td(style="padding: 10px; text-align: center;"):
                        if (
                            baseline_output_tokens > 0
                            and current_output_tokens > 0
                            and baseline_output_tokens < current_output_tokens
                        ):
                            html.strong(
                                _t=str(baseline_output_tokens),
                                style=f"color: {COLOR_GREEN};",
                            )
                        else:
                            html(str(baseline_output_tokens))
                    # current column with delta - only highlight if both non-zero and
                    # current is better
                    with html.td(style="padding: 10px; text-align: center;"):
                        if (
                            current_output_tokens > 0
                            and baseline_output_tokens > 0
                            and current_output_tokens < baseline_output_tokens
                        ):
                            html.strong(
                                _t=str(current_output_tokens),
                                style=f"color: {COLOR_GREEN};",
                            )
                        else:
                            html(str(current_output_tokens))
                        # add delta (negative is better - fewer tokens = lower cost)
                        if baseline_output_tokens > 0 or current_output_tokens > 0:
                            delta = current_output_tokens - baseline_output_tokens
                            if delta != 0:
                                html(" ")
                                delta_sign = "+" if delta > 0 else ""
                                # negative delta (fewer tokens) is green
                                delta_color = COLOR_RED if delta > 0 else COLOR_GREEN
                                html.strong(
                                    _t=f"({delta_sign}{delta})",
                                    style=f"color: {delta_color}; font-size: 0.9em;",
                                )

            # legend explaining technical metrics aggregation
            html.p(
                _t="Legend:",
                style="margin-top: 20px; margin-bottom: 10px; font-weight: bold;",
            )
            with html.ul(
                style="margin-left: 20px; margin-bottom: 30px; line-height: 1.8;"
            ):
                with html.li():
                    html.strong(_t="Total Cost")
                    html(
                        ": Sum of all API call costs across all models. Lower "
                        "is better."
                    )
                with html.li():
                    html.strong(_t="Total Duration")
                    html(
                        ": Sum of all request durations across all models. Lower "
                        "is better."
                    )
                with html.li():
                    html.strong(_t="Min/Max Duration")
                    html(": Minimum of all minimum durations (fastest request) and ")
                    html(
                        "maximum of all maximum durations (slowest request) across "
                        "all models."
                    )
                with html.li():
                    html.strong(_t="Total API Calls")
                    html(": Sum of all API calls made across all models.")
                with html.li():
                    html.strong(_t="Success/Failed Requests")
                    html(
                        ": Sum of successful and failed API requests. Higher success "
                        "is better."
                    )
                with html.li():
                    html.strong(_t="Tokens")
                    html(": Sum of input and output tokens across all models.")
                with html.li():
                    html.strong(_t="Aggregation Method")
                    html(
                        ": All values are SUMMED across models, except Min "
                        "(takes minimum) "
                    )
                    html(
                        "and Max (takes maximum). Duration averages are NOT shown to "
                        "avoid "
                    )
                    html("'average of averages' statistical error.")

    def _get_overall_comparison_data(self) -> dict | None:
        """Extract overall comparison data from diff object.

        Returns
        -------
        dict | None :
            Overall comparison data dict, or None if not available.

        """
        # check if overall comparison exists in diff_obj's to_dict() output
        # we need to call to_dict() and extract the overall_comparison field
        try:
            diff_dict = self.diff_obj.to_dict()
            return diff_dict.get(_explanations_cmp.KEY_OVERALL_COMPARISON)
        except Exception:
            return None

    @staticmethod
    def _html_leaderboard(
        html,
        leaderboard: list[dict],
        diff_to_idx: dict[_explanations_cmp.EvalResultDiff, int],
        metrics_meta: dict | None = None,
        model_key: str = "",
    ):
        """Generate HTML leaderboard table.

        Parameters
        ----------
        html : airium.Airium
            Airium HTML instance.
        leaderboard : list[dict]
            Leaderboard entries with wins, question, changed_metrics_count, etc.
        diff_to_idx : dict[_explanations_cmp.EvalResultDiff, int]
            Mapping from diff objects to their test case index (for links).
        metrics_meta : dict | None
            Metrics metadata containing higher_is_better information.
        model_key : str
            Model key for unique ID generation.

        """
        if not leaderboard:
            return

        # use pre-calculated wins and ranks from leaderboard entries
        # (calculated in _generate_leaderboard method)
        # entries are already sorted by rank difference
        enriched_leaderboard = []
        for entry in leaderboard:
            enriched_leaderboard.append(
                {
                    _explanations_cmp.KEY_QUESTION: entry[
                        _explanations_cmp.KEY_QUESTION
                    ],
                    _explanations_cmp.KEY_CHANGED_METRICS_COUNT: entry[
                        _explanations_cmp.KEY_CHANGED_METRICS_COUNT
                    ],
                    _explanations_cmp.KEY_BASELINE_WINS: entry[
                        _explanations_cmp.KEY_BASELINE_WINS
                    ],
                    _explanations_cmp.KEY_CURRENT_WINS: entry[
                        _explanations_cmp.KEY_CURRENT_WINS
                    ],
                    _explanations_cmp.KEY_BASELINE_RANK_AVG: entry[
                        _explanations_cmp.KEY_BASELINE_RANK_AVG
                    ],
                    _explanations_cmp.KEY_CURRENT_RANK_AVG: entry[
                        _explanations_cmp.KEY_CURRENT_RANK_AVG
                    ],
                    _explanations_cmp.KEY_DIFF: entry[_explanations_cmp.KEY_DIFF],
                }
            )

        section_id = f"top-test-cases-{model_key}" if model_key else ""
        with html.div(
            klass="leaderboard-section",
            id=section_id if section_id else None,
        ):
            html.h3(_t="Top Test Cases by Metric Changes")
            with html.p(style="margin-bottom: 15px;"):
                html("Test cases sorted by the ")
                html.b(
                    _t="highest absolute difference between Baseline Ranks "
                    "and Current Ranks"
                )
                html(
                    ". Test cases where the rank-based comparison shows "
                    "the most significant difference appear first."
                )

            # split leaderboard into visible (first 10) and collapsible (rest)
            visible_entries = enriched_leaderboard[:10]
            collapsible_entries = enriched_leaderboard[10:]

            # render first 10 rows in main table
            with html.table(klass="leaderboard-table"):
                # header
                with html.tr():
                    html.th(_t="Rank", style="width: 6%; text-align: center;")
                    html.th(_t="Question", style="width: 40%; text-align: center;")
                    html.th(
                        _t="Baseline Ranks", style="width: 12%; text-align: center;"
                    )
                    html.th(_t="Current Ranks", style="width: 12%; text-align: center;")
                    html.th(_t="Baseline Wins", style="width: 11%; text-align: center;")
                    html.th(_t="Current Wins", style="width: 11%; text-align: center;")
                    html.th(
                        _t="Changed Metrics", style="width: 8%; text-align: center;"
                    )

                # data - first 10 rows
                for position, entry in enumerate(visible_entries, start=1):
                    diff = entry[_explanations_cmp.KEY_DIFF]
                    idx = diff_to_idx.get(diff)
                    changed_count = entry[_explanations_cmp.KEY_CHANGED_METRICS_COUNT]
                    baseline_wins = entry[_explanations_cmp.KEY_BASELINE_WINS]
                    current_wins = entry[_explanations_cmp.KEY_CURRENT_WINS]
                    baseline_rank_avg = entry[_explanations_cmp.KEY_BASELINE_RANK_AVG]
                    current_rank_avg = entry[_explanations_cmp.KEY_CURRENT_RANK_AVG]

                    with html.tr():
                        with html.td(style="text-align: center;"):
                            html.strong(_t=f"#{position}")

                        with html.td():
                            if idx:
                                with html.a(href=f"#test-case-{idx}"):
                                    html(entry[_explanations_cmp.KEY_QUESTION])
                            else:
                                html(entry[_explanations_cmp.KEY_QUESTION])

                        # baseline ranks column
                        with html.td(style="text-align: center;"):
                            if baseline_rank_avg > 0 and (
                                current_rank_avg == 0
                                or baseline_rank_avg > current_rank_avg
                            ):
                                html.strong(
                                    _t=f"{baseline_rank_avg:.2f}",
                                    style=f"color: {COLOR_GREEN};",
                                )
                            elif baseline_rank_avg > 0:
                                html(f"{baseline_rank_avg:.2f}")
                            else:
                                html("-")

                        # current ranks column
                        with html.td(style="text-align: center;"):
                            if current_rank_avg > 0 and (
                                baseline_rank_avg == 0
                                or current_rank_avg > baseline_rank_avg
                            ):
                                html.strong(
                                    _t=f"{current_rank_avg:.2f}",
                                    style=f"color: {COLOR_GREEN};",
                                )
                            elif current_rank_avg > 0:
                                html(f"{current_rank_avg:.2f}")
                            else:
                                html("-")

                            # add delta for current ranks
                            if current_rank_avg > 0 and baseline_rank_avg > 0:
                                rank_delta = current_rank_avg - baseline_rank_avg
                                if rank_delta != 0:
                                    html(" ")
                                    delta_sign = "+" if rank_delta > 0 else ""
                                    # higher rank is better, positive delta is green
                                    delta_color = (
                                        COLOR_GREEN if rank_delta > 0 else COLOR_RED
                                    )
                                    html.strong(
                                        _t=f"({delta_sign}{rank_delta:.2f})",
                                        style=f"color: {delta_color};",
                                    )

                        with html.td(style="text-align: center;"):
                            if baseline_wins > current_wins:
                                html.strong(
                                    _t=str(baseline_wins),
                                    style=f"color: {COLOR_GREEN};",
                                )
                            elif baseline_wins < current_wins:
                                html.span(
                                    _t=str(baseline_wins), style=f"color: {COLOR_RED};"
                                )
                            else:
                                html(str(baseline_wins))

                        with html.td(style="text-align: center;"):
                            if current_wins > baseline_wins:
                                html.strong(
                                    _t=str(current_wins), style=f"color: {COLOR_GREEN};"
                                )
                            elif current_wins < baseline_wins:
                                html.span(
                                    _t=str(current_wins), style=f"color: {COLOR_RED};"
                                )
                            else:
                                html(str(current_wins))

                            # add delta for current wins
                            wins_delta = current_wins - baseline_wins
                            if wins_delta != 0:
                                html(" ")
                                delta_sign = "+" if wins_delta > 0 else ""
                                # higher is better, positive delta is green
                                delta_color = (
                                    COLOR_GREEN if wins_delta > 0 else COLOR_RED
                                )
                                html.strong(
                                    _t=f"({delta_sign}{wins_delta})",
                                    style=f"color: {delta_color};",
                                )

                        with html.td(style="text-align: center;"):
                            html(str(changed_count))

            # render collapsible section for rows 11+ if present
            if collapsible_entries:
                with html.details(style="margin-top: 10px;"):
                    with html.summary(
                        style=(
                            "cursor: pointer; padding: 10px; "
                            f"background-color: {COLOR_OFF_WHITE}; "
                            f"border: 1px solid {COLOR_BORDER_GRAY}; "
                            "border-radius: 4px; font-weight: bold;"
                        )
                    ):
                        case_word = "s" if len(collapsible_entries) != 1 else ""
                        html(
                            f"Show {len(collapsible_entries)} more "
                            f"test case{case_word}..."
                        )

                    with html.table(
                        klass="leaderboard-table", style="margin-top: 10px;"
                    ):
                        # header for collapsible table
                        with html.tr():
                            html.th(_t="Rank", style="width: 6%; text-align: center;")
                            html.th(
                                _t="Question", style="width: 40%; text-align: center;"
                            )
                            html.th(
                                _t="Baseline Ranks",
                                style="width: 12%; text-align: center;",
                            )
                            html.th(
                                _t="Current Ranks",
                                style="width: 12%; text-align: center;",
                            )
                            html.th(
                                _t="Baseline Wins",
                                style="width: 11%; text-align: center;",
                            )
                            html.th(
                                _t="Current Wins",
                                style="width: 11%; text-align: center;",
                            )
                            html.th(
                                _t="Changed Metrics",
                                style="width: 8%; text-align: center;",
                            )

                        # data - rows 11+
                        for position, entry in enumerate(collapsible_entries, start=11):
                            diff = entry[_explanations_cmp.KEY_DIFF]
                            idx = diff_to_idx.get(diff)
                            changed_count = entry[
                                _explanations_cmp.KEY_CHANGED_METRICS_COUNT
                            ]
                            baseline_wins = entry[_explanations_cmp.KEY_BASELINE_WINS]
                            current_wins = entry[_explanations_cmp.KEY_CURRENT_WINS]
                            baseline_rank_avg = entry[
                                _explanations_cmp.KEY_BASELINE_RANK_AVG
                            ]
                            current_rank_avg = entry[
                                _explanations_cmp.KEY_CURRENT_RANK_AVG
                            ]

                            with html.tr():
                                with html.td(style="text-align: center;"):
                                    html.strong(_t=f"#{position}")

                                with html.td():
                                    if idx:
                                        with html.a(href=f"#test-case-{idx}"):
                                            html(entry[_explanations_cmp.KEY_QUESTION])
                                    else:
                                        html(entry[_explanations_cmp.KEY_QUESTION])

                                # baseline ranks column
                                with html.td(style="text-align: center;"):
                                    if baseline_rank_avg > 0 and (
                                        current_rank_avg == 0
                                        or baseline_rank_avg > current_rank_avg
                                    ):
                                        html.strong(
                                            _t=f"{baseline_rank_avg:.2f}",
                                            style=f"color: {COLOR_GREEN};",
                                        )
                                    elif baseline_rank_avg > 0:
                                        html(f"{baseline_rank_avg:.2f}")
                                    else:
                                        html("-")

                                # current ranks column
                                with html.td(style="text-align: center;"):
                                    if current_rank_avg > 0 and (
                                        baseline_rank_avg == 0
                                        or current_rank_avg > baseline_rank_avg
                                    ):
                                        html.strong(
                                            _t=f"{current_rank_avg:.2f}",
                                            style=f"color: {COLOR_GREEN};",
                                        )
                                    elif current_rank_avg > 0:
                                        html(f"{current_rank_avg:.2f}")
                                    else:
                                        html("-")

                                    # add delta for current ranks
                                    if current_rank_avg > 0 and baseline_rank_avg > 0:
                                        rank_delta = (
                                            current_rank_avg - baseline_rank_avg
                                        )
                                        if rank_delta != 0:
                                            html(" ")
                                            delta_sign = "+" if rank_delta > 0 else ""
                                            # higher rank is better,
                                            # positive delta is green
                                            delta_color = (
                                                COLOR_GREEN
                                                if rank_delta > 0
                                                else COLOR_RED
                                            )
                                            html.strong(
                                                _t=f"({delta_sign}{rank_delta:.2f})",
                                                style=f"color: {delta_color};",
                                            )

                                with html.td(style="text-align: center;"):
                                    if baseline_wins > current_wins:
                                        html.strong(
                                            _t=str(baseline_wins),
                                            style=f"color: {COLOR_GREEN};",
                                        )
                                    elif baseline_wins < current_wins:
                                        html.span(
                                            _t=str(baseline_wins),
                                            style=f"color: {COLOR_RED};",
                                        )
                                    else:
                                        html(str(baseline_wins))

                                with html.td(style="text-align: center;"):
                                    if current_wins > baseline_wins:
                                        html.strong(
                                            _t=str(current_wins),
                                            style=f"color: {COLOR_GREEN};",
                                        )
                                    elif current_wins < baseline_wins:
                                        html.span(
                                            _t=str(current_wins),
                                            style=f"color: {COLOR_RED};",
                                        )
                                    else:
                                        html(str(current_wins))

                                    # add delta for current wins
                                    wins_delta = current_wins - baseline_wins
                                    if wins_delta != 0:
                                        html(" ")
                                        delta_sign = "+" if wins_delta > 0 else ""
                                        # higher is better, positive delta is green
                                        delta_color = (
                                            COLOR_GREEN if wins_delta > 0 else COLOR_RED
                                        )
                                        html.strong(
                                            _t=f"({delta_sign}{wins_delta})",
                                            style=f"color: {delta_color};",
                                        )

                                with html.td(style="text-align: center;"):
                                    html(str(changed_count))

            # LEGEND
            html.h4(_t="Legend:", style="margin-top: 20px; margin-bottom: 10px;")
            with html.ul(style="margin-left: 20px; line-height: 1.8;"):
                with html.li():
                    html.strong(_t="Rank")
                    html(
                        ": Position in the leaderboard "
                        "(1 = highest absolute difference between Baseline Ranks "
                        "and Current Ranks)."
                    )
                with html.li():
                    html.strong(_t="Question")
                    html(": The test case question. Click to jump to ")
                    html("the detailed test case comparison.")
                with html.li():
                    html.strong(_t="Baseline Ranks")
                    html(
                        ": Average rank of the baseline model for this "
                        "specific test case. "
                        "For each metric in the test case, baseline and current "
                        "models are ranked (best=2, worst=1), and the average "
                        "rank is calculated. "
                    )
                    html.strong(_t="Higher rank indicates better performance.")
                with html.li():
                    html.strong(_t="Current Ranks")
                    html(
                        ": Average rank of the current model for this "
                        "specific test case. "
                        "For each metric in the test case, baseline and current "
                        "models are ranked (best=2, worst=1), and the average "
                        "rank is calculated. "
                    )
                    html.strong(_t="Higher rank indicates better performance.")
                with html.li():
                    html.strong(_t="Baseline Wins")
                    html(": Number of metrics where the baseline model ")
                    html("performed better. ")
                with html.li():
                    html.strong(_t="Current Wins")
                    html(": Number of metrics where the current model ")
                    html("performed better. ")
                with html.li():
                    html.strong(_t="Changed Metrics")
                    html(": Total number of metrics with different values ")
                    html("between baseline and current models.")

    @staticmethod
    def _calculate_model_cmp_stats(
        model_diffs: list,
        metrics_meta: dict | None = None,
        baseline_model=None,
        current_model=None,
    ) -> dict:
        """Calculate comparison statistics across all test cases for a model pair.

        Parameters
        ----------
        model_diffs : list
            List of (idx, diff) tuples for this model.
        metrics_meta : dict | None
            Metrics metadata.
        baseline_model : ExplainableLlmModel | None
            Baseline explainable model (for extracting llm_model_meta).
        current_model : ExplainableLlmModel | None
            Current explainable model (for extracting llm_model_meta).

        Returns
        -------
        dict :
            Dictionary containing comparison statistics:
            - flipped_metrics_count: total number of metric flips
            - metrics_averages: dict mapping metric names to
              {baseline_avg, current_avg, diff}
            - empty_context_count: dict with old and new empty context counts
            - flipped_to_passed: number of test cases that flipped from failed to passed
            - flipped_to_failed: number of test cases that flipped from passed to failed
            - technical_metrics: dict with per-model technical performance stats

        """
        stats = {
            KEY_FLIPPED_METRICS_COUNT_STATS: 0,
            KEY_METRICS_AVERAGES: {},
            KEY_EMPTY_CONTEXT_COUNT: {KEY_BASELINE: 0, KEY_CURRENT: 0},
            KEY_FLIPPED_TO_PASSED: 0,
            KEY_FLIPPED_TO_FAILED: 0,
            KEY_METRICS_WINS_BASELINE: 0,
            KEY_METRICS_WINS_CURRENT: 0,
            KEY_TEST_CASE_WINS_BASELINE: 0,
            KEY_TEST_CASE_WINS_CURRENT: 0,
            KEY_TEST_CASE_RANKS_BASELINE: 0,
            KEY_TEST_CASE_RANKS_CURRENT: 0,
            KEY_METRICS_RANKS_BASELINE: 0.0,
            KEY_METRICS_RANKS_CURRENT: 0.0,
            KEY_TECHNICAL_METRICS: {
                KEY_BASELINE: {
                    KEY_COST_SUM: 0.0,
                    KEY_DURATION_SUM: 0.0,
                    KEY_DURATION_MIN: float("inf"),
                    KEY_DURATION_MAX: 0.0,
                },
                KEY_CURRENT: {
                    KEY_COST_SUM: 0.0,
                    KEY_DURATION_SUM: 0.0,
                    KEY_DURATION_MIN: float("inf"),
                    KEY_DURATION_MAX: 0.0,
                },
            },
        }

        # collect all metrics values per metric name
        # {metric_name: {baseline: [values], current: [values],
        #               baseline_better_wins: int, current_better_wins: int}}
        metrics_data = {}

        for idx, diff in model_diffs:
            # count flipped metrics
            if diff.diff_flipped_metrics:
                stats[KEY_FLIPPED_METRICS_COUNT_STATS] += len(diff.diff_flipped_metrics)

                # analyze flips: passed->failed or failed->passed
                # get metrics metadata to understand thresholds and higher_is_better
                test_case_metrics_meta = diff.baseline_test_case.get(
                    _explanations_cmp.KEY_METRICS_META, None
                )
                # if not in baseline, try current
                if not test_case_metrics_meta:
                    test_case_metrics_meta = diff.current_test_case.get(
                        _explanations_cmp.KEY_METRICS_META, None
                    )

                # if still no metadata, use the passed-in metrics_meta
                if not test_case_metrics_meta:
                    test_case_metrics_meta = metrics_meta

                for metric_key, metric_info in diff.diff_flipped_metrics.items():
                    baseline_value = metric_info.get(
                        _explanations_cmp.KEY_BASELINE_VALUE
                    )
                    current_value = metric_info.get(_explanations_cmp.KEY_CURRENT_VALUE)

                    # check if values are numeric
                    if not (
                        isinstance(baseline_value, (int, float))
                        and isinstance(current_value, (int, float))
                    ):
                        continue

                    # try to get metric metadata
                    if (
                        not test_case_metrics_meta
                        or metric_key not in test_case_metrics_meta
                    ):
                        # if no metadata, skip this metric
                        continue

                    metric_meta = test_case_metrics_meta[metric_key]
                    higher_is_better = True
                    threshold = 0.5

                    if hasattr(metric_meta, "higher_is_better"):
                        higher_is_better = metric_meta.higher_is_better
                        threshold = metric_meta.threshold
                    elif isinstance(metric_meta, dict):
                        higher_is_better = metric_meta.get("higher_is_better", True)
                        threshold = metric_meta.get("threshold", 0.5)

                    # determine if baseline was passing
                    baseline_passed = (
                        baseline_value >= threshold
                        if higher_is_better
                        else baseline_value <= threshold
                    )

                    # determine if current is passing
                    current_passed = (
                        current_value >= threshold
                        if higher_is_better
                        else current_value <= threshold
                    )

                    # count flips
                    if not baseline_passed and current_passed:
                        stats[KEY_FLIPPED_TO_PASSED] += 1
                    elif baseline_passed and not current_passed:
                        stats[KEY_FLIPPED_TO_FAILED] += 1

            # extract metrics from test cases (excluding non-metric fields)
            skip_list = [
                _explanations_cmp.KEY_KEY,
                _explanations_cmp.KEY_ACTUAL_DURATION,
                _explanations_cmp.KEY_ACTUAL_OUTPUT,
                _explanations_cmp.KEY_ACTUAL_OUTPUT_META,
                _explanations_cmp.KEY_CATEGORIES,
                _explanations_cmp.KEY_CONTEXT,
                _explanations_cmp.KEY_CORPUS,
                _explanations_cmp.KEY_COST,
                _explanations_cmp.KEY_EXPECTED_OUTPUT,
                _explanations_cmp.KEY_INPUT,
                _explanations_cmp.KEY_METRICS_META,
                _explanations_cmp.KEY_MODEL_KEY,
                _explanations_cmp.KEY_OUTPUT_CONDITION,
                _explanations_cmp.KEY_OUTPUT_CONSTRAINTS,
                _explanations_cmp.KEY_RELATIONSHIPS,
                _explanations_cmp.KEY_RESULT_ERR_MSG,
                _explanations_cmp.KEY_TEST_CASE_KEY,
                _explanations_cmp.KEY_TEST_KEY,
            ]

            baseline_metrics = {
                k: v for k, v in diff.baseline_test_case.items() if k not in skip_list
            }
            current_metrics = {
                k: v for k, v in diff.current_test_case.items() if k not in skip_list
            }

            # collect numeric metrics for averaging
            for metric_name, metric_value in baseline_metrics.items():
                if isinstance(metric_value, (int, float)):
                    if metric_name not in metrics_data:
                        metrics_data[metric_name] = {
                            KEY_BASELINE: [],
                            KEY_CURRENT: [],
                            KEY_BASELINE_BETTER_WINS: 0,
                            KEY_CURRENT_BETTER_WINS: 0,
                        }
                    metrics_data[metric_name][KEY_BASELINE].append(metric_value)

            for metric_name, metric_value in current_metrics.items():
                if isinstance(metric_value, (int, float)):
                    if metric_name not in metrics_data:
                        metrics_data[metric_name] = {
                            KEY_BASELINE: [],
                            KEY_CURRENT: [],
                            KEY_BASELINE_BETTER_WINS: 0,
                            KEY_CURRENT_BETTER_WINS: 0,
                        }
                    metrics_data[metric_name][KEY_CURRENT].append(metric_value)

            # check for empty contexts
            baseline_context = diff.baseline_test_case.get(
                _explanations_cmp.KEY_CONTEXT, []
            )
            current_context = diff.current_test_case.get(
                _explanations_cmp.KEY_CONTEXT, []
            )

            if not baseline_context or len(baseline_context) == 0:
                stats[KEY_EMPTY_CONTEXT_COUNT][KEY_BASELINE] += 1
            if not current_context or len(current_context) == 0:
                stats[KEY_EMPTY_CONTEXT_COUNT][KEY_CURRENT] += 1

            # extract per-test-case technical metrics (cost and duration)
            baseline_cost = diff.baseline_test_case.get(_explanations_cmp.KEY_COST, 0.0)
            current_cost = diff.current_test_case.get(_explanations_cmp.KEY_COST, 0.0)
            baseline_duration = diff.baseline_test_case.get(
                _explanations_cmp.KEY_ACTUAL_DURATION, 0.0
            )
            current_duration = diff.current_test_case.get(
                _explanations_cmp.KEY_ACTUAL_DURATION, 0.0
            )

            # aggregate cost
            if isinstance(baseline_cost, (int, float)):
                stats[KEY_TECHNICAL_METRICS][KEY_BASELINE][KEY_COST_SUM] += (
                    baseline_cost
                )
            if isinstance(current_cost, (int, float)):
                stats[KEY_TECHNICAL_METRICS][KEY_CURRENT][KEY_COST_SUM] += current_cost

            # aggregate duration (sum, min, max)
            if isinstance(baseline_duration, (int, float)) and baseline_duration > 0:
                stats[KEY_TECHNICAL_METRICS][KEY_BASELINE][KEY_DURATION_SUM] += (
                    baseline_duration
                )
                stats[KEY_TECHNICAL_METRICS][KEY_BASELINE][KEY_DURATION_MIN] = min(
                    stats[KEY_TECHNICAL_METRICS][KEY_BASELINE][KEY_DURATION_MIN],
                    baseline_duration,
                )
                stats[KEY_TECHNICAL_METRICS][KEY_BASELINE][KEY_DURATION_MAX] = max(
                    stats[KEY_TECHNICAL_METRICS][KEY_BASELINE][KEY_DURATION_MAX],
                    baseline_duration,
                )

            if isinstance(current_duration, (int, float)) and current_duration > 0:
                stats[KEY_TECHNICAL_METRICS][KEY_CURRENT][KEY_DURATION_SUM] += (
                    current_duration
                )
                stats[KEY_TECHNICAL_METRICS][KEY_CURRENT][KEY_DURATION_MIN] = min(
                    stats[KEY_TECHNICAL_METRICS][KEY_CURRENT][KEY_DURATION_MIN],
                    current_duration,
                )
                stats[KEY_TECHNICAL_METRICS][KEY_CURRENT][KEY_DURATION_MAX] = max(
                    stats[KEY_TECHNICAL_METRICS][KEY_CURRENT][KEY_DURATION_MAX],
                    current_duration,
                )

            # calculate metrics wins: count how many metrics each model won
            # iterate over all metrics that exist in both baseline and current
            all_metric_keys = set(baseline_metrics.keys()) & set(current_metrics.keys())

            # track per-test-case metric wins for test case wins
            test_case_baseline_wins = 0
            test_case_current_wins = 0

            for metric_key in all_metric_keys:
                baseline_value = baseline_metrics.get(metric_key)
                current_value = current_metrics.get(metric_key)

                # skip if values are None, not numeric, or equal
                if (
                    baseline_value is None
                    or current_value is None
                    or not isinstance(baseline_value, (int, float))
                    or not isinstance(current_value, (int, float))
                    or baseline_value == current_value
                ):
                    continue

                # get metric metadata to determine directionality
                # MUST have metrics_meta
                if not metrics_meta:
                    raise ValueError(
                        f"Metrics metadata is required for wins-based comparison "
                        f"but is missing. Cannot determine winner for metric "
                        f"'{metric_key}' without knowing if higher or lower is better."
                    )

                if metric_key not in metrics_meta:
                    raise ValueError(
                        f"Metrics metadata is missing for metric '{metric_key}'. "
                        f"Cannot determine winner without knowing if higher or "
                        f"lower is better. Available metrics in metadata: "
                        f"{list(metrics_meta.keys())}"
                    )

                metric_meta = metrics_meta[metric_key]

                if hasattr(metric_meta, "higher_is_better"):
                    higher_is_better = metric_meta.higher_is_better
                elif isinstance(metric_meta, dict):
                    if "higher_is_better" not in metric_meta:
                        raise ValueError(
                            f"Metric metadata for '{metric_key}' is missing the "
                            f"required 'higher_is_better' field. Metadata: "
                            f"{metric_meta}"
                        )
                    higher_is_better = metric_meta["higher_is_better"]
                else:
                    raise ValueError(
                        f"Metric metadata for '{metric_key}' has unexpected type: "
                        f"{type(metric_meta)}. Expected object with 'higher_is_better' "
                        f"attribute or dict with 'higher_is_better' key."
                    )

                # compare and count for both metrics wins and test case wins
                # also update per-metric wins counters
                if higher_is_better:
                    if baseline_value > current_value:
                        stats[KEY_METRICS_WINS_BASELINE] += 1
                        test_case_baseline_wins += 1
                        # update per-metric wins
                        if metric_key in metrics_data:
                            metrics_data[metric_key][KEY_BASELINE_BETTER_WINS] += 1
                    elif current_value > baseline_value:
                        stats[KEY_METRICS_WINS_CURRENT] += 1
                        test_case_current_wins += 1
                        # update per-metric wins
                        if metric_key in metrics_data:
                            metrics_data[metric_key][KEY_CURRENT_BETTER_WINS] += 1
                else:  # lower is better
                    if baseline_value < current_value:
                        stats[KEY_METRICS_WINS_BASELINE] += 1
                        test_case_baseline_wins += 1
                        # update per-metric wins
                        if metric_key in metrics_data:
                            metrics_data[metric_key][KEY_BASELINE_BETTER_WINS] += 1
                    elif current_value < baseline_value:
                        stats[KEY_METRICS_WINS_CURRENT] += 1
                        test_case_current_wins += 1
                        # update per-metric wins
                        if metric_key in metrics_data:
                            metrics_data[metric_key][KEY_CURRENT_BETTER_WINS] += 1

            # determine test case winner based on majority of metrics
            # ties (equal wins) don't count for either model
            if test_case_baseline_wins > test_case_current_wins:
                stats[KEY_TEST_CASE_WINS_BASELINE] += 1
            elif test_case_current_wins > test_case_baseline_wins:
                stats[KEY_TEST_CASE_WINS_CURRENT] += 1
            # else: tie - neither model wins this test case

            # calculate test case ranks: rank-based winner for this test case
            # for each metric in this test case, rank baseline vs current
            test_case_baseline_ranks = []
            test_case_current_ranks = []

            for metric_key in all_metric_keys:
                baseline_value = baseline_metrics.get(metric_key)
                current_value = current_metrics.get(metric_key)

                # skip if values are None or not numeric
                if (
                    baseline_value is None
                    or current_value is None
                    or not isinstance(baseline_value, (int, float))
                    or not isinstance(current_value, (int, float))
                ):
                    continue

                # get metric metadata to determine sort order
                if not metrics_meta or metric_key not in metrics_meta:
                    raise ValueError(
                        f"Metrics metadata required for rank-based comparison "
                        f"but missing for metric '{metric_key}'"
                    )

                metric_meta = metrics_meta[metric_key]
                if hasattr(metric_meta, "higher_is_better"):
                    higher_is_better = metric_meta.higher_is_better
                elif isinstance(metric_meta, dict):
                    if "higher_is_better" not in metric_meta:
                        raise ValueError(
                            f"Metric '{metric_key}' missing 'higher_is_better'"
                        )
                    higher_is_better = metric_meta["higher_is_better"]
                else:
                    raise ValueError(
                        f"Unexpected type for metric '{metric_key}' metadata"
                    )

                # rank these two values
                # higher is better: sort descending, invert ranks
                # lower is better: sort ascending, invert ranks
                values_with_origin = [
                    (baseline_value, "baseline"),
                    (current_value, "current"),
                ]
                values_with_origin.sort(key=lambda x: x[0], reverse=higher_is_better)

                # assign ranks with proper tie handling
                ranks_by_label = _assign_ranks_with_ties(values_with_origin)

                # extract ranks for baseline and current
                if "baseline" in ranks_by_label:
                    test_case_baseline_ranks.extend(ranks_by_label["baseline"])
                if "current" in ranks_by_label:
                    test_case_current_ranks.extend(ranks_by_label["current"])

            # determine test case winner based on average ranks
            # higher average rank = better
            if test_case_baseline_ranks and test_case_current_ranks:
                baseline_avg_rank = sum(test_case_baseline_ranks) / len(
                    test_case_baseline_ranks
                )
                current_avg_rank = sum(test_case_current_ranks) / len(
                    test_case_current_ranks
                )

                if baseline_avg_rank > current_avg_rank:
                    stats[KEY_TEST_CASE_RANKS_BASELINE] += 1
                elif current_avg_rank > baseline_avg_rank:
                    stats[KEY_TEST_CASE_RANKS_CURRENT] += 1
                # else: tie - neither model wins this test case

        # calculate averages for each metric and include per-metric wins counts
        for metric_name, values in metrics_data.items():
            baseline_values = values[KEY_BASELINE]
            current_values = values[KEY_CURRENT]

            baseline_avg = (
                sum(baseline_values) / len(baseline_values) if baseline_values else 0.0
            )
            current_avg = (
                sum(current_values) / len(current_values) if current_values else 0.0
            )
            diff = current_avg - baseline_avg

            stats[KEY_METRICS_AVERAGES][metric_name] = {
                KEY_BASELINE_AVG: baseline_avg,
                KEY_CURRENT_AVG: current_avg,
                KEY_DIFF_VALUE: diff,
                KEY_BASELINE_BETTER_WINS: values[KEY_BASELINE_BETTER_WINS],
                KEY_CURRENT_BETTER_WINS: values[KEY_CURRENT_BETTER_WINS],
            }

        # calculate rank-based comparison for metrics
        # for each metric, merge baseline and current values, rank them,
        # and average ranks
        metric_ranks_baseline = []
        metric_ranks_current = []

        for metric_name, values in metrics_data.items():
            baseline_values = values[KEY_BASELINE]
            current_values = values[KEY_CURRENT]

            # skip if no values
            if not baseline_values and not current_values:
                continue

            # get metric metadata to determine sort order
            if not metrics_meta:
                raise ValueError(
                    f"Metrics metadata is required for rank-based comparison "
                    f"but is missing. Cannot determine ranking for metric "
                    f"'{metric_name}' without knowing if higher or lower is "
                    f"better."
                )

            if metric_name not in metrics_meta:
                raise ValueError(
                    f"Metrics metadata is missing for metric '{metric_name}'. "
                    f"Cannot determine ranking without knowing if higher or "
                    f"lower is better. Available metrics in metadata: "
                    f"{list(metrics_meta.keys())}"
                )

            metric_meta = metrics_meta[metric_name]
            if hasattr(metric_meta, "higher_is_better"):
                higher_is_better = metric_meta.higher_is_better
            elif isinstance(metric_meta, dict):
                if "higher_is_better" not in metric_meta:
                    raise ValueError(
                        f"Metric metadata for '{metric_name}' is missing the "
                        f"required 'higher_is_better' field. Metadata: "
                        f"{metric_meta}"
                    )
                higher_is_better = metric_meta["higher_is_better"]
            else:
                raise ValueError(
                    f"Metric metadata for '{metric_name}' has unexpected "
                    f"type: {type(metric_meta)}. Expected object with "
                    f"'higher_is_better' attribute or dict with "
                    f"'higher_is_better' key."
                )

            # merge all values from both models with their origins
            all_values = []
            for v in baseline_values:
                all_values.append((v, "baseline"))
            for v in current_values:
                all_values.append((v, "current"))

            # sort based on higher_is_better
            # we want higher rank = better, so best performance gets max rank
            # if higher is better, sort descending (largest value first)
            # if lower is better, sort ascending (smallest value first)
            all_values.sort(key=lambda x: x[0], reverse=higher_is_better)

            # assign inverted ranks so higher rank = better
            # best performance gets rank n, worst gets rank 1
            # use helper function for proper tie handling
            ranks = _assign_ranks_with_ties(all_values)

            # calculate average rank for each model for this metric
            baseline_metric_rank = 0.0
            current_metric_rank = 0.0

            if "baseline" in ranks:
                baseline_metric_rank = sum(ranks["baseline"]) / len(ranks["baseline"])
                metric_ranks_baseline.append(baseline_metric_rank)

            if "current" in ranks:
                current_metric_rank = sum(ranks["current"]) / len(ranks["current"])
                metric_ranks_current.append(current_metric_rank)

            # store per-metric rank averages in metrics_averages
            if metric_name in stats[KEY_METRICS_AVERAGES]:
                stats[KEY_METRICS_AVERAGES][metric_name][KEY_BASELINE_RANK_AVG] = (
                    baseline_metric_rank
                )
                stats[KEY_METRICS_AVERAGES][metric_name][KEY_CURRENT_RANK_AVG] = (
                    current_metric_rank
                )

        # calculate overall average rank across all metrics
        if metric_ranks_baseline:
            stats[KEY_METRICS_RANKS_BASELINE] = sum(metric_ranks_baseline) / len(
                metric_ranks_baseline
            )
        if metric_ranks_current:
            stats[KEY_METRICS_RANKS_CURRENT] = sum(metric_ranks_current) / len(
                metric_ranks_current
            )

        # calculate duration averages
        test_case_count = len(model_diffs)
        if test_case_count > 0:
            stats[KEY_TECHNICAL_METRICS][KEY_BASELINE][KEY_DURATION_AVG] = (
                stats[KEY_TECHNICAL_METRICS][KEY_BASELINE][KEY_DURATION_SUM]
                / test_case_count
            )
            stats[KEY_TECHNICAL_METRICS][KEY_CURRENT][KEY_DURATION_AVG] = (
                stats[KEY_TECHNICAL_METRICS][KEY_CURRENT][KEY_DURATION_SUM]
                / test_case_count
            )

        # handle inf values for min duration (when no valid durations were found)
        if stats[KEY_TECHNICAL_METRICS][KEY_BASELINE][KEY_DURATION_MIN] == float("inf"):
            stats[KEY_TECHNICAL_METRICS][KEY_BASELINE][KEY_DURATION_MIN] = 0.0
        if stats[KEY_TECHNICAL_METRICS][KEY_CURRENT][KEY_DURATION_MIN] == float("inf"):
            stats[KEY_TECHNICAL_METRICS][KEY_CURRENT][KEY_DURATION_MIN] = 0.0

        # extract per-model technical metrics from llm_model_meta
        if baseline_model and baseline_model.llm_model_meta:
            meta = baseline_model.llm_model_meta
            if isinstance(meta, dict):
                # extract request stats
                stats[KEY_TECHNICAL_METRICS][KEY_BASELINE][KEY_SUCCESS_COUNT] = (
                    meta.get(models.ExplainableLlmModel.KEY_STATS_SUCCESS, 0)
                )
                stats[KEY_TECHNICAL_METRICS][KEY_BASELINE][KEY_FAILURE_COUNT] = (
                    meta.get(models.ExplainableLlmModel.KEY_STATS_FAILURE, 0)
                )
                stats[KEY_TECHNICAL_METRICS][KEY_BASELINE][KEY_RETRY_COUNT] = meta.get(
                    models.ExplainableLlmModel.KEY_STATS_RETRY, 0
                )
                stats[KEY_TECHNICAL_METRICS][KEY_BASELINE][KEY_TIMEOUT_COUNT] = (
                    meta.get(models.ExplainableLlmModel.KEY_STATS_TIMEOUT, 0)
                )

                # extract h2ogpte_perf_stats if available
                h2ogpte_stats = meta.get(
                    models.ExplainableLlmModel.KEY_H2OGPTE_STATS, {}
                )
                if h2ogpte_stats:
                    stats[KEY_TECHNICAL_METRICS][KEY_BASELINE][KEY_CALL_COUNT] = (
                        h2ogpte_stats.get("call_count", 0)
                    )
                    stats[KEY_TECHNICAL_METRICS][KEY_BASELINE][KEY_INPUT_TOKENS] = (
                        h2ogpte_stats.get("input_tokens", 0)
                    )
                    stats[KEY_TECHNICAL_METRICS][KEY_BASELINE][KEY_OUTPUT_TOKENS] = (
                        h2ogpte_stats.get("output_tokens", 0)
                    )
                    stats[KEY_TECHNICAL_METRICS][KEY_BASELINE][
                        KEY_TOKENS_PER_SECOND
                    ] = h2ogpte_stats.get("tokens_per_second", 0.0)
                    stats[KEY_TECHNICAL_METRICS][KEY_BASELINE][
                        KEY_TIME_TO_FIRST_TOKEN
                    ] = h2ogpte_stats.get("time_to_first_token", 0.0)

        if current_model and current_model.llm_model_meta:
            meta = current_model.llm_model_meta
            if isinstance(meta, dict):
                # extract request stats
                stats[KEY_TECHNICAL_METRICS][KEY_CURRENT][KEY_SUCCESS_COUNT] = meta.get(
                    models.ExplainableLlmModel.KEY_STATS_SUCCESS, 0
                )
                stats[KEY_TECHNICAL_METRICS][KEY_CURRENT][KEY_FAILURE_COUNT] = meta.get(
                    models.ExplainableLlmModel.KEY_STATS_FAILURE, 0
                )
                stats[KEY_TECHNICAL_METRICS][KEY_CURRENT][KEY_RETRY_COUNT] = meta.get(
                    models.ExplainableLlmModel.KEY_STATS_RETRY, 0
                )
                stats[KEY_TECHNICAL_METRICS][KEY_CURRENT][KEY_TIMEOUT_COUNT] = meta.get(
                    models.ExplainableLlmModel.KEY_STATS_TIMEOUT, 0
                )

                # extract h2ogpte_perf_stats if available
                h2ogpte_stats = meta.get(
                    models.ExplainableLlmModel.KEY_H2OGPTE_STATS, {}
                )
                if h2ogpte_stats:
                    stats[KEY_TECHNICAL_METRICS][KEY_CURRENT][KEY_CALL_COUNT] = (
                        h2ogpte_stats.get("call_count", 0)
                    )
                    stats[KEY_TECHNICAL_METRICS][KEY_CURRENT][KEY_INPUT_TOKENS] = (
                        h2ogpte_stats.get("input_tokens", 0)
                    )
                    stats[KEY_TECHNICAL_METRICS][KEY_CURRENT][KEY_OUTPUT_TOKENS] = (
                        h2ogpte_stats.get("output_tokens", 0)
                    )
                    stats[KEY_TECHNICAL_METRICS][KEY_CURRENT][KEY_TOKENS_PER_SECOND] = (
                        h2ogpte_stats.get("tokens_per_second", 0.0)
                    )
                    stats[KEY_TECHNICAL_METRICS][KEY_CURRENT][
                        KEY_TIME_TO_FIRST_TOKEN
                    ] = h2ogpte_stats.get("time_to_first_token", 0.0)

        # calculate totals for display in HTML
        stats[KEY_TOTAL_TEST_CASES] = len(model_diffs)
        stats[KEY_TOTAL_METRICS] = len(metrics_data)
        stats[KEY_TOTAL_CONTEXTS] = len(model_diffs)  # same as test cases

        return stats

    @staticmethod
    def _calculate_recommendation(
        stats: dict, metrics_meta: dict | None = None
    ) -> dict:
        """Calculate model recommendation using rank-based multi-factor scoring.

        Scoring weights (importance):
        - Test Case Ranks: high (50 points per rank difference)
        - Metrics Ranks: medium (30 points per rank difference)
        - Flipped metrics: medium (3 points per net flip)
        - Empty contexts: low (0.5 points per empty context difference)

        Parameters
        ----------
        stats : dict
            Comparison statistics from _calculate_model_comparison_stats.
        metrics_meta : dict | None
            Metrics metadata containing higher_is_better information.

        Returns
        -------
        dict :
            Dictionary with keys:
            - winner: "baseline", "current", or "tie"
            - explanation: str with detailed reasoning
            - confidence: str ("high", "medium", "low")

        """
        # extract rank-based metrics
        test_case_ranks_baseline = stats.get(KEY_TEST_CASE_RANKS_BASELINE, 0)
        test_case_ranks_current = stats.get(KEY_TEST_CASE_RANKS_CURRENT, 0)
        metrics_ranks_baseline = stats.get(KEY_METRICS_RANKS_BASELINE, 0.0)
        metrics_ranks_current = stats.get(KEY_METRICS_RANKS_CURRENT, 0.0)
        flipped_to_passed = stats[KEY_FLIPPED_TO_PASSED]
        flipped_to_failed = stats[KEY_FLIPPED_TO_FAILED]
        empty_context_baseline = stats[KEY_EMPTY_CONTEXT_COUNT][KEY_BASELINE]
        empty_context_current = stats[KEY_EMPTY_CONTEXT_COUNT][KEY_CURRENT]
        metrics_averages = stats[KEY_METRICS_AVERAGES]

        # initialize scores (positive = baseline better, negative = current better)
        baseline_score = 0.0
        current_score = 0.0
        reasons = []

        # TIER 1: Critical Blockers
        # check model_passes/failures if available
        if "model_passes" in metrics_averages:
            passes_baseline = metrics_averages["model_passes"][KEY_BASELINE_AVG]
            passes_current = metrics_averages["model_passes"][KEY_CURRENT_AVG]
            passes_diff = passes_current - passes_baseline

            # critical threshold: model_passes < 0.5 is concerning
            if passes_current < 0.5 and passes_baseline >= 0.5:
                baseline_score += 100  # strong penalty against current
                reasons.append(
                    f"Current model has critically low pass rate ({passes_current:.1%})"
                )
            elif passes_baseline < 0.5 and passes_current >= 0.5:
                current_score += 100  # strong penalty against baseline
                reasons.append(
                    f"Baseline model has critically low pass rate "
                    f"({passes_baseline:.1%})"
                )
            elif abs(passes_diff) > 0.15:  # >15% change is significant
                if passes_diff > 0:
                    current_score += 50
                    reasons.append(
                        f"Current model has {passes_diff:+.1%} higher pass rate"
                    )
                else:
                    baseline_score += 50
                    reasons.append(
                        f"Baseline model has {abs(passes_diff):.1%} higher pass rate"
                    )

        # TIER 2: Primary Indicators (Rank-Based)
        # test case ranks (HIGH importance - 50 points per test case rank diff)
        test_case_ranks_diff = test_case_ranks_current - test_case_ranks_baseline
        if test_case_ranks_diff > 0:
            current_score += abs(test_case_ranks_diff) * 50.0
            reasons.append(
                f"Current model won {test_case_ranks_diff} more test cases "
                f"using rank-based comparison "
                f"({test_case_ranks_current} vs {test_case_ranks_baseline})"
            )
        elif test_case_ranks_diff < 0:
            baseline_score += abs(test_case_ranks_diff) * 50.0
            reasons.append(
                f"Baseline model won {abs(test_case_ranks_diff)} more test cases "
                f"using rank-based comparison "
                f"({test_case_ranks_baseline} vs {test_case_ranks_current})"
            )

        # metrics ranks (MEDIUM importance - 30 points per rank unit)
        metrics_ranks_diff = metrics_ranks_current - metrics_ranks_baseline
        if abs(metrics_ranks_diff) > 0.1:  # meaningful difference threshold
            if metrics_ranks_diff > 0:
                current_score += abs(metrics_ranks_diff) * 30.0
                reasons.append(
                    f"Current model has higher average rank "
                    f"({metrics_ranks_current:.2f} vs {metrics_ranks_baseline:.2f})"
                )
            else:
                baseline_score += abs(metrics_ranks_diff) * 30.0
                reasons.append(
                    f"Baseline model has higher average rank "
                    f"({metrics_ranks_baseline:.2f} vs {metrics_ranks_current:.2f})"
                )

        # flipped metrics net balance (MEDIUM importance - 3 points per net flip)
        flipped_net = flipped_to_passed - flipped_to_failed
        if flipped_net > 0:
            current_score += abs(flipped_net) * 3.0
            reasons.append(
                f"Current model has net +{flipped_net} metrics flipped to passing "
                f"({flipped_to_passed} passed, {flipped_to_failed} failed)"
            )
        elif flipped_net < 0:
            baseline_score += abs(flipped_net) * 3.0
            reasons.append(
                f"Current model has net {flipped_net} metrics flipped to failing "
                f"({flipped_to_failed} failed, {flipped_to_passed} passed)"
            )

        # empty contexts (LOW importance - 0.5 points per empty context)
        empty_context_diff = empty_context_current - empty_context_baseline
        if empty_context_diff > 5:  # significant increase in empty contexts
            baseline_score += abs(empty_context_diff) * 0.5
            reasons.append(
                f"Current model has {empty_context_diff} more empty contexts "
                f"(retrieval degradation)"
            )
        elif empty_context_diff < -5:  # significant decrease
            current_score += abs(empty_context_diff) * 0.5
            reasons.append(
                f"Current model has {abs(empty_context_diff)} fewer empty contexts "
                f"(retrieval improvement)"
            )

        # TIER 3: Secondary Indicators
        # check critical quality metrics
        critical_metrics = [
            "groundedness",
            "answer_relevancy",
            "fairness_bias",
            "toxicity",
        ]
        for metric_name in critical_metrics:
            if metric_name in metrics_averages:
                baseline_avg = metrics_averages[metric_name][KEY_BASELINE_AVG]
                current_avg = metrics_averages[metric_name][KEY_CURRENT_AVG]
                diff = current_avg - baseline_avg

                # get metric directionality
                higher_is_better = True
                if metrics_meta and metric_name in metrics_meta:
                    metric_meta = metrics_meta[metric_name]
                    if hasattr(metric_meta, "higher_is_better"):
                        higher_is_better = metric_meta.higher_is_better
                    elif isinstance(metric_meta, dict):
                        higher_is_better = metric_meta.get("higher_is_better", True)

                # apply small bonus for improvements in critical metrics
                if abs(diff) > 0.05:  # >5% change is noticeable
                    if (higher_is_better and diff > 0) or (
                        not higher_is_better and diff < 0
                    ):
                        current_score += 10
                    elif (higher_is_better and diff < 0) or (
                        not higher_is_better and diff > 0
                    ):
                        baseline_score += 10

        # DETERMINE WINNER
        score_diff = current_score - baseline_score
        threshold_high = 50  # strong confidence
        threshold_medium = 20  # medium confidence

        if score_diff > threshold_high:
            winner = "current"
            confidence = "high"
        elif score_diff > threshold_medium:
            winner = "current"
            confidence = "medium"
        elif score_diff < -threshold_high:
            winner = "baseline"
            confidence = "high"
        elif score_diff < -threshold_medium:
            winner = "baseline"
            confidence = "medium"
        else:
            winner = "tie"
            confidence = "low"

        # BUILD EXPLANATION
        if winner == "tie":
            key_factors = ", ".join(reasons[:3]) if reasons else "balanced metrics"
            explanation = (
                f"Models show similar performance "
                f"(score difference: {score_diff:.1f}). "
                f"Key factors: {key_factors}. "
                f"Consider cost, latency, and specific use-case requirements."
            )
        else:
            winner_name = winner.capitalize()
            explanation = f"The {winner_name} model is recommended. "
            if reasons:
                # prioritize top 3 most important reasons
                top_reasons = reasons[:3]
                explanation += " • ".join(top_reasons)

        return {
            "winner": winner,
            "explanation": explanation,
            "confidence": confidence,
            "baseline_score": baseline_score,
            "current_score": current_score,
        }

    @staticmethod
    def _try_generate_chart(chart_func, *args, **kwargs) -> str | None:
        """Safely generate SVG chart with defensive error handling.

        Returns SVG string if successful, None if any error occurs.
        Never crashes HTML generation.

        Parameters
        ----------
        chart_func : callable
            Chart generation function to call.
        *args
            Positional arguments for chart function.
        **kwargs
            Keyword arguments for chart function.

        Returns
        -------
        str | None
            SVG string if successful, None on any error.

        """
        try:
            # validate labels exist and non-empty
            labels = kwargs.get("labels") or (args[0] if len(args) > 0 else None)
            if not labels or not isinstance(labels, list) or len(labels) == 0:
                return None

            # validate baseline/current values
            baseline_values = kwargs.get("baseline_values") or (
                args[1] if len(args) > 1 else None
            )
            current_values = kwargs.get("current_values") or (
                args[2] if len(args) > 2 else None
            )

            if baseline_values is None or current_values is None:
                return None
            if not isinstance(baseline_values, list) or not isinstance(
                current_values, list
            ):
                return None
            if len(baseline_values) == 0 or len(current_values) == 0:
                return None

            # validate lengths match
            if len(labels) != len(baseline_values) or len(labels) != len(
                current_values
            ):
                return None

            # validate all values are numeric (no NaN)
            for val in baseline_values + current_values:
                if not isinstance(val, (int, float)):
                    return None
                # defensive NaN check - only for float values
                if isinstance(val, float):
                    try:
                        if math.isnan(val):
                            return None
                    except (TypeError, ValueError):
                        # unexpected error during isnan check, fail safely
                        return None

            # call chart generation
            svg_result = chart_func(*args, **kwargs)

            # validate result is valid SVG
            if (
                not svg_result
                or not isinstance(svg_result, str)
                or "<svg" not in svg_result
            ):
                return None

            return svg_result

        except Exception:
            # fail silently - never crash HTML generation
            return None

    @staticmethod
    def _html_model_comparison_table(
        html, stats: dict, metrics_meta: dict | None = None, model_key: str = ""
    ):
        """Generate HTML tables comparing models across all test cases.

        Creates two tables:
        1. Summary table with flipped metrics count and empty context count
        2. Metrics averages table with all metric comparisons

        Parameters
        ----------
        html : airium.Airium
            Airium HTML instance.
        stats : dict
            Comparison statistics from _calculate_model_comparison_stats.
        metrics_meta : dict | None
            Metrics metadata containing higher_is_better information.
        model_key : str
            Model key for unique ID generation.

        """
        section_id = f"models-comparison-{model_key}" if model_key else ""
        with html.div(
            klass="model-overview",
            style="margin-top: 20px;",
            id=section_id if section_id else None,
        ):
            html.h3(_t="Models Comparison")

            # add visual cmp chart for model performance (only if meaningful data)
            test_case_wins_baseline = float(stats.get(KEY_TEST_CASE_WINS_BASELINE, 0))
            test_case_wins_current = float(stats.get(KEY_TEST_CASE_WINS_CURRENT, 0))
            metrics_wins_baseline = float(stats.get(KEY_METRICS_WINS_BASELINE, 0))
            metrics_wins_current = float(stats.get(KEY_METRICS_WINS_CURRENT, 0))

            # only generate chart if at least one value is non-zero
            if (
                test_case_wins_baseline > 0
                or test_case_wins_current > 0
                or metrics_wins_baseline > 0
                or metrics_wins_current > 0
            ):
                chart_svg = EvalResultsDiffHtml._try_generate_chart(
                    charts.generate_svg_grouped_bar_chart,
                    labels=["Test Case Wins", "Metrics Wins"],
                    baseline_values=[test_case_wins_baseline, metrics_wins_baseline],
                    current_values=[test_case_wins_current, metrics_wins_current],
                    title="Model Performance Comparison",
                    baseline_label="Baseline",
                    current_label="Current",
                    width=800,
                    height=400,
                    show_values=True,
                    show_grid=True,
                )

                if chart_svg:
                    with html.div(style="margin: 30px 0; text-align: center;"):
                        charts.add_svg_chart_to_html(html, chart_svg)

            # TABLE 1: Summary statistics
            with html.table(klass="model-overview-table", style="margin-bottom: 20px;"):
                # header
                with html.tr():
                    html.th(_t="", style="text-align: center;")
                    html.th(_t="Baseline", style="text-align: center;")
                    html.th(_t="Current", style="text-align: center;")

                # ROW 1: test case ranks row
                with html.tr():
                    total_test_cases = stats.get(KEY_TOTAL_TEST_CASES, 0)
                    html.td(
                        _t=f"Test Case Ranks ({total_test_cases})",
                        klass="field-label",
                        title=(
                            "The number of test cases won using rank-based "
                            "comparison. For each test case, metrics are ranked "
                            "between baseline and current models, and the model "
                            "with the higher average rank wins that test case. "
                            "Higher rank indicates better performance."
                        ),
                    )
                    with html.td(style="text-align: center;"):
                        baseline_tc_ranks = stats[KEY_TEST_CASE_RANKS_BASELINE]
                        current_tc_ranks = stats[KEY_TEST_CASE_RANKS_CURRENT]

                        # higher count is better
                        if baseline_tc_ranks > current_tc_ranks:
                            html.strong(
                                _t=str(baseline_tc_ranks),
                                style=f"color: {COLOR_GREEN};",
                            )
                        else:
                            html(str(baseline_tc_ranks))

                    with html.td(style="text-align: center;"):
                        if current_tc_ranks > baseline_tc_ranks:
                            html.strong(
                                _t=str(current_tc_ranks),
                                style=f"color: {COLOR_GREEN};",
                            )
                        else:
                            html(str(current_tc_ranks))

                        # add delta for test case ranks
                        tc_ranks_delta = current_tc_ranks - baseline_tc_ranks
                        if tc_ranks_delta != 0:
                            html(" ")
                            delta_sign = "+" if tc_ranks_delta > 0 else ""
                            # higher is better, positive delta is green
                            delta_color = (
                                COLOR_GREEN if tc_ranks_delta > 0 else COLOR_RED
                            )
                            html.strong(
                                _t=f"({delta_sign}{tc_ranks_delta})",
                                style=f"color: {delta_color};",
                            )

                # ROW 2: metrics ranks row
                with html.tr():
                    total_metrics = stats.get(KEY_TOTAL_METRICS, 0)
                    html.td(
                        _t=f"Metrics Ranks ({total_metrics})",
                        klass="field-label",
                        title=(
                            "Average rank of metric scores across all test "
                            "cases. For each metric, scores from both models "
                            "are merged and ranked. Higher average rank indicates "
                            "better overall performance. Ranks are calculated "
                            "per metric respecting higher_is_better property."
                        ),
                    )
                    with html.td(style="text-align: center;"):
                        baseline_rank = stats[KEY_METRICS_RANKS_BASELINE]
                        current_rank = stats[KEY_METRICS_RANKS_CURRENT]

                        # higher rank is better
                        if baseline_rank > 0 and (
                            current_rank == 0 or baseline_rank > current_rank
                        ):
                            html.strong(
                                _t=f"{baseline_rank:.2f}",
                                style=f"color: {COLOR_GREEN};",
                            )
                        elif baseline_rank > 0:
                            html(f"{baseline_rank:.2f}")
                        else:
                            html("-")

                    with html.td(style="text-align: center;"):
                        if current_rank > 0 and (
                            baseline_rank == 0 or current_rank > baseline_rank
                        ):
                            html.strong(
                                _t=f"{current_rank:.2f}",
                                style=f"color: {COLOR_GREEN};",
                            )
                        elif current_rank > 0:
                            html(f"{current_rank:.2f}")
                        else:
                            html("-")

                        # add delta for metrics ranks
                        if current_rank > 0 and baseline_rank > 0:
                            rank_delta = current_rank - baseline_rank
                            if rank_delta != 0:
                                html(" ")
                                delta_sign = "+" if rank_delta > 0 else ""
                                # higher rank is better, positive delta is green
                                delta_color = (
                                    COLOR_GREEN if rank_delta > 0 else COLOR_RED
                                )
                                html.strong(
                                    _t=f"({delta_sign}{rank_delta:.2f})",
                                    style=f"color: {delta_color};",
                                )

                # ROW 3: test case wins row
                with html.tr():
                    total_test_cases = stats.get(KEY_TOTAL_TEST_CASES, 0)
                    html.td(
                        _t=f"Test Case Wins ({total_test_cases})",
                        klass="field-label",
                        title=(
                            "The number of test cases where the model won "
                            "based on majority of metrics within that test case."
                        ),
                    )
                    with html.td(style="text-align: center;"):
                        baseline_test_case_wins = stats[KEY_TEST_CASE_WINS_BASELINE]
                        current_test_case_wins = stats[KEY_TEST_CASE_WINS_CURRENT]

                        if baseline_test_case_wins > current_test_case_wins:
                            html.strong(
                                _t=str(baseline_test_case_wins),
                                style=f"color: {COLOR_GREEN};",
                            )
                        else:
                            html(str(baseline_test_case_wins))

                    with html.td(style="text-align: center;"):
                        if current_test_case_wins > baseline_test_case_wins:
                            html.strong(
                                _t=str(current_test_case_wins),
                                style=f"color: {COLOR_GREEN};",
                            )
                        else:
                            html(str(current_test_case_wins))

                        # add delta for test case wins
                        tc_wins_delta = current_test_case_wins - baseline_test_case_wins
                        if tc_wins_delta != 0:
                            html(" ")
                            delta_sign = "+" if tc_wins_delta > 0 else ""
                            # higher is better, positive delta is green
                            delta_color = (
                                COLOR_GREEN if tc_wins_delta > 0 else COLOR_RED
                            )
                            html.strong(
                                _t=f"({delta_sign}{tc_wins_delta})",
                                style=f"color: {delta_color};",
                            )

                # ROW 4: metrics wins row
                with html.tr():
                    total_metrics = stats.get(KEY_TOTAL_METRICS, 0)
                    html.td(
                        _t=f"Metrics Wins ({total_metrics})",
                        klass="field-label",
                        title=(
                            "The number of metrics across all test cases where "
                            "the model scored better than the other model."
                        ),
                    )
                    with html.td(style="text-align: center;"):
                        baseline_wins = stats[KEY_METRICS_WINS_BASELINE]
                        current_wins = stats[KEY_METRICS_WINS_CURRENT]

                        if baseline_wins > current_wins:
                            html.strong(
                                _t=str(baseline_wins),
                                style=f"color: {COLOR_GREEN};",
                            )
                        else:
                            html(str(baseline_wins))

                    with html.td(style="text-align: center;"):
                        if current_wins > baseline_wins:
                            html.strong(
                                _t=str(current_wins),
                                style=f"color: {COLOR_GREEN};",
                            )
                        else:
                            html(str(current_wins))

                        # add delta for metrics wins
                        metrics_wins_delta = current_wins - baseline_wins
                        if metrics_wins_delta != 0:
                            html(" ")
                            delta_sign = "+" if metrics_wins_delta > 0 else ""
                            # higher is better, positive delta is green
                            delta_color = (
                                COLOR_GREEN if metrics_wins_delta > 0 else COLOR_RED
                            )
                            html.strong(
                                _t=f"({delta_sign}{metrics_wins_delta})",
                                style=f"color: {delta_color};",
                            )

                # ROW 5: flipped test cases count
                with html.tr():
                    total_test_cases = stats.get(KEY_TOTAL_TEST_CASES, 0)
                    html.td(
                        _t=f"Flipped Test Cases ({total_test_cases})",
                        klass="field-label",
                        title=(
                            "The number of test cases for whose a metric flipped from "
                            "failed to passed or vice versa."
                        ),
                    )
                    html.td(_t="-", style="text-align: center;")
                    with html.td(style="text-align: center;"):
                        flipped_count = stats[KEY_FLIPPED_METRICS_COUNT_STATS]
                        html(str(flipped_count))

                # ROW 5: flipped to passed
                with html.tr():
                    html.td(_t="Flipped to Passed", klass="field-label")
                    html.td(_t="-", style="text-align: center;")
                    with html.td(style="text-align: center;"):
                        flipped_passed = stats[KEY_FLIPPED_TO_PASSED]
                        if flipped_passed > 0:
                            html.strong(
                                _t=str(flipped_passed),
                                style=f"color: {COLOR_GREEN};",
                            )
                        else:
                            html(str(flipped_passed))

                # ROW 6: flipped to failed
                with html.tr():
                    html.td(_t="Flipped to Failed", klass="field-label")
                    html.td(_t="-", style="text-align: center;")
                    with html.td(style="text-align: center;"):
                        flipped_failed = stats[KEY_FLIPPED_TO_FAILED]
                        if flipped_failed > 0:
                            html.strong(
                                _t=str(flipped_failed),
                                style=f"color: {COLOR_RED};",
                            )
                        else:
                            html(str(flipped_failed))

                # ROW 7: empty context count
                with html.tr():
                    total_contexts = stats.get(KEY_TOTAL_CONTEXTS, 0)
                    html.td(
                        _t=f"Empty Contexts ({total_contexts})",
                        klass="field-label",
                        title=(
                            "The number of test cases for which RAG returned empty "
                            "retrived context."
                        ),
                    )
                    with html.td(style="text-align: center;"):
                        baseline_empty = stats[KEY_EMPTY_CONTEXT_COUNT][KEY_BASELINE]
                        if baseline_empty > 0:
                            html.strong(
                                _t=str(baseline_empty),
                                style=f"color: {COLOR_ORANGE};",
                            )
                        else:
                            html(str(baseline_empty))
                    with html.td(style="text-align: center;"):
                        # show current count with difference inline
                        current_empty = stats[KEY_EMPTY_CONTEXT_COUNT][KEY_CURRENT]
                        diff_empty = current_empty - baseline_empty

                        if current_empty > 0:
                            html.strong(
                                _t=str(current_empty),
                                style=f"color: {COLOR_ORANGE};",
                            )
                        else:
                            html(str(current_empty))

                        # add difference inline
                        if diff_empty != 0:
                            html(" ")
                            diff_sign = "+" if diff_empty > 0 else ""
                            diff_color = (
                                COLOR_RED
                                if diff_empty > 0
                                else COLOR_GREEN
                                if diff_empty < 0
                                else ""
                            )
                            if diff_color:
                                html.strong(
                                    _t=f"({diff_sign}{diff_empty})",
                                    style=f"color: {diff_color};",
                                )
                            else:
                                html(f"({diff_sign}{diff_empty})")

            # LEGEND
            html.p(_t="Legend:", style="margin-top: 20px; margin-bottom: 10px;")
            with html.ul(
                style="margin-left: 20px; margin-bottom: 30px; line-height: 1.8;"
            ):
                # TODO compact texts below
                with html.li():
                    html.strong(_t="Test Case Ranks")
                    html(
                        ": The number of test cases won by each model using "
                        "rank-based comparison. The value in parentheses indicates "
                        "the total number of comparable test cases. "
                        "For each test case, metrics are "
                        "ranked between baseline and current models (best "
                        "performance gets rank 2, other gets rank 1), then "
                        "averaged. The model with "
                    )
                    html.strong(_t="higher average rank wins that test case")
                    html(
                        ". Ties don't count. This provides a more nuanced comparison "
                        "than simple metric wins, as it considers relative "
                        "performance within each test case."
                    )
                with html.li():
                    html.strong(_t="Metrics Ranks")
                    html(
                        ": Average rank of metric scores across all test cases. "
                        "The value in parentheses indicates the total number of "
                        "metrics. "
                        "For each metric, all scores from both models are merged "
                        "and ranked together, with the best performance receiving "
                        "the highest rank. "
                    )
                    html.strong(_t="Higher average rank indicates better performance")
                    html(
                        ". This rank-based comparison method is robust to scale "
                        "differences between metrics and provides an intuitive "
                        "overall performance score."
                    )
                with html.li():
                    html.strong(_t="Test Case Wins")
                    html(": Number of test cases won by each model based on the ")
                    html.strong(_t="majority of metrics")
                    html(" within each test case. For each test case, the model ")
                    html("that wins more metrics (considering their directionality ")
                    html("- higher_is_better) wins that test case. Test cases with ")
                    html("ties are not counted.")
                with html.li():
                    html.strong(_t="Metrics Wins")
                    html(": Total count of ")
                    html.strong(_t="individual metric comparisons")
                    html(" won across all test cases. Each metric comparison ")
                    html("where one model scores better than the other (considering ")
                    html(
                        "the metric's directionality) increments that model's wins "
                        "count."
                    )
                with html.li():
                    html.strong(_t="Flipped Test Cases")
                    html(": Total number of test cases where at least one metric ")
                    html("flipped from passing to failing or vice versa (based on ")
                    html("the metric's threshold).")
                with html.li():
                    html.strong(_t="Flipped to Passed")
                    html(": Number of metrics that changed from failing (below ")
                    html("threshold) in Baseline to passing (above threshold) in ")
                    html("Current.")
                with html.li():
                    html.strong(_t="Flipped to Failed")
                    html(": Number of metrics that changed from passing (above ")
                    html("threshold) in Baseline to failing (below threshold) in ")
                    html("Current.")
                with html.li():
                    html.strong(_t="Empty Contexts")
                    html(": Number of test cases where the RAG system returned no ")
                    html("retrieved context chunks.")

            # TABLE 2: Metric averages
            # sort by highest absolute difference between baseline and current ranks
            sorted_metrics = sorted(
                stats[KEY_METRICS_AVERAGES].items(),
                key=lambda x: (
                    -abs(
                        x[1].get(KEY_BASELINE_RANK_AVG, 0.0)
                        - x[1].get(KEY_CURRENT_RANK_AVG, 0.0)
                    ),
                    x[0],  # then by metric name for stability
                ),
            )
            # if there are no DIFFERENT metrics, skip the table
            if sorted_metrics:
                html.h3(_t="Metrics Comparison")

                with html.table(klass="model-overview-table"):
                    # header
                    with html.tr():
                        html.th(_t="Metric", style="text-align: center;")
                        html.th(_t="Baseline Ranks", style="text-align: center;")
                        html.th(_t="Current Ranks", style="text-align: center;")
                        html.th(_t="Baseline Wins", style="text-align: center;")
                        html.th(_t="Current Wins", style="text-align: center;")
                        html.th(_t="Baseline Avg", style="text-align: center;")
                        html.th(_t="Current Avg", style="text-align: center;")

                    # metric averages
                    for metric_name, metric_stats in sorted_metrics:
                        baseline_avg = metric_stats[KEY_BASELINE_AVG]
                        current_avg = metric_stats[KEY_CURRENT_AVG]
                        diff = metric_stats[KEY_DIFF_VALUE]
                        baseline_better_wins = metric_stats[KEY_BASELINE_BETTER_WINS]
                        current_better_wins = metric_stats[KEY_CURRENT_BETTER_WINS]
                        baseline_rank_avg = metric_stats.get(KEY_BASELINE_RANK_AVG, 0.0)
                        current_rank_avg = metric_stats.get(KEY_CURRENT_RANK_AVG, 0.0)

                        # determine if higher is better for this metric
                        higher_is_better = True  # default
                        if metrics_meta and metric_name in metrics_meta:
                            metric_meta_obj = metrics_meta[metric_name]
                            if hasattr(metric_meta_obj, "higher_is_better"):
                                higher_is_better = metric_meta_obj.higher_is_better
                            elif isinstance(metric_meta_obj, dict):
                                higher_is_better = metric_meta_obj.get(
                                    "higher_is_better", True
                                )

                        with html.tr():
                            html.td(_t=metric_name, klass="field-label")

                            # baseline ranks column
                            with html.td(style="text-align: center;"):
                                if baseline_rank_avg > 0 and (
                                    current_rank_avg == 0
                                    or baseline_rank_avg > current_rank_avg
                                ):
                                    html.strong(
                                        _t=f"{baseline_rank_avg:.2f}",
                                        style=f"color: {COLOR_GREEN};",
                                    )
                                elif baseline_rank_avg > 0:
                                    html(f"{baseline_rank_avg:.2f}")
                                else:
                                    html("-")

                            # current ranks column
                            with html.td(style="text-align: center;"):
                                if current_rank_avg > 0 and (
                                    baseline_rank_avg == 0
                                    or current_rank_avg > baseline_rank_avg
                                ):
                                    html.strong(
                                        _t=f"{current_rank_avg:.2f}",
                                        style=f"color: {COLOR_GREEN};",
                                    )
                                elif current_rank_avg > 0:
                                    html(f"{current_rank_avg:.2f}")
                                else:
                                    html("-")

                                # add delta for current ranks
                                if current_rank_avg > 0 and baseline_rank_avg > 0:
                                    rank_delta = current_rank_avg - baseline_rank_avg
                                    if rank_delta != 0:
                                        html(" ")
                                        delta_sign = "+" if rank_delta > 0 else ""
                                        # higher rank is better, positive delta is green
                                        if rank_delta > 0:
                                            delta_color = COLOR_GREEN  # green
                                        else:
                                            delta_color = COLOR_RED  # red

                                        html.strong(
                                            _t=f"({delta_sign}{rank_delta:.2f})",
                                            style=f"color: {delta_color};",
                                        )

                            # baseline wins column
                            with html.td(style="text-align: center;"):
                                if baseline_better_wins > current_better_wins:
                                    html.strong(
                                        _t=str(baseline_better_wins),
                                        style=f"color: {COLOR_GREEN};",
                                    )
                                elif baseline_better_wins < current_better_wins:
                                    html.span(
                                        _t=str(baseline_better_wins),
                                        style=f"color: {COLOR_RED};",
                                    )
                                else:
                                    html(str(baseline_better_wins))

                            # current wins column
                            with html.td(style="text-align: center;"):
                                if current_better_wins > baseline_better_wins:
                                    html.strong(
                                        _t=str(current_better_wins),
                                        style=f"color: {COLOR_GREEN};",
                                    )
                                elif current_better_wins < baseline_better_wins:
                                    html.span(
                                        _t=str(current_better_wins),
                                        style=f"color: {COLOR_RED};",
                                    )
                                else:
                                    html(str(current_better_wins))

                                # add delta for current wins
                                wins_delta = current_better_wins - baseline_better_wins
                                if wins_delta != 0:
                                    html(" ")
                                    delta_sign = "+" if wins_delta > 0 else ""
                                    # higher wins is better, so positive delta is green
                                    if wins_delta > 0:
                                        delta_color = COLOR_GREEN  # green
                                    else:
                                        delta_color = COLOR_RED  # red

                                    html.strong(
                                        _t=f"({delta_sign}{wins_delta})",
                                        style=f"color: {delta_color};",
                                    )

                            with html.td(style="text-align: center;"):
                                html(f"{baseline_avg:.8f}")
                            with html.td(style="text-align: center;"):
                                # show current avg with difference inline
                                html(f"{current_avg:.8f} ")
                                diff_sign = "+" if diff > 0 else ""
                                # determine color based on higher_is_better
                                # and diff sign
                                if diff > 0:
                                    # positive change: green if higher is better,
                                    # else red
                                    diff_color = (
                                        COLOR_GREEN if higher_is_better else COLOR_RED
                                    )
                                elif diff < 0:
                                    # negative change: red if higher is better,
                                    # else green
                                    diff_color = (
                                        COLOR_RED if higher_is_better else COLOR_GREEN
                                    )
                                else:
                                    diff_color = ""

                                if diff != 0:
                                    if diff_color:
                                        html.strong(
                                            _t=f"({diff_sign}{diff:.8f})",
                                            style=f"color: {diff_color};",
                                        )
                                    else:
                                        html(f"({diff_sign}{diff:.8f})")

                # LEGEND for metrics averages table
                html.p(_t="Legend:", style="margin-top: 20px; margin-bottom: 10px;")
                with html.ul(style="margin-left: 20px; line-height: 1.8;"):
                    with html.li():
                        html.strong(_t="Baseline Model Ranks")
                        html(
                            ": Average rank of the Baseline model for this specific "
                            "metric across all test cases. For each metric, all scores "
                            "from both models are merged and ranked together, with the "
                            "best performance receiving the highest rank. "
                        )
                        html.strong(_t="Higher rank indicates better performance")
                        html(
                            ". This rank-based comparison is robust to scale "
                            "differences and provides a normalized performance "
                            "measure. "
                        )
                    with html.li():
                        html.strong(_t="Current Model Ranks")
                        html(
                            ": Average rank of the Current model for this specific "
                            "metric across all test cases. For each metric, all scores "
                            "from both models are merged and ranked together, with the "
                            "best performance receiving the highest rank. "
                        )
                        html.strong(_t="Higher rank indicates better performance")
                        html(
                            ". This rank-based comparison is robust to scale "
                            "differences and provides a normalized performance "
                            "measure. "
                        )
                    with html.li():
                        html.strong(_t="Baseline Model Wins")
                        html(
                            ": Number of test cases where the Baseline model scored "
                            "better for this specific metric (considering the metric's "
                            "directionality - higher_is_better). "
                        )
                    with html.li():
                        html.strong(_t="Current Model Wins")
                        html(
                            ": Number of test cases where the Current model scored "
                            "better for this specific metric (considering the metric's "
                            "directionality - higher_is_better). "
                        )
                    with html.li():
                        html.strong(_t="Baseline Model Avg")
                        html(
                            ": Average metric score for the Baseline model across all "
                            "test cases."
                        )
                    with html.li():
                        html.strong(_t="Current Model Avg")
                        html(
                            ": Average metric score for the Current model across all "
                            "test cases. The difference from Baseline is shown inline "
                            "(green for improvement, red for degradation, considering "
                            "the metric's directionality)."
                        )

            # add technical performance metrics section inside Models Comparison div
            EvalResultsDiffHtml._html_technical_metrics_section(
                html, stats, model_key=model_key
            )

    @staticmethod
    def _html_format_config_diff(html, diff_dict: dict):
        """Format configuration diff into readable HTML sections.

        Parameters
        ----------
        html : airium.Airium
            Airium HTML instance.
        diff_dict : dict
            DeepDiff dictionary with changes.

        """
        # section styles - removed background, keeping only border
        section_style = "margin-bottom: 15px; padding: 10px; border-radius: 4px;"

        # VALUES CHANGED
        values_changed = diff_dict.get("values_changed", {})
        if values_changed:
            style_changed = f"{section_style} border-left: 4px solid {COLOR_ORANGE};"
            with html.div(style=style_changed):
                html.strong(
                    _t=f"Values Changed ({len(values_changed)})",
                    style=f"font-size: 1.0em; color: {COLOR_ORANGE};",
                )
                with html.table(style="width: 100%; margin-top: 8px;"):
                    for path, change_info in sorted(values_changed.items()):
                        with html.tr():
                            with html.td(
                                style=(
                                    "font-family: monospace; padding: 4px; width: 70%;"
                                )
                            ):
                                html(path)
                            with html.td(style="padding: 4px;"):
                                old_val = change_info.get("old_value", "N/A")
                                new_val = change_info.get("new_value", "N/A")
                                style_old = (
                                    f"color: {COLOR_RED}; "
                                    f"text-decoration: line-through;"
                                )
                                with html.span(style=style_old):
                                    html(f"{json.dumps(old_val)}")
                                html(" → ")
                                with html.span(
                                    style=f"color: {COLOR_GREEN}; font-weight: bold;"
                                ):
                                    html(f"{json.dumps(new_val)}")

        # ITEMS ADDED
        items_added = diff_dict.get("dictionary_item_added", {})
        if items_added:
            style_added = f"{section_style} border-left: 4px solid {COLOR_GREEN};"
            with html.div(style=style_added):
                html.strong(
                    _t=f"Items Added ({len(items_added)})",
                    style=f"font-size: 1.0em; color: {COLOR_GREEN};",
                )
                with html.table(style="width: 100%; margin-top: 8px;"):
                    for path, value in sorted(items_added.items()):
                        with html.tr():
                            with html.td(
                                style=(
                                    "font-family: monospace; padding: 4px; width: 70%;"
                                )
                            ):
                                html(path)
                            with html.td(style=f"padding: 4px; color: {COLOR_GREEN};"):
                                html(f"{json.dumps(value)}")

        # ITEMS REMOVED
        items_removed = diff_dict.get("dictionary_item_removed", {})
        if items_removed:
            style_removed = f"{section_style} border-left: 4px solid {COLOR_RED};"
            with html.div(style=style_removed):
                html.strong(
                    _t=f"Items Removed ({len(items_removed)})",
                    style=f"font-size: 1.0em; color: {COLOR_RED};",
                )
                with html.table(style="width: 100%; margin-top: 8px;"):
                    for path, value in sorted(items_removed.items()):
                        with html.tr():
                            with html.td(
                                style=(
                                    "font-family: monospace; padding: 4px; width: 70%;"
                                )
                            ):
                                html(path)
                            with html.td(style=f"padding: 4px; color: {COLOR_RED};"):
                                html(f"{json.dumps(value)}")

        # ITERABLE ITEMS ADDED
        iter_added = diff_dict.get("iterable_item_added", {})
        if iter_added:
            style_iter_added = f"{section_style} border-left: 4px solid #17a2b8;"
            with html.div(style=style_iter_added):
                html.strong(
                    _t=f"List/Array Items Added ({len(iter_added)})",
                    style="font-size: 0.85em; color: #17a2b8;",
                )
                with html.table(style="width: 100%; margin-top: 8px;"):
                    for path, value in sorted(iter_added.items()):
                        with html.tr():
                            with html.td(
                                style=(
                                    "font-family: monospace; padding: 4px; width: 40%;"
                                )
                            ):
                                html(path)
                            with html.td(style="padding: 4px; color: #17a2b8;"):
                                html(f"{json.dumps(value)}")

        # ITERABLE ITEMS REMOVED
        iter_removed = diff_dict.get("iterable_item_removed", {})
        if iter_removed:
            style_iter_removed = f"{section_style} border-left: 4px solid {COLOR_RED};"
            with html.div(style=style_iter_removed):
                html.strong(
                    _t=f"List/Array Items Removed ({len(iter_removed)})",
                    style=f"font-size: 0.85em; color: {COLOR_RED};",
                )
                with html.table(style="width: 100%; margin-top: 8px;"):
                    for path, value in sorted(iter_removed.items()):
                        with html.tr():
                            with html.td(
                                style=(
                                    "font-family: monospace; padding: 4px; width: 40%;"
                                )
                            ):
                                html(path)
                            with html.td(style=f"padding: 4px; color: {COLOR_RED};"):
                                html(f"{json.dumps(value)}")

        # TYPE CHANGES
        type_changes = diff_dict.get("type_changes", {})
        if type_changes:
            style_type_changes = f"{section_style} border-left: 4px solid #fdcb6e;"
            with html.div(style=style_type_changes):
                html.strong(
                    _t=f"Type Changes ({len(type_changes)})",
                    style="font-size: 0.85em; color: #fdcb6e;",
                )
                with html.table(style="width: 100%; margin-top: 8px;"):
                    for path, change_info in sorted(type_changes.items()):
                        with html.tr():
                            with html.td(
                                style=(
                                    "font-family: monospace; padding: 4px; width: 40%;"
                                )
                            ):
                                html(path)
                            with html.td(style="padding: 4px;"):
                                old_type = change_info.get("old_type", "N/A")
                                new_type = change_info.get("new_type", "N/A")
                                html(f"{old_type} → {new_type}")

    @staticmethod
    def _html_model_config_comparison(
        html,
        baseline_model: models.ExplainableLlmModel | models.ExplainableRagModel | None,
        current_model: models.ExplainableLlmModel | models.ExplainableRagModel | None,
        model_key: str = "",
    ):
        """Generate HTML table comparing model configurations.

        Shows full configuration dictionary for baseline model and structured
        diff sections for current model highlighting specific types of changes.

        Parameters
        ----------
        html : airium.Airium
            Airium HTML instance.
        baseline_model : models.ExplainableLlmModel | models.ExplainableRagModel | None
            Baseline model instance.
        current_model : models.ExplainableLlmModel | models.ExplainableRagModel | None
            Current model instance.
        model_key : str
            Model key for unique ID generation.

        """
        # extract model_cfg from both models
        baseline_cfg = baseline_model.model_cfg if baseline_model else {}
        current_cfg = current_model.model_cfg if current_model else {}

        # if both configs are empty, don't show the table
        if not baseline_cfg and not current_cfg:
            return

        # use JSONComparator to find differences
        comparator = _explanations_diff_json.JSONComparator(baseline_cfg, current_cfg)

        # if no differences, don't show the table
        if not comparator.has_differences():
            return

        section_id = f"model-config-comparison-{model_key}" if model_key else ""
        with html.div(
            klass="model-config-comparison",
            style="margin-top: 20px;",
            id=section_id if section_id else None,
        ):
            html.h3(_t="Models Configuration Comparison")

            with html.table(klass="model-overview-table"):
                # header
                with html.tr():
                    html.th(
                        _t="Baseline Model Configuration", style="text-align: center;"
                    )
                    html.th(
                        _t="Current Model Configuration Diff",
                        style="text-align: center;",
                    )

                # BASELINE: show full configuration as formatted JSON
                with html.tr():
                    with html.td(style="vertical-align: top;"):
                        if baseline_cfg:
                            formatted_baseline = json.dumps(
                                baseline_cfg, indent=2, sort_keys=True
                            )
                            with html.pre(
                                style=(
                                    "background-color: #f5f5f5; "
                                    "padding: 10px; "
                                    "border-radius: 4px; "
                                    "font-size: 0.9em; "
                                    "overflow-x: auto; "
                                    "white-space: pre-wrap; "
                                    "word-wrap: break-word; "
                                    "max-width: 100%;"
                                )
                            ):
                                html(formatted_baseline)
                        else:
                            html.i(_t="<empty>")

                    # CURRENT: show structured diff sections
                    with html.td(style="vertical-align: top;"):
                        if current_cfg:
                            diff_dict = comparator.to_dict()
                            if diff_dict:
                                EvalResultsDiffHtml._html_format_config_diff(
                                    html, diff_dict
                                )
                            else:
                                html.i(_t="<no differences>")
                        else:
                            html.i(_t="<empty>")

    @staticmethod
    def _html_technical_metrics_section(html, stats: dict, model_key: str = ""):
        """Generate HTML section for technical performance metrics comparison.

        Parameters
        ----------
        html : airium.Airium
            Airium HTML instance.
        stats : dict
            Comparison statistics from _calculate_model_cmp_stats.
        model_key : str
            Model key for unique ID generation.

        """
        tech_metrics = stats.get(KEY_TECHNICAL_METRICS, {})
        baseline_metrics = tech_metrics.get(KEY_BASELINE, {})
        current_metrics = tech_metrics.get(KEY_CURRENT, {})

        section_id = f"technical-metrics-{model_key}" if model_key else ""
        with html.div(
            klass="technical-metrics-section",
            style="margin-top: 20px;",
            id=section_id if section_id else None,
        ):
            html.h3(_t="Technical Performance Metrics")

            with html.p(style="margin-bottom: 15px;"):
                html(
                    "Operational performance metrics aggregated across all test "
                    "cases. Lower costs and durations indicate better efficiency. "
                    "Metrics include per-test-case aggregations (cost, duration) "
                    "and per-model LLM statistics (tokens, request counts)."
                )

            # add operational efficiency chart (if meaningful data exists)
            baseline_cost_chart = baseline_metrics.get(KEY_COST_SUM, 0.0)
            current_cost_chart = current_metrics.get(KEY_COST_SUM, 0.0)
            baseline_dur_chart = baseline_metrics.get(KEY_DURATION_AVG, 0.0)
            current_dur_chart = current_metrics.get(KEY_DURATION_AVG, 0.0)

            # only generate if we have meaningful data
            if (baseline_cost_chart > 0 or current_cost_chart > 0) and (
                baseline_dur_chart > 0 or current_dur_chart > 0
            ):
                # scale cost to millidollars for better visualization
                baseline_cost_scaled = baseline_cost_chart * 1000
                current_cost_scaled = current_cost_chart * 1000

                chart_svg = EvalResultsDiffHtml._try_generate_chart(
                    charts.generate_svg_grouped_bar_chart,
                    labels=["Cost (millidollars)", "Avg Duration (seconds)"],
                    baseline_values=[baseline_cost_scaled, baseline_dur_chart],
                    current_values=[current_cost_scaled, current_dur_chart],
                    title="Operational Efficiency Comparison (Lower is Better)",
                    baseline_label="Baseline",
                    current_label="Current",
                    width=800,
                    height=400,
                    show_values=True,
                    show_grid=True,
                )

                if chart_svg:
                    with html.div(style="margin: 30px 0; text-align: center;"):
                        charts.add_svg_chart_to_html(html, chart_svg)

            with html.table(klass="model-overview-table"):
                # header
                with html.tr():
                    html.th(_t="Metric", style="text-align: left;")
                    html.th(_t="Baseline", style="text-align: center;")
                    html.th(_t="Current", style="text-align: center;")

                # ROW 1: Total Cost
                baseline_cost = baseline_metrics.get(KEY_COST_SUM, 0.0)
                current_cost = current_metrics.get(KEY_COST_SUM, 0.0)
                with html.tr():
                    html.td(_t="Total Cost (USD)", klass="field-label")
                    with html.td(style="text-align: center;"):
                        # lower is better
                        if baseline_cost > 0 and (
                            current_cost == 0 or baseline_cost < current_cost
                        ):
                            html.strong(
                                _t=f"${baseline_cost:.4f}",
                                style=f"color: {COLOR_GREEN};",
                            )
                        elif baseline_cost > 0:
                            html(f"${baseline_cost:.4f}")
                        else:
                            html("$0.0000")

                    with html.td(style="text-align: center;"):
                        if current_cost > 0 and (
                            baseline_cost == 0 or current_cost < baseline_cost
                        ):
                            html.strong(
                                _t=f"${current_cost:.4f}",
                                style=f"color: {COLOR_GREEN};",
                            )
                        elif current_cost > 0:
                            html(f"${current_cost:.4f}")
                        else:
                            html("$0.0000")

                        # add delta
                        if baseline_cost > 0 or current_cost > 0:
                            cost_delta = current_cost - baseline_cost
                            if abs(cost_delta) > 0.0001:
                                html(" ")
                                delta_sign = "+" if cost_delta > 0 else ""
                                # lower cost is better, negative delta is green
                                delta_color = (
                                    COLOR_RED if cost_delta > 0 else COLOR_GREEN
                                )
                                html.strong(
                                    _t=f"({delta_sign}${cost_delta:.4f})",
                                    style=f"color: {delta_color};",
                                )

                # ROW 2: Average Duration
                baseline_avg_dur = baseline_metrics.get(KEY_DURATION_AVG, 0.0)
                current_avg_dur = current_metrics.get(KEY_DURATION_AVG, 0.0)
                with html.tr():
                    html.td(_t="Avg Duration (s)", klass="field-label")
                    with html.td(style="text-align: center;"):
                        # lower is better
                        if baseline_avg_dur > 0 and (
                            current_avg_dur == 0 or baseline_avg_dur < current_avg_dur
                        ):
                            html.strong(
                                _t=f"{baseline_avg_dur:.2f}",
                                style=f"color: {COLOR_GREEN};",
                            )
                        elif baseline_avg_dur > 0:
                            html(f"{baseline_avg_dur:.2f}")
                        else:
                            html("0.00")

                    with html.td(style="text-align: center;"):
                        if current_avg_dur > 0 and (
                            baseline_avg_dur == 0 or current_avg_dur < baseline_avg_dur
                        ):
                            html.strong(
                                _t=f"{current_avg_dur:.2f}",
                                style=f"color: {COLOR_GREEN};",
                            )
                        elif current_avg_dur > 0:
                            html(f"{current_avg_dur:.2f}")
                        else:
                            html("0.00")

                        # add delta
                        if baseline_avg_dur > 0 or current_avg_dur > 0:
                            dur_delta = current_avg_dur - baseline_avg_dur
                            if abs(dur_delta) > 0.01:
                                html(" ")
                                delta_sign = "+" if dur_delta > 0 else ""
                                # lower duration is better, negative delta is green
                                delta_color = (
                                    COLOR_RED if dur_delta > 0 else COLOR_GREEN
                                )
                                html.strong(
                                    _t=f"({delta_sign}{dur_delta:.2f}s)",
                                    style=f"color: {delta_color};",
                                )

                # ROW 3: Min Duration
                baseline_min_dur = baseline_metrics.get(KEY_DURATION_MIN, 0.0)
                current_min_dur = current_metrics.get(KEY_DURATION_MIN, 0.0)
                # handle inf values
                if baseline_min_dur == float("inf"):
                    baseline_min_dur = 0.0
                if current_min_dur == float("inf"):
                    current_min_dur = 0.0
                with html.tr():
                    html.td(_t="Min Duration (s)", klass="field-label")
                    with html.td(style="text-align: center;"):
                        if baseline_min_dur > 0 and (
                            current_min_dur == 0 or baseline_min_dur < current_min_dur
                        ):
                            html.strong(
                                _t=f"{baseline_min_dur:.2f}",
                                style=f"color: {COLOR_GREEN};",
                            )
                        elif baseline_min_dur > 0:
                            html(f"{baseline_min_dur:.2f}")
                        else:
                            html("0.00")

                    with html.td(style="text-align: center;"):
                        if current_min_dur > 0 and (
                            baseline_min_dur == 0 or current_min_dur < baseline_min_dur
                        ):
                            html.strong(
                                _t=f"{current_min_dur:.2f}",
                                style=f"color: {COLOR_GREEN};",
                            )
                        elif current_min_dur > 0:
                            html(f"{current_min_dur:.2f}")
                        else:
                            html("0.00")

                        # add delta
                        if baseline_min_dur > 0 or current_min_dur > 0:
                            min_delta = current_min_dur - baseline_min_dur
                            if abs(min_delta) > 0.01:
                                html(" ")
                                delta_sign = "+" if min_delta > 0 else ""
                                # lower duration is better, negative delta is green
                                delta_color = (
                                    COLOR_RED if min_delta > 0 else COLOR_GREEN
                                )
                                html.strong(
                                    _t=f"({delta_sign}{min_delta:.2f}s)",
                                    style=f"color: {delta_color};",
                                )

                # ROW 4: Max Duration
                baseline_max_dur = baseline_metrics.get(KEY_DURATION_MAX, 0.0)
                current_max_dur = current_metrics.get(KEY_DURATION_MAX, 0.0)
                with html.tr():
                    html.td(_t="Max Duration (s)", klass="field-label")
                    with html.td(style="text-align: center;"):
                        if baseline_max_dur > 0 and (
                            current_max_dur == 0 or baseline_max_dur < current_max_dur
                        ):
                            html.strong(
                                _t=f"{baseline_max_dur:.2f}",
                                style=f"color: {COLOR_GREEN};",
                            )
                        elif baseline_max_dur > 0:
                            html(f"{baseline_max_dur:.2f}")
                        else:
                            html("0.00")

                    with html.td(style="text-align: center;"):
                        if current_max_dur > 0 and (
                            baseline_max_dur == 0 or current_max_dur < baseline_max_dur
                        ):
                            html.strong(
                                _t=f"{current_max_dur:.2f}",
                                style=f"color: {COLOR_GREEN};",
                            )
                        elif current_max_dur > 0:
                            html(f"{current_max_dur:.2f}")
                        else:
                            html("0.00")

                        # add delta
                        if baseline_max_dur > 0 or current_max_dur > 0:
                            max_delta = current_max_dur - baseline_max_dur
                            if abs(max_delta) > 0.01:
                                html(" ")
                                delta_sign = "+" if max_delta > 0 else ""
                                # lower duration is better, negative delta is green
                                delta_color = (
                                    COLOR_RED if max_delta > 0 else COLOR_GREEN
                                )
                                html.strong(
                                    _t=f"({delta_sign}{max_delta:.2f}s)",
                                    style=f"color: {delta_color};",
                                )

                # ROW 5: Success Count
                baseline_success = baseline_metrics.get(KEY_SUCCESS_COUNT, 0)
                current_success = current_metrics.get(KEY_SUCCESS_COUNT, 0)
                with html.tr():
                    html.td(_t="Successful Requests", klass="field-label")
                    with html.td(style="text-align: center;"):
                        # higher is better
                        if baseline_success > 0 and (
                            current_success == 0 or baseline_success > current_success
                        ):
                            html.strong(
                                _t=f"{baseline_success:,}",
                                style=f"color: {COLOR_GREEN};",
                            )
                        elif baseline_success > 0:
                            html(f"{baseline_success:,}")
                        else:
                            html("0")

                    with html.td(style="text-align: center;"):
                        if current_success > 0 and (
                            baseline_success == 0 or current_success > baseline_success
                        ):
                            html.strong(
                                _t=f"{current_success:,}",
                                style=f"color: {COLOR_GREEN};",
                            )
                        elif current_success > 0:
                            html(f"{current_success:,}")
                        else:
                            html("0")

                        # add delta
                        if baseline_success > 0 or current_success > 0:
                            success_delta = current_success - baseline_success
                            if success_delta != 0:
                                html(" ")
                                delta_sign = "+" if success_delta > 0 else ""
                                # higher success is better, positive delta is green
                                delta_color = (
                                    COLOR_GREEN if success_delta > 0 else COLOR_RED
                                )
                                html.strong(
                                    _t=f"({delta_sign}{success_delta:,})",
                                    style=f"color: {delta_color};",
                                )

                # ROW 6: Failure Count
                baseline_failure = baseline_metrics.get(KEY_FAILURE_COUNT, 0)
                current_failure = current_metrics.get(KEY_FAILURE_COUNT, 0)
                with html.tr():
                    html.td(_t="Failed Requests", klass="field-label")
                    with html.td(style="text-align: center;"):
                        # lower is better
                        if baseline_failure > 0 and (
                            current_failure == 0 or baseline_failure < current_failure
                        ):
                            html.strong(
                                _t=f"{baseline_failure:,}",
                                style=f"color: {COLOR_GREEN};",
                            )
                        elif baseline_failure > 0:
                            html(f"{baseline_failure:,}")
                        else:
                            html("0")

                    with html.td(style="text-align: center;"):
                        if current_failure > 0 and (
                            baseline_failure == 0 or current_failure < baseline_failure
                        ):
                            html.strong(
                                _t=f"{current_failure:,}",
                                style=f"color: {COLOR_GREEN};",
                            )
                        elif current_failure > 0:
                            html(f"{current_failure:,}")
                        else:
                            html("0")

                        # add delta
                        if baseline_failure > 0 or current_failure > 0:
                            failure_delta = current_failure - baseline_failure
                            if failure_delta != 0:
                                html(" ")
                                delta_sign = "+" if failure_delta > 0 else ""
                                # lower failure is better, negative delta is green
                                delta_color = (
                                    COLOR_RED if failure_delta > 0 else COLOR_GREEN
                                )
                                html.strong(
                                    _t=f"({delta_sign}{failure_delta:,})",
                                    style=f"color: {delta_color};",
                                )

                # ROW 7: Retry Count
                baseline_retry = baseline_metrics.get(KEY_RETRY_COUNT, 0)
                current_retry = current_metrics.get(KEY_RETRY_COUNT, 0)
                if baseline_retry > 0 or current_retry > 0:
                    with html.tr():
                        html.td(_t="Retried Requests", klass="field-label")
                        with html.td(style="text-align: center;"):
                            html(f"{baseline_retry:,}")
                        with html.td(style="text-align: center;"):
                            html(f"{current_retry:,}")

                # ROW 8: Timeout Count
                baseline_timeout = baseline_metrics.get(KEY_TIMEOUT_COUNT, 0)
                current_timeout = current_metrics.get(KEY_TIMEOUT_COUNT, 0)
                if baseline_timeout > 0 or current_timeout > 0:
                    with html.tr():
                        html.td(_t="Timed Out Requests", klass="field-label")
                        with html.td(style="text-align: center;"):
                            html(f"{baseline_timeout:,}")
                        with html.td(style="text-align: center;"):
                            html(f"{current_timeout:,}")

                # ROW 9: LLM Call Count
                baseline_calls = baseline_metrics.get(KEY_CALL_COUNT, 0)
                current_calls = current_metrics.get(KEY_CALL_COUNT, 0)
                if baseline_calls > 0 or current_calls > 0:
                    with html.tr():
                        html.td(_t="LLM Calls", klass="field-label")
                        with html.td(style="text-align: center;"):
                            html(f"{baseline_calls:,}")
                        with html.td(style="text-align: center;"):
                            html(f"{current_calls:,}")

                # ROW 10: Input Tokens
                baseline_in_tokens = baseline_metrics.get(KEY_INPUT_TOKENS, 0)
                current_in_tokens = current_metrics.get(KEY_INPUT_TOKENS, 0)
                if baseline_in_tokens > 0 or current_in_tokens > 0:
                    with html.tr():
                        html.td(_t="Input Tokens", klass="field-label")
                        with html.td(style="text-align: center;"):
                            html(f"{baseline_in_tokens:,}")
                        with html.td(style="text-align: center;"):
                            html(f"{current_in_tokens:,}")

                            # add delta (no color)
                            if baseline_in_tokens > 0 or current_in_tokens > 0:
                                in_tokens_delta = current_in_tokens - baseline_in_tokens
                                if in_tokens_delta != 0:
                                    html(" ")
                                    delta_sign = "+" if in_tokens_delta > 0 else ""
                                    html(f"({delta_sign}{in_tokens_delta:,})")

                # ROW 11: Output Tokens
                baseline_out_tokens = baseline_metrics.get(KEY_OUTPUT_TOKENS, 0)
                current_out_tokens = current_metrics.get(KEY_OUTPUT_TOKENS, 0)
                if baseline_out_tokens > 0 or current_out_tokens > 0:
                    with html.tr():
                        html.td(_t="Output Tokens", klass="field-label")
                        with html.td(style="text-align: center;"):
                            html(f"{baseline_out_tokens:,}")
                        with html.td(style="text-align: center;"):
                            html(f"{current_out_tokens:,}")

                            # add delta (no color)
                            if baseline_out_tokens > 0 or current_out_tokens > 0:
                                out_tokens_delta = (
                                    current_out_tokens - baseline_out_tokens
                                )
                                if out_tokens_delta != 0:
                                    html(" ")
                                    delta_sign = "+" if out_tokens_delta > 0 else ""
                                    html(f"({delta_sign}{out_tokens_delta:,})")

                # ROW 12: Tokens Per Second
                baseline_tps = baseline_metrics.get(KEY_TOKENS_PER_SECOND, 0.0)
                current_tps = current_metrics.get(KEY_TOKENS_PER_SECOND, 0.0)
                if baseline_tps > 0 or current_tps > 0:
                    with html.tr():
                        html.td(_t="Tokens Per Second", klass="field-label")
                        with html.td(style="text-align: center;"):
                            # higher is better
                            if baseline_tps > 0 and (
                                current_tps == 0 or baseline_tps > current_tps
                            ):
                                html.strong(
                                    _t=f"{baseline_tps:.2f}",
                                    style=f"color: {COLOR_GREEN};",
                                )
                            elif baseline_tps > 0:
                                html(f"{baseline_tps:.2f}")
                            else:
                                html("0.00")

                        with html.td(style="text-align: center;"):
                            if current_tps > 0 and (
                                baseline_tps == 0 or current_tps > baseline_tps
                            ):
                                html.strong(
                                    _t=f"{current_tps:.2f}",
                                    style=f"color: {COLOR_GREEN};",
                                )
                            elif current_tps > 0:
                                html(f"{current_tps:.2f}")
                            else:
                                html("0.00")

                            # add delta
                            if baseline_tps > 0 or current_tps > 0:
                                tps_delta = current_tps - baseline_tps
                                if abs(tps_delta) > 0.01:
                                    html(" ")
                                    delta_sign = "+" if tps_delta > 0 else ""
                                    # higher tps is better, positive delta is green
                                    delta_color = (
                                        COLOR_GREEN if tps_delta > 0 else COLOR_RED
                                    )
                                    html.strong(
                                        _t=f"({delta_sign}{tps_delta:.2f})",
                                        style=f"color: {delta_color};",
                                    )

                # ROW 13: Time To First Token
                baseline_ttft = baseline_metrics.get(KEY_TIME_TO_FIRST_TOKEN, 0.0)
                current_ttft = current_metrics.get(KEY_TIME_TO_FIRST_TOKEN, 0.0)
                if baseline_ttft > 0 or current_ttft > 0:
                    with html.tr():
                        html.td(_t="Time To First Token (s)", klass="field-label")
                        with html.td(style="text-align: center;"):
                            # lower is better
                            if baseline_ttft > 0 and (
                                current_ttft == 0 or baseline_ttft < current_ttft
                            ):
                                html.strong(
                                    _t=f"{baseline_ttft:.2f}",
                                    style=f"color: {COLOR_GREEN};",
                                )
                            elif baseline_ttft > 0:
                                html(f"{baseline_ttft:.2f}")
                            else:
                                html("0.00")

                        with html.td(style="text-align: center;"):
                            if current_ttft > 0 and (
                                baseline_ttft == 0 or current_ttft < baseline_ttft
                            ):
                                html.strong(
                                    _t=f"{current_ttft:.2f}",
                                    style=f"color: {COLOR_GREEN};",
                                )
                            elif current_ttft > 0:
                                html(f"{current_ttft:.2f}")
                            else:
                                html("0.00")

                            # add delta
                            if baseline_ttft > 0 or current_ttft > 0:
                                ttft_delta = current_ttft - baseline_ttft
                                if abs(ttft_delta) > 0.01:
                                    html(" ")
                                    delta_sign = "+" if ttft_delta > 0 else ""
                                    # lower ttft is better, negative delta is green
                                    delta_color = (
                                        COLOR_RED if ttft_delta > 0 else COLOR_GREEN
                                    )
                                    html.strong(
                                        _t=f"({delta_sign}{ttft_delta:.2f}s)",
                                        style=f"color: {delta_color};",
                                    )

            # LEGEND for technical metrics table
            html.p(_t="Legend:", style="margin-top: 20px; margin-bottom: 10px;")
            with html.ul(style="margin-left: 20px; line-height: 1.8;"):
                with html.li():
                    html.strong(_t="Total Cost (USD)")
                    html(
                        ": Total aggregated cost across all test cases. "
                        "Lower is better. "
                    )
                with html.li():
                    html.strong(_t="Avg Duration (s)")
                    html(": Average request duration in seconds. Lower is better. ")
                with html.li():
                    html.strong(_t="Min Duration (s)")
                    html(": Minimum request duration in seconds. Lower is better. ")
                with html.li():
                    html.strong(_t="Max Duration (s)")
                    html(": Maximum request duration in seconds. Lower is better. ")
                with html.li():
                    html.strong(_t="Successful Requests")
                    html(
                        ": Count of successfully completed LLM requests. "
                        "Higher is better. "
                    )
                with html.li():
                    html.strong(_t="Failed Requests")
                    html(": Count of failed LLM requests. Lower is better. ")
                with html.li():
                    html.strong(_t="Retried Requests")
                    html(
                        ": Count of requests that required retries. "
                        "Displayed only when non-zero."
                    )
                with html.li():
                    html.strong(_t="Timed Out Requests")
                    html(
                        ": Count of requests that exceeded timeout limits. "
                        "Displayed only when non-zero."
                    )
                with html.li():
                    html.strong(_t="LLM Calls")
                    html(
                        ": Total count of LLM API calls made. "
                        "Displayed only when non-zero."
                    )
                with html.li():
                    html.strong(_t="Input Tokens")
                    html(
                        ": Total input tokens sent to the LLM. "
                        "Displayed only when non-zero. "
                    )
                with html.li():
                    html.strong(_t="Output Tokens")
                    html(
                        ": Total output tokens generated by the LLM. "
                        "Displayed only when non-zero. "
                    )
                with html.li():
                    html.strong(_t="Tokens Per Second")
                    html(
                        ": Token generation throughput. "
                        "Higher is better (faster generation). "
                    )
                with html.li():
                    html.strong(_t="Time To First Token (s)")
                    html(
                        ": Latency before first token is received. "
                        "Lower is better (faster response initiation). "
                    )

    def _html_model_section(self, html, model_pair_key: str, model_diffs: list):
        """Generate HTML section for a model with overview and test cases.

        Parameters
        ----------
        html : airium.Airium
            Airium HTML instance.
        model_pair_key : str
            The unique model pair key identifier (baseline_model_key|current_model_key).
        model_diffs : list
            List of (idx, diff) tuples for this model.

        """
        # get model keys from first diff (all diffs in this section have same models)
        if not model_diffs:
            return

        _, first_diff = model_diffs[0]
        baseline_model_key = first_diff.baseline_test_case.get(
            _explanations_cmp.KEY_MODEL_KEY, ""
        )
        current_model_key = first_diff.current_test_case.get(
            _explanations_cmp.KEY_MODEL_KEY, ""
        )

        baseline_explainable_model = self.diff_obj._get_explainable_model_by_key(
            baseline_model_key, is_baseline=True
        )
        current_explainable_model = self.diff_obj._get_explainable_model_by_key(
            current_model_key, is_baseline=False
        )

        baseline_model_name = (
            baseline_explainable_model.llm_model_name
            if baseline_explainable_model
            else baseline_model_key
        )
        current_model_name = (
            current_explainable_model.llm_model_name
            if current_explainable_model
            else current_model_key
        )
        # format display names with model type
        baseline_display_name = self._format_model_display_name(
            baseline_explainable_model, baseline_model_name
        )
        current_display_name = self._format_model_display_name(
            current_explainable_model, current_model_name
        )

        with html.div(klass="model-header", id=f"model-{model_pair_key}"):
            with html.h2():
                html("Models Comparison: ")
                html.b(_t=f"{baseline_display_name}")
                html(" vs ")
                html.b(_t=f"{current_display_name}")

        # get metrics metadata from diff object
        metrics_meta = self.diff_obj.metrics_meta
        if not metrics_meta:
            raise ValueError(
                "Metrics metadata is not available in the evaluation results diff. "
                "This should have been set when the diff was created from the "
                "baseline explanation's explainer._metrics_meta."
            )

        # GENERATE comparison statistics first to get flipped metrics info
        comparison_stats = EvalResultsDiffHtml._calculate_model_cmp_stats(
            model_diffs,
            metrics_meta,
            baseline_explainable_model,
            current_explainable_model,
        )

        # store stats to diff object for test access
        self.diff_obj._stats = comparison_stats

        # GENERATE Models Comparison Summary - THE MOST IMPORTANT CONCLUSION
        recommendation_result = EvalResultsDiffHtml._calculate_recommendation(
            comparison_stats, metrics_meta
        )

        with html.div(
            klass="comparison-summary",
            style=(
                f"background-color: {COLOR_OFF_WHITE}; border-left: 4px solid #000000; "
                "padding: 20px; margin: 20px 0; font-size: 1.1em;"
            ),
        ):
            html.h3(_t="Summary", style="margin-top: 0; color: #000000;")

            if recommendation_result["winner"] == "current":
                # recommend current model
                with html.div(
                    style=(
                        f"background-color: #d4edda; border: 2px solid {COLOR_GREEN}; "
                        "border-radius: 5px; padding: 15px; margin-top: 10px;"
                    )
                ):
                    with html.p(
                        style=(
                            "margin: 0; font-weight: bold; color: #155724; "
                            "font-size: 1.2em;"
                        )
                    ):
                        html("✓ RECOMMENDATION: Use ")
                        html.strong(_t="CURRENT")
                        html(" model ")
                        html.strong(_t=current_display_name)
                    with html.p(style="margin: 10px 0 0 0; color: #155724;"):
                        html(recommendation_result["explanation"])
            elif recommendation_result["winner"] == "baseline":
                # recommend baseline model
                with html.div(
                    style=(
                        f"background-color: #d4edda; border: 2px solid {COLOR_GREEN}; "
                        "border-radius: 5px; padding: 15px; margin-top: 10px;"
                    )
                ):
                    with html.p(
                        style=(
                            "margin: 0; font-weight: bold; color: #155724; "
                            "font-size: 1.2em;"
                        )
                    ):
                        html("✓ RECOMMENDATION: Use ")
                        html.strong(_t="BASELINE")
                        html(" model ")
                        html.strong(_t=baseline_display_name)
                    with html.p(style="margin: 10px 0 0 0; color: #155724;"):
                        html(recommendation_result["explanation"])
            else:
                # no clear winner
                with html.div(
                    style="background-color: #e7f3ff; border: 2px solid #17a2b8; "
                    "border-radius: 5px; padding: 15px; margin-top: 10px;"
                ):
                    with html.p(
                        style=(
                            "margin: 0; font-weight: bold; color: #0c5460; "
                            "font-size: 1.2em;"
                        )
                    ):
                        html("⚖ RECOMMENDATION: No clear winner")
                    with html.p(style="margin: 10px 0 0 0; color: #0c5460;"):
                        html(recommendation_result["explanation"])

            html.h5(_t="Details:", style="margin-top: 1em;")
            with html.ul(style=" padding-left: 3em;"):
                with html.li(style="margin: 8px 0;"):
                    html.a(
                        href=f"#models-overview-{model_pair_key}",
                        _t="Models Overview",
                    )
                with html.li(style="margin: 8px 0;"):
                    html.a(
                        href=f"#models-comparison-{model_pair_key}",
                        _t="Models Comparison",
                    )
                with html.li(style="margin: 8px 0;"):
                    html.a(
                        href=f"#model-config-comparison-{model_pair_key}",
                        _t="Model Configuration Comparison",
                    )
                with html.li(style="margin: 8px 0;"):
                    html.a(
                        href=f"#technical-metrics-{model_pair_key}",
                        _t="Technical Performance Metrics",
                    )
                with html.li(style="margin: 8px 0;"):
                    html.a(
                        href=f"#top-test-cases-{model_pair_key}",
                        _t="Top Test Cases by Metric Changes",
                    )
                # link to first test case
                if model_diffs:
                    first_idx, _ = model_diffs[0]
                    with html.li(style="margin: 8px 0;"):
                        html.a(
                            href=f"#test-case-{first_idx}",
                            _t="Test Cases",
                        )

        with html.div(klass="model-overview", id=f"models-overview-{model_pair_key}"):
            html.h3(_t="Models Overview")

            model_fields = [
                ("Key", "key"),
                ("Name", "name"),
                ("LLM Model Name", "llm_model_name"),
                ("Model Type", "model_type"),
            ]

            # RAG-specific fields
            if baseline_explainable_model and isinstance(
                baseline_explainable_model, models.ExplainableRagModel
            ):
                model_fields.extend(
                    [
                        ("Collection ID", "collection_id"),
                        ("Collection Name", "collection_name"),
                        ("Documents Count", "documents"),
                    ]
                )

            with html.table(klass="model-overview-table"):
                # table header
                with html.tr():
                    html.th(_t="", style="text-align: center;")
                    html.th(_t="Baseline Model", style="text-align: center;")
                    html.th(_t="Current Model", style="text-align: center;")

                # table rows
                for field_label, field_key in model_fields:
                    baseline_value = (
                        getattr(baseline_explainable_model, field_key, "N/A")
                        if baseline_explainable_model
                        else "N/A"
                    )
                    current_value = (
                        getattr(current_explainable_model, field_key, "N/A")
                        if current_explainable_model
                        else "N/A"
                    )

                    # TODO list actual documents in a reasonable way
                    if field_key == "documents":
                        if baseline_value != "N/A" and isinstance(baseline_value, list):
                            baseline_value = len(baseline_value)
                        if current_value != "N/A" and isinstance(current_value, list):
                            current_value = len(current_value)

                    if field_key == "model_type":
                        if baseline_value != "N/A":
                            baseline_value = (
                                str(baseline_value.name)
                                if hasattr(baseline_value, "name")
                                else str(baseline_value)
                            )
                        if current_value != "N/A":
                            current_value = (
                                str(current_value.name)
                                if hasattr(current_value, "name")
                                else str(current_value)
                            )

                    # only show row if at least one value is not N/A
                    if baseline_value not in ["N/A", "", None] or current_value not in [
                        "N/A",
                        "",
                        None,
                    ]:
                        values_same = baseline_value == current_value
                        value_class = "value-same" if values_same else "value-different"

                        with html.tr():
                            html.td(_t=field_label, klass="field-label")
                            with html.td(klass=value_class):
                                html(str(baseline_value))
                            with html.td(klass=value_class):
                                html(str(current_value))

            # GENERATE model configuration comparison
            EvalResultsDiffHtml._html_model_config_comparison(
                html,
                baseline_explainable_model,
                current_explainable_model,
                model_key=model_pair_key,
            )

        # GENERATE comparison table for this model (stats already calculated above)
        EvalResultsDiffHtml._html_model_comparison_table(
            html, comparison_stats, metrics_meta, model_key=model_pair_key
        )

        # GENERATE leaderboard for this model comparison
        diff_to_idx = {diff: idx for idx, diff in model_diffs}
        diff_list = [diff for idx, diff in model_diffs]
        leaderboard = self.diff_obj._generate_leaderboard(
            diff_list, top_n=1_000, metrics_meta=metrics_meta
        )
        EvalResultsDiffHtml._html_leaderboard(
            html, leaderboard, diff_to_idx, metrics_meta, model_key=model_pair_key
        )

        # GENERATE test case sections for the model
        for idx, diff in model_diffs:
            self._html_diff_section(html, diff, idx)

    def _html_diff_section(
        self, html, diff: _explanations_cmp.EvalResultDiff, section_num: int
    ):
        with html.div(
            klass="diff-section w3-card-4 w3-white", id=f"test-case-{section_num}"
        ):
            with html.div(klass="question-header"):
                html.h3(_t="Test Case")
                with html.p():
                    html("Question: ")
                    html.b(_t=f"{diff.question}")

            # prepare metrics data
            skip_list = [
                _explanations_cmp.KEY_KEY,
                _explanations_cmp.KEY_ACTUAL_DURATION,
                _explanations_cmp.KEY_ACTUAL_OUTPUT,
                _explanations_cmp.KEY_ACTUAL_OUTPUT_META,
                _explanations_cmp.KEY_CATEGORIES,
                _explanations_cmp.KEY_CONTEXT,
                _explanations_cmp.KEY_CORPUS,
                _explanations_cmp.KEY_COST,
                _explanations_cmp.KEY_EXPECTED_OUTPUT,
                _explanations_cmp.KEY_INPUT,
                _explanations_cmp.KEY_METRICS,
                _explanations_cmp.KEY_METRICS_META,
                _explanations_cmp.KEY_MODEL_KEY,
                _explanations_cmp.KEY_OUTPUT_CONDITION,
                _explanations_cmp.KEY_OUTPUT_CONSTRAINTS,
                _explanations_cmp.KEY_RELATIONSHIPS,
                _explanations_cmp.KEY_RESULT_ERR_MSG,
                _explanations_cmp.KEY_TEST_CASE_KEY,
                _explanations_cmp.KEY_TEST_KEY,
            ]

            baseline_metrics = {
                k: v for k, v in diff.baseline_test_case.items() if k not in skip_list
            }
            current_metrics = {
                k: v for k, v in diff.current_test_case.items() if k not in skip_list
            }

            # TABLE 1: Metrics comparison (separate table)
            if baseline_metrics or current_metrics:
                # extract metrics metadata from diff object
                metrics_meta = self.diff_obj.metrics_meta

                with html.table(klass="comparison-table", style="margin-bottom: 20px;"):
                    # header
                    with html.tr():
                        html.th(_t="Metric", style="text-align: center;")
                        html.th(_t="Baseline Score", style="text-align: center;")
                        html.th(_t="Current Score", style="text-align: center;")

                    # metrics rows
                    all_metric_keys = sorted(
                        set(baseline_metrics.keys()) | set(current_metrics.keys())
                    )
                    for metric_key in all_metric_keys:
                        baseline_value = baseline_metrics.get(metric_key, None)
                        current_value = current_metrics.get(metric_key, None)

                        # skip if both values are None
                        if baseline_value is None and current_value is None:
                            continue

                        # check if metric flipped
                        is_flipped = metric_key in diff.diff_flipped_metrics

                        # determine if higher is better - MUST have metrics_meta
                        if not metrics_meta:
                            raise ValueError(
                                f"Metrics metadata is required for wins-based "
                                f"comparison but is missing. Cannot determine winner "
                                f"for metric '{metric_key}' without knowing if higher "
                                f"or lower is better."
                            )

                        if metric_key not in metrics_meta:
                            raise ValueError(
                                f"Metrics metadata is missing for metric "
                                f"'{metric_key}'. Cannot determine winner without "
                                f"knowing if higher or lower is better. Available "
                                f"metrics in metadata: {list(metrics_meta.keys())}"
                            )

                        metric_meta_obj = metrics_meta[metric_key]

                        if hasattr(metric_meta_obj, "higher_is_better"):
                            higher_is_better = metric_meta_obj.higher_is_better
                        elif isinstance(metric_meta_obj, dict):
                            if "higher_is_better" not in metric_meta_obj:
                                raise ValueError(
                                    f"Metric metadata for '{metric_key}' is missing "
                                    f"the required 'higher_is_better' field. "
                                    f"Metadata: {metric_meta_obj}"
                                )
                            higher_is_better = metric_meta_obj["higher_is_better"]
                        else:
                            raise ValueError(
                                f"Metric metadata for '{metric_key}' has unexpected "
                                f"type: {type(metric_meta_obj)}. Expected object with "
                                f"'higher_is_better' attribute or dict with "
                                f"'higher_is_better' key."
                            )

                        # determine who won this metric
                        baseline_won = False
                        current_won = False
                        if (
                            baseline_value is not None
                            and current_value is not None
                            and isinstance(baseline_value, (int, float))
                            and isinstance(current_value, (int, float))
                            and baseline_value != current_value
                        ):
                            if higher_is_better:
                                baseline_won = baseline_value > current_value
                                current_won = current_value > baseline_value
                            else:  # lower is better
                                baseline_won = baseline_value < current_value
                                current_won = current_value < baseline_value

                        with html.tr():
                            with html.td(klass="field-label"):
                                if is_flipped:
                                    html.strong(
                                        _t=metric_key, style=f"color: {COLOR_RED};"
                                    )
                                else:
                                    html(metric_key)

                            # baseline score with color
                            with html.td(style="text-align: center;"):
                                if baseline_value is not None:
                                    if baseline_won:
                                        html.strong(
                                            _t=f"{baseline_value:.8f}",
                                            style=f"color: {COLOR_GREEN};",
                                        )
                                    elif current_won:
                                        html.span(
                                            _t=f"{baseline_value:.8f}",
                                            style=f"color: {COLOR_RED};",
                                        )
                                    else:
                                        html(f"{baseline_value:.8f}")
                                else:
                                    html("N/A")

                            # current score with delta
                            with html.td(style="text-align: center;"):
                                if current_value is not None:
                                    html(f"{current_value:.8f}")

                                    # show difference if both values exist
                                    if baseline_value is not None:
                                        diff_value = current_value - baseline_value
                                        if diff_value != 0:
                                            html(" ")
                                            diff_sign = "+" if diff_value > 0 else ""

                                            # color based on higher_is_better
                                            if is_flipped:
                                                diff_color = COLOR_RED
                                            elif (
                                                higher_is_better and diff_value > 0
                                            ) or (
                                                not higher_is_better and diff_value < 0
                                            ):
                                                diff_color = COLOR_GREEN
                                            elif (
                                                higher_is_better and diff_value < 0
                                            ) or (
                                                not higher_is_better and diff_value > 0
                                            ):
                                                diff_color = COLOR_ORANGE
                                            else:
                                                diff_color = ""

                                            if diff_color:
                                                html.strong(
                                                    _t=f"({diff_sign}{diff_value:.8f})",
                                                    style=f"color: {diff_color};",
                                                )
                                            else:
                                                html(f"({diff_sign}{diff_value:.8f})")
                                else:
                                    html("N/A")

                # LEGEND for metrics table
                html.h4(_t="Legend:", style="margin-top: 10px; margin-bottom: 10px;")
                with html.ul(style="margin-left: 20px; line-height: 1.8;"):
                    with html.li():
                        html.strong(_t="Metric")
                        html(
                            ": The evaluation metric name. Shown in bold red if the "
                            "metric flipped between pass/fail status."
                        )
                    with html.li():
                        html.strong(_t="Baseline Score")
                        html(
                            ": The metric value for the baseline model. Shown in "
                            "bold green if baseline won, red if lost."
                        )
                    with html.li():
                        html.strong(_t="Current Score")
                        html(": The metric value for the current model, followed by ")
                        html("the delta from baseline in parentheses. ")
                    with html.li():
                        html.strong(_t="Delta Colors")
                        html(": ")
                        html.span(
                            _t="Green",
                            style=f"color: {COLOR_GREEN}; font-weight: bold;",
                        )
                        html(" = improvement, ")
                        html.span(
                            _t="Orange",
                            style=f"color: {COLOR_ORANGE}; font-weight: bold;",
                        )
                        html(" = regression, ")
                        html.span(
                            _t="Red", style=f"color: {COLOR_RED}; font-weight: bold;"
                        )
                        html(" = metric flipped to fail.")
                    with html.li():
                        html.strong(_t="Winner Determination")
                        html(
                            ": For each metric, the winner is determined by comparing "
                        )
                        html("scores while considering the metric's directionality ")
                        html("(higher is better vs. lower is better).")

            # TABLE 2: Test case details (original table without metrics)
            with html.table(klass="comparison-table"):
                # header
                with html.tr():
                    html.th(_t="", style="text-align: center;")
                    html.th(_t="Baseline Result", style="text-align: center;")
                    html.th(_t="Current Result", style="text-align: center;")

                # rows

                # get model names
                baseline_model_key = diff.baseline_test_case.get(
                    _explanations_cmp.KEY_MODEL_KEY, "N/A"
                )
                current_model_key = diff.current_test_case.get(
                    _explanations_cmp.KEY_MODEL_KEY, "N/A"
                )

                baseline_explainable_model = (
                    self.diff_obj._get_explainable_model_by_key(
                        baseline_model_key, is_baseline=True
                    )
                )
                current_explainable_model = self.diff_obj._get_explainable_model_by_key(
                    current_model_key, is_baseline=False
                )

                baseline_model_name = (
                    baseline_explainable_model.llm_model_name
                    if baseline_explainable_model
                    else baseline_model_key
                )
                current_model_name = (
                    current_explainable_model.llm_model_name
                    if current_explainable_model
                    else current_model_key
                )
                # format display names with model type
                baseline_display_name = self._format_model_display_name(
                    baseline_explainable_model, baseline_model_name
                )
                current_display_name = self._format_model_display_name(
                    current_explainable_model, current_model_name
                )

                # model row
                with html.tr():
                    with html.td(klass="field-label"):
                        html.strong(_t="Model")
                    with html.td():
                        html(f"{baseline_display_name}")
                    with html.td():
                        html(f"{current_display_name}")

                # question
                with html.tr():
                    with html.td(klass="field-label"):
                        html.strong(_t="Question")
                    with html.td(colspan="2"):
                        html(diff.question)

                # expected answer
                with html.tr():
                    with html.td(klass="field-label"):
                        html.strong(_t="Expected answer")
                    with html.td(colspan="2"):
                        html(diff.expected_answer)

                # actual answer row
                with html.tr():
                    with html.td(klass="field-label"):
                        html.strong(_t="Actual Answer")
                    with html.td():
                        EvalResultsDiffHtml._html_colorized_sentences(
                            html,
                            diff.baseline_diff_actual_output_meta,
                            diff.baseline_test_case.get(
                                _explanations_cmp.KEY_ACTUAL_OUTPUT, ""
                            ),
                        )
                    with html.td():
                        EvalResultsDiffHtml._html_colorized_sentences(
                            html,
                            diff.current_diff_actual_output_meta,
                            diff.current_test_case.get(
                                _explanations_cmp.KEY_ACTUAL_OUTPUT, ""
                            ),
                        )

                # retrieved Context row
                with html.tr():
                    with html.td(klass="field-label"):
                        html.strong(_t="Retrieved Context")
                    with html.td():
                        EvalResultsDiffHtml._html_context_chunks(
                            html,
                            diff.baseline_diff_retrieved_context,
                            diff.baseline_test_case.get(
                                _explanations_cmp.KEY_CONTEXT, []
                            ),
                        )
                    with html.td():
                        EvalResultsDiffHtml._html_context_chunks(
                            html,
                            diff.current_diff_retrieved_context,
                            diff.current_test_case.get(
                                _explanations_cmp.KEY_CONTEXT, []
                            ),
                        )

    @staticmethod
    def _similarity_to_color(similarity: float) -> str:
        """Convert similarity score [0.0, 1.0] to color gradient with non-linear
        degradation.

        Color gradient goes from:
        - 1.0-0.95 (very high similarity): green (#d4edda)
        - 0.95-0.90 (high similarity): green to green-yellow gradient
        - 0.90-0.80 (good similarity): yellow gradient
        - 0.80-0.70 (medium similarity): yellow to orange gradient
        - 0.70-0.50 (low similarity): orange gradient
        - <0.50 (very low similarity): orange to red gradient (#f8d7da)

        Parameters
        ----------
        similarity : float
            Similarity score in range [0.0, 1.0].

        Returns
        -------
        str :
            RGB color as hex string.

        """
        # clamp to [0.0, 1.0]
        similarity = max(0.0, min(1.0, similarity))

        # define color anchors (R, G, B) with faster degradation
        if similarity >= 0.95:
            # 1.0-0.95: Pure green (high similarity)
            # green: #d4edda (212, 237, 218)
            return "#d4edda"
        elif similarity >= 0.90:
            # 0.95-0.90: Green to green-yellow
            # green: #d4edda (212, 237, 218)
            # green-yellow: #dff0aa (223, 240, 170)
            t = (similarity - 0.90) / 0.05
            r = int(212 + (223 - 212) * (1 - t))
            g = int(237 + (240 - 237) * (1 - t))
            b = int(218 - (218 - 170) * (1 - t))
        elif similarity >= 0.80:
            # 0.90-0.80: Yellow (good similarity)
            # green-yellow: #dff0aa (223, 240, 170)
            # yellow: #fff3cd (255, 243, 205)
            t = (similarity - 0.80) / 0.10
            r = int(223 + (255 - 223) * (1 - t))
            g = int(240 + (243 - 240) * (1 - t))
            b = int(170 + (205 - 170) * (1 - t))
        elif similarity >= 0.70:
            # 0.80-0.70: Yellow to orange (medium similarity)
            # yellow: #fff3cd (255, 243, 205)
            # orange: #ffd580 (255, 213, 128)
            t = (similarity - 0.70) / 0.10
            r = 255
            g = int(243 - (243 - 213) * (1 - t))
            b = int(205 - (205 - 128) * (1 - t))
        elif similarity >= 0.50:
            # 0.70-0.50: Orange (low similarity)
            # orange: #ffd580 (255, 213, 128)
            # dark orange: #ffb366 (255, 179, 102)
            t = (similarity - 0.50) / 0.20
            r = 255
            g = int(213 - (213 - 179) * (1 - t))
            b = int(128 - (128 - 102) * (1 - t))
        else:
            # <0.50: Orange to red (very low similarity)
            # dark orange: #ffb366 (255, 179, 102)
            # red: #f8d7da (248, 215, 218)
            # t=0.0 -> red, t=1.0 -> orange
            t = similarity / 0.50
            r = int(248 + (255 - 248) * t)
            g = int(215 + (179 - 215) * t)
            b = int(218 + (102 - 218) * t)

        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def _html_colorized_sentences(html, diff_meta: dict, full_text: str):
        if not diff_meta or not diff_meta.get(_explanations_cmp.KEY_SENTENCES):
            html(full_text or "N/A")
            return

        sentences = diff_meta.get(_explanations_cmp.KEY_SENTENCES, [])
        sentence_similarity = diff_meta.get(
            _explanations_cmp.KEY_SENTENCE_SIMILARITY, {}
        )

        # if no similarity data, fall back to old behavior
        if not sentence_similarity:
            unique_sentences = set(
                diff_meta.get(_explanations_cmp.KEY_UNIQUE_SENTENCES, [])
            )
            for sentence in sentences:
                if sentence in unique_sentences:
                    # RED
                    with html.span(klass="sentence-unique"):
                        html(sentence)
                else:
                    # GREEN
                    with html.span(klass="sentence-common"):
                        html(sentence)
                html(" ")
        else:
            # use similarity scores to color sentences with gradient
            for sentence in sentences:
                similarity = sentence_similarity.get(sentence, 0.0)
                color = EvalResultsDiffHtml._similarity_to_color(similarity)

                with html.span(
                    style=f"background-color: {color}; padding: 2px 4px; "
                    f"margin: 2px; display: inline-block;",
                    title=f"Similarity: {similarity:.2f}",
                ):
                    html(sentence)
                html(" ")

    @staticmethod
    def _html_metrics(
        html,
        metrics: dict,
        comparison_metrics: dict,
        flipped_metrics: dict,
        metrics_meta: dict | None = None,
    ):
        """Render metrics with color coding showing baseline, current, and delta.

        Parameters
        ----------
        html : airium.Airium
            HTML builder object.
        metrics : dict
            Current metrics to display.
        comparison_metrics : dict
            Comparison metrics (from the other side) to check for changes.
        flipped_metrics : dict
            Dictionary of metrics that flipped (significant changes).
        metrics_meta : dict | None
            Metrics metadata containing higher_is_better information.

        """
        if not metrics:
            html("No metrics available")
            return

        with html.ul(style="margin: 5px 0; padding-left: 20px;"):
            for metric_name, metric_value in metrics.items():
                # check if this metric flipped (significant change)
                is_flipped = metric_name in flipped_metrics

                # check if value changed from comparison
                comparison_value = comparison_metrics.get(metric_name)
                is_changed = (
                    comparison_value is not None and comparison_value != metric_value
                )

                # determine if higher is better for this metric
                higher_is_better = True  # default
                if metrics_meta and metric_name in metrics_meta:
                    metric_meta_obj = metrics_meta[metric_name]
                    if hasattr(metric_meta_obj, "higher_is_better"):
                        higher_is_better = metric_meta_obj.higher_is_better
                    elif isinstance(metric_meta_obj, dict):
                        higher_is_better = metric_meta_obj.get("higher_is_better", True)

                # calculate delta and determine color
                delta = None
                delta_color = ""
                if (
                    is_changed
                    and isinstance(metric_value, (int, float))
                    and isinstance(comparison_value, (int, float))
                ):
                    delta = metric_value - comparison_value
                    # determine color based on higher_is_better and delta sign
                    if delta > 0:
                        # positive change: green if higher is better, else red
                        delta_color = COLOR_GREEN if higher_is_better else COLOR_RED
                    elif delta < 0:
                        # negative change: red if higher is better, else green
                        delta_color = COLOR_RED if higher_is_better else COLOR_GREEN

                if is_flipped:
                    css_class = "metric-flipped"
                    icon = " ⚠"
                elif is_changed:
                    css_class = "metric-changed"
                    icon = " 👁"
                else:
                    css_class = ""
                    icon = ""

                with html.li():
                    html(f"{icon} {metric_name}:")
                    html.br()
                    # show value with delta if changed
                    if delta is not None:
                        with html.span(klass=css_class):
                            html(f"{metric_value} ")
                        diff_sign = "+" if delta > 0 else ""
                        if delta_color:
                            html.strong(
                                _t=f"({diff_sign}{delta:.8f})",
                                style=f"color: {delta_color};",
                            )
                        else:
                            with html.span(klass=css_class):
                                html(f"({diff_sign}{delta:.8f})")
                    else:
                        html(f"{metric_value}")

    @staticmethod
    def _html_context_chunks(html, diff_meta: dict, context_list: list):
        if not context_list:
            html("No context available")
            return

        if not diff_meta or not diff_meta.get(_explanations_cmp.KEY_CHUNKS):
            for chunk in context_list:
                with html.div(klass="context-chunk"):
                    html(chunk)
            return

        chunk_similarity = diff_meta.get(_explanations_cmp.KEY_CHUNK_SIMILARITY, {})

        # if no similarity data, fall back to old behavior
        if not chunk_similarity:
            unique_chunks = set(diff_meta.get(_explanations_cmp.KEY_UNIQUE_CHUNKS, []))
            for chunk in context_list:
                if chunk in unique_chunks:
                    # RED
                    with html.div(klass="context-chunk context-unique"):
                        html(chunk)
                else:
                    # GREEN
                    with html.div(klass="context-chunk context-common"):
                        html(chunk)
        else:
            # use similarity scores to color chunks with gradient
            for chunk in context_list:
                similarity = chunk_similarity.get(chunk, 0.0)
                color = EvalResultsDiffHtml._similarity_to_color(similarity)

                with html.div(
                    klass="context-chunk",
                    style=f"background-color: {color};",
                    title=f"Similarity: {similarity:.2f}",
                ):
                    html(chunk)

    @staticmethod
    def html_h2o_sonar_pitch(brand_h2o_sonar: str) -> str:
        return (
            f"{brand_h2o_sonar} is Python package that enables a holistic, "
            f"low-risk, human-interpretable, fair, and trustable "
            f"approach to machine learning by implementing various "
            f"facets of Responsible AI. "
        )

    @staticmethod
    def html_footer(html, brand_h2o_sonar: str, branding: commons.Branding):
        with html.footer(klass="w3-container w3-padding-32 w3-dark-grey"):
            with html.div(klass="w3-row-padding"):
                with html.div(klass="w3-third"):
                    html.h3(_t=brand_h2o_sonar)
                    if branding == commons.Branding.H2O_SONAR:
                        with html.p():
                            html(
                                EvalResultsDiffHtml.html_h2o_sonar_pitch(
                                    brand_h2o_sonar
                                )
                            )
                with html.div(klass="w3-third"):
                    if branding == commons.Branding.H2O_SONAR:
                        html.h3(_t="Resources")
                    with html.ul(klass="w3-ul XXXw3-hoverable"):
                        if branding == commons.Branding.H2O_SONAR:
                            with html.li(klass="w3-padding-16"):
                                with html.div(
                                    klass="w3-left w3-margin-right", style="width:50px"
                                ):
                                    htmls.html_svg_h2oai_logo(html)
                                html.span(klass="w3-large", _t=brand_h2o_sonar)
                                html.br()
                                with html.span():
                                    html.a(
                                        href="https://github.com/h2oai/h2o-sonar",
                                        _t="GitHub&nbsp;repository",
                                    )
                        with html.li(klass="w3-padding-16"):
                            with html.div(
                                klass="w3-left w3-margin-right", style="width:50px"
                            ):
                                htmls.html_svg_h2oai_logo(html)
                            html.span(klass="w3-large", _t="H2O.ai")
                            html.br()
                            with html.span():
                                html("Democratize AI with")
                                html.a(href="https://h2o.ai/", _t="H2O.ai")
                            html.br()
                            html.br()
                            with html.span():
                                created_str = str(datetime.datetime.now())
                                created_str = created_str[: created_str.index(".")]
                                created_str = f"{created_str} T{time.strftime('%z')}"
                                html("Generated by ")
                                with html.b():
                                    html(f"v{sonar_version}")
                                html(" at ")
                                with html.b():
                                    html(created_str)
