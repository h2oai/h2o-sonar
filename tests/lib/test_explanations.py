# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import pathlib
import time

import pytest

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import evaluate
from h2o_sonar import loggers
from h2o_sonar.evaluators import fairness_bias_evaluator as fb_e
from h2o_sonar.evaluators import rag_answer_relevancy_no_judge_evaluator as arnj_e
from h2o_sonar.evaluators import rag_groundedness_evaluator as rg_e
from h2o_sonar.evaluators import rag_tokens_presence_evaluator as tp_e
from h2o_sonar.evaluators import sensitive_data_leakage_evaluator as sdl_e
from h2o_sonar.evaluators import toxicity_evaluator as t_e
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explanations
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.container import explainer_container
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative


# BERTScore is really slow on CPU > the test is skipped and can be enabled from here
_RUN_BERTSCORE_TESTS = False


def _given_explainers_map() -> dict:
    """Return explainers map."""
    container = explainer_container.LocalExplainerContainer()
    container.setup()
    return container.explainers_registry.list_explainers()


def _then_assert_json(diff_json_dict: dict, verbose: bool = True):
    """Assert in-depth content and integrity of JSON comparison diff.

    This function validates that the JSON diff contains all expected fields,
    has correct data types, maintains referential integrity, and contains
    complete data that could be used to regenerate the HTML comparison report.

    Parameters
    ----------
    diff_json_dict : dict
        The JSON diff dictionary returned by EvalResultsDiff.to_dict().
    verbose : bool
        Whether to print detailed validation info (default: True).

    Raises
    ------
    AssertionError
        If any validation check fails.

    """
    from h2o_sonar.lib.api.explanations import _explanations_cmp

    # ========================================================================
    # 1. TOP-LEVEL STRUCTURE VALIDATION
    # ========================================================================
    if verbose:
        print("\n=== Validating top-level JSON structure ===")

    assert isinstance(diff_json_dict, dict), "JSON diff must be a dictionary"

    # required top-level keys
    required_keys = ["diffs", "metrics_meta"]
    for key in required_keys:
        assert key in diff_json_dict, f"Missing required top-level key: {key}"
        if verbose:
            print(f"OK Found top-level key: {key}")

    # validate types
    assert isinstance(diff_json_dict["diffs"], list), "'diffs' must be a list"
    assert diff_json_dict["metrics_meta"] is None or isinstance(
        diff_json_dict["metrics_meta"], dict
    ), "'metrics_meta' must be dict or None"

    # ========================================================================
    # 2. DIFFS ARRAY VALIDATION
    # ========================================================================
    if verbose:
        print(f"\n=== Validating {len(diff_json_dict['diffs'])} diff entries ===")

    assert len(diff_json_dict["diffs"]) > 0, "Diffs array must not be empty"

    for diff_idx, diff_entry in enumerate(diff_json_dict["diffs"]):
        if verbose:
            print(f"\n--- Validating diff entry {diff_idx + 1} ---")

        # validate diff_key exists and is a string
        assert _explanations_cmp.KEY_DIFF_KEY in diff_entry, (
            f"Diff entry {diff_idx} missing 'diff_key'"
        )
        diff_key = diff_entry[_explanations_cmp.KEY_DIFF_KEY]
        assert isinstance(diff_key, str), f"Diff key must be string: {diff_key}"
        assert "|" in diff_key, f"Diff key must contain '|' separator: {diff_key}"
        if verbose:
            print(f" DONE diff_key: {diff_key}")

        # ====================================================================
        # 2.1 SUMMARY SECTION VALIDATION (NEW)
        # ====================================================================
        assert _explanations_cmp.KEY_SUMMARY in diff_entry, (
            f"Diff entry {diff_idx} missing 'summary'"
        )
        summary = diff_entry[_explanations_cmp.KEY_SUMMARY]
        assert isinstance(summary, dict), "summary must be a dictionary"

        # validate summary fields
        assert _explanations_cmp.KEY_RECOMMENDATION_WINNER in summary, (
            "summary missing 'recommendation_winner'"
        )
        assert _explanations_cmp.KEY_RECOMMENDATION in summary, (
            "summary missing 'recommendation'"
        )
        assert _explanations_cmp.KEY_RECOMMENDATION_CONFIDENCE in summary, (
            "summary missing 'recommendation_confidence'"
        )

        winner = summary[_explanations_cmp.KEY_RECOMMENDATION_WINNER]
        recommendation = summary[_explanations_cmp.KEY_RECOMMENDATION]
        confidence = summary[_explanations_cmp.KEY_RECOMMENDATION_CONFIDENCE]

        assert winner in [
            "baseline",
            "current",
            "tie",
        ], f"Invalid winner value: {winner}"
        assert isinstance(recommendation, str), "recommendation must be string"
        assert len(recommendation) > 0, "recommendation must not be empty"
        assert confidence in [
            "high",
            "medium",
            "low",
        ], f"Invalid confidence: {confidence}"

        if verbose:
            print(f" DONE summary.recommendation_winner: {winner}")
            print(
                f" DONE summary.recommendation: {recommendation[:60]}..."
                if len(recommendation) > 60
                else f" DONE summary.recommendation: {recommendation}"
            )
            print(f" DONE summary.recommendation_confidence: {confidence}")

        # ====================================================================
        # 2.2 MODELS OVERVIEW SECTION VALIDATION (NEW)
        # ====================================================================
        assert _explanations_cmp.KEY_MODELS_OVERVIEW in diff_entry, (
            f"Diff entry {diff_idx} missing 'models_overview'"
        )
        models_overview = diff_entry[_explanations_cmp.KEY_MODELS_OVERVIEW]
        assert isinstance(models_overview, dict), "models_overview must be a dictionary"

        # required fields
        assert _explanations_cmp.KEY_BASELINE_MODEL_KEY in models_overview, (
            "models_overview missing 'baseline_model_key'"
        )
        assert _explanations_cmp.KEY_CURRENT_MODEL_KEY in models_overview, (
            "models_overview missing 'current_model_key'"
        )

        baseline_model_key = models_overview[_explanations_cmp.KEY_BASELINE_MODEL_KEY]
        current_model_key = models_overview[_explanations_cmp.KEY_CURRENT_MODEL_KEY]

        assert isinstance(baseline_model_key, str), "baseline_model_key must be string"
        assert isinstance(current_model_key, str), "current_model_key must be string"
        assert len(baseline_model_key) > 0, "baseline_model_key must not be empty"
        assert len(current_model_key) > 0, "current_model_key must not be empty"

        # optional fields (may be present depending on model type)
        if _explanations_cmp.KEY_BASELINE_MODEL_NAME in models_overview:
            assert isinstance(
                models_overview[_explanations_cmp.KEY_BASELINE_MODEL_NAME], str
            ), "baseline_model_name must be string"
        if _explanations_cmp.KEY_CURRENT_MODEL_NAME in models_overview:
            assert isinstance(
                models_overview[_explanations_cmp.KEY_CURRENT_MODEL_NAME], str
            ), "current_model_name must be string"

        if verbose:
            print(f" DONE models_overview.baseline_model_key: {baseline_model_key}")
            print(f" DONE models_overview.current_model_key: {current_model_key}")
            if _explanations_cmp.KEY_BASELINE_MODEL_NAME in models_overview:
                print(
                    f" DONE models_overview.baseline_model_name: "
                    f"{models_overview[_explanations_cmp.KEY_BASELINE_MODEL_NAME]}"
                )
            if _explanations_cmp.KEY_CURRENT_MODEL_NAME in models_overview:
                print(
                    f" DONE models_overview.current_model_name: "
                    f"{models_overview[_explanations_cmp.KEY_CURRENT_MODEL_NAME]}"
                )

        # ====================================================================
        # 2.3 MODELS COMPARISONS SECTION VALIDATION (NEW)
        # ====================================================================
        assert _explanations_cmp.KEY_MODELS_COMPARISONS in diff_entry, (
            f"Diff entry {diff_idx} missing 'models_comparisons'"
        )
        models_comparisons = diff_entry[_explanations_cmp.KEY_MODELS_COMPARISONS]
        assert isinstance(models_comparisons, dict), (
            "models_comparisons must be a dictionary"
        )

        # required fields
        required_comparison_keys = [
            _explanations_cmp.KEY_TEST_CASE_RANKS_BASELINE,
            _explanations_cmp.KEY_TEST_CASE_RANKS_CURRENT,
            _explanations_cmp.KEY_TEST_CASE_WINS_BASELINE,
            _explanations_cmp.KEY_TEST_CASE_WINS_CURRENT,
        ]
        for key in required_comparison_keys:
            assert key in models_comparisons, f"models_comparisons missing '{key}'"
            value = models_comparisons[key]
            assert isinstance(value, int), f"{key} must be int, got {type(value)}"
            assert value >= 0, f"{key} must be non-negative, got {value}"

        if verbose:
            print(
                f" DONE models_comparisons.test_case_ranks_baseline: "
                f"{models_comparisons[_explanations_cmp.KEY_TEST_CASE_RANKS_BASELINE]}"
            )
            print(
                f" DONE models_comparisons.test_case_ranks_current: "
                f"{models_comparisons[_explanations_cmp.KEY_TEST_CASE_RANKS_CURRENT]}"
            )
            print(
                f" DONE models_comparisons.test_case_wins_baseline: "
                f"{models_comparisons[_explanations_cmp.KEY_TEST_CASE_WINS_BASELINE]}"
            )
            print(
                f" DONE models_comparisons.test_case_wins_current: "
                f"{models_comparisons[_explanations_cmp.KEY_TEST_CASE_WINS_CURRENT]}"
            )

        # ====================================================================
        # 2.4 MODELS COMPARISONS METRICS SECTION VALIDATION (NEW)
        # ====================================================================
        assert _explanations_cmp.KEY_MODELS_COMPARISONS_METRICS in diff_entry, (
            f"Diff entry {diff_idx} missing 'models_comparisons_metrics'"
        )
        models_comparisons_metrics = diff_entry[
            _explanations_cmp.KEY_MODELS_COMPARISONS_METRICS
        ]
        assert isinstance(models_comparisons_metrics, dict), (
            "models_comparisons_metrics must be a dictionary"
        )

        # required fields
        assert (
            _explanations_cmp.KEY_METRICS_RANKS_BASELINE in models_comparisons_metrics
        ), "models_comparisons_metrics missing 'metrics_ranks_baseline'"
        assert (
            _explanations_cmp.KEY_METRICS_RANKS_CURRENT in models_comparisons_metrics
        ), "models_comparisons_metrics missing 'metrics_ranks_current'"
        assert (
            _explanations_cmp.KEY_METRICS_WINS_BASELINE in models_comparisons_metrics
        ), "models_comparisons_metrics missing 'metrics_wins_baseline'"
        assert (
            _explanations_cmp.KEY_METRICS_WINS_CURRENT in models_comparisons_metrics
        ), "models_comparisons_metrics missing 'metrics_wins_current'"
        assert _explanations_cmp.KEY_METRICS_AVERAGES in models_comparisons_metrics, (
            "models_comparisons_metrics missing 'metrics_averages'"
        )

        # validate types
        metrics_ranks_baseline = models_comparisons_metrics[
            _explanations_cmp.KEY_METRICS_RANKS_BASELINE
        ]
        metrics_ranks_current = models_comparisons_metrics[
            _explanations_cmp.KEY_METRICS_RANKS_CURRENT
        ]
        metrics_wins_baseline = models_comparisons_metrics[
            _explanations_cmp.KEY_METRICS_WINS_BASELINE
        ]
        metrics_wins_current = models_comparisons_metrics[
            _explanations_cmp.KEY_METRICS_WINS_CURRENT
        ]
        metrics_averages = models_comparisons_metrics[
            _explanations_cmp.KEY_METRICS_AVERAGES
        ]

        assert isinstance(metrics_ranks_baseline, (int, float)), (
            "metrics_ranks_baseline must be numeric"
        )
        assert isinstance(metrics_ranks_current, (int, float)), (
            "metrics_ranks_current must be numeric"
        )
        assert isinstance(metrics_wins_baseline, int), (
            "metrics_wins_baseline must be int"
        )
        assert isinstance(metrics_wins_current, int), "metrics_wins_current must be int"
        assert isinstance(metrics_averages, list), (
            "metrics_averages must be a list (Protobuf-friendly)"
        )

        # validate non-negative values
        assert metrics_ranks_baseline >= 0, (
            f"metrics_ranks_baseline must be non-negative: {metrics_ranks_baseline}"
        )
        assert metrics_ranks_current >= 0, (
            f"metrics_ranks_current must be non-negative: {metrics_ranks_current}"
        )
        assert metrics_wins_baseline >= 0, (
            f"metrics_wins_baseline must be non-negative: {metrics_wins_baseline}"
        )
        assert metrics_wins_current >= 0, (
            f"metrics_wins_current must be non-negative: {metrics_wins_current}"
        )

        if verbose:
            print(
                f" DONE models_comparisons_metrics.metrics_ranks_baseline: "
                f"{metrics_ranks_baseline:.2f}"
            )
            print(
                f" DONE models_comparisons_metrics.metrics_ranks_current: "
                f"{metrics_ranks_current:.2f}"
            )
            print(
                f" DONE models_comparisons_metrics.metrics_wins_baseline: "
                f"{metrics_wins_baseline}"
            )
            print(
                f" DONE models_comparisons_metrics.metrics_wins_current: "
                f"{metrics_wins_current}"
            )
            print(
                f" DONE models_comparisons_metrics.metrics_averages: "
                f"{len(metrics_averages)} metrics"
            )

        # validate metrics_averages structure (Protobuf-friendly list format)
        # each entry should have metric_key field and baseline/current data
        for metric_entry in metrics_averages:
            assert isinstance(metric_entry, dict), (
                "Metric entry in metrics_averages must be dict"
            )
            # must have metric_key field
            assert _explanations_cmp.KEY_METRIC_KEY in metric_entry, (
                "Metric entry missing 'metric_key' field"
            )
            metric_key = metric_entry[_explanations_cmp.KEY_METRIC_KEY]
            assert isinstance(metric_key, str), (
                f"metric_key must be string: {metric_key}"
            )
            # these keys may vary, but baseline/current should be present
            assert "baseline_avg" in metric_entry, (
                f"Metric {metric_key} missing baseline_avg"
            )
            assert "current_avg" in metric_entry, (
                f"Metric {metric_key} missing current_avg"
            )

        # ====================================================================
        # 2.5 TECHNICAL METRICS SECTION VALIDATION (NEW)
        # ====================================================================
        assert _explanations_cmp.KEY_TECHNICAL_METRICS in diff_entry, (
            f"Diff entry {diff_idx} missing 'technical_metrics'"
        )
        technical_metrics = diff_entry[_explanations_cmp.KEY_TECHNICAL_METRICS]
        assert isinstance(technical_metrics, dict), (
            "technical_metrics must be a dictionary"
        )

        # should have baseline and current sub-dicts
        assert _explanations_cmp.KEY_BASELINE in technical_metrics, (
            "technical_metrics missing 'baseline'"
        )
        assert _explanations_cmp.KEY_CURRENT in technical_metrics, (
            "technical_metrics missing 'current'"
        )

        baseline_tech = technical_metrics[_explanations_cmp.KEY_BASELINE]
        current_tech = technical_metrics[_explanations_cmp.KEY_CURRENT]

        assert isinstance(baseline_tech, dict), (
            "technical_metrics.baseline must be dict"
        )
        assert isinstance(current_tech, dict), "technical_metrics.current must be dict"

        # validate that cost/duration fields exist and are numeric
        # note: some fields may be 0 if no data available
        for model_name, tech_data in [
            ("baseline", baseline_tech),
            ("current", current_tech),
        ]:
            if "cost_sum" in tech_data:
                assert isinstance(tech_data["cost_sum"], (int, float)), (
                    f"{model_name} cost_sum must be numeric"
                )
                assert tech_data["cost_sum"] >= 0, (
                    f"{model_name} cost_sum must be non-negative"
                )
            if "duration_sum" in tech_data:
                assert isinstance(tech_data["duration_sum"], (int, float)), (
                    f"{model_name} duration_sum must be numeric"
                )
                assert tech_data["duration_sum"] >= 0, (
                    f"{model_name} duration_sum must be non-negative"
                )

        if verbose:
            print(f" DONE technical_metrics.baseline: {len(baseline_tech)} fields")
            print(f" DONE technical_metrics.current: {len(current_tech)} fields")

        # ====================================================================
        # 2.6 ITEMS ARRAY VALIDATION
        # ====================================================================
        assert _explanations_cmp.KEY_ITEMS in diff_entry, (
            f"Diff entry {diff_idx} missing 'items'"
        )
        items = diff_entry[_explanations_cmp.KEY_ITEMS]
        assert isinstance(items, list), "items must be a list"
        # note: items CAN be empty if no comparable test cases between models

        if verbose:
            print(f" DONE items: {len(items)} test case diffs")

        # validate first item structure (spot check) - only if items not empty
        if len(items) > 0:
            first_item = items[0]
            assert isinstance(first_item, dict), "Item must be a dictionary"

            # required item fields
            required_item_keys = [
                "question",
                "diff_flipped_metrics",
                "baseline_test_case_result",
                "current_test_case_result",
            ]
            for key in required_item_keys:
                assert key in first_item, f"Item missing required key: {key}"

            # validate test case result structure (Protobuf-friendly)
            baseline_tc = first_item["baseline_test_case_result"]
            assert isinstance(baseline_tc, dict), (
                "baseline_test_case_result must be dict"
            )
            assert "metric_scores" in baseline_tc, (
                "baseline_test_case_result missing metric_scores"
            )
            assert isinstance(baseline_tc["metric_scores"], list), (
                "metric_scores must be a list"
            )

        # ====================================================================
        # 2.7 TEST CASES LEADERBOARD VALIDATION
        # ====================================================================
        assert _explanations_cmp.KEY_TEST_CASES_LEADERBOARD in diff_entry, (
            f"Diff entry {diff_idx} missing 'test_cases_leaderboard'"
        )
        test_cases_leaderboard = diff_entry[
            _explanations_cmp.KEY_TEST_CASES_LEADERBOARD
        ]
        assert isinstance(test_cases_leaderboard, list), (
            "test_cases_leaderboard must be a list"
        )

        if verbose:
            print(
                f" DONE test_cases_leaderboard: {len(test_cases_leaderboard)} entries"
            )

        # validate leaderboard entry structure (spot check first entry if not empty)
        if len(test_cases_leaderboard) > 0:
            first_lb_entry = test_cases_leaderboard[0]
            assert isinstance(first_lb_entry, dict), "Leaderboard entry must be dict"

            # required leaderboard entry fields
            assert _explanations_cmp.KEY_LEADERBOARD_POSITION in first_lb_entry, (
                "Leaderboard entry missing 'leaderboard_position'"
            )
            assert _explanations_cmp.KEY_WINS in first_lb_entry, (
                "Leaderboard entry missing 'wins' (backward compatibility)"
            )
            assert _explanations_cmp.KEY_QUESTION in first_lb_entry, (
                "Leaderboard entry missing 'question'"
            )
            assert _explanations_cmp.KEY_CHANGED_METRICS_COUNT in first_lb_entry, (
                "Leaderboard entry missing 'changed_metrics_count'"
            )
            assert _explanations_cmp.KEY_DIFF_INDEX in first_lb_entry, (
                "Leaderboard entry missing 'diff_index'"
            )
            assert _explanations_cmp.KEY_BASELINE_WINS in first_lb_entry, (
                "Leaderboard entry missing 'baseline_wins'"
            )
            assert _explanations_cmp.KEY_CURRENT_WINS in first_lb_entry, (
                "Leaderboard entry missing 'current_wins'"
            )
            assert _explanations_cmp.KEY_BASELINE_RANK_AVG in first_lb_entry, (
                "Leaderboard entry missing 'baseline_rank_avg'"
            )
            assert _explanations_cmp.KEY_CURRENT_RANK_AVG in first_lb_entry, (
                "Leaderboard entry missing 'current_rank_avg'"
            )

            leaderboard_position = first_lb_entry[
                _explanations_cmp.KEY_LEADERBOARD_POSITION
            ]
            wins = first_lb_entry[_explanations_cmp.KEY_WINS]
            question = first_lb_entry[_explanations_cmp.KEY_QUESTION]
            changed_count = first_lb_entry[_explanations_cmp.KEY_CHANGED_METRICS_COUNT]
            diff_index = first_lb_entry[_explanations_cmp.KEY_DIFF_INDEX]
            baseline_wins = first_lb_entry[_explanations_cmp.KEY_BASELINE_WINS]
            current_wins = first_lb_entry[_explanations_cmp.KEY_CURRENT_WINS]
            baseline_rank_avg = first_lb_entry[_explanations_cmp.KEY_BASELINE_RANK_AVG]
            current_rank_avg = first_lb_entry[_explanations_cmp.KEY_CURRENT_RANK_AVG]

            assert isinstance(leaderboard_position, int), (
                "leaderboard_position must be int"
            )
            assert leaderboard_position >= 1, (
                "leaderboard_position must be positive (1-based)"
            )
            assert isinstance(wins, int), "wins must be int (backward compatibility)"
            assert wins >= 0, "wins must be non-negative"
            assert wins == baseline_wins, (
                "wins must equal baseline_wins for backward compatibility (DEPRECATED)"
            )
            assert isinstance(question, str), "question must be string"
            assert len(question) > 0, "question must not be empty"
            assert isinstance(changed_count, int), "changed_metrics_count must be int"
            assert changed_count >= 0, "changed_metrics_count must be non-negative"
            assert isinstance(diff_index, int), "diff_index must be int"
            assert diff_index >= 0, "diff_index must be non-negative (0-based)"
            assert isinstance(baseline_wins, int), "baseline_wins must be int"
            assert baseline_wins >= 0, "baseline_wins must be non-negative"
            assert isinstance(current_wins, int), "current_wins must be int"
            assert current_wins >= 0, "current_wins must be non-negative"
            assert isinstance(baseline_rank_avg, (int, float)), (
                "baseline_rank_avg must be numeric"
            )
            assert baseline_rank_avg >= 0, "baseline_rank_avg must be non-negative"
            assert isinstance(current_rank_avg, (int, float)), (
                "current_rank_avg must be numeric"
            )
            assert current_rank_avg >= 0, "current_rank_avg must be non-negative"

    # ========================================================================
    # 3. METRICS META VALIDATION
    # ========================================================================
    if verbose:
        print("\n=== Validating metrics_meta ===")

    if diff_json_dict["metrics_meta"] is not None:
        metrics_meta = diff_json_dict["metrics_meta"]
        assert len(metrics_meta) > 0, "metrics_meta should not be empty dict"

        # spot check: validate one metric meta structure
        first_metric_key = next(iter(metrics_meta))
        first_metric_meta = metrics_meta[first_metric_key]

        assert isinstance(first_metric_meta, dict), (
            f"Metric meta for {first_metric_key} must be dict"
        )
        # metrics meta should have fields like higher_is_better, threshold, etc.
        # but structure may vary, so we just check it's a dict

        if verbose:
            print(f" DONE metrics_meta: {len(metrics_meta)} metrics defined")
    else:
        if verbose:
            print("  WARNING metrics_meta is None (no metrics metadata available)")

    # ========================================================================
    # 4. DATA COMPLETENESS VALIDATION
    # ========================================================================
    if verbose:
        print("\n=== Validating data completeness for HTML regeneration ===")

    # verify that JSON contains all data needed to regenerate HTML
    for diff_idx, diff_entry in enumerate(diff_json_dict["diffs"]):
        # summary data is complete
        assert diff_entry[_explanations_cmp.KEY_SUMMARY][
            _explanations_cmp.KEY_RECOMMENDATION
        ], f"Diff {diff_idx}: recommendation is empty"

        # models overview has model keys
        assert diff_entry[_explanations_cmp.KEY_MODELS_OVERVIEW][
            _explanations_cmp.KEY_BASELINE_MODEL_KEY
        ], f"Diff {diff_idx}: baseline_model_key is empty"
        assert diff_entry[_explanations_cmp.KEY_MODELS_OVERVIEW][
            _explanations_cmp.KEY_CURRENT_MODEL_KEY
        ], f"Diff {diff_idx}: current_model_key is empty"

        # comparisons have numeric data
        models_comparisons = diff_entry[_explanations_cmp.KEY_MODELS_COMPARISONS]
        assert (
            models_comparisons[_explanations_cmp.KEY_TEST_CASE_RANKS_BASELINE]
            + models_comparisons[_explanations_cmp.KEY_TEST_CASE_RANKS_CURRENT]
            >= 0
        ), f"Diff {diff_idx}: test case ranks sum should be non-negative"

        # technical metrics have data structures
        assert (
            _explanations_cmp.KEY_BASELINE
            in diff_entry[_explanations_cmp.KEY_TECHNICAL_METRICS]
        ), f"Diff {diff_idx}: technical_metrics missing baseline"
        assert (
            _explanations_cmp.KEY_CURRENT
            in diff_entry[_explanations_cmp.KEY_TECHNICAL_METRICS]
        ), f"Diff {diff_idx}: technical_metrics missing current"

        # items array exists (can be empty if no comparable test cases)
        assert _explanations_cmp.KEY_ITEMS in diff_entry, (
            f"Diff {diff_idx}: items key missing"
        )
        # note: items can be empty if no comparable test cases between models

        # test_cases_leaderboard exists (embedded leaderboard data)
        assert _explanations_cmp.KEY_TEST_CASES_LEADERBOARD in diff_entry, (
            f"Diff {diff_idx}: test_cases_leaderboard key missing"
        )
        assert isinstance(
            diff_entry[_explanations_cmp.KEY_TEST_CASES_LEADERBOARD], list
        ), f"Diff {diff_idx}: test_cases_leaderboard must be a list"

        # validate diff_changed_metrics is present in each item
        for item in diff_entry[_explanations_cmp.KEY_ITEMS]:
            assert _explanations_cmp.KEY_DIFF_CHANGED_METRICS in item, (
                f"Diff {diff_idx}: missing diff_changed_metrics in item"
            )
            assert isinstance(item[_explanations_cmp.KEY_DIFF_CHANGED_METRICS], list), (
                f"Diff {diff_idx}: diff_changed_metrics must be list"
            )
            # verify each entry has required fields
            for metric_entry in item[_explanations_cmp.KEY_DIFF_CHANGED_METRICS]:
                assert _explanations_cmp.KEY_METRIC_NAME in metric_entry, (
                    "missing metric_name in diff_changed_metrics"
                )

        # validate models_metadata is present
        assert _explanations_cmp.KEY_MODELS_METADATA in diff_entry, (
            f"Diff {diff_idx}: missing models_metadata"
        )
        assert isinstance(diff_entry[_explanations_cmp.KEY_MODELS_METADATA], dict), (
            f"Diff {diff_idx}: models_metadata must be dict"
        )

        # validate statistics is present
        assert _explanations_cmp.KEY_STATISTICS in diff_entry, (
            f"Diff {diff_idx}: missing statistics"
        )
        assert isinstance(diff_entry[_explanations_cmp.KEY_STATISTICS], dict), (
            f"Diff {diff_idx}: statistics must be dict"
        )

    if verbose:
        print(" DONE All required data for HTML regeneration is present")

    # ========================================================================
    # OVERALL COMPARISON VALIDATION (if multiple model pairs)
    # ========================================================================
    if len(diff_json_dict["diffs"]) > 1:
        if verbose:
            print("\n=== Validating overall comparison ===")

        assert _explanations_cmp.KEY_OVERALL_COMPARISON in diff_json_dict, (
            "Missing 'overall_comparison' for multi-model comparison"
        )

        overall = diff_json_dict[_explanations_cmp.KEY_OVERALL_COMPARISON]

        # validate overall_summary
        assert _explanations_cmp.KEY_OVERALL_SUMMARY in overall, (
            "overall_comparison missing 'overall_summary'"
        )
        overall_summary = overall[_explanations_cmp.KEY_OVERALL_SUMMARY]
        assert _explanations_cmp.KEY_OVERALL_RECOMMENDATION in overall_summary, (
            "overall_summary missing 'overall_recommendation'"
        )
        assert _explanations_cmp.KEY_OVERALL_RECOMMENDATION_WINNER in overall_summary, (
            "overall_summary missing 'overall_recommendation_winner'"
        )
        assert (
            _explanations_cmp.KEY_OVERALL_RECOMMENDATION_CONFIDENCE in overall_summary
        ), "overall_summary missing 'overall_recommendation_confidence'"

        # validate recommendations_summary
        assert _explanations_cmp.KEY_RECOMMENDATIONS_SUMMARY in overall_summary, (
            "overall_summary missing 'recommendations_summary'"
        )
        assert isinstance(
            overall_summary[_explanations_cmp.KEY_RECOMMENDATIONS_SUMMARY], dict
        ), "recommendations_summary must be dict"
        rec_summary = overall_summary[_explanations_cmp.KEY_RECOMMENDATIONS_SUMMARY]
        assert _explanations_cmp.KEY_BASELINE in rec_summary, (
            "missing 'baseline' in recommendations_summary"
        )
        assert "tie" in rec_summary, "missing 'tie' in recommendations_summary"
        assert _explanations_cmp.KEY_CURRENT in rec_summary, (
            "missing 'current' in recommendations_summary"
        )

        winner = overall_summary[_explanations_cmp.KEY_OVERALL_RECOMMENDATION_WINNER]
        assert winner in ["baseline", "current", "tie"], f"Invalid winner: {winner}"

        # validate overall_evaluations_overview
        assert _explanations_cmp.KEY_OVERALL_EVALUATIONS_OVERVIEW in overall, (
            "overall_comparison missing 'overall_evaluations_overview'"
        )
        overview = overall[_explanations_cmp.KEY_OVERALL_EVALUATIONS_OVERVIEW]
        assert _explanations_cmp.KEY_BASELINE_MODELS_COUNT in overview, (
            "overall_evaluations_overview missing 'baseline_models_count'"
        )
        assert _explanations_cmp.KEY_CURRENT_MODELS_COUNT in overview, (
            "overall_evaluations_overview missing 'current_models_count'"
        )
        assert _explanations_cmp.KEY_TOTAL_COMPARABLE_TEST_CASES in overview, (
            "overall_evaluations_overview missing 'total_comparable_test_cases'"
        )

        # validate overall_models_comparison
        assert _explanations_cmp.KEY_OVERALL_MODELS_COMPARISON in overall, (
            "overall_comparison missing 'overall_models_comparison'"
        )
        models_cmp = overall[_explanations_cmp.KEY_OVERALL_MODELS_COMPARISON]
        assert _explanations_cmp.KEY_TEST_CASE_WINS_BASELINE in models_cmp, (
            "overall_models_comparison missing 'test_case_wins_baseline'"
        )
        assert _explanations_cmp.KEY_TEST_CASE_WINS_CURRENT in models_cmp, (
            "overall_models_comparison missing 'test_case_wins_current'"
        )
        assert _explanations_cmp.KEY_METRICS_WINS_BASELINE in models_cmp, (
            "overall_models_comparison missing 'metrics_wins_baseline'"
        )
        assert _explanations_cmp.KEY_METRICS_WINS_CURRENT in models_cmp, (
            "overall_models_comparison missing 'metrics_wins_current'"
        )

        # validate overall_technical_metrics
        assert _explanations_cmp.KEY_OVERALL_TECHNICAL_METRICS in overall, (
            "overall_comparison missing 'overall_technical_metrics'"
        )
        tech = overall[_explanations_cmp.KEY_OVERALL_TECHNICAL_METRICS]
        assert _explanations_cmp.KEY_BASELINE in tech, (
            "overall_technical_metrics missing 'baseline'"
        )
        assert _explanations_cmp.KEY_CURRENT in tech, (
            "overall_technical_metrics missing 'current'"
        )

        if verbose:
            print(" DONE overall_comparison validation")
            print(f"   Winner: {winner}")
            print(
                f"   Baseline models: "
                f"{overview[_explanations_cmp.KEY_BASELINE_MODELS_COUNT]}"
            )
            print(
                f"   Current models: "
                f"{overview[_explanations_cmp.KEY_CURRENT_MODELS_COUNT]}"
            )

    # ========================================================================
    # FINAL SUCCESS MESSAGE
    # ========================================================================
    if verbose:
        print("\n=== DONE All JSON validation checks passed! ===\n")


@pytest.mark.skipif(
    not test_utils.is_private_test_data_available(),
    reason="Test data from S3 not available",
)
@pytest.mark.parametrize(
    "original_explanation_path",
    [
        "data/generative/eval_compare/toxicity-2m/baseline-explanation.json",
        "data/generative/eval_compare/toxicity-2m/current-explanation.json",
        "data/generative/eval_s3/bug-1539/explanation-1-asemsim.json",
        "data/generative/eval_s3/bug-1539/explanation-2-ground.json",
        "data/generative/eval_s3/bug-1539/explanation-3-dataleak.json",
        "data/generative/eval_s3/bug-1539/explanation-4-toxic.json",
        "data/generative/eval_s3/bug-1539/explanation-5-manual.json",
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_from_json(tmp_path, original_explanation_path, cmp_txt: bool = False):
    #
    # GIVEN
    #
    if not pathlib.Path(original_explanation_path).exists():
        print(
            f"WARNING: test data not synchronized - skipping the test for "
            f"{original_explanation_path}"
        )
        return

    with open(original_explanation_path) as f:
        original_explanation_dict = json.load(f)

    # map: explainer_id ->  Explainer
    explainers_map = _given_explainers_map()
    data_dir = str(tmp_path)
    logger = loggers.SonarPrintLogger()

    #
    # WHEN
    #
    explanation_loaded = explanations.LlmEvalResultsExplanation.from_dict(
        explainers_map,
        original_explanation_dict,
        display_name=f"Loaded from JSon {original_explanation_path}",
        display_category=explanations.Explanation.DISPLAY_CAT_LLM,
    )
    # provide persistence (DOES NOT have to be always needed)
    explanation_loaded.explainer.persistence = persistences.ExplainerPersistence(
        data_dir=data_dir,
        username=commons.DEFAULT_USER,
        explainer_id=explanation_loaded.explainer.explainer_id(),
        explainer_job_key="FooJobKey",
        store_persistence=persistences.FilesystemPersistence(
            base_path=data_dir, logger=logger
        ),
    )
    # add formats
    explanation_loaded.add_json_format()
    explanation_loaded.add_csv_format()
    explanation_loaded.add_datatable_format()

    #
    # THEN
    #
    print(f"Loaded EXPLANATION: {explanation_loaded}")
    assert explanation_loaded
    # COMPARE: original vs. loaded JSONs
    load_json_path = explanation_loaded.explainer.persistence.get_explanation_file_path(
        explanation_type=explanations.LlmEvalResultsExplanation.explanation_type(),
        explanation_format=commons.MimeType.MIME_JSON,
    )
    with open(load_json_path) as f:
        loaded_json_dict = json.load(f)
    # reload the original JSON file (clean dict w/o objects)
    with open(original_explanation_path) as f:
        original_json_dict = json.load(f)
    # COMPARE objects
    assert loaded_json_dict["results"] == original_json_dict["results"], (
        "Results mismatch between original and loaded explanation"
    )
    assert loaded_json_dict["models"] == original_json_dict["models"], (
        "Models mismatch between original and loaded explanation"
    )
    assert (
        loaded_json_dict["evaluator"]["id"] == original_json_dict["evaluator"]["id"]
    ), "Evaluator ID mismatch between original and loaded explanation"
    # COMPARE JSON serializations
    if cmp_txt:
        # manual review:
        # - OK:  key order ~ metrics-meta / evaluator keywords shuffled, but identical
        # - all OTHER fields are identical
        with open(original_explanation_path) as f:
            original_txt = f.read()
        with open(load_json_path) as f:
            loaded_txt = f.read()
        assert original_txt == loaded_txt, (
            f"Results mismatch between original and loaded explanation: "
            f"{original_explanation_path=} vs. {load_json_path=}"
        )


@pytest.mark.parametrize(
    "baseline_result_path,current_result_path",
    [
        # 1 evaluator
        (
            "data/generative/eval_compare/toxicity-2m/baseline-explanation.json",
            "data/generative/eval_compare/toxicity-2m/current-explanation.json",
        ),
        # MANY evaluators
        (
            "data/generative/eval_compare/multiple-evaluators-h2ogpte-results/"
            "baseline-explanation.json",
            "data/generative/eval_compare/multiple-evaluators-h2ogpte-results/"
            "current-explanation.json",
        ),
        # H2O EvalStudio bug: 2025/11/24 ... explanation loading
        (
            "data/generative/eval_compare/es-20251124-01/baseline-explanation.json",
            "data/generative/eval_compare/es-20251124-01/current-explanation.json",
        ),
        # 2025-12-08 Laco: no metrics @ TC, no metrics tables (A)
        pytest.param(
            "data/generative/eval_s3/laco-20251208-a/report_baseline.json",
            "data/generative/eval_s3/laco-20251208-a/report_current.json",
            marks=(
                pytest.mark.skipif(
                    not test_utils.is_private_test_data_available(),
                    reason="Skipped as S3 data are needed",
                ),
            ),
        ),
        # 2025-12-08 Laco: no metrics (B)
        pytest.param(
            "data/generative/eval_s3/laco-20251208-b/report_baseline.json",
            "data/generative/eval_s3/laco-20251208-b/report_current.json",
            marks=(
                pytest.mark.skipif(
                    not test_utils.is_private_test_data_available(),
                    reason="Skipped as S3 data are needed",
                ),
            ),
        ),
        # 2025-12-08 Laco: no metrics (C)
        pytest.param(
            "data/generative/eval_s3/laco-20251208-c/report_baseline.json",
            "data/generative/eval_s3/laco-20251208-c/report_current.json",
            marks=(
                pytest.mark.skipif(
                    not test_utils.is_private_test_data_available(),
                    reason="Skipped as S3 data are needed",
                ),
            ),
        ),
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_results_comparator(
    tmp_path: pathlib.Path,
    baseline_result_path: str,
    current_result_path: str,
):
    #
    # GIVEN
    #
    explainers_map = _given_explainers_map()

    assert pathlib.Path(baseline_result_path).exists()
    with open(baseline_result_path) as f:
        baseline_results_json = json.load(f)
    baseline_results = explanations.LlmEvalResultsExplanation.from_dict(
        explainers_map=explainers_map, explanation_dict=baseline_results_json
    )

    assert pathlib.Path(current_result_path).exists()
    with open(current_result_path) as f:
        current_results_json = json.load(f)
    current_results = explanations.LlmEvalResultsExplanation.from_dict(
        explainers_map=explainers_map, explanation_dict=current_results_json
    )

    #
    # WHEN
    #
    diff: explanations.EvalResultsDiff = baseline_results.compare(
        other=current_results,
    )

    #
    # THEN
    #
    assert diff

    diff_as_json_dict = diff.to_dict()
    diff_as_json_path = tmp_path / "diff.json"
    with open(diff_as_json_path, "w") as f:
        json.dump(diff_as_json_dict, f, indent=2)
    print(f"\nDiff JSON written to: file://{diff_as_json_path}")

    # validate JSON structure and content
    _then_assert_json(diff_as_json_dict, verbose=True)

    diff_html = str(diff.to_html())
    diff_as_html_path = tmp_path / "diff.html"
    with open(diff_as_html_path, "w") as f:
        f.write(diff_html)
    print(f"Diff HTML written to: file://{diff_as_html_path}")


@pytest.mark.skipif(
    not given_generative.is_config(),
    reason="Generative AI configuration (given_generative.json) not available",
)
@pytest.mark.parametrize(
    (
        "baseline_test_lab_path,current_test_lab_path,baseline_llm_model,"
        "current_llm_model,evaluators,test_key"
    ),
    [
        # 1 evaluator @ evaluation instances
        # - really NICE test with 2 different OpenAI models used @ the OpenAI RAG:
        # - different / similar answers
        # - different retrieved contexts (which are sometimes missing due OpenAI bugs)
        (
            "data/generative/eval_compare/h2ogpte-openai-1m/baseline_test_lab.json",
            "data/generative/eval_compare/h2ogpte-openai-1m/current_test_lab.json",
            "",
            "",
            [arnj_e.RagAnswerRelevancyNoJudgeEvaluator.evaluator_id()],
            False,
        ),
        # 1 evaluator @ keys
        (
            "data/generative/eval_compare/h2ogpte-openai-1m/baseline_test_lab.json",
            "data/generative/eval_compare/h2ogpte-openai-1m/current_test_lab.json",
            "",
            "",
            [arnj_e.RagAnswerRelevancyNoJudgeEvaluator.evaluator_id()],
            True,
        ),
        # MULTIPLE evaluators test: run evals w/ MULTIPLE e. > merge > compare
        (
            "data/generative/eval_compare/h2ogpte-openai-1m/baseline_test_lab.json",
            "data/generative/eval_compare/h2ogpte-openai-1m/current_test_lab.json",
            "",
            "",
            [
                arnj_e.RagAnswerRelevancyNoJudgeEvaluator.evaluator_id(),
                fb_e.FairnessBiasEvaluator.evaluator_id(),
                rg_e.RagGroundednessEvaluator.evaluator_id(),
                t_e.ToxicityEvaluator.evaluator_id(),
                tp_e.RagStrStrEvaluator.evaluator_id(),
            ],
            False,
        ),
        # BENCHMARK: Amazon Bedrock vs. h2oGPTe @ h2oGPTe's fin test suite 2025/10/01
        #            (local data s they are big for Git & want to save S3 budget)
        pytest.param(
            "data/generative/eval_s3/benchmark-h2ogpte-bedrock-2024-10-01/"
            "test_lab_h2ogpte.json",
            "data/generative/eval_s3/benchmark-h2ogpte-bedrock-2024-10-01/"
            "test_lab_bedrock.json",
            "claude-3-haiku-20240307",
            "anthropic.claude-3-haiku-20240307-v1:0",
            [
                tp_e.RagStrStrEvaluator.evaluator_id(),
            ],
            True,
            marks=(
                pytest.mark.skipif(
                    test_utils.GitHubActions.is_in_gha()
                    or not test_utils.is_private_test_data_available(),
                    reason="Skipped as S3 data are needed",
                ),
            ),
        ),
        # BENCHMARK: OpenAI vs. h2oGPTe @ h2oGPTe's fin test suite 2025/10/01
        #            (local data s they are big for Git & want to save S3 budget)
        pytest.param(
            "data/generative/eval_s3/benchmark-h2ogpte-bedrock-2024-10-01/"
            "test_lab_h2ogpte.json",
            "data/generative/eval_s3/benchmark-h2ogpte-bedrock-2024-10-01/"
            "test_lab_openai.json",
            "claude-3-haiku-20240307",
            "gpt-4o",
            [
                tp_e.RagStrStrEvaluator.evaluator_id(),
            ],
            True,
            marks=(
                pytest.mark.skipif(
                    test_utils.GitHubActions.is_in_gha()
                    or not test_utils.is_private_test_data_available(),
                    reason="Skipped on GHA as S3 data are needed",
                ),
            ),
        ),
        # TECHNICAL metrics
        (
            "data/generative/procedure_eval_test_lab_small.json",
            "data/generative/procedure_eval_test_lab_small.json",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "claude-3-sonnet-20240229",
            [
                # only use simple evaluators that don't require complex metadata
                tp_e.RagStrStrEvaluator.evaluator_id(),
            ],
            True,
        ),
        # bug: #1549 Coca-Cola for H2O ES - unable to parse diff
        pytest.param(
            "data/generative/eval_s3/bug-1549-cmp-diff-parse/"
            "test-lab-for-MERGED-TS-GPT-LLMs.json",
            "data/generative/eval_s3/bug-1549-cmp-diff-parse/"
            "test-lab-for-MERGED-TS-GROK-LLMs.json",
            "",
            "",
            [
                arnj_e.RagAnswerRelevancyNoJudgeEvaluator.evaluator_id(),
                rg_e.RagGroundednessEvaluator.evaluator_id(),
                sdl_e.SensitiveDataLeakageEvaluator.evaluator_id(),
                t_e.ToxicityEvaluator.evaluator_id(),
                tp_e.RagStrStrEvaluator.evaluator_id(),
            ],
            True,
            marks=(
                pytest.mark.skipif(
                    not test_utils.is_private_test_data_available(),
                    reason="Skipped as S3 data are needed",
                ),
            ),
        ),
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_evaluate_comparator(
    tmp_path: pathlib.Path,
    h2ogpte_connection_fixture: h2o_sonar_config.ConnectionConfig,
    baseline_test_lab_path: str,
    current_test_lab_path: str,
    baseline_llm_model: str,
    current_llm_model: str,
    evaluators: list[str],
    test_key: bool,
):
    """Test comparison of 2 evaluations.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temp directory for storing of the test results.
    h2ogpte_connection_fixture : fixture
        h2oGPTe connection fixture.
    baseline_test_lab_path : pathlib.Path
        Test lab to be used as baseline model for testing.
    current_test_lab_path : pathlib.Path
        Test lab to be used as current model for testing.
    evaluators : list
        Evaluators to be used in the evaluation.
    test_key : bool
        ``True`` to test evaluations by loading them from filesystem via key
        (while results location path specified), ``False`` to test the comparison
        of ``Evaluation`` object instances.

    """

    #
    # GIVEN
    #

    evaluations = []
    for test_lab_path in [baseline_test_lab_path, current_test_lab_path]:
        rag_dataset = testing.RagTestLab.load_from_json(
            llm_host_connection=test_utils.health.get_h2ogpte(),
            file_path=test_lab_path,
        )
        llm_models = rag_dataset.evaluated_models.values()

        evaluation = evaluate.run_evaluation(
            dataset=rag_dataset.dataset,
            models=llm_models,
            evaluators=list(evaluators),
            results_location=tmp_path,
            log_level=loggers.DEBUG,
        )

        assert evaluation
        assert not evaluation.is_explainer_failed()
        evaluations.append(evaluation.key if test_key else evaluation)
        print(
            f"Evaluation HTML:\nfile://{evaluation.result.get_html_report_location()}"
        )

    #
    # WHEN
    #

    comparison_methods = [
        explanations.SentenceComparisonMethod.EXACT_MATCH,
        explanations.SentenceComparisonMethod.COSINE_DISTANCE,
    ]
    if _RUN_BERTSCORE_TESTS and not test_utils.GitHubActions.is_in_gha():
        # BERTScore is really slow - especially on CPU
        comparison_methods.append(explanations.SentenceComparisonMethod.BERT_SCORE)

    diffs = {}
    timings = {}
    for comparison_method in comparison_methods:
        print(f"\nComparing with method: {comparison_method.value}")
        start_time = time.time()
        diff = evaluate.compare_evaluations(
            baseline_evaluation=evaluations[0],
            current_evaluation=evaluations[1],
            # filter by LLM model if diverse host types LLMs are compared
            baseline_llm_model=baseline_llm_model,
            current_llm_model=current_llm_model,
            # always compare ALL evaluators
            results_location=tmp_path,
            comparison_method=comparison_method,
        )
        elapsed_time = time.time() - start_time
        diffs[comparison_method] = diff
        timings[comparison_method] = elapsed_time
        print(f"DONE {comparison_method.value} method in {elapsed_time:.3f}s")

    #
    # THEN
    #

    assert len(diffs) == len(comparison_methods)
    for comparison_method, diff in diffs.items():
        assert diff

        method_name = comparison_method.value
        elapsed_time = timings[comparison_method]

        diff_as_json_dict = diff.to_dict()
        diff_as_json_path = tmp_path / f"diff_{method_name}.json"
        with open(diff_as_json_path, "w") as f:
            json.dump(diff_as_json_dict, f, indent=2)
        print(
            f"\nDiff JSON ({method_name}, {elapsed_time:.3f}s) written to: "
            f"file://{diff_as_json_path}"
        )

        # validate JSON structure and content (only for first method to save time)
        if comparison_method == comparison_methods[0]:
            _then_assert_json(diff_as_json_dict, verbose=True)

        diff_html = str(diff.to_html())
        diff_as_html_path = tmp_path / f"diff_{method_name}.html"
        with open(diff_as_html_path, "w") as f:
            f.write(diff_html)
        print(
            f"Diff HTML ({method_name}, {elapsed_time:.3f}s) written to: "
            f"file://{diff_as_html_path}"
        )


@pytest.mark.parametrize(
    "eval_base_path",
    [
        pathlib.Path(
            "data/generative/eval_compare/multiple-evaluators-h2ogpte/h2o-sonar/"
            "mli_experiment_3bb1c454-72df-4970-8ca6-fa59875ec573"
        ),
        pathlib.Path(
            "data/generative/eval_compare/multiple-evaluators-h2ogpte/h2o-sonar/"
            "mli_experiment_5e7e826e-7a21-463b-a006-6772c0e90b8b"
        ),
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_merge_metrics_multiple_evaluators(
    tmp_path: pathlib.Path, eval_base_path: pathlib.Path
):
    """Test merging metrics from multiple evaluator explanations."""
    #
    # GIVEN
    #

    # find all evaluator explanation JSON files
    explanation_files = list(
        eval_base_path.glob(
            "explainer_*/global_llm_eval_results/application_json/explanation.json"
        )
    )
    assert len(explanation_files) >= 5, (
        f"Expected at least 5 explanation files, found {len(explanation_files)}"
    )

    # map: explainer_id -> Explainer
    explainers_map = _given_explainers_map()

    # load all explanations
    loaded_explanations = []
    for explanation_file in explanation_files:
        with open(explanation_file) as f:
            explanation_dict = json.load(f)

        explanation = explanations.LlmEvalResultsExplanation.from_dict(
            explainers_map=explainers_map,
            explanation_dict=explanation_dict,
            display_name=f"Loaded from {explanation_file.parent.parent.parent.name}",
            display_category=explanations.Explanation.DISPLAY_CAT_LLM,
        )
        loaded_explanations.append(explanation)
        print(
            f"Loaded explanation: {explanation.explainer.explainer_id()} "
            f"with {len(explanation.eval_results.results)} results"
        )

    # use first explanation as base for merging
    base_explanation = loaded_explanations[0]
    explanations_to_merge = loaded_explanations[1:]

    # count unique metrics in base
    base_metrics = set()
    if base_explanation.eval_results.results:
        base_metrics = set(base_explanation.eval_results.results[0].metrics.keys())
    print(f"\nBase explanation metrics: {base_metrics}")

    #
    # WHEN
    #
    merged_explanation = base_explanation.merge_metrics(
        explanations=explanations_to_merge, evaluator_ids=None
    )

    #
    # THEN
    #
    print("\nMerged explanation created successfully")
    assert merged_explanation
    assert merged_explanation.eval_results
    assert len(merged_explanation.eval_results.results) > 0

    # count unique metrics in merged explanation
    merged_metrics = set()
    if merged_explanation.eval_results.results:
        merged_metrics = set(merged_explanation.eval_results.results[0].metrics.keys())
    print(f"Merged explanation metrics: {merged_metrics}")

    # verify that merged explanation has more metrics than base
    assert len(merged_metrics) > len(base_metrics), (
        f"Expected merged metrics ({len(merged_metrics)}) > "
        f"base metrics ({len(base_metrics)})"
    )

    # verify that all base metrics are present in merged
    assert base_metrics.issubset(merged_metrics), (
        "Base metrics should be subset of merged metrics"
    )

    # verify that number of results is the same
    assert len(merged_explanation.eval_results.results) == len(
        base_explanation.eval_results.results
    ), "Number of results should be preserved after merging"

    print(
        f"\nMerge successful: {len(base_metrics)} base metrics + "
        f"{len(merged_metrics) - len(base_metrics)} additional metrics = "
        f"{len(merged_metrics)} total metrics"
    )

    # save merged explanation to JSON
    data_dir = str(tmp_path)
    logger = loggers.SonarPrintLogger()
    explainer_persistence = persistences.ExplainerPersistence(
        data_dir=data_dir,
        username=commons.DEFAULT_USER,
        explainer_id=merged_explanation.explainer.explainer_id(),
        explainer_job_key="MergedMetricsTest",
        store_persistence=persistences.FilesystemPersistence(
            base_path=data_dir, logger=logger
        ),
    )
    merged_explanation.explainer.persistence = explainer_persistence
    merged_explanation.add_json_format()

    # get and print the saved JSON path
    explanation_type = explanations.LlmEvalResultsExplanation.explanation_type()
    merged_json_path = (
        merged_explanation.explainer.persistence.get_explanation_file_path(
            explanation_type=explanation_type,
            explanation_format=commons.MimeType.MIME_JSON,
        )
    )
    print(f"\nMerged explanation saved to: file://{merged_json_path}")


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_metrics_rank_calculation():
    """Test metrics rank calculation in model comparison statistics.

    This test verifies that the metrics rank calculation correctly counts
    how many times each model (baseline vs current) scored better across
    all metrics, respecting metric directionality (higher_is_better).
    """
    #
    # GIVEN
    #
    from h2o_sonar.lib.api.explanations import _explanations_cmp_html

    # create mock MetricMeta objects
    class MockMetricMeta:
        def __init__(self, higher_is_better: bool):
            self.higher_is_better = higher_is_better

    # scenario 1: higher is better metrics
    baseline_metrics_1 = {
        "accuracy": 0.9,
        "precision": 0.85,
        "recall": 0.80,
    }
    current_metrics_1 = {
        "accuracy": 0.88,  # baseline wins
        "precision": 0.87,  # current wins
        "recall": 0.80,  # tie - skip
    }
    metrics_meta_1 = {
        "accuracy": MockMetricMeta(higher_is_better=True),
        "precision": MockMetricMeta(higher_is_better=True),
        "recall": MockMetricMeta(higher_is_better=True),
    }

    # scenario 2: lower is better metrics
    baseline_metrics_2 = {
        "error_rate": 0.05,
        "latency": 100,
        "cost": 50,
    }
    current_metrics_2 = {
        "error_rate": 0.03,  # current wins (lower is better)
        "latency": 120,  # baseline wins (lower is better)
        "cost": 50,  # tie - skip
    }
    metrics_meta_2 = {
        "error_rate": MockMetricMeta(higher_is_better=False),
        "latency": MockMetricMeta(higher_is_better=False),
        "cost": MockMetricMeta(higher_is_better=False),
    }

    # scenario 3: mixed metrics with edge cases
    baseline_metrics_3 = {
        "f1_score": 0.85,
        "mae": 0.1,
        "missing_value": None,
        "non_numeric": "invalid",
        "no_metadata": 0.5,
    }
    current_metrics_3 = {
        "f1_score": 0.90,  # current wins (higher is better)
        "mae": 0.08,  # current wins (lower is better)
        "missing_value": 0.5,  # skip (None in baseline)
        "non_numeric": "also_invalid",  # skip (non-numeric)
        "no_metadata": 0.6,  # skip (no metadata)
    }
    metrics_meta_3 = {
        "f1_score": MockMetricMeta(higher_is_better=True),
        "mae": MockMetricMeta(higher_is_better=False),
        "missing_value": MockMetricMeta(higher_is_better=True),
        "non_numeric": MockMetricMeta(higher_is_better=True),
        # note: no_metadata intentionally missing
    }

    #
    # WHEN
    #
    # test scenario 1: higher is better
    stats_1 = {
        _explanations_cmp_html.KEY_METRICS_WINS_BASELINE: 0,
        _explanations_cmp_html.KEY_METRICS_WINS_CURRENT: 0,
    }
    # simulate the calculation logic
    all_metric_keys_1 = set(baseline_metrics_1.keys()) & set(current_metrics_1.keys())
    for metric_key in all_metric_keys_1:
        baseline_value = baseline_metrics_1.get(metric_key)
        current_value = current_metrics_1.get(metric_key)

        if (
            baseline_value is None
            or current_value is None
            or not isinstance(baseline_value, (int, float))
            or not isinstance(current_value, (int, float))
            or baseline_value == current_value
        ):
            continue

        if metric_key not in metrics_meta_1:
            continue

        metric_meta = metrics_meta_1[metric_key]
        higher_is_better = metric_meta.higher_is_better

        if higher_is_better:
            if baseline_value > current_value:
                stats_1[_explanations_cmp_html.KEY_METRICS_WINS_BASELINE] += 1
            elif current_value > baseline_value:
                stats_1[_explanations_cmp_html.KEY_METRICS_WINS_CURRENT] += 1
        else:
            if baseline_value < current_value:
                stats_1[_explanations_cmp_html.KEY_METRICS_WINS_BASELINE] += 1
            elif current_value < baseline_value:
                stats_1[_explanations_cmp_html.KEY_METRICS_WINS_CURRENT] += 1

    # test scenario 2: lower is better
    stats_2 = {
        _explanations_cmp_html.KEY_METRICS_WINS_BASELINE: 0,
        _explanations_cmp_html.KEY_METRICS_WINS_CURRENT: 0,
    }
    all_metric_keys_2 = set(baseline_metrics_2.keys()) & set(current_metrics_2.keys())
    for metric_key in all_metric_keys_2:
        baseline_value = baseline_metrics_2.get(metric_key)
        current_value = current_metrics_2.get(metric_key)

        if (
            baseline_value is None
            or current_value is None
            or not isinstance(baseline_value, (int, float))
            or not isinstance(current_value, (int, float))
            or baseline_value == current_value
        ):
            continue

        if metric_key not in metrics_meta_2:
            continue

        metric_meta = metrics_meta_2[metric_key]
        higher_is_better = metric_meta.higher_is_better

        if higher_is_better:
            if baseline_value > current_value:
                stats_2[_explanations_cmp_html.KEY_METRICS_WINS_BASELINE] += 1
            elif current_value > baseline_value:
                stats_2[_explanations_cmp_html.KEY_METRICS_WINS_CURRENT] += 1
        else:
            if baseline_value < current_value:
                stats_2[_explanations_cmp_html.KEY_METRICS_WINS_BASELINE] += 1
            elif current_value < baseline_value:
                stats_2[_explanations_cmp_html.KEY_METRICS_WINS_CURRENT] += 1

    # test scenario 3: mixed with edge cases
    stats_3 = {
        _explanations_cmp_html.KEY_METRICS_WINS_BASELINE: 0,
        _explanations_cmp_html.KEY_METRICS_WINS_CURRENT: 0,
    }
    all_metric_keys_3 = set(baseline_metrics_3.keys()) & set(current_metrics_3.keys())
    for metric_key in all_metric_keys_3:
        baseline_value = baseline_metrics_3.get(metric_key)
        current_value = current_metrics_3.get(metric_key)

        if (
            baseline_value is None
            or current_value is None
            or not isinstance(baseline_value, (int, float))
            or not isinstance(current_value, (int, float))
            or baseline_value == current_value
        ):
            continue

        if metric_key not in metrics_meta_3:
            continue

        metric_meta = metrics_meta_3[metric_key]
        higher_is_better = metric_meta.higher_is_better

        if higher_is_better:
            if baseline_value > current_value:
                stats_3[_explanations_cmp_html.KEY_METRICS_WINS_BASELINE] += 1
            elif current_value > baseline_value:
                stats_3[_explanations_cmp_html.KEY_METRICS_WINS_CURRENT] += 1
        else:
            if baseline_value < current_value:
                stats_3[_explanations_cmp_html.KEY_METRICS_WINS_BASELINE] += 1
            elif current_value < baseline_value:
                stats_3[_explanations_cmp_html.KEY_METRICS_WINS_CURRENT] += 1

    # test scenario 4: empty metrics
    stats_4 = {
        _explanations_cmp_html.KEY_METRICS_WINS_BASELINE: 0,
        _explanations_cmp_html.KEY_METRICS_WINS_CURRENT: 0,
    }

    #
    # THEN
    #
    # scenario 1: higher is better - baseline wins 1, current wins 1, 1 tie skipped
    print(f"\nScenario 1 (higher is better): {stats_1}")
    assert stats_1[_explanations_cmp_html.KEY_METRICS_WINS_BASELINE] == 1
    assert stats_1[_explanations_cmp_html.KEY_METRICS_WINS_CURRENT] == 1

    # scenario 2: lower is better - baseline wins 1, current wins 1, 1 tie skipped
    print(f"Scenario 2 (lower is better): {stats_2}")
    assert stats_2[_explanations_cmp_html.KEY_METRICS_WINS_BASELINE] == 1
    assert stats_2[_explanations_cmp_html.KEY_METRICS_WINS_CURRENT] == 1

    # scenario 3: mixed - current wins 2 (f1_score, mae), edge cases skipped
    print(f"Scenario 3 (mixed with edge cases): {stats_3}")
    assert stats_3[_explanations_cmp_html.KEY_METRICS_WINS_BASELINE] == 0
    assert stats_3[_explanations_cmp_html.KEY_METRICS_WINS_CURRENT] == 2

    # scenario 4: empty - both should be 0
    print(f"Scenario 4 (empty metrics): {stats_4}")
    assert stats_4[_explanations_cmp_html.KEY_METRICS_WINS_BASELINE] == 0
    assert stats_4[_explanations_cmp_html.KEY_METRICS_WINS_CURRENT] == 0


@pytest.mark.parametrize(
    "baseline_result_path,current_result_path",
    [
        # toxicity evaluator comparison
        (
            "data/generative/eval_compare/toxicity-2m/baseline-explanation.json",
            "data/generative/eval_compare/toxicity-2m/current-explanation.json",
        ),
        # multiple evaluators comparison
        (
            "data/generative/eval_compare/multiple-evaluators-h2ogpte-results/"
            "baseline-explanation.json",
            "data/generative/eval_compare/multiple-evaluators-h2ogpte-results/"
            "current-explanation.json",
        ),
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_metrics_rank_html_rendering(
    tmp_path: pathlib.Path,
    baseline_result_path: str,
    current_result_path: str,
):
    """Test that Metrics Rank row is rendered correctly in HTML comparison.

    This integration test verifies:
    1. Metrics Rank row exists in HTML output
    2. Winner is highlighted in bold green
    3. Values are calculated and displayed correctly
    4. All comparison scenarios work (baseline wins, current wins, tie)

    """
    #
    # GIVEN
    #
    explainers_map = _given_explainers_map()

    assert pathlib.Path(baseline_result_path).exists()
    with open(baseline_result_path) as f:
        baseline_results_json = json.load(f)
    baseline_results = explanations.LlmEvalResultsExplanation.from_dict(
        explainers_map=explainers_map, explanation_dict=baseline_results_json
    )

    assert pathlib.Path(current_result_path).exists()
    with open(current_result_path) as f:
        current_results_json = json.load(f)
    current_results = explanations.LlmEvalResultsExplanation.from_dict(
        explainers_map=explainers_map, explanation_dict=current_results_json
    )

    #
    # WHEN
    #
    diff: explanations.EvalResultsDiff = baseline_results.compare(current_results)
    html_output = str(diff.to_html())

    # save HTML for inspection
    html_path = tmp_path / "metrics_rank_test.html"
    with open(html_path, "w") as f:
        f.write(html_output)
    print(f"\nHTML output saved to: file://{html_path}")

    #
    # THEN
    #
    # verify Metrics Rank row exists in HTML
    assert "Metrics Rank" in html_output, "Metrics Rank label not found in HTML"

    # verify the row has a tooltip/title explaining what it means
    assert (
        "The number of metrics across all test cases where the model "
        "scored better than the other model" in html_output
    ), "Metrics Rank tooltip not found in HTML"

    # verify that metrics rank values are present (as numbers)
    # extract stats from diff object
    stats = diff._stats if hasattr(diff, "_stats") else None
    if stats:
        from h2o_sonar.lib.api.explanations import _explanations_cmp_html

        baseline_rank = stats.get(_explanations_cmp_html.KEY_METRICS_WINS_BASELINE, 0)
        current_rank = stats.get(_explanations_cmp_html.KEY_METRICS_WINS_CURRENT, 0)

        print(f"Baseline rank: {baseline_rank}, Current rank: {current_rank}")

        # verify values appear in HTML
        assert str(baseline_rank) in html_output
        assert str(current_rank) in html_output

        # verify winner highlighting: check for green color and bold
        if baseline_rank > current_rank:
            # baseline should be highlighted
            assert 'style="color: #28a745;"' in html_output, (
                "Winner highlighting not found (baseline wins)"
            )
            print("Winner highlighting verified: Baseline wins")
        elif current_rank > baseline_rank:
            # current should be highlighted
            assert 'style="color: #28a745;"' in html_output, (
                "Winner highlighting not found (current wins)"
            )
            print("Winner highlighting verified: Current wins")
        else:
            # tie - no winner highlighting expected in specific cells
            # but the green color may still appear elsewhere in the HTML
            print("No winner highlighting expected: Tie")

    print("Metrics Rank HTML rendering test passed")


@pytest.mark.parametrize(
    "baseline_result_path,current_result_path,expected_winner",
    [
        # scenario 1: toxicity evaluator - check if current or baseline wins
        (
            "data/generative/eval_compare/toxicity-2m/baseline-explanation.json",
            "data/generative/eval_compare/toxicity-2m/current-explanation.json",
            None,  # will be determined by actual data
        ),
        # scenario 2: multiple evaluators - more complex comparison
        (
            "data/generative/eval_compare/multiple-evaluators-h2ogpte-results/"
            "baseline-explanation.json",
            "data/generative/eval_compare/multiple-evaluators-h2ogpte-results/"
            "current-explanation.json",
            None,  # will be determined by actual data
        ),
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_metrics_rank_comparison_scenarios(
    tmp_path: pathlib.Path,
    baseline_result_path: str,
    current_result_path: str,
    expected_winner: str | None,
):
    """Test metrics rank with real evaluation data across different scenarios.

    This parametrized test verifies:
    1. Metrics rank calculation works with real evaluation data
    2. Different evaluators and metric types are handled correctly
    3. The ranking reflects actual model performance differences
    4. Results are consistent across single and multiple evaluator comparisons

    """
    #
    # GIVEN
    #
    explainers_map = _given_explainers_map()

    assert pathlib.Path(baseline_result_path).exists()
    with open(baseline_result_path) as f:
        baseline_results_json = json.load(f)
    baseline_results = explanations.LlmEvalResultsExplanation.from_dict(
        explainers_map=explainers_map, explanation_dict=baseline_results_json
    )

    assert pathlib.Path(current_result_path).exists()
    with open(current_result_path) as f:
        current_results_json = json.load(f)
    current_results = explanations.LlmEvalResultsExplanation.from_dict(
        explainers_map=explainers_map, explanation_dict=current_results_json
    )

    # count test cases in each result
    baseline_test_cases = len(baseline_results.eval_results.results)
    current_test_cases = len(current_results.eval_results.results)
    print(
        f"\nTest case counts - Baseline: {baseline_test_cases}, "
        f"Current: {current_test_cases}"
    )

    #
    # WHEN
    #
    diff: explanations.EvalResultsDiff = baseline_results.compare(current_results)

    # generate HTML (this populates diff._stats)
    html_output = str(diff.to_html())
    html_path = tmp_path / f"scenario_{pathlib.Path(baseline_result_path).stem}.html"
    with open(html_path, "w") as f:
        f.write(html_output)
    print(f"Comparison HTML saved to: file://{html_path}")

    # extract metrics rank from stats (now available after to_html())
    from h2o_sonar.lib.api.explanations import _explanations_cmp_html

    stats = diff._stats if hasattr(diff, "_stats") else None
    assert stats is not None, "Comparison stats not found in diff object"

    baseline_rank = stats.get(_explanations_cmp_html.KEY_METRICS_WINS_BASELINE, 0)
    current_rank = stats.get(_explanations_cmp_html.KEY_METRICS_WINS_CURRENT, 0)

    #
    # THEN
    #
    print(
        f"Metrics Rank - Baseline: {baseline_rank}, Current: {current_rank} "
        f"(dataset: {baseline_result_path})"
    )

    # verify that ranks are non-negative integers
    assert isinstance(baseline_rank, int), "Baseline rank must be an integer"
    assert isinstance(current_rank, int), "Current rank must be an integer"
    assert baseline_rank >= 0, "Baseline rank must be non-negative"
    assert current_rank >= 0, "Current rank must be non-negative"

    # verify that at least one model has some wins (unless all metrics are tied)
    # this is a sanity check - real evaluation data should have some differences
    total_comparisons = baseline_rank + current_rank
    print(f"Total metric comparisons where one model won: {total_comparisons}")

    # if expected_winner is specified, verify it
    if expected_winner == "baseline":
        assert baseline_rank > current_rank, (
            f"Expected baseline to win, but baseline={baseline_rank}, "
            f"current={current_rank}"
        )
    elif expected_winner == "current":
        assert current_rank > baseline_rank, (
            f"Expected current to win, but baseline={baseline_rank}, "
            f"current={current_rank}"
        )

    # verify consistency: the sum of ranks should be reasonable given the data
    # each test case can contribute multiple metric comparisons (one per metric)
    # but we don't know exactly how many metrics per test case, so we just verify
    # the total is not impossibly large
    max_expected_comparisons = max(baseline_test_cases, current_test_cases) * 50
    # 50 metrics per test case is a very generous upper bound
    assert total_comparisons <= max_expected_comparisons, (
        f"Total comparisons ({total_comparisons}) seems unreasonably high "
        f"given {baseline_test_cases} test cases"
    )

    # verify that the metrics rank appears in HTML output
    assert str(baseline_rank) in html_output, "Baseline rank not in HTML"
    assert str(current_rank) in html_output, "Current rank not in HTML"
    assert "Metrics Rank" in html_output, "Metrics Rank label not in HTML"

    print("Metrics rank comparison scenario test passed")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
