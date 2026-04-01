# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import loggers
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api.explanations import _explanations_base
from h2o_sonar.utils import caching
from h2o_sonar.utils import resource_mgmt


try:
    import nltk

    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    import sentence_transformers

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import bert_score

    HAS_BERT_SCORE = True
except ImportError:
    HAS_BERT_SCORE = False


# constants
KEY_ACTUAL_DURATION = "actual_duration"
KEY_ACTUAL_OUTPUT = "actual_output"
KEY_ACTUAL_OUTPUT_META = "actual_output_meta"
KEY_BASELINE_VALUE = "baseline_value"
KEY_BASELINE_WINS = "baseline_wins"
KEY_BASELINE_RANK_AVG = "baseline_rank_avg"
KEY_CURRENT_RANK_AVG = "current_rank_avg"
KEY_CATEGORIES = "categories"
KEY_CHANGED_METRICS_COUNT = "changed_metrics_count"
KEY_STATISTICS = "statistics"
KEY_FLIPPED_METRICS_COUNT_STATS = "flipped_metrics_count"
KEY_FLIPPED_TO_PASSED = "flipped_to_passed"
KEY_FLIPPED_TO_FAILED = "flipped_to_failed"
KEY_EMPTY_CONTEXT_COUNT = "empty_context_count"
KEY_TOTAL_TEST_CASES = "total_test_cases"
KEY_TOTAL_METRICS = "total_metrics"
KEY_CHUNKS = "chunks"
KEY_CHUNKS_COUNT = "chunks_count"
KEY_CHUNK_SIMILARITY = "chunk_similarity"
KEY_COMMON_CHUNKS = "common_chunks"
KEY_COMMON_COUNT = "common_count"
KEY_COMMON_SENTENCES = "common_sentences"
KEY_CONTEXT = "context"
KEY_CORPUS = "corpus"
KEY_COST = "cost"
KEY_CURRENT_VALUE = "current_value"
KEY_CURRENT_WINS = "current_wins"
KEY_DIFF = "diff"
KEY_DIFFS = "diffs"
KEY_EXPECTED_OUTPUT = "expected_output"
KEY_FLIPPED_METRICS_COUNT = "flipped_metrics_count"
KEY_IDENTICAL = "identical"
KEY_INPUT = "input"
KEY_KEY = "key"
KEY_LEADERBOARDS = "leaderboards"
KEY_MAGNITUDE = "magnitude"
KEY_METRICS = "metrics"
KEY_METRICS_META = "metrics_meta"
KEY_MODEL_KEY = "model_key"
KEY_OUTPUT_CONDITION = "output_condition"
KEY_OUTPUT_CONSTRAINTS = "output_constraints"
KEY_QUESTION = "question"
KEY_WINS = "wins"  # DEPRECATED: use leaderboard_position instead
KEY_LEADERBOARD_POSITION = "leaderboard_position"
KEY_DIFF_INDEX = "diff_index"
KEY_RELATIONSHIPS = "relationships"
KEY_RESULT_ERR_MSG = "result_error_message"
KEY_SENTENCES = "sentences"
KEY_SENTENCES_COUNT = "sentences_count"
KEY_SENTENCE_SIMILARITY = "sentence_similarity"
KEY_TEST_CASE_KEY = "test_case_key"
KEY_TEST_KEY = "test_key"
KEY_UNIQUE_CHUNKS = "unique_chunks"
KEY_UNIQUE_COUNT = "unique_count"
KEY_UNIQUE_SENTENCES = "unique_sentences"

# keys for protobuf-friendly JSON representation
KEY_METRIC_SCORES = "metric_scores"
KEY_METRIC_NAME = "metric_name"
KEY_METRIC_SCORE = "metric_score"
KEY_DIFF_KEY = "diff_key"
KEY_ITEMS = "items"
KEY_DIFF_CHANGED_METRICS = "diff_changed_metrics"

# keys for summary fields in JSON diff
KEY_SUMMARY = "summary"
KEY_RECOMMENDATION = "recommendation"
KEY_RECOMMENDATION_WINNER = "recommendation_winner"
KEY_RECOMMENDATION_EXPLANATION = "recommendation_explanation"
KEY_RECOMMENDATION_CONFIDENCE = "recommendation_confidence"
KEY_RECOMMENDATIONS_SUMMARY = "recommendations_summary"

# keys for models overview in JSON diff
KEY_MODELS_OVERVIEW = "models_overview"
KEY_BASELINE_MODEL_KEY = "baseline_model_key"
KEY_CURRENT_MODEL_KEY = "current_model_key"
KEY_BASELINE_MODEL_NAME = "baseline_model_name"
KEY_CURRENT_MODEL_NAME = "current_model_name"
KEY_BASELINE_COLLECTION_ID = "baseline_collection_id"
KEY_CURRENT_COLLECTION_ID = "current_collection_id"
KEY_MODELS_METADATA = "models_metadata"

# keys for models comparison in JSON diff
KEY_MODELS_COMPARISONS = "models_comparisons"
KEY_TEST_CASE_RANKS_BASELINE = "test_case_ranks_baseline"
KEY_TEST_CASE_RANKS_CURRENT = "test_case_ranks_current"
KEY_TEST_CASE_WINS_BASELINE = "test_case_wins_baseline"
KEY_TEST_CASE_WINS_CURRENT = "test_case_wins_current"

# keys for metrics comparison in JSON diff
KEY_MODELS_COMPARISONS_METRICS = "models_comparisons_metrics"
KEY_METRICS_RANKS_BASELINE = "metrics_ranks_baseline"
KEY_METRICS_RANKS_CURRENT = "metrics_ranks_current"
KEY_METRICS_WINS_BASELINE = "metrics_wins_baseline"
KEY_METRICS_WINS_CURRENT = "metrics_wins_current"
KEY_METRICS_AVERAGES = "metrics_averages"
KEY_METRIC_KEY = "metric_key"  # key field in metrics_averages list items
KEY_BASELINE = "baseline"
KEY_CURRENT = "current"

# keys for technical metrics in JSON diff
KEY_TECHNICAL_METRICS = "technical_metrics"

# keys for test cases leaderboard in JSON diff
KEY_TEST_CASES_LEADERBOARD = "test_cases_leaderboard"

# keys for overall evaluations comparison (across all model pairs)
KEY_OVERALL_COMPARISON = "overall_comparison"
KEY_OVERALL_SUMMARY = "overall_summary"
KEY_OVERALL_RECOMMENDATION = "overall_recommendation"
KEY_OVERALL_RECOMMENDATION_WINNER = "overall_recommendation_winner"
KEY_OVERALL_RECOMMENDATION_CONFIDENCE = "overall_recommendation_confidence"
KEY_OVERALL_EVALUATIONS_OVERVIEW = "overall_evaluations_overview"
KEY_OVERALL_MODELS_COMPARISON = "overall_models_comparison"
KEY_OVERALL_TECHNICAL_METRICS = "overall_technical_metrics"
KEY_BASELINE_MODELS_COUNT = "baseline_models_count"
KEY_CURRENT_MODELS_COUNT = "current_models_count"
KEY_BASELINE_MODEL_TYPES = "baseline_model_types"
KEY_CURRENT_MODEL_TYPES = "current_model_types"
KEY_BASELINE_COLLECTION_IDS = "baseline_collection_ids"
KEY_CURRENT_COLLECTION_IDS = "current_collection_ids"
KEY_BASELINE_UNIQUE_COLLECTIONS = "baseline_unique_collections"
KEY_CURRENT_UNIQUE_COLLECTIONS = "current_unique_collections"
KEY_TOTAL_COMPARABLE_TEST_CASES = "total_comparable_test_cases"
KEY_TOTAL_COMPARABLE_MODELS = "total_comparable_models"

# standard fields that are NOT metric scores in test case results
STANDARD_FIELDS = [
    "key",
    "input",
    "corpus",
    "context",
    "categories",
    "relationships",
    "expected_output",
    "output_constraints",
    "output_condition",
    "actual_output",
    "actual_duration",
    "cost",
    "model_key",
    "test_key",
    "test_case_key",
    "metrics",
    "metrics_meta",
    "actual_output_meta",
    "result_error_message",
]


def transform_test_case_result_to_protobuf_friendly(test_case_result: dict) -> dict:
    """Transform test case result dict to Protobuf-friendly format by moving
    metric scores into a separate list.

    All fields except STANDARD_FIELDS are considered metric scores and moved
    to the 'metric_scores' list.

    Parameters
    ----------
    test_case_result : dict
        Test case result dictionary with metrics as top-level keys.

    Returns
    -------
    dict
        Transformed dictionary with metrics in 'metric_scores' list.

    """
    result = {}
    metric_scores = []

    for key, value in test_case_result.items():
        if key in STANDARD_FIELDS:
            # keep standard fields as-is
            result[key] = value
        else:
            # move metric scores to list
            metric_scores.append({KEY_METRIC_NAME: key, KEY_METRIC_SCORE: value})

    # add metric scores list
    result[KEY_METRIC_SCORES] = metric_scores

    return result


def _assign_ranks_with_ties(
    values_with_labels: list[tuple[float, str]],
) -> dict[str, list[float]]:
    """Assign ranks with proper tie handling using average rank for tied values.

    For tied values, assigns the average of the ranks they would occupy.
    For example, if two values tie for ranks 3 and 4, both get rank 3.5.

    Parameters
    ----------
    values_with_labels : list[tuple[float, str]]
        List of (value, label) tuples, already sorted in ranking order
        (the best performance first).

    Returns
    -------
    dict[str, list[float]] :
        Dictionary mapping labels to lists of ranks assigned to them.

    Examples
    --------
    >>> values = [(0.9, "a"), (0.8, "b"), (0.8, "c"), (0.7, "d")]
    >>> _assign_ranks_with_ties(values)
    {'a': [4.0], 'b': [2.5], 'c': [2.5], 'd': [1.0]}
    """
    if not values_with_labels:
        return {}

    n = len(values_with_labels)
    ranks_by_label = {}

    i = 0
    while i < n:
        # find all values equal to current value
        current_value = values_with_labels[i][0]
        tie_start = i
        while i < n and values_with_labels[i][0] == current_value:
            i += 1
        tie_end = i

        # calculate average rank for tied values
        # ranks are inverted: position 0 gets rank n, position n-1 gets rank 1
        # for positions tie_start to tie_end-1, average the inverted ranks
        rank_sum = sum(n - pos for pos in range(tie_start, tie_end))
        avg_rank = rank_sum / (tie_end - tie_start)

        # assign average rank to all tied values
        for pos in range(tie_start, tie_end):
            _, label = values_with_labels[pos]
            if label not in ranks_by_label:
                ranks_by_label[label] = []
            ranks_by_label[label].append(avg_rank)

    return ranks_by_label


class EvalResultDiff:
    """Difference between two agent/RAG/LLM evaluation test cases ~ resolved and
    evaluated results. Test case diff items:

    - question/prompt/input (identical in both)
    - diff_flipped_metrics - list of metric which flipped w/ new state: pass/fail
    - baseline test case
        - test case key
        - ... all fields it had originally ...
        - diff_actual_output_meta - sentence level tokenization diff
          baseline vs. current
        - diff_retrieved_context - sentence level diff for every ctx item
          baseline vs. current
    - current test case
        - test case key
        - ... all fields it had originally ...
        - diff_actual_output_meta - sentence level tokenization diff
          current vs. baseline
        - diff_retrieved_context - sentence level diff for every ctx item
          current vs. baseline

    Sentence level diff metrics:

    - cosine distance of sentence embeddings

    """

    KEY_QUESTION = "question"
    KEY_DIFF_FLIPPED_METRICS = "diff_flipped_metrics"
    KEY_BASELINE_TEST_CASE_RESULT = "baseline_test_case_result"
    KEY_BASELINE_DIFF_AA_META = "baseline_diff_actual_output_meta"
    KEY_BASELINE_DIFF_RC = "baseline_diff_retrieved_context"
    KEY_CURRENT_TEST_CASE_RESULT = "current_test_case_result"
    KEY_CURRENT_DIFF_AA_META = "current_diff_actual_output_meta"
    KEY_CURRENT_DIFF_RC = "current_diff_retrieved_context"

    def __init__(
        self,
        question: str,
        expected_answer: str,
        diff_flipped_metrics: dict,
        diff_changed_metrics: dict,
        baseline_test_case_result: dict,
        baseline_diff_actual_output_meta: dict,
        baseline_diff_retrieved_context: dict,
        current_test_case_result: dict,
        current_diff_actual_output_meta: dict,
        current_diff_retrieved_context: dict,
    ):
        """Constructor for agent/RAG/LLM evaluation test case difference.

        Hints:

        - if embedding model is not available, then the diff will not
          contain sentence level similarity for actual answer and context chunks

        Parameters
        ----------
        question : str
            The question/prompt/input of the test case.
        expected_answer : str
            The expected answer to the question/prompt/input of the test case.
        diff_flipped_metrics : dict
            A dictionary of metrics that flipped with their new state (pass/fail).
        diff_changed_metrics : dict
            A dictionary of metrics that changed with their new state.
        baseline_test_case_result : LlmEvalResults.LlmEvalResultRow
            The baseline test case result data.
        baseline_diff_actual_output_meta : dict
            Sentence level tokenization diff - baseline vs. current.
        baseline_diff_retrieved_context : dict
            Sentence level diff for every context item - baseline vs. current.
        current_test_case_result : LlmEvalResults.LlmEvalResultRow
            The current test case result data.
        current_diff_actual_output_meta : dict
            Sentence level tokenization diff - current vs. baseline.
        current_diff_retrieved_context : dict
            Sentence level diff for every context item - current vs. baseline.

        """
        self.question = question
        self.expected_answer = expected_answer

        # map: metric name -> {baseline val, current val, threshold,
        #                      is baseline pass, is current pass}
        self.diff_flipped_metrics = diff_flipped_metrics
        self.diff_changed_metrics = diff_changed_metrics

        self.baseline_test_case = baseline_test_case_result
        # sentence level diff for actual output
        self.baseline_diff_actual_output_meta = baseline_diff_actual_output_meta
        # sentence level diff for context chunks
        self.baseline_diff_retrieved_context = baseline_diff_retrieved_context

        self.current_test_case = current_test_case_result
        # sentence level diff for actual output
        self.current_diff_actual_output_meta = current_diff_actual_output_meta
        # sentence level diff for context chunks
        self.current_diff_retrieved_context = current_diff_retrieved_context

    def to_dict(self):
        # transform diff_flipped_metrics from dict to list
        diff_flipped_metrics_list = []
        for metric_name, metric_info in self.diff_flipped_metrics.items():
            diff_flipped_metrics_list.append(
                {
                    KEY_METRIC_NAME: metric_name,
                    KEY_BASELINE_VALUE: metric_info.get(KEY_BASELINE_VALUE),
                    KEY_CURRENT_VALUE: metric_info.get(KEY_CURRENT_VALUE),
                }
            )

        # transform diff_changed_metrics from dict to list
        diff_changed_metrics_list = []
        for metric_name, metric_info in self.diff_changed_metrics.items():
            diff_changed_metrics_list.append(
                {
                    KEY_METRIC_NAME: metric_name,
                    KEY_BASELINE_VALUE: metric_info.get(KEY_BASELINE_VALUE),
                    KEY_CURRENT_VALUE: metric_info.get(KEY_CURRENT_VALUE),
                }
            )

        # transform test case results to protobuf-friendly format
        baseline_tc_result = transform_test_case_result_to_protobuf_friendly(
            self.baseline_test_case
        )
        current_tc_result = transform_test_case_result_to_protobuf_friendly(
            self.current_test_case
        )

        return {
            EvalResultDiff.KEY_QUESTION: self.question,
            EvalResultDiff.KEY_DIFF_FLIPPED_METRICS: diff_flipped_metrics_list,
            KEY_DIFF_CHANGED_METRICS: diff_changed_metrics_list,
            EvalResultDiff.KEY_BASELINE_TEST_CASE_RESULT: baseline_tc_result,
            EvalResultDiff.KEY_BASELINE_DIFF_AA_META: (
                self.baseline_diff_actual_output_meta
            ),
            EvalResultDiff.KEY_BASELINE_DIFF_RC: self.baseline_diff_retrieved_context,
            EvalResultDiff.KEY_CURRENT_TEST_CASE_RESULT: current_tc_result,
            EvalResultDiff.KEY_CURRENT_DIFF_AA_META: (
                self.current_diff_actual_output_meta
            ),
            EvalResultDiff.KEY_CURRENT_DIFF_RC: self.current_diff_retrieved_context,
        }

    def calculate_metric_change_magnitude(self, must_flip: bool = False) -> float:
        """Calculate the total magnitude of metric changes for this test case.

        Returns the sum of absolute differences for all numeric metrics that changed.
        For non-numeric metrics (like pass/fail), counts each change as 1.0.

        Returns
        -------
        float :
            Total magnitude of metric changes.

        """
        if must_flip and not self.diff_flipped_metrics:
            return 0.0

        total_magnitude = 0.0
        metrics_items = (
            self.diff_flipped_metrics.items()
            if must_flip
            else self.diff_changed_metrics.items()
        )
        for metric_key, metric_info in metrics_items:
            baseline_value = metric_info.get(KEY_BASELINE_VALUE)
            current_value = metric_info.get(KEY_CURRENT_VALUE)

            # try to calculate numeric difference
            try:
                if isinstance(baseline_value, (int, float)) and isinstance(
                    current_value, (int, float)
                ):
                    # numeric metrics - use absolute difference
                    total_magnitude += abs(current_value - baseline_value)
                else:
                    # non-numeric metrics (e.g., pass/fail) - count as 1.0 per change
                    total_magnitude += 1.0
            except (TypeError, ValueError):
                # if comparison fails, count as 1.0
                total_magnitude += 1.0

        return total_magnitude


class EvalResultsDiff:
    """Difference between two agent/RAG/LLM evaluation results:

    - Test cases with identical question/prompt/input are compared.
    - If LLM eval results file has > 1 explainable model,
      then also explainable model key must match
    - Result diff content - see ``EvalResultsDiff`` for details.

    """

    def __init__(
        self,
        diffs: dict[tuple[str, str], list[EvalResultDiff]],
        baseline_explainable_models: (
            list[models.ExplainableRagModel | models.ExplainableLlmModel] | None
        ) = None,
        current_explainable_models: (
            list[models.ExplainableRagModel | models.ExplainableLlmModel] | None
        ) = None,
        comparison_method: _explanations_base.SentenceComparisonMethod = (
            _explanations_base.SentenceComparisonMethod.BERT_SCORE
        ),
        metrics_meta: dict | None = None,
        branding: commons.Branding = commons.Branding.H2O_SONAR,
    ):
        """Evaluation results diffs constructor.

        Parameters
        ----------
        diffs : dict[tuple[str,str], list[EvalResultDiff]]
            Dictionary mapping (baseline_model_key, current_model_key) tuples to
            lists of EvalResultDiff objects. This groups test cases of
            comparable models together.
        baseline_explainable_models : list[ExplainableRagModel | ExplainableLlmModel]
            List of explainable models from baseline evaluation results.
        current_explainable_models : list[ExplainableRagModel | ExplainableLlmModel]
            List of explainable models from current evaluation results.
        comparison_method : _explanations_base.SentenceComparisonMethod
            Comparison method to use for text matching.
        metrics_meta : dict | None
            Metrics metadata dictionary (key -> MetricMeta object).
        branding : commons.Branding
            H2O Sonar vs. H2O Eval Studio branding.

        """

        self.comparison_method = comparison_method
        self.diffs = diffs
        self.baseline_explainable_models = baseline_explainable_models or []
        self.current_explainable_models = current_explainable_models or []
        self.metrics_meta = metrics_meta
        self.branding = branding

    def to_dict(self, test_cases_leaderboard_limit: int = 10_000):
        """Convert diffs to dictionary structure for JSON serialization.

        Transforms the structure to be Protobuf-friendly by converting
        dictionaries with dynamic keys to lists.

        Parameters
        ----------
        test_cases_leaderboard_limit : int
            Limit the number of test cases leader board to return.

        Returns
        -------
        dict :
            Dictionary with two keys:
            - "diffs": list of diff objects with "diff_key", "items", and
              "test_cases_leaderboard"
            - "metrics_meta": dictionary of metric metadata
              (key -> MetricMeta dict)

        """
        # import calculation methods from HTML module
        from h2o_sonar.lib.api.explanations._explanations_cmp_html import (
            EvalResultsDiffHtml,
        )

        # transform diffs dictionary to list
        result_diffs_list = []

        for model_pair, diff_list in self.diffs.items():
            key_str = f"{model_pair[0]}|{model_pair[1]}"
            baseline_model_key = model_pair[0]
            current_model_key = model_pair[1]

            # get explainable models for this pair
            baseline_model = self._get_explainable_model_by_key(
                baseline_model_key, is_baseline=True
            )
            current_model = self._get_explainable_model_by_key(
                current_model_key, is_baseline=False
            )

            # prepare model_diffs as list of (idx, diff) tuples for stats calculation
            model_diffs = [(idx, diff) for idx, diff in enumerate(diff_list, start=1)]

            # calculate comparison statistics
            stats = EvalResultsDiffHtml._calculate_model_cmp_stats(
                model_diffs=model_diffs,
                metrics_meta=self.metrics_meta,
                baseline_model=baseline_model,
                current_model=current_model,
            )

            # calculate recommendation
            recommendation = EvalResultsDiffHtml._calculate_recommendation(
                stats=stats,
                metrics_meta=self.metrics_meta,
            )

            # build summary section
            summary = {
                KEY_RECOMMENDATION_WINNER: recommendation.get("winner", "tie"),
                KEY_RECOMMENDATION: recommendation.get("explanation", ""),
                KEY_RECOMMENDATION_CONFIDENCE: recommendation.get("confidence", "low"),
            }

            # build models_overview section
            models_overview = {
                KEY_BASELINE_MODEL_KEY: baseline_model_key,
                KEY_CURRENT_MODEL_KEY: current_model_key,
            }

            # add model names and collection IDs if available
            if baseline_model:
                models_overview[KEY_BASELINE_MODEL_NAME] = baseline_model.llm_model_name
                if hasattr(baseline_model, "collection_id"):
                    # normalize collection_id to always be a list
                    cid = baseline_model.collection_id
                    models_overview[KEY_BASELINE_COLLECTION_ID] = (
                        cid if isinstance(cid, list) else [cid] if cid else []
                    )

            if current_model:
                models_overview[KEY_CURRENT_MODEL_NAME] = current_model.llm_model_name
                if hasattr(current_model, "collection_id"):
                    # normalize collection_id to always be a list
                    cid = current_model.collection_id
                    models_overview[KEY_CURRENT_COLLECTION_ID] = (
                        cid if isinstance(cid, list) else [cid] if cid else []
                    )

            # build models_comparisons section
            models_comparisons = {
                KEY_TEST_CASE_RANKS_BASELINE: stats.get(
                    KEY_TEST_CASE_RANKS_BASELINE, 0
                ),
                KEY_TEST_CASE_RANKS_CURRENT: stats.get(KEY_TEST_CASE_RANKS_CURRENT, 0),
                KEY_TEST_CASE_WINS_BASELINE: stats.get(KEY_TEST_CASE_WINS_BASELINE, 0),
                KEY_TEST_CASE_WINS_CURRENT: stats.get(KEY_TEST_CASE_WINS_CURRENT, 0),
            }

            # build models_comparisons_metrics section
            # transform metrics_averages from dict to list for Protobuf compatibility
            metrics_averages_dict = stats.get(KEY_METRICS_AVERAGES, {})
            metrics_averages_list = []
            for metric_name, metric_data in metrics_averages_dict.items():
                metric_entry = {KEY_METRIC_KEY: metric_name}
                # add all metric data fields to the entry
                metric_entry.update(metric_data)
                metrics_averages_list.append(metric_entry)

            models_comparisons_metrics = {
                KEY_METRICS_RANKS_BASELINE: stats.get(KEY_METRICS_RANKS_BASELINE, 0.0),
                KEY_METRICS_RANKS_CURRENT: stats.get(KEY_METRICS_RANKS_CURRENT, 0.0),
                KEY_METRICS_WINS_BASELINE: stats.get(KEY_METRICS_WINS_BASELINE, 0),
                KEY_METRICS_WINS_CURRENT: stats.get(KEY_METRICS_WINS_CURRENT, 0),
                KEY_METRICS_AVERAGES: metrics_averages_list,
            }

            # build technical_metrics section
            technical_metrics = stats.get(KEY_TECHNICAL_METRICS, {})

            # generate test cases leaderboard
            leaderboard = self._generate_leaderboard(
                diff_list,
                top_n=test_cases_leaderboard_limit,
                metrics_meta=self.metrics_meta,
            )
            test_cases_leaderboard = [
                {
                    KEY_LEADERBOARD_POSITION: entry[KEY_LEADERBOARD_POSITION],
                    KEY_WINS: entry[KEY_WINS],  # backward compatibility
                    KEY_QUESTION: entry[KEY_QUESTION],
                    KEY_CHANGED_METRICS_COUNT: entry[KEY_CHANGED_METRICS_COUNT],
                    KEY_DIFF_INDEX: entry[KEY_DIFF_INDEX],
                    KEY_BASELINE_WINS: entry[KEY_BASELINE_WINS],
                    KEY_CURRENT_WINS: entry[KEY_CURRENT_WINS],
                    KEY_BASELINE_RANK_AVG: entry[KEY_BASELINE_RANK_AVG],
                    KEY_CURRENT_RANK_AVG: entry[KEY_CURRENT_RANK_AVG],
                }
                for entry in leaderboard
            ]

            # extract and serialize model metadata
            models_metadata = {}
            if baseline_model:
                models_metadata[KEY_BASELINE] = {
                    "llm_model_meta": (
                        baseline_model.llm_model_meta
                        if hasattr(baseline_model, "llm_model_meta")
                        else None
                    ),
                    "model_type": (
                        str(baseline_model.model_type)
                        if hasattr(baseline_model, "model_type")
                        else None
                    ),
                }
            if current_model:
                models_metadata[KEY_CURRENT] = {
                    "llm_model_meta": (
                        current_model.llm_model_meta
                        if hasattr(current_model, "llm_model_meta")
                        else None
                    ),
                    "model_type": (
                        str(current_model.model_type)
                        if hasattr(current_model, "model_type")
                        else None
                    ),
                }

            # add statistics summary from stats dict
            statistics = {
                KEY_FLIPPED_METRICS_COUNT_STATS: stats.get(
                    KEY_FLIPPED_METRICS_COUNT_STATS, 0
                ),
                KEY_FLIPPED_TO_PASSED: stats.get(KEY_FLIPPED_TO_PASSED, 0),
                KEY_FLIPPED_TO_FAILED: stats.get(KEY_FLIPPED_TO_FAILED, 0),
                KEY_EMPTY_CONTEXT_COUNT: stats.get(KEY_EMPTY_CONTEXT_COUNT, {}),
                KEY_TOTAL_TEST_CASES: stats.get(KEY_TOTAL_TEST_CASES, 0),
                KEY_TOTAL_METRICS: stats.get(KEY_TOTAL_METRICS, 0),
            }

            # add diff with all new fields including embedded leaderboard
            result_diffs_list.append(
                {
                    KEY_DIFF_KEY: key_str,
                    KEY_SUMMARY: summary,
                    KEY_MODELS_OVERVIEW: models_overview,
                    KEY_MODELS_COMPARISONS: models_comparisons,
                    KEY_MODELS_COMPARISONS_METRICS: models_comparisons_metrics,
                    KEY_TECHNICAL_METRICS: technical_metrics,
                    KEY_TEST_CASES_LEADERBOARD: test_cases_leaderboard,
                    KEY_MODELS_METADATA: models_metadata,
                    KEY_STATISTICS: statistics,
                    KEY_ITEMS: [diff.to_dict() for diff in diff_list],
                }
            )

        # serialize metrics_meta if available
        metrics_meta_dict = None
        if self.metrics_meta:
            metrics_meta_dict = {
                key: metric.to_dict() for key, metric in self.metrics_meta.items()
            }

        # compute overall comparison if multiple model pairs
        overall_comparison = None
        if len(result_diffs_list) > 1:
            overall_comparison = self._compute_overall_comparison(result_diffs_list)

        # build result dict
        result = {
            KEY_DIFFS: result_diffs_list,
            KEY_METRICS_META: metrics_meta_dict,
        }

        # add overall comparison if computed
        if overall_comparison:
            result[KEY_OVERALL_COMPARISON] = overall_comparison

        return result

    def _compute_overall_comparison(self, diffs_list: list[dict]) -> dict:
        """Compute overall comparison statistics across all model pairs.

        Aggregates metrics from all model-pair comparisons to provide
        an overall assessment of baseline vs current evaluations.

        Parameters
        ----------
        diffs_list : list[dict]
            List of diff dictionaries (one per model pair), each containing
            models_overview, models_comparisons, technical_metrics, etc.

        Returns
        -------
        dict :
            Dictionary containing:
            - overall_summary: recommendation, winner, confidence
            - overall_evaluations_overview: model counts, types, collections
            - overall_models_comparison: aggregated test case/metric wins and ranks
            - overall_technical_metrics: aggregated costs, durations, counts

        """
        # initialize aggregation variables
        baseline_models = set()
        current_models = set()
        baseline_model_types = set()
        current_model_types = set()
        baseline_collections = set()
        current_collections = set()
        total_test_cases = 0

        # aggregated comparison metrics (SUM across all models)
        test_case_ranks_baseline = 0
        test_case_ranks_current = 0
        test_case_wins_baseline = 0
        test_case_wins_current = 0
        metrics_ranks_baseline = 0.0
        metrics_ranks_current = 0.0
        metrics_wins_baseline = 0
        metrics_wins_current = 0

        # technical metrics aggregation
        baseline_tech = {
            "cost_sum": 0.0,
            "duration_sum": 0.0,
            "duration_min": float("inf"),
            "duration_max": 0.0,
            "success_count": 0,
            "failure_count": 0,
            "call_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        current_tech = {
            "cost_sum": 0.0,
            "duration_sum": 0.0,
            "duration_min": float("inf"),
            "duration_max": 0.0,
            "success_count": 0,
            "failure_count": 0,
            "call_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        # iterate through diffs_list (per-model-pair data)
        for diff_entry in diffs_list:
            # extract data from each diff_entry
            models_overview = diff_entry[KEY_MODELS_OVERVIEW]
            models_comparison = diff_entry[KEY_MODELS_COMPARISONS]
            models_comparison_metrics = diff_entry[KEY_MODELS_COMPARISONS_METRICS]
            technical_metrics = diff_entry[KEY_TECHNICAL_METRICS]

            # collect model information
            baseline_model_key = models_overview[KEY_BASELINE_MODEL_KEY]
            current_model_key = models_overview[KEY_CURRENT_MODEL_KEY]
            baseline_models.add(baseline_model_key)
            current_models.add(current_model_key)

            # collect model types (names)
            if KEY_BASELINE_MODEL_NAME in models_overview:
                baseline_model_types.add(models_overview[KEY_BASELINE_MODEL_NAME])
            if KEY_CURRENT_MODEL_NAME in models_overview:
                current_model_types.add(models_overview[KEY_CURRENT_MODEL_NAME])

            # collect collection IDs
            if KEY_BASELINE_COLLECTION_ID in models_overview:
                for cid in models_overview[KEY_BASELINE_COLLECTION_ID]:
                    if cid:
                        baseline_collections.add(cid)
            if KEY_CURRENT_COLLECTION_ID in models_overview:
                for cid in models_overview[KEY_CURRENT_COLLECTION_ID]:
                    if cid:
                        current_collections.add(cid)

            # SUM comparison metrics
            test_case_ranks_baseline += models_comparison[KEY_TEST_CASE_RANKS_BASELINE]
            test_case_ranks_current += models_comparison[KEY_TEST_CASE_RANKS_CURRENT]
            test_case_wins_baseline += models_comparison[KEY_TEST_CASE_WINS_BASELINE]
            test_case_wins_current += models_comparison[KEY_TEST_CASE_WINS_CURRENT]

            metrics_ranks_baseline += models_comparison_metrics[
                KEY_METRICS_RANKS_BASELINE
            ]
            metrics_ranks_current += models_comparison_metrics[
                KEY_METRICS_RANKS_CURRENT
            ]
            metrics_wins_baseline += models_comparison_metrics[
                KEY_METRICS_WINS_BASELINE
            ]
            metrics_wins_current += models_comparison_metrics[KEY_METRICS_WINS_CURRENT]

            # SUM and MIN/MAX technical metrics for baseline
            if KEY_BASELINE in technical_metrics:
                baseline_data = technical_metrics[KEY_BASELINE]
                baseline_tech["cost_sum"] += baseline_data.get("cost_sum", 0.0)
                baseline_tech["duration_sum"] += baseline_data.get("duration_sum", 0.0)
                baseline_tech["duration_min"] = min(
                    baseline_tech["duration_min"],
                    baseline_data.get("duration_min", float("inf")),
                )
                baseline_tech["duration_max"] = max(
                    baseline_tech["duration_max"],
                    baseline_data.get("duration_max", 0.0),
                )
                baseline_tech["success_count"] += baseline_data.get("success_count", 0)
                baseline_tech["failure_count"] += baseline_data.get("failure_count", 0)
                baseline_tech["call_count"] += baseline_data.get("call_count", 0)
                baseline_tech["input_tokens"] += baseline_data.get("input_tokens", 0)
                baseline_tech["output_tokens"] += baseline_data.get("output_tokens", 0)

            # SUM and MIN/MAX technical metrics for current
            if KEY_CURRENT in technical_metrics:
                current_data = technical_metrics[KEY_CURRENT]
                current_tech["cost_sum"] += current_data.get("cost_sum", 0.0)
                current_tech["duration_sum"] += current_data.get("duration_sum", 0.0)
                current_tech["duration_min"] = min(
                    current_tech["duration_min"],
                    current_data.get("duration_min", float("inf")),
                )
                current_tech["duration_max"] = max(
                    current_tech["duration_max"],
                    current_data.get("duration_max", 0.0),
                )
                current_tech["success_count"] += current_data.get("success_count", 0)
                current_tech["failure_count"] += current_data.get("failure_count", 0)
                current_tech["call_count"] += current_data.get("call_count", 0)
                current_tech["input_tokens"] += current_data.get("input_tokens", 0)
                current_tech["output_tokens"] += current_data.get("output_tokens", 0)

            # count total test cases
            total_test_cases += len(diff_entry[KEY_ITEMS])

        # handle edge case where no durations were found
        if baseline_tech["duration_min"] == float("inf"):
            baseline_tech["duration_min"] = 0.0
        if current_tech["duration_min"] == float("inf"):
            current_tech["duration_min"] = 0.0

        # compute overall recommendation
        total_wins_baseline = test_case_wins_baseline + metrics_wins_baseline
        total_wins_current = test_case_wins_current + metrics_wins_current

        if total_wins_baseline > total_wins_current:
            winner = "baseline"
            confidence = (
                "high" if total_wins_baseline > total_wins_current * 1.5 else "medium"
            )
        elif total_wins_current > total_wins_baseline:
            winner = "current"
            confidence = (
                "high" if total_wins_current > total_wins_baseline * 1.5 else "medium"
            )
        else:
            winner = "tie"
            confidence = "low"

        # format recommendation text
        if winner == "baseline":
            winner_text = "baseline"
        elif winner == "current":
            winner_text = "current"
        else:
            winner_text = "tied"

        recommendation = (
            f"Overall evaluation shows {winner_text} models performing better across "
            f"{len(baseline_models)} model comparison(s). "
            f"Total wins: baseline={total_wins_baseline}, "
            f"current={total_wins_current}. "
            f"Based on aggregated metrics across {total_test_cases} test cases."
        )

        # calculate recommendations summary
        recommendations_summary = {KEY_BASELINE: 0, "tie": 0, KEY_CURRENT: 0}
        for diff_entry in diffs_list:
            rec_winner = diff_entry[KEY_SUMMARY].get(KEY_RECOMMENDATION_WINNER, "tie")
            if rec_winner in recommendations_summary:
                recommendations_summary[rec_winner] += 1

        # return aggregated data
        return {
            KEY_OVERALL_SUMMARY: {
                KEY_OVERALL_RECOMMENDATION: recommendation,
                KEY_OVERALL_RECOMMENDATION_WINNER: winner,
                KEY_OVERALL_RECOMMENDATION_CONFIDENCE: confidence,
                KEY_RECOMMENDATIONS_SUMMARY: recommendations_summary,
            },
            KEY_OVERALL_EVALUATIONS_OVERVIEW: {
                KEY_BASELINE_MODELS_COUNT: len(baseline_models),
                KEY_CURRENT_MODELS_COUNT: len(current_models),
                KEY_BASELINE_MODEL_TYPES: sorted(list(baseline_model_types)),
                KEY_CURRENT_MODEL_TYPES: sorted(list(current_model_types)),
                KEY_BASELINE_UNIQUE_COLLECTIONS: len(baseline_collections),
                KEY_CURRENT_UNIQUE_COLLECTIONS: len(current_collections),
                KEY_TOTAL_COMPARABLE_MODELS: len(baseline_models),
                KEY_TOTAL_COMPARABLE_TEST_CASES: total_test_cases,
            },
            KEY_OVERALL_MODELS_COMPARISON: {
                KEY_TEST_CASE_RANKS_BASELINE: test_case_ranks_baseline,
                KEY_TEST_CASE_RANKS_CURRENT: test_case_ranks_current,
                KEY_TEST_CASE_WINS_BASELINE: test_case_wins_baseline,
                KEY_TEST_CASE_WINS_CURRENT: test_case_wins_current,
                KEY_METRICS_RANKS_BASELINE: metrics_ranks_baseline,
                KEY_METRICS_RANKS_CURRENT: metrics_ranks_current,
                KEY_METRICS_WINS_BASELINE: metrics_wins_baseline,
                KEY_METRICS_WINS_CURRENT: metrics_wins_current,
            },
            KEY_OVERALL_TECHNICAL_METRICS: {
                KEY_BASELINE: baseline_tech,
                KEY_CURRENT: current_tech,
            },
        }

    @staticmethod
    def _generate_leaderboard(
        diff_list: list[EvalResultDiff],
        top_n: int = 10,
        metrics_meta: dict | None = None,
        min_rank_diff_threshold: float = 0.0,
    ) -> list[dict]:
        """Generate leaderboard of test cases with most significant metric changes.

        Calculates wins and ranks for each test case to enable JSON-based HTML
        regeneration without needing to recalculate from raw diffs.

        Test cases are sorted by absolute difference between baseline and current
        ranks (descending), then filtered by min_rank_diff_threshold, and finally
        limited to top_n entries.

        Parameters
        ----------
        diff_list : list[EvalResultDiff]
            List of test case diffs to rank.
        top_n : int
            Number of top test cases to include in leaderboard (default: 10).
        metrics_meta : dict | None
            Metrics metadata dictionary for determining higher_is_better.
        min_rank_diff_threshold : float
            Minimum absolute rank difference required to include test case in
            leaderboard. Only test cases where abs(baseline_rank_avg -
            current_rank_avg) >= threshold are included (default: 0.0).

        Returns
        -------
        list[dict] :
            List of leaderboard entries, each containing:
            - leaderboard_position: position in leaderboard (1-based)
            - wins: DEPRECATED - equals baseline_wins (for backward compatibility only)
            - question: test case question/prompt
            - changed_metrics_count: number of metrics that changed values
            - diff_index: index of the diff in the items array (0-based)
            - baseline_wins: count of metrics where baseline scored better
            - current_wins: count of metrics where current scored better
            - baseline_rank_avg: average rank for baseline across metrics
            - current_rank_avg: average rank for current across metrics
            - diff: reference to the diff object (for HTML generation, not serialized)

        """
        # standard fields to skip when extracting metrics
        skip_list = [
            KEY_KEY,
            KEY_ACTUAL_DURATION,
            KEY_ACTUAL_OUTPUT,
            KEY_ACTUAL_OUTPUT_META,
            KEY_CATEGORIES,
            KEY_CONTEXT,
            KEY_CORPUS,
            KEY_COST,
            KEY_EXPECTED_OUTPUT,
            KEY_INPUT,
            KEY_METRICS_META,
            KEY_MODEL_KEY,
            KEY_OUTPUT_CONDITION,
            KEY_OUTPUT_CONSTRAINTS,
            KEY_RELATIONSHIPS,
            KEY_RESULT_ERR_MSG,
            KEY_TEST_CASE_KEY,
            KEY_TEST_KEY,
        ]

        # calculate wins and ranks for each diff
        ranked_diffs = []
        for diff_index, diff in enumerate(diff_list):
            # count all metrics that changed values (not just flipped)
            changed_count = (
                len(diff.diff_changed_metrics) if diff.diff_changed_metrics else 0
            )
            if changed_count == 0:
                continue  # only include test cases with changes

            # extract metrics from baseline and current test cases
            baseline_metrics = {
                k: v for k, v in diff.baseline_test_case.items() if k not in skip_list
            }
            current_metrics = {
                k: v for k, v in diff.current_test_case.items() if k not in skip_list
            }

            # calculate wins
            baseline_wins = 0
            current_wins = 0
            all_metric_keys = set(baseline_metrics.keys()) & set(current_metrics.keys())

            for metric_key in all_metric_keys:
                baseline_value = baseline_metrics.get(metric_key)
                current_value = current_metrics.get(metric_key)

                # skip if not numeric or equal
                if (
                    baseline_value is None
                    or current_value is None
                    or not isinstance(baseline_value, (int, float))
                    or not isinstance(current_value, (int, float))
                    or baseline_value == current_value
                ):
                    continue

                # get metric directionality
                if not metrics_meta or metric_key not in metrics_meta:
                    continue  # skip if no metadata available

                metric_meta = metrics_meta[metric_key]

                # extract higher_is_better
                if hasattr(metric_meta, "higher_is_better"):
                    higher_is_better = metric_meta.higher_is_better
                elif (
                    isinstance(metric_meta, dict) and "higher_is_better" in metric_meta
                ):
                    higher_is_better = metric_meta["higher_is_better"]
                else:
                    continue  # skip if no directionality info

                # count wins
                if higher_is_better:
                    if baseline_value > current_value:
                        baseline_wins += 1
                    elif current_value > baseline_value:
                        current_wins += 1
                else:  # lower is better
                    if baseline_value < current_value:
                        baseline_wins += 1
                    elif current_value < baseline_value:
                        current_wins += 1

            # calculate ranks
            baseline_ranks = []
            current_ranks = []

            for metric_key in all_metric_keys:
                baseline_value = baseline_metrics.get(metric_key)
                current_value = current_metrics.get(metric_key)

                # skip if not numeric or equal
                if (
                    baseline_value is None
                    or current_value is None
                    or not isinstance(baseline_value, (int, float))
                    or not isinstance(current_value, (int, float))
                    or baseline_value == current_value
                ):
                    continue

                # get metric directionality
                if not metrics_meta or metric_key not in metrics_meta:
                    continue

                metric_meta = metrics_meta[metric_key]

                # extract higher_is_better
                if hasattr(metric_meta, "higher_is_better"):
                    higher_is_better = metric_meta.higher_is_better
                elif (
                    isinstance(metric_meta, dict) and "higher_is_better" in metric_meta
                ):
                    higher_is_better = metric_meta["higher_is_better"]
                else:
                    continue

                # rank these two values (n=2)
                values_with_origin = [
                    (baseline_value, "baseline"),
                    (current_value, "current"),
                ]
                values_with_origin.sort(key=lambda x: x[0], reverse=higher_is_better)

                # assign ranks with proper tie handling
                ranks_by_label = _assign_ranks_with_ties(values_with_origin)

                # extract ranks for baseline and current
                if "baseline" in ranks_by_label:
                    baseline_ranks.extend(ranks_by_label["baseline"])
                if "current" in ranks_by_label:
                    current_ranks.extend(ranks_by_label["current"])

            # calculate average ranks
            baseline_rank_avg = 0.0
            current_rank_avg = 0.0
            if baseline_ranks:
                baseline_rank_avg = sum(baseline_ranks) / len(baseline_ranks)
            if current_ranks:
                current_rank_avg = sum(current_ranks) / len(current_ranks)

            ranked_diffs.append(
                {
                    KEY_QUESTION: diff.question,
                    KEY_CHANGED_METRICS_COUNT: changed_count,
                    KEY_DIFF_INDEX: diff_index,  # 0-based index in items array
                    KEY_BASELINE_WINS: baseline_wins,
                    KEY_CURRENT_WINS: current_wins,
                    KEY_BASELINE_RANK_AVG: baseline_rank_avg,
                    KEY_CURRENT_RANK_AVG: current_rank_avg,
                    KEY_DIFF: diff,  # keep reference for HTML generation
                }
            )

        # sort by: absolute difference between baseline and current ranks (desc)
        ranked_diffs.sort(
            key=lambda x: -abs(x[KEY_BASELINE_RANK_AVG] - x[KEY_CURRENT_RANK_AVG])
        )

        # helper to check if entry meets rank difference threshold
        def meets_threshold(entry):
            rank_diff = abs(entry[KEY_BASELINE_RANK_AVG] - entry[KEY_CURRENT_RANK_AVG])
            if min_rank_diff_threshold > 0.0:
                # stricter threshold (subsumes zero-diff exclusion)
                return rank_diff >= min_rank_diff_threshold
            else:
                # ALWAYS exclude test cases with zero rank difference
                # (no performance difference between baseline and current)
                return rank_diff > 0.0

        # filter by rank difference threshold
        ranked_diffs = [entry for entry in ranked_diffs if meets_threshold(entry)]

        # take top N and add leaderboard position
        leaderboard = []
        for position, entry in enumerate(ranked_diffs[:top_n], start=1):
            leaderboard.append(
                {
                    KEY_LEADERBOARD_POSITION: position,
                    KEY_WINS: entry[
                        KEY_BASELINE_WINS
                    ],  # deprecated: backward compatibility
                    KEY_QUESTION: entry[KEY_QUESTION],
                    KEY_CHANGED_METRICS_COUNT: entry[KEY_CHANGED_METRICS_COUNT],
                    KEY_DIFF_INDEX: entry[KEY_DIFF_INDEX],
                    KEY_BASELINE_WINS: entry[KEY_BASELINE_WINS],
                    KEY_CURRENT_WINS: entry[KEY_CURRENT_WINS],
                    KEY_BASELINE_RANK_AVG: entry[KEY_BASELINE_RANK_AVG],
                    KEY_CURRENT_RANK_AVG: entry[KEY_CURRENT_RANK_AVG],
                    KEY_DIFF: entry[KEY_DIFF],
                }
            )

        return leaderboard

    def _get_explainable_model_by_key(
        self, model_key: str, is_baseline: bool = True
    ) -> models.ExplainableRagModel | models.ExplainableLlmModel | None:
        """Get explainable model by key.

        Parameters
        ----------
        model_key : str
            The model key to search for.
        is_baseline : bool
            Whether to search in baseline (True) or current (False) explainable models.

        Returns
        -------
        models.ExplainableRagModel | models.ExplainableLlmModel | None :
            The explainable model object or None if not found.
        """
        models_list = (
            self.baseline_explainable_models
            if is_baseline
            else self.current_explainable_models
        )
        for model in models_list:
            if model.key == model_key:
                return model
        return None

    def to_html(self) -> str:
        """Generate HTML representation of evaluation results differences.

        Returns
        -------
        str :
            HTML string representation of the diffs.
        """
        from h2o_sonar.lib.api.explanations._explanations_cmp_html import (
            EvalResultsDiffHtml,
        )

        html_generator = EvalResultsDiffHtml(self, branding=self.branding)
        return html_generator.to_html()


class EvalResultsExplanationsComparator:
    """Class which compares agent/RAG/LLM evaluation results explanations."""

    _e_model_baai_bge = caching.MODEL_BAAI_BGE_SMALL_EN

    def __init__(
        self,
        baseline_explanation,
        current_explanation,
        logger: loggers.SonarLogger | None = None,
        comparison_method: _explanations_base.SentenceComparisonMethod = (
            _explanations_base.SentenceComparisonMethod.COSINE_DISTANCE
        ),
        sentence_similarity_threshold: float = 0.9,
        branding: commons.Branding = commons.Branding.H2O_SONAR,
    ):
        """Constructor for agent/RAG/LLM results comparator.

        Parameters
        ----------
        baseline_explanation : explanations.LlmEvalResultsExplanation
            The baseline LLM evaluation results.
        current_explanation : explanations.LlmEvalResultsExplanation
            The current LLM evaluation results.
        logger : loggers.SonarLogger | None
            The logger instance.
        comparison_method : SentenceComparisonMethod
            The method to use for comparing sentences:
            - EXACT_MATCH: exact string matching
            - COSINE_DISTANCE: cosine distance of sentence embeddings (default)
            - BERT_SCORE: BERTScore contextual embeddings similarity
        sentence_similarity_threshold : float
            Threshold for determining if sentences are "common" (high similarity).
            Sentences with similarity >= threshold are considered common.
            Default is 0.9.
        branding : commons.Branding
            H2O Sonar vs. H2O Eval Studio branding.

        """
        self.logger = logger or loggers.SonarPrintLogger()

        self.baseline = baseline_explanation
        self.current = current_explanation
        self.comparison_method = comparison_method
        self.sentence_similarity_threshold = sentence_similarity_threshold
        self.branding = branding
        self._embedding_model = None

    @staticmethod
    def _are_explainable_models_equal(
        m1: models.ExplainableRagModel | models.ExplainableLlmModel,
        m2: models.ExplainableRagModel | models.ExplainableLlmModel,
    ):
        """Check whether the models are equal either by key or by value:

        - RAG vs. LLM models are distinguished
        - model host type is considered, model host is NOT
        - model cfg is NOT considered
        - explainable RAG model is compared based on:
          host type, LLM model, collection ID, name and corpus documents
        - explainable LLM model is compared based on:
          host type and LLM model

        Parameters
        ----------
        m1 : models.ExplainableRagModel | models.ExplainableLlmModel,
            The referential model.
        m2 : models.ExplainableRagModel | models.ExplainableLlmModel,
            The model to compare against.

        Returns
        -------
        bool :
           Whether the models are equal or not.

        """
        if isinstance(m1, models.ExplainableRagModel):
            if isinstance(m2, models.ExplainableRagModel):
                if m1.key == m2.key:
                    return True
                elif (
                    m1.model_type == m2.model_type
                    and m1.llm_model_name == m2.llm_model_name
                    and m1.collection_id == m2.collection_id
                    and m1.collection_name == m2.collection_name
                    and m1.documents == m2.documents
                ):
                    return True
        elif isinstance(m2, models.ExplainableLlmModel):
            if m1.key == m2.key:
                return True
            elif (
                m1.model_type == m2.model_type
                and m1.llm_model_name == m2.llm_model_name
            ):
                return True

        return False

    @staticmethod
    def _are_explainable_models_llm_equal(
        m1: models.ExplainableRagModel | models.ExplainableLlmModel,
        m2: models.ExplainableRagModel | models.ExplainableLlmModel,
        m1_llm_model: str,
        m2_llm_model: str,
    ):
        """Check whether the models are equal given the LLM Model names. The use case
        is comparison of explainable models on different host types (like h2oGPTe vs.
        AWS Bedrock) where LLM model names are different (names fixed by provider),
        collection IDs are (very) different, connections are different, ... but user
        know what they want to compare:

        - user provides LLM model NAME for both baseline and current models
        - LLM model names are used for filtering and equals of explainable models
        - RAG vs. LLM models are distinguished
        - model cfg is NOT considered
        - explainable RAG model is compared based on:
          LLM model name and collection name

        Parameters
        ----------
        m1 : models.ExplainableRagModel | models.ExplainableLlmModel,
            The referential model.
        m2 : models.ExplainableRagModel | models.ExplainableLlmModel,
            The model to compare against.
        m1_llm_model: str | None
            The LLM model name used for filtering and equals of explainable models.
        m2_llm_model: str | None
            The LLM model name used for filtering and equals of explainable models.

        Returns
        -------
        bool :
           Whether the models are equal or not.

        """
        if not m1_llm_model or not m2_llm_model:
            raise ValueError(
                f"LLM model names for both baseline and current models are not "
                f"available for {m1_llm_model=} and {m2_llm_model=}"
            )

        if isinstance(m1, models.ExplainableRagModel):
            if isinstance(m2, models.ExplainableRagModel):
                if m1.collection_name and m2.collection_name:
                    # smarter collection names comparison
                    m1_col_name = m1.collection_name.replace(m1_llm_model, "MODEL")
                    m2_col_name = m2.collection_name.replace(m2_llm_model, "MODEL")
                    if (
                        m1.llm_model_name == m1_llm_model
                        and m2.llm_model_name == m2_llm_model
                        and m1_col_name == m2_col_name
                    ):
                        return True
        elif isinstance(m2, models.ExplainableLlmModel):
            if m1.llm_model_name == m1_llm_model and m2.llm_model_name == m2_llm_model:
                return True

        return False

    def _find_comparable_models(
        self,
        baseline_llm_model: str = "",
        current_llm_model: str = "",
    ) -> list[tuple[str, str]]:
        """Extract intersection of models which can be compared:

         - if BASELINE has 1 model and CURRENT has 1 model,
           then compare those 2 - it means user compares a system BEFORE and AFTER
           a change in RAG/LLM configuration/LLM/prompt change/...
         - if BASELINE has N models and CURRENT has M models,
           then compare ONLY those models with the identical configuration, host type
           and LLM model.

        Returns
        -------
        list[tuple[str, str]]
            The list of comparable models from baseline and current results as tuples.

        """
        this = EvalResultsExplanationsComparator

        # extract unique model keys from both baseline and current results
        baseline_model_keys = set()
        for result in self.baseline.eval_results.results:
            if result.dataset_row.model_key:
                baseline_model_keys.add(result.dataset_row.model_key)

        current_model_keys = set()
        for result in self.current.eval_results.results:
            if result.dataset_row.model_key:
                current_model_keys.add(result.dataset_row.model_key)

        comparable_models = []

        # handle SPECIAL case: user provided LLM filtering hint (CHECK THIS FIRST)
        if baseline_llm_model and current_llm_model:
            for baseline_e_model in self.baseline.explainable_models:
                for current_e_model in self.current.explainable_models:
                    if this._are_explainable_models_llm_equal(
                        m1=baseline_e_model,
                        m2=current_e_model,
                        m1_llm_model=baseline_llm_model,
                        m2_llm_model=current_llm_model,
                    ):
                        comparable_models.append(
                            (baseline_e_model.key, current_e_model.key)
                        )
            # return early when LLM filtering is specified
            return comparable_models
        elif not baseline_llm_model and not current_llm_model:
            pass
        else:
            raise ValueError(
                f"Evaluations comparator must get either both baseline LLM model and "
                f"current LLM model or none: {baseline_llm_model=} "
                f" {current_llm_model=}"
            )

        # handle SPECIAL case: single model comparison: BEFORE vs. AFTER
        if len(baseline_model_keys) == 1 and len(current_model_keys) == 1:
            return [(list(baseline_model_keys)[0], list(current_model_keys)[0])]

        # find intersection of model keys for multimodel comparison
        # NEW APPROACH:
        # - models are comparable <=> intersection of their TCs is non-empty.
        #   TCs are equal if they have the same input/question and non-empty
        #   intersection of metrics.
        # OLD APPROACH (below):
        # - only compatible ~ same host type / LLM / corpus ~ are comparable,
        #   BUT it is NOT true, anything can be compared if the TCs intersection
        #   is NON-empty:
        # for baseline_e_model in self.baseline.explainable_models:
        #     for current_e_model in self.current.explainable_models:
        #         if this._are_explainable_models_equal(
        #             m1=baseline_e_model,
        #             m2=current_e_model,
        #         ):
        #             comparable_models.append(
        #                 (baseline_e_model.key, current_e_model.key)
        #             )

        # initialize comparable models: comparable <=> intersection of test cases
        # with the same input/question is non-empty
        for baseline_model_key in baseline_model_keys:
            for current_model_key in current_model_keys:
                # build maps of prompt -> test case for each model
                baseline_prompts = set()
                for result in self.baseline.eval_results.results:
                    if result.dataset_row.model_key == baseline_model_key:
                        baseline_prompts.add(result.dataset_row.i)

                current_prompts = set()
                for result in self.current.eval_results.results:
                    if result.dataset_row.model_key == current_model_key:
                        current_prompts.add(result.dataset_row.i)

                # check if models have non-empty intersection of test cases
                # (test cases with the same input/question)
                common_prompts = baseline_prompts & current_prompts
                if common_prompts:
                    # models are comparable if they have at least one common test case
                    comparable_models.append((baseline_model_key, current_model_key))

        return comparable_models

    def _compare_models(
        self,
        baseline_model_key: str,
        current_model_key: str,
        sentence_similarity_threshold: float = 0.9,
    ) -> tuple[list[EvalResultDiff], tuple]:
        """Compare two explainable models.

        Parameters
        ----------
        baseline_model_key : str
            The key of the model in the baseline results.
        current_model_key : str
            The key of the model in the current results.
        sentence_similarity_threshold : float
            Threshold for determining if sentences are "common" (high similarity).
            Default is 0.9.

        Returns
        -------
        tuple[list[EvalResultDiff], tuple] :
            A tuple containing:
            - list of EvalResultDiff objects for this model pair
            - tuple of (baseline_explainable_model, current_explainable_model)

        """
        # find comparable test cases between the two models
        comparable_test_cases = self._find_comparable_test_cases(
            baseline_model_key=baseline_model_key,
            current_model_key=current_model_key,
        )
        self.logger.info(
            f"Found {len(comparable_test_cases)} comparable results for "
            f"{baseline_model_key=} and {current_model_key=}"
        )
        if len(comparable_test_cases) == 0:
            # nothing to compare
            return [], (None, None)

        # create a list to hold all EvalResultDiff objects
        diffs = []

        # total test cases for progress reporting
        total_test_cases = len(comparable_test_cases)

        # compare each pair of test cases and create an EvalResultDiff for each
        for idx, (baseline_result, current_result) in enumerate(
            comparable_test_cases, start=1
        ):
            # get the test case keys
            baseline_tc_key = baseline_result.dataset_row.key
            current_tc_key = current_result.dataset_row.key

            # compare the test cases
            (
                baseline_diff_actual_output_meta,
                current_diff_actual_output_meta,
                baseline_diff_retrieved_context,
                current_diff_retrieved_context,
                tc_flipped,
                tc_changed,
                _,  # model-level flipped metrics handled separately
            ) = self._compare_test_case(
                baseline_model_key=baseline_model_key,
                current_model_key=current_model_key,
                baseline_test_case_key=baseline_tc_key,
                current_test_case_key=current_tc_key,
                sentence_similarity_threshold=sentence_similarity_threshold,
            )

            # get the question/prompt/input (identical in both)
            question = baseline_result.dataset_row.i

            # create EvalResultDiff for this test case pair
            diff = EvalResultDiff(
                question=question,
                expected_answer=baseline_result.dataset_row.expected_output,
                diff_flipped_metrics=tc_flipped or {},
                diff_changed_metrics=tc_changed or {},
                baseline_test_case_result=baseline_result.to_dict(),
                baseline_diff_actual_output_meta=baseline_diff_actual_output_meta or {},
                baseline_diff_retrieved_context=baseline_diff_retrieved_context or {},
                current_test_case_result=current_result.to_dict(),
                current_diff_actual_output_meta=current_diff_actual_output_meta or {},
                current_diff_retrieved_context=current_diff_retrieved_context or {},
            )
            diffs.append(diff)

            # log progress for BERTScore (slow method)
            if (
                self.comparison_method
                == _explanations_base.SentenceComparisonMethod.BERT_SCORE
            ):
                completed_pct = (idx / total_test_cases) * 100
                remaining_count = total_test_cases - idx
                self.logger.info(
                    f"BERTScore comparison progress: {idx}/{total_test_cases} "
                    f"({completed_pct:.1f}%) test cases completed, "
                    f"{remaining_count} remaining"
                )

        # find the corresponding explainable models
        baseline_explainable_model = None
        current_explainable_model = None
        for baseline_e_model in self.baseline.explainable_models:
            if baseline_e_model.key == baseline_model_key:
                baseline_explainable_model = baseline_e_model
                break
        for current_e_model in self.current.explainable_models:
            if current_e_model.key == current_model_key:
                current_explainable_model = current_e_model
                break

        return diffs, (baseline_explainable_model, current_explainable_model)

    def _find_comparable_test_cases(
        self, baseline_model_key: str, current_model_key: str
    ) -> list[tuple]:
        """Find comparable test cases between baseline and current models.

        Comparable test cases are those with:

        - correct model key
        - identical prompts

        Parameters
        ----------
        baseline_model_key : str
            The key of the model in the baseline results.
        current_model_key : str
            The key of the model in the current results.

        Returns
        -------
        list[tuple[LlmEvalResultsRow, LlmEvalResultsRow]]
            The list of comparable test cases from baseline and current models
            as tuples.

        """
        # build maps of prompt -> test case key for each model
        # m: prompt -> LLM evaluation result row
        baseline_prompt_to_tc = {}
        for result in self.baseline.eval_results.results:
            if result.dataset_row.model_key == baseline_model_key:
                baseline_prompt_to_tc[result.dataset_row.i] = result

        current_prompt_to_tc = {}
        for result in self.current.eval_results.results:
            if result.dataset_row.model_key == current_model_key:
                current_prompt_to_tc[result.dataset_row.i] = result

        # find results with identical prompts
        comparable_test_cases = []
        for prompt in baseline_prompt_to_tc:
            if prompt in current_prompt_to_tc:
                baseline_result = baseline_prompt_to_tc[prompt]
                current_result = current_prompt_to_tc[prompt]
                comparable_test_cases.append((baseline_result, current_result))

        return comparable_test_cases

    def _compare_test_case(
        self,
        baseline_model_key: str,
        current_model_key: str,
        baseline_test_case_key: str,
        current_test_case_key: str,
        sentence_similarity_threshold: float = 0.9,
    ) -> tuple:
        """Compare test cases in the baseline and current models.

        Parameters
        ----------
        baseline_model_key : str
            The key of the model in the baseline results.
        current_model_key : str
            The key of the model in the current results.
        baseline_test_case_key : str
            The key of the test case in the baseline model.
        current_test_case_key : str
            The key of the test case in the current model.
        sentence_similarity_threshold : float
            Threshold for determining if sentences are "common" (high similarity).
            Default is 0.9.

        Returns
        -------
        tuple :
            A tuple containing:
            - baseline_diff_actual_output_meta: sentence-level diff (old vs new)
            - current_diff_actual_output_meta: sentence-level diff (new vs old)
            - baseline_diff_retrieved_context: context chunks diff (old vs new)
            - current_diff_retrieved_context: context chunks diff (new vs old)
            - tc_flipped_metrics: metrics that flipped between old and new
            - flipped_metrics: (None, calculated at model level elsewhere)
            - changed_metrics: (None, calculated at model level elsewhere)

        """
        # find the actual result rows for comparison
        baseline_result = None
        current_result = None

        for result in self.baseline.eval_results.results:
            if (
                result.dataset_row.model_key == baseline_model_key
                and result.dataset_row.key == baseline_test_case_key
            ):
                baseline_result = result
                break

        for result in self.current.eval_results.results:
            if (
                result.dataset_row.model_key == current_model_key
                and result.dataset_row.key == current_test_case_key
            ):
                current_result = result
                break

        if not baseline_result or not current_result:
            return None, None, None, None, None, None

        # compare actual outputs at sentence level - bidirectional
        baseline_diff_actual_output_meta, current_diff_actual_output_meta = (
            EvalResultsExplanationsComparator._compare_sentences(
                baseline_output=baseline_result.dataset_row.actual_output,
                current_output=current_result.dataset_row.actual_output,
                comparison_method=self.comparison_method,
                embedding_model=self._embedding_model,
                sentence_similarity_threshold=sentence_similarity_threshold,
            )
        )

        # compare retrieved context chunks - bidirectional
        baseline_diff_retrieved_context, current_diff_retrieved_context = (
            EvalResultsExplanationsComparator._compare_context_chunks(
                baseline_context=baseline_result.dataset_row.context,
                current_context=current_result.dataset_row.context,
                comparison_method=self.comparison_method,
                embedding_model=self._embedding_model,
                sentence_similarity_threshold=sentence_similarity_threshold,
            )
        )

        # check for flipped metrics at test case level
        if not baseline_result.metrics.get(KEY_METRICS_META, None):
            baseline_result.metrics[KEY_METRICS_META] = (
                self.baseline.explainer._metrics_meta.key_to_metric
            )
        if not current_result.metrics.get(KEY_METRICS_META, None):
            current_result.metrics[KEY_METRICS_META] = (
                self.current.explainer._metrics_meta.key_to_metric
            )
        (tc_flipped_metrics, tc_changed_metrics) = (
            EvalResultsExplanationsComparator._compare_metrics(
                baseline_metrics=baseline_result.metrics,
                current_metrics=current_result.metrics,
            )
        )

        # model-level flipped metrics are calculated elsewhere
        flipped_metrics = None

        return (
            baseline_diff_actual_output_meta,
            current_diff_actual_output_meta,
            baseline_diff_retrieved_context,
            current_diff_retrieved_context,
            tc_flipped_metrics,
            tc_changed_metrics,
            flipped_metrics,
        )

    @staticmethod
    def _compare_sentences(
        baseline_output: str,
        current_output: str,
        comparison_method: _explanations_base.SentenceComparisonMethod = (
            _explanations_base.SentenceComparisonMethod.COSINE_DISTANCE
        ),
        embedding_model=None,
        sentence_similarity_threshold: float = 0.9,
    ) -> tuple[dict, dict] | tuple[None, None]:
        """Compare two outputs at sentence level - bidirectional.

        Uses NLTK's punkt tokenizer for sentence segmentation when available,
        falls back to simple period-based splitting otherwise.

        Parameters
        ----------
        baseline_output : str
            The baseline actual output.
        current_output : str
            The current actual output.
        comparison_method : SentenceComparisonMethod
            The method to use for comparing sentences:
            - EXACT_MATCH: exact string matching
            - COSINE_DISTANCE: cosine distance of sentence embeddings (default)
            - BERT_SCORE: BERTScore contextual embeddings similarity
        embedding_model : sentence_transformers.SentenceTransformer | None
            The embedding model for cosine distance comparison.
            Required when comparison_method is COSINE_DISTANCE.
        sentence_similarity_threshold : float
            Threshold for determining if sentences are "common" (high similarity).
            Sentences with similarity >= threshold are considered common.
            Default is 0.9.

        Returns
        -------
        tuple[dict, dict] | tuple[None, None]
            A tuple of (baseline_diff, current_diff) where:
            - baseline_diff: sentence-level diff from baseline perspective
              (baseline vs current)
            - current_diff: sentence-level diff from current perspective
              (current vs baseline)
            Returns (None, None) if outputs are identical.

        """
        if baseline_output == current_output:
            return None, None

        # sentence-level comparison using punkt tokenization
        if HAS_NLTK:
            # use NLTK punkt tokenizer for accurate sentence segmentation
            baseline_sentences = nltk.tokenize.sent_tokenize(baseline_output)
            current_sentences = nltk.tokenize.sent_tokenize(current_output)
        else:
            # FALLBACK: simple period-based splitting
            baseline_sentences = [
                s.strip() for s in baseline_output.split(".") if s.strip()
            ]
            current_sentences = [
                s.strip() for s in current_output.split(".") if s.strip()
            ]

        if comparison_method == _explanations_base.SentenceComparisonMethod.EXACT_MATCH:
            # EXACT MATCH: find common sentences using set intersection
            baseline_set = set(baseline_sentences)
            current_set = set(current_sentences)
            common_sentences = baseline_set.intersection(current_set)

            # build similarity dict: {sentence: 1.0 for exact match, 0.0 otherwise}
            baseline_similarity = {
                s: 1.0 if s in current_set else 0.0 for s in baseline_sentences
            }
            current_similarity = {
                s: 1.0 if s in baseline_set else 0.0 for s in current_sentences
            }

            # OLD perspective
            baseline_diff = {
                KEY_SENTENCES: baseline_sentences,
                KEY_SENTENCES_COUNT: len(baseline_sentences),
                KEY_COMMON_SENTENCES: list(common_sentences),
                KEY_COMMON_COUNT: len(common_sentences),
                KEY_UNIQUE_SENTENCES: [
                    s for s in baseline_sentences if s not in current_set
                ],
                KEY_UNIQUE_COUNT: len(
                    [s for s in baseline_sentences if s not in current_set]
                ),
                KEY_IDENTICAL: False,
                KEY_SENTENCE_SIMILARITY: baseline_similarity,
            }

            # NEW perspective
            current_diff = {
                KEY_SENTENCES: current_sentences,
                KEY_SENTENCES_COUNT: len(current_sentences),
                KEY_COMMON_SENTENCES: list(common_sentences),
                KEY_COMMON_COUNT: len(common_sentences),
                KEY_UNIQUE_SENTENCES: [
                    s for s in current_sentences if s not in baseline_set
                ],
                KEY_UNIQUE_COUNT: len(
                    [s for s in current_sentences if s not in baseline_set]
                ),
                KEY_IDENTICAL: False,
                KEY_SENTENCE_SIMILARITY: current_similarity,
            }

        elif (
            comparison_method
            == _explanations_base.SentenceComparisonMethod.COSINE_DISTANCE
        ):
            # COSINE DISTANCE: calculate semantic similarity using embeddings
            if embedding_model is None:
                raise ValueError(
                    "Embedding model required for COSINE_DISTANCE comparison method"
                )

            # embed all sentences
            baseline_embeddings = embedding_model.encode(baseline_sentences)
            current_embeddings = embedding_model.encode(current_sentences)

            # calculate similarity for each baseline sentence to all current sentences
            baseline_similarity = {}
            for i, baseline_sent in enumerate(baseline_sentences):
                # find max similarity to any current sentence
                max_sim = 0.0
                for j, current_sent in enumerate(current_sentences):
                    # cosine similarity = 1 - cosine distance
                    sim = 1.0 - nltk.cluster.cosine_distance(
                        baseline_embeddings[i], current_embeddings[j]
                    )
                    max_sim = max(max_sim, sim)
                baseline_similarity[baseline_sent] = max_sim

            # calculate similarity for each current sentence to all baseline sentences
            current_similarity = {}
            for j, current_sent in enumerate(current_sentences):
                # find max similarity to any baseline sentence
                max_sim = 0.0
                for i, baseline_sent in enumerate(baseline_sentences):
                    # cosine similarity = 1 - cosine distance
                    sim = 1.0 - nltk.cluster.cosine_distance(
                        baseline_embeddings[i], current_embeddings[j]
                    )
                    max_sim = max(max_sim, sim)
                current_similarity[current_sent] = max_sim

            # identify common and unique sentences based on similarity
            common_sentences = set()
            baseline_unique_sentences = []
            for s in baseline_sentences:
                if baseline_similarity[s] >= sentence_similarity_threshold:
                    common_sentences.add(s)
                else:
                    baseline_unique_sentences.append(s)

            current_unique_sentences = []
            for s in current_sentences:
                if current_similarity[s] >= sentence_similarity_threshold:
                    common_sentences.add(s)
                else:
                    current_unique_sentences.append(s)

            # OLD perspective
            baseline_diff = {
                KEY_SENTENCES: baseline_sentences,
                KEY_SENTENCES_COUNT: len(baseline_sentences),
                KEY_COMMON_SENTENCES: list(common_sentences),
                KEY_COMMON_COUNT: len(common_sentences),
                KEY_UNIQUE_SENTENCES: baseline_unique_sentences,
                KEY_UNIQUE_COUNT: len(baseline_unique_sentences),
                KEY_IDENTICAL: False,
                KEY_SENTENCE_SIMILARITY: baseline_similarity,
            }

            # NEW perspective
            current_diff = {
                KEY_SENTENCES: current_sentences,
                KEY_SENTENCES_COUNT: len(current_sentences),
                KEY_COMMON_SENTENCES: list(common_sentences),
                KEY_COMMON_COUNT: len(common_sentences),
                KEY_UNIQUE_SENTENCES: current_unique_sentences,
                KEY_UNIQUE_COUNT: len(current_unique_sentences),
                KEY_IDENTICAL: False,
                KEY_SENTENCE_SIMILARITY: current_similarity,
            }

        elif (
            comparison_method == _explanations_base.SentenceComparisonMethod.BERT_SCORE
        ):
            # BERT_SCORE: calculate semantic similarity using BERTScore
            if not HAS_BERT_SCORE:
                raise ImportError(
                    "bert_score is required for BERT_SCORE comparison method. "
                    "Install it with: pip install bert-score"
                )

            # batch comparisons for efficiency @ pairs
            baseline_cands = []
            baseline_refs = []
            for baseline_sent in baseline_sentences:
                for current_sent in current_sentences:
                    baseline_cands.append(baseline_sent)
                    baseline_refs.append(current_sent)

            if baseline_cands:
                p_old, r_old, f1_old = bert_score.score(
                    cands=baseline_cands,
                    refs=baseline_refs,
                    lang="en",
                    verbose=False,
                    batch_size=64,
                )

                # reshape results: for each baseline sentence, find max similarity
                baseline_similarity = {}
                idx = 0
                for baseline_sent in baseline_sentences:
                    max_sim = 0.0
                    for _ in current_sentences:
                        sim = f1_old[idx].item()
                        max_sim = max(max_sim, sim)
                        idx += 1
                    baseline_similarity[baseline_sent] = max_sim
            else:
                baseline_similarity = {s: 0.0 for s in baseline_sentences}

            # create all pairs: current sentences vs all baseline sentences
            current_cands = []
            current_refs = []
            for current_sent in current_sentences:
                for baseline_sent in baseline_sentences:
                    current_cands.append(current_sent)
                    current_refs.append(baseline_sent)

            # calculate all BERTScores in a single batch call
            if current_cands:
                p_new, r_new, f1_new = bert_score.score(
                    current_cands, current_refs, lang="en", verbose=False, batch_size=64
                )

                # reshape results: for each current sentence, find max similarity
                current_similarity = {}
                idx = 0
                for current_sent in current_sentences:
                    max_sim = 0.0
                    for _ in baseline_sentences:
                        sim = f1_new[idx].item()
                        max_sim = max(max_sim, sim)
                        idx += 1
                    current_similarity[current_sent] = max_sim
            else:
                current_similarity = {s: 0.0 for s in current_sentences}

            # identify common and unique sentences based on similarity
            common_sentences = set()
            baseline_unique_sentences = []
            for s in baseline_sentences:
                if baseline_similarity[s] >= sentence_similarity_threshold:
                    common_sentences.add(s)
                else:
                    baseline_unique_sentences.append(s)

            current_unique_sentences = []
            for s in current_sentences:
                if current_similarity[s] >= sentence_similarity_threshold:
                    common_sentences.add(s)
                else:
                    current_unique_sentences.append(s)

            # OLD perspective
            baseline_diff = {
                KEY_SENTENCES: baseline_sentences,
                KEY_SENTENCES_COUNT: len(baseline_sentences),
                KEY_COMMON_SENTENCES: list(common_sentences),
                KEY_COMMON_COUNT: len(common_sentences),
                KEY_UNIQUE_SENTENCES: baseline_unique_sentences,
                KEY_UNIQUE_COUNT: len(baseline_unique_sentences),
                KEY_IDENTICAL: False,
                KEY_SENTENCE_SIMILARITY: baseline_similarity,
            }

            # NEW perspective
            current_diff = {
                KEY_SENTENCES: current_sentences,
                KEY_SENTENCES_COUNT: len(current_sentences),
                KEY_COMMON_SENTENCES: list(common_sentences),
                KEY_COMMON_COUNT: len(common_sentences),
                KEY_UNIQUE_SENTENCES: current_unique_sentences,
                KEY_UNIQUE_COUNT: len(current_unique_sentences),
                KEY_IDENTICAL: False,
                KEY_SENTENCE_SIMILARITY: current_similarity,
            }

        else:
            raise ValueError(f"Unknown comparison method: {comparison_method}")

        return baseline_diff, current_diff

    @staticmethod
    def _compare_context_chunks(
        baseline_context: list[str],
        current_context: list[str],
        comparison_method: _explanations_base.SentenceComparisonMethod = (
            _explanations_base.SentenceComparisonMethod.COSINE_DISTANCE
        ),
        embedding_model=None,
        sentence_similarity_threshold: float = 0.9,
    ) -> tuple[dict, dict] | tuple[None, None]:
        """Compare retrieved context chunks between baseline and current results.

        Bidirectional comparison.

        NOTE: Chunks are compared as whole units, NOT tokenized into sentences.

        Parameters
        ----------
        baseline_context : list[str]
            The baseline context chunks.
        current_context : list[str]
            The current context chunks.
        comparison_method : SentenceComparisonMethod
            The method to use for comparing chunks:
            - EXACT_MATCH: exact string matching
            - COSINE_DISTANCE: cosine distance of chunk embeddings (default)
            - BERT_SCORE: BERTScore contextual embeddings similarity
        embedding_model : sentence_transformers.SentenceTransformer | None
            The embedding model for cosine distance comparison.
            Required when comparison_method is COSINE_DISTANCE.
        sentence_similarity_threshold : float
            Threshold for determining if chunks are "common" (high similarity).
            Chunks with similarity >= threshold are considered common.
            Default is 0.9.

        Returns
        -------
        tuple[dict, dict] | tuple[None, None] :
            A tuple of (baseline_diff, current_diff) where:
            - baseline_diff: context chunks diff from baseline perspective
              (baseline vs current)
            - current_diff: context chunks diff from current perspective
              (current vs baseline)
            Returns (None, None) if contexts are identical.

        """
        if baseline_context == current_context:
            return None, None

        # handle empty contexts: if one context is empty, all chunks in the other
        # context have 0.0 similarity (will be displayed as red)
        if not baseline_context or not current_context:
            if baseline_context and not current_context:
                # baseline has chunks, current is empty
                # -> all baseline chunks have 0.0 similarity
                baseline_similarity = {c: 0.0 for c in baseline_context}
                baseline_diff = {
                    KEY_CHUNKS: baseline_context,
                    KEY_CHUNKS_COUNT: len(baseline_context),
                    KEY_COMMON_CHUNKS: [],
                    KEY_COMMON_COUNT: 0,
                    KEY_UNIQUE_CHUNKS: baseline_context,
                    KEY_UNIQUE_COUNT: len(baseline_context),
                    KEY_IDENTICAL: False,
                    KEY_CHUNK_SIMILARITY: baseline_similarity,
                }
                current_diff = {
                    KEY_CHUNKS: [],
                    KEY_CHUNKS_COUNT: 0,
                    KEY_COMMON_CHUNKS: [],
                    KEY_COMMON_COUNT: 0,
                    KEY_UNIQUE_CHUNKS: [],
                    KEY_UNIQUE_COUNT: 0,
                    KEY_IDENTICAL: False,
                    KEY_CHUNK_SIMILARITY: {},
                }
                return baseline_diff, current_diff
            elif current_context and not baseline_context:
                # current has chunks, baseline is empty
                # -> all current chunks have 0.0 similarity
                current_similarity = {c: 0.0 for c in current_context}
                baseline_diff = {
                    KEY_CHUNKS: [],
                    KEY_CHUNKS_COUNT: 0,
                    KEY_COMMON_CHUNKS: [],
                    KEY_COMMON_COUNT: 0,
                    KEY_UNIQUE_CHUNKS: [],
                    KEY_UNIQUE_COUNT: 0,
                    KEY_IDENTICAL: False,
                    KEY_CHUNK_SIMILARITY: {},
                }
                current_diff = {
                    KEY_CHUNKS: current_context,
                    KEY_CHUNKS_COUNT: len(current_context),
                    KEY_COMMON_CHUNKS: [],
                    KEY_COMMON_COUNT: 0,
                    KEY_UNIQUE_CHUNKS: current_context,
                    KEY_UNIQUE_COUNT: len(current_context),
                    KEY_IDENTICAL: False,
                    KEY_CHUNK_SIMILARITY: current_similarity,
                }
                return baseline_diff, current_diff
            else:
                # both empty (should be caught by the equality check above)
                return None, None

        if comparison_method == _explanations_base.SentenceComparisonMethod.EXACT_MATCH:
            # EXACT MATCH: find common chunks using set intersection
            baseline_set = set(baseline_context)
            current_set = set(current_context)
            common_chunks = baseline_set.intersection(current_set)

            # build similarity dict: {chunk: 1.0 for exact match, 0.0 otherwise}
            baseline_similarity = {
                c: 1.0 if c in current_set else 0.0 for c in baseline_context
            }
            current_similarity = {
                c: 1.0 if c in baseline_set else 0.0 for c in current_context
            }

            # OLD perspective
            baseline_diff = {
                KEY_CHUNKS: baseline_context,
                KEY_CHUNKS_COUNT: len(baseline_context),
                KEY_COMMON_CHUNKS: list(common_chunks),
                KEY_COMMON_COUNT: len(common_chunks),
                KEY_UNIQUE_CHUNKS: list(baseline_set - current_set),
                KEY_UNIQUE_COUNT: len(baseline_set - current_set),
                KEY_IDENTICAL: False,
                KEY_CHUNK_SIMILARITY: baseline_similarity,
            }

            # NEW perspective
            current_diff = {
                KEY_CHUNKS: current_context,
                KEY_CHUNKS_COUNT: len(current_context),
                KEY_COMMON_CHUNKS: list(common_chunks),
                KEY_COMMON_COUNT: len(common_chunks),
                KEY_UNIQUE_CHUNKS: list(current_set - baseline_set),
                KEY_UNIQUE_COUNT: len(current_set - baseline_set),
                KEY_IDENTICAL: False,
                KEY_CHUNK_SIMILARITY: current_similarity,
            }

        elif (
            comparison_method
            == _explanations_base.SentenceComparisonMethod.COSINE_DISTANCE
        ):
            # COSINE DISTANCE: calculate semantic similarity using embeddings
            if embedding_model is None:
                raise ValueError(
                    "Embedding model required for COSINE_DISTANCE comparison method"
                )

            # embed all chunks (as whole units, NOT tokenized)
            baseline_embeddings = embedding_model.encode(baseline_context)
            current_embeddings = embedding_model.encode(current_context)

            # calculate similarity for each baseline chunk to all current chunks
            baseline_similarity = {}
            for i, baseline_chunk in enumerate(baseline_context):
                # find max similarity to any current chunk
                max_sim = 0.0
                for j, current_chunk in enumerate(current_context):
                    # cosine similarity = 1 - cosine distance
                    sim = 1.0 - nltk.cluster.cosine_distance(
                        baseline_embeddings[i], current_embeddings[j]
                    )
                    max_sim = max(max_sim, sim)
                baseline_similarity[baseline_chunk] = max_sim

            # calculate similarity for each current chunk to all baseline chunks
            current_similarity = {}
            for j, current_chunk in enumerate(current_context):
                # find max similarity to any baseline chunk
                max_sim = 0.0
                for i, baseline_chunk in enumerate(baseline_context):
                    # cosine similarity = 1 - cosine distance
                    sim = 1.0 - nltk.cluster.cosine_distance(
                        baseline_embeddings[i], current_embeddings[j]
                    )
                    max_sim = max(max_sim, sim)
                current_similarity[current_chunk] = max_sim

            # identify common and unique chunks based on similarity
            common_chunks = set()
            baseline_unique_chunks = []
            for c in baseline_context:
                if baseline_similarity[c] >= sentence_similarity_threshold:
                    common_chunks.add(c)
                else:
                    baseline_unique_chunks.append(c)

            current_unique_chunks = []
            for c in current_context:
                if current_similarity[c] >= sentence_similarity_threshold:
                    common_chunks.add(c)
                else:
                    current_unique_chunks.append(c)

            # OLD perspective
            baseline_diff = {
                KEY_CHUNKS: baseline_context,
                KEY_CHUNKS_COUNT: len(baseline_context),
                KEY_COMMON_CHUNKS: list(common_chunks),
                KEY_COMMON_COUNT: len(common_chunks),
                KEY_UNIQUE_CHUNKS: baseline_unique_chunks,
                KEY_UNIQUE_COUNT: len(baseline_unique_chunks),
                KEY_IDENTICAL: False,
                KEY_CHUNK_SIMILARITY: baseline_similarity,
            }

            # NEW perspective
            current_diff = {
                KEY_CHUNKS: current_context,
                KEY_CHUNKS_COUNT: len(current_context),
                KEY_COMMON_CHUNKS: list(common_chunks),
                KEY_COMMON_COUNT: len(common_chunks),
                KEY_UNIQUE_CHUNKS: current_unique_chunks,
                KEY_UNIQUE_COUNT: len(current_unique_chunks),
                KEY_IDENTICAL: False,
                KEY_CHUNK_SIMILARITY: current_similarity,
            }

        elif (
            comparison_method == _explanations_base.SentenceComparisonMethod.BERT_SCORE
        ):
            # BERT_SCORE: calculate semantic similarity using BERTScore
            if not HAS_BERT_SCORE:
                raise ImportError(
                    "bert_score is required for BERT_SCORE comparison method. "
                    "Install it with: pip install bert-score"
                )

            # batch comparisons for efficiency @ pairs
            baseline_cands = []
            baseline_refs = []
            for baseline_chunk in baseline_context:
                for current_chunk in current_context:
                    baseline_cands.append(baseline_chunk)
                    baseline_refs.append(current_chunk)

            if baseline_cands:
                p_old, r_old, f1_old = bert_score.score(
                    baseline_cands,
                    baseline_refs,
                    lang="en",
                    verbose=False,
                    batch_size=64,
                )

                # reshape results: for each baseline chunk, find max similarity
                baseline_similarity = {}
                idx = 0
                for baseline_chunk in baseline_context:
                    max_sim = 0.0
                    for _ in current_context:
                        sim = f1_old[idx].item()
                        max_sim = max(max_sim, sim)
                        idx += 1
                    baseline_similarity[baseline_chunk] = max_sim
            else:
                baseline_similarity = {c: 0.0 for c in baseline_context}

            # create all pairs: current chunks vs all baseline chunks
            current_cands = []
            current_refs = []
            for current_chunk in current_context:
                for baseline_chunk in baseline_context:
                    current_cands.append(current_chunk)
                    current_refs.append(baseline_chunk)

            # calculate all BERTScores in a single batch call
            if current_cands:
                p_new, r_new, f1_new = bert_score.score(
                    current_cands, current_refs, lang="en", verbose=False, batch_size=64
                )

                # reshape results: for each current chunk, find max similarity
                current_similarity = {}
                idx = 0
                for current_chunk in current_context:
                    max_sim = 0.0
                    for _ in baseline_context:
                        sim = f1_new[idx].item()
                        max_sim = max(max_sim, sim)
                        idx += 1
                    current_similarity[current_chunk] = max_sim
            else:
                current_similarity = {c: 0.0 for c in current_context}

            # identify common and unique chunks based on similarity
            common_chunks = set()
            baseline_unique_chunks = []
            for c in baseline_context:
                if baseline_similarity[c] >= sentence_similarity_threshold:
                    common_chunks.add(c)
                else:
                    baseline_unique_chunks.append(c)

            current_unique_chunks = []
            for c in current_context:
                if current_similarity[c] >= sentence_similarity_threshold:
                    common_chunks.add(c)
                else:
                    current_unique_chunks.append(c)

            # OLD perspective
            baseline_diff = {
                KEY_CHUNKS: baseline_context,
                KEY_CHUNKS_COUNT: len(baseline_context),
                KEY_COMMON_CHUNKS: list(common_chunks),
                KEY_COMMON_COUNT: len(common_chunks),
                KEY_UNIQUE_CHUNKS: baseline_unique_chunks,
                KEY_UNIQUE_COUNT: len(baseline_unique_chunks),
                KEY_IDENTICAL: False,
                KEY_CHUNK_SIMILARITY: baseline_similarity,
            }

            # NEW perspective
            current_diff = {
                KEY_CHUNKS: current_context,
                KEY_CHUNKS_COUNT: len(current_context),
                KEY_COMMON_CHUNKS: list(common_chunks),
                KEY_COMMON_COUNT: len(common_chunks),
                KEY_UNIQUE_CHUNKS: current_unique_chunks,
                KEY_UNIQUE_COUNT: len(current_unique_chunks),
                KEY_IDENTICAL: False,
                KEY_CHUNK_SIMILARITY: current_similarity,
            }

        else:
            raise ValueError(f"Unknown comparison method: {comparison_method}")

        return baseline_diff, current_diff

    @staticmethod
    def _compare_metrics(
        baseline_metrics: dict, current_metrics: dict
    ) -> tuple[dict, dict] | None:
        """Compare metrics between baseline and current results to find flipped metrics.

        Parameters
        ----------
        baseline_metrics : dict
            The baseline metrics.
        current_metrics : dict
            The current metrics.

        Returns
        -------
        tuple[dict, dict] | None :
            Flipped metrics information or None if no metrics flipped.

        """
        flipped = {}
        changed = {}

        metrics_meta = baseline_metrics.get(KEY_METRICS_META)
        if not metrics_meta:
            raise ValueError(
                "Metrics meta not available when comparing baseline and current "
                f"results metric scores: {baseline_metrics}"
            )

        for metric_key in metrics_meta:
            if metric_key in baseline_metrics and metric_key in current_metrics:
                baseline_value = baseline_metrics[metric_key]
                current_value = current_metrics[metric_key]
                if baseline_value != current_value:
                    changed[metric_key] = {
                        KEY_BASELINE_VALUE: baseline_value,
                        KEY_CURRENT_VALUE: current_value,
                    }
                    if commons.MetricMeta.is_metric_flip(
                        old_value=baseline_value,
                        new_value=current_value,
                        metric_meta=metrics_meta[metric_key],
                    ):
                        flipped[metric_key] = {
                            KEY_BASELINE_VALUE: baseline_value,
                            KEY_CURRENT_VALUE: current_value,
                        }

        return flipped if flipped else None, changed if changed else None

    def _calculate_model_flipped_metrics(
        self,
        baseline_model_key: str,
        current_model_key: str,
        comparable_test_cases: list[tuple[str, str]],
    ) -> dict:
        """Calculate model-level flipped metrics by aggregating test cases.

        Parameters
        ----------
        baseline_model_key : str
            The key of the model in the baseline results.
        current_model_key : str
            The key of the model in the current results.
        comparable_test_cases : list[tuple[str, str]]
            List of comparable test case pairs.

        Returns
        -------
        dict :
            Model-level flipped metrics summary.

        """
        baseline_model_metrics = {}
        current_model_metrics = {}

        for baseline_tc_key, current_tc_key in comparable_test_cases:
            # BASELINE result lookup
            for result in self.baseline.eval_results.results:
                if (
                    result.dataset_row.model_key == baseline_model_key
                    and result.dataset_row.key == baseline_tc_key
                ):
                    for metric_key, metric_value in result.metrics.items():
                        if metric_key not in baseline_model_metrics:
                            baseline_model_metrics[metric_key] = []
                        baseline_model_metrics[metric_key].append(metric_value)
                    break

            # CURRENT result lookup
            for result in self.current.eval_results.results:
                if (
                    result.dataset_row.model_key == current_model_key
                    and result.dataset_row.key == current_tc_key
                ):
                    for metric_key, metric_value in result.metrics.items():
                        if metric_key not in current_model_metrics:
                            current_model_metrics[metric_key] = []
                        current_model_metrics[metric_key].append(metric_value)
                    break

        # MODELS: AVG and compare ~ models get all TCs averaged
        flipped_model_metrics = {}
        for metric_key in baseline_model_metrics:
            if metric_key in current_model_metrics:
                baseline_values = baseline_model_metrics[metric_key]
                current_values = current_model_metrics[metric_key]

                try:
                    baseline_avg = sum(baseline_values) / len(baseline_values)
                    current_avg = sum(current_values) / len(current_values)

                    if baseline_avg != current_avg:
                        flipped_model_metrics[metric_key] = {
                            "baseline_average": baseline_avg,
                            "current_average": current_avg,
                            "difference": current_avg - baseline_avg,
                        }
                except (TypeError, ZeroDivisionError):
                    # non-numeric metrics or empty lists
                    continue

        return flipped_model_metrics

    def compare(
        self,
        baseline_llm_model: str = "",
        current_llm_model: str = "",
    ) -> EvalResultsDiff:
        """Compare two LLM evaluation results and return the diff.

        Returns
        -------
        explanations.EvalResultsDiff
            The differences between the old and new LLM evaluation results explanations.

        """
        comparable_models_keys = self._find_comparable_models(
            baseline_llm_model=baseline_llm_model,
            current_llm_model=current_llm_model,
        )

        if not comparable_models_keys:
            # no comparable models found, return empty diff
            raise ValueError(
                "No comparable models found between baseline and current results - "
                "compared result JSON files must either have exactly 1 model each "
                "(they are "
                "treated as before vs. after a RAG/LLM change) OR non-empty "
                "intersection of multiple models with identical configuration, host "
                "type and LLM model keys must be present in compared files."
            )

        # dic: (baseline_model_key, current_model_key) -> list[EvalResultDiff]
        diffs_dict = {}
        all_baseline_explainable_models = []
        all_current_explainable_models = []

        # cache punkt tokenizer (used by all methods, including EXACT)
        caching.cache_nltk_punkt(self.logger)

        # if using COSINE_DISTANCE, initialize embedding model
        if (
            self.comparison_method
            == _explanations_base.SentenceComparisonMethod.COSINE_DISTANCE
        ):
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError(
                    "sentence_transformers is required for COSINE_DISTANCE "
                    "comparison method. Install it with: "
                    "pip install sentence-transformers"
                )
            if not HAS_NLTK:
                raise ImportError(
                    "nltk is required for COSINE_DISTANCE comparison method. "
                    "Install it with: pip install nltk"
                )

            device = h2o_sonar_config.config.resolve_gpu_cpu_device(result_format="str")
            with resource_mgmt.PytorchModelLifeCycleManager(
                sentence_transformers.SentenceTransformer(
                    self._e_model_baai_bge,
                    device=device,
                    revision=caching.REVISIONS_FOR_MODEL.get(
                        self._e_model_baai_bge, "main"
                    ),
                )
            ) as embedding_model:
                self._embedding_model = embedding_model

                for baseline_model_key, current_model_key in comparable_models_keys:
                    self.logger.info(
                        f"Comparing model '{baseline_model_key}' in BASELINE "
                        f"results with model '{current_model_key}' in CURRENT results."
                    )
                    (
                        model_diffs,
                        (
                            baseline_explainable_model,
                            current_explainable_model,
                        ),
                    ) = self._compare_models(
                        baseline_model_key,
                        current_model_key,
                        sentence_similarity_threshold=(
                            self.sentence_similarity_threshold
                        ),
                    )

                    model_pair = (baseline_model_key, current_model_key)
                    diffs_dict[model_pair] = model_diffs

                    if baseline_explainable_model:
                        all_baseline_explainable_models.append(
                            baseline_explainable_model
                        )
                    if current_explainable_model:
                        all_current_explainable_models.append(current_explainable_model)

            # cleanup embedding model reference after context manager exits
            self._embedding_model = None
        else:
            # EXACT_MATCH mode - no embedding model needed
            for baseline_model_key, current_model_key in comparable_models_keys:
                self.logger.info(
                    f"Comparing model '{baseline_model_key}' in BASELINE results with "
                    f"model '{current_model_key}' in CURRENT results."
                )
                model_diffs, (baseline_explainable_model, current_explainable_model) = (
                    self._compare_models(
                        baseline_model_key,
                        current_model_key,
                        sentence_similarity_threshold=(
                            self.sentence_similarity_threshold
                        ),
                    )
                )

                # add to dictionary with model pair as key
                model_pair = (baseline_model_key, current_model_key)
                diffs_dict[model_pair] = model_diffs

                # collect explainable models
                if baseline_explainable_model:
                    all_baseline_explainable_models.append(baseline_explainable_model)
                if current_explainable_model:
                    all_current_explainable_models.append(current_explainable_model)

        # merge metrics_meta from both baseline and current explanation explainers
        merged_metrics_meta_dict = {}
        if self.baseline and hasattr(self.baseline, "explainer"):
            if hasattr(self.baseline.explainer, "_metrics_meta"):
                merged_metrics_meta_dict.update(
                    self.baseline.explainer._metrics_meta.key_to_metric
                )
        if self.current and hasattr(self.current, "explainer"):
            if hasattr(self.current.explainer, "_metrics_meta"):
                # merge current metrics (won't overwrite existing keys)
                for (
                    metric_key,
                    metric,
                ) in self.current.explainer._metrics_meta.key_to_metric.items():
                    if metric_key not in merged_metrics_meta_dict:
                        merged_metrics_meta_dict[metric_key] = metric

        # use merged metrics_meta
        metrics_meta = merged_metrics_meta_dict if merged_metrics_meta_dict else None

        # validate that every metric in the diffs has its specification in metrics_meta
        if metrics_meta:
            missing_metrics = []
            # check all metrics in all diffs
            for model_pair, diff_list in diffs_dict.items():
                for diff in diff_list:
                    # check metrics in flipped metrics
                    if diff.diff_flipped_metrics:
                        for metric_key in diff.diff_flipped_metrics.keys():
                            if metric_key not in metrics_meta:
                                if metric_key not in missing_metrics:
                                    missing_metrics.append(metric_key)
                    # check metrics in changed metrics
                    if diff.diff_changed_metrics:
                        for metric_key in diff.diff_changed_metrics.keys():
                            if metric_key not in metrics_meta:
                                if metric_key not in missing_metrics:
                                    missing_metrics.append(metric_key)

            # raise exception if any metrics don't have corresponding metadata
            if missing_metrics:
                raise ValueError(
                    f"Metrics found in comparison without corresponding metadata in "
                    f"metrics_meta: {missing_metrics}. This indicates that the "
                    f"comparison diffs contain metrics that are not defined in the "
                    f"merged metrics_meta from baseline and current explanations."
                )

        return EvalResultsDiff(
            diffs=diffs_dict,
            baseline_explainable_models=all_baseline_explainable_models,
            current_explainable_models=all_current_explainable_models,
            comparison_method=self.comparison_method,
            metrics_meta=metrics_meta,
            branding=self.branding,
        )
