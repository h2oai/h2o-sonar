# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import pytest

from h2o_sonar.lib.api.explanations import _explanations_cmp
from h2o_sonar.lib.api.explanations import _explanations_cmp_html


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_metrics_ranks_calculation_basic():
    """Test basic rank-based comparison calculation.

    This test verifies the rank-based comparison logic matches the R example
    from the requirements where scores are merged, ranked, and averaged.
    """
    #
    # GIVEN
    #

    # create mock MetricMeta objects
    class MockMetricMeta:
        def __init__(self, higher_is_better: bool):
            self.higher_is_better = higher_is_better

    # create mock test case diffs with metrics
    # scenario: 2 test cases, 1 metric each (accuracy - higher is better)
    # baseline: [0.8, 0.9]
    # current: [0.85, 0.75]
    # merged & sorted (descending): [0.9, 0.85, 0.8, 0.75]
    # inverted ranks (higher = better): [4, 3, 2, 1]
    # baseline ranks: [2, 4] -> avg = 3.0
    # current ranks: [3, 1] -> avg = 2.0
    # baseline should win (higher avg rank 3.0 > 2.0)

    class MockTestCase(dict):
        def __init__(self, accuracy):
            super().__init__()
            self["accuracy"] = accuracy
            self[_explanations_cmp.KEY_CONTEXT] = ["some context"]

    class MockDiff:
        def __init__(self, baseline_accuracy, current_accuracy):
            self.baseline_test_case = MockTestCase(baseline_accuracy)
            self.current_test_case = MockTestCase(current_accuracy)
            self.diff_flipped_metrics = {}

    model_diffs = [
        (0, MockDiff(0.8, 0.85)),
        (1, MockDiff(0.9, 0.75)),
    ]

    metrics_meta = {
        "accuracy": MockMetricMeta(higher_is_better=True),
    }

    #
    # WHEN
    #
    stats = _explanations_cmp_html.EvalResultsDiffHtml._calculate_model_cmp_stats(
        model_diffs=model_diffs, metrics_meta=metrics_meta
    )

    #
    # THEN
    #
    print(f"\nStats: {stats}")
    baseline_key = _explanations_cmp_html.KEY_METRICS_RANKS_BASELINE
    current_key = _explanations_cmp_html.KEY_METRICS_RANKS_CURRENT
    print(f"Baseline rank: {stats[baseline_key]}")
    print(f"Current rank: {stats[current_key]}")

    # verify ranks are calculated
    assert _explanations_cmp_html.KEY_METRICS_RANKS_BASELINE in stats
    assert _explanations_cmp_html.KEY_METRICS_RANKS_CURRENT in stats

    # expected ranks: baseline = 3.0, current = 2.0
    baseline_rank = stats[_explanations_cmp_html.KEY_METRICS_RANKS_BASELINE]
    current_rank = stats[_explanations_cmp_html.KEY_METRICS_RANKS_CURRENT]

    # verify ranks are floats
    assert isinstance(baseline_rank, float)
    assert isinstance(current_rank, float)

    # verify expected values
    msg = f"Expected baseline rank 3.0, got {baseline_rank}"
    assert baseline_rank == 3.0, msg
    msg = f"Expected current rank 2.0, got {current_rank}"
    assert current_rank == 2.0, msg

    # verify baseline wins (higher rank is better)
    assert baseline_rank > current_rank


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_metrics_ranks_calculation_lower_is_better():
    """Test rank-based comparison with lower_is_better metrics.

    This test verifies ranking works correctly when lower values are better.
    """
    #
    # GIVEN
    #

    class MockMetricMeta:
        def __init__(self, higher_is_better: bool):
            self.higher_is_better = higher_is_better

    class MockTestCase(dict):
        def __init__(self, error_rate):
            super().__init__()
            self["error_rate"] = error_rate
            self[_explanations_cmp.KEY_CONTEXT] = ["some context"]

    class MockDiff:
        def __init__(self, baseline_error, current_error):
            self.baseline_test_case = MockTestCase(baseline_error)
            self.current_test_case = MockTestCase(current_error)
            self.diff_flipped_metrics = {}

    # scenario: error_rate - lower is better
    # baseline: [0.1, 0.2]
    # current: [0.15, 0.05]
    # for lower_is_better, sort ascending: [0.05, 0.1, 0.15, 0.2]
    # inverted ranks (higher = better): [4, 3, 2, 1]
    # baseline ranks: [3, 1] -> avg = 2.0
    # current ranks: [4, 2] -> avg = 3.0
    # current should win (higher avg rank 3.0 > 2.0)

    model_diffs = [
        (0, MockDiff(0.1, 0.15)),
        (1, MockDiff(0.2, 0.05)),
    ]

    metrics_meta = {
        "error_rate": MockMetricMeta(higher_is_better=False),
    }

    #
    # WHEN
    #
    stats = _explanations_cmp_html.EvalResultsDiffHtml._calculate_model_cmp_stats(
        model_diffs=model_diffs, metrics_meta=metrics_meta
    )

    #
    # THEN
    #
    print(f"\nStats: {stats}")
    baseline_key = _explanations_cmp_html.KEY_METRICS_RANKS_BASELINE
    current_key = _explanations_cmp_html.KEY_METRICS_RANKS_CURRENT
    baseline_rank = stats[baseline_key]
    current_rank = stats[current_key]

    print(f"Baseline rank: {baseline_rank}")
    print(f"Current rank: {current_rank}")

    # expected ranks: baseline = 2.0, current = 3.0
    msg = f"Expected baseline rank 2.0, got {baseline_rank}"
    assert baseline_rank == 2.0, msg
    msg = f"Expected current rank 3.0, got {current_rank}"
    assert current_rank == 3.0, msg

    # verify current wins (higher rank is better)
    assert current_rank > baseline_rank


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_metrics_ranks_calculation_multiple_metrics():
    """Test rank-based comparison with multiple metrics.

    This test verifies that ranks are calculated per metric and then averaged.
    """
    #
    # GIVEN
    #

    class MockMetricMeta:
        def __init__(self, higher_is_better: bool):
            self.higher_is_better = higher_is_better

    class MockTestCase(dict):
        def __init__(self, accuracy, error_rate):
            super().__init__()
            self["accuracy"] = accuracy
            self["error_rate"] = error_rate
            self[_explanations_cmp.KEY_CONTEXT] = ["some context"]

    class MockDiff:
        def __init__(
            self,
            baseline_accuracy,
            baseline_error,
            current_accuracy,
            current_error,
        ):
            self.baseline_test_case = MockTestCase(baseline_accuracy, baseline_error)
            self.current_test_case = MockTestCase(current_accuracy, current_error)
            self.diff_flipped_metrics = {}

    # metric 1 (accuracy, higher is better): baseline better
    # baseline: [0.9], current: [0.8]
    # sorted descending: [0.9, 0.8]
    # inverted ranks (n=2, higher=better): [2, 1]
    # baseline=2, current=1

    # metric 2 (error_rate, lower is better): current better
    # baseline: [0.2], current: [0.1]
    # sorted ascending: [0.1, 0.2]
    # inverted ranks (n=2, higher=better): [2, 1]
    # current=2, baseline=1

    # overall average ranks:
    # baseline: (2 + 1) / 2 = 1.5
    # current: (1 + 2) / 2 = 1.5
    # tie

    model_diffs = [
        (0, MockDiff(0.9, 0.2, 0.8, 0.1)),
    ]

    metrics_meta = {
        "accuracy": MockMetricMeta(higher_is_better=True),
        "error_rate": MockMetricMeta(higher_is_better=False),
    }

    #
    # WHEN
    #
    stats = _explanations_cmp_html.EvalResultsDiffHtml._calculate_model_cmp_stats(
        model_diffs=model_diffs, metrics_meta=metrics_meta
    )

    #
    # THEN
    #
    print(f"\nStats: {stats}")
    baseline_key = _explanations_cmp_html.KEY_METRICS_RANKS_BASELINE
    current_key = _explanations_cmp_html.KEY_METRICS_RANKS_CURRENT
    baseline_rank = stats[baseline_key]
    current_rank = stats[current_key]

    print(f"Baseline rank: {baseline_rank}")
    print(f"Current rank: {current_rank}")

    # both should have rank 1.5 (tie)
    msg = f"Expected baseline rank 1.5, got {baseline_rank}"
    assert baseline_rank == 1.5, msg
    msg = f"Expected current rank 1.5, got {current_rank}"
    assert current_rank == 1.5, msg


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_metrics_ranks_calculation_empty_values():
    """Test rank-based comparison handles empty/missing values correctly."""
    #
    # GIVEN
    #

    class MockMetricMeta:
        def __init__(self, higher_is_better: bool):
            self.higher_is_better = higher_is_better

    class MockTestCase(dict):
        def __init__(self):
            super().__init__()
            self[_explanations_cmp.KEY_CONTEXT] = ["some context"]

    class MockDiff:
        def __init__(self):
            self.baseline_test_case = MockTestCase()
            self.current_test_case = MockTestCase()
            self.diff_flipped_metrics = {}

    model_diffs = [
        (0, MockDiff()),
    ]

    metrics_meta = {
        "accuracy": MockMetricMeta(higher_is_better=True),
    }

    #
    # WHEN
    #
    stats = _explanations_cmp_html.EvalResultsDiffHtml._calculate_model_cmp_stats(
        model_diffs=model_diffs, metrics_meta=metrics_meta
    )

    #
    # THEN
    #
    print(f"\nStats: {stats}")
    baseline_key = _explanations_cmp_html.KEY_METRICS_RANKS_BASELINE
    current_key = _explanations_cmp_html.KEY_METRICS_RANKS_CURRENT
    baseline_rank = stats[baseline_key]
    current_rank = stats[current_key]

    print(f"Baseline rank: {baseline_rank}")
    print(f"Current rank: {current_rank}")

    # both should be 0.0 (no metrics to rank)
    assert baseline_rank == 0.0
    assert current_rank == 0.0


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_test_case_ranks_calculation_basic():
    """Test basic test case rank-based comparison calculation.

    This test verifies that test case winners are determined by
    comparing average ranks of metrics within each test case.
    """
    #
    # GIVEN
    #

    class MockMetricMeta:
        def __init__(self, higher_is_better: bool):
            self.higher_is_better = higher_is_better

    class MockTestCase(dict):
        def __init__(self, accuracy):
            super().__init__()
            self["accuracy"] = accuracy
            self[_explanations_cmp.KEY_CONTEXT] = ["some context"]

    class MockDiff:
        def __init__(self, baseline_accuracy, current_accuracy):
            self.baseline_test_case = MockTestCase(baseline_accuracy)
            self.current_test_case = MockTestCase(current_accuracy)
            self.diff_flipped_metrics = {}

    # scenario: 2 test cases, 1 metric each (accuracy - higher is better)
    # test case 0: baseline=0.8, current=0.9 -> current wins (rank 2 > rank 1)
    # test case 1: baseline=0.9, current=0.7 -> baseline wins (rank 2 > rank 1)
    # test case ranks: baseline=1, current=1 (tie)

    model_diffs = [
        (0, MockDiff(0.8, 0.9)),
        (1, MockDiff(0.9, 0.7)),
    ]

    metrics_meta = {
        "accuracy": MockMetricMeta(higher_is_better=True),
    }

    #
    # WHEN
    #
    stats = _explanations_cmp_html.EvalResultsDiffHtml._calculate_model_cmp_stats(
        model_diffs=model_diffs, metrics_meta=metrics_meta
    )

    #
    # THEN
    #
    print(f"\nStats: {stats}")
    baseline_tc_key = _explanations_cmp_html.KEY_TEST_CASE_RANKS_BASELINE
    current_tc_key = _explanations_cmp_html.KEY_TEST_CASE_RANKS_CURRENT
    print(f"Baseline test case ranks: {stats[baseline_tc_key]}")
    print(f"Current test case ranks: {stats[current_tc_key]}")

    # verify test case ranks are calculated
    assert _explanations_cmp_html.KEY_TEST_CASE_RANKS_BASELINE in stats
    assert _explanations_cmp_html.KEY_TEST_CASE_RANKS_CURRENT in stats

    # expected: baseline=1, current=1 (each won one test case)
    baseline_tc_ranks = stats[_explanations_cmp_html.KEY_TEST_CASE_RANKS_BASELINE]
    current_tc_ranks = stats[_explanations_cmp_html.KEY_TEST_CASE_RANKS_CURRENT]

    # verify expected values
    msg = f"Expected baseline test case ranks 1, got {baseline_tc_ranks}"
    assert baseline_tc_ranks == 1, msg
    msg = f"Expected current test case ranks 1, got {current_tc_ranks}"
    assert current_tc_ranks == 1, msg


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_test_case_ranks_calculation_multiple_metrics():
    """Test test case ranks with multiple metrics per test case.

    This test verifies that within each test case, metrics are ranked
    and averaged to determine the winner.
    """
    #
    # GIVEN
    #

    class MockMetricMeta:
        def __init__(self, higher_is_better: bool):
            self.higher_is_better = higher_is_better

    class MockTestCase(dict):
        def __init__(self, accuracy, error_rate):
            super().__init__()
            self["accuracy"] = accuracy
            self["error_rate"] = error_rate
            self[_explanations_cmp.KEY_CONTEXT] = ["some context"]

    class MockDiff:
        def __init__(
            self,
            baseline_accuracy,
            baseline_error,
            current_accuracy,
            current_error,
        ):
            self.baseline_test_case = MockTestCase(baseline_accuracy, baseline_error)
            self.current_test_case = MockTestCase(current_accuracy, current_error)
            self.diff_flipped_metrics = {}

    # test case 0:
    # - accuracy (higher is better): baseline=0.9, current=0.8
    #   -> baseline gets rank 2, current gets rank 1
    # - error_rate (lower is better): baseline=0.2, current=0.1
    #   -> current gets rank 2, baseline gets rank 1
    # average ranks: baseline=(2+1)/2=1.5, current=(1+2)/2=1.5
    # -> tie, no winner for this test case

    # test case 1:
    # - accuracy: baseline=0.95, current=0.85
    #   -> baseline gets rank 2, current gets rank 1
    # - error_rate: baseline=0.1, current=0.2
    #   -> baseline gets rank 2, current gets rank 1
    # average ranks: baseline=(2+2)/2=2.0, current=(1+1)/2=1.0
    # -> baseline wins this test case

    model_diffs = [
        (0, MockDiff(0.9, 0.2, 0.8, 0.1)),
        (1, MockDiff(0.95, 0.1, 0.85, 0.2)),
    ]

    metrics_meta = {
        "accuracy": MockMetricMeta(higher_is_better=True),
        "error_rate": MockMetricMeta(higher_is_better=False),
    }

    #
    # WHEN
    #
    stats = _explanations_cmp_html.EvalResultsDiffHtml._calculate_model_cmp_stats(
        model_diffs=model_diffs, metrics_meta=metrics_meta
    )

    #
    # THEN
    #
    print(f"\nStats: {stats}")
    baseline_tc_key = _explanations_cmp_html.KEY_TEST_CASE_RANKS_BASELINE
    current_tc_key = _explanations_cmp_html.KEY_TEST_CASE_RANKS_CURRENT
    baseline_tc_ranks = stats[baseline_tc_key]
    current_tc_ranks = stats[current_tc_key]

    print(f"Baseline test case ranks: {baseline_tc_ranks}")
    print(f"Current test case ranks: {current_tc_ranks}")

    # expected: baseline=1, current=0 (baseline won 1, current won 0)
    msg = f"Expected baseline test case ranks 1, got {baseline_tc_ranks}"
    assert baseline_tc_ranks == 1, msg
    msg = f"Expected current test case ranks 0, got {current_tc_ranks}"
    assert current_tc_ranks == 0, msg


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_per_metric_ranks_calculation():
    """Test per-metric rank calculation and storage.

    This test verifies that average ranks are calculated and stored
    correctly for each individual metric in the metrics_averages dict.
    """
    #
    # GIVEN
    #

    class MockMetricMeta:
        def __init__(self, higher_is_better: bool):
            self.higher_is_better = higher_is_better

    class MockTestCase(dict):
        def __init__(self, accuracy, error_rate):
            super().__init__()
            self["accuracy"] = accuracy
            self["error_rate"] = error_rate
            self[_explanations_cmp.KEY_CONTEXT] = ["some context"]

    class MockDiff:
        def __init__(
            self,
            baseline_accuracy,
            baseline_error,
            current_accuracy,
            current_error,
        ):
            self.baseline_test_case = MockTestCase(baseline_accuracy, baseline_error)
            self.current_test_case = MockTestCase(current_accuracy, current_error)
            self.diff_flipped_metrics = {}

    # scenario: 2 test cases, 2 metrics
    # accuracy (higher is better):
    #   baseline: [0.9, 0.8], current: [0.85, 0.75]
    #   merged & sorted desc: [0.9, 0.85, 0.8, 0.75]
    #   inverted ranks: [4, 3, 2, 1]
    #   baseline ranks: [4, 2] -> avg = 3.0
    #   current ranks: [3, 1] -> avg = 2.0

    # error_rate (lower is better):
    #   baseline: [0.1, 0.2], current: [0.15, 0.05]
    #   merged & sorted asc: [0.05, 0.1, 0.15, 0.2]
    #   inverted ranks: [4, 3, 2, 1]
    #   current ranks: [4, 2] -> avg = 3.0
    #   baseline ranks: [3, 1] -> avg = 2.0

    model_diffs = [
        (0, MockDiff(0.9, 0.1, 0.85, 0.15)),
        (1, MockDiff(0.8, 0.2, 0.75, 0.05)),
    ]

    metrics_meta = {
        "accuracy": MockMetricMeta(higher_is_better=True),
        "error_rate": MockMetricMeta(higher_is_better=False),
    }

    #
    # WHEN
    #
    stats = _explanations_cmp_html.EvalResultsDiffHtml._calculate_model_cmp_stats(
        model_diffs=model_diffs, metrics_meta=metrics_meta
    )

    #
    # THEN
    #
    print(f"\nStats: {stats}")
    metrics_averages = stats[_explanations_cmp_html.KEY_METRICS_AVERAGES]

    # verify per-metric ranks are stored
    assert "accuracy" in metrics_averages
    assert "error_rate" in metrics_averages

    # accuracy ranks
    accuracy_baseline_rank = metrics_averages["accuracy"][
        _explanations_cmp_html.KEY_BASELINE_RANK_AVG
    ]
    accuracy_current_rank = metrics_averages["accuracy"][
        _explanations_cmp_html.KEY_CURRENT_RANK_AVG
    ]

    print(
        f"Accuracy - Baseline rank: {accuracy_baseline_rank}, "
        f"Current rank: {accuracy_current_rank}"
    )

    # expected: baseline=3.0, current=2.0
    msg = f"Expected accuracy baseline rank 3.0, got {accuracy_baseline_rank}"
    assert accuracy_baseline_rank == 3.0, msg
    msg = f"Expected accuracy current rank 2.0, got {accuracy_current_rank}"
    assert accuracy_current_rank == 2.0, msg

    # error_rate ranks
    error_rate_baseline_rank = metrics_averages["error_rate"][
        _explanations_cmp_html.KEY_BASELINE_RANK_AVG
    ]
    error_rate_current_rank = metrics_averages["error_rate"][
        _explanations_cmp_html.KEY_CURRENT_RANK_AVG
    ]

    print(
        f"Error Rate - Baseline rank: {error_rate_baseline_rank}, "
        f"Current rank: {error_rate_current_rank}"
    )

    # expected: baseline=2.0, current=3.0
    msg = f"Expected error_rate baseline rank 2.0, got {error_rate_baseline_rank}"
    assert error_rate_baseline_rank == 2.0, msg
    msg = f"Expected error_rate current rank 3.0, got {error_rate_current_rank}"
    assert error_rate_current_rank == 3.0, msg


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_metrics_ranks_calculation_with_ties():
    """Test rank-based comparison with identical scores (ties).

    This test verifies that when scores are identical, they receive
    identical ranks (average of the ranks they would occupy).
    This addresses the issue where M1: [0.0, 0.0, 0.0] and M2: [0.0, 0.0, 0.0]
    should be on par.
    """
    #
    # GIVEN
    #

    class MockMetricMeta:
        def __init__(self, higher_is_better: bool):
            self.higher_is_better = higher_is_better

    class MockTestCase(dict):
        def __init__(self, accuracy):
            super().__init__()
            self["accuracy"] = accuracy
            self[_explanations_cmp.KEY_CONTEXT] = ["some context"]

    class MockDiff:
        def __init__(self, baseline_accuracy, current_accuracy):
            self.baseline_test_case = MockTestCase(baseline_accuracy)
            self.current_test_case = MockTestCase(current_accuracy)
            self.diff_flipped_metrics = {}

    # scenario: all scores are identical (0.0)
    # baseline: [0.0, 0.0, 0.0]
    # current: [0.0, 0.0, 0.0]
    # merged & sorted: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # all 6 values are tied, so they should all get average rank:
    # ranks would be: [6, 5, 4, 3, 2, 1]
    # average = (6 + 5 + 4 + 3 + 2 + 1) / 6 = 21 / 6 = 3.5
    # both baseline and current should get rank 3.5

    model_diffs = [
        (0, MockDiff(0.0, 0.0)),
        (1, MockDiff(0.0, 0.0)),
        (2, MockDiff(0.0, 0.0)),
    ]

    metrics_meta = {
        "accuracy": MockMetricMeta(higher_is_better=True),
    }

    #
    # WHEN
    #
    stats = _explanations_cmp_html.EvalResultsDiffHtml._calculate_model_cmp_stats(
        model_diffs=model_diffs, metrics_meta=metrics_meta
    )

    #
    # THEN
    #
    print(f"\nStats: {stats}")
    baseline_key = _explanations_cmp_html.KEY_METRICS_RANKS_BASELINE
    current_key = _explanations_cmp_html.KEY_METRICS_RANKS_CURRENT
    baseline_rank = stats[baseline_key]
    current_rank = stats[current_key]

    print(f"Baseline rank: {baseline_rank}")
    print(f"Current rank: {current_rank}")

    # both should have rank 3.5 (perfect tie)
    expected_rank = 3.5
    msg = f"Expected baseline rank {expected_rank}, got {baseline_rank}"
    assert baseline_rank == expected_rank, msg
    msg = f"Expected current rank {expected_rank}, got {current_rank}"
    assert current_rank == expected_rank, msg

    # verify they are truly equal (no unfairness)
    assert baseline_rank == current_rank, (
        "Models should be on par with identical scores"
    )
