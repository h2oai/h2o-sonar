# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
"""Tests for chart integration in HTML comparison reports."""

from h2o_sonar.lib.api.explanations import _explanations_cmp_html
from h2o_sonar.utils import charts


class TestTryGenerateChart:
    """Test suite for _try_generate_chart helper function."""

    def test_valid_data_returns_svg(self):
        """Test that valid data generates SVG chart successfully."""
        # GIVEN
        labels = ["Test Case Wins", "Metrics Wins"]
        baseline_values = [5.0, 10.0]
        current_values = [7.0, 12.0]

        # WHEN
        result = _explanations_cmp_html.EvalResultsDiffHtml._try_generate_chart(
            charts.generate_svg_grouped_bar_chart,
            labels=labels,
            baseline_values=baseline_values,
            current_values=current_values,
            title="Test Chart",
        )

        # THEN
        print(f"Result type: {type(result)}")
        assert result is not None
        assert isinstance(result, str)
        assert "<svg" in result
        assert "Test Chart" in result

    def test_empty_labels_returns_none(self):
        """Test that empty labels list returns None."""
        # GIVEN
        labels = []
        baseline_values = [5.0]
        current_values = [7.0]

        # WHEN
        result = _explanations_cmp_html.EvalResultsDiffHtml._try_generate_chart(
            charts.generate_svg_grouped_bar_chart,
            labels=labels,
            baseline_values=baseline_values,
            current_values=current_values,
            title="Test Chart",
        )

        # THEN
        print(f"Result: {result}")
        assert result is None

    def test_none_labels_returns_none(self):
        """Test that None labels returns None."""
        # GIVEN
        labels = None
        baseline_values = [5.0]
        current_values = [7.0]

        # WHEN
        result = _explanations_cmp_html.EvalResultsDiffHtml._try_generate_chart(
            charts.generate_svg_grouped_bar_chart,
            labels=labels,
            baseline_values=baseline_values,
            current_values=current_values,
            title="Test Chart",
        )

        # THEN
        print(f"Result: {result}")
        assert result is None

    def test_none_baseline_values_returns_none(self):
        """Test that None baseline values returns None."""
        # GIVEN
        labels = ["A", "B"]
        baseline_values = None
        current_values = [5.0, 7.0]

        # WHEN
        result = _explanations_cmp_html.EvalResultsDiffHtml._try_generate_chart(
            charts.generate_svg_grouped_bar_chart,
            labels=labels,
            baseline_values=baseline_values,
            current_values=current_values,
            title="Test Chart",
        )

        # THEN
        print(f"Result: {result}")
        assert result is None

    def test_none_current_values_returns_none(self):
        """Test that None current values returns None."""
        # GIVEN
        labels = ["A", "B"]
        baseline_values = [5.0, 7.0]
        current_values = None

        # WHEN
        result = _explanations_cmp_html.EvalResultsDiffHtml._try_generate_chart(
            charts.generate_svg_grouped_bar_chart,
            labels=labels,
            baseline_values=baseline_values,
            current_values=current_values,
            title="Test Chart",
        )

        # THEN
        print(f"Result: {result}")
        assert result is None

    def test_empty_baseline_values_returns_none(self):
        """Test that empty baseline values list returns None."""
        # GIVEN
        labels = ["A", "B"]
        baseline_values = []
        current_values = [5.0, 7.0]

        # WHEN
        result = _explanations_cmp_html.EvalResultsDiffHtml._try_generate_chart(
            charts.generate_svg_grouped_bar_chart,
            labels=labels,
            baseline_values=baseline_values,
            current_values=current_values,
            title="Test Chart",
        )

        # THEN
        print(f"Result: {result}")
        assert result is None

    def test_empty_current_values_returns_none(self):
        """Test that empty current values list returns None."""
        # GIVEN
        labels = ["A", "B"]
        baseline_values = [5.0, 7.0]
        current_values = []

        # WHEN
        result = _explanations_cmp_html.EvalResultsDiffHtml._try_generate_chart(
            charts.generate_svg_grouped_bar_chart,
            labels=labels,
            baseline_values=baseline_values,
            current_values=current_values,
            title="Test Chart",
        )

        # THEN
        print(f"Result: {result}")
        assert result is None

    def test_mismatched_lengths_returns_none(self):
        """Test that mismatched list lengths returns None."""
        # GIVEN
        labels = ["A", "B"]
        baseline_values = [5.0]  # only 1 value
        current_values = [7.0, 8.0]

        # WHEN
        result = _explanations_cmp_html.EvalResultsDiffHtml._try_generate_chart(
            charts.generate_svg_grouped_bar_chart,
            labels=labels,
            baseline_values=baseline_values,
            current_values=current_values,
            title="Test Chart",
        )

        # THEN
        print(f"Result: {result}")
        assert result is None

    def test_nan_value_returns_none(self):
        """Test that NaN values return None."""
        # GIVEN
        labels = ["A", "B"]
        baseline_values = [5.0, float("nan")]
        current_values = [7.0, 8.0]

        # WHEN
        result = _explanations_cmp_html.EvalResultsDiffHtml._try_generate_chart(
            charts.generate_svg_grouped_bar_chart,
            labels=labels,
            baseline_values=baseline_values,
            current_values=current_values,
            title="Test Chart",
        )

        # THEN
        print(f"Result: {result}")
        assert result is None

    def test_string_value_returns_none(self):
        """Test that non-numeric string values return None."""
        # GIVEN
        labels = ["A", "B"]
        baseline_values = [5.0, "invalid"]
        current_values = [7.0, 8.0]

        # WHEN
        result = _explanations_cmp_html.EvalResultsDiffHtml._try_generate_chart(
            charts.generate_svg_grouped_bar_chart,
            labels=labels,
            baseline_values=baseline_values,
            current_values=current_values,
            title="Test Chart",
        )

        # THEN
        print(f"Result: {result}")
        assert result is None

    def test_non_list_labels_returns_none(self):
        """Test that non-list labels returns None."""
        # GIVEN
        labels = "invalid"
        baseline_values = [5.0, 7.0]
        current_values = [7.0, 8.0]

        # WHEN
        result = _explanations_cmp_html.EvalResultsDiffHtml._try_generate_chart(
            charts.generate_svg_grouped_bar_chart,
            labels=labels,
            baseline_values=baseline_values,
            current_values=current_values,
            title="Test Chart",
        )

        # THEN
        print(f"Result: {result}")
        assert result is None

    def test_non_list_values_returns_none(self):
        """Test that non-list values returns None."""
        # GIVEN
        labels = ["A", "B"]
        baseline_values = "invalid"
        current_values = [7.0, 8.0]

        # WHEN
        result = _explanations_cmp_html.EvalResultsDiffHtml._try_generate_chart(
            charts.generate_svg_grouped_bar_chart,
            labels=labels,
            baseline_values=baseline_values,
            current_values=current_values,
            title="Test Chart",
        )

        # THEN
        print(f"Result: {result}")
        assert result is None

    def test_zero_values_generates_chart(self):
        """Test that zero values still generate a valid chart."""
        # GIVEN
        labels = ["A", "B"]
        baseline_values = [0.0, 0.0]
        current_values = [0.0, 0.0]

        # WHEN
        result = _explanations_cmp_html.EvalResultsDiffHtml._try_generate_chart(
            charts.generate_svg_grouped_bar_chart,
            labels=labels,
            baseline_values=baseline_values,
            current_values=current_values,
            title="Test Chart",
        )

        # THEN
        print(f"Result type: {type(result)}")
        assert result is not None
        assert isinstance(result, str)
        assert "<svg" in result

    def test_mixed_zero_and_positive_values(self):
        """Test that mixed zero and positive values generate chart."""
        # GIVEN
        labels = ["A", "B"]
        baseline_values = [0.0, 5.0]
        current_values = [7.0, 0.0]

        # WHEN
        result = _explanations_cmp_html.EvalResultsDiffHtml._try_generate_chart(
            charts.generate_svg_grouped_bar_chart,
            labels=labels,
            baseline_values=baseline_values,
            current_values=current_values,
            title="Test Chart",
        )

        # THEN
        print(f"Result type: {type(result)}")
        assert result is not None
        assert isinstance(result, str)
        assert "<svg" in result

    def test_integer_values_work(self):
        """Test that integer values are accepted."""
        # GIVEN
        labels = ["A", "B"]
        baseline_values = [5, 10]
        current_values = [7, 12]

        # WHEN
        result = _explanations_cmp_html.EvalResultsDiffHtml._try_generate_chart(
            charts.generate_svg_grouped_bar_chart,
            labels=labels,
            baseline_values=baseline_values,
            current_values=current_values,
            title="Test Chart",
        )

        # THEN
        print(f"Result type: {type(result)}")
        assert result is not None
        assert isinstance(result, str)
        assert "<svg" in result

    def test_exception_in_chart_func_returns_none(self):
        """Test that exceptions in chart function are caught and return None."""
        # GIVEN

        def failing_chart_func(*args, **kwargs):
            raise ValueError("Simulated chart generation failure")

        labels = ["A", "B"]
        baseline_values = [5.0, 7.0]
        current_values = [7.0, 8.0]

        # WHEN
        result = _explanations_cmp_html.EvalResultsDiffHtml._try_generate_chart(
            failing_chart_func,
            labels=labels,
            baseline_values=baseline_values,
            current_values=current_values,
            title="Test Chart",
        )

        # THEN
        print(f"Result: {result}")
        assert result is None

    def test_invalid_svg_output_returns_none(self):
        """Test that invalid SVG output is rejected."""
        # GIVEN

        def invalid_svg_func(*args, **kwargs):
            return "not an svg string"

        labels = ["A", "B"]
        baseline_values = [5.0, 7.0]
        current_values = [7.0, 8.0]

        # WHEN
        result = _explanations_cmp_html.EvalResultsDiffHtml._try_generate_chart(
            invalid_svg_func,
            labels=labels,
            baseline_values=baseline_values,
            current_values=current_values,
            title="Test Chart",
        )

        # THEN
        print(f"Result: {result}")
        assert result is None

    def test_empty_svg_output_returns_none(self):
        """Test that empty SVG output is rejected."""
        # GIVEN

        def empty_svg_func(*args, **kwargs):
            return ""

        labels = ["A", "B"]
        baseline_values = [5.0, 7.0]
        current_values = [7.0, 8.0]

        # WHEN
        result = _explanations_cmp_html.EvalResultsDiffHtml._try_generate_chart(
            empty_svg_func,
            labels=labels,
            baseline_values=baseline_values,
            current_values=current_values,
            title="Test Chart",
        )

        # THEN
        print(f"Result: {result}")
        assert result is None

    def test_positional_args_work(self):
        """Test that positional arguments work correctly."""
        # GIVEN
        labels = ["A", "B"]
        baseline_values = [5.0, 7.0]
        current_values = [7.0, 8.0]

        # WHEN
        result = _explanations_cmp_html.EvalResultsDiffHtml._try_generate_chart(
            charts.generate_svg_grouped_bar_chart,
            labels,
            baseline_values,
            current_values,
            "Test Chart",
        )

        # THEN
        print(f"Result type: {type(result)}")
        assert result is not None
        assert isinstance(result, str)
        assert "<svg" in result
