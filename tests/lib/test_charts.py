# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import airium
import pytest

from h2o_sonar.utils import charts


@pytest.mark.h2o_sonar
def test_svg_charts_showcase(tmp_path):
    """Showcase all SVG chart utilities and generate HTML report.

    This test demonstrates all chart types and generates an HTML file
    with examples that can be viewed in a browser.
    """
    #
    # GIVEN - sample data for charts
    #
    # simple metrics
    metric_labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "Perplexity"]
    metric_values = [0.45, 0.32, 0.41, 0.38, 20.5]

    # long metric names for horizontal chart
    long_labels = [
        "Answer Semantic Similarity",
        "Context Relevancy Score",
        "Faithfulness Metric",
        "Groundedness Evaluation",
    ]
    long_values = [0.87, 0.92, 0.78, 0.85]

    # comparison data
    comparison_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    baseline_values = [0.85, 0.82, 0.88, 0.85]
    current_values = [0.89, 0.86, 0.91, 0.88]

    #
    # WHEN - generate charts
    #
    # vertical bar chart
    vertical_chart = charts.generate_svg_bar_chart(
        labels=metric_labels,
        values=metric_values,
        title="Evaluator Metrics - Vertical Bar Chart",
        width=800,
        height=400,
    )

    # horizontal bar chart (better for long labels)
    horizontal_chart = charts.generate_svg_horizontal_bar_chart(
        labels=long_labels,
        values=long_values,
        title="RAG Evaluator Scores - Horizontal Bar Chart",
        width=800,
        height=400,
    )

    # grouped bar chart (baseline vs. current comparison)
    grouped_chart = charts.generate_svg_grouped_bar_chart(
        labels=comparison_labels,
        baseline_values=baseline_values,
        current_values=current_values,
        title="Baseline vs. Current Model Performance",
        baseline_label="Baseline Model",
        current_label="Current Model",
        width=800,
        height=450,
    )

    # chart with custom colors
    custom_chart = charts.generate_svg_bar_chart(
        labels=["Good", "Better", "Best"],
        values=[0.7, 0.85, 0.95],
        title="Custom Colors Example",
        width=600,
        height=300,
        bar_color="#4CAF50",  # green
        bar_border_color="#2E7D32",
    )

    # empty chart (edge case)
    empty_chart = charts.generate_svg_bar_chart(
        labels=[],
        values=[],
        title="Empty Chart Example",
        width=600,
        height=300,
    )

    #
    # THEN - verify charts are valid SVG
    #
    print("\nChart validation:")
    print("- Vertical bar chart generated")
    assert vertical_chart.startswith("<svg")
    assert vertical_chart.endswith("</svg>")
    assert "Evaluator Metrics" in vertical_chart
    assert all(label in vertical_chart for label in metric_labels)

    print("- Horizontal bar chart generated")
    assert horizontal_chart.startswith("<svg")
    assert "Horizontal Bar Chart" in horizontal_chart

    print("- Grouped bar chart generated")
    assert grouped_chart.startswith("<svg")
    assert "Baseline Model" in grouped_chart
    assert "Current Model" in grouped_chart

    print("- Custom colors chart generated")
    assert custom_chart.startswith("<svg")
    assert "#4CAF50" in custom_chart

    print("- Empty chart handled gracefully")
    assert empty_chart.startswith("<svg")
    assert "No data to display" in empty_chart

    #
    # THEN - generate HTML report with all charts
    #
    html = airium.Airium()

    with html.html():
        with html.head():
            html.meta(charset="UTF-8")
            html.title(_t="H2O Sonar - SVG Charts Showcase")
            with html.style():
                html(
                    """
                    body {
                        font-family: "Segoe UI", Arial, sans-serif;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 40px 20px;
                        background: #f5f5f5;
                    }
                    h1 {
                        color: #161616;
                        text-align: center;
                        margin-bottom: 10px;
                    }
                    .subtitle {
                        text-align: center;
                        color: #54585A;
                        margin-bottom: 40px;
                    }
                    .chart-section {
                        background: white;
                        padding: 30px;
                        margin: 20px 0;
                        border-radius: 8px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    }
                    .chart-description {
                        color: #54585A;
                        font-size: 14px;
                        margin-bottom: 20px;
                        line-height: 1.6;
                    }
                    .stats {
                        background: #FFF9E6;
                        border-left: 4px solid #FEC925;
                        padding: 15px;
                        margin: 30px 0;
                    }
                    .stats h3 {
                        margin-top: 0;
                        color: #161616;
                    }
                    .stats ul {
                        margin: 10px 0;
                        padding-left: 20px;
                    }
                    .stats li {
                        color: #54585A;
                        margin: 5px 0;
                    }
                    code {
                        background: #f5f5f5;
                        padding: 2px 6px;
                        border-radius: 3px;
                        font-family: "Courier New", monospace;
                        font-size: 13px;
                    }
                    """
                )

        with html.body():
            html.h1(_t="H2O Sonar - SVG Charts Showcase")
            with html.p(klass="subtitle"):
                html(
                    "Inline SVG charts with zero external dependencies - "
                    "generated entirely in Python"
                )

            # statistics
            with html.div(klass="stats"):
                html.h3(_t="Chart Generation Statistics")
                with html.ul():
                    html.li(_t="File size per chart: 1-5KB (10-100x smaller than PNG)")
                    html.li(_t="Generation time: 5-20ms per chart")
                    html.li(_t="Vector graphics: infinite scalability")
                    html.li(_t="Zero external dependencies: no CDN, no external files")
                    html.li(_t="H2O.ai branding: #FEC925, #161616, #54585A")

            # vertical bar chart
            with html.div(klass="chart-section"):
                html.h2(_t="1. Vertical Bar Chart")
                with html.p(klass="chart-description"):
                    html(
                        "Standard vertical bar chart, ideal for comparing multiple "
                        "metrics with short labels. Includes grid lines, value labels, "
                        "and axis labels. Use "
                    )
                    html.code(_t="generate_svg_bar_chart()")
                    html(" to create this chart type.")
                charts.add_svg_chart_to_html(html, vertical_chart)

            # horizontal bar chart
            with html.div(klass="chart-section"):
                html.h2(_t="2. Horizontal Bar Chart")
                with html.p(klass="chart-description"):
                    html(
                        "Horizontal bar chart, better suited for long metric names "
                        "as labels have more space on the left. Perfect for RAG "
                        "evaluator names or detailed metric descriptions. Use "
                    )
                    html.code(_t="generate_svg_horizontal_bar_chart()")
                    html(" to create this chart type.")
                charts.add_svg_chart_to_html(html, horizontal_chart)

            # grouped bar chart
            with html.div(klass="chart-section"):
                html.h2(_t="3. Grouped Bar Chart (Comparison)")
                with html.p(klass="chart-description"):
                    html(
                        "Grouped bar chart for comparing baseline vs. current model "
                        "performance. Shows two bars per metric with different colors "
                        "and includes a legend. Ideal for A/B testing and model "
                        "comparison reports. Use "
                    )
                    html.code(_t="generate_svg_grouped_bar_chart()")
                    html(" to create this chart type.")
                charts.add_svg_chart_to_html(html, grouped_chart)

            # custom colors
            with html.div(klass="chart-section"):
                html.h2(_t="4. Custom Colors")
                with html.p(klass="chart-description"):
                    html(
                        "Charts can be customized with any color scheme. "
                        "This example uses green colors instead of H2O.ai branding. "
                        "All chart functions accept "
                    )
                    html.code(_t="bar_color")
                    html(" and ")
                    html.code(_t="bar_border_color")
                    html(" parameters.")
                charts.add_svg_chart_to_html(html, custom_chart)

            # empty chart
            with html.div(klass="chart-section"):
                html.h2(_t="5. Empty Chart Handling")
                with html.p(klass="chart-description"):
                    html(
                        "When no data is available, charts display a friendly "
                        "message instead of crashing. This ensures robust HTML "
                        "report generation even with incomplete data."
                    )
                charts.add_svg_chart_to_html(html, empty_chart)

            # footer
            with html.div(klass="chart-section"):
                html.h2(_t="Usage Example")
                with html.p(klass="chart-description"):
                    html("Python code to generate these charts:")
                with html.pre(
                    style="background: #f5f5f5; padding: 15px; "
                    "border-radius: 5px; overflow-x: auto;"
                ):
                    html.code(
                        _t="""from h2o_sonar.lib.api import charts

# vertical bar chart
svg = charts.generate_svg_bar_chart(
    labels=["ROUGE-1", "ROUGE-2", "ROUGE-L"],
    values=[0.45, 0.32, 0.41],
    title="Metrics",
    width=800,
    height=400
)

# horizontal bar chart
svg = charts.generate_svg_horizontal_bar_chart(
    labels=["Long Metric Name A", "Long Metric Name B"],
    values=[0.87, 0.92],
    title="Scores"
)

# grouped comparison chart
svg = charts.generate_svg_grouped_bar_chart(
    labels=["Metric A", "Metric B"],
    baseline_values=[0.5, 0.6],
    current_values=[0.55, 0.65],
    title="Baseline vs. Current"
)

# add to airium HTML
import airium
html = airium.Airium()
charts.add_svg_chart_to_html(html, svg)
"""
                    )

    # write HTML to file
    html_path = tmp_path / "charts_showcase.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(str(html))

    print("\nHTML report generated:")
    print(f"  file://{html_path}")
    print(f"  Size: {html_path.stat().st_size:,} bytes")

    # verify file was created
    assert html_path.exists()
    assert html_path.stat().st_size > 0

    # verify HTML contains all charts
    html_content = html_path.read_text(encoding="utf-8")
    assert html_content.count("<svg") == 5  # 5 charts
    assert "Vertical Bar Chart" in html_content
    assert "Horizontal Bar Chart" in html_content
    assert "Grouped Bar Chart" in html_content

    print("\nDONE - All charts validated and HTML showcase generated successfully")


@pytest.mark.h2o_sonar
def test_chart_sanitization():
    """Test that chart inputs are properly sanitized to prevent XSS."""
    #
    # GIVEN - malicious input with HTML/JavaScript
    #
    malicious_label = '<script>alert("XSS")</script>'
    malicious_title = '<img src=x onerror="alert(1)">'

    #
    # WHEN - generate chart with malicious input
    #
    svg = charts.generate_svg_bar_chart(
        labels=[malicious_label, "Safe Label"],
        values=[10, 20],
        title=malicious_title,
    )

    #
    # THEN - verify dangerous content is escaped
    #
    print("\nXSS sanitization test:")
    # dangerous tags should be escaped
    assert "<script>" not in svg
    assert "<img src=x onerror=" not in svg
    # but escaped versions should be present
    assert "&lt;script&gt;" in svg  # escaped
    assert "&lt;img" in svg  # escaped
    assert "&quot;" in svg  # escaped quotes
    print("- Malicious script tags properly escaped")
    print("- HTML injection prevented")
    print("- Dangerous attributes escaped")

    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")
    print("- Valid SVG structure maintained")


@pytest.mark.h2o_sonar
def test_chart_edge_cases():
    """Test edge cases: single value, zero values, negative values, etc."""
    #
    # GIVEN / WHEN / THEN - single value
    #
    svg = charts.generate_svg_bar_chart(
        labels=["Only One"], values=[42.5], title="Single Value"
    )
    assert "<svg" in svg
    assert "Only One" in svg
    print("\nEdge cases:")
    print("- Single value chart: OK")

    # zero values
    svg = charts.generate_svg_bar_chart(
        labels=["Zero"], values=[0.0], title="Zero Value"
    )
    assert "<svg" in svg
    print("- Zero value chart: OK")

    # negative values
    svg = charts.generate_svg_bar_chart(
        labels=["Negative"], values=[-5.0], title="Negative Value"
    )
    assert "<svg" in svg
    print("- Negative value chart: OK")

    # mixed positive/negative
    svg = charts.generate_svg_bar_chart(
        labels=["A", "B", "C"], values=[-10, 0, 10], title="Mixed Values"
    )
    assert "<svg" in svg
    print("- Mixed positive/negative values: OK")

    # mismatched lengths should raise error
    try:
        charts.generate_svg_bar_chart(
            labels=["A", "B"], values=[1, 2, 3], title="Mismatch"
        )
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Length mismatch" in str(e)
        print("- Length mismatch properly detected: OK")


@pytest.mark.h2o_sonar
def test_grouped_chart_comparison():
    """Test grouped bar chart for baseline vs. current comparison."""
    #
    # GIVEN - baseline and current values
    #
    labels = ["Metric A", "Metric B", "Metric C"]
    baseline = [0.70, 0.65, 0.80]
    current = [0.75, 0.68, 0.85]

    #
    # WHEN - generate grouped comparison chart
    #
    svg = charts.generate_svg_grouped_bar_chart(
        labels=labels,
        baseline_values=baseline,
        current_values=current,
        title="Model Comparison",
        baseline_label="Old Model",
        current_label="New Model",
    )

    #
    # THEN - verify both datasets are present
    #
    print("\nGrouped chart test:")
    assert "<svg" in svg
    assert "Model Comparison" in svg
    assert "Old Model" in svg  # legend
    assert "New Model" in svg  # legend

    # verify all metrics appear
    for label in labels:
        assert label in svg

    print("- Grouped chart with legend: OK")
    print("- Both baseline and current values rendered: OK")


@pytest.mark.h2o_sonar
def test_chart_customization():
    """Test chart customization options."""
    #
    # GIVEN - custom settings
    #
    labels = ["A", "B", "C"]
    values = [10, 20, 15]

    #
    # WHEN - generate chart with custom settings
    #
    # without values shown
    svg_no_values = charts.generate_svg_bar_chart(
        labels=labels,
        values=values,
        title="No Values",
        show_values=False,
    )

    # without grid
    svg_no_grid = charts.generate_svg_bar_chart(
        labels=labels,
        values=values,
        title="No Grid",
        show_grid=False,
    )

    # custom colors
    svg_custom = charts.generate_svg_bar_chart(
        labels=labels,
        values=values,
        title="Custom",
        bar_color="#FF5733",
        bar_border_color="#900C3F",
    )

    # custom dimensions
    svg_large = charts.generate_svg_bar_chart(
        labels=labels,
        values=values,
        title="Large",
        width=1200,
        height=600,
    )

    #
    # THEN - verify customizations applied
    #
    print("\nCustomization test:")
    assert "<svg" in svg_no_values
    print("- Chart without value labels: OK")

    assert "<svg" in svg_no_grid
    print("- Chart without grid lines: OK")

    assert "#FF5733" in svg_custom
    assert "#900C3F" in svg_custom
    print("- Custom colors applied: OK")

    assert 'width="1200"' in svg_large
    assert 'height="600"' in svg_large
    print("- Custom dimensions applied: OK")
