# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
"""Inline SVG chart generation for HTML reports.

This module provides utilities for generating embedded SVG charts with zero
external dependencies. All charts are generated as inline SVG strings that can
be directly embedded in HTML reports.

Examples
--------
Generate a simple bar chart:

>>> labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
>>> values = [0.45, 0.32, 0.41]
>>> svg = generate_svg_bar_chart(labels, values, "Metrics Comparison")

Generate a grouped comparison chart:

>>> svg = generate_svg_grouped_bar_chart(
...     labels=["Metric A", "Metric B"],
...     baseline_values=[0.5, 0.6],
...     current_values=[0.55, 0.65],
...     title="Baseline vs. Current"
... )

"""

import html as html_escape


# H2O.ai brand colors
H2O_YELLOW = "#FEC925"
H2O_BLACK = "#161616"
H2O_GRAY = "#54585A"
H2O_LIGHT_GRAY = "#E0E0E0"
H2O_YELLOW_LIGHT = "#FFD966"


def sanitize_text(text: str) -> str:
    """Sanitize text for safe embedding in SVG to prevent XSS.

    Parameters
    ----------
    text : str
        Text to sanitize.

    Returns
    -------
    str
        HTML-escaped text safe for SVG embedding.

    """
    return html_escape.escape(str(text))


def format_number(value: float, decimals: int = 2) -> str:
    """Format number for display in charts.

    Parameters
    ----------
    value : float
        Number to format.
    decimals : int
        Number of decimal places (default: 2).

    Returns
    -------
    str
        Formatted number string.

    """
    return f"{value:.{decimals}f}"


def truncate_label(label: str, max_length: int = 20) -> str:
    """Truncate label if too long, adding ellipsis.

    Parameters
    ----------
    label : str
        Label text to truncate.
    max_length : int
        Maximum length before truncation (default: 20).

    Returns
    -------
    str
        Truncated label with ellipsis if needed.

    """
    if len(label) <= max_length:
        return label
    return label[: max_length - 3] + "..."


def generate_svg_bar_chart(
    labels: list[str],
    values: list[float],
    title: str,
    width: int = 800,
    height: int = 400,
    show_values: bool = True,
    show_grid: bool = True,
    bar_color: str = H2O_YELLOW,
    bar_border_color: str = H2O_BLACK,
    bar_colors: list[str] | None = None,
) -> str:
    """Generate vertical bar chart as inline SVG.

    Parameters
    ----------
    labels : list[str]
        Labels for x-axis (one per bar).
    values : list[float]
        Values for each bar.
    title : str
        Chart title.
    width : int
        Chart width in pixels (default: 800).
    height : int
        Chart height in pixels (default: 400).
    show_values : bool
        Whether to show value labels on bars (default: True).
    show_grid : bool
        Whether to show horizontal grid lines (default: True).
    bar_color : str
        Bar fill color (default: H2O yellow). Used when bar_colors is None.
    bar_border_color : str
        Bar border color (default: H2O black).
    bar_colors : list[str] | None
        List of colors for each bar. If provided, overrides bar_color.
        Must match the length of labels/values.

    Returns
    -------
    str
        Complete SVG chart as string.

    Examples
    --------
    >>> labels = ["A", "B", "C"]
    >>> values = [10, 20, 15]
    >>> svg = generate_svg_bar_chart(labels, values, "Test Chart")
    >>> assert '<svg' in svg
    >>> assert 'Test Chart' in svg

    """
    if not labels or not values:
        return _generate_empty_chart(width, height, "No data to display")

    if len(labels) != len(values):
        raise ValueError(
            f"Length mismatch: {len(labels)} labels vs {len(values)} values"
        )

    # validate bar_colors if provided
    if bar_colors is not None:
        if len(bar_colors) != len(labels):
            raise ValueError(
                f"Length mismatch: {len(bar_colors)} colors vs {len(labels)} labels"
            )

    # sanitize inputs
    title_safe = sanitize_text(title)
    labels_safe = [sanitize_text(truncate_label(label)) for label in labels]

    # calculate dimensions
    padding = 60
    chart_width = width - 2 * padding
    chart_height = height - 2 * padding

    max_value = max(values) if values else 1
    min_value = min(values) if values else 0
    value_range = max_value - min_value if max_value != min_value else max_value

    # calculate bar dimensions
    num_bars = len(labels)
    bar_width = chart_width / (num_bars * 1.5)
    bar_spacing = bar_width * 0.5

    # start building SVG
    svg_parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
    ]

    # background
    svg_parts.append('  <rect width="100%" height="100%" fill="white"/>')

    # title
    svg_parts.append(
        f'  <text x="{width / 2}" y="30" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="16" font-weight="bold" '
        f'fill="{H2O_BLACK}">{title_safe}</text>'
    )

    # draw grid lines if enabled
    if show_grid:
        num_grid_lines = 5
        for i in range(num_grid_lines + 1):
            y = padding + (chart_height / num_grid_lines) * i
            grid_value = max_value - (value_range / num_grid_lines) * i

            # grid line
            svg_parts.append(
                f'  <line x1="{padding}" y1="{y}" x2="{width - padding}" y2="{y}" '
                f'stroke="{H2O_LIGHT_GRAY}" stroke-width="1" stroke-dasharray="2,2"/>'
            )

            # y-axis label
            svg_parts.append(
                f'  <text x="{padding - 10}" y="{y + 5}" text-anchor="end" '
                f'font-family="Arial, sans-serif" font-size="10" '
                f'fill="{H2O_GRAY}">{format_number(grid_value)}</text>'
            )

    # draw bars
    for i, (label, value) in enumerate(zip(labels_safe, values, strict=False)):
        x = padding + i * (bar_width + bar_spacing)

        # calculate bar height (handle negative values)
        if value_range > 0:
            bar_height = ((value - min_value) / value_range) * chart_height
        else:
            bar_height = chart_height * 0.5

        y = padding + chart_height - bar_height

        # determine bar color (use per-bar color if available, otherwise default)
        current_bar_color = bar_colors[i] if bar_colors else bar_color

        # bar rectangle
        svg_parts.append(
            f'  <rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" '
            f'fill="{current_bar_color}" stroke="{bar_border_color}" '
            f'stroke-width="1.5"/>'
        )

        # value label on top of bar
        if show_values:
            svg_parts.append(
                f'  <text x="{x + bar_width / 2}" y="{y - 5}" text-anchor="middle" '
                f'font-family="Arial, sans-serif" font-size="10" '
                f'fill="{H2O_BLACK}">{format_number(value)}</text>'
            )

        # x-axis label
        svg_parts.append(
            f'  <text x="{x + bar_width / 2}" y="{height - 20}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="10" '
            f'fill="{H2O_GRAY}">{label}</text>'
        )

    # x-axis line
    svg_parts.append(
        f'  <line x1="{padding}" y1="{padding + chart_height}" '
        f'x2="{width - padding}" y2="{padding + chart_height}" '
        f'stroke="{H2O_BLACK}" stroke-width="2"/>'
    )

    # y-axis line
    svg_parts.append(
        f'  <line x1="{padding}" y1="{padding}" '
        f'x2="{padding}" y2="{padding + chart_height}" '
        f'stroke="{H2O_BLACK}" stroke-width="2"/>'
    )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def generate_svg_horizontal_bar_chart(
    labels: list[str],
    values: list[float],
    title: str,
    width: int = 800,
    height: int = 400,
    show_values: bool = True,
    show_grid: bool = True,
    bar_color: str = H2O_YELLOW,
    bar_border_color: str = H2O_BLACK,
) -> str:
    """Generate horizontal bar chart as inline SVG.

    Horizontal bars are better for long label names as they have more space
    on the left side for labels.

    Parameters
    ----------
    labels : list[str]
        Labels for y-axis (one per bar).
    values : list[float]
        Values for each bar.
    title : str
        Chart title.
    width : int
        Chart width in pixels (default: 800).
    height : int
        Chart height in pixels (default: 400).
    show_values : bool
        Whether to show value labels on bars (default: True).
    show_grid : bool
        Whether to show vertical grid lines (default: True).
    bar_color : str
        Bar fill color (default: H2O yellow).
    bar_border_color : str
        Bar border color (default: H2O black).

    Returns
    -------
    str
        Complete SVG chart as string.

    Examples
    --------
    >>> labels = ["Long Metric Name A", "Long Metric Name B"]
    >>> values = [0.85, 0.92]
    >>> svg = generate_svg_horizontal_bar_chart(labels, values, "Metrics")
    >>> assert '<svg' in svg

    """
    if not labels or not values:
        return _generate_empty_chart(width, height, "No data to display")

    if len(labels) != len(values):
        raise ValueError(
            f"Length mismatch: {len(labels)} labels vs {len(values)} values"
        )

    # sanitize inputs
    title_safe = sanitize_text(title)
    labels_safe = [sanitize_text(label) for label in labels]

    # calculate dimensions
    padding_left = 180  # more space for labels
    padding_right = 40
    padding_top = 60
    padding_bottom = 40
    chart_width = width - padding_left - padding_right
    chart_height = height - padding_top - padding_bottom

    max_value = max(values) if values else 1

    # calculate bar dimensions
    num_bars = len(labels)
    bar_height = chart_height / (num_bars * 1.5)
    bar_spacing = bar_height * 0.5

    # start building SVG
    svg_parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
    ]

    # background
    svg_parts.append('  <rect width="100%" height="100%" fill="white"/>')

    # title
    svg_parts.append(
        f'  <text x="{width / 2}" y="30" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="16" font-weight="bold" '
        f'fill="{H2O_BLACK}">{title_safe}</text>'
    )

    # draw grid lines if enabled
    if show_grid:
        num_grid_lines = 5
        for i in range(num_grid_lines + 1):
            x = padding_left + (chart_width / num_grid_lines) * i
            grid_value = (max_value / num_grid_lines) * i

            # grid line
            svg_parts.append(
                f'  <line x1="{x}" y1="{padding_top}" '
                f'x2="{x}" y2="{padding_top + chart_height}" '
                f'stroke="{H2O_LIGHT_GRAY}" stroke-width="1" stroke-dasharray="2,2"/>'
            )

            # x-axis label
            svg_parts.append(
                f'  <text x="{x}" y="{padding_top + chart_height + 20}" '
                f'text-anchor="middle" font-family="Arial, sans-serif" '
                f'font-size="10" fill="{H2O_GRAY}">{format_number(grid_value)}</text>'
            )

    # draw bars
    for i, (label, value) in enumerate(zip(labels_safe, values, strict=False)):
        y = padding_top + i * (bar_height + bar_spacing)

        # calculate bar width
        bar_width_calc = (value / max_value) * chart_width if max_value > 0 else 0
        x = padding_left

        # bar rectangle
        svg_parts.append(
            f'  <rect x="{x}" y="{y}" width="{bar_width_calc}" height="{bar_height}" '
            f'fill="{bar_color}" stroke="{bar_border_color}" stroke-width="1.5"/>'
        )

        # value label at end of bar
        if show_values:
            label_x = x + bar_width_calc + 5
            svg_parts.append(
                f'  <text x="{label_x}" y="{y + bar_height / 2 + 4}" '
                f'text-anchor="start" font-family="Arial, sans-serif" '
                f'font-size="10" fill="{H2O_BLACK}">{format_number(value)}</text>'
            )

        # y-axis label (on left)
        svg_parts.append(
            f'  <text x="{padding_left - 10}" y="{y + bar_height / 2 + 4}" '
            f'text-anchor="end" font-family="Arial, sans-serif" '
            f'font-size="10" fill="{H2O_GRAY}">{label}</text>'
        )

    # x-axis line
    svg_parts.append(
        f'  <line x1="{padding_left}" y1="{padding_top + chart_height}" '
        f'x2="{padding_left + chart_width}" y2="{padding_top + chart_height}" '
        f'stroke="{H2O_BLACK}" stroke-width="2"/>'
    )

    # y-axis line
    svg_parts.append(
        f'  <line x1="{padding_left}" y1="{padding_top}" '
        f'x2="{padding_left}" y2="{padding_top + chart_height}" '
        f'stroke="{H2O_BLACK}" stroke-width="2"/>'
    )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def generate_svg_grouped_bar_chart(
    labels: list[str],
    baseline_values: list[float],
    current_values: list[float],
    title: str,
    baseline_label: str = "Baseline",
    current_label: str = "Current",
    width: int = 800,
    height: int = 400,
    show_values: bool = True,
    show_grid: bool = True,
    baseline_color: str = H2O_GRAY,
    current_color: str = H2O_YELLOW,
) -> str:
    """Generate grouped bar chart for baseline vs. current comparison.

    Parameters
    ----------
    labels : list[str]
        Labels for x-axis (one per group).
    baseline_values : list[float]
        Baseline values for each group.
    current_values : list[float]
        Current values for each group.
    title : str
        Chart title.
    baseline_label : str
        Legend label for baseline bars (default: "Baseline").
    current_label : str
        Legend label for current bars (default: "Current").
    width : int
        Chart width in pixels (default: 800).
    height : int
        Chart height in pixels (default: 400).
    show_values : bool
        Whether to show value labels on bars (default: True).
    show_grid : bool
        Whether to show horizontal grid lines (default: True).
    baseline_color : str
        Baseline bar color (default: H2O gray).
    current_color : str
        Current bar color (default: H2O yellow).

    Returns
    -------
    str
        Complete SVG chart as string.

    Examples
    --------
    >>> labels = ["Metric A", "Metric B"]
    >>> baseline = [0.5, 0.6]
    >>> current = [0.55, 0.65]
    >>> svg = generate_svg_grouped_bar_chart(
    ...     labels, baseline, current, "Comparison"
    ... )
    >>> assert '<svg' in svg

    """
    if not labels or not baseline_values or not current_values:
        return _generate_empty_chart(width, height, "No data to display")

    if len(labels) != len(baseline_values) or len(labels) != len(current_values):
        raise ValueError(
            f"Length mismatch: {len(labels)} labels, "
            f"{len(baseline_values)} baseline values, "
            f"{len(current_values)} current values"
        )

    # sanitize inputs
    title_safe = sanitize_text(title)
    labels_safe = [sanitize_text(truncate_label(label)) for label in labels]
    baseline_label_safe = sanitize_text(baseline_label)
    current_label_safe = sanitize_text(current_label)

    # calculate dimensions
    padding = 60
    legend_height = 40
    chart_width = width - 2 * padding
    chart_height = height - 2 * padding - legend_height

    # find max value across both datasets
    all_values = baseline_values + current_values
    max_value = max(all_values) if all_values else 1

    # calculate bar dimensions
    num_groups = len(labels)
    group_width = chart_width / (num_groups * 1.3)
    group_spacing = group_width * 0.3
    bar_width = group_width / 2.2

    # start building SVG
    svg_parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
    ]

    # background
    svg_parts.append('  <rect width="100%" height="100%" fill="white"/>')

    # title
    svg_parts.append(
        f'  <text x="{width / 2}" y="30" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="16" font-weight="bold" '
        f'fill="{H2O_BLACK}">{title_safe}</text>'
    )

    # draw grid lines if enabled
    if show_grid:
        num_grid_lines = 5
        for i in range(num_grid_lines + 1):
            y = padding + (chart_height / num_grid_lines) * i
            grid_value = max_value * (1 - i / num_grid_lines)

            # grid line
            svg_parts.append(
                f'  <line x1="{padding}" y1="{y}" x2="{width - padding}" y2="{y}" '
                f'stroke="{H2O_LIGHT_GRAY}" stroke-width="1" stroke-dasharray="2,2"/>'
            )

            # y-axis label
            svg_parts.append(
                f'  <text x="{padding - 10}" y="{y + 5}" text-anchor="end" '
                f'font-family="Arial, sans-serif" font-size="10" '
                f'fill="{H2O_GRAY}">{format_number(grid_value)}</text>'
            )

    # draw grouped bars
    for i, (label, baseline_val, current_val) in enumerate(
        zip(labels_safe, baseline_values, current_values, strict=False)
    ):
        group_x = padding + i * (group_width + group_spacing)

        # baseline bar
        baseline_height = (
            (baseline_val / max_value) * chart_height if max_value > 0 else 0
        )
        baseline_y = padding + chart_height - baseline_height

        svg_parts.append(
            f'  <rect x="{group_x}" y="{baseline_y}" '
            f'width="{bar_width}" height="{baseline_height}" '
            f'fill="{baseline_color}" stroke="{H2O_BLACK}" stroke-width="1.5"/>'
        )

        if show_values:
            svg_parts.append(
                f'  <text x="{group_x + bar_width / 2}" y="{baseline_y - 5}" '
                f'text-anchor="middle" font-family="Arial, sans-serif" '
                f'font-size="9" fill="{H2O_BLACK}">{format_number(baseline_val)}</text>'
            )

        # current bar
        current_x = group_x + bar_width * 1.1
        current_height = (
            (current_val / max_value) * chart_height if max_value > 0 else 0
        )
        current_y = padding + chart_height - current_height

        svg_parts.append(
            f'  <rect x="{current_x}" y="{current_y}" '
            f'width="{bar_width}" height="{current_height}" '
            f'fill="{current_color}" stroke="{H2O_BLACK}" stroke-width="1.5"/>'
        )

        if show_values:
            svg_parts.append(
                f'  <text x="{current_x + bar_width / 2}" y="{current_y - 5}" '
                f'text-anchor="middle" font-family="Arial, sans-serif" '
                f'font-size="9" fill="{H2O_BLACK}">{format_number(current_val)}</text>'
            )

        # group label
        label_x = group_x + group_width / 2
        svg_parts.append(
            f'  <text x="{label_x}" y="{height - legend_height - 5}" '
            f'text-anchor="middle" font-family="Arial, sans-serif" '
            f'font-size="10" fill="{H2O_GRAY}">{label}</text>'
        )

    # axes
    svg_parts.append(
        f'  <line x1="{padding}" y1="{padding + chart_height}" '
        f'x2="{width - padding}" y2="{padding + chart_height}" '
        f'stroke="{H2O_BLACK}" stroke-width="2"/>'
    )
    svg_parts.append(
        f'  <line x1="{padding}" y1="{padding}" '
        f'x2="{padding}" y2="{padding + chart_height}" '
        f'stroke="{H2O_BLACK}" stroke-width="2"/>'
    )

    # legend
    legend_y = height - legend_height / 2
    legend_x_start = width / 2 - 100

    # baseline legend
    svg_parts.append(
        f'  <rect x="{legend_x_start}" y="{legend_y - 8}" '
        f'width="16" height="16" fill="{baseline_color}" '
        f'stroke="{H2O_BLACK}" stroke-width="1"/>'
    )
    svg_parts.append(
        f'  <text x="{legend_x_start + 22}" y="{legend_y + 4}" '
        f'font-family="Arial, sans-serif" font-size="12" '
        f'fill="{H2O_BLACK}">{baseline_label_safe}</text>'
    )

    # current legend
    legend_x_current = legend_x_start + 120
    svg_parts.append(
        f'  <rect x="{legend_x_current}" y="{legend_y - 8}" '
        f'width="16" height="16" fill="{current_color}" '
        f'stroke="{H2O_BLACK}" stroke-width="1"/>'
    )
    svg_parts.append(
        f'  <text x="{legend_x_current + 22}" y="{legend_y + 4}" '
        f'font-family="Arial, sans-serif" font-size="12" '
        f'fill="{H2O_BLACK}">{current_label_safe}</text>'
    )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def _generate_empty_chart(width: int, height: int, message: str) -> str:
    """Generate empty chart with message.

    Parameters
    ----------
    width : int
        Chart width in pixels.
    height : int
        Chart height in pixels.
    message : str
        Message to display.

    Returns
    -------
    str
        SVG chart with message.

    """
    message_safe = sanitize_text(message)
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="14" '
        f'fill="{H2O_GRAY}">{message_safe}</text>'
        f"</svg>"
    )


def add_svg_chart_to_html(
    html, chart_svg: str, container_class: str = "chart-container"
) -> None:
    """Add SVG chart to airium HTML object.

    Parameters
    ----------
    html : airium.Airium
        Airium HTML object.
    chart_svg : str
        SVG chart string generated by one of the chart generation functions.
    container_class : str
        CSS class for container div (default: "chart-container").

    Examples
    --------
    >>> import airium
    >>> html = airium.Airium()
    >>> svg = generate_svg_bar_chart(["A"], [10], "Test")
    >>> add_svg_chart_to_html(html, svg)

    """
    with html.div(klass=container_class, style="text-align: center; margin: 20px 0;"):
        html(chart_svg)
