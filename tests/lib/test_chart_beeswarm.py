# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import pandas
import pytest
from matplotlib import pyplot

from h2o_sonar.lib.api import plots
from tests import test_utils


@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "csv_path",
    [
        "data/predictive/shap-wages-regression.csv",
        "data/predictive/shap-cc-binary-classification.csv",
    ],
)
def test_beeswarm_from_shap_values_csv(tmp_path, csv_path):
    """Load pre-computed SHAP values from CSV and render a beeswarm chart to PNG.

    The CSV contains both raw feature values and their corresponding SHAP
    contributions.  Feature columns are plain names (e.g. ``EDUCATION``);
    SHAP columns carry a ``_shap`` suffix (e.g. ``EDUCATION_shap``).
    ``base_value`` and ``WAGE`` are metadata columns that are excluded from
    the chart.
    """
    #
    # GIVEN - load SHAP values CSV and split into contributions / frame
    #
    path = test_utils.find_locally(csv_path)
    df = pandas.read_csv(path)

    shap_cols = [c for c in df.columns if c.endswith("_shap")]
    feature_names = [c.removesuffix("_shap") for c in shap_cols]

    contributions = df[shap_cols].copy()
    contributions.columns = feature_names

    frame = df[feature_names].copy()

    #
    # WHEN - render beeswarm chart from SHAP contributions
    #
    fig = plots.ScatterFeatImpPlot.plot(
        contributions=contributions,
        frame=frame,
        chart_title="SHAP Feature Importance",
        hard_asserts=True,
    )

    png_path = tmp_path / "beeswarm.png"
    fig.savefig(str(png_path), dpi=120, bbox_inches="tight")
    pyplot.close(fig)

    #
    # THEN - verify a non-empty PNG was written
    #
    print(f"\nBeeswarm chart saved to: file://{png_path}")
    assert png_path.exists(), f"PNG file not created at {png_path}"
    assert png_path.stat().st_size > 0, "PNG file is empty"
    print(f"PNG size: {png_path.stat().st_size:,} bytes")
    print("DONE")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
