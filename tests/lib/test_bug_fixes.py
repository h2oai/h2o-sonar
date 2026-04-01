# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import html as html_lib

import airium
import pandas as pd
import pytest

from h2o_sonar import interpret
from h2o_sonar import loggers
from h2o_sonar.explainers import dt_surrogate_explainer as dt_surrogate
from h2o_sonar.lib.api import commons
from tests.lib import test_containers


def test_bug_mli_2_606(tmpdir):
    # GIVEN
    dataset_path = "./data/predictive/creditcard.csv"
    target_col = "default payment next month"
    df = pd.read_csv(dataset_path)
    model = test_containers.SimpleMockModel(
        dataset_path=dataset_path, target_col=target_col
    )

    # WHEN
    interpretation = interpret.run_interpretation(
        dataset=df,
        model=model,
        target_col=target_col,
        results_location=str(tmpdir),
        log_level=loggers.DEBUG,
        explainers=[
            commons.ExplainerToRun(
                explainer_id=dt_surrogate.DecisionTreeSurrogateExplainer.explainer_id(),
                params="",
            )
        ],
    )

    # THEN
    assert interpretation
    print(f"{interpretation.key=}")
    assert interpretation.key


def test_bug_airium():
    # GIVEN
    html = airium.Airium()
    html("<!DOCTYPE html>")
    with html.html(lang="en"):
        with html.head():
            html.meta(charset="utf-8")
            html.title(_t="Interpretation Report")

        with html.body():
            with html.h1():
                html("H2O Sonar Model Interpretation Report")

        # HTML escaping
        html(html_lib.escape("<class ABC>"))

    # WHEN
    html_str = str(html)

    # THEN
    print(html_str)
    assert "<class" not in html_str


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
