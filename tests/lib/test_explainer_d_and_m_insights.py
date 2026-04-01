# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import logging
import os
import subprocess

import pytest

from h2o_sonar import interpret
from h2o_sonar.explainers import dataset_and_model_insights_explainer as explainer
from tests import test_utils
from tests.lib import test_containers


def _then_assert_interpretation(interpretation, should_return_problem_count: int):
    assert interpretation
    assert interpretation.is_explainer_scheduled()
    assert interpretation.is_explainer_finished()
    assert interpretation.is_explainer_successful()
    assert interpretation.get_scheduled_explainer_ids()
    assert interpretation.get_finished_explainer_ids()
    assert interpretation.get_successful_explainer_ids()

    failed_explainers = interpretation.get_failed_explainer_ids()
    assert not failed_explainers
    assert interpretation.get_explainer_result_metadata(
        explainer.DatasetAndModelInsightsExplainer().explainer_id()
    )
    assert interpretation.get_explainer_result(
        explainer.DatasetAndModelInsightsExplainer.explainer_id()
    ).summary()
    if should_return_problem_count > 0:
        assert interpretation.result.problems
    else:
        assert interpretation.result.problems == []

    assert len(interpretation.result.problems) == should_return_problem_count


@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "should_return_problem_count,dataset_path,target_col",
    [
        (2, "data/predictive/creditcard100_pred_missing_values.csv", "predictions"),
        (0, "data/predictive/creditcard100_pred.csv", "predictions"),
    ],
    ids=["creditcard100_pred_missing_values.csv", "creditcard100_pred.csv"],
)
def test_dataset_insights_mock_model_dataset_path(
    tmpdir, should_return_problem_count: int, dataset_path: str, target_col: str
):
    #
    # GIVEN
    #
    dataset_path = test_utils.find_locally(dataset_path)
    mock_model = test_containers.SimpleMockModel(
        dataset_path=dataset_path, target_col=target_col
    )

    #
    # WHEN
    #
    assert interpret.describe_explainer(explainer.DatasetAndModelInsightsExplainer)
    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        model=mock_model,
        target_col=target_col,
        explainers=[explainer.DatasetAndModelInsightsExplainer.explainer_id()],
        results_location=tmpdir,
        log_level=logging.DEBUG,
    )

    #
    # THEN
    #
    _then_assert_interpretation(interpretation, should_return_problem_count)


@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "should_return_problem_count,dataset_path,target_col",
    [
        (2, "data/predictive/creditcard100_pred_missing_values.csv", "predictions"),
    ],
)
def test_no_model(
    tmpdir, should_return_problem_count: int, dataset_path: str, target_col: str
):
    """Test interpretation WITHOUT the model (and target column and other model
    related options).

    """
    #
    # GIVEN
    #
    dataset_path = test_utils.find_locally(dataset_path)

    #
    # WHEN
    #
    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        # NO model
        model=None,
        # target column is typically needed for the analysis
        target_col="predictions",
        explainers=[explainer.DatasetAndModelInsightsExplainer.explainer_id()],
        results_location=tmpdir,
        log_level=logging.DEBUG,
    )

    #
    # THEN
    #
    _then_assert_interpretation(interpretation, should_return_problem_count)


@pytest.mark.h2o_sonar
@pytest.mark.cli
def test_no_model_cli(tmpdir):
    #
    # GIVEN
    #
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    target_col = "default payment next month"

    cli_cmd = (
        ["h2o-sonar"]
        if os.system("which h2o-sonar") == 0
        else ["python", "h2o_sonar/h2o_sonar_cli.py"]
    )
    child_env = os.environ.copy()
    # add the root of the repo to Python path so that the CLI can load model class
    child_env["PYTHONPATH"] = "."

    # WHEN
    cmd = cli_cmd + [
        "run",
        "interpretation",
        # model NOT specified X target column SPECIFIED as it is needed for the analysis
        "--dataset",
        dataset_path,
        "--target-col",
        target_col,
        "--results-location",
        tmpdir,
        "--explainers",
        explainer.DatasetAndModelInsightsExplainer.explainer_id(),
    ]

    print(f"\nRunning interpretation via CLI:\n{cmd}\n")
    p = subprocess.Popen(cmd, env=child_env)
    p.wait()

    # THEN
    p_fs = str(os.popen(f"find {tmpdir}").read())
    print(p_fs)
    assert "problems_and_actions.json" in p_fs


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
