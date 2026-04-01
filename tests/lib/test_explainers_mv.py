# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import pytest

from h2o_sonar.lib.api import commons
from tests import test_utils
from tests.lib import given_generative
from tests.lib import test_explainer_adversarial_similarity
from tests.lib import test_explainer_backtesting
from tests.lib import test_explainer_calibration_score
from tests.lib import test_explainer_drift
from tests.lib import test_explainer_segment_performance
from tests.lib import test_explainer_size_dependency


@pytest.mark.skip(
    reason=(
        "Test which runs all MV explainers for documentations and demo purposes. "
        "It might be disabled as individual tests are run anyway + it requires "
        "pre-build Driverless AI experiments."
    )
)
@pytest.mark.skipif(
    not test_utils.is_local_dai_running(), reason="Driverless AI server is not running"
)
@pytest.mark.skipif(not test_utils.is_mojo_supported(), reason="MOJO is not supported")
@pytest.mark.h2o_sonar
@pytest.mark.h2o_model_validation
@pytest.mark.parametrize(
    (
        "experiment_type,mojo_path,dataset_path,target_col,"
        "ts_model_uuid,ts_dataset_uuid,ts_testset_uuid,ts_target_col,ts_time_col,"
        "m_model_uuid,m_dataset_uuid,m_target_col"
    ),
    [
        (
            commons.ExperimentType.binomial,
            "data/predictive/models/creditcard-binomial.mojo",
            "data/predictive/creditcard.csv",
            "default payment next month",
            given_generative.DAI_EXPERIMENT_TS.get("model"),
            given_generative.DAI_EXPERIMENT_TS.get("dataset"),
            given_generative.DAI_EXPERIMENT_TS.get("testset"),
            given_generative.DAI_EXPERIMENT_TS.get("target_col"),
            given_generative.DAI_EXPERIMENT_TS.get("time_col"),
            given_generative.DAI_EXPERIMENT_M.get("model"),
            given_generative.DAI_EXPERIMENT_M.get("dataset"),
            given_generative.DAI_EXPERIMENT_M.get("target_col"),
        ),
    ],
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
def test_explainers(
    tmp_path,
    experiment_type,
    mojo_path,
    dataset_path,
    target_col,
    ts_model_uuid,
    ts_dataset_uuid,
    ts_testset_uuid,
    ts_target_col,
    ts_time_col,
    m_model_uuid,
    m_dataset_uuid,
    m_target_col,
):
    # explainers are ordered from the fastest to the slowest

    test_explainer_drift.test_explainer(
        dai_connection=None,
        tmp_path=tmp_path,
        dataset_path=dataset_path,
        another_dataset_path=dataset_path,
        target_col=target_col,
        local_artifacts=True,
    )
    test_explainer_backtesting.test_explainer(
        tmp_path=tmp_path,
        dai_connection=given_generative.DAI_WORKER_CONNECTION,
        model_uuid=ts_model_uuid,
        dataset_uuid=ts_dataset_uuid,
        target_col=ts_target_col,
        time_col=ts_time_col,
    )
    test_explainer_calibration_score.test_explainer(
        tmp_path=tmp_path,
        dai_connection=given_generative.DAI_WORKER_CONNECTION,
        model_uuid=m_model_uuid,
        dataset_uuid=m_dataset_uuid,
        target_col=m_target_col,
    )
    test_explainer_segment_performance.test_explainer(
        tmp_path=tmp_path,
        dai_connection=given_generative.DAI_WORKER_CONNECTION,
        model_uuid=ts_model_uuid,
        dataset_uuid=ts_dataset_uuid,
        testset_uuid=ts_testset_uuid,
        target_col=ts_target_col,
    )
    test_explainer_size_dependency.test_explainer(
        tmp_path=tmp_path,
        dai_connection=given_generative.DAI_WORKER_CONNECTION,
        model_uuid=ts_model_uuid,
        dataset_uuid=ts_dataset_uuid,
        testset_uuid=ts_testset_uuid,
        target_col=ts_target_col,
        time_col=ts_time_col,
    )
    test_explainer_adversarial_similarity.test_explainer(
        tmp_path=tmp_path,
        dai_connection=None,
        dataset_path=dataset_path,
        another_dataset_path=dataset_path,
        target_col=target_col,
    )


@pytest.mark.skip(reason="Test which is used to debug MV Jupyter Notebook.")
@pytest.mark.h2o_sonar
@pytest.mark.h2o_model_validation
def test_mv_jupyter_notebook():
    #
    # GIVEN
    #
    from h2o_sonar import interpret
    from h2o_sonar.explainers.backtesting_explainer import BacktestingExplainer
    from h2o_sonar.explainers.segment_performance_explainer import (
        SegmentPerformanceExplainer,
    )
    from h2o_sonar.explainers.size_dependency_explainer import SizeDependencyExplainer

    dai_worker_connection = given_generative.DAI_WORKER_CONNECTION

    model_uuid = "b78cb888-f658-11ed-9ecf-0242709d15f7"
    dataset_uuid = "a407dd4c-f658-11ed-9ecf-0242709d15f7"
    testset_uuid = "a4077500-f658-11ed-9ecf-0242709d15f7"
    target_col = "Weekly_Sales"
    time_col = "Date"

    # 2) use HANDLERs to reference ^ datasets and model to be explained
    model_handle = commons.ResourceHandle(
        connection_key=dai_worker_connection.key,
        resource_key=model_uuid,
    )
    dataset_handle = commons.ResourceHandle(
        connection_key=dai_worker_connection.key,
        resource_key=dataset_uuid,
    )
    testset_handle = commons.ResourceHandle(
        connection_key=dai_worker_connection.key,
        resource_key=testset_uuid,
    )

    #
    # WHEN
    #
    interpretation = interpret.run_interpretation(
        dataset=dataset_handle,
        testset=testset_handle,
        model=model_handle,
        target_col=target_col,
        # schedule for run all MV explainers + specify parameters they need
        explainers=[
            commons.ExplainerToRun(
                explainer_id=BacktestingExplainer.explainer_id(),
                params={
                    BacktestingExplainer.PARAM_WORKER: dai_worker_connection.key,
                    BacktestingExplainer.PARAM_TIME_COLUMN: time_col,
                },
            ),
            commons.ExplainerToRun(
                explainer_id=SizeDependencyExplainer.explainer_id(),
                params={
                    SizeDependencyExplainer.PARAM_WORKER: dai_worker_connection.key,
                    SizeDependencyExplainer.PARAM_TIME_COLUMN: time_col,
                },
            ),
            commons.ExplainerToRun(
                explainer_id=SegmentPerformanceExplainer.explainer_id(),
                params={
                    SegmentPerformanceExplainer.PARAM_WORKER: dai_worker_connection.key,
                },
            ),
        ],
        results_location="results-all-mv-explainers",
    )

    #
    # THEN
    #
    assert interpretation


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
