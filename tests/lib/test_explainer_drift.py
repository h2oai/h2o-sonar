# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import logging
import os
import subprocess

import datatable
import pytest

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import interpret
from h2o_sonar.explainers import drift_explainer
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.integrations import mv_adapter
from h2o_sonar.utils import io
from tests import test_utils
from tests.lib import given_generative
from tests.lib import test_mv_export_import


# aliases
DriftExplainer = drift_explainer.DriftDetectionExplainer


def _given_another_dataset(tmp_path, dataset_path):
    # get another dataset by randomization/sub-setting
    another_dataset = datatable.fread(dataset_path)[:500, :]
    another_dataset_path = str(tmp_path / "another_dataset.csv")
    another_dataset.to_csv(another_dataset_path)

    return another_dataset_path


@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({mv_adapter.PACKAGE_NAME}),
    reason="H2O Model Validation Python package is not installed",
)
@pytest.mark.h2o_sonar
@pytest.mark.h2o_model_validation
@pytest.mark.parametrize(
    "dai_connection,dataset_path,another_dataset_path,target_col",
    [
        # LOCAL dataset
        (
            None,
            "data/predictive/creditcard.csv",
            "data/predictive/creditcard.csv",
            "default payment next month",
        ),
        # REMOTE dataset
        pytest.param(
            given_generative.DAI_WORKER_CONNECTION,
            given_generative.DAI_EXPERIMENT_TS.get("dataset"),
            given_generative.DAI_EXPERIMENT_TS.get("testset"),
            given_generative.DAI_EXPERIMENT_TS.get("target_col"),
            marks=pytest.mark.skipif(
                True, reason="This test requires a DAI server w/ dataset w/ known UUID"
            ),
        ),
        # HARD-CODED Walmart time-series experiment @ H2O AIEM
        pytest.param(
            given_generative.AIEM_DAI_WORKER_CONNECTION,
            given_generative.AIEM_DAI_EXPERIMENT_TS.get("dataset"),
            given_generative.AIEM_DAI_EXPERIMENT_TS.get("dataset"),
            given_generative.AIEM_DAI_EXPERIMENT_TS.get("target_col"),
            marks=pytest.mark.skipif(
                True, reason="This test requires a DAI server w/ dataset w/ known UUID"
            ),
        ),
    ],
)
def test_explainer(
    tmp_path,
    dai_connection: h2o_sonar_config.ConnectionConfig | None,
    dataset_path,
    another_dataset_path,
    target_col,
):
    #
    # GIVEN
    #
    # NO model
    # dataset
    if dai_connection is None:
        dataset = test_utils.find_locally(dataset_path)
        another_dataset = _given_another_dataset(tmp_path, dataset_path)
        explainer_to_run = commons.ExplainerToRun(
            explainer_id=DriftExplainer.explainer_id(),
            params={
                DriftExplainer.PARAM_DRIFT_THRESHOLD: 0.05,
            },
        )
    else:
        # configure DAI server connection
        h2o_sonar_config.config.add_connection(dai_connection)

        # save the configuration for debugging purposes
        h2o_sonar_config.config.save(
            config_path=str(tmp_path / "h2o-sonar-config.json"),
            encryption_key="m1-s3cr3t-k3y",
        )

        dataset = commons.ResourceHandle(
            connection_key=dai_connection.key,
            resource_key=dataset_path,
        )
        another_dataset = commons.ResourceHandle(
            connection_key=dai_connection.key,
            resource_key=another_dataset_path,
        )
        explainer_to_run = commons.ExplainerToRun(
            explainer_id=DriftExplainer.explainer_id(),
            params={
                DriftExplainer.PARAM_DRIFT_THRESHOLD: 0.05,
                DriftExplainer.PARAM_WORKER: dai_connection.key,
            },
        )
    # interpretation parameters as JSon for debugging purposes
    args_as_json_str = io.to_run_interpretation_args_json(
        dataset=str(dataset),
        testset=str(another_dataset),
        target_col=target_col,
        explainers=[explainer_to_run],
        results_location=str(tmp_path),
        log_level=logging.DEBUG,
    )
    print(f"Interpretation parameters as JSon:\n{args_as_json_str}")

    # container
    container = interpret.resolve_container()

    #
    # WHEN
    #
    interpretation = interpret.run_interpretation(
        dataset=dataset,
        testset=another_dataset,
        model=None,
        target_col=target_col,
        explainers=[explainer_to_run],
        results_location=str(tmp_path),
        log_level=logging.DEBUG,
        container=container,
    )

    #
    # THEN
    #
    print(f"\n{interpretation}")
    # find failed explainers
    assert interpretation
    assert interpretation.result.explainers
    assert len(interpretation.result.explainers) == 1
    failed_explainers = interpretation.get_failed_explainer_ids()
    if dai_connection is None and failed_explainers:
        # dump debugging details
        filesytem_tree_dump = str(os.popen(f"find {tmp_path}").read())
        print(filesytem_tree_dump)
        # lookup H2O Sonar log
        for line in filesytem_tree_dump.splitlines():
            if "h2o-sonar.log" in line:
                print(line)
                h2o_sonar_log_content = str(os.popen(f"cat {line}").read())
                print(f"= {line} ===")
                print(h2o_sonar_log_content)
                print(f"^ {line} ^^^")
    assert not failed_explainers, f"Failed explainers: {failed_explainers}"

    # result
    result = interpretation.get_explainer_result(DriftExplainer.explainer_id())
    print(f"Explainer result: {result}")
    assert result
    print(f"Explainer result SUMMARY: {result.summary()}")
    assert result.summary()
    print(f"Explainer result DATA: {result.data().to_dict()}")
    assert result.data()
    print("Explainer result PLOT...")
    result.plot()
    print("Explainer result LOG...")
    result.log(path=str(tmp_path / "feature-importance-demo.log"))

    # HTML report
    test_utils.assert_html_report_images(interpretation)

    # MV result export/import
    zip_path = test_mv_export_import.get_mvresult_zip_path_for_interpretation(
        interpretation=interpretation,
        explainer_id=DriftExplainer.explainer_id(),
        explainer_job_id=next(iter(interpretation.result.explainers.keys())),
    )
    test_mv_export_import.do_test_import_export(
        tmp_path=tmp_path,
        zip_or_dir_path=zip_path,
        raise_exception=True,
    )

    return interpretation


@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({mv_adapter.PACKAGE_NAME}),
    reason="H2O Model Validation Python package is not installed",
)
@pytest.mark.parametrize(
    "dataset_path,target_col",
    [
        # LOCAL dataset
        (
            "data/predictive/creditcard.csv",
            "default payment next month",
        ),
    ],
)
@pytest.mark.h2o_sonar
def test_cli(tmp_path, dataset_path: str, target_col: str):
    """Test Drift explainer via H2O Sonar CLI."""
    #
    # GIVEN
    #
    dataset_path = test_utils.find_locally(dataset_path)
    another_dataset_path = _given_another_dataset(tmp_path, dataset_path)

    #
    # WHEN
    #
    (cmd, child_env) = test_utils.given_base_cli_cmd()
    cmd = cmd + [
        "run",
        "interpretation",
        "--explainers",
        DriftExplainer.explainer_id(),
        "--dataset",
        dataset_path,
        "--testset",
        another_dataset_path,
        "--target-col",
        target_col,
        "--results-location",
        str(tmp_path),
    ]

    print(f"\nRunning interpretation via CLI:\n{cmd}\n")
    p = subprocess.Popen(cmd, env=child_env)
    p.wait()

    # THEN
    filesytem_tree_dump = str(os.popen(f"find {tmp_path}").read())
    print(filesytem_tree_dump)
    # DIA
    assert "feature_importance_class_0.jay" in filesytem_tree_dump


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
