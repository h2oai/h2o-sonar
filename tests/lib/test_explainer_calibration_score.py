# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import logging
import os
import subprocess

import pytest

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import interpret
from h2o_sonar.explainers import calibration_score_explainer as explainer
from h2o_sonar.lib.api import commons
from tests import test_utils
from tests.lib import given_generative
from tests.lib import test_cli as test_cli_test
from tests.lib import test_mv_export_import


# aliases
CalibScoreExplainer = explainer.CalibrationScoreExplainer


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.skip(
    reason=(
        "This test requires a running Driverless AI server with a prebuilt "
        "time series experiment"
    )
)
@pytest.mark.h2o_model_validation
@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "dai_connection,model_uuid,dataset_uuid,target_col",
    [
        # HARD-CODED experiment
        (
            given_generative.DAI_WORKER_CONNECTION,
            given_generative.DAI_EXPERIMENT_M.get("model"),
            given_generative.DAI_EXPERIMENT_M.get("dataset"),
            given_generative.DAI_EXPERIMENT_M.get("target_col"),
        ),
        # HARD-CODED CC multinomial experiment @ H2O AIEM
        (
            given_generative.AIEM_DAI_WORKER_CONNECTION,
            given_generative.AIEM_DAI_EXPERIMENT_M.get("model"),
            given_generative.AIEM_DAI_EXPERIMENT_M.get("dataset"),
            given_generative.AIEM_DAI_EXPERIMENT_M.get("target_col"),
        ),
    ],
    ids=["local_cc_multinomial", "aiem_cc_multinomial"],
)
def test_explainer(
    tmp_path,
    dai_connection: h2o_sonar_config.ConnectionConfig | None,
    model_uuid: str,
    dataset_uuid: str,
    target_col: str,
):
    """Test Calibration Score explainer via H2O Sonar Python API."""
    #
    # GIVEN
    #

    assert model_uuid
    assert dataset_uuid
    assert target_col

    # configure local DAI server connection
    h2o_sonar_config.config.add_connection(dai_connection)

    model_handle = commons.ResourceHandle(
        connection_key=dai_connection.key,
        resource_key=model_uuid,
    )
    dataset_handle = commons.ResourceHandle(
        connection_key=dai_connection.key,
        resource_key=dataset_uuid,
    )

    #
    # WHEN
    #
    interpretation = interpret.run_interpretation(
        dataset=str(dataset_handle),
        model=str(model_handle),
        target_col=target_col,
        explainers=[
            commons.ExplainerToRun(
                explainer_id=CalibScoreExplainer.explainer_id(),
                params={
                    CalibScoreExplainer.PARAM_WORKER: dai_connection.key,
                },
            )
        ],
        results_location=str(tmp_path),
        log_level=logging.DEBUG,
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
    assert not failed_explainers, f"Failed explainers: {failed_explainers}"

    # result
    result = interpretation.get_explainer_result(
        explainer.CalibrationScoreExplainer.explainer_id()
    )
    print(f"Explainer result: {result}")
    assert result
    print(f"Explainer result SUMMARY: {result.summary()}")
    assert result.summary()
    print(f"Explainer result DATA: {result.data()}")
    assert result.data()
    print("Explainer result PLOT...")
    plot_file_path = str(tmp_path / "plot.png")
    result.plot(file_path=plot_file_path)
    assert os.path.isfile(plot_file_path)
    print("Explainer result LOG...")
    result.log(path=str(tmp_path / "feature-importance-demo.log"))

    # HTML report
    test_utils.assert_html_report_images(interpretation)

    # MV result export/import
    zip_path = test_mv_export_import.get_mvresult_zip_path_for_interpretation(
        interpretation=interpretation,
        explainer_id=CalibScoreExplainer.explainer_id(),
        explainer_job_id=next(iter(interpretation.result.explainers.keys())),
    )
    test_mv_export_import.do_test_import_export(
        tmp_path=tmp_path,
        zip_or_dir_path=zip_path,
        raise_exception=True,
    )


@pytest.mark.skip(
    reason=(
        "This test requires a running Driverless AI server with a prebuilt "
        "time series experiment"
    )
)
@pytest.mark.h2o_model_validation
@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "model_uuid,dataset_uuid,target_col",
    [
        # HARD-CODED experiment
        (
            given_generative.DAI_EXPERIMENT_M.get("model"),
            given_generative.DAI_EXPERIMENT_M.get("dataset"),
            given_generative.DAI_EXPERIMENT_M.get("target_col"),
        ),
    ],
    ids=["local_cc_multinomial"],
)
def test_cli(
    tmp_path,
    model_uuid,
    dataset_uuid,
    target_col,
):
    """Test Calibration Score explainer via H2O Sonar CLI."""
    #
    # GIVEN
    #
    encryption_key = "TEST_ENCRYPTION_KEY"

    assert model_uuid
    assert dataset_uuid
    assert target_col

    # configure local DAI server connection
    h2o_sonar_config_path = test_cli_test.given_h2o_sonar_base_config(
        tmpdir=str(tmp_path)
    )
    connection_config_json = json.dumps(
        given_generative.DAI_WORKER_CONNECTION.to_dict(False)
    )

    model_handle = commons.ResourceHandle(
        connection_key=given_generative.DAI_WORKER_CONNECTION.key,
        resource_key=model_uuid,
    )
    dataset_handle = commons.ResourceHandle(
        connection_key=given_generative.DAI_WORKER_CONNECTION.key,
        resource_key=dataset_uuid,
    )

    (cmd, child_env) = test_cli_test.given_base_cli_cmd()
    cmd = cmd + [
        "add",
        "config",
        "--config-type",
        "CONNECTION",
        "--config-value",
        connection_config_json,
        "--config-path",
        h2o_sonar_config_path,
        "--encryption-key",
        encryption_key,
    ]
    print(f"\nRunning H2O Sonar to add DAI connection configuration:\n{cmd}\n")
    p = subprocess.Popen(cmd, env=child_env)
    p.wait()
    p_dump = str(os.popen(f"cat {h2o_sonar_config_path}").read())
    print(f"Config file:\n{p_dump}")
    assert given_generative.DAI_WORKER_CONNECTION.key in p_dump
    assert "encrypted" in p_dump

    explainer_pars = {
        CalibScoreExplainer.explainer_id(): {
            CalibScoreExplainer.PARAM_WORKER: given_generative.DAI_WORKER_CONNECTION.key
        }
    }

    #
    # WHEN
    #
    (cmd, child_env) = test_cli_test.given_base_cli_cmd()
    cmd = cmd + [
        "run",
        "interpretation",
        "--explainers",
        CalibScoreExplainer.explainer_id(),
        "--explainers-pars",
        str(explainer_pars),
        "--model",
        str(model_handle),
        "--dataset",
        str(dataset_handle),
        "--target-col",
        target_col,
        "--results-location",
        str(tmp_path),
        "--config-path",
        h2o_sonar_config_path,
        "--encryption-key",
        encryption_key,
    ]

    print(f"\nRunning interpretation via CLI:\n{cmd}\n")
    p = subprocess.Popen(cmd, env=child_env)
    p.wait()

    # THEN
    filesytem_tree_dump = str(os.popen(f"find {tmp_path}").read())
    print(filesytem_tree_dump)
    # assert files which must present in the explainer sandbox
    assert "brier_scores.csv" in filesytem_tree_dump


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
