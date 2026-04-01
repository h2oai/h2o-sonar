# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import logging
import os
import subprocess

import pytest

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import interpret
from h2o_sonar.explainers import dia_explainer as explainer
from tests import test_utils
from tests.lib import test_explainer_dia


ENCRYPTION_KEY = "m1-s3cr3t-k3y"


@pytest.mark.skipif(not test_utils.is_mojo_supported(), reason="MOJO is not supported")
@pytest.mark.parametrize(
    "dataset,target_col",
    [
        pytest.param(
            "pd_ice_creditcard_train.csv",
            "LIMIT_BAL",
            marks=pytest.mark.skipif(
                test_utils.health.get_h2ogpte() is None
                or not test_utils.health.get_h2ogpte().token,
                reason="API key not available",
            ),
        ),
    ],
)
@pytest.mark.h2o_sonar
def test_report_h2ogpte_upload(tmp_path, dataset, target_col):
    #
    # GIVEN
    #

    # configure h2oGPTe server connection
    h2o_sonar_config.config.add_connection(test_utils.health.get_h2ogpte())

    # save the configuration for debugging purposes
    h2o_sonar_config.config.save(
        config_path=str(tmp_path / "h2o-sonar-config.json"),
        encryption_key=ENCRYPTION_KEY,
    )

    dataset_path, explainable_model, target_col = test_explainer_dia.setup_dai(
        dataset=dataset, classification=False
    )

    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        model=explainable_model,
        target_col=target_col,
        results_location=tmp_path,
        log_level=logging.DEBUG,
        explainers=[explainer.DiaExplainer.explainer_id()],
    )
    html_report_path = interpretation.result.get_html_report_location()
    print(f"HTML report path: {html_report_path}")
    assert html_report_path
    pdf_report_path = interpretation.result.get_pdf_report_location()
    print(f"PDF report path: {pdf_report_path}")

    #
    # WHEN
    #
    (up_id, up_url) = interpret.upload_interpretation(
        interpretation_result=interpretation,
        connection=test_utils.health.get_h2ogpte().key,
    )

    #
    # THEN
    #
    assert up_id
    print(up_id)
    assert up_url
    print(up_url)


@pytest.mark.skipif(not test_utils.is_mojo_supported(), reason="MOJO is not supported")
@pytest.mark.parametrize(
    "dataset,target_col",
    [
        pytest.param(
            "pd_ice_creditcard_train.csv",
            "LIMIT_BAL",
            marks=pytest.mark.skipif(
                test_utils.health.get_h2ogpte() is None
                or not test_utils.health.get_h2ogpte().token,
                reason="API key not available",
            ),
        ),
    ],
)
@pytest.mark.h2o_sonar
def test_interpretation_with_h2ogpte_upload(tmp_path, dataset, target_col):
    #
    # GIVEN
    #

    # configure h2oGPT E server connection
    h2o_sonar_config.config.add_connection(test_utils.health.get_h2ogpte())

    # save the configuration for debugging purposes
    h2o_sonar_config.config.save(
        config_path=str(tmp_path / "h2o-sonar-config.json"),
        encrypt=True,
        encryption_key=ENCRYPTION_KEY,
    )

    dataset_path, explainable_model, target_col = test_explainer_dia.setup_dai(
        dataset=dataset, classification=False
    )

    #
    # WHEN
    #
    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        model=explainable_model,
        target_col=target_col,
        results_location=tmp_path,
        log_level=logging.DEBUG,
        explainers=[explainer.DiaExplainer.explainer_id()],
        upload_to=h2o_sonar_config.config.get_connection(
            connection_key=test_utils.health.get_h2ogpte().key
        ),
    )

    #
    # THEN
    #
    assert interpretation.result.upload_url
    print(
        f"TEST: interpretation report uploaded to:\n"
        f"  {interpretation.result.upload_url}"
    )


@pytest.mark.skipif(
    not test_utils.is_mojo_supported(),
    reason="MOJO is not supported on this platform",
)
@pytest.mark.skipif(
    not test_utils.health.is_h2ogpte(),
    reason="h2oGPTE service is not reachable",
)
@pytest.mark.h2o_sonar
def test_cli(tmp_path):
    """Test Drift explainer via H2O Sonar CLI."""
    #
    # GIVEN
    #
    # configure h2oGPT E server connection
    h2o_sonar_config.config.add_connection(test_utils.health.get_h2ogpte())
    h2o_sonar_config_path = tmp_path / "h2o-sonar-config.json"
    h2o_sonar_config.config.save(
        config_path=str(tmp_path / h2o_sonar_config_path),
        encrypt=True,
        encryption_key=ENCRYPTION_KEY,
    )
    print(
        json.dumps(
            h2o_sonar_config.config.to_dict(
                encrypt=True, encryption_key=ENCRYPTION_KEY
            ),
            indent=4,
        )
    )

    dataset_path = test_utils.find_locally("data/predictive/creditcard_str_10k.csv")
    model_path = test_utils.find_locally(
        "data/predictive/models/creditcard-regression-str.mojo"
    )
    target_col = "LIMIT_BAL"

    #
    # WHEN: interpretation w/ upload
    #
    (cmd, child_env) = test_utils.given_base_cli_cmd()
    cmd = cmd + [
        "run",
        "interpretation",
        "--explainers",
        explainer.DiaExplainer.explainer_id(),
        "--dataset",
        dataset_path,
        "--model",
        model_path,
        "--target-col",
        target_col,
        "--results-location",
        str(tmp_path),
        "--upload-to",
        test_utils.health.get_h2ogpte().key,
        "--config-path",
        str(h2o_sonar_config_path),
        "--encryption-key",
        ENCRYPTION_KEY,
    ]

    print(f"\nRunning interpretation via CLI:\n{cmd}\n")
    p = subprocess.Popen(cmd, env=child_env)
    p.wait()

    #
    # THEN
    #
    filesytem_tree_dump = str(os.popen(f"find {tmp_path}").read())
    print(filesytem_tree_dump)
    # DIA
    assert "dia-0-n.png" in filesytem_tree_dump

    #
    # WHEN: upload only
    #
    for ll in filesytem_tree_dump.split("\n"):
        if ll.endswith("interpretation.html"):
            (cmd, child_env) = test_utils.given_base_cli_cmd()
            cmd = cmd + [
                "upload",
                "interpretation",
                "--interpretation",
                ll,
                "--upload-to",
                test_utils.health.get_h2ogpte().key,
                "--config-path",
                str(h2o_sonar_config_path),
                "--encryption-key",
                ENCRYPTION_KEY,
            ]

            print(f"\nRunning interpretation via CLI:\n{cmd}\n")
            p = subprocess.Popen(cmd, env=child_env)
            p.wait()


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
