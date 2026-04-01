# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import logging
import os
import subprocess
from typing import Any

import datatable
import pytest

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import interpret
from h2o_sonar.explainers import adversarial_similarity_explainer as explainer
from h2o_sonar.lib.api import commons
from tests import test_utils
from tests.lib import given_generative as g_i
from tests.lib import test_cli as test_of_cli
from tests.lib import test_mv_export_import


# aliases
AdversarialExplainer = explainer.AdversarialSimilarityExplainer


def _given_artifacts(dataset_path: str, tmp_path) -> tuple[str, str, Any]:
    # container
    container = interpret.resolve_container()

    # primary dataset
    primary_dataset_path = test_utils.find_locally(dataset_path)

    # secondary dataset by randomization/sub-setting
    secondary_dataset = datatable.fread(dataset_path)[:500, :]
    secondary_dataset_path = str(tmp_path / "another_dataset.csv")
    secondary_dataset.to_csv(secondary_dataset_path)

    return primary_dataset_path, secondary_dataset_path, container


@pytest.mark.skipif(
    not test_utils.is_local_dai_running(), reason="Driverless AI server is not running"
)
@pytest.mark.h2o_model_validation
@pytest.mark.h2o_sonar
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
            g_i.DAI_WORKER_CONNECTION,
            g_i.DAI_EXPERIMENT_TS.get("dataset"),
            g_i.DAI_EXPERIMENT_TS.get("testset"),
            g_i.DAI_EXPERIMENT_TS.get("target_col"),
            marks=pytest.mark.skipif(
                True, reason="This test requires a DAI server w/ dataset w/ known UUID"
            ),
        ),
        # HARD-CODED Walmart time-series experiment @ H2O AIEM
        pytest.param(
            g_i.AIEM_DAI_WORKER_CONNECTION,
            g_i.AIEM_DAI_EXPERIMENT_TS.get("dataset"),
            g_i.AIEM_DAI_EXPERIMENT_TS.get("testset"),
            g_i.AIEM_DAI_EXPERIMENT_TS.get("target_col"),
            marks=pytest.mark.skipif(
                True, reason="This test requires a DAI server w/ dataset w/ known UUID"
            ),
        ),
    ],
    ids=["LOCAL", "REMOTE", "AIEM"],
)
def test_explainer(
    tmp_path,
    dai_connection: h2o_sonar_config.ConnectionConfig | None,
    dataset_path: str,
    another_dataset_path: str,
    target_col: str,
):
    """Test Adversarial Similarity explainer via H2O Sonar Python API."""
    #
    # GIVEN
    #

    # configure local DAI server connection
    h2o_sonar_config.config.add_connection(dai_connection)

    # artifacts
    # NO model
    # dataset
    if dai_connection is None:
        (
            primary_dataset,
            secondary_dataset,
            # NO model
            container,
        ) = _given_artifacts(dataset_path, tmp_path)
    else:
        primary_dataset = commons.ResourceHandle(
            connection_key=dai_connection.key,
            resource_key=dataset_path,
        )
        secondary_dataset = commons.ResourceHandle(
            connection_key=dai_connection.key,
            resource_key=another_dataset_path,
        )
        container = None

    #
    # WHEN
    #
    interpretation = interpret.run_interpretation(
        dataset=primary_dataset,
        testset=secondary_dataset,
        model=None,
        target_col=target_col,
        explainers=[
            commons.ExplainerToRun(
                explainer_id=AdversarialExplainer.explainer_id(),
                params={
                    AdversarialExplainer.PARAM_WORKER: dai_connection.key,
                    # enable Shapleys to force create of the MV artifact(s)
                    AdversarialExplainer.PARAM_SHAPLEY_VALUES: True,
                },
            )
        ],
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
    assert not failed_explainers, f"Failed explainers: {failed_explainers}"

    # result
    result = interpretation.get_explainer_result(
        explainer.AdversarialSimilarityExplainer.explainer_id()
    )
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
        explainer_id=AdversarialExplainer.explainer_id(),
        explainer_job_id=next(iter(interpretation.result.explainers.keys())),
    )
    test_mv_export_import.do_test_import_export(
        tmp_path=tmp_path,
        zip_or_dir_path=zip_path,
        raise_exception=True,
    )


@pytest.mark.skipif(
    not test_utils.is_local_dai_running(), reason="Driverless AI server is not running"
)
@pytest.mark.h2o_model_validation
@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "dataset_path,target_col",
    [
        # LOCAL dataset(s)
        (
            "data/predictive/creditcard.csv",
            "default payment next month",
        ),
    ],
)
def test_cli(tmp_path, dataset_path, target_col):
    """Test Adversarial Similarity explainer via H2O Sonar CLI."""
    #
    # GIVEN
    #
    encryption_key = "TEST_ENCRYPTION_KEY"

    h2o_sonar_config_path = test_of_cli.given_h2o_sonar_base_config(
        tmpdir=str(tmp_path),
        h2o_auto_start=True,
        custom_explainers=[],
    )

    # configure local DAI server connection
    connection_config_json = json.dumps(g_i.DAI_WORKER_CONNECTION.to_dict(False))
    (cmd, child_env) = test_utils.given_base_cli_cmd()
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
    assert g_i.DAI_WORKER_CONNECTION.key in p_dump
    assert "encrypted" in p_dump

    explainer_pars = {
        AdversarialExplainer.explainer_id(): {
            AdversarialExplainer.PARAM_WORKER: g_i.DAI_WORKER_CONNECTION.key
        }
    }

    # artifacts
    (
        primary_dataset_path,
        secondary_dataset_path,
        # NO model
        container,
    ) = _given_artifacts(dataset_path, tmp_path)

    #
    # WHEN
    #
    (cmd, child_env) = test_utils.given_base_cli_cmd()
    cmd = cmd + [
        "run",
        "interpretation",
        "--explainers",
        AdversarialExplainer.explainer_id(),
        "--explainers-pars",
        str(explainer_pars),
        "--dataset",
        primary_dataset_path,
        "--testset",
        secondary_dataset_path,
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
    assert (
        "global_grouped_bar_chart/application_vnd_h2oai_json_datatable_jay"
        "/feature_importance_class_0.jay"
    ) in filesytem_tree_dump


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
