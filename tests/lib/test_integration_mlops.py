# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import os
import subprocess

import pytest

from h2o_sonar.explainers import fi_naive_shapley_explainer as explainer
from tests import test_utils


# MLOps integration
#
# - REQUEST:
#     - interface: CLI
#     - explainers: feature importance explainer ONLY (do not run other explainers)
#     - dataset: .csv
#     - model: Driverless AI MOJO
# - RESPONSE:
#     - results to be pushed to specific REST interface (connector)
#


@pytest.mark.h2o_sonar
@pytest.mark.skipif(not test_utils.is_mojo_supported(), reason="MOJO is not supported")
def test_cli(tmpdir):
    # GIVEN
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    model_path = test_utils.find_locally(
        "data/predictive/models/creditcard-binomial.mojo"
    )
    target_column = "default payment next month"

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
        "--dataset",
        dataset_path,
        "--target-col",
        target_column,
        "--model",
        model_path,
        "--explainers",
        explainer.NaiveShapleyMojoFeatureImportanceExplainer.explainer_id(),
        "--results-location",
        tmpdir,
    ]
    print(f"\nRunning interpretation via CLI:\n{cmd}\n")
    p = subprocess.Popen(cmd, env=child_env)
    p.wait()

    # THEN
    p_tree = str(os.popen(f"find {tmpdir}").read())
    print(p_tree)
    # .
    # ├── h2o-sonar
    # │   └── mli_experiment_e16866f1-23a2-4ab9-88ea-a3c6166a46c9
    # │       ├── explainer_..._NaiveShapleyMojoFeatureImportanceExplainer_116...1be5
    # │       │   ├── global_feature_importance
    # │       │   │   ├── application_json
    # │       │   │   │   ├── explanation.json
    # │       │   │   │   └── feature_importance_class_0.json
    # │       │   │   ├── application_json.meta
    # │       │   │   ├── application_vnd_h2oai_json_csv
    # │       │   │   │   ├── explanation.json
    # │       │   │   │   └── feature_importance_class_0.csv
    # │       │   │   ├── application_vnd_h2oai_json_csv.meta
    # │       │   │   ├── application_vnd_h2oai_json_datatable_jay
    # │       │   │   │   ├── explanation.json
    # │       │   │   │   └── feature_importance_class_0.jay
    # │       │   │   └── application_vnd_h2oai_json_datatable_jay.meta
    # │       │   ├── local_feature_importance
    # │       │   │   ├── application_vnd_h2oai_json_datatable_jay
    # │       │   │   │   ├── explanation.json
    # │       │   │   │   ├── feature_importance_class_0.jay
    # │       │   │   │   └── y_hat.bin
    # │       │   │   └── application_vnd_h2oai_json_datatable_jay.meta
    # │       │   ├── log
    # │       │   │   └── explainer_run_116fb9e3-cc75-4e4a-9a6e-08af24b11be5.log
    # │       │   ├── result_descriptor.json
    # │       │   └── work
    # │       │       ├── shapley_formatted_orig_feat.zip
    # │       │       ├── shapley.orig.feat.bin
    # │       │       ├── shapley.orig.feat.csv
    # │       │       └── y_hat.bin
    # │       ├── interpretation.html
    # │       └── interpretation.json
    # └── h2o-sonar.log
    # FI
    assert "feature_importance_class_0.json" in p_tree


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
