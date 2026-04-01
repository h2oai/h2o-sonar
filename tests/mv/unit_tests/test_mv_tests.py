# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import os.path

import datatable
import pytest

from h2o_sonar.lib.integrations import mv_adapter
from tests import test_utils


@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({mv_adapter.PACKAGE_NAME}),
    reason="H2O Model Validation Python package is not installed",
)
@pytest.mark.h2o_sonar
@pytest.mark.h2o_model_validation
def test_drift(tmp_path):
    from h2o_mv.core.mv_client import MVClient
    from h2o_mv.core.mv_database import DatabaseName
    from h2o_mv.core.mv_states import MVTestState
    from h2o_mv.recipes.drift import Drift
    from h2o_mv.recipes.drift import DriftSettings

    #
    # GIVEN
    client = MVClient(data_folder=tmp_path)
    client.select_database(name=DatabaseName.TEST)

    raw_dataset_path = "data/predictive/creditcard.csv"
    dataset_path = test_utils.find_locally(raw_dataset_path)
    another_dataset = datatable.fread(dataset_path)[:500, :]
    another_dataset_path = str(tmp_path / "another_dataset.csv")
    another_dataset.to_csv(another_dataset_path)

    assert os.path.isfile(dataset_path)
    assert os.path.isfile(another_dataset_path)

    # client-based initialization
    trn_ds = client.import_local_dataset(dataset_path)
    tst_ds = client.import_local_dataset(another_dataset_path)

    #
    # WHEN
    #
    mv_drift = Drift(name="mv_drift")
    mv_drift.settings = DriftSettings(primary_dataset=trn_ds, secondary_dataset=tst_ds)
    mv_drift.run()

    #
    # THEN
    #
    print(f"{mv_drift=}")
    print(f"{mv_drift.results=}")
    print(f"{mv_drift.results.psi_scores=}")
    assert mv_drift.results.psi_scores
    assert mv_drift.state == MVTestState.Done
    assert mv_drift.progress == 100
    client.delete_database()


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
