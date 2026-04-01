# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import os
import tempfile

import numpy as np
import pandas

from h2o_sonar.methods.core import _data
from h2o_sonar.methods.core._mli import MLI
from h2o_sonar.methods.surrogates._random_forest_h2o import RandomForestH2O
from tests.base_h2o_test import BaseH2OTest
from tests.conftest import get_h2o3_config
from tests.test_utils import find_locally
from tests.test_utils import rm_test_dir


class TestRandomForestH2O(BaseH2OTest):
    def test_params(self):
        tmp_dir = tempfile.mkdtemp(prefix="rf_h2o_")
        try:
            # GIVEN
            mli = MLI(work_dir=tmp_dir, seed=1234, config=get_h2o3_config())

            # Model as data
            model = mli.wrap(
                "test_model_data",
                data=_data.PersistedData(
                    find_locally("data/predictive/creditcard_with_preds.csv")
                ),
            )

            # WHEN
            rf = RandomForestH2O(max_depth=4, nfolds=3)

            # THEN
            # Check correct params are passed into DT and estimator
            assert rf.tree_parameters["nfolds"] == 3, (
                "nfolds for tree_parameters should be 3!"
            )
            assert rf.estimator._parms["nfolds"] == 3, (
                "nfolds for estimator._parms should be 3!"
            )
            assert rf.tree_parameters["max_depth"] == 4, (
                "max_depth for tree_parameters should be 4!"
            )
            assert rf.estimator._parms["max_depth"] == 4, (
                "max_depth for estimator._parms should be 4!"
            )

            rf.fit(model, response_column="p_DEFAULT_NEXT_MONTH")
        finally:
            rm_test_dir(tmp_dir)

    def test_basic_fit_predict(self):
        tmp_dir = tempfile.mkdtemp(prefix="rf_h2o_")
        try:
            # GIVEN
            mli = MLI(work_dir=tmp_dir, seed=1234, config=get_h2o3_config())

            # Model as data
            data = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
            labels = [1, 2, 3]
            model = mli.wrap("test_model_basic", data, labels)

            # WHEN
            rf = RandomForestH2O(max_depth=3, nfolds=0)
            rf.fit(model, response_column=3)
            predictions = rf.predict(data)

            # THEN
            np.testing.assert_almost_equal(predictions, labels, decimal=1)
        finally:
            rm_test_dir(tmp_dir)

    def test_fit_predict_pandas(self):
        tmp_dir = tempfile.mkdtemp(prefix="rf_surrogate_h2o_")
        try:
            # GIVEN
            mli = MLI(work_dir=tmp_dir, seed=1234, config=get_h2o3_config())

            # Model as data
            preds = [1, 2, 3]
            d = {
                "col1": [1, 2, 3],
                "col2": [2, 3, 4],
                "col3": [3, 4, 5],
                "preds": preds,
            }
            df = pandas.DataFrame(data=d)
            model = mli.wrap("test_model_pandas", df)

            # WHEN
            rf = RandomForestH2O(max_depth=3, nfolds=0)
            rf.fit(model, response_column="preds")
            predictions = rf.predict(df)

            # THEN
            np.testing.assert_almost_equal(predictions, preds, decimal=1)
        finally:
            rm_test_dir(tmp_dir)

    def test_fit_predict_numpy(self):
        tmp_dir = tempfile.mkdtemp(prefix="rf_surrogate_h2o_")
        try:
            # GIVEN
            mli = MLI(work_dir=tmp_dir, seed=1234, config=get_h2o3_config())
            # Model as data
            data = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
            preds = np.array([1, 2, 3])
            model = mli.wrap("test_model_data", data, preds)

            # WHEN
            rf_h2o = RandomForestH2O(max_depth=3, nfolds=0)
            rf_h2o.fit(model, response_column=3)
            predictions = rf_h2o.predict(data)

            # THEN
            np.testing.assert_almost_equal(predictions, preds, decimal=1)
        finally:
            rm_test_dir(tmp_dir)

    def test_fit_predict_disk(self):
        tmp_dir = tempfile.mkdtemp(prefix="rf_surrogate_h2o_")
        try:
            # GIVEN
            mli = MLI(work_dir=tmp_dir, seed=1234, config=get_h2o3_config())
            data = pandas.DataFrame(
                data={"C1": [1, 2, 3], "C2": [2, 3, 4], "C3": [3, 4, 5]}
            )
            preds = np.array([1, 2, 3])

            # Model as data
            model = mli.wrap(
                "test_model_data",
                data=_data.PersistedData(
                    find_locally("data/predictive/basic_dt_data.csv")
                ),
            )

            # WHEN
            rf = RandomForestH2O(max_depth=3, nfolds=0)
            rf.fit(model, response_column=3)
            predictions = rf.predict(data)

            # THEN
            np.testing.assert_almost_equal(predictions, preds, decimal=1)
        finally:
            rm_test_dir(tmp_dir)

    def test_fail_pred_before_fit(self):
        # GIVEN
        # Model as data

        # WHEN
        rf = RandomForestH2O(max_depth=3, nfolds=0)
        with self.assertRaises(ValueError):
            rf.predict(None)

    def test_fail_save_before_fit(self):
        # GIVEN
        # Model as data

        # WHEN
        rf = RandomForestH2O(max_depth=3, nfolds=0)
        with self.assertRaises(ValueError):
            rf.save()

    def test_download_mojo(self):
        tmp_dir = tempfile.mkdtemp(prefix="rf_surrogate_h2o_")
        try:
            # GIVEN
            mli = MLI(work_dir=tmp_dir, seed=1234, config=get_h2o3_config())

            # Model as data
            model = mli.wrap(
                "test_model_data",
                data=_data.PersistedData(
                    find_locally("data/predictive/basic_dt_data.csv")
                ),
            )

            # WHEN
            rf = RandomForestH2O(max_depth=3, nfolds=0)
            rf.fit(model, response_column=3)
            mojo_name = "rfsurr_mojo.zip"
            rf.save_mojo(path=tmp_dir + "/" + mojo_name)

            # THEN
            assert mojo_name in os.listdir(tmp_dir), (
                f"Expected "
                f"{mojo_name} in {tmp_dir} but got {os.listdir(tmp_dir)} in "
                f"{tmp_dir}"
            )
        finally:
            rm_test_dir(tmp_dir)
