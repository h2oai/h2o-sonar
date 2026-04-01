# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import os
import tempfile

import datatable
import numpy as np
import pandas
import pytest

from h2o_sonar.config import H2o3Config
from h2o_sonar.methods.core import _data
from h2o_sonar.methods.core._mli import MLI
from h2o_sonar.methods.core._mli import MLIDataBackend
from h2o_sonar.methods.surrogates._decision_tree_h2o import DecisionTreeH2O
from tests.base_h2o_test import BaseH2OTest
from tests.conftest import get_h2o3_config
from tests.test_utils import find_locally
from tests.test_utils import GitHubActions
from tests.test_utils import rm_test_dir


try:
    import h2o

    HAS_H2O = True
except ImportError:
    HAS_H2O = False


CAT_ENCODING = "onehotexplicit"


class TestDecisionTreeH2O(BaseH2OTest):
    def test_params(self):
        tmp_dir = tempfile.mkdtemp(prefix="dt_h2o_")
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
            dt = DecisionTreeH2O(
                max_depth=4, nfolds=3, categorical_encoding=CAT_ENCODING
            )

            # THEN
            # Check correct params are passed into DT and estimator
            assert dt.tree_parameters["nfolds"] == 3, (
                "nfolds for tree_parameters should be 3!"
            )
            assert dt.estimator._parms["nfolds"] == 3, (
                "nfolds for estimator._parms should be 3!"
            )
            assert dt.tree_parameters["max_depth"] == 4, (
                "max_depth for tree_parameters should be 4!"
            )
            assert dt.estimator._parms["max_depth"] == 4, (
                "max_depth for estimator._parms should be 4!"
            )
            assert dt.estimator._parms["categorical_encoding"] == CAT_ENCODING, (
                f"categorical_encoding for estimator._parms should be {CAT_ENCODING}"
            )

            dt.fit(model, response_column="p_DEFAULT_NEXT_MONTH")
        finally:
            rm_test_dir(tmp_dir)

    def test_basic_fit_predict(self):
        tmp_dir = tempfile.mkdtemp(prefix="dt_h2o_")
        try:
            # GIVEN
            mli = MLI(work_dir=tmp_dir, seed=1234, config=get_h2o3_config())

            # Model as data
            data = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
            labels = [1, 2, 3]
            model = mli.wrap("test_model_basic", data, labels)

            # WHEN
            dt = DecisionTreeH2O(max_depth=3, nfolds=0)
            dt.fit(model, response_column=3)
            predictions = dt.predict(data)

            # THEN
            np.testing.assert_almost_equal(predictions, labels, decimal=1)
        finally:
            rm_test_dir(tmp_dir)

    def test_fit_predict_pandas(self):
        tmp_dir = tempfile.mkdtemp(prefix="dt_surrogate_h2o_")
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
            dt = DecisionTreeH2O(max_depth=3, nfolds=0)
            dt.fit(model, response_column="preds")
            predictions = dt.predict(df)

            # THEN
            np.testing.assert_almost_equal(predictions, preds, decimal=1)
        finally:
            rm_test_dir(tmp_dir)

    def test_fit_predict_datatable(self):
        tmp_dir = tempfile.mkdtemp(prefix="dt_surrogate_h2o_")
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
            df = datatable.Frame(pandas.DataFrame(data=d))
            model = mli.wrap(
                "test_model_pandas", df, data_backend=MLIDataBackend.DATATABLE
            )

            # WHEN
            dt = DecisionTreeH2O(max_depth=3, nfolds=0)
            dt.fit(model, response_column="preds")
            predictions = dt.predict(df)

            # THEN
            np.testing.assert_almost_equal(predictions, preds, decimal=1)
        finally:
            rm_test_dir(tmp_dir)

    def test_fit_predict_pandas_weights(self):
        tmp_dir = tempfile.mkdtemp(prefix="dt_surrogate_h2o_")
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
            dt = DecisionTreeH2O(max_depth=3, nfolds=0)
            dt.fit(model, response_column="preds", weights_column="col3")
            predictions = dt.predict(df)

            # THEN
            np.testing.assert_almost_equal(predictions, preds, decimal=1)
        finally:
            rm_test_dir(tmp_dir)

    def test_fit_predict_datatable_weights(self):
        tmp_dir = tempfile.mkdtemp(prefix="dt_surrogate_h2o_")
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
            df = datatable.Frame(pandas.DataFrame(data=d))
            model = mli.wrap(
                "test_model_pandas", df, data_backend=MLIDataBackend.DATATABLE
            )

            # WHEN
            dt = DecisionTreeH2O(max_depth=3, nfolds=0)
            dt.fit(model, response_column="preds", weights_column="col3")
            predictions = dt.predict(df)

            # THEN
            np.testing.assert_almost_equal(predictions, preds, decimal=1)
        finally:
            rm_test_dir(tmp_dir)

    def test_fit_predict_numpy(self):
        tmp_dir = tempfile.mkdtemp(prefix="dt_surrogate_h2o_")
        try:
            # GIVEN
            mli = MLI(work_dir=tmp_dir, seed=1234, config=get_h2o3_config())
            # Model as data
            data = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
            preds = np.array([1, 2, 3])
            model = mli.wrap("test_model_data", data, preds)

            # WHEN
            dt_h2o = DecisionTreeH2O(max_depth=3, nfolds=0)
            dt_h2o.fit(model, response_column=3)
            predictions = dt_h2o.predict(data)

            # THEN
            np.testing.assert_almost_equal(predictions, preds, decimal=1)
        finally:
            rm_test_dir(tmp_dir)

    def test_fit_predict_disk(self):
        tmp_dir = tempfile.mkdtemp(prefix="dt_surrogate_h2o_")
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
            dt = DecisionTreeH2O(max_depth=3, nfolds=0)
            dt.fit(model, response_column=3)
            predictions = dt.predict(data)

            # THEN
            np.testing.assert_almost_equal(predictions, preds, decimal=1)
        finally:
            rm_test_dir(tmp_dir)

    def test_fail_pred_before_fit(self):
        # GIVEN
        # Model as data

        # WHEN
        dt = DecisionTreeH2O(max_depth=3, nfolds=0)
        with self.assertRaises(ValueError):
            dt.predict(None)

    def test_fail_save_before_fit(self):
        # GIVEN
        # Model as data

        # WHEN
        dt = DecisionTreeH2O(max_depth=3, nfolds=0)
        with self.assertRaises(ValueError):
            dt.save()

    def test_h2o_params(self):
        mli = MLI(config=get_h2o3_config())

        # Model as data
        data = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        labels = [1, 2, 3]
        model = mli.wrap("test_model_basic", data, labels)

        # WHEN
        dt = DecisionTreeH2O(max_depth=3, nfolds=0)
        dt.fit(model, response_column=3)

        # THEN
        # verify that get_h2o3_config() returns expected memory settings
        config = get_h2o3_config()
        self.assertIn(H2o3Config.KEY_MIN_MEM_SIZE, config)
        self.assertIn(H2o3Config.KEY_MAX_MEM_SIZE, config)

    def test_download_mojo(self):
        tmp_dir = tempfile.mkdtemp(prefix="dt_surrogate_h2o_")
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
            dt = DecisionTreeH2O(max_depth=3, nfolds=0)
            dt.fit(model, response_column=3)
            mojo_name = "dtsurr_mojo.zip"
            dt.save_mojo(path=tmp_dir + "/" + mojo_name)

            # THEN
            assert mojo_name in os.listdir(tmp_dir), (
                f"Expected "
                f"{mojo_name} in {tmp_dir} but got {os.listdir(tmp_dir)} in "
                f"{tmp_dir}"
            )
        finally:
            rm_test_dir(tmp_dir)

    def test_download_model_details(self):
        tmp_dir = tempfile.mkdtemp(prefix="dt_surrogate_h2o_")
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
            dt = DecisionTreeH2O(max_depth=3, nfolds=0)
            dt.fit(
                model, response_column=3, validation_frame=model.data_as_model
            )  # Use train as validation set to get metrics for train (work around)
            model_name = "dtModel.json"

            # Load model details
            dt.load_model_details()

            # Save model details
            dt.save_model_details(path=tmp_dir)

            # THEN
            assert model_name in os.listdir(tmp_dir), (
                f"Expected "
                f"{model_name} in {tmp_dir} but got {os.listdir(tmp_dir)} in "
                f"{tmp_dir}"
            )
        finally:
            rm_test_dir(tmp_dir)

    def test_download_dt_paths_frame(self):
        tmp_dir = tempfile.mkdtemp(prefix="dt_surrogate_h2o_")
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
            dt = DecisionTreeH2O(max_depth=3, nfolds=0)
            dt.fit(model, response_column=3)
            dt_paths_frame_name = "dtPathsFrame.csv"
            dt.save_dt_paths_frame(
                input_df=model.data_as_model,
                path=tmp_dir + "/" + dt_paths_frame_name,
            )

            # THEN
            assert dt_paths_frame_name in os.listdir(tmp_dir), (
                f"Expected "
                f"{dt_paths_frame_name} in {tmp_dir} but got {os.listdir(tmp_dir)} "
                f"in {tmp_dir}"
            )
        finally:
            rm_test_dir(tmp_dir)

    @pytest.mark.skipif(
        not HAS_H2O,
        reason="ML interpret Python package is not installed",
    )
    @pytest.mark.skipif(
        GitHubActions.is_in_gha(),
        reason=(
            "Skipped on GHA as this test has high memory usage (ran on MMC/self-hosted)"
        ),
    )
    def test_issue_500(self):
        tmp_dir = tempfile.mkdtemp(prefix="dt_surrogate_h2o_")
        try:
            # GIVEN
            mli = MLI(work_dir=tmp_dir, seed=1234, config=get_h2o3_config())

            # Model as data
            model = mli.wrap(
                "test_model_data",
                data=_data.PersistedData("data/predictive/dry_bean_dataset.csv"),
            )

            # WHEN

            dt = DecisionTreeH2O(
                max_depth=3,
                nfolds=0,
                ignored_columns=["Area", "Perimeter", "ConvexArea"],
            )
            valid_frame = h2o.import_file("data/predictive/dry_bean_dataset.csv")
            dt.fit(model, response_column="Class", validation_frame=valid_frame)
            dt_paths_frame_name = "dtSurrogate.json"

            # Load JSON tree
            dt.load_dt_tree_json()

            # Save JSON tree
            dt.save_dt_tree_json(path=tmp_dir)

            # THEN
            assert dt_paths_frame_name in os.listdir(tmp_dir), (
                f"Expected "
                f"{dt_paths_frame_name} in {tmp_dir} but got {os.listdir(tmp_dir)} "
                f"in {tmp_dir}"
            )
        finally:
            rm_test_dir(tmp_dir)

    def test_download_dt_tree_json(self):
        tmp_dir = tempfile.mkdtemp(prefix="dt_surrogate_h2o_")
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
            dt = DecisionTreeH2O(max_depth=3, nfolds=0)
            dt.fit(model, response_column=3)
            dt_paths_frame_name = "dtSurrogate.json"

            # Load JSON tree
            dt.load_dt_tree_json()

            # Save JSON tree
            dt.save_dt_tree_json(path=tmp_dir)

            # THEN
            assert dt_paths_frame_name in os.listdir(tmp_dir), (
                f"Expected "
                f"{dt_paths_frame_name} in {tmp_dir} but got {os.listdir(tmp_dir)} "
                f"in {tmp_dir}"
            )
        finally:
            rm_test_dir(tmp_dir)

    def test_const_col(self):
        tmp_dir = tempfile.mkdtemp(prefix="dt_surrogate_h2o_")
        try:
            # GIVEN
            mli = MLI(work_dir=tmp_dir, seed=1234, config=get_h2o3_config())

            # Model as data
            model = mli.wrap(
                "test_model_data",
                data=_data.PersistedData(
                    find_locally("data/predictive/creditcard_const_target.csv")
                ),
            )

            # WHEN
            dt = DecisionTreeH2O(max_depth=3, nfolds=0)
            dt.fit(model, response_column=3)

            # THEN
            # Main goal of this test is to ensure a constant response column does not
            # throw an exception ...
        finally:
            rm_test_dir(tmp_dir)
