# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
from functools import partial
from unittest import TestCase

import numpy as np
import pandas as pd

from h2o_sonar import loggers as logging
from h2o_sonar.methods._pd import PD
from tests.methods.ice_pd_test_commons import FooScorerRegrSeries
from tests.methods.ice_pd_test_commons import FooScorerSumRegrFrame
from tests.methods.ice_pd_test_commons import IceStrategyFactory


class TestPdQuantile(TestCase):
    """Test Partial Dependency (PD) implementation."""

    def setUp(self):
        logging.setLevel(logging.DEBUG)

        self.strategy = IceStrategyFactory()

        # data
        self.df2x2 = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
        # data: 3x3 for value checks
        self.df3x3 = pd.DataFrame({"f1": [1, 2, 3], "F": [1, 3, 5], "f2": [0.5, 2, 3]})
        # data: 1x5 dataframe
        self.df1x5 = pd.DataFrame([[i for i in range(1, 6)]], columns=list("abFdY"))
        # data: 5x5 dataframe w/ i..i+5 rows, F is target feature
        self.df5x5 = pd.DataFrame(
            [[i + j for i in range(1, 6)] for j in range(0, 5)],
            columns=list("abFdY"),
        )

        # prediction method lambda
        self.score_foo = partial(FooScorerRegrSeries().score_batch, fast_approx=True)
        self.score_sum = partial(FooScorerSumRegrFrame().score_batch, fast_approx=True)

        # test persistence in the current directory (avoid making garbage)
        self.test_current_dir_persistence = True
        # visual check
        self.test_visual_check = False

    def test_quantile_binning_subset_cols(self):
        # GIVEN
        X = pd.DataFrame()
        X["num1"] = np.random.randint(1000, 1000000, 20000)
        X["num2"] = np.random.randint(1000, 1000000, 20000)
        X["num3"] = np.random.randint(1000, 1000000, 20000)
        meta = {PD.KEY_QUANTILE_BINS: {"num1": 10, "num2": 4}}

        # WHEN
        json = (
            self.strategy.get_pd()
            .explain(
                ["num1", "num2", "num3"],
                X,
                predict_method=self.score_foo,
                features_meta=meta,
            )
            .to_json()
        )

        # THEN
        self.assertIsNotNone(json)
        logging.debug(f"PD JSon:\n{json}")
        self.assertTrue(PD.JSON_PD_DATA in json)
        self.assertTrue(json[PD.JSON_PD_DATA])
        json_num1_bins = json[PD.JSON_PD_DATA][0]["bins"]
        json_num2_bins = json[PD.JSON_PD_DATA][1]["bins"]
        self.assertEqual(10, len(json_num1_bins))
        self.assertEqual(4, len(json_num2_bins))

    def test_quantile_binning_all_cols(self):
        # GIVEN
        X = pd.DataFrame()
        X["num1"] = np.random.randint(1000, 1000000, 20000)
        X["num2"] = np.random.randint(1000, 1000000, 20000)
        X["num3"] = np.random.randint(1000, 1000000, 20000)
        meta = {PD.KEY_QUANTILE_BINS: {"num1": 10, "num2": 4, "num3": 3}}

        # WHEN
        json = (
            self.strategy.get_pd()
            .explain(
                ["num1", "num2", "num3"],
                X,
                predict_method=self.score_foo,
                features_meta=meta,
            )
            .to_json()
        )

        # THEN
        self.assertIsNotNone(json)
        logging.debug(f"PD JSon:\n{json}")
        self.assertTrue(PD.JSON_PD_DATA in json)
        self.assertTrue(json[PD.JSON_PD_DATA])
        json_num1_bins = json[PD.JSON_PD_DATA][0]["bins"]
        json_num2_bins = json[PD.JSON_PD_DATA][1]["bins"]
        json_num3_bins = json[PD.JSON_PD_DATA][2]["bins"]
        self.assertEqual(10, len(json_num1_bins))
        self.assertEqual(4, len(json_num2_bins))
        self.assertEqual(3, len(json_num3_bins))

    def test_quantile_binning_pass_single_none(self):
        # GIVEN
        X = pd.DataFrame()
        X["num1"] = np.random.randint(1000, 1000000, 20000)
        X["num2"] = np.random.randint(1000, 1000000, 20000)
        X["num3"] = np.random.randint(1000, 1000000, 20000)
        meta = {PD.KEY_QUANTILE_BINS: {"num1": 10, "num2": None, "num3": 3}}

        # WHEN
        json = (
            self.strategy.get_pd()
            .explain(
                ["num1", "num2", "num3"],
                X,
                predict_method=self.score_foo,
                features_meta=meta,
            )
            .to_json()
        )

        # THEN
        self.assertIsNotNone(json)
        logging.debug(f"PD JSon:\n{json}")
        self.assertTrue(PD.JSON_PD_DATA in json)
        self.assertTrue(json[PD.JSON_PD_DATA])
        json_num1_bins = json[PD.JSON_PD_DATA][0]["bins"]
        json_num2_bins = json[PD.JSON_PD_DATA][1]["bins"]
        json_num3_bins = json[PD.JSON_PD_DATA][2]["bins"]
        self.assertEqual(10, len(json_num1_bins))
        self.assertEqual(10, len(json_num2_bins))
        self.assertEqual(3, len(json_num3_bins))

    def test_quantile_binning_pass_all_none(self):
        # GIVEN
        X = pd.DataFrame()
        X["num1"] = np.random.randint(1000, 1000000, 20000)
        X["num2"] = np.random.randint(1000, 1000000, 20000)
        X["num3"] = np.random.randint(1000, 1000000, 20000)
        meta = {PD.KEY_QUANTILE_BINS: {"num1": None, "num2": None, "num3": None}}

        # WHEN
        json = (
            self.strategy.get_pd()
            .explain(
                ["num1", "num2", "num3"],
                X,
                predict_method=self.score_foo,
                features_meta=meta,
            )
            .to_json()
        )

        # THEN
        self.assertIsNotNone(json)
        logging.debug(f"PD JSon:\n{json}")
        self.assertTrue(PD.JSON_PD_DATA in json)
        self.assertTrue(json[PD.JSON_PD_DATA])
        json_num1_bins = json[PD.JSON_PD_DATA][0]["bins"]
        json_num2_bins = json[PD.JSON_PD_DATA][1]["bins"]
        json_num3_bins = json[PD.JSON_PD_DATA][2]["bins"]
        self.assertEqual(10, len(json_num1_bins))
        self.assertEqual(10, len(json_num2_bins))
        self.assertEqual(10, len(json_num3_bins))

    def test_quantile_binning_create_dict_single_quantile(self):
        # GIVEN
        X = pd.DataFrame()
        X["num1"] = np.random.randint(1000, 1000000, 20000)
        X["num2"] = np.random.randint(1000, 1000000, 20000)
        X["num3"] = np.random.randint(1000, 1000000, 20000)
        # This mimics an API call that can be done from a FE -> BE, e.g., DAI
        features = list(X.columns)  # FE provides list of features
        quantile = [10] * len(features)  # FE provides an int, which represents
        # the quantile value, which will be constant
        # across all features
        feature_dict = dict(zip(features, quantile, strict=False))
        meta = {PD.KEY_QUANTILE_BINS: feature_dict}

        # WHEN
        json = (
            self.strategy.get_pd()
            .explain(
                ["num1", "num2", "num3"],
                X,
                predict_method=self.score_foo,
                features_meta=meta,
            )
            .to_json()
        )

        # THEN
        self.assertIsNotNone(json)
        logging.debug(f"PD JSon:\n{json}")
        self.assertTrue(PD.JSON_PD_DATA in json)
        self.assertTrue(json[PD.JSON_PD_DATA])
        json_num1_bins = json[PD.JSON_PD_DATA][0]["bins"]
        json_num2_bins = json[PD.JSON_PD_DATA][1]["bins"]
        json_num3_bins = json[PD.JSON_PD_DATA][2]["bins"]
        self.assertEqual(10, len(json_num1_bins))
        self.assertEqual(10, len(json_num2_bins))
        self.assertEqual(10, len(json_num3_bins))

    def test_quantile_binning_backward_compatability(self):
        # GIVEN
        X = pd.DataFrame()
        X["num1"] = np.random.randint(1000, 1000000, 20000)
        X["num2"] = np.random.randint(1000, 1000000, 20000)
        X["num3"] = np.random.randint(1000, 1000000, 20000)

        meta = {PD.KEY_QUANTILE_BINS: ["num1", "num2", "num3"]}

        # WHEN
        json = (
            self.strategy.get_pd()
            .explain(
                ["num1", "num2", "num3"],
                X,
                predict_method=self.score_foo,
                features_meta=meta,
            )
            .to_json()
        )

        # THEN
        self.assertIsNotNone(json)
        logging.debug(f"PD JSon:\n{json}")
        self.assertTrue(PD.JSON_PD_DATA in json)
        self.assertTrue(json[PD.JSON_PD_DATA])
        json_num1_bins = json[PD.JSON_PD_DATA][0]["bins"]
        json_num2_bins = json[PD.JSON_PD_DATA][1]["bins"]
        json_num3_bins = json[PD.JSON_PD_DATA][2]["bins"]
        self.assertEqual(10, len(json_num1_bins))
        self.assertEqual(10, len(json_num2_bins))
        self.assertEqual(10, len(json_num3_bins))
