# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import unittest
from functools import partial

import pandas as pd
from datatable import fread

from h2o_sonar import loggers as logging
from h2o_sonar.methods._h_statistic import HStatistic
from h2o_sonar.methods._pd import PD
from h2o_sonar.methods.core.method import Method
from tests.methods.ice_pd_test_commons import DATASET_DATE_BINNING_PATH
from tests.methods.ice_pd_test_commons import FooScorerRegrSeries
from tests.test_utils import find_locally


class TestDateAwareBinning(unittest.TestCase):
    """Test date-aware binning for PD, ICE and H-statistic."""

    # override
    def setUp(self):
        logging.setLevel(logging.DEBUG)

    @staticmethod
    def _given_pandas(path: str):
        df = pd.read_csv(find_locally(path), index_col=0)
        logging.debug(f"Data ({len(df.index)}):\n{df.to_string()}")
        return df

    @staticmethod
    def _given_datatable(path: str):
        df = fread(file=find_locally(path))
        logging.debug(f"Data:\n{df.names}\n{df.ltypes}")
        return df

    def test_negative_init(self):
        df = self._given_datatable(DATASET_DATE_BINNING_PATH)

        with self.assertRaises(ValueError):
            Method.create_date_aware_bins(None, df)
        with self.assertRaises(ValueError):
            Method.create_date_aware_bins([], df)
        with self.assertRaises(ValueError):
            Method.create_date_aware_bins(["ID"], None)
        with self.assertRaises(ValueError):
            Method.create_date_aware_bins(["ID"], df, None, -1)
        with self.assertRaises(ValueError):
            Method.create_date_aware_bins(["ID"], df, None, 5, -1)
        with self.assertRaises(ValueError):
            Method.create_date_aware_bins(["ID"], df, None, 5, 3, None)

    def test_binning_int(self):
        # GIVEN
        df = self._given_datatable(DATASET_DATE_BINNING_PATH)

        # WHEN
        bins, oor_bins = Method.create_date_aware_bins(
            ["ID", "INT_DATE"], df, features_meta={"date": "INT_DATE"}
        )

        # THEN
        logging.debug(f"Date bins    : {bins}")
        self.assertEqual(1, len(bins))
        self.assertEqual(Method.DEFAULT_GRID_RESOLUTION, len(bins[0]))
        self.assertIsInstance(bins[0][0], int)
        logging.debug(f"OOR Date bins: {oor_bins}")
        self.assertIsNone(oor_bins)

    def test_binning_str(self):
        # GIVEN
        df = self._given_datatable(DATASET_DATE_BINNING_PATH)

        # WHEN
        bins, oor_bins = Method.create_date_aware_bins(
            ["ID", "STR_DATE"],
            df,
            features_meta={"date": "STR_DATE"},
            date_format="%Y-%m-%d",
        )

        # THEN
        logging.debug(f"Date bins    : {bins}")
        self.assertEqual(1, len(bins))
        self.assertEqual(Method.DEFAULT_GRID_RESOLUTION, len(bins[0]))
        self.assertIsInstance(bins[0][0], str)
        logging.debug(f"OOR Date bins: {oor_bins}")
        self.assertIsNone(oor_bins)

    def test_binning_multiple_features(self):
        # GIVEN
        df = self._given_datatable(DATASET_DATE_BINNING_PATH)

        # WHEN
        bins, oor_bins = Method.create_date_aware_bins(
            ["ID", "INT_DATE", "STR_DATE"],
            df,
            features_meta={"date": ["INT_DATE", "STR_DATE"]},
            date_format=["%Y%m%d", "%Y-%m-%d"],
        )

        # THEN
        logging.debug(f"Date bins    : {bins}")
        self.assertEqual(2, len(bins))
        self.assertEqual(Method.DEFAULT_GRID_RESOLUTION, len(bins[0]))
        self.assertIsInstance(bins[0][0], int)
        self.assertIsInstance(bins[1][0], str)
        logging.debug(f"OOR Date bins: {oor_bins}")
        self.assertIsNone(oor_bins)

    def test_binning_nometa(self):
        # GIVEN
        df = self._given_datatable(DATASET_DATE_BINNING_PATH)

        # WHEN
        bins, oor_bins = Method.create_date_aware_bins(
            ["INT_DATE", "STR_DATE"], df, date_format=["%Y%m%d", "%Y-%m-%d"]
        )

        # THEN
        logging.debug(f"Date bins    : {bins}")
        self.assertEqual(2, len(bins))
        self.assertEqual(Method.DEFAULT_GRID_RESOLUTION, len(bins[0]))
        self.assertIsInstance(bins[0][0], int)
        self.assertIsInstance(bins[1][0], str)
        logging.debug(f"OOR Date bins: {oor_bins}")
        self.assertIsNone(oor_bins)

    def test_oor_binning_int(self):
        # GIVEN
        df = self._given_datatable(DATASET_DATE_BINNING_PATH)
        oor_resolution = 3

        # WHEN
        bins, oor_bins = Method.create_date_aware_bins(
            ["INT_DATE"], df, out_of_range_resolution=oor_resolution
        )

        # THEN
        logging.debug(f"Date bins    : {bins}")
        self.assertEqual(1, len(bins))
        self.assertEqual(Method.DEFAULT_GRID_RESOLUTION, len(bins[0]))
        logging.debug(f"OOR Date bins: {oor_bins}")
        self.assertEqual(1, len(oor_bins))
        self.assertEqual(2 * oor_resolution, len(oor_bins[0]))
        self.assertIsInstance(oor_bins[0][0], int)

    def test_oor_binning_multiple_features(self):
        # GIVEN
        df = self._given_datatable(DATASET_DATE_BINNING_PATH)
        oor_resolution = 3

        # WHEN
        bins, oor_bins = Method.create_date_aware_bins(
            ["ID", "INT_DATE", "STR_DATE"],
            df,
            features_meta={"date": ["INT_DATE", "STR_DATE"]},
            out_of_range_resolution=oor_resolution,
            date_format=["%Y%m%d", "%Y-%m-%d"],
        )

        # THEN
        logging.debug(f"Date bins    : {bins}")
        self.assertEqual(2, len(bins))
        self.assertEqual(Method.DEFAULT_GRID_RESOLUTION, len(bins[0]))
        logging.debug(f"OOR Date bins: {oor_bins}")
        self.assertEqual(2, len(oor_bins))
        self.assertEqual(2 * oor_resolution, len(oor_bins[0]))
        self.assertIsInstance(oor_bins[0][0], int)
        self.assertIsInstance(oor_bins[1][0], str)

    def test_pd_int(self):
        # GIVEN
        df3x3 = pd.DataFrame(
            {
                "f1": [1, 2, 3],
                "D": [20160101, 20180505, 20191212],
                "f2": [1, 2, 3],
            }
        )
        features = ["D"]
        score_foo = partial(FooScorerRegrSeries().score_batch, fast_approx=True)

        # WHEN
        ice = PD("I").explain(
            features,
            df3x3,
            predict_method=score_foo,
            features_meta={
                Method.KEY_DATE_FEATURES: features,
                Method.KEY_DATE_FEATURES_FORMAT: [Method.DEFAULT_DATE_FEATURE_FORMAT],
            },
        )

        # THEN
        logging.debug(ice)
        self.assertEqual(ice.features, features)

    def test_pd_str(self):
        # GIVEN
        df3x3 = pd.DataFrame(
            {
                "f1": [1, 2, 3],
                "D": ["2016-01-01", "2018-05-05", "2019-12-12"],
                "f2": [1, 2, 3],
            }
        )
        features = ["D"]
        score_foo = partial(FooScorerRegrSeries().score_batch, fast_approx=True)

        # WHEN
        ice = PD("I").explain(
            features,
            df3x3,
            predict_method=score_foo,
            features_meta={
                Method.KEY_DATE_FEATURES: features,
                Method.KEY_DATE_FEATURES_FORMAT: ["%Y-%m-%d"],
            },
        )

        # THEN
        logging.debug(ice)
        self.assertEqual(ice.features, features)

    def test_pd_n_dim(self):
        # GIVEN
        fs = [("C", "DI", "DS")]
        x = pd.DataFrame(
            {
                "C": [1, 2, 3],
                "DI": [20160101, 20180505, 20191212],
                "DS": ["2016-01-01", "2018-05-05", "2019-12-12"],
                "E": [1, 2, 3],
            }
        )
        score_foo = partial(FooScorerRegrSeries().score_batch, fast_approx=True)
        resolution = 5

        # WHEN
        pdp = PD("N").explain(
            fs,
            x,
            predict_method=score_foo,
            grid_resolution=resolution,
            features_meta={
                PD.KEY_DATE_FEATURES: ["DI", "DS"],
                PD.KEY_DATE_FEATURES_FORMAT: [
                    PD.DEFAULT_DATE_FEATURE_FORMAT,
                    "%Y-%m-%d",
                ],
            },
        )

        # THEN
        self.assertIsNotNone(pdp)
        logging.debug(f"n-PDs: {pdp}")
        self.assertIsNotNone(pdp.explanations())
        # n-PD
        exs = pdp.explanations()[fs[0]][PD.LABEL_REGRESSION]
        logging.debug(f"n-PD cols: {exs.columns.values}")
        self.assertEqual(45, len(exs.columns))

    def test_h_stat_dates(self):
        # GIVEN
        fs = ["DI", "DS"]
        x = pd.DataFrame(
            {
                "C": [1, 2, 3],
                "DI": [20160101, 20180505, 20191212],
                "DS": ["2016-01-01", "2018-05-05", "2019-12-12"],
                "E": [1, 2, 3],
            }
        )
        score_foo = partial(FooScorerRegrSeries().score_batch, fast_approx=True)

        # WHEN
        hstat = HStatistic("I").explain(
            fs,
            x,
            predict_method=score_foo,
            features_meta={
                PD.KEY_DATE_FEATURES: ["DI", "DS"],
                PD.KEY_DATE_FEATURES_FORMAT: [
                    PD.DEFAULT_DATE_FEATURE_FORMAT,
                    "%Y-%m-%d",
                ],
            },
        )

        # THEN
        self.assertIsNotNone(hstat)
        logging.debug(f"H-stat: {hstat}")
        exs = hstat.explanations()
        self.assertIsNotNone(exs)
        self.assertEqual(1, len(exs))
        self.assertGreater(0.5, exs[("DI", "DS")][PD.LABEL_REGRESSION])
