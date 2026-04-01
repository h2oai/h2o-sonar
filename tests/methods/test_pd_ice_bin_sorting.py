# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
from functools import partial
from unittest import TestCase

import pandas as pd

from h2o_sonar import loggers as logging
from h2o_sonar.methods._pd import PD
from tests.methods.ice_pd_test_commons import FooScorerSumRegrFrame
from tests.methods.ice_pd_test_commons import IceStrategyFactory


class TestPdIceBinSorting(TestCase):
    """Test bin categorical and numerical sorting."""

    def setUp(self):
        logging.setLevel(logging.DEBUG)

        self.strategy = IceStrategyFactory()

        # data
        self.x5x3 = pd.DataFrame(
            {
                "F": [1, 2, 3, 4, 5],
                "C": ["cat", "dog", "cat", "sheep", "cat"],
                "G": [50, 40, 30, 20, 10],
            }
        )

    @staticmethod
    def predict_method(x):
        return x["G"]

    def _when_ice(
        self,
        feature,
        feature_bin: list,
        expected_bin: list,
        features_meta: dict = None,
        do_sort: bool = True,
        oor: int = 0,
    ):
        # GIVEN
        fs = [feature]
        bins = [feature_bin]

        # WHEN
        ice = self.strategy.get_ice().explain(
            fs,
            self.x5x3,
            predict_method=TestPdIceBinSorting.predict_method,
            mins=[0],
            maxs=[55],
            bins=bins,
            bins_sort=do_sort,
            features_meta=features_meta,
            out_of_range_resolution=oor,
        )

        # THEN
        self.assertIsNotNone(ice)
        logging.debug(f"ICEs:\n{ice}")
        self.assertListEqual(
            expected_bin,
            ice.explanations()[fs[0]][PD.LABEL_REGRESSION].columns.values.tolist(),
        )

        return ice

    def test_ice_no_sort(self):
        self._when_ice("F", [5, 1, 3, 7, 8], [5, 1, 3, 7, 8], None, False)

    def test_ice_num_sort(self):
        self._when_ice("F", [5, 1, 3, 7, 8], [1, 3, 5, 7, 8])

    def test_ice_cat_sort(self):
        self._when_ice(
            "C",
            ["0", "a", "-1", "-2", "22", "b"],
            ["-1", "-2", "0", "22", "a", "b"],
            {PD.KEY_CATEGORICAL_FEATURES: ["C"]},
        )

    def test_ice_num_in_cat_sort(self):
        """If feature is categorical, but bins are numbers in strings, then use
        numerical ordering.

        """
        self._when_ice("F", ["2", "-1", "0", "-2", "1"], ["-1", "-2", "0", "1", "2"])

    def _when_pd(
        self,
        feature,
        feature_bin: list,
        expected_bin: list,
        features_meta: dict = None,
        do_sort: bool = True,
        oor: int = 0,
    ):
        # GIVEN
        fs = [feature]
        bins = [feature_bin]

        # WHEN
        pdp = self.strategy.get_pd().explain(
            fs,
            self.x5x3,
            predict_method=TestPdIceBinSorting.predict_method,
            bins=bins,
            bins_sort=do_sort,
            features_meta=features_meta,
            out_of_range_resolution=oor,
        )

        # THEN
        self.assertIsNotNone(pdp)
        logging.debug(f"PDs:\n{pdp}")
        self.assertListEqual(
            expected_bin,
            pdp.explanations()[fs[0]][PD.LABEL_REGRESSION].columns.values.tolist(),
        )

        return pdp

    def test_pd_bin_num_sort(self):
        self._when_pd("F", [5, 1, 3, 7, 8], [1, 3, 5, 7, 8])

    def test_pd_cat_sort(self):
        self._when_pd(
            "C",
            ["0", "a", "-1", "-2", "22", "b"],
            ["-1", "-2", "0", "22", "a", "b"],
            {PD.KEY_CATEGORICAL_FEATURES: ["C"]},
        )

    def test_pd_num_in_cat_sort(self):
        """If feature is categorical, but bins are numbers in strings, then use
        numerical ordering.

        """
        self._when_pd(
            "F",
            ["2", "-1", "0", "-2", "1"],
            ["-1", "-2", "0", "1", "2"],
            {PD.KEY_CATEGORICAL_FEATURES: ["F"]},
        )

    def test_pd_n_dim_regr_cat(self):
        # GIVEN
        X = pd.DataFrame(
            {
                "F": [6, 1, 2, 3, 4, 5],
                "G": ["cat", "dog", "cat9", "sheep", "cat5", "dog4"],
                "H": [50, 20, 0, 55, 45, 35],
            }
        )
        fs = [("F", "G"), "G"]
        n_predict_method = partial(
            FooScorerSumRegrFrame().score_batch, fast_approx=True
        )

        # WHEN
        npd = self.strategy.get_pd().explain(
            fs, X, predict_method=n_predict_method, bins_sort=True
        )

        # THEN
        self.assertIsNotNone(npd)
        logging.debug(f"n-PDs: {npd}")
        logging.debug(
            npd.explanations()[("F", "G")][PD.LABEL_REGRESSION].columns.values.tolist()
        )
        self.assertListEqual(
            [
                (1, "cat"),
                (1, "cat5"),
                (1, "cat9"),
                (1, "dog"),
                (1, "dog4"),
                (1, "sheep"),
                (2, "cat"),
                (2, "cat5"),
                (2, "cat9"),
                (2, "dog"),
                (2, "dog4"),
                (2, "sheep"),
                (3, "cat"),
                (3, "cat5"),
                (3, "cat9"),
                (3, "dog"),
                (3, "dog4"),
                (3, "sheep"),
                (4, "cat"),
                (4, "cat5"),
                (4, "cat9"),
                (4, "dog"),
                (4, "dog4"),
                (4, "sheep"),
                (5, "cat"),
                (5, "cat5"),
                (5, "cat9"),
                (5, "dog"),
                (5, "dog4"),
                (5, "sheep"),
                (6, "cat"),
                (6, "cat5"),
                (6, "cat9"),
                (6, "dog"),
                (6, "dog4"),
                (6, "sheep"),
            ],
            npd.explanations()[("F", "G")][PD.LABEL_REGRESSION].columns.values.tolist(),
        )
        self.assertListEqual(
            ["cat", "cat5", "cat9", "dog", "dog4", "sheep"],
            npd.explanations()["G"][PD.LABEL_REGRESSION].columns.values.tolist(),
        )

    def test_pd_oor_num_sort(self):
        pdp = self._when_pd(
            "F",
            [5, 1, 3, 7, 8],
            [-3, -1, 1, 3, 5, 7, 8, 10, 12],
            oor=2,
        )

        self.assertListEqual(
            [True, True, False, False, False, False, False, True, True],
            pdp.explanations()["F"][PD.LABEL_REGRESSION]
            .loc[PD.COL_OOR]
            .values.tolist(),
        )
