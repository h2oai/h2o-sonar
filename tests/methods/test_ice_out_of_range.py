# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import unittest
from functools import partial

import pandas as pd

from h2o_sonar import loggers as logging
from h2o_sonar.methods import _ice as e_ice
from tests.methods import ice_pd_test_commons


class TestIceOutOfRange(unittest.TestCase):
    """Test ICE for out of range values."""

    def setUp(self):
        logging.setLevel(logging.DEBUG)

        self.strategy = ice_pd_test_commons.IceStrategyFactory()

        # data: 3x3 for value checks
        self.x3x3 = pd.DataFrame({"f1": [1, 2, 3], "F": [1, 3, 5], "f2": [0.5, 2, 3]})
        self.x3x3_mins = [1, 2, 0.5]
        self.x3x3_maxs = [3, 5, 3]
        self.x18x3 = pd.DataFrame(
            {
                "f1": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 3],
                "F": [
                    "cat",
                    "dog",
                    "cat9",
                    "sheep",
                    "cat5",
                    "dog4",
                    "cat1",
                    "dog",
                    "cat1",
                    "1sheep",
                    "2cat",
                    "1dog",
                    "3cat",
                    "2dog",
                    "cat",
                    "sheep5",
                    "cat",
                    "dog8",
                ],
                "f2": [
                    50,
                    40,
                    30,
                    20,
                    10,
                    0,
                    55,
                    45,
                    35,
                    25,
                    15,
                    3,
                    50,
                    40,
                    30,
                    20,
                    10,
                    0,
                ],
            }
        )

        # prediction method lambda
        self.score_sum = partial(
            ice_pd_test_commons.FooScorerSumRegrFrame().score_batch,
            fast_approx=True,
        )

    def test_negative_init(self):
        with self.assertRaises(ValueError):
            # missing standard deviations
            self.strategy.get_ice().explain(
                ["F"],
                self.x3x3,
                predict_method=self.score_sum,
                out_of_range_resolution=3,
            )

    def test_default_oor(self):
        logging.debug("# ICE: default out of range ###")

        # GIVEN
        logging.debug(f"X:\n{self.x3x3}")
        features = ["F"]

        # WHEN
        ice = self.strategy.get_ice().explain(
            features,
            self.x3x3,
            mins=[1],
            maxs=[5],
            predict_method=self.score_sum,
        )

        # THEN
        logging.debug(f"OORs:\n{ice}")
        self.assertIsNotNone(ice.explanations())
        self.assertEqual(
            5,
            len(ice.explanations()[features[0]][e_ice.ICE.LABEL_REGRESSION].columns),
        )

    def test_custom_oor(self):
        # GIVEN
        logging.debug(f"X:\n{self.x3x3}")
        features = ["F"]
        oor_resolution = 5

        # WHEN
        ice = self.strategy.get_ice().explain(
            features,
            self.x3x3,
            mins=[1],
            maxs=[5],
            stds=[2],
            predict_method=self.score_sum,
            out_of_range_resolution=oor_resolution,
        )

        # THEN
        logging.debug(f"OORs:\n{ice}")
        self.assertIsNotNone(ice.explanations())
        self.assertEqual(
            5 + 2 * oor_resolution,
            len(ice.explanations()[features[0]][e_ice.ICE.LABEL_REGRESSION].columns),
        )

    def test_no_bins_no_resolution_oor_only(self):
        # GIVEN
        logging.debug(f"X:\n{self.x3x3}")
        features = ["F"]
        oor_n = 2

        # WHEN
        ice = self.strategy.get_ice().explain(
            features,
            self.x3x3,
            mins=[1],
            maxs=[5],
            stds=[2],
            predict_method=self.score_sum,
            grid_resolution=0,
            out_of_range_resolution=oor_n,
        )

        # THEN
        self.assertNotEqual(ice.explanations(), None)
        self.assertEqual(1, len(ice.explanations()))
        logging.debug(f"OORs {oor_n}\n{ice}")
        self.assertEqual(
            2 * oor_n,
            len(ice.explanations()[features[0]][e_ice.ICE.LABEL_REGRESSION].columns),
        )
        self.assertListEqual(
            [-1, -3, 7, 9],
            list(
                ice.explanations()[features[0]][
                    e_ice.ICE.LABEL_REGRESSION
                ].columns.values
            ),
        )
        oors = ice.explanations()[features[0]][e_ice.ICE.LABEL_REGRESSION]
        self.assertEqual(0.5, oors.loc[0][-1])
        self.assertEqual(1, oors.loc[1][-3])
        self.assertEqual(13, oors.loc[2][7])

    def test_oor_bins(self):
        # GIVEN
        data = self.x18x3
        oor_n = 2
        features = ["f1"]
        bins = [[2, 4]]

        def foo_predict_method(x):
            return (x["f1"] + x["f2"]) / 100

        # WHEN
        ice = self.strategy.get_ice()
        ice.explain(
            features,
            data,
            out_of_range_resolution=oor_n,
            mins=[1],
            maxs=[6],
            stds=[1.6449566416599],
            predict_method=foo_predict_method,
            bins=bins,
        )

        # THEN
        self.assertNotEqual(ice.explanations(), None)
        self.assertEqual(1, len(ice.explanations()))
        self.assertListEqual(
            [2, 4, 0, -1, 7, 8],
            list(
                ice.explanations()[features[0]][
                    e_ice.ICE.LABEL_REGRESSION
                ].columns.values
            ),
        )
        logging.debug(f"OORs n={oor_n}\n{ice}")
        self.assertEqual(
            2 * oor_n + len(bins[0]),
            len(
                ice.explanations()[features[0]][
                    e_ice.ICE.LABEL_REGRESSION
                ].columns.values
            ),
        )
        oors = ice.explanations()[features[0]][e_ice.ICE.LABEL_REGRESSION]
        self.assertEqual(0.58, oors.loc[0][8])
        self.assertEqual(0.39, oors.loc[1][-1])

    def test_multinomial_oor(self):
        # GIVEN
        df = self.x18x3
        oor_n = 2
        features = ["f1"]
        bins = [[2, 4]]
        scorer = ice_pd_test_commons.FooScorerConstPredictFrame(df.shape[0], 3)

        # WHEN
        ice = self.strategy.get_ice()
        ice.explain(
            features,
            df,
            predict_method=scorer.score_batch,
            out_of_range_resolution=oor_n,
            mins=[1],
            maxs=[6],
            stds=[1.6449566416599],
            bins=bins,
        )

        # THEN
        exs = ice.explanations()
        self.assertIsNotNone(exs)
        logging.debug(f"m-ICEs: {ice}")
        class_0 = e_ice.ICE.LABEL_PREFIX_CLASS + "0"
        class_1 = e_ice.ICE.LABEL_PREFIX_CLASS + "1"
        class_2 = e_ice.ICE.LABEL_PREFIX_CLASS + "2"
        self.assertEqual(len(features), len(exs))
        self.assertListEqual(features, list(exs.keys()))
        self.assertEqual(3, len(exs[features[0]]))
        self.assertListEqual([class_0, class_1, class_2], list(exs[features[0]].keys()))

        logging.debug(f"m-ICE OOR bins: {exs[features[0]][class_0].columns.values}")
        self.assertListEqual(
            [2, 4, 0, -1, 7, 8],
            list(exs[features[0]][class_0].columns.values),
        )

        self.assertEqual((18, 6), exs[features[0]][class_0].shape)
        self.assertEqual(290, exs["f1"][class_1].loc[0][8])
        self.assertEqual(129, exs["f1"][class_1].loc[1][0])

    def test_multidimensional_oor_bins(self):
        # GIVEN
        logging.debug(f"X:\n{self.x3x3}")
        features = [("f1", "f2"), "F"]
        x = pd.DataFrame({"f1": [1, 2, 3], "f2": [0.5, 2, 4], "F": [1, 3, 5]})
        bins = [([1, 3], [2, 4]), [3, 5]]
        oor_resolution = 2

        # WHEN
        ice = self.strategy.get_ice().explain(
            features,
            x,
            out_of_range_resolution=oor_resolution,
            bins=bins,
            mins=[(1, 0.5), 5],
            maxs=[(3, 4), 5],
            stds=[(0.3, 0.4), 0.5],
            predict_method=self.score_sum,
        )

        # THEN
        logging.debug(f"OORs:\n{ice}")
        self.assertIsNotNone(ice.explanations())
        logging.debug(
            "{} cols: {}".format(
                str(features[0]),
                ice.explanations()[features[0]][
                    e_ice.ICE.LABEL_REGRESSION
                ].columns.values,
            )
        )
        self.assertEqual(
            4 + 32,
            len(ice.explanations()[features[0]][e_ice.ICE.LABEL_REGRESSION].columns),
        )
        self.assertTrue(
            (3, 4.8)
            in ice.explanations()[features[0]][e_ice.ICE.LABEL_REGRESSION].columns
        )
        self.assertTrue(
            (0, 4.4)
            in ice.explanations()[features[0]][e_ice.ICE.LABEL_REGRESSION].columns
        )
        self.assertEqual(
            1 + 2 * oor_resolution,
            len(ice.explanations()[features[1]][e_ice.ICE.LABEL_REGRESSION].columns),
        )
        self.assertTrue(
            4 in ice.explanations()[features[1]][e_ice.ICE.LABEL_REGRESSION].columns
        )
        self.assertTrue(
            5 in ice.explanations()[features[1]][e_ice.ICE.LABEL_REGRESSION].columns
        )

    def test_multidimensional_oor(self):
        # GIVEN
        logging.debug(f"X:\n{self.x3x3}")
        features = [("f1", "f2"), "F"]
        x = pd.DataFrame({"f1": [1, 2, 3], "f2": [0.5, 2, 4], "F": [1, 3, 5]})

        # WHEN
        ice = self.strategy.get_ice().explain(
            features,
            x,
            out_of_range_resolution=2,
            grid_resolution=2,
            mins=[(1, 0.5), 1],
            maxs=[(3, 4), 5],
            stds=[(0.3, 0.4), 0.5],
            predict_method=self.score_sum,
        )

        # THEN
        logging.debug(f"OORs:\n{ice}")
        self.assertIsNotNone(ice.explanations())
        logging.debug(
            "{} cols: {}".format(
                str(features[0]),
                ice.explanations()[features[0]][
                    e_ice.ICE.LABEL_REGRESSION
                ].columns.values,
            )
        )
        self.assertEqual(
            4 + 32,
            len(ice.explanations()[features[0]][e_ice.ICE.LABEL_REGRESSION].columns),
        )
        self.assertTrue(
            (3, 4.8)
            in ice.explanations()[features[0]][e_ice.ICE.LABEL_REGRESSION].columns
        )
        self.assertTrue(
            (5, 4.4)
            in ice.explanations()[features[0]][e_ice.ICE.LABEL_REGRESSION].columns
        )
