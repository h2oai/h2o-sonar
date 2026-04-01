# Copyright (C) 2018-2026 H2O.ai, Inc. All rights reserved
import os
import tempfile
from functools import partial
from unittest import TestCase

import pandas as pd

from h2o_sonar import loggers as logging
from h2o_sonar.errors import MliJsonSerializationError
from h2o_sonar.methods._ice import ICE
from tests.methods.ice_pd_test_commons import FooScorerConstPredictFrame
from tests.methods.ice_pd_test_commons import FooScorerSumRegrFrame
from tests.methods.ice_pd_test_commons import IceStrategyFactory


class TestMultiDimensionalIce(TestCase):
    """Test N-dimensional ICE implementation."""

    def setUp(self):
        logging.setLevel(logging.DEBUG)

        self.strategy = IceStrategyFactory()

        # data
        self.df5x5 = pd.DataFrame(
            [[i + j for i in range(1, 6)] for j in range(0, 5)],
            columns=list("aFGHb"),
        )

        # prediction
        self.score_foo_regr = partial(
            FooScorerSumRegrFrame().score_batch, fast_approx=True
        )

    def test_negative(self):
        with self.assertRaises(ValueError):
            ICE("-").explain(
                [("F", "G")], self.df5x5, predict_method=self.score_foo_regr
            )
        with self.assertRaises(ValueError):
            ICE("-").explain(
                [("F", "G")],
                self.df5x5,
                predict_method=self.score_foo_regr,
                maxs=[(1, 2)],
            )
        with self.assertRaises(ValueError):
            ICE("-").explain(
                [("F", "G", "H")],
                self.df5x5,
                predict_method=self.score_foo_regr,
                maxs=[(1, 2)],
                mins=[(0, 0)],
            )

    def test_ice_regr(self):
        # GIVEN
        fs = [("F", "G", "H")]
        x = self.df5x5
        mins = [(x["F"].min(), x["G"].min(), x["H"].min())]
        maxs = [(x["F"].max(), x["G"].max(), x["H"].max())]
        logging.debug(f"X:\n{x}")
        logging.debug(f"Mins: {mins}")
        logging.debug(f"Maxs: {maxs}")

        # WHEN
        ice = self.strategy.get_ice().explain(
            fs, x, predict_method=self.score_foo_regr, maxs=maxs, mins=mins
        )

        # THEN
        self.assertIsNotNone(ice)
        logging.debug(f"n-ICEs: {ice}")
        self.assertIsNotNone(ice.explanations())
        self.assertEqual(1, len(ice.explanations()))
        # n-ICE
        exs = ice.explanations()[fs[0]][ICE.LABEL_REGRESSION]
        logging.debug(f"n-ICE cols:\n{exs.columns.values}")
        self.assertEqual(5**3, len(exs.columns))
        self.assertEqual((2, 3, 4), exs.columns.values[0])
        self.assertEqual((6, 7, 8), exs.columns.values[-1])

    def test_ice_regr_bins(self):
        # GIVEN
        fs = [("F", "G", "H")]
        x = self.df5x5
        logging.debug(f"X:\n{x}")
        bins = [([1, 2], [3, 4, 5], [6, 7])]

        # WHEN
        ice = self.strategy.get_ice().explain(
            fs, x, bins=bins, predict_method=self.score_foo_regr
        )

        # THEN
        self.assertIsNotNone(ice)
        logging.debug(f"n-ICEs: {ice}")
        self.assertIsNotNone(ice.explanations())
        self.assertEqual(1, len(ice.explanations()))
        # n-ICE
        exs = ice.explanations()[fs[0]][ICE.LABEL_REGRESSION]
        logging.debug(f"n-ICE cols: {exs.columns.values}")
        self.assertEqual(12, len(exs.columns))

    def test_ice_mixed_regr(self):
        # GIVEN
        fs = ["F", "G", ("F", "G")]
        x = self.df5x5
        bins = [[1, 2], [10, 20], ([0.1, 0.2], [0.3, 0.4, 0.5])]

        # WHEN
        ice = self.strategy.get_ice().explain(
            fs, x, bins=bins, predict_method=self.score_foo_regr
        )

        # THEN
        self.assertIsNotNone(ice)
        logging.debug(f"n-ICEs: {ice}")
        self.assertIsNotNone(ice.explanations())
        self.assertEqual(3, len(ice.explanations()))
        # n-ICE
        exs = ice.explanations()[fs[2]][ICE.LABEL_REGRESSION]
        logging.debug(f"n-ICE cols: {exs.columns.values}")
        self.assertEqual(6, len(exs.columns))
        # ICE
        exs = ice.explanations()[fs[0]][ICE.LABEL_REGRESSION]
        logging.debug(f"ICE cols: {exs.columns.values}")
        self.assertEqual(len(bins[0]), len(exs.columns))

    def test_ice_multinomial_json(self):
        # GIVEN
        fs = [("F", "G", "H")]
        x = self.df5x5
        logging.debug(f"X:\n{x}")
        classes = 3
        scorer = FooScorerConstPredictFrame(self.df5x5.shape[0], classes)
        bins = [([1, 2], [3, 4, 5], [6, 7])]

        # WHEN
        ice = self.strategy.get_ice().explain(
            fs, x, bins=bins, predict_method=scorer.score_batch
        )

        # THEN
        self.assertIsNotNone(ice)
        logging.debug(f"n-ICEs: {ice}")
        self.assertIsNotNone(ice.explanations())
        self.assertEqual(1, len(ice.explanations()))
        # multinomial
        self.assertEqual(classes, len(ice.explanations()[fs[0]]))
        # n-ICE
        exs = ice.explanations()[fs[0]][ICE.LABEL_REGRESSION]
        logging.debug(f"n-ICE cols: {exs.columns.values}")
        self.assertEqual(12, len(exs.columns))

        # TEST: JSon save / load

        # WHEN
        logging.debug("SAVING JSon")
        tmp_dir = tempfile.mkdtemp(prefix="mli_unit_n_ice_m_class_")
        tmp_file_name = os.path.join(tmp_dir, "unit-n_ice-save-basic.json")

        # THEN
        with self.assertRaises(MliJsonSerializationError):
            ice.save_json(tmp_file_name)

    def test_ice_multinomial_residuals_bins(self):
        # GIVEN
        x = pd.DataFrame({"f1": [1, 2, 3], "f2": [1, 3, 5], "f3": [8, 6, 4]})
        y = pd.DataFrame(
            {
                "class-1": [1, 2, 3],
                "class-2": [100, 200, 300],
                "class-3": [1000, 2000, 3000],
            }
        )
        bins = [([1, 3], [2, 4])]
        features = [("f1", "f2")]
        scorer = FooScorerConstPredictFrame(x.shape[0], 3)

        # WHEN
        ice = self.strategy.get_ice()
        ice.explain(
            features,
            x,
            predict_method=scorer.score_batch,
            Y=y,
            target_transform=abs,
            bins=bins,
        )

        # THEN
        exs = ice.explanations()
        self.assertIsNotNone(exs)
        logging.debug(f"m-ICEs residuals:\n{ice}")
        self.assertEqual(len(features), len(exs))
        class_0 = ICE.LABEL_PREFIX_CLASS + "0"
        class_1 = ICE.LABEL_PREFIX_CLASS + "1"
        class_2 = ICE.LABEL_PREFIX_CLASS + "2"
        self.assertListEqual(features, list(exs.keys()))
        self.assertEqual(3, len(exs[features[0]]))
        self.assertListEqual([class_0, class_1, class_2], list(exs[features[0]].keys()))
        self.assertEqual((3, 4), exs[features[0]][class_1].shape)
        self.assertEqual(10, exs[features[0]][class_0].loc[0][1])
        self.assertEqual(167, exs[features[0]][class_1].loc[1][3])
        self.assertEqual(2972, exs[features[0]][class_2].loc[2][2])

    def test_ice_multinomial_residuals(self):
        # GIVEN
        x = pd.DataFrame({"f1": [1, 2, 3], "f2": [1, 3, 5], "f3": [8, 6, 4]})
        y = pd.DataFrame(
            {
                "class-1": [1, 2, 3],
                "class-2": [100, 200, 300],
                "class-3": [1000, 2000, 3000],
            }
        )
        features = [("f1", "f2")]
        scorer = FooScorerConstPredictFrame(x.shape[0], 3)

        # WHEN
        ice = self.strategy.get_ice()
        ice.explain(
            features,
            x,
            mins=[(1, 1)],
            maxs=[(3, 5)],
            predict_method=scorer.score_batch,
            Y=y,
            target_transform=abs,
            grid_resolution=2,
        )

        # THEN
        exs = ice.explanations()
        self.assertIsNotNone(exs)
        logging.debug(f"m-ICEs residuals:\n{ice}")
        self.assertEqual(len(features), len(exs))
        class_0 = ICE.LABEL_PREFIX_CLASS + "0"
        class_1 = ICE.LABEL_PREFIX_CLASS + "1"
        class_2 = ICE.LABEL_PREFIX_CLASS + "2"
        self.assertListEqual(features, list(exs.keys()))
        self.assertEqual(3, len(exs[features[0]]))
        self.assertListEqual([class_0, class_1, class_2], list(exs[features[0]].keys()))
        self.assertEqual((3, 4), exs[features[0]][class_1].shape)
        self.assertEqual(10, exs[features[0]][class_0].loc[0][1])
        self.assertEqual(167, exs[features[0]][class_1].loc[1][3])
        self.assertEqual(2972, exs[features[0]][class_2].loc[2][2])
