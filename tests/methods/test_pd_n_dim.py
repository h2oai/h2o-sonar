# Copyright (C) 2018-2026 H2O.ai, Inc. All rights reserved
import os
import tempfile
from functools import partial
from unittest import TestCase

import pandas as pd

from h2o_sonar import loggers as logging
from h2o_sonar.errors import MliJsonSerializationError
from h2o_sonar.methods._pd import PD
from tests.methods.ice_pd_test_commons import FooScorerConstPredictFrame
from tests.methods.ice_pd_test_commons import FooScorerSumRegrFrame
from tests.methods.ice_pd_test_commons import IceStrategyFactory


class TestMultiDimensionalPd(TestCase):
    """Test N-dimensional PD implementation."""

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

    def test_pd_regr(self):
        # GIVEN
        fs = [("F", "G", "H")]
        x = self.df5x5

        # WHEN
        pdp = self.strategy.get_pd().explain(fs, x, predict_method=self.score_foo_regr)

        # THEN
        self.assertIsNotNone(pdp)
        logging.debug(f"n-PDs: {pdp}")
        self.assertIsNotNone(pdp.explanations())
        # n-PD
        exs = pdp.explanations()[fs[0]][PD.LABEL_REGRESSION]
        logging.debug(f"n-PD cols: {exs.columns.values}")
        self.assertEqual(self.df5x5.shape[1] ** 3, len(exs.columns))

    def test_pd_regr_bins(self):
        # GIVEN
        fs = [("F", "G", "H")]
        x = self.df5x5
        bins = [([1, 2], [3, 4, 5], [6, 7])]

        # WHEN
        pdp = self.strategy.get_pd().explain(
            fs, x, bins=bins, predict_method=self.score_foo_regr
        )

        # THEN
        self.assertIsNotNone(pdp)
        logging.debug(f"n-PDs: {pdp}")
        self.assertIsNotNone(pdp.explanations())
        # n-PD
        exs = pdp.explanations()[fs[0]][PD.LABEL_REGRESSION]
        logging.debug(f"n-PD cols: {exs.columns.values}")
        self.assertEqual(12, len(exs.columns))

    def __give_when_pd_mixed_regr(self, features, bins, center):
        # GIVEN
        x = self.df5x5

        # WHEN
        npd = self.strategy.get_pd().explain(
            features,
            x,
            bins=bins,
            predict_method=self.score_foo_regr,
            center=center,
        )

        return npd

    def test_pd_mixed_regr(self):
        # GIVEN
        fs = ["F", "G", ("F", "G")]
        f_bin = [1, 2]
        g_bin = [10, 20]
        bins = [f_bin, g_bin, (f_bin, g_bin)]

        # WHEN
        npd = self.__give_when_pd_mixed_regr(fs, bins, False)

        # THEN
        self.assertIsNotNone(npd)
        logging.debug(f"n-PDs: {npd}")
        self.assertIsNotNone(npd.explanations())
        self.assertEqual(3, len(npd.explanations()))
        # n-PD
        exs = npd.explanations()[fs[2]][PD.LABEL_REGRESSION]
        logging.debug(f"n-PD cols: {exs.columns.values}")
        self.assertEqual(4, len(exs.columns))
        self.assertEqual(
            27,
            npd.explanations()[fs[2]][PD.LABEL_REGRESSION].loc["mean"][(1, 10)],
        )
        # PD
        exs = npd.explanations()[fs[0]][PD.LABEL_REGRESSION]
        logging.debug(f"PD cols: {exs.columns.values}")
        self.assertEqual(len(bins[0]), len(exs.columns))

    def test_n_dim_pd_mixed_regr_center(self):
        # GIVEN
        fs = ["F", "G", ("F", "G")]
        f_bin = [1, 2]
        g_bin = [10, 20]
        bins = [f_bin, g_bin, (f_bin, g_bin)]

        # WHEN
        npd = self.__give_when_pd_mixed_regr(fs, bins, True)

        # THEN
        self.assertIsNotNone(npd)
        logging.debug(f"n-PDs: {npd}")
        self.assertIsNotNone(npd.explanations())
        self.assertEqual(3, len(npd.explanations()))
        # n-PD
        exs = npd.explanations()[fs[2]][PD.LABEL_REGRESSION]
        logging.debug(f"n-PD cols: {exs.columns.values}")
        self.assertEqual(4, len(exs.columns))
        self.assertEqual(
            -5.5,
            npd.explanations()[fs[2]][PD.LABEL_REGRESSION].loc["mean"][(1, 10)],
        )
        # PD
        exs = npd.explanations()[fs[0]][PD.LABEL_REGRESSION]
        logging.debug(f"PD cols: {exs.columns.values}")
        self.assertEqual(len(bins[0]), len(exs.columns))
        self.assertEqual(2, len(exs.columns))

    def test_n_dim_regr_categorical(self):
        # GIVEN
        x = pd.DataFrame(
            {
                "F": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 3],
                "G": [
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
                "H": [
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
        fs = [("F", "G"), "G"]

        # WHEN
        npd = self.strategy.get_pd().explain(fs, x, predict_method=self.score_foo_regr)

        # THEN
        self.assertIsNotNone(npd)
        logging.debug(f"n-PDs: {npd}")
        self.assertIsNotNone(npd.explanations())
        self.assertEqual(2, len(npd.explanations()))
        # n-PD
        exs = npd.explanations()[fs[0]][PD.LABEL_REGRESSION]
        logging.debug(f"n-PD cols: {exs.columns.values}")
        self.assertEqual(60, len(exs.columns))
        self.assertEqual(
            27.555555555555557,
            npd.explanations()[fs[0]][PD.LABEL_REGRESSION].loc["mean"][(1, "cat")],
        )
        # PD
        exs = npd.explanations()[fs[1]][PD.LABEL_REGRESSION]
        logging.debug(f"PD cols: {exs.columns.values}")
        self.assertEqual(10, len(exs.columns))

    def test_pd_multinomial(self):
        # GIVEN
        fs = [("F", "G", "H")]
        x = self.df5x5
        logging.debug(f"X:\n{x}")
        classes = 3
        scorer = FooScorerConstPredictFrame(self.df5x5.shape[0], classes)
        bins = [([1, 2], [3, 4, 5], [6, 7])]

        # WHEN
        npd = self.strategy.get_pd().explain(
            fs, x, bins=bins, predict_method=scorer.score_batch
        )

        # THEN
        self.assertIsNotNone(npd)
        logging.debug(f"n-PDs: {npd}")
        self.assertIsNotNone(npd.explanations())
        self.assertEqual(1, len(npd.explanations()))
        # multinomial
        self.assertEqual(classes, len(npd.explanations()[fs[0]]))
        # n-PD
        exs = npd.explanations()[fs[0]][PD.LABEL_REGRESSION]
        logging.debug(f"n-PD cols: {exs.columns.values}")
        self.assertEqual(12, len(exs.columns))

        # TEST: JSon serialization

        # WHEN
        logging.debug("SAVING JSon")
        tmp_dir = tempfile.mkdtemp(prefix="mli_unit_n_pd_")
        tmp_file_name = os.path.join(tmp_dir, "unit-n_pd-save-basic.json")

        # THEN
        with self.assertRaises(MliJsonSerializationError):
            npd.save_json(tmp_file_name)

    def test_pd_multinomial_residuals_bins(self):
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
        pdp = self.strategy.get_pd()
        pdp.explain(
            features,
            x,
            predict_method=scorer.score_batch,
            Y=y,
            target_transform=abs,
            bins=bins,
        )

        # THEN
        exs = pdp.explanations()
        self.assertIsNotNone(exs)
        logging.debug(f"m-ICEs residuals:\n{pd}")
        self.assertEqual(len(features), len(exs))
        class_0 = PD.LABEL_PREFIX_CLASS + "0"
        class_1 = PD.LABEL_PREFIX_CLASS + "1"
        class_2 = PD.LABEL_PREFIX_CLASS + "2"
        self.assertListEqual(features, list(exs.keys()))
        self.assertEqual(3, len(exs[features[0]]))
        self.assertListEqual([class_0, class_1, class_2], list(exs[features[0]].keys()))
        self.assertEqual((7, 4), exs[features[0]][class_1].shape)
        self.assertEqual(10.0, exs[features[0]][class_0].loc[PD.COL_RESIDUAL_MEAN][1])
        self.assertEqual(167.0, exs[features[0]][class_1].loc[PD.COL_RESIDUAL_MEAN][3])
        self.assertEqual(1973.0, exs[features[0]][class_2].loc[PD.COL_RESIDUAL_MEAN][2])

    def test_pd_multinomial_residuals(self):
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
        pdp = self.strategy.get_pd()
        pdp.explain(
            features,
            x,
            predict_method=scorer.score_batch,
            Y=y,
            target_transform=abs,
            grid_resolution=2,
        )

        # THEN
        exs = pdp.explanations()
        self.assertIsNotNone(exs)
        logging.debug(f"m-ICEs residuals:\n{pdp}")
        self.assertEqual(len(features), len(exs))
        class_0 = PD.LABEL_PREFIX_CLASS + "0"
        class_1 = PD.LABEL_PREFIX_CLASS + "1"
        class_2 = PD.LABEL_PREFIX_CLASS + "2"
        self.assertListEqual(features, list(exs.keys()))
        self.assertEqual(3, len(exs[features[0]]))
        self.assertListEqual([class_0, class_1, class_2], list(exs[features[0]].keys()))
        self.assertEqual((7, 4), exs[features[0]][class_1].shape)
        self.assertEqual(10.0, exs[features[0]][class_0].loc[PD.COL_RESIDUAL_MEAN][1])
        self.assertEqual(167.0, exs[features[0]][class_1].loc[PD.COL_RESIDUAL_MEAN][3])
        self.assertEqual(1973.0, exs[features[0]][class_2].loc[PD.COL_RESIDUAL_MEAN][2])
