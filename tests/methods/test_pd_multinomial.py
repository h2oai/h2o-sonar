# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import os
import tempfile
from unittest import TestCase

import pandas as pd

from h2o_sonar import loggers as logging
from h2o_sonar.errors import MliJsonSerializationError
from h2o_sonar.methods._pd import PD
from tests.methods.ice_pd_test_commons import FooScorerConstPredictFrame
from tests.methods.ice_pd_test_commons import IceStrategyFactory


class TestPdMultinomial(TestCase):
    """
    Test PD implementation for multinomial predictions.

    """

    def setUp(self):
        logging.setLevel(logging.DEBUG)

        self.strategy = IceStrategyFactory()

        # data
        self.df2x2 = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
        # data: 3x3 for value checks
        self.df3x3 = pd.DataFrame({"f1": [1, 2, 3], "f2": [1, 3, 5], "f3": [8, 6, 4]})

    def test_binomial_single_feature_many_instances(self):
        # GIVEN
        df = self.df2x2
        bins = [[1, 3, 5]]
        features = ["f1"]
        scorer = FooScorerConstPredictFrame(df.shape[0], 2)

        # WHEN
        pdp = self.strategy.get_pd()
        pdp.explain(features, df, predict_method=scorer.score_batch, bins=bins)

        # THEN
        exs = pdp.explanations()
        self.assertIsNotNone(exs)
        logging.debug(f"m-PDs: {pdp}")
        class_0 = PD.LABEL_PREFIX_CLASS + "0"
        class_1 = PD.LABEL_PREFIX_CLASS + "1"
        self.assertEqual(len(features), len(exs))
        self.assertListEqual(features, list(exs.keys()))
        self.assertEqual(2, len(exs[features[0]]))
        self.assertListEqual([class_0, class_1], list(exs[features[0]].keys()))
        self.assertEqual((4, 3), exs[features[0]][class_0].shape)
        self.assertEqual(2.5, exs["f1"][class_0].loc["mean"][1])
        self.assertEqual(6.5, exs["f1"][class_0].loc["mean"][3])
        self.assertEqual(10.5, exs["f1"][class_0].loc["mean"][5])
        self.assertEqual((4, 3), exs[features[0]][class_1].shape)
        self.assertEqual(4.5, exs["f1"][class_1].loc["mean"][1])
        self.assertEqual(0.7071067811865476, exs["f1"][class_1].loc["sd"][3])
        self.assertEqual(0.5, exs["f1"][class_1].loc["sem"][5])

    def test_multinomial_multiple_feature_many_instances(self):
        # GIVEN
        df = self.df3x3
        bins = [[1, 3, 5], [7, 9]]
        features = ["f1", "f2"]
        scorer = FooScorerConstPredictFrame(df.shape[0], 3)

        # WHEN
        pdp = self.strategy.get_pd()
        pdp.explain(features, df, predict_method=scorer.score_batch, bins=bins)

        # THEN
        exs = pdp.explanations()
        self.assertIsNotNone(exs)
        logging.debug(f"m-PDs: {pdp}")
        class_0 = PD.LABEL_PREFIX_CLASS + "0"
        class_1 = PD.LABEL_PREFIX_CLASS + "1"
        class_2 = PD.LABEL_PREFIX_CLASS + "2"
        self.assertEqual(len(features), len(exs))
        self.assertListEqual(features, list(exs.keys()))
        self.assertEqual(3, len(exs[features[0]]))
        self.assertListEqual([class_0, class_1, class_2], list(exs[features[0]].keys()))

        self.assertEqual((4, 3), exs[features[0]][class_0].shape)
        self.assertEqual(3, exs["f1"][class_0].loc["mean"][1])
        self.assertEqual(12, exs["f1"][class_0].loc["mean"][3])
        self.assertEqual(21, exs["f1"][class_0].loc["mean"][5])
        self.assertEqual((4, 3), exs[features[0]][class_1].shape)
        self.assertEqual(6, exs["f1"][class_1].loc["mean"][1])
        self.assertEqual(1, exs["f1"][class_1].loc["sd"][3])
        self.assertEqual(0.5773502691896258, exs["f1"][class_1].loc["sem"][5])
        self.assertEqual((4, 3), exs[features[0]][class_2].shape)
        self.assertEqual(9, exs["f1"][class_2].loc["mean"][1])
        self.assertEqual(1, exs["f1"][class_1].loc["sd"][3])
        self.assertEqual(0.5773502691896258, exs["f1"][class_2].loc["sem"][5])

        self.assertEqual((4, 2), exs[features[1]][class_0].shape)
        self.assertEqual(30, exs["f2"][class_0].loc["mean"][7])
        self.assertEqual(39, exs["f2"][class_0].loc["mean"][9])
        self.assertEqual((4, 2), exs[features[1]][class_2].shape)
        self.assertEqual(36, exs["f2"][class_2].loc["mean"][7])
        self.assertEqual(1, exs["f2"][class_2].loc["sd"][7])
        self.assertEqual(0.5773502691896258, exs["f2"][class_2].loc["sem"][7])

        # TEST: JSon serialization

        # WHEN
        logging.debug("SAVING JSon")
        tmp_dir = tempfile.mkdtemp(prefix="mli_unit_m_pd_")
        tmp_file_name = os.path.join(tmp_dir, "unit-m_pd-save-basic.json")

        # THEN
        with self.assertRaises(MliJsonSerializationError):
            pdp.save_json(tmp_file_name)

    def test_multinomial_multiple_f_many_i_residuals(self):
        # GIVEN
        X = self.df3x3
        Y = pd.DataFrame(
            {
                "class-1": [1, 2, 3],
                "class-2": [100, 200, 300],
                "class-3": [1000, 2000, 3000],
            }
        )
        bins = [[1, 3, 5], [7, 9]]
        features = ["f1", "f2"]
        scorer = FooScorerConstPredictFrame(X.shape[0], 3)

        # WHEN
        pdp = self.strategy.get_pd()
        pdp.explain(
            features,
            X,
            predict_method=scorer.score_batch,
            bins=bins,
            Y=Y,
            target_transform=abs,
        )

        # THEN
        exs = pdp.explanations()
        self.assertIsNotNone(exs)
        logging.debug(f"m-PDs residuals:\n{pdp}")
        class_0 = PD.LABEL_PREFIX_CLASS + "0"
        class_1 = PD.LABEL_PREFIX_CLASS + "1"
        class_2 = PD.LABEL_PREFIX_CLASS + "2"
        self.assertEqual(len(features), len(exs))
        self.assertListEqual(features, list(exs.keys()))
        self.assertEqual(3, len(exs[features[0]]))
        self.assertListEqual([class_0, class_1, class_2], list(exs[features[0]].keys()))

        self.assertEqual((7, 3), exs[features[0]][class_0].shape)
        self.assertEqual(1.0, exs["f1"][class_0].loc[PD.COL_RESIDUAL_MEAN][1])
        self.assertEqual(99.0, exs["f1"][class_1].loc[PD.COL_RESIDUAL_SD][3])
        self.assertEqual(
            576.7729189204362, exs["f1"][class_2].loc[PD.COL_RESIDUAL_SEM][5]
        )

    def test_multinomial_ice_cache(self):
        # GIVEN
        df = self.df3x3
        bins = [[1, 3, 5], [7, 9]]
        features = ["f1", "f2"]
        scorer = FooScorerConstPredictFrame(df.shape[0], 3)
        class_0 = PD.LABEL_PREFIX_CLASS + "0"
        class_2 = PD.LABEL_PREFIX_CLASS + "2"

        # WHEN
        pdp = self.strategy.get_pd()
        pdp.explain(
            features,
            df,
            ice_cache={},
            predict_method=scorer.score_batch,
            bins=bins,
        )

        # THEN
        exs = pdp.explanations()
        self.assertIsNotNone(exs)
        logging.debug(f"m-PDs: {pdp}")
        self.assertEqual(len(features), len(exs))
        self.assertListEqual(features, list(exs.keys()))
        cached_ice = pdp.explanations(kind="ice")
        logging.debug(f"Cached m_ICE:\n{cached_ice}")
        self.assertIsNotNone(cached_ice)
        self.assertEqual(2, len(cached_ice))
        self.assertEqual(3, len(cached_ice["f1"]))
        self.assertEqual(3, len(cached_ice["f2"]))
        self.assertEqual((3, 3), cached_ice["f1"][class_0].shape)
        self.assertEqual((3, 2), cached_ice["f2"][class_2].shape)

    def test_multinomial_ice_filter_n_cache(self):
        logging.debug("# m-PD: ICE caching ###")
        # GIVEN
        df = self.df3x3
        bins = [[1, 3, 5], [7, 9]]
        features = ["f1", "f2"]
        scorer = FooScorerConstPredictFrame(df.shape[0], 3)
        class_0 = PD.LABEL_PREFIX_CLASS + "0"
        class_2 = PD.LABEL_PREFIX_CLASS + "2"

        # WHEN
        pdp = self.strategy.get_pd()
        pdp.explain(
            features,
            df,
            ice_cache={"f1": {class_0: [0, 2]}, "f2": {class_2: [1]}},
            predict_method=scorer.score_batch,
            bins=bins,
        )

        # THEN
        exs = pdp.explanations()
        self.assertIsNotNone(exs)
        logging.debug(f"m-PDs: {pdp}")
        self.assertEqual(len(features), len(exs))
        self.assertListEqual(features, list(exs.keys()))
        cached_ice = pdp.explanations(kind="ice")
        logging.debug(f"Cached m_ICE:\n{cached_ice}")
        self.assertIsNotNone(cached_ice)
        self.assertEqual(2, len(cached_ice))
        self.assertEqual(1, len(cached_ice["f1"]))
        self.assertEqual(1, len(cached_ice["f2"]))
        self.assertEqual((2, 3), cached_ice["f1"][class_0].shape)
        self.assertEqual((1, 2), cached_ice["f2"][class_2].shape)
