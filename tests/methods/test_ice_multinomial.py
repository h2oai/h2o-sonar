# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import os
import tempfile
from unittest import TestCase

import pandas as pd

from h2o_sonar import loggers as logging
from h2o_sonar.errors import MliJsonSerializationError
from h2o_sonar.methods._ice import ICE
from tests.methods.ice_pd_test_commons import FooScorerConstPredictFrame
from tests.methods.ice_pd_test_commons import IceStrategyFactory


class TestIceMultinomial(TestCase):
    """Test ICE implementation for multinomial predictions."""

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
        ice = self.strategy.get_ice()
        ice.explain(features, df, predict_method=scorer.score_batch, bins=bins)

        # THEN
        exs = ice.explanations()
        self.assertIsNotNone(exs)
        logging.debug(f"m-ICEs: {ice}")
        class_0 = ICE.LABEL_PREFIX_CLASS + "0"
        class_1 = ICE.LABEL_PREFIX_CLASS + "1"
        self.assertEqual(len(features), len(exs))
        self.assertListEqual(features, list(exs.keys()))
        self.assertEqual(2, len(exs[features[0]]))
        self.assertListEqual([class_0, class_1], list(exs[features[0]].keys()))
        self.assertEqual((2, 3), exs[features[0]][class_0].shape)
        self.assertEqual(2, exs["f1"][class_0].loc[0][1])
        self.assertEqual(6, exs["f1"][class_0].loc[0][3])
        self.assertEqual(10, exs["f1"][class_0].loc[0][5])
        self.assertEqual((2, 3), exs[features[0]][class_1].shape)
        self.assertEqual(4, exs["f1"][class_1].loc[0][1])
        self.assertEqual(5, exs["f1"][class_1].loc[1][1])

    def test_multinomial_multi_feature_many_instances(self):
        logging.debug("# ICE@Pandas: Many features, many instances ###")
        # GIVEN
        df = self.df3x3
        bins = [[1, 3, 5], [7, 9]]
        features = ["f1", "f2"]
        scorer = FooScorerConstPredictFrame(df.shape[0], 3)

        # WHEN
        ice = self.strategy.get_ice()
        ice.explain(features, df, predict_method=scorer.score_batch, bins=bins)

        # THEN
        exs = ice.explanations()
        self.assertIsNotNone(exs)
        logging.debug(f"m-ICEs: {ice}")
        self.assertEqual(len(features), len(exs))

    def test_multinomial_multi_fs_many_is_residuals_json(self):
        logging.debug("# ICE@Pandas: May features, many instances, RESIDUALS")
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
        ice = self.strategy.get_ice()
        ice.explain(
            features,
            X,
            predict_method=scorer.score_batch,
            Y=Y,
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
        self.assertEqual((3, 3), exs[features[0]][class_1].shape)
        self.assertEqual(1, exs[features[0]][class_0].loc[0][1])
        self.assertEqual(185, exs[features[0]][class_1].loc[1][3])
        self.assertEqual(2972, exs[features[0]][class_2].loc[2][5])

        # TEST: JSon serialization

        # WHEN
        logging.debug("SAVING JSon")
        tmp_dir = tempfile.mkdtemp(prefix="mli_unit_m_ice_")
        tmp_file_name = os.path.join(tmp_dir, "unit-m_ice-save-basic.json")

        # THEN
        with self.assertRaises(MliJsonSerializationError):
            ice.save_json(tmp_file_name)
