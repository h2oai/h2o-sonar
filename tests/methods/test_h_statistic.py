# Copyright (C) 2018-2026 H2O.ai, Inc. All rights reserved
import tempfile
from unittest import TestCase

import pandas

from h2o_sonar import loggers as logging
from h2o_sonar.methods import _h_statistic
from h2o_sonar.methods import _pd
from h2o_sonar.methods.core import _mli
from tests.conftest import get_h2o3_config
from tests.methods.ice_pd_test_commons import IceStrategyFactory
from tests.test_utils import rm_test_dir


class TestHStatistic(TestCase):
    """Test Friedman's H-statistic implementation."""

    def setUp(self):
        logging.setLevel(logging.DEBUG)

        self.strategy = IceStrategyFactory()

        # data
        self.df5x5 = pandas.DataFrame(
            {
                "a": [9.0, 8.0, 7.0, 6.0, 5.0],
                "F": [3.0, 4.0, 5.0, 4.0, 3.0],
                "G": [10.0, 20.0, 30.0, 40.0, 50.0],
                "H": [0.1, 0.2, 0.3, 0.4, 0.5],
                "b": [5.0, 6.0, 7.0, 8.0, 9.0],
            }
        )
        self.classes = 3

    @staticmethod
    def foo_predict_method_regr(x):
        y_hat = abs(x["a"] + x["F"] + x["G"] * x["H"] - x["b"]) ** (1 / 2)
        # logging.debug("Y_hat:\n{}".format(y_hat))
        return y_hat

    @staticmethod
    def foo_predict_method_cat_regr(x):
        y_hat = abs(x["F"] + x["H"]) ** (1 / 2)
        logging.debug(f"Y_hat:\n{y_hat}")
        return y_hat

    def foo_predict_method_multinomial(self, x):
        y_class = TestHStatistic.foo_predict_method_regr(x)
        y_hat = pandas.DataFrame({"c1": y_class, "c2": y_class * 2, "c3": y_class / 3})
        assert y_hat.shape[1] == self.classes
        logging.debug(f"Y_hat:\n{y_hat}")
        return y_hat

    def _when_foo_predict_regr(self, features, X, bins):
        return self.strategy.get_h_statistic().explain(
            features,
            X,
            predict_method=TestHStatistic.foo_predict_method_regr,
            bins=bins,
        )

    def _when_foo_predict_multinomial(self, features, X, bins):
        return self.strategy.get_h_statistic().explain(
            features,
            X,
            predict_method=self.foo_predict_method_multinomial,
            bins=bins,
        )

    def test_negative_init(self):
        with self.assertRaises(ValueError):
            _h_statistic.HStatistic(None)
        with self.assertRaises(ValueError):
            _h_statistic.HStatistic("")

    def test_negative(self):
        with self.assertRaises(ValueError):
            # interaction of at least 2 features
            _h_statistic.HStatistic("-").explain(["F"], self.df5x5)
        with self.assertRaises(ValueError):
            # either interpretable model or predict method needed
            _h_statistic.HStatistic("-").explain(["F", "G"], self.df5x5)

    def test_2_features_2x2_values_regr_bins(self):
        # GIVEN
        X = self.df5x5
        logging.debug(f"X:\n{X}")
        features = list("FG")
        bins = [[10.0, 1.0], [2.0, 20.0]]

        # WHEN
        h_statistic = self._when_foo_predict_regr(features, X, bins)

        # THEN
        self.assertIsNotNone(h_statistic)
        logging.debug(f"H-statistic:{h_statistic}")
        self.assertIsNotNone(h_statistic.explanations())
        self.assertEqual(1, len(h_statistic.explanations()))
        self.assertGreater(
            1, h_statistic.explanations()[("F", "G")][_pd.PD.LABEL_REGRESSION]
        )
        self.assertEqual(
            0.08272287837657792,
            h_statistic.explanations()[("F", "G")][_pd.PD.LABEL_REGRESSION],
        )

    def test_2_features_3x3_values_regr_bins(self):
        # GIVEN
        X = self.df5x5
        logging.debug(f"X:\n{X}")
        features = list("FG")
        bins = [[1, 3, 5], [2, 8, 13]]

        # WHEN
        h_statistic = self._when_foo_predict_regr(features, X, bins)

        # THEN
        self.assertIsNotNone(h_statistic)
        logging.debug(f"Scorer calls: {h_statistic.diagnostics.total_scorer_calls}")
        logging.debug(f"H-statistic:{h_statistic}")
        self.assertIsNotNone(h_statistic.explanations())
        self.assertEqual(1, len(h_statistic.explanations()))
        self.assertGreater(
            1, h_statistic.explanations()[("F", "G")][_pd.PD.LABEL_REGRESSION]
        )
        self.assertEqual(
            0.09363519802353176,
            h_statistic.explanations()[("F", "G")][_pd.PD.LABEL_REGRESSION],
        )

    def test_2_features_2x3_values_regr_bins(self):
        # GIVEN
        X = self.df5x5
        logging.debug(f"X:\n{X}")
        features = list("FG")
        bins = [[1, 3], [2, 8, 5]]

        # WHEN
        h_statistic = self._when_foo_predict_regr(features, X, bins)

        # THEN
        self.assertIsNotNone(h_statistic)
        logging.debug(f"Scorer calls: {h_statistic.diagnostics.total_scorer_calls}")
        logging.debug(f"H-statistic:{h_statistic}")
        self.assertIsNotNone(h_statistic.explanations())
        self.assertEqual(1, len(h_statistic.explanations()))
        self.assertGreater(
            1, h_statistic.explanations()[("F", "G")][_pd.PD.LABEL_REGRESSION]
        )
        self.assertEqual(
            0.14792426794770042,
            h_statistic.explanations()[("F", "G")][_pd.PD.LABEL_REGRESSION],
        )

    def test_3_features_2x2_values_regr_bins(self):
        # GIVEN
        X = self.df5x5
        logging.debug(f"X:\n{X}")
        features = list("FGH")
        bins = [[1, 3], [2, 8], [13, 16]]

        # WHEN
        h_statistic = self._when_foo_predict_regr(features, X, bins)

        # THEN
        self.assertIsNotNone(h_statistic)
        logging.debug(f"Scorer calls: {h_statistic.diagnostics.total_scorer_calls}")
        logging.debug(f"H-statistic:{h_statistic}")
        self.assertIsNotNone(h_statistic.explanations())
        self.assertEqual(3, len(h_statistic.explanations()))
        self.assertGreater(
            1, h_statistic.explanations()[("F", "G")][_pd.PD.LABEL_REGRESSION]
        )
        self.assertEqual(
            0.11175305034852595,
            h_statistic.explanations()[("F", "G")][_pd.PD.LABEL_REGRESSION],
        )
        self.assertEqual(
            0.013857157530249646,
            h_statistic.explanations()[("F", "H")][_pd.PD.LABEL_REGRESSION],
        )
        self.assertEqual(
            0.8207840640422803,
            h_statistic.explanations()[("G", "H")][_pd.PD.LABEL_REGRESSION],
        )

    def test_3_features_all_values_regr_bins(self):
        # GIVEN
        X = self.df5x5
        logging.debug(f"X:\n{X}")
        features = list("FGH")
        bins = [list(set(X["F"].tolist())), X["G"].tolist(), X["H"].tolist()]

        # WHEN
        h_statistic = self._when_foo_predict_regr(features, X, bins)

        # THEN
        self.assertIsNotNone(h_statistic)
        logging.debug(f"Scorer calls: {h_statistic.diagnostics.total_scorer_calls}")
        logging.debug(f"H-statistic:{h_statistic}")
        self.assertIsNotNone(h_statistic.explanations())
        self.assertEqual(3, len(h_statistic.explanations()))
        self.assertGreater(
            1, h_statistic.explanations()[("F", "G")][_pd.PD.LABEL_REGRESSION]
        )

    def test_2_features_2x2_values_multinomial(self):
        # GIVEN
        X = self.df5x5
        logging.debug(f"X:\n{X}")
        features = list("FG")
        bins = [[10.0, 1.0], [2.0, 20.0]]

        # WHEN
        h_statistic = self._when_foo_predict_multinomial(features, X, bins)

        # THEN
        self.assertIsNotNone(h_statistic)
        logging.debug(f"Scorer calls: {h_statistic.diagnostics.total_scorer_calls}")
        if self.strategy.is_1_predict():
            self.assertEqual(6, h_statistic.diagnostics.total_scorer_calls)
        else:
            self.assertEqual(8, h_statistic.diagnostics.total_scorer_calls)
        logging.debug(f"H-statistic:{h_statistic}")
        self.assertIsNotNone(h_statistic.explanations())
        self.assertEqual(1, len(h_statistic.explanations()))
        # multinomial
        self.assertEqual(
            self.classes + 3,  # + avg/sd/sem
            len(h_statistic.explanations()[("F", "G")]),
        )
        self.assertGreater(
            1, h_statistic.explanations()[("F", "G")][_pd.PD.LABEL_REGRESSION]
        )
        self.assertEqual(
            0.08272287837657792,
            h_statistic.explanations()[("F", "G")][_pd.PD.LABEL_REGRESSION],
        )
        self.assertEqual(
            0.08272287837657792,
            h_statistic.explanations()[("F", "G")][_pd.PD.COL_MEAN],
        )

    def test_2_features_resolution_regr(self):
        # GIVEN
        X = self.df5x5
        logging.debug(f"X:\n{X}")
        features = list("FG")

        # WHEN
        h_statistic = self._when_foo_predict_regr(features, X, None)

        # THEN
        self.assertIsNotNone(h_statistic)
        logging.debug(f"Scorer calls: {h_statistic.diagnostics.total_scorer_calls}")
        logging.debug(f"H-statistic:{h_statistic}")
        self.assertIsNotNone(h_statistic.explanations())
        self.assertEqual(1, len(h_statistic.explanations()))
        self.assertGreater(
            1, h_statistic.explanations()[("F", "G")][_pd.PD.LABEL_REGRESSION]
        )
        # IMPROVE paper calculation > assert H-stat value (many calculations)

    def test_regr_categorical(self):
        # GIVEN
        X = pandas.DataFrame(
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
        logging.debug(f"X:\n{X}")
        features = list("FG")

        # WHEN
        h_statistic = self.strategy.get_h_statistic().explain(
            features,
            X,
            predict_method=TestHStatistic.foo_predict_method_cat_regr,
        )

        # THEN
        self.assertIsNotNone(h_statistic)
        logging.debug(f"H-statistic:{h_statistic}")
        self.assertIsNotNone(h_statistic.explanations())
        self.assertEqual(1, len(h_statistic.explanations()))
        self.assertGreater(
            1, h_statistic.explanations()[("F", "G")][_pd.PD.LABEL_REGRESSION]
        )

    def test_interpretable_model_integration(self):
        logging.debug("# H-stat: interpretable model integration ###")

        # GIVEN
        X = self.df5x5
        logging.debug(f"X:\n{X}")
        features = list("FG")
        bins = [[10.0, 1.0], [2.0, 20.0]]
        # MLI: has a default working directory
        tmp_dir = tempfile.mkdtemp(prefix="h_stat_unit_ice_im_")
        mli = _mli.MLI(work_dir=tmp_dir, config=get_h2o3_config())
        i_model = _mli.InterpretableModel(
            mli,
            "IM for H-stat",
            predict_method=TestHStatistic.foo_predict_method_regr,
        )

        try:
            # WHEN
            h_statistic = _h_statistic.HStatistic("Test", i_model).explain(
                features, X, bins=bins
            )

            # THEN
            self.assertIsNotNone(h_statistic)
            logging.debug(f"H-statistic:{h_statistic}")
            self.assertIsNotNone(h_statistic.explanations())
            self.assertEqual(1, len(h_statistic.explanations()))
            self.assertGreater(
                1, h_statistic.explanations()[("F", "G")][_pd.PD.LABEL_REGRESSION]
            )
            self.assertEqual(
                0.08272287837657792,
                h_statistic.explanations()[("F", "G")][_pd.PD.LABEL_REGRESSION],
            )
        finally:
            rm_test_dir(tmp_dir)
