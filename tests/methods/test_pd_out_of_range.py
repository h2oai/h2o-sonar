# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import unittest
from functools import partial

import pandas as pd

from h2o_sonar import loggers as logging
from h2o_sonar.methods._pd import PD
from tests.methods.ice_pd_test_commons import FooScorerConstPredictFrame
from tests.methods.ice_pd_test_commons import FooScorerSumRegrFrame
from tests.methods.ice_pd_test_commons import IceStrategyFactory


class TestPdOutOfRange(unittest.TestCase):
    """Test Partial Dependency for out of range values."""

    def setUp(self):
        logging.setLevel(logging.DEBUG)

        self.strategy = IceStrategyFactory()

        # data: 3x3 for value checks
        self.x3x3 = pd.DataFrame({"f1": [1, 2, 3], "F": [1, 3, 5], "f2": [0.5, 2, 3]})
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
        self.x3x3const = pd.DataFrame(
            {"f1": [1, 2, 3], "fc": [42, 42, 42], "f2": [0.5, 2, 3]}
        )

        # prediction method lambda
        self.score_sum = partial(FooScorerSumRegrFrame().score_batch, fast_approx=True)

    def test_no_oor_by_default(self):
        # GIVEN
        logging.debug(f"X:\n{self.x3x3}")
        features = ["F"]

        # WHEN
        pdp = self.strategy.get_pd().explain(
            features, self.x3x3, predict_method=self.score_sum
        )

        # THEN
        self.assertIsNotNone(pdp.explanations())
        logging.debug(f"OORs:\n{pdp}")
        self.assertEqual(
            (4, 5), pdp.explanations()[features[0]][PD.LABEL_REGRESSION].shape
        )

    def test_custom_oor(self):
        # GIVEN
        logging.debug(f"X:\n{self.x3x3}")
        features = ["F"]
        oor_resolution = 5

        # WHEN
        pdp = self.strategy.get_pd().explain(
            features,
            self.x3x3,
            predict_method=self.score_sum,
            out_of_range_resolution=oor_resolution,
        )

        # THEN
        self.assertIsNotNone(pdp.explanations())
        logging.debug(f"OORs:\n{pdp}")
        oor = pdp.explanations()[features[0]][PD.LABEL_REGRESSION]
        self.assertEqual(5 + 2 * oor_resolution, len(oor.columns))
        logging.debug(f"OOR hints:\n{oor.loc[PD.COL_OOR]}")
        self.assertEqual(4, oor.shape[0])
        self.assertEqual(False, oor.loc[PD.COL_OOR].iloc[0])
        self.assertEqual(True, oor.loc[PD.COL_OOR].iloc[5])
        self.assertEqual(True, oor.loc[PD.COL_OOR].iloc[10])
        self.assertEqual(True, oor.loc[PD.COL_OOR].iloc[14])

    def test_oor_of_constant_feature_w_bins(self):
        # GIVEN
        df: pd.DataFrame = self.x3x3const
        logging.debug(f"X:\n{df}")
        features: list = ["fc"]
        bins: list = [[df[features[0]][0]]]
        oor_n: int = 3

        # WHEN
        pdp = self.strategy.get_pd().explain(
            features,
            df,
            bins=bins,
            predict_method=self.score_sum,
            out_of_range_resolution=oor_n,
        )

        # THEN
        self.assertIsNotNone(pdp.explanations())
        self.assertEqual(1, len(pdp.explanations()))
        logging.debug(f"OORs {oor_n}\n{pdp}")

    def test_oor_of_constant_feature(self):
        # GIVEN
        df: pd.DataFrame = self.x3x3const
        logging.debug(f"X:\n{df}")
        features: list = ["fc"]
        oor_n: int = 3

        # WHEN
        pdp = self.strategy.get_pd().explain(
            features,
            df,
            predict_method=self.score_sum,
            out_of_range_resolution=oor_n,
        )

        # THEN
        self.assertIsNotNone(pdp.explanations())
        logging.debug(f"OORs {oor_n}\n{pdp}")
        self.assertEqual(1, len(pdp.explanations()))
        self.assertEqual(
            1,
            len(pdp.explanations()[features[0]][PD.LABEL_REGRESSION].columns),
        )

    def test_oor_of_constant_features(self):
        # GIVEN
        df: pd.DataFrame = self.x3x3const
        logging.debug(f"X:\n{df}")
        features: list = ["fc", "f1"]
        oor_resolution: int = 3

        # WHEN
        pdp = self.strategy.get_pd().explain(
            features,
            df,
            predict_method=self.score_sum,
            out_of_range_resolution=oor_resolution,
        )

        # THEN
        self.assertIsNotNone(pdp.explanations())
        logging.debug(f"OORs {oor_resolution}\n{pdp}")
        self.assertEqual(2, len(pdp.explanations()))
        self.assertEqual(
            1,
            len(pdp.explanations()[features[0]][PD.LABEL_REGRESSION].columns),
        )
        self.assertEqual(
            3 + 2 * oor_resolution,
            len(pdp.explanations()[features[1]][PD.LABEL_REGRESSION].columns),
        )

    def test_no_bins_no_resolution_constant_feature_oor_only(self):
        # GIVEN
        df = self.x3x3const
        logging.debug(f"X:\n{df}")
        features = ["fc"]
        oor_n = 2

        # WHEN
        pdp = self.strategy.get_pd().explain(
            features,
            df,
            predict_method=self.score_sum,
            grid_resolution=0,
            out_of_range_resolution=oor_n,
        )

        # THEN
        self.assertIsNotNone(pdp.explanations())
        logging.debug(f"OORs {oor_n}\n{pdp}")
        self.assertEqual(1, len(pdp.explanations()))

    def test_no_bins_no_resolution_oor_only(self):
        # GIVEN
        df = self.x3x3
        logging.debug(f"X:\n{df}")
        features = ["F"]
        oor_n = 2

        # WHEN
        pdp = self.strategy.get_pd().explain(
            features,
            df,
            predict_method=self.score_sum,
            grid_resolution=0,
            out_of_range_resolution=oor_n,
        )

        # THEN
        self.assertIsNotNone(pdp.explanations())
        self.assertEqual(1, len(pdp.explanations()))
        logging.debug(f"OORs {oor_n}\n{pdp}")
        self.assertEqual(
            2 * oor_n,
            len(pdp.explanations()[features[0]][PD.LABEL_REGRESSION].columns),
        )
        self.assertEqual(
            4, pdp.explanations()[features[0]][PD.LABEL_REGRESSION].shape[0]
        )
        self.assertListEqual(
            [-1, -3, 7, 9],
            list(pdp.explanations()[features[0]][PD.LABEL_REGRESSION].columns.values),
        )
        oors = pdp.explanations()[features[0]][PD.LABEL_REGRESSION]
        self.assertEqual(2.833_333_333_333_333_5, oors.loc["mean"][-1])
        self.assertEqual(2.254_624_876_411_447, oors.loc["sd"][-3])
        self.assertEqual(1.301_708_279_317_775_9, oors.loc["sem"][7])

    def test_oor_with_bins(self):
        # GIVEN
        x = self.x18x3
        oor_n = 2
        features = ["f1"]
        bins = [[5, 10]]

        def foo_predict_method(x):
            logging.debug(f"Predict method X:\n{x}")
            y_hat = (x["f1"] + x["f2"]) / 100
            logging.debug(f"Predict method Y^:\n{y_hat}")
            return y_hat

        # WHEN
        pdp = self.strategy.get_pd()

        pdp.explain(
            features,
            x,
            out_of_range_resolution=oor_n,
            predict_method=foo_predict_method,
            bins=bins,
        )

        # THEN
        self.assertIsNotNone(pdp.explanations())
        self.assertEqual(1, len(pdp.explanations()))
        self.assertListEqual(
            [5, 10, 2, -1, 13, 16],
            list(pdp.explanations()[features[0]][PD.LABEL_REGRESSION].columns.values),
        )
        logging.debug(f"OORs n={oor_n}\n")
        logging.debug(f"Expected bins: \n{2 * oor_n + len(bins[0])}")
        logging.debug(f"PD\n{pdp}")
        self.assertEqual(
            2 * oor_n + len(bins[0]),
            len(pdp.explanations()[features[0]][PD.LABEL_REGRESSION].columns),
        )
        oors = pdp.explanations()[features[0]][PD.LABEL_REGRESSION]
        self.assertEqual(0.285_555_555_555_555_56, oors.loc["mean"][2])
        self.assertEqual(0.178_892_745_462_427_75, oors.loc["sd"][13])

        self.assertEqual(
            1 if pdp.opt_1_prediction else 2 * oor_n + len(bins[0]),
            pdp.diagnostics.total_scorer_calls,
        )

    def test_oor_and_residuals(self):
        # GIVEN
        logging.debug(f"X:\n{self.x3x3}")
        y = pd.DataFrame({"Y": [10, 10, 10]})
        logging.debug(f"Y:\n{y}")
        features = ["F"]

        # WHEN
        pdp = self.strategy.get_pd().explain(
            features, self.x3x3, Y=y, predict_method=self.score_sum
        )

        # THEN
        self.assertIsNotNone(pdp.explanations())
        self.assertEqual(
            5,
            len(pdp.explanations()[features[0]][PD.LABEL_REGRESSION].columns),
        )
        logging.debug(f"OORs:\n{pdp}")

    def test_oor_multinomial(self):
        # GIVEN
        df = self.x18x3
        oor_n = 2
        features = ["f1"]
        bins = [[5, 10]]
        scorer = FooScorerConstPredictFrame(df.shape[0], 3)

        # WHEN
        pdp = self.strategy.get_pd()
        pdp.explain(
            features,
            df,
            predict_method=scorer.score_batch,
            out_of_range_resolution=oor_n,
            bins=bins,
        )

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

        logging.debug(f"m-PD OOR bins: {exs[features[0]][class_0].columns.values}")
        self.assertListEqual(
            [5, 10, 2, -1, 13, 16],
            list(exs[features[0]][class_0].columns.values),
        )

        self.assertEqual((4, 6), exs[features[0]][class_0].shape)
        self.assertEqual(244.5, exs["f1"][class_1].loc["mean"][13])
        self.assertEqual(
            1.258_305_739_211_791_8,
            exs["f1"][class_1].loc["sem"][16],
        )

    def test_multidimensional_oor_bins(self):
        # GIVEN
        logging.debug(f"X:\n{self.x3x3}")
        features = [("f1", "f2"), "F"]
        x = pd.DataFrame({"f1": [1, 2, 3], "f2": [0.5, 2, 4], "F": [1, 3, 5]})
        bins = [([1, 3], [2, 4]), [3, 5]]

        # WHEN
        pdp = self.strategy.get_pd().explain(
            features,
            x,
            out_of_range_resolution=2,
            bins=bins,
            predict_method=self.score_sum,
        )

        # THEN
        logging.debug(f"OORs:\n{pdp}")
        self.assertIsNotNone(pdp.explanations())
        logging.debug(
            f"{str(features[0])} cols: "
            f"{pdp.explanations()[features[0]][PD.LABEL_REGRESSION].columns.values}"
        )
        self.assertEqual(
            4 + 32,
            len(pdp.explanations()[features[0]][PD.LABEL_REGRESSION].columns),
        )
        self.assertTrue(
            (5, 6.828_427_124_746_19)
            in pdp.explanations()[features[0]][PD.LABEL_REGRESSION].columns
        )
        self.assertEqual(
            2 + 2**2,
            len(pdp.explanations()[features[1]][PD.LABEL_REGRESSION].columns),
        )
        self.assertEqual(
            2,
            pdp.explanations()[features[1]][PD.LABEL_REGRESSION].columns[2],
        )
        self.assertEqual(
            7,
            pdp.explanations()[features[1]][PD.LABEL_REGRESSION].columns[5],
        )

    def test_multidimensional_oor(self):
        # GIVEN
        logging.debug(f"X:\n{self.x3x3}")
        features = [("f1", "f2"), "F"]
        x = pd.DataFrame({"f1": [1, 2, 3], "f2": [0.5, 2, 4], "F": [1, 3, 5]})

        # WHEN
        pdp = self.strategy.get_pd().explain(
            features,
            x,
            out_of_range_resolution=2,
            grid_resolution=2,
            predict_method=self.score_sum,
        )

        # THEN
        logging.debug(f"OORs:\n{pdp}")
        self.assertIsNotNone(pdp.explanations())
        logging.debug(
            f"{str(features[0])} cols: "
            f"{pdp.explanations()[features[0]][PD.LABEL_REGRESSION].columns.values}"
        )
        self.assertEqual(
            4, pdp.explanations()[features[0]][PD.LABEL_REGRESSION].shape[0]
        )
        self.assertEqual(
            4 + 32,
            len(pdp.explanations()[features[0]][PD.LABEL_REGRESSION].columns),
        )
        self.assertTrue(
            (5.0, 7.511_884_584_284_246)
            in pdp.explanations()[features[0]][PD.LABEL_REGRESSION].columns
        )
        self.assertEqual(
            2 + 2**2,
            len(pdp.explanations()[features[1]][PD.LABEL_REGRESSION].columns),
        )
        self.assertEqual(
            -1, pdp.explanations()[features[1]][PD.LABEL_REGRESSION].columns[2]
        )
        self.assertEqual(
            9, pdp.explanations()[features[1]][PD.LABEL_REGRESSION].columns[5]
        )

    @unittest.skip(
        "Flaky test: re-ordering of non-OOR categorical features is non-deterministic"
    )
    def test_oor_str_force_categorical(self):
        self._test_oor_force_categorical(
            feature="F",
            expected_bins=[
                "cat",
                "cat1",
                "dog",
                "1dog",
                "1sheep",
                "2cat",
                "2dog",
                "3cat",
                "cat5",
                "cat9",
                "UNSEEN",
            ],
        )

    def test_oor_int_force_categorical(self):
        self._test_oor_force_categorical(
            feature="f1", expected_bins=[3, 1, 2, 4, 5, 6, 0, -1, -2, 7, 8, 9]
        )

    def _test_oor_force_categorical(self, feature: str, expected_bins: list):
        # GIVEN
        x = self.x18x3
        features = [feature]
        features_meta = {"categorical": [feature]}

        # WHEN
        logging.debug(f"X:\n{x}")
        pdp = self.strategy.get_pd().explain(
            features=features,
            X=x,
            predict_method=self.score_sum,
            out_of_range_resolution=3,
            features_meta=features_meta,
        )

        # THEN
        self.assertIsNotNone(pdp.explanations())
        logging.debug(f"OORs:\n{pdp}")
        actual_bins = list(pdp.explanations()[features[0]][PD.LABEL_REGRESSION].columns)
        logging.debug(f"OOR bins:\n{actual_bins}")
        self.assertEqual(
            (4, len(expected_bins)),
            pdp.explanations()[features[0]][PD.LABEL_REGRESSION].shape,
        )
        self.assertEqual(expected_bins, actual_bins)

    def test_oor_datetime_date(self):
        feature = "f_date"
        self._test_oor_datetime(
            feature=feature,
            features_meta={
                PD.KEY_DATE_FEATURES: [feature],
                PD.KEY_DATE_FEATURES_FORMAT: ["%Y-%m-%d"],
            },
            expected_bins=[
                "2016-04-04",
                "2016-10-30",
                "2017-05-28",
                "2017-12-24",
                "2018-07-21",
                "2019-02-16",
                "2019-09-14",
                "2020-04-10",
                "2020-11-06",
                "2021-06-04",
            ],
        )

    def test_oor_datetime_str(self):
        feature = "f_date"
        self._test_oor_datetime(
            feature=feature,
            features_meta=None,
            expected_bins=[
                "2016-04-04",
                "2017-08-19",
                "2019-05-09",
                "2021-06-04",
                "UNSEEN",
            ],
        )

    def _test_oor_datetime(self, feature, features_meta, expected_bins):
        # GIVEN
        x = pd.DataFrame(
            {
                feature: [
                    "2017-08-19",
                    "2016-04-04",
                    "2019-05-09",
                    "2021-06-04",
                ],
                "f_str": [
                    "2017-08-19",
                    "2016-04-04",
                    "2019-05-09",
                    "2021-06-04",
                ],
            }
        )
        features = [feature]

        # WHEN
        logging.debug(f"X:\n{x}")
        pdp = self.strategy.get_pd().explain(
            features=features,
            X=x,
            predict_method=self.score_sum,
            out_of_range_resolution=3,
            features_meta=features_meta,
        )

        # THEN
        self.assertIsNotNone(pdp.explanations())
        logging.debug(f"OORs:\n{pdp}")
        actual_bins = list(pdp.explanations()[features[0]][PD.LABEL_REGRESSION].columns)

        logging.debug(f"OOR bins:\n{actual_bins}")
        self.assertEqual(
            (4, len(expected_bins)),
            pdp.explanations()[features[0]][PD.LABEL_REGRESSION].shape,
        )
        self.assertEqual(expected_bins, actual_bins)
