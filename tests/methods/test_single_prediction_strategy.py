# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import unittest
from functools import partial

import datatable as dt
import pandas as pd

from h2o_sonar import loggers as logging
from h2o_sonar.methods._ice import ICE
from tests.methods.ice_pd_test_commons import FooScorerSumRegrFrame
from tests.methods.ice_pd_test_commons import IceStrategyFactory


class TestSinglePredictionStrategy(unittest.TestCase):
    """Test single vs. per-bin predict method invocation strategy for
    H-statistic/PD/ICE.

    Default ICE strategy is tested by regular test cases. This test case
    switches ICE strategy to NON-DEFAULT and runs selected tests from other
    test cases to ensure that computation is correct.

    """

    def setUp(self):
        logging.setLevel(logging.DEBUG)

        # prediction
        self.score_sum = partial(FooScorerSumRegrFrame().score_batch, fast_approx=True)

    def _ice_for_1_row(self, strategy):
        # GIVEN
        x = pd.DataFrame({"F": [1, 1, 1], "G": [1, 1, 1], "H": [1, 1, 1]})
        bins = [[10, 20], [30]]
        features = ["F", "G"]

        # WHEN
        ice = ICE("I")
        ice.opt_1_prediction = strategy
        ice.explain(features, x, predict_method=self.score_sum, bins=bins)
        exs = ice.explanations()

        # THEN
        logging.debug(f"{ice}")
        self.assertIsNotNone(exs, "Explanations must be cached")
        self.assertEqual(
            len(features),
            len(exs),
            "Target features size and result size must fit",
        )

        exs = exs[features[0]][ICE.LABEL_REGRESSION]
        logging.debug(f"ICE explanation for {features[0]}:\n{exs}")
        self.assertIsInstance(exs, pd.DataFrame)
        self.assertEqual(exs.shape[0], 3)
        self.assertEqual(exs.shape[1], len(bins[0]))

        self.assertEqual(
            1 if ice.opt_1_prediction else 3, ice.diagnostics.total_scorer_calls
        )

    def test_sanity_allow_one_predict(self):
        self._ice_for_1_row(IceStrategyFactory.OPT_1_PREDICT_STRATEGY)

    def test_sanity_forbid_one_predict(self):
        self._ice_for_1_row(IceStrategyFactory.OPT_MULTIPLE_PREDICTS_STRATEGY)

    @staticmethod
    def predict_method_dt_friendly(x: pd.DataFrame):
        """The purpose of this predict method is to verify X's datatable
        friendliness.

        """
        logging.debug(f"Predict X.dtypes:\n{x.dtypes}")
        logging.debug(f"Predict X:\n{x}")

        x_dt = dt.Frame(x)
        assert x.shape == x_dt.shape

        return x["F"]

    def test_1_frame_datatable_friendliness(self):
        """The purpose of this test is to determine how to construct 1
        frame in Pandas so that datatable is able to convert it on work with it.

        """
        # GIVEN
        x = pd.DataFrame(
            {
                "F": [1, 2, 3, 4, 5, 6],
                "G": ["cat", "dog", "cat", "sheep", "cat", "dog"],
                "H": [50, 40, 30, 20, 10, 0],
            }
        )
        logging.debug(f"X.dtypes:\n{x.dtypes}")
        bins = [[1, 3, 6], ["cat", "dog", "sheep"], [0, 30, 50]]
        features = ["F", "G", "H"]

        # WHEN
        ice = ICE("I")
        ice.opt_1_prediction = IceStrategyFactory.OPT_1_PREDICT_STRATEGY
        ice.explain(
            features,
            x,
            predict_method=self.predict_method_dt_friendly,
            bins=bins,
        )
        exs = ice.explanations()

        # THEN
        logging.debug(f"{ice}")
        self.assertIsNotNone(exs, "Explanations must be cached")
        self.assertEqual(
            len(features),
            len(exs),
            "Target features size and result size must fit",
        )

        exs = exs[features[0]][ICE.LABEL_REGRESSION]
        logging.debug(f"ICE explanation for {features[0]}:\n{exs}")
        self.assertIsInstance(exs, pd.DataFrame)

    def test_ice_regression(self):
        from tests.methods.test_ice_regression import TestIceRegression

        test = TestIceRegression()
        test.setUp()
        test.strategy.set_non_default_ice_strategy()

        test.test_default_resolution()
        test.test_no_bins_no_resolution()
        test.test_predict_method_return_values()
        test.test_resolution()
        test.test_run_pandas_many_instances_many_features()
        test.test_run_pandas_single_instance_single_feature()

    def test_ice_multinomial(self):
        from tests.methods.test_ice_multinomial import TestIceMultinomial

        test = TestIceMultinomial()
        test.setUp()
        test.strategy.set_non_default_ice_strategy()

        test.test_binomial_single_feature_many_instances()
        test.test_multinomial_multi_feature_many_instances()
        test.test_multinomial_multi_fs_many_is_residuals_json()

    def test_ice_multidimensional_and_residuals(self):
        from tests.methods.test_ice_n_dim import TestMultiDimensionalIce

        test = TestMultiDimensionalIce()
        test.setUp()
        test.strategy.set_non_default_ice_strategy()

        test.test_ice_mixed_regr()
        test.test_ice_multinomial_residuals()
        test.test_ice_multinomial_residuals_bins()
        test.test_ice_regr()
        test.test_ice_regr_bins()

    def test_ice_oor(self):
        from tests.methods.test_ice_out_of_range import TestIceOutOfRange

        test = TestIceOutOfRange()
        test.setUp()
        test.strategy.set_non_default_ice_strategy()

        test.test_no_bins_no_resolution_oor_only()
        test.test_oor_bins()
        test.test_default_oor()
        test.test_custom_oor()
        test.test_multidimensional_oor()
        test.test_multidimensional_oor_bins()
        test.test_multinomial_oor()

    def test_pd_regression(self):
        from tests.methods.test_pd_regression import TestPdRegression

        test = TestPdRegression()
        test.setUp()
        test.strategy.set_non_default_ice_strategy()

        test.test_predict_method_return_values()
        test.test_json_bins_type_conversion_driven_by_feature_type()
        test.test_no_bins_no_resolution_no_y_no_nothing()
        test.test_run_pandas_many_features_and_bins()
        test.test_run_pandas_many_features_grid_bins()
        test.test_run_pandas_single_feature()

    def test_pd_multinomial(self):
        from tests.methods.test_pd_multinomial import TestPdMultinomial

        test = TestPdMultinomial()
        test.setUp()
        test.strategy.set_non_default_ice_strategy()

        test.test_binomial_single_feature_many_instances()
        test.test_multinomial_ice_cache()
        test.test_multinomial_ice_filter_n_cache()
        test.test_multinomial_multiple_f_many_i_residuals()
        test.test_multinomial_multiple_feature_many_instances()

    def test_pd_multidimensional(self):
        from tests.methods.test_pd_n_dim import TestMultiDimensionalPd

        test = TestMultiDimensionalPd()
        test.setUp()
        test.strategy.set_non_default_ice_strategy()

        test.test_pd_multinomial()
        test.test_n_dim_pd_mixed_regr_center()
        test.test_n_dim_regr_categorical()
        test.test_pd_mixed_regr()
        test.test_pd_multinomial_residuals()
        test.test_pd_multinomial_residuals_bins()
        test.test_pd_regr()
        test.test_pd_regr_bins()

    def test_pd_residuals(self):
        from tests.methods.test_pd_residuals import TestPdResiduals

        test = TestPdResiduals()
        test.setUp()
        test.strategy.set_non_default_ice_strategy()

        test.test_abs()
        test.test_identity()
        test.test_multinomial()
        test.test_json()

    def test_pd_oor(self):
        from tests.methods.test_pd_out_of_range import TestPdOutOfRange

        test = TestPdOutOfRange()
        test.setUp()
        test.strategy.set_non_default_ice_strategy()

        test.test_multidimensional_oor_bins()
        test.test_multidimensional_oor()
        test.test_custom_oor()
        test.test_no_bins_no_resolution_oor_only()
        test.test_no_oor_by_default()
        test.test_oor_and_residuals()
        test.test_oor_multinomial()
        test.test_oor_with_bins()

    def test_robustness(self):
        from tests.methods.test_pd_robustness import TestPdIceRobustness

        test = TestPdIceRobustness()
        test.setUp()
        test.strategy.set_non_default_ice_strategy()

        test.test_ice_cleaned_single_feature_titanic()
        test.test_pd_cleaned_single_feature_titanic()
        test.test_pd_loan()
        test.test_pd_raw_all_features_titanic()
        test.test_pd_raw_cat_feature_titanic()

    def test_bins_sorting(self):
        from tests.methods.test_pd_ice_bin_sorting import TestPdIceBinSorting

        test = TestPdIceBinSorting()
        test.setUp()
        test.strategy.set_non_default_ice_strategy()

        test.test_ice_cat_sort()
        test.test_ice_no_sort()
        test.test_ice_num_in_cat_sort()
        test.test_ice_num_sort()
        test.test_pd_bin_num_sort()
        test.test_pd_cat_sort()
        test.test_pd_n_dim_regr_cat()
        test.test_pd_num_in_cat_sort()
        test.test_pd_oor_num_sort()

    def test_h_stat(self):
        from tests.methods.test_h_statistic import TestHStatistic

        test = TestHStatistic()
        test.setUp()
        test.strategy.set_non_default_ice_strategy()

        test.test_2_features_2x2_values_multinomial()
        test.test_2_features_2x2_values_regr_bins()
        test.test_2_features_2x3_values_regr_bins()
        test.test_2_features_2x3_values_regr_bins()
        test.test_2_features_3x3_values_regr_bins()
        test.test_2_features_resolution_regr()
        test.test_3_features_2x2_values_regr_bins()
        test.test_3_features_all_values_regr_bins()
        test.test_regr_categorical()
