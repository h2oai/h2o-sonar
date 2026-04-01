# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import os
import tempfile
from functools import partial

import pandas as pd
import pytest

from h2o_sonar import loggers as logging
from h2o_sonar.errors import MliError
from h2o_sonar.methods._ice import ICE
from h2o_sonar.methods.core._mli import InterpretableModel
from h2o_sonar.methods.core._mli import MLI
from tests import test_utils
from tests.base_h2o_test import BaseH2OTest
from tests.conftest import get_h2o3_config
from tests.methods.ice_pd_test_commons import FooScorerRegrSeries
from tests.methods.ice_pd_test_commons import FooScorerSumRegrFrame
from tests.methods.ice_pd_test_commons import IceStrategyFactory


try:
    import h2o

    HAS_H2O = True
except ImportError:
    HAS_H2O = False


class TestIceRegression(BaseH2OTest):
    """Test Individual Conditional Expectation (ICE) implementation.

    Inherits from BaseH2OTest which provides automatic H2O-3 cleanup
    after each test method and after the entire test class completes.
    """

    def setUp(self):
        super().setUp()
        logging.setLevel(logging.DEBUG)

        self.strategy = IceStrategyFactory()

        # data
        self.df2x2 = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
        # data: 3x3 for value checks
        self.df3x3 = pd.DataFrame({"f1": [1, 2, 3], "F": [1, 3, 5], "f2": [1, 2, 3]})
        # data: 1x5 dataframe
        self.df1x5 = pd.DataFrame([[i for i in range(1, 6)]], columns=list("abFdY"))
        # data: 5x5 dataframe w/ i..i+5 rows, F is target feature
        self.df5x5 = pd.DataFrame(
            [[i + j for i in range(1, 6)] for j in range(0, 5)],
            columns=list("abFdY"),
        )

        # prediction
        self.score_foo = partial(FooScorerRegrSeries().score_batch, fast_approx=True)
        self.score_sum = partial(FooScorerSumRegrFrame().score_batch, fast_approx=True)

        # test persistence in the current directory (avoid making garbage)
        self.test_current_dir_persistence = True
        # visual check
        self.test_visual_check = False

    def test_negative_init(self):
        with self.assertRaises(ValueError):
            ICE(None)
        with self.assertRaises(ValueError):
            ICE("")

        with self.assertRaises(ValueError):
            ICE("No min").explain(
                ["F"], self.df3x3, predict_method=self.score_foo, maxs=[1]
            )
            ICE("!= min").explain(
                ["F1", "F2"],
                self.df3x3,
                predict_method=self.score_foo,
                maxs=[1, 2],
                mins=[3],
            )
            ICE("No max").explain(
                ["F"], self.df3x3, predict_method=self.score_foo, mins=[1]
            )
            ICE("!= max").explain(
                ["F1", "F2"],
                self.df3x3,
                predict_method=self.score_foo,
                maxs=[1],
                mins=[3, 4],
            )

    def test_init_name_n_type(self):
        # GIVEN
        name = "Name"

        # WHEN
        ice = ICE(name)

        # THEN
        self.assertEqual(ice.method_name, name)
        self.assertEqual(ice.method_type, "ice")

    def test_non_existent_feature(self):
        # GIVEN
        name = "Name"
        fs = ["NONEXISTENT", "F"]

        # THEN
        with self.assertRaises(ValueError):
            # WHEN
            ICE(name).explain(
                fs,
                self.df3x3,
                bins=[[1, 2], [2, 3]],
                predict_method=self.score_foo,
            )

    def test_init_name_n_type_feature(self):
        # GIVEN
        name = "Name"
        fs = ["F"]

        # WHEN
        ice = ICE(name).explain(
            fs, self.df3x3, bins=[[1]], predict_method=self.score_foo
        )

        # THEN
        self.assertEqual(ice.features, fs)

    @pytest.mark.skipif(
        not HAS_H2O,
        reason="H2O Python package is not installed",
    )
    def test_run_negative(self):
        # data
        empty_data = pd.DataFrame()
        bins = pd.DataFrame()

        # negative feature(s)
        with self.assertRaises(ValueError):
            ICE("Name").explain(None, self.df2x2, predict_method=self.score_foo)
        with self.assertRaises(ValueError):
            ICE("Name").explain([], self.df2x2, predict_method=self.score_foo)
        # negative scorer
        with self.assertRaises((ValueError, TypeError)):
            ICE("Name").explain(["F"], None, self.df2x2, bins=bins)
        # negative data
        with self.assertRaises((ValueError, TypeError)):
            ICE("Name").explain(["F"], None, predict_method=self.score_foo, bins=bins)
        with self.assertRaises((ValueError, TypeError)):
            ICE("Name").explain(
                ["F"], empty_data, predict_method=self.score_foo, bins=bins
            )
        # features and column labels mismatch
        with self.assertRaises(ValueError):
            ICE("Name").explain(["WRONG"], self.df2x2, predict_method=self.score_foo)
        with self.assertRaises(ValueError):
            h2oframe = h2o.H2OFrame(self.df2x2)
            ICE("Name").explain(["WRONG"], h2oframe, predict_method=self.score_foo)

    def test_assert_ice_values_single_feature_many_instances(self):
        """Basic ICE test."""
        logging.debug("# ICE@Pandas: Single feature, many instances ###")

        # GIVEN
        bins = [[1, 3, 5]]
        features = ["F"]

        # WHEN
        _, explanations = self._when_ice(features, self.df3x3, bins)

        # THEN
        self.assertIsNotNone(explanations, "Explanations must be cached")
        self.assertEqual(
            len(features),
            len(explanations),
            "Target features size and result size must fit",
        )

        explanation = explanations[features[0]][ICE.LABEL_REGRESSION]
        logging.debug(f"ICE explanation for {features[0]}:\n{str(explanation)}")
        self.assertIsInstance(explanation, pd.DataFrame, "Result to be dictionary")
        self.assertEqual(
            explanation.shape[0], 3, "3 instances, 3 ICE explanations in rows"
        )
        self.assertEqual(
            explanation.shape[1],
            len(bins[0]),
            "3 bins, 3 ICE values each explanation",
        )

        # assert actual values
        ice_target = pd.DataFrame({1: [3, 5, 7], 3: [5, 7, 9], 5: [7, 9, 11]})
        for c in ice_target.columns.values:
            for r in ice_target.index.values:
                self.assertEqual(
                    ice_target[c][r],
                    explanation[c][r],
                    "Calculated ICE value are incorrect",
                )
        # loop above replaces assert below: The truth value of a DF is ambiguous
        # self.assertEqual(ice_target, explanation, "ICE values are incorrect")

        if self.test_visual_check:
            # visual check
            import matplotlib.pyplot as plt

            explanation.transpose().plot.line()
            plt.show()

    def test_run_pandas_single_instance_single_feature(self):
        logging.debug("# ICE@Pandas: Single feature, single instance ###")

        # GIVEN
        logging.debug("ICE data:\n" + self.df1x5.to_string())
        bins = [[1, 2, 3]]
        features = ["F"]

        # WHEN
        _, explanations = self._when_ice(features, self.df1x5, bins)

        # THEN
        self.assertIsNotNone(explanations, "Explanations must be cached")
        self.assertIsInstance(explanations, dict, "Result to be dictionary")

        explanation = explanations[features[0]][ICE.LABEL_REGRESSION]
        logging.debug(f"ICE explanation for {features[0]}:\n{str(explanation)}")
        self.assertIsInstance(explanation, pd.DataFrame, "Result to be Pandas")
        self.assertEqual(
            explanation.shape[0], 1, "1 instances, 1 ICE explanation in 1 row"
        )
        self.assertEqual(
            explanation.shape[1],
            len(bins[0]),
            "3 bins, 3 ICE values each explanation",
        )

    def test_run_pandas_many_instances_many_features(self):
        logging.debug("# ICE@Pandas: Many features, may instances ###")

        # GIVEN
        logging.debug("ICE data:\n" + self.df5x5.to_string())
        bins = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        target_features = ["b", "F", "d"]

        # WHEN
        _, explanations = self._when_ice(target_features, self.df5x5, bins)

        # THEN
        self.assertIsNotNone(explanations, "Explanations must be cached")
        self.assertIsInstance(explanations, dict, "Result to be dictionary")
        self.assertEqual(
            len(target_features),
            len(explanations),
            "Target features size and result size must fit",
        )
        for f in target_features:
            logging.debug(
                f"ICE explanation for {f} is:\n"
                f"{str(explanations[f][ICE.LABEL_REGRESSION])}"
            )
            self.assertIsInstance(
                explanations[f][ICE.LABEL_REGRESSION],
                pd.DataFrame,
                "Result to be Pandas",
            )
            self.assertEqual(
                explanations[f][ICE.LABEL_REGRESSION].shape[0],
                5,
                "5 instances, 5 ICE explanations in a row",
            )
            self.assertEqual(
                explanations[f][ICE.LABEL_REGRESSION].shape[1],
                len(bins),
                "3 bins, 3 ICE values each explanation",
            )

    def _when_ice(self, target_features, df, bins):
        if isinstance(df, pd.DataFrame):
            logging.debug("ICE data:\n" + df.to_string())

        # ICE
        ice = self.strategy.get_ice()
        i = ice.explain(target_features, df, predict_method=self.score_sum, bins=bins)

        self.assertIsNotNone(i, "Run must return ICE instance")
        self.assertIsInstance(i, ICE, "Run must return ICE instance")

        return ice, ice.explanations()

    def test_no_bins_no_resolution(self):
        with self.assertRaises(ValueError):
            ICE("Name").explain(
                ["F"],
                self.df2x2,
                predict_method=self.score_foo,
                bins=None,
                grid_resolution=None,
            )

    def test_resolution(self):
        logging.debug("# ICE: RESOLUTION @ single feature, single instance ###")

        # GIVEN
        logging.debug("ICE data:\n" + self.df5x5.to_string())
        bins = [[1, 2, 3]]
        features = ["F"]
        ice = self.strategy.get_ice()

        # WHEN
        # NOTHING (no bins, no resolution)
        my_5x5 = self.df5x5.copy()
        my_5x5[features[0]] = [1, 50, 70, 90, 100]
        mins = [my_5x5[features[0]].min()]
        maxs = [my_5x5[features[0]].max()]
        logging.debug("ICE zoom data:\n" + my_5x5.to_string())
        exs = ice.explain(
            features,
            my_5x5,
            mins=mins,
            maxs=maxs,
            predict_method=self.score_sum,
        ).explanations()
        # THEN
        self.assertIsNotNone(exs, "Explanations must be cached")
        self.assertIsInstance(exs, dict, "Result to be dictionary")
        ex = exs[features[0]][ICE.LABEL_REGRESSION]
        self.assertEqual(ex.shape[0], 5, "5 instances > 5 ICE expl. > 5 row")
        self.assertEqual(ex.shape[1], 10, "100/10 > 10 ICE values / row")

        # WHEN
        # bins ONLY
        exs = ice.explain(
            features, self.df5x5, predict_method=self.score_sum, bins=bins
        ).explanations()
        # THEN
        self.assertIsNotNone(exs, "Explanations must be cached")
        self.assertIsInstance(exs, dict, "Result to be dictionary")
        ex = exs[features[0]][ICE.LABEL_REGRESSION]
        self.assertEqual(ex.shape[0], 5, "5 instances > 5 ICE expl. > 5 row")
        self.assertEqual(ex.shape[1], len(bins[0]), "3 bins > 3 ICE values / row")

        # WHEN
        # resolution ONLY
        resolution = 12
        exs = ice.explain(
            features,
            my_5x5,
            predict_method=self.score_sum,
            mins=mins,
            maxs=maxs,
            grid_resolution=resolution,
        ).explanations()
        # THEN
        self.assertIsNotNone(exs, "Explanations must be cached")
        self.assertIsInstance(exs, dict, "Result to be dictionary")
        ex = exs[features[0]][ICE.LABEL_REGRESSION]
        self.assertEqual(ex.shape[0], 5, "5 instances > 5 ICE expl. > 5 row")
        self.assertEqual(ex.shape[1], resolution, "resolution ICE values / row")

        # WHEN
        # bins and resolution (bins win)
        exs = ice.explain(
            features,
            self.df5x5,
            predict_method=self.score_sum,
            bins=bins,
            grid_resolution=2,
        ).explanations()
        # THEN
        self.assertIsNotNone(exs, "Explanations must be cached")
        self.assertIsInstance(exs, dict, "Result to be dictionary")
        ex = exs[features[0]][ICE.LABEL_REGRESSION]
        self.assertEqual(ex.shape[0], 5, "5 instances > 5 ICE expl. > 5 row")
        self.assertEqual(ex.shape[1], len(bins[0]), "3 bins > 3 ICE values / row")

        # negative
        with self.assertRaises(ValueError):
            ice.explain(
                features,
                self.df5x5,
                predict_method=self.score_sum,
                grid_resolution=-1,
            )
        with self.assertRaises(ValueError):
            ice.explain(
                features,
                self.df5x5,
                predict_method=self.score_sum,
                grid_resolution=0,
            )

            # big resolution
            resolution = 1000
            exs = ice.explain(
                features,
                self.df5x5,
                predict_method=self.score_sum,
                grid_resolution=resolution,
            ).explanations()
            self.assertIsNotNone(exs, "Explanations must be cached")
            self.assertIsInstance(exs, dict, "Result to be dictionary")
            ex = exs[features[0]][ICE.LABEL_REGRESSION]
            self.assertEqual(ex.shape[0], 5, "5 instances > 5 ICE expl. > 5 row")
            self.assertEqual(ex.shape[1], resolution, "res > 1000 ICE values / row")

            # min == max
            my_5x5 = self.df5x5.copy()
            my_5x5[features[0]] = [10, 10, 10, 10, 10]
            logging.debug("ICE min==max data:\n" + my_5x5.to_string())
            exs = ice.explain(
                features,
                my_5x5,
                predict_method=self.score_sum,
                grid_resolution=3,
            ).explanations()
            self.assertIsNotNone(exs, "Explanations must be cached")
            self.assertIsInstance(exs, dict, "Result to be dictionary")
            ex = exs[features[0]][ICE.LABEL_REGRESSION]
            self.assertEqual(ex.shape[0], 5, "5 instances > 5 ICE expl. > 5 row")
            self.assertEqual(ex.shape[1], 1, "(10,10,3)->[10] = 1 ICE value / row")

    def test_explanations_negative(self):
        ice = ICE("Negative explanations")
        with self.assertRaises(MliError):
            ice.explanations()

    @pytest.mark.skipif(
        not HAS_H2O,
        reason="H2O Python package is not installed",
    )
    def test_h2oframe_many_feature(self):
        logging.debug("# ICE@H2OFrame: Many features, many instances ###")

        # GIVEN
        # H2OFrame data
        df = h2o.H2OFrame(self.df5x5)
        bins = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        target_features = ["b", "F", "d"]

        # WHEN
        # run Pandas runner w/ h2oframe
        _, explanations = self._when_ice(target_features, df, bins)

        # THEN
        self.assertIsNotNone(explanations, "Explanations must be cached")

        self.assertIsInstance(explanations, dict, "Result to be dictionary")
        self.assertEqual(
            len(target_features),
            len(explanations),
            "Target features size and result size must fit",
        )
        for f in target_features:
            logging.debug(
                f"ICE explanation for {f} is:\n"
                f"{str(explanations[f][ICE.LABEL_REGRESSION])}"
            )
            self.assertIsInstance(
                explanations[f][ICE.LABEL_REGRESSION],
                pd.DataFrame,
                "Result to be Pandas",
            )
            self.assertEqual(
                explanations[f][ICE.LABEL_REGRESSION].shape[0],
                5,
                "5 instances, 5 ICE explanations in a row",
            )
            self.assertEqual(
                explanations[f][ICE.LABEL_REGRESSION].shape[1],
                len(bins),
                "3 bins, 3 ICE values each methods",
            )

        logging.debug("# DONE ICE@H2OFrame: Many features, many instances ###")

    def test_save_and_load_json(self):
        # GIVEN
        bins = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        target_features = ["b", "F", "d"]
        ice, explanations = self._when_ice(target_features, self.df5x5, bins)

        # WHEN
        # save
        logging.debug("SAVING JSon")
        tmp_dir = tempfile.mkdtemp(prefix="mli_unit_ice_")
        tmp_file_name = os.path.join(tmp_dir, "unit-ice-save-basic.json")

        try:
            ice.save_json(tmp_file_name)

            # NEGATIVE: save check overwrite
            with self.assertRaises(FileExistsError):
                ice.save_json(tmp_file_name)

            # load
            logging.debug("LOADING JSon...")
            ice.evict_explanations()
            ice.load_json(tmp_file_name)
            l_explanations = ice.explanations()

            # THEN
            self.assertIsInstance(explanations, dict, "Result to be dictionary")
            for f in l_explanations:
                logging.debug(
                    f"ICE explanation for {f} is:\n"
                    f"{str(l_explanations[f][ICE.LABEL_REGRESSION])}"
                )

            # assert save/load roundtrip
            self.assertEqual(
                len(explanations),
                len(l_explanations),
                "Explanations cannot have different size after SAVE/LOAD",
            )
            self.assertEqual(
                explanations.keys(),
                l_explanations.keys(),
                "Explanations keys cannot be different after SAVE/LOAD",
            )
            for f in l_explanations:
                logging.debug(f"SRC: {f}\n{str(explanations[f][ICE.LABEL_REGRESSION])}")
                logging.debug(
                    f"S/L: {f}\n{str(l_explanations[f][ICE.LABEL_REGRESSION])}"
                )
                self.assertEqual(
                    explanations[f][ICE.LABEL_REGRESSION].shape,
                    l_explanations[f][ICE.LABEL_REGRESSION].shape,
                )
                for c in explanations[f][ICE.LABEL_REGRESSION].columns.values:
                    for r in explanations[f][ICE.LABEL_REGRESSION].index.values:
                        self.assertEqual(
                            explanations[f][ICE.LABEL_REGRESSION][c][r],
                            l_explanations[f][ICE.LABEL_REGRESSION][c][r],
                        )
        finally:
            test_utils.rm_test_dir(tmp_dir)

        # NEGATIVE: load non-existent file
        with self.assertRaises(FileNotFoundError):
            ice.load_json("/THIS-IS-NON-EXISTENT-FILE")

        # save/load to/from CURRENT dir
        if self.test_current_dir_persistence:
            tmp_file = os.path.join(os.getcwd(), ice.default_json_file_name)
            try:
                # save to current dir
                logging.debug("SAVING to current directory: " + os.getcwd())
                ice.save_json()
                # load from current dir
                logging.debug("LOADING from current directory: " + os.getcwd())
                ice.load_json()
                self.assertIsInstance(explanations, dict, "Result to be dictionary")
                self.assertEqual(len(explanations), len(ice.explanations()))
            finally:
                os.remove(tmp_file)

    def test_predict_method_return_values(self):
        # predict method which returns pd Series
        # WHEN
        ice = self.strategy.get_ice().explain(
            ["f1", "f2"],
            self.df3x3,
            bins=[[1], [2]],
            predict_method=lambda x: pd.Series(x["F"]),
        )
        # THEN
        self.assertIsNotNone(ice)
        logging.debug(f"Series:\n{ice}")
        self.assertEqual(2, len(ice.explanations()))
        self.assertIsInstance(
            ice.explanations()["f1"][ICE.LABEL_REGRESSION],
            pd.DataFrame,
            "Result to be DataFrame",
        )
        self.assertEqual(3, ice.explanations()["f1"][ICE.LABEL_REGRESSION][1][1])

        # predict method which returns pd DataFrame 1 column
        # WHEN
        ice = self.strategy.get_ice().explain(
            ["f1", "f2"],
            self.df3x3,
            bins=[[1], [2]],
            predict_method=lambda x: x["F"],
        )
        # THEN
        self.assertIsNotNone(ice)
        logging.debug(f"Frame[1]:\n{ice}")
        self.assertIsInstance(
            ice.explanations()["f1"][ICE.LABEL_REGRESSION],
            pd.DataFrame,
            "Result to be DataFrame",
        )
        self.assertEqual(3, ice.explanations()["f1"][ICE.LABEL_REGRESSION][1][1])

    def test_default_resolution(self):
        # WHEN
        ice = self.strategy.get_ice()
        f1_min = 1
        f1_max = 3
        exs = ice.explain(
            ["f1", "f2"],
            self.df3x3,
            mins=[f1_min, 1],
            maxs=[f1_max, 3],
            predict_method=lambda x: pd.Series(x["F"]),
        )

        # THEN
        self.assertIsNotNone(exs)
        logging.debug(f"ICE:\n{exs}")
        logging.debug(
            f"Columns:\n{exs.explanations()['f1'][ICE.LABEL_REGRESSION].columns.values}"
        )
        self.assertEqual(2, len(ice.explanations()))
        self.assertEqual(
            f1_max - f1_min + 1,
            len(exs.explanations()["f1"][ICE.LABEL_REGRESSION].columns),
        )
        self.assertEqual(
            1, exs.explanations()["f1"][ICE.LABEL_REGRESSION].columns.values[0]
        )
        self.assertEqual(
            3, exs.explanations()["f1"][ICE.LABEL_REGRESSION].columns.values[2]
        )
        self.assertEqual(3, exs.explanations()["f1"][ICE.LABEL_REGRESSION][1][1])

        logging.debug(
            f"Diagnostics: predict method invocations "
            f"{ice.diagnostics.total_scorer_calls} ~ "
            f"{ice.diagnostics.scorer_calls_history}"
        )
        if not self.strategy.is_1_predict():
            self.assertEqual(
                2 * self.df3x3.shape[1],
                ice.diagnostics.total_scorer_calls,
            )
        else:
            self.assertEqual(1, ice.diagnostics.total_scorer_calls)

    def test_interpretable_model_integration(self):
        logging.debug("# ICE: interpretable model integration ###")

        # GIVEN
        x = self.df5x5
        target_features = ["F"]
        mins = [self.df5x5[target_features[0]].min()]
        maxs = [self.df5x5[target_features[0]].max()]

        # MLI: has a default working directory
        tmp_dir = tempfile.mkdtemp(prefix="mli_unit_ice_im_")
        mli = MLI(work_dir=tmp_dir, config=get_h2o3_config())
        i_model = InterpretableModel(mli, "IM for ICE", predict_method=self.score_foo)

        # A) predict() and working dir used from IM
        ice = None
        try:
            # WHEN
            ice = ICE("IM based", i_model)
            exs = (
                ice.explain(target_features, x, mins=mins, maxs=maxs)
                .save_json()
                .explanations()
            )

            # THEN
            self.assertEqual(len(exs), 1)
            # 5 instances, no bins specified > resolutions makes 10 bins
            self.assertEqual(
                exs[target_features[0]][ICE.LABEL_REGRESSION].shape,
                (5, self.df5x5.shape[1]),
            )
        finally:
            os.remove(os.path.join(tmp_dir, ice.default_json_file_name))

        # B) predict() override (doesn't have to be specified on IM)
        try:
            # WHEN
            ice = ICE("Predict override", i_model)
            ice.explain(
                target_features,
                x,
                mins=mins,
                maxs=maxs,
                predict_method=self.score_foo,
            ).save_json()
        finally:
            test_utils.rm_test_dir(tmp_dir)

        # C) no IM, all override (all other tests in this test case)
        try:
            # WHEN
            ice = self.strategy.get_ice()
            tmp_file_name = os.path.join(
                tempfile.mkdtemp(prefix="mli_unit_ice_im_"),
                "unit-ice-save-no-im.json",
            )
            ice.explain(
                target_features,
                x,
                mins=mins,
                maxs=maxs,
                predict_method=self.score_foo,
            ).save_json(tmp_file_name)
        finally:
            test_utils.rm_test_dir(tmp_dir)
