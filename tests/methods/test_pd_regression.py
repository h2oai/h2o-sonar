# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import os
import tempfile
from functools import partial

import pandas as pd
import pytest

from h2o_sonar import loggers as logging
from h2o_sonar.errors import MliError
from h2o_sonar.errors import MliUnsupportedDataFormatError
from h2o_sonar.methods._ice import ICE
from h2o_sonar.methods._pd import PD
from h2o_sonar.methods.core._mli import InterpretableModel
from h2o_sonar.methods.core._mli import MLI
from tests.base_h2o_test import BaseH2OTest
from tests.conftest import get_h2o3_config
from tests.methods.ice_pd_test_commons import FooScorerRegrSeries
from tests.methods.ice_pd_test_commons import FooScorerSumRegrFrame
from tests.methods.ice_pd_test_commons import IceStrategyFactory
from tests.test_utils import rm_test_dir


try:
    import h2o

    HAS_H2O = True
except ImportError:
    HAS_H2O = False


class TestPdRegression(BaseH2OTest):
    """Test Partial Dependency (PD) implementation.

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
        self.df3x3 = pd.DataFrame({"f1": [1, 2, 3], "F": [1, 3, 5], "f2": [0.5, 2, 3]})
        # data: 1x5 dataframe
        self.df1x5 = pd.DataFrame([[i for i in range(1, 6)]], columns=list("abFdY"))
        # data: 5x5 dataframe w/ i..i+5 rows, F is target feature
        self.df5x5 = pd.DataFrame(
            [[i + j for i in range(1, 6)] for j in range(0, 5)],
            columns=list("abFdY"),
        )

        # prediction method lambda
        self.score_foo = partial(FooScorerRegrSeries().score_batch, fast_approx=True)
        self.score_sum = partial(FooScorerSumRegrFrame().score_batch, fast_approx=True)

        # test persistence in the current directory (avoid making garbage)
        self.test_current_dir_persistence = True
        # visual check
        self.test_visual_check = False

    def test_negative_init(self):
        with self.assertRaises(ValueError):
            PD(None)
        with self.assertRaises(ValueError):
            PD("")

    def test_no_bins_no_resolution_no_y_no_nothing(self):
        """If no bins and no resolution and no Y, then there is nothing to do."""
        with self.assertRaises(ValueError):
            PD("No bins & no resolution").explain(
                ["F"],
                self.df2x2,
                predict_method=self.score_foo,
                grid_resolution=0,
            )

    def test_init_name_type(self):
        # GIVEN
        name = "Name"

        # WHEN
        pdp = PD(name)

        # THEN
        self.assertEqual(pdp.method_name, name)
        self.assertEqual(pdp.method_type, "pd")

    def test_init_features(self):
        # GIVEN
        name = "Name"
        fs = ["F"]

        # WHEN
        pdp = PD(name).explain(fs, self.df3x3, predict_method=self.score_foo)

        # THEN
        self.assertEqual(pdp.features, fs)

    def test_non_existent_feature(self):
        # GIVEN
        name = "Name"
        fs = ["NONEXISTENT", "F"]

        # THEN
        with self.assertRaises(ValueError):
            # WHEN
            PD(name).explain(fs, self.df3x3, predict_method=self.score_foo)

    def test_run_negative(self):
        empty_data = pd.DataFrame()
        bins = []

        # negative feature(s)
        with self.assertRaises(ValueError):
            PD("Name").explain(
                None, self.df2x2, predict_method=self.score_foo, bins=bins
            )
        with self.assertRaises(ValueError):
            PD("Name").explain([], self.df2x2, predict_method=self.score_foo, bins=bins)
        with self.assertRaises(ValueError):
            PD("Name").explain(
                ["wrong-type"],
                self.df2x2,
                predict_method=self.score_foo,
                bins=bins,
            )
        # negative scorer
        with self.assertRaises(ValueError):
            PD("Name").explain(["F"], self.df2x2, bins=bins)
        # negative data
        with self.assertRaises(ValueError):
            PD("Name").explain(["F"], None, predict_method=self.score_foo, bins=bins)
        with self.assertRaises(ValueError):
            PD("Name").explain(
                ["F"], empty_data, predict_method=self.score_foo, bins=bins
            )

    @pytest.mark.skipif(
        not HAS_H2O,
        reason="H2O Python package is not installed",
    )
    def test_run_pandas_single_feature(self):
        self._run_single_feature(False)

    @pytest.mark.skipif(
        not HAS_H2O,
        reason="H2O Python package is not installed",
    )
    def test_run_h2oframe_single_feature(self):
        self._run_single_feature(True)

    def _run_single_feature(self, h2oframe):
        # GIVEN
        features = ["F"]

        # WHEN
        _, explanations = self._run_pdp(features, None, h2oframe)

        # THEN
        logging.debug(
            f"PD explanation for {features[0]}:\n"
            f"{explanations[features[0]][PD.LABEL_REGRESSION]}"
        )
        self.assertIsInstance(explanations, dict, "Result to be dictionary")
        explanation = explanations[features[0]][PD.LABEL_REGRESSION]
        self.assertIsInstance(explanation, pd.DataFrame, "Result to be Pandas")
        self.assertEqual(
            explanation.shape[0],
            4,
            "1 feature gives 4 rows: PDs, SDs, SEMs, OOR hints",
        )
        expected_result_cols_shape = 5
        self.assertEqual(
            explanation.shape[1],
            expected_result_cols_shape,
            "3 bins, 3 PD values (x chart axis)",
        )

    @pytest.mark.skipif(
        not HAS_H2O,
        reason="H2O Python package is not installed",
    )
    def test_run_pandas_many_features_and_bins(self):
        # GIVEN
        bins = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        features = ["b", "F", "d"]

        # WHEN
        _, explanations = self._run_pdp(features, bins)

        # THEN
        self.assertIsInstance(explanations, dict, "Result to be dictionary")
        for f in features:
            logging.debug(
                f"PD explanation for {f} is:\n"
                f"{str(explanations[f][PD.LABEL_REGRESSION])}"
            )
            self.assertIsInstance(
                explanations[f][PD.LABEL_REGRESSION],
                pd.DataFrame,
                "Result to be Pandas",
            )
            self.assertEqual(
                explanations[f][PD.LABEL_REGRESSION].shape[0],
                4,
                "1 feature gives 3 rows: PDs + SDs + SEMs + OOR hints",
            )
            self.assertEqual(
                explanations[f][PD.LABEL_REGRESSION].shape[1],
                len(bins),
                "3 bins, 3 PD values (x chart axis)",
            )

    @pytest.mark.skipif(
        not HAS_H2O,
        reason="H2O Python package is not installed",
    )
    def test_run_pandas_many_features_grid_bins(self):
        # GIVEN
        features = ["b", "F", "d"]

        # WHEN
        _, explanations = self._run_pdp(features, None)

        # THEN
        self.assertIsInstance(explanations, dict, "Result to be dictionary")
        for f in features:
            logging.debug(
                f"PD explanation for {f} is:\n{explanations[f][PD.LABEL_REGRESSION]}"
            )
            self.assertIsInstance(
                explanations[f][PD.LABEL_REGRESSION],
                pd.DataFrame,
                "Result to be Pandas",
            )
            self.assertEqual(
                explanations[f][PD.LABEL_REGRESSION].shape[0],
                4,
                "1 feature gives 3 rows: PDs + SDs + SEMs + OOR hints",
            )
            self.assertEqual(
                self.df5x5.shape[1],
                explanations[f][PD.LABEL_REGRESSION].shape[1],
                "3 bins, 3 PD values (x chart axis)",
            )

    def _run_pdp(self, target_features, bins, h2oframe=False):
        if h2oframe:
            # H2OFrame data - convert to pandas before calling explain()
            h2o_frame = h2o.H2OFrame(self.df5x5)
            df = h2o_frame.as_data_frame()
        else:
            logging.debug("PD data:\n" + self.df2x2.to_string())
            df = self.df5x5

        # PD
        pdp = self.strategy.get_pd()
        i = pdp.explain(target_features, df, predict_method=self.score_sum, bins=bins)

        self.assertIsNotNone(i, "Run must return PD instance")
        self.assertIsInstance(i, PD, "Run must return PD instance")

        # check that explanations are cached
        explanations = pdp.explanations()
        self.assertIsNotNone(explanations, "Explanations must be cached")

        return pdp, explanations

    def test_explanations_negative(self):
        pdp = PD("Negative explanations")
        with self.assertRaises(MliError):
            pdp.explanations()

    @pytest.mark.skipif(
        not HAS_H2O,
        reason="H2O Python package is not installed",
    )
    def test_save_and_load_json(self):
        # GIVEN
        bins = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        features = ["b", "F", "d"]

        # WHEN
        pdp, explanations = self._run_pdp(features, bins)
        for f in features:
            logging.debug(
                f"S/L explanation for {f} is:\n"
                f"{str(explanations[f][PD.LABEL_REGRESSION])}"
            )

        # save
        logging.debug("SAVING JSon")
        tmp_dir = tempfile.mkdtemp(prefix="mli_unit_pd_")
        tmp_file_name = os.path.join(tmp_dir, "unit-pd-save-basic.json")
        try:
            pdp.save_json(tmp_file_name)

            # NEGATIVE: save check overwrite
            with self.assertRaises(FileExistsError):
                pdp.save_json(tmp_file_name)

            # THEN
            # load
            logging.debug("LOADING JSon...")
            pdp.evict_explanations()
            pdp.load_json(tmp_file_name)
            l_explanations = pdp.explanations()

            self.assertIsInstance(explanations, dict, "Result to be dictionary")
            for f in l_explanations:
                logging.debug(
                    f"PD explanation for {f} is:\n"
                    f"{str(l_explanations[f][PD.LABEL_REGRESSION])}"
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
                logging.debug(f"SRC: {f}\n{str(explanations[f][PD.LABEL_REGRESSION])}")
                logging.debug(
                    f"S/L: {f}\n{str(l_explanations[f][PD.LABEL_REGRESSION])}"
                )
                self.assertEqual(
                    explanations[f][PD.LABEL_REGRESSION].shape,
                    l_explanations[f][PD.LABEL_REGRESSION].shape,
                )
                for c in explanations[f][PD.LABEL_REGRESSION].columns.values:
                    for r in explanations[f][PD.LABEL_REGRESSION].index.values:
                        self.assertEqual(
                            explanations[f][PD.LABEL_REGRESSION][c][r],
                            l_explanations[f][PD.LABEL_REGRESSION][c][r],
                        )
        finally:
            rm_test_dir(tmp_dir)

        # NEGATIVE: load non-existent file
        with self.assertRaises(FileNotFoundError):
            pdp.load_json("/THIS-IS-NON-EXISTENT-FILE")

        # save/load to/from CURRENT dir
        if self.test_current_dir_persistence:
            tmp_file = os.path.join(os.getcwd(), pdp.default_json_file_name)
            try:
                # save to current dir
                logging.debug("SAVING to current directory: " + os.getcwd())
                pdp.save_json()
                # load from current dir
                logging.debug("LOADING from current directory: " + os.getcwd())
                pdp.load_json()
                self.assertIsInstance(explanations, dict, "Result to be dictionary")
                self.assertEqual(len(explanations), len(pdp.explanations()))
            finally:
                os.remove(tmp_file)

    def test_save_and_append(self):
        # GIVEN
        bins = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        features = ["b", "F", "d"]
        pdp = self.strategy.get_pd()
        tmp_dir = tempfile.mkdtemp(prefix="mli_unit_pdp_append")
        tmp_file_name = os.path.join(tmp_dir, "unit-pdp-append.json")

        # WHEN
        try:
            for i, feature in enumerate(features):
                pdp.explain(
                    [feature],
                    self.df5x5,
                    predict_method=self.score_sum,
                    bins=[bins[i]],
                    ice_cache_path=tmp_file_name,
                )
                logging.debug(f"PDP {i}: {pdp}")

            # THEN
            logging.debug("LOADING cached ICE...")
            ice = ICE("ICE")
            ice.load(tmp_file_name)
            logging.debug(f"ICE: {ice}")

            self.assertIsNotNone(ice.explanations())
            exs = ice.explanations()
            self.assertEqual(3, len(exs))
        finally:
            rm_test_dir(tmp_dir)

    def test_assert_pdp_values_single_feature_many_instances(self):
        logging.debug("# PD: values and visual check ###")

        # GIVEN
        df = self.df3x3
        if isinstance(df, pd.DataFrame):
            logging.debug("ICE data:\n" + df.head(10).to_string())
        bins = [[1, 3, 5]]
        features = ["F"]

        # WHEN
        # ICE
        ice = ICE("ICE plot")
        explanations = ice.explain(
            features, df, predict_method=self.score_sum, bins=bins
        ).explanations()

        ice_e = explanations[features[0]][PD.LABEL_REGRESSION]
        logging.debug(f"ICE for {features[0]}:\n{str(ice_e.head(10))}")

        pdp = self.strategy.get_pd()
        # assert PD calculations
        pdp_e = pdp.explain(
            features, df, predict_method=self.score_sum, bins=bins
        ).explanations()[features[0]][PD.LABEL_REGRESSION]

        # THEN
        logging.debug(f"PD for {features[0]}:\n{str(pdp_e)}")
        self.assertEqual(len(pdp_e.columns), 3, "3 bins expected")
        self.assertEqual(len(pdp_e.index), 4, "4 rows expected: MEAN, SD, SEM, OOR")
        self.assertEqual(pdp_e[1][PD.COL_MEAN], 4.833_333_333_333_333, "wrong MEAN")
        self.assertEqual(pdp_e[3][PD.COL_SD], 2.254_624_876_411_447, "wrong SD")
        self.assertEqual(pdp_e[5][PD.COL_SEM], 1.301_708_279_317_775_9, "wrong SEM")

        logging.debug(
            f"Diagnostics: predict method invocations "
            f"{ice.diagnostics.total_scorer_calls} ~ "
            f"{ice.diagnostics.scorer_calls_history}"
        )
        self.assertEqual(
            1 if pdp.opt_1_prediction else 3, ice.diagnostics.total_scorer_calls
        )

        # plot
        if self.test_visual_check:
            # visual check
            import matplotlib.pyplot as plt

            ax = plt.gca()
            ice_e.transpose().plot(kind="line", ax=ax)
            pdp_e.transpose().plot(
                kind="line",
                style=["ro-", "bs-", "y^-"],
                title="PD + ICE",
                ax=ax,
            )
            plt.show()

    def test_predict_method_return_values(self):
        # predict method which returns pd Series
        # WHEN
        pdp = self.strategy.get_pd().explain(
            ["f1", "f2"],
            self.df3x3,
            bins=[[1], [2]],
            predict_method=lambda x: pd.Series(x["F"]),
        )
        # THEN
        self.assertIsNotNone(pdp)
        logging.debug(f"Series:\n{pdp}")
        self.assertEqual(2, len(pdp.explanations()))
        self.assertIsInstance(
            pdp.explanations()["f1"][PD.LABEL_REGRESSION],
            pd.DataFrame,
            "Result to be DataFrame",
        )
        self.assertEqual(
            3, pdp.explanations()["f1"][PD.LABEL_REGRESSION][1][PD.COL_MEAN]
        )

        # predict method which returns pd DataFrame 1 column
        # WHEN
        pdp = PD("Frame w/ 1 column predict method").explain(
            ["f1", "f2"],
            self.df3x3,
            bins=[[1], [2]],
            predict_method=lambda x: x["F"],
        )
        # THEN
        self.assertIsNotNone(pdp)
        logging.debug(f"Frame[1]:\n{pdp}")
        self.assertIsInstance(
            pdp.explanations()["f1"][PD.LABEL_REGRESSION],
            pd.DataFrame,
            "Result to be DataFrame",
        )
        self.assertEqual(
            3, pdp.explanations()["f1"][PD.LABEL_REGRESSION][1][PD.COL_MEAN]
        )

    def test_ice_cache(self):
        # WHEN
        pdp = self.strategy.get_pd().explain(
            ["f1", "f2"],
            self.df3x3,
            predict_method=self.score_foo,
            ice_cache={},
        )

        # THEN
        self.assertIsNotNone(pdp)
        logging.debug(f"PD:\n{pdp}")
        cached_ice = pdp.explanations(kind="ice")
        logging.debug(f"Cached ICE:\n{cached_ice}")
        self.assertIsNotNone(cached_ice)
        self.assertEqual(2, len(cached_ice))
        self.assertEqual(1, len(cached_ice["f1"]))
        self.assertEqual(1, len(cached_ice["f2"]))
        # int feature
        self.assertEqual((3, 3), cached_ice["f1"][PD.LABEL_REGRESSION].shape)
        # float feature
        self.assertEqual((3, 10), cached_ice["f2"][PD.LABEL_REGRESSION].shape)

    def test_negative_ice_cache(self):
        def negative_ice_cache(ice_mask):
            PD("ICE cached by PD").explain(
                ["f1", "f2"],
                self.df3x3,
                predict_method=self.score_foo,
                ice_cache=ice_mask,
            )

        with self.assertRaises(ValueError):
            negative_ice_cache(["wrong type"])
        with self.assertRaises(ValueError):
            negative_ice_cache({"wrong_feature_name": {PD.LABEL_REGRESSION: [0]}})
        with self.assertRaises(ValueError):
            negative_ice_cache({"f1": ["wrong type"]})
        with self.assertRaises(ValueError):
            negative_ice_cache({"f1": {PD.LABEL_REGRESSION: ("wrong type", 1)}})
        with self.assertRaises(ValueError):
            negative_ice_cache({"f1": {PD.LABEL_REGRESSION: []}})

    def test_ice_filter_n_cache(self):
        # WHEN
        pdp = self.strategy.get_pd().explain(
            ["f1", "f2"],
            self.df3x3,
            predict_method=self.score_foo,
            ice_cache={
                "f1": {PD.LABEL_REGRESSION: [0, 2]},
                "f2": {PD.LABEL_REGRESSION: [1]},
            },
        )

        # THEN
        self.assertIsNotNone(pdp)
        logging.debug(f"PD:\n{pdp}")
        cached_ice = pdp.explanations(kind="ice")
        logging.debug(f"Cached ICE:\n{cached_ice}")
        self.assertIsNotNone(cached_ice)
        self.assertEqual(2, len(cached_ice))
        self.assertEqual(1, len(cached_ice["f1"]))
        self.assertEqual(1, len(cached_ice["f2"]))
        self.assertEqual((2, 3), cached_ice["f1"][PD.LABEL_REGRESSION].shape)
        self.assertEqual((1, 10), cached_ice["f2"][PD.LABEL_REGRESSION].shape)

    def test_json_bins_type_conversion_driven_by_feature_type(self):
        # GIVEN
        X = pd.DataFrame({"num": [1, 2, 3], "cat": [1, 3, 5]})
        bins = [[1, 2, 3], [1, 2, 3]]
        meta = {PD.KEY_CATEGORICAL_FEATURES: ["cat"]}

        # WHEN
        json = (
            self.strategy.get_pd()
            .explain(
                ["num", "cat"],
                X,
                bins=bins,
                predict_method=self.score_foo,
                features_meta=meta,
            )
            .to_json()
        )

        # THEN
        self.assertIsNotNone(json)
        logging.debug(f"PD JSon:\n{json}")
        self.assertTrue(PD.JSON_PD_DATA in json)
        self.assertTrue(json[PD.JSON_PD_DATA])
        json_bins = json[PD.JSON_PD_DATA][0][PD.JSON_DATA][0]
        logging.debug(f"int 2 float bins:\n{json_bins}")
        self.assertListEqual([1.0, 2.0, 3.0], json_bins)
        json_bins = json[PD.JSON_PD_DATA][1][PD.JSON_DATA][0]
        logging.debug(f"str 2 float bins:\n{json_bins}")
        self.assertListEqual(["1", "2", "3"], json_bins)

    def test_interpretable_model_integration(self):
        logging.debug("# PD: interpretable model integration ###")

        # GIVEN
        x = self.df3x3
        features = ["F"]

        # MLI: has a default working directory
        tmp_dir = tempfile.mkdtemp(prefix="mli_unit_pdp_im_")
        mli = MLI(work_dir=tmp_dir, config=get_h2o3_config())
        i_model = InterpretableModel(mli, "IM for PD", predict_method=self.score_foo)

        # A) predict() and working dir used from IM
        pdp = None
        try:
            # WHEN
            pdp = PD("IM based", i_model)
            exs = pdp.explain(features, x).save_json().explanations()

            # THEN
            self.assertEqual(len(exs), 1)
            self.assertEqual((4, 5), exs[features[0]][PD.LABEL_REGRESSION].shape)
        finally:
            if pdp:
                os.remove(os.path.join(tmp_dir, pdp.default_json_file_name))

        # B) predict() override (doesn't have to be specified on IM)
        try:
            # WHEN
            pdp = PD("Predict override", i_model)
            pdp.explain(features, x, predict_method=self.score_foo).save_json()
        finally:
            rm_test_dir(tmp_dir)

        # C) no IM, all override (all other tests in this test case)
        try:
            # WHEN
            pdp = PD("Predict override")
            tmp_file_name = (
                tempfile.mkdtemp(prefix="mli_unit_pdp_im_") + "unit-pdp-save-no-im.json"
            )
            pdp.explain(features, x, predict_method=self.score_foo).save_json(
                tmp_file_name
            )
        finally:
            rm_test_dir(tmp_dir)

    def test_negative_create_unique_bins(self):
        # X
        with self.assertRaises(MliUnsupportedDataFormatError):
            PD.create_unique_bins(["a"], None)
        with self.assertRaises(ValueError):
            PD.create_unique_bins(["a"], pd.DataFrame())
        # features
        with self.assertRaises(ValueError):
            PD.create_unique_bins(None, self.df2x2)
        with self.assertRaises(ValueError):
            PD.create_unique_bins([], self.df2x2)
        # grid resolution
        with self.assertRaises(ValueError):
            PD.create_unique_bins(["a"], self.df2x2, -3)

    @pytest.mark.skipif(
        not HAS_H2O,
        reason="H2O Python package is not installed",
    )
    def test_create_unique_bins(self):
        from h2o.utils.typechecks import assert_is_type

        # GIVEN
        features = ["a"]

        # WHEN
        bins = PD.create_unique_bins(features, self.df5x5)

        # THEN
        self.assertIsNotNone(bins)
        assert_is_type(bins, list, "Bins must be list")
        logging.debug(f"Bins: {bins}")
        self.assertEqual(len(features), len(bins))
        assert_is_type(bins[0], list, "Bins item must be list")
        self.assertEqual(5, len(bins[0]))

    @pytest.mark.skipif(
        not HAS_H2O,
        reason="H2O Python package is not installed",
    )
    def test_create_unique_bins_grid(self):
        from h2o.utils.typechecks import assert_is_type

        # GIVEN
        features = ["a", "b"]
        X = pd.DataFrame(
            {"a": [i for i in range(25)], "b": [i * 10 for i in range(25)]}
        )

        # WHEN
        bins = PD.create_unique_bins(features, X)

        # THEN
        self.assertIsNotNone(bins)
        assert_is_type(bins, list, "Bins must be list")
        logging.debug(f"Bins: {bins}")
        self.assertEqual(len(features), len(bins))
        assert_is_type(bins[0], list, "Bins item must be list")
        self.assertEqual(10, len(bins[0]))
