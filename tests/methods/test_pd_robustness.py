# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import os
import tempfile
from functools import partial
from unittest import TestCase

import pandas as pd
import pytest

from h2o_sonar import loggers as logging
from h2o_sonar.errors import MliPredictMethodError
from h2o_sonar.methods._pd import PD
from tests.methods.ice_pd_test_commons import DATASET_CLEANED_TITANIC_PATH
from tests.methods.ice_pd_test_commons import DATASET_LOAN_10_PATH
from tests.methods.ice_pd_test_commons import DATASET_RAW_TITANIC_PATH
from tests.methods.ice_pd_test_commons import FooScorerRegrSeries
from tests.methods.ice_pd_test_commons import IceStrategyFactory
from tests.test_utils import find_locally
from tests.test_utils import rm_test_dir


try:
    import h2o  # noqa: F401

    HAS_H2O = True
except ImportError:
    HAS_H2O = False

MSG_BROKEN_SCORER = "Unable to score"


class TestPdIceRobustness(TestCase):
    """Test PD/ICE robustness - in particular:

    - not cleaned dataset w/ N/As, boolean and categorical values
    - unstable predict method

    """

    # override
    def setUp(self):
        logging.setLevel(logging.DEBUG)

        self.strategy = IceStrategyFactory()
        self.tmp_dir = tempfile.mkdtemp(prefix="mli_unit_pd_")

        self.df2x2 = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})

        # prediction method lambda
        self.score_titanic = partial(FooTitanicScorer().score_batch, fast_approx=True)
        self.score_broken = partial(FooBrokenScorer().score_batch, fast_approx=True)
        self.score_foo = partial(FooScorerRegrSeries().score_batch, fast_approx=True)

        # visual check
        self.test_visual_check = False

    # override
    def tearDown(self):
        rm_test_dir(self.tmp_dir)

    @staticmethod
    def _given_trimmed_dataset(path):
        df = pd.read_csv(find_locally(path), index_col=0)
        # trim dataset
        shard_size = 100
        df = df.loc[[i for i in range(1, df.shape[0], shard_size)]]
        logging.debug(f"Trimmed data ({len(df.index)}):\n{df.to_string()}")
        return df

    @staticmethod
    def _given_dataset(path):
        df = pd.read_csv(find_locally(path), index_col=0)
        # logging.debug("Data ({}):\n{}".format(len(df.index), df.to_string()))
        return df

    def test_broken_scorer(self):
        with self.assertRaises(MliPredictMethodError):
            try:
                # GIVEN
                self.strategy.get_pd().explain(
                    ["f1"], self.df2x2, predict_method=self.score_broken
                )
                # THEN
            except MliPredictMethodError as e:
                self.assertEqual("Predict method failed: Unable to score", str(e))
                logging.debug(f"Exception message: {e}")
                raise

    def test_duplicate_features(self):
        # THEN
        with self.assertRaises(ValueError):
            # WHEN
            self.strategy.get_pd().explain(
                ["f1", "f1"],
                self.df2x2,
                predict_method=lambda x: pd.Series([x]),
            )

    @pytest.mark.skipif(
        not HAS_H2O,
        reason="H2O Python package is not installed",
    )
    def test_ice_cleaned_single_feature_titanic(self):
        logging.debug("# ICE: cleaned Titanic single feature ###")

        # GIVEN
        df = TestPdIceRobustness._given_trimmed_dataset(DATASET_CLEANED_TITANIC_PATH)
        features = ["Age"]
        # use default resolution
        bins = None
        mins = [df["Age"].min()]
        maxs = [df["Age"].max()]

        # WHEN
        ice = self.strategy.get_ice()
        explanations = ice.explain(
            features,
            df,
            predict_method=self.score_titanic,
            bins=bins,
            mins=mins,
            maxs=maxs,
        ).explanations()
        ice_e = explanations[features[0]][PD.LABEL_REGRESSION]

        # THEN
        self.assertIsNotNone(explanations)
        logging.debug(f"ICE explainer for {features[0]}:\n{str(ice_e.head(10))}")
        self.assertEqual(len(ice_e.columns), PD.DEFAULT_GRID_RESOLUTION)
        self.assertGreater(ice_e[17][101], 80, "Mean WRONG")
        # plot
        if self.test_visual_check:
            # visual check
            import matplotlib.pyplot as plt

            ax = plt.gca()
            ice_e.transpose().plot(kind="line", ax=ax)
            plt.show()

    def test_pd_cleaned_single_feature_titanic(self):
        # GIVEN
        df = TestPdIceRobustness._given_trimmed_dataset(DATASET_CLEANED_TITANIC_PATH)
        target_features = ["Age"]
        # use default resolution
        bins = None

        # WHEN
        pdp = self.strategy.get_pd()
        pdp_e = pdp.explain(
            target_features, df, predict_method=self.score_titanic, bins=bins
        ).explanations()[target_features[0]][PD.LABEL_REGRESSION]
        # AND: test Titanic test persistence w/ text labels
        tmp_file_name = os.path.join(self.tmp_dir, "unit-pdp-save-titanic.json")
        pdp.save_json(tmp_file_name)

        # THEN
        logging.debug(f"PD methods:\n {str(pdp)}")
        # default resolution is 10
        self.assertEqual(len(pdp_e.columns), PD.DEFAULT_GRID_RESOLUTION)
        self.assertGreater(pdp_e[17.0]["mean"], 85, "Mean WRONG")

        # plot
        if self.test_visual_check:
            # visual check
            import matplotlib.pyplot as plt

            fs = ["Pclass"]
            pdp.explain(
                fs, df, predict_method=self.score_titanic, bins=bins
            ).explanations()[fs[0]].transpose().plot(
                kind="line",
                style=["ro-", "bs-", "y^-"],
                title="Titanic Class PD",
            )
            plt.show()

    def test_pd_raw_cat_feature_titanic(self):
        logging.debug("# PD: raw Titanic single CAT features ###")

        # GIVEN
        df = TestPdIceRobustness._given_trimmed_dataset(DATASET_RAW_TITANIC_PATH)
        # use default resolution
        mins = [df["Name"].min()]
        maxs = [df["Name"].max()]
        logging.debug(f"Mins/maxs: {mins}/{maxs}")

    def test_pd_raw_all_features_titanic(self):
        logging.debug("# PD: raw Titanic all features ###")

        # GIVEN
        df = TestPdIceRobustness._given_dataset(DATASET_RAW_TITANIC_PATH)
        target_features = df.columns.values.tolist()

        # WHEN
        pdp = self.strategy.get_pd()
        exs = pdp.explain(
            target_features, df, predict_method=self.score_titanic
        ).explanations()

        # THEN
        logging.debug(f"Titanic explanations:\n {str(pdp)}")
        self.assertEqual(len(exs), len(target_features), "Missing PD for a feature")

    def test_pd_loan(self):
        logging.debug("# PD: raw Loan all features ###")

        # GIVEN
        df = TestPdIceRobustness._given_dataset(DATASET_LOAN_10_PATH)
        target_features = df.columns.values.tolist()

        # WHEN
        pdp = self.strategy.get_pd()
        exs = pdp.explain(
            target_features, df, predict_method=self.score_foo
        ).explanations()

        # THEN
        logging.debug(f"Loan explanations:\n {str(pdp)}")
        self.assertEqual(len(exs), len(target_features), "Missing PD for a feature")


class FooBrokenScorer:
    """
    Fake scorer which fails to predict.
    """

    def __init__(self):
        self.s = 0.0

    @staticmethod
    def score(data):
        return data

    @staticmethod
    def score_batch(data, fast_approx):
        raise RuntimeError(MSG_BROKEN_SCORER)


class FooTitanicScorer:
    """
    Fake scorer for Titanic w/ class and method signature intentionally made
    similar to DAI scoring pipeline.
    """

    def __init__(self):
        self.s = 0.0

    @staticmethod
    def score(data):
        return data

    @staticmethod
    def score_batch(data, fast_approx):
        from h2o.utils.typechecks import assert_is_type

        assert_is_type(data, pd.DataFrame)
        prediction = (100 - data["Age"]) + (data["Fare"] / 10)
        logging.debug(f"Titanic prediction:\n{prediction}")
        return prediction
