# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import numpy as np
import pandas as pd

from h2o_sonar import loggers as logging
from h2o_sonar.methods._h_statistic import HStatistic
from h2o_sonar.methods._pd import ICE
from h2o_sonar.methods._pd import PD


# loan dataset
DATASET_LOAN_PATH = "benchmarks/data/predictive/loan.csv"
DATASET_LOAN_EXAMPLES = 887_379
DATASET_LOAN_10_PATH = "data/predictive/pd_ice_loan_10_rows.csv"
DATASET_LOAN_10_EXAMPLES = 10

# credit card dataset
DATASET_CC_PATH = "data/predictive/pd_ice_creditcard_train.csv"
DATASET_CC_EXAMPLES = 23999
DATASET_CC_10_PATH = "data/predictive/pd_ice_creditcard_10_rows.csv"
DATASET_CC_10_EXAMPLES = 10
DATASET_CC_1_PATH = "data/predictive/pd_ice_creditcard_1_row.csv"
DATASET_CC_1_EXAMPLES = 1

# Titanic dataset
DATASET_CLEANED_TITANIC_PATH = "data/predictive/pd_ice_titanic_cleaned.csv"
DATASET_RAW_TITANIC_PATH = "data/predictive/pd_ice_titanic_raw.csv"

# Date features dataset (derived from Bank Marketing, CC BY 4.0 license)
DATASET_DATE_FEATURES_PATH = "data/predictive/pd_ice_date_features_train.csv"

# Binning
DATASET_DATE_BINNING_PATH = "data/predictive/pd_ice_date_binning.csv"


class IceStrategyFactory:
    """Class used to test ICE computation strategy for H-stat/PD/ICE."""

    # ICE computation strategy instance limits
    OPT_1_PREDICT_STRATEGY = True
    OPT_MULTIPLE_PREDICTS_STRATEGY = False

    @property
    def opt_1_prediction(self):
        return self.__strategy

    @opt_1_prediction.setter
    def opt_1_prediction(self, allow_1_predict_strategy: bool):
        self.__strategy = allow_1_predict_strategy

    def __init__(self):
        """Set default ICE strategy by default."""
        self.__strategy = ICE("I").opt_1_prediction

    def set_non_default_ice_strategy(self):
        if ICE("I").opt_1_prediction == self.OPT_1_PREDICT_STRATEGY:
            self.__strategy = self.OPT_MULTIPLE_PREDICTS_STRATEGY
        else:
            self.__strategy = self.OPT_1_PREDICT_STRATEGY

    def is_1_predict(self):
        return self.__strategy

    def get_ice(self):
        """ICE factory for computation strategy initialization, ..."""
        ice = ICE("I")
        ice.opt_1_prediction = self.__strategy
        return ice

    def get_pd(self):
        """PD factory for computation strategy initialization, ..."""
        pdp = PD("P")
        pdp.opt_1_prediction = self.__strategy
        return pdp

    def get_h_statistic(self):
        """H-stat factory for computation strategy initialization, ..."""
        hstat = HStatistic("H")
        hstat.opt_1_prediction = self.__strategy
        return hstat


class FooScorerConstPredictFrame:
    def __init__(self, example_count, cols):
        """Fake binomial/multinomial/regression prediction scorer.

        Parameters
        ----------
        example_count: int
            number of rows in the column returned
        cols:
            columns to return - 1 regression, 2 - binomial, >2 multinomial

        Returns
        -------
        Fake prediction as DataFrame

        """
        self.examples = example_count
        self.cols = cols

        self.seq = 1
        # series OK
        index = np.arange(example_count)
        self.prediction = pd.DataFrame(
            42, index=index, columns=["rnd_" + str(i) for i in range(cols)]
        )
        logging.debug(f"Scorer result shape: {self.prediction.shape}")

    def score(self, _):
        # returns Pandas frame
        return [self.prediction]

    def score_batch(self, X):
        if X.shape[0] == self.prediction.shape[0]:
            prediction = self.__get_frame(self.prediction, self.examples)
        else:
            count = int(X.shape[0] / self.examples)
            prediction = pd.DataFrame(
                columns=["rnd_" + str(i) for i in range(self.cols)]
            )
            for _ in range(count):
                df = self.__get_frame(self.prediction, self.examples)
                prediction = prediction.append(df.copy())

        logging.debug(f"Prediction:\n{prediction}")

        return prediction

    def __get_frame(self, prediction, examples):
        for i in prediction.columns.values:
            r = []
            for j in range(examples):
                self.seq += 1
                r.append(self.seq)
            prediction[i] = r

        return prediction


class FooScorerConstRegrFrame:
    def __init__(self, example_count):
        # series OK
        index = np.arange(example_count)
        self.prediction = pd.DataFrame(42, index=index, columns=[example_count])
        logging.debug(f"Scorer result shape: {self.prediction.shape}")

    def score(self, _):
        # returns Pandas frame
        return [self.prediction]

    def score_batch(self, _):
        # returns Pandas frame
        return self.prediction


class FooScorerSumRegrFrame:
    """Fake scorer whose prediction is example.sum() w/ class and method
    signature intentionally made similar to DAI scoring pipeline.

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
        # returns Pandas Series
        prediction = data.sum(axis=1)
        return prediction


class FooScorerRegrSeries:
    """Fake scorer w/ class and method signature intentionally made similar to
    DAI scoring pipeline.

    """

    def __init__(self):
        self.s = 0.0

    @staticmethod
    def score(data):
        return data

    def score_batch(self, data, fast_approx):
        predictions = [i + self.s for i in range(0, data.shape[0])]
        self.s = self.s + data.shape[0]
        # returns Python list
        return pd.Series(predictions)
