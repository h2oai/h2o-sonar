# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import copy
import os
import pickle
import warnings
from abc import ABC
from abc import abstractmethod

import datatable
import pandas

from h2o_sonar.methods.core.method import Method


class SurrogateModel(Method, ABC):
    """Interface class for all surrogate models."""

    def __init__(self, **kwargs):
        """
        Parameters supported in kwargs:

        Parameters
        ----------
        fold_column : str
            Column with cross-validation fold index assignment
            per observation.

        ignored_columns : array
            Columns not used for fitting. Defaults to none.

        response_column : str, int
            Response variable column.

        seed : int
            Seed for random numbers.

        weights_column : str
            Column with observation weights. Giving some
            observation a weight of zero is equivalent to excluding it from the
            data set. Giving an observation a relative weight of 2 is equivalent
            to repeating that row twice. Negative weights are not allowed.
            Note: Weights are per-row observation weights and do not increase
            the size of the data frame. During training, rows with higher
            weights matter more, due to the larger loss function pre-factor.

        """
        Method.__init__(self, "Surrogate Method", "surrogate")
        self.common_parameters = {
            "fold_column",
            "ignored_columns",
            "response_column",
            "weights_column",
            "seed",
        }
        self._parms = {}
        self._set_params(self.common_parameters, **kwargs)

    #
    # COMMON PROPERTIES
    #
    @property
    def fold_column(self):
        return self._parms.get("fold_column")

    @fold_column.setter
    def fold_column(self, fold_column):
        self._parms["fold_column"] = fold_column

    @property
    def ignored_columns(self):
        return self._parms.get("ignored_columns")

    @ignored_columns.setter
    def ignored_columns(self, ignored_columns):
        self._parms["ignored_columns"] = ignored_columns

    @property
    def response_column(self):
        return self._parms.get("response_column")

    @response_column.setter
    def response_column(self, response_column):
        self._parms["response_column"] = response_column

    @property
    def seed(self):
        return self._parms.get("seed")

    @seed.setter
    def seed(self, seed):
        self._parms["seed"] = seed

    @property
    def weights_column(self):
        return self._parms.get("weights_column")

    @weights_column.setter
    def weights_column(self, weights_column):
        self._parms["weights_column"] = weights_column

    #
    # COMMON METHODS FOR ALL SURROGATES
    #

    def fit(self, model, **kwargs):
        """Fit the surrogate model using data passed inside the model parameter.
        Different surrogate models require different data to be passed, please
        consult concrete docs for more information.

        This method does generic data pre-processing and validation common for
        all method and then calls the actual implementation.

        Parameters
        ----------
        model : h2o_sonar.core.InterpretableModel
            Object representing the model used as the surrogates base

        kwargs : dict
            All the arguments supported by a given surrogate.
            Please check the Surrogate and concrete algorithms documentations.

        """
        fit_params = copy.deepcopy(self._parms)
        fit_params.update(kwargs)
        self._validate(model, **fit_params)
        self._fit_impl(model, **fit_params)

    def _validate(self, model, **params):
        """Set of validations common to all surrogate methods.

        Parameters
        ----------
        model: h2o_sonar.core.InterpretableModel
            See fit().

        """
        #
        # RESPONSE COLUMN
        #
        response_column = params.get("response_column")
        if not response_column or response_column == "":
            raise ValueError("Response column is required.")

        #
        # WEIGHTS
        #
        weight_col = params.get("weight_column")
        weights_present = weight_col and weight_col != ""

        if weights_present and model.col_type(weight_col) in [str, object]:
            raise ValueError(
                "Weight column cannot be a string/obj type for surrogate model"
            )

        #
        # RESPONSE
        #
        response_column = params.get("response_column")
        if not response_column:
            raise ValueError("Response column is required but was not defined.")

        total_columns = (
            len(model.data_as_model.names)
            if isinstance(model.data_as_model, datatable.Frame)
            else len(model.data_as_model.columns)
        )
        if isinstance(response_column, int) and response_column >= total_columns:
            raise ValueError(
                f"Column index {response_column} exceeds the number of columns in "
                f"the data set {total_columns}."
            )

        if isinstance(response_column, str) and response_column not in (
            model.data_as_model.names
            if isinstance(model.data_as_model, datatable.Frame)
            else model.data_as_model.columns
        ):
            raise ValueError(
                f"Column {response_column} is not present in the data set."
            )

        if model.col_type(response_column) not in [str, object]:
            variance = model.col_variance(column=response_column)
            if round(variance, 10) == 0.0:
                warnings.warn(
                    "Cannot run MLI on constant prediction column."
                    f"Prediction column variance is: {variance}",
                    stacklevel=2,
                )

        #
        # WEIGHTS + RESPONSE MIX
        #
        if weights_present:
            if isinstance(model.data_as_model, pandas.DataFrame):
                variance = model.col_variance(
                    column=response_column,
                    col_filter=lambda frame: frame[weight_col] > 0,
                )
            elif isinstance(model.data_as_model, datatable.Frame):
                variance = model.col_variance(
                    data_column=model.data_as_model[datatable.f[weight_col] > 0, :][
                        :, response_column
                    ],
                    column=response_column,
                )
            else:
                raise ValueError(f"Unsupported data type: {type(model.data_as_model)}")
            if (round(variance), 10) == 0.0:
                raise ValueError(
                    "Weight column and response column creates constant target."
                    "Cannot proceed with surrogate DRF model."
                )

        #
        # MODEL SPECIFIC VALIDATIONS
        #
        self._validate_impl(model, **params)

    def _validate_impl(self, model, **kwargs):
        """Model specific validations. Should be overridden in concrete classes as
        needed.

        Parameters
        ----------
        model : h2o_sonar.core.InterpretableModel
            See fit()

        kwargs : dict
            All the arguments supported by a given surrogate.
            Please check the Surrogate and concrete algorithms documentations.

        """
        pass

    #
    # ABSTRACT METHODS
    #

    @abstractmethod
    def _fit_impl(self, model, **kwargs):
        """This method contains should contain the actual, method specific,
        implementation called by fit().

        Parameters
        ----------
        model : h2o_sonar.core.InterpretableModel
            See fit()

        kwargs : dict
            All the arguments supported by a given surrogate.
            Please check the Surrogate and concrete algorithms documentations.

        """
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save(self):
        """Saves the model to the disk using work_dir/name from the model object."""
        pass

    @abstractmethod
    def load_internal(self):
        """Load model specific parts of the object."""
        pass

    @classmethod
    def load(cls, work_dir, name):
        """Load a surrogate model for a given name from work_dir/name path.

        Parameters
        ----------
        work_dir : str
            Mli work_dir

        name : str
            Model name used during fitting

        Returns
        -------
        h2o_sonar.core.h2o_sonar.InterpretableModel :

        """
        with open(
            os.path.join(
                work_dir,
                name,
                cls.surrogate_type(),
                f"{cls.surrogate_type()}.pickle",
            ),
            "rb",
        ) as load_file:
            model = pickle.load(load_file)

        model.load_internal()

        return model

    #
    # STATIC ABSTRACT METHODS
    #

    @staticmethod
    # @abstractmethod ... 3.8 compatibility
    def surrogate_type():
        """Name of the surrogate. Used mainly for save/load and printing."""
        pass

    #
    # STATIC METHODS
    #

    @staticmethod
    def can_interpret(model, **kwargs):
        """This surrogate model can interpret InterpretableModel objects containing
        data and predictions fields.

        Parameters
        ----------
        model : h2o_sonar.core.h2o_sonar.InterpretableModel
            Model object to be interpreted.

        Returns
        -------
        bool :

        """
        return model.data_as_model is not None and "response_column" in kwargs

    @staticmethod
    def _model_path(work_dir, name, model_name="h2o-model"):
        return os.path.join(work_dir, name, model_name)

    #
    # PICKLE METHODS
    #

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["estimator"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
