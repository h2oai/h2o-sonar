# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import os
from enum import Enum

import datatable
import numpy
import pandas

from h2o_sonar import loggers as logging
from h2o_sonar.lib.api import commons
from h2o_sonar.methods.core import _data
from h2o_sonar.methods.utils import h2o_utils
from h2o_sonar.methods.utils.h2o_utils import to_h2oframe


try:
    import h2o

    HAS_H2O = True
except ImportError:
    HAS_H2O = False


def default_pred(model, X):
    """Default prediction method used if user doesn't pass a different one.
    Separate method and not a lambda so it can be pickled.

    Parameters
    ----------
    model: Object
        Model objects containing predict() method

    X:
        Python object which can be handled by `model.predict()` method

    """
    return model.predict(X)


def default_activation(X):
    """Default activation method used if user doesn't pass a different one.
    Separate method and not a lambda so it can be pickled.

    This method will be called, if necessary, after predictions, using the
    result of `predict_method()` as input.

    Parameters
    ----------
    X:
        Result of `predict_method()`

    """
    return X


class MLIDataBackend(Enum):
    H2O = 1
    PANDAS = 2
    DATATABLE = 3


class MLI:
    """Entry point to the world of interpretability.

    This class contains all the information about the working directory and
    exposes methods to prepare your data for interpretation.

    """

    def __init__(
        self, work_dir="/tmp/", log_level=logging.NOTSET, seed=None, config=None
    ):
        """Parameters
        ----------
        work_dir: str
            Working directory for MLI, all results, models and
            temporary data will be stored here

        seed: int
            Seed used by algorithms for reproducible results

        config: dict
            Configuration used by MLI and MLI algorithms:
            - h2o_url: URL to a running H2O-3 instance, used by H2O backends
            - h2o_port: port to a running H2O-3 instance under h2o_url,
            used by H2O backends

        """
        if config is None:
            config = {}
        h2o_utils.assert_is_type(config, dict)
        h2o_utils.assert_is_type(work_dir, str)
        h2o_utils.assert_is_type(seed, None, int)

        if os.path.exists(work_dir) and not os.path.isdir(work_dir):
            raise ValueError(f"Working directory {work_dir} has to be a directory")

        logging.setLevel(log_level)

        logging.debug("Created MLI object with the following configuration: ")
        logging.debug(json.dumps(config, indent=2))

        self.config = config
        self.work_dir = work_dir
        self.seed = seed

    def wrap(
        self,
        name,
        data=None,
        predictions=None,
        model=None,
        predict_method=default_pred,
        activation_method=default_activation,
        fit_method=None,
        data_backend=MLIDataBackend.H2O,
    ):
        """Wraps your data in an interpretable object. Depending on which
        parameters were passed you will have access to different
        interpretability methods. Consult each method's class to learn what
        kind of information
        it requires.

        Parameters
        ----------
        name: str
            Name of the model, used mainly for persistence so should
            be unique within the workspace

        data: numpy.ndarray, pandas, datatable or Python list
            Observations used for training

        predictions: numpy.ndarray, pandas, datatable or Python list
            Predictions obtained from the model we are trying to interpret

        model: Object
            Model object capable of predictions on data, passed to the
            predict_method for certain interpretation methods

        predict_method: lambda
            Method consuming the model and data producing
            predictions using the model

        activation_method: lambda
            A linking method between the results of
            predict_method and whatever output domain we are expecting

        fit_method: lambda
            A method capable of fitting a new model

        data_backend: h2o_sonar.core.h2o_sonar.MLIDataBackend
            The dominating backend for MLI. Use the backend
            corresponding to algorithm backends you will use mostly to reduce
            the amount of data type transformations.

        Returns
        -------
        h2o_sonar.core.h2o_sonar.InterpretableModel

        """
        return InterpretableModel(
            self,
            name,
            data,
            predictions,
            model,
            predict_method,
            activation_method,
            fit_method,
            data_backend,
        )


class InterpretableModel:
    """Wrapper around data we are interpreting.

    This is the main class used by all the methods in this package for
    interpretation. Depending on what kind of data was provided different
    interpretation methods will be available.

    """

    def __init__(
        self,
        mli,
        name,
        data=None,
        predictions=None,
        model=None,
        predict_method=lambda model, x: model.predict(x),
        activation_method=lambda x: x,
        fit_method=None,
        data_backend=MLIDataBackend.H2O,
    ):
        """Constructor.

        Parameters
        ----------
        mli: h2o_sonar.core.MLI
            MLI object containing work_dir, seed and config

        name: str
            Name of the model, used mainly for persistence so
            should be unique within the workspace

        data: pd.DataFrame, dt.Frame, h2o.H2OFrame or PersistedData
            Observations used for training

        predictions: pd.DataFrame, dt.Frame, h2o.H2OFrame or PersistedData
            Predictions obtained from the model we are trying to interpret

        model: Object
            Model object capable of predictions on data,
            passed to the predict_method for certain interpretation methods

        predict_method: lambda
            Method consuming the model and data
            producing predictions using the model

        activation_method: lambda
            A linking method between the results
            of predict_method and whatever output domain we are expecting

        fit_method: lambda
            A method capable of fitting a new model

        """
        if not HAS_H2O:
            commons.raise_opt_import_err("h2o")

        h2o_utils.assert_is_type(mli, MLI)
        h2o_utils.assert_is_type(name, str)
        h2o_utils.assert_is_type(
            data,
            None,
            str,
            _data.PersistedData,
            pandas.DataFrame,
            numpy.ndarray,
            list,
            h2o.H2OFrame,
            datatable.Frame,
        )
        h2o_utils.assert_is_type(
            predictions,
            None,
            str,
            _data.PersistedData,
            pandas.DataFrame,
            numpy.ndarray,
            list,
            h2o.H2OFrame,
            datatable.Frame,
        )

        if data is None and predictions is not None:
            raise ValueError("When setting predictions you must also supply the data!")

        if (
            data is not None
            and predictions is not None
            and not isinstance(data, type(predictions))
        ):
            raise ValueError("Data and predictions need to be of same type.")

        self.mli = mli
        self.name = name

        # TODO: probably will need to remove frames when this object gets GC'd
        self.delete_data_as_model = False
        self.data_as_model = pandas.DataFrame()
        self._set_data(data, predictions, data_backend, mli.config)

        self.model = model
        self.predict_method = predict_method

        # TODO: if predict_method is not None then we should double check model is
        #  also not None and contains the predict method
        self.activation_method = activation_method
        self.fit_method = fit_method

        self.data_backend = data_backend

    def empty(self):
        return len(self.data_as_model) == 0

    def col_type(self, column):
        """Checks if a given column is part of the data set and returns its type.

        Parameters
        ----------
        column: str, int
            Column name to be checked.

        Returns
        -------
        pandas.core.series.Series

        """
        InterpretableModel._valid_col_name(column, self.data_as_model)

        data_column = self.find_column(column)

        if hasattr(data_column, "dtype"):
            return data_column.dtype

        if hasattr(data_column, "type"):
            return data_column.type

        if hasattr(data_column, "stypes"):
            return data_column.stype

        raise ValueError("Unsupported data type.")

    def col_variance(self, column=None, col_filter=None, data_column=None):
        """Checks if a given column is part of the data set and returns its
        variance if so. If `col_filter` is not passed then the variance is
        computed on all the rows. Otherwise only on the rows which pass the
        filtering condition.

        Parameters
        ----------
        column: str, int
            Column for which the variance is to be computed.

        col_filter: lambda
            Expression filtering the rows of `col_name`

        data_column: pandas.DataFrame, h2o.H2OFrame, datatable.Frame
            Column frame for which the variance is to be computed

        Returns
        -------
        float

        """
        if not HAS_H2O:
            commons.raise_opt_import_err("h2o")

        InterpretableModel._valid_col_name(column, self.data_as_model)

        if not data_column and column:
            data_column = self.find_column(column)

        if col_filter:
            return data_column[col_filter(self.data_as_model)].var()
        types = (pandas.DataFrame, pandas.Series, h2o.H2OFrame)
        if isinstance(data_column, types):
            return data_column.var()
        return data_column.sd()[0, 0] ** 2

    def find_column(self, column):
        """Returns the column by name or index. Column can be part of the data set
        or the predictions.

        Parameters
        ----------
        column: str, int
            Column name or index

        Returns
        -------
        pandas.core.series.Series
        """
        if not HAS_H2O:
            commons.raise_opt_import_err("h2o")

        if isinstance(self.data_as_model, datatable.Frame):
            if column in self.data_as_model.names:
                return self.data_as_model[column]
        elif isinstance(self.data_as_model, (pandas.DataFrame, h2o.H2OFrame)):
            if column in self.data_as_model.columns:
                return self.data_as_model[column]
        else:
            raise ValueError(f"Unsupported data type: {type(self.data_as_model)}")

        data_cols = numpy.shape(self.data_as_model)[1]
        if isinstance(column, int) and data_cols - column > 0:
            col_name = self._idx2col(self.data_as_model, column)
            return self.data_as_model[col_name]

        raise ValueError(f"Response column '{column}' not in model.")

    def _set_data(self, data, predictions, data_backend, config=None):
        """This method sets the data_as_model appropriately.
        The logic should be as follow:

        If the PANDAS backend is used we should concatenate both data and
        predictions into a single pandas.DataFrame. This is currently done
        but first converting both to separate pandas.DataFrame and
        concatenating. A faster way probably is available should we need one.

        If the H2O backend is used then:
        - if only data is passed as H2OFrame throw exception is predictions
        are passed (we assume both data and preds to be in the passed data)
        - return data as is
        - if it's not an H2OFrame then make sure data and preds are of same
        type (unless preds are None then we don't care). Concat both and
        make a single H2OFrame. We first convert both to Pandas for convenience.
        A more direct conversion would probably be faster - can add if needed.

        """
        if data is None and predictions is None:
            return

        if config is None:
            config = {}

        if data_backend == MLIDataBackend.PANDAS:
            self._set_as_pandas(data, predictions)
        elif data_backend == MLIDataBackend.H2O:
            self._set_as_h2oframe(config, data, predictions)
        elif data_backend == MLIDataBackend.DATATABLE:
            self._set_as_datatable(data, predictions)

    def _set_as_h2oframe(self, config, data, predictions):
        if not HAS_H2O:
            commons.raise_opt_import_err("h2o")

        if isinstance(data, h2o.H2OFrame):
            if predictions is None:
                # Assume data+preds are in a single frame
                self.data_as_model = data
            elif isinstance(predictions, h2o.H2OFrame):
                self.data_as_model = data.cbind(predictions)
                self.delete_data_as_model = True
            else:
                raise ValueError("Data/prediction need to be of same type.")
        else:
            if predictions is not None:
                if isinstance(data, _data.PersistedData):
                    raise ValueError(
                        "Currently when passing location to "
                        "data we do not support passing "
                        "predictions. Both data and "
                        "predictions should be in the same "
                        "file."
                    )
                if not isinstance(data, type(predictions)):
                    raise ValueError(
                        "When passing data and predictions, "
                        "both need to be of same type."
                    )

            if isinstance(data, (_data.PersistedData, str)):
                data, delete = InterpretableModel.to_h2o_frame(data, config)
                self.data_as_model = data
                self.delete_data_as_model = delete
            else:
                # Going through Pandas for convenience
                # A more direct approach might be possible
                data = InterpretableModel.to_pandas(data)
                predictions = InterpretableModel.to_pandas(
                    predictions, column_prefix="p"
                )
                preds = pandas.concat([data, predictions], axis=1, sort=False)
                data, delete = InterpretableModel.to_h2o_frame(preds, config)
                self.data_as_model = data
                self.delete_data_as_model = delete

    def _set_as_pandas(self, data, predictions):
        if isinstance(data, str):
            raise ValueError("Passing path to data using PANDAS backend not supported.")
        data = InterpretableModel.to_pandas(data)
        predictions = InterpretableModel.to_pandas(predictions)
        self.data_as_model = pandas.concat([data, predictions], axis=1, sort=False)

    def _set_as_datatable(self, data, predictions):
        if isinstance(data, str):
            raise ValueError(
                "Passing path to data using DATATABLE backend not supported."
            )
        data = InterpretableModel.to_datatable(data)
        predictions = InterpretableModel.to_datatable(predictions)
        self.data_as_model = datatable.cbind(data, predictions)

    @staticmethod
    def prepare_data(data, config=None, data_backend=MLIDataBackend.H2O):
        """Prepare data for future user. Currently, we support only h2o.H2OFrame
        and pandas.DataFrame. Ideally this will change in the future.

        Parameters
        ----------
        data: h2o.H2OFrame or anything that pandas.DataFrame() allows

        data_backend: MLIDataBackend
            The dominating backend for MLI. Use the backend
            corresponding to algorithm backends you will use mostly to reduce
            the amount of data type transformations.

        config: dict
            Additional config for data preparation. For instance for
            H2O backend user might want to pass a custom IP/port values.

        """
        if data is None:
            return None
        if data_backend == MLIDataBackend.H2O:
            return InterpretableModel.to_h2o_frame(data, config)
        if data_backend == MLIDataBackend.PANDAS:
            return InterpretableModel.to_pandas(data), False
        if data_backend == MLIDataBackend.DATATABLE:
            return InterpretableModel.to_datatable(data)
        raise ValueError("Unsupported data backend.")

    @staticmethod
    def _idx2col(data, column):
        """Get a column using numerical index.

        Parameters
        ----------
        data:
            Columnar data containing columns attribute.

        column: int
            Column index.

        Returns
        -------
        pandas.core.series.Series :

        """
        h2o_utils.assert_is_type(column, int)
        return data.columns[column]

    @staticmethod
    def _valid_col_name(column, data):
        """All the checks making sure a given column is in the data set.

        Parameters
        ----------
        column: str, int
            Column's name or index.

        data: pandas.DataFrame
            Data set to be queried.

        """
        if not HAS_H2O:
            commons.raise_opt_import_err("h2o")

        if not column or column == "":
            raise ValueError("Column name cannot be empty.")
        if isinstance(data, datatable.Frame):
            if isinstance(column, str) and (column not in data.names):
                raise ValueError(f"Column {column} not part of the data set.")
        elif isinstance(data, (pandas.DataFrame, h2o.H2OFrame)):
            if isinstance(column, str) and (column not in data.columns):
                raise ValueError(f"Column {column} not part of the data set.")
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    @staticmethod
    def to_h2o_frame(data, config=None):
        if not HAS_H2O:
            commons.raise_opt_import_err("h2o")

        if data is None:
            return None, False

        if isinstance(data, (str, _data.PersistedData)):
            h2o_utils.ensure_h2o3_running(False)
            return to_h2oframe(data)
        if not isinstance(data, h2o.H2OFrame):
            h2o_utils.ensure_h2o3_running(False)
            return to_h2oframe(InterpretableModel.to_pandas(data))
        return data, False

    @staticmethod
    def to_pandas(data, column_prefix="c"):
        """Convert data into a pandas.DataFrame

        If data isn't pandas.DataFrame, datatable.Frame or h2o.H2OFrame assume
        it does not contain column headers, and we'll need to generate them using
        column_prefix and column number (starting from 0).

        Parameters
        ----------
        data: anything which pandas.DataFrame supports
            Data to be converted.

        column_prefix: str
            A string prefix for the column names

        Returns
        -------
        pandas.DataFrame :
            Data frame.

        """
        h2o_utils.assert_is_type(column_prefix, str)

        if data is None:
            return None

        if isinstance(data, pandas.DataFrame):
            return data

        if isinstance(data, datatable.Frame):
            return data.to_pandas()

        if not HAS_H2O:
            commons.raise_opt_import_err("h2o")
        if isinstance(data, h2o.H2OFrame):
            return data.as_data_frame(use_pandas=True)

        # assumes other data types don't have explicit column names
        column_nr = 1 if len(numpy.shape(data)) == 1 else numpy.shape(data)[1]
        columns = [column_prefix + str(x) for x in range(column_nr)]
        frame = pandas.DataFrame(data, columns=columns)
        frame.columns = list(map(str, frame.columns))
        return frame

    @staticmethod
    def to_datatable(data, column_prefix="c"):
        """Convert data into a datatable.Frame

        If data isn't pandas.DataFrame, datatable.Frame or h2o.H2OFrame assume
        it does not contain column headers, and we'll need to generate them using
        column_prefix and column number (starting from 0).

        Parameters
        ----------
        data: anything which datatable.Frame supports
            Data to be converted.

        column_prefix: str
            A string prefix for the column names

        Returns
        -------
        datatable.Frame :
            Data frame.

        """
        h2o_utils.assert_is_type(column_prefix, str)

        if data is None:
            return None

        if isinstance(data, datatable.Frame):
            return data

        if isinstance(data, pandas.DataFrame):
            return datatable.Frame(data)

        if not HAS_H2O:
            commons.raise_opt_import_err("h2o")
        if isinstance(data, h2o.H2OFrame):
            return datatable.Frame(data.as_data_frame(use_pandas=True))

        # assumes other data types don't have explicit column names
        column_nr = 1 if len(numpy.shape(data)) == 1 else numpy.shape(data)[1]
        columns = [column_prefix + str(x) for x in range(column_nr)]
        frame = datatable.Frame(data, names=columns)
        frame.names = list(map(str, frame.names))
        return frame
