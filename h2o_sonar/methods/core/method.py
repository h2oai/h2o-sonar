# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import datetime
import json
from abc import ABC
from abc import abstractmethod

import datatable
import pandas
from pandas.api.types import is_bool_dtype
from pandas.api.types import is_numeric_dtype

from h2o_sonar import errors
from h2o_sonar.methods.utils import _method_diagnostics


class FeatureTypes:
    KEY_ID_FEATURES = "id"

    KEY_CATEGORICAL_FEATURES = "categorical"
    KEY_NUMERIC_FEATURES = "numeric"
    KEY_CATNUM_FEATURES = "catnum"

    KEY_DATE_FEATURES = "date"
    KEY_TIME_FEATURES = "time"
    KEY_DATE_TIME_FEATURES = "datetime"
    KEY_TEXT_FEATURES = "text"
    KEY_IMAGE_FEATURES = "image"

    KEY_DATE_FEATURES_FORMAT = "date-format"
    DEFAULT_DATE_FEATURE_FORMAT = "%Y%m%d"

    KEY_QUANTILE_BINS = "quantile-bin"


class FeaturesMetadata(FeatureTypes):
    """Utility class to build dictionary with features metadata. For instances as
    used/determined by a machine learning model. Every feature used by model is
    marked with its type (numeric, categorical or both) and characteristic (date,
    time, datetime, text, image, ID).

    """

    # NAMING CONVENTION through the sources:
    #   feature_meta ~ dict,
    #   features_metadata ~ class

    @staticmethod
    def create_blank_dict():
        return {
            FeaturesMetadata.KEY_ID_FEATURES: [],
            FeaturesMetadata.KEY_CATEGORICAL_FEATURES: [],
            FeaturesMetadata.KEY_NUMERIC_FEATURES: [],
            FeaturesMetadata.KEY_CATNUM_FEATURES: [],
            FeaturesMetadata.KEY_DATE_FEATURES: [],
            FeaturesMetadata.KEY_TIME_FEATURES: [],
            FeaturesMetadata.KEY_DATE_TIME_FEATURES: [],
            FeaturesMetadata.KEY_TEXT_FEATURES: [],
            FeaturesMetadata.KEY_IMAGE_FEATURES: [],
            FeaturesMetadata.KEY_DATE_FEATURES_FORMAT: [],
            FeaturesMetadata.KEY_QUANTILE_BINS: {},
        }

    @property
    def id_features(self) -> list:
        """ID features."""
        return self._features_meta.get(FeaturesMetadata.KEY_CATEGORICAL_FEATURES, [])

    @property
    def categorical_features(self) -> list:
        """Categorical features - can overlap with numeric features."""
        return self._features_meta.get(FeaturesMetadata.KEY_CATEGORICAL_FEATURES, [])

    @property
    def numeric_features(self) -> list:
        """Numeric features (can overlap with categorical features)"""
        return self._features_meta.get(FeaturesMetadata.KEY_NUMERIC_FEATURES, [])

    @property
    def categorical_numeric_features(self) -> list:
        return self._features_meta.get(FeaturesMetadata.KEY_CATNUM_FEATURES, [])

    @property
    def date_features(self) -> list:
        return self._features_meta.get(FeaturesMetadata.KEY_DATE_FEATURES, [])

    @property
    def format_date_features(self) -> list:
        """Format for date features - index of the format corresponds to the
        index of date feature."""
        return self._features_meta.get(FeaturesMetadata.KEY_DATE_FEATURES_FORMAT, [])

    @property
    def time_features(self) -> list:
        return self._features_meta.get(FeaturesMetadata.KEY_TIME_FEATURES, [])

    @property
    def date_time_features(self) -> list:
        return self._features_meta.get(FeaturesMetadata.KEY_DATE_TIME_FEATURES, [])

    @property
    def text_features(self) -> list:
        """Text features - dataset column is used as text feature by the model."""
        return self._features_meta.get(FeaturesMetadata.KEY_TEXT_FEATURES, [])

    @property
    def image_features(self) -> list:
        """Image features - column contains images and is used by the model."""
        return self._features_meta.get(FeaturesMetadata.KEY_IMAGE_FEATURES, [])

    @property
    def qtile_binning_features(self) -> dict:
        """Quantile binning specification for given features - key is the feature,
        value is quantile binning specification (the number of quantile bins to
        create e.g. 4 for quartiles)"""
        return self._features_meta.get(FeaturesMetadata.KEY_QUANTILE_BINS, [])

    def __init__(self, features_meta: dict | None = None):
        self._features_meta = features_meta or FeaturesMetadata.create_blank_dict()

    def empty(self) -> bool:
        """Return ``True`` if no feature metadata are set."""
        return not any(self._features_meta.values())

    def __str__(self):
        return str(self.to_json(2))

    def set(self, features_meta: dict):
        self._features_meta = features_meta

    def add(self, feature_type: str, feature_name: str):
        self._features_meta[feature_type] = feature_name

    def get(self, feature_name: str, default_value):
        return self._features_meta.get(feature_name, default_value)

    def to_dict(self):
        return self._features_meta.copy()

    def to_json(self, indent=None):
        return json.dumps(self.to_dict(), indent=indent)


class Method(ABC, FeatureTypes):
    """Abstract class for all MLI objects exposing interpretation mechanisms.

    Attributes:
    -----------
    method_type : str
        Method type.
    method_name : str
        Method name.
    interpretable_model : InterpretableModel
        MLI interpretable model.

    """

    # constants

    DEFAULT_GRID_RESOLUTION = 10

    KEY_CAT_WITH_NUM_BIN = "categorical_with_numeric_bin"

    LABEL_PREFIX_CLASS = "p_"
    LABEL_REGRESSION = LABEL_PREFIX_CLASS + str(0)

    # List of values that should be interpreted as missing values. Applies
    # both to numeric and string columns. Note that 'nan' is always
    # interpreted as a missing value for numeric columns.
    MISSING_VALUES = [
        "",
        "?",
        "None",
        "nan",
        "NA",
        "N/A",
        "unknown",
        "inf",
        "-inf",
        "1.7976931348623157e+308",
        "-1.7976931348623157e+308",
    ]

    @property
    def method_type(self):
        """Method type e.g. 'loco' or 'ice'."""
        return self._method_type

    @property
    def method_name(self):
        """Method name."""
        return self._method_name

    @property
    def interpretable_model(self):
        """Interpretable model."""
        return self._i_model

    @property
    def diagnostics(self):
        """Method diagnostics data."""
        return self._diagnostics

    def __init__(self, method_name, method_type, interpretable_model=None):
        """
        Create new methods instance.

        Parameters
        ----------
        method_type: str
            Method type.
        method_name: str
            Method name.
        interpretable_model: InterpretableModel
            interpretable model whose predict_method and current directory
            should be used

        """
        if not isinstance(method_name, str):
            raise ValueError("Method name cannot be string")
        if not method_name:
            raise ValueError("Method name cannot be empty")
        self._method_name = method_name

        if not isinstance(method_type, str):
            raise ValueError("Method type cannot be string")
        if not method_type:
            raise ValueError("Name cannot be empty")
        self._method_type = method_type

        self._i_model = interpretable_model
        self._diagnostics = _method_diagnostics.MethodDiagnostics()

    def _set_params(self, parameter_names, **kwargs):
        for p_name, p_value in kwargs.items():
            if p_name in parameter_names:
                setattr(self, p_name, p_value)

    def _method_precondition(self, predict_method):
        if predict_method is None:
            if self._i_model is None:
                raise ValueError("Predict method must be specified")
            if self._i_model.predict_method is None:
                raise ValueError("Predict method must be specified")

            predict_method = self._i_model.predict_method

        return predict_method

    @staticmethod
    def opt_import_err_msg(
        pckg_names: list[str] | str,
        method_name: str = "",
        method_type: str = "",
    ):
        log_name = (
            f"{method_name}/{method_type}: " if method_name or method_type else ""
        )
        if isinstance(pckg_names, list):
            pckg_names_fmt = ", ".join([f"'{p}'" for p in pckg_names])
            return (
                f"{log_name}{', '.join(pckg_names_fmt)} Python packages are required, "
                f"but not installed"
            )
        return (
            f"{log_name}: '{pckg_names}' Python package is required, but not installed"
        )

    def _opt_import_err_msg(self, pckg_names: list[str] | str) -> str:
        return Method.opt_import_err_msg(
            pckg_names=pckg_names,
            method_name=self._method_name,
            method_type=self.method_type,
        )

    def _raise_opt_import_err(self, pckg_names: list[str] | str):
        raise ImportError(self._opt_import_err_msg(pckg_names))

    #
    # ABSTRACT METHODS
    #

    @abstractmethod
    def explain(self, model, **kwargs):
        pass

    #
    # STATIC HELPERS
    #

    @staticmethod
    def is_missing_value(value):
        """Determine whether input represents a missing value.

        Parameters
        ----------
        value:
            Input value.

        Returns
        -------
        bool:
            `True` in case of missing value, `False` otherwise.

        """
        return str(value) in Method.MISSING_VALUES

    @staticmethod
    def create_date_aware_bins(
        features: list,
        frame,
        features_meta: dict = None,
        grid_resolution: int = DEFAULT_GRID_RESOLUTION,
        out_of_range_resolution: int = 0,
        date_format: str | list[str] = FeatureTypes.DEFAULT_DATE_FEATURE_FORMAT,
    ):
        """Create date aware bins (for basic formats) with given grid resolution.

        Parameters
        ----------
        features: list[int or str]
            A list of features for which date aware bins should be created.
        frame: datatable.Frame or pandas.core.frame.DataFrame
            Original data for which should be partial dependence computed.
        grid_resolution: int
            The number of equally spaced points used to create bins if
            the number of unique values is big.
        features_meta: dict
            Optional features metadata allowing to indicate whether given
            feature is date (use ``date`` key and list of feature names)
        out_of_range_resolution: int
            Number of out of range bins to create below / above the
            binning interval.
        date_format: str or [str]
            Pandas (Python string format based) date format to be used to
            decode featurs. Optinal list allows to specify per-feature date
            format.
            https://docs.python.org/3/library/datetime.html\
            #strftime-and-strptime-behavior

        Returns
        -------
        bins, oor_bins: tuple(list[list[object]], list[list[object]])
            Data values for each target feature for which we want to compute
            partial dependence, vector if for single target feature,
            otherwise a matrix.

        """
        frame = Method._check_date_aware_bins(
            features,
            frame,
            grid_resolution,
            out_of_range_resolution,
            date_format,
        )

        meta_filter = []
        for feature in features:
            if feature not in frame.names:
                raise ValueError(
                    f"Feature '{feature}' is not label of any input data column"
                )

            if (
                features_meta
                and Method.KEY_DATE_FEATURES in features_meta
                and feature in features_meta[Method.KEY_DATE_FEATURES]
            ):
                meta_filter.append(feature)

        if features_meta and meta_filter:
            features = meta_filter

        if not features:
            raise ValueError("At least one date feature must be specified")

        result_bins = []
        result_oor_bins = []
        for i, feature in enumerate(features):
            bins, oor_bins = Method._feature_date_aware_bins(
                frame,
                feature,
                grid_resolution,
                out_of_range_resolution,
                date_format if isinstance(date_format, str) else date_format[i],
            )
            result_bins.append(bins)
            if out_of_range_resolution:
                result_oor_bins.append(oor_bins)

        return (
            result_bins,
            (result_oor_bins if out_of_range_resolution else None),
        )

    @staticmethod
    def _check_date_aware_bins(
        features: list,
        X,
        grid_resolution: int,
        out_of_range_resolution: int,
        date_format: str | list[str],
    ):
        """Data aware bins parameter check."""
        if X is None:
            raise ValueError("Frame must be specified")
        if isinstance(X, pandas.DataFrame):
            # IMPROVE implement InterpretableModel.to_datatable
            X = datatable.Frame(X)
        elif not isinstance(X, datatable.Frame):
            raise errors.MliUnsupportedDataFormatError(
                f"Unsupported X data type: {type(X)}",
                "Use Pandas or datatable frame",
            )

        if X.shape == (0, 0):
            raise ValueError("Data frame cannot be empty")

        if not features:
            raise ValueError("At least one feature must be specified")
        if grid_resolution < 1:
            raise ValueError("Grid resolution must be positive integer")
        if out_of_range_resolution < 0:
            raise ValueError("Out of range resolution must be positive integer")
        if not date_format:
            raise ValueError("A date format must be specified")
        if not isinstance(date_format, (str, list)):
            raise ValueError("A date format must be either string or list of strings")

        return X

    @staticmethod
    def _feature_date_aware_bins(
        X,
        feature,
        grid_resolution: int = DEFAULT_GRID_RESOLUTION,
        out_of_range_resolution: int = 0,
        date_format: str = "%Y%m%d",
    ):
        """Calculate bins and out of range bins for particular feature."""
        unique_dates = datatable.unique(X[:, feature]).to_pandas()[feature]
        unique_dates = unique_dates[~unique_dates.apply(Method.is_missing_value)]
        unique_dates_time = pandas.to_datetime(unique_dates, format=date_format)

        bins = list(
            pandas.date_range(
                unique_dates_time.min(),
                unique_dates_time.max(),
                periods=grid_resolution,
            )
        )
        std_seconds = unique_dates_time.apply(lambda x: x.timestamp()).std()
        oor_bins = [
            unique_dates_time.min() - datetime.timedelta(seconds=(i + 1) * std_seconds)
            for i in range(out_of_range_resolution)
        ]
        oor_bins += [
            unique_dates_time.max() + datetime.timedelta(seconds=(i + 1) * std_seconds)
            for i in range(out_of_range_resolution)
        ]

        if X.ltypes[X.names.index(feature)] == datatable.ltype.int:
            bins = [int(x.strftime(date_format)) for x in bins]
            oor_bins = [int(x.strftime(date_format)) for x in oor_bins]
        else:
            bins = [x.strftime(date_format) for x in bins]
            oor_bins = [x.strftime(date_format) for x in oor_bins]

        return bins, oor_bins

    @staticmethod
    def _is_categorical(column):
        """Determine whether Pandas frame column represents categorical feature or not.

        Parameters
        ----------
        column: pandas.DataFrame
            Feature type to be checked.

        Returns
        -------
        bool:
            True if feature is categorical, else false.

        """
        if not is_numeric_dtype(column) or is_bool_dtype(column):
            return True
        return False
