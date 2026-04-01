# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.


import numpy
import pandas

from h2o_sonar.lib.api import commons


try:
    from sklearn.preprocessing import LabelEncoder

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class MultiColumnLabelEncoderAbc:
    """Abstract base class for multi-column label encoders."""

    def fit(self, dframe):
        """Fit label encoder to Pandas columns."""
        raise NotImplementedError

    def fit_transform(self, dframe):
        """Fit label encoder and return encoded labels."""
        raise NotImplementedError

    def transform(self, dframe):
        """Transform labels to normalized encoding."""
        raise NotImplementedError

    def inverse_transform(self, dframe):
        """Transform labels back to original encoding."""
        raise NotImplementedError


def get_multi_column_label_encoder(columns=None) -> MultiColumnLabelEncoderAbc:
    if not HAS_SKLEARN:
        commons.raise_opt_import_err("scikit-learn")

    class MultiColumnLabelEncoder(MultiColumnLabelEncoderAbc, LabelEncoder):
        """Wraps sklearn ``LabelEncoder`` functionality for use on multiple columns of a
        Pandas ``DataFrame``.

        """

        def __init__(self, encoder_columns=None):
            """Multi column label encoder constructor.

            Parameters
            ----------
            encoder_columns : list | None
              Columns to label encode.

            """
            self.columns = encoder_columns
            # encoders
            self.all_classes_ = None
            self.all_encoders_ = None
            self.all_labels_ = None
            # actually encoded columns
            self.encoded_columns: list = []

        def fit(self, dframe):
            """Fit label encoder to Pandas columns.
            Access individual column classes via indexing `self.all_classes_`.
            Access individual column encoders via indexing `self.all_encoders_`

            """
            # if columns are provided, iterate through and get `classes_`
            if self.columns is not None:
                # ndarray to hold LabelEncoder().classes_ for each
                # column; should match the shape of specified `columns`
                self.all_classes_ = numpy.ndarray(
                    shape=self.columns.shape, dtype=object
                )
                self.all_encoders_ = numpy.ndarray(
                    shape=self.columns.shape, dtype=object
                )
                for idx, column in enumerate(self.columns):
                    # fit LabelEncoder to get `classes_` for the column
                    le = LabelEncoder()
                    le.fit(dframe.loc[:, column].values)
                    # append the `classes_` to our ndarray container
                    self.all_classes_[idx] = (
                        column,
                        numpy.array(le.classes_.tolist(), dtype=object),
                    )
                    # append this column's encoder
                    self.all_encoders_[idx] = le
                    self.encoded_columns.append(column)
            else:
                # no columns specified; assume all are to be encoded
                self.columns = dframe.iloc[:, :].columns
                self.all_classes_ = numpy.ndarray(
                    shape=self.columns.shape, dtype=object
                )
                for idx, column in enumerate(self.columns):
                    le = LabelEncoder()
                    le.fit(dframe.loc[:, column].values)
                    self.all_classes_[idx] = (
                        column,
                        numpy.array(le.classes_.tolist(), dtype=object),
                    )
                    self.all_encoders_[idx] = le
                    self.encoded_columns.append(column)

            return self

        def fit_transform(self, dframe):
            """Fit label encoder and return encoded labels.
            Access individual column classes via indexing `self.all_classes_`
            Access individual column encoders via indexing `self.all_encoders_`
            Access individual column encoded labels via indexing `self.all_labels_`

            """
            # if columns are provided, iterate through and get `classes_`
            if self.columns is not None:
                # ndarray to hold LabelEncoder().classes_ for each
                # column; should match the shape of specified `columns`
                self.all_classes_ = numpy.ndarray(
                    shape=self.columns.shape, dtype=object
                )
                self.all_encoders_ = numpy.ndarray(
                    shape=self.columns.shape, dtype=object
                )
                self.all_labels_ = numpy.ndarray(shape=self.columns.shape, dtype=object)
                for idx, column in enumerate(self.columns):
                    # instantiate LabelEncoder
                    le = LabelEncoder()
                    # fit and transform labels in the column
                    dframe.loc[:, column] = le.fit_transform(
                        dframe.loc[:, column].values
                    )
                    # append the `classes_` to our ndarray container
                    self.all_classes_[idx] = (
                        column,
                        numpy.array(le.classes_.tolist(), dtype=object),
                    )
                    self.all_encoders_[idx] = le
                    self.all_labels_[idx] = le
                    self.encoded_columns.append(column)
            else:
                # no columns specified; assume all are to be encoded
                self.columns = dframe.iloc[:, :].columns
                self.all_classes_ = numpy.ndarray(
                    shape=self.columns.shape, dtype=object
                )
                for idx, column in enumerate(self.columns):
                    le = LabelEncoder()
                    dframe.loc[:, column] = le.fit_transform(
                        dframe.loc[:, column].values
                    )
                    self.all_classes_[idx] = (
                        column,
                        numpy.array(le.classes_.tolist(), dtype=object),
                    )
                    self.all_encoders_[idx] = le
                    self.encoded_columns.append(column)
            return dframe.loc[:, self.columns].values

        def transform(self, dframe):
            """Transform labels to normalized encoding."""
            if self.columns is not None:
                for idx, column in enumerate(self.columns):
                    dframe.loc[:, column] = self.all_encoders_[idx].transform(
                        dframe.loc[:, column].values
                    )
            else:
                self.columns = dframe.iloc[:, :].columns
                for idx, column in enumerate(self.columns):
                    dframe.loc[:, column] = self.all_encoders_[idx].transform(
                        dframe.loc[:, column].values
                    )
            return dframe.loc[:, self.columns].values

        def inverse_transform(self, dframe):
            """Transform labels back to original encoding."""
            if self.columns is not None:
                for idx, column in enumerate(self.columns):
                    dframe.loc[:, column] = self.all_encoders_[idx].inverse_transform(
                        dframe.loc[:, column].values
                    )
            else:
                self.columns = dframe.iloc[:, :].columns
                for idx, column in enumerate(self.columns):
                    dframe.loc[:, column] = self.all_encoders_[idx].inverse_transform(
                        dframe.loc[:, column].values
                    )
            return dframe.loc[:, self.columns].values

    return MultiColumnLabelEncoder(columns)


def categorical_encoder(
    X: pandas.DataFrame,
) -> tuple[pandas.DataFrame, MultiColumnLabelEncoderAbc, list]:
    categorical_variables = list(X.select_dtypes(["object"]).columns)
    mcle = None
    if categorical_variables:
        mcle = get_multi_column_label_encoder(
            columns=numpy.asarray(categorical_variables)
        )
    if mcle:
        X[categorical_variables] = mcle.fit_transform(X)

    return X, mcle, categorical_variables
