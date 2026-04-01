# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import os
import pathlib
import tempfile
import zipfile
from enum import auto
from enum import Enum

import datatable
import numpy
import pandas

from h2o_sonar import loggers
from h2o_sonar.lib.api import commons
from h2o_sonar.utils import preprocessing
from h2o_sonar.utils import sampling


try:
    import h2o

    HAS_H2O = True
except ImportError:
    HAS_H2O = False


class ExplainableDatasetType(Enum):
    remote = auto()
    filesystem = auto()  # CSV, jay, ...
    datatable = auto()
    pandas = auto()
    unknown = auto()
    h2o3 = auto()


class ExplainableColumnMeta:
    """Dataset column metadata."""

    def __init__(
        self,
        name: str = "",
        data_type: str = "",
        logical_types: list | None = None,
        values_format: str = "",
        is_id: bool = False,
        is_numeric: bool = False,
        is_categorical: bool = False,
        count: int = 0,
        frequency: int = 0,
        unique: int = 0,
        max_value: int | None = None,
        min_value: int | None = None,
        mean: float | None = None,
        std: float | None = None,
        histogram_counts: list | None = None,
        histogram_ticks: list | None = None,
        properties: dict | None = None,
    ):
        """Constructor.

        Parameters
        ----------
        name : str
          Column name.
        data_type : str
          Column type represented as string in lowercase.
        logical_types : list
          Logical column types.
        values_format
          Value format e.g., data time format.
        is_id : bool
          ``True`` if the column is ID (frequency is equal to the number of columns).
        is_numeric : bool
          ``True`` if the column (can be/was used as) numeric.
        is_categorical : bool
          ``True`` if the column (can be/was used as) categorical. Not that column
          might be/used as both categorical and numeric.
        count : int
          Number of dataset rows.
        frequency : int
          Frequency of the most frequent value.
        unique : int
          Number of unique values in the column.
        max_value : int | None
          Maximum column value.
        min_value : int | None
          Minimum column value.
        mean :  float | None
          Mean column value.
        std :  float | None
          Standard deviation of column values.
        histogram_counts : list
          Histogram counts for the column.
        histogram_ticks : list
          Histogram ticks (corresponding to histogram values) for the column.
        properties: dict = {},
          Additional column metadata (extensibility point for forward compatibility).

        """
        self.name = name or ""
        self.data_type = data_type or ""
        self.logical_types = logical_types or []
        self.format = values_format or ""
        self.is_id = is_id
        self.is_numeric = is_numeric
        self.is_categorical = is_categorical
        self.count = count or 0
        self.frequency = frequency or 0
        self.unique = unique or 0
        self.max = max_value
        self.min = min_value
        self.mean = mean
        self.std = std
        self.histogram_counts = histogram_counts or []
        self.histogram_ticks = histogram_ticks or []

        # EXTENSIBILITY POINT
        self.properties = properties or {}

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "logical_types": self.logical_types,
            "format": self.format,
            "is_id": self.is_id,
            "is_numeric": self.is_numeric,
            "is_categorical": self.is_categorical,
            "count": self.count,
            "frequency": self.frequency,
            "unique": self.unique,
            "max": self.max,
            "min": self.min,
            "mean": self.mean,
            "std": self.std,
            "histogram_counts": self.histogram_counts,
            "histogram_ticks": self.histogram_ticks,
        }


class ExplainableDatasetMeta:
    """Dataset metadata - this class provides a uniform API to get basic EDA dataset
    metadata regardless dataset source, provider or implementation.

    """

    KEY_SHAPE = "shape"
    KEY_ROW_COUNT = "row_count"
    KEY_COLUMN_NAMES = "column_names"
    KEY_COLUMN_TYPES = "column_types"
    KEY_COLUMN_UNIQUES = "column_uniques"
    KEY_COLUMNS_CAT = "columns_cat"
    KEY_COLUMNS_NUM = "columns_num"
    KEY_FILE_PATH = "file_path"
    KEY_FILE_NAME = "file_name"
    KEY_FILE_SIZE = "file_size"
    KEY_MISSING_VALUES = "missing_values"
    KEY_COLUMNS_META = "columns_meta"
    KEY_ORIGINAL_DATASET_SAMPLED = "original_dataset_sampled"
    KEY_ORIGINAL_DATASET_PATH = "original_dataset_path"
    KEY_ORIGINAL_DATASET_SIZE = "original_dataset_size"
    KEY_ORIGINAL_DATASET_SHAPE = "original_dataset_shape"

    def __init__(
        self,
        shape: tuple | None = None,
        columns_meta: list[ExplainableColumnMeta] | None = None,
        column_names: list | None = None,
        column_types: list | None = None,
        column_uniques: list | None = None,
        columns_cat: list | None = None,
        columns_num: list | None = None,
        file_name: str = "",
        file_path: str = "",
        file_size: int = 0,
        key: str = "",
        missing_values: list | None = None,
    ):
        # per-column metadata
        self.columns_meta = columns_meta or []
        self.column_2_meta = {}
        # dataset shape
        self.shape = shape

        # INDEX: number of dataset rows
        self.row_count = shape[0] if shape else None
        # INDEX: dataset column names
        self.column_names = column_names or []
        # INDEX: dataset column types
        self.column_types = column_types or []
        # INDEX: number of unique values per column
        self.column_uniques = column_uniques or []
        # INDEX: list of categorical columns names
        self.columns_cat = columns_cat or []
        # INDEX: list of numeric columns names
        self.columns_num = columns_num or []

        # dataset file metadata (in case that dataset source was the file)
        self.file_name = file_name or ""
        self.file_path = file_path or ""
        self.file_size = file_size or 0

        # sampling: if this is a sampled dataset, then path / shape of the orig dataset
        self.original_dataset_sampled = False
        self.original_dataset_path = self.file_path
        self.original_dataset_size = self.file_size
        self.original_dataset_shape = self.shape
        # ^ attrs equal to the dataset values if the dataset was not sampled

        # dataset key/ID/UUID
        self.key = key or ""
        # missing values
        self.missing_values = missing_values or [
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

        self.__init_indices()

    def __init_indices(self):
        if (
            self.columns_meta
            and not self.column_names
            and not self.column_types
            and not self.column_uniques
            and not self.columns_cat
            and not self.columns_num
        ):
            for c in self.columns_meta:
                self.column_2_meta[c.name] = c
                self.column_names.append(c.name)
                self.column_types.append(c.data_type)
                self.column_uniques.append(c.unique)
                if c.is_numeric:
                    self.columns_num.append(c.name)
                if c.is_categorical:
                    self.columns_cat.append(c.name)
        else:
            self.column_2_meta = {cm.name: cm for cm in self.columns_meta}

    def __str__(self):
        return str(self.to_json(2))

    def has_column(self, column_name: str):
        return column_name in self.column_names

    def is_numeric_column(self, column_name: str):
        return column_name in self.columns_num

    def is_categorical_column(self, column_name: str):
        return column_name in self.columns_cat

    def get_column_meta(self, column_name: str):
        return self.column_2_meta.get(column_name, None)

    def copy(self):
        return ExplainableDatasetMeta(
            shape=self.shape,
            column_names=(
                self.column_names.copy() if self.column_names is not None else None
            ),
            column_types=(
                self.column_types.copy() if self.column_types is not None else None
            ),
            column_uniques=(
                self.column_uniques.copy() if self.column_uniques is not None else None
            ),
            columns_cat=(
                self.columns_cat.copy() if self.columns_cat is not None else None
            ),
            columns_num=(
                self.columns_num.copy() if self.columns_num is not None else None
            ),
            file_name=self.file_name,
            file_path=self.file_path,
            file_size=self.file_size,
            key=self.key,
            missing_values=(
                self.missing_values.copy() if self.missing_values is not None else None
            ),
            columns_meta=(
                self.columns_meta.copy() if self.columns_meta is not None else None
            ),
        )

    def to_dict(self):
        return {
            ExplainableDatasetMeta.KEY_SHAPE: str(self.shape),
            ExplainableDatasetMeta.KEY_ROW_COUNT: self.row_count,
            ExplainableDatasetMeta.KEY_COLUMN_NAMES: self.column_names,
            ExplainableDatasetMeta.KEY_COLUMN_TYPES: self.column_types,
            ExplainableDatasetMeta.KEY_COLUMN_UNIQUES: self.column_uniques,
            ExplainableDatasetMeta.KEY_COLUMNS_CAT: self.columns_cat,
            ExplainableDatasetMeta.KEY_COLUMNS_NUM: self.columns_num,
            ExplainableDatasetMeta.KEY_FILE_PATH: self.file_path,
            ExplainableDatasetMeta.KEY_FILE_NAME: self.file_name,
            ExplainableDatasetMeta.KEY_FILE_SIZE: self.file_size,
            ExplainableDatasetMeta.KEY_MISSING_VALUES: self.missing_values,
            ExplainableDatasetMeta.KEY_COLUMNS_META: (
                [cm.to_dict() for cm in self.columns_meta]
                if self.columns_meta is not None
                else []
            ),
            ExplainableDatasetMeta.KEY_ORIGINAL_DATASET_SAMPLED: (
                self.original_dataset_sampled
            ),
            ExplainableDatasetMeta.KEY_ORIGINAL_DATASET_PATH: (
                self.original_dataset_path
            ),
            ExplainableDatasetMeta.KEY_ORIGINAL_DATASET_SIZE: (
                self.original_dataset_size
            ),
            ExplainableDatasetMeta.KEY_ORIGINAL_DATASET_SHAPE: (
                self.original_dataset_shape
            ),
        }

    def to_json(self, indent=None):
        return json.dumps(self.to_dict(), indent=indent)


class ExplainableDataset:
    """Dataset with metadata - this class provides a uniform API to get dataset data
    regardless dataset source, provider or implementation.

    """

    KEY_METADATA = "metadata"
    KEY_DATA = "data"

    COL_BIAS = "bias"

    @staticmethod
    def frame_2_pandas(
        frame,
        columns: list | None = None,
        trim_to_columns: list | None = None,
    ) -> pandas.DataFrame:
        """Convert frame to Pandas.

        Parameters
        ----------
        frame :
          A frame to be converted.
        columns : list | None
          Optional list of column names to be used for newly created frame - column
          names are overwritten by this list.
        trim_to_columns : list | None
          Remove all columns that are not on this list from the result frame.

        Returns
        -------
        pandas.DataFrame :
          Pandas frame.

        """
        if frame is not None:
            if isinstance(frame, numpy.ndarray):
                pandas_frame = pandas.DataFrame(data=frame, columns=columns)
                return (
                    pandas_frame[trim_to_columns] if trim_to_columns else pandas_frame
                )
            elif isinstance(frame, datatable.Frame):
                frame = frame[:, trim_to_columns] if trim_to_columns else frame
                return frame.to_pandas()
            elif isinstance(frame, pandas.DataFrame):
                return frame[trim_to_columns] if trim_to_columns else frame
            elif not isinstance(frame, pandas.DataFrame):
                return frame.as_data_frame(use_pandas=True)
        return frame

    @staticmethod
    def frame_2_datatable(
        frame,
        columns: list | None = None,
        trim_to_columns: list | None = None,
    ) -> datatable.Frame:
        """Convert frame to datatable.

        Parameters
        ----------
        frame :
          A frame to be converted.
        columns : list | None
          Optional list of column names to be used for newly created frame - column
          names are overwritten by this list.
        trim_to_columns : list | None
          Remove all columns that are not on this list from the result frame.

        Returns
        -------
        pandas.DataFrame :
          Pandas frame.

        """
        if frame is not None:
            if isinstance(frame, numpy.ndarray):
                return (
                    datatable.Frame(frame, names=columns)[:, trim_to_columns]
                    if trim_to_columns
                    else datatable.Frame(frame, names=columns)
                )
            elif isinstance(frame, pandas.DataFrame):
                return (
                    datatable.Frame(frame)[:, trim_to_columns]
                    if trim_to_columns
                    else datatable.Frame(frame)
                )
            elif isinstance(frame, datatable.Frame):
                return frame[:, trim_to_columns] if trim_to_columns else frame
            elif not isinstance(frame, datatable.Frame):
                return (
                    datatable.Frame(frame.as_data_frame(use_pandas=True))[
                        :trim_to_columns
                    ]
                    if trim_to_columns
                    else datatable.Frame(frame.as_data_frame(use_pandas=True))
                )
        return frame

    @staticmethod
    def frame_2_numpy(frame, flatten: bool = False) -> numpy.ndarray | None:
        if frame is not None:
            if isinstance(frame, (datatable.Frame, pandas.DataFrame)):
                np_frame = frame.to_numpy()
            elif not isinstance(frame, numpy.ndarray):
                np_frame = numpy.ndarray(frame)
            else:
                np_frame = frame

            if flatten and np_frame.ndim == 2:
                return np_frame.flatten()
            else:
                return np_frame

        return frame

    @staticmethod
    def is_bias_col(col_name) -> bool:
        if col_name == ExplainableDataset.COL_BIAS or (
            col_name and col_name.startswith(f"{ExplainableDataset.COL_BIAS}.")
        ):
            return True
        return False

    @property
    def meta(self) -> ExplainableDatasetMeta:
        return self._meta

    @property
    def data(self) -> datatable.Frame:
        return self._data

    def __init__(self, data=None, meta=None, logger=None):
        self._data: datatable.Frame | None = data
        self._meta: ExplainableDatasetMeta | None = meta
        self.logger = logger or loggers.SonarPrintLogger()
        self.log_name = "ExplainableDataset"

    def __str__(self):
        return str(self.to_json(2))

    def prepare(
        self,
        drop_na_rows: bool = True,
        used_features: list | None = None,
        le_cat_variables: bool = True,
        cleaned_frame_type: (
            type[pandas.DataFrame] | type[datatable.Frame]
        ) = datatable.Frame,
        update: bool = False,
    ) -> tuple[
        datatable.Frame | pandas.DataFrame,
        list,
        preprocessing.MultiColumnLabelEncoderAbc,
        int,
    ]:
        """Method with commonly need actions to preprocess an explainable dataset.
        3rd party libraries often require, e.g., numeric features only, examples
        without N/A or undefined values, ... which this method ensures.

        Parameters
        ----------
        drop_na_rows : bool
          Drop rows with N/A values.
        used_features : list | None
          Trim dataset to used features.
        le_cat_variables : bool
          Do label encode non-numerical columns.
        cleaned_frame_type :
          Frame type to return - Pandas or datatable.
        update : bool
          If ``True``, set ``data`` field of this ``ExplainableDataset`` instance,
          else return cleaned dataset and keep ``data`` field intact.

        Returns
        -------
        Tuple[datatable.Frame, list[str], Any, int] :
          Result frame; non-numeric column names (label encoded); label encoder;
          number of dropped rows with N/A values.

        """
        dropped_rows_count: int = 0
        mcle = None

        # trim dataset to used features
        frame = self.data[:, used_features] if used_features else self.data

        # PERFORMANCE: avoid datatable > Pandas > datatable conversion
        frame = ExplainableDataset.frame_2_pandas(frame)

        if drop_na_rows:
            dropped_rows_count = frame.shape[0]
            frame = frame.dropna()
            dropped_rows_count = dropped_rows_count - frame.shape[0]
            if dropped_rows_count:
                self.logger.warning(
                    f"{self.log_name} cleaner: dropped {dropped_rows_count} "
                    f"rows with N/As",
                )

        # (label) encode categorical variables
        cat_variables = list(frame.select_dtypes(["object"]).columns)
        if le_cat_variables and cat_variables:
            mcle = preprocessing.get_multi_column_label_encoder(
                columns=numpy.asarray(cat_variables)
            )
            frame[cat_variables] = mcle.fit_transform(frame)
            self.logger.warning(
                f"{self.log_name} cleaner: label encoded {cat_variables} columns"
            )
        # ensure boolean columns to be of the int type
        int_variables = list(frame.select_dtypes(["bool"]).columns)
        if int_variables:
            for int_variable in int_variables:
                frame[int_variable] = frame[int_variable].astype(int)

        dt_frame = datatable.Frame(frame)

        if update:
            self._data = dt_frame

        return (
            dt_frame if cleaned_frame_type is datatable.Frame else frame,
            cat_variables,
            mcle,
            dropped_rows_count,
        )

    def transform(self, *args, **kwargs):
        """Transform the explainable dataset - sanitize, sample - and return new
        explainable dataset instance.

        """
        del args
        del kwargs

        # TODO: identity as tentative implementation
        return ExplainableDataset(
            data=self._data, meta=self._meta.copy() if self._meta else None
        )

    def sample(self, *args, **kwargs):
        """Sample the explainable dataset and return new instance."""
        del args
        del kwargs

        # TODO: identity as tentative implementation
        return ExplainableDataset(
            data=self._data, meta=self._meta.copy() if self._meta else None
        )

    def to_dict(self):
        return {
            ExplainableDataset.KEY_DATA: (
                str(type(self._data)) if self._data else "None"
            ),
            ExplainableDataset.KEY_METADATA: (
                self._meta.to_dict() if self._meta else "None"
            ),
        }

    def to_json(self, indent=None):
        return json.dumps(self.to_dict(), indent=indent)


class ExplainableDatatableDataset(ExplainableDataset):
    """Datatable based dataset."""

    def __init__(self, frame: datatable.Frame):
        ExplainableDataset.__init__(self)
        self._data = frame
        num_uniques = self._data.nunique().to_list()
        column_uniques = [item for sublist in num_uniques for item in sublist]
        column_types = [str(x).replace("ltype.", "") for x in self._data.ltypes]
        column_names = list(self._data.names)
        columns_meta = []
        for i, column_name in enumerate(column_names):
            is_id = False
            if (
                column_types[i] in ["int"]
                and column_uniques[i]
                and column_uniques[i] == frame.shape[0]
            ):
                is_id = True
            columns_meta.append(
                ExplainableColumnMeta(
                    name=column_name,
                    data_type=column_types[i],
                    unique=column_uniques[i],
                    is_numeric=column_types[i] in ["int", "real"],
                    is_categorical=column_types[i] not in ["int", "real"],
                    is_id=is_id,
                    count=column_uniques[i],
                )
            )
        shape = self._data.shape
        # fill dataset metadata (possibly on demand)
        self._meta = ExplainableDatasetMeta(
            shape=shape,
            column_names=column_names,
            column_types=column_types,
            column_uniques=column_uniques,
            columns_meta=columns_meta,
        )


class ExplainableDatasetHandle(commons.ResourceHandle):
    """Handle to a REMOTE dataset hosted by a remote system described by its
    connection configuration.

    ``ExplainableDatasetHandle`` differs from the ``ExplainerDataset`` in that it
    doesn't provide the actual dataset data, but only the metadata required to
    access the dataset.

    """

    @staticmethod
    def from_string(
        str_handle: str, h2o_sonar_config=None
    ) -> "ExplainableDatasetHandle":
        """Create a new instance of the dataset handle from the string."""
        (
            connection_key,
            resource_key,
            resource_version,
        ) = commons.ResourceHandle.parse_string_handle(str_handle)

        # validate connection name
        if h2o_sonar_config:
            if not h2o_sonar_config.has_connection(connection_key):
                raise ValueError(
                    f"Connection key '{connection_key}' not found in "
                    f"the H2O Sonar config"
                )

        return ExplainableDatasetHandle(
            connection_key=connection_key,
            dataset_key=resource_key,
            dataset_version=resource_version,
        )

    def __init__(
        self,
        connection_key: str,
        dataset_key: str,
        dataset_version: str = "",
    ):
        """Constructor.

        Parameters
        ----------
        connection_key: str
            Key of the connection configuration defined in the H2O Sonar config.
        dataset_key: str
            Key which uniquely identifies the dataset on the host system.
        dataset_version: str
            Optional dataset version which might be needed to uniquely identify
            the dataset on the host system.

        """
        commons.ResourceHandle.__init__(
            self,
            connection_key=connection_key,
            resource_key=dataset_key,
            version=dataset_version,
        )


#
# API
#


class DatasetApi:
    """Dataset API interface provides uniform API allowing explainers to use
    any dataset regardless format or location details.

    """

    def __init__(self, logger: loggers.SonarLogger | None = None):
        self.logger = logger or loggers.SonarPrintLogger()

    @staticmethod
    def create_dataset(
        dataset_src,
        dataset_type: ExplainableDatasetType = ExplainableDatasetType.unknown,
        target_col: str = "",
        sampled_dataset_path: str = "",
        sample_num_rows: int | None = None,
        sampler: sampling.DatasetSampler | None = None,
        **extra_params,
    ) -> ExplainableDataset:
        """Create explainable model.

        Parameters
        ----------
        dataset_src : ExplainableDataset | datatable.Frame | str | dict
          | pandas.DataFrame | Any
          Create dataset from given source: explainable dataset instance, datatable
          frame, H2OFrame, Pandas DataFrame, string (expect path to CSV, .jay or any
          other file type supported by datatable), dictionary (used to construct frame).
        dataset_type : ExplainableDatasetType
          Optional dataset type hint, which can be used to construct the dataset
          correctly.
        sampled_dataset_path : str
          Optional file path, which can be used to create a new file with the sampled
          dataset (if the datasets are sampled and if the sampling will be needed).
        target_col : str
          Optional target column name.
        sample_num_rows : int | None
          If ``None``, then automatically sample based on the dataset and RAM size.
          If > 0, then do sample the ``dataset`` to ``sample_num_rows`` number of rows.
          If == 0, then do NOT sample.
        sampler : DatasetSampler | None
          Sampling method (implementation) to be used - see ``h2o_sonar.utils.sampling``
          module (documentation) for available sampling methods. Use a sampler instance
          to use the specific sampling method.

        """
        del extra_params

        # data
        explainable_dataset = DatasetApi._create_dataset_data(
            dataset_src=dataset_src,
            dataset_type=dataset_type,
            target_col=target_col,
            sampled_dataset_path=sampled_dataset_path,
            sample_num_rows=sample_num_rows,
            sampler=sampler,
        )

        return explainable_dataset

    @staticmethod
    def _create_dataset_data(
        dataset_src,
        dataset_type: ExplainableDatasetType = ExplainableDatasetType.unknown,
        target_col: str = "",
        sampled_dataset_path: str = "",
        sample_num_rows: int | None = None,
        sampler: sampling.DatasetSampler | None = None,
    ) -> ExplainableDataset:
        if isinstance(dataset_src, ExplainableDataset):
            return dataset_src
        elif isinstance(dataset_src, datatable.Frame):
            if target_col and dataset_src[target_col].stype == datatable.bool8:
                # SHOULD NOT be here dataset_src[target_col].stype = ...?
                dataset_src[target_col] = datatable.int8
            return ExplainableDatatableDataset(dataset_src)
        elif isinstance(dataset_src, (str, pathlib.Path)):
            dataset_src = str(dataset_src)
            if os.path.isfile(dataset_src):
                # sampling and OOM protection
                if sample_num_rows is None:
                    sampler = sampler or sampling.StratifiedDatasetSampling()
                elif sample_num_rows == 0:
                    sampler = sampling.NoDatasetSampling(check_ram=True)
                else:
                    sampler = sampler or sampling.StratifiedDatasetSampling()

                # sample the dataset (if needed)
                sampled_dataset_path = sampled_dataset_path or os.path.join(
                    tempfile.mkdtemp(), "sampled_dataset"
                )
                (sampled, data, path) = sampler.sample_dataset(
                    dataset=dataset_src,
                    sampling_limit=sample_num_rows,
                    target_col=target_col,
                    sampled_dataset_path=sampled_dataset_path,
                )

                if target_col and data[target_col].stype == datatable.bool8:
                    # SHOULD NOT be here data[target_col].stype = ...? / NO SCALE OP
                    data[target_col] = datatable.int8

                e_dataset = ExplainableDatatableDataset(data)
                e_dataset.meta.original_dataset_sampled = sampled
                if sampled:
                    e_dataset.meta.file_path = path
                    e_dataset.meta.file_size = pathlib.Path(path).stat().st_size
                    e_dataset.meta.original_dataset_path = dataset_src
                    e_dataset.meta.original_dataset_size = (
                        pathlib.Path(dataset_src).stat().st_size
                    )
                    e_dataset.meta.original_dataset_shape = None
                else:
                    e_dataset.meta.file_path = dataset_src
                    e_dataset.meta.file_size = pathlib.Path(dataset_src).stat().st_size

                return e_dataset
            else:
                raise ValueError(
                    f"Unable to create dataset from path: '{dataset_src}' - file does "
                    f"not exist"
                )
        elif isinstance(dataset_src, dict):
            return ExplainableDatatableDataset(datatable.Frame(dataset_src))
        elif isinstance(dataset_src, pandas.DataFrame):
            return ExplainableDatatableDataset(datatable.Frame(dataset_src))
        elif HAS_H2O and isinstance(dataset_src, h2o.H2OFrame):
            # convert H2O frame to pandas then to datatable
            pandas_df = dataset_src.as_data_frame()
            return ExplainableDatatableDataset(datatable.Frame(pandas_df))
        else:
            raise ValueError(
                f"Unable to create explainable dataset - unknown dataset source: "
                f"{dataset_src} and type: '{dataset_type}'"
            )

    #
    # utilities
    #

    @staticmethod
    def write_dataset(
        dataset: datatable.Frame | pandas.DataFrame | pandas.Series | numpy.ndarray,
        path: str,
    ):
        if not path:
            raise ValueError("Unable to write dataset - path is empty")
        if dataset is None:
            raise ValueError("Dataset cannot be written - frame is empty")
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        if isinstance(dataset, datatable.Frame):
            DatasetApi.write_datatable_dataset(dataset, path)
        elif isinstance(dataset, (pandas.DataFrame, pandas.Series, numpy.ndarray)):
            DatasetApi.write_pandas_dataset(dataset, path)
        else:
            raise ValueError(
                f"Cannot write dataset - type {dataset.__class__.__name__} is not "
                f"supported: {dataset}"
            )

    @staticmethod
    def write_pandas_dataset(
        data: pandas.DataFrame | pandas.Series | numpy.ndarray, path: str
    ):
        table = datatable.Frame(data)
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        # IMPROVE: set _strategy= to tune how exactly will be the frame written
        table.to_jay(path)

    @staticmethod
    def write_datatable_dataset(dataset: datatable.Frame, path: str):
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        assert not any([datatable.ltype.obj == x for x in dataset.ltypes]), (
            f"Datatable drops object columns, ensure original Pandas data frame has no "
            f"object columns: {dataset.ltypes}"
        )
        dataset.materialize()
        # IMPROVE: set _strategy= to tune how exactly will be the frame written
        dataset.to_jay(path)

    @staticmethod
    def write_csv(
        dataset: datatable.Frame | pandas.DataFrame, path: str, bom: bool = False
    ):
        if not path:
            raise ValueError("Unable to write dataset - path is empty")
        if not dataset:
            raise ValueError("Dataset cannot be written - frame is empty")
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        if isinstance(dataset, pandas.DataFrame):
            # IMPROVE: set _strategy= to tune how exactly will be the frame written
            return datatable.Frame(dataset).to_csv(path, bom=bom)
        elif isinstance(dataset, datatable.Frame):
            # IMPROVE: set _strategy= to tune how exactly will be the frame written
            return dataset.to_csv(path, bom=bom)
        else:
            raise ValueError(
                "Dataset must be either a Pandas DataFrame or datatable Frame"
            )

    @staticmethod
    def zip_csv(csv_file_path):
        dst = csv_file_path.replace(".csv", ".zip")
        zf = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED)
        zf.write(csv_file_path)
        zf.close()
        return dst


def filter_importance_greater_than_zero(
    frame: datatable.Frame, label: str | None = None, skip_bias: bool = True
) -> datatable.Frame:
    """Filter out all columns with 0s values.

    Parameters
    ----------
    frame : datatable.Frame
      Frame to filter.
    label :
      Label for which to pull bias.
    skip_bias : bool
      If bias columns presents, do skip it.

    Returns
    -------
      Filtered frame.

    """
    feature_list = list(frame.names)

    bias_var: str = ""
    if skip_bias:
        bias_var = (
            ExplainableDataset.COL_BIAS
            if label is None or ExplainableDataset.COL_BIAS in feature_list
            else f"{ExplainableDataset.COL_BIAS}.{label}"
        )
        if bias_var in feature_list:
            feature_list.remove(bias_var)
        else:
            loggers.SonarPrintLogger().warning(
                f"Bias '{bias_var}' for label '{label}' not in Shapley frame with "
                f"columns '{feature_list}'"
            )
    summed_array = frame[:, feature_list].sum().to_numpy()[0]

    # drop all features with 0 importance - if all features are not 0s (constant models)
    if summed_array.any():
        non_zero_importance = [
            col_name for i, col_name in enumerate(feature_list) if summed_array[i] != 0
        ]
        if skip_bias and bias_var in feature_list:
            non_zero_importance.append(bias_var)
        frame = frame[:, non_zero_importance]

    return frame
