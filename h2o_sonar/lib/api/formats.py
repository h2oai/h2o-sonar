# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import abc
import json
import os
import traceback
from abc import ABC
from typing import Any

import datatable
import datatable as dt
import pandas as pd
from datatable import f as dtf

from h2o_sonar import errors
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api.commons import FilterEntry
from h2o_sonar.lib.api.commons import MimeType
from h2o_sonar.lib.integrations import mv_adapter


# constants
PersistenceDataType = persistences.PersistenceDataType


# dict: explanation_format -> class
_rml_byor_explanation_formats: dict = dict()


def get_custom_explanation_formats():
    if not _rml_byor_explanation_formats:
        # find all ABC format subclasses
        classes = [ExplanationFormat]
        while classes:
            parent = classes.pop()
            for child in parent.__subclasses__():
                if child not in _rml_byor_explanation_formats:
                    _rml_byor_explanation_formats[
                        f"{child.__module__}.{child.__name__}"
                    ] = child
                    classes.append(child)
        _rml_byor_explanation_formats.pop(
            f"{ExplanationFormat.__module__}.{ExplanationFormat.__name__}",
            None,
        )

    return _rml_byor_explanation_formats


class ExplanationFormat(ABC):
    """Base class of explanation representation.

    Representation is serialization of explanation in a format like JSon or CSV.
    Representation has a MIME type. It can be formed by one or more files, but at least
    one file must be provided.

    """

    """Format MIME type identifier."""
    mime: str = None

    KEY_ID = "id"
    KEY_ACTION = "action"
    KEY_ACTUAL = "actual"
    KEY_ACTION_TYPE = "action_type"
    KEY_BIAS = "bias"
    KEY_CATEGORICAL = "categorical"
    KEY_DATA = "data"
    KEY_METADATA = "metadata"
    KEY_DATA_HISTOGRAM = "data_histogram"
    KEY_DATA_HISTOGRAM_CAT = "data_histogram_categorical"
    KEY_DATA_HISTOGRAM_NUM = "data_histogram_numerical"
    KEY_DEFAULT_CLASS = "default_class"
    KEY_DOC = "documentation"
    KEY_EXPLAINER_JOB_KEY = "explainer_job_key"
    KEY_FEATURES = "features"
    KEY_FEATURE_TYPE = "feature_type"
    KEY_FEATURE_VALUE = "feature_value"
    KEY_FILES = "files"
    KEY_FILES_DETAILS = "files_details"
    KEY_IS_MULTI = "is_multinomial"
    KEY_ITEM_ORDER = "order"
    KEY_LABEL = "label"
    KEY_METRICS = "metrics"
    KEY_KEYWORDS = "keywords"
    KEY_MLI_KEY = "mli_key"
    KEY_NUMERIC = "numeric"
    KEY_DATE = "date"
    KEY_TIME = "time"
    KEY_DATE_TIME = "datetime"
    KEY_ON_DEMAND = "on_demand"
    KEY_ON_DEMAND_PARAMS = "on_demand_params"
    KEY_PAGE_OFFSET = "page_offset"
    KEY_PAGE_SIZE = "page_size"
    KEY_RAW_FEATURES = "raw_features"
    KEY_ROWS_PER_PAGE = "rows_per_page"
    KEY_RUNNING_ACTION = "running-action"
    KEY_SCOPE = "scope"
    KEY_SYNC_ON_DEMAND = "synchronous_on_demand_exec"
    KEY_TOTAL_ROWS = "total_rows"
    KEY_Y_FILE = "y_file"
    KEY_VALUE = "value"
    KEY_FILES_NUMCAT_ASPECT = "files_numcat_aspect"

    SCOPE_GLOBAL = "global"
    SCOPE_LOCAL = "local"

    FEATURE_TYPE_NUM = "numeric"
    FEATURE_TYPE_CAT = "categorical"
    FEATURE_TYPE_DATE = "date"
    FEATURE_TYPE_TIME = "time"
    FEATURE_TYPE_DATETIME = "datetime"
    FEATURE_TYPE_CAT_NUM = "catnum"

    KEYWORD_RESIDUALS = "residuals"

    LABEL_REGRESSION = "None (Regression)"

    DEFAULT_PAGE_SIZE = 20

    FILE_PREFIX_EXPLANATION_IDX = (
        f"{persistences.ExplainerPersistence.FILE_EXPLANATION}."
    )

    @classmethod
    def is_on_demand(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
    ) -> tuple[bool, dict | None]:
        """Returns ``True`` in case that there is no pre-computed (cached) local
        explanation and it must be calculated on demand.

        Returns
        -------
        bool:
          ``True`` if the representation is calculated on demand.
        dict:
          On-demand calculation parameters.

        """
        return False, None

    @classmethod
    def load_meta(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        explanation_format: str,
    ) -> dict:
        """Load representation metadata with class identifier and MIME."""
        meta_path = persistence.get_explanation_meta_path(
            explanation_type, explanation_format
        )
        return persistence.store.load_json(meta_path)

    @classmethod
    def is_paged(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
    ) -> bool:
        """Returns ``True`` in case that representation supports paging."""
        return False

    @classmethod
    def get_page(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        page_offset: int,
        page_size: int,
        result_format: str,
        explanation_filter: list[FilterEntry],
    ) -> str:
        """Get global explanation page."""
        raise errors.MliError(
            f"{cls.__name__} representation does not support paged explanations"
        )

    @classmethod
    def get_local_explanation(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        row: int,
        explanation_filter: list[FilterEntry],
        **extra_params,
    ) -> str:
        """Get local explanation for given dataset row and feature/class/...
        specified by explanation filter. Local explanation is returned as string.

        """
        raise errors.MliError(
            f"{cls.__name__} representation does not support local explanations"
        )

    @property
    def explanation(self):
        return self._explanation

    @property
    def index_file_name(self) -> str:
        """Get (mandatory) index file name which typically references all other files
        along with various metadata.

        """
        return self._idx_file_name

    @property
    def file_names(self) -> list[str]:
        """Get file names which form the representation.

        Hints:

        * representation is formed by flat structure of files without directories
        * representation data are not kept in memory - list of file names is sufficient

        """
        return self._file_names

    def __init__(
        self,
        explanation,
        format_data,
        format_file: str | None,
        extra_format_files: list | None = None,
        file_extension: str = "bin",
        persistence: persistences.Persistence | None = None,
    ):
        """Create new representation.

        Parameters
        ----------
        explanation: explanations.Explanation
          Explanation.
        format_data:
          (Index) file data. Either `file data` or `file` to be provided.
        format_file:
          Source file to copy and make it representation's index file. Either
          `file data` or `file` to be provided.
        extra_format_files:
          Additional files to copy to the representation's directory.
        file_extension: str
          (Index) file extension.
        persistence : persistences.Persistence | None
          Persistence store to save and load explanation representations.

        """
        self._persistence = persistence or persistences.FilesystemPersistence()
        self._explanation = explanation
        self._file_names: list[str] = list()

        if format_data and format_file:
            raise ValueError("Either data or file to copy to be specified, not both")
        if not self.mime:
            raise ValueError(
                "Format MIME type must be specified by representation class"
            )
        if not file_extension:
            raise ValueError("Format file extension must be specified")

        self._idx_file_name: str = f"{self.FILE_PREFIX_EXPLANATION_IDX}{file_extension}"
        self._check_new_file(format_data, format_file, self._idx_file_name)

        if format_data:
            self.add_data(format_data=format_data, file_name=self._idx_file_name)
        else:
            self.add_file(format_file=format_file, file_name=self._idx_file_name)

        if extra_format_files:
            for extra_file in extra_format_files:
                try:
                    extra_base_dir = persistences.Persistence.key_folder(format_file)
                    self.add_file(
                        format_file=extra_file,
                        file_name=extra_file.replace(
                            f"{extra_base_dir}/" if extra_base_dir else "", ""
                        ),
                    )
                except Exception:
                    pass

        self._add_meta()

        explanation.add_format(self)

    def __str__(self):
        result: str = (
            f"{self.__class__.__name__}\n"
            f"  MIME             : {self.mime}\n"
            f"  index file name  : {self._idx_file_name}\n"
            f"  files            : {self._file_names}\n"
            f"  explanation type : {self._explanation.__class__.__name__}\n"
        )
        return result

    KEY_NAME = "name"
    KEY_FULLNAME = "full_name"
    KEY_MIME = "mime"

    def _check_new_file(
        self,
        format_data: str | None,
        format_file: str | None,
        file_name: str,
    ):
        if (not format_data and not format_file) or (format_data and format_file):
            raise ValueError("Either format file or data must be specified")
        if file_name in self.file_names:
            raise ValueError(
                f"Format file `{file_name}` already exist in the representation "
            )

    def _check_save_load(self):
        if self.explanation:
            if self.explanation.explainer:
                if self.explanation.explainer.persistence:
                    return
                else:
                    raise RuntimeError(
                        "Unable to save representation as persistence is not set"
                    )
            else:
                raise RuntimeError(
                    "Unable to save representation as explanation is not set"
                )
        else:
            raise RuntimeError(
                "Unable to save representation as explanation is not set"
            )

    def _pre_add_data(self, format_data, file_name: str):
        self._check_save_load()
        self._check_new_file(format_data, None, file_name)
        self._file_names.append(file_name)
        dir_path = self.explanation.explainer.persistence.get_explanation_dir_path(
            self.explanation.explanation_type(), self.mime
        )
        self.explanation.explainer.persistence.store.make_dir(dir_path)
        return dir_path

    def _pre_add_file(self, format_file: str, file_name: str):
        self._check_save_load()
        self._check_new_file(None, format_file, file_name)
        self._file_names.append(file_name)
        return persistences.Persistence.make_key(
            self.explanation.explainer.persistence.get_explanation_dir_path(
                self.explanation.explanation_type(), self.mime
            ),
            file_name,
        )

    def _pre_get_data(self, file_name: str):
        self._check_save_load()
        if file_name not in self._file_names:
            raise ValueError(f"File name `{file_name}` not known to representation")

        return self.explanation.explainer.persistence.get_explanation_file_path(
            explanation_type=self.explanation.explanation_type(),
            explanation_format=self.mime,
            explanation_file=file_name,
        )

    @staticmethod
    def _add_frame_to_store(
        path: str,
        format_data: datatable.Frame,
        persistence: persistences.Persistence,
    ):
        temp_path = (
            persistences.Persistence.make_temp_file("frame.jay")
            if persistences.PersistenceType.file_system != persistence.type
            else path
        )
        format_data.to_jay(temp_path)
        if temp_path != path:
            binary_data = persistence.internal.load(temp_path)
            persistence.save(key=path, data=binary_data)
            persistences.Persistence.delete_temp_dir(os.path.dirname(temp_path))

    def add_data(self, format_data: str, file_name: str | None = None):
        """Add TEXT data as new explanation representation file.  Child classes with
        binary data to override this class.

        Parameters
        ----------
        format_data:
          Data to store as new explanation's format file.
        file_name: str
          Representation file name or file relative path.
        """
        self._pre_add_data(format_data, file_name if file_name else self._idx_file_name)
        file_path = self.explanation.explainer.persistence.get_explanation_file_path(
            self.explanation.explanation_type(),
            self.mime,
            explanation_file=file_name,
        )
        self._persistence.save(key=file_path, data=format_data)
        return self

    def update_data(self, format_data: str, file_name: str | None = None):
        self._check_save_load()
        file_path = self.explanation.explainer.persistence.get_explanation_file_path(
            self.explanation.explanation_type(),
            self.mime,
            explanation_file=file_name,
        )
        if self._persistence.is_file(file_path):
            self._persistence.update(key=file_path, data=format_data)
        else:
            raise errors.MliError(
                f"Unable to update file {file_path} as it doesn't exist"
            )
        return self

    def _add_meta(self):
        """Metadata file is used to determine which representation persisted data."""
        fullname = f"{self.__class__.__module__}.{self.__class__.__name__}"
        metadata: dict = {
            ExplanationFormat.KEY_NAME: self.__class__.__name__,
            ExplanationFormat.KEY_FULLNAME: fullname,
            ExplanationFormat.KEY_MIME: self.mime,
        }
        meta_path = self.explanation.explainer.persistence.get_explanation_meta_path(
            self.explanation.explanation_type(), self.mime
        )
        self._persistence.save_json(key=meta_path, data=metadata)

    def add_file(self, format_file: str, file_name: str | None = None) -> str:
        """Copy file to representation as new explanation representation file.

        Parameters
        ----------
        format_file:
          Source file to store (copy) as new explanation's format file.
        file_name: str
          Representation file name or file relative path.
        """
        file_path = self._pre_add_file(
            format_file, file_name if file_name else self._idx_file_name
        )
        self._persistence.make_dir(persistences.Persistence.key_folder(file_path))
        self._persistence.copy_file(from_key=format_file, to_key=file_path)
        return file_path

    def get_data(self, file_name: str | None = None):
        file_path = self._pre_get_data(file_name if file_name else self._idx_file_name)
        self._persistence.load(key=file_path)


class GrammarOfMliFormat:
    """Format class which is child of Grammar of MLI format class is supported in
    H2O Sonar UI - there is UI component which will render such format in an
    (interactive) chart.

    """

    _is_gom_ui_supported = True

    @classmethod
    def is_grammar_of_mli(cls) -> bool:
        """Will representation be rendered in UI?"""
        return cls._is_gom_ui_supported


class ExplanationFormatUtils:
    @staticmethod
    def get_page(data, page_offset: int, page_size: int):
        """Get page of given data.

        Parameters
        ----------
        data:
          Data to page.
        page_offset: int
          Positive integer or 0 with page offset.
        page_size: int
          Page size, returns all data entries if 0 or negative integer.

        """
        if isinstance(data, list):
            if not data:
                return data
            # don't fail, be robust
            page_offset = page_offset if page_offset >= 0 else 0
            if page_offset > len(data):
                raise ValueError(
                    f"Page offset out of range (data/offset): {len(data)}/{page_offset}"
                )
            page_size = page_size if page_size >= 0 else 0
            if not page_offset and (not page_size or len(data) < page_size):
                return data
            return data[page_offset : page_offset + page_size]

        raise ValueError(f"Unsupported type to do paging: {type(data)}")


class TextCustomExplanationFormat(ExplanationFormat):
    mime = MimeType.MIME_TEXT

    FILTER_FEATURE = "explain_feature"
    FILTER_CLASS = "explain_class"
    FILTER_NUMCAT = "explain_numcat"

    FILE_IS_ON_DEMAND = "IS_ON_DEMAND"

    def __init__(
        self,
        explanation,
        format_data: str,
        format_file: str | None,
        extra_format_files: list | None = None,
        persistence: persistences.Persistence | None = None,
    ):
        ExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=format_data,
            format_file=format_file,
            file_extension=MimeType.ext_for_mime(self.mime),
            extra_format_files=extra_format_files,
            persistence=persistence,
        )

    def add_data(self, format_data: str, file_name: str | None = None):
        self._pre_add_data(format_data, file_name if file_name else self._idx_file_name)
        file_path = self.explanation.explainer.persistence.get_explanation_file_path(
            self.explanation.explanation_type(),
            self.mime,
            explanation_file=file_name,
        )
        self._persistence.save(key=file_path, data=format_data)

        return self

    def add_file(self, format_file: str, file_name: str | None = None):
        file_path = self._pre_add_file(
            format_file, file_name if file_name else self._idx_file_name
        )
        self._persistence.make_dir(persistences.Persistence.key_folder(file_path))
        self._persistence.copy_file(from_key=format_file, to_key=file_path)
        return self

    def set_on_demand(self, is_on_demand: bool, mime: str = ""):
        """Indicate that representation is on-demand."""
        if is_on_demand:
            path = self.explanation.explainer.persistence.get_explanation_file_path(
                explanation_type=self.explanation.explanation_type(),
                explanation_format=mime or TextCustomExplanationFormat.mime,
                explanation_file=TextCustomExplanationFormat.FILE_IS_ON_DEMAND,
            )
            self._persistence.touch(path)

    @staticmethod
    def set_index_commons(
        index_dict: dict,
        classes: list[str],
        default_class: str = "",
        metrics: list | None = None,
        keywords: int | None = None,
        doc: str = "",
        total_rows: int | None = None,
    ):
        index_dict[ExplanationFormat.KEY_METRICS] = [] if not metrics else metrics
        index_dict[ExplanationFormat.KEY_KEYWORDS] = [] if not keywords else keywords
        if default_class:
            index_dict[ExplanationFormat.KEY_DEFAULT_CLASS] = default_class
        elif classes and len(classes):
            index_dict[ExplanationFormat.KEY_DEFAULT_CLASS] = classes[0]
        if total_rows is not None:
            index_dict[ExplanationFormat.KEY_TOTAL_ROWS] = total_rows
        if doc:
            index_dict[ExplanationFormat.KEY_DOC] = doc

    def update_index_file(
        self,
        index_dict: dict,
        metrics: list | None = None,
        total_rows: int | None = None,
    ):
        if total_rows is not None:
            index_dict[ExplanationFormat.KEY_TOTAL_ROWS] = total_rows
        if metrics is not None:
            index_dict[ExplanationFormat.KEY_METRICS] = metrics
        self.update_data(json.dumps(index_dict, indent=4))
        return self

    def get_data(
        self,
        file_name: str | None = None,
        data_type: PersistenceDataType | None = None,
    ):
        file_path = self._pre_get_data(file_name if file_name else self._idx_file_name)
        return self._persistence.load(key=file_path, data_type=data_type)

    @classmethod
    def is_on_demand(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
    ) -> tuple[bool, dict | None]:
        return (
            persistence.store.is_file(
                persistence.get_explanation_file_path(
                    explanation_type=explanation_type,
                    explanation_format=TextCustomExplanationFormat.mime,
                    explanation_file=TextCustomExplanationFormat.FILE_IS_ON_DEMAND,
                )
            ),
            {},
        )


class DatatableCustomExplanationFormat(ExplanationFormat):
    mime = MimeType.MIME_DATATABLE

    def __init__(
        self,
        explanation,
        frame: dt.Frame,
        frame_file: str,
        persistence: persistences.Persistence | None = None,
    ):
        ExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=frame,
            format_file=frame_file,
            file_extension=MimeType.ext_for_mime(MimeType.MIME_DATATABLE),
            persistence=persistence,
        )

    # override
    def add_data(self, format_data: dt.Frame, file_name: str | None = None):
        self._check_save_load()

        persist = self.explanation.explainer.persistence
        persist.store.make_dir(
            persist.get_explanation_dir_path(
                self.explanation.explanation_type(), self.mime
            )
        )
        self._add_frame_to_store(
            path=persist.get_explanation_file_path(
                self.explanation.explanation_type(), self.mime
            ),
            format_data=format_data,
            persistence=persist.store,
        )

    # override
    def get_data(self, file_name: str | None = None):
        del file_name
        self._check_save_load()

        persist = self.explanation.explainer.persistence
        return dt.fread(
            persist.get_explanation_file_path(
                self.explanation.explanation_type(), self.mime
            )
        )


class CsvFormatCustomExplanationFormat(ExplanationFormat):
    mime = MimeType.MIME_CSV

    def __init__(
        self,
        explanation,
        frame: dt.Frame,
        frame_file: str,
        persistence: persistences.Persistence | None = None,
    ):
        ExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=frame,
            format_file=frame_file,
            file_extension=MimeType.ext_for_mime(MimeType.MIME_CSV),
            persistence=persistence,
        )

    # override
    def add_data(self, format_data: dt.Frame, file_name: str | None = None):
        self._check_save_load()

        persist = self.explanation.explainer.persistence
        persist.store.make_dir(
            persist.get_explanation_dir_path(
                self.explanation.explanation_type(), self.mime
            )
        )
        self._add_frame_to_store(
            path=persist.get_explanation_file_path(
                self.explanation.explanation_type(), self.mime
            ),
            format_data=format_data,
            persistence=persist.store,
        )

    # override
    def get_data(self, file_name: str | None = None):
        del file_name
        self._check_save_load()

        persist = self.explanation.explainer.persistence
        return dt.fread(
            persist.get_explanation_file_path(
                self.explanation.explanation_type(), self.mime
            )
        )


class CustomJsonFormat(TextCustomExplanationFormat):
    """Representation of custom JSon format."""

    mime = MimeType.MIME_JSON

    def __init__(
        self,
        explanation,
        json_data: str = None,
        json_file: str = None,
        persistence: persistences.Persistence | None = None,
    ):
        TextCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=json_data,
            format_file=json_file,
            persistence=persistence,
        )

    @staticmethod
    def validate_data(json_data: str) -> str:
        return json_data

    @classmethod
    def load_index_file(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        mime: str = MimeType.MIME_JSON,
    ) -> dict:
        """Load index file and check parameters.

        Returns
        -------
        dict:
          Index file as dictionary.

        """
        #  index file
        idx_path = persistence.get_explanation_file_path(explanation_type, mime)
        return persistence.store.load_json(key=idx_path)


class CustomCsvFormat(TextCustomExplanationFormat):
    """Representation of custom JSon format."""

    mime = MimeType.MIME_CSV

    def __init__(
        self,
        explanation,
        frame: dt.Frame,
        persistence: persistences.Persistence | None = None,
    ):
        TextCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=frame.to_csv(),
            format_file=None,
            persistence=persistence,
        )

    @staticmethod
    def validate_data(json_data: str) -> str:
        return json_data


class IceJsonDatatableFormat(TextCustomExplanationFormat, GrammarOfMliFormat):
    """Individual conditional explanation as per-feature and class datatable frames
    with JSon index file.

    JSon representation index file example:

    .. code-block:: text

        {
            "features": {
                "PAY_0": {
                    "order": 0,
                    "feature_type": ["categorical"],
                    "files": {
                        "rec_class": "ice_feature_0_class_0.jay"
                        "blue_class": "ice_feature_0_class_1.jay"
                        "white_class": "ice_feature_0_class_2.jay"
                    }
                },
                ...
            },
            "metrics": [{"RMSE": 0.03}],
            "y_file": "y_hat.jay",
            "on_demand": false
        }

    or (if on demand e.g. in case of sampled dataset):

    .. code-block:: text

        {
            "on_demand": true
            "on_demand_parameters": ...
        }

    Datatable representation data file example:

    .. code-block:: python

        > datatable.fread("ice_feature_0_class_0.jay")

    .. code-block:: text

           |       -2        -1         0         1         2         7
        -- + --------  --------  --------  --------  --------  --------
         0 | 0.390716  0.390716  0.390716  0.390716  0.531548  0.531548
         1 | 0.38681   0.38681   0.38681   0.38681   0.508216  0.508216
         2 | 0.425908  0.425908  0.425908  0.425908  0.536061  0.536061
         ...

    Remarks:

    * ``y_file``    ... datatable frame with predictions for every X dataset instance
    * ``on_demand`` ... `true` if there is no cached ICE and it must be computed

    """

    mime = MimeType.MIME_JSON_DATATABLE

    KEY_COL_NAME = "column_name"
    KEY_PREDICTION = "prediction"
    KEY_FEATURE_VALUE = "feature_value"
    KEY_BIN = "bin"
    KEY_BINS = "bins"
    KEY_BINS_NUMCAT_ASPECT = "bins_numcat_aspect"
    KEY_ICE = "ice"

    FILE_Y_FILE = "y_hat.jay"

    def __init__(
        self,
        explanation,
        json_data: str = None,
        json_file: str = None,
        persistence: persistences.Persistence | None = None,
    ):
        TextCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=IceJsonDatatableFormat.validate_data(json_data),
            format_file=json_file,
            persistence=persistence,
        )

    @staticmethod
    def validate_data(json_data: str) -> str:
        return json_data

    @staticmethod
    def serialize_index_file(
        features: list[str],
        classes: list[str],
        default_class: str = "",
        features_meta: dict | None = None,
        metrics: list | None = None,
        doc: str = "",
        y_file: str | None = None,
    ) -> tuple[dict, str]:
        return PartialDependenceJSonFormat.serialize_index_file(
            features=features,
            classes=classes,
            default_class=default_class,
            features_meta=features_meta,
            metrics=metrics,
            doc=doc,
            data_file_prefix="ice",
            data_file_suffix="jay",
            y_file=y_file,
        )

    @staticmethod
    def serialize_on_demand_index_file(on_demand_params: dict) -> str:
        return json.dumps(
            {
                IceJsonDatatableFormat.KEY_ON_DEMAND: True,
                IceJsonDatatableFormat.KEY_ON_DEMAND_PARAMS: on_demand_params,
            }
        )

    @classmethod
    def load_index_file(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        mime: str = MimeType.MIME_JSON_DATATABLE,
    ) -> dict:
        """Load index file and check parameters.

        Returns
        -------
        dict:
          Index file as dictionary.

        """
        #  index file
        idx_path = persistence.get_explanation_file_path(explanation_type, mime)
        return persistence.store.load_json(key=idx_path)

    @classmethod
    def is_on_demand(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
    ) -> tuple[bool, dict | None]:
        idx_dict = IceJsonDatatableFormat.load_index_file(persistence, explanation_type)

        if idx_dict and idx_dict.get(IceJsonDatatableFormat.KEY_ON_DEMAND, False):
            return (
                True,
                idx_dict.get(IceJsonDatatableFormat.KEY_ON_DEMAND_PARAMS, None),
            )
        return False, None

    def add_data_frame(self, format_data: dt.Frame, file_name: str | None = None):
        self._pre_add_data(format_data=format_data, file_name=file_name)

        persist = self.explanation.explainer.persistence
        persist.store.make_dir(
            persist.get_explanation_dir_path(
                self.explanation.explanation_type(), self.mime
            )
        )
        self._add_frame_to_store(
            path=persist.get_explanation_file_path(
                self.explanation.explanation_type(),
                self.mime,
                explanation_file=file_name,
            ),
            format_data=format_data,
            persistence=persist.store,
        )

    @classmethod
    def get_local_explanation(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        dataset_path: str,
        row: int,
        explanation_filter: list[FilterEntry],
        **extra_params,
    ) -> str:
        """Get ICE.

        Parameters
        ----------
        persistence :
          Persistence object initialized for explainer/MLI run.
        explanation_type : str
          Explanation type ~ explanation ID.
        dataset_path :
          Dataset path.
        row: int
          Local explanation to be provided for given row.
        explanation_filter : list[FilterEntry]
          Required filter entries:
            `feature`
            `class`

        Returns
        -------
        str :
          JSon representation of the local explanation.

        JSon ICE representation:

        .. code-block:: text
           :linenos:

           {
                prediction: float,
                data: [
                    {
                        bin: any,
                        ice: float,
                    }
                ]
           }

        """
        logger = extra_params.get("logger", None)
        # persistence and explanation type is checked by caller
        if row is None or not isinstance(row, int) or row < 0:
            raise ValueError(f"Row index must be 0 or positive integer, not '{row}'")
        if not explanation_filter:
            raise ValueError(
                f"ICE filter parameters '{cls.FILTER_FEATURE}' and "
                f"'{cls.FILTER_CLASS}' are required"
            )
        e_filter: dict = dict()
        for i in explanation_filter:
            e_filter[i.filter_by] = i.value
        filter_feature: str = e_filter.get(cls.FILTER_FEATURE, "")
        if not filter_feature:
            raise ValueError(f"ICE filter parameter '{cls.FILTER_FEATURE}' is required")
        filter_cls: str = e_filter.get(cls.FILTER_CLASS, "")
        if not filter_cls:
            raise ValueError(f"ICE filter parameter '{cls.FILTER_CLASS}' is required")

        idx_dict = IceJsonDatatableFormat.load_index_file(persistence, explanation_type)

        # get a datatable file for feature and class
        if filter_feature not in idx_dict[cls.KEY_FEATURES]:
            raise ValueError(
                f"ICE feature filter parameter '{cls.FILTER_FEATURE}' value "
                f"'{filter_feature}' is not available in the "
                f"representation"
            )
        # translate class and check it
        pd_filter_cls = filter_cls
        if (
            idx_dict.get("labels_2_pd_map", None)
            and idx_dict["labels_2_pd_map"].get(filter_cls, None) is not None
        ):
            filter_cls = idx_dict["labels_2_pd_map"][filter_cls]
        # else: no class mapping
        if filter_cls not in idx_dict[cls.KEY_FEATURES][filter_feature][cls.KEY_FILES]:
            raise ValueError(
                f"ICE class filter parameter '{cls.FILTER_CLASS}' value "
                f"'{filter_cls}' is not available in the "
                f"representation "
                f"{idx_dict[cls.KEY_FEATURES][filter_feature][cls.KEY_FILES]}"
            )
        key_files = cls.KEY_FILES
        if explanation_filter:
            for f in explanation_filter:
                if f.filter_by == IceJsonDatatableFormat.FILTER_NUMCAT:
                    numcat_aspect = f.value
                    if numcat_aspect != idx_dict[cls.KEY_FEATURES][filter_feature][
                        cls.KEY_FEATURE_TYPE
                    ][0] and idx_dict[cls.KEY_FEATURES][filter_feature].get(
                        cls.KEY_FILES_NUMCAT_ASPECT
                    ):
                        key_files = cls.KEY_FILES_NUMCAT_ASPECT
                    break

        df_path: str = ""
        try:
            df_file: str = idx_dict[cls.KEY_FEATURES][filter_feature][key_files][
                filter_cls
            ]
            df_path = persistence.get_explanation_file_path(
                explanation_type=explanation_type,
                explanation_format=cls.mime,
                explanation_file=df_file,
            )
            df_ice = dt.fread(df_path)
            row_ice = df_ice[row, :]

            # original dataset prediction
            y_hat_path = idx_dict.get(cls.KEY_Y_FILE, None)
            if y_hat_path:
                y_hat_frame = dt.fread(
                    persistence.get_explanation_file_path(
                        explanation_type=explanation_type,
                        explanation_format=cls.mime,
                        explanation_file=y_hat_path,
                    )
                )
                if len(idx_dict[cls.KEY_FEATURES][filter_feature][key_files]) <= 2:
                    prediction = y_hat_frame[row, 0]
                else:
                    # predictions frame uses classes, not pd_*
                    prediction = y_hat_frame[row, pd_filter_cls]
            else:
                raise RuntimeError(
                    "ICE: Unable to load original dataset predictions - no path to "
                    "cached predictions frame in the index file"
                )
        except Exception as ex:
            raise RuntimeError(
                f"Unable to load ICE for row {filter_feature}/{filter_cls}/{row} "
                f"from {df_path}: {ex}\n{traceback.format_exc()}"
            )

        # feature value
        feature_value = None
        try:
            if dataset_path:
                df = dt.fread(dataset_path)
                # IMPROVE: feature / frame columns sanitization
                feature_value = df[row, filter_feature]
        except Exception as ex:
            if logger:
                logger.warning(
                    f"ICE representation not able to pull feature {filter_feature} "
                    f"value: {ex}:\n{traceback.format_exc()}"
                )

        return cls.mli_ice_explanation_to_json(
            ice_df=row_ice.to_pandas(),
            filter_feature=filter_feature,
            prediction=prediction,
            feature_value=feature_value,
            logger=logger,
        )

    @classmethod
    def mli_ice_explanation_to_json(
        cls,
        ice_df: pd.DataFrame,
        filter_feature: str,
        prediction,
        feature_value,
        logger=None,
    ) -> str:
        result_dict = dict()
        result_dict[cls.KEY_PREDICTION] = prediction
        result_dict[cls.KEY_FEATURE_VALUE] = feature_value
        result_dict[cls.KEY_DATA] = list()

        ice_f = dict()
        ice_f[cls.KEY_COL_NAME] = filter_feature
        ice_f[cls.KEY_DATA] = list()

        try:
            bins = ice_df.columns.values.tolist()
            ice_f[cls.KEY_DATA].append(bins)
            values = ice_df.iloc[0].values.tolist()
            for i, value in enumerate(values):
                result_dict[cls.KEY_DATA].append(
                    {cls.KEY_BIN: bins[i], cls.KEY_ICE: value}
                )
            if logger:
                logger.info(f"ICE:\n{result_dict}")
            return json.dumps(result_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to convert ICE for UI: {e}")

    @classmethod
    def merge_format(
        cls,
        from_path: str,
        to_path: str,
        overwrite: bool = True,
        discriminant: str = "",
        is_numcat_merge: bool = False,
        persistence: persistences.Persistence | None = None,
    ):
        """Merge ``from`` representation files to ``to`` representation files.

        Parameters
        ----------
        from_path : str
          Directory with the source representation to merge.
        to_path : str
          Directory with the target representation where should be new explanations
          merged.
        overwrite : bool
          Overwrite explanations if they already exist in the target representation.
          Use ``False`` to keep existing target explanations in case of a clash.
        discriminant: str
          Delimiter to make data file names unique (if needed).
        is_numcat_merge : bool
          ``True`` if this is num/cat update, ``False`` otherwise.
        persistence : persistences.Persistence | None
          Persistence store to save and load representations.

        """
        _persistence = persistence or persistences.FilesystemPersistence()

        if not persistence.is_dir(from_path):
            raise ValueError(
                f"Source directory to merge {cls.__name__} "
                f"formats does not exist: {from_path}"
            )
        if not persistence.is_dir(to_path):
            raise ValueError(
                f"Target directory to merge {cls.__name__} "
                f"formats does not exist: {to_path}"
            )

        idx_ext = MimeType.ext_for_mime(cls.mime)
        data_ext = MimeType.ext_for_mime(MimeType.MIME_DATATABLE)
        idx_file = f"{cls.FILE_PREFIX_EXPLANATION_IDX}{idx_ext}"
        from_idx_path = persistences.Persistence.make_key(from_path, idx_file)
        to_idx_path = persistences.Persistence.make_key(to_path, idx_file)

        if not persistence.is_file(from_idx_path):
            raise ValueError(
                f"Source format index file to merge {cls.__name__} "
                f"format does not exist: {from_idx_path}"
            )
        if not persistence.is_file(to_idx_path):
            raise ValueError(
                f"Target format index file to merge {cls.__name__} "
                f"format does not exist: {to_idx_path}"
            )

        files_key = cls.KEY_FILES_NUMCAT_ASPECT if is_numcat_merge else cls.KEY_FILES

        from_idx_dict = _persistence.load_json(from_idx_path)
        to_idx_dict = _persistence.load_json(to_idx_path)

        # on-demand merge resolution:
        # - if target is on-demand ICE, then on-demand, features are not merged,
        #   BUT on-demand BINS must be merged from new
        # - if source is on-demand ICE (target not), then target is changed
        #   to on-demand and new parameters are used
        if to_idx_dict.get(IceJsonDatatableFormat.KEY_ON_DEMAND, False):
            # merge bins from new to to
            if (
                from_idx_dict.get(IceJsonDatatableFormat.KEY_ON_DEMAND, False)
                and from_idx_dict.get(IceJsonDatatableFormat.KEY_ON_DEMAND_PARAMS, {})
                and from_idx_dict[IceJsonDatatableFormat.KEY_ON_DEMAND_PARAMS].get(
                    IceJsonDatatableFormat.KEY_BINS, {}
                )
            ):
                if not to_idx_dict.get(IceJsonDatatableFormat.KEY_ON_DEMAND_PARAMS, {}):
                    to_idx_dict[IceJsonDatatableFormat.KEY_ON_DEMAND_PARAMS] = {}

                bins_key = (
                    IceJsonDatatableFormat.KEY_BINS_NUMCAT_ASPECT
                    if is_numcat_merge
                    else IceJsonDatatableFormat.KEY_BINS
                )

                if not to_idx_dict[IceJsonDatatableFormat.KEY_ON_DEMAND_PARAMS].get(
                    bins_key, {}
                ):
                    to_idx_dict[IceJsonDatatableFormat.KEY_ON_DEMAND_PARAMS][
                        bins_key
                    ] = {}
                for bin_ in from_idx_dict[IceJsonDatatableFormat.KEY_ON_DEMAND_PARAMS][
                    IceJsonDatatableFormat.KEY_BINS
                ]:
                    if (
                        bin_
                        in to_idx_dict[IceJsonDatatableFormat.KEY_ON_DEMAND_PARAMS][
                            bins_key
                        ]
                        and not overwrite
                    ):
                        continue
                    to_idx_dict[IceJsonDatatableFormat.KEY_ON_DEMAND_PARAMS][bins_key][
                        bin_
                    ] = from_idx_dict[IceJsonDatatableFormat.KEY_ON_DEMAND_PARAMS][
                        IceJsonDatatableFormat.KEY_BINS
                    ][bin_]
            # save updated index file
            _persistence.save_json(data=to_idx_dict, key=to_idx_path)
            return
        if from_idx_dict.get(IceJsonDatatableFormat.KEY_ON_DEMAND, False):
            # switch target to on-demand (no need to merge features)
            to_idx_dict[IceJsonDatatableFormat.KEY_ON_DEMAND] = True
            to_idx_dict[IceJsonDatatableFormat.KEY_ON_DEMAND_PARAMS] = from_idx_dict[
                IceJsonDatatableFormat.KEY_ON_DEMAND_PARAMS
            ]
            # save updated index file
            _persistence.save_json(data=to_idx_dict, key=to_idx_path)
            return

        # cached: add/overwrite new feature entries from from to to (purge > add)
        for feature_name in from_idx_dict[cls.KEY_FEATURES]:
            # purge
            if feature_name in to_idx_dict[cls.KEY_FEATURES]:
                if overwrite:
                    # purge dict + files of feature to be overwritten (added later)
                    if is_numcat_merge:
                        if (
                            cls.KEY_FILES_NUMCAT_ASPECT
                            in to_idx_dict[cls.KEY_FEATURES][feature_name]
                        ):
                            files = to_idx_dict[cls.KEY_FEATURES][feature_name][
                                cls.KEY_FILES_NUMCAT_ASPECT
                            ]
                            for clazz in files:
                                _persistence.delete_file(
                                    persistences.Persistence.make_key(
                                        to_path, files[clazz]
                                    )
                                )
                            del to_idx_dict[cls.KEY_FEATURES][feature_name][
                                cls.KEY_FILES_NUMCAT_ASPECT
                            ]
                        # feature type to stay as is
                    else:
                        files = to_idx_dict[cls.KEY_FEATURES][feature_name][
                            cls.KEY_FILES
                        ]
                        for clazz in files:
                            _persistence.delete_file(
                                persistences.Persistence.make_key(to_path, files[clazz])
                            )
                        del to_idx_dict[cls.KEY_FEATURES][feature_name][cls.KEY_FILES]
                        # reset feature type
                        to_idx_dict[cls.KEY_FEATURES][feature_name][
                            cls.KEY_FEATURE_TYPE
                        ] = []
                else:
                    continue

            # append new/overwrite feature
            if feature_name in to_idx_dict[cls.KEY_FEATURES]:
                if not to_idx_dict[cls.KEY_FEATURES][feature_name][
                    cls.KEY_FEATURE_TYPE
                ]:
                    to_idx_dict[cls.KEY_FEATURES][feature_name][
                        cls.KEY_FEATURE_TYPE
                    ] = from_idx_dict[cls.KEY_FEATURES][feature_name][
                        cls.KEY_FEATURE_TYPE
                    ].copy()
                # feature/aspect is always loaded from ``files``
                to_idx_dict[cls.KEY_FEATURES][feature_name][files_key] = from_idx_dict[
                    cls.KEY_FEATURES
                ][feature_name][cls.KEY_FILES].copy()
            else:
                to_idx_dict[cls.KEY_FEATURES][feature_name] = from_idx_dict[
                    cls.KEY_FEATURES
                ][feature_name].copy()
            # copy data files + fix index file name
            files = to_idx_dict[cls.KEY_FEATURES][feature_name][files_key]
            for clazz in files:
                src_data_path = persistences.Persistence.make_key(
                    from_path, files[clazz]
                )
                dst_data_file = files[clazz].replace(
                    f".{data_ext}", f"_{discriminant}.{data_ext}"
                )
                dst_data_path = persistences.Persistence.make_key(
                    to_path, dst_data_file
                )
                _persistence.copy_file(from_key=src_data_path, to_key=dst_data_path)
                files[clazz] = dst_data_file

        for i, feature_name in enumerate(to_idx_dict[cls.KEY_FEATURES]):
            to_idx_dict[cls.KEY_FEATURES][feature_name][cls.KEY_ITEM_ORDER] = i

        # save updated index file
        _persistence.save_json(data=to_idx_dict, key=to_idx_path)


class LocalOnDemandTextFormat(TextCustomExplanationFormat):
    """Local (single row) on-demand representation."""

    mime = MimeType.MIME_TEXT

    @classmethod
    def get_local_explanation(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        row: int,
        explanation_filter: list[FilterEntry],
        **extra_params,
    ) -> str:
        """Load index file and check parameters.

        Returns
        -------
        str:
          Local explanation as string - can be any (on)structured format.

        """
        file_path = persistence.get_explanation_file_path(explanation_type, cls.mime)
        return persistence.store.load(file_path)

    def __init__(
        self,
        explanation,
        format_data: str,
        persistence: persistences.Persistence | None = None,
    ):
        TextCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=format_data,
            format_file=None,
            persistence=persistence,
        )

    @staticmethod
    def validate_data(dt_data: dt.Frame):
        return dt_data


class LocalOnDemandHtmlFormat(TextCustomExplanationFormat):
    """Local (single row) on-demand representation."""

    mime = MimeType.MIME_HTML

    @classmethod
    def get_local_explanation(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        row: int,
        explanation_filter: list[FilterEntry],
        **extra_params,
    ) -> str:
        """Load index file and check parameters.

        Returns
        -------
        str:
          Local explanation as string - can be any (on)structured format.

        """
        file_path = persistence.get_explanation_file_path(explanation_type, cls.mime)
        return persistence.store.load(file_path)

    def __init__(
        self,
        explanation,
        format_data: str,
        persistence: persistences.Persistence | None = None,
    ):
        TextCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=format_data,
            format_file=None,
            persistence=persistence,
        )

    def set_on_demand(self, is_on_demand: bool, mime: str = ""):
        TextCustomExplanationFormat.set_on_demand(
            self, is_on_demand=is_on_demand, mime=MimeType.MIME_HTML
        )

    @staticmethod
    def validate_data(dt_data: dt.Frame):
        return dt_data


class HtmlFormat(TextCustomExplanationFormat):
    """HTML representation.

    Example local (single row) on-demand NLP HTML explanation:

    .. code-block:: text

        <feature-text min="-10.0" max="5.0">
          Sentence with <word value="-0.9485">dummy word</word>.
        </feature-text>


    """

    mime = MimeType.MIME_HTML

    EL_FEATURE_TEXT = "feature-text"
    EL_WORD = "word"
    ATT_MAX = "max"
    ATT_MIN = "min"
    ATT_VALUE = "value"

    MINIMAL_HTML = "<!DOCTYPE html>\n<html lang='en'><head></head><body></body></html>"

    def __init__(
        self,
        explanation,
        format_data: str,
        format_file: str | None = None,
        extra_format_files: list | None = None,
        persistence: persistences.Persistence | None = None,
    ):
        TextCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=format_data,
            format_file=format_file,
            extra_format_files=extra_format_files,
            persistence=persistence,
        )

    @classmethod
    def is_on_demand(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
    ) -> tuple[bool, dict | None]:
        idx_dict = GlobalFeatImpJSonFormat.load_index_file(
            persistence, explanation_type, mime=HtmlFormat.mime
        )

        if idx_dict and idx_dict.get(HtmlFormat.KEY_ON_DEMAND, False):
            return True, idx_dict.get(HtmlFormat.KEY_ON_DEMAND_PARAMS, None)
        return False, None

    @staticmethod
    def validate_data(dt_data: dt.Frame):
        return dt_data


class Global3dDataJSonFormat(TextCustomExplanationFormat):
    """Representation of global 3D data (3D bar charts, heatmaps, ...) as JSon.

    JSon representation index file example:

    .. code-block:: text

        {
            "features": {
                "PAY_0 and AGE": {
                    "order": 0,
                    "feature_names: ["PAY_0", "AGE"],
                    "files": {
                        "red_class": "data3d_feature_0_class_0.json"
                        "green_class": "data3d_feature_0_class_1.json"
                        "blue_class": "data3d_feature_0_class_2.json"
                    }
                },
                ...
            },
            "metrics": [{"R2": 0.96}, {"RMSE": 0.03}],
            "documentation": "PD for 2 features..."
        }

    JSon representation data file example:

    .. code-block:: text

        "data_dictionary": {
            {
                "feature_1_bin_1": {
                    "feature_2_bin_1": 1,
                    "feature_2_bin_2": 2,
                    "feature_2_bin_3": 3
                },
                "feature_1_bin_2": {
                    "feature_2_bin_1": 1,
                    "feature_2_bin_2": 2,
                    "feature_2_bin_3": 3
                },
                "feature_1_bin_3": {
                    "feature_2_bin_1": 1,
                    "feature_2_bin_2": 2,
                    "feature_2_bin_3": 3
                }
            }
        }

    Where:

    * ``data_dictionary`` is dictionary which might be used to easily construct
      data frame where column and row labels represent bin values
    * ``data`` key is not intentionally used to be used in the future for
      Grammar of MLI/Vega friendly representations (like in case of other formats).

    """

    mime = MimeType.MIME_JSON

    KEY_FEATURE_NAMES = "feature_names"

    def __init__(
        self,
        explanation,
        json_data: str = None,
        json_file: str = None,
        persistence: persistences.Persistence | None = None,
    ):
        TextCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=Global3dDataJSonFormat.validate_data(json_data),
            format_file=json_file,
            persistence=persistence,
        )

    @staticmethod
    def validate_data(json_data: str) -> str:
        return json_data

    @staticmethod
    def serialize_index_file(
        features: list[str],
        features_names: list[list[str]],
        classes: list[str],
        default_class: str = "",
        metrics: list | None = None,
        keywords: list | None = None,
        doc: str = "",
        data_file_prefix: str = "data3d",
        data_file_suffix: str = "json",
        y_file: str | None = None,
    ) -> tuple[dict, str]:
        """JSon index file serialization to string.

        Parameters
        ----------
        features : list
          Feature tuples.
        features_names : list
          Per-feature tuple feature names.
        classes : list
          Classes.
        default_class : str
          Class to be shown as default (the first one) e.g. the class of interest in
          case of binomial experiment interpretation.
        metrics : list
          Optional list of metrics e.g. ``[{"RMSE": 0.02}, {"SD": 3.1}]``
        keywords : list[str]
          Optional list of keywords indicating representation features, properties
          and aspects.
        doc : str
          Chart documentation.
        data_file_prefix : str
          Prefix for data file names.
        data_file_suffix : str
          Suffix for data file names.
        y_file : str
          Predictions file.

        Returns
        -------
        Tuple[dict, str] :
          Dictionary with mapping of features and classes to file names AND JSon
          serialization (as string).

        """
        if not features:
            raise ValueError(
                "At least one feature must provided to serialize 3D data JSon index "
                "file"
            )
        if not classes:
            raise ValueError(
                "At least one class must provided to serialize 3D data JSon index file"
            )
        if len(features) != len(features_names):
            raise ValueError(
                f"Features and features name lists lengths must match "
                f"({len(features)} != {len(features_names)}) - features names "
                f"list must provide feature names for ever features list item"
            )

        dj = Global3dDataJSonFormat
        index_dict: dict = dict()
        features_dict: dict = dict()
        for i_f, feature in enumerate(features):
            features_dict[feature] = fd = dict()
            fd[dj.KEY_ITEM_ORDER] = i_f
            fd[dj.KEY_FEATURE_NAMES] = features_names[i_f]
            fd[dj.KEY_FILES] = dict()
            for i_c, cls in enumerate(classes):
                features_dict[feature][dj.KEY_FILES][cls] = (
                    f"{data_file_prefix}_feature_{i_f}_class_{i_c}.{data_file_suffix}"
                )
        index_dict[dj.KEY_FEATURES] = features_dict
        TextCustomExplanationFormat.set_index_commons(
            index_dict=index_dict,
            classes=classes,
            default_class=default_class,
            metrics=metrics,
            keywords=keywords,
            doc=doc,
        )
        if y_file:
            index_dict[dj.KEY_Y_FILE] = y_file

        return index_dict, json.dumps(index_dict, indent=4)

    @classmethod
    def load_index_file(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        mime: str = MimeType.MIME_JSON,
    ) -> dict:
        #  index file
        idx_path = persistence.get_explanation_file_path(explanation_type, mime)
        return persistence.store.load_json(idx_path)


class Global3dDataJSonCsvFormat(TextCustomExplanationFormat):
    """Representation of global 3D data (3D bar charts, heatmaps, ...) as CSV files
    with JSon index.

    JSon representation index file example:

    .. code-block:: text

        {
            "features": {
                "PAY_0 and AGE": {
                    "order": 0,
                    "feature_names: ["PAY_0", "AGE"],
                    "files": {
                        "red_class": "data3d_feature_0_class_0.csv"
                        "green_class": "data3d_feature_0_class_1.csv"
                        "blue_class": "data3d_feature_0_class_2.csv"
                    }
                },
                ...
            },
            "metrics": [{"R2": 0.96}, {"RMSE": 0.03}],
            "documentation": "PD for 2 features..."
        }

    CSV representation data file example:

    .. code-block:: text

        ,feature_1_bin_1,feature_1_bin_2,feature_1_bin_3
        feature_2_bin_1,1,1,1
        feature_2_bin_2,2,2,2
        feature_2_bin_3,3,3,3

    """

    mime = MimeType.MIME_JSON_CSV

    KEY_FEATURE_NAMES = "feature_names"

    def __init__(
        self,
        explanation,
        json_data: str = None,
        json_file: str = None,
        persistence: persistences.Persistence | None = None,
    ):
        TextCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=Global3dDataJSonFormat.validate_data(json_data),
            format_file=json_file,
            persistence=persistence,
        )

    @staticmethod
    def validate_data(json_data: str) -> str:
        return json_data

    @staticmethod
    def serialize_index_file(
        features: list[str],
        features_names: list[list[str]],
        classes: list[str],
        default_class: str = "",
        metrics: list | None = None,
        keywords: list | None = None,
        doc: str = "",
        data_file_prefix: str = "data3d",
        data_file_suffix: str = "csv",
        y_file: str | None = None,
    ) -> tuple[dict, str]:
        """JSon index file serialization to string.

        Parameters
        ----------
        features : list
          Feature tuples.
        features_names : list
          Per-feature tuple feature names.
        classes : list
          Classes.
        default_class : str
          Class to be shown as default (the first one) e.g. the class of interest in
          case of binomial experiment interpretation.
        metrics : list
          Optional list of metrics e.g. ``[{"RMSE": 0.02}, {"SD": 3.1}]``
        keywords : list[str]
          Optional list of keywords indicating representation features, properties
          and aspects.
        doc : str
          Chart documentation.
        data_file_prefix : str
          Prefix for data file names.
        data_file_suffix : str
          Suffix for data file names.
        y_file : str
          Predictions file.

        Returns
        -------
        Tuple[dict, str] :
          Dictionary with mapping of features and classes to file names AND JSon
          serialization (as string).

        """
        if not features:
            raise ValueError(
                "At least one feature must provided to serialize 3D data JSon index "
                "file"
            )
        if not classes:
            raise ValueError(
                "At least one class must provided to serialize 3D data JSon index file"
            )
        if len(features) != len(features_names):
            raise ValueError(
                f"Features and features name lists lengths must match "
                f"({len(features)} != {len(features_names)}) - features names "
                f"list must provide feature names for ever features list item"
            )

        dj = Global3dDataJSonFormat
        index_dict: dict = dict()
        features_dict: dict = dict()
        for i_f, feature in enumerate(features):
            features_dict[feature] = fd = dict()
            fd[dj.KEY_ITEM_ORDER] = i_f
            fd[dj.KEY_FEATURE_NAMES] = features_names[i_f]
            fd[dj.KEY_FILES] = dict()
            for i_c, cls in enumerate(classes):
                features_dict[feature][dj.KEY_FILES][cls] = (
                    f"{data_file_prefix}_feature_{i_f}_class_{i_c}.{data_file_suffix}"
                )
        index_dict[dj.KEY_FEATURES] = features_dict
        TextCustomExplanationFormat.set_index_commons(
            index_dict=index_dict,
            classes=classes,
            default_class=default_class,
            metrics=metrics,
            keywords=keywords,
            doc=doc,
        )
        if y_file:
            index_dict[dj.KEY_Y_FILE] = y_file

        return index_dict, json.dumps(index_dict, indent=4)

    @classmethod
    def load_index_file(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        mime: str = MimeType.MIME_JSON,
    ) -> dict:
        #  index file
        idx_path = persistence.get_explanation_file_path(explanation_type, mime)
        return persistence.store.load_json(idx_path)


class IceDatatableFormat(DatatableCustomExplanationFormat):
    """Individual conditional explanation as datatable.

    Canonical representation (datatable frame, ltypes) for 1D ICE:

    .. code-block:: text

        | Required column    | Type  | Description            |
        |--------------------|-------|------------------------|
        | feature_name       | str   | Feature name.          |
        | feature_type       | str   | Feature type.          |
        | instance_id        | int   | Instance.              |
        | bin_value          | str   | Bin value.             |
        | prediction         | real  | Prediction.            |

    Hints:

    * ``bin_value`` is converted to string (can be converted back using feature_type)

    ... other optional columns are allowed

    """

    mime = MimeType.MIME_DATATABLE

    COL_F_NAME = "feature_name"
    COL_F_LTYPE = "feature_type"
    COL_INSTANCE = "instance"
    COL_BIN_VALUE = "bin_value"
    COL_PREDICTION = "prediction"

    def __init__(
        self,
        explanation,
        frame: dt.Frame,
        frame_file: str = None,
        persistence: persistences.Persistence | None = None,
    ):
        DatatableCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            frame=IceDatatableFormat.validate_data(frame),
            frame_file=frame_file,
            persistence=persistence,
        )

    @staticmethod
    def validate_data(dt_data: dt.Frame):
        return dt_data


class IceCsvFormat(CsvFormatCustomExplanationFormat):
    mime = MimeType.MIME_CSV

    def __init__(
        self,
        explanation,
        frame: dt.Frame,
        frame_file: str = None,
        persistence: persistences.Persistence | None = None,
    ):
        CsvFormatCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            frame=IceCsvFormat.validate_data(frame),
            frame_file=frame_file,
            persistence=persistence,
        )

    @staticmethod
    def validate_data(dt_data: dt.Frame):
        return dt_data


class PartialDependenceDatatableFormat(DatatableCustomExplanationFormat):
    """Representation of partial dependence (PD) explanation as datatable.

    Canonical representation (datatable frame, ltypes) for 1D PD:

    .. code-block:: text

        | Required column    | Type  | Description            |
        |--------------------|-------|------------------------|
        | feature_name       | str   | Feature name.          |
        | feature_type       | str   | Feature type.          |
        | bin_value          | str   | Bin value              |
        | mean               | real  | Mean.                  |
        | sd                 | real  | Standard deviation.    |
        | sem                | real  | Standard mean error.   |
        | is_oor             | bool  | Is out of range value? |

    Hints:

    * ``bin_value`` is converted to string (can be converted back using feature_type).

    ... other optional columns are allowed

    """

    mime = MimeType.MIME_DATATABLE

    COL_F_NAME = "feature_name"
    COL_F_LTYPE = "feature_type"
    COL_BIN_VALUE = "bin_value"
    COL_MEAN = "mean"
    COL_SD = "sd"
    COL_SEM = "sem"
    COL_IS_OOR = "is_oor"

    def __init__(
        self,
        explanation,
        frame: dt.Frame,
        frame_file: str = None,
        persistence: persistences.Persistence | None = None,
    ):
        DatatableCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            frame=PartialDependenceDatatableFormat.validate_data(frame),
            frame_file=frame_file,
            persistence=persistence,
        )

    @staticmethod
    def validate_data(dt_data: dt.Frame):
        return dt_data


class PartialDependenceCsvFormat(CsvFormatCustomExplanationFormat):
    def __init__(
        self,
        explanation,
        frame: dt.Frame,
        frame_file: str = None,
        persistence: persistences.Persistence | None = None,
    ):
        CsvFormatCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            frame=PartialDependenceCsvFormat.validate_data(frame),
            frame_file=frame_file,
            persistence=persistence,
        )

    @staticmethod
    def validate_data(dt_data: dt.Frame):
        return dt_data


class PartialDependenceJSonFormat(TextCustomExplanationFormat, GrammarOfMliFormat):
    """Representation of partial dependence (PD) explanation as JSon.

    JSon representation index file example:

    .. code-block:: text

        {
            "features": {
                "PAY_0": {
                    "order": 0,
                    "feature_type": ["categorical"],
                    "files": {
                        "red_class": "pd_feature_0_class_0.json"
                        "green_class": "pd_feature_0_class_1.json"
                        "blue_class": "pd_feature_0_class_2.json"
                    }
                },
                ...
            },
            "metrics": [{"R2": 0.96}, {"RMSE": 0.03}]
        }

    JSon representation data file example:

    .. code-block:: text

        {
            "data": [{
                "bin": -2,
                "pd": 0.3710160553455353,
                "sd": 0.029299162328243256,,
                "out_of_range": false
            }, {
                "bin": -1,
                "pd": 0.3710160553455353,
                "sd": 0.029299162328243256,,
                "out_of_range": false
            },
            ...
        }

    """

    mime = MimeType.MIME_JSON

    KEY_BIN = "bin"
    KEY_PD = "pd"
    KEY_SD = "sd"
    KEY_OOR = "oor"
    KEY_FREQUENCY = "frequency"
    KEY_X = "x"

    def __init__(
        self,
        explanation,
        json_data: str = None,
        json_file: str = None,
        persistence: persistences.Persistence | None = None,
    ):
        TextCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=PartialDependenceJSonFormat.validate_data(json_data),
            format_file=json_file,
            persistence=persistence,
        )

    @staticmethod
    def validate_data(json_data: str) -> str:
        return json_data

    @staticmethod
    def serialize_index_file(
        features: list[str],
        classes: list[str],
        default_class: str = "",
        features_meta: dict | None = None,
        metrics: list | None = None,
        keywords: list | None = None,
        doc: str = "",
        data_file_prefix: str = "pd",
        data_file_suffix: str = "json",
        y_file: str | None = None,
    ) -> tuple[dict, str]:
        """JSon index file serialization to string.

        Parameters
        ----------
        features : list
          Features.
        classes : list
          Classes.
        default_class : str
          Class to be shown as default (the first one) e.g. the class of interest in
          case of binomial experiment interpretation.
        features_meta : dict
            Features metadata allowing to indicate that given feature is
            categorical (use ``categorical`` key and list of feature names),
            (use ``date`` key and list of feature names, to specify format use
            ``date-format`` and list of Python date formats) or ``numerical``
            (default).
        metrics : list
          Optional list of PD related metrics e.g. ``[{"RMSE": 0.02}, {"SD": 3.1}]``
        keywords : list[str]
          Optional list of keywords indicating representation features, properties
          and aspects.
        doc : str
          Chart documentation.
        data_file_prefix : str
          Prefix for data file names.
        data_file_suffix : str
          Suffix for data file names.
        y_file : str
          Predictions file.

        Returns
        -------
        Tuple[dict, str] :
          Dictionary with mapping of features and classes to file names AND JSon
          serialization (as string).

        """
        if not features:
            raise ValueError(
                "At least one feature must provided to serialize PD/ICE JSon index file"
            )
        if not classes:
            raise ValueError(
                "At least one class must provided to serialize PD/ICE JSon index file"
            )

        pdj = PartialDependenceJSonFormat
        index_dict: dict = dict()
        features_dict: dict = dict()
        for i_f, feature in enumerate(features):
            features_dict[feature] = fd = dict()
            fd[pdj.KEY_ITEM_ORDER] = i_f
            fd[pdj.KEY_FEATURE_TYPE] = pdj._serializable_feature_type(
                feature, features_meta
            )
            fd[pdj.KEY_FILES] = dict()
            for i_c, cls in enumerate(classes):
                features_dict[feature][pdj.KEY_FILES][cls] = (
                    f"{data_file_prefix}_feature_{i_f}_class_{i_c}.{data_file_suffix}"
                )
        index_dict[pdj.KEY_FEATURES] = features_dict
        TextCustomExplanationFormat.set_index_commons(
            index_dict=index_dict,
            classes=classes,
            default_class=default_class,
            metrics=metrics,
            keywords=keywords,
            doc=doc,
        )
        if y_file:
            index_dict[pdj.KEY_Y_FILE] = y_file

        return index_dict, json.dumps(index_dict, indent=4)

    @classmethod
    def load_index_file(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        mime: str = MimeType.MIME_JSON,
    ) -> dict:
        #  index file
        idx_path = persistence.get_explanation_file_path(explanation_type, mime)
        return persistence.store.load_json(idx_path)

    @staticmethod
    def _serializable_feature_type(feature: str, feature_meta: dict) -> list[str]:
        result = list()
        pdj = PartialDependenceJSonFormat

        if feature_meta:
            # use date, date time, and time as categoricals for now since handling of
            # dates in PD is not done in any specific way and may lead to odd results
            valid_cat_types = sum(
                [
                    feature_meta.get(pdj.KEY_CATEGORICAL, []),
                    feature_meta.get(pdj.KEY_DATE, []),
                    feature_meta.get(pdj.KEY_DATE_TIME, []),
                    feature_meta.get(pdj.KEY_TIME, []),
                ],
                [],
            )
            if feature in valid_cat_types:
                result.append(pdj.FEATURE_TYPE_CAT)
            if feature in feature_meta.get(pdj.KEY_NUMERIC, []):
                result.append(pdj.FEATURE_TYPE_NUM)

        if not result:
            result.append(pdj.FEATURE_TYPE_NUM)
        return result

    @classmethod
    def get_bins(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        feature: str,
    ) -> list:
        """Get bins for given feature.

        Parameters
        ----------
        persistence:
          Persistence object initialized for explainer/MLI run.
        explanation_type: str
          Explanation type ~ explanation ID.
        feature: str
          Feature for which to get bins.

        Returns
        -------
        list:
          Bins.

        """
        idx_dict = IceJsonDatatableFormat.load_index_file(
            persistence, explanation_type, cls.mime
        )
        if feature not in idx_dict[cls.KEY_FEATURES]:
            raise ValueError(
                f"ICE filter parameter '{IceJsonDatatableFormat.FILTER_FEATURE}' "
                f"parameter '{feature}' is not available in the representation"
            )

        pd_path: str = ""
        try:
            clazz = next(iter(idx_dict[cls.KEY_FEATURES][feature][cls.KEY_FILES]))
            pd_file: str = idx_dict[cls.KEY_FEATURES][feature][cls.KEY_FILES][clazz]
            pd_path = persistence.get_explanation_file_path(
                explanation_type=explanation_type,
                explanation_format=MimeType.MIME_JSON,
                explanation_file=pd_file,
            )
            # get bins
            pd_dict = persistence.store.load_json(pd_path)
            bins: list = list()
            if pd_dict:
                for entry in pd_dict[cls.KEY_DATA]:
                    bins.append(entry[cls.KEY_BIN])
            return bins
        except Exception as ex:
            raise RuntimeError(
                f"Unable to load PD bins for row {feature} from {pd_path}: {ex}"
                f"\n{traceback.format_exc()}"
            )

    @classmethod
    def get_numcat_aspects(cls, feature, idx: dict) -> list[str]:
        """Get available num/cat aspects for given feature:

        * [] ... invalid feature
        * ``["numeric"]`` ... numeric PD only
        * ``["categorical"]`` ... categorical PD only
        * ``["numeric", "categorical"]`` ... numeric and categorical PD

        Parameters
        ----------
        feature: str
          Feature name for which to determine available aspects.
        idx: dict
          PD JSon index file (``explanation.json``).

        Returns
        -------
        list[str]:
          Available num/cat aspects.

        """
        aspects: list = []

        if (
            feature
            and idx
            and idx.get(cls.KEY_FEATURES)
            and idx[cls.KEY_FEATURES].get(feature)
        ):
            # determine default aspect
            aspects.append(
                idx[cls.KEY_FEATURES][feature].get(
                    cls.KEY_FEATURE_TYPE, [cls.FEATURE_TYPE_CAT]
                )[0]
            )
            # check whatever alternative aspect exists
            if cls.KEY_FILES_NUMCAT_ASPECT in idx[cls.KEY_FEATURES][feature]:
                if cls.FEATURE_TYPE_CAT in aspects:
                    aspects.append(cls.FEATURE_TYPE_NUM)
                else:
                    aspects.append(cls.FEATURE_TYPE_CAT)

        return aspects

    @classmethod
    def get_numcat_missing_aspect(cls, feature: str, idx: dict):
        """Return (missing) aspect to be calculated.

        Parameters
        ----------
        feature: str
          Feature name for which to determine available aspects.
        idx: dict
          PD JSon index file (``explanation.json``).

        Returns
        -------
        str:
          Aspect to calculate or ``""`` (no aspect is missing).

        """
        numcat_aspects: list = PartialDependenceJSonFormat.get_numcat_aspects(
            feature=feature, idx=idx
        )
        if numcat_aspects:
            if cls.FEATURE_TYPE_NUM not in numcat_aspects:
                return cls.FEATURE_TYPE_NUM
            if cls.FEATURE_TYPE_CAT not in numcat_aspects:
                return cls.FEATURE_TYPE_CAT
        else:
            raise errors.MliError(
                f"Invalid feature name '{feature}' when determining missing num/cat "
                f"aspects of {cls.__name__}"
            )

        return ""

    @classmethod
    def set_merge_status(
        cls,
        dir_path: str,
        mli_key: str,
        explainer_job_key: str,
        clear: bool = False,
        action: str = "update_explanation",
        action_type: str = "add_aspect",
        persistence: persistences.Persistence | None = None,
    ):
        """Add (``clear=False``) or remove running interpretation update.

        Parameters
        ----------
        dir_path: str
          Directory with index file where the status should be set.
        mli_key: str
          MLI key of the interpretation which will update another representation.
        explainer_job_key: str
          Explainer job key of the interpretation which will update another
          representation.
        clear: bool
          Add (``clear=False``) or remove (``clear=True``) indicator in
          representation's dict.
        action: str
          Running action identifier e.g. update explanation.
        action_type: str
          Action (sub)type identifier e.g. add feature, add numeric/categorical view.
        persistence : persistences.Persistence | None
          Persistence store to save and load explanation representations.

        """
        _persistence = persistence or persistences.FilesystemPersistence()

        if not _persistence.is_dir(dir_path):
            raise ValueError(
                f"Representations directory to set status of {cls.__name__} "
                f"format does not exist: {dir_path}"
            )

        idx_ext = MimeType.ext_for_mime(cls.mime)
        idx_file = f"{cls.FILE_PREFIX_EXPLANATION_IDX}{idx_ext}"
        idx_path = persistences.Persistence.make_key(dir_path, idx_file)

        if not persistence.is_file(idx_path):
            raise ValueError(
                f"Representation format index file to set status of {cls.__name__} "
                f"format does not exist: {idx_path}"
            )

        idx_dict = _persistence.load_json(idx_path)
        if clear:
            if cls.KEY_RUNNING_ACTION in idx_dict:
                del idx_dict[cls.KEY_RUNNING_ACTION]
        else:
            idx_dict[cls.KEY_RUNNING_ACTION] = {
                cls.KEY_ACTION: action,
                cls.KEY_ACTION_TYPE: action_type,
                cls.KEY_MLI_KEY: mli_key,
                cls.KEY_EXPLAINER_JOB_KEY: explainer_job_key,
            }

        # save updated index file
        _persistence.save_json(data=idx_dict, key=idx_path)

    @classmethod
    def merge_format(
        cls,
        from_path: str,
        to_path: str,
        overwrite: bool = True,
        discriminant: str = "",
        is_numcat_merge: bool = False,
        persistence: persistences.Persistence | None = None,
    ):
        """Merge ``from`` representation files to ``to`` representation files.

        Parameters
        ----------
        from_path: str
          Directory with the source representation to merge.
        to_path: str
          Directory with the target representation where should be new explanations
          merged.
        overwrite: bool
          Overwrite explanations if they already exist in the target representation.
          Use ``False`` to keep existing target explanations in case of a clash.
        discriminant: str
          Delimiter to make data file names unique (if needed).
        is_numcat_merge: bool
          ``True`` if this is num/cat update, ``False`` otherwise.
        persistence : persistences.Persistence | None
          Persistence store to save and load explanation representations.

        """
        _persistence = persistence or persistences.FilesystemPersistence()

        if not _persistence.is_dir(from_path):
            raise ValueError(
                f"Source directory to merge {cls.__name__} "
                f"formats does not exist: {from_path}"
            )
        if not _persistence.is_dir(to_path):
            raise ValueError(
                f"Target directory to merge {cls.__name__} "
                f"formats does not exist: {to_path}"
            )

        idx_ext = MimeType.ext_for_mime(cls.mime)
        data_ext = MimeType.ext_for_mime(cls.mime)
        idx_file = f"{cls.FILE_PREFIX_EXPLANATION_IDX}{idx_ext}"
        from_idx_path = persistences.Persistence.make_key(from_path, idx_file)
        to_idx_path = persistences.Persistence.make_key(to_path, idx_file)

        if not persistence.is_file(from_idx_path):
            raise ValueError(
                f"Source format index file to merge {cls.__name__} "
                f"formats does not exist: {from_idx_path}"
            )
        if not persistence.is_file(to_idx_path):
            raise ValueError(
                f"Target format index file to merge {cls.__name__} "
                f"formats does not exist: {to_idx_path}"
            )

        files_key = cls.KEY_FILES_NUMCAT_ASPECT if is_numcat_merge else cls.KEY_FILES

        from_idx_dict = _persistence.load_json(from_idx_path)
        to_idx_dict = _persistence.load_json(to_idx_path)

        # cached: add/overwrite new feature entries from from to to (purge > add)
        for feature_name in from_idx_dict[cls.KEY_FEATURES]:
            # purge
            if feature_name in to_idx_dict[cls.KEY_FEATURES]:
                if overwrite:
                    # purge dict + files of feature to be overwritten (added later)
                    if is_numcat_merge:
                        if (
                            cls.KEY_FILES_NUMCAT_ASPECT
                            in to_idx_dict[cls.KEY_FEATURES][feature_name]
                        ):
                            files = to_idx_dict[cls.KEY_FEATURES][feature_name][
                                cls.KEY_FILES_NUMCAT_ASPECT
                            ]
                            for clazz in files:
                                _persistence.delete_file(
                                    persistences.Persistence.make_key(
                                        to_path, files[clazz]
                                    )
                                )
                            del to_idx_dict[cls.KEY_FEATURES][feature_name][
                                cls.KEY_FILES_NUMCAT_ASPECT
                            ]
                        # feature type MUST stay as is
                    else:
                        files = to_idx_dict[cls.KEY_FEATURES][feature_name][
                            cls.KEY_FILES
                        ]
                        for clazz in files:
                            _persistence.delete_file(
                                persistences.Persistence.make_key(to_path, files[clazz])
                            )
                        del to_idx_dict[cls.KEY_FEATURES][feature_name][cls.KEY_FILES]
                        # reset feature type
                        to_idx_dict[cls.KEY_FEATURES][feature_name][
                            cls.KEY_FEATURE_TYPE
                        ] = []
                else:
                    continue

            # append new/overwrite feature
            if feature_name in to_idx_dict[cls.KEY_FEATURES]:
                if not to_idx_dict[cls.KEY_FEATURES][feature_name][
                    cls.KEY_FEATURE_TYPE
                ]:
                    to_idx_dict[cls.KEY_FEATURES][feature_name][
                        cls.KEY_FEATURE_TYPE
                    ] = from_idx_dict[cls.KEY_FEATURES][feature_name][
                        cls.KEY_FEATURE_TYPE
                    ].copy()
                # feature/aspect is always loaded from ``files``
                to_idx_dict[cls.KEY_FEATURES][feature_name][files_key] = from_idx_dict[
                    cls.KEY_FEATURES
                ][feature_name][cls.KEY_FILES].copy()
            else:
                to_idx_dict[cls.KEY_FEATURES][feature_name] = from_idx_dict[
                    cls.KEY_FEATURES
                ][feature_name].copy()
            # copy data files + fix index file name
            files = to_idx_dict[cls.KEY_FEATURES][feature_name][files_key]
            for clazz in files:
                src_data_path = persistences.Persistence.make_key(
                    from_path, files[clazz]
                )
                dst_data_file = files[clazz].replace(
                    f".{data_ext}", f"_{discriminant}.{data_ext}"
                )
                dst_data_path = persistences.Persistence.make_key(
                    to_path, dst_data_file
                )
                _persistence.copy_file(from_key=src_data_path, to_key=dst_data_path)
                files[clazz] = dst_data_file

        for i, feature_name in enumerate(to_idx_dict[cls.KEY_FEATURES]):
            to_idx_dict[cls.KEY_FEATURES][feature_name][cls.KEY_ITEM_ORDER] = i

        # save updated index file
        _persistence.save_json(data=to_idx_dict, key=to_idx_path)


class LocalFeatImpDatatableFormat(DatatableCustomExplanationFormat, GrammarOfMliFormat):
    """Local feature importance datatable representation.

    * feature importance for all classes

    Canonical representation (datatable frame):

    .. code-block:: text

        | Columns            | Rows                                |
        |--------------------|-------------------------------------|
        | feature names      | per-dataset row feature importance  |

    Example:

    .. code-block:: text

          |     activity   ...   max_speed
        --- + ------------ ...  -----------
        0 |    -0.0143614        -0.142553
        . |     ...               ...
        9 |     0.0156479        -0.231883

    """

    mime = MimeType.MIME_DATATABLE

    FILE_EXT = MimeType.EXT_DATATABLE

    def __init__(
        self,
        explanation,
        frame: dt.Frame,
        frame_file: str | None = None,
        persistence: persistences.Persistence | None = None,
    ):
        DatatableCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            frame=LocalFeatImpDatatableFormat.validate_data(frame),
            frame_file=frame_file,
            persistence=persistence,
        )

    @classmethod
    def get_local_explanation(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        row: int,
        explanation_filter: list[FilterEntry],
        **extra_params,
    ) -> str:
        """Get local feature importance explanation.

        Parameters
        ----------
        persistence:
          Persistence object initialized for explainer/MLI run.
        explanation_type: str
          Explanation type ~ explanation ID.
        row: int
          Local explanation to be provided for given row.
        explanation_filter: list[FilterEntry]
          Filter (unused in case of feature importance).

        Returns
        -------
        str:
          JSon representation of the local explanation.

        JSon representation:

        .. code-block:: text
           :linenos:

           {
                data: [
                    {
                        label: str,
                        value: num,
                        scope: str,
                    }+
                ]
           }

        Where:

        * ``label`` is feature name
        * ``value`` is feature importance
        * ``scope`` is ``local``

        """
        if row is None or not isinstance(row, int) or row < 0:
            raise ValueError(f"Row index must be 0 or positive integer, not '{row}'")

        df_path: str = ""
        try:
            df_path = persistence.get_explanation_file_path(
                explanation_type=explanation_type,
                explanation_format=cls.mime,
                explanation_file=(
                    f"{persistences.ExplainerPersistence.FILE_EXPLANATION}."
                    f"{LocalFeatImpDatatableFormat.FILE_EXT}"
                ),
            )
            df = dt.fread(df_path)
            row_dict = df[row, :].to_dict()
            data: list = []
            gfi = GlobalFeatImpJSonFormat
            for feature in row_dict:
                data.append(
                    {
                        gfi.KEY_LABEL: feature,
                        gfi.KEY_VALUE: row_dict[feature][0],
                        gfi.KEY_SCOPE: gfi.SCOPE_LOCAL,
                    }
                )

            return json.dumps({cls.KEY_DATA: data})

        except Exception as ex:
            raise RuntimeError(
                f"Unable to load local feature importance explanation "
                f"from {df_path}: {ex}\n{traceback.format_exc()}"
            )

    @staticmethod
    def validate_data(frame_data: dt.Frame) -> dt.Frame:
        return frame_data


class GlobalFeatImpDatatableFormat(DatatableCustomExplanationFormat):
    """Global feature importance datatable representation.

    Canonical representation (datatable frame, ltypes):

    .. code-block:: text

        | Required column    | Type  | Description        |
        |--------------------|-------|--------------------|
        | feature_name       | str   | Feature name.      |
        | feature_importance | real  | Feature importance |

    ... other optional columns are allowed

    """

    mime = MimeType.MIME_DATATABLE

    COL_NAME = "feature_name"
    COL_IMPORTANCE = "feature_importance"

    @staticmethod
    def from_lists(explanation, features: list, importances: list):
        if features and importances and len(features) == len(importances):
            return GlobalFeatImpDatatableFormat(
                explanation=explanation,
                frame=dt.Frame(
                    {
                        GlobalFeatImpDatatableFormat.COL_NAME: features,
                        GlobalFeatImpDatatableFormat.COL_IMPORTANCE: importances,
                    }
                ),
                frame_file=None,
            )

        raise ValueError(
            f"Features names ({len(features)})/importances ({len(importances)})"
            f" empty or have different lengths"
        )

    def __init__(
        self,
        explanation,
        frame: dt.Frame,
        frame_file: str | None,
        persistence: persistences.Persistence | None = None,
    ):
        DatatableCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            frame=GlobalFeatImpDatatableFormat.validate_data(frame),
            frame_file=frame_file,
            persistence=persistence,
        )

    @staticmethod
    def validate_data(frame_data: dt.Frame) -> dt.Frame:
        return frame_data


class GlobalFeatImpJSonDatatableFormat(TextCustomExplanationFormat):
    """Global feature importance JSon (index file) and datatable (data files)
    representation.

    The typical use of JSon+datatable feature importance representation:

    .. code-block:: python

        featImpJsonDt = GlobalFeatImpJSonDatatableFormat(...create...)
        # ... get other representations for free:
        featImpJSon = GlobalFeatImpJSonFormat.fromJSonDatatable(featImpJsonDt)
        featImpJSonCsv = GlobalFeatImpJSonCsvSonFormat.fromJSonDatatable(featImpJsonDt)

    JSon representation index file example:

    .. code-block:: text

        {
            "files": {
                "red_class": "feature_importance_class_0.jay"
                "green_class": "feature_importance_class_1.jay"
                "blue_class": "feature_importance_class_2.jay"
                ...
            },
            "metrics": [{"R2": 0.96}, {"RMSE": 0.03}],
            "total_rows": 592,
        }

    Datatable representation data file spec (datatable frame, ltypes; other optional
    columns are allowed):

    .. code-block:: text

        | Required column    | Type  | Description                           |
        |--------------------|-------|---------------------------------------|
        | feature_name       | str   | Feature name.                         |
        | feature_importance | real  | Feature importance                    |
        | global_scope       | bool  | Global/local feature importance scope |

    Datatable representation data file example:

    .. code-block:: text

           | feature_name  feature_importance  global_scope
        -- + ------------  ------------------  ------------
         0 | feature-a                    1.1             1
         1 | feature-b                    2.2             1

    """

    mime = MimeType.MIME_JSON_DATATABLE

    COL_NAME = "feature_name"
    COL_IMPORTANCE = "feature_importance"
    COL_GLOBAL_SCOPE = "global_scope"

    @staticmethod
    def from_lists(explanation, features: list, importances: list):
        if features and importances and len(features) == len(importances):
            return GlobalFeatImpDatatableFormat(
                explanation=explanation,
                frame=dt.Frame(
                    {
                        GlobalFeatImpDatatableFormat.COL_NAME: features,
                        GlobalFeatImpDatatableFormat.COL_IMPORTANCE: importances,
                    }
                ),
                frame_file=None,
            )

        raise ValueError(
            f"Features names ({len(features)})/importances ({len(importances)})"
            f" empty or have different lengths"
        )

    def __init__(
        self,
        explanation,
        json_data: str = None,
        json_file: str = None,
        persistence: persistences.Persistence | None = None,
    ):
        TextCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=GlobalFeatImpJSonDatatableFormat.validate_data(json_data),
            format_file=json_file,
            persistence=persistence,
        )

    @staticmethod
    def validate_data(json_data):
        return json_data

    @staticmethod
    def serialize_index_file(
        classes: list[str],
        default_class: str = "",
        metrics: list | None = None,
        doc: str = "",
        total_rows: int | None = None,
        data_file_prefix: str = "feature_importance",
        data_file_suffix: str = MimeType.EXT_DATATABLE,
    ) -> tuple[dict, str]:
        return GlobalFeatImpJSonFormat.serialize_index_file(
            classes=classes,
            default_class=default_class,
            metrics=metrics,
            doc=doc,
            total_rows=total_rows,
            data_file_prefix=data_file_prefix,
            data_file_suffix=data_file_suffix,
        )

    @staticmethod
    def dict_to_data_frame(
        feature_importances: dict[str, float], scope: str = "global"
    ) -> dt.Frame:
        """(Typical) feature importance dictionary to data frame conversion.

        Parameters
        ----------
        feature_importances: dict
          Feature importances as dictionary of feature name to importance.
        scope: str
          ``global`` or ``local``.

        Returns
        -------
        dt.Frame:
          Data file.

        """
        if not feature_importances:
            raise ValueError(
                "At least one feature importance must provided to serialize global "
                "feature importance JSon index file"
            )
        gom_data_frame: dt.Frame = dt.Frame(
            {
                GlobalFeatImpJSonDatatableFormat.COL_NAME: [],
                GlobalFeatImpJSonDatatableFormat.COL_IMPORTANCE: [],
                GlobalFeatImpJSonDatatableFormat.COL_GLOBAL_SCOPE: [],
            }
        )
        for feature in feature_importances:
            gom_data_frame.rbind(
                dt.Frame(
                    {
                        GlobalFeatImpJSonDatatableFormat.COL_NAME: [feature],
                        GlobalFeatImpJSonDatatableFormat.COL_IMPORTANCE: [
                            feature_importances[feature]
                        ],
                        GlobalFeatImpJSonDatatableFormat.COL_GLOBAL_SCOPE: [
                            True if scope == "global" else False
                        ],
                    }
                )
            )

        return gom_data_frame

    def add_data_frame(self, format_data: dt.Frame, file_name: str | None = None):
        self._pre_add_data(format_data=format_data, file_name=file_name)

        persist = self.explanation.explainer.persistence
        persist.store.make_dir(
            persist.get_explanation_dir_path(
                self.explanation.explanation_type(), self.mime
            )
        )
        self._add_frame_to_store(
            path=persist.get_explanation_file_path(
                self.explanation.explanation_type(),
                self.mime,
                explanation_file=file_name,
            ),
            format_data=format_data,
            persistence=persist.store,
        )

    def get_data(self, file_name: str | None = None):
        if not file_name or file_name == self.index_file_name:
            return TextCustomExplanationFormat.get_data(self, file_name)
        elif file_name.endswith(MimeType.EXT_DATATABLE):
            persist = self.explanation.explainer.persistence
            return dt.fread(
                persist.get_explanation_file_path(
                    self.explanation.explanation_type(),
                    GlobalFeatImpJSonDatatableFormat.mime,
                    file_name,
                )
            )
        else:
            return TextCustomExplanationFormat.get_data(self, file_name)

    @classmethod
    def is_paged(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
    ) -> bool:
        is_on_demand, _ = cls.is_on_demand(persistence, explanation_type)
        if is_on_demand:
            return False
        return True

    @classmethod
    def get_page(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        page_offset: int,
        page_size: int,
        result_format: str,
        explanation_filter: list[FilterEntry],
    ) -> str:
        if not GlobalFeatImpJSonDatatableFormat.is_paged(
            persistence=persistence, explanation_type=explanation_type
        ):
            raise errors.MliError(
                f"{cls.__name__} representation does not support paged explanations"
            )
        gfc = GlobalFeatImpJSonDatatableFormat
        if not explanation_filter:
            raise ValueError(f"Filter parameter '{gfc.FILTER_CLASS}' is required")
        e_filter: dict = dict()
        for i in explanation_filter:
            e_filter[i.filter_by] = i.value
        filter_cls: str = e_filter.get(gfc.FILTER_CLASS, "")
        if not filter_cls:
            raise ValueError(f"Filter parameter '{gfc.FILTER_CLASS}' is required")

        # find frame using an index file
        idx_dict = IceJsonDatatableFormat.load_index_file(
            persistence=persistence, explanation_type=explanation_type
        )
        if filter_cls not in idx_dict[gfc.KEY_FILES]:
            raise ValueError(
                f"Filter parameter '{gfc.FILTER_CLASS}' parameter "
                f"'{filter_cls}' is not available in the "
                f"representation ({list(idx_dict[gfc.KEY_FILES].keys())})"
            )

        df_path: str = ""
        try:
            df_file: str = idx_dict[gfc.KEY_FILES][filter_cls]
            df_path = persistence.get_explanation_file_path(
                explanation_type=explanation_type,
                explanation_format=MimeType.MIME_JSON_DATATABLE,
                explanation_file=df_file,
            )
            df_e: dt.Frame = dt.fread(df_path)
            if page_offset < df_e.shape[0]:
                page_size = (
                    page_size
                    if page_size and (page_offset + page_size) < df_e.shape[0]
                    else df_e.shape[0] - page_offset
                )
                df_page: dt.Frame = df_e[page_offset : page_offset + page_size, :]
                if MimeType.EXT_CSV == result_format:
                    return df_page.to_csv(path=None)
                else:
                    # JSon is fallback
                    return GlobalFeatImpJSonFormat.from_dataframe_to_json(df_page)
            else:
                raise errors.MliError(
                    f"Unable to load {cls.__name__} explanation for class {filter_cls} "
                    f"and page {page_offset}[{page_size}] as page is out of range "
                    f"(frame has {df_e.shape[0]} rows)"
                )
        except Exception as ex:
            raise errors.MliError(
                f"Unable to load {cls.__name__} explanation for class '{filter_cls}' "
                f"and page {page_offset}[{page_size}] from {df_path}: {ex}\n"
                f"{traceback.format_exc()}"
            )


class LocalFeatImpJSonDatatableFormat(GlobalFeatImpJSonDatatableFormat):
    mime = MimeType.MIME_JSON_DATATABLE

    def __init__(
        self,
        explanation,
        json_data: str = None,
        json_file: str = None,
        persistence: persistences.Persistence | None = None,
    ):
        GlobalFeatImpJSonDatatableFormat.__init__(
            self,
            explanation=explanation,
            json_data=json_data,
            json_file=json_file,
            persistence=persistence,
        )

    @staticmethod
    def serialize_index_file(
        classes: list[str],
        default_class: str = "",
        metrics: list | None = None,
        doc: str = "",
        total_rows: int | None = None,
        data_file_prefix: str = "feature_importance",
        data_file_suffix: str = MimeType.EXT_DATATABLE,
    ) -> tuple[dict, str]:
        return GlobalFeatImpJSonFormat.serialize_index_file(
            classes=classes,
            default_class=default_class,
            metrics=metrics,
            doc=doc,
            total_rows=total_rows,
            data_file_prefix=data_file_prefix,
            data_file_suffix=data_file_suffix,
        )

    @classmethod
    def is_on_demand(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
    ) -> tuple[bool, dict | None]:
        idx_dict = GlobalFeatImpJSonFormat.load_index_file(
            persistence,
            explanation_type,
            mime=LocalFeatImpJSonDatatableFormat.mime,
        )

        if idx_dict and idx_dict.get(LocalFeatImpJSonFormat.KEY_ON_DEMAND, False):
            return (
                True,
                idx_dict.get(LocalFeatImpJSonFormat.KEY_ON_DEMAND_PARAMS, None),
            )
        return False, None

    def add_data_frame(self, format_data: dt.Frame, file_name: str | None = None):
        self._pre_add_data(format_data=format_data, file_name=file_name)

        persist = self.explanation.explainer.persistence
        persist.store.make_dir(
            persist.get_explanation_dir_path(
                self.explanation.explanation_type(), self.mime
            )
        )
        self._add_frame_to_store(
            path=persist.get_explanation_file_path(
                self.explanation.explanation_type(),
                self.mime,
                explanation_file=file_name,
            ),
            format_data=format_data,
            persistence=persist.store,
        )

    def get_data(self, file_name: str | None = None):
        if not file_name or file_name == self.index_file_name:
            return TextCustomExplanationFormat.get_data(self, file_name)
        elif file_name.endswith(MimeType.EXT_DATATABLE):
            persist = self.explanation.explainer.persistence
            return dt.fread(
                persist.get_explanation_file_path(
                    self.explanation.explanation_type(), self.mime, file_name
                )
            )
        else:
            return TextCustomExplanationFormat.get_data(self, file_name)


class LocalFeatImpWithYhatsJSonDatatableFormat(LocalFeatImpJSonDatatableFormat):
    FILE_Y_HAT = "y_hat.bin"
    KEY_Y_HAT = "y_hat"


class GlobalFeatImpJSonCsvFormat(GlobalFeatImpJSonDatatableFormat):
    mime = MimeType.MIME_JSON_CSV

    def __init__(
        self,
        explanation,
        json_data: str = None,
        json_file: str = None,
        persistence: persistences.Persistence | None = None,
    ):
        GlobalFeatImpJSonDatatableFormat.__init__(
            self,
            explanation=explanation,
            json_data=GlobalFeatImpJSonCsvFormat.validate_data(json_data),
            json_file=json_file,
            persistence=persistence,
        )

    @staticmethod
    def validate_data(json_data):
        return json_data

    @staticmethod
    def serialize_index_file(
        classes: list[str],
        default_class: str = "",
        metrics: list | None = None,
        doc: str = "",
        total_rows: int | None = None,
        data_file_prefix: str = "feature_importance",
        data_file_suffix: str = "csv",
    ) -> tuple[dict, str]:
        return GlobalFeatImpJSonFormat.serialize_index_file(
            classes=classes,
            default_class=default_class,
            metrics=metrics,
            doc=doc,
            total_rows=total_rows,
            data_file_prefix=data_file_prefix,
            data_file_suffix=data_file_suffix,
        )

    @staticmethod
    def from_json_datatable(
        json_dt_format: GlobalFeatImpJSonDatatableFormat,
    ) -> "GlobalFeatImpJSonCsvFormat":
        if json_dt_format:
            # index file
            jd_idx_str = json_dt_format.get_data(
                file_name=(
                    f"{GlobalFeatImpJSonFormat.FILE_PREFIX_EXPLANATION_IDX}"
                    f"{MimeType.EXT_JSON}"
                )
            )
            jd_idx_dict: dict = json.loads(jd_idx_str)
            idx_dict: dict = dict()
            idx_dict[GlobalFeatImpJSonFormat.KEY_FILES] = dict()
            idx_dict[GlobalFeatImpJSonFormat.KEY_METRICS] = (
                jd_idx_dict[GlobalFeatImpJSonFormat.KEY_METRICS].copy()
                if GlobalFeatImpJSonFormat.KEY_METRICS in jd_idx_dict
                else []
            )
            idx_dict[GlobalFeatImpJSonFormat.KEY_DOC] = (
                jd_idx_dict[GlobalFeatImpJSonFormat.KEY_DOC]
                if GlobalFeatImpJSonFormat.KEY_DOC in jd_idx_dict
                else ""
            )
            if GlobalFeatImpJSonFormat.KEY_TOTAL_ROWS in jd_idx_dict:
                idx_dict[GlobalFeatImpJSonFormat.KEY_TOTAL_ROWS] = jd_idx_dict[
                    GlobalFeatImpJSonFormat.KEY_TOTAL_ROWS
                ]
            for clazz in jd_idx_dict[GlobalFeatImpJSonFormat.KEY_FILES]:
                idx_dict[GlobalFeatImpJSonFormat.KEY_FILES][clazz] = jd_idx_dict[
                    GlobalFeatImpJSonFormat.KEY_FILES
                ][clazz].replace(f".{MimeType.EXT_DATATABLE}", f".{MimeType.EXT_CSV}")

            result = GlobalFeatImpJSonCsvFormat(
                explanation=json_dt_format.explanation,
                json_data=json.dumps(idx_dict, indent=4),
            )

            # data files (
            tmp_dir = persistences.Persistence.make_temp_dir()
            for clazz in jd_idx_dict[GlobalFeatImpJSonFormat.KEY_FILES]:
                data_file: str = jd_idx_dict[GlobalFeatImpJSonFormat.KEY_FILES][clazz]
                frame: dt.Frame = json_dt_format.get_data(data_file)
                path = persistences.Persistence.make_key(tmp_dir, data_file)
                frame.to_csv(path)
                result.add_file(
                    format_file=path,
                    file_name=data_file.replace(
                        f".{MimeType.EXT_DATATABLE}", f".{MimeType.EXT_CSV}"
                    ),
                )
            persistences.Persistence.delete_temp_dir(tmp_dir)

            return result

        raise ValueError("Valid and non-empty representation must be provided")


class GlobalFeatImpJSonFormat(TextCustomExplanationFormat, GrammarOfMliFormat):
    """Representation of global feature importance explanation as JSon.

    JSon representation index file example:

    .. code-block:: text

        {
            "files": {
                "red_class": "feature_importance_class_0.json"
                "green_class": "feature_importance_class_1.json"
                "blue_class": "feature_importance_class_2.json"
                ...
            },
            "metrics": [{"R2": 0.96}, {"RMSE": 0.03}],
            "total_rows": 592,
        }

    JSon representation data file example:

    .. code-block:: text

        {
            data: [
                {
                    label: str,
                    value: num,
                    scope: str,
                }+
            ]
            bias: num
        }

    Where:

    * ``label`` is feature name
    * ``value`` is feature importance
    * ``scope`` is either ``local`` or ``global``

    """

    mime = MimeType.MIME_JSON

    KEY_LABEL = "label"
    KEY_VALUE = "value"

    def __init__(
        self,
        explanation,
        json_data: str = None,
        json_file: str = None,
        persistence: persistences.Persistence | None = None,
    ):
        TextCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=GlobalFeatImpJSonFormat.validate_data(json_data),
            format_file=json_file,
            persistence=persistence,
        )

    @staticmethod
    def validate_data(json_data: str) -> str:
        return json_data

    @staticmethod
    def serialize_index_file(
        classes: list[str],
        default_class: str = "",
        metrics: list | None = None,
        keywords: list | None = None,
        doc: str = "",
        total_rows: int | None = None,
        data_file_prefix: str = "feature_importance",
        data_file_suffix: str = "json",
    ) -> tuple[dict, str]:
        """JSon index file serialization to string.

        Parameters
        ----------
        classes: list
          Classes.
        default_class: str
          Class to be shown as default (the first one) e.g. the class of interest in
          case of binomial experiment interpretation.
        metrics: list
          Optional list of PD related metrics e.g. ``[{"RMSE": 0.02}, {"SD": 3.1}]``.
        keywords : list[str]
          Optional list of keywords indicating representation features, properties
          and aspects.
        doc: str
          Documentation.
        total_rows: int
          Total number of rows (which can be used for pagination).
        data_file_prefix: str
          Prefix for data file names.
        data_file_suffix: str
          Suffix for data file names.

        Returns
        -------
        Tuple[dict, str]:
          Dictionary with mapping of classes to file names AND JSon serialization
          (as string).

        """

        pdj = PartialDependenceJSonFormat
        index_dict: dict = dict()
        index_dict[pdj.KEY_FILES] = dict()
        for i_c, cls in enumerate(classes):
            index_dict[pdj.KEY_FILES][cls] = (
                f"{data_file_prefix}_class_{i_c}.{data_file_suffix}"
            )
        TextCustomExplanationFormat.set_index_commons(
            index_dict=index_dict,
            classes=classes,
            default_class=default_class,
            metrics=metrics,
            keywords=keywords,
            doc=doc,
            total_rows=total_rows,
        )

        return index_dict, json.dumps(index_dict, indent=4)

    @staticmethod
    def serialize_data_file(
        feature_importances: dict[str, float],
        scope: str = "global",
        bias: float | None = None,
    ) -> str:
        """JSon data file serialization to string.

        Parameters
        ----------
        feature_importances: dict
          Feature importances as dictionary of feature name to importance.
        scope: str
          ``global`` or ``local``.
        bias: optional str
          Bias value.

        Returns
        -------
        str:
          Data file serialization.

        """
        if not feature_importances:
            raise ValueError(
                "At least one feature importance must provided to serialize global "
                "feature importance JSon index file"
            )
        data_dict: dict = dict()
        data = list()
        for feature in feature_importances:
            data.append(
                {
                    GlobalFeatImpJSonFormat.KEY_LABEL: feature,
                    GlobalFeatImpJSonFormat.KEY_VALUE: feature_importances[feature],
                    GlobalFeatImpJSonFormat.KEY_SCOPE: scope,
                }
            )
        data_dict[GlobalFeatImpJSonFormat.KEY_DATA] = data
        if bias:
            data_dict[GlobalFeatImpJSonFormat.KEY_BIAS] = bias
        return json.dumps(data_dict)

    @staticmethod
    def from_dataframe_to_json(frame: dt.Frame, bias_col: str = None) -> str:
        featimp_dict: dict = dict()

        # the bias value should be stored differently in the output json
        bias_value = None
        if (
            bias_col
            and bias_col in frame[GlobalFeatImpDatatableFormat.COL_NAME].to_list()[0]
        ):
            bias_value = frame[
                dtf[GlobalFeatImpDatatableFormat.COL_NAME] == bias_col,
                dtf[GlobalFeatImpDatatableFormat.COL_IMPORTANCE],
            ].to_list()[0][0]
            del frame[dtf[GlobalFeatImpDatatableFormat.COL_NAME] == bias_col, :]

        for row in range(frame.shape[0]):
            featimp_dict[frame[row, GlobalFeatImpJSonDatatableFormat.COL_NAME]] = frame[
                row, GlobalFeatImpJSonDatatableFormat.COL_IMPORTANCE
            ]
        return GlobalFeatImpJSonFormat.serialize_data_file(
            feature_importances=featimp_dict, bias=bias_value
        )

    @staticmethod
    def from_json_datatable(
        json_dt_format: GlobalFeatImpJSonDatatableFormat, bias_col: str = None
    ) -> "GlobalFeatImpJSonFormat":
        if json_dt_format:
            # index file
            jd_idx_str = json_dt_format.get_data(
                file_name=(
                    f"{GlobalFeatImpJSonFormat.FILE_PREFIX_EXPLANATION_IDX}"
                    f"{MimeType.EXT_JSON}"
                )
            )
            jd_idx_dict: dict = json.loads(jd_idx_str)
            idx_dict: dict = dict()
            idx_dict[GlobalFeatImpJSonFormat.KEY_FILES] = dict()
            idx_dict[GlobalFeatImpJSonFormat.KEY_METRICS] = (
                jd_idx_dict[GlobalFeatImpJSonFormat.KEY_METRICS].copy()
                if GlobalFeatImpJSonFormat.KEY_METRICS in jd_idx_dict
                else []
            )
            idx_dict[GlobalFeatImpJSonFormat.KEY_DOC] = (
                jd_idx_dict[GlobalFeatImpJSonFormat.KEY_DOC]
                if GlobalFeatImpJSonFormat.KEY_DOC in jd_idx_dict
                else ""
            )
            if GlobalFeatImpJSonFormat.KEY_TOTAL_ROWS in jd_idx_dict:
                idx_dict[GlobalFeatImpJSonFormat.KEY_TOTAL_ROWS] = jd_idx_dict[
                    GlobalFeatImpJSonFormat.KEY_TOTAL_ROWS
                ]
            for clazz in jd_idx_dict[GlobalFeatImpJSonFormat.KEY_FILES]:
                idx_dict[GlobalFeatImpJSonFormat.KEY_FILES][clazz] = jd_idx_dict[
                    GlobalFeatImpJSonFormat.KEY_FILES
                ][clazz].replace(f".{MimeType.EXT_DATATABLE}", f".{MimeType.EXT_JSON}")

            result = GlobalFeatImpJSonFormat(
                explanation=json_dt_format.explanation,
                json_data=json.dumps(idx_dict, indent=4),
            )

            # data files
            for clazz in jd_idx_dict[GlobalFeatImpJSonFormat.KEY_FILES]:
                data_file: str = jd_idx_dict[GlobalFeatImpJSonFormat.KEY_FILES][clazz]
                frame: dt.Frame = json_dt_format.get_data(data_file)
                data_file_str = GlobalFeatImpJSonFormat.from_dataframe_to_json(
                    frame=frame, bias_col=bias_col
                )
                result.add_data(
                    format_data=data_file_str,
                    file_name=data_file.replace(
                        f".{MimeType.EXT_DATATABLE}", f".{MimeType.EXT_JSON}"
                    ),
                )

            return result

        raise ValueError("Valid and non-empty representation must be provided")

    @classmethod
    def load_index_file(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        mime: str = MimeType.MIME_JSON,
    ) -> dict:
        """Load index file and check parameters.

        Returns
        -------
        dict:
          Index file as dictionary.

        """
        #  index file
        idx_path = persistence.get_explanation_file_path(explanation_type, mime)
        return persistence.store.load_json(idx_path)

    @classmethod
    def get_global_explanation(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
    ) -> str:
        """Get global feature importance explanation.

        Parameters
        ----------
        persistence:
          Persistence object initialized for explainer/MLI run.
        explanation_type: str
          Explanation type ~ explanation ID.

        """
        # persistence and explanation type is checked by caller
        idx_dict = GlobalFeatImpJSonFormat.load_index_file(
            persistence, explanation_type
        )

        filter_cls = next(iter(idx_dict[cls.KEY_FILES]))
        file_path: str = ""
        try:
            file_path = idx_dict[cls.KEY_FILES][filter_cls]
            file_path = persistence.get_explanation_file_path(
                explanation_type=explanation_type,
                explanation_format=cls.mime,
                explanation_file=file_path,
            )
            explanation_dict = persistence.store.load_json(file_path)
            return json.dumps(explanation_dict)
        except Exception as ex:
            raise RuntimeError(
                f"Unable to load global feature importance "
                f"from {file_path}: {ex}\n{traceback.format_exc()}"
            )


class LocalFeatImpJSonFormat(TextCustomExplanationFormat, GrammarOfMliFormat):
    """Representation of local feature importance explanation as JSon. See
    `GlobalFeatImpJSonFormat` for structure of the index file and data.

    """

    KEY_Y = "prediction"
    mime = MimeType.MIME_JSON

    def __init__(
        self,
        explanation,
        json_data: str = None,
        json_file: str = None,
        persistence: persistences.Persistence | None = None,
    ):
        TextCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=LocalFeatImpJSonFormat.validate_data(json_data),
            format_file=json_file,
            persistence=persistence,
        )

    @staticmethod
    def serialize_index_file(
        classes: list[str],
        default_class: str = "",
        metrics: list | None = None,
        doc: str = "",
    ) -> tuple[dict, str]:
        return GlobalFeatImpJSonFormat.serialize_index_file(
            classes=classes,
            default_class=default_class,
            metrics=metrics,
            doc=doc,
        )

    @classmethod
    def load_index_file(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        mime: str = MimeType.MIME_JSON,
    ) -> dict:
        idx_path = persistence.get_explanation_file_path(explanation_type, mime)
        return persistence.store.load_json(idx_path)

    @classmethod
    def is_on_demand(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
    ) -> tuple[bool, dict | None]:
        idx_dict = LocalFeatImpJSonFormat.load_index_file(persistence, explanation_type)

        if idx_dict and idx_dict.get(LocalFeatImpJSonFormat.KEY_ON_DEMAND, False):
            return (
                True,
                idx_dict.get(LocalFeatImpJSonFormat.KEY_ON_DEMAND_PARAMS, None),
            )
        return False, None

    @staticmethod
    def sort_data(json_dict: dict):
        """Sort local feature importance explanation data by (abs) value:

        .. code-block:: text

            {'data': [{'label': .,'value': .,'scope': .}, ...

        """
        if json_dict and json_dict.get(LocalFeatImpJSonFormat.KEY_DATA):

            def sort_by_value(item):
                value = item[GlobalFeatImpJSonFormat.KEY_VALUE] or 0.0
                return abs(value)

            json_dict.get(LocalFeatImpJSonFormat.KEY_DATA).sort(
                key=sort_by_value, reverse=True
            )

        return json_dict

    @staticmethod
    def merge_local_and_global_page(
        global_page: dict,
        local_page: dict,
        mli_key: str = "",
        explainer_job_key: str = "",
        bias_key: str = "",
        logger=None,
    ):
        """Use this method to merge local and global explanations page (especially
        if frontend is not able to process local explanations only.

        Local explanations page is expected to be sorted (as required) and it defines
        order of entries in the merged page. Merged result contains global explanation
        entry followed by local exp entry.

        """
        global_page_data_dict = {}
        if (
            global_page
            and global_page.get(GlobalFeatImpJSonFormat.KEY_DATA, None)
            and len(global_page.get(GlobalFeatImpJSonFormat.KEY_DATA))
        ):
            global_page_data_dict = {
                e[GlobalFeatImpJSonFormat.KEY_LABEL]: e
                for e in global_page.get(GlobalFeatImpJSonFormat.KEY_DATA, [])
            }

        try:
            if (
                local_page
                and local_page.get(LocalFeatImpJSonFormat.KEY_DATA, None)
                and global_page_data_dict
                and len(local_page[LocalFeatImpJSonFormat.KEY_DATA])
                == len(global_page_data_dict)
            ):
                merged_data = []
                for i, entry in enumerate(
                    local_page.get(LocalFeatImpJSonFormat.KEY_DATA)
                ):
                    if (
                        global_page_data_dict.get(
                            entry[GlobalFeatImpJSonFormat.KEY_LABEL], None
                        )
                        is None
                    ):
                        raise RuntimeError(
                            f"No global explanation for local: {entry} ({i})"
                        )
                    if (
                        bias_key is not None
                        and entry.get(GlobalFeatImpJSonFormat.KEY_LABEL, None)
                        == bias_key
                    ):
                        bias = entry.get(GlobalFeatImpJSonFormat.KEY_VALUE, None)
                        if bias is not None:
                            local_page[GlobalFeatImpJSonFormat.KEY_BIAS] = bias
                    else:
                        merged_data.append(
                            global_page_data_dict[
                                entry[GlobalFeatImpJSonFormat.KEY_LABEL]
                            ]
                        )
                        merged_data.append(entry)

                local_page[LocalFeatImpJSonFormat.KEY_DATA] = merged_data
        except Exception as ex:
            # frontend should be able to handle LOCAL only (merged result is better)
            if logger:
                logger.warning(
                    f"Unable to merge local feature importance "
                    f"{mli_key}/{explainer_job_key}: {ex}\n{traceback.format_exc()}"
                )
        # either merged pages OR local page as fallback
        return local_page

    @staticmethod
    def validate_data(json_data: str) -> str:
        return json_data


class LocalNlpLocoJSonFormat(TextCustomExplanationFormat):
    """Representation of local LOCO explanation as JSon. See
    `GlobalNlpLocoJSonFormat` for structure of the index file and data.

    """

    mime = MimeType.MIME_JSON

    def __init__(
        self,
        explanation,
        json_data: str = None,
        json_file: str = None,
        persistence: persistences.Persistence | None = None,
    ):
        TextCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=LocalFeatImpJSonFormat.validate_data(json_data),
            format_file=json_file,
            persistence=persistence,
        )

    @staticmethod
    def serialize_index_file(
        classes: list[str],
        default_class: str = "",
        metrics: list | None = None,
        doc: str = "",
    ) -> tuple[dict, str]:
        return GlobalFeatImpJSonFormat.serialize_index_file(
            classes=classes,
            default_class=default_class,
            metrics=metrics,
            doc=doc,
        )

    @classmethod
    def load_index_file(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        mime: str = MimeType.MIME_JSON,
    ) -> dict:
        idx_path = persistence.get_explanation_file_path(explanation_type, mime)
        return persistence.store.load_json(idx_path)

    @classmethod
    def is_on_demand(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
    ) -> tuple[bool, dict | None]:
        idx_dict = LocalFeatImpJSonFormat.load_index_file(persistence, explanation_type)

        if idx_dict and idx_dict.get(LocalFeatImpJSonFormat.KEY_ON_DEMAND, False):
            return (
                True,
                idx_dict.get(LocalFeatImpJSonFormat.KEY_ON_DEMAND_PARAMS, None),
            )
        return False, None

    @staticmethod
    def sort_data(json_dict: dict):
        """Sort local feature importance explanation data by (abs) value:

        .. code-block:: text

            {'data': [{'label': .,'value': .,'scope': .}, ...

        """
        if json_dict and json_dict.get(LocalFeatImpJSonFormat.KEY_DATA):

            def sort_by_value(item):
                value = item[GlobalFeatImpJSonFormat.KEY_VALUE] or 0.0
                return abs(value)

            json_dict.get(LocalFeatImpJSonFormat.KEY_DATA).sort(
                key=sort_by_value, reverse=True
            )

        return json_dict

    @staticmethod
    def merge_local_and_global_page(
        global_page: dict,
        local_page: dict,
        mli_key: str = "",
        explainer_job_key: str = "",
        bias_key: str = "",
        logger=None,
    ):
        """Use this method to merge local and global explanations page (especially
        if frontend is not able to process local explanations only.

        Local explanations page is expected to be sorted (as required) and it defines
        order of entries in the merged page. Merged result contains global explanation
        entry followed by local exp entry.

        """
        global_page_data_dict = {}
        if (
            global_page
            and global_page.get(GlobalFeatImpJSonFormat.KEY_DATA, None)
            and len(global_page.get(GlobalFeatImpJSonFormat.KEY_DATA))
        ):
            global_page_data_dict = {
                e[GlobalFeatImpJSonFormat.KEY_LABEL]: e
                for e in global_page.get(GlobalFeatImpJSonFormat.KEY_DATA, [])
            }

        try:
            if (
                local_page
                and local_page.get(LocalFeatImpJSonFormat.KEY_DATA, None)
                and global_page_data_dict
                and len(local_page[LocalFeatImpJSonFormat.KEY_DATA])
                == len(global_page_data_dict)
            ):
                merged_data = []
                for i, entry in enumerate(
                    local_page.get(LocalFeatImpJSonFormat.KEY_DATA)
                ):
                    if (
                        global_page_data_dict.get(
                            entry[GlobalFeatImpJSonFormat.KEY_LABEL], None
                        )
                        is None
                    ):
                        raise RuntimeError(
                            f"No global explanation for local: {entry} ({i})"
                        )
                    if (
                        bias_key is not None
                        and entry.get(GlobalFeatImpJSonFormat.KEY_LABEL, None)
                        == bias_key
                    ):
                        bias = entry.get(GlobalFeatImpJSonFormat.KEY_VALUE, None)
                        if bias is not None:
                            local_page[GlobalFeatImpJSonFormat.KEY_BIAS] = bias
                    else:
                        merged_data.append(
                            global_page_data_dict[
                                entry[GlobalFeatImpJSonFormat.KEY_LABEL]
                            ]
                        )
                        merged_data.append(entry)

                local_page[LocalFeatImpJSonFormat.KEY_DATA] = merged_data
        except Exception as ex:
            # frontend should be able to handle LOCAL only (merged result is better)
            if logger:
                logger.warning(
                    f"Unable to merge local NLP LOCO {mli_key}/{explainer_job_key}: "
                    f"{ex}\n{traceback.format_exc()}"
                )
        # either merged pages OR local page as fallback
        return local_page

    @staticmethod
    def validate_data(json_data: str) -> str:
        return json_data


class GlobalSummaryFeatImpJsonDatatableFormat(
    TextCustomExplanationFormat, GrammarOfMliFormat
):
    """Representation of global **summary** feature importance explanation as JSon.

    JSon representation index file example:

    .. code-block:: text

        {
            "files": {
                "red_class": "feature_importance_summary_class_0.jay"
                "green_class": "feature_importance_summary_class_1.jay"
                "blue_class": "feature_importance_summary_class_2.jay"
                ...
            },
            "metrics": [{"R2": 0.96}, {"RMSE": 0.03}],
            "total_rows": 25,
        }

    Where:

    * ``total_rows`` is number of features.

    Getting data file:

    .. code-block:: python

        > datatable.fread("feature_importance_summary_class_2.jay")

    JSon representation data file example:

    .. code-block:: text

           |  feature   shapley_value   count   avg_high_value   clazz   order
        -- + --------- --------------- ------- ---------------- ------- -------
         0 |  PAY_0      0.390716        0      0.390716         "red"   0
         1 |  PAY_0     -0.386815       25      0.38681          "red"   0
         ...
         . |  AGE        0.425908       17      0.425908         "red"   1
         ...

    Where:

    * ``feature`` is feature name (y-axis)
    * ``shapley_value`` is Shapley value (x-axis)
    * ``count`` frequency of the Shapley value (height, normalized to [0, 1])
    * ``avg_high_value`` average feature value height (color) normalized to
      [0, 1] (if feature value is low, it's 0, if it's high, then it's 1) in case of
      numerical features, ``None`` in case of categorical features.
    * ``order`` feature order to ensure "order by feature importance" paging

    """

    mime = MimeType.MIME_JSON_DATATABLE

    KEY_FEATURE = "feature"
    KEY_SHAPLEY = "shapley_value"
    KEY_FREQUENCY = "count"
    KEY_HIGH_VALUE = "avg_high_value"
    KEY_ORDER = "order"

    def __init__(
        self,
        explanation,
        json_data: str = None,
        json_file: str = None,
        persistence: persistences.Persistence | None = None,
    ):
        TextCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=IceJsonDatatableFormat.validate_data(json_data),
            format_file=json_file,
            persistence=persistence,
        )

    @staticmethod
    def validate_data(json_data: str) -> str:
        return json_data

    @staticmethod
    def serialize_index_file(
        classes: list[str],
        default_class: str = "",
        metrics: list | None = None,
        keywords: list | None = None,
        doc: str = "",
        total_rows: int | None = None,
        data_file_prefix: str = "summary_feature_importance",
        data_file_suffix: str = "jay",
    ) -> tuple[dict, str]:
        """JSon index file serialization to string.

        Parameters
        ----------
        classes: list
          Classes.
        default_class: str
          Class to be shown as default (the first one) e.g. the class of interest in
          case of binomial experiment interpretation.
        metrics: list
          Optional list of PD related metrics e.g. ``[{"RMSE": 0.02}, {"SD": 3.1}]``.
        keywords : list[str]
          Optional list of keywords indicating representation features, properties
          and aspects.
        doc: str
          Documentation.
        total_rows: int
          Total number of rows (which can be used for pagination).
        data_file_prefix: str
          Prefix for data file names.
        data_file_suffix: str
          Suffix for data file names.

        Returns
        -------
        Tuple[dict, str]:
          Dictionary with mapping of classes to file names AND JSon serialization
          (as string).

        """

        pdj = PartialDependenceJSonFormat
        index_dict: dict = dict()
        index_dict[pdj.KEY_FILES] = dict()
        for i_c, cls in enumerate(classes):
            index_dict[pdj.KEY_FILES][cls] = (
                f"{data_file_prefix}_class_{i_c}.{data_file_suffix}"
            )
        TextCustomExplanationFormat.set_index_commons(
            index_dict=index_dict,
            classes=classes,
            default_class=default_class,
            metrics=metrics,
            keywords=keywords,
            doc=doc,
            total_rows=total_rows,
        )

        return index_dict, json.dumps(index_dict, indent=4)

    @classmethod
    def load_index_file(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        mime: str = MimeType.MIME_JSON_DATATABLE,
    ) -> dict:
        """Load index file and check parameters.

        Returns
        -------
        dict:
          Index file as dictionary.

        """
        #  index file
        idx_path = persistence.get_explanation_file_path(explanation_type, mime)
        return persistence.store.load_json(idx_path)

    def add_data_frame(self, format_data: dt.Frame, file_name: str | None = None):
        self._pre_add_data(format_data=format_data, file_name=file_name)

        persist = self.explanation.explainer.persistence
        persist.store.make_dir(
            persist.get_explanation_dir_path(
                self.explanation.explanation_type(), self.mime
            )
        )
        self._add_frame_to_store(
            path=persist.get_explanation_file_path(
                self.explanation.explanation_type(),
                self.mime,
                explanation_file=file_name,
            ),
            format_data=format_data,
            persistence=persist.store,
        )


class GlobalSummaryFeatImpJsonFormat(TextCustomExplanationFormat, GrammarOfMliFormat):
    """Representation of global **summary** feature importance explanation as JSon.

    JSon representation index file example:

    .. code-block:: text

        {
            "files": {
                "red_class": {
                    "0": "feature_importance_class_0_offset_0.json",
                    "10": "feature_importance_class_0_offset_10.json",
                    "20": "feature_importance_class_0_offset_20.json"
                },
                "green_class": {
                    ...
                },
                "blue_class":  {
                    "0": "feature_importance_class_2_offset_0.json",
                    "10": "feature_importance_class_2_offset_10.json",
                    "20": "feature_importance_class_2_offset_20.json"
                },
                ...
            },
            "metrics": [{"R2": 0.96}, {"RMSE": 0.03}],
            "total_rows": 25,
            "rows_per_page": 10
        }

    Where:

    * Every class dictionary has per-page offset key with the JSon file containing
      chart for given page. Offset is based on the number of rows (features) per page.
    * ``total_rows`` is number of features.
    * ``rows_per_page`` is number of features in every file (created per page)

    JSon representation data file example:

    .. code-block:: text

        {
            data: [
                {
                    feature: str,
                    shapley_value: num,
                    count: num,
                    avg_high_value: num,
                    order: num,
                }+
            ]
        }

    Where:

    * ``feature`` is feature name (y-axis)
    * ``shapley_value`` is Shapley value (x-axis)
    * ``count`` frequency of the Shapley value (height, normalized to [0, 1])
    * ``avg_high_value`` average feature value height (color) normalized to
      [0, 1] (if feature value is low, it's 0, if it's high, then it's 1) in case of
      numerical features, ``None`` in case of categorical features.
    * ``order`` is feature order (global feature importance).

    """

    mime = MimeType.MIME_JSON

    KEY_FEATURE = GlobalSummaryFeatImpJsonDatatableFormat.KEY_FEATURE
    KEY_SHAPLEY = GlobalSummaryFeatImpJsonDatatableFormat.KEY_SHAPLEY
    KEY_FREQUENCY = GlobalSummaryFeatImpJsonDatatableFormat.KEY_FREQUENCY
    KEY_HIGH_VALUE = GlobalSummaryFeatImpJsonDatatableFormat.KEY_HIGH_VALUE
    KEY_ORDER = GlobalSummaryFeatImpJsonDatatableFormat.KEY_ORDER

    KEY_FEATURES_PER_PAGE = "features_per_page"

    # 10 features per page
    DEFAULT_PAGE_SIZE = 10

    DATA_FILE_PREFIX = "summary_feature_importance"

    def __init__(
        self,
        explanation,
        json_data: str = None,
        json_file: str = None,
        persistence: persistences.Persistence | None = None,
    ):
        TextCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=GlobalFeatImpJSonFormat.validate_data(json_data),
            format_file=json_file,
            persistence=persistence,
        )

    @staticmethod
    def validate_data(json_data: str) -> str:
        return json_data

    @staticmethod
    def from_json_datatable(
        json_dt_format: GlobalSummaryFeatImpJsonDatatableFormat,
        page_size: int,
        total_rows: int = -1,
        persistence: persistences.Persistence | None = None,
        index_extensions: dict | None = None,
    ) -> tuple["GlobalSummaryFeatImpJsonFormat", dict]:
        if json_dt_format:
            # features per-page index for local explanation (class > offset > features)
            features_per_page = {}

            # index file
            me = GlobalSummaryFeatImpJsonFormat
            jd_idx_str = json_dt_format.get_data(
                file_name=f"{me.FILE_PREFIX_EXPLANATION_IDX}{MimeType.EXT_JSON}"
            )
            jd_idx_dict: dict = json.loads(jd_idx_str)

            (idx_dict, idx_str) = me.serialize_index_file(
                classes=[c for c in jd_idx_dict[me.KEY_FILES].keys()],
                metrics=(
                    jd_idx_dict[me.KEY_METRICS].copy()
                    if me.KEY_METRICS in jd_idx_dict
                    else []
                ),
                default_class=(
                    jd_idx_dict[me.KEY_DEFAULT_CLASS]
                    if me.KEY_DEFAULT_CLASS in jd_idx_dict
                    else ""
                ),
                doc=(jd_idx_dict[me.KEY_DOC] if me.KEY_DOC in jd_idx_dict else ""),
                total_rows=(
                    total_rows
                    if total_rows >= 0
                    else (
                        jd_idx_dict[me.KEY_TOTAL_ROWS]
                        if me.KEY_TOTAL_ROWS in jd_idx_dict
                        else None
                    )
                ),
                rows_per_page=page_size,
            )
            if index_extensions:
                idx_dict.update(index_extensions)
                idx_str = json.dumps(idx_dict, indent=4)

            result = GlobalSummaryFeatImpJsonFormat(
                explanation=json_dt_format.explanation,
                json_data=idx_str,
                persistence=persistence,
            )

            # data file
            persistence = json_dt_format.explanation.explainer.persistence
            src_dir_path = persistence.get_explanation_dir_path(
                explanation_type=json_dt_format.explanation.explanation_type(),
                explanation_format=MimeType.MIME_JSON_DATATABLE,
            )
            dst_dir_path = persistence.get_explanation_dir_path(
                explanation_type=json_dt_format.explanation.explanation_type(),
                explanation_format=MimeType.MIME_JSON,
            )
            # current feature and current page features
            page_size = page_size or 1_000_000
            # source: data file is per-class (no paging)
            for clazz in idx_dict[GlobalFeatImpJSonFormat.KEY_FILES]:
                src_data_file_name = jd_idx_dict[GlobalFeatImpJSonFormat.KEY_FILES][
                    clazz
                ]
                src_df_key = persistences.Persistence.make_key(
                    src_dir_path, src_data_file_name
                )
                src_df = dt.fread(
                    persistence.store.load(src_df_key)
                    if persistences.PersistenceType.file_system
                    != persistence.store.type
                    else src_df_key
                )
                features_per_page[clazz] = {}

                # target: create data file per-class and per-page
                for page_number, offset in enumerate(
                    idx_dict[GlobalFeatImpJSonFormat.KEY_FILES][clazz]
                ):
                    page_feature_offset = page_number * page_size
                    page_end_feature_offset = page_feature_offset + page_size
                    page_file_name = idx_dict[GlobalFeatImpJSonFormat.KEY_FILES][clazz][
                        offset
                    ]
                    page_df = src_df[
                        (dt.f["order"] >= page_feature_offset)
                        & (dt.f["order"] < page_end_feature_offset),
                        :,
                    ]
                    # convert src DF page to dst JSon dict file
                    if page_df.shape[0]:
                        features_per_page[clazz][offset] = dt.unique(
                            page_df[me.KEY_FEATURE]
                        ).to_list()[0]
                        page_df_dict = page_df.to_dict()
                        page_data = []
                        page_dict = {me.KEY_DATA: page_data}
                        for r in range(len(page_df_dict[me.KEY_FEATURE])):
                            page_data.append(
                                {
                                    me.KEY_FEATURE: page_df_dict[me.KEY_FEATURE][r],
                                    me.KEY_SHAPLEY: page_df_dict[me.KEY_SHAPLEY][r],
                                    me.KEY_FREQUENCY: page_df_dict[me.KEY_FREQUENCY][r],
                                    me.KEY_HIGH_VALUE: page_df_dict[me.KEY_HIGH_VALUE][
                                        r
                                    ],
                                }
                            )
                        persistence.store.save_json(
                            key=persistences.Persistence.make_key(
                                dst_dir_path, page_file_name
                            ),
                            data=page_dict,
                        )

            return result, features_per_page

        raise ValueError("Valid and non-empty representation must be provided")

    @staticmethod
    def serialize_index_file(
        classes: list[str],
        default_class: str = "",
        metrics: list | None = None,
        keywords: list | None = None,
        doc: str = "",
        total_rows: int | None = None,
        rows_per_page: int | None = None,
        data_file_prefix: str = DATA_FILE_PREFIX,
        data_file_suffix: str = "json",
    ) -> tuple[dict, str]:
        cef = ExplanationFormat
        index_dict: dict = dict()
        if rows_per_page:
            index_dict[GlobalSummaryFeatImpJsonFormat.KEY_ROWS_PER_PAGE] = rows_per_page
        if total_rows and rows_per_page:
            page_count = int(total_rows / rows_per_page)
            if total_rows % rows_per_page:
                page_count += 1
            page_offsets = [str(o * rows_per_page) for o in range(page_count)]
        else:
            page_offsets = ["0"]

        index_dict[cef.KEY_FILES] = dict()
        for i_c, cls in enumerate(classes):
            index_dict[cef.KEY_FILES][cls] = {}
            for i_o, page_offset in enumerate(page_offsets):
                index_dict[cef.KEY_FILES][cls][page_offset] = (
                    f"{data_file_prefix}_class_{i_c}_offset_{i_o}.{data_file_suffix}"
                )

        TextCustomExplanationFormat.set_index_commons(
            index_dict=index_dict,
            classes=classes,
            default_class=default_class,
            metrics=metrics,
            keywords=keywords,
            doc=doc,
            total_rows=total_rows,
        )

        return index_dict, json.dumps(index_dict, indent=4)

    @staticmethod
    def serialize_data_file(
        feature_importances: dict[str, float],
        scope: str = "global",
        bias: float | None = None,
    ) -> str:
        """JSon data file serialization to string.

        Parameters
        ----------
        feature_importances: dict
          Feature importances as dictionary of feature name to importance.
        scope: str
          ``global`` or ``local``.
        bias: optional str
          Bias value.

        Returns
        -------
        str:
          Data file serialization.

        """
        if not feature_importances:
            raise ValueError(
                "At least one feature importance must provided to serialize global "
                "feature importance JSon index file"
            )
        data_dict: dict = dict()
        data = list()
        for feature in feature_importances:
            data.append(
                {
                    GlobalFeatImpJSonFormat.KEY_LABEL: feature,
                    GlobalFeatImpJSonFormat.KEY_VALUE: feature_importances[feature],
                    GlobalFeatImpJSonFormat.KEY_SCOPE: scope,
                }
            )
        data_dict[GlobalFeatImpJSonFormat.KEY_DATA] = data
        if bias:
            data_dict[GlobalFeatImpJSonFormat.KEY_BIAS] = bias
        return json.dumps(data_dict)

    @classmethod
    def is_paged(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
    ) -> bool:
        del persistence
        del explanation_type

        return True

    @classmethod
    def load_index_file(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        mime: str = MimeType.MIME_JSON,
    ) -> dict:
        #  index file
        idx_path = persistence.get_explanation_file_path(explanation_type, mime)
        return persistence.store.load_json(idx_path)

    @classmethod
    def get_page(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        page_offset: int,
        page_size: int,
        result_format: str,
        explanation_filter: list[FilterEntry],
    ) -> str:
        """Representation expect JSon+datatable representation to exist and use
        it to construct the page as expected

        """
        gfc = GlobalFeatImpJSonDatatableFormat
        if not explanation_filter:
            raise ValueError(f"Filter parameter '{gfc.FILTER_CLASS}' is required")
        e_filter: dict = dict()
        for i in explanation_filter:
            e_filter[i.filter_by] = i.value
        filter_cls: str = e_filter.get(gfc.FILTER_CLASS, "")
        if not filter_cls:
            raise ValueError(f"Filter parameter '{gfc.FILTER_CLASS}' is required")

        # load index file to find page for class and specific offset
        idx_dict = GlobalSummaryFeatImpJsonFormat.load_index_file(
            persistence=persistence, explanation_type=explanation_type
        )
        if filter_cls not in idx_dict[gfc.KEY_FILES]:
            raise ValueError(
                f"Filter parameter '{gfc.FILTER_CLASS}' parameter "
                f"'{filter_cls}' is not available in the "
                f"representation (try {list(idx_dict[gfc.KEY_FILES].keys())})"
            )
        # other than pre-generated pages are not supported (page size param ignored)
        if str(page_offset) not in idx_dict[gfc.KEY_FILES][filter_cls]:
            raise ValueError(
                f"Page offset {page_offset} parameter is not supported for class "
                f"'{filter_cls}' in the representation "
                f"(try {list(idx_dict[gfc.KEY_FILES][filter_cls].keys())})"
            )

        try:
            # load page as is (do NOT parse it ~ performance) and return it
            explanation_dir_path = persistence.get_explanation_dir_path(
                explanation_type=explanation_type,
                explanation_format=MimeType.MIME_JSON,
            )
            explanation_data_file = persistences.Persistence.make_key(
                explanation_dir_path,
                idx_dict[gfc.KEY_FILES][filter_cls][str(page_offset)],
            )
            return persistence.store.load(explanation_data_file)
        except Exception as ex:
            raise errors.MliError(
                f"Unable to load {cls.__name__} explanation for class '{filter_cls}' "
                f"and page {page_offset}[{page_size}]: {ex}\n"
                f"{traceback.format_exc()}"
            )


class LocalSummaryFeatImplJSonFormat(TextCustomExplanationFormat):
    """Local (on-demand) representation of summary feature importance as JSon."""

    mime = MimeType.MIME_JSON

    def __init__(
        self,
        explanation,
        json_data: str = None,
        json_file: str = None,
        persistence: persistences.Persistence | None = None,
    ):
        TextCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=LocalSummaryFeatImplJSonFormat.validate_data(json_data),
            format_file=json_file,
            persistence=persistence,
        )

    @staticmethod
    def validate_data(json_data: str) -> str:
        return json_data

    @staticmethod
    def serialize_index_file(
        classes: list[str],
        default_class: str = "",
        metrics: list | None = None,
        doc: str = "",
        data_file_prefix: str = "dt",
        data_file_suffix: str = "json",
    ) -> tuple[dict, str]:
        return GlobalSummaryFeatImpJsonFormat.serialize_index_file(
            classes=classes,
            default_class=default_class,
            metrics=metrics,
            doc=doc,
            data_file_prefix=data_file_prefix,
            data_file_suffix=data_file_suffix,
        )

    @classmethod
    def load_index_file(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        mime: str = MimeType.MIME_JSON_DATATABLE,
    ) -> dict:
        #  index file
        idx_path = persistence.get_explanation_file_path(explanation_type, mime)
        return persistence.store.load_json(idx_path)

    @staticmethod
    def serialize_on_demand_index_file(on_demand_params: dict) -> str:
        return json.dumps(
            {
                LocalSummaryFeatImplJSonFormat.KEY_ON_DEMAND: True,
                LocalSummaryFeatImplJSonFormat.KEY_ON_DEMAND_PARAMS: on_demand_params,
            }
        )

    @classmethod
    def is_paged(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
    ) -> bool:
        del persistence
        del explanation_type

        return True

    @classmethod
    def is_on_demand(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
    ) -> tuple[bool, dict | None]:
        idx_dict = LocalSummaryFeatImplJSonFormat.load_index_file(
            persistence=persistence,
            explanation_type=explanation_type,
            mime=LocalDtJSonFormat.mime,
        )

        if idx_dict and idx_dict.get(
            LocalSummaryFeatImplJSonFormat.KEY_ON_DEMAND, False
        ):
            return (
                True,
                idx_dict.get(LocalSummaryFeatImplJSonFormat.KEY_ON_DEMAND_PARAMS, None),
            )
        return False, None


class GlobalGroupedBarChartJSonDatatableFormat(TextCustomExplanationFormat):
    """Global grouped bar chart JSon (index file) and datatable (data files)
    representation.

    """

    mime = MimeType.MIME_JSON_DATATABLE

    COL_X = "x"
    COL_Y_GROUP_1 = "y_group_1"
    COL_Y_GROUP_2 = "y_group_2"

    def __init__(
        self,
        explanation,
        json_data: str = None,
        json_file: str = None,
        persistence: persistences.Persistence | None = None,
    ):
        TextCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=GlobalFeatImpJSonDatatableFormat.validate_data(json_data),
            format_file=json_file,
            persistence=persistence,
        )

    @staticmethod
    def validate_data(json_data):
        return json_data

    @staticmethod
    def serialize_index_file(
        classes: list[str],
        default_class: str = "",
        metrics: list | None = None,
        doc: str = "",
        total_rows: int | None = None,
        data_file_prefix: str = "feature_importance",
        data_file_suffix: str = MimeType.EXT_DATATABLE,
    ) -> tuple[dict, str]:
        return GlobalFeatImpJSonFormat.serialize_index_file(
            classes=classes,
            default_class=default_class,
            metrics=metrics,
            doc=doc,
            total_rows=total_rows,
            data_file_prefix=data_file_prefix,
            data_file_suffix=data_file_suffix,
        )

    @classmethod
    def load_index_file(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        mime: str = MimeType.MIME_JSON_DATATABLE,
    ) -> dict:
        #  index file
        idx_path = persistence.get_explanation_file_path(explanation_type, mime)
        return persistence.store.load_json(idx_path)

    def add_data_frame(self, format_data: dt.Frame, file_name: str | None = None):
        self._pre_add_data(format_data=format_data, file_name=file_name)

        persist = self.explanation.explainer.persistence
        persist.store.make_dir(
            persist.get_explanation_dir_path(
                self.explanation.explanation_type(), self.mime
            )
        )
        self._add_frame_to_store(
            path=persist.get_explanation_file_path(
                self.explanation.explanation_type(),
                self.mime,
                explanation_file=file_name,
            ),
            format_data=format_data,
            persistence=persist.store,
        )

    def get_data(self, file_name: str | None = None):
        if not file_name or file_name == self.index_file_name:
            return TextCustomExplanationFormat.get_data(self, file_name)
        elif file_name.endswith(MimeType.EXT_DATATABLE):
            persist = self.explanation.explainer.persistence
            return dt.fread(
                persist.get_explanation_file_path(
                    self.explanation.explanation_type(),
                    GlobalFeatImpJSonDatatableFormat.mime,
                    file_name,
                )
            )
        else:
            return TextCustomExplanationFormat.get_data(self, file_name)

    @classmethod
    def is_paged(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
    ) -> bool:
        return False


class GlobalNlpLocoJSonFormat(TextCustomExplanationFormat, GrammarOfMliFormat):
    """Representation of global feature importance explanation as JSon.

    JSon representation index file example:

    .. code-block:: text

        {
            "files": {
                "red_class": "feature_importance_class_0.json"
                "green_class": "feature_importance_class_1.json"
                "blue_class": "feature_importance_class_2.json"
                ...
            },
            "filters": [
                {
                    "type": "text_features",
                    "name": "TEXT FEATURES",
                    "description": "Model text features",
                    "values": ["description", "review"]
                }
            ],
            "metrics": [{"R2": 0.96}, {"RMSE": 0.03}],
            "total_rows": 592,
        }

    JSon representation data file example:

    .. code-block:: text

        {
            data: [
                {
                    label: str,
                    value: num,
                    scope: str,
                }+
            ]
            bias: num
        }

    Where:

    * ``label`` is feature name
    * ``value`` is feature importance
    * ``scope`` is either ``local`` or ``global``

    """

    mime = MimeType.MIME_JSON

    KEY_DESCRIPTION = "description"
    KEY_FILTERS = "filters"
    KEY_LABEL = "label"
    KEY_NAME = "name"
    KEY_TYPE = "type"
    KEY_VALUE = "value"
    KEY_VALUES = "values"

    FILTER_TYPE_TEXT_FEATURES = "text_feature"

    def __init__(
        self,
        explanation,
        json_data: str = None,
        json_file: str = None,
        persistence: persistences.Persistence | None = None,
    ):
        TextCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=GlobalFeatImpJSonFormat.validate_data(json_data),
            format_file=json_file,
            persistence=persistence,
        )

    @staticmethod
    def validate_data(json_data: str) -> str:
        return json_data

    @staticmethod
    def serialize_index_file(
        classes: list[str],
        default_class: str = "",
        filters: list | None = None,
        metrics: list | None = None,
        keywords: list | None = None,
        doc: str = "",
        total_rows: int | None = None,
        data_file_prefix: str = "feature_importance",
        data_file_suffix: str = "json",
    ) -> tuple[dict, str]:
        """JSon index file serialization to string.

        Parameters
        ----------
        classes: list
          Classes.
        default_class: str
          Class to be shown as default (the first one) e.g. the class of interest in
          case of binomial experiment interpretation.
        filters: list
          Optional list of per-filter items used to filter data entries.
        metrics: list
          Optional list of PD related metrics e.g. ``[{"RMSE": 0.02}, {"SD": 3.1}]``.
        keywords : list[str]
          Optional list of keywords indicating representation features, properties
          and aspects.
        doc: str
          Documentation.
        total_rows: int
          Total number of rows (which can be used for pagination).
        data_file_prefix: str
          Prefix for data file names.
        data_file_suffix: str
          Suffix for data file names.

        Returns
        -------
        Tuple[dict, str]:
          Dictionary with mapping of classes to file names AND JSon serialization
          (as string).

        """

        pdj = PartialDependenceJSonFormat
        index_dict: dict = dict()
        index_dict[pdj.KEY_FILES] = dict()
        for i_c, cls in enumerate(classes):
            index_dict[pdj.KEY_FILES][cls] = (
                f"{data_file_prefix}_class_{i_c}.{data_file_suffix}"
            )
        index_dict[GlobalNlpLocoJSonFormat.KEY_FILTERS] = [] if not filters else filters

        TextCustomExplanationFormat.set_index_commons(
            index_dict=index_dict,
            classes=classes,
            default_class=default_class,
            metrics=metrics,
            doc=doc,
            total_rows=total_rows,
            keywords=keywords,
        )

        return index_dict, json.dumps(index_dict, indent=4)

    @staticmethod
    def serialize_data_file(
        feature_importances: dict[str, float],
        scope: str = "global",
        bias: float | None = None,
    ) -> str:
        """JSon data file serialization to string.

        Parameters
        ----------
        feature_importances: dict
          Feature importances as dictionary of feature name to importance.
        scope: str
          ``global`` or ``local``.
        bias: optional str
          Bias value.

        Returns
        -------
        str:
          Data file serialization.

        """
        if not feature_importances:
            raise ValueError(
                "At least one feature importance must provided to serialize global "
                "feature importance JSon index file"
            )
        data_dict: dict = dict()
        data = list()
        for feature in feature_importances:
            data.append(
                {
                    GlobalFeatImpJSonFormat.KEY_LABEL: feature,
                    GlobalFeatImpJSonFormat.KEY_VALUE: feature_importances[feature],
                    GlobalFeatImpJSonFormat.KEY_SCOPE: scope,
                }
            )
        data_dict[GlobalFeatImpJSonFormat.KEY_DATA] = data
        if bias:
            data_dict[GlobalFeatImpJSonFormat.KEY_BIAS] = bias
        return json.dumps(data_dict)

    @staticmethod
    def from_dataframe_to_json(frame: dt.Frame, bias_col: str = None) -> str:
        featimp_dict: dict = dict()

        # the bias value should be stored differently in the output json
        bias_value = None
        if (
            bias_col
            and bias_col in frame[GlobalFeatImpDatatableFormat.COL_NAME].to_list()[0]
        ):
            bias_value = frame[
                dtf[GlobalFeatImpDatatableFormat.COL_NAME] == bias_col,
                dtf[GlobalFeatImpDatatableFormat.COL_IMPORTANCE],
            ].to_list()[0][0]
            del frame[dtf[GlobalFeatImpDatatableFormat.COL_NAME] == bias_col, :]

        for row in range(frame.shape[0]):
            featimp_dict[frame[row, GlobalFeatImpJSonDatatableFormat.COL_NAME]] = frame[
                row, GlobalFeatImpJSonDatatableFormat.COL_IMPORTANCE
            ]
        return GlobalFeatImpJSonFormat.serialize_data_file(
            feature_importances=featimp_dict, bias=bias_value
        )

    @staticmethod
    def from_json_datatable(
        json_dt_format: GlobalFeatImpJSonDatatableFormat, bias_col: str = None
    ) -> "GlobalFeatImpJSonFormat":
        if json_dt_format:
            # index file
            jd_idx_str: str = json_dt_format.get_data(
                file_name=(
                    f"{GlobalFeatImpJSonFormat.FILE_PREFIX_EXPLANATION_IDX}"
                    f"{MimeType.EXT_JSON}"
                )
            )
            jd_idx_dict: dict = json.loads(jd_idx_str)
            idx_dict: dict = dict()
            idx_dict[GlobalFeatImpJSonFormat.KEY_FILES] = dict()
            idx_dict[GlobalFeatImpJSonFormat.KEY_METRICS] = (
                jd_idx_dict[GlobalFeatImpJSonFormat.KEY_METRICS].copy()
                if GlobalFeatImpJSonFormat.KEY_METRICS in jd_idx_dict
                else []
            )
            idx_dict[GlobalFeatImpJSonFormat.KEY_DOC] = (
                jd_idx_dict[GlobalFeatImpJSonFormat.KEY_DOC]
                if GlobalFeatImpJSonFormat.KEY_DOC in jd_idx_dict
                else ""
            )
            if GlobalFeatImpJSonFormat.KEY_TOTAL_ROWS in jd_idx_dict:
                idx_dict[GlobalFeatImpJSonFormat.KEY_TOTAL_ROWS] = jd_idx_dict[
                    GlobalFeatImpJSonFormat.KEY_TOTAL_ROWS
                ]
            for clazz in jd_idx_dict[GlobalFeatImpJSonFormat.KEY_FILES]:
                idx_dict[GlobalFeatImpJSonFormat.KEY_FILES][clazz] = jd_idx_dict[
                    GlobalFeatImpJSonFormat.KEY_FILES
                ][clazz].replace(f".{MimeType.EXT_DATATABLE}", f".{MimeType.EXT_JSON}")

            result = GlobalFeatImpJSonFormat(
                explanation=json_dt_format.explanation,
                json_data=json.dumps(idx_dict, indent=4),
            )

            # data files
            for clazz in jd_idx_dict[GlobalFeatImpJSonFormat.KEY_FILES]:
                data_file: str = jd_idx_dict[GlobalFeatImpJSonFormat.KEY_FILES][clazz]
                frame: dt.Frame = json_dt_format.get_data(data_file)
                data_file_str = GlobalFeatImpJSonFormat.from_dataframe_to_json(
                    frame=frame, bias_col=bias_col
                )
                result.add_data(
                    format_data=data_file_str,
                    file_name=data_file.replace(
                        f".{MimeType.EXT_DATATABLE}", f".{MimeType.EXT_JSON}"
                    ),
                )

            return result

        raise ValueError("Valid and non-empty representation must be provided")

    @classmethod
    def load_index_file(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        mime: str = MimeType.MIME_JSON,
    ) -> dict:
        """Load index file and check parameters.

        Returns
        -------
        dict:
          Index file as dictionary.

        """
        #  index file
        idx_path = persistence.get_explanation_file_path(explanation_type, mime)
        return persistence.store.load_json(idx_path)

    @classmethod
    def is_paged(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
    ) -> bool:
        del persistence
        del explanation_type

        return True

    @classmethod
    def get_page(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        page_offset: int,
        page_size: int,
        result_format: str,
        explanation_filter: list[FilterEntry],
    ) -> str:
        if not GlobalNlpLocoJSonFormat.is_paged(
            persistence=persistence, explanation_type=explanation_type
        ):
            raise errors.MliError(
                f"{cls.__name__} representation does not support paged explanations"
            )
        gfc = GlobalNlpLocoJSonFormat
        if not explanation_filter:
            raise ValueError(f"Filter parameter '{gfc.FILTER_CLASS}' is required")
        e_filter: dict = dict()
        for i in explanation_filter:
            e_filter[i.filter_by] = i.value
        filter_cls: str = e_filter.get(gfc.FILTER_CLASS, "")
        if not filter_cls:
            raise ValueError(f"Filter parameter '{gfc.FILTER_CLASS}' is required")
        filter_text_feature: str = e_filter.get(gfc.FILTER_TYPE_TEXT_FEATURES, "")

        # find frame using index file
        idx_dict = GlobalNlpLocoJSonFormat.load_index_file(
            persistence=persistence, explanation_type=explanation_type
        )
        if filter_cls not in idx_dict[gfc.KEY_FILES]:
            raise ValueError(
                f"Filter parameter '{gfc.FILTER_CLASS}' parameter "
                f"'{filter_cls}' is not available in the "
                f"representation ({list(idx_dict[gfc.KEY_FILES].keys())})"
            )

        json_path: str = ""
        try:
            json_file: str = idx_dict[gfc.KEY_FILES][filter_cls]
            json_path = persistence.get_explanation_file_path(
                explanation_type=explanation_type,
                explanation_format=MimeType.MIME_JSON,
                explanation_file=json_file,
            )
            json_dict = persistence.store.load_json(json_path)
            data: list = json_dict[GlobalNlpLocoJSonFormat.KEY_DATA]
            if filter_text_feature:
                filtered_data = [
                    t
                    for t in data
                    if t.get(GlobalNlpLocoJSonFormat.KEY_LABEL, "").startswith(
                        f"{filter_text_feature}("
                    )
                ]
                if len(filtered_data) > 0:
                    data = filtered_data
            if page_offset < len(data):
                data_json_page = data[page_offset : page_offset + page_size]
                json_dict[GlobalNlpLocoJSonFormat.KEY_DATA] = data_json_page
                return json.dumps(json_dict)
            else:
                if not page_offset:
                    return json.dumps({GlobalNlpLocoJSonFormat.KEY_DATA: []})
                # raise error only for pages with offset > 0
                raise errors.MliError(
                    f"Unable to load {cls.__name__} explanation for class {filter_cls} "
                    f"and page {page_offset}[{page_size}] as page is out of range "
                    f"(frame has {len(data)} rows)"
                )
        except Exception as ex:
            raise errors.MliError(
                f"Unable to load {cls.__name__} explanation for class '{filter_cls}' "
                f"and page {page_offset}[{page_size}] from {json_path}: {ex}\n"
                f"{traceback.format_exc()}"
            )

    @classmethod
    def get_global_explanation(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
    ) -> str:
        """Get global feature importance explanation.

        Parameters
        ----------
        persistence:
          Persistence object initialized for explainer/MLI run.
        explanation_type: str
          Explanation type ~ explanation ID.

        """
        # persistence and explanation type is checked by caller
        idx_dict = GlobalFeatImpJSonFormat.load_index_file(
            persistence, explanation_type
        )

        filter_cls = next(iter(idx_dict[cls.KEY_FILES]))
        file_path: str = ""
        try:
            file_path = idx_dict[cls.KEY_FILES][filter_cls]
            file_path = persistence.get_explanation_file_path(
                explanation_type=explanation_type,
                explanation_format=cls.mime,
                explanation_file=file_path,
            )
            explanation_dict = persistence.store.load_json(file_path)
            return json.dumps(explanation_dict)
        except Exception as ex:
            raise RuntimeError(
                f"Unable to load global feature importance "
                f"from {file_path}: {ex}\n{traceback.format_exc()}"
            )


class GlobalScatterPlotJSonFormat(GlobalFeatImpJSonFormat, GrammarOfMliFormat):
    def __init__(
        self,
        explanation,
        json_data: str = None,
        json_file: str = None,
        persistence: persistences.Persistence | None = None,
    ):
        GlobalFeatImpJSonFormat.__init__(
            self,
            explanation=explanation,
            json_data=json_data,
            json_file=json_file,
            persistence=persistence,
        )


class GlobalLinePlotJSonFormat(GlobalFeatImpJSonFormat, GrammarOfMliFormat):
    def __init__(
        self,
        explanation,
        json_data: str = None,
        json_file: str = None,
        persistence: persistences.Persistence | None = None,
    ):
        GlobalFeatImpJSonFormat.__init__(
            self,
            explanation=explanation,
            json_data=json_data,
            json_file=json_file,
            persistence=persistence,
        )


class GlobalDtJSonFormat(TextCustomExplanationFormat, GrammarOfMliFormat):
    """Representation of decision tree as JSon.

    JSon representation index file example:

    .. code-block:: text

        {
            "files": {
                "red_class": "dt_class_0.json"
                "green_class": "dt_class_1.json"
                "blue_class": "dt_class_2.json"
                ...
            },
            "metrics": [
              {"Training RMSE": 0.96},
              {"CV RMSE": 0.97},
              {"NFolds": 3},
              {"R2": 0.96}
            ]
        }

    JSon representation data file example:

    .. code-block:: text
       :linenos:

       {
            data: [
                {
                  key: str,
                  name: str,
                  parent: str,
                  edge_in: str,
                  edge_weight: num,
                  leaf_path: bool,
                  total_weight: num,
                  weight: num,
                }+
            ]
       }

    """

    mime = MimeType.MIME_JSON

    KEY_KEY = "key"
    KEY_NAME = "name"
    KEY_EDGE_IN = "edge_in"
    KEY_EDGE_WEIGHT = "edge_weight"
    KEY_TOTAL_WEIGHT = "total_weight"
    KEY_WEIGHT = "weight"
    KEY_LEAF_PATH = "leaf_path"
    KEY_CHILDREN = "children"
    KEY_PARENT = "parent"

    class TreeNode:
        def __init__(
            self,
            name: str,
            parent: Any | None,
            edge_in: str | None,
            edge_weight: float | None,
            total_weight: float | None,
            weight: float | None,
            leaf_path: bool = False,
            key: str = "0",
        ):
            """Tree node use to build object representation.

            Parameters
            ----------
            name: str
              Node name.
            parent: TreeNode | None
              Parent node, ``None`` if this node is root.
            edge_in: str | None
              Parent edge label.
            edge_weight: float
              Parent edge weight (% of rows in a node)
            total_weight: float
              Total weight (Total # of rows in root node).
            weight: float
               Weight in node (# of rows in a node).
            leaf_path: bool
              ``True`` of part of path, else ``False``.

            """
            self.key = key
            self.name = name
            self.parent = parent
            self.edge_in = edge_in
            self.edge_weight = edge_weight
            self.leaf_path = leaf_path
            self.children = list()
            self.total_weight = total_weight
            self.weight = weight

            if parent:
                parent.children.append(self)

        def to_dict(self) -> dict:
            result: dict = dict()
            result[GlobalDtJSonFormat.KEY_KEY] = self.key
            result[GlobalDtJSonFormat.KEY_NAME] = self.name
            result[GlobalDtJSonFormat.KEY_PARENT] = (
                self.parent.key if self.parent else None
            )
            result[GlobalDtJSonFormat.KEY_EDGE_IN] = self.edge_in
            result[GlobalDtJSonFormat.KEY_EDGE_WEIGHT] = self.edge_weight
            result[GlobalDtJSonFormat.KEY_TOTAL_WEIGHT] = self.total_weight
            result[GlobalDtJSonFormat.KEY_WEIGHT] = self.weight
            result[GlobalDtJSonFormat.KEY_LEAF_PATH] = self.leaf_path

            return result

    def __init__(
        self,
        explanation,
        json_data: str = None,
        json_file: str = None,
        persistence: persistences.Persistence | None = None,
    ):
        TextCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=GlobalFeatImpJSonFormat.validate_data(json_data),
            format_file=json_file,
            persistence=persistence,
        )

    @staticmethod
    def serialize_index_file(
        classes: list[str],
        default_class: str = "",
        metrics: list | dict | None = None,
        doc: str = "",
        data_file_prefix: str = "dt",
        data_file_suffix: str = "json",
    ) -> tuple[dict, str]:
        """JSon index file serialization to string.

        Parameters
        ----------
        classes: list
          Classes.
        default_class: str
          Class to be shown as default (the first one) e.g. the class of interest in
          case of binomial experiment interpretation.
        metrics: list
          Optional list of PD related metrics e.g. ``[{"RMSE": 0.02}, {"SD": 3.1}]``
          in case of binomial/regression or dictionary (per class key, metrics list
          as value) in case of multinomial.
        doc: str
          Documentation.
        data_file_prefix: str
          Prefix for data file names.
        data_file_suffix: str
          Suffix for data file names.

        Returns
        -------
        Tuple[dict, str]:
          Dictionary with mapping of classes to file names AND JSon serialization
          (as string).

        """
        return GlobalFeatImpJSonFormat.serialize_index_file(
            classes=classes,
            default_class=default_class,
            metrics=metrics,
            doc=doc,
            data_file_prefix=data_file_prefix,
            data_file_suffix=data_file_suffix,
        )

    @staticmethod
    def serialize_data_file(dt_root_node) -> str:
        """JSon data file serialization to string.

        Parameters
        ----------
        dt_root_node: TreeNode
          Object representation root node.

        Returns
        -------
        str:
          Data file serialization.

        """
        gom_dict = GlobalDtJSonFormat._object_to_data_file(dt_root_node)
        return json.dumps(gom_dict)

    @classmethod
    def load_index_file(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        mime: str = MimeType.MIME_JSON,
    ) -> dict:
        #  index file
        idx_path = persistence.get_explanation_file_path(explanation_type, mime)
        return persistence.store.load_json(idx_path)

    @staticmethod
    def _object_to_data_file(tree) -> dict:
        """Depth-first walk through the decision tree."""
        result: dict = dict()
        result[GlobalDtJSonFormat.KEY_DATA] = list()
        if tree:
            GlobalDtJSonFormat._object_to_data_file_down(
                node=tree, data=result[GlobalDtJSonFormat.KEY_DATA]
            )
        return result

    @staticmethod
    def _object_to_data_file_down(node: TreeNode, data: list):
        data.append(node.to_dict())
        if node.children:
            for child in node.children:
                GlobalDtJSonFormat._object_to_data_file_down(child, data)

    @staticmethod
    def validate_data(json_data: str) -> str:
        return json_data


class LocalDtJSonFormat(TextCustomExplanationFormat):
    """Local representation of decision tree as JSon.

    JSon representation index file example:

    .. code-block:: text

        {
            "files": {
                "red_class": "dt_class_0.json"
                "green_class": "dt_class_1.json"
                "blue_class": "dt_class_2.json"
                ...
            },
            "metrics": [
              {"Training RMSE": 0.96},
              {"CV RMSE": 0.97},
              {"NFolds": 3},
              {"R2": 0.96}
            ]
        }

    JSon representation data file example:

    .. code-block:: text
       :linenos:

       {
            data: [
                {
                  key: str,
                  name: str,
                  parent: str,
                  edge_in: str,
                  edge_weight: num,
                  leaf_path: bool,
                  total_weight: num,
                  weight: num
                }+
            ]
       }

    or (if on demand e.g. in case of sampled dataset):

    .. code-block:: text

        {
            "on_demand": true
            "on_demand_parameters": ...
        }

    Remarks:

    * ``leaf_path`` ... ``true`` if local path (hint in the leaf defines path to
       the root), else global explanation. In other words return the whole tree
       with *leaf selected*.
    * ``on_demand`` ... ``true`` if there is no cached ICE and it must be computed.

    """

    mime = MimeType.MIME_JSON

    def __init__(
        self,
        explanation,
        json_data: str = None,
        json_file: str = None,
        persistence: persistences.Persistence | None = None,
    ):
        TextCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=IceJsonDatatableFormat.validate_data(json_data),
            format_file=json_file,
            persistence=persistence,
        )

    @staticmethod
    def validate_data(json_data: str) -> str:
        return json_data

    @staticmethod
    def serialize_index_file(
        classes: list[str],
        default_class: str = "",
        metrics: list | None = None,
        doc: str = "",
        data_file_prefix: str = "dt",
        data_file_suffix: str = "json",
    ) -> tuple[dict, str]:
        return GlobalDtJSonFormat.serialize_index_file(
            classes=classes,
            default_class=default_class,
            metrics=metrics,
            doc=doc,
            data_file_prefix=data_file_prefix,
            data_file_suffix=data_file_suffix,
        )

    @staticmethod
    def serialize_on_demand_index_file(on_demand_params: dict) -> str:
        return json.dumps(
            {
                IceJsonDatatableFormat.KEY_ON_DEMAND: True,
                IceJsonDatatableFormat.KEY_ON_DEMAND_PARAMS: on_demand_params,
            }
        )

    @classmethod
    def is_on_demand(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
    ) -> tuple[bool, dict | None]:
        idx_dict = IceJsonDatatableFormat.load_index_file(
            persistence=persistence,
            explanation_type=explanation_type,
            mime=LocalDtJSonFormat.mime,
        )

        if idx_dict and idx_dict.get(IceJsonDatatableFormat.KEY_ON_DEMAND, False):
            return (
                True,
                idx_dict.get(IceJsonDatatableFormat.KEY_ON_DEMAND_PARAMS, None),
            )
        return False, None

    @staticmethod
    def dt_path_to_node_key(path: str) -> str:
        # convert path to node key > load representation > set leaf to true
        key = "0"
        for step in path:
            if "L" == step:
                key = f"{key}.0"
            else:
                key = f"{key}.1"
        return key

    @staticmethod
    def dt_set_tree_path(key: str, tree: dict):
        path = "0"
        keys_to_set = [path]
        if len(key) > 1:
            directions: list = key.split(".")
            for direction in directions[1:]:
                path = f"{path}.{direction}"
                keys_to_set.append(path)

        for path_key in keys_to_set:
            for node in tree["data"]:
                if path_key == node.get(GlobalDtJSonFormat.KEY_KEY, None):
                    node[GlobalDtJSonFormat.KEY_LEAF_PATH] = True

        return tree

    @classmethod
    def get_local_explanation(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        row: int,
        explanation_filter: list[FilterEntry],
        **extra_params,
    ) -> str:
        """Get local DT explanation.

        Parameters
        ----------
        persistence:
          Persistence object initialized for explainer/MLI run.
        explanation_type: str
          Explanation type ~ explanation ID.
        row: int
          Local explanation to be provided for given row.
        explanation_filter: list[FilterEntry]
          Required filter entries:
            `feature`
            `class`

        Returns
        -------
        str:
          JSon representation of the local explanation.

        JSon DT representation:

        .. code-block:: text
           :linenos:

           {
                data: [
                    {
                      key: str,
                      name: str,
                      parent: str,
                      edge_in: str,
                      edge_weight: num,
                      leaf_path: bool,
                      total_weight: num,
                      weight: num,
                    }+
                ]
           }

        """
        raise errors.MliError(
            f"{cls.__name__} representation does not support local explanations"
        )


class TextFormat(TextCustomExplanationFormat):
    """Text representation."""

    mime = MimeType.MIME_TEXT

    def __init__(
        self,
        explanation,
        format_data: str,
        persistence: persistences.Persistence | None = None,
    ):
        TextCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=format_data,
            format_file=None,
            persistence=persistence,
        )

    @staticmethod
    def validate_data(dt_data: dt.Frame):
        return dt_data


class DiaTextFormat(TextCustomExplanationFormat):
    """Disparate Impact Analysis (DIA) text representation."""

    mime = MimeType.MIME_TEXT

    def __init__(
        self,
        explanation,
        format_data: str,
        persistence: persistences.Persistence | None = None,
    ):
        TextCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=format_data,
            format_file=None,
            persistence=persistence,
        )

    @staticmethod
    def validate_data(dt_data: dt.Frame):
        return dt_data


class SaTextFormat(TextCustomExplanationFormat):
    """Sensitivity Analysis (SA) text representation."""

    mime = MimeType.MIME_TEXT

    def __init__(
        self,
        explanation,
        format_data: str,
        persistence: persistences.Persistence | None = None,
    ):
        TextCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=format_data,
            format_file=None,
            persistence=persistence,
        )

    @staticmethod
    def validate_data(dt_data: dt.Frame):
        return dt_data


class DocxFormat(ExplanationFormat, GrammarOfMliFormat):
    """Open ``docx`` document."""

    mime = MimeType.MIME_DOCX

    def __init__(
        self,
        explanation,
        format_file: str,
        persistence: persistences.Persistence | None = None,
    ):
        ExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=None,
            format_file=format_file,
            file_extension=MimeType.ext_for_mime(self.mime),
            persistence=persistence,
        )

    @staticmethod
    def validate_data(dt_data: dt.Frame):
        return dt_data


class MarkdownFormat(ExplanationFormat, GrammarOfMliFormat):
    """Markdown representation (text and images)."""

    mime = MimeType.MIME_MARKDOWN

    def __init__(
        self,
        explanation,
        format_file: str,
        extra_format_files: list | None = None,
        persistence: persistences.Persistence | None = None,
    ):
        ExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=None,
            format_file=format_file,
            extra_format_files=extra_format_files,
            file_extension=MimeType.ext_for_mime(self.mime),
            persistence=persistence,
        )

    @staticmethod
    def validate_data(dt_data: dt.Frame):
        return dt_data


class EvalStudioMarkdownFormat(MarkdownFormat):
    """EvalStudio Markdown representation (text and images)."""

    mime = MimeType.MIME_EVALSTUDIO_MARKDOWN

    def __init__(
        self,
        explanation,
        format_file: str,
        extra_format_files: list | None = None,
        persistence: persistences.Persistence | None = None,
    ):
        MarkdownFormat.__init__(
            self,
            explanation=explanation,
            format_file=format_file,
            extra_format_files=extra_format_files,
            persistence=persistence,
        )


class ModelValidationResultArchiveFormat(ExplanationFormat):
    """Model Validation test result archived in a ZIP."""

    mime = MimeType.MIME_ZIP

    def __init__(
        self,
        explanation,
        mv_test_type: str | Any,
        mv_test_name: str,
        mv_test_id: str,
        mv_test_results,  # h2o_mv.core.mv_test.MVTestResult
        mv_test_settings,  # h2o_mv.core.mv_test.MVTestSettings
        mv_test_artifacts: dict,  # list[h2o_mv.core.mv_test.ArtifactInfo]
        mv_test_log,  # h2o_mv.core.mv_test.MVTestLog
        mv_client=None,  # MV DBs (obj & RDBMs) access
        persistence: persistences.Persistence | None = None,
        logger=None,
    ):
        ExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data="archive on the way",
            format_file=None,
            file_extension=MimeType.ext_for_mime(WorkDirArchiveZipFormat.mime),
            persistence=persistence,
        )

        mv_result_dir_path = self._export_mvr_to_work(
            mv_test_type=(
                mv_test_type
                if isinstance(mv_test_type, str)
                else f"{type(mv_test_type)}"
            ),
            mv_test_name=mv_test_name,
            mv_test_id=mv_test_id,
            mv_test_results=mv_test_results,
            mv_test_settings=mv_test_settings,
            mv_test_artifacts=mv_test_artifacts,
            mv_test_log=mv_test_log,
            mv_client=mv_client,
            logger=logger,
        )

        persistence = self.explanation.explainer.persistence
        archive_dir = persistence.get_explanation_dir_path(
            explanation_type=self.explanation.explanation_type(),
            explanation_format=self.mime,
        )
        persistence.store.make_dir(archive_dir)
        archive_path: str = persistence.get_explanation_file_path(
            explanation_type=self.explanation.explanation_type(),
            explanation_format=self.mime,
        )

        self._persistence.delete_file(archive_path)
        self._persistence.make_dir_zip_archive(
            src_key=mv_result_dir_path,
            zip_key=archive_path,
            file_filter=lambda x: False,
        )

        # purge exported MV result
        if os.path.isdir(mv_result_dir_path):
            persistences.Persistence.delete_temp_dir(mv_result_dir_path)

    def _export_mvr_to_work(
        self,
        mv_test_type: str,
        mv_test_name: str,
        mv_test_id: str,
        mv_test_results,
        mv_test_settings,
        mv_test_artifacts: dict,
        mv_test_log,
        mv_client,
        logger=None,
    ):
        # store MV result to the work/mv-result directory
        persistence = self.explanation.explainer.persistence
        mv_persistence = mv_adapter.MvResultPersistence(
            target_dir_path=persistence.get_explainer_working_dir(),
            mv_client=mv_client,
            logger=logger,
        )
        (export_dir_path, _) = mv_persistence.export_mv_test(
            mv_test_type=mv_test_type,
            mv_test_name=mv_test_name,
            mv_test_id=mv_test_id,
            mv_test_results=mv_test_results,
            mv_test_settings=mv_test_settings,
            mv_test_artifacts=mv_test_artifacts,
            mv_test_log=mv_test_log,
        )

        return export_dir_path


class WorkDirArchiveZipFormat(ExplanationFormat, GrammarOfMliFormat):
    """Working directory ZIP archive representation. Just instantiate this class,
    and it will create the ZIP representation (no need to add files/data). Note
    that the archive is created exactly on the time of instantiation.

    """

    mime = MimeType.MIME_ZIP

    def __init__(
        self,
        explanation,
        file_filter=lambda x: False,
        persistence: persistences.Persistence | None = None,
    ):
        ExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data="archive on the way",
            format_file=None,
            file_extension=MimeType.ext_for_mime(WorkDirArchiveZipFormat.mime),
            persistence=persistence,
        )

        persistence = self.explanation.explainer.persistence
        archive_dir = persistence.get_explanation_dir_path(
            explanation_type=self.explanation.explanation_type(),
            explanation_format=self.mime,
        )
        persistence.store.make_dir(archive_dir)
        archive_path: str = persistence.get_explanation_file_path(
            explanation_type=self.explanation.explanation_type(),
            explanation_format=self.mime,
        )

        self._persistence.delete_file(archive_path)
        self._persistence.make_dir_zip_archive(
            src_key=persistence.get_explainer_working_dir(),
            zip_key=archive_path,
            file_filter=file_filter,
        )


class CustomArchiveZipFormat(ExplanationFormat, GrammarOfMliFormat):
    """Custom ZIP archive representation."""

    mime = MimeType.MIME_ZIP

    def __init__(
        self,
        explanation,
        format_file: str,
        persistence: persistences.Persistence | None = None,
    ):
        ExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data="",
            format_file=format_file,
            file_extension=MimeType.ext_for_mime(WorkDirArchiveZipFormat.mime),
            persistence=persistence,
        )


class LlmLeaderboardJSonFormat(abc.ABC):
    mime = MimeType.MIME_JSON

    KEY_ALL_METRICS = "ALL_METRICS"  # symbolic / virtual key for all metrics


class LlmHeatmapLeaderboardJSonFormat(
    LlmLeaderboardJSonFormat, TextCustomExplanationFormat
):
    """Representation of LLM Heatmap Leaderboard explanation as JSon.

    JSon representation index file example:

    .. code-block:: text

        {
            "files": {
                "ragas": "leaderboard_0.json"
                "answer_relevance": "leaderboard_1.json"
                ...
                "ALL_METRICS": "leaderboard_n.json"
            },
            ...
        }

    JSon representation data file example:

    .. code-block:: text

        {
            "data": {
                "h2oai/h2ogpt-4096-llama2-70b-chat": {
                    "answer_similarity": 1
                },
                "h2oai/h2ogpt-4096-llama2-70b-chat-4bit": {
                    "answer_similarity": 1
                },
                ...
                "gpt-4-32k-0613": {
                    "answer_similarity": 1
                }
            },
            "eda": {
                ...
            }
        }

    """

    mime = MimeType.MIME_JSON

    KEY_DEFAULT_METRIC = "default_metric"
    KEY_EDA = "eda"

    def __init__(
        self,
        explanation,
        json_data: str = None,
        json_file: str = None,
        persistence: persistences.Persistence | None = None,
    ):
        TextCustomExplanationFormat.__init__(
            self,
            explanation=explanation,
            format_data=GlobalFeatImpJSonFormat.validate_data(json_data),
            format_file=json_file,
            persistence=persistence,
        )

    @staticmethod
    def validate_data(json_data: str) -> str:
        return json_data

    @staticmethod
    def serialize_index_file(
        metrics: list[str],
        default_metric: str = "",
        eda: dict | None = None,
        doc: str = "",
        data_file_prefix: str = "leaderboard",
        data_file_suffix: str = "json",
    ) -> tuple[dict, str]:
        """JSon index file serialization to string.

        Parameters
        ----------
        metrics: list
          Metrics.
        default_metric: str
          Metric to be shown as default (the first one).
        eda: dict
            EDA data.
        doc: str
          Documentation.
        data_file_prefix: str
          Prefix for data file names.
        data_file_suffix: str
          Suffix for data file names.

        Returns
        -------
        Tuple[dict, str]:
          Dictionary with mapping of classes to file names AND JSon serialization
          (as string).

        """

        pdj = PartialDependenceJSonFormat
        index_dict: dict = dict()
        index_dict[pdj.KEY_FILES] = dict()
        for i_c, cls in enumerate(metrics):
            index_dict[pdj.KEY_FILES][cls] = (
                f"{data_file_prefix}_{i_c}.{data_file_suffix}"
            )

        index_dict[ExplanationFormat.KEY_METRICS] = [] if not metrics else metrics
        if default_metric:
            index_dict[LlmHeatmapLeaderboardJSonFormat.KEY_DEFAULT_METRIC] = (
                default_metric
            )
        elif metrics and len(metrics):
            index_dict[LlmHeatmapLeaderboardJSonFormat.KEY_DEFAULT_METRIC] = metrics[0]

        if eda is not None:
            index_dict[LlmHeatmapLeaderboardJSonFormat.KEY_EDA] = eda
        if doc:
            index_dict[ExplanationFormat.KEY_DOC] = doc

        return index_dict, json.dumps(index_dict, indent=4)

    @classmethod
    def load_index_file(
        cls,
        persistence: persistences.ExplainerPersistence,
        explanation_type: str,
        mime: str = MimeType.MIME_JSON,
    ) -> dict:
        """Load index file and check parameters.

        Returns
        -------
        dict:
          Index file as dictionary.

        """
        #  index file
        idx_path = persistence.get_explanation_file_path(explanation_type, mime)
        return persistence.store.load_json(key=idx_path)
