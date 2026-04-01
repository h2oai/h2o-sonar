# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import abc
import base64
import enum
import fnmatch
import getpass
import glob
import json
import math
import os
import pathlib
import re
import shutil
import sys
import tempfile
import time
import traceback
import uuid
import zipfile
from abc import ABC
from abc import abstractmethod
from typing import Any

import numpy
import pandas

from h2o_sonar import errors
from h2o_sonar import loggers as loggers
from h2o_sonar.lib.api import commons


class PersistenceType(enum.Enum):
    file_system = enum.auto()
    in_memory = enum.auto()
    database = enum.auto()


class PersistenceDataType(enum.Enum):
    binary = enum.auto()
    datatable = enum.auto()
    text = enum.auto()
    json = enum.auto()


class Persistence(abc.ABC):
    """Key/value-based persistence API interface provides uniform store-agnostic API
    allowing explainers to use chosen *store type* regardless container runtime
    or technology to store *explainer results* (explanations). It aims to
    enable writing *identical* code regardless explanation data is stored/loaded
    to/from filesystem, memory or DB.

    Interface and implementations are based on *opaque* string keys (which might be
    filesystem paths, dictionary keys or NoSQL database keys) and data types
    (text, binary, ...). On implementation initialization is be set base in-memory
    reference, filesystem path or DB connection information.

    There are the following special types of data which are written to filesystem
    (network or memory) regardless chosen store type:

    - temporary files (explainer work/ directory)
    - log filed (explainer log/ directory)

    Therefore, an explainer *sandbox* is always created on the file-system,
    but it might be located in user specified directory (in case of file-system store)
    or system temp directory (in case of in-memory or database store).

    The persistence API is written with security (barriers) and performance in mind.

    """

    PREFIX_INTERNAL_STORE = "h2o_sonar-of-"

    @staticmethod
    def flush_dir_for_file(file_path: str) -> bool:
        if os.path.isfile(file_path):
            return True
        bft_step = 0.1
        bft = 60 * int(1 / bft_step)
        for _ in range(bft):
            if os.path.isfile(file_path):
                return True
            else:
                dirfd = os.open(os.path.dirname(file_path), os.O_DIRECTORY)
                os.fsync(dirfd)
                os.close(dirfd)
                time.sleep(bft_step)
        return False

    @staticmethod
    def is_binary_file(key: str) -> bool:
        for suffix in [".txt", ".log", ".json", ".md", ".csv", ".xml"]:
            if key.endswith(suffix):
                return True
        return True

    @staticmethod
    def safe_name(key: str) -> str:
        """Encode name to be store (file-sytem) safe (can be decoded if needed)."""
        if key:
            # = padding escaped
            return (
                base64.urlsafe_b64encode(key.encode("utf-8"))
                .decode("utf-8")
                .replace("=", "_")
            )
        return key

    @staticmethod
    def make_key(*args) -> str:
        """Assemble key (path) from the string arguments given to this function
        (equivalent of ``os.path.join()``).

        """
        if len(args):
            return os.path.join(*args)
        return ""

    @staticmethod
    def key_folder(key: str | pathlib.Path) -> str:
        """Get (parent) folder key for given key
        (equivalent of ``os.path.dirname()``).

        """
        return os.path.dirname(Persistence.check_key(key))

    @staticmethod
    def check_key(key: str | pathlib.Path) -> str:
        """Check and fix key."""
        if not key:
            raise ValueError(f"Attempt to use invalid persistence key: '{key}'")
        return str(key)

    @staticmethod
    def make_temp_dir() -> str:
        # TEMP DIRECTORY: operations in the temp directory are required
        # regardless the persistence store location/technology
        return tempfile.mkdtemp()

    @staticmethod
    def make_temp_file(file_name: str) -> str:
        # TEMP DIRECTORY: operations in the temp directory are required
        # regardless the persistence store location/technology
        return os.path.join(tempfile.mkdtemp(), file_name)

    @staticmethod
    def delete_temp_dir(tmp_dir_path: str | pathlib.Path):
        shutil.rmtree(str(tmp_dir_path))

    @property
    def type(self):
        raise NotImplementedError

    def __init__(
        self,
        logger=None,
    ):
        """Persistence constructor."""
        self.internal = None
        self._internal_root_path: str = ""
        self.logger = logger or loggers.SonarPrintLogger()

    def _init_internal_store(self, internal_store=None):
        """Internal store initialization.

        Parameters
        ----------
        internal_store : FilesystemPersistence
          Optional persistence with path to the data directory path in case of
          file-system persistence (directory in system/user temp if not specified
          e.g. in case of in-memory or database persistence).

        """
        if internal_store:
            self.internal = internal_store
            self._internal_root_path = internal_store.base_path
        else:
            # user specified (file-system) or system temp (in-memory, database)
            self._internal_root_path: str = tempfile.mkdtemp(
                prefix=f"{Persistence.PREFIX_INTERNAL_STORE}{getpass.getuser()}-"
            )
            self.internal = FilesystemPersistence(
                self._internal_root_path, logger=self.logger
            )

    def path_to_internal(self, path: str | pathlib.Path) -> str:
        return (
            os.path.join(self._internal_root_path, path[1:] if path[0] == "/" else path)
            if path
            else self._internal_root_path
        )

    def getcwl(self):
        """Get current working location - directory, memory key or DB locator."""
        raise NotImplementedError

    def touch(self, key: str | pathlib.Path):
        raise NotImplementedError

    def exists(self, key: str | pathlib.Path) -> bool:
        raise NotImplementedError

    def is_file(self, key: str | pathlib.Path) -> bool:
        raise NotImplementedError

    def is_dir(self, key: str | pathlib.Path) -> bool:
        raise NotImplementedError

    def is_dir_or_file(self, key: str | pathlib.Path) -> bool:
        return self.is_dir(key) or self.is_file(key)

    def list_dir(self, key: str | pathlib.Path) -> list:
        raise NotImplementedError

    def list_files_by_wildcard(self, key: str | pathlib.Path, wildcard: str) -> list:
        raise NotImplementedError

    def make_dir(self, key: str | pathlib.Path):
        raise NotImplementedError

    def load(
        self,
        key: str | pathlib.Path,
        data_type: PersistenceDataType = PersistenceDataType.binary,
    ) -> Any:
        raise NotImplementedError

    def save(
        self,
        key: str | pathlib.Path,
        data,
        data_type: PersistenceDataType = PersistenceDataType.binary,
    ):
        raise NotImplementedError

    def update(
        self,
        key: str | pathlib.Path,
        data,
        data_type: PersistenceDataType = PersistenceDataType.binary,
    ):
        self.save(key=key, data=data, data_type=data_type)

    def load_json(self, key: str | pathlib.Path) -> dict:
        raise NotImplementedError

    def copy_file(
        self,
        from_key: str | pathlib.Path,
        to_key: str | pathlib.Path,
    ):
        raise NotImplementedError

    def make_dir_zip_archive(
        self, src_key: str, zip_key: str, file_filter=lambda x: False
    ):
        """Make ZIP archive of given source directory.

        Parameters
        ----------
        src_key : str
          Source key (directory path).
        zip_key : str
          ZIP key (ZIP file path).
        file_filter : Callable
          File filter.

        """
        raise NotImplementedError

    def delete_file(self, key: str | pathlib.Path) -> bool:
        raise NotImplementedError

    def delete_tree(self, key: str | pathlib.Path) -> bool:
        raise NotImplementedError

    def delete_dir_contents(self, key: str | pathlib.Path, logger=None):
        raise NotImplementedError

    def delete(self, key: str | pathlib.Path) -> bool:
        raise NotImplementedError


class FilesystemPersistence(Persistence):
    """File-system store persistence."""

    @staticmethod
    def get_default_cwl():
        """Get default current working location when no specified by the user."""
        return os.getcwd()

    @staticmethod
    def flush_dir_for_file(file_path: str) -> bool:
        if os.path.isfile(file_path):
            return True
        bft_step = 0.1
        bft = 60 * int(1 / bft_step)
        for _ in range(bft):
            if os.path.isfile(file_path):
                return True
            else:
                dirfd = os.open(os.path.dirname(file_path), os.O_DIRECTORY)
                os.fsync(dirfd)
                os.close(dirfd)
                time.sleep(bft_step)
        return False

    @property
    def type(self):
        return PersistenceType.file_system

    def __init__(self, base_path: pathlib.Path | str | None = None, logger=None):
        Persistence.__init__(self, logger)
        base_path = base_path or FilesystemPersistence.get_default_cwl()
        self.base_path = Persistence.check_key(base_path)
        self._init_internal_store(internal_store=self)

    def getcwl(self):
        return self.base_path

    def touch(self, key: str | pathlib.Path):
        with open(Persistence.check_key(key), "a"):
            os.utime(key, None)

    def exists(self, key: str | pathlib.Path) -> bool:
        return os.path.exists(Persistence.check_key(key))

    def is_file(self, key: str | pathlib.Path) -> bool:
        return os.path.isfile(Persistence.check_key(key))

    def is_dir(self, key: str | pathlib.Path) -> bool:
        return os.path.isdir(Persistence.check_key(key))

    def list_dir(self, key: str | pathlib.Path) -> list:
        return os.listdir(Persistence.check_key(key))

    def list_files_by_wildcard(self, key: str | pathlib.Path, wildcard: str) -> list:
        key = Persistence.check_key(key)
        if key and self.is_dir(key):
            wild_path = Persistence.make_key(key, wildcard)
            return glob.glob(wild_path)
        return list()

    def make_dir(self, key: str | pathlib.Path):
        os.makedirs(Persistence.check_key(key), exist_ok=True)

    def load(
        self,
        key: str | pathlib.Path,
        data_type: PersistenceDataType | None = None,
    ) -> Any:
        key = Persistence.check_key(key)
        binary_flag: str = ""
        if data_type:
            if PersistenceDataType.binary == data_type:
                binary_flag = "b"
        elif Persistence.is_binary_file(key):
            binary_flag = "b"
        with open(Persistence.check_key(key), f"r{binary_flag}") as text_file:
            return text_file.read()

    def save(
        self,
        key: str | pathlib.Path,
        data,
        data_type: PersistenceDataType = PersistenceDataType.text,
    ):
        with open(
            Persistence.check_key(key),
            (
                "w"
                if data_type in [PersistenceDataType.text, PersistenceDataType.json]
                else "wb"
            ),
        ) as text_file:
            text_file.write(data)

    def update(
        self,
        key: str | pathlib.Path,
        data,
        data_type: PersistenceDataType = PersistenceDataType.binary,
    ):
        with open(Persistence.check_key(key), "r+") as text_file:
            text_file.seek(0)
            text_file.write(data)
            text_file.truncate()

    @staticmethod
    def save_json(
        key: str | pathlib.Path,
        data: dict | list,
        indent: int = 4,
        save_explainer_params=False,
    ) -> dict:
        key = Persistence.check_key(key)
        # ATOMIC write to file ... :-/ tricks
        tmp_name: str = f"{key}.{str(uuid.uuid4())}.tmp"
        if save_explainer_params:
            current_data = None
            if os.path.exists(key):
                with open(key) as json_file:
                    current_data = json.loads(json_file.read())
                    # context manager flush is NOT guaranteed for libs
                    json_file.flush()
                    # fdatasync() (fast, no meta) vs. os.fsync() (integrity)
                    os.fsync(json_file.fileno())
                if current_data:
                    data.update(current_data)
                tmp_name = key
        with open(tmp_name, mode="w") as json_file:
            json_file.write(json.dumps(data, indent=indent, cls=RobustEncoder))
            # context manager flush is NOT guaranteed for libs
            json_file.flush()
            # fdatasync() (fast, no meta) vs. os.fsync() (integrity)
            os.fsync(json_file.fileno())
        # flush tmp file existence
        if FilesystemPersistence.flush_dir_for_file(tmp_name):
            shutil.move(tmp_name, key)
            # flushed rename
            FilesystemPersistence.flush_dir_for_file(key)
            return data
        else:
            raise RuntimeError(
                f"Unable to atomically save JSon entity file by renaming "
                f"'{tmp_name}' (existence flush timeout)"
            )

    def load_json(self, key: str | pathlib.Path) -> dict | list:
        key = Persistence.check_key(key)
        if os.path.isfile(key):
            with open(key) as json_file:
                return json.load(json_file)
        raise errors.MliError(f"Unable to load JSon file - {key} does not exist")

    def copy_file(
        self,
        from_key: str | pathlib.Path,
        to_key: str | pathlib.Path,
    ):
        shutil.copyfile(Persistence.check_key(from_key), Persistence.check_key(to_key))

    def make_dir_zip_archive(
        self,
        src_key: str | pathlib.Path,
        zip_key: str | pathlib.Path,
        file_filter=lambda x: False,
    ):
        """Create ZIP archive of given directory.

        Parameters
        ----------
        src_key: src
          Absolute path to directory to be archived.
        zip_key: src
          ZIP archive path.
        file_filter:
          Function to be used for filtering - it gets relative path from the
          `src_dir_path` as parameter and returns boolean indicating whether to keep
          (`False`) or filter file out (`True`).

        """
        src_key = Persistence.make_key(src_key)
        if not src_key or not self.is_dir(src_key):
            raise ValueError(f"Directory {src_key} to be zipped does not exist")
        src_dir_path = os.path.abspath(src_key)
        src_dir_name = os.path.basename(src_key)
        zip_key = Persistence.make_key(zip_key)

        # scan and filter
        paths = []
        for root, dirs, files in os.walk(src_dir_path):
            for f_name in files:
                path = Persistence.make_key(root, f_name)
                # relative path from the src dir path is used for filtering
                filter_path = path[len(src_dir_path) + 1 :]
                if not file_filter(filter_path):
                    paths.append(path)

        # zip
        zip_root_path = src_dir_path[: -len(src_dir_name)]
        with zipfile.ZipFile(zip_key, "w") as zf:
            for file in paths:
                zip_file_path = file[len(zip_root_path) :]
                zf.write(file, arcname=zip_file_path)

        return zip_key

    def delete_file(self, key: str | pathlib.Path) -> bool:
        key = Persistence.check_key(key)
        if os.path.exists(key):
            os.remove(key)
            return True
        return False

    def delete_tree(self, key: str | pathlib.Path):
        shutil.rmtree(Persistence.check_key(key))

    def delete_dir_contents(self, key: str | pathlib.Path, logger=None):
        for filename in os.listdir(key):
            file_path = Persistence.make_key(key, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as ex:
                logger = logger or self.logger
                logger.warning(
                    f"Interpretation persistence failed to dir file {file_path}: {ex}",
                )

    def delete(self, key: str | pathlib.Path) -> bool:
        return self.delete_tree(key)


class InMemoryPersistence(Persistence):
    """In-memory key-based store persistence."""

    class Directory:
        pass

    DIR = Directory()

    @staticmethod
    def get_default_cwl():
        """Get default current working location when no specified by the user."""
        return os.getcwd()

    @property
    def type(self):
        return PersistenceType.in_memory

    def __init__(self):
        Persistence.__init__(self)
        self._init_internal_store()
        self.memory_store = {}
        self.base_path = InMemoryPersistence.get_default_cwl()

    def __str__(self):
        return str(list(self.memory_store.keys()))

    def getcwl(self):
        return self.base_path

    def touch(self, key: str | pathlib.Path):
        self.memory_store[Persistence.check_key(key)] = None

    def exists(self, key: str | pathlib.Path) -> bool:
        return Persistence.check_key(key) in self.memory_store

    def is_file(self, key: str | pathlib.Path) -> bool:
        return self.exists(Persistence.check_key(key))

    def is_dir(self, key: str | pathlib.Path) -> bool:
        # in-memory directory might be key + instance of a Directory() class
        return self.exists(Persistence.check_key(key))

    def list_dir(self, key: str | pathlib.Path) -> list:
        result = []
        key = Persistence.check_key(key)
        if key and self.memory_store:
            for k in self.memory_store.keys():
                if k.startswith(key) and len(k) > len(key):
                    tail = k[len(key) :]
                    tail = tail[1:] if tail[0] == "/" else tail
                    if "/" not in tail:
                        result.append(tail)
        return result

    def list_files_by_wildcard(self, key: str | pathlib.Path, wildcard: str) -> list:
        result = []
        key = Persistence.check_key(key)
        if key and wildcard is not None and self.memory_store:
            for k in self.memory_store.keys():
                if k.startswith(key) and len(k) > len(key):
                    tail = k[len(key) :]
                    tail = tail[1:] if tail[0] == "/" else tail
                    if fnmatch.fnmatch(tail, wildcard):
                        result.append(k)
        return result

    def make_dir(self, key: str | pathlib.Path):
        # in-memory directory might be key + instance value indicating directory
        self.memory_store[Persistence.check_key(key)] = InMemoryPersistence.DIR

    def load(
        self,
        key: str | pathlib.Path,
        data_type: PersistenceDataType = PersistenceDataType.binary,
    ) -> Any:
        data = self.memory_store.get(Persistence.check_key(key), None)
        if PersistenceType.file_system != self.type and data is None:
            # fallback: try load from internal storage work/ if no value is in memory
            if self.internal.exists(key):
                return self.internal.load(key)

        return data

    def save(
        self,
        key: str | pathlib.Path,
        data,
        data_type: PersistenceDataType = PersistenceDataType.binary,
    ):
        self.memory_store[Persistence.check_key(key)] = data

    def save_json(
        self,
        key: str | pathlib.Path,
        data: dict,
        indent: int = 4,
        save_explainer_params=False,
    ):
        if save_explainer_params:
            if os.path.exists(key):
                current_data = None
                with open(key) as json_file:
                    current_data = json.loads(json_file.read())
                    # context manager flush is NOT guaranteed for libs
                    json_file.flush()
                if current_data:
                    data.update(current_data)
        self.save(
            key=key,
            data=json.dumps(data, indent=indent),
            data_type=PersistenceDataType.json,
        )

    def load_json(self, key: str | pathlib.Path) -> dict:
        return json.loads(self.load(key=key, data_type=PersistenceDataType.json))

    def copy_file(
        self,
        from_key: str | pathlib.Path,
        to_key: str | pathlib.Path,
    ):
        self.save(to_key, self.load(from_key))

    def make_dir_zip_archive(
        self,
        src_key: str | pathlib.Path,
        zip_key: str | pathlib.Path,
        file_filter=lambda x: False,
    ):
        # zip to temp file and then copy to memory
        internal_zip_key = os.path.join(tempfile.mkdtemp(), "archive.zip")
        self.internal.make_dir_zip_archive(
            src_key=src_key,
            zip_key=internal_zip_key,
            file_filter=file_filter,
        )
        self.copy_file(from_key=internal_zip_key, to_key=zip_key)
        self.internal.delete_tree(os.path.dirname(internal_zip_key))

    def delete_file(self, key: str | pathlib.Path) -> bool:
        return self.delete(key)

    def delete_tree(self, key: str | pathlib.Path):
        raise NotImplementedError

    def delete_dir_contents(self, key: str | pathlib.Path, logger=None):
        raise NotImplementedError

    def delete(self, key: str | pathlib.Path) -> bool:
        key = Persistence.check_key(key)
        if self.exists(key):
            return True if self.memory_store.pop(key, None) else False
        return False


class JsonPersistableExplanations(ABC):
    """Interface for classes implementing explanations JSon file persistence.

    Examples
    --------
    .. code-block:: text

         ice = ICE("Step by step ICE loading")
         ice.load_json("cache/ice.json")

         es = ice.explanations()

         es = ICE("On the fly").explain(
           ["Feature"],
           X,
           predict_method=scorer
        ).save_json()

    """

    @property
    def default_json_file_name(self):
        return self._default_json_file_name

    @default_json_file_name.setter
    def default_json_file_name(self, default_file_name=None):
        self._default_json_file_name = (
            self._default_json_file_name
            if default_file_name is None
            else default_file_name
        )

    class PandasJSonEncoder(json.JSONEncoder):
        """Custom Pandas DataFrames serializer."""

        def default(self, o):
            if isinstance(o, pandas.DataFrame):
                return o.to_dict()

            return json.JSONEncoder.default(self, o)

    def __init__(self):
        self._default_json_file_name = "explanations.json"

    @abstractmethod
    def save_json(self, path=None):
        """Save explanations as JSon file.

        Parameters
        ----------
        path : str
            Local file path where to store explanations. If path isn't
            specified, then explanations are stored to 'explanations.json' in
            the current directory

        """
        pass

    @staticmethod
    def check_explanations_serializability(explanations):
        if explanations:
            # IMPROVE: multinomial and multidimensional PD/ICE serialization is
            # not supported (yet)
            for feature in explanations.keys():
                if isinstance(feature, tuple):
                    raise errors.MliJsonSerializationError(
                        "Multidimensional ICE/PD de/serialization is not supported"
                    )
                if len(explanations[feature]) > 2:
                    # regression and binary classification (only positive
                    # class serialized) are supported
                    raise errors.MliJsonSerializationError(
                        "Multinomial ICE/PD de/serialization is not supported"
                    )

    def _save_json(self, explanations, path=None, overwrite=False):
        if explanations is None:
            raise errors.MliJsonSerializationError(
                "No explanations - call run() function first"
            )

        path = self._default_json_file_name if not path else path

        if not overwrite and os.path.exists(path):
            raise FileExistsError(f"File with explanations already exists: {path}")

        with open(path, "w", encoding="utf-8") as fp:
            json.dump(
                explanations,
                fp,
                cls=JsonPersistableExplanations.PandasJSonEncoder,
            )

    @abstractmethod
    def load_json(self, path=None):
        """Load explanations from JSon file.

        Parameters
        ----------
        path: str
            Local file path from where to loadJson explanations. If path
            isn't specified, then explanations are loaded from
            ``explanations.json`` in the current directory.

        Returns
        -------
        dict :
            Explanations deserialized from JSon.

        """
        pass

    def _check_json_path_existence(self, path):
        path = self._default_json_file_name if not path else path

        if not os.path.exists(path):
            raise FileNotFoundError(f"File with explanations doesn't exist: {path}")

        return path

    def _load_json(self, path=None):
        path = self._check_json_path_existence(path)

        explanations = {}
        with open(path, encoding="utf-8") as fp:
            explanations_as_dict = json.load(fp)
            # TODO: IMPROVE post processing is SLOW & :-/ (consider V dicts)
            for feature in explanations_as_dict:
                explanations[feature] = {}
                for clazz in explanations_as_dict[feature]:
                    explanations[feature][clazz] = pandas.DataFrame(
                        explanations_as_dict[feature][clazz]
                    )
                    # TODO: IMPROVE post processing as keys are str by default
                    explanations[feature][clazz].index = explanations[feature][
                        clazz
                    ].index.astype(int)
                    explanations[feature][clazz].columns = explanations[feature][
                        clazz
                    ].columns.astype(int)

        return explanations


class NanEncoder(json.JSONEncoder):
    def encode(self, obj):
        encoded_obj = NanEncoder.__enc_nan(obj)
        return super().encode(encoded_obj)

    @staticmethod
    def __enc_nan(obj):
        if isinstance(obj, dict):
            return {k: NanEncoder.__enc_nan(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [NanEncoder.__enc_nan(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple([NanEncoder.__enc_nan(v) for v in obj])

        if isinstance(obj, float) and math.isnan(obj):
            return commons.SafeJavaScript.NAN
        if isinstance(obj, float) and math.isinf(obj):
            return (
                commons.SafeJavaScript.INF
                if obj > 0
                else commons.SafeJavaScript.NEG_INF
            )
        if isinstance(obj, numpy.ndarray) and numpy.isnan(obj):
            return commons.SafeJavaScript.NAN
        if isinstance(obj, numpy.ndarray) and numpy.isinf(obj):
            return (
                commons.SafeJavaScript.INF
                if obj > 0
                else commons.SafeJavaScript.NEG_INF
            )

        return obj


class RobustEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            if obj is None:
                return "None"
            elif numpy.issubdtype(obj, numpy.integer):
                return int(obj)
            elif numpy.issubdtype(obj, numpy.floating):
                if numpy.isnan(obj):
                    return numpy.finfo(obj.dtype).max
                else:
                    return float(obj)
            elif isinstance(obj, numpy.ndarray):
                return obj.tolist()
        except Exception as _:
            del _

        try:
            if isinstance(obj, pandas.DataFrame) or isinstance(obj, pandas.Series):
                return json.dumps(obj.to_dict())
        except Exception as _:
            del _
        try:
            encoded = super().default(obj)
        except Exception as ex:
            msg = (
                f"RobustEncoder error: {ex} with object: {obj}:\n"
                f"{traceback.format_exc()}"
            )
            print(msg)
            sys.stdout.flush()
            sys.stderr.write(msg)
            sys.stderr.flush()
            raise
        return encoded


class InterpretationPersistence:
    """Interpretation persistence - class used to manage interpretation files
    and directories within base data directory (or equivalent on particular store
    type).

    Once extended to actual writing/reading of files it should also simplify store
    switch - like remote/multinode/distributed.

    Filesystem structure:

    <base data dir>/
        mli_experiment_<UUID>/    ... MLI interpretation (bulk explainers run)
        explanation_<job UUID>/   .. ad-hoc

    Examples
    --------

    # MLI interpretation
    mli_experiment_4d774e62-3c67-11ea-9c7e-106530ed5ceb/

    # Ad hoc explainer run
    explanation_4d774e62-3c67-11ea-9c7e-106530ed5ceb/

    """

    KEY_E_PARAMS = "explainers_parameters"
    KEY_RESULT = "result"

    DIR_AUTOML_EXPERIMENT = "h2oai_experiment_"
    DIR_MLI_EXPERIMENT = "mli_experiment_"
    DIR_MLI_TS_EXPERIMENT = "mli_experiment_timeseries_"
    DIR_AD_HOC_EXPLANATION = "explanation_"

    FILE_INTERPRETATION_JSON = "interpretation.json"
    FILE_INTERPRETATION_HTML = "interpretation.html"
    FILE_INTERPRETATION_HTML_4_PDF = "interpretation-detailed.html"
    FILE_INTERPRETATION_PDF = "interpretation-detailed.pdf"
    FILE_PROGRESS_JSON = "progress.json"
    FILE_H2O_SONAR_HTML = "h2o-sonar.html"
    FILE_MLI_EXPERIMENT_LOG = "mli_experiment_log_"

    FILE_COMMON_PARAMS = "explainers_common_parameters.json"
    FILE_EXPERIMENT_ID_COLS = "experiment_id_columns.json"

    FILE_EXPERIMENT_TS = "IS_TIMESERIES"
    FILE_EXPERIMENT_IMAGE = "IS_IMAGE"

    FILE_PREFIX_DATASET = "dataset_"

    @staticmethod
    def is_safe_name(name: str) -> bool:
        """Check whether given nameis formed by alphanumeric chars
        (and therefore filesystem safe).

        """
        if name == InterpretationPersistence.to_alphanum_name(name):
            return True
        return False

    @staticmethod
    def to_alphanum_name(name: str):
        """Convert given name to filesystem save string formed by alphanumeric
        characters.

        """
        if name is None or not name:
            return ""
        else:
            # TODO add -
            return re.sub("[^0-9a-zA-Z_]+", "_", name)

    @staticmethod
    def get_async_log_file_name(mli_key: str):
        return f"{InterpretationPersistence.FILE_MLI_EXPERIMENT_LOG}-{mli_key}.log"

    @staticmethod
    def get_mli_dir_name(data_dir: str, username: str, mli_key: str):
        return Persistence.make_key(
            data_dir,
            username,
            f"{InterpretationPersistence.DIR_MLI_EXPERIMENT}{mli_key}",
        )

    @staticmethod
    def get_ad_hoc_mli_dir_name(data_dir: str, username: str, explainer_job_key: str):
        return Persistence.make_key(
            data_dir,
            username,
            f"{InterpretationPersistence.DIR_AD_HOC_EXPLANATION}{explainer_job_key}",
        )

    @staticmethod
    def get_base_dir(data_dir: str, dir_name: str):
        return Persistence.make_key(data_dir, dir_name)

    def rm_base_dir(self, logger=None):
        self.rm_dir(dir_path=self._base_dir, logger=logger)

    @staticmethod
    def to_server_path(data_dir: str, path: str):
        """Return bare server path without data directory"""
        if data_dir and path:
            bare_path: str = path.replace(data_dir, "")
            # :-/ Win
            return bare_path[1:] if bare_path.startswith("/") else bare_path
        return None

    @staticmethod
    def to_server_file_path(data_dir: str, path: str):
        """Return bare server path without data directory"""
        if data_dir and path:
            bare_path: str = path.replace(data_dir, "")
            # :-/ Win
            bare_path = bare_path[1:] if bare_path.startswith("/") else bare_path
            return f"files/{bare_path}"
        return None

    @staticmethod
    def list_interpretations(
        data_dir: str,
        username: str,
        store_persistence: Persistence,
        paths: bool = True,
    ):
        """List interpretations.

        Parameters
        ----------
        data_dir : str
          H2O Sonar results directory.
        username : str
          Username.
        store_persistence : Persistence
          Handle to the store persistence.
        paths : bool
          Return list of paths (e.g. file-systems) if ``True`` (default),
          else return interpretation UUIDs.

        """
        user_dir = Persistence.make_key(
            data_dir,
            username,
        )
        interpretations_dirs: list = store_persistence.list_files_by_wildcard(
            key=user_dir,
            wildcard=f"{InterpretationPersistence.DIR_MLI_EXPERIMENT}*",
        )

        # get UUID keys of interpretations
        keys = [x[-36:] for x in interpretations_dirs]

        return (
            keys
            if not paths
            else [
                Persistence.make_key(
                    user_dir, f"{InterpretationPersistence.DIR_MLI_EXPERIMENT}{key}"
                )
                for key in keys
            ]
        )

    @property
    def data_dir(self) -> str:
        return self._data_dir

    @property
    def user_dir(self) -> str:
        return self._user_dir

    @property
    def base_dir(self) -> str:
        return self._base_dir

    @property
    def tmp_dir(self) -> str:
        return self._tmp_dir

    @property
    def mli_key(self) -> str:
        return self._mli_key

    @property
    def ad_hoc_job_key(self) -> str:
        return self._ad_hoc_job_key

    def __init__(
        self,
        data_dir: str,
        username: str,
        mli_key: str = None,
        ad_hoc_explainer_job_key: str = None,
        store_persistence: Persistence | None = None,
        logger=None,
    ):
        """MLI persistence - either MLI interpretation on ad-hoc explainer run key.

        Parameters
        ----------
        data_dir: str
          Base data dir.
        username: str
          Username.
        mli_key: str
          MLI key in case of MLI interpretation.
        ad_hoc_explainer_job_key: str
          Explainer run key in case of ad-hoc explainer run.
        store_persistence : Persistence | None
          Store persistence (low level).
        logger :
          Logger.

        """
        if not data_dir:
            raise ValueError("Data directory dir must be specified")

        self.store = store_persistence or FilesystemPersistence(
            base_path=data_dir,
        )

        self.logger = logger or loggers.SonarPrintLogger()

        username = username or ""

        self._data_dir: str = data_dir
        self._username: str = username
        self._user_dir: str = Persistence.make_key(self._data_dir, self._username)

        if mli_key:
            self._base_dir: str = InterpretationPersistence.get_mli_dir_name(
                data_dir=data_dir, username=username, mli_key=mli_key
            )
            self._tmp_dir: str = Persistence.make_key(self._base_dir, "tmp")
            self._mli_key: str = mli_key
        elif ad_hoc_explainer_job_key:
            self._base_dir = InterpretationPersistence.get_ad_hoc_mli_dir_name(
                data_dir=data_dir,
                username=username,
                explainer_job_key=ad_hoc_explainer_job_key,
            )
            self._tmp_dir: str = Persistence.make_key(self._base_dir, "tmp")
            self._ad_hoc_job_key: str = ad_hoc_explainer_job_key
        else:
            raise ValueError("Either MLI or ad-hoc run key to be specified")

    def make_base_dir(self):
        self.store.make_dir(self._base_dir)
        if PersistenceType.file_system != self.store.type:
            self.store.internal.make_dir(
                self.store.internal.path_to_internal(self._base_dir)
            )

    def make_tmp_dir(self):
        self.store.make_dir(self._tmp_dir)
        if PersistenceType.file_system != self.store.type:
            self.store.internal.make_dir(
                self.store.internal.path_to_internal(self._tmp_dir)
            )

    def make_interpretation_sandbox(self):
        """Create interpretation directory as well as common files."""
        self.make_base_dir()
        self.make_tmp_dir()

    def rm_dir(self, dir_path):
        try:
            if self.store.is_dir(dir_path):
                self.store.delete_tree(dir_path)
        except Exception as ex:
            self.logger.warning(
                f"MLI persistence failed to delete dir {dir_path}: {ex}",
            )

    def get_base_dir_file(self, file_name: str) -> str:
        return Persistence.make_key(self._base_dir, file_name)

    def create_dataset_path(self) -> str:
        return self.get_base_dir_file(
            f"{InterpretationPersistence.FILE_PREFIX_DATASET}{uuid.uuid4()}.csv"
        )

    def save_message_entity(self, entity, path: str):
        if entity:
            self.store.save_json(data=entity.dump(), key=path)
        else:
            raise RuntimeError(
                f"Unable to atomically save JSon entity file {path} - no entity"
            )

    def load_message_entity(self, path: str) -> dict:
        if self.store.is_file(path):
            return self.store.load_json(path)
        raise errors.MliError(
            f"Unable to load JSon entity - file {path} does not exist"
        )

    def get_html_path(self) -> str:
        return Persistence.make_key(
            self._base_dir, InterpretationPersistence.FILE_INTERPRETATION_HTML
        )

    def get_html_4_pdf_path(self) -> str:
        return Persistence.make_key(
            self._base_dir, InterpretationPersistence.FILE_INTERPRETATION_HTML_4_PDF
        )

    def get_pdf_path(self) -> str:
        return Persistence.make_key(
            self._base_dir, InterpretationPersistence.FILE_INTERPRETATION_PDF
        )

    def save_as_html(self, interpretation_html: str):
        """Save interpretation as HTML."""
        self.store.save(key=self.get_html_path(), data=interpretation_html)

    def save_as_pdf(self, interpretation):
        """Save interpretation as PDF."""
        # create HTML for PDF
        self.store.save(
            key=self.get_html_4_pdf_path(), data=interpretation.to_html_4_pdf()
        )
        # use pandoc to create PDF
        interpretation.to_pdf(
            input_path=self.get_html_4_pdf_path(),
            output_path=self.get_pdf_path(),
        )

    def get_json_path(self) -> str:
        return Persistence.make_key(
            self._base_dir, InterpretationPersistence.FILE_INTERPRETATION_JSON
        )

    def save_as_json(self, interpretation_dict: dict):
        """Save interpretation as JSon."""
        self.store.save_json(key=self.get_json_path(), data=interpretation_dict)

    def is_common_params(self):
        return self.store.is_file(
            Persistence.make_key(
                self._base_dir, InterpretationPersistence.FILE_COMMON_PARAMS
            )
        )

    def save_common_params(self, entity: commons.CommonInterpretationParams):
        """Save ``CommonExplainerParameters`` entity to interpretation root dir."""
        self.save_message_entity(
            entity=entity,
            path=Persistence.make_key(
                self._base_dir, InterpretationPersistence.FILE_COMMON_PARAMS
            ),
        )

    def load_common_params(
        self, patch_sequential_execution: bool | None = None
    ) -> commons.CommonInterpretationParams:
        """Load ``CommonExplainerParameters`` entity from interpretation root dir."""
        path: str = Persistence.make_key(
            self._base_dir, InterpretationPersistence.FILE_COMMON_PARAMS
        )
        json_dict = self.store.load_json(path)
        dai_params = (
            commons.CommonInterpretationParams.load(json_dict) if json_dict else None
        )
        if patch_sequential_execution is not None and dai_params:
            dai_params.sequential_execution = patch_sequential_execution

        return dai_params

    def save_experiment_type_hints(
        self, is_timeseries: bool = False, is_image: bool = False
    ):
        """Write hint (in backward compatible manner) indicating experiment type (like
        timeseries or image) to interpretation directory (IID is default).

        Parameters
        ----------
        is_timeseries : bool
          Write time series hint.
        is_image : bool
          Write image hint.

        """
        if is_timeseries:
            self.store.touch(
                Persistence.make_key(
                    self._base_dir, InterpretationPersistence.FILE_EXPERIMENT_TS
                )
            )
        if is_image:
            self.store.touch(
                Persistence.make_key(
                    self._base_dir,
                    InterpretationPersistence.FILE_EXPERIMENT_IMAGE,
                )
            )

    def load_is_timeseries_experiment(self):
        return self.store.is_file(
            Persistence.make_key(
                self._base_dir, InterpretationPersistence.FILE_EXPERIMENT_TS
            )
        )

    def load_is_image_experiment(self):
        return self.store.is_file(
            Persistence.make_key(
                self._base_dir, InterpretationPersistence.FILE_EXPERIMENT_IMAGE
            )
        )

    def load_explainers_params(self, explainer_id: str = "") -> dict:
        """Load explainers parameters dictionary from interpretation JSon."""
        explainers_params = {}
        if pathlib.Path(self.get_json_path()).exists():
            as_dict = self.store.load_json(key=self.get_json_path())
            explainers_params = as_dict.get(
                InterpretationPersistence.KEY_RESULT, {}
            ).get(InterpretationPersistence.KEY_E_PARAMS, {})

        if explainers_params and explainer_id:
            return explainers_params.get(explainer_id, {})

        return {}

    def get_experiment_id_cols_path(self) -> str:
        return Persistence.make_key(
            self.base_dir, InterpretationPersistence.FILE_EXPERIMENT_ID_COLS
        )

    def make_dir_zip_archive(
        self,
        src_dir_path: str | pathlib.Path,
        zip_path: str | pathlib.Path,
        file_filter=lambda x: False,
    ):
        self.store.make_dir_zip_archive(
            src_key=src_dir_path,
            zip_key=zip_path,
            file_filter=file_filter,
        )

    def resolve_model_path(self, model_path: str):
        """Resolve fitted model path as there are several combinations of DAI
        configuration and experiment creation (path):

        - fitted model path MAY have <username> prefix, based on whether it was
          created in 1.8.x version or with `config.per_user_directories=True/False`
        - current user directory may be either data directory, or may have username
          in path based on `config.per_user_directories` configuration item value

        Parameters
        ----------
        model_path: str
          (Un)fitted model *relative* path as present on model entity
          as `model.fitted_model_path`.

        """
        # (un)fitted model path CAN contains username directory
        #   (<username>/)?h2oai_experiment_<UUID>/fitted_model.pickle
        resolved_path = Persistence.make_key(self.data_dir, model_path)
        if self.store.is_file(resolved_path):
            return resolved_path

        # fallback to user dir
        return Persistence.make_key(self.user_dir, model_path)


class ExplainerPersistence(InterpretationPersistence):
    """Explainer persistence.

    Filesystem structure:

    .. code-block:: text

        mli_experiment_<UUID>/ (MLI interpretation) OR explanation_<job UUID>/ (ad hoc)
            explainer_<explainer ID>_<job UUID>/
                <explanation name>/
                    explanation.<extension>
                    ... extra files completing main explanation file allowed in this dir
                 work/
                    ... directory which can be used for intermediary results persistence

    Web access:

    .. code-block:: text

        http://<HOST>:<PORT>/files/mli_experiment_<UUID>/...
        http://<HOST>:<PORT>/files/explanation_<UUID>/...

    Hints:

    * Explainer and explanation names are checked to contain safe characters
      only (alpha, num, ``.``, ``_`` and ``-``). IDs are preserved
      (filesystem/runtime match).
    * Format identifiers (MIME types) are processed to contain safe characters only.
    * explanation.<extension> is "index file" - directory may contain also other files
      which form/support the explanations
    * Explainer may be executed multiple times within one MLI interpretation,
      therefore uniqueness is guaranteed by job UUID.
    * Datatable explanation is canonical (always present), others are optional.

    Examples
    --------

    .. code-block:: text

        # MLI interpretation
        mli_experiment_4d774e62-3c67-11ea-9c7e-106530ed5ceb/

            # OOTB PD explainer
            explainer_h2oaicore.h2o_sonar.oss.byor.explainers.pd.PD_4d774e62-3c67...06530ed5ceb/
                global_partial_dependence/
                    application_vnd_h2oai_datatable_jay/
                        explanation.jay
                    application_json/
                        explanation.json
                local_individual_conditional_explanation/
                    application_vnd_h2oai_datatable_jay/
                        explanation.jay
                    application_json/
                        explanation.json
                        feature_1_class_1_pd.json
                        ...
                        feature_n_class_n_pd.json

            # hot deployed feature importance explainer
            explainer_False_test_kernel_shap_f72edb06_...er.TestKernelShap_4d7...d5ceb/
                local_feature_importance/
                    application_vnd_h2oai_datatable_jay/
                        explanation.jay
                    application_json/
                        explanation.json

        # Ad hoc explainer run
        explanation_4d774e62-3c67-11ea-9c7e-106530ed5ceb/

            # OOTB feature importance explainer
            explainer_h2oaicore.h2o_sonar.oss.byor.explainers.kernel_shap.KernelShap_4d7...ceb/
                global_feature_importance/
                    application_vnd_h2oai_datatable_jay/
                        explanation.jay
                    application_json/
                        explanation.json

    """

    DIR_EXPLAINER = "explainer_"
    DIR_WORK = "work"
    DIR_PROBLEMS = "problems"
    DIR_INSIGHTS = "insights"
    DIR_LOG = "log"

    FILE_RESULT_DESCRIPTOR = "result_descriptor.json"
    FILE_PROBLEMS = "problems_and_actions.json"
    FILE_INSIGHTS = "insights_and_actions.json"
    FILE_EXPLANATION = "explanation"
    FILE_EXPLAINER_PICKLE = "explainer.pickle"
    FILE_ON_DEMAND_EXPLANATION_SUFFIX = "on_demand_explanation.txt"
    FILE_DONE_DONE = "EXPLAINER_DONE"
    FILE_DONE_FAILED = "EXPLAINER_FAILED"

    EXPLAINER_LOG_PREFIX = "explainer_run_"
    EXPLAINER_LOG_SUFFIX_ANON = "_anonymized.log"

    @staticmethod
    def make_dir(target_dir):
        ExplainerPersistence.makedirs(target_dir, exist_ok=True)

    @staticmethod
    def makedirs(path: str, exist_ok=True):
        """Avoid some inefficiency in ``os.makedirs()``.

        Parameters
        ----------
        path : str
          Path to directory/ies to create.
        exist_ok : bool
          Fail if directory exists.

        Returns
        -------
        str :
          Path to newly create directory.

        """
        if os.path.isdir(path) and os.path.exists(path):
            assert exist_ok, f"Directory '{path}' already exists"
        else:
            os.makedirs(path, exist_ok=exist_ok)

        return path

    @staticmethod
    def get_dirs_for_explainer_id(
        data_dir: str,
        username: str,
        mli_key: str,
        explainer_id: str,
        explainer_job_key: str | None = None,
    ) -> list:
        # TODO consolidate this method / avoid duplicated code across persistence

        # MLI interpretation
        persistence = ExplainerPersistence(
            data_dir=data_dir,
            username=username,
            mli_key=mli_key,
            explainer_id=explainer_id,
            explainer_job_key="",
        )
        dirs: list = Persistence.list_files_by_wildcard(
            key=persistence.base_dir,
            wildcard=(
                f"{ExplainerPersistence.DIR_EXPLAINER}"
                f"{InterpretationPersistence.to_alphanum_name(explainer_id)}*"
            ),
        )
        if not dirs:
            # fallback to ad_hoc explainer
            persistence = ExplainerPersistence(
                data_dir=data_dir,
                username=username,
                mli_key=explainer_job_key if explainer_job_key else None,
                explainer_id=explainer_id,
                explainer_job_key=explainer_job_key if explainer_job_key else mli_key,
            )
            dirs: list = Persistence.list_files_by_wildcard(
                key=persistence.base_dir,
                wildcard=(
                    f"{ExplainerPersistence.DIR_EXPLAINER}"
                    f"{InterpretationPersistence.to_alphanum_name(explainer_id)}*"
                ),
            )

        return dirs

    @staticmethod
    def get_key_for_explainer_dir(explainer_dir_path: str) -> str | None:
        # 63afb83c-9a80-11ea-a109-207918bc8e4b
        key_lng: int = 36
        if explainer_dir_path and len(explainer_dir_path) > 36:
            if explainer_dir_path.endswith("/") or explainer_dir_path.endswith("\\"):
                key_lng += 1
            if "_" == explainer_dir_path[-key_lng - 1 : -key_lng]:
                return explainer_dir_path[-key_lng:]

        return ExplainerPersistence._legacy_get_key_for_explainer_dir(
            explainer_dir_path
        )

    @staticmethod
    def _legacy_get_key_for_explainer_dir(
        explainer_dir_path: str,
    ) -> str | None:
        # betiwuwo
        key_lng: int = 8
        if explainer_dir_path and len(explainer_dir_path) > 8:
            if explainer_dir_path.endswith("/") or explainer_dir_path.endswith("\\"):
                key_lng += 1
            return explainer_dir_path[-key_lng:]
        return None

    @staticmethod
    def get_locators_for_explainer_id(
        data_dir: str,
        username: str,
        mli_key: str,
        explainer_id: str,
        explainer_job_key: str | None = None,
    ) -> list[tuple[str, str]] | None:
        explainer_dirs: list = ExplainerPersistence.get_dirs_for_explainer_id(
            data_dir=data_dir,
            username=username,
            mli_key=mli_key,
            explainer_id=explainer_id,
            explainer_job_key=explainer_job_key,
        )
        if explainer_dirs:
            result: list[tuple[str, str]] = list()
            for e_dir in explainer_dirs:
                e_key = ExplainerPersistence.get_key_for_explainer_dir(e_dir)
                result.append((e_dir, f"{e_key}"))
            return result

        return None

    @property
    def explainer_id(self) -> str:
        return self._explainer_id

    @property
    def explainer_job_key(self) -> str:
        return self._explainer_job_key

    @property
    def username(self) -> str:
        return self._username

    def __init__(
        self,
        data_dir: str,
        username: str,
        explainer_id: str,
        explainer_job_key: str,
        mli_key: str = None,
        store_persistence: Persistence | None = None,
    ):
        """Persistence for a particular instances of explainer.

        Parameters
        ----------
        data_dir: str
          Base data dir.
        username: str,
          Username.
        explainer_job_key: str
          Explainer run job key.
        mli_key: str
          MLI key in case of MLI interpretation, `None` (or explainer run job key) in
          case of ad-hoc explainer.
        explainer_id: str
          Explainer ID - unique explainer identifier.
        store_persistence : Persistence | None
          Store persistence (low level).

        """
        username = username or ""

        self.store = store_persistence or FilesystemPersistence(
            base_path=data_dir,
        )

        if not mli_key or mli_key == explainer_job_key:
            InterpretationPersistence.__init__(
                self,
                data_dir=data_dir,
                username=username,
                ad_hoc_explainer_job_key=explainer_job_key,
                store_persistence=self.store,
            )
        else:
            InterpretationPersistence.__init__(
                self,
                data_dir=data_dir,
                username=username,
                mli_key=mli_key,
                store_persistence=self.store,
            )

        if not self._base_dir:
            raise ValueError("Custom explainer persistence base dir must be specified")
        if not explainer_id:
            raise ValueError(
                f"Custom explainer ID '{explainer_id}' must be specified "
                f"for persistence"
            )

        self._username = username
        self._explainer_id = explainer_id
        self._explainer_job_key = explainer_job_key

    @staticmethod
    def save_json(data: dict, path: str):
        # ATOMIC write to file ... :-/
        tmp_name: str = f"{path}.{str(uuid.uuid4())}.tmp"
        with open(tmp_name, mode="w") as json_file:
            json_file.write(json.dumps(data, indent=4, cls=RobustEncoder))
            # context manager flush is NOT guaranteed for libs
            json_file.flush()
            # fdatasync() (fast, no meta) vs. os.fsync() (integrity)
            os.fsync(json_file.fileno())
        # flush tmp file existence
        if Persistence.flush_dir_for_file(tmp_name):
            shutil.move(tmp_name, path)
            # flushed rename
            Persistence.flush_dir_for_file(path)
        else:
            raise RuntimeError(
                f"Unable to atomically save JSon entity file by renaming "
                f"'{tmp_name}' (existence flush timeout)"
            )

    def __get_explainer_dir(self, base_dir=None):
        """Internal get explainer dir to get path to work/ and log/ directories."""
        return Persistence.make_key(
            base_dir or self.store.internal.base_path,
            f"{ExplainerPersistence.DIR_EXPLAINER}"
            f"{InterpretationPersistence.to_alphanum_name(self._explainer_id)}"
            f"_{self._explainer_job_key}",
        )

    def get_explainer_dir(self) -> str:
        return self.__get_explainer_dir(self._base_dir)

    def get_relative_path(self, path: str, base_entity: str = "interpretation"):
        if path and len(path) > len(self.base_dir) and path.startswith(self.base_dir):
            if base_entity == "interpretation":
                return path[len(self.base_dir) + 1 :]
        else:
            return path

        raise ValueError(
            f"Unknown base entity '{base_entity}' on making the following path "
            f"relative: '{path}'"
        )

    def get_explainer_dir_archive(self) -> str:
        return Persistence.make_key(
            self._base_dir,
            f"{ExplainerPersistence.DIR_EXPLAINER}"
            f"{InterpretationPersistence.to_alphanum_name(self._explainer_id)}"
            f"_{self._explainer_job_key}.zip",
        )

    def rm_explainer_dir(self):
        self.store.delete_tree(self.get_explainer_dir())

    def get_explainer_working_dir(self) -> str:
        explainer_working_dir = Persistence.make_key(
            self.get_explainer_dir(), ExplainerPersistence.DIR_WORK
        )
        if PersistenceType.file_system == self.store.type:
            return explainer_working_dir

        return self.store.internal.path_to_internal(explainer_working_dir)

    def get_explainer_working_file(self, file_name: str) -> str:
        return Persistence.make_key(self.get_explainer_working_dir(), file_name)

    def get_evaluator_working_file(self, file_name: str) -> str:
        return self.get_explainer_working_file(file_name)

    def get_explainer_problems_dir(self) -> str:
        explainer_problems_dir = Persistence.make_key(
            self.get_explainer_dir(), ExplainerPersistence.DIR_PROBLEMS
        )
        if PersistenceType.file_system == self.store.type:
            return explainer_problems_dir

        return self.store.internal.path_to_internal(explainer_problems_dir)

    def get_explainer_insights_dir(self) -> str:
        explainer_insights_dir = Persistence.make_key(
            self.get_explainer_dir(), ExplainerPersistence.DIR_INSIGHTS
        )
        if PersistenceType.file_system == self.store.type:
            return explainer_insights_dir

        return self.store.internal.path_to_internal(explainer_insights_dir)

    def get_explainer_problems_file(self, file_name: str) -> str:
        return Persistence.make_key(self.get_explainer_problems_dir(), file_name)

    def get_explainer_insights_file(self, file_name: str) -> str:
        return Persistence.make_key(self.get_explainer_insights_dir(), file_name)

    def get_explainer_log_dir(self) -> str:
        explainer_log_dir = Persistence.make_key(
            self.get_explainer_dir(), ExplainerPersistence.DIR_LOG
        )
        if PersistenceType.file_system == self.store.type:
            return explainer_log_dir

        return self.store.internal.path_to_internal(explainer_log_dir)

    def get_explainer_log_file(self) -> str:
        return (
            f"{ExplainerPersistence.EXPLAINER_LOG_PREFIX}{self.explainer_job_key}.log"
        )

    def get_explainer_log_path(self) -> str:
        return Persistence.make_key(
            self.get_explainer_log_dir(), self.get_explainer_log_file()
        )

    def get_explainer_ann_log_file(self) -> str:
        return self.get_explainer_log_file().replace(
            ".log", ExplainerPersistence.EXPLAINER_LOG_SUFFIX_ANON
        )

    def get_explainer_ann_log_path(self) -> str:
        return self.get_explainer_log_path().replace(
            ".log", ExplainerPersistence.EXPLAINER_LOG_SUFFIX_ANON
        )

    def get_explanation_meta_path(
        self, explanation_type: str, explanation_format: str
    ) -> str:
        return (
            f"{self.get_explanation_dir_path(explanation_type, explanation_format)}"
            f".meta"
        )

    def get_explanation_dir_path(
        self, explanation_type: str, explanation_format: str
    ) -> str:
        """Get explanation directory path.

        Parameters
        ----------
        explanation_type : str
          Explanation identifier returned by ``explanation_type()``.
        explanation_format : str
          Format MIME type.

        Returns
        -------
        str :
          Path to the directory with the explanation.

        """
        if not explanation_type:
            raise ValueError("Explanation type must be non-empty")
        if not explanation_format:
            raise ValueError("Explanation format must be non-empty")
        return Persistence.make_key(
            self.get_explainer_dir(),
            InterpretationPersistence.to_alphanum_name(explanation_type),
            InterpretationPersistence.to_alphanum_name(explanation_format),
        )

    def get_explanation_file_path(
        self,
        explanation_type: str,
        explanation_format: str,
        explanation_file: str = None,
    ) -> str:
        if explanation_file:
            file: str = explanation_file
        else:
            file = (
                f"{ExplainerPersistence.FILE_EXPLANATION}."
                f"{commons.MimeType.ext_for_mime(explanation_format)}"
            )

        return Persistence.make_key(
            self.get_explanation_dir_path(explanation_type, explanation_format),
            file,
        )

    def get_result_descriptor_file_path(self) -> str:
        return Persistence.make_key(
            self.get_explainer_dir(),
            ExplainerPersistence.FILE_RESULT_DESCRIPTOR,
        )

    def load_result_descriptor(self) -> dict:
        return (
            self.store.load_json(self.get_result_descriptor_file_path())
            if self.store.exists(self.get_result_descriptor_file_path())
            else {}
        )

    def make_explainer_dir(self):
        self.store.make_dir(self.get_explainer_dir())

    def make_explainer_working_dir(self):
        if PersistenceType.file_system == self.store.type:
            self.store.make_dir(self.get_explainer_working_dir())
        else:
            self.store.internal.make_dir(self.get_explainer_working_dir())

    def make_explainer_problems_dir(self):
        if PersistenceType.file_system == self.store.type:
            self.store.make_dir(self.get_explainer_problems_dir())
        else:
            self.store.internal.make_dir(self.get_explainer_problems_dir())

    def make_explainer_insights_dir(self):
        if PersistenceType.file_system == self.store.type:
            self.store.make_dir(self.get_explainer_insights_dir())
        else:
            self.store.internal.make_dir(self.get_explainer_insights_dir())

    def save_problems(self, problems: list[dict]):
        """Save model problems."""
        self.store.save_json(
            key=self.get_explainer_problems_file(ExplainerPersistence.FILE_PROBLEMS),
            data=problems,
        )

    def load_problems(self) -> list[dict]:
        """Load model problems."""
        path: str = self.get_explainer_problems_file(ExplainerPersistence.FILE_PROBLEMS)
        return self.store.load_json(path)

    def save_insights(self, insights: list[dict]):
        """Save insights."""
        self.store.save_json(
            key=self.get_explainer_insights_file(ExplainerPersistence.FILE_INSIGHTS),
            data=insights,
        )

    def load_insights(self) -> list[dict]:
        """Load insights."""
        path: str = self.get_explainer_insights_file(ExplainerPersistence.FILE_INSIGHTS)
        return self.store.load_json(path)

    def make_explainer_log_dir(self):
        if PersistenceType.file_system == self.store.type:
            self.store.make_dir(self.get_explainer_log_dir())
        else:
            self.store.internal.make_dir(self.get_explainer_log_dir())

    def make_explainer_sandbox(self, dai_params=None):
        """Create explainer working dir and log directories as well as common files.

        Parameters
        ----------
        dai_params: CommonDaiExplainerParameters
          Common explainer parameters to be stored in the root of the
          interpretation (if it already doesn't exist).

        """
        self.make_explainer_working_dir()
        self.make_explainer_problems_dir()
        self.make_explainer_insights_dir()
        self.make_explainer_log_dir()
        if dai_params and not self.is_common_params():
            self.save_common_params(dai_params)

    def resolve_mli_path(self, mli_key: str, username: str):
        """Resolve MLI interpretation directory as it should be in the directory with
        username in path, but potentially it will be possible to create it in
        directory without it using ``config.per_user_directories`` (or can be migrated
        from 1.8.x).

        """
        resolved_path = Persistence.make_key(
            self.data_dir,
            f"{InterpretationPersistence.DIR_MLI_EXPERIMENT}{mli_key}",
        )
        if self.store.is_file(
            Persistence.make_key(resolved_path, ExplainerPersistence.DIR_WORK)
        ):
            return resolved_path

        return Persistence.make_key(
            self.data_dir,
            username,
            f"{InterpretationPersistence.DIR_MLI_EXPERIMENT}{mli_key}",
        )


class PersistenceApi(abc.ABC):
    """Factory which creates ``Persistence`` implementations for various store types
    and purposes which are available in specific runtime and/or container(s).

    """

    def __init__(self, logger: loggers.SonarLogger | None = None):
        self.logger = logger or loggers.SonarPrintLogger()

    def get_cwl(
        self,
        persistence_type: PersistenceType = PersistenceType.file_system,
    ):
        if PersistenceType.in_memory == persistence_type:
            return InMemoryPersistence.get_default_cwl()
        elif PersistenceType.database == persistence_type:
            raise NotImplementedError

        return FilesystemPersistence.get_default_cwl()

    def create_persistence(
        self,
        persistence_type: PersistenceType = PersistenceType.file_system,
        base_path: str = "",
        connection_string: str = "",
    ) -> InMemoryPersistence | FilesystemPersistence:
        """Create persistence of given *store* type - file-system,
        in-memory or DB. Default store persistence is file-system persistence with
        base in the current directory.

        Parameters
        ----------
        persistence_type : PersistenceType
          Type of the persistence to create.
        base_path : str
          Optional root path of the persistence on the host store (where meaningful
          e.g. file-system).
        connection_string : str
          Option connection string (where meaningful e.g. database).

        Returns
        -------
        Any :
          Persistence to load/store container and explainer artifacts.

        """
        del connection_string

        if PersistenceType.in_memory == persistence_type:
            return InMemoryPersistence()
        elif PersistenceType.database == persistence_type:
            raise NotImplementedError

        return FilesystemPersistence(base_path)

    def create_interpretation_persistence(
        self,
        store_persistence: Persistence,
        base_path: str | pathlib.Path,
        interpretation_key: str,
        username: str = "",
    ) -> InterpretationPersistence:
        """Create interpretation persistence *atop* given store persistence e.g.
        to store interpretations in-memory.

        """
        return InterpretationPersistence(
            store_persistence=store_persistence,
            data_dir=Persistence.make_key(base_path),
            username=username or getpass.getuser(),
            mli_key=interpretation_key,
        )

    def create_explainer_persistence(
        self,
        store_persistence: Persistence,
        base_path: str | pathlib.Path,
        interpretation_key: str,
        explainer_id: str,
        explainer_job_key: str,
        username: str = "",
    ) -> ExplainerPersistence:
        """Create *explainer* persistence *atop* given store persistence e.g.
        to store explainer data to database.

        """
        return ExplainerPersistence(
            store_persistence=store_persistence,
            data_dir=Persistence.make_key(base_path),
            username=username or getpass.getuser(),
            explainer_id=explainer_id,
            explainer_job_key=explainer_job_key,
            mli_key=interpretation_key,
        )
