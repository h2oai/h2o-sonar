# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
"""H2O Model Validation adaptor."""

import dataclasses
import datetime
import enum
import json
import os
import pathlib
import pprint
import sys
import tempfile
import traceback
import uuid
import zipfile
from typing import Any
from urllib.parse import urlparse

import datatable
import numpy as np
import pandas

from h2o_sonar import config as config
from h2o_sonar import errors
from h2o_sonar import loggers
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import persistences


try:
    import h2o_mv
    from h2o_mv.core import mv_test

    HAS_H2O_MV = True
except ImportError:
    HAS_H2O_MV = False


PACKAGE_NAME = "h2o-mv"


class ExplainerToMvTestAdapter:
    """H2O Sonar ``Explainer`` to H2O Model Validation ``MVTest`` adapter."""

    # MV module name might be different from the package name (as it was in the past)
    MV_PYTHON_MODULE_NAME = PACKAGE_NAME

    def __init__(self):
        # H2O MV test class
        self.mv_client = None
        self.persistence = None
        self.h2o_sonar_config = None
        self.logger = loggers.SonarPrintLogger()

    def check_mv_compatibility(self, explainer) -> None:
        if not HAS_H2O_MV:
            raise ImportError(
                f"{commons.opt_import_err_msg(PACKAGE_NAME)} - NOT COMPATIBLE"
            )

        # enable explainer ONLY if all required modules are installed
        self.h2o_sonar_config = self.h2o_sonar_config or explainer.config
        required_modules = set()
        required_modules.add(ExplainerToMvTestAdapter.MV_PYTHON_MODULE_NAME)
        if not explainer.check_required_modules(required_modules):
            raise errors.ExplainerCompatibilityError(
                f"{explainer.log_name} not compatible as the following required Python "
                f"module is not installed: "
                f"{ExplainerToMvTestAdapter.MV_PYTHON_MODULE_NAME}"
            )

    def setup(
        self,
        h2o_sonar_config,
        persistence: persistences.ExplainerPersistence,
        logger: loggers.SonarLogger,
    ):
        self.persistence = persistence
        self.h2o_sonar_config = h2o_sonar_config
        self.logger = logger or loggers.SonarPrintLogger()

        # set MV data dir to interpretation dir so that it can be reused by explainers
        self.mv_client = h2o_mv.core.mv_client.MVClient(data_folder=persistence.tmp_dir)
        self.mv_client.select_database(name=h2o_mv.core.mv_database.DatabaseName.TEST)

    def _get_dai_conn_from_sonar_cfg_by_key(
        self, connection_key
    ) -> config.ConnectionConfig | None:
        for c in self.h2o_sonar_config.connections:
            if c.key == connection_key:
                if c.connection_type in [
                    config.ConnectionConfigType.DRIVERLESS_AI.name,
                    config.ConnectionConfigType.DRIVERLESS_AI_STEAM.name,
                    config.ConnectionConfigType.DRIVERLESS_AI_AIEM.name,
                ]:
                    return c
                else:
                    raise errors.MliError(
                        f"Only Driverless AI connection is supported, but "
                        f"{c.connection_type} was provided."
                    )
        return None

    def _probe_dai_connectivity(
        self,
        dai_connection_key: str,
        log_name: str,
        logger: loggers.SonarLogger | None = None,
    ):
        """Lookup Driverless AI connection by key in the H2O Sonar configuration
        and check network connectivity to the Driverless AI server.

        """
        logger = logger or loggers.SonarPrintLogger()

        if not dai_connection_key:
            raise errors.MliError(
                f"{log_name}: required worker connection key not specified "
                f"in the explainer parameters"
            )
        # h2o_mv.platforms.driverless.platform.DriverlessPlatform:
        dai_connection = self._get_dai_conn_from_sonar_cfg_by_key(dai_connection_key)
        if not dai_connection:
            raise errors.MliError(
                f"{log_name}: connection '{dai_connection_key}' "
                f"not found in the H2O Sonar configuration. "
            )
        # probe DAI connectivity, but do NOT if DAI is hosted by Steam or AIEM
        if (
            dai_connection.connection_type
            == config.ConnectionConfigType.DRIVERLESS_AI.name
        ):
            parsed_url = urlparse(dai_connection.server_url)
            if not commons.is_port_used(
                hostname=parsed_url.hostname, port=parsed_url.port, logger=logger
            ):
                raise errors.MliError(
                    f"{log_name}: cannot connect to Driverless AI instance "
                    f"'{dai_connection_key}' at {dai_connection.server_url}"
                )

        return dai_connection

    def _get_mv_dai_worker_by_key(self, connection_key: str, log_name: str):
        """Get H2O MV worker connection from the H2O Sonar configuration by name
        and convert it to the MV structure.

        """
        c = self._get_dai_conn_from_sonar_cfg_by_key(connection_key)
        if c:
            # convert H2O Sonar connection to H2O MV connection
            if not HAS_H2O_MV:
                raise ImportError(
                    f"{commons.opt_import_err_msg(PACKAGE_NAME)} - NOT COMPATIBLE"
                )

            t_dai_platform = h2o_mv.platforms.driverless.platform

            if c.connection_type == config.ConnectionConfigType.DRIVERLESS_AI.name:
                # IMPROVE: does this call has side effects and needs to be kept?
                dai_worker = t_dai_platform.DriverlessPlatform(
                    # a custom name of the connection
                    name=c.name,
                    # a custom description of the connection
                    description=c.description,
                    # Driverless AI server URL like
                    # - http://host:12345
                    address=c.server_url,
                    # Driverless AI username
                    username=c.username,
                    mv_type=h2o_mv.core.mv_types.MVType.DAI,
                )
                self.logger.debug(
                    f"{log_name}: MV's standalone Driverless AI platform created: "
                    f"{dai_worker} "
                )
                dai_credentials = t_dai_platform.DriverlessCredentials(
                    # Driverless AI server address like
                    # - http://host:12345
                    address=c.server_url,
                    # Driverless AI username
                    username=c.username,
                    # Driverless AI password
                    password=c.password,
                )
                self.mv_client._cache.save_credentials(dai_credentials)
            elif (
                c.connection_type
                == config.ConnectionConfigType.DRIVERLESS_AI_STEAM.name
            ):
                dai_worker = t_dai_platform.DriverlessPlatform(
                    name=c.name,
                    address=c.server_url,
                    username=c.username,
                    description=c.description,
                    mv_type=h2o_mv.core.mv_types.MVType.DAI_STEAM,
                )
                self.logger.debug(
                    f"{log_name}: MV's Steam hosted Driverless AI platform created: "
                    f"{dai_worker} "
                )
                dai_credentials = t_dai_platform.DriverlessSteamCredentials(
                    # Steam server URL (REQUIRED by parent Platform class) like
                    # - https://steam.CLUSTER.h2o.ai/
                    address=c.server_url,
                    # Steam user login mail (REQUIRED by parent Platform class)
                    # - username is NOT actually used in the authentication process
                    username="firstname.lastname@acme.com",
                    # Steam's refresh token as pass (REQUIRED by parent Platform class)
                    password=c.token,
                    # Steam server URL like
                    # - https://steam.CLUSTER.h2o.ai/
                    url=c.server_url,
                    # Steam's unique name given by user to the Driverless AI instance
                    instance_name=c.server_id,
                    # Steam's refresh token as pass (REQUIRED by parent Platform class)
                    # - get it from: H2O.ai Cloud > Enterprise Steam > Configurations >
                    #   Personal Access Token > Get token (NEW toke generated on access)
                    refresh_token=c.token,
                    # OPTIONAL function used to refresh access/refresh token
                    token_provider=None,
                    cacert=None,
                    verify_ssl=True,
                    no_proxy=True,
                )
                self.mv_client._cache.save_credentials(dai_credentials)
            elif (
                c.connection_type == config.ConnectionConfigType.DRIVERLESS_AI_AIEM.name
            ):
                dai_worker = t_dai_platform.DriverlessPlatform(
                    name=c.name,
                    address=c.server_url,
                    username=c.username,
                    description=c.description,
                    mv_type=h2o_mv.core.mv_types.MVType.DAI_AIEM,
                )
                self.logger.debug(
                    f"{log_name}: MV's AIEM hosted Driverless AI platform created: "
                    f"{dai_worker} "
                )

                dai_credentials = t_dai_platform.DriverlessEngineCredentials(
                    # AIEM server URL (REQUIRED by parent Platform class) like
                    # - https://enginemanager.h2o.ai/
                    address=c.server_url,
                    # AIEM valid login email (REQUIRED by parent Platform class)
                    username=c.username,
                    # an H2O.ai environment name:
                    # - https://CLUSTER.h2o.ai
                    # see https://CLUSTER.h2o.ai/cli-and-api-access
                    environment=c.environment_url,
                    # H2O.ai refresh token for given environment
                    # - H2O.ai Cloud > User > CLI & API Access > API Token (generation)
                    # - see https://CLUSTER.h2o.ai/cli-and-api-access
                    platform_token=c.token,
                    # OPTIONAL function used to refresh access/refresh token
                    token_provider=None,
                    # path to h2o-cli-config.toml file (AIEM client config)
                    # - h2o-cli-config.toml can be created using 'h2o' CLI tool
                    # - doc: https://docs.h2o.ai/h2o-ai-cloud/developerguide/cli
                    # - spec: https://github.com/h2oai/model-validation
                    #     #driverless-ai-connection-via-h2o-engine-manager
                    config_path="",
                    # workspace name
                    workspace_id=c.realm_name or "default",
                    # ID of the Driverless AI engine in AIEM
                    engine_id=c.server_id,
                )
                self.mv_client._cache.save_credentials(dai_credentials)
            else:
                raise f"Unsupported Driverless AI connection type: {c.connection_type}"

            # register platform (not needed for MV explainers D/AS run)
            dai_worker = self.mv_client.add_connection(dai_credentials)

            if not dai_worker:
                raise errors.MliError(
                    f"{log_name}: Driverless AI worker '{connection_key}' "
                    f"not initialized by H2O Model Validation. Can you connect to "
                    f"Driverless AI server? Do you have the latest Driverless AI "
                    f"client installed?. "
                )

            return dai_worker

        raise errors.MliError(
            f"{log_name}: required worker connection key "
            f"'{connection_key}' not found in the explainer parameters"
        )

    def _dump_mv_test_log_to_explainer_log(self, mv_test_log):
        if not HAS_H2O_MV:
            raise ImportError(
                f"{commons.opt_import_err_msg(PACKAGE_NAME)} - NOT COMPATIBLE"
            )

        if mv_test_log and isinstance(mv_test_log, mv_test.MVTestLog):
            for msg in mv_test_log.messages:
                text = f"[MVTestLog] {msg.text}"
                if msg.level == mv_test.MVTestLogLevel.STATUS:
                    self.logger.info(text)
                elif msg.level == mv_test.MVTestLogLevel.INFO:
                    self.logger.info(text)
                elif msg.level == mv_test.MVTestLogLevel.WARNING:
                    self.logger.warning(text)
                elif msg.level == mv_test.MVTestLogLevel.ERROR:
                    self.logger.error(text)
                else:
                    self.logger.debug(text)

    @staticmethod
    def assert_mv_test_status(explainer, mv_test):
        if not HAS_H2O_MV:
            raise ImportError(
                f"{commons.opt_import_err_msg(PACKAGE_NAME)} - NOT COMPATIBLE"
            )

        if mv_test.state != h2o_mv.core.mv_states.MVTestState.Done:
            err_msg = (
                f"{explainer.log_name} did not finish - it finished with state: "
                f"{mv_test.state}"
            )
            explainer.logger.error(err_msg)
            raise errors.MliError(err_msg)

        if mv_test.progress < 100:
            err_msg = (
                f"{explainer.log_name} did not finish - it finished with progress: "
                f"{mv_test.progress}%"
            )
            explainer.logger.error(err_msg)
            raise errors.MliError(err_msg)


# custom JSon serializer (encoder)
class MvResultJSonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.void):
            return None
        else:
            return super().default(obj)


#
# IMPROVE: MV import/export should be moved to MV repository ~ classes being i/e
#


class MvResultPersistence:
    """This class provides portable filesystem export and import of MV test results.
     The design enables easy export of the results as a ZIP archive and import from
     the ZIP (or filesystem structure) to the runtime (MV) data class instances.

     Result data are stored:

     * in the filesystem
     * either as a directory structure or ZIP archive (zipped directory structure)
     * with per-test directory JSon index file representing either a data class
       instance or a dictionary
     * JSon index having per test result field key and value either holding
       the data or pointing to another JSon index file in the ZIP archive/directory
       structure.

    Filesystem structure:

     .. code-block:: text

        MVTest/
            MVTestResults/
                report/
                    AGE/
                        binned_cat_view/
                            index.json
                        ...
                        numerical_view/
                        index.json
                    PAY_ATM6/
                    index.json
                index.json
                psi_scores.csv
            MVTestArtifacts/
                ...
                index.json
            MVTestLog/
                log.json
            MVTestSettings/
                ...
                index.json
            index.json

    """

    FORMAT_DATETIME = "%Y/%d/%m %H:%M:%S.%f"

    KEY_COLUMNS = "_columns"
    KEY_COLUMN_SUMMARIES = "_column_summaries"
    KEY_COL_HISTOGRAM_STATS = "column_histogram_stats"
    KEY_DATA = "data"
    KEY_DIR = "dir"
    KEY_ERROR = "error"
    KEY_FILENAME = "filename"
    KEY_HASH = "hash"
    KEY_LEVEL = "level"
    KEY_MSGS = "messages"
    KEY_MV_FPATH = "mv_fpath"
    KEY_MV_ID = "mvid"
    KEY_MV_NAME = "mv_name"
    KEY_MV_TYPE = "mv_type"
    KEY_NAME = "name"
    KEY_N_COLS = "n_cols"
    KEY_N_ROWS = "n_rows"
    KEY_ORIGIN_OBJ_KEY = "origin_obj_key"
    KEY_PATH = "path"
    KEY_PLATFORM_MVID = "platform_mvid"
    KEY_PLATFORM_OBJ_KEY = "platform_obj_key"
    KEY_SAMPLE_TABLE = "sample_table"
    KEY_SHAPE = "shape"
    KEY_SIZE = "size"
    KEY_SIZE_STR = "size_str"
    KEY_SUMMARY = "_summary"
    KEY_TEXT = "text"
    KEY_TS = "timestamp"
    KEY_TYPE = "type"

    DIR_MVTEST = "MVTest"
    DIR_MVARTIFACTS = "MVTestArtifacts"
    DIR_MVRESULTS = "MVTestResults"
    DIR_MVLOG = "MVTestLog"
    DIR_MVSETTINGS = "MVTestSettings"

    FILE_INDEX = "index.json"

    TYPE_NONE_STR = f"{type(None)}"

    TYPE_SHAPE_STR = "<custom-type 'shape'>"
    TYPE_MV_ARTIFACTS = "<class 'h2o_mv.core.mv_test.MVTestArtifacts'>"
    TYPE_MV_ARTIFACT_INFO = "<class 'h2o_mv.core.mv_test.ArtifactInfo'>"

    __STR2TYPE = {}
    __TYPE2STR = {}
    __PRIMITIVE_TYPES_STR = []
    __NUMPY_PRIMITIVE_TYPES_STR = []

    @staticmethod
    def _supported_types() -> list:
        return [
            str,
            int,
            float,
            bool,
            list,
            tuple,
            dict,
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.float_,
            np.float16,
            np.float32,
            np.float64,
            np.ndarray,
            np.bool_,
            np.void,
            pandas.DataFrame,
            pandas.Series,
            datatable.Frame,
            pandas.Int64Index,
            datetime.datetime,
        ]

    @staticmethod
    def _primitive_types() -> list:
        return [
            None,
            str,
            int,
            float,
            bool,
        ]

    @staticmethod
    def _primitive_types_str() -> list:
        if not MvResultPersistence.__PRIMITIVE_TYPES_STR:
            MvResultPersistence.__PRIMITIVE_TYPES_STR = [
                f"{t}" for t in MvResultPersistence._primitive_types()
            ]
        return MvResultPersistence.__PRIMITIVE_TYPES_STR

    @staticmethod
    def _numpy_primitive_types() -> list:
        return [
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.float_,
            np.float16,
            np.float32,
            np.float64,
            np.bool_,
        ]

    @staticmethod
    def _numpy_primitive_types_str() -> list:
        if not MvResultPersistence.__NUMPY_PRIMITIVE_TYPES_STR:
            MvResultPersistence.__NUMPY_PRIMITIVE_TYPES_STR = [
                f"{t}" for t in MvResultPersistence._numpy_primitive_types()
            ]
        return MvResultPersistence.__NUMPY_PRIMITIVE_TYPES_STR

    @staticmethod
    def _str_2_type() -> dict:
        if not MvResultPersistence.__STR2TYPE:
            MvResultPersistence.__STR2TYPE = {
                f"{t}": t for t in MvResultPersistence._supported_types()
            }
        return MvResultPersistence.__STR2TYPE

    @staticmethod
    def _type_2_str() -> dict:
        if not MvResultPersistence.__TYPE2STR:
            MvResultPersistence.__TYPE2STR = {
                MvResultPersistence._str_2_type()[k]: k
                for k in MvResultPersistence._str_2_type()
            }

        return MvResultPersistence.__TYPE2STR

    @staticmethod
    def split_full_type(type_str: str) -> tuple[str, str]:
        if type_str:
            if type_str.startswith("<class '") and type_str.endswith("'>"):
                type_str = type_str[len("<class '") :]
                type_str = type_str[: -len("'>")]

            if type_str.startswith("<enum '") and type_str.endswith("'>"):
                type_str = type_str[len("<enum '") :]
                type_str = type_str[: -len("'>")]

            if "." in type_str:
                split = type_str.split(".")
                if len(split) >= 2:
                    return ".".join(split[:-1]), split[-1]
                return "", type_str
            else:
                return "", type_str
        return "", ""

    def __init__(
        self, target_dir_path: str | pathlib.Path, mv_client=None, logger=None
    ):
        """Create new instance of the MV result persistence.

        Parameters
        ----------
        target_dir_path : str | pathlib.Path
            Destination directory to export to or import from. Directory might be
            also used as working directory.
        mv_client :
            Optional MV client allowing to get access to MV databases - both object
            store and RDBMS - and use it e.g. to download artifacts.
        logger :
            H2O Sonar logger.

        """
        self.log_name = "MVTest import/export"

        if not target_dir_path:
            raise ValueError(
                f"{self.log_name}: directory for the import/export not specified: "
                f"{target_dir_path}"
            )

        if isinstance(target_dir_path, str):
            target_dir_path = pathlib.Path(target_dir_path)

        if not isinstance(target_dir_path, pathlib.Path):
            raise ValueError(
                f"{self.log_name}: directory for the import/export must be either "
                f"string or pathlib.Path, but it is: {type(target_dir_path)}"
            )
        elif not target_dir_path.exists():
            raise ValueError(
                f"{self.log_name}: directory for the import/export does not exist: "
                f"{target_dir_path}"
            )
        elif not target_dir_path.is_dir():
            raise ValueError(
                f"{self.log_name}: directory for the import/export is not directory: "
                f"{target_dir_path}"
            )

        # target export/import directory
        self.target_dir = None
        self._dir_test = None
        self._dir_result = None
        self._dir_log = None
        self._dir_settings = None
        self._dir_artifacts = None

        self.set_target_dir(target_dir_path)

        self.mv_client = mv_client

        self.logger = logger or loggers.SonarPrintLogger()

    def set_target_dir(self, target_dir_path: pathlib.Path):
        self.target_dir = pathlib.Path(target_dir_path)
        self.set_test_dir_name()

    def set_test_dir_name(self, custom_test_dir: str = DIR_MVTEST):
        self._dir_test = self.target_dir / custom_test_dir

        self._dir_artifacts = self._dir_test / MvResultPersistence.DIR_MVARTIFACTS
        self._dir_log = self._dir_test / MvResultPersistence.DIR_MVLOG
        self._dir_result = self._dir_test / MvResultPersistence.DIR_MVRESULTS
        self._dir_settings = self._dir_test / MvResultPersistence.DIR_MVSETTINGS

    #
    # EXPORT
    #

    @staticmethod
    def _is_type_primitive(type_str: str):
        return type_str in MvResultPersistence._primitive_types()

    @staticmethod
    def _is_dict_primitive(d: dict):
        return all([isinstance(v, (str, int, float, bool)) for v in d.values()])

    @staticmethod
    def _is_list_primitive(ll: list):
        return all([isinstance(v, (str, int, float, bool)) for v in ll])

    @staticmethod
    def _is_tuple_primitive(ll: tuple):
        return all([isinstance(v, (str, int, float, bool)) for v in ll])

    @staticmethod
    def _normalize_numpy_obj(obj):
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.void):
            return None
        # elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
        #    return {'real': obj.real, 'imag': obj.imag}
        else:
            raise TypeError(f"Unknown numpy type to be normalized: {type(obj)}")

    @staticmethod
    def _normalize_numpy_dict_keys(d: dict) -> dict:
        """Convert numpy keys to strings."""
        return {MvResultPersistence._normalize_numpy_obj(k): v for k, v in d.items()}

    @staticmethod
    def _export_dataclass_2_dict(dc: dataclasses.dataclass):
        # note that dataclasses are export to dict as a dict of dicts RECURSIVELY
        return {k: v for k, v in dataclasses.asdict(dc).items()}

    @staticmethod
    def _export_class_2_dict(cls_instance):
        """Export any (non-dataclass) class instance to the dictionary."""
        return {k: v for k, v in cls_instance.__dict__.items()}

    def _export_class(
        self, cls_instance, export_dir_path: pathlib.Path, field_name: str
    ):
        cls_dict = MvResultPersistence._export_class_2_dict(cls_instance)

        exported_cls_path = export_dir_path / field_name
        os.makedirs(exported_cls_path, exist_ok=True)

        self._export_dc_or_dict(
            export_type=f"{type(cls_instance)}",
            export_data=cls_dict,
            export_dir_path=exported_cls_path,
        )

        return exported_cls_path

    @staticmethod
    def _export_enum_value(enum_value: enum.Enum) -> str:
        return enum_value.name

    def _import_enum(
        self,
        field_name: str,
        field_type: str,
        field_value: str,
        index_file_path: str | pathlib.Path,
        cls,
    ):
        enum_value = None
        try:
            (m, t) = MvResultPersistence.split_full_type(field_type)
            enum_cls = getattr(sys.modules[m], t)
            raw_enum_value = enum_cls[field_value]
            if isinstance(raw_enum_value, enum.Enum):
                enum_value = raw_enum_value
        except Exception as ex:
            self.logger.error(
                f"Unable to instantiate unsupported enum data type "
                f"'{field_type}' when importing field '{field_name}' "
                f"of {cls} defined by {index_file_path}: "
                f"{ex}\n{traceback.format_exc()}"
            )

        return enum_value

    @staticmethod
    def _export_save_frame(
        data: pandas.DataFrame | pandas.Series | datatable.Frame,
        field_name: str,
        export_dir_path: str | pathlib.Path,
    ) -> str:
        export_path = os.path.join(str(export_dir_path), f"{field_name}.csv")
        if isinstance(data, (pandas.DataFrame, pandas.Series)):
            # due to loss of float precision bug in Pandas, CSV export is made via
            # datatable:
            # https://stackoverflow.com/questions/12877189/float64-with-pandas-to-csv
            # data.to_csv(export_path, index=False)
            datatable.Frame(data).to_csv(export_path)
        elif isinstance(data, datatable.Frame):
            # due to bug (see above) in Pandas, CSV export is made via datatable
            datatable.Frame(data).to_csv(export_path)
        else:
            raise ValueError(f"Unsupported export data frame type: {type(data)}")
        return export_path

    @staticmethod
    def _import_dataclass_from_dict(dataclass_cls, fields_dict: dict):
        field_set = {f.name for f in dataclasses.fields(dataclass_cls) if f.init}
        filtered_field_dict = {k: v for k, v in fields_dict.items() if k in field_set}
        return dataclass_cls(**filtered_field_dict)

    def _export_create_dirs(self):
        self._dir_artifacts.mkdir(parents=True, exist_ok=True)
        self._dir_log.mkdir(parents=True, exist_ok=True)
        self._dir_result.mkdir(parents=True, exist_ok=True)
        self._dir_settings.mkdir(parents=True, exist_ok=True)

    def _export_save_dict(
        self,
        data_as_dict: dict,
        field_name: str,
        export_dir_path: str | pathlib.Path,
    ) -> str:
        export_path_recursion = os.path.join(str(export_dir_path), field_name)
        os.makedirs(export_path_recursion, exist_ok=True)

        self._export_dc_or_dict(
            export_type=MvResultPersistence._type_2_str()[dict],
            export_data=data_as_dict,
            export_dir_path=export_path_recursion,
        )

        return export_path_recursion

    def _export_save_list(
        self,
        data: list,
        field_name: str,
        export_dir_path: str | pathlib.Path,
        export_type: str,
    ) -> str:
        export_path_recursion = os.path.join(str(export_dir_path), field_name)
        os.makedirs(export_path_recursion, exist_ok=True)

        # turn list into dict and export
        list_as_dict = {str(i): v for i, v in enumerate(data)}

        self._export_dc_or_dict(
            export_type=export_type,
            export_data=list_as_dict,
            export_dir_path=export_path_recursion,
        )

        return export_path_recursion

    @staticmethod
    def _export_save_index(export_dir_path, idx_dict: dict):
        # save index file
        idx_json_path = os.path.join(
            str(export_dir_path), MvResultPersistence.FILE_INDEX
        )

        try:
            with open(idx_json_path, "w") as file_handle:
                json.dump(idx_dict, file_handle, indent=4, cls=MvResultJSonEncoder)
        except Exception as ex:
            idx_dict_pprint = pprint.pformat(idx_dict, indent=4)
            err_msg = (
                f"Index file {idx_json_path} file serialization failed: "
                f"{ex}\n{traceback.format_exc()}\n\n{idx_dict_pprint}"
            )
            raise RuntimeError(err_msg)

    def _export_dc_or_dict(
        self,
        export_type: str,
        export_data: dict | Any,
        export_dir_path: str | pathlib.Path,
    ) -> tuple[str | pathlib.Path, dict]:
        """Export data class or dictionary to the filesystem."""
        if not HAS_H2O_MV:
            raise ImportError(
                f"{commons.opt_import_err_msg(PACKAGE_NAME)} - NOT COMPATIBLE"
            )

        cls = MvResultPersistence

        if not isinstance(export_dir_path, pathlib.Path):
            export_dir_path = pathlib.Path(export_dir_path)

        # index dictionary for the incoming dictionary / data class
        idx_dict = {
            MvResultPersistence.KEY_TYPE: export_type,
            MvResultPersistence.KEY_DATA: {},
        }

        # handle (top level) MV types which are support
        if isinstance(export_data, h2o_mv.core.mv_test.MVTestLog):
            data_dict = MvResultPersistence._export_mv_test_log_to_dict(export_data)
            MvResultPersistence._export_save_index(
                export_dir_path=export_dir_path, idx_dict=data_dict
            )
            return export_dir_path, data_dict

        # ensure data representation as a dictionary
        if isinstance(export_data, dict):
            data_dict = export_data
        elif dataclasses.is_dataclass(export_data):
            data_dict = MvResultPersistence._export_dataclass_2_dict(export_data)
        else:
            raise RuntimeError(f"Unsupported type for the export: {type(export_data)}")

        for k in data_dict.keys():
            # per-type serialization
            v = data_dict[k]

            if v is None:
                idx_dict[MvResultPersistence.KEY_DATA][k] = cls._export_none()
            elif isinstance(v, enum.Enum):
                # enum must be tested BEFORE other types (multiple parents cls e.g. str)
                idx_dict[MvResultPersistence.KEY_DATA][k] = cls._export_enum(v)
            elif isinstance(v, (str, int, float, bool)):
                idx_dict[MvResultPersistence.KEY_DATA][k] = cls._export_primitive_val(v)
            elif isinstance(v, datetime.datetime):
                idx_dict[MvResultPersistence.KEY_DATA][k] = cls._export_datetime(v)
            elif isinstance(
                v,
                (
                    np.int_,
                    np.intc,
                    np.intp,
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,
                    np.uint8,
                    np.uint16,
                    np.uint32,
                    np.uint64,
                ),
            ):
                idx_dict[MvResultPersistence.KEY_DATA][k] = {
                    MvResultPersistence.KEY_TYPE: f"{type(v)}",
                    MvResultPersistence.KEY_DATA: int(v),
                }
            elif isinstance(v, (np.float_, np.float16, np.float32, np.float64)):
                idx_dict[MvResultPersistence.KEY_DATA][k] = {
                    MvResultPersistence.KEY_TYPE: f"{type(v)}",
                    MvResultPersistence.KEY_DATA: float(v),
                }
            elif isinstance(v, (pandas.DataFrame, pandas.Series, datatable.Frame)):
                idx_dict[MvResultPersistence.KEY_DATA][k] = cls._export_frame(
                    v=v, field_name=k, export_dir_path=export_dir_path
                )
            elif isinstance(v, pandas.Int64Index):
                idx_dict[MvResultPersistence.KEY_DATA][k] = {
                    MvResultPersistence.KEY_TYPE: f"{type(v)}",
                    MvResultPersistence.KEY_DATA: v.to_list(),
                }
            elif isinstance(v, dict):
                self._export_dict(
                    k=k,
                    v=v,
                    idx_dict=idx_dict[MvResultPersistence.KEY_DATA],
                    export_dir_path=export_dir_path,
                )
            elif isinstance(v, list):
                self._export_list(
                    k=k,
                    v=v,
                    idx_dict=idx_dict[MvResultPersistence.KEY_DATA],
                    export_dir_path=export_dir_path,
                )
            elif isinstance(v, tuple):
                self._export_tuple(
                    k=k,
                    v=v,
                    idx_dict=idx_dict[MvResultPersistence.KEY_DATA],
                    export_dir_path=export_dir_path,
                )
            elif dataclasses.is_dataclass(v):
                # recursively export data class
                field_export_dir_path = export_dir_path / k
                field_export_dir_path.mkdir(parents=True, exist_ok=True)
                self._export_dc_or_dict(
                    export_type=f"{type(v)}",
                    export_data=v,
                    export_dir_path=field_export_dir_path,
                )
                idx_dict[MvResultPersistence.KEY_DATA][k] = {
                    MvResultPersistence.KEY_TYPE: f"{type(v)}",
                    MvResultPersistence.KEY_FILENAME: k,
                }
            # generic non-dataclass class instance export - known types ONLY
            else:
                if not isinstance(
                    v,
                    (
                        h2o_mv.core.mv_model.MVModel,
                        h2o_mv.core.mv_dataset.MVDataset,
                        h2o_mv.platforms.summaries.ColumnSummaries,
                    ),
                ):
                    self.logger.error(
                        f"Unsupported export data type - using GENERIC JSon "
                        f"export / JSon serializer for: {type(v)}"
                    )

                # recursively export custom class
                self._export_class(
                    cls_instance=v,
                    export_dir_path=export_dir_path,
                    field_name=k,
                )
                # reference subtype index from the current index
                idx_dict[MvResultPersistence.KEY_DATA][k] = {
                    MvResultPersistence.KEY_TYPE: f"{type(v)}",
                    MvResultPersistence.KEY_DIR: k,
                }

        # ensure serializable keys and values
        idx_dict[MvResultPersistence.KEY_DATA] = (
            MvResultPersistence._normalize_numpy_dict_keys(
                idx_dict[MvResultPersistence.KEY_DATA]
            )
        )

        MvResultPersistence._export_save_index(
            export_dir_path=export_dir_path, idx_dict=idx_dict
        )

        return export_dir_path, idx_dict

    @staticmethod
    def _export_none(v=None) -> dict | None:
        if v is None:
            return {
                MvResultPersistence.KEY_TYPE: f"{type(v)}",
                MvResultPersistence.KEY_DATA: v,
            }
        return None

    @staticmethod
    def _export_primitive_val(v: str | int | float | bool):
        return MvResultPersistence._export_none(v) or {
            MvResultPersistence.KEY_TYPE: f"{type(v)}",
            MvResultPersistence.KEY_DATA: v,
        }

    @staticmethod
    def _export_enum(v):
        if v is not None:
            enum_type = f"{type(v)}"
            enum_type = enum_type.replace("enum '", f"enum '{type(v).__module__}.")
            return {
                MvResultPersistence.KEY_TYPE: enum_type,
                MvResultPersistence.KEY_DATA: MvResultPersistence._export_enum_value(v),
            }
        return MvResultPersistence._export_none()

    @staticmethod
    def _export_datetime(v: datetime):
        return MvResultPersistence._export_none(v) or {
            MvResultPersistence.KEY_TYPE: f"{type(v)}",
            MvResultPersistence.KEY_DATA: v.strftime(
                MvResultPersistence.FORMAT_DATETIME
            ),
        }

    @staticmethod
    def _export_primitive_list(v: list):
        return MvResultPersistence._export_none(v) or {
            MvResultPersistence.KEY_TYPE: f"{type(v)}",
            MvResultPersistence.KEY_DATA: v,
        }

    @staticmethod
    def _export_primitive_tuple(v: tuple):
        return MvResultPersistence._export_none(v) or {
            MvResultPersistence.KEY_TYPE: f"{type(v)}",
            MvResultPersistence.KEY_DATA: v,
        }

    @staticmethod
    def _export_primitive_dict(v: dict):
        cls = MvResultPersistence
        return cls._export_none(v) or {
            cls.KEY_TYPE: f"{type(v)}",
            cls.KEY_DATA: cls._normalize_numpy_dict_keys(v),
        }

    def _export_list(self, k, v: list, idx_dict: dict, export_dir_path: pathlib.Path):
        if MvResultPersistence._is_list_primitive(v):
            # list of primitive values
            idx_dict[k] = MvResultPersistence._export_primitive_list(v)
        else:
            # list of non-primitive values
            export_path = self._export_save_list(
                data=v,
                field_name=k,
                export_dir_path=export_dir_path,
                export_type=MvResultPersistence._type_2_str()[list],
            )
            idx_dict[k] = {
                MvResultPersistence.KEY_TYPE: f"{type(v)}",
                MvResultPersistence.KEY_FILENAME: os.path.basename(export_path),
            }

    def _export_tuple(self, k, v: tuple, idx_dict: dict, export_dir_path: pathlib.Path):
        if MvResultPersistence._is_tuple_primitive(v):
            # list of primitive values
            idx_dict[k] = MvResultPersistence._export_primitive_tuple(v)
        else:
            # list of non-primitive values
            export_path = self._export_save_list(
                data=list(v),
                field_name=k,
                export_dir_path=export_dir_path,
                export_type=MvResultPersistence._type_2_str()[tuple],
            )
            idx_dict[k] = {
                MvResultPersistence.KEY_TYPE: f"{type(v)}",
                MvResultPersistence.KEY_FILENAME: os.path.basename(export_path),
            }

    def _export_dict(self, k, v: dict, idx_dict: dict, export_dir_path: pathlib.Path):
        if MvResultPersistence._is_dict_primitive(v):
            # list of primitive values
            idx_dict[k] = MvResultPersistence._export_primitive_dict(v)
        else:
            # list of non-primitive values
            export_path = self._export_save_dict(
                data_as_dict=v, field_name=k, export_dir_path=export_dir_path
            )
            idx_dict[k] = {
                MvResultPersistence.KEY_TYPE: f"{type(v)}",
                MvResultPersistence.KEY_FILENAME: os.path.basename(export_path),
            }

    @staticmethod
    def _export_shape(v: tuple):
        return MvResultPersistence._export_none(v) or {
            MvResultPersistence.KEY_TYPE: MvResultPersistence.TYPE_SHAPE_STR,
            MvResultPersistence.KEY_DATA: v,
        }

    @staticmethod
    def _export_frame(
        v: pandas.DataFrame | pandas.Series | datatable.Frame,
        field_name: str,
        export_dir_path: pathlib.Path,
    ):
        if v is not None:
            export_path = MvResultPersistence._export_save_frame(
                data=v, field_name=field_name, export_dir_path=export_dir_path
            )
            return {
                MvResultPersistence.KEY_TYPE: f"{type(v)}",
                MvResultPersistence.KEY_FILENAME: os.path.basename(export_path),
            }
        return MvResultPersistence._export_none()

    @staticmethod
    def _export_mv_test_log_msg_to_dict(mv_test_log_msg):
        cls = MvResultPersistence
        return {
            cls.KEY_TEXT: cls._export_primitive_val(mv_test_log_msg.text),
            cls.KEY_LEVEL: cls._export_enum(mv_test_log_msg.level),
            cls.KEY_TS: cls._export_datetime(mv_test_log_msg.timestamp),
        }

    @staticmethod
    def _export_mv_test_log_to_dict(mv_test_log) -> dict:
        messages = []
        idx_dict = {
            MvResultPersistence.KEY_TYPE: f"{type(mv_test_log)}",
            MvResultPersistence.KEY_DATA: {
                MvResultPersistence.KEY_MSGS: messages,
            },
        }
        for m in mv_test_log.messages:
            messages.append(MvResultPersistence._export_mv_test_log_msg_to_dict(m))

        return idx_dict

    def export_mv_test(
        self,
        mv_test_type: str,
        mv_test_name: str,
        mv_test_id: str,
        mv_test_results=None,
        mv_test_settings=None,
        mv_test_artifacts: dict | None = None,
        mv_test_log=None,
        export_dir_name: str = DIR_MVTEST,
        fail_fast: bool = False,
    ):
        """Export instances created by the MVTest.

        Parameters
        ----------
        mv_test_type : str
            Python type of the MV test.
        mv_test_name : str
            Name of the MV test.
        mv_test_id : str
            ID of the MV test.
        mv_test_results :
            MV test results - ``h2o_mv.core.mv_test.MVTestResult``.
        mv_test_settings :
            MV test settings - ``h2o_mv.core.mv_test.MVTestSettings``.
        mv_test_artifacts : dict | None
            Artifacts created by the MV test - dictionary of
            artifact name to ``h2o_mv.core.mv_test.ArtifactInfo``.
        mv_test_log :
            MV test log - ``h2o_mv.core.mv_test.MVTestLog``.
        export_dir_name : str
            A name of the directory where the result will be exported. The directory
            will be created in the ``export_dir_path``.
        fail_fast : bool
            Don't be robust, but throw an exception on the first error.

        Returns
        -------
        Tuple[str | pathlib.Path, dict] :
            Directory with MV test outputs saved on the filesystem and a dictionary
            with the index file content.

        """
        if not HAS_H2O_MV:
            raise ImportError(
                f"{commons.opt_import_err_msg(PACKAGE_NAME)} - NOT COMPATIBLE"
            )

        self.set_test_dir_name(export_dir_name)

        self._export_create_dirs()

        cls = MvResultPersistence
        idx_dict = {
            cls.KEY_TYPE: mv_test_type or "",
            cls.KEY_NAME: mv_test_name or "",
            cls.KEY_MV_ID: mv_test_id or "",
            cls.KEY_DATA: {
                cls.DIR_MVLOG: {},
                cls.DIR_MVRESULTS: {},
                cls.DIR_MVSETTINGS: {},
                cls.DIR_MVARTIFACTS: {},
            },
        }

        def __handle_export_error(err_msg: str, key: str, exc):
            err_msg = f"{err_msg}: {exc}\n{traceback.format_exc()}"
            self.logger.error(err_msg)
            idx_dict[cls.KEY_DATA][key] = {
                cls.KEY_ERROR: err_msg,
            }
            if fail_fast:
                raise exc

        if mv_test_log:
            try:
                self._export_mv_class(
                    mv_test_output=mv_test_log, export_dir_path=self._dir_log
                )
                idx_dict[cls.KEY_DATA][cls.DIR_MVLOG] = {
                    cls.KEY_TYPE: f"{type(mv_test_log)}",
                    cls.KEY_DIR: cls.DIR_MVLOG,
                }
            except Exception as ex:
                __handle_export_error(
                    err_msg="Unable to export MV test log", key=cls.DIR_MVLOG, exc=ex
                )
        if mv_test_results:
            try:
                self._export_mv_class(
                    mv_test_output=mv_test_results, export_dir_path=self._dir_result
                )
                idx_dict[cls.KEY_DATA][cls.DIR_MVRESULTS] = {
                    cls.KEY_TYPE: f"{type(mv_test_results)}",
                    cls.KEY_DIR: cls.DIR_MVRESULTS,
                }
            except Exception as ex:
                __handle_export_error(
                    err_msg="Unable to export MV test results",
                    key=cls.DIR_MVRESULTS,
                    exc=ex,
                )
        if mv_test_settings:
            try:
                self._export_mv_class(
                    mv_test_output=mv_test_settings, export_dir_path=self._dir_settings
                )
                idx_dict[cls.KEY_DATA][cls.DIR_MVSETTINGS] = {
                    cls.KEY_TYPE: f"{type(mv_test_settings)}",
                    cls.KEY_DIR: cls.DIR_MVSETTINGS,
                }
            except Exception as ex:
                __handle_export_error(
                    err_msg="Unable to export MV test settings",
                    key=cls.DIR_MVSETTINGS,
                    exc=ex,
                )
        if mv_test_artifacts:
            if not self.mv_client:
                if self.logger:
                    self.logger.error(
                        f"Unable to export MV artifacts ({mv_test_artifacts}) as "
                        f"MV client required to access MV Object store was not set"
                    )
            else:
                try:
                    self._export_mv_artifacts(
                        mv_test_artifacts=mv_test_artifacts,
                        export_dir_path=self._dir_artifacts,
                    )
                    idx_dict[cls.KEY_DATA][cls.DIR_MVARTIFACTS] = {
                        cls.KEY_TYPE: f"{mv_test.MVTestArtifacts}",
                        cls.KEY_DIR: cls.DIR_MVARTIFACTS,
                    }
                except Exception as ex:
                    __handle_export_error(
                        err_msg="Unable to export MV test artifacts",
                        key=cls.DIR_MVARTIFACTS,
                        exc=ex,
                    )

        self._export_save_index(export_dir_path=self._dir_test, idx_dict=idx_dict)

        return self._dir_test, idx_dict

    def _export_mv_class(
        self,
        mv_test_output,
        export_dir_path: pathlib.Path,
    ) -> tuple[str | pathlib.Path, dict]:
        """Export an MV test output instance, like ``MVTestResult``, to a directory
        as an index file (JSon) and a hierarchy of directories and files.

        The index file contains a list of all files and their types. Data are
        serialized as follows:

        - primitive type (str, int, float and bool)
          is stored in the INDEX file as ``data`` key value.
        - data frame (Pandas, Datatable)
          is stored as a CSV file.
        - list of primitive types (str, int, float and bool)
          is stored in the index file as ``data`` key value.
        - dictionary of primitive types (str, int, float and bool)
          is stored in the index file as ``data`` key value.
        - dictionary or list of non-primitive types
          is stored to a subdirectory. Each dictionary/list item is stored in
          a separate file (if non-primitive) or as ``data`` key value (if primitive
          type).

        Parameters
        ----------
        mv_test_output : MvResultPersistence | dataclasses.dataclass | dict
            A model validation test output.

        Returns
        -------
        Tuple[str | pathlib.Path, dict] :
            Path to the stored MV test output and a dictionary with the index file
            content.

        """

        return self._export_dc_or_dict(
            export_type=f"{type(mv_test_output)}",
            export_dir_path=export_dir_path,
            export_data=mv_test_output,
        )

    def _export_mv_artifacts(
        self,
        mv_test_artifacts: dict | None,
        export_dir_path: pathlib.Path,
    ) -> tuple:
        """Export MV artifacts by creating JSon index and downloading artifacts
        one by one to files. MV artifacts are created by the following tests:

        - Adversarial MVTest

        Parameters
        ----------
        mv_test_artifacts : dict | None
            MV test artifacts metadata - dictionary of
            ``str`` (name) to ``h2o_mv.core.mv_test.ArtifactInfo``.
        export_dir_path : pathlib.Path
            Base directory for the artifacts export.

        Returns
        -------
        Tuple[pathlib.Path, dict] :
            Path to the artifacts directory and index represented as the dictionary

        """
        idx_dict = {
            MvResultPersistence.KEY_TYPE: MvResultPersistence.TYPE_MV_ARTIFACTS,
            MvResultPersistence.KEY_DATA: [],
        }
        mv_test_artifacts = mv_test_artifacts or {}
        for name in mv_test_artifacts:
            export_file_name = str(uuid.uuid4())
            export_file_path = export_dir_path / export_file_name
            a_dict = {
                MvResultPersistence.KEY_TYPE: MvResultPersistence.TYPE_MV_ARTIFACT_INFO,
                # MV ArtifactInfo attributes
                MvResultPersistence.KEY_MV_NAME: mv_test_artifacts[name].name,
                MvResultPersistence.KEY_MV_FPATH: mv_test_artifacts[name].fpath,
                MvResultPersistence.KEY_MV_ID: mv_test_artifacts[name].obj_mvid,
                # file used to store the artifact
                MvResultPersistence.KEY_FILENAME: export_file_name,
            }
            # download the data
            try:
                self.mv_client.mvdb.obj_storage.download(
                    mvid=mv_test_artifacts[name].obj_mvid,
                    dst_path=export_file_path,
                )
            except Exception as ex:
                self.logger.error(
                    f"Unable to export artifact data/file for the artifact {name}: {ex}"
                    f"\n{traceback.format_exc()}"
                )
                # skip this artifact, but don't fail, just report an error
                continue

            # add new artifact
            idx_dict[MvResultPersistence.KEY_DATA].append(a_dict)

        # save index file
        MvResultPersistence._export_save_index(
            export_dir_path=export_dir_path,
            idx_dict=idx_dict,
        )

        return self._dir_artifacts, idx_dict

    #
    # IMPORT
    #

    def _import_mv_artifacts(self, import_dir_path: pathlib.Path) -> dict:
        """Import MV test artifacts and return artifact info for every uploaded
        artifacts.

        Parameters
        ----------
        import_dir_path : pathlib.Path
            Directory with MV test artifacts to be imported.

        Returns
        -------
        dict :
            Dictionary of artifact name to ``h2o_mv.core.mv_test.ArtifactInfo`` for
            successfully uploaded artifacts.

        """
        if not HAS_H2O_MV:
            raise ImportError(
                f"{commons.opt_import_err_msg(PACKAGE_NAME)} - NOT COMPATIBLE"
            )

        index_file_path = import_dir_path / MvResultPersistence.FILE_INDEX
        if not index_file_path.exists():
            raise ValueError(
                f"MV test artifacts index file cannot be imported as it does not exist:"
                f" {import_dir_path}"
            )

        with open(index_file_path) as file_handle:
            try:
                idx_dict = json.load(file_handle)
            except Exception as ex:
                raise RuntimeError(
                    f"MV test artifacts JSon file '{index_file_path}' parsing failed: "
                    f"{ex}\n{traceback.format_exc()}"
                )

        if f"{mv_test.MVTestArtifacts}" != idx_dict.get(
            MvResultPersistence.KEY_TYPE, ""
        ):
            raise ValueError(
                f"MV test artifacts index file cannot be imported as it does not "
                f"contain MV test artifacts data - type mismatch "
                f"'{idx_dict.get(MvResultPersistence.KEY_TYPE)}' in {index_file_path}"
            )

        mv_result = {}
        for a_dict in idx_dict.get(MvResultPersistence.KEY_DATA, []):
            if f"{mv_test.ArtifactInfo}" != a_dict.get(
                MvResultPersistence.KEY_TYPE, ""
            ):
                self.logger.error(
                    f"MV test artifact info cannot be imported - type mismatch "
                    f"'{a_dict.get(MvResultPersistence.KEY_TYPE)}' in {index_file_path}"
                )
                continue

            a = mv_test.ArtifactInfo(
                name=a_dict.get(MvResultPersistence.KEY_MV_NAME, ""),
                fpath=a_dict.get(MvResultPersistence.KEY_MV_FPATH, ""),
                obj_mvid=a_dict.get(MvResultPersistence.KEY_MV_ID, ""),
            )

            filename = a_dict.get(MvResultPersistence.KEY_FILENAME)
            if filename:
                file_path = import_dir_path / filename
                if file_path.exists():
                    # upload the data
                    try:
                        self.mv_client.mvdb.obj_storage.upload(
                            src_path=file_path,
                            mvid=a.obj_mvid,
                        )
                    except Exception as ex:
                        self.logger.error(
                            f"Unable to import artifact data/file for the artifact "
                            f"{a.name}: {ex}"
                            f"\n{traceback.format_exc()}"
                        )
                        # skip this artifact, but don't fail, just report an error
                        continue

                    mv_result[a.name] = a
                else:
                    self.logger.error(
                        f"MV test artifact cannot be imported - missing file "
                        f"'{file_path}' in {index_file_path}"
                    )
            else:
                self.logger.error(
                    f"MV test artifact cannot be imported - missing file name "
                    f" in {index_file_path}"
                )

        return mv_result

    def _is_import_dir(self):
        root_idx_path = self._dir_test / MvResultPersistence.FILE_INDEX
        if (
            self._dir_test.exists()
            and (
                self._dir_artifacts.exists()
                or self._dir_log.exists()
                or self._dir_result.exists()
                or self._dir_settings.exists()
            )
            and root_idx_path.exists()
        ):
            return True

        return False

    def import_mv_test(
        self, zip_path: str | pathlib.Path = "", fail_fast: bool = False
    ):
        """Import MVTest related objects - results, settings, artifacts and log.

        Parameters
        ----------
        zip_path : str | pathlib.Path
          Path to the ZIP archive with the MV export is stored.
        fail_fast : bool
            Don't be robust, but throw an exception on the first error.

        Returns
        -------
        dict :
            A dictionary with imported instances.

        """
        # if ZIP archive path specified, then extract ZIP to the working directory
        if zip_path:
            zip_path = pathlib.Path(zip_path)
            if zip_path.is_file():
                with zipfile.ZipFile(zip_path, "r") as zip_file:
                    zip_file.extractall(self.target_dir)
            else:
                raise ValueError(
                    f"ZIP archive path '{zip_path}' is not a file or does not exist"
                )

        if not self._is_import_dir():
            raise ValueError(
                f"Directory {self._dir_test} does not contain MV test export "
                f"directory structure with {MvResultPersistence.DIR_MVTEST} directory "
                f"and per-test field subdirectories for results, settings, artifacts "
                f"and log."
            )

        cls = MvResultPersistence

        # load index file to get name, type, MV ID, ...
        root_idx_path = self._dir_test / MvResultPersistence.FILE_INDEX
        with open(root_idx_path) as file_handle:
            try:
                idx_dict = json.load(file_handle)
            except Exception as ex:
                raise RuntimeError(
                    f"JSon file '{root_idx_path}' parsing failed: "
                    f"{ex}\n{traceback.format_exc()}"
                )

        results_dict = {
            cls.KEY_TYPE: idx_dict.get(cls.KEY_TYPE, ""),
            cls.KEY_NAME: idx_dict.get(cls.KEY_NAME, ""),
            cls.KEY_MV_ID: idx_dict.get(cls.KEY_MV_ID, ""),
            cls.KEY_DATA: {
                cls.DIR_MVRESULTS: {},
                cls.DIR_MVSETTINGS: {},
                cls.DIR_MVARTIFACTS: {},
                cls.DIR_MVLOG: {},
            },
        }

        def __handle_import_error(err_msg: str, mv_test_field_key: str, exc):
            err_msg = f"{err_msg}: {exc}\n{traceback.format_exc()}"
            self.logger.warning(err_msg)
            results_dict[cls.KEY_DATA][mv_test_field_key][cls.KEY_ERROR] = err_msg
            if fail_fast:
                raise exc

        if self._dir_log.exists():
            key = cls.DIR_MVLOG
            try:
                obj = self._import_mv_test_log(self._dir_log)
                results_dict[cls.KEY_DATA][key] = {
                    cls.KEY_TYPE: f"{type(obj)}",
                    cls.KEY_DATA: obj,
                }
            except Exception as ex:
                __handle_import_error(
                    err_msg="Unable to import MV test log",
                    mv_test_field_key=key,
                    exc=ex,
                )

        if self._dir_result.exists():
            key = cls.DIR_MVRESULTS
            try:
                obj = self._import_result_or_settings(
                    self._dir_result, dir_in_zip=cls.DIR_MVRESULTS
                )
                results_dict[cls.KEY_DATA][key] = {
                    cls.KEY_TYPE: f"{type(obj)}",
                    cls.KEY_DATA: obj,
                }
            except Exception as ex:
                __handle_import_error(
                    err_msg="Unable to import MV test results",
                    mv_test_field_key=key,
                    exc=ex,
                )

        if self._dir_settings.exists():
            key = cls.DIR_MVSETTINGS
            try:
                obj = self._import_result_or_settings(
                    self._dir_settings, dir_in_zip=cls.DIR_MVSETTINGS
                )
                results_dict[cls.KEY_DATA][key] = {
                    cls.KEY_TYPE: f"{type(obj)}",
                    cls.KEY_DATA: obj,
                }
            except Exception as ex:
                __handle_import_error(
                    err_msg="Unable to import MV test settings",
                    mv_test_field_key=key,
                    exc=ex,
                )

        if self._dir_artifacts.exists():
            if not self.mv_client:
                if self.logger:
                    self.logger.error(
                        f"Unable to import MV artifacts from {self._dir_artifacts} as "
                        f"MV client required to access MV Object store was not set "
                        f"and therefore artifacts cannot be uploaded to the MV Object "
                        f"store"
                    )
            else:
                key = cls.DIR_MVARTIFACTS
                try:
                    obj = self._import_mv_artifacts(self._dir_artifacts)
                    results_dict[cls.KEY_DATA][key] = {
                        cls.KEY_TYPE: f"{type(obj)}",
                        cls.KEY_DATA: obj,
                    }
                except Exception as ex:
                    __handle_import_error(
                        err_msg="Unable to import MV test artifacts",
                        mv_test_field_key=key,
                        exc=ex,
                    )

        return results_dict

    def _import_mv_test_log(self, import_dir_path: pathlib.Path):
        if not HAS_H2O_MV:
            raise ImportError(
                f"{commons.opt_import_err_msg(PACKAGE_NAME)} - NOT COMPATIBLE"
            )

        index_file_path = import_dir_path / MvResultPersistence.FILE_INDEX
        if not index_file_path.exists():
            raise ValueError(
                f"MV test log index file cannot be imported as it does not exist:"
                f" {import_dir_path}"
            )

        with open(index_file_path) as file_handle:
            try:
                idx_dict = json.load(file_handle)
            except Exception as ex:
                raise RuntimeError(
                    f"MV test log JSon file '{index_file_path}' parsing failed: "
                    f"{ex}\n{traceback.format_exc()}"
                )

        if f"{mv_test.MVTestLog}" != idx_dict.get(MvResultPersistence.KEY_TYPE, ""):
            raise ValueError(
                f"MV test log index file cannot be imported as it does not contain "
                f"MV test log data - type mismatch "
                f"'{idx_dict.get(MvResultPersistence.KEY_TYPE)}' in {index_file_path}"
            )

        msgs = idx_dict.get(MvResultPersistence.KEY_DATA, {}).get(
            MvResultPersistence.KEY_MSGS, {}
        )
        if not msgs:
            raise ValueError(
                f"MV test log index file cannot be imported as it does not contain "
                f"messages in {index_file_path}"
            )

        mv_result = mv_test.MVTestLog()
        mv_result.messages = []
        for m in msgs:
            log_msg = mv_test.MVTestLogMessage(
                text=m.get(MvResultPersistence.KEY_TEXT, {}).get(
                    MvResultPersistence.KEY_DATA, ""
                ),
            )

            f_descr = m.get(MvResultPersistence.KEY_LEVEL, {})
            try:
                log_msg.level = self._import_enum(
                    field_name=MvResultPersistence.DIR_MVLOG,
                    field_type=f_descr.get(
                        MvResultPersistence.KEY_TYPE,
                        f"{mv_test.MVTestLogLevel.__module__}."
                        f"{mv_test.MVTestLogLevel.__name__}",
                    ),
                    field_value=f_descr.get(
                        MvResultPersistence.KEY_DATA, mv_test.MVTestLogLevel.DEBUG.name
                    ),
                    index_file_path=index_file_path,
                    cls=mv_result,
                )
            except Exception as ex:
                self.logger.error(
                    f"Unable to import MV test log message level: {ex}"
                    f"\n{traceback.format_exc()}"
                )
                log_msg.level = mv_test.MVTestLogLevel.DEBUG

            f_descr = m.get(MvResultPersistence.KEY_TS, {})
            try:
                log_msg.timestamp = datetime.datetime.strptime(
                    f_descr.get(MvResultPersistence.KEY_DATA, ""),
                    MvResultPersistence.FORMAT_DATETIME,
                )
            except Exception as ex:
                self.logger.error(
                    f"Unable to import MV test log message timestamp "
                    f"'{m.get(MvResultPersistence.KEY_DATA)}': {ex}"
                    f"\n{traceback.format_exc()}"
                )
                # default timestamp is now

            mv_result.messages.append(log_msg)

        return mv_result

    def _import_result_or_settings(
        self, zip_or_dir_path: str | pathlib.Path, dir_in_zip: str = DIR_MVRESULTS
    ) -> dataclasses.dataclass:
        """Import ``MVTestResults`` or ``MVTestSettings`` instance from a ZIP archive
        or a directory. Create a ``MVTestResult`` or a ``MVTestSettings`` instance
        from the data.

        Parameters
        ----------
        zip_or_dir_path : str | pathlib.Path
          Path to the ZIP archive or a directory where the MV result is stored
          (directory must contain the ``index.json`` file).
        dir_in_zip : str
            Directory in the ZIP archive where the MV result or settings are stored.

        Returns
        -------
        dataclasses.dataclass :
          A ``MVTestResult`` or ``MVTestSettings`` instance.

        """
        input_path = pathlib.Path(zip_or_dir_path)
        if input_path.is_dir():
            return self._import_from_dir(input_path)
        elif input_path.is_file():
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_dir_path = pathlib.Path(tmpdir)
                with zipfile.ZipFile(input_path, "r") as zip_file:
                    zip_file.extractall(tmp_dir_path)
                    import_path = (
                        tmp_dir_path / MvResultPersistence.DIR_MVTEST / dir_in_zip
                    )
                    try:
                        return self._import_from_dir(import_path)
                    except Exception as ex:
                        raise ex
        else:
            raise ValueError(f"Unsupported input path: {input_path}")

    def _import_load_dict(
        self,
        base_dir: pathlib.Path,
        sub_dir_name: str,
    ) -> dict:
        # recursive load
        return self._import_dc_or_dict(
            base_dir=base_dir / sub_dir_name,
            force_type=self._type_2_str()[dict],
        )

    def _import_load_list(self, import_dir_path: pathlib.Path) -> list:
        # recursive load
        return self._import_dc_or_dict(
            base_dir=import_dir_path,
            force_type=self._type_2_str()[list],
        )

    def _import_load_tuple(self, import_dir_path: pathlib.Path) -> tuple:
        # recursive load
        return self._import_dc_or_dict(
            base_dir=import_dir_path,
            force_type=self._type_2_str()[tuple],
        )

    def _import_dc_or_dict(self, base_dir: pathlib.Path, force_type=None):
        index_file_path = base_dir / MvResultPersistence.FILE_INDEX
        if not index_file_path.exists():
            raise ValueError(f"Index file does not exist: {index_file_path}")

        with open(index_file_path) as file_handle:
            try:
                index_dict = json.load(file_handle)
            except Exception as ex:
                raise RuntimeError(
                    f"JSon file '{index_file_path}' parsing failed: "
                    f"{ex}\n{traceback.format_exc()}"
                )

        # instantiate type for which has been the index file created
        full_type_name = index_dict[MvResultPersistence.KEY_TYPE]
        (package_name, type_name) = MvResultPersistence.split_full_type(full_type_name)

        if force_type is not None:
            assert force_type == full_type_name, (
                f"Forced type mismatch (forced/actual): "
                f"{force_type} vs. {full_type_name}"
            )

        self.logger.debug(f"Importing MV result type: {full_type_name}")
        if self._type_2_str()[dict] == full_type_name:
            cls = dict
        elif self._type_2_str()[list] == full_type_name:
            cls = list
        elif self._type_2_str()[tuple] == full_type_name:
            cls = tuple
        else:
            # OPTIONAL: force import to ensure smooth dynamic class instantiation
            _ = [
                h2o_mv.recipes.adversarial.results,
                h2o_mv.recipes.backtesting.results,
                h2o_mv.recipes.calibscore.results,
                h2o_mv.recipes.drift.results,
                h2o_mv.recipes.segperf.results,
                h2o_mv.recipes.sizedep.results,
            ]

            cls = getattr(sys.modules[package_name], type_name)

        self.logger.debug(f"MV result OR settings class: {cls}")
        obj = cls()

        # data (fields) are stored in the index file
        if not index_dict.get(MvResultPersistence.KEY_DATA):
            raise ValueError(
                f"Index file {index_file_path} does not contain "
                f"'{MvResultPersistence.KEY_DATA}' key"
            )
        for field_name in index_dict[MvResultPersistence.KEY_DATA].keys():
            self.logger.debug(f"  Injecting '{field_name}' field to {cls}...")

            # resolve & instantiate data value
            field_descr = index_dict[MvResultPersistence.KEY_DATA][field_name]
            field_type = field_descr.get(MvResultPersistence.KEY_TYPE)
            if field_type == MvResultPersistence.TYPE_NONE_STR:
                field_value = None
            elif field_type in MvResultPersistence._primitive_types_str():
                field_value = field_descr[MvResultPersistence.KEY_DATA]
            elif field_type in MvResultPersistence._numpy_primitive_types_str():
                # numpy types: https://numpy.org/doc/stable/user/basics.types.html
                field_value = MvResultPersistence._str_2_type()[field_type](
                    field_descr[MvResultPersistence.KEY_DATA]
                )
            elif field_type == self._type_2_str()[datetime.datetime]:
                field_value = datetime.datetime.strptime(
                    field_descr[MvResultPersistence.KEY_DATA],
                    MvResultPersistence.FORMAT_DATETIME,
                )
            elif field_type == self._type_2_str()[dict]:
                if MvResultPersistence.KEY_FILENAME in field_descr:
                    field_value = self._import_load_dict(
                        base_dir=base_dir,
                        sub_dir_name=field_descr[MvResultPersistence.KEY_FILENAME],
                    )
                elif MvResultPersistence.KEY_DATA in field_descr:
                    field_value = field_descr[MvResultPersistence.KEY_DATA]
                else:
                    raise ValueError(
                        f"Unsupported data type '{field_type}' when importing "
                        f"field '{field_name}' of {cls} defined by {index_file_path} - "
                        f"neither 'filename' nor 'data' key found to load the value"
                    )
            elif field_type == self._type_2_str()[list]:
                if MvResultPersistence.KEY_FILENAME in field_descr:
                    field_value = self._import_load_list(
                        import_dir_path=base_dir / field_name,
                    )
                elif MvResultPersistence.KEY_DATA in field_descr:
                    field_value = field_descr[MvResultPersistence.KEY_DATA]
                else:
                    raise ValueError(
                        f"Unsupported data type '{field_type}' when importing "
                        f"field '{field_name}' of {cls} defined by {index_file_path} - "
                        f"neither 'filename' nor 'data' key found to load the value"
                    )
            elif field_type == self._type_2_str()[tuple]:
                if MvResultPersistence.KEY_FILENAME in field_descr:
                    field_value = self._import_load_list(
                        import_dir_path=base_dir / field_name,
                    )
                elif MvResultPersistence.KEY_DATA in field_descr:
                    field_value = field_descr[MvResultPersistence.KEY_DATA]
                else:
                    raise ValueError(
                        f"Unsupported data type '{field_type}' when importing "
                        f"field '{field_name}' of {cls} defined by {index_file_path} - "
                        f"neither 'filename' nor 'data' key found to load the value"
                    )
            elif field_type == self._type_2_str()[datatable.Frame]:
                if MvResultPersistence.KEY_FILENAME in field_descr:
                    field_value = datatable.fread(
                        base_dir / field_descr[MvResultPersistence.KEY_FILENAME]
                    )
                else:
                    raise ValueError(
                        f"Unsupported data type '{field_type}' when importing "
                        f"field '{field_name}' of {cls} defined by {index_file_path} - "
                        f"'filename' key NOT found to load the value"
                    )
            elif field_type == self._type_2_str()[pandas.DataFrame]:
                if MvResultPersistence.KEY_FILENAME in field_descr:
                    # due to precision bug in Pandas, CSV export is made via datatable:
                    # field_value = pandas.read_csv(
                    #     base_dir / field_descr[MvResultPersistence.KEY_FILENAME]
                    # )
                    field_value = datatable.fread(
                        base_dir / field_descr[MvResultPersistence.KEY_FILENAME]
                    ).to_pandas()
                else:
                    raise ValueError(
                        f"Unsupported data type '{field_type}' when importing "
                        f"field '{field_name}' of {cls} defined by {index_file_path} - "
                        f"'filename' key NOT found to load the value"
                    )
            elif field_type == self._type_2_str()[pandas.Series]:
                if MvResultPersistence.KEY_FILENAME in field_descr:
                    # due to precision bug in Pandas, CSV export is made via datatable:
                    # field_value = pandas.read_csv(
                    #     base_dir / field_descr[MvResultPersistence.KEY_FILENAME]
                    # )
                    field_value = (
                        datatable.fread(
                            base_dir / field_descr[MvResultPersistence.KEY_FILENAME]
                        )
                        .to_pandas()
                        .squeeze()
                    )
                else:
                    raise ValueError(
                        f"Unsupported data type '{field_type}' when importing "
                        f"field '{field_name}' of {cls} defined by {index_file_path} - "
                        f"'filename' key NOT found to load the value"
                    )
            elif field_type == self._type_2_str()[pandas.Int64Index]:
                if MvResultPersistence.KEY_DATA in field_descr:
                    field_value = pandas.Int64Index(
                        field_descr[MvResultPersistence.KEY_DATA]
                    )
                else:
                    raise ValueError(
                        f"Unsupported data type '{field_type}' when importing "
                        f"field '{field_name}' of {cls} defined by {index_file_path} - "
                        f"'data' key NOT found to load the value"
                    )
            else:  # dataclasses and regular classes
                field_value = None
                # import any data class
                if field_type and field_type.startswith("<class '"):
                    try:
                        (
                            field_package_name,
                            field_type_name,
                        ) = MvResultPersistence.split_full_type(field_type)
                        dc_cls = getattr(
                            sys.modules[field_package_name], field_type_name
                        )
                        if dataclasses.is_dataclass(dc_cls):
                            # recursively import data class
                            import_dir_path = base_dir / field_name
                            field_value = self._import_dc_or_dict(
                                base_dir=import_dir_path,
                            )
                        else:
                            # instantiate regular class
                            import_dir_path = base_dir / field_name
                            field_value = self._import_dc_or_dict(
                                base_dir=import_dir_path,
                            )
                    except Exception as ex:
                        self.logger.error(
                            f"Unable to instantiate unsupported dataclasses data type "
                            f"'{field_type}' when importing field '{field_name}' "
                            f"of {cls} defined by {index_file_path}: "
                            f"{ex}\n{traceback.format_exc()}"
                        )
                # import any enum
                elif field_type and field_type.startswith("<enum '"):
                    field_value = self._import_enum(
                        field_name=field_name,
                        field_type=field_type,
                        field_value=field_descr[MvResultPersistence.KEY_DATA],
                        index_file_path=index_file_path,
                        cls=cls,
                    )

                if field_value is None:
                    raise ValueError(
                        f"Unsupported data type '{field_type}' when importing "
                        f"field '{field_name}' of {cls} defined by {index_file_path}"
                    )

            if force_type == self._type_2_str()[dict]:
                obj[field_name] = field_value
            elif force_type == self._type_2_str()[list]:
                obj.append(field_value)
            elif force_type == self._type_2_str()[tuple]:
                obj = obj + (field_value,)
            else:
                # make injection of fields ROBUST - do NOT crash in case that the
                # field has been renamed or removed (in H2O MV) after the export
                if hasattr(obj, field_name):
                    setattr(obj, field_name, field_value)
                else:
                    self.logger.error(
                        f"MV test importer is Unable to inject field '{field_name}' to"
                        f" {cls} as it does not exist in the class anymore"
                    )

        return obj

    def _import_from_dir(self, input_dir_path: pathlib.Path):
        return self._import_dc_or_dict(input_dir_path)
