# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import enum
import json
import os
import pathlib
import sys
import uuid

import toml

import h2o_sonar.utils.caching
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import persistences
from h2o_sonar.utils import crypto


try:
    import torch

    HAS_PKG_TORCH = True
except ImportError:
    HAS_PKG_TORCH = False
    torch = None


# constants
DEP_DETOXIFY = "detoxify==0.5.2"
DEP_NLTK = "nltk==3.9.1"
DEP_OPTIMUM = "optimum==1.17.1"
DEP_SENTENCE_TRANSFORMERS = "sentence_transformers==5.1.2"
DEP_TRANSFORMERS = "transformers==4.38.2"
DEP_UMAP = "umap-learn==0.5.9.post2"


class ConfigItemType(enum.Enum):
    """Configuration item types."""

    CONNECTION = enum.auto()
    LICENSE = enum.auto()


class ConfigKeys:
    KEY_NAME = "name"
    KEY_DESCRIPTION = "description"
    KEY_ENCRYPTED = "encrypted"

    @staticmethod
    def _resolve_encryptable_value(
        config_dict: dict[str, str],
        key: str,
        decrypt: bool = True,
        encryption_key: str = "",
    ) -> str:
        dict_value = config_dict.get(key, "")
        if dict_value and isinstance(dict_value, dict):
            if ConnectionConfig.KEY_ENCRYPTED in dict_value.keys():
                encrypted_value = dict_value.get(ConnectionConfig.KEY_ENCRYPTED, "")
                if encrypted_value:
                    if decrypt:
                        return crypto.decrypt(
                            crypto.resolve_encryption_key(encryption_key),
                            dict_value[ConnectionConfig.KEY_ENCRYPTED],
                        )
                return encrypted_value

        return dict_value


class H2o3Config:
    """H2O-3 configuration keys used by H2O Sonar."""

    KEY_MIN_MEM_SIZE = "h2o_min_mem_size"
    DEFAULT_MIN_MEM_SIZE = "2G"
    KEY_MAX_MEM_SIZE = "h2o_max_mem_size"
    DEFAULT_MAX_MEM_SIZE = "4G"
    KEY_HOST = "h2o_host"
    DEFAULT_HOST = "localhost"
    KEY_PORT = "h2o_port"
    DEFAULT_PORT = 12349
    KEY_AUTO_START = "h2o_auto_start"
    DEFAULT_AUTO_START = True
    KEY_AUTO_CLEANUP = "h2o_auto_cleanup"
    DEFAULT_AUTO_CLEANUP = True
    KEY_AUTO_STOP = "h2o_auto_stop"
    DEFAULT_AUTO_STOP = False

    KEYS = [
        KEY_MIN_MEM_SIZE,
        KEY_MAX_MEM_SIZE,
        KEY_HOST,
        KEY_PORT,
        KEY_AUTO_START,
        KEY_AUTO_CLEANUP,
        KEY_AUTO_STOP,
    ]


class ConnectionConfigType(enum.Enum):
    """Predefined connection types."""

    # predictive autoMLs
    DRIVERLESS_AI = enum.auto()  # remote or local Driverless AI server installation
    DRIVERLESS_AI_STEAM = enum.auto()  # Driverless AI in H2O Enterprise Steam
    DRIVERLESS_AI_AIEM = enum.auto()  # Driverless AI in H2O AI Engine Manager
    H2O_3 = enum.auto()

    # RAGs and LLMs
    H2O_GPT = enum.auto()  # H2O GPT
    H2O_GPT_E = enum.auto()  # H2O GPT Enterprise
    H2O_LLM_OPS = enum.auto()  # H2O LLMOps
    OLLAMA = enum.auto()  # ollama
    OPENAI_RAG = enum.auto()  # OpenAI RAG API
    OPENAI_CHAT = enum.auto()  # OpenAI Chat API
    AZURE_OPENAI_CHAT = enum.auto()  # Microsoft Azure hosted OpenAI Chat API
    # AZURE_OPENAI_RAG = enum.auto() # MS Azure hosted OpenAI Assistant API w/ retrieval
    ANTHROPIC_CHAT = enum.auto()  # Anthropic Claude Chat API
    HF_SPACES = enum.auto()  # HuggingFace Spaces API
    AMAZON_BEDROCK = enum.auto()  # Amazon Bedrock


class TokenUseType(enum.Enum):
    """Predefined token use types."""

    API_KEY = enum.auto()  # API key
    ACCESS_TOKEN = enum.auto()  # (personal) access token
    REFRESH_TOKEN = enum.auto()  # (client) refresh token
    SECRET = enum.auto()  # (client) secret


class ConnectionConfig(ConfigKeys):
    """A generic purpose connection configuration which can be used to connect to
    various H2O.ai and 3rd party products and services.

    Overview of the fields which are important for the H2O.ai authentication methods:

    Username/password authentication method requires (Keycloak):
    - authentication server URL: auth_server_url
    - client ID: client_id
    - client secret: token + token_use_type==SECRET
    - username and password (on getting the access token)

    Refresh token authentication method requires (Keycloak):
    - authentication server URL: auth_server_url
    - client ID: client_id
    - refresh token: token + token_use_type==REFRESH_TOKEN
    - realm: realm_name

    H2O token provider authentication method requires:
    - environment URL: environment_url
    - refresh token: token + token_use_type==REFRESH_TOKEN

    """

    KEY_KEY = "key"
    KEY_TYPE = "connection_type"
    KEY_SERVER_URL = "server_url"
    KEY_SERVER_ID = "server_id"
    KEY_AUTH_SERVER_URL = "auth_server_url"
    KEY_ENV_URL = "environment_url"
    KEY_REALM_NAME = "realm_name"
    KEY_CLIENT_ID = "client_id"
    KEY_TOKEN = "token"
    KEY_TOKEN_USE_TYPE = "token_use_type"
    KEY_USERNAME = "username"
    KEY_PASSWORD = "password"
    KEY_EXTRA_PARAMS = "extra_params"

    # fields which are automatically be encrypted/decrypted on save/load
    ENCRYPTED_FIELDS = [KEY_TOKEN, KEY_PASSWORD]

    def __init__(
        self,
        connection_type: str,  # see also ConnectionConfigType
        name: str,
        description: str,
        server_url: str | None = "",
        server_id: str | None = "",
        # authentication
        auth_server_url: str | None = "",
        environment_url: str | None = "",
        client_id: str | None = "",
        realm_name: str | None = "",
        token: str | None = "",
        token_use_type: str | None = "",  # see also TokenUseType
        username: str | None = "",
        password: str | None = "",
        extra_params: dict | None = None,
        key: str | None = "",
    ):
        """Connection configuration constructor.

        Parameters
        ----------

        key : str
            Unique connection key. If key is not specified, then it will be
            automatically generated by H2O Sonar. This key is used to identify
            and/or reference the connection.
        connection_type : str
            Connection type - see ``ConnectionConfigType`` for valid string values.
        name : str
            Custom connection name.
        description : str
            Custom connection description.
        server_url : str | None
            Server URL. For example:
            - https://enginemanager.cloud.h2o.ai/ (H2O AIEM hosted Driverless AI)
            - https://steam.cloud.h2o.ai/ (H2O Enterprise Steam hosted Driverless AI)
            - https://host:12345/ (a standalone Driverless AI installation)
        server_id : str | None
            Server ID used to identify the server in case it does not have a fixed URL.
            For example:
            - "new-dai-engine-42" (ID assigned by H2O AIEM to Driverless AI)
            - "my-driverless-ai" (custom name given to Driverless AI in H2O Steam)
        auth_server_url : str | None
            Authentication server URL.
        environment_url : str | None
            Environment URL used in the H2O token provider authentication method. For
            example:
            - https://cloud.h2o.ai/ (H2O.ai cloud)
            - https://qa.acme.com/ (an on-premise QA cloud)
            - https://dev.acme.com/ (an on-premise development cloud)
        realm_name : str | None
            Realm name (authentication domain) or workspace in case of Driverless AI
            hosted by H2O AIEM connection.
        client_id : str | None
            Client ID.
        token : str | None
            A client refresh, access or a secret token:
            - H2O AIEM:
              Get/generate token: H2O.ai Cloud > User > CLI & API Access > API Token.
              Use ``TokenUseType.REFRESH_TOKEN`` as the ``token_use_type``.
            - H2O Enterprise Steam:
              Get/generate token: H2O.ai Cloud > Enterprise Steam > Configurations >
              Personal Access Token > Get token.
              Use ``TokenUseType.REFRESH_TOKEN`` as the ``token_use_type``.
        token_use_type : str | None
            Client use type - see ``TokenUseType`` for valid string values.
        username : str | None
            Username.
        password : str | None
            Password.
        extra_params : dict
            Extra parameters which can be used to store additional connection
            configuration information

        """
        self.key = key or str(uuid.uuid4())
        self.connection_type = connection_type
        self.name = name
        self.description = description
        self.server_url = server_url
        self.server_id = server_id
        self.auth_server_url = auth_server_url
        self.environment_url = environment_url
        self.realm_name = realm_name
        self.client_id = client_id
        self.token = token
        self.token_use_type = token_use_type
        self.username = username
        self.password = password
        self.extra_params = extra_params or {}

    def to_dict(self, encrypt: bool = True, encryption_key: str = "") -> dict:
        return {
            ConnectionConfig.KEY_KEY: self.key,
            ConnectionConfig.KEY_TYPE: self.connection_type,
            ConnectionConfig.KEY_NAME: self.name,
            ConnectionConfig.KEY_DESCRIPTION: self.description,
            ConnectionConfig.KEY_AUTH_SERVER_URL: self.auth_server_url,
            ConnectionConfig.KEY_ENV_URL: self.environment_url,
            ConnectionConfig.KEY_SERVER_URL: self.server_url,
            ConnectionConfig.KEY_SERVER_ID: self.server_id,
            ConnectionConfig.KEY_REALM_NAME: self.realm_name,
            ConnectionConfig.KEY_CLIENT_ID: self.client_id,
            ConnectionConfig.KEY_TOKEN: (
                {
                    ConnectionConfig.KEY_ENCRYPTED: crypto.encrypt(
                        encryption_key=crypto.resolve_encryption_key(encryption_key),
                        data=self.token,
                    )
                }
                if encrypt
                else self.token
            ),
            ConnectionConfig.KEY_TOKEN_USE_TYPE: self.token_use_type,
            ConnectionConfig.KEY_USERNAME: self.username,
            ConnectionConfig.KEY_PASSWORD: (
                {
                    ConnectionConfig.KEY_ENCRYPTED: crypto.encrypt(
                        encryption_key=crypto.resolve_encryption_key(encryption_key),
                        data=self.password,
                    )
                }
                if encrypt
                else self.password
            ),
            ConnectionConfig.KEY_EXTRA_PARAMS: self.extra_params,
        }

    @staticmethod
    def from_dict(
        config_dict: dict, decrypt: bool = True, encryption_key: str = ""
    ) -> "ConnectionConfig":
        """Create the connection configuration from dictionary.

        JSon example w/ unencrypted fields::

            {
                "key": "096ca3c2-4715-11ee-9e2f-10828613f8ad",
                "connection_type": "ML_API",
                "name": "My connection name",
                "description": "My connection description.",
                "server_url": "http://localhost:8080",
                "server_id": "my-model-validation-dai",
                "auth_server_url": "http://localhost:8080/auth",
                "environment_url": "https://cloud.h2o.ai/",
                "realm_name": "my_realm",
                "client_id": "my_client_id",
                "token": "",
                "token_use_type": "",
                "username": "sonaruser",
                "password": "s3cr3tpa33word"
            }

        JSon example w/ encrypted fields::

            {
                "key": "096ca3c2-4715-11ee-9e2f-10828613f8ad",
                "connection_type": "ML_API",
                "name": "My connection name",
                "description": "My connection description.",
                "server_url": "http://localhost:8080",
                "server_id": "my-model-validation-dai",
                "auth_server_url": "http://localhost:8080/auth",
                "environment_url": "https://cloud.h2o.ai/",
                "realm_name": "my_realm",
                "client_id": "my_client_id",
                "token": {
                    "encrypted": "gAAAAABkTfy18iis3ya8nitGi...URMxE14aJJk="
                },
                "token_use_type": "REFRESH_TOKEN",
                "username": "sonaruser",
                "password": {
                    "encrypted": "py18iis3ya8nitG="
                }
            }

        """
        token = ConnectionConfig._resolve_encryptable_value(
            config_dict=config_dict,
            key=ConnectionConfig.KEY_TOKEN,
            decrypt=decrypt,
            encryption_key=encryption_key,
        )
        password = ConnectionConfig._resolve_encryptable_value(
            config_dict=config_dict, key=ConnectionConfig.KEY_PASSWORD, decrypt=decrypt
        )

        return ConnectionConfig(
            key=config_dict.get(ConnectionConfig.KEY_KEY) or str(uuid.uuid4()),
            connection_type=config_dict[ConnectionConfig.KEY_TYPE],
            name=config_dict.get(ConnectionConfig.KEY_NAME, "Connection"),
            description=config_dict.get(ConnectionConfig.KEY_DESCRIPTION, ""),
            server_url=config_dict.get(ConnectionConfig.KEY_SERVER_URL, ""),
            server_id=config_dict.get(ConnectionConfig.KEY_SERVER_ID, ""),
            auth_server_url=config_dict.get(ConnectionConfig.KEY_AUTH_SERVER_URL, ""),
            environment_url=config_dict.get(ConnectionConfig.KEY_ENV_URL, ""),
            realm_name=config_dict.get(ConnectionConfig.KEY_REALM_NAME, ""),
            client_id=config_dict.get(ConnectionConfig.KEY_CLIENT_ID, ""),
            token=token,
            token_use_type=config_dict.get(ConnectionConfig.KEY_TOKEN_USE_TYPE, ""),
            username=config_dict.get(ConnectionConfig.KEY_USERNAME, ""),
            password=password,
            extra_params=config_dict.get(ConnectionConfig.KEY_EXTRA_PARAMS, {}),
        )

    def __hash__(self):
        return hash(json.dumps(self.to_dict(encrypt=False), sort_keys=True))

    def __eq__(self, other):
        return json.dumps(self.to_dict(encrypt=False), sort_keys=True) == json.dumps(
            other.to_dict(encrypt=False), sort_keys=True
        )


class ProductLicenseConfig(enum.Enum):
    """Predefined license types."""

    DRIVERLESS_AI = enum.auto()


class LicenseConfig(ConfigKeys):
    """A product license configuration."""

    KEY_KEY = "key"
    KEY_PRODUCT = "product"
    KEY_LICENSE_FILE = "license_file"
    KEY_LICENSE = "license"

    # fields to be automatically encrypted/decrypted on save/load
    ENCRYPTED_FIELDS = [KEY_LICENSE]

    def __init__(
        self,
        product: str,  # see also ProductLicenseConfig
        name: str,
        description: str,
        license: str = "",
        license_file: str = "",
        key: str = "",
    ):
        """License configuration constructor.

        Parameters
        ----------
        key : str
            Unique license key. If key is not specified, then it will be
            automatically generated.
        product : str
            Product name of the product whose license is being configured.
        name : str
            License name.
        description : str
            License description.
        license : str
            License key.
        license_file : str
            License file path.

        """
        self.key = key or str(uuid.uuid4())
        self.product = product
        self.name = name
        self.description = description
        self.license = license
        self.license_file = license_file

    def to_dict(self, encrypt: bool = True, encryption_key: str = "") -> dict:
        return {
            LicenseConfig.KEY_KEY: self.key,
            LicenseConfig.KEY_PRODUCT: self.product,
            LicenseConfig.KEY_NAME: self.name,
            LicenseConfig.KEY_DESCRIPTION: self.description,
            LicenseConfig.KEY_LICENSE: (
                {
                    ConnectionConfig.KEY_ENCRYPTED: crypto.encrypt(
                        crypto.resolve_encryption_key(encryption_key), self.license
                    ),
                }
                if encrypt
                else self.license
            ),
            LicenseConfig.KEY_LICENSE_FILE: self.license_file,
        }

    @staticmethod
    def from_dict(
        config_dict: dict, decrypt: bool = True, encryption_key: str = ""
    ) -> "LicenseConfig":
        license_value = LicenseConfig._resolve_encryptable_value(
            config_dict=config_dict,
            key=LicenseConfig.KEY_LICENSE,
            decrypt=decrypt,
            encryption_key=encryption_key,
        )

        return LicenseConfig(
            key=config_dict.get(LicenseConfig.KEY_KEY) or str(uuid.uuid4()),
            product=config_dict.get(LicenseConfig.KEY_PRODUCT, "PRODUCT"),
            name=config_dict.get(LicenseConfig.KEY_NAME, "License"),
            description=config_dict.get(LicenseConfig.KEY_DESCRIPTION, ""),
            license=license_value,
            license_file=config_dict.get(LicenseConfig.KEY_LICENSE_FILE, ""),
        )


class EvaluationJudgeType(enum.Enum):
    h2ogpt = enum.auto()  # h2oGPT hosted LLM
    h2ogpte = enum.auto()  # h2oGPTe RAG
    h2ogpte_llm = enum.auto()  # h2oGPTe hosted LLM
    h2ollmops = enum.auto()  # H2O LLMOps hosted LLM
    openai_rag = enum.auto()  # OpenAI RAG
    openai_llm = enum.auto()  # OpenAI hosted LLM
    azure_openai_llm = enum.auto()  # Microsoft Azure OpenAI hosted LLM
    anthropic_llm = enum.auto()  # Anthropic Claude hosted LLM
    ollama = enum.auto()  # ollama
    custom = enum.auto()


class EvaluationJudgeConfig:
    """Evaluation judge configuration."""

    KEY_NAME = "name"
    KEY_DESCRIPTION = "description"
    KEY_JUDGE_TYPE = "judge_type"
    KEY_LLM_MODEL_NAME = "llm_model_name"
    KEY_CONNECTION = "connection"
    KEY_COLLECTION_ID = "collection_id"
    KEY_KEY = "key"

    def __init__(
        self,
        name: str,
        description: str,
        judge_type: str,
        connection: ConnectionConfig,
        llm_model_name: str = "",
        collection_id: str = "",
        key: str = "",
    ):
        self.name = name
        self.description = description
        self.judge_type = judge_type
        self.llm_model_name = llm_model_name
        self.connection = connection
        self.collection_id = collection_id
        self.key = key or str(uuid.uuid4())

    def __str__(self):
        return json.dumps(self.to_dict())

    def to_dict(self) -> dict:
        return {
            EvaluationJudgeConfig.KEY_NAME: self.name,
            EvaluationJudgeConfig.KEY_DESCRIPTION: self.description,
            EvaluationJudgeConfig.KEY_JUDGE_TYPE: self.judge_type,
            EvaluationJudgeConfig.KEY_LLM_MODEL_NAME: self.llm_model_name,
            EvaluationJudgeConfig.KEY_COLLECTION_ID: self.collection_id,
            EvaluationJudgeConfig.KEY_CONNECTION: (
                self.connection.key if self.connection else ""
            ),
            EvaluationJudgeConfig.KEY_KEY: self.key,
        }

    @staticmethod
    def from_dict(config_dict: dict) -> "EvaluationJudgeConfig":
        connection = config.get_connection(
            config_dict.get(EvaluationJudgeConfig.KEY_CONNECTION, "")
        )

        return EvaluationJudgeConfig(
            name=config_dict.get(EvaluationJudgeConfig.KEY_NAME, str(uuid.uuid4())),
            description=config_dict.get(EvaluationJudgeConfig.KEY_DESCRIPTION, ""),
            judge_type=config_dict.get(EvaluationJudgeConfig.KEY_JUDGE_TYPE, ""),
            connection=connection,
            llm_model_name=config_dict.get(
                EvaluationJudgeConfig.KEY_LLM_MODEL_NAME, ""
            ),
            collection_id=config_dict.get(EvaluationJudgeConfig.KEY_COLLECTION_ID, ""),
            key=config_dict.get(EvaluationJudgeConfig.KEY_KEY, ""),
        )


class H2oSonarConfig:
    """H2O Sonar configuration with global configuration items which impact H2O Sonar
    behavior, methods and explainers.

    Configuration priority from lowest to highest:

    - default value
    - configuration file
    - environment variable
    - command line argument
    - configuration item

    """

    MP_START_METHOD_SPAWN = "spawn"
    MP_START_METHOD_FORK = "fork"
    MP_START_METHOD_FORKSERVER = "forkserver"

    VALUE_STR_TRUE = "true"
    VALUE_STR_FALSE = "false"

    VALUE_CPU = "cpu"
    VALUE_GPU = "gpu"
    VALUE_MPS = "mps"
    VALUE_NPU = "npu"
    VALUE_CUDA = "cuda"
    VALUE_AUTO = ""

    CFG_PER_EXPLAINER_LOGGER = commons.Param(
        param_name="per_explainer_logger",
        description=(
            "Create new logger for each explainer (which logs to explainer "
            "sandbox) or reuse one logger and use library logger for all "
            "log messages."
        ),
        param_type=commons.ParamType.bool,
        default_value=True,
    )
    CFG_ENABLE_DATASET_DOWNLOADING = commons.Param(
        param_name="enable_dataset_downloading",
        description=(
            "Privacy options which controls whether potentially sensitive dataset "
            "data can be stored in explainer snapshot ZIP archives."
        ),
        param_type=commons.ParamType.bool,
        default_value=True,
    )
    CFG_CUSTOM_EXPLAINERS = commons.Param(
        param_name="custom_explainers",
        description=(
            'List of custom "Bring Your Own Explainer" string locators to be registered'
            " on H2O Sonar run. The location has the following structure: "
            '"[PACKAGE and MODULE]::[EXPLAINER-CLASS-NAME]" where PACKAGE and MODULE '
            "is dot (.) separated path to the the module (installed on PYTHONPATH) "
            "and EXPLAINER-CLASS-NAME is the name of explainer class. "
            'Example: [ "my_package.explainer_module::MyExplainerClass", '
            '"their_package.explainer_module::TheirExplainerClass"]'
        ),
        param_type=commons.ParamType.customlist,
        default_value=[],
    )
    CFG_H2O_HOST = commons.Param(
        param_name="h2o_host",
        description=(
            "The host of the H2O-3 server that should be used for the explanation "
            "that requires it."
        ),
        param_type=commons.ParamType.str,
        default_value=H2o3Config.DEFAULT_HOST,
    )
    CFG_H2O_PORT = commons.Param(
        param_name="h2o_port",
        description=(
            "The port of the H2O-3 server that should be used for the explanation "
            "that requires it."
        ),
        param_type=commons.ParamType.int,
        default_value=H2o3Config.DEFAULT_PORT,
    )
    CFG_H2O_AUTO_START = commons.Param(
        param_name="h2o_auto_start",
        description=(
            "Automatically start H2O-3 server on the interpretation start (True), "
            "or do not start the server (False)."
        ),
        param_type=commons.ParamType.bool,
        default_value=H2o3Config.DEFAULT_AUTO_START,
    )
    CFG_H2O_AUTO_CLEANUP = commons.Param(
        param_name="h2o_auto_cleanup",
        description=(
            "Automatically remove all data from the H2O-3 server on"
            "the interpretation end (True), or do not remove all data from"
            "the server (False)."
        ),
        param_type=commons.ParamType.bool,
        default_value=H2o3Config.DEFAULT_AUTO_CLEANUP,
    )
    CFG_H2O_AUTO_STOP = commons.Param(
        param_name="h2o_auto_stop",
        description=(
            "Automatically stop H2O-3 server on the interpretation end (True), "
            "or do not stop the server (False)."
        ),
        param_type=commons.ParamType.bool,
        default_value=H2o3Config.DEFAULT_AUTO_STOP,
    )
    CFG_H2O_MIN_MEM_SIZE = commons.Param(
        param_name="h2o_min_mem_size",
        description=(
            "Minimum memory specification for H2O-3 server started by H2O Sonar."
        ),
        param_type=commons.ParamType.int,
        default_value=H2o3Config.DEFAULT_MIN_MEM_SIZE,
    )
    CFG_H2O_MAX_MEM_SIZE = commons.Param(
        param_name="h2o_max_mem_size",
        description=(
            "Maximum memory specification for H2O-3 server started by H2O Sonar."
        ),
        param_type=commons.ParamType.int,
        default_value=H2o3Config.DEFAULT_MAX_MEM_SIZE,
    )
    CFG_DO_SAMPLE = commons.Param(
        param_name="mli_sample",
        description="Choose whether to run all explainers on the sampled dataset.",
        param_type=commons.ParamType.bool,
        default_value=True,
    )
    CFG_SAMPLE_SIZE = commons.Param(
        param_name="mli_sample_size",
        description=(
            "The sample size, number of rows, to be used for the surrogate models."
        ),
        param_type=commons.ParamType.int,
        default_value=0,
    )
    CFG_NUM_QUANTILES = commons.Param(
        param_name="mli_num_quantiles",
        description="The default number of bins for quantile binning.",
        param_type=commons.ParamType.int,
        default_value=10,
    )
    CFG_CREATE_HTML_REPRESENTATIONS = commons.Param(
        param_name="create_html_representations",
        description=(
            "Indicate that explainers can create HTML representation (True), "
            "or request to skip it (False) from performance/resource consumption "
            "reasons."
        ),
        param_type=commons.ParamType.bool,
        default_value=True,
    )
    CFG_LOOK_AND_FEEL = commons.Param(
        param_name="look_and_feel",
        description=(
            "Charts theme (look and feel) - one of: 'h2o_sonar', 'blue', "
            "'driverless_ai'."
        ),
        param_type=commons.ParamType.str,
        default_value=commons.LookAndFeel.H2O_SONAR_THEME,
    )
    CFG_DEVICE = commons.Param(
        param_name="device",
        description=(
            "Device to be used for the calculations. The value of this configuration "
            "item might be ``cpu`` or ``gpu``."
        ),
        param_type=commons.ParamType.str,
        default_value=VALUE_AUTO,
    )
    CFG_ENABLE_SLOW_PERTURBATORS = commons.Param(
        param_name="enable_slow_perturbators",
        description=(
            "Enable slow (agent-based, model-based, resource intensive) perturbators "
            "which are by default skipped and not listed."
        ),
        param_type=commons.ParamType.bool,
        default_value=False,
    )
    CFG_FORCE_EVAL_JUDGE = commons.Param(
        param_name="force_eval_judge",
        description=(
            "Force the use of custom evaluation judge for the evaluation of "
            "the models over the judges used by evaluators by default. For example "
            "to use a local judge in order to avoid sending sensitive data to "
            "a 3rd party or to the cloud. The value of this configuration item "
            "might be ``false``, ``true`` or configuration key of the custom "
            "evaluation judge. "
            "Forcing the use of a custom evaluation judge will automatically "
            "reconfigure the embeddings calculation in evaluations to a local model "
            "to ensure privacy safety."
        ),
        param_type=commons.ParamType.str,
        default_value=VALUE_STR_FALSE,
    )
    # multiprocessing start method - one of: 'spawn', 'fork', 'forkserver'
    #  https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    #  https://github.com/pytest-dev/pytest/issues/11174
    #  https://stackoverflow.com/questions/64095876/multiprocessing-fork-vs-spawn
    CFG_MP_START_METHOD = commons.Param(
        param_name="multiprocessing_start_method",
        description=(
            "Multiprocessing start method - one of: 'spawn', 'fork', 'forkserver' "
            "or `None` (default)."
        ),
        param_type=commons.ParamType.str,
        default_value=MP_START_METHOD_SPAWN,
    )
    CFG_MODEL_CACHE_DIR = commons.Param(
        param_name="model_cache_dir",
        description=(
            "Directory where the models are cached. If not specified, the models "
            "are cached in a default directory in user home which follows operating "
            "system conventions."
        ),
        param_type=commons.ParamType.str,
        default_value=h2o_sonar.utils.caching.DEFAULT_SONAR_CACHE_MODEL_DIR,
    )
    # HTTPS SSL certificate verification configuration for HTTP requests:
    #   https://requests.readthedocs.io/en/latest/user/advanced/#ssl-cert-verification
    # HTTPS clients SSL certificate verification configuration :
    # - OK: requests library - verify parameter & CA_BUNDLE env var
    # - OK: h2oGPTe client - verify parameter
    # - OK: OpenAI client - CA_BUNDLE env var
    # - OK: h2oGPT client - OpenAI client ^
    # - OK: Azure OpenAI client - OpenAI (sub) client ^
    # - OK: fairness bias evaluator - requests library ^
    # To be configured:
    # - HuggingFace client @ ragas library (once HF models are cached, it is OK)
    CFG_HTTP_SSL_CERT_VERIFY = commons.Param(
        param_name="http_ssl_cert_verify",
        description=(
            "SSL certificate verification for HTTPS requests. If set to ``false``, "
            "then SSL certificate verification is disabled. If set to ``true``, "
            "then SSL certificate verification is enabled. If set to the path (string) "
            "to a ``CA_BUNDLE`` file or directory with certificates of trusted CAs, "
            "then they will be used for the verification (in this case the directory "
            "must have been processed using the c_rehash utility supplied with "
            "OpenSSL)."
        ),
        param_type=commons.ParamType.str,
        default_value=VALUE_STR_TRUE,
    )
    CFG_BRANDING = commons.Param(
        param_name="branding",
        description=(
            "Branding for HTML reports. If not specified (empty string). "
            "Valid values: 'H2O_SONAR', 'EVAL_STUDIO', or '' (empty for auto)."
        ),
        param_type=commons.ParamType.str,
        default_value="",
    )

    # interpretation parameters declaration for documentation, report and introspection
    _cfg_items = [
        CFG_PER_EXPLAINER_LOGGER,
        CFG_ENABLE_DATASET_DOWNLOADING,
        CFG_CUSTOM_EXPLAINERS,
        CFG_H2O_HOST,
        CFG_H2O_PORT,
        CFG_H2O_AUTO_START,
        CFG_H2O_AUTO_CLEANUP,
        CFG_H2O_AUTO_STOP,
        CFG_H2O_MIN_MEM_SIZE,
        CFG_H2O_MAX_MEM_SIZE,
        CFG_DO_SAMPLE,
        CFG_SAMPLE_SIZE,
        CFG_NUM_QUANTILES,
        CFG_CREATE_HTML_REPRESENTATIONS,
        CFG_LOOK_AND_FEEL,
        CFG_DEVICE,
        CFG_ENABLE_SLOW_PERTURBATORS,
        CFG_FORCE_EVAL_JUDGE,
        CFG_MP_START_METHOD,
        CFG_MODEL_CACHE_DIR,
        CFG_HTTP_SSL_CERT_VERIFY,
        CFG_BRANDING,
    ]

    KEY_CUSTOM_EXPLAINERS = "custom_explainers"
    KEY_PER_EXPLAINER_LOGGER = "per_explainer_logger"
    KEY_CREATE_HTML_REPRESENTATIONS = "create_html_representations"
    KEY_LOOK_AND_FEEL = "look_and_feel"
    KEY_DEVICE = "device"
    KEY_ENABLE_SLOW_PERTURBATORS = "enable_slow_perturbators"
    KEY_CONNECTIONS = "connections"
    KEY_LICENSES = "licenses"
    KEY_EVALUATION_JUDGES = "evaluation_judges"
    KEY_FORCE_EVAL_JUDGE = "force_eval_judge"
    KEY_MODEL_CACHE_DIR = "model_cache_dir"
    KEY_HTTP_SSL_CERT_VERIFY = "http_ssl_cert_verify"
    KEY_BRANDING = "branding"

    @property
    def http_ssl_cert_verify(self):
        if str(self._http_ssl_cert_verify).lower() == H2oSonarConfig.VALUE_STR_TRUE:
            return True
        elif str(self._http_ssl_cert_verify).lower() == H2oSonarConfig.VALUE_STR_FALSE:
            return False

        return self._http_ssl_cert_verify

    @http_ssl_cert_verify.setter
    def http_ssl_cert_verify(self, value):
        # if certificate verification is newly going to use a file, then reconfigure
        if (
            str(value).lower()
            not in [H2oSonarConfig.VALUE_STR_TRUE, H2oSonarConfig.VALUE_STR_FALSE]
            and pathlib.Path(value).is_file()
        ):
            # set the env var for the requests library w/ the path to CA_BUNDLE file,
            # once env var is set, OpenAI/requests library will use this certificate
            # file for validation automatically and will resolve the issue.
            os.environ["REQUESTS_CA_BUNDLE"] = str(value)

        self._http_ssl_cert_verify = value

    def __init__(
        self,
        connections: list[ConnectionConfig] | None = None,
        licenses: list[LicenseConfig] | None = None,
        evaluation_judges: list[EvaluationJudgeConfig] | None = None,
        ignore_env: bool | None = False,
    ):
        # connections
        self.connections = connections or []

        # licenses
        self.licenses = licenses or []

        # evaluation judges
        self.evaluation_judges = evaluation_judges or []

        # logging
        self.per_explainer_logger = (
            H2oSonarConfig.CFG_PER_EXPLAINER_LOGGER.default_value
        )

        # datasets
        self.enable_dataset_downloading = (
            H2oSonarConfig.CFG_ENABLE_DATASET_DOWNLOADING.default_value
        )

        # BYOE recipes: list of strings like "my_package.my_module.py::MyExplainer"
        self.custom_explainers: list[str] = (
            H2oSonarConfig.CFG_CUSTOM_EXPLAINERS.default_value
        )

        # H2O-3 server
        self.h2o_host: str = H2oSonarConfig.CFG_H2O_HOST.default_value
        self.h2o_port: int = H2oSonarConfig.CFG_H2O_PORT.default_value
        self.h2o_auto_start: bool = H2oSonarConfig.CFG_H2O_AUTO_START.default_value
        self.h2o_auto_cleanup: bool = H2oSonarConfig.CFG_H2O_AUTO_CLEANUP.default_value
        self.h2o_auto_stop: bool = H2oSonarConfig.CFG_H2O_AUTO_STOP.default_value

        # sampling
        self.mli_sample: bool = H2oSonarConfig.CFG_DO_SAMPLE.default_value
        self.mli_sample_size: int = H2oSonarConfig.CFG_SAMPLE_SIZE.default_value

        # binning
        self.mli_num_quantiles: int = H2oSonarConfig.CFG_NUM_QUANTILES.default_value

        # NLP
        self.mli_nlp_tokenizer: str = "tfidf"
        self.mli_nlp_surrogate_tokenizer: str = "Linear Model + TF-IDF"
        self.mli_nlp_surrogate_tokens: int = 100
        self.mli_nlp_min_df: float = 3.0
        self.mli_nlp_max_df: float = 0.9
        self.mli_nlp_min_ngram: int = 1
        self.mli_nlp_max_ngram: int = 1
        self.mli_nlp_use_stop_words: bool = True
        self.mli_nlp_stop_words: str = "english"
        self.mli_nlp_append_to_english_stop_words: bool = False
        self.mli_nlp_min_token_mode: str = "top"

        # representations
        self.create_html_representations = (
            H2oSonarConfig.CFG_CREATE_HTML_REPRESENTATIONS.default_value
        )

        # look & feel
        self.look_and_feel = H2oSonarConfig.CFG_LOOK_AND_FEEL.default_value

        # device
        self.device = H2oSonarConfig.CFG_DEVICE.default_value

        # enable slow perturbators
        self.enable_slow_perturbators = (
            H2oSonarConfig.CFG_ENABLE_SLOW_PERTURBATORS.default_value
        )

        # forced evaluation judge
        self.force_eval_judge = H2oSonarConfig.CFG_FORCE_EVAL_JUDGE.default_value

        # multiprocessing start method
        # + avoid DEADLOCKS in case of multiprocessing + fork (considered dangerous)
        self.mp_start_method = H2oSonarConfig.CFG_MP_START_METHOD.default_value

        # cache
        self.model_cache_dir = H2oSonarConfig.CFG_MODEL_CACHE_DIR.default_value

        # HTTP/HTTPS SSL certificate verification
        self._http_ssl_cert_verify = (
            H2oSonarConfig.CFG_HTTP_SSL_CERT_VERIFY.default_value
        )

        # branding
        self.branding = H2oSonarConfig.CFG_BRANDING.default_value

        # extensibility: any custom GLOBAL parameters can be set/get by any component
        self.global_params = {}

        # introspection: parameters indexation
        self.cfg_items_dict = {}
        for p in H2oSonarConfig._cfg_items:
            self.cfg_items_dict[p.param_name] = p

        # debugging & profiling
        self.enable_profiler = False

        # override default values from environment variables
        if not ignore_env:
            self.env_and_override()

    def add_connection(self, connection_config: ConnectionConfig):
        if not connection_config:
            raise ValueError("Connection configuration cannot be added - it is None.")
        if not connection_config.key:
            raise ValueError(
                "Connection configuration cannot be added - it has no key."
            )
        c = self.get_connection(connection_config.key)
        if c:
            print(
                "WARNING: Connection configuration cannot be added - it already exists."
            )
            return c

        self.connections.append(connection_config)
        return connection_config

    def get_connection(
        self, connection_key: str, connection_type: str = ""
    ) -> ConnectionConfig | None:
        if self.connections:
            for c in self.connections:
                if c.key == connection_key:
                    return c
        return None

    def add_license(self, license_config: LicenseConfig):
        self.licenses.append(license_config)
        return license_config

    def get_license(self, license_key: str) -> LicenseConfig | None:
        if self.licenses:
            for lc in self.licenses:
                if lc.key == license_key:
                    return lc
        return None

    def add_evaluation_judge(self, evaluation_judge_config: EvaluationJudgeConfig):
        self.evaluation_judges.append(evaluation_judge_config)
        return evaluation_judge_config

    def get_evaluation_judge(self, judge_key: str = "") -> EvaluationJudgeConfig | None:
        if self.evaluation_judges:
            if not judge_key:
                return self.evaluation_judges[0]

            for jc in self.evaluation_judges:
                if jc.key == judge_key:
                    return jc
        return None

    def describe_config_items(self) -> dict[str, commons.Param]:
        return self.cfg_items_dict

    def describe_config_item(self, config_item_name: str) -> commons.Param | None:
        return self.cfg_items_dict.get(config_item_name, None)

    def save(
        self,
        config_path: str,
        config_data: dict | None = None,
        encrypt: bool = True,
        encryption_key: str = "",
    ):
        """Save configuration to JSon file.

        Parameters
        ----------
        config_path : str
          Path to the configuration file.
        config_data : dict
          Optional dictionary with configuration data
        encrypt : bool
            Whether to encrypt sensitive data.
        encryption_key : str
            Optional encryption key to encrypt sensitive data. If not specified, then
            shell environment variable ``H2O_SONAR_ENCRYPTION_KEY`` will be used.

        """
        persistences.ExplainerPersistence.save_json(
            data=(
                self.to_dict(encrypt=encrypt, encryption_key=encryption_key)
                if not config_data
                else config_data
            ),
            path=config_path,
        )

    @staticmethod
    def _decrypt_json_value(connection, field_key, encryption_key):
        encrypted_val = connection[field_key].get(ConnectionConfig.KEY_ENCRYPTED, {})
        if encrypted_val:
            decrypted_val = crypto.decrypt(
                crypto.resolve_encryption_key(encryption_key),
                encrypted_val,
            )
            if decrypted_val:
                connection[field_key] = decrypted_val

    @staticmethod
    def _decrypt_loaded_json(json_dict: dict, encryption_key: str) -> dict | None:
        if encryption_key:
            if json_dict:
                licenses = json_dict.get(H2oSonarConfig.KEY_LICENSES, [])
                if licenses:
                    for license_cfg in licenses:
                        for key in license_cfg.keys():
                            if key in ConnectionConfig.ENCRYPTED_FIELDS:
                                H2oSonarConfig._decrypt_json_value(
                                    license_cfg, key, encryption_key
                                )

                connections = json_dict.get(H2oSonarConfig.KEY_CONNECTIONS, [])
                if connections:
                    for connection_cfg in connections:
                        for key in connection_cfg.keys():
                            if key in ConnectionConfig.ENCRYPTED_FIELDS:
                                H2oSonarConfig._decrypt_json_value(
                                    connection_cfg, key, encryption_key
                                )

        return json_dict

    @staticmethod
    def _instantiate_loaded_json(
        json_dict: dict, decrypt: bool = True, encryption_key: str = ""
    ) -> dict | None:
        if json_dict:
            connections = json_dict.get(H2oSonarConfig.KEY_CONNECTIONS, [])
            json_dict[H2oSonarConfig.KEY_CONNECTIONS] = [
                ConnectionConfig.from_dict(
                    config_dict=c, decrypt=decrypt, encryption_key=encryption_key
                )
                for c in connections
            ]
            licenses = json_dict.get(H2oSonarConfig.KEY_LICENSES, [])
            json_dict[H2oSonarConfig.KEY_LICENSES] = [
                LicenseConfig.from_dict(
                    config_dict=c, decrypt=decrypt, encryption_key=encryption_key
                )
                for c in licenses
            ]
            evaluation_judges = json_dict.get(H2oSonarConfig.KEY_EVALUATION_JUDGES, [])
            json_dict[H2oSonarConfig.KEY_EVALUATION_JUDGES] = [
                EvaluationJudgeConfig.from_dict(c) for c in evaluation_judges
            ]

        return json_dict

    @staticmethod
    def load(config_path: str, encryption_key: str = "") -> dict | None:
        """Load JSon/TOML file with configuration items specified in the
        configuration file.

        Parameters
        ----------
        config_path : str
          Path to the configuration file.
        encryption_key : str
            Optional encryption key to decrypt/encrypt sensitive data. If not
            specified, then shell environment variable ``H2O_SONAR_ENCRYPTION_KEY``
            will be used.

        Returns
        -------
        dict :
          Dictionary with the configuration if file can be found and parsed,
          ``None`` otherwise.

        """
        json_dict = None

        if os.path.isfile(config_path):
            with open(config_path) as json_file:
                try:
                    json_dict = json.load(json_file)
                except json.decoder.JSONDecodeError:
                    json_dict = toml.load(json_file)

        json_dict = H2oSonarConfig._decrypt_loaded_json(
            json_dict=json_dict, encryption_key=encryption_key
        )
        json_dict = H2oSonarConfig._instantiate_loaded_json(
            json_dict=json_dict, encryption_key=encryption_key
        )

        return json_dict

    def load_and_override(self, config_path: str, encryption_key: str = ""):
        """Load JSon/TOML file and override this instance configuration items with
        configuration items specified in the configuration file.

        Parameters
        ----------
        config_path : str
          Path to the configuration file.
        encryption_key : str
            Optional encryption key to decrypt encrypted fields in the configuration
            file. If not specified, then shell environment variable
            ``H2O_SONAR_ENCRYPTION_KEY`` will be used.

        """
        json_dict = self.load(config_path, encryption_key)

        # override
        if json_dict:
            self.h2o_host = json_dict.get(H2o3Config.KEY_HOST, self.h2o_host)
            self.h2o_port = json_dict.get(H2o3Config.KEY_PORT, self.h2o_port)

            self.h2o_auto_start = json_dict.get(
                H2o3Config.KEY_AUTO_START, self.h2o_auto_start
            )
            self.h2o_auto_cleanup = json_dict.get(
                H2o3Config.KEY_AUTO_CLEANUP, self.h2o_auto_cleanup
            )
            self.h2o_auto_stop = json_dict.get(
                H2o3Config.KEY_AUTO_STOP, self.h2o_auto_stop
            )

            self.custom_explainers = json_dict.get(
                H2oSonarConfig.KEY_CUSTOM_EXPLAINERS, self.custom_explainers
            )
            self.look_and_feel = json_dict.get(
                H2oSonarConfig.KEY_LOOK_AND_FEEL, self.look_and_feel
            )
            self.device = json_dict.get(H2oSonarConfig.KEY_DEVICE, self.device)
            self.enable_slow_perturbators = json_dict.get(
                H2oSonarConfig.KEY_ENABLE_SLOW_PERTURBATORS,
                self.enable_slow_perturbators,
            )
            self.force_eval_judge = json_dict.get(
                H2oSonarConfig.KEY_FORCE_EVAL_JUDGE, self.force_eval_judge
            )
            self.mp_start_method = json_dict.get(
                H2oSonarConfig.CFG_MP_START_METHOD.param_name, self.mp_start_method
            )
            self.model_cache_dir = json_dict.get(
                H2oSonarConfig.KEY_MODEL_CACHE_DIR, self.model_cache_dir
            )
            self.http_ssl_cert_verify = json_dict.get(
                H2oSonarConfig.KEY_HTTP_SSL_CERT_VERIFY, self.http_ssl_cert_verify
            )
            self.branding = json_dict.get(H2oSonarConfig.KEY_BRANDING, self.branding)
            self.per_explainer_logger = json_dict.get(
                H2oSonarConfig.KEY_PER_EXPLAINER_LOGGER, self.per_explainer_logger
            )
            self.create_html_representations = json_dict.get(
                H2oSonarConfig.KEY_CREATE_HTML_REPRESENTATIONS,
                self.create_html_representations,
            )
            self.connections = json_dict.get(
                H2oSonarConfig.KEY_CONNECTIONS,
                self.connections,
            )
            self.licenses = json_dict.get(H2oSonarConfig.KEY_LICENSES, self.licenses)
            self.evaluation_judges = json_dict.get(
                H2oSonarConfig.KEY_EVALUATION_JUDGES, self.evaluation_judges
            )

            return

        raise FileNotFoundError(
            f"Unable to load the configuration file - {config_path} does not exist"
        )

    ENV_VAR_CFG_PREFIX = "H2O_SONAR_CFG_"

    def env_and_override(self):
        """Get configuration from environment variables following the naming
        convention and override this instance configuration.

        """
        for cfg_item in self._cfg_items:
            env_var_name = (
                f"{H2oSonarConfig.ENV_VAR_CFG_PREFIX}{cfg_item.param_name.upper()}"
            )
            env_var_value = os.getenv(env_var_name)
            if env_var_value is not None:
                if cfg_item.param_type == commons.ParamType.bool:
                    env_var_value = env_var_value.lower() in [
                        "true",
                        "1",
                        "yes",
                        "y",
                    ]
                elif cfg_item.param_type == commons.ParamType.int:
                    try:
                        env_var_value = int(env_var_value)
                    except ValueError:
                        print(
                            f"Error: Could not convert environment variable "
                            f"{env_var_name} value '{env_var_value}' to int. "
                            f"Skipping.",
                            file=sys.stderr,
                        )
                        continue
                elif cfg_item.param_type == commons.ParamType.float:
                    try:
                        env_var_value = float(env_var_value)
                    except ValueError:
                        print(
                            f"Error: Could not convert environment variable "
                            f"{env_var_name} value '{env_var_value}' to float. "
                            f"Skipping.",
                            file=sys.stderr,
                        )
                        continue
                elif cfg_item.param_type == commons.ParamType.str:
                    try:
                        env_var_value = str(env_var_value)
                    except ValueError:
                        print(
                            f"Error: Could not convert environment variable "
                            f"{env_var_name} value '{env_var_value}' to string. "
                            f"Skipping.",
                            file=sys.stderr,
                        )
                        continue
                else:
                    print(
                        f"Error: Unsupported parameter type {cfg_item.param_type} "
                        f"for environment variable {env_var_name}. Skipping.",
                        file=sys.stderr,
                    )
                    continue

                setattr(self, cfg_item.param_name, env_var_value)

    def to_dict(self, encrypt: bool = True, encryption_key: str = "") -> dict:
        connections_dict = [
            c.to_dict(encrypt, encryption_key) for c in self.connections
        ]
        licenses_dict = [c.to_dict(encrypt, encryption_key) for c in self.licenses]
        evaluation_judges_dict = [c.to_dict() for c in self.evaluation_judges]

        return {
            H2o3Config.KEY_HOST: self.h2o_host,
            H2o3Config.KEY_PORT: self.h2o_port,
            H2o3Config.KEY_AUTO_START: self.h2o_auto_start,
            H2o3Config.KEY_AUTO_CLEANUP: self.h2o_auto_cleanup,
            H2o3Config.KEY_AUTO_STOP: self.h2o_auto_stop,
            H2o3Config.KEY_MIN_MEM_SIZE: H2o3Config.DEFAULT_MIN_MEM_SIZE,
            H2o3Config.KEY_MAX_MEM_SIZE: H2o3Config.DEFAULT_MAX_MEM_SIZE,
            H2oSonarConfig.KEY_CUSTOM_EXPLAINERS: self.custom_explainers,
            H2oSonarConfig.KEY_LOOK_AND_FEEL: self.look_and_feel,
            H2oSonarConfig.KEY_DEVICE: self.device,
            H2oSonarConfig.KEY_ENABLE_SLOW_PERTURBATORS: self.enable_slow_perturbators,
            H2oSonarConfig.KEY_FORCE_EVAL_JUDGE: self.force_eval_judge,
            H2oSonarConfig.CFG_MP_START_METHOD.param_name: self.mp_start_method,
            H2oSonarConfig.KEY_MODEL_CACHE_DIR: str(self.model_cache_dir),
            H2oSonarConfig.KEY_HTTP_SSL_CERT_VERIFY: str(self.http_ssl_cert_verify),
            H2oSonarConfig.KEY_BRANDING: self.branding,
            H2oSonarConfig.KEY_PER_EXPLAINER_LOGGER: self.per_explainer_logger,
            H2oSonarConfig.KEY_CREATE_HTML_REPRESENTATIONS: (
                self.create_html_representations
            ),
            H2oSonarConfig.KEY_CONNECTIONS: connections_dict,
            H2oSonarConfig.KEY_LICENSES: licenses_dict,
            H2oSonarConfig.KEY_EVALUATION_JUDGES: evaluation_judges_dict,
        }

    def to_toml(self):
        raise NotImplementedError

    def from_toml(self):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def resolve_gpu_cpu_device(self, result_format: str = "torch"):
        """Resolve a GPU/CPU device for the evaluation.

        Parameters
        ----------
        result_format : str
            Result format - one of "str" or "torch".

        Returns
        -------
        str | torch.device | None :
            Device string, torch device object, or None if unable to resolve.

        """
        device = None

        # IF torch is not available, THEN gracefully return string representations
        if self.device == H2oSonarConfig.VALUE_CPU:
            if result_format == "torch" and HAS_PKG_TORCH:
                return torch.device("cpu")
            else:
                return H2oSonarConfig.VALUE_CPU
        elif self.device == H2oSonarConfig.VALUE_GPU:
            if result_format == "torch" and HAS_PKG_TORCH:
                return torch.device("cuda")  # can be cuda:0, cuda:1, etc.
            else:
                return H2oSonarConfig.VALUE_CUDA
        elif "cuda" in self.device:
            if result_format == "torch" and HAS_PKG_TORCH:
                return torch.device(self.device)
            else:
                return self.device

        return device


#
# H2O Sonar configuration singleton
#

config = H2oSonarConfig()
