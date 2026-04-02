# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import os
import pathlib
import socket

from h2o_sonar import config as h2o_sonar_config


# LLMs (constants remain hardcoded as they're not in JSON)
LLM_CAPYBARA = "NousResearch/Nous-Capybara-34b"
LLM_CLAUDE_SONNET = "claude-sonnet-4-5-20250929"
LLM_CLAUDE_SONNET_37 = "claude-3-7-sonnet-20250219"
LLM_CLAUDE_SONNET_37_LITE = "claude-3-7-sonnet-20250219-litellm"
LLM_CLAUDE_SONNET_45 = LLM_CLAUDE_SONNET
LLM_CLAUDE_SONNET_46 = "claude-sonnet-4-6"
LLM_DANUBE_3 = "h2oai/h2o-danube3-4b-chat"
LLM_GEMINI_FLASH = "gemini-2.5-flash"
LLM_GPT_35_TURBO = "gpt-35-turbo-1106"
LLM_GPT_4 = "gpt-4"  # genai.OpenAiAssistantsRagClientVersion1.DEFAULT_LLM_MODEL
LLM_GPT_4O_MINI = "gpt-4o-mini"
LLM_GROK_FAST = "grok-4-fast"
LLM_LLAMA31_70B = "meta-llama/Meta-Llama-3.1-70B-Instruct"
LLM_LLAMA_13B = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LLM_LLAMA_70B = "meta-llama/Llama-3.3-70B-Instruct"
LLM_MIXTRAL_8x22B = "mistralai/Mixtral-8x22B-Instruct-v0.1"
LLM_MIXTRAL_8x7B = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# test suites
TS_ARABIC_10P = "data/generative/eval_llm/arabic_mmlu_test_suite_10p.json"
TS_H2OGPTE_BENCH = "data/generative/h2ogpte_benchmark_test_suite.json"
TS_SR = "data/generative/sr1107_test_suite.json"
TS_SR_171OP = "data/generative/sr1107_test_suite_171.json"

# judge LLMs
# H2OGPTE_JUDGE_LLM_MODEL_NAME = LLM_CLAUDE_SONNET_46  # valid model for C_D
H2OGPTE_JUDGE_LLM_MODEL_NAME = LLM_CLAUDE_SONNET_45  # valid model for I_D

# H2O.ai cloud clusters
CLOUD_CLUSTER_D = "CLOUD_D"
CLOUD_CLUSTER_C_QA = "CLOUD_C_QA"
CLOUD_CLUSTER_I_D = "CLOUD_I_D"
CLOUD_CLUSTER_H2OAI = "CLOUD_PRODUCTION"

# API key env variable names
KEY_H2OGPT_API_KEY = "H2O_GPT_API_KEY"
KEY_H2OGPTE_API_KEY_D = "H2O_GPTE_API_KEY_D"
KEY_H2OGPTE_API_KEY_C_D = "H2O_GPTE_API_KEY_C_D"
KEY_H2OGPTE_API_KEY_I_D = "H2O_GPTE_API_KEY_I_D"
KEY_H2OGPTE_API_KEY_AZURE_J = "H2O_GPTE_API_KEY_AZURE_J"
KEY_H2OGPTE_API_KEY_H_W = "H2O_GPTE_API_KEY_H_W"
KEY_H2OGPTE_API_KEY_G_T = "H2O_GPTE_API_KEY_G_T"
KEY_H2OGPTE_API_KEY_G = "H2O_GPTE_API_KEY_G"
KEY_OPENAI_API_KEY = "OPENAI_API_KEY"
KEY_AZURE_OPENAI_API_KEY = "AZURE_OPENAI_API_KEY"
KEY_ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
KEY_AMAZON_BEDROCK_ACCESS_KEY = "AMAZON_BEDROCK_ACCESS_KEY_ID"
KEY_AMAZON_BEDROCK_SECRET_ACCESS_KEY = "AMAZON_BEDROCK_SECRET_ACCESS_KEY"
KEY_AMAZON_BEDROCK_SESSION_TOKEN = "AMAZON_BEDROCK_SESSION_TOKEN"

# test data sets
DIR_TEST_RAG_DOCS_CACHE = "data/generative/rag_docs"

#
# DYNAMIC CONFIGURATION - JSON CONFIG KEYS
#

# top-level service configuration keys
CFG_KEY_H2O_DRIVERLESS_AI = "h2o_driverless_ai"
CFG_KEY_H2O_STEAM = "h2o_steam"
CFG_KEY_H2O_AIEM = "h2o_aiem"
CFG_KEY_H2O_GPT = "h2o_gpt"
CFG_KEY_H2O_GPTE = "h2o_gpte"
CFG_KEY_OPENAI = "openai"
CFG_KEY_ANTHROPIC = "anthropic"
CFG_KEY_AZURE_OPENAI = "azure_openai"
CFG_KEY_AMAZON_BEDROCK = "amazon_bedrock"
CFG_KEY_H2O_LLMOPS = "h2o_llmops"
CFG_KEY_OLLAMA = "ollama"

# second-level configuration structure keys
CFG_KEY_CONNECTIONS = "connections"
CFG_KEY_EXPERIMENTS = "experiments"

#
# DYNAMIC CONFIGURATION - FUNCTIONS
#


def _resolve_env_variable(value):
    """Resolve environment variable references in the format 'env-variable://VAR_NAME'.

    Parameters
    ----------
    value : str | dict | list
        Value that may contain environment variable references.

    Returns
    -------
    str | dict | list
        Value with environment variables resolved.
    """
    if isinstance(value, str) and value.startswith("env-variable://"):
        env_var_name = value.replace("env-variable://", "")
        return os.getenv(env_var_name)
    elif isinstance(value, dict):
        return {k: _resolve_env_variable(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_variable(item) for item in value]

    return value


def _safe_nested_get(dct, *keys, default=None):
    """Safely get nested dict values, handling None intermediates.

    Parameters
    ----------
    dct : dict
        Root dictionary to traverse.
    *keys : str
        Sequence of keys to traverse.
    default : Any
        Value to return if any key missing or value is None.

    Returns
    -------
    Any
        Retrieved value or default.

    """
    result = dct
    for key in keys:
        if not isinstance(result, dict):
            return default
        result = result.get(key, default)
        if result is None:
            return default
    return result


def _load_json_config():
    """Load configuration from the JSON file.

    Returns
    -------
    dict
        Configuration dictionary loaded from JSON file, or empty dict if file doesn't
        exist.
    """
    json_path = pathlib.Path(__file__).parent / "given_generative.json"
    if not json_path.exists():
        print(
            f"WARNING: Configuration file {json_path} not found. "
            "Tests requiring external service connections will be skipped."
        )
        return {}
    try:
        with open(json_path) as f:
            return json.load(f)
    except Exception as e:
        print(
            f"WARNING: Failed to load configuration from {json_path}: {e}. "
            "Tests requiring external service connections will be skipped."
        )
        return {}


def _create_connection_from_json(conn_data):
    """Create ConnectionConfig object from JSON data with env variables resolved.

    Parameters
    ----------
    conn_data : dict
        Connection configuration dictionary from JSON.

    Returns
    -------
    h2o_sonar_config.ConnectionConfig
        Connection configuration object.
    """
    # use ConnectionConfig constants for field keys to ensure consistency
    cc = h2o_sonar_config.ConnectionConfig

    # Resolve environment variables for sensitive fields
    resolved_data = {
        cc.KEY_TYPE: conn_data.get(cc.KEY_TYPE, ""),
        cc.KEY_NAME: conn_data.get(cc.KEY_NAME, ""),
        cc.KEY_DESCRIPTION: conn_data.get(cc.KEY_DESCRIPTION, ""),
        cc.KEY_SERVER_URL: conn_data.get(cc.KEY_SERVER_URL, ""),
        cc.KEY_SERVER_ID: conn_data.get(cc.KEY_SERVER_ID, ""),
        cc.KEY_AUTH_SERVER_URL: conn_data.get(cc.KEY_AUTH_SERVER_URL, ""),
        cc.KEY_ENV_URL: conn_data.get(cc.KEY_ENV_URL, ""),
        cc.KEY_REALM_NAME: conn_data.get(cc.KEY_REALM_NAME, ""),
        cc.KEY_CLIENT_ID: conn_data.get(cc.KEY_CLIENT_ID, ""),
        cc.KEY_TOKEN: _resolve_env_variable(conn_data.get(cc.KEY_TOKEN, "")),
        cc.KEY_TOKEN_USE_TYPE: conn_data.get(cc.KEY_TOKEN_USE_TYPE, ""),
        cc.KEY_USERNAME: _resolve_env_variable(conn_data.get(cc.KEY_USERNAME, "")),
        cc.KEY_PASSWORD: _resolve_env_variable(conn_data.get(cc.KEY_PASSWORD, "")),
        cc.KEY_EXTRA_PARAMS: _resolve_env_variable(
            conn_data.get(cc.KEY_EXTRA_PARAMS, {})
        ),
    }

    return cc(**resolved_data)


# Load configuration from JSON
_config = _load_json_config()


def is_config() -> bool:
    """Indicate whether the services config was successfully loaded and initialized.

    Returns
    -------
    bool
        True if configuration is available, False otherwise.
    """
    return bool(_config)


#
# local/remove Driverless AI @ username/password authentication experiments
#

DAI_WORKER_CONNECTION = _create_connection_from_json(
    # hint: arrays usage below ~ [0], are intentional for infra privacy
    _safe_nested_get(
        _config, CFG_KEY_H2O_DRIVERLESS_AI, CFG_KEY_CONNECTIONS, default=[{}]
    )[0]
)

# hard-coded Driverless AI time series experiment hosted by DAI_WORKER_CONNECTION
DAI_EXPERIMENT_SONAR_TS = _safe_nested_get(
    _config, CFG_KEY_H2O_DRIVERLESS_AI, CFG_KEY_EXPERIMENTS, default=[{}]
)[0]
# hard-coded Driverless AI time series experiment hosted by DAI_WORKER_CONNECTION
DAI_EXPERIMENT_ALIEN_TS = _safe_nested_get(
    _config, CFG_KEY_H2O_DRIVERLESS_AI, CFG_KEY_EXPERIMENTS, default=[{}, {}]
)[1]
# hard-coded Driverless AI multinomial experiment hosted by DAI_WORKER_CONNECTION
DAI_EXPERIMENT_SONAR_M = _safe_nested_get(
    _config, CFG_KEY_H2O_DRIVERLESS_AI, CFG_KEY_EXPERIMENTS, default=[{}, {}, {}]
)[2]
# hard-coded Driverless AI multinomial experiment hosted by DAI_WORKER_CONNECTION
DAI_EXPERIMENT_ALIEN_M = _safe_nested_get(
    _config, CFG_KEY_H2O_DRIVERLESS_AI, CFG_KEY_EXPERIMENTS, default=[{}, {}, {}, {}]
)[3]

if "sonar" == socket.gethostname():
    DAI_EXPERIMENT_TS = DAI_EXPERIMENT_SONAR_TS
    DAI_EXPERIMENT_M = DAI_EXPERIMENT_SONAR_M
else:  # alien host Driverless AI experiments
    DAI_EXPERIMENT_TS = DAI_EXPERIMENT_ALIEN_TS
    DAI_EXPERIMENT_M = DAI_EXPERIMENT_ALIEN_M

#
# H2O Enterprise Steam Driverless AI experiments
#

_STEAM_CLOUD_HOST = CLOUD_CLUSTER_C_QA

if _STEAM_CLOUD_HOST == CLOUD_CLUSTER_D:
    STEAM_DAI_WORKER_CONNECTION = _create_connection_from_json(
        _safe_nested_get(_config, CFG_KEY_H2O_STEAM, CFG_KEY_CONNECTIONS, default=[{}])[
            0
        ]
    )
    # hard-coded Driverless AI time series experiment hosted by H2O Enterprise Steam
    STEAM_DAI_EXPERIMENT_TS = _safe_nested_get(
        _config, CFG_KEY_H2O_STEAM, CFG_KEY_EXPERIMENTS, default=[{}]
    )[0]

elif _STEAM_CLOUD_HOST == CLOUD_CLUSTER_C_QA:
    STEAM_DAI_WORKER_CONNECTION = _create_connection_from_json(
        _safe_nested_get(
            _config, CFG_KEY_H2O_STEAM, CFG_KEY_CONNECTIONS, default=[None, {}]
        )[1]
    )

    # hard-coded Driverless AI time series experiment hosted by H2O Enterprise Steam
    STEAM_DAI_EXPERIMENT_TS = _safe_nested_get(
        _config, CFG_KEY_H2O_STEAM, CFG_KEY_EXPERIMENTS, default=[{}, {}]
    )[1]
else:
    raise ValueError(f"Unknown H2O Steam cloud cluster: {_STEAM_CLOUD_HOST}")

#
# H2O AIEM Driverless AI experiments
#

_AIEM_CLOUD_HOST = CLOUD_CLUSTER_C_QA

if _AIEM_CLOUD_HOST == CLOUD_CLUSTER_C_QA:
    AIEM_DAI_WORKER_CONNECTION = _create_connection_from_json(
        _safe_nested_get(_config, CFG_KEY_H2O_AIEM, CFG_KEY_CONNECTIONS, default=[{}])[
            0
        ]
    )

    # hard-coded Driverless AI time series experiment hosted by H2O AIEM
    AIEM_DAI_EXPERIMENT_TS = _safe_nested_get(
        _config, CFG_KEY_H2O_AIEM, CFG_KEY_EXPERIMENTS, default=[{}]
    )[0]

    # hard-coded Driverless AI multinomial experiment hosted by H2O AIEM
    AIEM_DAI_EXPERIMENT_M = _safe_nested_get(
        _config, CFG_KEY_H2O_AIEM, CFG_KEY_EXPERIMENTS, default=[{}, {}]
    )[1]

elif _AIEM_CLOUD_HOST == CLOUD_CLUSTER_D:
    AIEM_DAI_WORKER_CONNECTION = _create_connection_from_json(
        _safe_nested_get(
            _config, CFG_KEY_H2O_AIEM, CFG_KEY_CONNECTIONS, default=[None, {}]
        )[1]
    )

    # hard-coded Driverless AI time series experiment hosted by H2O AIEM
    AIEM_DAI_EXPERIMENT_TS = _safe_nested_get(
        _config, CFG_KEY_H2O_AIEM, CFG_KEY_EXPERIMENTS, default=[{}, {}, {}]
    )[2]

    # hard-coded Driverless AI multinomial experiment hosted by H2O AIEM
    AIEM_DAI_EXPERIMENT_M = _safe_nested_get(
        _config, CFG_KEY_H2O_AIEM, CFG_KEY_EXPERIMENTS, default=[{}, {}, {}, {}]
    )[3]

else:
    raise ValueError(f"Unknown H2O AIEM cloud cluster: {_AIEM_CLOUD_HOST}")

#
# h2oGPT: OSS LLMs host
#

H2OGPT_PUBLIC = _create_connection_from_json(
    _safe_nested_get(_config, CFG_KEY_H2O_GPT, CFG_KEY_CONNECTIONS, default=[{}])[0]
)


#
# h2oGPTe: Enterprise h2oGPT
#

H2OGPTE_D = _create_connection_from_json(
    _safe_nested_get(_config, CFG_KEY_H2O_GPTE, CFG_KEY_CONNECTIONS, default=[{}])[0]
)

H2OGPTE_C_D = _create_connection_from_json(
    _safe_nested_get(
        _config, CFG_KEY_H2O_GPTE, CFG_KEY_CONNECTIONS, default=[None, {}]
    )[1]
)

# GPU Box hosted h2oGPTe
H2OGPTE_G = _create_connection_from_json(
    _safe_nested_get(
        _config, CFG_KEY_H2O_GPTE, CFG_KEY_CONNECTIONS, default=[None, None, {}]
    )[2]
)

# stable h2oGPTe used for HAIC testing
H2OGPTE_I_D = _create_connection_from_json(
    _safe_nested_get(
        _config, CFG_KEY_H2O_GPTE, CFG_KEY_CONNECTIONS, default=[None, None, None, {}]
    )[3]
)

# J's Azure hosted instance (system-wide certificate needed)
H2OGPTE_AZURE_J = _create_connection_from_json(
    _safe_nested_get(
        _config,
        CFG_KEY_H2O_GPTE,
        CFG_KEY_CONNECTIONS,
        default=[None, None, None, None, {}],
    )[4]
)

# H2O W events instance
H2OGPTE_H2O_W = _create_connection_from_json(
    _safe_nested_get(
        _config,
        CFG_KEY_H2O_GPTE,
        CFG_KEY_CONNECTIONS,
        default=[None, None, None, None, None, {}],
    )[5]
)

# H2O G-T events instance
H2OGPTE_G_T = _create_connection_from_json(
    _safe_nested_get(
        _config,
        CFG_KEY_H2O_GPTE,
        CFG_KEY_CONNECTIONS,
        default=[None, None, None, None, None, None, {}],
    )[6]
)


#
# OpenAI RAG
#

OPENAI_RAG = _create_connection_from_json(
    _safe_nested_get(_config, CFG_KEY_OPENAI, CFG_KEY_CONNECTIONS, default=[{}])[0]
)

OPENAI_LLM = _create_connection_from_json(
    _safe_nested_get(_config, CFG_KEY_OPENAI, CFG_KEY_CONNECTIONS, default=[None, {}])[
        1
    ]
)


# Anthropic Claude
ANTHROPIC_LLM = _create_connection_from_json(
    _safe_nested_get(_config, CFG_KEY_ANTHROPIC, CFG_KEY_CONNECTIONS, default=[{}])[0]
)


# OpenAI @ Microsoft Azure
#

AZURE_OPENAI_LLM = _create_connection_from_json(
    _safe_nested_get(_config, CFG_KEY_AZURE_OPENAI, CFG_KEY_CONNECTIONS, default=[{}])[
        0
    ]
)

# Amazon Bedrock
#
AMAZON_BEDROCK = _create_connection_from_json(
    _safe_nested_get(
        _config, CFG_KEY_AMAZON_BEDROCK, CFG_KEY_CONNECTIONS, default=[{}]
    )[0]
)


#
# H2O LLMOps hosted models
#

H2O_LLMOPS = _create_connection_from_json(
    _safe_nested_get(_config, CFG_KEY_H2O_LLMOPS, CFG_KEY_CONNECTIONS, default=[{}])[0]
)

#
# ollama hosted models
#

OLLAMA_LOCAL = _create_connection_from_json(
    _safe_nested_get(_config, CFG_KEY_OLLAMA, CFG_KEY_CONNECTIONS, default=[{}])[0]
)

OLLAMA_REMOTE = _create_connection_from_json(
    _safe_nested_get(_config, CFG_KEY_OLLAMA, CFG_KEY_CONNECTIONS, default=[None, {}])[
        1
    ]
)
