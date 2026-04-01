# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import hashlib
import importlib.metadata
import os
import pathlib
import platform
import shutil

import pandas
from packaging import version

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import loggers
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.integrations import genai
from h2o_sonar.methods.utils import h2o_utils
from h2o_sonar.utils import _profiling
from h2o_sonar.utils import preprocessing
from tests.lib import given_generative
from tests.lib.given_generative import KEY_AMAZON_BEDROCK_ACCESS_KEY
from tests.lib.given_generative import KEY_AMAZON_BEDROCK_SECRET_ACCESS_KEY
from tests.lib.given_generative import KEY_AMAZON_BEDROCK_SESSION_TOKEN


class GitHubActions:
    """GitHub Actions utilities."""

    ENV_GHA = "GITHUB_ACTIONS"

    # for GHA workers CPU/RAM flavors see ./.github/workflows/*.yml
    DEFAULT_H2O3_MIN_MEM_SIZE = "1G"  # 500M
    DEFAULT_H2O3_MAX_MEM_SIZE = "2G"  # 1G

    @staticmethod
    def is_in_gha():
        return os.getenv(GitHubActions.ENV_GHA, "").lower() == "true"

    @staticmethod
    def configure_h2o_sonar():
        """Modify H2O Sonar (singleton) configuration for running on GitHub
        Actions while considering GitHub Actions workers (HW) flavors.

        """
        pass

    @staticmethod
    def is_low_memory_worker() -> bool:
        (_, total_mem) = _profiling.get_mem_profile()
        return True if total_mem is not None and total_mem < 10000 else False

    @staticmethod
    def get_h2o3_config() -> dict:
        """Get H2O-3 server configuration for GitHub Actions (workers)."""
        t_h2o3cfg = h2o_sonar_config.H2o3Config

        if GitHubActions.is_low_memory_worker():
            return {
                t_h2o3cfg.KEY_MIN_MEM_SIZE: GitHubActions.DEFAULT_H2O3_MIN_MEM_SIZE,
                t_h2o3cfg.KEY_MAX_MEM_SIZE: GitHubActions.DEFAULT_H2O3_MAX_MEM_SIZE,
            }

        return {
            t_h2o3cfg.KEY_MIN_MEM_SIZE: t_h2o3cfg.DEFAULT_MIN_MEM_SIZE,
            t_h2o3cfg.KEY_MAX_MEM_SIZE: t_h2o3cfg.DEFAULT_MAX_MEM_SIZE,
        }

    @staticmethod
    def configure_h2o3(h2o3_config: dict) -> dict:
        """Modify H2O-3 server/cluster configuration for running on
        GitHub Actions while considering GitHub Actions workers (HW) flavors.

        """
        h2o3_config_overrides = GitHubActions.get_h2o3_config()

        for k in h2o3_config_overrides:
            h2o3_config[k] = h2o3_config_overrides[k]

        return h2o3_config


def h2o3_init_for_tests():
    """Initialization of the H2O-3 server."""

    h2o_utils.ensure_h2o3_running(
        h2o3_config_overrides={
            h2o_sonar_config.H2o3Config.KEY_MIN_MEM_SIZE: (
                GitHubActions.DEFAULT_H2O3_MIN_MEM_SIZE
                if GitHubActions.is_low_memory_worker()
                else h2o_sonar_config.H2o3Config.DEFAULT_MIN_MEM_SIZE
            ),
            h2o_sonar_config.H2o3Config.KEY_MAX_MEM_SIZE: (
                GitHubActions.DEFAULT_H2O3_MAX_MEM_SIZE
                if GitHubActions.is_low_memory_worker()
                else h2o_sonar_config.H2o3Config.DEFAULT_MAX_MEM_SIZE
            ),
        }
    )


def rm_test_dir(tmp_dir: str):
    """Remove test directory.

    Parameters
    ----------
    tmp_dir: str
        Directory to remove.

    """
    if os.path.exists(tmp_dir) and os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)


def find_subdir(start_dir: pathlib.Path, subdir_name: str) -> pathlib.Path | None:
    """Recursively search for a subdirectory with a specific name.

    Parameters
    ----------
    start_dir : pathlib.Path
        The directory to start the search from.
    subdir_name : str
        The name of the subdirectory to find.

    Returns
    -------
    pathlib.Path | None :
        The path of the found subdirectory or None if not found.
    """
    if not isinstance(start_dir, pathlib.Path):
        raise TypeError("start_dir must be a pathlib.Path object")
    if not start_dir.is_dir():
        raise ValueError(
            f"Warning: Starting directory '{start_dir}' does not exist or is "
            f"not a directory."
        )
    if not subdir_name:
        raise ValueError("Warning: subdir_name cannot be empty.")

    # recursively search
    try:
        for item in start_dir.rglob(subdir_name):
            if item.is_dir():
                return item
    except PermissionError:
        print(
            f"Warning: Permission denied while searching in '{start_dir}' or "
            f"its subdirectories."
        )
        return None
    except Exception as e:
        raise e

    return None


def find_locally(path: str) -> str:
    """Looks for a file in 3 places:

    * path that was passed
    * one directory above
    * two directories above

    Returns the first one that exists

    Parameters
    ----------
    path: str
        Directory to be found

    Returns
    -------
    str

    """
    if os.path.exists(path):
        return path

    if os.path.exists(f"../{path}"):
        return f"../{path}"

    if os.path.exists(f"../../{path}"):
        return f"../../{path}"

    raise ValueError(f"Could not find path [{path}].")


def dump_in_memory_persistence(
    persistence: persistences.InMemoryPersistence, do_assert: bool = False
):
    if do_assert:
        assert persistence, "In memory persistence not is None"
        assert persistence.memory_store.keys(), "In-memory store is empty"

    memory_store = persistence.memory_store
    if isinstance(persistence, persistences.InMemoryPersistence):
        store_keys = list(memory_store.keys())
        print(f"Store ({persistence.type}):")
        for k in store_keys:
            v = memory_store[k]
            print(
                f"  {k}\n    {type(v)}"
                f"[{len(v) if persistences.InMemoryPersistence.DIR != v else 0}]"
            )
    else:
        print(f"Store ({persistence.type}): SKIPPING keys dump")


def is_linux() -> bool:
    return platform.system().lower() == "linux"


def is_private_test_data_available() -> bool:
    """Check whether private test data is available (S3 access configured)."""
    private_data_path = pathlib.Path("data/generative/eval_s3")
    return (private_data_path / "bug-1539").is_dir()


def is_mojo_supported() -> bool:
    """MOJO is not supported on GitHub Actions as Driverless AI license cannot be
    used there.

    """
    if not are_python_modules_installed({"daimojo"}):
        return False

    is_dai_license = os.path.isfile(os.path.expanduser("~/.driverlessai/license.sig"))

    return is_dai_license and not GitHubActions.is_in_gha() and is_linux()


def is_sklearn_1_1_2() -> bool:
    """scikit-learn version cannot be enforced in case of Driverless AI "
    tests, and it would lead to version to version pickle de/serialization errors."

    """
    import sklearn

    return sklearn.__version__ == "1.1.2"


def get_version_specific_scikit_model(scikit_model_path: str) -> str:
    """Get version specific scikit-learn model pickle path. If there is no pickle
    for a new scikit-learn version, then use
    ``test_models_sklearn.py::test_pickle_sklearn_model`` to create a new one.

    Parameters
    ----------
    scikit_model_path: str
        Path to the scikit-learn model pickle.

    Returns
    -------
    str
        Version specific scikit-learn model pickle.

    """
    import sklearn

    default_model_name = "creditcard-binomial-sklearn-gbm.pkl"
    if default_model_name in scikit_model_path:
        version_specific_name = "creditcard-binomial-sklearn-1.1.1-gbm.pkl"
        if version.parse(sklearn.__version__) >= version.parse("1.8.0"):
            version_specific_name = "creditcard-binomial-sklearn-1.8.0-gbm.pkl"
        elif version.parse(sklearn.__version__) >= version.parse("1.5.1"):
            version_specific_name = "creditcard-binomial-sklearn-1.5.1-gbm.pkl"
        elif version.parse(sklearn.__version__) >= version.parse("1.3.0"):
            version_specific_name = "creditcard-binomial-sklearn-1.3.0-gbm.pkl"
        elif version.parse(sklearn.__version__) >= version.parse("1.1.2"):
            version_specific_name = "creditcard-binomial-sklearn-1.1.2-gbm.pkl"

        return scikit_model_path.replace(default_model_name, version_specific_name)

    raise ValueError(f"Unknown sklearn test model: {scikit_model_path}")


def are_python_modules_installed(module_names: set[str]) -> bool:
    """Check whether given Python modules are installed"""
    if module_names:
        installed = {
            dist.name.lower().replace("-", "_")
            for dist in importlib.metadata.distributions()
        }
        module_names = {n.replace("-", "_") for n in module_names}
        missing = module_names - installed
        if missing:
            return False
    return True


def is_local_dai_running(port: int = 12345) -> bool:
    return commons.is_port_used(port=port, logger=loggers.SonarPrintLogger())


def create_sklearn_model(dataset_name: str, target_col: str):
    """Create scikit-learn model.

    Parameters
    ----------
    dataset_name : str
      Filename of the dataset in ``data/`` directory.
    target_col : str
      Target column.

    Returns
    -------
    Tuple[str, ExplainableModel, str] :
      Resolved path to dataset, explainable model and target column.

    """
    import sklearn

    # dataset
    dataset_path = find_locally(f"data/{dataset_name}")
    x_train = pandas.read_csv(dataset_path)
    (X, y) = x_train.drop(target_col, axis=1), x_train[target_col]
    used_features = X.columns.to_list()
    (X, _, _) = preprocessing.categorical_encoder(X)

    # model
    model = sklearn.ensemble.GradientBoostingClassifier(learning_rate=0.1)
    model.fit(X, y)
    explainable_model = models.ModelApi().create_model(
        model_src=model,
        target_col=target_col,
        used_features=used_features,
    )

    return dataset_path, explainable_model, target_col


def assert_interpretation(interpretation):
    from h2o_sonar.explainers import dia_explainer
    from h2o_sonar.explainers import dt_surrogate_explainer
    from h2o_sonar.explainers import pd_ice_explainer
    from h2o_sonar.explainers import summary_shap_explainer

    explainers = [
        dia_explainer.DiaExplainer,
        pd_ice_explainer.PdIceExplainer,
        summary_shap_explainer.SummaryShapleyExplainer,
        dt_surrogate_explainer.DecisionTreeSurrogateExplainer,
    ]

    print(f"Interpretation:\n{interpretation}")
    assert interpretation, "Interpretation cannot be None"
    assert interpretation.result, "Interpretation result cannot be None"
    assert interpretation.result.explainers
    assert len(interpretation.result.explainers) > 5
    failed_explainers = interpretation.get_failed_explainer_ids()
    assert not failed_explainers, f"Failed explainers: {failed_explainers}"

    for explainer in explainers:
        actual_param_names = [
            explainer.parameters()[i].param_name
            for i in range(len(explainer.parameters()))
        ]
        result = interpretation.get_explainer_result(explainer.explainer_id())
        assert result, f"{explainer.explainer_id()} result cannot be None"
        assert result.summary()
        assert actual_param_names == list(result.params().keys())


def assert_html_report_images(interpretation):
    html_report_str = interpretation.to_html()
    assert html_report_str
    # image paths asserts
    html_report_path = interpretation.result.html_location
    html_report_base_dir = os.path.dirname(html_report_path)
    # image: <img src="PATH-TO-PARSE-OUT" alt="...
    html_report_lines = html_report_str.splitlines()
    for line in html_report_lines:
        if '<img src="' in line and 'alt="' in line:
            img_path = line.split('<img src="')[1].split('" alt="')[0]
            assert os.path.exists(os.path.join(html_report_base_dir, img_path))


def given_base_cli_cmd() -> tuple[list[str], dict]:
    cli_cmd = (
        ["h2o-sonar"]
        if os.system("which h2o-sonar") == 0
        else ["python", "h2o_sonar/h2o_sonar_cli.py"]
    )
    child_env = os.environ.copy()
    # add the root of the repo to Python path so that the CLI can load model class
    child_env["PYTHONPATH"] = "."

    return cli_cmd, child_env


class Health:
    """Health check for the RAGs, LLMs, judges, ...

    Health check is stateful - it's initialized only **once** and then used through
    whole Python VM life cycle. Singleton pattern is not used as race conditions are
    not expected (test are sequential and the results are always consistent).

    """

    def __init__(self):
        self._is_openai = None
        self._is_azure_openai = None
        self._is_anthropic = None
        self._is_h2ollmops = None
        self._is_h2ogpt = None
        self._is_h2ogpte = None
        self._is_ollama = None
        self._is_bedrock = None

        self._openai_azure_connection = None
        self._anthropic_connection = None

        self._h2ogpt_connection = None
        self._h2ogpt_models = []

        self._h2ogpte_connection = None
        self._h2ogpte_models = []

        self._ollama_connection = None
        self._ollama_models = []

        self._judge_cfg = None

    def is_openai(self) -> bool:
        if self._is_openai is None:
            if not are_python_modules_installed({"openai"}) or not os.getenv(
                given_generative.KEY_OPENAI_API_KEY
            ):
                self._is_openai = False
            else:
                self._is_openai = True

        return self._is_openai

    def is_bedrock(self) -> bool:
        if self._is_bedrock is None:
            if not are_python_modules_installed({"boto3"}) or not (
                os.getenv(KEY_AMAZON_BEDROCK_SESSION_TOKEN)
                and os.getenv(KEY_AMAZON_BEDROCK_ACCESS_KEY)
                and os.getenv(KEY_AMAZON_BEDROCK_SECRET_ACCESS_KEY)
            ):
                self._is_bedrock = False
            else:
                self._is_bedrock = True

        return self._is_bedrock

    def is_h2ollmops(self) -> bool:
        if self._is_h2ollmops is None:
            # H2O LLMOps hosted LLM models are deployed only temporarily > DISABLED
            self._is_h2ollmops = False

        return self._is_h2ollmops

    # OpenAI @ Azure

    def is_azure_openai(self) -> bool:
        if self._is_azure_openai is None:
            if not are_python_modules_installed({"openai"}) or not os.getenv(
                given_generative.KEY_AZURE_OPENAI_API_KEY
            ):
                self._is_azure_openai = False
            else:
                self._is_azure_openai = True

        return self._is_azure_openai

    def get_openai_azure(self):
        if not self._openai_azure_connection:
            self._openai_azure_connection = given_generative.AZURE_OPENAI_LLM
        return self._openai_azure_connection

    # Anthropic Claude

    def is_anthropic(self) -> bool:
        if self._is_anthropic is None:
            if not are_python_modules_installed({"anthropic"}) or not os.getenv(
                given_generative.KEY_ANTHROPIC_API_KEY
            ):
                self._is_anthropic = False
            else:
                self._is_anthropic = True

        return self._is_anthropic

    def get_anthropic(self):
        if not self._anthropic_connection:
            self._anthropic_connection = given_generative.ANTHROPIC_LLM
        return self._anthropic_connection

    # h2oGPT

    def is_h2ogpt(self) -> bool:
        if self._is_h2ogpt is None:
            if not os.getenv(given_generative.KEY_H2OGPT_API_KEY):
                self._is_h2ogpt = False
            else:
                self._is_h2ogpt = True

        return self._is_h2ogpt

    def get_h2ogpt(self):
        """Get h2oGPT server to use for the testing. h2oGPT servers and/or hosted LLMs
        are unstable and causing hangs, times and test failures - this variable allows
        to switch all test to the currently working server.

        Returns
        -------
        h2o_sonar_config.ConnectionConfig
            h2oGPT server to use for the testing.

        """
        if not self._h2ogpt_connection:
            self._h2ogpt_connection = given_generative.H2OGPT_PUBLIC

        return self._h2ogpt_connection

    def get_h2ogpt_models(self, h2ogpt_models: list[str] | None = 0):
        if not self._h2ogpt_models:
            self._h2ogpt_models = [
                given_generative.LLM_GPT_4,
                given_generative.LLM_LLAMA_13B,
                given_generative.LLM_LLAMA_70B,
            ]

        if h2ogpt_models:
            return [model for model in self._h2ogpt_models if model in h2ogpt_models]

        return self._h2ogpt_models

    # h2oGPTe
    t_gg = given_generative
    H2OGPTE_URL_2_ENV_VAR = {
        t_gg.H2OGPTE_D.server_url: t_gg.KEY_H2OGPTE_API_KEY_D,
        t_gg.H2OGPTE_C_D.server_url: t_gg.KEY_H2OGPTE_API_KEY_C_D,
        t_gg.H2OGPTE_G.server_url: t_gg.KEY_H2OGPTE_API_KEY_G,
        t_gg.H2OGPTE_I_D.server_url: t_gg.KEY_H2OGPTE_API_KEY_I_D,
        t_gg.H2OGPTE_H2O_W.server_url: t_gg.KEY_H2OGPTE_API_KEY_H_W,
        t_gg.H2OGPTE_G_T.server_url: t_gg.KEY_H2OGPTE_API_KEY_G_T,
    }

    # h2oGPTe servers and/or proxied/hosted LLMs are unstable and causing hangs, times
    # and test failures. Also, API keys are frequently purged. This variable allows to
    # switch all test to the currently working LLM models.
    H2OGPTE_URL_2_LLM = {
        t_gg.H2OGPTE_D.server_url: [],
        t_gg.H2OGPTE_C_D.server_url: [
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "h2oai/h2o-danube3-4b-chat",
            "Qwen/Qwen2-VL-7B-Instruct",
            "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
            "Qwen/Qwen2.5-72B-Instruct",
            "Qwen/Qwen2-VL-72B-Instruct",
            "h2oai/h2ovl-mississippi-2b",
            "mistralai/Pixtral-12B-2409",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "meta-llama/Meta-Llama-3.3-70B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "upstage/SOLAR-10.7B-Instruct-v1.0",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "google/gemma-2-27b-it",
            "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
            "meta-llama/Llama-3.2-3B-Instruct-Turbo",
            "mistral-tiny",
            "mistral-small-latest",
            "mistral-large-latest",
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash-latest",
            "claude-3-haiku-20240307",
            "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet-20241022",
            "Qwen/QwQ-32B-Preview",
        ],
        t_gg.H2OGPTE_G.server_url: [],
        t_gg.H2OGPTE_I_D.server_url: [
            "h2oai/h2o-danube3-4b-chat",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "OpenGVLab/InternVL-Chat-V1-5",
            "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
            "mistral-tiny",
            "mistral-small-latest",
            "mistral-medium-latest",
            "mistral-large-latest",
            "claude-3-sonnet-20240229",
            "claude-3-5-sonnet-20240620",
            "claude-3-haiku-20240307",
            "gpt-4o",
            "gpt-4o-mini",
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash-latest",
        ],
        t_gg.H2OGPTE_H2O_W.server_url: [],
        t_gg.H2OGPTE_G_T.server_url: [],
    }

    def __configured_h2ogpte(self) -> h2o_sonar_config.ConnectionConfig:
        self._h2ogpte_connection = given_generative.H2OGPTE_C_D

        return self._h2ogpte_connection

    def is_h2ogpte(self) -> bool:
        if self._is_h2ogpte is None:
            # check if test services config is available
            if not given_generative.is_config():
                self._is_h2ogpte = False
            else:
                env_var_name = Health.H2OGPTE_URL_2_ENV_VAR.get(
                    self.__configured_h2ogpte().server_url
                )

                if not env_var_name or not os.getenv(env_var_name):
                    self._is_h2ogpte = False
                else:
                    self._is_h2ogpte = True

        return self._is_h2ogpte

    def get_h2ogpte(self):
        """Get h2oGPTe server to use for the testing. h2oGPTe servers and/or
        proxied/hosted LLMs are unstable and causing hangs, times and test failures.
        Also, API keys are frequently purged. This variable allows to switch all test to
        the currently working server.

        Returns
        -------
        h2o_sonar_config.ConnectionConfig | None
            h2oGPTe server to use for the testing, or None if config not available.

        """
        # check if test services config is available
        if not given_generative.is_config():
            print(
                "WARNING: Test services config not available. "
                "get_h2ogpte() returning None."
            )
            return None

        if not self._h2ogpte_connection:
            # IMPROVE: ping all servers / do a sanity call to find a working one
            configured_connection = self.__configured_h2ogpte()
            if not configured_connection.token:
                env_var_name = Health.H2OGPTE_URL_2_ENV_VAR.get(
                    configured_connection.server_url
                )
                raise ValueError(
                    f"API key for h2oGPTe server '{configured_connection.name}' "
                    f"is not set using environment variable: {env_var_name} "
                )

            self._h2ogpte_connection = configured_connection

        return self._h2ogpte_connection

    def get_h2ogpte_llm(self):
        return self.get_h2ogpte()

    def get_h2ogpte_models(self, h2ogpte_models: list[str] | None = 0) -> list[str]:
        """Get non-hanging h2oGPTe models to use for the testing.

        Parameters
        ----------
        h2ogpte_models : list[str] | None
            List of h2oGPTe models to use for the testing - the method will verify
            that the models are available on the server.

        Returns
        -------
        list[str]
            List of h2oGPTe models to use for the testing.

        """
        if not self._h2ogpte_models:
            self._h2ogpte_models = Health.H2OGPTE_URL_2_LLM.get(
                self.get_h2ogpte().server_url
            )
            # fallback
            if not self._h2ogpte_models:
                self._h2ogpte_models = [
                    given_generative.LLM_CAPYBARA,
                    given_generative.LLM_MIXTRAL_8x7B,
                ]

        if h2ogpte_models:
            return [model for model in self._h2ogpte_models if model in h2ogpte_models]

        return self._h2ogpte_models

    # ollama

    def is_ollama(self) -> bool:
        if self._is_ollama is None:
            # ollama must be deployed (locally)
            self._is_ollama = False
            # self._is_ollama = True
        return self._is_ollama

    def get_ollama(self):
        if self._ollama_connection is None:
            # self._ollama_connection = given_generative.OLLAMA_LOCAL
            self._ollama_connection = given_generative.OLLAMA_REMOTE
        return self._ollama_connection

    def get_ollama_models(self):
        if self._ollama_models is None:
            self._ollama_models = [
                "llama2:latest",
                "llama3:latest",
                "mistral:latest",
                "phi:latest",
                "zephyr:latest",
            ]
        return self._ollama_models

    def get_judge_cfg(
        self, floss: bool = False
    ) -> h2o_sonar_config.EvaluationJudgeConfig:
        """Get judge configuration for the tests.

        Parameters
        ----------
        floss : bool
            Use FLOSS judge (like llama) if True, otherwise use high-quality judge like
            GPT 4o+.

        """
        if floss:
            # judge_llm_model = given_generative.LLM_LLAMA_70B
            print("WARNING: h2oGPTe no longer hosts FLOSS judges - fallback to ")
            judge_llm_model = given_generative.H2OGPTE_JUDGE_LLM_MODEL_NAME
            judge_connection = self.get_h2ogpte_llm()
            judge_type = h2o_sonar_config.EvaluationJudgeType.h2ogpte_llm.name
        else:
            if self.is_openai():
                # inject GPT 3.5 judge (default is tested by test_evaluate.py)
                judge_llm_model = given_generative.LLM_GPT_4
                judge_connection = given_generative.OPENAI_LLM
                judge_type = h2o_sonar_config.EvaluationJudgeType.openai_llm.name
            else:
                # inject ALTERNATIVE judge
                judge_llm_model = given_generative.LLM_GPT_4
                judge_connection = self.get_h2ogpte_llm()
                judge_type = h2o_sonar_config.EvaluationJudgeType.h2ogpte_llm.name

        return h2o_sonar_config.EvaluationJudgeConfig(
            name=(
                f"CUSTOM judge EvalStudio: "
                f"{judge_llm_model} @ {judge_connection.connection_type}"
            ),
            description=(
                f"Custom LLM judge to be used by evaluators: {judge_llm_model} "
                f"hosted by {judge_connection.name}"
            ),
            judge_type=judge_type,
            connection=judge_connection,
            llm_model_name=judge_llm_model,
        )

    @staticmethod
    def probe_judge(
        judge_type: h2o_sonar_config.EvaluationJudgeType,
        judge_connection: h2o_sonar_config.ConnectionConfig,
        judge_llm_model: str,
    ) -> tuple[str, h2o_sonar_config.ConnectionConfig, str]:
        try:
            client_health_check = genai.get_client_for_connection(judge_connection)
            answer_health_check = client_health_check.health_check(judge_llm_model)
            print(f"Health check answer ({judge_llm_model}): {answer_health_check}")
            return judge_type.name, judge_connection, judge_llm_model
        except Exception as ex:
            msg = f"Judge LLM health check for {judge_llm_model} failed: {ex}"
            print(msg)
            raise RuntimeError(msg)


# RAG, LLM and judge heat checks
health = Health()


def file_hash_sha256(filepath: str) -> str:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    hasher = hashlib.sha256()
    with open(filepath, "rb") as file:
        while True:
            chunk = file.read(65536)
            if not chunk:
                break
            hasher.update(chunk)

    return hasher.hexdigest()
