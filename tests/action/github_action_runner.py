# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import os
import pathlib

from evaluators import rag_ragas_evaluator

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import interpret
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.integrations import genai
from h2o_sonar.utils import testing


# shell environment: INPUT_ prefix is added by the GH Action runtime
ENV_SERVER_URL = "INPUT_H2OGPTE_SERVER_URL"
ENV_API_KEY = "INPUT_H2OGPTE_API_KEY"
ENV_TEST_CONFIG_PATH = "INPUT_TEST_CONFIG_PATH"


def run_ragas_explainer(
    server_url: str,
    api_key: str,
    test_suite_path: str,
    llm_model_names: list | None = None,
):
    # h2oGPTe server
    h2ogpte_connection = h2o_sonar_config.ConnectionConfig(
        connection_type=h2o_sonar_config.ConnectionConfigType.H2O_GPT_E.name,
        name="H2O GPT Enterprise",
        description="H2O GPT Enterprise.",
        server_url=server_url,
        token=api_key,
        token_use_type=h2o_sonar_config.TokenUseType.API_KEY.name,
    )
    llm_model_names = genai.H2oGpteRagClient(h2ogpte_connection).list_llm_model_names()

    # test configuration (RAG product test suite): docs + prompts + expected answers
    if not test_suite_path or not pathlib.Path(test_suite_path).exists():
        raise ValueError("Test suite configuration path not defined or does not exist.")

    rag_test_suite = testing.RagTestSuiteConfig.load_from_json(test_suite_path)

    # OPTIONAL DESCOPE: faster / smaller test (debugging)
    # llm_model_names = llm_model_names[:1]
    # rag_tests_config.test_cases = rag_tests_config.test_cases[:1]

    # test lab (RAG product)
    test_lab = testing.RagTestLab.from_rag_test_suite(
        rag_connection=h2ogpte_connection,
        rag_test_suite=rag_test_suite,
        llm_model_names=llm_model_names,
        docs_cache_dir="rag_docs",
    )
    # test lab: DEPLOY the h2oGPTe server (docs sync: S3 > filesystem cache > h2oGPTe)
    test_lab.build()

    # test lab: complete dataset w/ ACTUAL values from the h2oGPTe server (answers, ...)
    test_lab.complete_dataset(save_as_you_go="testlab_wip.json")
    # backup fully resolved dataset
    test_lab.save_as_json("testlab_with_actual_values.json")

    #
    # WHEN
    #

    interpretation = interpret.run_interpretation(
        # dataset w/ prompts, constraints and model keys
        dataset=test_lab.dataset,
        # models to be evaluated / compared to get leaderboard
        models=test_lab.rag_models.values(),
        # evaluators
        explainers=[
            rag_ragas_evaluator.RagasEvaluator().explainer_id(),
        ],
        # where to save the report (current directory)
        results_location="results",
    )

    #
    # THEN
    #
    print(f"{interpretation}")
    assert interpretation
    assert not interpretation.get_failed_explainer_ids()

    # get RAGAS evaluator persistence (will be used to get the leaderboard as JSon)
    ep = persistences.ExplainerPersistence(
        data_dir=interpretation.result.results_location,
        mli_key=interpretation.key,
        username=commons.DEFAULT_USER,
        explainer_id=rag_ragas_evaluator.RagasEvaluator.explainer_id(),
        explainer_job_key=next(iter(interpretation.result.explainers)),
    )
    # get path to the heatmap leaderboard JSon index file
    json_index_path = ep.get_explanation_file_path(
        explanation_type=e10s.LlmHeatmapLeaderboardExplanation.explanation_type(),
        explanation_format=f5s.LlmHeatmapLeaderboardJSonFormat.mime,
    )

    print(f"JSON index path: {json_index_path}")
    # you MAY load the JSon index file and get the paths for per-metric leaderboards


if __name__ == "__main__":
    # ensure h2oGPTe connection information (GH Action input parameters)
    h2ogpte_server_url = os.getenv(ENV_SERVER_URL, "")
    if not h2ogpte_server_url:
        raise ValueError(
            f"h2oGPTe server URL not defined in the environment (GitHub Action "
            f"parameter) - please set {ENV_SERVER_URL}"
        )
    h2ogpte_api_key = os.getenv(ENV_API_KEY, "")
    if not h2ogpte_api_key:
        raise ValueError(
            f"h2oGPTe API key not defined in the environment (GitHub Action "
            f"parameter) - please set {ENV_API_KEY}"
        )

    test_suite_env_path = os.getenv(ENV_TEST_CONFIG_PATH, "")

    run_ragas_explainer(
        server_url=h2ogpte_server_url,
        api_key=h2ogpte_api_key,
        test_suite_path=test_suite_env_path,
        llm_model_names=None,  # specify custom set of model names valid for h2oGPTe
    )
