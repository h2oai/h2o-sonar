# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import pytest

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar.lib.api import judges
from h2o_sonar.lib.integrations import genai
from tests import test_utils
from tests.lib import given_generative


@pytest.mark.skipif(
    test_utils.GitHubActions.is_in_gha(),
    reason="Skipped on GHA as this test was flaky there",
)
@pytest.mark.parametrize(
    "judge_type,for_config",
    [
        # h2oGPT RETIRED: (h2o_sonar_config.EvaluationJudgeType.h2ogpt, False),
        pytest.param(
            h2o_sonar_config.EvaluationJudgeType.h2ogpte,
            False,
            marks=pytest.mark.skipif(
                not given_generative.is_config(),
                reason="Test services config not available",
            ),
        ),
        pytest.param(
            h2o_sonar_config.EvaluationJudgeType.h2ogpte_llm,
            False,
            marks=pytest.mark.skipif(
                not given_generative.is_config(),
                reason="Test services config not available",
            ),
        ),
        pytest.param(
            h2o_sonar_config.EvaluationJudgeType.openai_rag,
            False,
            marks=pytest.mark.skipif(
                not test_utils.health.is_openai(),
                reason="Valid OpenAI key is not available",
            ),
        ),
        # h2oGPT RETIRED: (h2o_sonar_config.EvaluationJudgeType.h2ogpt, False),
        pytest.param(
            h2o_sonar_config.EvaluationJudgeType.h2ogpte,
            True,
            marks=pytest.mark.skipif(
                not given_generative.is_config(),
                reason="Test services config not available",
            ),
        ),
        pytest.param(
            h2o_sonar_config.EvaluationJudgeType.h2ogpte_llm,
            True,
            marks=pytest.mark.skipif(
                not given_generative.is_config(),
                reason="Test services config not available",
            ),
        ),
        pytest.param(
            h2o_sonar_config.EvaluationJudgeType.openai_rag,
            True,
            marks=pytest.mark.skipif(
                not test_utils.health.is_openai(),
                reason="Valid OpenAI key is not available",
            ),
        ),
        pytest.param(
            h2o_sonar_config.EvaluationJudgeType.ollama,
            False,
            marks=pytest.mark.skipif(
                not test_utils.health.is_ollama(),
                reason="ollama deployment is not available",
            ),
        ),
        pytest.param(
            h2o_sonar_config.EvaluationJudgeType.anthropic_llm,
            False,
            marks=pytest.mark.skipif(
                not test_utils.health.is_anthropic(),
                reason="Anthropic API key is not available",
            ),
        ),
        pytest.param(
            h2o_sonar_config.EvaluationJudgeType.anthropic_llm,
            True,
            marks=[
                pytest.mark.skipif(
                    not test_utils.health.is_anthropic(),
                    reason="Anthropic API key is not available",
                ),
                pytest.mark.skipif(
                    not given_generative.is_config(),
                    reason="Test services config not available",
                ),
            ],
        ),
    ],
)
@pytest.mark.flaky
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_get_judge_for_connection(
    tmp_path, judge_type: h2o_sonar_config.EvaluationJudgeType, for_config: bool
):
    #
    # GIVEN
    #
    print(f"Testing judge type: {judge_type}")

    if judge_type in [
        h2o_sonar_config.EvaluationJudgeType.ollama,
    ]:
        judge_llm_model_name = "llama3"
    elif judge_type == h2o_sonar_config.EvaluationJudgeType.openai_rag:
        judge_llm_model_name = "gpt-3.5-turbo-1106"  # openai has "3.5", h2ogpte "35"
    elif judge_type == h2o_sonar_config.EvaluationJudgeType.h2ogpt:
        # v model is changed based on models provided by h2oGPTe and/or performance
        judge_llm_model_name = test_utils.health.LLM_LLAMA31_70B
    elif judge_type == h2o_sonar_config.EvaluationJudgeType.anthropic_llm:
        judge_llm_model_name = given_generative.LLM_CLAUDE_SONNET
    else:
        judge_llm_model_name = given_generative.H2OGPTE_JUDGE_LLM_MODEL_NAME

    corpus_doc_path = tmp_path / "corpus.txt"
    with open(corpus_doc_path, "w") as f:
        f.write(
            "Most human adults have 32 teeth. These are permanent teeth, also known as "
            "secondary teeth. They begin to erupt around age 6 and continue to come "
            "in until the early 20s. Wisdom teeth are the last teeth to erupt, "
            "usually between the ages of 17 and 25. However, some people don't have "
            "all their wisdom teeth, or they may not erupt completely through the "
            "gums."
        )

    client = None
    collection_id = None
    try:
        if judge_type in [
            h2o_sonar_config.EvaluationJudgeType.h2ogpt,
        ]:
            connection = test_utils.health.get_h2ogpt()
        elif judge_type in [
            h2o_sonar_config.EvaluationJudgeType.h2ogpte_llm,
        ]:
            connection = test_utils.health.get_h2ogpte()
        elif judge_type in [
            h2o_sonar_config.EvaluationJudgeType.h2ogpte,
        ]:
            connection = test_utils.health.get_h2ogpte()
            client = genai.get_client_for_connection(connection)
            [collection_id, collection_url] = client.create_collection(
                doc_paths=[corpus_doc_path]
            )
            print(f"Created h2oGPTe collection ID: {collection_id} / {collection_url}")
        elif judge_type in [
            h2o_sonar_config.EvaluationJudgeType.openai_rag,
        ]:
            connection = given_generative.OPENAI_RAG
            client = genai.get_client_for_connection(connection)
            collection_id = client.create_collection(doc_paths=[corpus_doc_path])
            print(f"Created h2oGPTe collection ID: {collection_id}")
        elif judge_type in [
            h2o_sonar_config.EvaluationJudgeType.anthropic_llm,
        ]:
            connection = test_utils.health.get_anthropic()
        elif judge_type in [
            h2o_sonar_config.EvaluationJudgeType.ollama,
        ]:
            connection = test_utils.health.get_ollama()
        else:
            raise ValueError(f"Unknown judge type: {judge_type}")

        # connection must be in H2O Sonar config
        h2o_sonar_config.config.add_connection(connection)

        #
        # WHEN
        #
        if for_config:
            judge_cfg = h2o_sonar_config.config.add_evaluation_judge(
                h2o_sonar_config.EvaluationJudgeConfig(
                    name="TEST CUSTOM evaluation judge",
                    description="Custom evaluation judge for the test.",
                    judge_type=judge_type.name,
                    connection=connection,
                    llm_model_name=judge_llm_model_name,
                )
            )
            print(f"Creating judge for CONFIG: {judge_cfg}...")
            judge = judges.get_evaluation_judge_for_config(judge_cfg)
        else:
            print(f"Creating judge for CONNECTION: {judge_type}...")
            judge = judges.get_evaluation_judge_for_connection(
                connection=connection,
                judge_type=judge_type.name,
                llm_model_name=judge_llm_model_name,
            )
        h2o_sonar_config.config.save(
            config_path=str(tmp_path / "h2o_sonar_config_with_judge.json"),
            encrypt=False,
        )
        print(
            f"H2O Sonar config to be used for judgetest:"
            f"\n{h2o_sonar_config.config.to_dict(encrypt=False)}"
        )
        # USE the judge
        answers = judge.evaluate(
            prompts=[
                "How many teeth does a human have? Answer in number of teeth. No text.",
            ],
        )

        #
        # THEN
        #
        print(f"Judge answer     : {answers}")
        assert answers is not None
        answer = answers[0]
        print(f"Judge answer type: {type(answer)}")
        assert isinstance(answer, genai.LlmHostClient.LlmRagAnswer)
        print(f"Judge answer     : {answer.prompt}")
        print(f"Judge answer     : {answer.answer}")

    finally:
        # purge resources created during the test
        print(f"Purging resources for judge type {judge_type}")
        if client and collection_id:
            if judge_type in [
                h2o_sonar_config.EvaluationJudgeType.h2ogpte,
                h2o_sonar_config.EvaluationJudgeType.openai_rag,
            ]:
                # client remembers collection it created and documents it uploaded
                print(f"Purging collection {collection_id} and uploaded documents")
                client.purge_collections()
                client.purge_uploaded_docs()


@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
