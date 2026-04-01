# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
#
# Kaggle LLM Science Exam:
# - https://www.kaggle.com/competitions/kaggle-llm-science-exam
#
import pytest

from h2o_sonar import evaluate
from h2o_sonar.evaluators import (
    rag_answer_correctness_evaluator as answer_correctness_evaluator,
)
from h2o_sonar.evaluators import rag_answer_similarity_evaluator as similarity_evaluator
from h2o_sonar.evaluators import (
    rag_context_relevancy_evaluator as context_relevancy_evaluator,
)
from h2o_sonar.evaluators import rag_ragas_evaluator
from h2o_sonar.lib.api import models
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.skip(reason="Test lab builder requires h2oGPTe server")
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_build_test_lab(tmp_path):
    #
    # GIVEN
    #

    # h2oGPTe server
    h2ogpte_connection = test_utils.health.get_h2ogpte()

    # collection ID w/ uploaded encyclopedia documents (h2oGPTe server specific)
    collection_id = "2c200000-fdef-4594-96f0-72fd0fc4216d"
    collection_name = "test-wiki"

    # models (fastest h2oGPT and GPT)
    llm_model_names = [
        given_generative.H2OGPTE_JUDGE_LLM_MODEL_NAME,
        "h2oai/h2ogpt-4096-llama2-13b-chat",
    ]
    # llm_model_names = genai.get_h2ogpte_llm_models(h2ogpte_connection)

    test_suite = testing.RagTestSuiteConfig.load_from_json(
        test_utils.find_locally(
            # "data/generative/kaggle_llm_science_exam_test_suite_small.json"
            "data/generative/kaggle_llm_science_exam_test_suite_small_25.json"  # 7:35
        )
    )

    # OPTIONAL DESCOPE: faster / smaller test (debugging)
    # llm_model_names = llm_model_names[:5]
    # test_suite.test_cases = test_suite.test_cases[:10]

    #
    # WHEN
    #

    # test lab (RAG product)
    test_lab = testing.RagTestLab.from_rag_test_suite(
        rag_connection=h2ogpte_connection,
        rag_test_suite=test_suite,
        rag_model_type=models.ExplainableModelType.h2ogpte,
        llm_model_names=llm_model_names,
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )
    # test lab: do NOT build > known collection ID > no need to chat w/ h2oGPTe server
    # test_lab.build()
    test_lab.bind(
        collection_id=collection_id,
        collection_name=collection_name,
        corpus=["https://www.wikipedia.org"],
    )

    # test lab: complete dataset w/ ACTUAL values from the h2oGPTe server (answers, ...)
    test_lab.complete_dataset(
        complete_context=True, save_as_you_go=tmp_path / "wip_testlab.json"
    )
    # backup fully resolved dataset
    test_lab.save_as_json(tmp_path / "test_lab_with_actual_values.json")


@pytest.mark.skip(reason="Test is long running + requires OpenAI API key $")
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_load_lab(tmp_path):
    """Evaluate KGM model from:
    https://www.kaggle.com/competitions/kaggle-llm-science-exam/leaderboard

    """
    #
    # GIVEN
    #

    # h2oGPTe server
    h2ogpte_connection = test_utils.health.get_h2ogpte()

    # 4x evaluators, 25 prompts duration = 32:00
    test_lab_path = "data/generative/kaggle_llm_science_exam_test_lab_4x_25.json"
    # pre-built TEST LAB config
    # "kaggle_llm_science_exam_test_lab.json"
    # "data/generative/kaggle_llm_science_exam_test_lab_2x.json"
    # "data/generative/kaggle_llm_science_exam_test_lab_2x_small_3.json"
    # 4x evaluators duration = 4:20
    # "data/generative/kaggle_llm_science_exam_test_lab_2x_small_25.json"
    # "data/generative/kaggle_llm_science_exam_test_lab_2x_small_200.json"

    # test lab (load cfg w/ actual values - build/chat not needed)
    test_lab = testing.RagTestLab.load_from_json(
        llm_host_connection=h2ogpte_connection,
        file_path=test_lab_path,
        docs_cache_dir="data/generative/h2ogpte-documents",
    )

    #
    # WHEN
    #

    evaluation = evaluate.run_evaluation(
        # dataset w/ prompts, constraints and model keys
        dataset=test_lab.dataset,
        # models to be evaluated / compared to get leaderboard
        models=list(test_lab.evaluated_models.values()),
        # evaluators
        evaluators=[
            rag_ragas_evaluator.RagasEvaluator().evaluator_id(),
            answer_correctness_evaluator.AnswerCorrectnessEvaluator().evaluator_id(),
            similarity_evaluator.AnswerSemanticSimilarityEvaluator().evaluator_id(),
            context_relevancy_evaluator.ContextRelevancyEvaluator().evaluator_id(),
        ],
        # where to save the report
        results_location=tmp_path,
    )

    #
    # THEN
    #

    print(f"{evaluation}")
    assert not evaluation.get_failed_evaluator_ids()


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
