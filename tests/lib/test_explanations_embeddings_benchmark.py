# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import pathlib
import time

import pytest

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import evaluate
from h2o_sonar import loggers
from h2o_sonar.evaluators import answer_accuracy_evaluator as e_aa
from h2o_sonar.evaluators import (
    answer_semantic_similarity_per_sentence_evaluator as e_assprs,
)
from h2o_sonar.evaluators import rag_chunk_relevancy_evaluator as e_rch
from h2o_sonar.evaluators import rag_context_mean_reciprocal_rank_evaluator as e_mrr
from h2o_sonar.evaluators import rag_groundedness_evaluator as e_rg
from h2o_sonar.evaluators import rag_tokens_presence_evaluator as e_tm
from h2o_sonar.evaluators import rag_tokens_presence_evaluator as evaluator
from h2o_sonar.evaluators import rouge_evaluator as e_r
from h2o_sonar.lib.api import explanations
from h2o_sonar.lib.api import models
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative
from tests.lib import test_explanations


"""Comparison of Chinese vs. non-Chinese embedding models:"""

# data directories
DIR_BASE_DATA = "data/generative/eval_s3/benchmark-embeddings-2025-01-07"

# test suites
SUITE_SR_11_7 = "test-suite-sr-11-7-50p.json"
SUITE_SINGTEL = "test-suite-singtel-multichoice-74p.json"
SUITE_CALL_CENTER = "test-suite-call-center-50p.json"

# LLMs
# claude-3-7-sonnet-20250219 > claude-sonnet-4-5-20250929
LLM_JOBY_AZURE_37 = "claude-3-7-sonnet-20250219"
LLM_JOBY_AZURE_45 = "claude-sonnet-4-5-20250929"

# embeddings
EMBED_EN_CN = "BAAI/bge-large-en-v1.5"
EMBED_EN_NEW = "mixedbread-ai/mxbai-embed-large-v1"
EMBED_MULTI_CN = "BAAI/bge-m3"
EMBED_MULTI_NEW = "google/embeddinggemma-300m-qat-q8_0-unquantized"


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.skip(reason="Upgrade of test suites w/ legacy constraints notation")
@pytest.mark.parametrize(
    "test_suite_path",
    [
        pathlib.Path(DIR_BASE_DATA) / SUITE_CALL_CENTER,
        pathlib.Path(DIR_BASE_DATA) / SUITE_SR_11_7,
        pathlib.Path(DIR_BASE_DATA) / SUITE_SINGTEL,
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_legacy_test_suites_constraints_to_conditions(
    tmp_path: pathlib.Path, test_suite_path: pathlib.Path
):
    #
    # GIVEN
    #
    with open(test_suite_path) as f:
        test_suite_dict = json.load(f)
    assert test_suite_dict

    #
    # WHEN
    #
    for t in test_suite_dict.get("tests", {}):
        for tc in t.get("test_cases", {}):
            cs = tc.get("constraints")
            if not cs:
                print(f"SKIPPING test as constraints are MISSING: '{cs}'")
                continue
            cd = tc.get("condition")
            if cd:
                print(f"SKIPPING test as condition is present: '{cd}'")
                continue

            condition = evaluator.constraints_to_condition(cs)
            tc["condition"] = condition
            print(f"CONVERSION: '{cs}' =-> '{condition}'")

    #
    # THEN
    #
    new_test_suite_path = tmp_path / test_suite_path.name
    with open(new_test_suite_path, "w") as f:
        json.dump(test_suite_dict, f, indent=4)
    print(f"DONE new test suite: file://{new_test_suite_path}")


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.skip(reason="Embeddings performance comparison test lab builder")
@pytest.mark.parametrize(
    "h2ogpte_connection,collection_id,llm_model_names,test_suite_filename",
    [
        # ########################################################
        # h2oGPTe: h2ogpte.az-gpte.qa-azure.h2o.dev @ Azure (Joby)
        # ########################################################
        # Call center (BGE EN) bge-large-en-v1.5
        (
            given_generative.H2OGPTE_AZURE_J,
            "450ed57a-9de3-4846-a6e4-01d8688c4ec9",
            [LLM_JOBY_AZURE_37],
            SUITE_CALL_CENTER,
        ),
        # Call center (MXBAI EN) embeddinggemma-300m-qat-q8_0-unquantized
        (
            given_generative.H2OGPTE_AZURE_J,
            "6595bd2b-4b03-4329-b7d9-20c47dfcb72d",
            [LLM_JOBY_AZURE_37],
            SUITE_CALL_CENTER,
        ),
        # Call center (BGE MULTILINGUAL) bge-m3
        (
            given_generative.H2OGPTE_AZURE_J,
            "5a195e7a-d00b-4b39-aeb1-d61d66ca4fe7",
            [LLM_JOBY_AZURE_37],
            SUITE_CALL_CENTER,
        ),
        # Call center (MXBAI MULTILINGUAL) mxbai-embed-large-v1
        (
            given_generative.H2OGPTE_AZURE_J,
            "1d5bd782-90b6-4697-bbb8-d1c4ca81f8e3",
            [LLM_JOBY_AZURE_37],
            SUITE_CALL_CENTER,
        ),
        # ########################################################
        # Singtel multichoice (BGE EN) bge-large-en-v1.5
        (
            given_generative.H2OGPTE_AZURE_J,
            "656f99ba-9224-4b16-9f95-7bb47d410d07",
            [LLM_JOBY_AZURE_45],
            SUITE_SINGTEL,
        ),
        # Singtel multichoice (MXBAI EN) embeddinggemma-300m-qat-q8_0-unquantized
        (
            given_generative.H2OGPTE_AZURE_J,
            "9c10ad1b-a526-4f85-8360-c3dd1e1219b1",
            [LLM_JOBY_AZURE_45],
            SUITE_SINGTEL,
        ),
        # Singtel multichoice (BGE MULTILINGUAL) bge-m3
        (
            given_generative.H2OGPTE_AZURE_J,
            "9dc9efbf-c818-48fd-9e5e-96bf9e9f7e0c",
            [LLM_JOBY_AZURE_45],
            SUITE_SINGTEL,
        ),
        # Singtel multichoice (MXBAI MULTILINGUAL) mxbai-embed-large-v1
        (
            given_generative.H2OGPTE_AZURE_J,
            "6d4d4af0-e8f2-4aea-9eec-3980f0e0a293",
            [LLM_JOBY_AZURE_45],
            SUITE_SINGTEL,
        ),
        # ########################################################
        # SR 11-7 (BGE EN) bge-large-en-v1.5
        (
            given_generative.H2OGPTE_AZURE_J,
            "d3020997-052d-43fb-9405-74a60781d6a6",
            [LLM_JOBY_AZURE_37],
            SUITE_SR_11_7,
        ),
        # SR 11-7 (MXBAI EN) embeddinggemma-300m-qat-q8_0-unquantized
        (
            given_generative.H2OGPTE_AZURE_J,
            "7e42e741-4156-43d0-a602-3c0afaeebb58",
            [LLM_JOBY_AZURE_37],
            SUITE_SR_11_7,
        ),
        # SR 11-7 (BGE MULTILINGUAL) bge-m3
        (
            given_generative.H2OGPTE_AZURE_J,
            "3460acbd-db16-4efb-bccd-4e62324f5d23",
            [LLM_JOBY_AZURE_37],
            SUITE_SR_11_7,
        ),
        # SR 11-7 (MXBAI MULTILINGUAL) mxbai-embed-large-v1
        (
            given_generative.H2OGPTE_AZURE_J,
            "071b7a61-a540-4f5b-b8ca-0595f709d24d",
            [LLM_JOBY_AZURE_37],
            SUITE_SR_11_7,
        ),
        # #####################################################
        # h2oGPTe: i-d
        # #####################################################
        # SR 11-7 (BGE)
        (
            given_generative.H2OGPTE_I_D,
            "9204bf9e-e523-4461-a0a9-1353e3598291",
            ["claude-sonnet-4-5-20250929"],
            SUITE_SR_11_7,
        ),
        # SR 11-7 (MXBAI)
        (
            given_generative.H2OGPTE_I_D,
            "...",
            ["claude-sonnet-4-5-20250929"],
            SUITE_SR_11_7,
        ),
        # ############################################
        # h2oGPTe: c-d
        # ############################################
        # SR 11-7 (BGE)
        (
            given_generative.H2OGPTE_C_D,
            "02cfb22e-a9d7-49ef-8b7a-3042821d5943",
            ["claude-sonnet-4-5-20250929"],
            SUITE_SR_11_7,
        ),
        # SR 11-7 (MXBAI)
        (
            given_generative.H2OGPTE_C_D,
            "c8d3d0d3-18e4-4153-9544-2a065fcef424",
            ["claude-sonnet-4-5-20250929"],
            SUITE_SR_11_7,
        ),
        # Singtel Report multi-choice (BGE)
        (
            given_generative.H2OGPTE_C_D,
            "72e096da-a9dc-4b88-bac8-593b471c1b01",
            ["claude-sonnet-4-5-20250929"],
            SUITE_SR_11_7,
        ),
        # Singtel Report multi-choice (MXBAI)
        (
            given_generative.H2OGPTE_C_D,
            "02cfb22e-a9d7-49ef-8b7a-3042821d5943",
            ["claude-sonnet-4-5-20250929"],
            SUITE_SR_11_7,
        ),
        # CALL CENTER (BGE)
        (
            given_generative.H2OGPTE_C_D,
            "bea0d0da-79ea-4eab-9e32-1d011b4751ad",
            ["claude-sonnet-4-5-20250929"],
            SUITE_SR_11_7,
        ),
        # CALL CENTER (MXBAI)
        (
            given_generative.H2OGPTE_C_D,
            "eb9eb8e8-0919-4fee-99fe-a07e04a407d1",
            ["claude-sonnet-4-5-20250929"],
            SUITE_SR_11_7,
        ),
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_complete_lab_on_given_collection(
    tmp_path: pathlib.Path,
    h2ogpte_connection: h2o_sonar_config.ConnectionConfig,
    collection_id: str,
    llm_model_names: list[str],
    test_suite_filename: str,
):
    """Test lab completed on existing h2oGPTe collection."""

    #
    # GIVEN
    #

    #
    # WHEN: string OR dictionary {test case key: collection ID}
    #

    # 1) test suite
    test_suite_path = pathlib.Path(DIR_BASE_DATA) / test_suite_filename
    test_suite = testing.RagTestSuiteConfig.load_from_json(test_suite_path)

    # 2) complete the test lab using the predefined collection ID
    test_lab = testing.RagTestLab.from_rag_test_suite(
        rag_connection=h2ogpte_connection,
        rag_test_suite=test_suite,
        rag_model_type=models.ExplainableModelType.h2ogpte,
        llm_model_names=llm_model_names,
        predefined_collection_id=collection_id,
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )
    # check that the collection ID is set
    for em in test_lab.evaluated_models.values():
        assert em.collection_id
    # 3) build() test lab ~ just set the collection ID
    test_lab.build()
    test_lab_path = tmp_path / f"test-lab-BEFORE-{test_suite_filename}"
    test_lab.save_as_json(test_lab_path)
    # 4) complete the test lab
    test_lab.complete_dataset()
    test_lab_path = tmp_path / f"test-lab-{test_suite_filename}"
    test_lab.save_as_json(test_lab_path)

    #
    # THEN
    #
    assert test_lab
    print(f"Test lab:\n  file://{test_lab_path}")


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.skip(reason="Test lab model configuration injector")
@pytest.mark.parametrize(
    "name,test_lab_base_path",
    [
        (
            "singtel-multichoice-74",
            pathlib.Path(DIR_BASE_DATA) / "results" / "singtel" / "h2ogpte-azure-joby",
        ),
        (
            "sr-11-7-50",
            pathlib.Path(DIR_BASE_DATA) / "results" / "sr" / "h2ogpte-azure-joby",
        ),
        (
            "call-center-50",
            pathlib.Path(DIR_BASE_DATA)
            / "results"
            / "call-center"
            / "h2ogpte-azure-joby",
        ),
    ],
)
def test_inject_lab_model_cfg(
    tmp_path: pathlib.Path, name: str, test_lab_base_path: pathlib.Path
):
    #
    # GIVEN
    #

    test_lab_names = [
        f"en-bge-test-lab-{name}p.json",
        f"en-mxbai-test-lab-{name}p.json",
        f"multi-bge-test-lab-{name}p.json",
        f"multi-mxbai-test-lab-{name}p.json",
    ]
    embeddings = [
        EMBED_EN_CN,
        EMBED_MULTI_NEW,
        EMBED_MULTI_CN,
        EMBED_EN_NEW,
    ]

    #
    # WHEN
    #
    print("Injecting...")
    for e, test_lab_name in enumerate(test_lab_names):
        test_lab_path = test_lab_base_path / test_lab_name

        if not test_lab_path.exists():
            print(f"SKIPPING non-existent test lab: {test_lab_name}")
            continue

        with open(test_lab_path) as f:
            test_lab_dict = json.load(f)

        if len(test_lab_dict.get("models", [])) > 0:
            if not test_lab_dict["models"][0].get("model_cfg"):
                test_lab_dict["models"][0]["model_cfg"] = {}
            test_lab_dict["models"][0]["model_cfg"]["embedding_model"] = embeddings[e]

        #
        # THEN
        #
        new_test_lab_path = tmp_path / test_lab_name
        with open(new_test_lab_path, "w") as f:
            json.dump(test_lab_dict, f, indent=4)
            print(f"DONE new test lab: file://{new_test_lab_path}")


DIR_RESULT_JOBY_BASE = str(
    pathlib.Path.home() / "h2o-eval-studio-EMBEDDINGS-benchmark-2026-01-22/ALL/results"
)
DIR_RESULTS_JOBY_SR = f"{DIR_RESULT_JOBY_BASE}/sr/h2ogpte-azure-joby"
DIR_RESULTS_JOBY_CC = f"{DIR_RESULT_JOBY_BASE}/call-center/h2ogpte-azure-joby"
DIR_RESULTS_JOBY_S = f"{DIR_RESULT_JOBY_BASE}/singtel/h2ogpte-azure-joby"


@pytest.mark.skip(reason="Evaluations comparison run for embeddings benchmark")
@pytest.mark.parametrize(
    (
        "baseline_test_lab_path,current_test_lab_path,baseline_llm_model,"
        "current_llm_model,evaluators,test_key,display_name"
    ),
    [
        # Call center ENGLISH: run evaluations > merge > compare
        (
            f"{DIR_RESULTS_JOBY_CC}/en-bge-test-lab-call-center-50p.json",
            f"{DIR_RESULTS_JOBY_CC}/en-mxbai-test-lab-call-center-50p.json",
            LLM_JOBY_AZURE_37,
            LLM_JOBY_AZURE_37,
            [
                # AA
                e_aa.AnswerAccuracyEvaluator.evaluator_id(),
                e_assprs.AnswerSemanticSimilarityPerSentenceEvaluator.evaluator_id(),
                e_r.RougeEvaluator.evaluator_id(),
                e_rg.RagGroundednessEvaluator.evaluator_id(),
                e_tm.RagStrStrEvaluator.evaluator_id(),
                # CTX
                e_rch.ContextChunkRelevancyEvaluator.evaluator_id(),
                e_mrr.MeanReciprocalRankEvaluator.evaluator_id(),
            ],
            False,
            "CALL-CENTER-EN",
        ),
        # Call center MULTILINGUAL: run evaluations > merge > compare
        (
            f"{DIR_RESULTS_JOBY_CC}/multi-bge-test-lab-call-center-50p.json",
            f"{DIR_RESULTS_JOBY_CC}/multi-gemma-test-lab-call-center-50p.json",
            LLM_JOBY_AZURE_37,
            LLM_JOBY_AZURE_37,
            [
                # AA
                e_aa.AnswerAccuracyEvaluator.evaluator_id(),
                e_assprs.AnswerSemanticSimilarityPerSentenceEvaluator.evaluator_id(),
                e_r.RougeEvaluator.evaluator_id(),
                e_rg.RagGroundednessEvaluator.evaluator_id(),
                e_tm.RagStrStrEvaluator.evaluator_id(),
                # CTX
                e_rch.ContextChunkRelevancyEvaluator.evaluator_id(),
                e_mrr.MeanReciprocalRankEvaluator.evaluator_id(),
            ],
            False,
            "CALL-CENTER-MULTI",
        ),
        # ##################################################################
        # Singtel multichoice ENGLISH: run evaluations > merge > compare
        (
            f"{DIR_RESULTS_JOBY_S}/en-bge-test-lab-singtel-multichoice-74p.json",
            f"{DIR_RESULTS_JOBY_S}/en-mxbai-test-lab-singtel-multichoice-74p.json",
            LLM_JOBY_AZURE_45,
            LLM_JOBY_AZURE_45,
            [
                # AA
                # TOO SHORT AA: e_aa.AnswerAccuracyEvaluator.evaluator_id(),
                e_assprs.AnswerSemanticSimilarityPerSentenceEvaluator.evaluator_id(),
                e_r.RougeEvaluator.evaluator_id(),
                e_rg.RagGroundednessEvaluator.evaluator_id(),
                e_tm.RagStrStrEvaluator.evaluator_id(),
                # CTX
                e_rch.ContextChunkRelevancyEvaluator.evaluator_id(),
                e_mrr.MeanReciprocalRankEvaluator.evaluator_id(),
            ],
            False,
            "SINGTEL-EN",
        ),
        # Singtel multichoice MULTILINGUAL: run evaluations > merge > cmp
        (
            f"{DIR_RESULTS_JOBY_S}/multi-bge-test-lab-singtel-multichoice-74p.json",
            f"{DIR_RESULTS_JOBY_S}/multi-gemma-test-lab-singtel-multichoice-74p.json",
            LLM_JOBY_AZURE_45,
            LLM_JOBY_AZURE_45,
            [
                # AA
                # TOO SHORT AA: e_aa.AnswerAccuracyEvaluator.evaluator_id(),
                e_assprs.AnswerSemanticSimilarityPerSentenceEvaluator.evaluator_id(),
                e_r.RougeEvaluator.evaluator_id(),
                e_rg.RagGroundednessEvaluator.evaluator_id(),
                e_tm.RagStrStrEvaluator.evaluator_id(),
                # CTX
                e_rch.ContextChunkRelevancyEvaluator.evaluator_id(),
                e_mrr.MeanReciprocalRankEvaluator.evaluator_id(),
            ],
            False,
            "SINGTEL-MULTI",
        ),
        # ##################################################################
        # SR 11-7 ENGLISH: run evaluations > merge > compare
        (
            f"{DIR_RESULTS_JOBY_SR}/en-bge-test-lab-sr-11-7-50p.json",
            f"{DIR_RESULTS_JOBY_SR}/en-mxbai-test-lab-sr-11-7-50p.json",
            LLM_JOBY_AZURE_37,
            LLM_JOBY_AZURE_37,
            [
                # AA
                e_aa.AnswerAccuracyEvaluator.evaluator_id(),
                e_assprs.AnswerSemanticSimilarityPerSentenceEvaluator.evaluator_id(),
                e_r.RougeEvaluator.evaluator_id(),
                e_rg.RagGroundednessEvaluator.evaluator_id(),
                e_tm.RagStrStrEvaluator.evaluator_id(),
                # CTX
                e_rch.ContextChunkRelevancyEvaluator.evaluator_id(),
                e_mrr.MeanReciprocalRankEvaluator.evaluator_id(),
            ],
            False,
            "SR-EN",
        ),
        # SR 11-7 MULTILINGUAL: run evaluations > merge > compare
        (
            f"{DIR_RESULTS_JOBY_SR}/multi-bge-test-lab-sr-11-7-50p.json",
            f"{DIR_RESULTS_JOBY_SR}/multi-gemma-test-lab-sr-11-7-50p.json",
            LLM_JOBY_AZURE_37,
            LLM_JOBY_AZURE_37,
            [
                # AA
                e_aa.AnswerAccuracyEvaluator.evaluator_id(),
                e_assprs.AnswerSemanticSimilarityPerSentenceEvaluator.evaluator_id(),
                e_r.RougeEvaluator.evaluator_id(),
                e_rg.RagGroundednessEvaluator.evaluator_id(),
                e_tm.RagStrStrEvaluator.evaluator_id(),
                # CTX
                e_rch.ContextChunkRelevancyEvaluator.evaluator_id(),
                e_mrr.MeanReciprocalRankEvaluator.evaluator_id(),
            ],
            False,
            "SR-MULTI",
        ),
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_evaluate_and_compare(
    tmp_path: pathlib.Path,
    baseline_test_lab_path: str,
    current_test_lab_path: str,
    baseline_llm_model: str,
    current_llm_model: str,
    evaluators: list[str],
    test_key: bool,
    display_name: str,
):
    """Test comparison of 2 evaluations.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temp directory for storing of the test results.
    baseline_test_lab_path : pathlib.Path
        Test lab to be used as baseline model for testing.
    current_test_lab_path : pathlib.Path
        Test lab to be used as current model for testing.
    evaluators : list
        Evaluators to be used in the evaluation.
    test_key : bool
        ``True`` to test evaluations by loading them from filesystem via key
        (while results location path specified), ``False`` to test the comparison
        of ``Evaluation`` object instances.
    display_name : str
        Display name to use in the pytest results (tmp/) to identify the comparison.

    """

    #
    # GIVEN
    #

    evaluations = []
    for test_lab_path in [baseline_test_lab_path, current_test_lab_path]:
        rag_dataset = testing.RagTestLab.load_from_json(
            llm_host_connection=test_utils.health.get_h2ogpt(),
            file_path=test_lab_path,
        )
        llm_models = rag_dataset.evaluated_models.values()

        evaluation = evaluate.run_evaluation(
            dataset=rag_dataset.dataset,
            models=llm_models,
            evaluators=list(evaluators),
            results_location=tmp_path,
            log_level=loggers.DEBUG,
        )

        assert evaluation
        assert not evaluation.is_explainer_failed()
        evaluations.append(evaluation.key if test_key else evaluation)
        print(
            f"Evaluation HTML:\nfile://{evaluation.result.get_html_report_location()}"
        )

    #
    # WHEN
    #

    comparison_methods = [
        explanations.SentenceComparisonMethod.EXACT_MATCH,
        explanations.SentenceComparisonMethod.COSINE_DISTANCE,
    ]
    # IMPROVE: BERTScore skipped (see original test in test_explanations.py)

    diffs = {}
    timings = {}
    for comparison_method in comparison_methods:
        print(f"\nComparing with method: {comparison_method.value}")
        start_time = time.time()
        diff = evaluate.compare_evaluations(
            baseline_evaluation=evaluations[0],
            current_evaluation=evaluations[1],
            # filter by LLM model if diverse host types LLMs are compared
            baseline_llm_model=baseline_llm_model,
            current_llm_model=current_llm_model,
            # always compare ALL evaluators
            results_location=tmp_path,
            comparison_method=comparison_method,
        )
        elapsed_time = time.time() - start_time
        diffs[comparison_method] = diff
        timings[comparison_method] = elapsed_time
        print(f"DONE {comparison_method.value} method in {elapsed_time:.3f}s")

    #
    # THEN
    #
    cmp_denominator = tmp_path / f"{display_name}.json"
    with open(cmp_denominator, "w") as f:
        json.dump({"comparison": display_name}, f, indent=2)

    assert len(diffs) == len(comparison_methods)
    for comparison_method, diff in diffs.items():
        assert diff

        method_name = comparison_method.value
        elapsed_time = timings[comparison_method]

        diff_as_json_dict = diff.to_dict()
        diff_as_json_path = tmp_path / f"diff_{method_name}.json"
        with open(diff_as_json_path, "w") as f:
            json.dump(diff_as_json_dict, f, indent=2)
        print(
            f"\nDiff JSON ({method_name}, {elapsed_time:.3f}s) written to: "
            f"file://{diff_as_json_path}"
        )

        # validate JSON structure and content (only for first method to save time)
        if comparison_method == comparison_methods[0]:
            test_explanations._then_assert_json(diff_as_json_dict, verbose=True)

        diff_html = str(diff.to_html())
        diff_as_html_path = tmp_path / f"diff_{method_name}.html"
        with open(diff_as_html_path, "w") as f:
            f.write(diff_html)
        print(
            f"Diff HTML ({method_name}, {elapsed_time:.3f}s) written to: "
            f"file://{diff_as_html_path}"
        )


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
