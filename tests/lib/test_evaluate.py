# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
#
# This test module is the ONLY module which runs (ALL) evaluators on CI:
#
# - fast, small and cheap: 7'30"
# - it runs all evaluators on the h2oGPTe server
# - it requires OpenAPI key
# - ^ therefore:
#   - only 2 LLM models are evaluated
#   - only 2 tests (2 docs w/ 2 test cases each) are evaluated for RAG
#   - only 1 test w/ 3 test cases are evaluated for LLM
# - test suite has as much as possible features tested (constraints, regexps, ...)
#
# Other evaluators are typically skipped/commented out as their run would be expensive
# (OpenAI cost), take hours (building bigger lab) and utilize the HW resources.
#
#
import json
import logging
import pathlib
import pprint

import pytest

from h2o_sonar import config
from h2o_sonar import evaluate
from h2o_sonar import loggers
from h2o_sonar.evaluators import agent_sanity_check_evaluator as asc_e
from h2o_sonar.evaluators import answer_accuracy_evaluator as a_aaea
from h2o_sonar.evaluators import bertscore_evaluator as bs_e
from h2o_sonar.evaluators import bleu_evaluator as b_e
from h2o_sonar.evaluators import classification_evaluator as c_e
from h2o_sonar.evaluators import contact_information_byop_evaluator as ci_p_e
from h2o_sonar.evaluators import encoding_guardrail_evaluator as en_g_e
from h2o_sonar.evaluators import fairness_bias_evaluator as fb_e
from h2o_sonar.evaluators import gptscore_machine_translation_evaluator as gs_mt_e
from h2o_sonar.evaluators import gptscore_question_answering_evaluator as gs_qa_e
from h2o_sonar.evaluators import gptscore_summary_with_reference_evaluator as gs_sr_e
from h2o_sonar.evaluators import gptscore_summary_without_reference_evaluator as gs_o_r
from h2o_sonar.evaluators import gptscore_summary_without_reference_evaluator as gs_s_e
from h2o_sonar.evaluators import language_mismatch_byop_evaluator as lm_p_e
from h2o_sonar.evaluators import looping_detection_evaluator as ld_e
from h2o_sonar.evaluators import perplexity_evaluator as ppx_e
from h2o_sonar.evaluators import pii_leakage_evaluator as pii_e
from h2o_sonar.evaluators import rag_answer_correctness_evaluator as ac_e
from h2o_sonar.evaluators import rag_answer_relevancy_evaluator as ar_e
from h2o_sonar.evaluators import rag_answer_relevancy_no_judge_evaluator as ar_n_e
from h2o_sonar.evaluators import rag_answer_similarity_evaluator as as_e
from h2o_sonar.evaluators import rag_chunk_relevancy_evaluator as rc_e
from h2o_sonar.evaluators import rag_context_mean_reciprocal_rank_evaluator as rc_m_e
from h2o_sonar.evaluators import rag_context_precision_evaluator as cp_e
from h2o_sonar.evaluators import rag_context_recall_evaluator as crc_e
from h2o_sonar.evaluators import rag_context_relevancy_evaluator as cr_e
from h2o_sonar.evaluators import rag_faithfulness_evaluator as f_e
from h2o_sonar.evaluators import rag_groundedness_evaluator as rg_e
from h2o_sonar.evaluators import rag_hallucination_evaluator as hal_e
from h2o_sonar.evaluators import rag_ragas_evaluator
from h2o_sonar.evaluators import rag_tokens_presence_evaluator as tp_e
from h2o_sonar.evaluators import rouge_evaluator as r_e
from h2o_sonar.evaluators import sensitive_data_leakage_evaluator as sdl_e
from h2o_sonar.evaluators import sexism_byop_evaluator as s_p_e
from h2o_sonar.evaluators import stereotype_byop_evaluator as st_p_e
from h2o_sonar.evaluators import summarization_byop_evaluator as su_p_e
from h2o_sonar.evaluators import summarization_evaluator as s_e
from h2o_sonar.evaluators import toxicity_evaluator as t_e
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.integrations import genai
from h2o_sonar.utils import perturbations
from h2o_sonar.utils import progress
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative
from tests.lib import then_eval


@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_integrity_check():
    #
    # GIVEN
    #
    evaluator_classes = [
        a_aaea.AnswerAccuracyEvaluator,
        ac_e.AnswerCorrectnessEvaluator,
        ar_e.AnswerRelevancyEvaluator,
        ar_n_e.RagAnswerRelevancyNoJudgeEvaluator,
        as_e.AnswerSemanticSimilarityEvaluator,
        asc_e.AgentSanityCheckEvaluator,
        b_e.BleuEvaluator,
        bs_e.BertscoreEvaluator,
        c_e.ClassificationEvaluator,
        ci_p_e.ContactInformationByopEvaluator,
        cp_e.ContextPrecisionEvaluator,
        cr_e.ContextRelevancyEvaluator,
        crc_e.ContextRecallEvaluator,
        en_g_e.EncodingGuardrailEvaluator,
        f_e.FaithfulnessEvaluator,
        fb_e.FairnessBiasEvaluator,
        gs_mt_e.GptScoreMachineTranslationEvaluator,
        gs_o_r.GptScoreSummaryWithoutReferenceEvaluator,
        gs_qa_e.GptScoreQuestionAnsweringEvaluator,
        gs_s_e.GptScoreSummaryWithoutReferenceEvaluator,
        gs_sr_e.GptScoreSummaryWithReferenceEvaluator,
        hal_e.RagHallucinationEvaluator,
        ld_e.LoopingDetectionEvaluator,
        lm_p_e.LanguageMismatchByopEvaluator,
        pii_e.PiiLeakageEvaluator,
        ppx_e.PerplexityEvaluator,
        r_e.RougeEvaluator,
        rag_ragas_evaluator.RagasEvaluator,
        rc_e.ContextChunkRelevancyEvaluator,
        rc_m_e.MeanReciprocalRankEvaluator,
        rg_e.RagGroundednessEvaluator,
        s_e.SummarizationEvaluator,
        s_p_e.SexismByopEvaluator,
        sdl_e.SensitiveDataLeakageEvaluator,
        st_p_e.StereotypeByopEvaluator,
        su_p_e.SummarizationByopEvaluator,
        t_e.ToxicityEvaluator,
        tp_e.RagStrStrEvaluator,
    ]

    #
    # WHEN
    #

    print(f"Checking integrity of {len(evaluator_classes)} evaluators")
    failed_integrity_checks = []
    for evaluator_class in evaluator_classes:
        print(f"Checking integrity of {evaluator_class}")

        #
        # THEN
        #

        try:
            evaluator_class()
        except Exception as e:
            print(f"  {evaluator_class}: {e} ({type(e)})")
            failed_integrity_checks.append((evaluator_class, e))

    if failed_integrity_checks:
        print(
            f"\nFailed integrity checks "
            f"{len(failed_integrity_checks)}/{len(evaluator_classes)}:"
        )
        for evaluator_class, e in failed_integrity_checks:
            print(f"  {evaluator_class}:\n    {e}")


@pytest.mark.parametrize(
    "portable",
    [
        True,
        False,
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_list_evaluators(portable: bool):
    #
    # GIVEN
    #

    #
    # WHEN
    #
    evaluator_descrs = evaluate.list_evaluators(portable=portable)

    #
    # THEN
    #
    print(f"Evaluators [{len(evaluator_descrs)}]:")
    assert len(evaluator_descrs) > 0

    e_ids = [e.id for e in evaluator_descrs]
    print(e_ids)
    for e_id in e_ids:
        assert ".evaluators." in e_id
        assert ".explainers." not in e_id
    e_ids_str = str(e_ids)
    assert "RagasEvaluator" in e_ids_str
    assert "AnswerCorrectnessEvaluator" in e_ids_str
    assert "ContextRelevancyEvaluator" in e_ids_str
    assert "AnswerRelevancyEvaluator" in e_ids_str
    assert "ContextPrecisionEvaluator" in e_ids_str
    assert "ContextRecallEvaluator" in e_ids_str
    assert "FaithfulnessEvaluator" in e_ids_str
    assert "ToxicityEvaluator" in e_ids_str

    non_portable_values = [float("-inf"), float("inf"), "Infinity", "-Infinity"]

    for e_descr in evaluator_descrs:
        print(e_descr.brief_description)

        # ensure no Inf/-Inf
        if portable:
            for p in e_descr.parameters:
                assert p.min_ not in non_portable_values
                assert p.max_ not in non_portable_values
            for k in e_descr.metrics_meta.key_to_metric:
                m_m = e_descr.metrics_meta.key_to_metric[k]
                assert m_m.threshold not in non_portable_values
                if isinstance(m_m.value_range, tuple) and len(m_m.value_range) == 2:
                    assert m_m.value_range[0] not in non_portable_values, e_descr.id
                    assert m_m.value_range[1] not in non_portable_values, e_descr.id
        # ensure brief description
        assert e_descr.brief_description

        # assert GET of evaluator descriptor
        d_e = evaluate.describe_evaluator(evaluator=e_descr.id, portable=portable)
        assert d_e
        pprint.pprint(d_e)
        if portable:
            for p in d_e["parameters"]:
                assert p["min_"] not in non_portable_values
                assert p["max_"] not in non_portable_values
            for m_m in d_e["metrics_meta"]:
                assert m_m["threshold"] not in non_portable_values
                if (
                    isinstance(m_m["value_range"], tuple)
                    and len(m_m["value_range"]) == 2
                ):
                    assert m_m["value_range"][0] not in non_portable_values, f"{d_e}"
                    assert m_m["value_range"][1] not in non_portable_values, f"{d_e}"
        assert (
            d_e.get(explainers.ExplainerDescriptor.KEY_BRIEF_DESCRIPTION)
            == e_descr.brief_description
        )


def _given_all_hosts_completion(
    tmp_path,
    rag_or_llm: commons.ModelTypeExplanation,
    test_suite_path: str,
    connection_type: config.ConnectionConfigType,
) -> tuple[
    list,
    progress.ProgressCallbackContext,
    progress.ProgressCallbackContext,
    testing.RagTestLab,
    bool,
    pathlib.Path,
]:
    #
    # GIVEN
    #
    skip_context_dependent_evaluators = (
        connection_type == config.ConnectionConfigType.OPENAI_RAG
    )
    evaluators = [
        # agentic: no agentic data, no check, just sanity run
        asc_e.AgentSanityCheckEvaluator().evaluator_id(),
        # RAG
        hal_e.RagHallucinationEvaluator().evaluator_id(),
        ld_e.LoopingDetectionEvaluator().evaluator_id(),
        pii_e.PiiLeakageEvaluator().evaluator_id(),
        ppx_e.PerplexityEvaluator().evaluator_id(),
        rc_m_e.MeanReciprocalRankEvaluator().evaluator_id(),
        rg_e.RagGroundednessEvaluator().evaluator_id(),
        sdl_e.SensitiveDataLeakageEvaluator().evaluator_id(),
        tp_e.RagStrStrEvaluator().evaluator_id(),
        # red teaming - no attack, no action
        en_g_e.EncodingGuardrailEvaluator().evaluator_id(),
        # classification problem evaluation: no value in this suite, just sanity run
        c_e.ClassificationEvaluator().evaluator_id(),
        # summarization: ROUGE, BLEU, ... will detect failure - sanity run
        a_aaea.AnswerAccuracyEvaluator().evaluator_id(),
        r_e.RougeEvaluator().evaluator_id(),
        b_e.BleuEvaluator().evaluator_id(),
        bs_e.BertscoreEvaluator().evaluator_id(),
        # s_e.SummarizationEvaluator().evaluator_id(),  # slow & inefficient
        # HF models
        fb_e.FairnessBiasEvaluator().evaluator_id(),
        t_e.ToxicityEvaluator().evaluator_id(),
    ]
    if test_utils.health.is_openai():
        # evaluators which require OpenAI key
        evaluators += [
            # RAGAS
            rag_ragas_evaluator.RagasEvaluator().evaluator_id(),
            ac_e.AnswerCorrectnessEvaluator().evaluator_id(),
            as_e.AnswerSemanticSimilarityEvaluator().evaluator_id(),
            cr_e.ContextRelevancyEvaluator().evaluator_id(),
            ar_e.AnswerRelevancyEvaluator().evaluator_id(),
            cp_e.ContextPrecisionEvaluator().evaluator_id(),
            crc_e.ContextRecallEvaluator().evaluator_id(),
            f_e.FaithfulnessEvaluator().evaluator_id(),
            # BYOP
            ci_p_e.ContactInformationByopEvaluator().evaluator_id(),
            lm_p_e.LanguageMismatchByopEvaluator().evaluator_id(),
            # p_p_e.ParameterizableByopEvaluator().evaluator_id(),  # needs prompt
            s_p_e.SexismByopEvaluator().evaluator_id(),
            st_p_e.StereotypeByopEvaluator().evaluator_id(),
            su_p_e.SummarizationByopEvaluator().evaluator_id(),
        ]
        if skip_context_dependent_evaluators:
            # No context => RagHallucination fails
            evaluators.remove(hal_e.RagHallucinationEvaluator.evaluator_id())

    if config.ConnectionConfigType.H2O_GPT == connection_type:
        target_host_connection = test_utils.health.get_h2ogpt()
        llm_model_type = models.ExplainableModelType.h2ogpt
        llm_model_names = test_utils.health.get_h2ogpt_models(
            genai.H2oGptLlmClient(target_host_connection).list_llm_model_names()
        )
    elif config.ConnectionConfigType.H2O_GPT_E == connection_type:
        # h2oGPTe server (must be accessible from the CI)
        target_host_connection = test_utils.health.get_h2ogpte()
        # LLM models to evaluate
        llm_model_type = (
            models.ExplainableModelType.h2ogpte
            if rag_or_llm == commons.ModelTypeExplanation.RAG
            else models.ExplainableModelType.h2ogpte_llm
        )
        llm_model_names = genai.H2oGpteRagClient(
            target_host_connection
        ).list_llm_model_names()
        # probe: llm_model_names = test_utils.health.get_h2ogpte_models(llm_model_names)
        # filter out hanging models
        llm_model_names = [m for m in llm_model_names if "Qwen" not in m]
    elif config.ConnectionConfigType.AZURE_OPENAI_CHAT == connection_type:
        # MS Azure hosted Open AI chat server
        target_host_connection = given_generative.AZURE_OPENAI_LLM
        llm_model_type = models.ExplainableModelType.azure_openai_llm
        # MS Azure uses deployment name instead of the model (hardcoded in the client)
        llm_model_names = genai.MsAzureOpenAiLlmClient(
            connection=target_host_connection,
        ).list_llm_model_names()
    elif config.ConnectionConfigType.H2O_LLM_OPS == connection_type:
        target_host_connection = given_generative.H2O_LLMOPS
        llm_model_type = models.ExplainableModelType.h2ollmops
        llm_model_names = genai.H2oLlmOpsClient(
            connection=target_host_connection,
        ).list_llm_model_names()
    elif config.ConnectionConfigType.OPENAI_CHAT == connection_type:
        # Open AI chat server
        target_host_connection = given_generative.OPENAI_LLM
        llm_model_type = models.ExplainableModelType.openai_llm
        llm_model_names = genai.OpenAiLlmClient(
            connection=target_host_connection,
        ).list_llm_model_names()
        # filter out visual models like dall-e models
        llm_model_names = [m for m in llm_model_names if "gpt-" in m.lower()]
    elif config.ConnectionConfigType.ANTHROPIC_CHAT == connection_type:
        # Anthropic Claude chat server
        target_host_connection = given_generative.ANTHROPIC_LLM
        llm_model_type = models.ExplainableModelType.anthropic_llm
        llm_model_names = genai.AnthropicClaudeLlmClient(
            connection=target_host_connection,
        ).list_llm_model_names()
    elif config.ConnectionConfigType.OPENAI_RAG == connection_type:
        # Open AI RAG server
        target_host_connection = given_generative.OPENAI_RAG
        llm_model_type = models.ExplainableModelType.openai_rag
        llm_model_names = genai.OpenAiAssistantsRagClient(
            target_host_connection
        ).list_llm_model_names()
    elif config.ConnectionConfigType.AMAZON_BEDROCK == connection_type:
        target_host_connection = given_generative.AMAZON_BEDROCK
        llm_model_type = models.ExplainableModelType.amazon_bedrock_rag
        llm_model_names = ["anthropic.claude-3-haiku-20240307-v1:0"]
    else:
        raise ValueError(f"Unknown connection type: {connection_type}")

    # test suite
    test_suite = testing.RagTestSuiteConfig.load_from_json(
        # small(er) test suite w/ many features
        test_utils.find_locally(test_suite_path)
    )

    # DESCOPE to be fast & cheap @ CI
    llm_model_names = llm_model_names[:3]
    # test_suite.test_cases = test_suite.test_cases[:3]

    print(f"Tested LLM models: {llm_model_names}")

    # test lab
    if rag_or_llm == commons.ModelTypeExplanation.RAG:
        test_lab = testing.RagTestLab.from_rag_test_suite(
            rag_connection=target_host_connection,
            rag_test_suite=test_suite,
            llm_model_names=llm_model_names,
            rag_model_type=llm_model_type,
            docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
            predefined_collection_id=(
                genai.AmazonBedrockRagClient(
                    given_generative.AMAZON_BEDROCK, None
                )._resolve_collection_id("knowledge-base-es-test")
                if config.ConnectionConfigType.AMAZON_BEDROCK == connection_type
                else None
            ),
        )
    else:
        test_lab = testing.RagTestLab.from_llm_test_suite(
            llm_host_connection=target_host_connection,
            llm_test_suite=test_suite,
            llm_model_names=llm_model_names,
            llm_model_type=llm_model_type,
            work_dir=tmp_path,
        )
    try:
        # progress: 3 stages - build, complete, evaluate
        progress_callback = progress.LoggingProgressCallbackContext(
            logger=test_lab.logger,
            prefix="[TEST E2E progress callback]",
            name="Test E2E progress callback",
        )
        lab_build_progress = progress_callback.get_sub_callback_for_progress(
            min_progress=0.0, max_progress=0.33, verbose_children=False
        )
        lab_completion_progress = progress_callback.get_sub_callback_for_progress(
            min_progress=0.34, max_progress=0.66, verbose_children=False
        )
        eval_progress = progress_callback.get_sub_callback_for_progress(
            min_progress=0.67, max_progress=1.0, verbose_children=False
        )

        # test lab:
        #     DEPLOY the h2oGPTe server (docs sync: S3 > filesystem cache > h2oGPT2)
        test_lab.build(progress_callback=lab_build_progress)

        # test lab:
        #     complete dataset w/ ACTUAL values from the h2oGPTe server (answers, ...)
        test_lab.complete_dataset(
            complete_context=3,
            progress_callback=lab_completion_progress,
            save_as_you_go=tmp_path / "wip_testlab.json",
            parallelize=testing.TestLab.PARALLEL_RUN,
            retry_on_error=3,
        )
        # backup fully resolved dataset
        test_lab_path = tmp_path / "test_lab.json"
        test_lab.save_as_json(test_lab_path)

        return (
            evaluators,
            progress_callback,
            eval_progress,
            test_lab,
            skip_context_dependent_evaluators,
            test_lab_path,
        )
    finally:
        test_lab.purge()


@pytest.mark.skip("Used by ::test_all_evaluators - just for genaiclient extra testing")
@pytest.mark.parametrize(
    "rag_or_llm,test_suite_path,connection_type",
    [
        # h2oGPT retired
        # (
        #    commons.ModelTypeExplanation.LLM,
        #     "data/generative/ci_llm_test_suite.json",
        #     config.ConnectionConfigType.H2O_GPT,
        # ),
        (
            commons.ModelTypeExplanation.RAG,
            "data/generative/ci_rag_test_suite.json",
            config.ConnectionConfigType.H2O_GPT_E,
        ),
        (
            commons.ModelTypeExplanation.LLM,
            "data/generative/ci_llm_test_suite.json",
            config.ConnectionConfigType.H2O_GPT_E,
        ),
        pytest.param(
            commons.ModelTypeExplanation.LLM,
            "data/generative/ci_llm_test_suite.json",
            config.ConnectionConfigType.OPENAI_CHAT,
            marks=pytest.mark.skipif(
                not test_utils.health.is_openai(),
                reason="Valid OpenAI key is not available",
            ),
        ),
        pytest.param(
            commons.ModelTypeExplanation.RAG,
            "data/generative/ci_rag_test_suite.json",
            config.ConnectionConfigType.OPENAI_RAG,
            marks=pytest.mark.skipif(
                not test_utils.health.is_openai(),
                reason="Valid OpenAI key is not available",
            ),
        ),
        pytest.param(
            commons.ModelTypeExplanation.LLM,
            "data/generative/ci_llm_test_suite.json",
            config.ConnectionConfigType.ANTHROPIC_CHAT,
            marks=pytest.mark.skipif(
                not test_utils.health.is_anthropic(),
                reason="Valid Anthropic key is not available",
            ),
        ),
        (
            commons.ModelTypeExplanation.LLM,
            "data/generative/ci_llm_test_suite.json",
            config.ConnectionConfigType.AZURE_OPENAI_CHAT,
        ),
        pytest.param(
            commons.ModelTypeExplanation.RAG,
            "data/generative/ci_rag_test_suite_bedrock.json",
            config.ConnectionConfigType.AMAZON_BEDROCK,
            marks=pytest.mark.skipif(
                not test_utils.health.is_bedrock(),
                reason="Valid AWS Bedrock key is not available",
            ),
        ),
        pytest.param(
            commons.ModelTypeExplanation.LLM,
            "data/generative/ci_llm_test_suite.json",
            config.ConnectionConfigType.H2O_LLM_OPS,
            marks=pytest.mark.skipif(
                not test_utils.health.is_h2ollmops(),
                reason="H2O LLMOps models are deployed temporarily",
            ),
        ),
    ],
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_all_hosts_completion(
    tmp_path,
    rag_or_llm: commons.ModelTypeExplanation,
    test_suite_path: str,
    connection_type: config.ConnectionConfigType,
):
    #
    # GIVEN & WHEN
    #
    (_, _, _, _, _, test_lab_path) = _given_all_hosts_completion(
        tmp_path=tmp_path,
        rag_or_llm=rag_or_llm,
        test_suite_path=test_suite_path,
        connection_type=connection_type,
    )

    #
    # THEN
    #
    assert test_lab_path.exists()


@pytest.mark.parametrize(
    "rag_or_llm,test_suite_path,connection_type",
    [
        # h2oGPT is retired
        # (
        #    commons.ModelTypeExplanation.LLM,
        #     "data/generative/ci_llm_test_suite.json",
        #     config.ConnectionConfigType.H2O_GPT,
        # ),
        (
            commons.ModelTypeExplanation.RAG,
            "data/generative/ci_rag_test_suite.json",
            config.ConnectionConfigType.H2O_GPT_E,
        ),
        (
            commons.ModelTypeExplanation.LLM,
            "data/generative/ci_llm_test_suite.json",
            config.ConnectionConfigType.H2O_GPT_E,
        ),
        pytest.param(
            commons.ModelTypeExplanation.LLM,
            "data/generative/ci_llm_test_suite.json",
            config.ConnectionConfigType.OPENAI_CHAT,
            marks=pytest.mark.skipif(
                not test_utils.health.is_openai(),
                reason="Valid OpenAI key is not available",
            ),
        ),
        pytest.param(
            commons.ModelTypeExplanation.RAG,
            "data/generative/ci_rag_test_suite.json",
            config.ConnectionConfigType.OPENAI_RAG,
            marks=pytest.mark.skipif(
                not test_utils.health.is_openai(),
                reason="Valid OpenAI key is not available",
            ),
        ),
        pytest.param(
            commons.ModelTypeExplanation.LLM,
            "data/generative/ci_llm_test_suite.json",
            config.ConnectionConfigType.ANTHROPIC_CHAT,
            marks=pytest.mark.skipif(
                not test_utils.health.is_anthropic(),
                reason="Valid Anthropic key is not available",
            ),
        ),
        (
            commons.ModelTypeExplanation.LLM,
            "data/generative/ci_llm_test_suite.json",
            config.ConnectionConfigType.AZURE_OPENAI_CHAT,
        ),
        pytest.param(
            commons.ModelTypeExplanation.RAG,
            "data/generative/ci_rag_test_suite_bedrock.json",
            config.ConnectionConfigType.AMAZON_BEDROCK,
            marks=pytest.mark.skipif(
                not test_utils.health.is_bedrock(),
                reason="Valid AWS Bedrock key is not available",
            ),
        ),
        pytest.param(
            commons.ModelTypeExplanation.LLM,
            "data/generative/ci_llm_test_suite.json",
            config.ConnectionConfigType.H2O_LLM_OPS,
            marks=pytest.mark.skipif(
                not test_utils.health.is_h2ollmops(),
                reason="H2O LLMOps models are deployed temporarily",
            ),
        ),
    ],
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_all_evaluators(
    tmp_path,
    rag_or_llm: commons.ModelTypeExplanation,
    test_suite_path: str,
    connection_type: config.ConnectionConfigType,
):
    #
    # GIVEN
    #
    (
        evaluators,
        progress_callback,
        eval_progress,
        test_lab,
        skip_context_dependent_evaluators,
        _,
    ) = _given_all_hosts_completion(
        tmp_path=tmp_path,
        rag_or_llm=rag_or_llm,
        test_suite_path=test_suite_path,
        connection_type=connection_type,
    )

    #
    # WHEN
    #
    try:
        evaluation = evaluate.run_evaluation(
            # dataset w/ prompts, constraints and model keys
            dataset=test_lab.dataset,
            # models to be evaluated / compared to get leaderboard
            models=list(test_lab.evaluated_models.values()),
            # evaluators
            evaluators=evaluators,
            # where to save the report
            results_location=tmp_path,
            # progress
            progress_callback=eval_progress,
            # log
            log_level=logging.INFO,
        )

        #
        # THEN
        #

        print(f"{evaluation}")

        # it is EXPECTED that summarization evaluator fails for short inputs)
        if evaluation.is_explainer_failed():
            assert len(evaluation.get_failed_evaluator_ids()) == 1
            assert "Summarization" in evaluation.get_failed_evaluator_ids()[0]

        # assert result
        if (
            rag_or_llm == commons.ModelTypeExplanation.RAG
            and not skip_context_dependent_evaluators
        ):
            evaluator_id = hal_e.RagHallucinationEvaluator.evaluator_id()
            result = evaluation.get_evaluator_result(evaluator_id)
            print(result)
            assert result
            # get explanation file path
            if evaluator_id in evaluators:
                t_llmheatmap = e10s.LlmHeatmapLeaderboardExplanation
                path = evaluation.get_explanation_file_path(
                    evaluator_id=evaluator_id,
                    explanation_type=t_llmheatmap.explanation_type(),
                    explanation_format=f5s.LlmHeatmapLeaderboardJSonFormat.mime,
                )
                print(f"Explanation file path: {path}")
                assert pathlib.Path(path).exists()

                # result: leaderboard
                result = evaluation.get_explainer_result(evaluator_id)
                # result: data
                data = result.data()
                print(f"Data:\n{data}")
                assert data
                # result: summary
                summary = result.summary()
                print(f"Summary:\n{summary}")
                assert summary
                # result: plot / log / zip
                result.plot(file_path=tmp_path / "my_plot.png")
                result.log(path=tmp_path / "my_log.txt")
                result.zip(file_path=tmp_path / "my_result.zip")
        else:
            evaluator_id = tp_e.RagStrStrEvaluator.evaluator_id()
            result = evaluation.get_evaluator_result(evaluator_id)
            print(result)
            assert result

        assert evaluation.progress == 1.0
        assert evaluation.progress_callback.progress == 1.0
        assert progress_callback.progress == 1.0

        # assert leaderboard JSon representation data and meta
        if evaluation.is_evaluator_failed():
            # remove Summarization evaluator from the list
            evaluators.remove(s_e.SummarizationEvaluator().evaluator_id())
        for evaluator_id in evaluators:
            then_eval.then_leaderboard_json(evaluation, evaluator_id)

        print(
            f"Explanations:\n"
            f"  HTML: file://{evaluation.result.get_html_report_location()}\n"
        )
    finally:
        test_lab.purge()


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_negative_evaluate_lab_of_failures(tmp_path):
    #
    # GIVEN
    #

    rag_dataset = testing.RagTestLab.load_from_json(
        llm_host_connection=test_utils.health.get_h2ogpt(),
        file_path="data/generative/ci_rag_test_lab_llm_failures.json",
    )
    llm_models = rag_dataset.evaluated_models.values()

    #
    # WHEN
    #

    evaluation = evaluate.run_evaluation(
        dataset=rag_dataset.dataset,
        models=llm_models,
        evaluators=[tp_e.RagStrStrEvaluator.evaluator_id()],
        results_location=tmp_path,
        log_level=loggers.DEBUG,
    )

    # THEN
    print(f"Evaluation:\n{evaluation}")
    print(f"Evaluation error message:\n{evaluation.error}")

    assert evaluation, f"Evaluation must not be null: {evaluation}"
    assert evaluation.result.problems, (
        f"Problems must not be empty: {evaluation.result.problems}"
    )
    assert evaluation.error, f"Error message must not be empty: {evaluation.error}"
    assert evaluation.status == commons.ExplainerJobStatus.FAILED, (
        f"Evaluation status must be FAILED: {evaluation.status}"
    )
    print(f"HTML report:\nfile://{evaluation.result.get_html_report_location()}")


@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_list_perturbators():
    #
    # GIVEN, WHEN
    #
    perturbators = evaluate.list_perturbators()

    #
    # THEN
    #
    print(perturbators)
    for p in perturbators:
        print(p.perturbator_id)
        assert p.perturbator_id
    assert len(perturbators) > 0


@pytest.mark.parametrize(
    "perturbators_count",
    [
        1,
        5,
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_perturb_string(perturbators_count):
    #
    # GIVEN
    #
    input_content = (
        "This is the short, witty, easy to understand text to be randomly perturbed "
        "using a suite of precise, reliable and ugly perturbators so of which need to "
        "have a lot of ys and zs in it so that qwerty perturbator can do itz job: "
        "yyYzZzz."
    )
    all_perturbator_ids = [
        evaluate.list_perturbators()[i + 1].perturbator_id
        for i in range(perturbators_count)
    ]
    # skip CopyPerturbator as it does NOT perturb the text
    perturbator_ids_to_run = [
        perturbator_id
        for perturbator_id in all_perturbator_ids
        if perturbator_id != perturbations.CopyPerturbator().perturbator_id()
    ]
    perturbators_2_run = [
        commons.PerturbatorToRun(
            perturbator_id=perturbator_id,
            intensity=commons.PerturbationIntensity.LOW.name,
        )
        for perturbator_id in perturbator_ids_to_run
    ]

    #
    # WHEN
    #
    perturbed_text = evaluate.perturb(
        content=input_content, perturbators=perturbators_2_run
    )

    #
    # THEN
    #
    print(
        f"Content perturbed by {perturbators_count} perturbators: "
        f"{perturbator_ids_to_run}"
    )
    print(f"  ORIGINAL : {input_content}")
    print(f"  PERTURBED: {perturbed_text}")
    assert input_content != perturbed_text


@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_perturb_test_case():
    #
    # GIVEN
    #
    content = "This is the text to be perturbed using a perturbator."
    test_case = testing.RagTestCaseConfig(prompt=content)
    perturbator_id = evaluate.list_perturbators()[1].perturbator_id

    #
    # WHEN
    #
    raised_errors = []
    perturbed_test_case = evaluate.perturb(
        content=test_case,
        perturbators=[
            commons.PerturbatorToRun(
                perturbator_id=perturbator_id,
                intensity=commons.PerturbationIntensity.LOW.name,
            )
        ],
        raised_errors=raised_errors,
    )

    #
    # THEN
    #
    print(f"Original and perturbed content ({perturbator_id}):")
    print(content)
    print(perturbed_test_case.prompt)
    assert content != perturbed_test_case.prompt

    print(json.dumps(perturbed_test_case.to_dict(), indent=2))
    assert perturbations.CAT_PERTURBED in test_case.categories

    print(f"Raised errors: {raised_errors}")
    assert not raised_errors


@pytest.mark.parametrize(
    "in_place,test_suite_path,raised_errors",
    [
        # (True, "data/generative/ci_llm_test_suite.json", None),
        # (False, "data/generative/ci_llm_test_suite.json", None),
        # negative test for ES (in_place=False)
        (False, "data/generative/ci_rag_test_suite.json", []),
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_perturb_test_suite(
    tmp_path, in_place: bool, test_suite_path: str, raised_errors: list | None
):
    #
    # GIVEN
    #
    test_suite = testing.RagTestSuiteConfig.load_from_json(
        test_utils.find_locally(test_suite_path)
    )
    test_suite.save_as_json(tmp_path / "original_test_suite.json")
    original_cardinality = len(test_suite.test_cases)
    if raised_errors is not None:
        # the goal is to let perturbations fail, do NOT throw exception, but collect
        # errors to the list
        gathered_errors = []
        intensity = commons.PerturbationIntensity.LOW.name
        perturbators = [
            commons.PerturbatorToRun(
                perturbator_id=perturbations.CommaPerturbator.perturbator_id(),
                intensity=intensity,
            ),
            commons.PerturbatorToRun(
                perturbator_id=perturbations.WordSwapPerturbator.perturbator_id(),
                intensity=intensity,
            ),
            commons.PerturbatorToRun(
                perturbator_id=perturbations.AntonymPerturbator.perturbator_id(),
                intensity=intensity,
            ),
            commons.PerturbatorToRun(
                perturbator_id=perturbations.SynonymPerturbator.perturbator_id(),
                intensity=intensity,
            ),
            commons.PerturbatorToRun(
                perturbator_id=perturbations.QwertyPerturbator.perturbator_id(),
                intensity=intensity,
            ),
            commons.PerturbatorToRun(
                perturbator_id=(
                    perturbations.RandomCharacterInsertPerturbator.perturbator_id()
                ),
                intensity=intensity,
            ),
            commons.PerturbatorToRun(
                perturbator_id=(
                    perturbations.RandomCharacterDeletePerturbator.perturbator_id()
                ),
                intensity=intensity,
            ),
            commons.PerturbatorToRun(
                perturbator_id=(
                    perturbations.RandomCharacterReplacementPerturbator.perturbator_id()
                ),
                intensity=intensity,
            ),
            commons.PerturbatorToRun(
                perturbator_id=(
                    perturbations.KeywordTyposCharacterPerturbator.perturbator_id()
                ),
                intensity=intensity,
            ),
            commons.PerturbatorToRun(
                perturbator_id=(
                    perturbations.RandomOCRCharacterPerturbator.perturbator_id()
                ),
                intensity=intensity,
            ),
        ]
    else:
        gathered_errors = None
        perturbators = [
            commons.PerturbatorToRun(
                perturbator_id=perturbations.CommaPerturbator.perturbator_id(),
                # high intensity would make pert. impossible for other perturbators
                intensity=commons.PerturbationIntensity.MEDIUM.name,
            ),
            commons.PerturbatorToRun(
                perturbator_id=perturbations.WordSwapPerturbator.perturbator_id(),
                intensity=commons.PerturbationIntensity.HIGH.name,
            ),
            commons.PerturbatorToRun(
                perturbator_id=(
                    perturbations.RandomCharacterInsertPerturbator.perturbator_id()
                ),
                intensity=commons.PerturbationIntensity.LOW.name,
            ),
        ]
    perturbator_ids = [p.perturbator_id for p in perturbators]

    #
    # WHEN
    #
    perturbed_suite = evaluate.perturb(
        content=test_suite,
        perturbators=perturbators,
        in_place=in_place,
        raised_errors=gathered_errors,
    )
    perturbed_suite.save_as_json(tmp_path / "perturbed_test_suite.json")

    #
    # THEN
    #

    print(
        f"Original and perturbed content ({perturbator_ids}):"
        f"\n  Raised errors ({len(gathered_errors)}): "
        f"{raised_errors} -> {gathered_errors}"
    )
    if raised_errors is not None:
        assert gathered_errors
        return

    if in_place:
        for test_case in perturbed_suite.test_cases:
            print(test_case.prompt)
        assert original_cardinality == len(perturbed_suite.test_cases)
    else:
        perturbed = 0
        for i in range(original_cardinality):
            test_case = test_suite.test_cases[i]
            perturbed_test_case = perturbed_suite.test_cases[original_cardinality + i]
            print("===================================================================")
            print(test_case.prompt)
            print("-------------------------------------------------------------------")
            print(perturbed_test_case.prompt)
            if test_case.prompt != perturbed_test_case.prompt:
                perturbed += 1
                assert perturbations.CAT_PERTURBED in perturbed_test_case.categories
                for p in perturbators:
                    match = ""
                    for c in perturbed_test_case.categories:
                        if p.perturbator_id in c:
                            match = c
                            break
                    assert match
                assert len(perturbed_test_case.relationships)
        assert original_cardinality * 2 == len(perturbed_suite.test_cases)
        assert perturbed == original_cardinality

        print(f"\nOriginal cardinality/perturbed: {perturbed}/{original_cardinality}")


@pytest.mark.parametrize(
    "in_place,test_suite_path",
    [
        (False, "data/generative/ci_llm_test_suite.json"),
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_perturb_test_suite_twice(tmp_path, in_place: bool, test_suite_path: str):
    #
    # GIVEN
    #
    test_suite = testing.RagTestSuiteConfig.load_from_json(
        test_utils.find_locally(test_suite_path)
    )
    test_suite.save_as_json(tmp_path / "original_test_suite.json")
    original_cardinality = len(test_suite.test_cases)
    perturbators = [
        commons.PerturbatorToRun(
            perturbator_id=perturbations.CommaPerturbator.perturbator_id(),
            # high intensity would make perturbations impossible for other perturbators
            intensity=commons.PerturbationIntensity.MEDIUM.name,
        ),
        commons.PerturbatorToRun(
            perturbator_id=(
                perturbations.RandomCharacterDeletePerturbator.perturbator_id()
            ),
            intensity=commons.PerturbationIntensity.LOW.name,
        ),
    ]
    perturbator_ids = [p.perturbator_id for p in perturbators]

    #
    # WHEN
    #
    perturbed_suite = evaluate.perturb(
        content=test_suite,
        perturbators=perturbators,
        in_place=in_place,
    )
    perturbed_suite.save_as_json(tmp_path / "perturbed_test_suite.json")

    perturbed_twice_suite = evaluate.perturb(
        content=perturbed_suite,
        perturbators=perturbators,
        in_place=in_place,
    )
    perturbed_twice_suite.save_as_json(tmp_path / "perturbed_twice_test_suite.json")
    #
    # THEN
    #

    print(f"Original and perturbed content ({perturbator_ids}):")
    if in_place:
        for test_case in perturbed_suite.test_cases:
            print(test_case.prompt)
        assert original_cardinality == len(perturbed_suite.test_cases)
    else:
        perturbed = 0
        for i in range(original_cardinality):
            test_case = test_suite.test_cases[i]
            perturbed_test_case = perturbed_suite.test_cases[original_cardinality + i]
            print("===================================================================")
            print(test_case.prompt)
            print("-------------------------------------------------------------------")
            print(perturbed_test_case.prompt)
            if test_case.prompt != perturbed_test_case.prompt:
                perturbed += 1
                assert perturbations.CAT_PERTURBED in perturbed_test_case.categories
                for p in perturbators:
                    match = ""
                    for c in perturbed_test_case.categories:
                        if p.perturbator_id in c:
                            match = c
                            break
                    assert match
                assert len(perturbed_test_case.relationships)
        assert original_cardinality * 2 == len(perturbed_suite.test_cases)
        assert perturbed == original_cardinality

        print(f"\nOriginal cardinality/perturbed: {original_cardinality}/{perturbed}")

    print(f"Perturbed and perturbed_twice content ({perturbator_ids}):")
    if in_place:
        for test_case in perturbed_twice_suite.test_cases:
            print(test_case.prompt)
        assert len(perturbed_suite.test_cases) == len(perturbed_twice_suite.test_cases)
    else:
        perturbed_twice = 0
        for i in range(len(perturbed_suite.test_cases)):
            test_case = perturbed_suite.test_cases[i]
            perturbed_test_case = perturbed_twice_suite.test_cases[
                original_cardinality + i
            ]
            print("===================================================================")
            print(test_case.prompt)
            print("-------------------------------------------------------------------")
            print(perturbed_test_case.prompt)
            if test_case.prompt != perturbed_test_case.prompt:
                perturbed_twice += 1
                assert perturbations.CAT_PERTURBED in perturbed_test_case.categories
                for p in perturbators:
                    match = ""
                    for c in perturbed_test_case.categories:
                        if p.perturbator_id in c:
                            match = c
                            break
                    assert match
                assert len(perturbed_test_case.relationships)
        assert len(perturbed_suite.test_cases) * 2 == len(
            perturbed_twice_suite.test_cases
        )
        assert perturbed_twice == len(perturbed_suite.test_cases)

        print(
            f"\nPerturbed/Perturbed twice: {len(perturbed_suite.test_cases)}/"
            f"{perturbed_twice}"
        )


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.parametrize(
    "in_place,test_lab_path",
    [
        (True, "data/generative/kaggle_llm_science_exam_test_lab_2x_small_3.json"),
        (False, "data/generative/kaggle_llm_science_exam_test_lab_2x_small_3.json"),
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_perturb_llm_dataset(in_place: bool, test_lab_path: str):
    #
    # GIVEN
    #
    test_lab = testing.RagTestLab.load_from_json(
        llm_host_connection=test_utils.health.get_h2ogpte(),
        file_path=test_utils.find_locally(test_lab_path),
    )
    original_cardinality = test_lab.dataset.shape()[0]
    perturbator_id = evaluate.list_perturbators()[4].perturbator_id
    original_dataset = test_lab.dataset

    #
    # WHEN
    #
    perturbed_dataset = evaluate.perturb(
        content=original_dataset,
        perturbators=[
            commons.PerturbatorToRun(
                perturbator_id=perturbator_id,
                intensity=commons.PerturbationIntensity.MEDIUM.name,
            )
        ],
        in_place=in_place,
    )

    #
    # THEN
    #
    print(f"Original and perturbed content ({perturbator_id}):")
    if in_place:
        for i in perturbed_dataset.inputs:
            print(i.i)
        assert original_cardinality == original_dataset.shape()[0]
    else:
        perturbed = 0
        for i in range(original_cardinality):
            row = original_dataset.inputs[i]
            perturbed_test_case = perturbed_dataset.inputs[original_cardinality + i]
            print("===================================================================")
            print(row.i)
            print("-------------------------------------------------------------------")
            print(perturbed_test_case.i)
            if row.i != perturbed_test_case.i:
                perturbed += 1
        assert original_cardinality * 2 == perturbed_dataset.shape()[0]
        assert perturbed == original_cardinality

        print(f"\nOriginal cardinality/perturbed: {perturbed}/{original_cardinality}")


@pytest.mark.skip(reason="Not actual test, just a helper to cache models")
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_cache_models():
    #
    # WHEN
    #
    evaluate.cache_models()

    #
    # THEN
    #
    assert True


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
