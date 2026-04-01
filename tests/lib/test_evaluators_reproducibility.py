# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import pathlib
import traceback
import uuid

import pytest

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import evaluate
from h2o_sonar import interpret
from h2o_sonar import loggers
from h2o_sonar.evaluators import answer_accuracy_evaluator
from h2o_sonar.evaluators import (
    answer_semantic_similarity_per_sentence_evaluator as assprs,
)
from h2o_sonar.evaluators import perplexity_evaluator
from h2o_sonar.evaluators import rag_answer_similarity_evaluator
from h2o_sonar.evaluators import rouge_evaluator
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import evaluators
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import persistences
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative


def _given_deterministic_evaluator_classes() -> list:
    container = interpret.resolve_container()
    evaluator_classes_by_id = container.explainers_registry.list_explainers()
    deterministic_evaluators_ids = [
        e.id
        for e in evaluate.list_evaluators(
            keywords=[evaluators.KEYWORD_METHOD_TYPE_DETERMINISTIC]
        )
    ]
    return [
        evaluator_classes_by_id[eid]
        for eid in deterministic_evaluators_ids
        if eid in evaluator_classes_by_id
    ]


def _given_test_labs_with_aa() -> list[str]:
    return [
        "./data/generative/conferences/atlanta-2024/bank_teller_test_lab.json",
        "./data/generative/conferences/atlanta-2024/pii_test_lab.json",
        "./data/generative/conferences/atlanta-2024/ragas_no_or_h2ogpte_test_lab.json",
        "./data/generative/conferences/atlanta-2024/sensitive_data_test_lab.json",
        "./data/generative/conferences/atlanta-2024/sr1107_test_lab_large.json",
        "./data/generative/conferences/atlanta-2024/sr1107_test_lab_small.json",
        "./data/generative/bugs/82a80332-b363-4cf2-8735-44fb69cbabb2_test_lab.json",
        "./data/generative/ci_llm_test_lab_h2ogpt_args.json",
        "./data/generative/ci_llm_test_lab_h2ogpte_args.json",
        "./data/generative/ci_llm_test_lab_ollama_args.json",
        "./data/generative/ci_rag_test_lab_bedrock.json",
        "./data/generative/ci_rag_test_lab_h2ogpte_args.json",
        "./data/generative/ci_rag_test_lab_openai_args.json",
        "./data/generative/class_multi_test_lab.json",
        "./data/generative/dummy_summarization_test_lab_small.json",
        "./data/generative/dummy_translation_test_lab_small.json",
        "./data/generative/eval_agent/llm_bank_teller_1p/test_lab.json",
        "./data/generative/eval_agent/rag_fbi_agent_1p/test_lab.json",
        "./data/generative/eval_llm/arabic_mmlu_test_lab_10p.json",
        "./data/generative/eval_llm/bank_teller_h2ogpt_test_lab.json",
        "./data/generative/eval_llm/bank_teller_test_lab.json",
        "./data/generative/eval_llm/encoded_bug_1415_test_lab.json",
        "./data/generative/eval_llm/encoded_pii_perturbed_test_lab.json",
        "./data/generative/eval_llm/encoded_pii_perturbed_test_lab_1p.json",
        "./data/generative/eval_llm/h2ogpte_benchmark_test_lab_micro.json",
        "./data/generative/eval_llm/json_schema_test_lab.json",
        "./data/generative/eval_llm/minimal_llm_test_lab_1p.json",
        "./data/generative/eval_llm/perturbed_test_lab_2p.json",
        "./data/generative/eval_llm/pii_test_lab.json",
        "./data/generative/eval_llm/red_teaming_test_lab_broken_some.json",
        "./data/generative/eval_llm/sensitive_data_test_lab.json",
        "./data/generative/fairness_bias_test_lab_1p.json",
        "./data/generative/h2ogpte_benchmark_test_lab_small.json",
        "./data/generative/h2ogpte_benchmark_test_lab_top.json",
        "./data/generative/h2ogpte_benchmark_test_lab_top_openai_20231212.json",
        "./data/generative/kaggle_llm_science_exam_class_bin_test_lab.json",
        "./data/generative/kaggle_llm_science_exam_class_multi_test_lab.json",
        "./data/generative/kaggle_llm_science_exam_test_lab_2x_small_200.json",
        "./data/generative/kaggle_llm_science_exam_test_lab_2x_small_25.json",
        "./data/generative/kaggle_llm_science_exam_test_lab_2x_small_3.json",
        (
            "./data/generative/"
            "kaggle_llm_science_exam_test_lab_2x_small_wrong_answers.json"
        ),
        "./data/generative/kaggle_llm_science_exam_test_lab_4x_25.json",
        "./data/generative/kaggle_llm_science_exam_test_lab_cosmos_25x2.json",
        "./data/generative/kaggle_llm_science_exam_test_lab_cosmos_5x2.json",
        "./data/generative/kaggle_llm_science_exam_test_lab_h2o.json",
        "./data/generative/kaggle_llm_science_exam_test_lab_h2o_small.json",
        "./data/generative/kaggle_llm_science_exam_test_lab_leak.json",
        "./data/generative/kims_summarization_negative_test_lab.json",
        "./data/generative/minimal_rag_test_lab_1p.json",
        "./data/generative/ragas-custom-judge_test_lab_1p.json",
        "./data/generative/rouge_long_ea_test_lab_1p.json",
        "./data/generative/rouge_long_aa_test_lab_1p.json",
        "./data/generative/rouge_long_ea_aa_test_lab_1p.json",
        "./data/generative/sr1107_test_lab_15m.json",
        "./data/generative/sr1107_test_lab_171.json",
        "./data/generative/sr1107_test_lab_3m.json",
        "./data/generative/sr1107_test_lab_7p_perturbed.json",
        "./data/generative/summarization_frank_small_test_lab.json",
        "./data/generative/talk2report_prompts_test_lab.json",
        "./data/generative/talk2report_prompts_test_lab_5x5.json",
        "./data/generative/talk2report_prompts_test_lab_small.json",
        "./data/generative/toxicity_test_lab_2x_3.json",
    ]


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_deterministic_evaluators_list():
    #
    # WHEN
    #
    evaluator_classes = _given_deterministic_evaluator_classes()

    #
    # THEN
    #
    print(f"Deterministic evaluators ({len(evaluator_classes)}):")
    assert evaluator_classes
    for e in evaluator_classes:
        print(f"- {e.explainer_id()}: {e.__name__}")
        assert evaluators.KEYWORD_METHOD_TYPE_DETERMINISTIC in e._keywords


@pytest.mark.skipif(
    test_utils.GitHubActions.is_in_gha(),
    reason="Skipped on GHA as this test has high resource usage",
)
@pytest.mark.parametrize(
    "evaluator_classes,test_lab_paths,multiplier,expected_sha256,expected_scores",
    [
        # SMOKE a few deterministic evaluators, a few test labs
        (
            [
                rouge_evaluator.RougeEvaluator,
                perplexity_evaluator.PerplexityEvaluator,
                assprs.AnswerSemanticSimilarityPerSentenceEvaluator,
                answer_accuracy_evaluator.AnswerAccuracyEvaluator,
                rag_answer_similarity_evaluator.AnswerSemanticSimilarityEvaluator,
            ],
            [
                "./data/generative/eval_llm/h2ogpte_benchmark_test_lab_micro.json",
                "./data/generative/conferences/atlanta-2024/sr1107_test_lab_small.json",
                "./data/generative/eval_llm/arabic_mmlu_test_lab_10p.json",
            ],
            1,
            # CPU - reproducible if ALWAYS run on CPU (GPU has different SHA) 72"
            "f03bc83e9e49ce77b0f10bd2727cf145b4cf17a262e6fdbdd482924f9d559dff",
            # GPU - reproducible if ALWAYS run on GPU/CUDA (CPU has different SHA) 33"
            # "9d5227bb820b84ef2949fd5feed37f15f469e820d8a7e6ddd3180b29c7d1d670"
            {
                "./data/generative/eval_llm/h2ogpte_benchmark_test_lab_micro.json": {
                    "h2o_sonar.evaluators.rouge_evaluator.RougeEvaluator": {
                        "h2oai/h2ogpt-4096-llama2-70b-chat": {
                            "rouge_1": 0.09236410011158935,
                            "rouge_2": 0.05428115965523763,
                            "rouge_l": 0.08227323449705086,
                        },
                        "h2oai/h2ogpt-4096-llama2-70b-chat-4bit": {
                            "rouge_1": 0.042897213178407494,
                            "rouge_2": 0.033074870296477286,
                            "rouge_l": 0.042897213178407494,
                        },
                        "h2oai/h2ogpt-4096-llama2-13b-chat": {
                            "rouge_1": 0.057478058503699525,
                            "rouge_2": 0.039207213435888566,
                            "rouge_l": 0.05353328532815712,
                        },
                    },
                    "h2o_sonar.evaluators.perplexity_evaluator.PerplexityEvaluator": {
                        "h2oai/h2ogpt-4096-llama2-70b-chat": {
                            "perplexity": 20.32140874923188
                        },
                        "h2oai/h2ogpt-4096-llama2-70b-chat-4bit": {
                            "perplexity": 25.498576439324683
                        },
                        "h2oai/h2ogpt-4096-llama2-13b-chat": {
                            "perplexity": 27.60516558181412
                        },
                    },
                    (
                        "h2o_sonar.evaluators.answer_semantic_similarity_per_sentence"
                        "_evaluator.AnswerSemanticSimilarityPerSentenceEvaluator"
                    ): {
                        "h2oai/h2ogpt-4096-llama2-70b-chat": {
                            "mean_answer_similarity": 0.8063910196956002,
                            "min_answer_similarity": 0.7559340409741804,
                        },
                        "h2oai/h2ogpt-4096-llama2-70b-chat-4bit": {
                            "mean_answer_similarity": 0.8086532062635344,
                            "min_answer_similarity": 0.705561928086324,
                        },
                        "h2oai/h2ogpt-4096-llama2-13b-chat": {
                            "mean_answer_similarity": 0.8041212220386504,
                            "min_answer_similarity": 0.7479509678711627,
                        },
                    },
                    (
                        "h2o_sonar.evaluators.answer_accuracy_evaluator"
                        ".AnswerAccuracyEvaluator"
                    ): {
                        "h2oai/h2ogpt-4096-llama2-70b-chat": {
                            "answer_accuracy": 0.7561344011571925
                        },
                        "h2oai/h2ogpt-4096-llama2-70b-chat-4bit": {
                            "answer_accuracy": 0.705561928086324
                        },
                        "h2oai/h2ogpt-4096-llama2-13b-chat": {
                            "answer_accuracy": 0.7479509678711627
                        },
                    },
                },
                (
                    "./data/generative/conferences/atlanta-2024"
                    "/sr1107_test_lab_small.json"
                ): {
                    "h2o_sonar.evaluators.rouge_evaluator.RougeEvaluator": {
                        "h2oai/h2ogpt-4096-llama2-70b-chat": {
                            "rouge_1": 0.37061069771350147,
                            "rouge_2": 0.2520334524253897,
                            "rouge_l": 0.30810311184142963,
                        },
                        "h2oai/h2ogpt-4096-llama2-13b-chat": {
                            "rouge_1": 0.44380246913580246,
                            "rouge_2": 0.3157782316739423,
                            "rouge_l": 0.3985679012345679,
                        },
                        "mistral-medium": {
                            "rouge_1": 0.5621456311993397,
                            "rouge_2": 0.41805979016000666,
                            "rouge_l": 0.510155753122505,
                        },
                        "h2oai/h2ogpt-gm-7b-mistral-chat-sft-dpo-v1": {
                            "rouge_1": 0.5946192430460556,
                            "rouge_2": 0.43492063492063493,
                            "rouge_l": 0.5946192430460556,
                        },
                        "h2oai/h2ogpt-gm-experimental": {
                            "rouge_1": 0.611531007751938,
                            "rouge_2": 0.46783625730994144,
                            "rouge_l": 0.5906976744186047,
                        },
                        "h2oai/h2ogpt-32k-codellama-34b-instruct": {
                            "rouge_1": 0.5946192430460556,
                            "rouge_2": 0.4997854997854998,
                            "rouge_l": 0.5946192430460556,
                        },
                        "HuggingFaceH4/zephyr-7b-beta": {
                            "rouge_1": 0.32357435831606485,
                            "rouge_2": 0.219726736195327,
                            "rouge_l": 0.2747412786610674,
                        },
                        "01-ai/Yi-34B-Chat": {
                            "rouge_1": 0.32992002314575625,
                            "rouge_2": 0.2117758784425451,
                            "rouge_l": 0.28446547769121083,
                        },
                        "claude-2.1": {
                            "rouge_1": 0.27644668407099343,
                            "rouge_2": 0.2113902286989767,
                            "rouge_l": 0.24698075405257724,
                        },
                        "mistralai/Mixtral-8x7B-Instruct-v0.1": {
                            "rouge_1": 0.34342250501733257,
                            "rouge_2": 0.2317481016730322,
                            "rouge_l": 0.28887771392081735,
                        },
                        "gpt-3.5-turbo-0613": {
                            "rouge_1": 0.35612039313597604,
                            "rouge_2": 0.23490148952584414,
                            "rouge_l": 0.3129549255100768,
                        },
                        "gpt-3.5-turbo-16k-0613": {
                            "rouge_1": 0.33419972640218876,
                            "rouge_2": 0.21136928667049149,
                            "rouge_l": 0.28279699476689896,
                        },
                        "gpt-35-turbo-1106": {
                            "rouge_1": 0.5045964619723097,
                            "rouge_2": 0.34497176946841374,
                            "rouge_l": 0.44386439153748247,
                        },
                        "gpt-4-1106-preview": {
                            "rouge_1": 0.47646276075300387,
                            "rouge_2": 0.3224811471591696,
                            "rouge_l": 0.44053461704042896,
                        },
                        "gemini-pro": {
                            "rouge_1": 0.4893990116371752,
                            "rouge_2": 0.31722516166960607,
                            "rouge_l": 0.4406185238322971,
                        },
                    },
                    "h2o_sonar.evaluators.perplexity_evaluator.PerplexityEvaluator": {
                        "h2oai/h2ogpt-4096-llama2-70b-chat": {
                            "perplexity": 2.903635488251016
                        },
                        "h2oai/h2ogpt-4096-llama2-13b-chat": {
                            "perplexity": 2.1178675896167523
                        },
                        "mistral-medium": {"perplexity": 1.8750369099544801},
                        "h2oai/h2ogpt-gm-7b-mistral-chat-sft-dpo-v1": {
                            "perplexity": 1.271003625941413
                        },
                        "h2oai/h2ogpt-gm-experimental": {
                            "perplexity": 1.2234868882962309
                        },
                        "h2oai/h2ogpt-32k-codellama-34b-instruct": {
                            "perplexity": 1.3133756481302583
                        },
                        "HuggingFaceH4/zephyr-7b-beta": {
                            "perplexity": 6.096579568483328
                        },
                        "01-ai/Yi-34B-Chat": {"perplexity": 4.145129857876982},
                        "claude-2.1": {"perplexity": 4.296065460638246},
                        "mistralai/Mixtral-8x7B-Instruct-v0.1": {
                            "perplexity": 5.050599715707967
                        },
                        "gpt-3.5-turbo-0613": {"perplexity": 3.1697671349702734},
                        "gpt-3.5-turbo-16k-0613": {"perplexity": 3.4444292045298863},
                        "gpt-35-turbo-1106": {"perplexity": 1.983316597157465},
                        "gpt-4-1106-preview": {"perplexity": 2.2146400296219833},
                        "gemini-pro": {"perplexity": 1.8069485818057187},
                    },
                    (
                        "h2o_sonar.evaluators.answer_semantic_similarity_per_sentence"
                        "_evaluator.AnswerSemanticSimilarityPerSentenceEvaluator"
                    ): {
                        "h2oai/h2ogpt-4096-llama2-70b-chat": {
                            "mean_answer_similarity": 0.8948524844637534,
                            "min_answer_similarity": 0.8396056604680879,
                        },
                        "h2oai/h2ogpt-4096-llama2-13b-chat": {
                            "mean_answer_similarity": 0.907992571596532,
                            "min_answer_similarity": 0.8725255871872543,
                        },
                        "mistral-medium": {
                            "mean_answer_similarity": 0.912302500609865,
                            "min_answer_similarity": 0.8595836764789789,
                        },
                        "h2oai/h2ogpt-gm-7b-mistral-chat-sft-dpo-v1": {
                            "mean_answer_similarity": 0.9426429985367194,
                            "min_answer_similarity": 0.9278683237855126,
                        },
                        "h2oai/h2ogpt-gm-experimental": {
                            "mean_answer_similarity": 0.9306100724863762,
                            "min_answer_similarity": 0.9128401013457559,
                        },
                        "h2oai/h2ogpt-32k-codellama-34b-instruct": {
                            "mean_answer_similarity": 0.9395440244983648,
                            "min_answer_similarity": 0.9247693497471582,
                        },
                        "HuggingFaceH4/zephyr-7b-beta": {
                            "mean_answer_similarity": 0.8853270159175183,
                            "min_answer_similarity": 0.8366201540501721,
                        },
                        "01-ai/Yi-34B-Chat": {
                            "mean_answer_similarity": 0.8937231007070499,
                            "min_answer_similarity": 0.8399538571346007,
                        },
                        "claude-2.1": {
                            "mean_answer_similarity": 0.8770235013061812,
                            "min_answer_similarity": 0.7926640145482474,
                        },
                        "mistralai/Mixtral-8x7B-Instruct-v0.1": {
                            "mean_answer_similarity": 0.8831737534783125,
                            "min_answer_similarity": 0.8293148580812889,
                        },
                        "gpt-3.5-turbo-0613": {
                            "mean_answer_similarity": 0.8924463408356013,
                            "min_answer_similarity": 0.8335476868875359,
                        },
                        "gpt-3.5-turbo-16k-0613": {
                            "mean_answer_similarity": 0.8916602506695138,
                            "min_answer_similarity": 0.830557639928814,
                        },
                        "gpt-35-turbo-1106": {
                            "mean_answer_similarity": 0.90350171100528,
                            "min_answer_similarity": 0.8620693105385616,
                        },
                        "gpt-4-1106-preview": {
                            "mean_answer_similarity": 0.8977261561602982,
                            "min_answer_similarity": 0.8297340700018072,
                        },
                        "gemini-pro": {
                            "mean_answer_similarity": 0.892531739619539,
                            "min_answer_similarity": 0.8600010050580197,
                        },
                    },
                    (
                        "h2o_sonar.evaluators.answer_accuracy_evaluator"
                        ".AnswerAccuracyEvaluator"
                    ): {
                        "h2oai/h2ogpt-4096-llama2-70b-chat": {
                            "answer_accuracy": 0.8450306289012791
                        },
                        "h2oai/h2ogpt-4096-llama2-13b-chat": {
                            "answer_accuracy": 0.8725255871872543
                        },
                        "mistral-medium": {"answer_accuracy": 0.8595836764789789},
                        "h2oai/h2ogpt-gm-7b-mistral-chat-sft-dpo-v1": {
                            "answer_accuracy": 0.9278683237855126
                        },
                        "h2oai/h2ogpt-gm-experimental": {
                            "answer_accuracy": 0.9128401013457559
                        },
                        "h2oai/h2ogpt-32k-codellama-34b-instruct": {
                            "answer_accuracy": 0.9247693497471582
                        },
                        "HuggingFaceH4/zephyr-7b-beta": {
                            "answer_accuracy": 0.8420450950228192
                        },
                        "01-ai/Yi-34B-Chat": {"answer_accuracy": 0.8573633222779855},
                        "claude-2.1": {"answer_accuracy": 0.7991169194247516},
                        "mistralai/Mixtral-8x7B-Instruct-v0.1": {
                            "answer_accuracy": 0.8347397990539362
                        },
                        "gpt-3.5-turbo-0613": {"answer_accuracy": 0.8584742508645679},
                        "gpt-3.5-turbo-16k-0613": {
                            "answer_accuracy": 0.855484203905846
                        },
                        "gpt-35-turbo-1106": {"answer_accuracy": 0.8620693105385616},
                        "gpt-4-1106-preview": {"answer_accuracy": 0.8637500883887214},
                        "gemini-pro": {"answer_accuracy": 0.8600010050580197},
                    },
                },
                "./data/generative/eval_llm/arabic_mmlu_test_lab_10p.json": {
                    "h2o_sonar.evaluators.rouge_evaluator.RougeEvaluator": {
                        "claude-3-7-sonnet-20250219-litellm-databrick": {
                            "rouge_1": 0.012132169290456473,
                            "rouge_2": 0.0,
                            "rouge_l": 0.012132169290456473,
                        },
                        "meta-llama/Meta-Llama-3.1-8B-Instruct": {
                            "rouge_1": 0.0025477707006369425,
                            "rouge_2": 0.0012903225806451613,
                            "rouge_l": 0.0025477707006369425,
                        },
                        "h2oai/h2o-danube3-4b-chat": {
                            "rouge_1": 0.04212765957446809,
                            "rouge_2": 0.0,
                            "rouge_l": 0.04212765957446809,
                        },
                    },
                    "h2o_sonar.evaluators.perplexity_evaluator.PerplexityEvaluator": {
                        "claude-3-7-sonnet-20250219-litellm-databrick": {
                            "perplexity": 3.3418649981136936
                        },
                        "meta-llama/Meta-Llama-3.1-8B-Instruct": {
                            "perplexity": 6.967694304343442
                        },
                        "h2oai/h2o-danube3-4b-chat": {"perplexity": 3.671781912420829},
                    },
                    (
                        "h2o_sonar.evaluators.answer_semantic_similarity_per_sentence"
                        "_evaluator.AnswerSemanticSimilarityPerSentenceEvaluator"
                    ): {
                        "claude-3-7-sonnet-20250219-litellm-databrick": {
                            "mean_answer_similarity": 0.8067816414043028,
                            "min_answer_similarity": 0.7703517362788602,
                        },
                        "meta-llama/Meta-Llama-3.1-8B-Instruct": {
                            "mean_answer_similarity": 0.9090561578084904,
                            "min_answer_similarity": 0.9014081094548176,
                        },
                        "h2oai/h2o-danube3-4b-chat": {
                            "mean_answer_similarity": 0.9010312684876016,
                            "min_answer_similarity": 0.8893710949155414,
                        },
                    },
                    (
                        "h2o_sonar.evaluators.answer_accuracy_evaluator"
                        ".AnswerAccuracyEvaluator"
                    ): {
                        "claude-3-7-sonnet-20250219-litellm-databrick": {
                            "answer_accuracy": 0.661413159839753
                        },
                        "meta-llama/Meta-Llama-3.1-8B-Instruct": {
                            "answer_accuracy": 0.8137581151773235
                        },
                        "h2oai/h2o-danube3-4b-chat": {
                            "answer_accuracy": 0.7777733486518074
                        },
                    },
                },
            },
        ),
        # ####################################
        # # ROUGE evaluator, ALL test labs 93"
        # (
        #     [rouge_evaluator.RougeEvaluator],
        #     _given_test_labs_with_aa(),
        #     1,
        #     "49f01ef5578d06cb7c2161f9765ba115abcaa5074dc513a16e5885258ae9960b",
        # ),
        # # ROUGE evaluator, 100MB test lab
        # (
        #     [rouge_evaluator.RougeEvaluator],
        #     [
        #         "./data/generative/rouge_long_aa_test_lab_1p.json",
        #     ],
        #     100,
        #     "d902292e80750b98d06c49ae7361a72b4bb6e6410be30f914f5c8a22da629858",
        # ),
        # ####################################
        # # Perplexity evaluator - 100MB test lab - set device to CPU and GPU > compare
        # (
        #     [perplexity_evaluator.PerplexityEvaluator],
        #     [
        #         "./data/generative/rouge_long_aa_test_lab_1p.json",
        #     ],
        #     100,
        #     # CPU - reproducible if ALWAYS run on CPU (GPU has different SHA)
        #     "132cade20c5c48b14486147a9947e6bfdcd8e53e190603a0b8642ab8c55e4a9d",
        #     # GPU - reproducible if ALWAYS run on GPU/CUDA (CPU has different SHA)
        #     # "6c850626de15f26e8f16f1652e61bd24c0af63cfa450d4c24cf21100ca05b065",
        # ),
        # # TODO - never ran: all deterministic evaluators @ all test labs > 1x
        # (
        #     _given_deterministic_evaluator_classes(),
        #     _given_test_labs_with_aa(),
        #     1,
        #     "...",
        # ),
    ],
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_reproducibility_test_labs(
    tmp_path,
    evaluator_classes,
    test_lab_paths: list[str],
    multiplier: int,
    expected_sha256: str,
    expected_scores: dict,
) -> None:
    """This test runs ROUGE eval on ALL test labs in the data/generative/*.json."""
    try:
        test_key = str(uuid.uuid4())
        # map: test_lab_path => evaluator ID => last seen ROUGE scores
        all_deterministic_scores = {}
        # list of tuples (expected_answer, actual_answer)
        answers = []
        failed_test_labs = []
        test_cases_count = 0

        test_labs_scores_path = f"/tmp/DETERMINISTIC_scores_{test_key}.json"

        h2ogpte_connection = test_utils.health.get_h2ogpte()

        for e, test_lab_path in enumerate(test_lab_paths):
            print(f"DETERMINISTIC test lab path [{e}]: {test_lab_path}", flush=True)

            test_lab = testing.RagTestLab.load_from_json(
                llm_host_connection=h2ogpte_connection,
                file_path=test_lab_path,
                docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
            )
            the_dataset = test_lab.dataset
            test_cases_count += len(the_dataset.inputs)
            for r in the_dataset.inputs:
                answers.append(
                    (r.expected_output * multiplier, r.actual_output * multiplier)
                )
            the_models = list(test_lab.evaluated_models.values())

            for evaluator_class in evaluator_classes:
                #
                # WHEN
                #
                if (
                    evaluators.KEYWORD_REQUIRES_OPENAI_KEY in evaluator_class._keywords
                    and not test_utils.Health().is_openai()
                ):
                    print(
                        f"SKIPPING: evaluator {evaluator_class.explainer_id()} "
                        f"which requires OpenAI key which is NOT set"
                    )
                    continue
                evaluator_id = evaluator_class.explainer_id()

                print(
                    f"RUNNING DETERMINISTIC evaluator [{e}]: {evaluator_id} "
                    f"on test lab: {test_lab_path}",
                    flush=True,
                )
                evaluation = evaluate.run_evaluation(
                    dataset=the_dataset,
                    models=the_models,
                    evaluators=[evaluator_id],
                    results_location=tmp_path,
                    log_level=loggers.DEBUG,
                )

                # evaluation result
                print(f"DETERMINISTIC evaluation [{e}]:\n{evaluation}")
                print(f"HTML:\nfile://{evaluation.result.get_html_report_location()}")

                # lookup scores
                jobs = evaluation.get_jobs_for_explainer_id(evaluator_id)
                if not jobs:
                    err_msg = (
                        f"SKIPPING: No jobs found for evaluator {evaluator_id} "
                        f"test lab {test_lab_path}: "
                        f"file://{evaluation.result.get_html_report_location()}"
                    )
                    print(err_msg)
                    failed_test_labs.append(
                        (
                            "NO JOBS",
                            evaluator_id,
                            test_lab_path,
                            evaluation.result.get_html_report_location(),
                        )
                    )
                    # raise RuntimeError(err_msg)
                    continue

                ep = persistences.ExplainerPersistence(
                    data_dir=str(tmp_path),
                    mli_key=evaluation.key,
                    username=commons.DEFAULT_USER,
                    explainer_id=evaluator_id,
                    explainer_job_key=jobs[0].key,
                )

                # lookup leaderboard path
                json_leaderboard_path = None
                leaderboard_types = [
                    e10s.LlmHeatmapLeaderboardExplanation,
                    e10s.LlmBoolLeaderboardExplanation,
                    e10s.LlmClassifierLeaderboardExplanation,
                ]
                for l_type in leaderboard_types:
                    potential_json_leaderboard_path = ep.get_explanation_file_path(
                        explanation_type=l_type.explanation_type(),
                        explanation_format=f5s.LlmLeaderboardJSonFormat.mime,
                    )
                    if pathlib.Path(potential_json_leaderboard_path).exists():
                        json_leaderboard_path = potential_json_leaderboard_path
                        break
                if not json_leaderboard_path:
                    raise RuntimeError(
                        f"Cannot find leaderboard path for evaluator {evaluator_id}"
                        f" and test lab {test_lab_path}"
                    )

                with open(json_leaderboard_path) as f:
                    json_leaderboard = json.load(f)
                print(
                    f"DETERMINISTIC leaderboard for {evaluator_id} and {test_lab_path}:"
                    f"\n{json_leaderboard}"
                )
                assert json_leaderboard
                assert json_leaderboard[f5s.ExplanationFormat.KEY_FILES][
                    e10s.AbcHeatmapExplanation.METRIC_ALL
                ]
                all_metrics_file = json_leaderboard[f5s.ExplanationFormat.KEY_FILES][
                    e10s.AbcHeatmapExplanation.METRIC_ALL
                ]
                json_leaderboard_data_path = json_leaderboard_path.replace(
                    "explanation.json", all_metrics_file
                )
                print(f"Leaderboard data path: {json_leaderboard_data_path}")
                with open(json_leaderboard_data_path) as f:
                    json_leaderboard_data = f.read()
                print(f"Leaderboard data:\n{json_leaderboard_data}")
                assert json_leaderboard_data

                # reproducibility check @ average scores
                with open(json_leaderboard_data_path) as f:
                    jlbd = json.load(f)
                print(
                    f"DETERMINISTIC leaderboard data: {json.dumps(jlbd, indent=2)}",
                    flush=True,
                )
                try:
                    metrics_scores = jlbd[f5s.ExplanationFormat.KEY_DATA]
                    print(f"DETERMINISTIC scores: {metrics_scores}")

                    if str(test_lab_path) not in all_deterministic_scores:
                        all_deterministic_scores[str(test_lab_path)] = {}
                    if evaluator_id not in all_deterministic_scores[str(test_lab_path)]:
                        all_deterministic_scores[str(test_lab_path)][evaluator_id] = {}
                    all_deterministic_scores[str(test_lab_path)][evaluator_id] = (
                        metrics_scores
                    )

                    # save last seen metricscores for all test labs to a file
                    print(
                        f"Saving DETERMINISTIC scores for all test labs to:"
                        f" {test_labs_scores_path}"
                    )
                    with open(test_labs_scores_path, "w") as f:
                        json.dump(all_deterministic_scores, f, indent=2)
                except KeyError as e:
                    print(
                        f"SKIPPING: no DETERMINISTIC scores for test lab "
                        f"{test_lab_path}"
                    )
                    failed_test_labs.append(
                        (
                            "NO SCORES",
                            evaluator_id,
                            test_lab_path,
                            evaluation.result.get_html_report_location(),
                        )
                    )
                    raise e

        print(f"Failed DETERMINISTIC test labs ({len(failed_test_labs)}):")
        for reason, evaluator_id, path, html in failed_test_labs:
            print(f"- {reason}: {evaluator_id} {path} (HTML: file://{html})")

        # calculate SHA first
        test_lab_scores_sha = test_utils.file_hash_sha256(str(test_labs_scores_path))

        # compare expected vs actual scores ONLY if SHA is different
        differences = []
        total_metrics = 0
        different_metrics = 0

        if test_lab_scores_sha != expected_sha256:
            for test_lab_path, test_lab_evaluators in all_deterministic_scores.items():
                if test_lab_path not in expected_scores:
                    differences.append(
                        f"  ! Test lab not in expected scores: {test_lab_path}"
                    )
                    continue

                for evaluator_id, models in test_lab_evaluators.items():
                    if evaluator_id not in expected_scores[test_lab_path]:
                        differences.append(
                            f"  ! Evaluator not in expected scores: {evaluator_id}"
                        )
                        continue

                    for model_name, metrics in models.items():
                        if (
                            model_name
                            not in expected_scores[test_lab_path][evaluator_id]
                        ):
                            differences.append(
                                f"  ! Model not in expected scores: {model_name}"
                            )
                            continue

                        expected_model_scores = expected_scores[test_lab_path][
                            evaluator_id
                        ][model_name]
                        for metric_name, actual_value in metrics.items():
                            total_metrics += 1
                            expected_value = expected_model_scores.get(metric_name)

                            if expected_value is None:
                                differences.append(
                                    f"  ! {evaluator_id} | {model_name} | "
                                    f"{metric_name}: "
                                    f"MISSING in expected scores"
                                )
                                different_metrics += 1
                            elif actual_value != expected_value:
                                diff = actual_value - expected_value
                                diff_pct = (
                                    (diff / expected_value * 100)
                                    if expected_value != 0
                                    else float("inf")
                                )
                                differences.append(
                                    f"  ! {evaluator_id} | {model_name} | "
                                    f"{metric_name}: "
                                    f"expected={expected_value:.10f}, "
                                    f"actual={actual_value:.10f}, "
                                    f"diff={diff:+.10f} ({diff_pct:+.4f}%)"
                                )
                                different_metrics += 1

        # build the summary string
        summary_lines = [
            "\nDETERMINISTIC test summary:",
            f"- evaluators : {len(evaluator_classes)}",
        ]
        for e_c in evaluator_classes:
            summary_lines.append(f"    {e_c.evaluator_id()}")
        summary_lines.extend(
            [
                f"- test labs  : {len(test_lab_paths)}",
                f"- device     : {h2o_sonar_config.config.resolve_gpu_cpu_device()}",
                f"- failed test labs: {len(failed_test_labs)}",
                f"- test cases : {test_cases_count}",
                f"- shortest EA: {min([len(a[1]) for a in answers]):,}",
                f"- longest EA : {max([len(a[1]) for a in answers]):,}",
                f"- shortest AA: {min([len(a[0]) for a in answers]):,}",
                f"- longest AA : {max([len(a[0]) for a in answers]):,}",
            ]
        )

        # add SHA and metric comparison results
        summary_lines.append(f"- DETERMINISTIC scores SHA256: {test_lab_scores_sha}")
        if test_lab_scores_sha == expected_sha256:
            summary_lines.append("- SHA matches: all metrics are identical")
        else:
            summary_lines.append("- SHA differs: comparing metrics...")
            if differences:
                summary_lines.append(
                    f"- different metrics: {different_metrics}/{total_metrics}"
                )
                summary_lines.append("- metric differences:")
                summary_lines.extend(differences[:50])  # limit to first 50 differences
                if len(differences) > 50:
                    summary_lines.append(
                        f"  ... and {len(differences) - 50} more differences"
                    )
            else:
                summary_lines.append(
                    f"- all metrics match: {total_metrics}/{total_metrics}"
                )
        summary = "\n".join(summary_lines)

        # print the summary
        print(summary)

        # assert with summary in the message
        assert test_lab_scores_sha == expected_sha256, (
            f"SHA validation failed!\n{summary}"
        )

    except Exception as ex:
        print(
            f"FAILED: {ex}\n{traceback.format_exc()}",
            flush=True,
        )
        raise ex


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
