# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
#
# LLM/RAG gear: conversion, imports, helpers, loading, saving, ...
#
import json
import math
import os
import traceback

import datatable
import pytest

from h2o_sonar import evaluate
from h2o_sonar import loggers
from h2o_sonar.evaluators import bleu_evaluator as b_e
from h2o_sonar.evaluators import classification_evaluator as c_e
from h2o_sonar.evaluators import contact_information_byop_evaluator as ci_p_e
from h2o_sonar.evaluators import encoding_guardrail_evaluator as e_g_e
from h2o_sonar.evaluators import fairness_bias_evaluator as fb_e
from h2o_sonar.evaluators import gptscore_machine_translation_evaluator as gpt_mt_e
from h2o_sonar.evaluators import gptscore_question_answering_evaluator as gpt_qa_e
from h2o_sonar.evaluators import gptscore_summary_with_reference_evaluator as gpt_sum_w
from h2o_sonar.evaluators import (
    gptscore_summary_without_reference_evaluator as gpt_s_wo,
)
from h2o_sonar.evaluators import language_mismatch_byop_evaluator as lm_p_e
from h2o_sonar.evaluators import looping_detection_evaluator as ld_e
from h2o_sonar.evaluators import parameterizable_byop_evaluator as p_p_e
from h2o_sonar.evaluators import perplexity_evaluator as ppx_e
from h2o_sonar.evaluators import pii_leakage_evaluator as pii_e
from h2o_sonar.evaluators import rag_answer_correctness_evaluator as ac_e
from h2o_sonar.evaluators import rag_answer_relevancy_evaluator as ar_e
from h2o_sonar.evaluators import rag_answer_relevancy_no_judge_evaluator as ar_e_nj
from h2o_sonar.evaluators import rag_answer_similarity_evaluator as as_e
from h2o_sonar.evaluators import rag_chunk_relevancy_evaluator as c_r_e
from h2o_sonar.evaluators import rag_context_precision_evaluator as cp_e
from h2o_sonar.evaluators import rag_context_recall_evaluator as crc_e
from h2o_sonar.evaluators import rag_context_relevancy_evaluator as cr_e
from h2o_sonar.evaluators import rag_faithfulness_evaluator as f_e
from h2o_sonar.evaluators import rag_groundedness_evaluator as g_e
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
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import evaluators as e8s
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import results
from h2o_sonar.utils import testing
from tests import test_utils


_ALL_EVALUATORS = [
    ld_e.LoopingDetectionEvaluator,
    ar_e_nj.RagAnswerRelevancyNoJudgeEvaluator,
    c_r_e.ContextChunkRelevancyEvaluator,
    g_e.RagGroundednessEvaluator,
    ac_e.AnswerCorrectnessEvaluator,
    ar_e.AnswerRelevancyEvaluator,
    as_e.AnswerSemanticSimilarityEvaluator,
    b_e.BleuEvaluator,
    c_e.ClassificationEvaluator,
    ci_p_e.ContactInformationByopEvaluator,
    cp_e.ContextPrecisionEvaluator,
    cr_e.ContextRelevancyEvaluator,
    crc_e.ContextRecallEvaluator,
    f_e.FaithfulnessEvaluator,
    fb_e.FairnessBiasEvaluator,
    gpt_mt_e.GptScoreMachineTranslationEvaluator,
    gpt_qa_e.GptScoreQuestionAnsweringEvaluator,
    gpt_sum_w.GptScoreSummaryWithReferenceEvaluator,
    gpt_s_wo.GptScoreSummaryWithoutReferenceEvaluator,
    hal_e.RagHallucinationEvaluator,
    lm_p_e.LanguageMismatchByopEvaluator,
    p_p_e.ParameterizableByopEvaluator,
    pii_e.PiiLeakageEvaluator,
    ppx_e.PerplexityEvaluator,
    r_e.RougeEvaluator,
    rag_ragas_evaluator.RagasEvaluator,
    s_e.SummarizationEvaluator,
    s_p_e.SexismByopEvaluator,
    sdl_e.SensitiveDataLeakageEvaluator,
    st_p_e.StereotypeByopEvaluator,
    su_p_e.SummarizationByopEvaluator,
    t_e.ToxicityEvaluator,
    tp_e.RagStrStrEvaluator,
    e_g_e.EncodingGuardrailEvaluator,
]


#
# KEYWORD GROUPS sanity tests
#


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_purpose_keyword_group():
    #
    # GIVEN
    #
    keyword_group = e8s.KEYWORD_GROUPS.get_group(prefix=e8s.PREFIX_ES_PURPOSE)

    #
    # WHEN
    #
    for e in _ALL_EVALUATORS:
        #
        # THEN
        #

        assert keyword_group.is_member(e._keywords), (
            f"Evaluator {e.evaluator_id()} with keywords {e._keywords} "
            f"is not in group {keyword_group.name}"
        )


#
# METRICS METADATA
#


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_evaluation_metrics_meta():
    #
    # GIVEN
    #
    es = _ALL_EVALUATORS

    #
    # WHEN
    #

    #
    # THEN
    #
    for e in es:
        print(f"{e.evaluator_id()}")
        print(f"  {e._metrics_meta}")
        assert e._metrics_meta
        assert e._metrics_meta.size() > 0

        if e in [
            rag_ragas_evaluator.RagasEvaluator,
            t_e.ToxicityEvaluator,
            tp_e.RagStrStrEvaluator,
            ci_p_e.ContactInformationByopEvaluator,
            lm_p_e.LanguageMismatchByopEvaluator,
            p_p_e.ParameterizableByopEvaluator,
            s_p_e.SexismByopEvaluator,
            st_p_e.StereotypeByopEvaluator,
            su_p_e.SummarizationByopEvaluator,
            pii_e.PiiLeakageEvaluator,
            e_g_e.EncodingGuardrailEvaluator,
            sdl_e.SensitiveDataLeakageEvaluator,
        ]:
            assert e._metrics_meta.size() >= 5, f"Expected at least 5 metrics: {e}"
        elif e in [
            c_e.ClassificationEvaluator,
            r_e.RougeEvaluator,
            b_e.BleuEvaluator,
            s_e.SummarizationEvaluator,
            gpt_mt_e.GptScoreMachineTranslationEvaluator,
            gpt_qa_e.GptScoreQuestionAnsweringEvaluator,
            gpt_sum_w.GptScoreSummaryWithReferenceEvaluator,
            gpt_s_wo.GptScoreSummaryWithoutReferenceEvaluator,
            ld_e.LoopingDetectionEvaluator,
        ]:
            assert e._metrics_meta.size() >= 3, f"Expected at least 3 metrics: {e}"
        elif e in [
            c_r_e.ContextChunkRelevancyEvaluator,
        ]:
            assert e._metrics_meta.size() > 1, f"Expected exactly 2 metrics: {e}"
        else:
            assert e._metrics_meta.size() == 1, f"Expected exactly 1 metric: {e}"

        print("  Metrics:")
        for m in e._metrics_meta.get_metric_keys():
            print(f"    {m}")


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_results_denormalization():
    #
    # GIVEN
    #
    data = {
        "data": {
            "mistralai/Mixtral-8x7B-Instruct-v0.1": {
                "answer_relevancy": 0.9421541759377194,
                "context_precision": 0.9999999999166667,
                "faithfulness": 0.7777777777777778,
                "context_recall": 1.0,
                "ragas": 0.9201511904967471,
            },
            "h2oai/h2ogpt-4096-llama2-70b-chat": {
                "answer_relevancy": 0.9500240535342429,
                "context_precision": 0.8333333332666667,
                "faithfulness": "NaN",
                "context_recall": commons.SafeJavaScript.INF,
                "ragas": commons.SafeJavaScript.NEG_INF,
            },
            "h2oai/h2ogpt-4096-llama2-13b-chat": {
                "answer_relevancy": 0.9128441230237397,
                "context_precision": 0.8333333332666667,
                "faithfulness": 0.6666666666666666,
                "context_recall": 1.0,
                "ragas": 0.8341192677148174,
            },
        },
        "metadata": {
            "ragas": {
                "key": "ragas",
                "display_name": "RAGAS",
                "data_type": "float",
                "display_value": "{v:.4f}",
                "description": "RAGAs (RAG Assessment) metric is ...",
                "value_range": [0.0, 1.0],
                "value_enum": None,
                "higher_is_better": True,
                "threshold": 0.75,
                "is_primary_metric": True,
                "parent_metric": "",
                "exclude": False,
            },
            "faithfulness": {
                "key": "faithfulness",
                "display_name": "Faithfulness",
                "data_type": "float",
                "display_value": "{v:.4f}",
                "description": "Faithfulness (generation) metric measures ...",
                "value_range": [0.0, 1.0],
                "value_enum": None,
                "higher_is_better": True,
                "threshold": 0.75,
                "is_primary_metric": False,
                "parent_metric": "",
                "exclude": False,
            },
            "answer_relevancy": {
                "key": "answer_relevancy",
                "display_name": "Answer relevancy",
                "data_type": "float",
                "display_value": "{v:.4f}",
                "description": "Answer relevancy metric (retrieval+generation) ...",
                "value_range": [0.0, 1.0],
                "value_enum": None,
                "higher_is_better": True,
                "threshold": 0.75,
                "is_primary_metric": False,
                "parent_metric": "",
                "exclude": False,
            },
            "context_precision": {
                "key": "context_precision",
                "display_name": "Context precision",
                "data_type": "float",
                "display_value": "{v:.4f}",
                "description": "Context precision metric (retrieval) evaluator ...",
                "value_range": [0.0, 1.0],
                "value_enum": None,
                "higher_is_better": True,
                "threshold": 0.75,
                "is_primary_metric": False,
                "parent_metric": "",
                "exclude": False,
            },
            "context_recall": {
                "key": "context_recall",
                "display_name": "Context recall",
                "data_type": "float",
                "display_value": "{v:.4f}",
                "description": "Context recall metric (retrieval) measures ...",
                "value_range": [0.0, 1.0],
                "value_enum": None,
                "higher_is_better": True,
                "threshold": 0.75,
                "is_primary_metric": False,
                "parent_metric": "",
                "exclude": False,
            },
        },
    }

    #
    # WHEN
    #

    results.LeaderboardResult._str_nan_inf_to_math(data)

    #
    # THEN
    #
    print(json.dumps(data, indent=4))
    assert not isinstance(
        data["data"]["h2oai/h2ogpt-4096-llama2-70b-chat"]["faithfulness"],
        str,
    )
    assert not isinstance(
        data["data"]["h2oai/h2ogpt-4096-llama2-70b-chat"]["context_recall"],
        str,
    )
    assert not isinstance(
        data["data"]["h2oai/h2ogpt-4096-llama2-70b-chat"]["ragas"],
        str,
    )
    assert math.isnan(data["data"]["h2oai/h2ogpt-4096-llama2-70b-chat"]["faithfulness"])


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_load_llm_dataset():
    #
    # GIVEN
    #
    prompts_path = test_utils.find_locally(
        "data/generative/talk2report_prompts_dataset.json"
    )

    #
    # WHEN
    #
    llm_dataset = datasets.LlmDataset.load_from_json(prompts_path)

    #
    # THEN
    assert llm_dataset
    print(json.dumps(llm_dataset.to_dict(), indent=4))
    assert len(llm_dataset.to_dict().get(datasets.LlmDataset.KEY_INPUTS, [])) > 1


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_llm_dataset_json_columns_encoding(tmp_path):
    """Test encoding of the fields (listed below) to JSon and back when
    exporting to / importing from datatable:

    - LlmDataset.COL_CORPUS,
    - LlmDataset.COL_CATEGORIES,
    - LlmDataset.COL_OUTPUT_CONSTRAINTS,
    - LlmDataset.COL_CONTEXT,

    Data structures which use the encoding:

    - LlmDataset
        - does encoding
    - LlmEvalResult
        - does encoding
    - TestLab
        - test lab has `raw_dataset` and `dataset` fields which are LlmDataset
        - no need to test it as it is covered by ^ tests

    Hint:

    In the past were ^ fields lists of strings which were converted to single string
    with `SEPARATOR_CATS = "@H2OS@CS@"` separator escaping `,` which would cause
    problems in CSV format.

    """

    #
    # GIVEN
    #

    # LlmDataset
    llm_dataset = datasets.LlmDataset.load_from_json(
        test_utils.find_locally(
            "data/generative/kaggle_llm_science_exam_dataset_h2o_small.json"
        )
    )
    # LlmEvalResult (created from the LlmDataset)
    eval_results = datasets.LlmEvalResults()
    for i, ii in enumerate(llm_dataset.inputs):
        eval_results.add_result(
            datasets.LlmEvalResults.LlmEvalResultRow(
                ii,
                {
                    "foo_metric": i * 0.1,
                },
            )
        )

    #
    # WHEN
    #

    # serialize and save
    llm_dataset_csv_path = str(tmp_path / "llm_dataset.csv")
    eval_results_csv_path = str(tmp_path / "llm_eval_result.csv")

    llm_dataset.save_as_json(tmp_path / "llm_dataset.json")
    llm_dataset.to_datatable().to_csv(llm_dataset_csv_path)
    llm_dataset.to_datatable().to_jay(str(tmp_path / "llm_dataset.bin"))

    eval_results.save_as_json(tmp_path / "llm_eval_result.json")
    eval_results.to_datatable().to_csv(eval_results_csv_path)
    eval_results.to_datatable().to_jay(str(tmp_path / "llm_eval_result.bin"))

    # load and deserialize
    loaded_llm_dataset = datasets.LlmDataset.from_datatable_dict(
        datatable.fread(llm_dataset_csv_path).to_dict()
    )
    loaded_eval_results_dt = datatable.fread(eval_results_csv_path).to_dict()

    #
    # THEN
    #

    expected_corpus = ["https://www.wikipedia.org/"]
    expected_categories = ["question_answering", "kaggle"]
    expected_constraints = ["MOND", "2"]

    # assert LLM dataset
    assert loaded_llm_dataset
    input_dict = loaded_llm_dataset.inputs[0].to_dict()
    assert input_dict[datasets.LlmDataset.COL_CORPUS] == expected_corpus
    assert input_dict[datasets.LlmDataset.COL_CATEGORIES] == expected_categories
    assert isinstance(input_dict[datasets.LlmDataset.COL_CONTEXT], list)
    assert len(input_dict[datasets.LlmDataset.COL_CONTEXT]) == 5
    assert (
        input_dict[datasets.LlmDataset.COL_OUTPUT_CONSTRAINTS] == expected_constraints
    )

    # assert LLM eval results (datatable)
    assert loaded_eval_results_dt
    assert loaded_eval_results_dt[datasets.LlmDataset.COL_CORPUS][0] == json.dumps(
        expected_corpus
    )
    assert loaded_eval_results_dt[datasets.LlmDataset.COL_CATEGORIES][0] == json.dumps(
        expected_categories
    )
    assert loaded_eval_results_dt[datasets.LlmDataset.COL_OUTPUT_CONSTRAINTS][
        0
    ] == json.dumps(expected_constraints)
    assert isinstance(loaded_eval_results_dt[datasets.LlmDataset.COL_CONTEXT][0], str)
    assert loaded_eval_results_dt[datasets.LlmDataset.COL_CONTEXT][0].startswith("[")


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_llm_results_explanation(tmp_path):
    """Method:

    - LlmDataset    ... is the test set which contains test data (datatable frame)
                    ... self.inputs == list[LlmDatasetRow]
    - <calculation> ... of one or more metrics ~ per metric column frame
    - LlmEvalResult ... is the LlmDataset with added metric columns
                    ... self.results == list[LlmEvalResultRow]
                    ... self.add_result(
                          LllDatasetRow, {metric_1_id: value, metric_2_id: ...})
                    ... serialization to all formats (JSon, CSV, HTML, ...)
                    ... INPUT of all explanations & formats

    """
    #
    # GIVEN
    #
    llm_dataset = datasets.LlmDataset.load_from_json(
        test_utils.find_locally(
            "data/generative/kaggle_llm_science_exam_dataset_h2o_small.json"
        )
    )

    #
    # WHEN
    #
    eval_result = datasets.LlmEvalResults()
    for i, ii in enumerate(llm_dataset.inputs):
        eval_result.add_result(
            datasets.LlmEvalResults.LlmEvalResultRow(
                ii,
                {
                    "context_recall": i * 0.1,
                    "faithfulness": i * 1.1,
                },
            )
        )

    #
    # THEN
    #
    eval_result.save_as_json(tmp_path / "llm_eval_result.json")
    eval_result.to_datatable().to_csv(str(tmp_path / "llm_eval_result.csv"))
    eval_result.to_datatable().to_jay(str(tmp_path / "llm_eval_result.bin"))
    assert eval_result.to_dict()


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_load_suite_without_keys(tmp_path):
    #
    # GIVEN
    #
    no_keys_suite_path = (
        "data/generative/h2ogpte_benchmark_test_suite_no_constraints.json"
    )

    #
    # WHEN
    #
    test_suite = testing.RagTestSuiteConfig.load_from_json(
        test_utils.find_locally(no_keys_suite_path)
    )
    suite_path = test_suite.save_as_json(tmp_path / "test_suite_with_keys.json")

    #
    # THEN
    #
    print(test_suite)
    assert test_suite
    assert len(test_suite.test_cases) == 2
    with open(suite_path) as f:
        suite_str = f.read()
    assert "key" in suite_str


@pytest.mark.skip(reason="Leaderboard JSon representations merger")
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_merge_leaderboards(tmp_path):
    #
    # GIVEN
    #
    json_dir_path = (
        "/home/user/h/mli/eval-studio-gallery/reports/h2ogpte-arno-benchmark"
        "/report-ragas-small/explainer_h2o_sonar_explainers_llm_ragas_evaluator_"
        "RagasEvaluator_6d0fcdb9-d4b2-439f-800c-c609175bf0b6"
        "/global_heatmap_leaderboard/application_json"
    )

    with open(json_dir_path + "/explanation.json") as f:
        idx_dict = json.load(f)

    #
    # WHEN
    #
    # map: model > metric_id > value
    merged_metrics = {}

    # iterate map: "files" > metrics > file
    for metric_id in idx_dict["files"]:
        file_name = idx_dict["files"][metric_id]
        # load data file
        with open(json_dir_path + "/" + file_name) as f:
            data_dict = json.load(f)

        # merge
        for model_id in data_dict:
            if merged_metrics.get(model_id) is None:
                merged_metrics[model_id] = {}
            merged_metrics[model_id][metric_id] = data_dict[model_id][metric_id]

    # save merged metrics
    with open(tmp_path / "merged_metrics.json", mode="w") as f:
        json.dump(merged_metrics, f, indent=4)

    #
    # THEN
    #
    print(json.dumps(merged_metrics, indent=4))


@pytest.mark.skip(reason="Data transcoder from JSon to JSon for frontend")
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_extract_prompts_from_suite(tmp_path):
    #
    # GIVEN
    #
    test_suite = testing.RagTestSuiteConfig.load_from_json(
        test_utils.find_locally(
            "data/generative/kaggle_llm_science_exam_test_suite.json"
        )
    )

    #
    # WHEN
    #
    ui_format = {"prompts": []}
    for i, tc in enumerate(test_suite.test_cases):
        ui_format["prompts"].append(
            {
                "id": i + 1,
                "prompt": tc.prompt,
                "answer": tc.expected_output,
                "category": tc.categories[0] if tc.categories else "",
            }
        )

    with open(tmp_path / "prompts.json", mode="w") as f:
        json.dump(ui_format, f, indent=4)

    #
    # THEN
    #
    print(ui_format)
    assert ui_format


@pytest.mark.skip(reason="LLM dataset 2 test config converter")
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_llm_dataset_2_testconfig(tmp_path):
    #
    # GIVEN
    #
    prompts_path = test_utils.find_locally(
        "data/generative/talk2report_prompts_dataset.json"
    )
    llm_dataset = datasets.LlmDataset.load_from_json(prompts_path)
    assert llm_dataset
    print(json.dumps(llm_dataset.to_dict(), indent=4))
    assert len(llm_dataset.to_dict().get(datasets.LlmDataset.KEY_INPUTS, [])) > 1

    #
    # WHEN
    #
    test_suite = testing.RagTestSuiteConfig.from_llm_dataset(llm_dataset)

    #
    # THEN
    print(test_suite.to_dict())
    assert test_suite
    test_suite.save_as_json(tmp_path / "rag_test_suite.json")


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_heatmap_leaderboard_explanation(tmp_path):
    #
    # GIVEN
    #
    llm_model_names = [
        "h2oai/h2ogpt-4096-llama2-7b-chat",
        "HuggingFaceH4/zephyr-7b-alpha",
        "gpt-3.5-turbo-0613",
    ]

    llm_models = [
        models.ExplainableRagModel(
            connection=test_utils.health.get_h2ogpte(),
            model_type=models.ExplainableModelType.h2ogpte,
            name="Mocked RAG model",
            collection_id="",
            collection_name="",
            llm_model_name=llm_model_name,
            documents=None,
            logger=loggers.SonarPrintLogger(),
        )
        for llm_model_name in llm_model_names
    ]

    llm_models_dict = {llm_model.key: llm_model for llm_model in llm_models}

    # explanation
    t_ragas_evaluator = rag_ragas_evaluator.RagasEvaluator
    explanation = e10s.LlmHeatmapLeaderboardExplanation(
        evaluator="<explainer>",
        eval_results=datasets.LlmEvalResults(),
        metrics_meta=commons.MetricsMeta(
            metrics=[
                t_ragas_evaluator.METRIC_META_CONTEXT_RECALL,
                t_ragas_evaluator.METRIC_META_FAITHFULNESS,
            ]
        ),
        display_name="Test Gear Evaluator",
        display_category="Test Gear Evaluator",
        key_2_evaluated_model=llm_models_dict,
        logger=loggers.SonarPrintLogger(),
    )

    #
    # WHEN
    #

    for llm_model in llm_models:
        for i in range(3):
            for mdelta in [0.1, 0.2, 0.3]:
                metrics_id = (
                    t_ragas_evaluator.METRIC_META_CONTEXT_RECALL.key
                    if i % 2 == 0
                    else t_ragas_evaluator.METRIC_META_FAITHFULNESS.key
                )
                value = 0.4 + mdelta

                r = datasets.LlmEvalResults.LlmEvalResultRow(
                    dataset_row=datasets.LlmDataset.LlmDatasetRow(
                        i=f"PROMPT {i}",
                        corpus=[f"DOC {i}"],
                        model_key=llm_model.key,
                    ),
                    metrics={
                        metrics_id: value,
                    },
                )

                explanation.add_col_value(
                    llm_model_name=llm_model.llm_model_name,
                    docs=r.dataset_row.corpus[0],
                    prompt=r.dataset_row.i,
                    metrics_id=metrics_id,
                    value=value,
                    result_row=r,
                )

    #
    # THEN
    #

    print(json.dumps(explanation.data_dict, indent=4))

    # HTML
    table_html = explanation.as_html(
        sort_by_metric_id=t_ragas_evaluator.METRIC_META_CONTEXT_RECALL.key
    )
    print(table_html)
    assert table_html
    with open(tmp_path / "metrics_heatmap.html", mode="w") as f:
        f.write(table_html)


@pytest.mark.skip(reason="Dataset builder for talk2report which chats with h2oGPTe")
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_build_talk2report_dataset(tmp_path):
    # TODO rewrite to new LLM helper methods

    #
    # GIVEN
    #
    prompts_str = [
        # model compliance
        "Did interpretation find any model problem?",
        "How many model problems were found?",
        "Were there any HIGH/MEDIUM/LOW severity problems?",
        "Suggest how to solve the HIGH SEVERITY problem.",
        # "Suggest how to solve <XYZ> problem?",
        "Create action plan as bullet list to solve the LOW SEVERITY problem.",
        "Is the model fair?",
        "Which features lead to the highest model error?",
        "Summarize the report.",
        # model understanding
        "What is the target column of the model?",
        "Which features are used by the model?",
        "What is the most important original feature of the model?",
        "What is the most important transformed feature of the model?",
        "What are the 3 most important original features of the model?",
        "What are the 3 most important transformed features of the model?",
        "What were the columns of the training dataset?",
        "Which explainers were run by the interpretation?",
    ]

    prompts = datasets.LlmDataset()

    # get the answer from h2oGPTE
    from h2ogpte import H2OGPTE

    gpte_remote_address = "https://h2ogpte.h2o.ai"
    gpte_api_key = os.getenv("H2O_GPT_E_API_KEY")
    assert gpte_api_key
    collection_name = "DeepEval 2023/11/03 (manual upload)"
    collection_id = ""
    client = H2OGPTE(address=gpte_remote_address, api_key=gpte_api_key)
    print("Recent collections:")
    recent_collections = client.list_recent_collections(0, 1000)
    for c in recent_collections:
        if c.name == collection_name and c.document_count:
            collection_id = c.id
            break
    assert collection_id
    chat_session_id = client.create_chat_session(collection_id)

    # TODO for llm=... circle through different LLMs: session.query(p, llm=llm)
    for q in prompts_str:
        try:
            print(f"Q: {q}", flush=True)
            with client.connect(chat_session_id) as session:
                actual_output = session.query(q).content
                print(f"A: {actual_output}", flush=True)
        except Exception as e:
            print(
                f"Failed to get answer for prompt '{q}' with: {e}"
                f"\n{traceback.format_exc()}"
            )
            actual_output = ""

        prompts.add_input(
            i=q,
            actual_output=actual_output,
        )

    # save inputs to JSon file
    prompts_json_path = tmp_path / "inputs.json"
    with open(prompts_json_path, "w") as f:
        json.dump(prompts.to_dict(), f, indent=4)


@pytest.mark.skip(reason="Test to run ALL evaluators")
@pytest.mark.skipif(not test_utils.is_mojo_supported(), reason="MOJO is not supported")
@pytest.mark.parametrize(
    "dataset_path,mojo_path,target_col,use_explainable_model",
    [
        # CC numeric dataset & MOJO
        (
            "data/predictive/creditcard.csv",
            "data/predictive/models/creditcard-binomial.mojo",
            "default payment next month",
            True,
        ),
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_talk2report_all_evaluators(
    tmpdir, dataset_path, mojo_path, target_col, use_explainable_model
):
    """Run all DeepEval-based evaluators using keyword."""

    import daimojo

    #
    # GIVEN
    #
    # dataset
    dataset_path = test_utils.find_locally(dataset_path)
    # model
    mojo_path = test_utils.find_locally(mojo_path)
    model = daimojo.model(mojo_path)
    # container: production / default

    #
    # WHEN
    #
    evaluation = evaluate.run_evaluation(
        dataset=dataset_path,
        models=[model],
        evaluator_keywords=[evaluate.KEYWORD_LLM],
        results_location=tmpdir,
        log_level=loggers.DEBUG,
    )

    #
    # THEN
    #
    print(f"\n{evaluation}")
    # find failed evaluators
    assert evaluation
    assert evaluation.result.explainers
    assert len(evaluation.result.explainers) > 5
    failed_evaluators = evaluation.get_failed_evaluator_ids()
    assert not failed_evaluators, f"Failed evaluators: {failed_evaluators}"


@pytest.mark.parametrize(
    "status_int,status_enum",
    [
        (-4, commons.ExplainerJobStatus.SYNCING),
        (-3, commons.ExplainerJobStatus.SCHEDULED),
        (-2, commons.ExplainerJobStatus.UNKNOWN),
        (-1, commons.ExplainerJobStatus.IN_PROGRESS),
        (-1, commons.ExplainerJobStatus.RUNNING),
        (0, commons.ExplainerJobStatus.FINISHED),
        (0, commons.ExplainerJobStatus.SUCCESS),
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_status_code_enum(status_int, status_enum):
    assert status_enum == commons.ExplainerJobStatus.from_int(status_int)


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_build_llm_dataset():
    #
    # GIVEN
    #
    dataset = datasets.LlmDataset()

    #
    # WHEN
    #
    dataset.add_input(
        i="What is the target column of the model?",
        actual_output="default payment next month",
    )

    #
    # THEN
    #
    assert dataset
    assert len(dataset.inputs) == 1
    assert dataset.inputs[0].i == "What is the target column of the model?"
    assert dataset.inputs[0].actual_output == "default payment next month"


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
