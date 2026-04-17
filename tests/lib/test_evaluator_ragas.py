# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import math
import pathlib
import uuid

import pytest

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import evaluate
from h2o_sonar import loggers
from h2o_sonar.evaluators import rag_answer_correctness_evaluator as ac_e
from h2o_sonar.evaluators import rag_answer_relevancy_evaluator as ar_e
from h2o_sonar.evaluators import rag_answer_similarity_evaluator as as_e
from h2o_sonar.evaluators import rag_context_precision_evaluator as cp_e
from h2o_sonar.evaluators import rag_context_recall_evaluator as crc_e
from h2o_sonar.evaluators import rag_context_relevancy_evaluator as cr_e
from h2o_sonar.evaluators import rag_faithfulness_evaluator as f_e
from h2o_sonar.evaluators import rag_ragas_evaluator as ragas_e
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets as d6s
from h2o_sonar.lib.api import evaluators as e8s
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import judges
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences as p10s
from h2o_sonar.lib.integrations import genai
from h2o_sonar.lib.integrations import ragas_adapter
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative
from tests.lib import then_eval


def _given_huggingface_dataset_for_ragas():
    """Create a HuggingFace dataset from a LLM dataset."""
    from datasets import Dataset

    # H2O Sonar LLM dataset
    llm_dataset = d6s.LlmDataset.load_from_json(
        test_utils.find_locally(
            # "data/generative/kaggle_llm_science_exam_dataset_h2o_small.json"  # > rows
            "data/generative/ragas-custom-judge_dataset_1p.json"  # 2x rows
        )
    )
    llm_dt_dict = llm_dataset.to_datatable_dict()

    #
    # HuggingFace dataset for RAGAS library
    #

    # normalize CONTEXTS
    contexts = []
    for c in llm_dt_dict[d6s.LlmDataset.KEY_CONTEXT]:
        if isinstance(c, list):
            if not c or (len(c) == 1 and c[0] == []):
                # prevents empty context like [[]], but []
                contexts.append(["NO CONTEXT"])
            else:
                contexts.append(c)
        else:
            # string > [string]
            contexts.append([c])

    # normalize GROUND TRUTHS
    ground_truths = [[c] for c in llm_dt_dict[d6s.LlmDataset.KEY_EXPECTED_OUTPUT]]

    # create HuggingFace dataset
    ragas_hf_dict = {
        ragas_e.RagasEvaluator.KEY_QUESTION: llm_dt_dict[d6s.LlmDataset.KEY_INPUT],
        ragas_e.RagasEvaluator.KEY_GROUND_TRUTHS: ground_truths,
        ragas_e.RagasEvaluator.KEY_ANSWER: llm_dt_dict[
            d6s.LlmDataset.KEY_ACTUAL_OUTPUT
        ],
        ragas_e.RagasEvaluator.KEY_CONTEXTS: contexts,
    }

    hf_dataset = Dataset.from_dict(ragas_hf_dict)

    return hf_dataset


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.skip(reason="Test CUSTOM LLM judge instead of OpenAI models @ ragas API")
@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({"ragas"}),
    reason="Ragas Python package is not installed",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_custom_llm_and_embeddings_via_evaluate(tmp_path):
    #
    # GIVEN custom judge ~ H2O Sonar configured
    #
    custom_judge = judges.get_evaluation_judge_for_connection(
        connection=test_utils.health.get_h2ogpt(),
        # llama2 70b ... 25" > score == 0.5
        # llm_model_name="h2oai/h2ogpt-4096-llama2-70b-chat",
        # llama2 13b ... 6' (slow) > score == NaN
        # llm_model_name="h2oai/h2ogpt-4096-llama2-13b-chat",
        # GPT 3.5 ... 6" (faster) > score == 1.0
        llm_model_name="gpt-3.5-turbo-0613",
        # GPT 4 ... 40" (faster) > score == 0.33
        # llm_model_name="gpt-4o",
        judge_type=h2o_sonar_config.EvaluationJudgeType.h2ogpt.name,
    )

    #
    # GIVEN dataset
    #
    hf_dataset = _given_huggingface_dataset_for_ragas()

    #
    # GIVEN custom LLM
    #
    custom_llm_for_ragas = ragas_adapter.get_ragas_to_sonar_llm_adapter(custom_judge)

    #
    # WHEN
    #
    import ragas
    from ragas.metrics import context_precision  # fast metrics

    results = ragas.evaluate(
        dataset=hf_dataset,
        metrics=[context_precision],
        llm=custom_llm_for_ragas,
        embeddings=None,
    )

    #
    # THEN
    #
    print(results)
    assert results


@pytest.mark.skip(reason="Test lab builder")
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_build_test_lab(tmp_path):
    #
    # GIVEN
    #

    # h2oGPTe server
    h2ogpte_connection = test_utils.health.get_h2ogpte()

    # LLM models to validate: all / non-OpenAI / OpenAI only
    llm_model_names = genai.H2oGpteRagClient(h2ogpte_connection).list_llm_model_names()
    # llm_model_names = [given_generative.H2OGPTE_JUDGE_LLM_MODEL_NAME]
    # filtering in/out non-OpenAI models
    # llm_model_names = [m for m in llm_model_names if "gpt-4-" in m]

    test_suite = testing.RagTestSuiteConfig.load_from_json(
        # "data/generative/talk2report_prompts_test_suite.json"
        # 26s (parallel)
        # "data/generative/h2ogpte_benchmark_test_suite_min.json"
        # "data/generative/h2ogpte_benchmark_test_suite_top.json"
        # 2 tests, 3 models, 6 shards > 3'20s (parallel) / 18'+ (sequential)
        # "data/generative/h2ogpte_benchmark_test_suite_demo.json"
        # 2h36'47s (parallel) / 6h+ (sequential)
        test_utils.find_locally(
            "data/generative/h2ogpte_benchmark_test_suite_openai.json"
        )
        # 1x prompt 1x test
        # "data/generative/ragas-custom-judge_test_suite_1p.json"
    )

    # OPTIONAL DESCOPE: faster / smaller test (debugging)
    # llm_model_names = llm_model_names[:5]
    # test_suite.test_cases = test_suite.test_cases[:10]

    print(
        f"Building test lab for {len(llm_model_names)} LLM models and "
        f"{len(test_suite.test_cfgs)} tests:\n"
        f"  - expecting {len(llm_model_names) * len(test_suite.test_cfgs)} SHARDS"
    )

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

    # test lab: DEPLOY the h2oGPTe server (docs sync: S3 > filesystem cache > h2oGPT2)
    test_lab.build()

    # test lab: complete dataset w/ ACTUAL values from the h2oGPTe server (answers, ...)
    test_lab.complete_dataset(
        complete_context=15,
        save_as_you_go=tmp_path / "wip_testlab.json",
        parallelize=-testing.TestLab.PARALLEL_RUN,
        purge_workdir=False,
    )

    #
    # THEN
    #

    # keep (as collections creation takes time) or purge
    # test_lab.purge()

    # backup fully resolved dataset
    test_lab.save_as_json(tmp_path / "test_lab_with_actual_values.json")


@pytest.mark.skip(reason="Test lab builder from shards")
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_build_test_lab_from_shards(tmp_path):
    #
    # GIVEN
    #

    # h2oGPTe server
    h2ogpte_connection = test_utils.health.get_h2ogpte()

    # SKIP: LLM models to validate

    test_suite = testing.RagTestSuiteConfig.load_from_json(
        test_utils.find_locally(
            "data/generative/h2ogpte_benchmark_test_suite_openai.json"
        )
    )

    #
    # WHEN
    #

    # test lab (RAG product)
    test_lab = testing.RagTestLab.from_rag_test_suite(
        rag_connection=h2ogpte_connection,
        rag_test_suite=test_suite,
        rag_model_type=models.ExplainableModelType.h2ogpte,
        llm_model_names=[],  # LLM models will be completed from shards
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )

    # test lab: complete from shards
    test_lab.complete_from_shards(
        execution_dir_path=(
            "/home/user/h/mli/git/h2o-sonar/data/generative/rag_docs"
            "/execution_1702937491.6930892"
        )
    )

    #
    # THEN
    #

    # backup fully resolved dataset
    test_lab.save_as_json(tmp_path / "test_lab_with_actual_values.json")


@pytest.mark.parametrize(
    "values,harmonic_mean",
    [
        ([1, 4, 4], 2),
        ([2, 5, 7, 9], 4.1930116472545755),
        ([5, 6, 7, 8], 6.303939962476548),
        ([0.94439, 0.19999, 0.57999, 1.0], 0.45540958172596185),
        ([0.9035336428568099, 0.0, 1.0, 1.0], 0),
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_harmonic_mean(values, harmonic_mean):
    assert commons.harmonic_mean(values) == harmonic_mean


@pytest.mark.skip(reason="RAGAS metric calculation")
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_ragas_injection():
    scores_dict = scores_dict = {
        "answer_relevancy": [
            0.94439,
        ],
        "context_precision": [
            0.19999,
        ],
        "faithfulness": [
            0.57999,
        ],
        "context_recall": [
            1.0,
        ],
    }

    ragas_component = {ragas_e.RagasEvaluator.METRIC_RAGAS: []}
    for i in range(len(scores_dict[ragas_e.RagasEvaluator.METRIC_ANSWER_RELEVANCY])):
        ragas_inputs = []
        for m in scores_dict:
            ragas_inputs.append(scores_dict[m][i])
        ragas_metric = commons.harmonic_mean(ragas_inputs)
        ragas_component[ragas_e.RagasEvaluator.METRIC_RAGAS].append(ragas_metric)

    scores_dict[ragas_e.RagasEvaluator.METRIC_RAGAS] = ragas_component[
        ragas_e.RagasEvaluator.METRIC_RAGAS
    ]

    print(json.dumps(scores_dict, indent=4))


@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_ragas_metric():
    """RAGAS metric calculation."""
    #
    # GIVEN
    #
    component_metrics = {
        "answer_relevancy": [
            0.9391049463094591,
            0.859587688739864,
            0.9383272750726624,
            0.9164858620403605,
            0.9303184114370175,
            0.9303184114370175,
            0.8886642180215701,
            0.8886019531845094,
            0.8502063010724319,
            0.8502063010724319,
            0.9116515389876034,
            0.911638641155494,
            0.8473023170877894,
            0.8472943162374582,
            0.8387382943155971,
            0.845273123040629,
            0.9067896146129723,
            0.906761057941294,
            0.9031601318495,
            0.9037456093362737,
        ],
        "context_precision": [
            0.9999999999,
            0.9999999999,
            0.9999999999,
            0.9999999999,
            0.9999999999,
            0.9999999999,
            0.9999999999,
            0.9999999999,
            0.9999999999,
            0.9999999999,
            0.9999999999,
            0.9999999999,
            0.9999999999,
            0.9999999999,
            0.9999999999,
            0.9999999999,
            0.9999999999,
            0.9999999999,
            0.9999999999,
            0.9999999999,
        ],
        "faithfulness": [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.75,
            0.75,
            0.5,
            0.5,
            1.0,
            1.0,
            1.0,
            1.0,
            0.75,
            1.0,
            1.0,
            1.0,
            0.6666666666666667,
            1.0,
        ],
        "context_recall": [
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
        ],
    }

    #
    # WHEN
    #
    # calculate the harmonic mean for all rows
    ragas_component = {"ragas": []}
    for i in range(len(component_metrics["answer_relevancy"])):
        ragas_inputs = []
        for m in component_metrics:
            ragas_inputs.append(component_metrics[m][i])
        ragas_metric = commons.harmonic_mean(ragas_inputs)
        ragas_component["ragas"].append(ragas_metric)

    component_metrics["ragas"] = ragas_component["ragas"]

    #
    # THEN
    #
    print(json.dumps(ragas_component, indent=4))
    print(json.dumps(component_metrics, indent=4))


@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_load_dataset():
    #
    # GIVEN
    #
    prompts_path = test_utils.find_locally(
        "data/generative/talk2report_prompts_dataset.json"
    )

    #
    # WHEN
    #
    llm_dataset = d6s.LlmDataset.load_from_json(prompts_path)

    #
    # THEN
    assert llm_dataset
    print(json.dumps(llm_dataset.to_dict(), indent=4))
    assert len(llm_dataset.to_dict().get(d6s.LlmDataset.KEY_INPUTS, [])) > 1


@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({"datasets"}),
    reason="HuggingFace 'datasets' Python package is not installed",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_genai_2_hf_dataset():
    evaluator = ragas_e.RagasEvaluator

    #
    # GIVEN
    #
    prompts_path = test_utils.find_locally(
        "data/generative/talk2report_prompts_dataset.json"
    )
    prompts_dataset = d6s.LlmDataset.load_from_json(prompts_path)
    prompts_dict = prompts_dataset.to_datatable_dict()

    #
    # WHEN
    #

    from datasets import Dataset

    # LLM dataset > RAGAS @ HF dataset conversion
    ragas_hf_dict = {
        evaluator.KEY_QUESTION: prompts_dict[d6s.LlmDataset.KEY_INPUT],
        evaluator.KEY_GROUND_TRUTHS: prompts_dict[d6s.LlmDataset.KEY_EXPECTED_OUTPUT],
        evaluator.KEY_ANSWER: prompts_dict[d6s.LlmDataset.KEY_ACTUAL_OUTPUT],
        evaluator.KEY_CONTEXTS: prompts_dict[d6s.LlmDataset.KEY_CONTEXT],
    }

    hf_dataset = Dataset.from_dict(ragas_hf_dict)

    print(hf_dataset.features)
    print(hf_dataset)


@pytest.mark.skip(reason="RAGAS library sanity check")
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_ragas_lib():
    #
    # GIVEN
    #
    from datasets import load_dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy
    from ragas.metrics import context_precision
    from ragas.metrics import context_recall
    from ragas.metrics import faithfulness

    # https://huggingface.co/datasets/explodinggradients/fiqa
    # https://huggingface.co/datasets/explodinggradients/fiqa/viewer/ragas_eval
    fiqa_eval = load_dataset("explodinggradients/fiqa", "ragas_eval")

    #
    # WHEN
    #

    result = evaluate(
        fiqa_eval["baseline"].select(range(3)),  # selecting only 3
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
        ],
    )

    #
    # THEN
    #
    print(result)
    assert result


def _given_judge_mixtral() -> tuple[str, h2o_sonar_config.ConnectionConfig, str]:
    return (
        h2o_sonar_config.EvaluationJudgeType.h2ogpte_llm.name,
        test_utils.health.get_h2ogpte(),
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
    )


def _given_judge_capybara() -> tuple[str, h2o_sonar_config.ConnectionConfig, str]:
    return (
        h2o_sonar_config.EvaluationJudgeType.h2ogpte_llm.name,
        test_utils.health.get_h2ogpte(),
        "NousResearch/Nous-Capybara-34B",
    )


def _given_judge_llama2_70b() -> tuple[str, h2o_sonar_config.ConnectionConfig, str]:
    return (
        h2o_sonar_config.EvaluationJudgeType.h2ogpte_llm.name,
        test_utils.health.get_h2ogpte(),
        "h2oai/h2ogpt-4096-llama2-70b-chat",
    )


def _given_judge_gpt35() -> tuple[str, h2o_sonar_config.ConnectionConfig, str]:
    return (
        h2o_sonar_config.EvaluationJudgeType.h2ogpte_llm.name,
        test_utils.health.get_h2ogpte(),
        given_generative.H2OGPTE_JUDGE_LLM_MODEL_NAME,
    )


def _test_load_lab(
    tmp_path, h2ogpte_connection, test_lab_path, custom_judge, evaluator_classes
):
    try:
        #
        # GIVEN
        #
        evaluator_id = evaluator_classes[0].evaluator_id()

        evaluators = [
            commons.EvaluatorToRun(
                evaluator_id=clz.evaluator_id(),
                params={
                    clz.PARAM_SAVE_LLM_RESULT: True,
                    clz.PARAM_METRIC_THRESHOLD: 0.3,
                },
            )
            for clz in evaluator_classes
        ]

        # test lab (load cfg w/ actual values - build/chat not needed)
        test_lab = testing.RagTestLab.load_from_json(
            llm_host_connection=h2ogpte_connection,
            file_path=test_lab_path,
            docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
        )

        # CUSTOM judge - OPTION "force_custom_judge_with_config":
        # - add judge configuration to H2O Sonar config
        # - use evaluator's parameters to request the use of 3rd party judge
        # CUSTOM judge - OPTION "use_custom_judge_param":
        # - add judge configuration to H2O Sonar config
        # - force custom judge using H2O Sonar config
        if custom_judge:
            # (judge_type, judge_connection, judge_llm_model) = _given_judge_mixtral()
            # (judge_type, judge_connection, judge_llm_model) = _given_judge_capybara()
            # (judge_type, judge_connection, judge_llm_model)= _given_judge_llama2_70b()
            (judge_type, judge_connection, judge_llm_model) = _given_judge_gpt35()

            print(f"TEST will use custom judge LLM model: {judge_llm_model}")
            h2o_sonar_config.config.add_connection(judge_connection)
            eval_judge_cfg = h2o_sonar_config.config.add_evaluation_judge(
                h2o_sonar_config.EvaluationJudgeConfig(
                    name="CUSTOM judge EvalStudio TEST",
                    description="Custom LLM judge to be used by evaluators.",
                    judge_type=judge_type,
                    connection=judge_connection,
                    llm_model_name=judge_llm_model,
                )
            )

            if custom_judge == "use_custom_judge_param":
                evaluators = [
                    commons.EvaluatorToRun(
                        evaluator_id=clz.evaluator_id(),
                        params={
                            clz.PARAM_EVAL_JUDGE_CFG_KEY: eval_judge_cfg.key,
                        },
                    )
                    for clz in evaluator_classes
                ]
            elif custom_judge == "force_custom_judge_with_config":
                # request 1st judge from the H2O Sonar config
                h2o_sonar_config.config.force_eval_judge = "true"
            else:
                raise ValueError(f"Unknown custom_judge test option: {custom_judge}")

        #
        # WHEN
        #

        evaluation = evaluate.run_evaluation(
            # dataset w/ prompts, constraints and model keys
            dataset=test_lab.dataset,
            # models to be evaluated / compared to get leaderboard
            models=list(test_lab.evaluated_models.values()),
            # evaluators
            evaluators=evaluators,
            # where to save the report
            results_location=tmp_path,
        )

        #
        # THEN
        #

        print(f"{evaluation}")
        assert not evaluation.get_failed_evaluator_ids()
        assert evaluation.get_successful_evaluator_ids()

        # assert result
        result = evaluation.get_evaluator_result(evaluator_id)
        print(result)

        then_eval.then_leaderboard_json(evaluation, evaluator_id)

        assert result
        # get explanation file path
        if evaluator_id in evaluators:
            path = evaluation.get_explanation_file_path(
                evaluator_id=evaluator_id,
                explanation_type=(
                    e10s.LlmHeatmapLeaderboardExplanation.explanation_type()
                ),
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

        print(
            f"Explanations:\n"
            f"  HTML: file://{evaluation.result.get_html_report_location()}\n"
        )
    finally:
        h2o_sonar_config.config.force_eval_judge = (
            h2o_sonar_config.H2oSonarConfig.CFG_FORCE_EVAL_JUDGE.default_value
        )


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.skip(reason="RAGAS based evaluators covered by CI suite (OpenAI cost)")
@pytest.mark.parametrize(
    "test_lab_path,custom_judge,evaluator_classes",
    [
        # default OpenAI judge
        (
            #
            # RAG test labs:
            #
            "data/generative/h2ogpte_benchmark_test_lab_small.json",
            # "data/generative/h2ogpte_benchmark_test_lab_top.json",  # NICE HTML report
            #
            # LLM test labs:
            #
            # LLM compatible evaluators ~ w/o contexts:
            # "data/generative/eval_llm/h2ogpte_benchmark_test_lab_micro.json"
            #
            # CUSTOM judge
            #
            None,
            # EVALUATORS: AS fastest (3s @ cosine)
            #
            [
                ac_e.AnswerCorrectnessEvaluator,
                as_e.AnswerSemanticSimilarityEvaluator,
                ar_e.AnswerRelevancyEvaluator,
                cr_e.ContextRelevancyEvaluator,
                cp_e.ContextPrecisionEvaluator,
                crc_e.ContextRecallEvaluator,
                f_e.FaithfulnessEvaluator,
                ragas_e.RagasEvaluator,
            ],
        ),
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_load_lab(
    tmp_path, h2ogpte_connection_fixture, test_lab_path, custom_judge, evaluator_classes
):
    _test_load_lab(
        tmp_path=tmp_path,
        h2ogpte_connection=h2ogpte_connection_fixture,
        test_lab_path=test_lab_path,
        custom_judge=custom_judge,
        evaluator_classes=evaluator_classes,
    )


# Custom JUDGE testing SUMMARY:
#
# Mixtral-8x7B-Instruct-v0.1 -> WIP: slow & hanging
# - #1 in KGM benchmark of privacy safe LLMs
# - performance
#   - pretty slow
#   - CP @ 1p/1m: 5'
#
# NousResearch/Nous-Capybara-34B -> WIP: model HANGS (h2oGPTe)
# - performance
#   - CR @ 1p/1m: ?
# - correctness:
#   - ALL evaluators ?
#
# h2ogpt-4096-llama2-70b-chat -> CONCLUSION: CANNOT BE USED
# - #5 in KGM benchmark of privacy safe LLMs
# - performance
#   - pretty fast + much faster than 13b on our infrastructure
#   - CR  @ 1p/1m: 30"
#   - ALL @ 1p/1m: 2'31"
# - BLOCKERS:
#   - Context precision
#     - llama cannot handle CP prompts - NaN - evaluator fails to produce a score
#   - Faithfulness
#     - llama cannot handle CP prompts - NaN - evaluator fails to produce a score
#   - RAGAs
#     - Faithfulness and Context Precision are required for RAGAs - no score
#
# gpt-3.5-turbo-0613 -> WORKS (commercial model)
# - commercial model, used to verify that the custom judge is working
# - provides baseline performance for the on-premise/open custom judges.
# - performance
#   - fast: CR @ 1p/1m: 20"
# - correctness:
#   - ALL evaluators OK
#
# Additional note:
#
# - see h2o_sonar.lib.api.judges for more details
#
@pytest.mark.parametrize(
    "test_lab_path,custom_judge,evaluator_classes",
    [
        # custom judge: evaluator argument
        # (
        #     #     "data/generative/ragas-custom-judge_test_lab_1p.json",
        #     "use_custom_judge_param",
        #     [
        #         crc_e.ContextRecallEvaluator,
        #     ],
        # ),
        # custom judge: forced from H2O Sonar config
        # - tested WITHOUT OpenAI API key!
        (
            "data/generative/ragas-custom-judge_test_lab_1p.json",
            "force_custom_judge_with_config",
            [
                crc_e.ContextRecallEvaluator,
                ac_e.AnswerCorrectnessEvaluator,
                as_e.AnswerSemanticSimilarityEvaluator,
                ar_e.AnswerRelevancyEvaluator,
                cr_e.ContextRelevancyEvaluator,  # fast
                cp_e.ContextPrecisionEvaluator,
                f_e.FaithfulnessEvaluator,
                ragas_e.RagasEvaluator,
            ],
        ),
    ],
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_load_lab_judges(
    tmp_path, h2ogpte_connection_fixture, test_lab_path, custom_judge, evaluator_classes
):
    _test_load_lab(
        tmp_path=tmp_path,
        h2ogpte_connection=h2ogpte_connection_fixture,
        test_lab_path=test_lab_path,
        custom_judge=custom_judge,
        evaluator_classes=evaluator_classes,
    )


@pytest.mark.skip(reason="RAGAS based evaluators covered by CI suite (OpenAI cost)")
@pytest.mark.parametrize(
    "test_lab_path,custom_judge,evaluator_classes",
    [
        # NaN bug
        (
            "data/generative/bugs/bug-ragas-nan-2025-04-10-test-lab.json",
            None,
            [
                # BUG: 1 or more answer relevancy scores was NaN
                ar_e.AnswerRelevancyEvaluator,
                # BUG: therefore RAGAs value was NaN
                # ragas_e.RagasEvaluator,
            ],
        ),
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_load_lab_nan(
    tmp_path, h2ogpte_connection_fixture, test_lab_path, custom_judge, evaluator_classes
):
    _test_load_lab(
        tmp_path=tmp_path,
        h2ogpte_connection=h2ogpte_connection_fixture,
        test_lab_path=test_lab_path,
        custom_judge=custom_judge,
        evaluator_classes=evaluator_classes,
    )


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.parametrize(
    "evaluator,nan_tolerance",
    [
        (ragas_e.RagasEvaluator(), 0.0),
        (ragas_e.RagasEvaluator(), 0.1),
        (ragas_e.RagasEvaluator(), 0.5),
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_nan_tolerance(
    tmp_path: pathlib.Path, evaluator: e8s.Evaluator, nan_tolerance: float
):
    """Test calculation of RAGAs family metrics evan if the result metric scores
    contain NaN values.

    """
    #
    # GIVEN
    #
    evaluator.persistence = p10s.ExplainerPersistence(
        data_dir=str(tmp_path),
        username="tester",
        explainer_id=evaluator.explainer_id(),
        explainer_job_key=f"{uuid.uuid4()}",
        mli_key=f"{uuid.uuid4()}",
        store_persistence=p10s.FilesystemPersistence(
            base_path=tmp_path, logger=loggers.SonarPrintLogger()
        ),
    )
    evaluator.persistence.make_explainer_sandbox()
    connection = test_utils.health.get_h2ogpte()
    # NaN bug
    # v ... authentic dataset (privacy safe data run @ their deployment)
    llm_eval_results_path = test_utils.find_locally(
        "data/generative/bugs/bug-ragas-nan-2025-04-10-results-dataset.json"
    )
    print(f"Loading repro dataset from: {llm_eval_results_path}")
    eval_results = d6s.LlmEvalResults.load_from_json(llm_eval_results_path)
    # normalization - remove non-metric values
    garbage_keys = ["key", "test_case_key", "metrics_meta"]
    for r in eval_results.results:
        for k in garbage_keys:
            if k in r.metrics:
                del r.metrics[k]
        for m in r.metrics:
            if r.metrics[m] == "NaN":
                r.metrics[m] = float("nan")
    # map: model key -> evaluated model
    key_2_evaluated_model = {}
    with open(llm_eval_results_path) as f:
        llm_eval_results_json = json.load(f)
    if llm_eval_results_json.get("models"):
        for m_json in llm_eval_results_json["models"]:
            m = models.ExplainableRagModel.from_dict(
                as_dict=m_json, connection=connection
            )
            key_2_evaluated_model[m.key] = m

    #
    # WHEN
    #

    heatmap_explanation = e10s.LlmHeatmapLeaderboardExplanation.from_eval_results(
        evaluator=evaluator,
        eval_results=eval_results,
        metrics_meta=evaluator._metrics_meta,
        key_2_evaluated_model=key_2_evaluated_model,
        llm_host=commons.LlmModelHostType.RAG,
        display_name="REPRO heatmap leaderboard",
        display_category=e10s.GlobalSummaryFeatImpExplanation.DISPLAY_CAT_LLM,
        nan_tolerance=nan_tolerance,
        logger=loggers.SonarPrintLogger(),
    )
    heatmap_explanation.add_json_format(threshold=0.75)
    heatmap_explanation.add_markdown_format(
        sort_by_metric_id=evaluator._metrics_meta.get_primary_metric().key
    )
    heatmap_explanation.add_evalstudio_markdown_format(
        sort_by_metric_id=evaluator._metrics_meta.get_primary_metric().key
    )

    # HTML explanation
    html_explanation = e10s.GlobalHtmlFragmentExplanation(
        evaluator=evaluator,
        display_name="LLM heatmap leaderboard as HTML",
        display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
    )
    html_explanation.add_html_format(
        str(
            heatmap_explanation.as_html(
                sort_by_metric_id=evaluator._metrics_meta.get_primary_metric().key,
            )
        ),
    )

    #
    # THEN
    #

    # ASSERT heatmap leaderboard JSon
    subdir_name = "application_json"
    t_dir = test_utils.find_subdir(start_dir=tmp_path, subdir_name=subdir_name)
    if not t_dir:
        raise FileNotFoundError(
            f"THEN test section cannot find {subdir_name}/ directory in {tmp_path}"
        )
    ragas_leaderboard_path = t_dir / "leaderboard_5.json"
    with open(ragas_leaderboard_path) as f:
        ragas_leaderboard_dict = json.load(f)
    assert "ragas" in ragas_leaderboard_dict["data"]["claude-3-7-sonnet-20250219"]
    if nan_tolerance in [0.0, 0.1]:
        assert math.isnan(
            float(ragas_leaderboard_dict["data"]["claude-3-7-sonnet-20250219"]["ragas"])
        )
    else:
        assert isinstance(
            ragas_leaderboard_dict["data"]["claude-3-7-sonnet-20250219"]["ragas"], float
        )
    # ASSERT EvalStudio markdown
    subdir_name = "application_vnd_h2oai_evalstudio_leaderboard_markdown"
    t_dir = test_utils.find_subdir(start_dir=tmp_path, subdir_name=subdir_name)
    if not t_dir:
        raise FileNotFoundError(
            f"THEN test section cannot find {subdir_name}/ directory in {tmp_path}"
        )
    md_es_path = t_dir / "explanation.md"
    with open(md_es_path) as f:
        md_es_dict = f.read()
    if nan_tolerance in [0.0, 0.1]:
        assert "nan" in md_es_dict
    else:
        assert "nan" not in md_es_dict

    # ASSERT text Markdown
    subdir_name = "text_markdown"
    t_dir = test_utils.find_subdir(start_dir=tmp_path, subdir_name=subdir_name)
    if not t_dir:
        raise FileNotFoundError(
            f"THEN test section cannot find {subdir_name}/ directory in {tmp_path}"
        )
    md_es_path = t_dir / "explanation.md"
    with open(md_es_path) as f:
        md_es_dict = f.read()
    if nan_tolerance in [0.0, 0.1]:
        assert "nan" in md_es_dict
    else:
        assert "nan" not in md_es_dict

    # ASSERT HTML
    subdir_name = "text_html"
    t_dir = test_utils.find_subdir(start_dir=tmp_path, subdir_name=subdir_name)
    if not t_dir:
        raise FileNotFoundError(
            f"THEN test section cannot find {subdir_name}/ directory in {tmp_path}"
        )
    html_path = t_dir / "explanation.html"
    with open(html_path) as f:
        html_dict = f.read()
    ragas_metric_score = "0.62169"
    if nan_tolerance in [0.0, 0.1]:
        assert ragas_metric_score not in html_dict
    else:
        assert ragas_metric_score in html_dict

    # ASSERT problems
    print(f"Evaluator Problems[{len(evaluator.explain_problems())}]...")
    for p in evaluator.explain_problems():
        print(f"  Problem: {p}")
    if nan_tolerance in [0.1, 0.5]:
        assert len(evaluator.explain_problems()) == 1


@pytest.mark.skip(reason="RAGAS evaluator is long running and requires OpenAI API key")
@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({"ragas"}),
    reason="Ragas Python package is not installed",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_evaluator(tmp_path):
    #
    # GIVEN
    #

    # RAG or LLM
    is_rag = True

    # h2oGPTe server
    h2ogpte_connection = test_utils.health.get_h2ogpte()
    llm_model_names = genai.H2oGpteRagClient(h2ogpte_connection).list_llm_model_names()

    test_suite = testing.RagTestSuiteConfig.load_from_json(
        test_utils.find_locally(
            # "data/generative/ci_rag_test_suite.json"
            # "data/generative/ci_llm_test_suite.json"
            "data/generative/talk2report_prompts_test_suite_small.json"
        )
    )

    # OPTIONAL DESCOPE: faster / smaller test (debugging)
    llm_model_names = llm_model_names[:3]
    # test_suite.test_cases = test_suite.test_cases[:3]

    # test lab (RAG product)
    if is_rag:
        test_lab = testing.RagTestLab.from_rag_test_suite(
            rag_connection=h2ogpte_connection,
            rag_test_suite=test_suite,
            rag_model_type=models.ExplainableModelType.h2ogpte,
            llm_model_names=llm_model_names,
            docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
        )
    else:
        test_lab = testing.RagTestLab.from_llm_test_suite(
            llm_host_connection=h2ogpte_connection,
            llm_test_suite=test_suite,
            llm_model_type=models.ExplainableModelType.h2ogpte_llm,
            llm_model_names=llm_model_names,
            work_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
        )
    # test lab: DEPLOY the h2oGPTe (docs sync: S3 > filesystem cache > h2oGPT2)
    test_lab.build()
    # test lab: complete dataset w/ ACTUAL values from the the h2oGPTe server
    test_lab.complete_dataset(
        complete_context=True,
        save_as_you_go=tmp_path / "wip_testlab.json",
        parallelize=testing.TestLab.PARALLEL_RUN,
    )
    # backup fully resolved dataset
    test_lab.save_as_json(tmp_path / "test_lab.json")

    #
    # WHEN
    #

    evaluator = ragas_e.RagasEvaluator

    assert evaluate.describe_evaluator(evaluator)

    evaluation = evaluate.run_evaluation(
        dataset=test_lab.dataset,
        models=test_lab.evaluated_models.values(),
        evaluators=[
            commons.EvaluatorToRun(
                evaluator_id=evaluator.evaluator_id(),
                params=None,
            )
        ],
        results_location=tmp_path,
        log_level=loggers.DEBUG,
    )

    #
    # THEN
    #

    print(f"Evaluation:\n{evaluation}")

    assert evaluation
    assert not evaluation.is_evaluator_failed()


@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({"ragas"}),
    reason="Package 'ragas' is not installed",
)
@pytest.mark.skipif(
    not test_utils.health.is_openai(),
    reason="OpenAI API key is not set or OpenAI package is not installed",
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_bug_861_nan(tmp_path: pathlib.Path, h2ogpte_connection_fixture):
    """Test NaN in the Faithfulness evaluator @ Mistral 7B output where test repro
    is LLM dataset.

    """
    #
    # GIVEN
    #
    llm_dataset_path = test_utils.find_locally(
        "data/generative/bugs/bug_861_dataset.json"
    )
    llm_eval_results = d6s.LlmEvalResults.load_from_json(llm_dataset_path)
    llm_dataset = llm_eval_results.to_llm_dataset()

    foo_model = models.ExplainableRagModel(
        key=llm_dataset.inputs[0].model_key,
        collection_name="Foo RAG collection name",
        collection_id="foo-collection-id",
        name="Foo mistralai/Mixtral-8x7B-Instruct-v0.1",
        connection=h2ogpte_connection_fixture,
        model_type=models.ExplainableModelType.h2ogpte,
        llm_model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
    )

    #
    # WHEN
    #
    evaluator = f_e.FaithfulnessEvaluator

    assert evaluate.describe_evaluator(evaluator)

    evaluation = evaluate.run_evaluation(
        dataset=llm_dataset,
        models=[foo_model],
        evaluators=[evaluator.evaluator_id()],
        results_location=tmp_path,
        log_level=loggers.DEBUG,
    )

    #
    # THEN
    #
    print(f"Evaluation:\n{evaluation}")

    assert evaluation
    assert not evaluation.is_evaluator_failed()

    leaderboard_as_json = e10s.LlmLeaderboardExplanation.get_leaderboard_data_path(
        evaluation,
        evaluator_id=evaluation.result.get_evaluator_jobs()[0].evaluator_id(),
        explanation_format=f5s.LlmLeaderboardJSonFormat.mime,
        metric=f5s.LlmLeaderboardJSonFormat.KEY_ALL_METRICS,
    )

    print(
        f"Explanations:\n"
        f"  HTML          : file://{evaluation.result.get_html_report_location()}\n"
        f"  Lead JSon data: file://{leaderboard_as_json}\n"
    )

    with open(leaderboard_as_json) as f:
        leaderboard_dict = json.load(f)

    print(json.dumps(leaderboard_dict, indent=4))
    assert leaderboard_dict
    model_key = next(iter(leaderboard_dict.keys()))
    print(f"Model key: {model_key}")
    assert model_key
    metric_key = next(iter(leaderboard_dict[model_key].keys()))
    print(f"Metric key: {metric_key}")
    assert metric_key
    metric_value = leaderboard_dict[model_key][metric_key]["faithfulness"]
    print(f"Metric value: {metric_value} ({type(metric_value)})")
    if isinstance(metric_value, float):
        assert not math.isnan(metric_value)
    elif isinstance(metric_value, str):
        assert metric_value == "NaN"
    else:
        raise AssertionError(f"Unexpected metric value type: {type(metric_value)}")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
