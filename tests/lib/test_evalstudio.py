# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import logging
import random
import time
import traceback

import pandas
import pytest
from sklearn import ensemble

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import evaluate
from h2o_sonar import interpret
from h2o_sonar import loggers
from h2o_sonar.evaluators import pii_leakage_evaluator
from h2o_sonar.evaluators import rag_answer_correctness_evaluator
from h2o_sonar.evaluators import rag_answer_relevancy_evaluator
from h2o_sonar.evaluators import rag_answer_similarity_evaluator
from h2o_sonar.evaluators import rag_context_precision_evaluator
from h2o_sonar.evaluators import rag_context_recall_evaluator
from h2o_sonar.evaluators import rag_context_relevancy_evaluator
from h2o_sonar.evaluators import rag_faithfulness_evaluator
from h2o_sonar.evaluators import rag_groundedness_evaluator as g_e
from h2o_sonar.evaluators import rag_hallucination_evaluator
from h2o_sonar.evaluators import rag_ragas_evaluator
from h2o_sonar.evaluators import rag_tokens_presence_evaluator
from h2o_sonar.evaluators import sensitive_data_leakage_evaluator
from h2o_sonar.evaluators import toxicity_evaluator
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import judges
from h2o_sonar.lib.api import models as m4s
from h2o_sonar.lib.api import models as sonar_models
from h2o_sonar.utils import perturbations
from h2o_sonar.utils import preprocessing
from h2o_sonar.utils import progress
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative


# aliases
ragas_evaluator = rag_ragas_evaluator
ctx_precision_evaluator = rag_context_precision_evaluator
pii_evaluator = pii_leakage_evaluator
certs_evaluator = sensitive_data_leakage_evaluator
toxic_evaluator = toxicity_evaluator


@pytest.mark.parametrize(
    "ragas_family_evaluator_cls",
    [
        rag_answer_correctness_evaluator.AnswerCorrectnessEvaluator,
        rag_answer_relevancy_evaluator.AnswerRelevancyEvaluator,  # reported bug
        rag_answer_similarity_evaluator.AnswerSemanticSimilarityEvaluator,
        rag_context_precision_evaluator.ContextPrecisionEvaluator,
        rag_context_recall_evaluator.ContextRecallEvaluator,
        rag_context_relevancy_evaluator.ContextRelevancyEvaluator,
        rag_faithfulness_evaluator.FaithfulnessEvaluator,
        ragas_evaluator.RagasEvaluator,
    ],
)
@pytest.mark.generative
def test_duplicate_primary_metrics(ragas_family_evaluator_cls):
    #
    # GIVEN
    #
    evaluator = ragas_family_evaluator_cls()

    #
    # WHEN
    #
    descriptor_dict = evaluator.as_descriptor().dump()

    #
    # THEN
    #
    print(json.dumps(descriptor_dict, indent=2))
    primary_metrics = []
    for m in descriptor_dict.get("metrics_meta"):
        print(f"  metric: {m.get('key')} -> {m.get('is_primary_metric')}")
        if m.get("is_primary_metric"):
            primary_metrics.append(m.get("key"))
    assert len(primary_metrics) == 1, (
        f"Evaluator {evaluator.evaluator_id()} has multiple primary metrics: "
        f"{primary_metrics}"
    )


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_evaluation_jobs_api(tmp_path):
    """Test evaluation jobs API:

    - IDs of ALL evaluators which user wants to run
    - IDs of incompatible evaluators which were discarded by compatibility checks
    - IDs of compatible evaluators which were scheduled to run
    - ... EVALUATION ...
    - IDs of evaluators which failed
    - IDs of evaluators which succeeded
    - IDs of evaluators which finished

    Test uses RAG vs. LLM for compatibility testing - LLM evaluation is run,
    but also RAG evaluators are scheduled to run (which don't have sufficient data)

    """
    #
    # GIVEN
    #
    given_es_to_run = [
        rag_hallucination_evaluator.RagHallucinationEvaluator,  # RAG
        rag_tokens_presence_evaluator.RagStrStrEvaluator,  # LLM + RAG
        pii_evaluator.PiiLeakageEvaluator,  # LLM + RAG
        toxic_evaluator.ToxicityEvaluator,  # LLM + RAG
    ]
    given_llm_es_ids = []
    given_rag_only_es_ids = []
    for e in given_es_to_run:
        if e._rag and not e._llm:
            given_rag_only_es_ids.append(e.evaluator_id())
        else:
            given_llm_es_ids.append(e.evaluator_id())

    # test lab
    test_lab_path = "data/generative/ci_llm_test_lab_ollama_args.json"
    test_lab = testing.RagTestLab.load_from_json(
        llm_host_connection=test_utils.health.get_h2ogpte(),
        file_path=test_utils.find_locally(test_lab_path),
    )

    #
    # WHEN
    #
    evaluation = evaluate.run_evaluation(
        dataset=test_lab.dataset,
        models=test_lab.evaluated_models.values(),
        evaluators=[e.evaluator_id() for e in given_es_to_run],
        results_location=tmp_path,
    )

    #
    # THEN
    #
    print("THEN")
    assert evaluation

    # IDs
    print(
        f"ALL evaluators [{len(evaluation.get_all_evaluator_ids())}]:"
        f"\n{evaluation.get_all_evaluator_ids()}"
    )
    print(
        f"INCOMPATIBLE evaluators [{len(evaluation.get_incompatible_evaluator_ids())}]:"
        f"\n{evaluation.get_incompatible_evaluator_ids()}"
    )
    print(
        f"SCHEDULED evaluators [{len(evaluation.get_scheduled_explainer_ids())}]:"
        f"\n{evaluation.get_scheduled_explainer_ids()}"
    )
    print(
        f"FINISHED evaluators [{len(evaluation.get_finished_evaluator_ids())}]:"
        f"\n{evaluation.get_finished_explainer_ids()}"
    )
    print(
        f"FAILED evaluators [{len(evaluation.get_failed_evaluator_ids())}]:"
        f"\n{evaluation.get_failed_evaluator_ids()}"
    )
    print(
        f"SUCCEEDED evaluators [{len(evaluation.get_successful_evaluator_ids())}]:"
        f"\n{evaluation.get_successful_evaluator_ids()}"
    )
    assert 4 == len(evaluation.get_all_evaluator_ids())
    assert 1 == len(evaluation.get_incompatible_evaluator_ids())
    assert 3 == len(evaluation.get_scheduled_evaluator_ids())
    assert 3 == len(evaluation.get_finished_evaluator_ids())
    assert 0 == len(evaluation.get_failed_evaluator_ids())
    assert 3 == len(evaluation.get_successful_evaluator_ids())

    # jobs: all finished evaluators must have jobs
    print(f"JOBS [{len(evaluation.result.get_evaluator_jobs())}]:")
    assert 3 == len(evaluation.result.get_evaluator_jobs())
    for j in evaluation.result.get_evaluator_jobs():
        print(f"  {j.explainer_descriptor.id}")
        jobs = evaluation.get_jobs_for_explainer_id(j.explainer_descriptor.id)
        assert evaluation.get_jobs_for_evaluator_id(j.explainer_descriptor.id)
        assert jobs
        assert len(jobs) == 1
        assert j.key == jobs[0].key
    print("Evaluator IDs by status:")
    for s in commons.ExplainerJobStatus:
        es = evaluation.get_evaluator_ids_by_status(s.value)
        print(f"  status: {s} -> {es}")
    print("Evaluation JOBs by status:")
    for s in commons.ExplainerJobStatus:
        js = evaluation.get_evaluator_jobs_by_status(s.value)
        print(f"  status: {s} -> {js}")


@pytest.mark.skip(reason="Test requires h2oGPTe server")
@pytest.mark.parametrize(
    "dataset_type,model_type,models_type,explainers,expected_e_count",
    [
        # TEST: RAG evaluator @ LLM model -> NO evaluator run
        (
            commons.ModelTypeExplanation.RAG,  # dataset
            commons.ModelTypeExplanation.LLM,  # model
            None,  # models
            [
                # incompatible: RAG evaluator @ LLM model (30s)
                rag_context_recall_evaluator.ContextRecallEvaluator().evaluator_id(),
            ],
            0,
        ),
        # TEST: RAG evaluator @ LLM models -> NO evaluator run
        (
            commons.ModelTypeExplanation.RAG,  # dataset
            None,  # model
            commons.ModelTypeExplanation.LLM,  # models
            [
                # incompatible: RAG evaluator @ LLM model (30s)
                rag_context_recall_evaluator.ContextRecallEvaluator().evaluator_id(),
            ],
            0,
        ),
        # TEST: RAG+LLM evaluator @ LLM models -> 1 evaluator run
        (
            commons.ModelTypeExplanation.RAG,  # dataset
            None,  # model
            commons.ModelTypeExplanation.LLM,  # models
            [
                # compatible: RAG+LLM evaluator @ LLM model
                rag_tokens_presence_evaluator.RagStrStrEvaluator().evaluator_id(),
            ],
            1,
        ),
        # TEST: LLM ONLY evaluator @ RAG model -> 1 evaluator run
        # TODO do NOT have LLM ONLY evaluator -> foo evaluator + custom test container
        # (
        #     commons.ModelTypeExplanation.RAG,  # dataset
        #     commons.ModelTypeExplanation.LLM,  # model
        #     None,  # models
        #     [
        #         # compatible: RAG+LLM evaluator @ LLM model
        #         rag_tokens_presence_evaluator.RagStrStrEvaluator().evaluator_id(),
        #     ],
        #     1
        # ),
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_compatibility(
    tmp_path,
    dataset_type: commons.ModelTypeExplanation,
    model_type: commons.ModelTypeExplanation,
    models_type: commons.ModelTypeExplanation,
    explainers: list[str],
    expected_e_count: int,
):
    """Test that Explainer metadata (_llm/_rag) is used for the compatibility check:

    - NEGATIVE tests ~ incompatible evaluators w/ incompatible models ONLY
      (positive are all other regular tests)

    """
    #
    # GIVEN
    #
    h2ogpte_connection = test_utils.health.get_h2ogpte()
    target_col = ""
    iid_dataset_path = ""
    explainable_iid_model = None
    llm_test_lab = testing.RagTestLab.load_from_json(
        llm_host_connection=h2ogpte_connection,
        file_path="data/generative/eval_llm/h2ogpte_benchmark_test_lab_micro.json",
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )
    rag_test_lab = testing.RagTestLab.load_from_json(
        llm_host_connection=h2ogpte_connection,
        file_path="data/generative/h2ogpte_benchmark_test_lab_top.json",
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )
    llm_test_lab.integrity_check()
    rag_test_lab.integrity_check()
    if (
        dataset_type == commons.ModelTypeExplanation.IID
        or model_type == commons.ModelTypeExplanation.IID
        or models_type == commons.ModelTypeExplanation.IID
    ):
        # BINOMIAL dataset & model
        target_col = "default payment next month"
        iid_dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
        df = pandas.read_csv(iid_dataset_path)
        (x, y) = df.drop(target_col, axis=1), df[target_col]
        (x, _, _) = preprocessing.categorical_encoder(x)
        # scikit-learn model
        iid_model = ensemble.GradientBoostingClassifier()
        iid_model.fit(x, y)
        explainable_iid_model = sonar_models.ModelApi().create_model(
            model_src=iid_model,
            target_col=target_col,
            used_features=list(x.columns),
        )

    # dataset

    if dataset_type == commons.ModelTypeExplanation.IID:
        dataset = iid_dataset_path
    elif dataset_type == commons.ModelTypeExplanation.LLM:
        dataset = llm_test_lab.dataset
    elif dataset_type == commons.ModelTypeExplanation.RAG:
        dataset = rag_test_lab.dataset
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # model

    if model_type is None:
        model = None
    elif model_type == commons.ModelTypeExplanation.IID:
        model = explainable_iid_model
    elif model_type == commons.ModelTypeExplanation.LLM:
        if dataset_type == commons.ModelTypeExplanation.LLM:
            model = list(llm_test_lab.evaluated_models.values())[0]
        elif dataset_type == commons.ModelTypeExplanation.RAG:
            # RAG lab patch: RAG model -> LLM model (keep UUIDs)
            llm_base_models = {}
            for m in rag_test_lab.evaluated_models.values():
                llm_base_m = sonar_models.ExplainableLlmModel(
                    connection=m.connection,
                    model_type=sonar_models.ExplainableModelType.h2ogpte_llm,
                    llm_model_name=m.llm_model_name,
                    key=m.key,
                )
                llm_base_models[llm_base_m.name] = llm_base_m

            rag_test_lab.evaluated_models = llm_base_models
            model = list(rag_test_lab.evaluated_models.values())[0]
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    elif model_type == commons.ModelTypeExplanation.RAG:
        if dataset_type == commons.ModelTypeExplanation.RAG:
            model = list(rag_test_lab.evaluated_models.values())[0]
        elif dataset_type == commons.ModelTypeExplanation.LLM:
            # LLM lab patch: LLM model -> RAG model (keep UUIDs)
            rag_base_models = {}
            for m in llm_test_lab.evaluated_models.values():
                rag_base_m = sonar_models.ExplainableRagModel(
                    connection=m.connection,
                    model_type=sonar_models.ExplainableModelType.h2ogpte_llm,
                    llm_model_name=m.llm_model_name,
                    key=m.key,
                )
                rag_base_models[rag_base_m.name] = rag_base_m

            llm_test_lab.evaluated_models = rag_base_models
            model = list(llm_test_lab.evaluated_models.values())[0]
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # models
    if models_type is None:
        models = None
    elif models_type == commons.ModelTypeExplanation.IID:
        models = [explainable_iid_model]
    elif models_type == commons.ModelTypeExplanation.LLM:
        if dataset_type == commons.ModelTypeExplanation.LLM:
            models = list(llm_test_lab.evaluated_models.values())
        elif dataset_type == commons.ModelTypeExplanation.RAG:
            # RAG dataset patch: RAG models -> LLM models (keep UUIDs)
            llm_proxy_models = {}
            for m in rag_test_lab.evaluated_models.values():
                llm_proxy_m = sonar_models.ExplainableLlmModel(
                    connection=m.connection,
                    model_type=sonar_models.ExplainableModelType.h2ogpte_llm,
                    llm_model_name=m.llm_model_name,
                    key=m.key,
                )
                llm_proxy_models[m.name] = llm_proxy_m

            rag_test_lab.evaluated_models = llm_proxy_models
            models = list(rag_test_lab.evaluated_models.values())
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    elif models_type == commons.ModelTypeExplanation.RAG:
        if dataset_type == commons.ModelTypeExplanation.RAG:
            models = list(rag_test_lab.evaluated_models.values())
        elif dataset_type == commons.ModelTypeExplanation.LLM:
            # LLM lab patch: LLM models -> RAG model (keep UUIDs)
            rag_proxy_models = {}
            for m in llm_test_lab.evaluated_models.values():
                rag_base_m = sonar_models.ExplainableRagModel(
                    connection=m.connection,
                    model_type=sonar_models.ExplainableModelType.h2ogpte_llm,
                    llm_model_name=m.llm_model_name,
                    key=m.key,
                )
                rag_proxy_models[m.name] = rag_base_m

            llm_test_lab.evaluated_models = rag_proxy_models
            models = list(llm_test_lab.evaluated_models.values())
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    else:
        raise ValueError(f"Unknown models type: {models_type}")

    # integrity check
    llm_test_lab.integrity_check()
    rag_test_lab.integrity_check()

    #
    # WHEN
    #

    interpretation = interpret.run_interpretation(
        target_col=target_col,
        dataset=dataset,
        model=model,
        models=models,
        explainers=explainers,
        results_location=tmp_path,
    )

    #
    # THEN
    #

    print(f"{interpretation}")
    assert not interpretation.get_failed_evaluator_ids()
    assert expected_e_count == len(interpretation.get_successful_explainer_ids())
    # result
    if expected_e_count:
        result = interpretation.get_explainer_result(explainers[0])
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
        f"  HTML: file://{interpretation.result.get_html_report_location()}\n"
    )


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
# @pytest.mark.skipif(
#     test_utils.GitHubActions.is_in_gha(),
#     reason="Skipped on GHA as this test is flaky there while passing locally",
# )
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_async_evaluate(tmp_path):
    try:
        #
        # GIVEN
        #
        print("=" * 80)

        # h2oGPTe server
        h2ogpte_connection = test_utils.health.get_h2ogpte()

        # test lab w/ actual values including contexts
        # EVALUATOR: strstr lab
        test_lab_path = "data/generative/h2ogpte_benchmark_test_lab_top.json"
        # EVALUATOR: RAGAS lab
        # test_lab_path = "data/generative/h2ogpte_benchmark_test_lab_small.json"
        print(f"GIVEN: Loading test lab from: {test_lab_path}")

        # test lab (load cfg w/ actual values - build/chat not needed)
        test_lab = testing.RagTestLab.load_from_json(
            llm_host_connection=h2ogpte_connection,
            file_path=test_lab_path,
            docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
        )

        #
        # WHEN ... run evaluation asynchronously
        #
        print("=" * 80)

        # JUDGE: force custom evaluation judge - it can be:
        # - bool:
        #   "false" means to use the default judge,
        #   "true" means to use the custom judge (the first judge found in the config)
        # - str: judge configuration key: the key of the custom judge configuration
        #   to be used for the evaluation and instantiated by the evaluator
        h2o_sonar_config.config.add_connection(h2ogpte_connection)
        eval_judge_cfg = h2o_sonar_config.config.add_evaluation_judge(
            h2o_sonar_config.EvaluationJudgeConfig(
                name="TEST EvalStudio CUSTOM judge",
                description="Custom evaluation judge for the test.",
                judge_type=h2o_sonar_config.EvaluationJudgeType.h2ogpte_llm.name,
                connection=h2ogpte_connection,
                llm_model_name=given_generative.H2OGPTE_JUDGE_LLM_MODEL_NAME,
            )
        )
        eval_judge = judges.get_evaluation_judge_for_config(eval_judge_cfg)
        assert eval_judge
        print(f"WHEN: Evaluation judge configured: {eval_judge}")
        # JUDGE: force use of the custom evaluation judge over the default one
        h2o_sonar_config.config.force_eval_judge = "true"

        async_evaluations = []
        progress_callbacks = []
        print("WHEN: Starting 5x async evaluations...")
        for i in range(5):
            print(f"WHEN: STARTING async evaluation #{i}")

            # progress: 1 stage - evaluate
            progress_callback_name = f"[TEST E2E progress callback #{i}]"
            progress_callback = progress.LoggingProgressCallbackContext(
                # every interpretation to have its own logger to avoid race conditions
                logger=loggers.SonarPrintLogger(),
                prefix=progress_callback_name,
                name=progress_callback_name,
            )
            progress_callbacks.append(progress_callback)

            print(f"WHEN: run_evaluation #{i} (async=True)...")
            async_evaluation = evaluate.run_evaluation(
                # dataset w/ prompts, constraints and model keys
                dataset=test_lab.dataset,
                # models to be evaluated / compared to get leaderboard
                models=list(test_lab.evaluated_models.values()),
                # evaluators
                evaluators=[
                    # EVALUATOR: RAGAs (slow)
                    # ragas_evaluator.RagasEvaluator.evaluator_id(),
                    # EVALUATOR: toxicity (slow)
                    # toxic_evaluator.ToxicityEvaluator().evaluator_id(),
                    # EVALUATOR: context precision (mid)
                    # ctx_p_evaluator.ContextPrecisionEvaluator().evaluator_id(),
                    # EVALUATOR: strstr (fast)
                    rag_tokens_presence_evaluator.RagStrStrEvaluator().evaluator_id(),
                    # EVALUATOR: PII (fast)
                    # pii_leakage_evaluator.PiiLeakageEvaluator().evaluator_id(),
                    # EVALUATOR: data (fast)
                    # certs_evaluator.SensitiveDataLeakageEvaluator().evaluator_id(),
                ],
                # where to save the report
                results_location=tmp_path,
                # logging
                log_level=logging.DEBUG,
                # ASYNCHRONOUS evaluation
                run_asynchronously=True,
                # progress
                progress_callback=progress_callback,
            )
            print(f"WHEN: STARTED evaluation #{i}: key={async_evaluation.key}")
            print(f"WHEN:   status={async_evaluation.status}")
            async_evaluations.append(async_evaluation)

        # caller to persist only evaluation ID and results directory path
        evaluation = None
        print("=" * 80)
        print("Waiting for all evaluations to complete...")
        for e, async_evaluation in enumerate(async_evaluations):
            print(f"WAIT: Processing async evaluation #{e}")
            print(f"WAIT:   ID: {async_evaluation.key}")
            print(
                f"WAIT:   Results location: {async_evaluation.result.results_location}"
            )
            print(f"WAIT:   Status BEFORE wait: {async_evaluation.status}")

            evaluation_key = async_evaluation.key
            results_path = async_evaluation.result.results_location

            # WAIT for the evaluation to finish
            print("WAIT: Calling wait_for_evaluation()")
            wait_start_time = time.time()
            evaluation = evaluate.wait_for_evaluation(
                evaluation_key=evaluation_key,
                results_location=results_path,
                wait_steps=100_000,
                wait_step_seconds=1.0,
                logger=loggers.SonarPrintLogger(),
            )
            wait_duration = time.time() - wait_start_time
            print(f"WAIT: wait_for_evaluation returned after {wait_duration:.2f}s")
            print(f"WAIT:   Status AFTER wait: {evaluation.status}")
            print(
                f"WAIT:   Finished evaluators: "
                f"{evaluation.get_finished_evaluator_ids()}"
            )
            print(f"WAIT:   Failed evaluators: {evaluation.get_failed_evaluator_ids()}")
            print(
                f"WAIT:   Successful evaluators: "
                f"{evaluation.get_successful_evaluator_ids()}"
            )

            if evaluation.get_failed_evaluator_ids():
                print(f"WAIT: ERROR - Evaluation #{e} FAILED!")
                raise RuntimeError(
                    f"Evaluation #{e} FAILED - the following evaluators failed: "
                    f"{evaluation.get_failed_evaluator_ids()}, evaluation: "
                    f"{evaluation}"
                )
            print(f"WAIT: Evaluation #{e} completed successfully")

        #
        # THEN ... get the leaderboard JSon index file path
        #
        print("=" * 80)

        print(f"THEN: evaluation instance: {evaluation is not None}")
        assert evaluation
        evaluation_key = evaluation.key
        results_path = evaluation.result.results_location
        print(f"THEN: evaluation_key: {evaluation_key}")
        print(f"THEN: results_path: {results_path}")

        if commons.ExplainerJobStatus.is_job_failed(evaluation.status):
            raise RuntimeError(f"Evaluation FAILED with status: {evaluation.status}")
        e_jobs = list(evaluation.result.explainers.values())
        print(f"THEN: number of  jobs: {len(e_jobs)}")
        if not e_jobs:
            raise RuntimeError("No evaluator has been executed")

        #
        # GET the LEADERBOARDS as JSon and Markdown
        #

        # get THE FIRST evaluator leaderboard DATA file as JSon
        print("THEN: Loading evaluation from disk...")
        loaded_evaluation = evaluate.get_evaluation(
            evaluation_key=evaluation_key,
            results_location=results_path,
        )

        leaderboard_as_json = e10s.LlmLeaderboardExplanation.get_leaderboard_data_path(
            loaded_evaluation,
            evaluator_id=loaded_evaluation.result.get_evaluator_jobs()[
                0
            ].evaluator_id(),
            explanation_format=f5s.LlmLeaderboardJSonFormat.mime,
            metric=f5s.LlmLeaderboardJSonFormat.KEY_ALL_METRICS,
        )
        leaderboard_as_md = e10s.LlmLeaderboardExplanation.get_leaderboard_data_path(
            loaded_evaluation,
            evaluator_id=loaded_evaluation.result.get_evaluator_jobs()[
                0
            ].evaluator_id(),
            explanation_format=f5s.EvalStudioMarkdownFormat.mime,
        )

        # progress
        print("=" * 80)
        print("THEN: PROGRESS VERIFICATION - waiting for callbacks to reach 100% ...")
        print(f"THEN: Progress callback(s) - waiting for {len(progress_callbacks)}:")
        for t in range(60):  # wait 1' for callback pollers to deliver the 100% progress
            all_finished = True
            print(f"  WAITING {t}s...")
            for e, progress_callback in enumerate(progress_callbacks):
                print(
                    f"    #{e} {progress_callback.name}: {progress_callback.progress}"
                )
                print(f"PROGRESS:   Callback #{e}: {progress_callback.progress}")
                if progress_callback.progress < 1.0:
                    print(f"      evaluation PROGRESS #{e} did NOT reach 100% yet")
                    all_finished = False
            if all_finished:
                print("PROGRESS: All callbacks reached 100%")
                break

            time.sleep(1.0)
        print(f"THEN: Progress callback(s) - asserts for {len(progress_callbacks)}:")
        for e, progress_callback in enumerate(progress_callbacks):
            print(f"  #{e} {progress_callback.name}: {progress_callback.progress}")
            assert progress_callback.progress == 1.0, (
                f"Evaluation progress callback #{e} did not reach 100%: "
                f"{progress_callback.progress * 100.0:.1f}%"
            )

        print("=" * 80)
        print(
            f"Explanations:\n"
            f"  HTML     : file://{evaluation.result.get_html_report_location()}\n"
            f"  JSon data: file://{leaderboard_as_json}\n"
            f"  MD data  : file://{leaderboard_as_md}\n"
        )
    except Exception as ex:
        print("=" * 80)
        print(f"TEST FAILED: {ex}\n {traceback.format_exc()}")
        raise
    finally:
        h2o_sonar_config.config.force_eval_judge = "false"


@pytest.mark.skip(reason="This is a tool to generate mock data for ES workflows")
@pytest.mark.parametrize(
    "do_perturb,do_eval",
    [
        (False, False),
        (True, False),
        (True, True),
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_es_workflows_robustness_prompts_mock(do_perturb, do_eval, tmp_path):
    """This is a tool to generate mock data for H2O Eval Studio workflows - robustness
    testing step..

    Parameters
    ----------
    do_perturb : bool
        Whether to generate perturbed prompts.
    do_eval : bool
        Whether to generate evaluation metrics.
    tmp_path : Path
        Path to the temporary directory

    """
    #
    # GIVEN
    #
    lab_path = "data/generative/conferences/atlanta-2024/sr1107_test_lab_small.json"
    with open(lab_path) as f:
        lab = json.load(f)

    #
    # WHEN
    #
    es_tcs = []

    perturbator = perturbations.CommaPerturbator() if do_perturb else None

    for e, i in enumerate(lab["dataset"]["inputs"]):
        prompt = i["input"]
        output = i["actual_output"]

        eval_id = g_e.RagGroundednessEvaluator.evaluator_id() if do_eval else ""
        eval_name = g_e.RagGroundednessEvaluator().display_name if do_eval else ""
        eval_metrics_name = (
            g_e.RagGroundednessEvaluator()
            .get_evaluation_metrics()
            .get_primary_metric()
            .display_name
            if do_eval
            else ""
        )
        eval_metrics_score = random.uniform(0, 1) if do_eval else 0.0
        o_eval_metrics_passed = eval_metrics_score > 0.5 if do_eval else False

        es_tcs.append(
            {
                "id": f"{e}",
                "prompt": prompt,
                "modelOutput": output,
                "perturbatorId": "",
                "perturbatorName": "",
                "evaluatorId": eval_id,
                "evaluatorName": eval_name,
                "evalMetricName": eval_metrics_name,
                "evalMetricScore": eval_metrics_score,
                "evalMetricPassed": o_eval_metrics_passed,
                "flip": False,
            }
        )

        if perturbator:
            perturbed_prompt = perturbator.perturb(
                text=prompt,
                intensity=commons.PerturbationIntensity.HIGH,
                raised_errors=[],
            )

            eval_id = g_e.RagGroundednessEvaluator.evaluator_id() if do_eval else ""
            eval_name = g_e.RagGroundednessEvaluator()._display_name if do_eval else ""
            eval_metrics_name = (
                g_e.RagGroundednessEvaluator()
                .get_evaluation_metrics()
                .get_primary_metric()
                .display_name
                if do_eval
                else ""
            )
            eval_metrics_score = random.uniform(0, 1) if do_eval else 0.0
            eval_metrics_passed = eval_metrics_score > 0.5 if do_eval else False

            es_tcs.append(
                {
                    "id": f"{e}.1",
                    "prompt": perturbed_prompt,
                    "modelOutput": output if do_eval else "",
                    "perturbatorId": perturbator.perturbator_id(),
                    "perturbatorName": perturbator.display_name,
                    "evaluatorId": eval_id,
                    "evaluatorName": eval_name,
                    "evalMetricName": eval_metrics_name,
                    "evalMetricScore": eval_metrics_score,
                    "evalMetricPassed": eval_metrics_passed,
                    "flip": (
                        True
                        if (o_eval_metrics_passed and not eval_metrics_passed)
                        or (not o_eval_metrics_passed and eval_metrics_passed)
                        else False
                    ),
                }
            )

    #
    # THEN
    #
    print(json.dumps(es_tcs, indent=2))
    # save it to tmpdir
    with open(tmp_path / "es_tcs.json", "w") as f:
        json.dump(es_tcs, f, indent=2)
    print(f"Saved to file://{tmp_path / 'es_tcs.json'}")


@pytest.mark.generative
def test_evaluation_model_loading():
    #
    # GIVEN
    #
    results_location = "data/generative/bugs/bug-1648"
    evaluation_key = "7ee992f2-e30f-485e-8a06-2bc0e0978e49"

    #
    # WHEN
    #
    print(f"Loading the EVALUATION {evaluation_key} from {results_location}")

    loaded_evaluation = evaluate.get_evaluation(
        evaluation_key=evaluation_key,
        results_location=results_location,
    )

    #
    # THEN
    #
    print(f"Loaded EVALUATION: {loaded_evaluation is not None}")
    assert loaded_evaluation
    print(
        f"Loaded MODELS:\n"
        f"{json.dumps(loaded_evaluation.result.models[0].to_dict(), indent=2)}"
    )
    assert loaded_evaluation.result.models

    model_type = loaded_evaluation.result.models[0].model_type
    print(f"MODEL TYPE: {model_type}")
    assert m4s.ExplainableModelType.is_rag(model_type)


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
