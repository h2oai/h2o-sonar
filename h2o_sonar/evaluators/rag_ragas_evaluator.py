# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import os
import traceback

from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets as d6s
from h2o_sonar.lib.api import evaluators
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import judges
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import problems
from h2o_sonar.lib.api import results
from h2o_sonar.lib.integrations import ragas_adapter


try:
    import ragas
    from ragas import metrics as ragas_metrics

    HAS_RAGAS = True
except ImportError:
    HAS_RAGAS = False

try:
    from datasets import Dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


# disable ragas library telemetry
os.environ["RAGAS_DO_NOT_TRACK"] = "true"


class RagasEvaluator(evaluators.Evaluator):
    _display_name = "Ragas"
    _tagline = "Evaluate RAG retrieval and generation using four metrics."

    # Technical hints:
    # - supported Python runtimes: 3.11
    # - ragas library versions:
    #   - ragas==0.0.21 ... used prior CUSTOM LLM judge/embeddings implementation
    #   - ragas==0.1.3  ... used to support CUSTOM LLM judge/embeddings

    # COMPATIBILITY: LLM/RAG evaluation
    _rag = True

    # GLOBAL: average metric of all dataset rows
    _global_explanation = True
    # LOCAL: metric values for all dataset rows
    _local_explanation = True
    _explanation_types = [
        e10s.LlmEvalResultsExplanation,
        e10s.LlmHeatmapLeaderboardExplanation,
        e10s.WorkDirArchiveExplanation,
    ]

    _keywords = [
        evaluators.KEYWORD_LLM,
        evaluators.KEYWORD_EVALUATES_RAG,
        evaluators.KEYWORD_RQ_P,
        evaluators.KEYWORD_RQ_EA,
        evaluators.KEYWORD_RQ_AA,
        evaluators.KEYWORD_RQ_RC,
        evaluators.KEYWORD_RQ_J,
        evaluators.KEYWORD_REQUIRES_OPENAI_KEY,
        evaluators.KEYWORD_SR_11_7_CS,
        evaluators.KEYWORD_SR_11_7_OA,
        evaluators.KEYWORD_NIST_AI_RMF_S,
        evaluators.KEYWORD_PROBLEM_TYPE_IR,
        evaluators.KEYWORD_PROBLEM_TYPE_QA,
        evaluators.KEYWORD_ES_GENERATE,
        evaluators.KEYWORD_ES_RETRIEVE,
        evaluators.KEYWORD_METHOD_JUDGE,
        evaluators.KEYWORD_METHOD_TYPE_NON_DETERMINISTIC,
    ]

    METRIC_ANSWER_RELEVANCY = "answer_relevancy"
    METRIC_ANSWER_CORRECTNESS = "answer_correctness"
    METRIC_ANSWER_SIMILARITY = "answer_similarity"
    METRIC_CONTEXT_RECALL = "context_recall"
    METRIC_CONTEXT_PRECISION = "context_precision"
    METRIC_CONTEXT_RELEVANCY = "context_relevancy"
    METRIC_FAITHFULNESS = "faithfulness"
    METRIC_RAGAS = "ragas"

    # ragas evaluator can evaluate 6 metrics above, however, in case of ragas
    # metric only 1+4 metrics are calculated; other evaluators use various
    # combinations of these metrics - this is why metrics are defined as constants
    # and default metadata is defined for 1+4 metrics only
    METRIC_META_ANSWER_RELEVANCY = commons.MetricMeta(
        key=METRIC_ANSWER_RELEVANCY,
        display_name="Answer relevancy",
        description=(
            "Answer relevancy metric (retrieval+generation) is assessing how pertinent "
            "the generated answer is to the given prompt. A lower score indicates "
            "answers which are incomplete or contain redundant information. "
            "This metric is computed using the question and the answer. "
            "Higher the better. An answer is deemed relevant when it directly and "
            "appropriately addresses the original question. "
            "To calculate this score, the LLM is prompted to generate an appropriate "
            "question for the generated answer multiple times, and the mean cosine "
            "similarity of generated questions with the original question is measured."
        ),
        higher_is_better=True,
        threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
        is_primary_metric=False,
    )
    METRIC_META_ANSWER_CORRECTNESS = commons.MetricMeta(
        key=METRIC_ANSWER_CORRECTNESS,
        display_name="Answer correctness",
        description=(
            "The assessment of answer correctness metric involves gauging the accuracy "
            "of the generated answer when compared to the ground truth. This "
            "evaluation relies on the ground truth and the answer, with scores ranging "
            "from 0 to 1. A higher score indicates a closer alignment between the "
            "generated answer and the ground truth, signifying better correctness. "
            "Answer correctness metric encompasses two critical aspects:"
            "semantic similarity between the generated answer and the ground truth, "
            "as well as factual similarity. These aspects are combined using "
            "a weighted scheme to formulate the answer correctness score."
        ),
        higher_is_better=True,
        threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
        is_primary_metric=False,
    )
    METRIC_META_ANSWER_SIMILARITY = commons.MetricMeta(
        key=METRIC_ANSWER_SIMILARITY,
        display_name="Answer similarity",
        description=(
            "The concept of answer semantic similarity pertains to the assessment of "
            "the semantic resemblance between the generated answer and "
            "the ground truth. This evaluation is based on the ground truth and "
            "the answer, with values falling within the range of 0 to 1. "
            "A higher score signifies a better alignment between the generated answer "
            "and the ground truth. Semantic similarity between answers can offer "
            "valuable insights into the quality of the generated response. This "
            "evaluation utilizes a cross-encoder model to calculate the semantic "
            "similarity score."
        ),
        higher_is_better=True,
        threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
        is_primary_metric=False,
    )
    METRIC_META_CONTEXT_RECALL = commons.MetricMeta(
        key=METRIC_CONTEXT_RECALL,
        display_name="Context recall",
        description=(
            "Context recall metric (retrieval) measures the extent to which the "
            "retrieved context aligns with the expected answer, treated as the ground "
            "truth. "
            "It is computed based on the ground truth and the retrieved context. "
            "Higher the better. Each sentence in the expected answer is analyzed "
            "to determine whether it can be attributed to the retrieved context "
            "or not:"
            " (expected answer sentences that can be attributed to context / "
            "expected answer sentences count)"
        ),
        higher_is_better=True,
        threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
        is_primary_metric=False,
    )
    METRIC_META_CONTEXT_PRECISION = commons.MetricMeta(
        key=METRIC_CONTEXT_PRECISION,
        display_name="Context precision",
        description=(
            "Context precision metric (retrieval) evaluator uses a metric that "
            "evaluates whether all of the ground-truth relevant items present "
            "in the contexts are ranked higher or not - ideally all the relevant "
            "chunks must appear at the top of the context - ranged high."
        ),
        higher_is_better=True,
        threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
        is_primary_metric=False,
    )
    METRIC_META_CONTEXT_RELEVANCY = commons.MetricMeta(
        key=METRIC_CONTEXT_RELEVANCY,
        display_name="Context relevancy",
        description=(
            "Context relevancy metric gauges the relevancy of the retrieved context, "
            "calculated based on both the question and contexts. The values fall "
            "within the range of (0, 1), with higher values indicating better "
            "relevancy. "
            "Ideally, the retrieved context should exclusively contain essential "
            "information to address the provided query. To compute this, "
            "evaluator initially estimate the value of by identifying sentences within "
            "the retrieved context that are relevant for answering the given question. "
            "The final score is determined by the following formula: "
            "ctx relevancy = (number of question relevant sentences / "
            "total number of context sentences)."
        ),
        higher_is_better=True,
        threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
        is_primary_metric=False,
    )
    METRIC_META_FAITHFULNESS = commons.MetricMeta(
        key=METRIC_FAITHFULNESS,
        display_name="Faithfulness",
        description=(
            "Faithfulness (generation) metric measures the factual consistency of "
            "the generated answer against the given context. It is calculated from "
            "answer and retrieved context. Higher the better. "
            "The generated answer is regarded as faithful if all the claims that "
            "are made in the answer can be inferred from the given context: "
            "(number of claims inferable from the context / claims in the answer)."
        ),
        higher_is_better=True,
        threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
        is_primary_metric=False,
    )
    METRIC_META_RAGAS = commons.MetricMeta(
        key=METRIC_RAGAS,
        display_name="RAGAS",
        description=(
            "RAGAs (RAG Assessment) metric is a harmonic mean of the following "
            "metrics: faithfulness, answer relevancy, context precision and context "
            "recall."
        ),
        higher_is_better=True,
        threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
        is_primary_metric=True,
    )

    _metrics_meta = commons.MetricsMeta(
        [
            commons.MetricMeta.clone(METRIC_META_RAGAS),
            commons.MetricMeta.clone(METRIC_META_FAITHFULNESS, False),
            commons.MetricMeta.clone(METRIC_META_ANSWER_RELEVANCY, False),
            commons.MetricMeta.clone(METRIC_META_CONTEXT_PRECISION, False),
            commons.MetricMeta.clone(METRIC_META_CONTEXT_RECALL, False),
        ]
    )

    _parameters = [
        evaluators.Evaluator._get_custom_param_metric_threshold(
            _metrics_meta.get_primary_metric()
        ),
        evaluators.Evaluator._PARAM_NAN_TOLERANCE,
        evaluators.Evaluator._PARAM_SAVE_LLM_RESULT,
        evaluators.Evaluator._PARAM_EVAL_JUDGE,
        evaluators.Evaluator._get_custom_param_min_test_case(),
    ]

    _modules_needed_by_name = ["ragas==0.1.3"]

    _brief_description = """RAGAs (RAG Assessment) is a framework that helps you
evaluate your Retrieval Augmented Generation (RAG) pipelines. RAG refers to LLM
applications that use external data to enhance the context. Evaluation
and quantifying the performance of your pipeline can be hard. This is
where Ragas (RAG Assessment) comes in. RAGAs metrics score includes
both performance of the **retrieval** and **generation** components of
the RAG pipeline.
Therefore RAGAs score represents the **overall quality** of the answer
considering both the retrieval and the answer generation itself.

- Harmonic mean of Faithfulness, Answer Relevancy, Context precision,
  and Context Recall metrics.
- Compatibility: RAG evaluation only.
- Based on
  [RAGAs library](https://docs.ragas.io/en/latest/concepts/metrics/index.html)"""
    _description = evaluators.Evaluator._description_builder(
        brief=f"""{_brief_description}

**Method**:

- RAGAs metric score is calculated as
  [harmonic mean](https://en.wikipedia.org/wiki/Harmonic_mean) of the four
  metrics calculated by the following evaluators:
   - Faithfulness Evaluator (generation)
   - Answer Relevancy Evaluator (retrieval+generation)
   - Context Precision Evaluator (retrieval)
   - Context Recall Evaluator (retrieval)
- Faithfulness covers generation answer quality, Answer Relevancy covers
  answer generation and retrieval quality.
  Context Precision and Context Recall evaluate the retrieval quality.

```math
                      4
   RAGAS = --------------------------
             1     1      1      1
            --- + ---- + ---- + ----
             F     AR     CP     CR
```

- Where:
    - `F` is the Faithfulness metric.
    - `AR` is the Answer Relevancy metric.
    - `CP` is the Context Precision metric.
    - `CR` is the Context Recall metric.

See also:

- Paper: *"RAGAS: Automated Evaluation of Retrieval Augmented Generation"*:
  https://arxiv.org/abs/2309.15217
- 3rd party metric documentation:
  https://docs.ragas.io/en/latest/concepts/metrics/index.html
- 3rd party library used: https://github.com/explodinggradients/ragas""",
        metrics_meta=commons.MetricsMeta(
            metrics=[
                METRIC_META_RAGAS,
            ]
        ),
        keywords=_keywords,
        parameters=_parameters,
        leaderboard_type=e10s.LlmHeatmapLeaderboardExplanation.explanation_type(),
    )

    # HF/RAGAS dataset format
    KEY_QUESTION = "question"
    KEY_GROUND_TRUTHS = "ground_truths"
    KEY_ANSWER = "answer"
    KEY_CONTEXTS = "contexts"

    def __init__(self):
        evaluators.Evaluator.__init__(self)

        self.args = None
        self.problems = []
        self.custom_llm_judge = None
        self.log_name = "RAGAS evaluator"

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        **evaluator_params,
    ) -> bool:
        evaluators.Evaluator.check_compatibility(self, params, **evaluator_params)

        if not HAS_RAGAS:
            self.logger.error(self._check_compatibility_pckg_err_msg("ragas"))
            return False

        if not HAS_DATASETS:
            self.logger.error(self._check_compatibility_pckg_err_msg("datasets"))
            return False

        # IMPROVE check dataset columns presence & actual values (non-empty)
        # IMPROVE document column names in the evaluator description

        if not evaluators.Evaluator._check_llm_dataset_compatibility(
            self,
            params=params,
            evaluator_keywords=self.keywords,
            check_empty_contexts=evaluators.KEYWORD_RQ_RC in self.keywords,
            fail_on_all_empty_contexts=evaluators.KEYWORD_RQ_RC in self.keywords,
        ):
            return False

        return True

    def setup(self, model, persistence, **kwargs):
        evaluators.Evaluator.setup(self, model, persistence, **kwargs)

        self._resolve_evaluator_params()

    def evaluate(self, llm_testset, explanations_types=None, **kwargs) -> list:
        return self.eval_custom_metrics(
            llm_testset,
            metrics_threshold=self.args.get(
                evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
                evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
            ),
            save_llm_result=self.args.get(
                RagasEvaluator.PARAM_SAVE_LLM_RESULT,
                RagasEvaluator.DEFAULT_SAVE_LLM_RESULT,
            ),
            custom_eval_judge_cfg_key=self._resolve_judge_key(),
            metrics_to_run=commons.MetricsMeta(
                [RagasEvaluator.METRIC_META_RAGAS],
            ),
            nan_tolerance=self.args.get(
                evaluators.Evaluator.PARAM_NAN_TOLERANCE,
                evaluators.Evaluator.DEFAULT_NAN_TOLERANCE,
            ),
        )

    def eval_custom_metrics(
        self,
        llm_testset,
        metrics_threshold: float,
        save_llm_result: bool,
        custom_eval_judge_cfg_key: str,
        metrics_to_run: commons.MetricsMeta,
        evaluator: evaluators.Evaluator | None = None,
        nan_tolerance: float = evaluators.Evaluator.DEFAULT_NAN_TOLERANCE,
    ) -> list:
        if evaluator:
            self.args = evaluator.args

        #
        # CUSTOM EVALUATION (LLM) JUDGE
        #
        if custom_eval_judge_cfg_key:
            # LLM judge materialization
            eval_judge_cfg = self.config.get_evaluation_judge(
                judge_key=custom_eval_judge_cfg_key
            )
            if not eval_judge_cfg:
                valid_judge_keys = [j.key for j in self.config.evaluation_judges]
                raise ValueError(
                    f"Custom LLM judge for key: "
                    f"'{custom_eval_judge_cfg_key}' not found in H2O Sonar "
                    f"configuration. Valid keys are: {valid_judge_keys}"
                )
            eval_judge = judges.get_evaluation_judge_for_config(eval_judge_cfg)
            custom_ragas_judge = ragas_adapter.get_ragas_to_sonar_llm_adapter(
                eval_judge
            )
            # run custom judge health check
            try:
                custom_ragas_judge.health_check()
            except Exception as ex:
                raise ValueError(
                    f"Custom LLM judge '{eval_judge_cfg.name}' health check failed: "
                    f"{ex}\n{traceback}"
                )
            # EMBEDDINGS materialization
            custom_embeddings_model = ragas_adapter.get_ragas_privacy_safe_embeddings()
        else:
            eval_judge_cfg = None
            custom_ragas_judge = None
            custom_embeddings_model = None

        #
        # EVALUATION
        #
        (
            eval_results,
            sort_by_metric,
            key_2_evaluated_model,
            llm_host,
        ) = self._calculate_metrics(
            llm_testset=llm_testset,
            metrics_threshold=metrics_threshold,
            metrics_to_run=metrics_to_run,
            eval_judge=custom_ragas_judge,
            eval_embeddings_model=custom_embeddings_model,
        )

        #
        # NORMALIZATION of the evaluation results to the common EXPLANATIONS/FORMAT(s)
        #

        # EXPLANATIONS
        explanations = []

        # EXPLANATION: all data (per prompt metrics)
        if save_llm_result:
            eval_results_explanation = e10s.LlmEvalResultsExplanation(
                evaluator=self,
                display_name="Evaluation metrics data",
                display_category=e10s.Explanation.DISPLAY_CAT_LLM,
                eval_results=eval_results,
            )
            # FORMATS of the explanation: JSon, CSV, DataTable
            eval_results_explanation.add_json_format()
            eval_results_explanation.add_csv_format()
            eval_results_explanation.add_datatable_format()
            explanations.append(eval_results_explanation)

        # METRICS META: complete base metrics set needed by composite metric like ragas
        if metrics_to_run.contains(RagasEvaluator.METRIC_META_RAGAS.key):
            for m_key in RagasEvaluator._metrics_meta.key_to_metric:
                if not metrics_to_run.contains(m_key):
                    metrics_to_run.add_metric(
                        RagasEvaluator._metrics_meta.get_metric(m_key)
                    )

        # EXPLANATION: heatmap leaderboard
        heatmap_explanation = e10s.LlmHeatmapLeaderboardExplanation.from_eval_results(
            evaluator=self,
            eval_results=eval_results,
            metrics_meta=metrics_to_run,
            key_2_evaluated_model=key_2_evaluated_model,
            llm_host=llm_host,
            display_name="LLM heatmap leaderboard",
            display_category=e10s.GlobalSummaryFeatImpExplanation.DISPLAY_CAT_LLM,
            nan_tolerance=nan_tolerance,
            logger=self.logger,
        )
        heatmap_explanation.add_json_format(
            threshold=self.args.get(
                evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
                self._metrics_meta.get_primary_metric().threshold,
            )
        )
        heatmap_explanation.add_markdown_format(sort_by_metric_id=sort_by_metric)
        heatmap_explanation.add_evalstudio_markdown_format(
            sort_by_metric_id=sort_by_metric
        )
        explanations.append(heatmap_explanation)

        # PROBLEMS for alerts and actionability
        self._diagnose_problems(
            eval_results=eval_results,
            key_2_evaluated_model=key_2_evaluated_model,
            leaderboard_explanation=heatmap_explanation,
            metrics_to_run=metrics_to_run,
            evaluator=evaluator,
        )

        # INSIGHTS
        RagasEvaluator._diagnose_insights(
            leaderboard_explanation=heatmap_explanation,
            metrics_to_run=metrics_to_run,
            metric_id=sort_by_metric,
        )

        # EXPLANATION: HTML fragment
        if self.config and self.config.create_html_representations:
            try:
                html_explanation = e10s.GlobalHtmlFragmentExplanation(
                    evaluator=self,
                    display_name="LLM heatmap leaderboard as HTML",
                    display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
                )
                html_explanation.add_html_format(
                    str(
                        heatmap_explanation.as_html(
                            sort_by_metric_id=sort_by_metric,
                            additional_details={
                                "Judge used by evaluator": (
                                    "Default OpenAI GPT LLM."
                                    if not eval_judge_cfg
                                    else (
                                        f"Custom '{eval_judge_cfg.name}' with "
                                        f" '{eval_judge_cfg.llm_model_name}' LLM."
                                    )
                                ),
                            },
                        )
                    ),
                )
                explanations.append(html_explanation)
            except Exception as ex:
                self.logger.warning(
                    f"{self.log_name}: HTML fragment explanation creation failed: "
                    f"{ex}\n{traceback.format_exc()}"
                )

        # EXPLANATION: ZIP archive of all artifacts created by the evaluator
        explanations.append(
            self.create_explanation_workdir_archive(
                display_name=f"Archive of {self._display_name} artifacts",
                display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
            )
        )

        return explanations

    def _calculate_metrics(
        self,
        llm_testset,
        metrics_threshold: float,
        metrics_to_run: commons.MetricsMeta,
        eval_judge=None,
        eval_embeddings_model=None,
    ) -> tuple[d6s.LlmEvalResults, str, dict, commons.LlmModelHostType]:
        # IMPROVE: add critique (custom)
        # from ragas.metrics.critique import AspectCritique

        metric_to_module = {
            RagasEvaluator.METRIC_ANSWER_RELEVANCY: ragas_metrics.answer_relevancy,
            RagasEvaluator.METRIC_ANSWER_CORRECTNESS: ragas_metrics.answer_correctness,
            RagasEvaluator.METRIC_ANSWER_SIMILARITY: ragas_metrics.answer_similarity,
            RagasEvaluator.METRIC_CONTEXT_RECALL: ragas_metrics.context_recall,
            RagasEvaluator.METRIC_CONTEXT_PRECISION: ragas_metrics.context_precision,
            RagasEvaluator.METRIC_CONTEXT_RELEVANCY: ragas_metrics.context_relevancy,
            RagasEvaluator.METRIC_FAITHFULNESS: ragas_metrics.faithfulness,
        }

        llm_dataset = d6s.LlmDataset.from_datatable_dict(llm_testset.to_dict())
        llm_dt_dict = llm_dataset.to_datatable_dict()

        # LLM dataset > RAGAS @ HF dataset conversion
        self.report_progress(0.01, "Converting LLM dataset to HF dataset.")
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
        # > 1 ground truth (~ expected answer) is supported by RAGAS
        ground_truths = [[c] for c in llm_dt_dict[d6s.LlmDataset.KEY_EXPECTED_OUTPUT]]

        ragas_hf_dict = {
            RagasEvaluator.KEY_QUESTION: llm_dt_dict[d6s.LlmDataset.KEY_INPUT],
            RagasEvaluator.KEY_GROUND_TRUTHS: ground_truths,
            RagasEvaluator.KEY_ANSWER: llm_dt_dict[d6s.LlmDataset.KEY_ACTUAL_OUTPUT],
            RagasEvaluator.KEY_CONTEXTS: contexts,
        }

        hf_dataset = Dataset.from_dict(ragas_hf_dict)

        metrics_keys_to_run = [key for key in metrics_to_run.get_metric_keys()]
        self.report_progress(
            0.10, f"Running RAGAS to evaluate '{metrics_keys_to_run}' metrics..."
        )
        if RagasEvaluator.METRIC_RAGAS in metrics_keys_to_run:
            # RAGAS-based evaluation
            result = ragas.evaluate(
                hf_dataset, llm=eval_judge, embeddings=eval_embeddings_model
            )
            scores_dict = result.scores.to_dict()
            # calculate RAGAS metrics @ harmonic mean of metrics on the row
            RagasEvaluator.__add_ragas_metric(scores_dict)
        else:
            metrics_par = [
                metric_to_module[m]
                for m in metrics_keys_to_run
                if m in metric_to_module
            ]
            # RAGAS-based evaluation
            result = ragas.evaluate(
                hf_dataset,
                metrics=metrics_par,
                llm=eval_judge,
                embeddings=eval_embeddings_model,
            )
            scores_dict = result.scores.to_dict()
        self.report_progress(
            0.90, f"RAGAS '{metrics_keys_to_run}' metrics calculation DONE"
        )

        # inject result metrics to testset > evaluation result
        self.report_progress(0.95, "Injecting evaluation metrics to the result...")
        eval_results = d6s.LlmEvalResults()
        result_metric_ids = scores_dict.keys()
        for i, r in enumerate(llm_dataset.inputs):
            # handle actual answer retrieval error ~ RAG/LLM client crash
            is_internal_err = evaluators.Evaluator._is_internal_err_answer(
                r.actual_output
            )

            # handle empty actual output
            is_empty_actual_output = not r.actual_output or not isinstance(
                r.actual_output, str
            )
            if is_empty_actual_output:
                self.logger.warning(
                    f"{self.log_name}: Row {i + 1} - Empty actual output detected. "
                    f"Setting worst metric values."
                )

            # handle empty expected output
            is_empty_expected_output = not r.expected_output or not isinstance(
                r.expected_output, str
            )
            if is_empty_expected_output:
                description = (
                    f"{self.log_name}: Row {i + 1} - Empty expected output detected. "
                    f"Setting worst metric values."
                )
                self.logger.warning(description)

                explanation_type = e10s.GlobalHtmlFragmentExplanation.explanation_type()
                self.add_problem(
                    problems.ProblemAndAction(
                        description=description,
                        severity=problems.ProblemSeverity.low,
                        evaluator_id=self.evaluator_id(),
                        evaluator_name=self._display_name,
                        problem_attrs={
                            problems.ProblemAndAction.ATTR_ROW_KEYS: [
                                (r.key, r.model_key)
                            ],
                            problems.ProblemAndAction.ATTR_TEST_CASE_KEYS: [r.key],
                            problems.ProblemAndAction.ATTR_EVALUATOR_NAME: (
                                self._display_name
                            ),
                        },
                        explanation_type=explanation_type,
                        explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
                        explanation_mime=f5s.HtmlFormat.mime,
                        resources=[],
                    )
                )

            # metrics dictionary
            metrics_dict = {}
            for m in result_metric_ids:
                # set WORST metrics values in case of internal error or empty output
                metrics_dict[m] = (
                    0.0
                    if (
                        is_internal_err
                        or is_empty_actual_output
                        or is_empty_expected_output
                    )
                    else scores_dict[m][i]
                )
            # add result row
            eval_results.add_result(
                d6s.LlmEvalResults.LlmEvalResultRow(
                    dataset_row=r,
                    metrics=metrics_dict,
                )
            )

        #
        # ADDITIONAL characteristics of the evaluation result
        #

        sort_by_metric_id = (
            RagasEvaluator.METRIC_RAGAS
            if RagasEvaluator.METRIC_RAGAS in scores_dict
            else next(iter(scores_dict.keys()))
        )

        metrics_to_run.set_threshold(metrics_threshold)

        # RAG models
        key_2_evaluated_model = {m.key: m for m in self.models}
        llm_host = (
            commons.LlmModelHostType.RAG
            if isinstance(
                next(iter(key_2_evaluated_model.values())), models.ExplainableRagModel
            )
            else commons.LlmModelHostType.SERVICE
        )

        self.report_progress(0.95, "Evaluator DONE")

        return (
            eval_results,
            sort_by_metric_id,
            key_2_evaluated_model,
            llm_host,
        )

    @staticmethod
    def __add_ragas_metric(scores_dict: dict):
        ragas_component = {RagasEvaluator.METRIC_RAGAS: []}
        for i in range(len(scores_dict[RagasEvaluator.METRIC_ANSWER_RELEVANCY])):
            ragas_inputs = []
            for m in [
                RagasEvaluator.METRIC_ANSWER_RELEVANCY,
                RagasEvaluator.METRIC_CONTEXT_PRECISION,
                RagasEvaluator.METRIC_FAITHFULNESS,
                RagasEvaluator.METRIC_CONTEXT_RECALL,
            ]:
                ragas_inputs.append(scores_dict[m][i])
            ragas_metric = commons.harmonic_mean(ragas_inputs)
            ragas_component[RagasEvaluator.METRIC_RAGAS].append(ragas_metric)

        scores_dict[RagasEvaluator.METRIC_RAGAS] = ragas_component[
            RagasEvaluator.METRIC_RAGAS
        ]

    def _diagnose_problems(
        self,
        eval_results: d6s.LlmEvalResults,
        key_2_evaluated_model: dict,
        leaderboard_explanation: e10s.LlmHeatmapLeaderboardExplanation,
        metrics_to_run: commons.MetricsMeta,
        evaluator: evaluators.Evaluator | None = None,
    ):
        # low test case count
        evaluators.Evaluator._diagnose_low_test_case_problem(
            evaluator or self,
            eval_results=eval_results,
            models=self.models,
            test_case_minimum=self.args.get(evaluators.Evaluator.PARAM_MIN_TEST_CASES),
        )

        # perturbation flips
        evaluators.Evaluator._diagnose_perturbation_problems(
            evaluator or self,
            eval_results=eval_results,
            key_2_evaluated_model=key_2_evaluated_model,
        )

        # threshold failures
        problems.problems_for_heat_leaderboard(
            evaluator=evaluator or self,
            leaderboard=leaderboard_explanation,
            metric_threshold=self.args.get(
                evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
                self._metrics_meta.get_primary_metric().threshold,
            ),
            primary_metric_meta=metrics_to_run.get_primary_metric(),
            problem_type="accuracy",
            problem_code=problems.AVIDProblemCode.P0200_MODEL,
            actions_description=(
                "Use three-pronged approach: data & training, model refinement, "
                "and evaluation. First, ensure high-quality training data with "
                "diverse retrieval sources and well-labeled generations. Techniques "
                "like curriculum learning can gradually challenge the LLM during "
                "training. Secondly, explore improvements in the retriever's "
                "ability to find relevant context and the LLM's generation of "
                "factually consistent answers. Finally, consider using more nuanced "
                "metrics beyond just RAGAs score, like human evaluation or "
                "task-specific measures, to get a clearer picture of the model's "
                "performance. This multi-faceted approach can significantly improve "
                "evaluation scores and the quality of RAG outputs."
            ),
            explanation_type=e10s.GlobalHtmlFragmentExplanation.explanation_type(),
            explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
            explanation_mime=f5s.HtmlFormat.mime,
        )

    @staticmethod
    def _diagnose_insights(
        leaderboard_explanation: e10s.LlmHeatmapLeaderboardExplanation,
        metric_id: str,
        metrics_to_run: commons.MetricsMeta,
    ):
        t_html_fragment = e10s.GlobalHtmlFragmentExplanation

        leaderboard_explanation.get_insights(
            metrics_meta=metrics_to_run,
            metric_id=metric_id,
            insight_type="accuracy",
            explanation_type=t_html_fragment.explanation_type(),
            explanation_name=t_html_fragment.__name__,
            explanation_mime=f5s.HtmlFormat.mime,
        )

    def get_result(
        self,
    ) -> results.LeaderboardResult:
        return results.LeaderboardResult(
            persistence=self.persistence,
            explainer_id=RagasEvaluator.explainer_id(),
            h2o_sonar_config=self.config,
            logger=self.logger,
            explanation=e10s.LlmHeatmapLeaderboardExplanation,
            explanation_format=f5s.CustomJsonFormat,
        )
