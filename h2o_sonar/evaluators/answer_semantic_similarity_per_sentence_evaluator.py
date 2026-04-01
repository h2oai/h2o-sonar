# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import traceback

import airium
import numpy as np

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import evaluators
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import problems
from h2o_sonar.lib.api import results
from h2o_sonar.utils import caching
from h2o_sonar.utils import progress as progress_utils
from h2o_sonar.utils import resource_mgmt
from h2o_sonar.utils import tokenization


try:
    import nltk

    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    import sentence_transformers

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class AnswerSemanticSimilarityPerSentenceEvaluator(evaluators.Evaluator):
    _display_name = "Answer semantic sentence similarity"

    METRIC_MEAN_ANSWER_SIMILARITY = "mean_answer_similarity"
    METRIC_MIN_ANSWER_SIMILARITY = "min_answer_similarity"

    _metrics_meta = commons.MetricsMeta(
        metrics=[
            commons.MetricMeta(
                key=METRIC_MEAN_ANSWER_SIMILARITY,
                display_name="Mean Answer Similarity",
                description=(
                    "The mean of the maximum similarity between the actual answer "
                    "sentences and the expected answer sentences."
                ),
                higher_is_better=True,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=True,
            ),
            commons.MetricMeta(
                key=METRIC_MIN_ANSWER_SIMILARITY,
                display_name="Min Answer Similarity",
                description=(
                    "The minimum similarity between the actual answer sentences "
                    "and the expected answer sentences."
                ),
                higher_is_better=True,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=False,
            ),
        ]
    )

    # COMPATIBILITY: LLM/RAG evaluation
    _rag = True
    _llm = True

    # GLOBAL: metric value for all dataset rows
    _global_explanation = True
    # LOCAL: metric value for particular row
    _local_explanation = True
    _explanation_types = [
        e10s.LlmEvalResultsExplanation,
        e10s.LlmHeatmapLeaderboardExplanation,
        e10s.WorkDirArchiveExplanation,
    ]

    _parameters = [
        evaluators.Evaluator._get_custom_param_metric_threshold(
            _metrics_meta.get_primary_metric()
        ),
        evaluators.Evaluator._PARAM_SAVE_LLM_RESULT,
        evaluators.Evaluator._PARAM_SENTENCE_LEVEL_METRICS,
    ]

    _keywords = [
        evaluators.KEYWORD_GPU_OPT,
        evaluators.KEYWORD_LLM,
        evaluators.KEYWORD_EVALUATES_LLM,
        evaluators.KEYWORD_EVALUATES_RAG,
        evaluators.KEYWORD_RQ_EA,
        evaluators.KEYWORD_RQ_AA,
        evaluators.KEYWORD_SR_11_7_CS,
        evaluators.KEYWORD_SR_11_7_OA,
        evaluators.KEYWORD_NIST_AI_RMF_S,
        evaluators.KEYWORD_PROBLEM_TYPE_IR,
        evaluators.KEYWORD_PROBLEM_TYPE_QA,
        evaluators.KEYWORD_EVALUATOR_ROLE_REGULATOR,
        evaluators.KEYWORD_ES_GENERATE,
        evaluators.KEYWORD_METHOD_SEMANTIC_SIMILARITY,
        evaluators.KEYWORD_METHOD_TYPE_DETERMINISTIC,
    ]

    _modules_needed_by_name = [h2o_sonar_config.DEP_SENTENCE_TRANSFORMERS]

    # models used by the evaluator
    _e_model_baai_bge = caching.MODEL_BAAI_BGE_SMALL_EN

    _brief_description = """Answer Semantic Similarity Evaluator assesses the semantic
resemblance between the generated answer and the expected answer (ground truth).
"""
    _description = evaluators.Evaluator._description_builder(
        brief=f"""{_brief_description}

**Method**:

- The answer similarity per sentence metrics are calculated as:

```math
answer similarity = {{max({{S(emb(a), emb(e)) : for all e in E}}): for all a in A}}
mean answer similarity = mean(answer similarity)
min answer similarity = min(answer similarity)
```

- Where:
    - `A` is the actual answer.
    - `emb(a)` is a vector embedding of the actual answer sentence.
    - `E` is the expected answer.
    - `emb(e)` is a vector embedding of the expected answer sentence.
    - `S(emb(a), emb(e))` is the 1 - cosine distance between the embedding of the actual
      answer sentence `a` and the expected answer sentence `e`.
- The evaluator uses **embeddings**
  [{_e_model_baai_bge}](https://huggingface.co/{_e_model_baai_bge}) (where
  BGE stands for "BAAI General Embedding" which refers to a suite of open-source text
  embedding models developed by the Beijing Academy of Artificial Intelligence (BAAI)).
""",
        metrics_meta=_metrics_meta,
        keywords=_keywords,
        parameters=_parameters,
        extra_insights=(
            "\n- The least similar actual answer sentence (in case that "
            "the output metric score is below the threshold)."
        ),
        leaderboard_type=e10s.LlmHeatmapLeaderboardExplanation.explanation_type(),
    )

    COL_INPUT = datasets.LlmDataset.KEY_INPUT
    COL_CONTEXT = datasets.LlmDataset.KEY_CONTEXT
    COL_EXPECTED_OUTPUT = datasets.LlmDataset.KEY_EXPECTED_OUTPUT
    COL_ACTUAL_OUTPUT = datasets.LlmDataset.KEY_ACTUAL_OUTPUT
    COL_MODEL = "model"
    COL_SCORE = "score"

    def __init__(self):
        evaluators.Evaluator.__init__(self)

        self.args = None
        self.problems = []
        self.log_name = "Answer semantic similarity per sentence"
        self._embedding_model = None

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        **evaluator_params,
    ) -> bool:
        evaluators.Evaluator.check_compatibility(self, params, **evaluator_params)

        if not HAS_NLTK:
            self.logger.warning(self._check_compatibility_pckg_err_msg("nltk"))
            return False
        if not HAS_SENTENCE_TRANSFORMERS:
            self.logger.warning(
                self._check_compatibility_pckg_err_msg("sentence_transformers")
            )
            return False

        if not self.models:
            self.logger.warning(
                f"{self.log_name}: no RAG/LLM models found for evaluation: "
                f"{[m.key for m in self.models]} - NOT COMPATIBLE"
            )
            return False

        if not evaluators.Evaluator._check_llm_dataset_compatibility(
            self, params=params, evaluator_keywords=self._keywords
        ):
            return False

        # check that at least one row has both actual answer and expected answer
        if not self._check_llm_dataset_field_presence(
            params=params,
            require_actual_answer=True,
            require_expected_answer=True,
        ):
            return False

        return True

    def setup(self, model, persistence, **kwargs):
        evaluators.Evaluator.setup(self, model, persistence, **kwargs)

        self._resolve_evaluator_params()

    def evaluate(self, llm_testset, explanations_types=None, **kwargs) -> list:
        save_llm_result = self.args.get(
            evaluators.Evaluator.PARAM_SAVE_LLM_RESULT,
            evaluators.Evaluator.DEFAULT_SAVE_LLM_RESULT,
        )

        return self._evaluate(
            llm_testset=llm_testset,
            save_llm_result=save_llm_result,
        )

    def _info_about_prompt_and_llm(self, row, html=None) -> str:
        key_2_evaluated_model = {m.key: m for m in self.models}
        if html is None:
            # enc/dec to make Cythonizer happy
            return (
                (
                    f"The following dataset row was skipped - it may impact model "
                    f"metric(s) values and influence its position in leaderboards"
                    f' - prompt: "{row.i}", '
                    f"LLM: {key_2_evaluated_model[row.model_key].llm_model_name}"
                )
                .encode("utf-8")
                .decode("utf-8")
            )

        html(
            "The following dataset row was skipped - it may impact the model "
            "metric(s) values and influence its position in leaderboards - prompt: "
        )
        with html.b():
            with html.i():
                html(f'"{row.i}"')
        html(", LLM: ")
        with html.code():
            html(key_2_evaluated_model[row.model_key].llm_model_name)
        return str(html)

    def _evaluate(
        self,
        llm_testset,
        save_llm_result: bool,
    ) -> list:
        #
        # EVALUATION
        #

        key_2_evaluated_model = {m.key: m for m in self.models}

        # LLM host: RAG or service
        llm_host = (
            commons.LlmModelHostType.RAG
            if isinstance(
                next(iter(key_2_evaluated_model.values())), models.ExplainableRagModel
            )
            else commons.LlmModelHostType.SERVICE
        )

        eval_results = self._calculate_metrics(llm_testset=llm_testset)

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

        # THRESHOLD for the metric
        metrics_threshold = self.args.get(
            evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
            evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
        )
        self._metrics_meta.set_threshold(metrics_threshold)

        # EXPLANATION: heatmap leaderboard
        heatmap_explanation = e10s.LlmHeatmapLeaderboardExplanation.from_eval_results(
            evaluator=self,
            eval_results=eval_results,
            metrics_meta=self._metrics_meta,
            key_2_evaluated_model=key_2_evaluated_model,
            llm_host=llm_host,
            display_name="LLM heatmap leaderboard",
            display_category=e10s.GlobalSummaryFeatImpExplanation.DISPLAY_CAT_LLM,
            logger=self.logger,
        )
        heatmap_explanation.add_json_format(
            threshold=self.args.get(
                evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
                self._metrics_meta.get_primary_metric().threshold,
            )
        )
        heatmap_explanation.add_markdown_format(
            sort_by_metric_id=self.METRIC_MEAN_ANSWER_SIMILARITY
        )
        heatmap_explanation.add_evalstudio_markdown_format(
            sort_by_metric_id=self.METRIC_MEAN_ANSWER_SIMILARITY
        )
        explanations.append(heatmap_explanation)

        # PROBLEMS for alerts and actionability
        self._diagnose_problems(
            eval_results=eval_results,
            key_2_evaluated_model=key_2_evaluated_model,
        )

        # INSIGHTS
        self._diagnose_insights(leaderboard_explanation=heatmap_explanation)

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
                            sort_by_metric_id=self.METRIC_MEAN_ANSWER_SIMILARITY
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

    @staticmethod
    def embed(embedding_model, text):
        txts = nltk.sent_tokenize(text)
        return txts, embedding_model.encode(txts)

    def _calculate_answer_similarity(self, embedding_model, row):
        this = AnswerSemanticSimilarityPerSentenceEvaluator

        explanation_type = e10s.GlobalHtmlFragmentExplanation.explanation_type()

        try:
            tok_ao, emb_ao = this.embed(embedding_model, row.actual_output)
        except Exception as ex:
            d_prefix = "Error during vector embedding of the actual output: "
            description = (
                f'{d_prefix}"{row.actual_output}". '
                f"{self._info_about_prompt_and_llm(row)}."
            )
            self.logger.error(f"{description} - {ex}\n{traceback.format_exc()}")

            html = airium.Airium()
            html(d_prefix)
            with html.b():
                with html.i():
                    html(f'"{row.actual_output}"')
            html(".")
            self._info_about_prompt_and_llm(row, html)
            html(".")

            self.add_problem(
                problems.ProblemAndAction(
                    description=description,
                    description_html=html,
                    evaluator_id=self.evaluator_id(),
                    problem_attrs={
                        problems.ProblemAndAction.ATTR_ROW_KEYS: [
                            (row.key, row.model_key)
                        ],
                        problems.ProblemAndAction.ATTR_TEST_CASE_KEYS: [row.key],
                        problems.ProblemAndAction.ATTR_EVALUATOR_NAME: (
                            self._display_name
                        ),
                    },
                    evaluator_name=self._display_name,
                    explanation_type=explanation_type,
                    explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
                    explanation_mime=f5s.HtmlFormat.mime,
                    resources=[],
                )
            )
            return float("NaN"), float("NaN"), None

        try:
            tok_eo, emb_eo = this.embed(embedding_model, row.expected_output)
        except Exception as ex:
            d_prefix = "Error during vector embedding of the expected answer: "
            description = (
                f'{d_prefix}"{row.expected_output}".'
                f" {self._info_about_prompt_and_llm(row)}."
            )
            self.logger.error(f"{description} - {ex}\n{traceback.format_exc()}")

            html = airium.Airium()
            html(d_prefix)
            with html.b():
                with html.i():
                    html(f'"{row.expected_output}"')
            html(".")
            self._info_about_prompt_and_llm(row, html)
            html(".")

            self.add_problem(
                problems.ProblemAndAction(
                    description=description,
                    description_html=html,
                    evaluator_id=self.evaluator_id(),
                    problem_attrs={
                        problems.ProblemAndAction.ATTR_ROW_KEYS: [
                            (row.key, row.model_key)
                        ],
                        problems.ProblemAndAction.ATTR_TEST_CASE_KEYS: [row.key],
                        problems.ProblemAndAction.ATTR_EVALUATOR_NAME: (
                            self._display_name
                        ),
                    },
                    evaluator_name=self._display_name,
                    explanation_type=explanation_type,
                    explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
                    explanation_mime=f5s.HtmlFormat.mime,
                    resources=[],
                )
            )

            return float("NaN"), float("NaN"), None

        # handle edge cases: empty actual or expected output embeddings
        if len(emb_ao) == 0 or len(emb_eo) == 0:
            d_prefix = "Empty embeddings detected because "
            if len(emb_ao) == 0 and len(emb_eo) == 0:
                detail = "both actual and expected output are empty"
            elif len(emb_ao) == 0:
                detail = "actual output is empty"
            else:
                detail = "expected output is empty"

            description = f"{d_prefix}{detail}. {self._info_about_prompt_and_llm(row)}."
            self.logger.warning(description)

            html = airium.Airium()
            html(d_prefix)
            with html.b():
                html(detail)
            html(". ")
            self._info_about_prompt_and_llm(row, html)
            html(".")

            self.add_problem(
                problems.ProblemAndAction(
                    description=description,
                    description_html=html,
                    actions_description=(
                        "The evaluator requires both the expected and actual answers "
                        "to be non-empty to calculate embeddings and function "
                        "correctly."
                    ),
                    evaluator_id=self.evaluator_id(),
                    problem_attrs={
                        problems.ProblemAndAction.ATTR_ROW_KEYS: [
                            (row.key, row.model_key)
                        ],
                        problems.ProblemAndAction.ATTR_TEST_CASE_KEYS: [row.key],
                        problems.ProblemAndAction.ATTR_EVALUATOR_NAME: (
                            self._display_name
                        ),
                    },
                    evaluator_name=self._display_name,
                    explanation_type=explanation_type,
                    explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
                    explanation_mime=f5s.HtmlFormat.mime,
                    resources=[],
                )
            )

            return float("NaN"), float("NaN"), None

        metric_results = []
        for i, a in enumerate(emb_ao):
            max_j = 0
            max_j_sim = float("-inf")
            for j, e in enumerate(emb_eo):
                s = 1 - nltk.cluster.cosine_distance(a, e)
                if s > max_j_sim:
                    max_j_sim = s
                    max_j = j

            metric_results.append(((i, max_j), max_j_sim))

        metrics = np.array([v for _, v in metric_results])
        mean_answer_similarity = np.mean(metrics)
        min_answer_similarity = np.min(metrics)
        actual_output_meta = None
        if self.args.get(
            self.PARAM_SENTENCE_LEVEL_METRICS, self.DEFAULT_SENTENCE_LEVEL_METRICS
        ):
            text_fragments = []
            for (i, j), val in metric_results:
                text_fragments.append(
                    tokenization.TextFragment(
                        text=tok_ao[i],
                        metrics={self.METRIC_MEAN_ANSWER_SIMILARITY: val},
                        meta={},
                    )
                )
            actual_output_meta = tokenization.Tokenization(
                tokenization=tokenization.TOKENIZATION_TYPE_S_PUNKT,
                data=text_fragments,
            )
        return mean_answer_similarity, min_answer_similarity, actual_output_meta

    def _calculate_metrics(self, llm_testset):
        llm_dataset = datasets.LlmDataset.from_datatable_dict(llm_testset.to_dict())

        # evaluator runs only ONE metric at a time
        self.report_progress(0.01, "Configuring metrics...")

        eval_results = datasets.LlmEvalResults()

        caching.cache_nltk_punkt(self.logger)

        device = h2o_sonar_config.config.resolve_gpu_cpu_device(result_format="str")
        with resource_mgmt.PytorchModelLifeCycleManager(
            sentence_transformers.SentenceTransformer(
                AnswerSemanticSimilarityPerSentenceEvaluator._e_model_baai_bge,
                device=device,
                revision=caching.REVISIONS_FOR_MODEL.get(
                    AnswerSemanticSimilarityPerSentenceEvaluator._e_model_baai_bge,
                    "main",
                ),
            )
        ) as embedding_model:
            # for every test case run metric (row by row)
            for e, r in enumerate(llm_dataset.inputs):
                # progress
                self.report_progress(
                    progress=progress_utils.ProgressCallbackContext.progress_for_steps(
                        e + 1, len(llm_dataset.inputs)
                    ),
                    message=evaluators.Evaluator._eval_row_progress_msg(
                        metric_name=self.METRIC_MEAN_ANSWER_SIMILARITY,
                        device=device,
                        row=e + 1,
                        total_rows=len(llm_dataset.inputs),
                    ),
                )

                # handle actual answer retrieval error ~ RAG/LLM client crash
                if evaluators.Evaluator._is_internal_err_answer(r.actual_output):
                    # set WORST metrics values
                    eval_results.add_result(
                        datasets.LlmEvalResults.LlmEvalResultRow(
                            dataset_row=r,
                            metrics={
                                self.METRIC_MEAN_ANSWER_SIMILARITY: 0.0,
                                self.METRIC_MIN_ANSWER_SIMILARITY: 0.0,
                            },
                            actual_output_meta=[
                                tokenization.Tokenization(
                                    tokenization=tokenization.TOKENIZATION_TYPE_F,
                                    data=[],
                                )
                            ],
                        )
                    )
                    continue

                # handle valid actual answer
                mean_as, min_as, result_tokenization = (
                    self._calculate_answer_similarity(embedding_model, r)
                )
                if mean_as == mean_as and min_as == min_as:  # not NaN
                    eval_results.add_result(
                        datasets.LlmEvalResults.LlmEvalResultRow(
                            dataset_row=r,
                            metrics={
                                self.METRIC_MEAN_ANSWER_SIMILARITY: mean_as,
                                self.METRIC_MIN_ANSWER_SIMILARITY: min_as,
                            },
                            actual_output_meta=(
                                [result_tokenization] if result_tokenization else []
                            ),
                        )
                    )
                else:
                    # IMPROVE: create a problem for the row
                    self.logger.warning(
                        f"{self.log_name}: Row {e + 1} - "
                        "Answer semantic similarity metrics are NaN. "
                        f"Actual output: {r.actual_output}, "
                        f"Expected output: {r.expected_output}."
                    )
                    eval_results.add_result(
                        datasets.LlmEvalResults.LlmEvalResultRow(
                            dataset_row=r,
                            metrics={
                                self.METRIC_MEAN_ANSWER_SIMILARITY: 0.0,
                                self.METRIC_MIN_ANSWER_SIMILARITY: 0.0,
                            },
                            actual_output_meta=[
                                tokenization.Tokenization(
                                    tokenization=tokenization.TOKENIZATION_TYPE_F,
                                    data=[],
                                )
                            ],
                        )
                    )

        return eval_results

    def _diagnose_problems(
        self,
        eval_results: datasets.LlmEvalResults,
        key_2_evaluated_model: dict,
    ):
        # perturbation flips
        self._diagnose_perturbation_problems(
            eval_results=eval_results,
            key_2_evaluated_model=key_2_evaluated_model,
        )

    def _diagnose_insights(
        self, leaderboard_explanation: e10s.LlmHeatmapLeaderboardExplanation
    ):
        pass

    def get_result(
        self,
    ) -> results.LeaderboardResult:
        return results.LeaderboardResult(
            persistence=self.persistence,
            explainer_id=AnswerSemanticSimilarityPerSentenceEvaluator.explainer_id(),
            h2o_sonar_config=self.config,
            logger=self.logger,
            explanation=e10s.LlmHeatmapLeaderboardExplanation,
            explanation_format=f5s.CustomJsonFormat,
        )
