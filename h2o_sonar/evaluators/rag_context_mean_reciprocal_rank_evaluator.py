# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import math
import traceback

import airium

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


class MeanReciprocalRankEvaluator(evaluators.Evaluator):
    """Context Mean Reciprocal Rank (MRR) evaluator assesses the retrieval quality
    of a RAG system.

    """

    _display_name = "Context mean reciprocal rank"
    _tagline = "Assess mean reciprocal rank of the retrieved context."

    METRIC_MRR = "mean_reciprocal_rank"

    PARAM_RELEVANT_CHUNK_THRESHOLD = "mrr_relevant_chunk_threshold"
    PARAM_RELEVANT_CHUNK_THRESHOLD_DEFAULT = 0.7

    PARAM_RELEVANT_CHUNK_OOR_IDX = "mrr_relevant_chunk_oor_idx"
    PARAM_RELEVANT_CHUNK_OOR_IDX_DEFAULT = 10

    _metrics_meta = commons.MetricsMeta(
        metrics=[
            commons.MetricMeta(
                key=METRIC_MRR,
                display_name="Mean Reciprocal Rank",
                description=(
                    "Mean reciprocal rank metric score given the first relevant "
                    "retrieved context chunk."
                ),
                higher_is_better=True,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=False,
            ),
        ]
    )

    # COMPATIBILITY: LLM/RAG evaluation
    _rag = True

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
        evaluators.EvaluatorParam(
            param_name=PARAM_RELEVANT_CHUNK_THRESHOLD,
            description=(
                # IMPROVE formula:
                #   0.5 * S(ctx chunk, query) + f"0.5 * S(ctx chunk, expected answer)
                f"Threshold for the relevance score of the retrieved context chunk. "
                f"The relevance score is calculated as: S(ctx chunk, query). "
                f"The threshold value should be between 0.0 and 1.0 "
                f"(default: {PARAM_RELEVANT_CHUNK_THRESHOLD_DEFAULT})."
            ),
            param_type=commons.EvaluatorParamType.float,
            default_value=PARAM_RELEVANT_CHUNK_THRESHOLD_DEFAULT,
            src=evaluators.EvaluatorParam.SRC_EVALUATOR_PARAMS,
        ),
        evaluators.EvaluatorParam(
            param_name=PARAM_RELEVANT_CHUNK_OOR_IDX,
            description=(
                f"Threshold for the index of the relevant chunk in the retrieved "
                f"context. If the first relevant chunk is at an index higher than this "
                f"value, it is considered out of range and the reciprocal rank for "
                f"that query is set to 0.0. The value should be a positive integer "
                f"(default: {PARAM_RELEVANT_CHUNK_OOR_IDX_DEFAULT})."
            ),
            param_type=commons.EvaluatorParamType.int,
            default_value=PARAM_RELEVANT_CHUNK_OOR_IDX_DEFAULT,
            src=evaluators.EvaluatorParam.SRC_EVALUATOR_PARAMS,
        ),
        evaluators.Evaluator._get_custom_param_metric_threshold(
            _metrics_meta.get_primary_metric()
        ),
        evaluators.Evaluator._PARAM_SAVE_LLM_RESULT,
    ]

    _keywords = [
        evaluators.KEYWORD_GPU_OPT,
        evaluators.KEYWORD_LLM,
        evaluators.KEYWORD_EVALUATES_RAG,
        evaluators.KEYWORD_RQ_RC,
        evaluators.KEYWORD_RQ_P,
        # IMPROVE formula: evaluators.KEYWORD_RQ_EA,
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

    _brief_description = """Mean Reciprocal Rank Evaluator assesses the performance of
    the retrieval component of a RAG system by measuring the average of the reciprocal
    ranks of the first relevant document retrieved for a set of queries. It helps to
    evaluate how effectively the retrieval component of a RAG system provides relevant
    context for generating accurate and contextually appropriate responses."""
    # IMPROVE formula:
    # relevance score = 0.5 * S(ctx chunk, query) + 0.5 * S(ctx chunk, expected answer)
    _description = evaluators.Evaluator._description_builder(
        brief=f"""{_brief_description}

**Method**:

- The evaluator brings mean reciprocal rank (MRR) metric.
- Relevant retrieved context chunk is defined as the chunk that contains the answer
  to the query. The relevance score is calculated as:

```math
relevance score = max( S(ctx chunk sentence, query) )
```

- Where S(a, b) is the similarity score between texts a and b, calculated as
  1 - cosine distance between their vector embeddings.
- For a single query, the reciprocal rank is the inverse of the rank of the first
  relevant document retrieved:

```math
reciprocal rank = 1 / rank of the first chunk with relevance score >= threshold
```

- If the first relevant document is at rank 1, the reciprocal rank is 1.0 (best
  score). If no relevant document is retrieved, the reciprocal rank is 0.0 (worst
  score). If the first relevant document is at rank 5, the reciprocal rank
  is 1 / 5 i.e. 0.2.
- Threshold for the relevance score is set to 0.7 by default, but can be
  adjusted using the evaluator parameter.
- Mean reciprocal rank (MRR) is the average of the reciprocal ranks across all
  queries:

```math
mean reciprocal rank = sum(reciprocal rank for query in queries) / |queries|
```

- The evaluator uses **embeddings**
  [{_e_model_baai_bge}](https://huggingface.co/{_e_model_baai_bge}) (where BGE
  stands for "BAAI General Embedding" which refers to a suite of open-source text
  embedding models developed by the Beijing Academy of Artificial Intelligence (BAAI)).
""",
        metrics_meta=_metrics_meta,
        keywords=_keywords,
        parameters=_parameters,
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
        self.log_name = "MRR"

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        **evaluator_params,
    ) -> bool:
        evaluators.Evaluator.check_compatibility(self, params, **evaluator_params)

        if not HAS_SENTENCE_TRANSFORMERS:
            self.logger.error(
                self._check_compatibility_pckg_err_msg("sentence_transformers")
            )
            return False
        if not HAS_NLTK:
            self.logger.error(self._check_compatibility_pckg_err_msg("nltk"))
            return False

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
        save_llm_result = self.args.get(
            evaluators.Evaluator.PARAM_SAVE_LLM_RESULT,
            evaluators.Evaluator.DEFAULT_SAVE_LLM_RESULT,
        )

        return self._evaluate(
            llm_testset=llm_testset,
            save_llm_result=save_llm_result,
        )

    def _info_about_prompt_and_llm(self, row, html=None):
        key_2_evaluated_model = {m.key: m for m in self.models}
        if html is None:
            return (
                f'Additional details - prompt: "{row.i}", '
                f"LLM: {key_2_evaluated_model[row.model_key].llm_model_name}"
            )

        html("Additional details - prompt: ")
        with html.b():
            with html.i():
                html(f'"{row.i}"')
        html(", LLM: ")
        with html.code():
            html(key_2_evaluated_model[row.model_key].llm_model_name)

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

        eval_results = self._calculate_reciprocal_ranks(llm_testset=llm_testset)

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
        heatmap_explanation.add_markdown_format(sort_by_metric_id=self.METRIC_MRR)
        heatmap_explanation.add_evalstudio_markdown_format(
            sort_by_metric_id=self.METRIC_MRR
        )
        explanations.append(heatmap_explanation)

        # EXPLANATION: HTML fragment
        if self.config and self.config.create_html_representations:
            try:
                html_explanation = e10s.GlobalHtmlFragmentExplanation(
                    evaluator=self,
                    display_name="LLM heatmap leaderboard as HTML",
                    display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
                )
                html_explanation.add_html_format(
                    str(heatmap_explanation.as_html(sort_by_metric_id=self.METRIC_MRR)),
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
    def split_sentences(text: str) -> list[str]:
        sentences = nltk.tokenize.sent_tokenize(text)
        sentences = [sent for sent in sentences if len(sent) > 10]
        return sentences

    def _calculate_reciprocal_rank(
        self,
        model,
        row: datasets.LlmDataset.LlmDatasetRow,
        chunk_relevance_threshold: float,
        chunk_oor_idx_threshold: int = 10,
    ):
        import nltk

        # handle empty contexts
        if not row.context or len(row.context) == 0:
            return {
                self.METRIC_MRR: 0.0,
            }

        explanation_type = e10s.GlobalHtmlFragmentExplanation.explanation_type()

        chunk_embedded_sentences = []
        for e, c in enumerate(row.context):
            # CHUNK 2 sentences w/ embedding
            try:
                chunk_embedded_sentences.append(model.encode(self.split_sentences(c)))
            except Exception as ex:
                d_prefix = (
                    "Error during the vector embedding calculation of the retrieved "
                    "context chunk - "
                )
                description = (
                    f'{d_prefix}"{c}". {self._info_about_prompt_and_llm(row)}.'
                )
                self.logger.error(
                    f"{self.log_name}: {description} - {ex}\n{traceback.format_exc()}"
                )

                html = airium.Airium()
                html(d_prefix)
                with html.b():
                    with html.i():
                        html(f'"{c}"')
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
                return float("NaN")
        # QUESTION 2 embedding
        try:
            question_embedded = model.encode(row.i)
        except Exception as ex:
            d_prefix = "Error during vector embedding calculation of the query - "
            description = (
                f'{d_prefix}"{row.i}". {self._info_about_prompt_and_llm(row)}.'
            )
            self.logger.error(
                f"{self.log_name}: {description} - {ex}\n{traceback.format_exc()}"
            )

            html = airium.Airium()
            html(d_prefix)
            with html.b():
                with html.i():
                    html(f'"{row.i}"')
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
            return float("NaN")

        chunk_embedded_sentences = [
            ces for ces in chunk_embedded_sentences if len(ces) > 0
        ]

        if len(chunk_embedded_sentences) == 0 or len(question_embedded) == 0:
            d_prefix = (
                "Embedding of the retrieved context chunks or the query is empty - "
                "the evaluation dataset row will be skipped. "
            )
            description = f"{d_prefix}{self._info_about_prompt_and_llm(row)}."

            html = airium.Airium()
            html(d_prefix)
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
            return float("NaN")

        # MRR calculation
        context_chunk_relevancy = [
            max(
                1 - nltk.cluster.cosine_distance(question_embedded, ctx_sent)
                for ctx_sent in context_chunk
            )
            for context_chunk in chunk_embedded_sentences
        ]

        # find the first relevant chunk
        first_relevant_chunk_idx = 0
        for e, r in enumerate(context_chunk_relevancy):
            if r >= chunk_relevance_threshold:
                first_relevant_chunk_idx = e + 1
                break
        first_relevant_chunk_idx = (
            first_relevant_chunk_idx
            if (first_relevant_chunk_idx <= chunk_oor_idx_threshold)
            else 0
        )

        reciprocal_rank_metric_score = (
            1.0 / float(first_relevant_chunk_idx)
            if first_relevant_chunk_idx > 0
            else 0.0
        )

        return {
            self.METRIC_MRR: reciprocal_rank_metric_score,
        }

    def _calculate_reciprocal_ranks(self, llm_testset):
        self.report_progress(0.01, "Configuring metrics...")

        # ensure `punkt` is downloaded
        caching.cache_nltk_punkt(self.logger)

        llm_dataset = datasets.LlmDataset.from_datatable_dict(llm_testset.to_dict())
        eval_results = datasets.LlmEvalResults()
        model = caching.MODEL_BAAI_BGE_SMALL_EN
        device = h2o_sonar_config.config.resolve_gpu_cpu_device(result_format="str")
        chunk_relevance_threshold = self.args.get(
            self.PARAM_RELEVANT_CHUNK_THRESHOLD,
            self.PARAM_RELEVANT_CHUNK_THRESHOLD_DEFAULT,
        )
        if chunk_relevance_threshold < 0.0 or chunk_relevance_threshold > 1.0:
            chunk_relevance_threshold = self.PARAM_RELEVANT_CHUNK_THRESHOLD_DEFAULT
        chunk_oor_idx_threshold = self.args.get(
            self.PARAM_RELEVANT_CHUNK_OOR_IDX,
            self.PARAM_RELEVANT_CHUNK_OOR_IDX_DEFAULT,
        )
        if chunk_oor_idx_threshold <= 0:
            chunk_oor_idx_threshold = self.PARAM_RELEVANT_CHUNK_OOR_IDX_DEFAULT
        with resource_mgmt.PytorchModelLifeCycleManager(
            sentence_transformers.SentenceTransformer(
                model,
                device=device,
                revision=caching.REVISIONS_FOR_MODEL.get(model, "main"),
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
                        metric_name=self.METRIC_MRR,
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
                                self.METRIC_MRR: 0.0,
                            },
                        )
                    )
                    continue

                reciprocal_rank = self._calculate_reciprocal_rank(
                    model=embedding_model,
                    row=r,
                    chunk_relevance_threshold=chunk_relevance_threshold,
                    chunk_oor_idx_threshold=chunk_oor_idx_threshold,
                )

                def _is_nan(*xs):
                    for x in xs:
                        if x != x:
                            return True
                    return False

                # skip NaN results
                if isinstance(reciprocal_rank, float) and math.isnan(reciprocal_rank):
                    continue

                if not _is_nan(reciprocal_rank.values()):  # not NaN
                    eval_results.add_result(
                        datasets.LlmEvalResults.LlmEvalResultRow(
                            dataset_row=r,
                            metrics=reciprocal_rank,
                        )
                    )

        return eval_results

    def get_result(
        self,
    ) -> results.LeaderboardResult:
        return results.LeaderboardResult(
            persistence=self.persistence,
            explainer_id=MeanReciprocalRankEvaluator.explainer_id(),
            h2o_sonar_config=self.config,
            logger=self.logger,
            explanation=e10s.LlmHeatmapLeaderboardExplanation,
            explanation_format=f5s.CustomJsonFormat,
        )
