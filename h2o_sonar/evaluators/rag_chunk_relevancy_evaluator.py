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
    import sentence_transformers

    HAS_REQUIRED_PACKAGES = True
except ImportError:
    HAS_REQUIRED_PACKAGES = False


class ContextChunkRelevancyEvaluator(evaluators.Evaluator):
    _display_name = "Context relevancy (soft recall and precision)"
    _tagline = "Assess precision and relevancy of the retrieved context."

    METRIC_RECALL_RELEVANCY = "recall_relevancy"
    METRIC_PRECISION_RELEVANCY = "precision_relevancy"

    _metrics_meta = commons.MetricsMeta(
        metrics=[
            commons.MetricMeta(
                key=METRIC_RECALL_RELEVANCY,
                display_name="Recall Relevancy",
                description="Maximum retrieved context chunk relevancy.",
                higher_is_better=True,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=False,
            ),
            commons.MetricMeta(
                key=METRIC_PRECISION_RELEVANCY,
                display_name="Precision Relevancy",
                description="Average retrieved context chunk relevancy.",
                higher_is_better=True,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=True,
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

    _brief_description = """Context Relevancy Evaluator assesses the context relevancy
in a Retrieval Augmented Generation (RAG) pipeline.
Context Relevancy (Soft Recall and Precision) Evaluator measures the relevancy
of the retrieved context based on the question and context sentences and produces
two metrics - **precision** and **recall relevancy**."""
    _description = evaluators.Evaluator._description_builder(
        brief=f"""{_brief_description}

**Method**:

- The evaluator brings two metrics calculated as:

```math
chunk context relevancy(ch) = max( {{S(emb(q), emb(s)): for all s in ch}} )

recall relevancy = max( {{chunk context relevancy(ch): for all ch in rc}} )
precision relevancy = avg( {{chunk context relevancy(ch): for all ch in rc}} )
```

- Where:
    - `rc` is the retrieved context.
    - `ch` is a chunk of the retrieved context.
    - `emb(s)` is a vector embedding of the retrieved context chunk sentence.
    - `emb(q)` is a vector embedding of the query.
    - `S(question, s)` is the 1 - cosine distance between the `question` and
      the retrieved context sentence `s`.
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
        self.log_name = "ContextChunkRelevancy"

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        **evaluator_params,
    ) -> bool:
        evaluators.Evaluator.check_compatibility(self, params, **evaluator_params)

        if not HAS_REQUIRED_PACKAGES:
            self.logger.warning(
                self._check_compatibility_pckg_err_msg(
                    ["nltk", "sentence-transformers"]
                )
            )
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
            sort_by_metric_id=self.METRIC_PRECISION_RELEVANCY
        )
        heatmap_explanation.add_evalstudio_markdown_format(
            sort_by_metric_id=self.METRIC_PRECISION_RELEVANCY
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
                    str(
                        heatmap_explanation.as_html(
                            sort_by_metric_id=self.METRIC_PRECISION_RELEVANCY
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

    # Split docs into sentences
    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """Split the data into sentences"""
        sentences = nltk.tokenize.sent_tokenize(text)
        sentences = [sent for sent in sentences if len(sent) > 10]
        return sentences

    def _calculate_context_relevancy(self, model, row):
        explanation_type = e10s.GlobalHtmlFragmentExplanation.explanation_type()

        context_embedded_sentences = []
        for c in row.context:
            try:
                context_embedded_sentences.append(model.encode(self.split_sentences(c)))
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
        try:
            query_embedded = model.encode(row.i)
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

        context_embedded_sentences = [
            ces for ces in context_embedded_sentences if len(ces) > 0
        ]

        if len(context_embedded_sentences) == 0 or len(query_embedded) == 0:
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

        context_chunk_relevancy = [
            max(
                1 - nltk.cluster.cosine_distance(query_embedded, ctx_sent)
                for ctx_sent in context_chunk
            )
            for context_chunk in context_embedded_sentences
        ]

        least_relevant_chunk_idx = min(
            range(len(context_chunk_relevancy)),
            key=lambda i: context_chunk_relevancy[i],
        )
        metric_result = context_chunk_relevancy[least_relevant_chunk_idx]

        threshold = self.args.get(
            evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
            self._metrics_meta.get_primary_metric().threshold,
        )
        if metric_result < threshold:
            least_rel_chunk = row.context[least_relevant_chunk_idx]

            d_prefix = (
                f"The least relevant context chunk identified by "
                f"the {self._display_name} evaluator is: "
            )

            description = (
                f'{d_prefix}"{least_rel_chunk}". '
                f"{self._info_about_prompt_and_llm(row)}."
            )

            html = airium.Airium()
            html(d_prefix)
            with html.b():
                with html.i():
                    html(f'"{least_rel_chunk}"')
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
                        # input dataset ~ test lab ~ key is the test case key
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

        return {
            self.METRIC_RECALL_RELEVANCY: max(context_chunk_relevancy),
            self.METRIC_PRECISION_RELEVANCY: (
                sum(context_chunk_relevancy) / len(context_chunk_relevancy)
            ),
        }

    def _calculate_metrics(self, llm_testset):
        # ensure `punkt` is downloaded
        caching.cache_nltk_punkt(self.logger)

        llm_dataset = datasets.LlmDataset.from_datatable_dict(llm_testset.to_dict())

        # evaluator runs only ONE metric at a time
        self.report_progress(0.01, "Configuring metrics...")

        eval_results = datasets.LlmEvalResults()

        model = caching.MODEL_BAAI_BGE_SMALL_EN

        device = h2o_sonar_config.config.resolve_gpu_cpu_device(result_format="str")
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
                        metric_name=(
                            f"{self.METRIC_RECALL_RELEVANCY} and "
                            f"{self.METRIC_PRECISION_RELEVANCY}"
                        ),
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
                                self.METRIC_RECALL_RELEVANCY: 0.0,
                                self.METRIC_PRECISION_RELEVANCY: 0.0,
                            },
                        )
                    )
                    continue

                result = self._calculate_context_relevancy(embedding_model, r)

                def _is_nan(*xs):
                    for x in xs:
                        if x != x:
                            return True
                    return False

                # skip NaN results
                if isinstance(result, float) and math.isnan(result):
                    continue

                if not _is_nan(result.values()):  # not NaN
                    eval_results.add_result(
                        datasets.LlmEvalResults.LlmEvalResultRow(
                            dataset_row=r,
                            metrics=result,
                        )
                    )

        return eval_results

    def get_result(
        self,
    ) -> results.LeaderboardResult:
        return results.LeaderboardResult(
            persistence=self.persistence,
            explainer_id=ContextChunkRelevancyEvaluator.explainer_id(),
            h2o_sonar_config=self.config,
            logger=self.logger,
            explanation=e10s.LlmHeatmapLeaderboardExplanation,
            explanation_format=f5s.CustomJsonFormat,
        )
