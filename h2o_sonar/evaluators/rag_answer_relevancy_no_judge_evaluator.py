# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
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


class RagAnswerRelevancyNoJudgeEvaluator(evaluators.Evaluator):
    _display_name = "Answer relevancy (sentence similarity)"
    _tagline = (
        "Evaluate the pertinence of actual answers to questions based on their "
        "similarity."
    )

    METRIC_ANSWER_RELEVANCY = "answer_relevancy"

    _metrics_meta = commons.MetricsMeta(
        metrics=[
            commons.MetricMeta(
                key=METRIC_ANSWER_RELEVANCY,
                display_name="Answer Relevancy",
                description=(
                    "Answer Relevancy metric determines whether the RAG/LLM outputs "
                    "relevant information by comparing the actual answer sentences "
                    "to the question."
                ),
                higher_is_better=True,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=True,
            )
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
        evaluators.KEYWORD_RQ_P,
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
        evaluators.KEYWORD_CAP_AH,
    ]

    _modules_needed_by_name = [h2o_sonar_config.DEP_SENTENCE_TRANSFORMERS]

    # models used by the evaluator
    _e_model_baai_bge = caching.MODEL_BAAI_BGE_SMALL_EN

    _brief_description = """The Answer Relevancy (Sentence Similarity) Evaluator
assesses how relevant the actual answer is by computing the similarity between
the question and the actual answer sentences."""
    _description = evaluators.Evaluator._description_builder(
        brief=f"""{_brief_description}

**Method**:

- The metric is calculated as maximum similarity between the question and the actual
  answer sentences:

```math
answer relevancy = max( {{S(emb(question), emb(a)): for all a in actual answer}} )
```

- Where:
    - `A` is the actual answer.
    - `a` is a sentence in the actual answer.
    - `emb(a)` is a vector embedding of the actual answer sentence.
    - `emb(question)` is a vector embedding of the question.
    - `S(q, a)` is the 1 - cosine distance between the question `q` and the actual
      answer sentence `a`.
- The evaluator uses **embeddings**
  [{_e_model_baai_bge}](https://huggingface.co/{_e_model_baai_bge}) (where
  BGE stands for "BAAI General Embedding" which refers to a suite of open-source text
  embedding models developed by the Beijing Academy of Artificial Intelligence(BAAI)).
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
        self.log_name = "AnswerRelevancySentenceSimilarity"

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        **evaluator_params,
    ) -> bool:
        evaluators.Evaluator.check_compatibility(self, params, **evaluator_params)

        if not HAS_NLTK:
            self.logger.error(self._check_compatibility_pckg_err_msg("nltk"))
            return False
        if not HAS_SENTENCE_TRANSFORMERS:
            self.logger.error(
                self._check_compatibility_pckg_err_msg("sentence_transformers")
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

        # check that at least one row has actual answer
        if not self._check_llm_dataset_field_presence(
            params=params,
            require_actual_answer=True,
            require_expected_answer=False,
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
            return (
                (
                    f'Additional details - prompt: "{row.i}", '
                    f"LLM: {key_2_evaluated_model[row.model_key].llm_model_name}"
                )
                .encode("utf-8")
                .decode("utf-8")
            )

        html("Additional details - prompt: ")
        with html.b():
            with html.i():
                html(f'"{row.i}"')
        html(", LLM: ")
        with html.code():
            html(key_2_evaluated_model[row.model_key].llm_model_name)

        return ""

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
            sort_by_metric_id=self.METRIC_ANSWER_RELEVANCY
        )
        heatmap_explanation.add_evalstudio_markdown_format(
            sort_by_metric_id=self.METRIC_ANSWER_RELEVANCY
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
                            sort_by_metric_id=self.METRIC_ANSWER_RELEVANCY
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

    # split docs into sentences
    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """Split the data into sentences"""
        sentences = nltk.tokenize.sent_tokenize(text)
        sentences = [sent for sent in sentences if len(sent) > 10]
        return sentences

    def _calculate_answer_relevancy(self, model, row):
        explanation_type = e10s.GlobalHtmlFragmentExplanation.explanation_type()

        output_sentences = self.split_sentences(row.actual_output)
        try:
            output_embedded_sentences = model.encode(output_sentences)
        except Exception as ex:
            d_prefix = (
                "Error during the vector embedding calculation of the actual answer - "
            )
            description = (
                f'{d_prefix}"{output_sentences}". '
                f"{self._info_about_prompt_and_llm(row)}."
            )
            self.logger.error(
                f"{self.log_name}: {description} - {ex}\n{traceback.format_exc()}"
            )

            html = airium.Airium()
            html(d_prefix)
            with html.b():
                with html.i():
                    html(f'"{output_sentences}"')
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
            return float("NaN")

        query_embedding = model.encode(row.i)
        sentence_relevancy = [
            (1 - nltk.cluster.cosine_distance(query_embedding, sent))
            for sent in output_embedded_sentences
        ]
        actual_output_meta = None
        if self.args.get(
            self.PARAM_SENTENCE_LEVEL_METRICS, self.DEFAULT_SENTENCE_LEVEL_METRICS
        ):
            all_actual_answer_sentences = nltk.sent_tokenize(row.actual_output)
            text_fragments = []
            for aa in all_actual_answer_sentences:
                try:
                    i = output_sentences.index(aa)
                    text_fragments.append(
                        tokenization.TextFragment(
                            text=aa,
                            metrics={
                                self.METRIC_ANSWER_RELEVANCY: sentence_relevancy[i]
                            },
                            meta={},
                        )
                    )
                except ValueError:
                    # short sentence -> without metric
                    text_fragments.append(
                        tokenization.TextFragment(text=aa, metrics={}, meta={})
                    )

            actual_output_meta = tokenization.Tokenization(
                tokenization=tokenization.TOKENIZATION_TYPE_S_PUNKT, data=text_fragments
            )

        return max(sentence_relevancy), actual_output_meta

    def _calculate_metrics(self, llm_testset):
        # ensure `punkt` is downloaded
        caching.cache_nltk_punkt(self.logger)

        llm_dataset = datasets.LlmDataset.from_datatable_dict(llm_testset.to_dict())

        # evaluator runs only ONE metric at a time
        self.report_progress(0.01, "Configuring metrics...")

        eval_results = datasets.LlmEvalResults()

        device = h2o_sonar_config.config.resolve_gpu_cpu_device(result_format="str")
        with resource_mgmt.PytorchModelLifeCycleManager(
            sentence_transformers.SentenceTransformer(
                RagAnswerRelevancyNoJudgeEvaluator._e_model_baai_bge,
                device=device,
                revision=caching.REVISIONS_FOR_MODEL.get(
                    RagAnswerRelevancyNoJudgeEvaluator._e_model_baai_bge, "main"
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
                        metric_name=self.METRIC_ANSWER_RELEVANCY,
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
                            metrics={self.METRIC_ANSWER_RELEVANCY: 0.0},
                        )
                    )
                    continue

                # handle empty actual output
                if not r.actual_output or not isinstance(r.actual_output, str):
                    self.logger.warning(
                        f"{self.log_name}: Row {e + 1} - Empty actual output detected. "
                        f"Setting worst metric value."
                    )
                    # set WORST metrics values
                    eval_results.add_result(
                        datasets.LlmEvalResults.LlmEvalResultRow(
                            dataset_row=r,
                            metrics={self.METRIC_ANSWER_RELEVANCY: 0.0},
                        )
                    )
                    continue

                result, actual_output_meta = self._calculate_answer_relevancy(
                    embedding_model, r
                )
                if result == result:  # not NaN
                    eval_results.add_result(
                        datasets.LlmEvalResults.LlmEvalResultRow(
                            dataset_row=r,
                            metrics={self.METRIC_ANSWER_RELEVANCY: result},
                            actual_output_meta=(
                                [actual_output_meta] if actual_output_meta else []
                            ),
                        )
                    )

        return eval_results

    def get_result(
        self,
    ) -> results.LeaderboardResult:
        return results.LeaderboardResult(
            persistence=self.persistence,
            explainer_id=RagAnswerRelevancyNoJudgeEvaluator.explainer_id(),
            h2o_sonar_config=self.config,
            logger=self.logger,
            explanation=e10s.LlmHeatmapLeaderboardExplanation,
            explanation_format=f5s.CustomJsonFormat,
        )
