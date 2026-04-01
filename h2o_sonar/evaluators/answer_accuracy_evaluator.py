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


class AnswerAccuracyEvaluator(evaluators.Evaluator):
    _display_name = "Answer accuracy (semantic similarity)"
    _tagline = (
        "Evaluate actual answers by comparing them to expected answers using "
        "semantic similarity."
    )

    METRIC_ANSWER_ACCURACY = "answer_accuracy"

    _metrics_meta = commons.MetricsMeta(
        metrics=[
            commons.MetricMeta(
                key=METRIC_ANSWER_ACCURACY,
                display_name="Answer Accuracy",
                description=(
                    "Answer Accuracy metric determines how closely the actual answer "
                    "matches the expected answer by **comparing** the actual answer "
                    "sentences to the expected answer sentences using semantic "
                    "similarity."
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

    SHORT_STRING_METRIC_NORMALIZED_EDIT_DISTANCE = "normalized_edit_distance"
    SHORT_STRING_METRIC_EXACT_MATCH = "exact_match"
    SHORT_STRING_METRIC_TOKEN_JACCARD = "token_jaccard"
    SHORT_STRING_METRIC_EMBEDDINGS = "embeddings"
    DEFAULT_SHORT_STRING_METRIC = SHORT_STRING_METRIC_NORMALIZED_EDIT_DISTANCE

    PARAM_SHORT_STRING_METRIC = "short_string_metric"
    PARAM_SHORT_STRING_THRESHOLD = "short_string_threshold"
    DEFAULT_SHORT_STRING_THRESHOLD = 10

    _parameters = [
        evaluators.Evaluator._get_custom_param_metric_threshold(
            _metrics_meta.get_primary_metric()
        ),
        evaluators.Evaluator._PARAM_SAVE_LLM_RESULT,
        evaluators.Evaluator._PARAM_SENTENCE_LEVEL_METRICS,
        evaluators.Evaluator._get_custom_param_min_test_case(),
        evaluators.EvaluatorParam(
            param_name=PARAM_SHORT_STRING_METRIC,
            description=(
                "Metric to use for short strings (length <= short_string_threshold). "
                f"Options: '{SHORT_STRING_METRIC_NORMALIZED_EDIT_DISTANCE}' (default, "
                "good for handling typos and case differences), "
                f"'{SHORT_STRING_METRIC_EXACT_MATCH}' (strict matching, good for "
                f"Yes/No answers), '{SHORT_STRING_METRIC_TOKEN_JACCARD}' "
                f"(token overlap, good for short multi-word phrases), "
                f"'{SHORT_STRING_METRIC_EMBEDDINGS}' (force embeddings anyway, may "
                f"result in NaN for very short strings)"
            ),
            param_type=commons.EvaluatorParamType.str,
            default_value=DEFAULT_SHORT_STRING_METRIC,
            src=evaluators.EvaluatorParam.SRC_EVALUATOR_PARAMS,
        ),
        evaluators.EvaluatorParam(
            param_name=PARAM_SHORT_STRING_THRESHOLD,
            description=(
                "Character length threshold below which to use short string metric "
                "instead of embedding-based similarity. When either the expected or "
                "actual answer is at or below this threshold, the short_string_metric "
                "is used."
            ),
            param_type=commons.EvaluatorParamType.int,
            default_value=DEFAULT_SHORT_STRING_THRESHOLD,
            src=evaluators.EvaluatorParam.SRC_EVALUATOR_PARAMS,
        ),
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
        evaluators.KEYWORD_CAP_AH,
    ]

    _modules_needed_by_name = [
        h2o_sonar_config.DEP_SENTENCE_TRANSFORMERS,
        h2o_sonar_config.DEP_NLTK,
    ]

    _e_model_baai_bge = caching.MODEL_BAAI_BGE_SMALL_EN

    _brief_description = """Answer Accuracy Evaluator assesses how closely the
actual answer matches the expected answer. It measures semantic similarity between
the expected answer and actual answer sentences - as the actual answer generated by
the RAG/LLM model **should match** the expected answer."""
    _description = evaluators.Evaluator._description_builder(
        brief=f"""{_brief_description}

**Method**:

- The answer accuracy metric is calculated as:

```math
answer_acc = min( {{ max( {{S(emb(a), emb(e)): for all e in E}} ): for all a in A }} )
```

- Where:
    - `A` is the actual answer.
    - `emb(a)` is a vector embedding of the actual answer sentence.
    - `E` is the expected answer.
    - `emb(e)` is a vector embedding of the expected answer sentence.
    - `S(a, e)` is the 1 - cosine distance between the actual answer sentence `a`
      and the expected answer sentence `e`.
- The evaluator uses **embeddings**
  [{_e_model_baai_bge}](https://huggingface.co/{_e_model_baai_bge}) (where BGE
  stands for "BAAI General Embedding" which refers to a suite of open-source text
  embedding models developed by the Beijing Academy of Artificial Intelligence (BAAI)).
- For short answers (either expected or actual ≤ `short_string_threshold` characters),
  embedding-based similarity is not ideal as it cannot be calculated. Instead,
  the evaluator uses a fallback metric specified by `short_string_metric`:
    - **normalized_edit_distance** (default): Normalized Levenshtein distance,
      good for handling typos and case differences.
    - **exact_match**: Strict case-insensitive matching, ideal for Yes/No answers.
    - **token_jaccard**: Token overlap similarity, suitable for short multi-word
      phrases.
    - **embeddings**: Force embeddings anyway which may and probably will result
      in `NaN`s metric scores.
- This ensures accurate evaluation when either answer is short, like "Yes", "No",
  "$25", "90 days", etc., which would otherwise be filtered out during sentence
  tokenization.
""",
        metrics_meta=_metrics_meta,
        keywords=_keywords,
        parameters=_parameters,
        extra_insights=(
            "\n- The least accurate actual answer sentence (in case that "
            "the output metric score is below the threshold)."
        ),
        leaderboard_type=e10s.LlmHeatmapLeaderboardExplanation.explanation_type(),
    )

    COL_INPUT = datasets.LlmDataset.KEY_INPUT
    COL_EXPECTED_OUTPUT = datasets.LlmDataset.KEY_EXPECTED_OUTPUT
    COL_ACTUAL_OUTPUT = datasets.LlmDataset.KEY_ACTUAL_OUTPUT
    COL_MODEL = "model"
    COL_SCORE = "score"

    def __init__(self):
        evaluators.Evaluator.__init__(self)

        self.args = None
        self.problems = []
        self.log_name = "Answer Accuracy"

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
            check_empty_contexts=False,
            fail_on_all_empty_contexts=False,
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
        caching.cache_nltk_punkt(self.logger)

    def evaluate(self, llm_testset, explanations_types=None, **kwargs) -> list:
        save_llm_result = self.args.get(
            evaluators.Evaluator.PARAM_SAVE_LLM_RESULT,
            evaluators.Evaluator.DEFAULT_SAVE_LLM_RESULT,
        )

        return self._evaluate(
            llm_testset=llm_testset,
            save_llm_result=save_llm_result,
        )

    _MSG_ADDITIONAL_DETAILS_PREFIX = "Additional details - prompt: "

    def _get_prompt_and_llm_info_str(self, row) -> str:
        """Get prompt and LLM information as a formatted string."""
        key_2_evaluated_model = {m.key: m for m in self.models}
        return (
            f'{self._MSG_ADDITIONAL_DETAILS_PREFIX}"{row.i}"'
            f", LLM: {key_2_evaluated_model[row.model_key].llm_model_name}"
        )

    def _add_prompt_and_llm_info_html(self, row, html) -> None:
        """Add prompt and LLM information to an HTML object."""
        key_2_evaluated_model = {m.key: m for m in self.models}
        html(self._MSG_ADDITIONAL_DETAILS_PREFIX)
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
            sort_by_metric_id=self.METRIC_ANSWER_ACCURACY
        )
        heatmap_explanation.add_evalstudio_markdown_format(
            sort_by_metric_id=self.METRIC_ANSWER_ACCURACY
        )
        explanations.append(heatmap_explanation)

        # PROBLEMS for alerts and actionability
        self._diagnose_problems(
            eval_results=eval_results,
            key_2_evaluated_model=key_2_evaluated_model,
            leaderboard_explanation=heatmap_explanation,
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
                            sort_by_metric_id=self.METRIC_ANSWER_ACCURACY
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
    def split_sentences(text: str) -> list[str]:
        """Split the text into sentences."""
        sentences = nltk.tokenize.sent_tokenize(text)
        sentences = [sent for sent in sentences if len(sent) > 10]
        return sentences

    @staticmethod
    def _calculate_normalized_edit_distance(str1: str, str2: str) -> float:
        """Normalized edit distance (Levenshtein) in [0, 1] range for short strings."""
        if not str1 and not str2:
            return 1.0  # both empty = perfect match
        if not str1 or not str2:
            return 0.0  # one empty = no match

        max_len = max(len(str1), len(str2))
        distance = nltk.edit_distance(str1.lower(), str2.lower())
        return 1.0 - (distance / max_len)

    @staticmethod
    def _calculate_exact_match(str1: str, str2: str) -> float:
        """Exact match (case-insensitive, stripped) for short strings."""
        return 1.0 if str1.strip().lower() == str2.strip().lower() else 0.0

    @staticmethod
    def _calculate_token_jaccard(str1: str, str2: str) -> float:
        """Jaccard similarity on whitespace-tokenized strings for short strings."""
        tokens1 = set(str1.lower().split())
        tokens2 = set(str2.lower().split())

        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        return len(intersection) / len(union)

    def _calculate_short_string_similarity(
        self, actual: str, expected: str, metric_type: str
    ) -> float | None:
        """Calculate similarity for short strings using configured specified metric.

        Returns:
            Similarity score between 0.0 and 1.0, or None if embeddings should be used
        """
        if metric_type == self.SHORT_STRING_METRIC_NORMALIZED_EDIT_DISTANCE:
            return self._calculate_normalized_edit_distance(actual, expected)
        elif metric_type == self.SHORT_STRING_METRIC_EXACT_MATCH:
            return self._calculate_exact_match(actual, expected)
        elif metric_type == self.SHORT_STRING_METRIC_TOKEN_JACCARD:
            return self._calculate_token_jaccard(actual, expected)
        elif metric_type == self.SHORT_STRING_METRIC_EMBEDDINGS:
            # Return None to signal that embeddings should be used
            return None
        else:
            self.logger.warning(
                f"Unknown short string metric '{metric_type}', "
                f"defaulting to {self.SHORT_STRING_METRIC_NORMALIZED_EDIT_DISTANCE}"
            )
            return self._calculate_normalized_edit_distance(actual, expected)

    def _calculate_answer_accuracy(self, model, row):
        explanation_type = e10s.GlobalHtmlFragmentExplanation.explanation_type()

        # check if expected answer is short - use short string metric if so
        short_string_threshold = self.args.get(
            self.PARAM_SHORT_STRING_THRESHOLD,
            self.DEFAULT_SHORT_STRING_THRESHOLD,
        )
        short_string_metric = self.args.get(
            self.PARAM_SHORT_STRING_METRIC,
            self.DEFAULT_SHORT_STRING_METRIC,
        )

        is_expected_short = len(row.expected_output.strip()) <= short_string_threshold
        is_actual_short = len(row.actual_output.strip()) <= short_string_threshold

        # if either answer is short, use short string metric instead of embeddings
        # this prevents short outputs from being filtered out by sentence tokenization
        if is_expected_short or is_actual_short:
            similarity = self._calculate_short_string_similarity(
                row.actual_output, row.expected_output, short_string_metric
            )

            # if similarity is None, it means "embeddings" option was selected
            # so we should proceed with normal embedding-based calculation
            if similarity is not None:
                self.logger.info(
                    f"Using short string metric '{short_string_metric}' for "
                    f"expected='{row.expected_output}' "
                    f"(len={len(row.expected_output)}) vs actual='{row.actual_output}' "
                    f"(len={len(row.actual_output)}): similarity={similarity:.3f}"
                )

                # no sentence-level breakdown for short strings
                return similarity, None
            else:
                self.logger.info(
                    f"Short string metric is '{short_string_metric}' (embeddings) - "
                    f"proceeding with embedding-based calculation for "
                    f"expected='{row.expected_output}' "
                    f"(len={len(row.expected_output)}) vs actual='{row.actual_output}' "
                    f"(len={len(row.actual_output)})"
                )

        # embed expected answer sentences
        expected_sentences = self.split_sentences(row.expected_output)
        try:
            expected_embedded_sentences = model.encode(expected_sentences)
        except Exception as ex:
            d_prefix = "Error during vector embedding of the expected answer: "
            description = (
                f'{d_prefix}"{expected_sentences}". '
                f"{self._get_prompt_and_llm_info_str(row)}."
            )
            self.logger.error(f"{description} - {ex}\n{traceback.format_exc()}")

            html = airium.Airium()
            html(d_prefix)
            with html.b():
                with html.i():
                    html(f'"{expected_sentences}"')
            html(".")
            self._add_prompt_and_llm_info_html(row, html)
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
            return float("NaN"), None

        # Embed actual answer sentences
        output_sentences = self.split_sentences(row.actual_output)
        try:
            output_embedded_sentences = model.encode(output_sentences)
        except Exception as ex:
            d_prefix = "Error during vector embedding of the actual answer: "
            description = (
                f'{d_prefix}"{output_sentences}". '
                f"{self._get_prompt_and_llm_info_str(row)}."
            )
            self.logger.error(f"{description} - {ex}\n{traceback.format_exc()}")

            html = airium.Airium()
            html(d_prefix)
            with html.b():
                with html.i():
                    html(f'"{output_sentences}"')
            html(".")
            self._add_prompt_and_llm_info_html(row, html)
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
            return float("NaN"), None

        if len(expected_embedded_sentences) == 0 or len(output_embedded_sentences) == 0:
            d_prefix = (
                "Embedding of the expected answer or actual answer is empty - the "
                "evaluation dataset row will be skipped. "
            )
            description = f"{d_prefix}{self._get_prompt_and_llm_info_str(row)}."

            html = airium.Airium()
            html(d_prefix)
            self._add_prompt_and_llm_info_html(row, html)
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
            return float("NaN"), None

        # Calculate sentence-level accuracy: for each actual answer sentence,
        # find maximum similarity with any expected answer sentence
        sentence_accuracy = [
            max(
                1 - nltk.cluster.cosine_distance(actual_sent, expected_sent)
                for expected_sent in expected_embedded_sentences
            )
            for actual_sent in output_embedded_sentences
        ]

        # Find the least accurate sentence
        least_accurate_sentence_idx = min(
            range(len(sentence_accuracy)), key=lambda s: sentence_accuracy[s]
        )
        metric_result = sentence_accuracy[least_accurate_sentence_idx]

        threshold = self.args.get(
            evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
            self._metrics_meta.get_primary_metric().threshold,
        )
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
                            metrics={self.METRIC_ANSWER_ACCURACY: sentence_accuracy[i]},
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

        if metric_result < threshold:
            least_accurate_sent = output_sentences[least_accurate_sentence_idx]

            d_prefix = (
                "The least accurate sentence identified by the Answer Accuracy "
                "evaluator is: "
            )

            description = (
                f'{d_prefix}"{least_accurate_sent}". '
                f"{self._get_prompt_and_llm_info_str(row)}."
            )

            html = airium.Airium()
            html(d_prefix)
            with html.b():
                with html.i():
                    html(f'"{least_accurate_sent}"')
            html(".")
            self._add_prompt_and_llm_info_html(row, html)
            html(".")

            self.add_problem(
                problems.ProblemAndAction(
                    description=description,
                    description_html=html,
                    evaluator_id=self.evaluator_id(),
                    problem_code=problems.AVIDProblemCode.P0100_DATA,
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

        return metric_result, actual_output_meta

    def _calculate_metrics(self, llm_testset):
        llm_dataset = datasets.LlmDataset.from_datatable_dict(llm_testset.to_dict())

        # evaluator runs only ONE metric at a time
        self.report_progress(0.01, "Configuring metrics...")

        eval_results = datasets.LlmEvalResults()

        device = h2o_sonar_config.config.resolve_gpu_cpu_device(result_format="str")
        with resource_mgmt.PytorchModelLifeCycleManager(
            sentence_transformers.SentenceTransformer(
                AnswerAccuracyEvaluator._e_model_baai_bge,
                device=device,
                revision=caching.REVISIONS_FOR_MODEL.get(
                    AnswerAccuracyEvaluator._e_model_baai_bge, "main"
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
                        metric_name=self.METRIC_ANSWER_ACCURACY,
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
                                self.METRIC_ANSWER_ACCURACY: 0.0,
                            },
                        )
                    )
                    continue

                # handle empty actual or expected output
                if (
                    not r.actual_output
                    or not isinstance(r.actual_output, str)
                    or not r.expected_output
                    or not isinstance(r.expected_output, str)
                ):
                    d_prefix = "Empty or invalid field detected: "
                    if not r.actual_output or not isinstance(r.actual_output, str):
                        detail = "actual output is empty or invalid"
                    else:
                        detail = "expected output is empty or invalid"

                    description = (
                        f"{d_prefix}{detail}. Dataset row skipped - prompt: "
                        f'"{r.i}", model: {r.model_key}.'
                    )
                    self.logger.warning(description)

                    html = airium.Airium()
                    html(d_prefix)
                    with html.b():
                        html(detail)
                    html(". Dataset row skipped - prompt: ")
                    with html.b():
                        with html.i():
                            html(f'"{r.i}"')
                    html(f", model: {r.model_key}.")

                    self.add_problem(
                        problems.ProblemAndAction(
                            description=description,
                            description_html=html,
                            actions_description=(
                                "The evaluator requires both the expected and actual "
                                "answers to be non-empty strings to calculate answer "
                                "accuracy."
                            ),
                            evaluator_id=self.evaluator_id(),
                            problem_attrs={
                                problems.ProblemAndAction.ATTR_ROW_KEYS: [
                                    (r.key, r.model_key)
                                ],
                                problems.ProblemAndAction.ATTR_TEST_CASE_KEYS: [r.key],
                                problems.ProblemAndAction.ATTR_EVALUATOR_NAME: (
                                    self._display_name
                                ),
                            },
                            evaluator_name=self._display_name,
                            explanation_type=(
                                e10s.GlobalHtmlFragmentExplanation.explanation_type()
                            ),
                            explanation_name=(
                                e10s.GlobalHtmlFragmentExplanation.__name__
                            ),
                            explanation_mime=f5s.HtmlFormat.mime,
                            resources=[],
                        )
                    )

                    # set WORST metrics values
                    eval_results.add_result(
                        datasets.LlmEvalResults.LlmEvalResultRow(
                            dataset_row=r,
                            metrics={
                                self.METRIC_ANSWER_ACCURACY: 0.0,
                            },
                        )
                    )
                    continue

                result, result_tokenization = self._calculate_answer_accuracy(
                    embedding_model, r
                )
                if not math.isnan(result):
                    eval_results.add_result(
                        datasets.LlmEvalResults.LlmEvalResultRow(
                            dataset_row=r,
                            metrics={self.METRIC_ANSWER_ACCURACY: result},
                            actual_output_meta=(
                                [result_tokenization] if result_tokenization else []
                            ),
                        )
                    )

        return eval_results

    def _diagnose_problems(
        self,
        eval_results: datasets.LlmEvalResults,
        key_2_evaluated_model: dict,
        leaderboard_explanation: e10s.LlmHeatmapLeaderboardExplanation,
    ):
        # perturbation flips
        self._diagnose_perturbation_problems(
            eval_results=eval_results,
            key_2_evaluated_model=key_2_evaluated_model,
        )

        # threshold failures
        problems.problems_for_heat_leaderboard(
            evaluator=self,
            leaderboard=leaderboard_explanation,
            metric_threshold=self.args.get(
                evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
                self._metrics_meta.get_primary_metric().threshold,
            ),
            primary_metric_meta=self._metrics_meta.get_primary_metric(),
            problem_type="accuracy",
            actions_description=(
                "Focus on three key areas: training data quality, model fine-tuning, "
                "and output validation. First, ensure training data includes diverse, "
                "high-quality examples that cover the expected answer patterns. "
                "Second, fine-tune the model on task-specific data to improve "
                "alignment with expected outputs. Third, implement output validation "
                "mechanisms to verify generated answers against expected criteria."
            ),
            explanation_type=e10s.GlobalHtmlFragmentExplanation.explanation_type(),
            explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
            explanation_mime=f5s.HtmlFormat.mime,
        )

    def _diagnose_insights(
        self, leaderboard_explanation: e10s.LlmHeatmapLeaderboardExplanation
    ):
        t_html_fragment = e10s.GlobalHtmlFragmentExplanation

        leaderboard_explanation.get_insights(
            metrics_meta=self._metrics_meta,
            metric_name_protection=True,
            extra_description_worst=(
                "Focus on three key areas: training data quality, model fine-tuning, "
                "and output validation. First, ensure training data includes diverse, "
                "high-quality examples that cover the expected answer patterns. "
                "Second, fine-tune the model on task-specific data to improve "
                "alignment with expected outputs. Third, implement output validation "
                "mechanisms to verify generated answers against expected criteria."
            ),
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
            explainer_id=AnswerAccuracyEvaluator.explainer_id(),
            h2o_sonar_config=self.config,
            logger=self.logger,
            explanation=e10s.LlmHeatmapLeaderboardExplanation,
            explanation_format=f5s.CustomJsonFormat,
        )
