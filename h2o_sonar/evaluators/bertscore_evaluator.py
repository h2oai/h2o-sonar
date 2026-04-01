# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import traceback

import airium

from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import evaluators
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import problems
from h2o_sonar.lib.api import results


try:
    import bert_score

    HAS_BERT_SCORE = True
except ImportError:
    HAS_BERT_SCORE = False


class BertscoreEvaluator(evaluators.Evaluator):
    _display_name = "BERTScore"
    _tagline = "Evaluate actual and expected answers using BERT embeddings."

    METRIC_BERTSCORE_PRECISION = "bertscore_precision"
    METRIC_BERTSCORE_RECALL = "bertscore_recall"
    METRIC_BERTSCORE_F1 = "bertscore_f1"

    _metrics_meta = commons.MetricsMeta(
        metrics=[
            commons.MetricMeta(
                key=METRIC_BERTSCORE_PRECISION,
                display_name="BERTScore Precision",
                description=(
                    "BERTScore Precision measures how much of the actual answer is "
                    "relevant to the expected answer based on contextual embeddings."
                ),
                higher_is_better=True,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=False,
            ),
            commons.MetricMeta(
                key=METRIC_BERTSCORE_RECALL,
                display_name="BERTScore Recall",
                description=(
                    "BERTScore Recall measures how much of the expected answer is "
                    "covered by the actual answer based on contextual embeddings."
                ),
                higher_is_better=True,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=False,
            ),
            commons.MetricMeta(
                key=METRIC_BERTSCORE_F1,
                display_name="BERTScore F1",
                description=(
                    "BERTScore F1 is the harmonic mean of precision and recall, "
                    "providing a balanced measure of semantic similarity between "
                    "actual and expected answer using contextual embeddings."
                ),
                higher_is_better=True,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=True,
            ),
        ]
    )

    # COMPATIBILITY: LLM model explanations only
    _llm = True
    _rag = True

    # GLOBAL: leaderboard as global explanation
    _global_explanation = True

    # EXPLANATION TYPES created by the evaluator
    _explanation_types = [
        e10s.LlmEvalResultsExplanation,
        e10s.LlmHeatmapLeaderboardExplanation,
    ]

    _parameters = [
        evaluators.Evaluator._get_custom_param_metric_threshold(
            _metrics_meta.get_primary_metric()
        ),
        evaluators.Evaluator._PARAM_SAVE_LLM_RESULT,
        evaluators.Evaluator._get_custom_param_min_test_case(),
    ]

    _keywords = [
        evaluators.KEYWORD_LLM,
        evaluators.KEYWORD_EVALUATES_LLM,
        evaluators.KEYWORD_EVALUATES_RAG,
        evaluators.KEYWORD_RQ_EA,
        evaluators.KEYWORD_RQ_AA,
        evaluators.KEYWORD_SR_11_7_CS,
        evaluators.KEYWORD_SR_11_7_OA,
        evaluators.KEYWORD_NIST_AI_RMF_S,
        evaluators.KEYWORD_PROBLEM_TYPE_SUM,
        evaluators.KEYWORD_ES_SUMMARIZE,
        evaluators.KEYWORD_METHOD_SEMANTIC_SIMILARITY,
        evaluators.KEYWORD_METHOD_TYPE_DETERMINISTIC,
    ]

    _modules_needed_by_name = ["bert-score"]

    _brief_description = """BERTScore leverages pre-trained contextual embeddings from
BERT to evaluate the semantic similarity between actual and expected answers. Unlike
traditional n-gram based metrics like ROUGE and BLEU, BERTScore captures semantic
meaning and is more robust to paraphrasing and word choice variations.

- Compatibility: RAG and LLM models."""
    _description = evaluators.Evaluator._description_builder(
        brief=f"""{_brief_description}

**Method**:

- BERTScore computes token-level similarity using contextual embeddings from pre-trained
  BERT models, then aggregates these similarities to produce precision, recall,
  and F1 scores.
- `Precision` measures how much of the actual answer is relevant to the expected answer.
- `Recall` measures how much of the expected answer is covered by the actual answer.
- `F1` is the harmonic mean of precision and recall, providing a balanced evaluation
  metric.
- The evaluator uses the default BERT model optimized for English text evaluation.
- BERTScore is particularly effective for tasks where semantic similarity is more
  important than exact word matching, such as summarization, paraphrasing, and machine
  translation.

See also:

- BERTScore paper: https://arxiv.org/abs/1904.09675
- 3rd party library: https://pypi.org/project/bert-score/
- Source code: https://github.com/Tiiiger/bert_score""",
        metrics_meta=_metrics_meta,
        keywords=_keywords,
        parameters=_parameters,
        leaderboard_type=e10s.LlmHeatmapLeaderboardExplanation.explanation_type(),
    )

    def __init__(self):
        evaluators.Evaluator.__init__(self)

        self.args = None
        self.problems = []
        self.log_name = "BERTScore evaluator"

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        **evaluator_params,
    ) -> bool:
        if not HAS_BERT_SCORE:
            self.logger.warning(self._check_compatibility_pckg_err_msg("bert-score"))
            return False

        evaluators.Evaluator.check_compatibility(self, params, **evaluator_params)

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

        self.log_name = f"BERTScore evaluator {self.mli_key}/{self.key}"

    def evaluate(self, llm_testset, **kwargs) -> list:
        save_llm_result = self.args.get(
            evaluators.Evaluator.PARAM_SAVE_LLM_RESULT,
            evaluators.Evaluator.DEFAULT_SAVE_LLM_RESULT,
        )

        metrics_threshold = self.args.get(
            evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
            evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
        )
        self._metrics_meta.set_threshold(metrics_threshold)

        # RAG models: key -> model
        key_2_evaluated_model = {m.key: m for m in self.models}

        llm_dataset = datasets.LlmDataset.from_datatable_dict(llm_testset.to_dict())

        eval_results = datasets.LlmEvalResults()
        for r in llm_dataset.inputs:
            # handle actual answer retrieval error ~ RAG/LLM client crash
            if evaluators.Evaluator._is_internal_err_answer(r.actual_output):
                # set WORST metrics values
                eval_results.add_result(
                    datasets.LlmEvalResults.LlmEvalResultRow(
                        dataset_row=r,
                        metrics={
                            self.METRIC_BERTSCORE_PRECISION: 0.0,
                            self.METRIC_BERTSCORE_RECALL: 0.0,
                            self.METRIC_BERTSCORE_F1: 0.0,
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
                            "answers to be non-empty strings to calculate BERTScore."
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
                        explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
                        explanation_mime=f5s.HtmlFormat.mime,
                        resources=[],
                    )
                )

                # set WORST metrics values
                eval_results.add_result(
                    datasets.LlmEvalResults.LlmEvalResultRow(
                        dataset_row=r,
                        metrics={
                            self.METRIC_BERTSCORE_PRECISION: 0.0,
                            self.METRIC_BERTSCORE_RECALL: 0.0,
                            self.METRIC_BERTSCORE_F1: 0.0,
                        },
                    )
                )
                continue

            # Calculate BERTScore
            # bert_score.score returns (precision, recall, f1) tensors
            P, R, F1 = bert_score.score(
                [r.actual_output],
                [r.expected_output],
                lang="en",
                verbose=False,
            )

            # add to result
            eval_results.add_result(
                datasets.LlmEvalResults.LlmEvalResultRow(
                    dataset_row=r,
                    metrics={
                        self.METRIC_BERTSCORE_PRECISION: P.item(),
                        self.METRIC_BERTSCORE_RECALL: R.item(),
                        self.METRIC_BERTSCORE_F1: F1.item(),
                    },
                )
            )

        #
        # NORMALIZATION of the evaluation RESULTS
        #

        sort_by_metric = self._metrics_meta.get_primary_metric().key

        # EXPLANATIONS
        explanations = []

        # EXPLANATION: all data (per prompt metrics)
        if save_llm_result:
            eval_results_explanation = e10s.LlmEvalResultsExplanation(
                evaluator=self,
                display_name="BERTScore evaluation results",
                display_category=e10s.Explanation.DISPLAY_CAT_LLM,
                eval_results=eval_results,
            )
            # FORMATS of the explanation: JSon, CSV, DataTable
            eval_results_explanation.add_json_format()
            eval_results_explanation.add_csv_format()
            eval_results_explanation.add_datatable_format()
            explanations.append(eval_results_explanation)

        # EXPLANATION: heatmap leaderboard
        heatmap_explanation = e10s.LlmHeatmapLeaderboardExplanation.from_eval_results(
            evaluator=self,
            eval_results=eval_results,
            metrics_meta=self._metrics_meta,
            key_2_evaluated_model=key_2_evaluated_model,
            display_name=f"{self._display_name} leaderboard",
            display_category=e10s.GlobalSummaryFeatImpExplanation.DISPLAY_CAT_LLM,
            llm_host=(
                commons.LlmModelHostType.RAG
                if isinstance(
                    next(iter(key_2_evaluated_model.values())),
                    models.ExplainableRagModel,
                )
                else commons.LlmModelHostType.SERVICE
            ),
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
            sort_by_metric_id=self._metrics_meta.get_primary_metric().key
        )
        explanations.append(heatmap_explanation)

        # PROBLEMS for alerts and actionability
        self._diagnose_problems(
            eval_results=eval_results,
            key_2_evaluated_model=key_2_evaluated_model,
            leaderboard_explanation=heatmap_explanation,
        )

        # INSIGHTS
        self._diagnose_insights(
            leaderboard_explanation=heatmap_explanation,
        )

        # EXPLANATION: HTML fragment
        if self.config and self.config.create_html_representations:
            try:
                html_explanation = e10s.GlobalHtmlFragmentExplanation(
                    evaluator=self,
                    display_name=f"{self._display_name} leaderboard as HTML",
                    display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
                )
                html_explanation.add_html_format(
                    str(
                        heatmap_explanation.as_html(
                            sort_by_metric_id=sort_by_metric,
                        )
                    )
                )
                explanations.append(html_explanation)
            except Exception as ex:
                self.logger.warning(
                    f"{self.log_name}: HTML fragment explanation creation failed: "
                    f"{ex}\n{traceback.format_exc()}"
                )

        return explanations

    def _diagnose_problems(
        self,
        eval_results: datasets.LlmEvalResults,
        key_2_evaluated_model: dict,
        leaderboard_explanation: e10s.LlmHeatmapLeaderboardExplanation,
    ):
        # perturbation flips
        self._diagnose_perturbation_problems(
            eval_results=eval_results, key_2_evaluated_model=key_2_evaluated_model
        )

        # low test case count
        self._diagnose_low_test_case_problem(
            eval_results=eval_results,
            models=self.models,
            test_case_minimum=self.args.get(evaluators.Evaluator.PARAM_MIN_TEST_CASES),
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
            problem_type="semantic similarity",
            problem_code=problems.AVIDProblemCode.P0200_MODEL,
            actions_description=(
                "To improve semantic similarity scores, consider the following "
                "approaches: 1) Prompt-engineer or fine-tune the LLM to better match "
                "the content of expected answers. 2) Ensure training data / corpus "
                "includes diverse examples that demonstrate the desired semantic "
                "patterns. 3) Evaluate whether expected answers accurately represent "
                "the desired output quality and revise them if needed."
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
            extra_description_best=(
                "This model produces responses that are most semantically similar to "
                "the expected responses based on the BERTScore metric, which uses "
                "contextual embeddings to measure semantic similarity rather than just "
                "surface-level word overlap. This indicates the model captures "
                "the meaning and intent of expected answers more effectively."
            ),
            insight_type="semantic similarity",
            explanation_type=t_html_fragment.explanation_type(),
            explanation_name=t_html_fragment.__name__,
            explanation_mime=f5s.HtmlFormat.mime,
        )

    def get_result(
        self,
    ) -> results.LeaderboardResult:
        return results.LeaderboardResult(
            persistence=self.persistence,
            explainer_id=self.explainer_id(),
            h2o_sonar_config=self.config,
            logger=self.logger,
            explanation=e10s.LlmHeatmapLeaderboardExplanation,
            explanation_format=f5s.CustomJsonFormat,
        )
