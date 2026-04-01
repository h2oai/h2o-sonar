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
    import rouge_score.rouge_scorer

    HAS_ROUGE_SCORE = True
except ImportError:
    HAS_ROUGE_SCORE = False


class RougeEvaluator(evaluators.Evaluator):
    _display_name = "ROUGE"
    _tagline = "Assess the fidelity of generated texts to the reference texts."

    METRIC_ROUGE_1 = "rouge_1"
    METRIC_ROUGE_2 = "rouge_2"
    METRIC_ROUGE_L = "rouge_l"

    _metrics_meta = commons.MetricsMeta(
        metrics=[
            commons.MetricMeta(
                key=METRIC_ROUGE_1,
                display_name="ROUGE-1",
                description=(
                    "ROUGE-1 metric measures the overlap of 1-grams (individual words) "
                    "between the generated and the reference summaries."
                ),
                higher_is_better=True,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=False,
            ),
            commons.MetricMeta(
                key=METRIC_ROUGE_2,
                display_name="ROUGE-2",
                description=(
                    "ROUGE-1 metric measures the overlap of 2-grams (pairs of "
                    "consecutive words) between the generated and the reference "
                    "summaries."
                ),
                higher_is_better=True,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=False,
            ),
            commons.MetricMeta(
                key=METRIC_ROUGE_L,
                display_name="ROUGE-L",
                description=(
                    "ROUGE-L metric considers the longest common subsequence (LCS) "
                    "between the generated and reference summaries."
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
        evaluators.KEYWORD_METHOD_NGRAM,
        evaluators.KEYWORD_METHOD_TYPE_DETERMINISTIC,
    ]

    _modules_needed_by_name = ["rouge_score==0.1.2"]

    _brief_description = """ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
is a set of evaluation metrics used to assess the quality of generated summaries
compared to reference summaries. There are several variations of ROUGE metrics,
including `ROUGE-1`, `ROUGE-2`, and `ROUGE-L`.

- Compatibility: RAG and LLM models."""
    _description = evaluators.Evaluator._description_builder(
        brief=f"""{_brief_description}

**Method**:

- The evaluator reports `F1 score` between the generated (actual answer) and
  reference (generated answer) n-grams.
- `ROUGE-1` measures the overlap of 1-grams (individual words) between the generated
  and the reference summaries.
- `ROUGE-2` extends the evaluation to 2-grams (pairs of consecutive words).
- `ROUGE-L` considers the longest common subsequence (LCS) between the generated and
  reference summaries.
- These ROUGE metrics provide a quantitative evaluation of the similarity between
  the generated and reference texts to assess the effectiveness of
  text summarization algorithms.

See also:

- 3rd party library ROUGE: https://pypi.org/project/rouge-score/
- 3rd party ROUGE source code:
  https://github.com/google-research/google-research/tree/master/rouge""",
        metrics_meta=_metrics_meta,
        keywords=_keywords,
        parameters=_parameters,
        leaderboard_type=e10s.LlmHeatmapLeaderboardExplanation.explanation_type(),
    )

    def __init__(self):
        evaluators.Evaluator.__init__(self)

        self.args = None
        self.problems = []
        self.log_name = "ROUGE evaluator"

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        **evaluator_params,
    ) -> bool:
        if not HAS_ROUGE_SCORE:
            self.logger.warning(self._check_compatibility_pckg_err_msg("rouge-score"))
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

        self.log_name = f"ROUGE evaluator {self.mli_key}/{self.key}"

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
                            self.METRIC_ROUGE_1: 0.0,
                            self.METRIC_ROUGE_2: 0.0,
                            self.METRIC_ROUGE_L: 0.0,
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
                            "answers to be non-empty strings to calculate ROUGE scores."
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
                            self.METRIC_ROUGE_1: 0.0,
                            self.METRIC_ROUGE_2: 0.0,
                            self.METRIC_ROUGE_L: 0.0,
                        },
                    )
                )
                continue

            rouge = rouge_score.rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            ).score(r.expected_output, r.actual_output)

            # add to result
            eval_results.add_result(
                datasets.LlmEvalResults.LlmEvalResultRow(
                    dataset_row=r,
                    metrics={
                        self.METRIC_ROUGE_1: rouge["rouge1"].fmeasure,
                        self.METRIC_ROUGE_2: rouge["rouge2"].fmeasure,
                        self.METRIC_ROUGE_L: rouge["rougeL"].fmeasure,
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
                display_name="Rouge evaluation results",
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
            problem_type="summarization",
            problem_code=problems.AVIDProblemCode.P0200_MODEL,
            actions_description=(
                "To improve summarizations, focus on three key areas:  refinement, "
                "evaluation, and training data. The LLM can be equipped with "
                "auto-refinement modules that assess its own summaries and identify "
                "areas for improvement, like missing key points. Additionally, "
                "using metrics that go beyond surface-level similarity to "
                "human-written summaries can guide the training process. Finally, "
                "incorporating diverse and high-quality summaries into the training "
                "data provides the LLM with better examples to learn from, leading "
                "to more comprehensive and informative summaries."
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
                "This model produces responses that most closely resemble the expected "
                "responses based on the ROUGE metric, which is typically used to "
                "measure the quality of machine-generated text, especially summaries."
            ),
            insight_type="summarization",
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
