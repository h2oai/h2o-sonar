# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import traceback

import datatable

from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import evaluators
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import problems
from h2o_sonar.lib.api import results


try:
    import sklearn

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class ClassificationEvaluator(evaluators.Evaluator):
    _display_name = "Classification"
    _tagline = "Evaluate binomial and multinomial classification."

    METRIC_TP = "tp"
    METRIC_TN = "tn"
    METRIC_FP = "fp"
    METRIC_FN = "fn"

    _metrics_meta = commons.MetricsMeta(
        metrics=[
            e10s.LlmClassifierLeaderboardExplanation.METRIC_META_ACCURACY,
            e10s.LlmClassifierLeaderboardExplanation.METRIC_META_PRECISION,
            e10s.LlmClassifierLeaderboardExplanation.METRIC_META_RECALL,
            e10s.LlmClassifierLeaderboardExplanation.METRIC_META_F1,
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
        evaluators.KEYWORD_PROBLEM_TYPE_CLS,
        evaluators.KEYWORD_PROBLEM_TYPE_BIN,
        evaluators.KEYWORD_PROBLEM_TYPE_MUL,
        evaluators.KEYWORD_ES_CLASSIFY,
        evaluators.KEYWORD_METHOD_TYPE_DETERMINISTIC,
        evaluators.KEYWORD_METHOD_RULE_BASED,
    ]

    _modules_needed_by_name = []

    _brief_description = """Binomial and multinomial classification evaluator for
LLM models and RAG systems which are used to classify data into two or more classes.

- Compatibility: RAG and LLM models."""
    _description = evaluators.Evaluator._description_builder(
        brief=f"""{_brief_description}

**Method**:

- The evaluator matches expected answer (label) and actual answers (prediction) for
  each test case and calculates the confusion matrix and metrics metrics such as
  accuracy, precision, recall, and F1 score for each model."

```math
            | TP | + | TN |
accuracy = -----------------
            all predictions

                 | TP |
precision = -----------------
            | TP |  + | FP |

              | TP |
recall = -----------------
          | TP | + | FN |

      2 * (precision * recall)
F1 = --------------------------
         precision + recall
```

- Where:
    - `TP` - true positives.
    - `TN` - true negatives.
    - `FP` - false positives.
    - `FN` - false negatives.
""",
        metrics_meta=_metrics_meta,
        keywords=_keywords,
        parameters=_parameters,
        leaderboard_type=e10s.LlmClassifierLeaderboardExplanation.explanation_type(),
    )

    def __init__(self):
        evaluators.Evaluator.__init__(self)

        self.args = None
        self.problems = []
        self.log_name = "Classification evaluator"

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        **evaluator_params,
    ) -> bool:
        evaluators.Evaluator.check_compatibility(self, params, **evaluator_params)

        if not HAS_SKLEARN:
            self.logger.warning(self._check_compatibility_pckg_err_msg("scikit-learn"))
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

        return True

    def setup(self, model, persistence, **kwargs):
        evaluators.Evaluator.setup(self, model, persistence, **kwargs)

        self._resolve_evaluator_params()

        self.log_name = f"Classification evaluator {self.mli_key}/{self.key}"

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

        #
        # BINOMIAL vs. MULTINOMIAL classification
        #
        class_count = llm_testset[
            :, datatable.nunique(datatable.f[datasets.LlmDataset.COL_EXPECTED_OUTPUT])
        ][0, 0]
        if class_count < 2:
            raise ValueError(
                f"{self.log_name}: expected output column must have at least 2 "
                f"unique values for binomial or multinomial classification "
                f"evaluation, but got {class_count} "
                f"value{'' if class_count in [1, -1] else 's'}"
            )

        # BINOMIAL classification: determine classes
        classes_df = datatable.unique(
            llm_testset[:, datasets.LlmDataset.COL_EXPECTED_OUTPUT]
        )
        classes = classes_df.to_dict().get(datasets.LlmDataset.COL_EXPECTED_OUTPUT, [])
        if not classes:
            raise ValueError(
                f"{self.log_name}: unable to get classes from the LLM dataset: "
                f"{classes}"
            )
        self.logger.info(f"{self.log_name}: classes: {classes}")

        # calculate metrics per model:
        # map: model key -> confusion matrix
        model_2_confusion_matrix = {}
        # map: model key -> metric ID -> metric value
        model_2_metrics = {}

        t_cls_leaderboard = e10s.LlmClassifierLeaderboardExplanation

        for model_key in key_2_evaluated_model:
            llm_model_name = key_2_evaluated_model[model_key].llm_model_name

            model_testset = llm_testset[
                datatable.f[datasets.LlmDataset.COL_MODEL_KEY] == model_key, :
            ]

            # CONFUSION MATRIX per-model
            labels_col = model_testset[
                :, datasets.LlmDataset.COL_EXPECTED_OUTPUT
            ].to_list()[0]
            preds_col = model_testset[
                :, datasets.LlmDataset.COL_ACTUAL_OUTPUT
            ].to_list()[0]
            labels_col_unique = list(set(labels_col))
            labels_col_unique.sort()
            unexpected_col_unique = list(set(preds_col) - set(labels_col))
            confusion_matrix = sklearn.metrics.confusion_matrix(
                y_true=labels_col,
                y_pred=preds_col,
                labels=(labels_col_unique + unexpected_col_unique),
            )
            # x-axis labels ~ classes list, y-axis labels ~ classes list
            model_2_confusion_matrix[llm_model_name] = confusion_matrix
            self.logger.info(f"{self.log_name}: confusion matrix:\n{confusion_matrix}")

            # METRICS: precision, recall, f1, accuracy
            precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(
                labels_col, preds_col, labels=classes, average="macro"
            )
            self.logger.info(
                f"{self.log_name}: precision={precision}, recall={recall}, f1={f1}"
            )

            # METRICS: accuracy
            accuracy = sklearn.metrics.accuracy_score(labels_col, preds_col)
            self.logger.info(f"{self.log_name}: accuracy={accuracy}")

            model_2_metrics[llm_model_name] = {
                t_cls_leaderboard.METRIC_ACCURACY: float(accuracy),
                t_cls_leaderboard.METRIC_PRECISION: float(precision),
                t_cls_leaderboard.METRIC_RECALL: float(recall),
                t_cls_leaderboard.METRIC_F1: float(f1),
            }

        self.logger.info(f"model_2_metrics:\n{json.dumps(model_2_metrics, indent=2)}")

        #
        # NORMALIZATION of the evaluation RESULTS
        #

        sort_by_metric = self._metrics_meta.get_primary_metric().key

        # RESULTS
        eval_results = datasets.LlmEvalResults()
        for e, r in enumerate(llm_dataset.inputs):
            # handle actual answer retrieval error ~ RAG/LLM client crash
            if evaluators.Evaluator._is_internal_err_answer(r.actual_output):
                # set WORST metrics values
                metrics_dict = {
                    t_cls_leaderboard.METRIC_ACCURACY: 0.0,
                    t_cls_leaderboard.METRIC_PRECISION: 0.0,
                    t_cls_leaderboard.METRIC_RECALL: 0.0,
                    t_cls_leaderboard.METRIC_F1: 0.0,
                }
            else:
                # metric: pass/fail for row (acc/prec/rec/f1 are aggregated per-model)
                metric_value = 1.0 if r.actual_output == r.expected_output else 0.0
                metrics_dict = {
                    # TODO: prompt scope metrics values of evaluation scope metrics
                    t_cls_leaderboard.METRIC_ACCURACY: metric_value,
                    t_cls_leaderboard.METRIC_PRECISION: metric_value,
                    t_cls_leaderboard.METRIC_RECALL: metric_value,
                    # WARNING: F1 should be NaN for wrong class, but is 0.0
                    # (avoid interop issues)
                    t_cls_leaderboard.METRIC_F1: metric_value,
                }

            eval_results.add_result(
                datasets.LlmEvalResults.LlmEvalResultRow(
                    dataset_row=r,
                    metrics=metrics_dict,
                )
            )

        # EXPLANATIONS
        explanations = []

        # EXPLANATION: all data (per prompt metrics)
        if save_llm_result:
            eval_results_explanation = e10s.LlmEvalResultsExplanation(
                evaluator=self,
                display_name="Classification evaluation results",
                display_category=e10s.Explanation.DISPLAY_CAT_LLM,
                eval_results=eval_results,
            )
            # FORMATS of the explanation: JSon, CSV, DataTable
            eval_results_explanation.add_json_format()
            eval_results_explanation.add_csv_format()
            eval_results_explanation.add_datatable_format()
            explanations.append(eval_results_explanation)

        # EXPLANATION: classification leaderboard
        cls_explanation = e10s.LlmClassifierLeaderboardExplanation.from_eval_results(
            evaluator=self,
            eval_results=eval_results,
            model_2_metrics=model_2_metrics,
            model_2_confusion_matrix=model_2_confusion_matrix,
            classes=classes,
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
        #   code to be within classifier leaderboard explanation
        cls_explanation.add_json_format(
            threshold=self.args.get(
                evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
                self._metrics_meta.get_primary_metric().threshold,
            )
        )
        cls_explanation.add_markdown_format(sort_by_metric_id=sort_by_metric)
        cls_explanation.add_evalstudio_markdown_format(
            sort_by_metric_id=self._metrics_meta.get_primary_metric().key
        )
        explanations.append(cls_explanation)

        # PROBLEMS for alerts and actionability
        self._diagnose_problems(
            eval_results=eval_results,
            key_2_evaluated_model=key_2_evaluated_model,
            leaderboard_explanation=cls_explanation,
        )

        # INSIGHTS
        ClassificationEvaluator._diagnose_insights(
            leaderboard_explanation=cls_explanation,
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
                        cls_explanation.as_html(
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

    @staticmethod
    def _diagnose_insights(
        leaderboard_explanation: e10s.LlmClassifierLeaderboardExplanation,
    ):
        t_html_fragment = e10s.GlobalHtmlFragmentExplanation

        leaderboard_explanation.get_insights(
            extra_description_best=(
                "This model produces responses that most closely resemble the expected "
                "classification responses."
            ),
            insight_type="accuracy",  # + classification
            explanation_type=t_html_fragment.explanation_type(),
            explanation_name=t_html_fragment.__name__,
            explanation_mime=f5s.HtmlFormat.mime,
        )

    def _diagnose_problems(
        self,
        eval_results: datasets.LlmEvalResults,
        key_2_evaluated_model: dict,
        leaderboard_explanation: e10s.LlmClassifierLeaderboardExplanation,
    ):
        # perturbation flips
        self._diagnose_perturbation_problems(
            eval_results=eval_results,
            key_2_evaluated_model=key_2_evaluated_model,
            # row metrics are different from aggregated model metrics
            metrics_meta=e10s.LlmBoolLeaderboardExplanation.LEADERBOARD_METRICS_META,
        )

        # low test case count
        self._diagnose_low_test_case_problem(
            eval_results=eval_results,
            models=self.models,
            test_case_minimum=self.args.get(evaluators.Evaluator.PARAM_MIN_TEST_CASES),
        )

        # threshold failures
        problems.problems_for_cls_leaderboard(
            evaluator=self,
            leaderboard=leaderboard_explanation,
            metric_threshold=self.args.get(
                evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
                self._metrics_meta.get_primary_metric().threshold,
            ),
            primary_metric_meta=self._metrics_meta.get_primary_metric(),
            problem_type="classification",
            problem_code=problems.AVIDProblemCode.P0200_MODEL,
            actions_description=(
                "RAG/LLM can enhance classification accuracy through several "
                "methods: firstly, focusing on high-quality training data with "
                "clear labels for both retrieval and generation steps is crucial. "
                "Secondly, incorporating techniques like curriculum learning can "
                "gradually expose the model to more complex classification tasks. "
                "Additionally, employing external knowledge sources during retrieval "
                "and refining the answer generation process with techniques focusing "
                "on high-probability classifications can improve final outputs. "
            ),
            explanation_type=e10s.GlobalHtmlFragmentExplanation.explanation_type(),
            explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
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
