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
from h2o_sonar.lib.api import insights
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


class QuestionsDriftEvaluator(evaluators.Evaluator):
    _display_name = "Questions Drift"
    _tagline = "Detect semantic drift in questions over time."

    METRIC_QUESTIONS_DRIFT = "questions_drift"
    PARAM_DRIFT_THRESHOLD = "drift_threshold"
    PARAM_SPLIT_RATIO = "split_ratio"

    DEFAULT_DRIFT_THRESHOLD = 0.1
    DEFAULT_SPLIT_RATIO = 0.5
    DEFAULT_MIN_TEST_CASES = 10

    _e_model_baai_bge = caching.MODEL_BAAI_BGE_SMALL_EN

    _metrics_meta = commons.MetricsMeta(
        metrics=[
            commons.MetricMeta(
                key=METRIC_QUESTIONS_DRIFT,
                display_name="Questions Drift",
                description=(
                    "Mean embedding distance (cosine distance between centroids) of "
                    "question embeddings from two temporal groups. Lower values "
                    "indicate less drift (0 = no drift). Theoretical range: "
                    "[0.0, 2.0]. Practical range for real-world scenarios: [0.0, 0.6]."
                ),
                higher_is_better=False,
                value_range=(0.0, 2.0),
                threshold=DEFAULT_DRIFT_THRESHOLD,
                is_primary_metric=True,
            )
        ]
    )

    # COMPATIBILITY: LLM/RAG evaluation
    _llm = True
    _rag = True

    # GLOBAL: metric value for all dataset rows
    _global_explanation = True
    # LOCAL: no per-row metric
    _local_explanation = False
    _explanation_types = [
        e10s.LlmEvalResultsExplanation,
        e10s.LlmHeatmapLeaderboardExplanation,
        e10s.WorkDirArchiveExplanation,
    ]

    _parameters = [
        evaluators.EvaluatorParam(
            param_name=PARAM_DRIFT_THRESHOLD,
            description=(
                "Threshold for drift detection. Values above this threshold indicate "
                "significant drift. Default: 0.1. Typical range: 0.05-0.15."
            ),
            param_type=commons.EvaluatorParamType.float,
            default_value=DEFAULT_DRIFT_THRESHOLD,
            src=evaluators.EvaluatorParam.SRC_EVALUATOR_PARAMS,
        ),
        evaluators.EvaluatorParam(
            param_name=PARAM_SPLIT_RATIO,
            description=(
                "Ratio for splitting test cases into two temporal groups. "
                "Default: 0.5 (equal split). Range: 0.1-0.9."
            ),
            param_type=commons.EvaluatorParamType.float,
            default_value=DEFAULT_SPLIT_RATIO,
            src=evaluators.EvaluatorParam.SRC_EVALUATOR_PARAMS,
        ),
        evaluators.Evaluator._PARAM_SAVE_LLM_RESULT,
        evaluators.Evaluator._get_custom_param_min_test_case(
            minimum=DEFAULT_MIN_TEST_CASES
        ),
    ]

    _keywords = [
        evaluators.KEYWORD_GPU_OPT,
        evaluators.KEYWORD_LLM,
        evaluators.KEYWORD_EVALUATES_LLM,
        evaluators.KEYWORD_EVALUATES_RAG,
        evaluators.KEYWORD_RQ_P,
        evaluators.KEYWORD_SR_11_7_OGA,
        evaluators.KEYWORD_NIST_AI_RMF_VR,
        evaluators.KEYWORD_ES_GENERATE,
        evaluators.KEYWORD_METHOD_SEMANTIC_SIMILARITY,
        evaluators.KEYWORD_METHOD_TYPE_DETERMINISTIC,
    ]

    _modules_needed_by_name = [
        h2o_sonar_config.DEP_SENTENCE_TRANSFORMERS,
        h2o_sonar_config.DEP_NLTK,
    ]

    _brief_description = """Questions Drift Evaluator detects semantic drift in
input questions over time using mean embedding distance (cosine distance between
centroids). When the space of questions changes or untrained users engage in
unexpected ways, this evaluator identifies the drift."""

    _description = evaluators.Evaluator._description_builder(
        brief=f"""{_brief_description}

**Method**:

**Important**: This evaluator assumes test cases are **ordered chronologically**.
If your dataset is not temporally sorted, drift detection results will not be
meaningful. Ensure your dataset rows are sorted by time before evaluation.

- The metric score is calculated for every model questions (test cases) in the test lab.
- Test cases are split into 2 groups based on `split_ratio` (default: 0.5).
- Questions from each group are embedded using BAAI BGE model.
- Centroids (mean embeddings) are calculated for each group.
- Drift is computed as cosine distance between centroids:

```math
drift = cosine_distance(mean(emb(group1)), mean(emb(group2)))
```

**Note**: This evaluator uses a simplified approach based on centroid distance rather
than kernel-based Maximum Mean Discrepancy (MMD). This is appropriate and efficient
for high-dimensional embeddings and effectively detects major semantic shifts.

**Metric Range**: Theoretical range is [0.0, 2.0] based on cosine distance formula
(1 - cosine_similarity). In practice, embeddings rarely exceed 0.6 for real-world
question sets. Interpretation guide:

- 0.0 - 0.05: Negligible drift (questions remain highly consistent)
- 0.05 - 0.15: Low drift (slight variation, within normal range)
- 0.15 - 0.30: Moderate drift (noticeable topic or style shift)
- 0.30 - 0.50: High drift (significant semantic change)
- 0.50+: Extreme drift (very rare, dramatically different topics)

**Short Questions**: The evaluator can process questions of any length, including
very short questions (1-2 words). However, embedding quality and drift detection
reliability may be reduced for extremely short text. Empty questions (None or empty
strings) are automatically filtered out before processing.

- The evaluator uses **embeddings**
  [{_e_model_baai_bge}](https://huggingface.co/{_e_model_baai_bge}) (where BGE
  stands for "BAAI General Embedding" which refers to a suite of open-source text
  embedding models developed by the Beijing Academy of Artificial Intelligence (BAAI)).

**Reproducibility**: This evaluator is **reproducible**. It uses deterministic embedding
models and consistent splitting logic.
""",
        metrics_meta=_metrics_meta,
        keywords=_keywords,
        parameters=_parameters,
        leaderboard_type=e10s.LlmHeatmapLeaderboardExplanation.explanation_type(),
    )

    COL_INPUT = datasets.LlmDataset.KEY_INPUT
    COL_MODEL = "model"
    COL_DRIFT = "drift"

    def __init__(self):
        evaluators.Evaluator.__init__(self)

        self.args = None
        self.problems = []
        self.log_name = "Questions Drift"

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

        # questions/prompts (input field) are always present in LlmDataset by design
        # so no need to check explicitly

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

        drift_threshold = self.args.get(
            self.PARAM_DRIFT_THRESHOLD,
            self.DEFAULT_DRIFT_THRESHOLD,
        )

        split_ratio = self.args.get(
            self.PARAM_SPLIT_RATIO,
            self.DEFAULT_SPLIT_RATIO,
        )

        # validate parameters
        if not (0.1 <= split_ratio <= 0.9):
            self.logger.warning(
                f"{self.log_name}: split_ratio {split_ratio} out of range [0.1, 0.9], "
                f"using default {self.DEFAULT_SPLIT_RATIO}"
            )
            split_ratio = self.DEFAULT_SPLIT_RATIO

        return self._evaluate(
            llm_testset=llm_testset,
            save_llm_result=save_llm_result,
            drift_threshold=drift_threshold,
            split_ratio=split_ratio,
        )

    def _evaluate(
        self,
        llm_testset,
        save_llm_result: bool,
        drift_threshold: float,
        split_ratio: float,
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

        eval_results = self._calculate_metrics(
            llm_testset=llm_testset,
            split_ratio=split_ratio,
        )

        #
        # NORMALIZATION of the evaluation results to the common EXPLANATIONS/FORMAT(s)
        #

        # EXPLANATIONS
        explanations = []

        # EXPLANATION: all data (per model metrics)
        if save_llm_result:
            eval_results_explanation = e10s.LlmEvalResultsExplanation(
                evaluator=self,
                display_name="Questions drift evaluation results",
                display_category=e10s.Explanation.DISPLAY_CAT_LLM,
                eval_results=eval_results,
            )
            # FORMATS of the explanation: JSon, CSV, DataTable
            eval_results_explanation.add_json_format()
            eval_results_explanation.add_csv_format()
            eval_results_explanation.add_datatable_format()
            explanations.append(eval_results_explanation)

        # THRESHOLD for the metric
        self._metrics_meta.set_threshold(drift_threshold)

        # EXPLANATION: heatmap leaderboard
        heatmap_explanation = e10s.LlmHeatmapLeaderboardExplanation.from_eval_results(
            evaluator=self,
            eval_results=eval_results,
            metrics_meta=self._metrics_meta,
            key_2_evaluated_model=key_2_evaluated_model,
            llm_host=llm_host,
            display_name="Questions drift leaderboard",
            display_category=e10s.GlobalSummaryFeatImpExplanation.DISPLAY_CAT_LLM,
            logger=self.logger,
        )
        heatmap_explanation.add_json_format(threshold=drift_threshold)
        heatmap_explanation.add_markdown_format(
            sort_by_metric_id=self.METRIC_QUESTIONS_DRIFT
        )
        heatmap_explanation.add_evalstudio_markdown_format(
            sort_by_metric_id=self.METRIC_QUESTIONS_DRIFT
        )
        explanations.append(heatmap_explanation)

        # PROBLEMS for alerts and actionability
        self._diagnose_problems(
            eval_results=eval_results,
            key_2_evaluated_model=key_2_evaluated_model,
            leaderboard_explanation=heatmap_explanation,
            drift_threshold=drift_threshold,
        )

        # INSIGHTS
        self._diagnose_insights(
            leaderboard_explanation=heatmap_explanation,
            drift_threshold=drift_threshold,
        )

        # EXPLANATION: HTML fragment
        if self.config and self.config.create_html_representations:
            try:
                html_explanation = e10s.GlobalHtmlFragmentExplanation(
                    evaluator=self,
                    display_name="Questions drift leaderboard as HTML",
                    display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
                )

                # check if we have valid drift scores or only NaN
                leaderboard_data = heatmap_explanation.as_dict()[0].get(
                    f5s.ExplanationFormat.KEY_DATA, {}
                )
                has_valid_scores = any(
                    not (
                        isinstance(model_data.get(self.METRIC_QUESTIONS_DRIFT), float)
                        and np.isnan(model_data.get(self.METRIC_QUESTIONS_DRIFT))
                    )
                    for model_data in leaderboard_data.values()
                )

                if has_valid_scores:
                    # generate full leaderboard table
                    html_content = str(
                        heatmap_explanation.as_html(
                            sort_by_metric_id=self.METRIC_QUESTIONS_DRIFT
                        )
                    )
                else:
                    # all scores are NaN - show fallback message
                    html = airium.Airium()
                    with html.div(
                        klass="w3-panel w3-pale-yellow w3-leftbar w3-border-yellow"
                    ):
                        with html.p():
                            with html.b():
                                html("Questions Drift Evaluation: No Valid Scores")
                        with html.p():
                            html(
                                "The evaluator could not calculate drift scores. "
                                "This typically occurs when there are insufficient "
                                "test cases. See the Problems section for details."
                            )
                    html_content = str(html)

                html_explanation.add_html_format(html_content)
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
        split_ratio: float,
    ) -> datasets.LlmEvalResults:
        llm_dataset = datasets.LlmDataset.from_datatable_dict(llm_testset.to_dict())

        self.report_progress(0.01, "Configuring metrics...")

        self.logger.info(
            f"{self.log_name}: This evaluator assumes test cases are ordered by time. "
            f"If your dataset is not temporally sorted, results may not be meaningful."
        )

        eval_results = datasets.LlmEvalResults()

        device = h2o_sonar_config.config.resolve_gpu_cpu_device(result_format="str")

        # group rows by model_key
        model_groups = {}
        for r in llm_dataset.inputs:
            if r.model_key not in model_groups:
                model_groups[r.model_key] = []
            model_groups[r.model_key].append(r)

        with resource_mgmt.PytorchModelLifeCycleManager(
            sentence_transformers.SentenceTransformer(
                QuestionsDriftEvaluator._e_model_baai_bge,
                device=device,
                revision=caching.REVISIONS_FOR_MODEL.get(
                    QuestionsDriftEvaluator._e_model_baai_bge, "main"
                ),
            )
        ) as embedding_model:
            # calculate drift for each model
            for i, (model_key, rows) in enumerate(model_groups.items()):
                self.report_progress(
                    progress=progress_utils.ProgressCallbackContext.progress_for_steps(
                        i + 1, len(model_groups)
                    ),
                    message=evaluators.Evaluator._eval_row_progress_msg(
                        metric_name=self.METRIC_QUESTIONS_DRIFT,
                        device=device,
                        row=i + 1,
                        total_rows=len(model_groups),
                    ),
                )

                drift_score = self._calculate_drift_for_model(
                    embedding_model=embedding_model,
                    rows=rows,
                    split_ratio=split_ratio,
                )

                # add result for each row in this model group
                # include NaN results so leaderboard can be created
                for r in rows:
                    eval_results.add_result(
                        datasets.LlmEvalResults.LlmEvalResultRow(
                            dataset_row=r,
                            metrics={self.METRIC_QUESTIONS_DRIFT: drift_score},
                        )
                    )

        return eval_results

    # minimum questions per group for reliable embeddings
    _MIN_GROUP_SIZE = 3
    # minimum recommended question length
    _MIN_RECOMMENDED_LENGTH = 5

    def _calculate_drift_for_model(
        self,
        embedding_model,
        rows: list,
        split_ratio: float,
    ) -> float:
        """
        Calculate drift score for a single model.

        Parameters
        ----------
        embedding_model : sentence_transformers.SentenceTransformer
            The embedding model to use
        rows : list
            List of dataset rows for this model
        split_ratio : float
            Ratio for splitting into two groups

        Returns
        -------
        float
            Drift score (cosine distance between centroids) or NaN if error
        """
        explanation_type = e10s.GlobalHtmlFragmentExplanation.explanation_type()

        try:
            # extract questions
            questions = [r.i for r in rows]

            # filter out empty or whitespace-only questions
            valid_questions = [q for q in questions if isinstance(q, str) and q.strip()]

            # check for very short questions and log warning
            short_questions = [
                q
                for q in valid_questions
                if len(q.strip()) < QuestionsDriftEvaluator._MIN_RECOMMENDED_LENGTH
            ]
            if short_questions:
                short_pct = (
                    (len(short_questions) / len(valid_questions)) * 100
                    if valid_questions
                    else 0
                )
                if short_pct > 20:  # warn if more than 20% are very short
                    self.logger.warning(
                        f"{self.log_name}: {len(short_questions)} questions "
                        f"({short_pct:.1f}%) are very short "
                        f"(< {QuestionsDriftEvaluator._MIN_RECOMMENDED_LENGTH} chars). "
                        f"Embedding quality and drift detection reliability may be "
                        f"reduced for very short text."
                    )

            if len(valid_questions) < self.args.get(
                evaluators.Evaluator.PARAM_MIN_TEST_CASES,
                self.DEFAULT_MIN_TEST_CASES,
            ):
                min_tcs = self.args.get(
                    evaluators.Evaluator.PARAM_MIN_TEST_CASES,
                    self.DEFAULT_MIN_TEST_CASES,
                )
                d_prefix = (
                    f"Insufficient test cases for drift detection. "
                    f"Found {len(valid_questions)}, minimum required: "
                    f"{min_tcs}"
                )
                description = f"{d_prefix}."
                self.logger.warning(description)

                html = airium.Airium()
                html(d_prefix)
                html(".")

                self.add_problem(
                    problems.ProblemAndAction(
                        description=description,
                        description_html=html,
                        evaluator_id=self.evaluator_id(),
                        problem_attrs={
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

            # split into two temporal groups
            split_idx = int(len(valid_questions) * split_ratio)

            # ensure both groups have data
            if split_idx == 0 or split_idx >= len(valid_questions):
                d_prefix = (
                    f"Invalid split resulting in empty group. "
                    f"Split index: {split_idx}, total questions: {len(valid_questions)}"
                )
                description = f"{d_prefix}."
                self.logger.warning(description)

                html = airium.Airium()
                html(d_prefix)
                html(".")

                self.add_problem(
                    problems.ProblemAndAction(
                        description=description,
                        description_html=html,
                        evaluator_id=self.evaluator_id(),
                        problem_attrs={
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

            group1_questions = valid_questions[:split_idx]
            group2_questions = valid_questions[split_idx:]

            # validate minimum group sizes for statistical stability
            if (
                len(group1_questions) < QuestionsDriftEvaluator._MIN_GROUP_SIZE
                or len(group2_questions) < QuestionsDriftEvaluator._MIN_GROUP_SIZE
            ):
                d_prefix = (
                    f"Groups too small for reliable drift detection. "
                    f"Group sizes: {len(group1_questions)} and "
                    f"{len(group2_questions)}, minimum required per group: "
                    f"{QuestionsDriftEvaluator._MIN_GROUP_SIZE}"
                )
                description = f"{d_prefix}."
                self.logger.warning(description)

                html = airium.Airium()
                html(d_prefix)
                html(".")

                self.add_problem(
                    problems.ProblemAndAction(
                        description=description,
                        description_html=html,
                        evaluator_id=self.evaluator_id(),
                        problem_attrs={
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

            # generate embeddings for both groups
            group1_embeddings = embedding_model.encode(group1_questions)
            group2_embeddings = embedding_model.encode(group2_questions)

            # calculate centroids (mean embeddings)
            centroid1 = np.mean(group1_embeddings, axis=0)
            centroid2 = np.mean(group2_embeddings, axis=0)

            # compute drift as cosine distance between centroids
            drift_score = nltk.cluster.cosine_distance(centroid1, centroid2)

            return float(drift_score)

        except Exception as ex:
            d_prefix = "Error calculating drift: "
            description = f"{d_prefix}{ex}"
            self.logger.error(f"{description}\n{traceback.format_exc()}")

            html = airium.Airium()
            html(d_prefix)
            with html.code():
                html(str(ex))

            self.add_problem(
                problems.ProblemAndAction(
                    description=description,
                    description_html=html,
                    evaluator_id=self.evaluator_id(),
                    problem_attrs={
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

    def _diagnose_problems(
        self,
        eval_results: datasets.LlmEvalResults,
        key_2_evaluated_model: dict,
        leaderboard_explanation: e10s.LlmHeatmapLeaderboardExplanation,
        drift_threshold: float,
    ):
        # perturbation problems
        self._diagnose_perturbation_problems(
            eval_results=eval_results,
            key_2_evaluated_model=key_2_evaluated_model,
        )

        # low test case count
        self._diagnose_low_test_case_problem(
            eval_results=eval_results,
            models=self.models,
            test_case_minimum=self.args.get(evaluators.Evaluator.PARAM_MIN_TEST_CASES),
        )

        # threshold failures - custom problem generation for dataset-level drift
        self._diagnose_drift_threshold_problems(
            leaderboard=leaderboard_explanation,
            drift_threshold=drift_threshold,
        )

    def _diagnose_drift_threshold_problems(
        self,
        leaderboard: e10s.LlmHeatmapLeaderboardExplanation,
        drift_threshold: float,
    ):
        """Generate problems for drift threshold violations.

        Note: Questions Drift is a dataset-level metric, so we generate one problem
        per unique drift score (typically one, since all models get the same score).
        """
        leaderboard_data = leaderboard.as_dict()[0].get(f5s.ExplanationFormat.KEY_DATA)
        if not leaderboard_data:
            return

        primary_metric_meta = self._metrics_meta.get_primary_metric()

        # collect unique drift scores (typically just one for dataset-level metric)
        drift_scores_seen = set()

        for model_id in leaderboard_data:
            metric_score = leaderboard_data[model_id].get(primary_metric_meta.key)

            # check if this drift score exceeds threshold
            if (
                metric_score is not None
                and metric_score > drift_threshold
                and metric_score not in drift_scores_seen
            ):
                drift_scores_seen.add(metric_score)

                # create dataset-level problem message (not model-specific)
                html = airium.Airium()
                html("Question drift")
                with html.b(klass="w3-black"):
                    html("&nbsp;exceeded threshold&nbsp;")
                with html.code():
                    html(f"&nbsp;{drift_threshold}")
                html(" for metric ")
                with html.code():
                    html(f"{primary_metric_meta.display_name}")
                html(" with drift score ")
                with html.code():
                    html(f"&nbsp;{insights.r(metric_score)}.")
                html(" Metric details: ")
                with html.i():
                    html(primary_metric_meta.description)

                actions_description = (
                    "Significant drift detected in question semantics. This may "
                    "indicate: "
                    "1) test questions changing unexpectedly over time, "
                    "2) new user populations with different question patterns, "
                    "3) topic expansion in the question space, or "
                    "4) quality issues in test case generation. "
                    "Recommended actions: review recent questions for consistency, "
                    "analyze the semantic shift to understand topic changes, "
                    "consider updating evaluation datasets to reflect new patterns, "
                    "and validate that question generation process maintains quality."
                )

                t_problems = problems.ProblemAndAction
                problem = problems.ProblemAndAction(
                    description=(
                        f"Question drift exceeded threshold {drift_threshold} "
                        f"for metric '{primary_metric_meta.display_name}' "
                        f"with drift score {insights.r(metric_score)}. This is "
                        f"a dataset-level metric measuring semantic drift in the "
                        f"input questions themselves (not model performance). "
                        f"Metric details: {primary_metric_meta.description}"
                    ),
                    description_html=html,
                    severity=(
                        problems.ProblemSeverity.medium
                        if metric_score < drift_threshold * 2.0
                        else problems.ProblemSeverity.high
                    ),
                    problem_type="drift",
                    problem_attrs={
                        t_problems.ATTR_M_ID: primary_metric_meta.key,
                        t_problems.ATTR_M_NAME: primary_metric_meta.display_name,
                        t_problems.ATTR_M_THRESHOLD: drift_threshold,
                        t_problems.ATTR_M_SCORE: metric_score,
                        t_problems.ATTR_EVALUATOR_NAME: self._display_name,
                    },
                    problem_code=problems.AVIDProblemCode.P0100_DATA,
                    actions_description=actions_description,
                    explainer_id=self.explainer_id(),
                    explainer_name=self._display_name,
                    explanation_type=e10s.GlobalHtmlFragmentExplanation.explanation_type(),
                    explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
                    explanation_mime=f5s.HtmlFormat.mime,
                )

                self.add_problem(problem)

    def _diagnose_insights(
        self,
        leaderboard_explanation: e10s.LlmHeatmapLeaderboardExplanation,
        drift_threshold: float,
    ):
        """Generate dataset-level insights about question drift.

        Questions Drift is calculated on the test questions/inputs themselves,
        not on model performance. Therefore, all models have the SAME drift score.

        Model-specific insights (best/worst model) are not meaningful here.
        The drift score indicates whether the question semantics have shifted
        over time in the test dataset, regardless of which model is being evaluated.

        This method generates ONE insight per evaluation run (dataset-level),
        not per model.
        """
        # extract drift scores from leaderboard
        # since all models have same drift score, we only need to check one unique score
        leaderboard_data = leaderboard_explanation.as_dict()[0].get("data", {})

        drift_score = None
        for model_data in leaderboard_data.values():
            score = model_data.get(self.METRIC_QUESTIONS_DRIFT)
            if score is not None and not (isinstance(score, float) and np.isnan(score)):
                drift_score = score
                break

        # if no valid drift score, no insight to generate
        if drift_score is None:
            return

        # generate dataset-level insight based on drift score
        t_html_fragment = e10s.GlobalHtmlFragmentExplanation
        evaluator_name = self._display_name

        # determine if drift is significant
        is_drifted = drift_score >= drift_threshold

        if is_drifted:
            # high drift - questions have drifted significantly
            description = (
                f"Questions have drifted significantly with drift score "
                f"{insights.r(drift_score)}, exceeding threshold {drift_threshold}. "
                f"The semantic content of questions has shifted between the two "
                f"temporal groups, indicating a change in question distribution."
            )

            html = airium.Airium()
            html("Questions have")
            with html.b(klass="w3-red"):
                html("&nbsp;drifted significantly&nbsp;")
            html("with drift score ")
            with html.code():
                html(f"{insights.r(drift_score)}")
            html(", exceeding threshold ")
            with html.code():
                html(f"{drift_threshold}")
            html(
                ". The semantic content of questions has shifted between "
                "temporal groups."
            )

            actions_description = (
                "Investigate the root cause of question drift. Has the user base "
                "changed? Are users asking about different topics or domains? "
                "Consider whether the model needs retraining or fine-tuning on the "
                "new question distribution. Review the questions in both temporal "
                "groups to understand the semantic shift."
            )

            insight_type = "data_quality"
        else:
            # low drift - questions are stable
            description = (
                f"Questions are stable with low drift score {insights.r(drift_score)}, "
                f"below threshold {drift_threshold}. The semantic content of questions "
                f"has remained consistent between the two temporal groups."
            )

            html = airium.Airium()
            html("Questions are")
            with html.b(klass="w3-green"):
                html("&nbsp;stable&nbsp;")
            html("with low drift score ")
            with html.code():
                html(f"{insights.r(drift_score)}")
            html(", below threshold ")
            with html.code():
                html(f"{drift_threshold}")
            html(
                ". The semantic content has remained consistent between "
                "temporal groups."
            )

            actions_description = (
                "No immediate action required. Questions remain semantically "
                "consistent over time. Continue monitoring drift in future "
                "evaluations to detect any changes in question distribution."
            )

            insight_type = "data_quality"

        # create and add the insight
        insight = insights.InsightAndAction(
            description=description,
            description_html=html,
            insight_type=insight_type,
            insight_attrs={
                "drift_score": drift_score,
                "drift_threshold": drift_threshold,
                "is_drifted": is_drifted,
                "evaluator_name": evaluator_name,
            },
            actions_description=actions_description,
            evaluator_id=self.evaluator_id(),
            evaluator_name=evaluator_name,
            explanation_type=t_html_fragment.explanation_type(),
            explanation_name=t_html_fragment.__name__,
            explanation_mime=f5s.HtmlFormat.mime,
            resources=[],
        )

        self.add_insight(insight)

    def get_result(
        self,
    ) -> results.LeaderboardResult:
        return results.LeaderboardResult(
            persistence=self.persistence,
            explainer_id=QuestionsDriftEvaluator.explainer_id(),
            h2o_sonar_config=self.config,
            logger=self.logger,
            explanation=e10s.LlmHeatmapLeaderboardExplanation,
            explanation_format=f5s.CustomJsonFormat,
        )
