# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import traceback

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
    import transformers

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class RagHallucinationEvaluator(evaluators.Evaluator):
    _display_name = "Hallucination"
    _tagline = (
        "Assess the answers with respect to retrieved contexts to detect fabricated "
        "facts."
    )

    METRIC_HALLUCINATION = "hallucination"

    # suggested by DeepEval for this metric
    DEFAULT_METRIC_THRESHOLD = 0.5

    _metrics_meta = commons.MetricsMeta(
        metrics=[
            commons.MetricMeta(
                key=METRIC_HALLUCINATION,
                display_name="Hallucination",
                description=(
                    "Hallucination metric determines whether the RAG outputs factually "
                    "correct information by comparing the **actual answer** to the "
                    "retrieved **context**. If there are facts in the output that are "
                    "not present in the retrieved context, then the model is "
                    "considered to be hallucinating - fabricates facts that are not "
                    "supported by the context."
                ),
                higher_is_better=True,
                threshold=DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=True,
            )
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
        evaluators.Evaluator._get_custom_param_min_test_case(),
        evaluators.Evaluator._PARAM_SENTENCE_LEVEL_METRICS,
    ]

    _keywords = [
        evaluators.KEYWORD_GPU_OPT,
        evaluators.KEYWORD_LLM,
        evaluators.KEYWORD_EVALUATES_RAG,
        evaluators.KEYWORD_RQ_RC,
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

    # models used by the evaluator
    _e_model_flan = caching.MODEL_GOOGLE_FLAN_T5_BASE
    _e_model_vectara = caching.MODEL_VECTARA_HALLUCINATION
    _vectara_readme = (
        "https://huggingface.co/vectara/hallucination_evaluation_model/blob/"
        + caching.REVISIONS_FOR_MODEL.get(_e_model_vectara, "main")
        + "/README.md"
    )

    _brief_description = """Hallucination Evaluator assesses the hallucination of
the base **LLM model** in a Retrieval Augmented Generation (RAG) pipeline. It
evaluates whether the actual answer is factually correct information by **comparing**
the actual answer to the retrieved context - as the actual answer generated
by the LLM model **must be based on** the retrieved context.  If there are
facts in the output that are not present in the retrieved context, then
the model is considered to be **hallucinating** - **fabricates or discard facts** that
are not supported by the context.

- Fine-tuned flan-t5-base model assessing retrieved context and actual answer
  similarity.
- Compatibility: RAG evaluation only."""
    _description = evaluators.Evaluator._description_builder(
        brief=f"""{_brief_description}

**Method**:

- The evaluation uses
  [{_e_model_vectara}]({_e_model_vectara}) hallucination evaluation a fine-tuned
  [{_e_model_flan}](https://huggingface.co/{_e_model_flan}) model to calculate
  a score that measures the extent of hallucination in the generated answer from
  the retrieved context.
- The hallucination score is calculated as maximum of the hallucination score of the
  retrieved context chunks and the actual answer:

```math
hallucination = max( {{ hallucination_score(c, a): for all c in retrieved_context }} )
```

- Where:
    - `a` is the actual answer.
    - `c` is the retrieved context chunk.
    - `retrieved_context` is the retrieved context.
    - `hallucination_score(c, a)` is the hallucination score of the retrieved context
       chunk `c` and actual answer `a` by the `{caching.MODEL_VECTARA_HALLUCINATION}`
       model (higher is better).

See also:

* Model: {_e_model_vectara} (Hughes Hallucination Evaluation Model, factual consistency
  score `[0.0, 1.0]`, higher is better).""",
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
        self.log_name = "RAG Hallucination"

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        **evaluator_params,
    ) -> bool:
        evaluators.Evaluator.check_compatibility(self, params, **evaluator_params)

        if not HAS_TRANSFORMERS:
            self.logger.error(self._check_compatibility_pckg_err_msg("transformers"))
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
        caching.cache_nltk_punkt(self.logger)
        caching.cache_vectara_hallucination_model(self.logger)

    def evaluate(self, llm_testset, explanations_types=None, **kwargs) -> list:
        save_llm_result = self.args.get(
            evaluators.Evaluator.PARAM_SAVE_LLM_RESULT,
            evaluators.Evaluator.DEFAULT_SAVE_LLM_RESULT,
        )

        return self._evaluate(
            llm_testset=llm_testset,
            save_llm_result=save_llm_result,
        )

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
            sort_by_metric_id=self.METRIC_HALLUCINATION
        )
        heatmap_explanation.add_evalstudio_markdown_format(
            sort_by_metric_id=self.METRIC_HALLUCINATION
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
                            sort_by_metric_id=self.METRIC_HALLUCINATION
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
    def _calculate_for_single_context_document(model, context, output):
        # From README.md:
        # > You may run into a warning message that "Token indices sequence length is
        #  longer than the specified maximum sequence length". Please ignore it which is
        #  inherited from the foundation, T5-base.

        # input to predict: list[Tuple(premise, hypothesis)]
        return model.predict([(context, output)])[0]

    @staticmethod
    def _calculate_hallucination_helper(model, contexts, output):
        max_score = 0
        for c in contexts:
            score = RagHallucinationEvaluator._calculate_for_single_context_document(
                model=model, context=c, output=output
            )
            if isinstance(score, dict):
                if score.get("score", None) is not None:
                    score = float(score["score"])
                else:
                    raise RuntimeError(
                        f"Hallucination score returned as dictionary: "
                        f"'{score}' while max_score is  {max_score}"
                    )
            if score > max_score:
                max_score = score

        return float(max_score)

    def _calculate_hallucination(self, model, row):
        hallucination = RagHallucinationEvaluator._calculate_hallucination_helper(
            model, row.context, row.actual_output
        )

        actual_output_meta = None
        if self.args.get(
            self.PARAM_SENTENCE_LEVEL_METRICS, self.DEFAULT_SENTENCE_LEVEL_METRICS
        ):
            all_actual_output_sentences = nltk.sent_tokenize(row.actual_output)
            text_fragments = []
            for aa in all_actual_output_sentences:
                try:
                    hm = RagHallucinationEvaluator._calculate_hallucination_helper(
                        model, row.context, aa
                    )
                    text_fragments.append(
                        tokenization.TextFragment(
                            text=aa,
                            metrics={self.METRIC_HALLUCINATION: hm},
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

        return hallucination, actual_output_meta

    def _calculate_metrics(self, llm_testset):
        llm_dataset = datasets.LlmDataset.from_datatable_dict(llm_testset.to_dict())

        # evaluator runs only ONE metric at a time
        self.report_progress(0.01, "Configuring metrics...")

        metrics = []
        actual_output_metas = []

        device = h2o_sonar_config.config.resolve_gpu_cpu_device(
            result_format="torch",
        )
        with resource_mgmt.PytorchModelLifeCycleManager(
            # GPU
            transformers.pipeline(
                task="text-classification",
                model=caching.MODEL_VECTARA_HALLUCINATION,
                tokenizer=transformers.AutoTokenizer.from_pretrained(
                    caching.MODEL_GOOGLE_FLAN_T5_BASE,
                ),
                trust_remote_code=True,
                revision=caching.REVISIONS_FOR_MODEL.get(
                    caching.MODEL_VECTARA_HALLUCINATION, "main"
                ),
                padding=True,
                truncation=True,
                device=device,
            )
            # CPU
            # AutoModelForSequenceClassification.from_pretrained(
            #     caching.MODEL_VECTARA_HALLUCINATION,
            #     trust_remote_code=True,
            #     revision=caching.REVISIONS_FOR_MODEL.get(
            #         caching.MODEL_VECTARA_HALLUCINATION, "main"
            #     ),
            # )
        ) as hallucination_model:
            # for every test case run metric (row by row)
            for e, r in enumerate(llm_dataset.inputs):
                # progress
                self.report_progress(
                    progress=progress_utils.ProgressCallbackContext.progress_for_steps(
                        e + 1, len(llm_dataset.inputs)
                    ),
                    message=evaluators.Evaluator._eval_row_progress_msg(
                        metric_name=self.METRIC_HALLUCINATION,
                        device=device,
                        row=e + 1,
                        total_rows=len(llm_dataset.inputs),
                    ),
                )

                # handle actual answer retrieval error ~ RAG/LLM client crash
                if evaluators.Evaluator._is_internal_err_answer(r.actual_output):
                    # set WORST metrics values
                    metrics.append(0.0)
                    actual_output_metas.append([])
                else:
                    metric, actual_output_meta = self._calculate_hallucination(
                        hallucination_model, r
                    )
                    metrics.append(metric)
                    actual_output_metas.append(actual_output_meta)

        # inject result metrics to testset > evaluation result
        eval_results = datasets.LlmEvalResults()
        for i, rr in enumerate(llm_dataset.inputs):
            # add result row
            eval_results.add_result(
                datasets.LlmEvalResults.LlmEvalResultRow(
                    dataset_row=rr,
                    metrics={self.METRIC_HALLUCINATION: metrics[i]},
                    actual_output_meta=(
                        [actual_output_metas[i]] if actual_output_metas[i] else []
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
            problem_type="accuracy",
            problem_code=problems.AVIDProblemCode.P0200_MODEL,
            actions_description=(
                "Focus on three areas: training data, grounding techniques, and "
                "factual consistency checks. First, curating training data to include "
                "reliable sources and diverse perspectives can minimize exposure to "
                "factual inaccuracies. Second, incorporating grounding techniques "
                "like entity verification and external knowledge bases can help "
                "the model anchor its responses in real-world information. Finally, "
                "implementing factual consistency checks during generation, such "
                "as cross-referencing with knowledge bases or reliable sources, "
                "can help identify and flag potential hallucinations before they "
                "reach the user. This three-pronged approach can significantly "
                "reduce the presence of hallucinations in RAG model outputs."
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
                "Focus on three areas: training data, grounding techniques, and "
                "factual consistency checks. First, curating training data to include "
                "reliable sources and diverse perspectives can minimize exposure to "
                "factual inaccuracies. Second, incorporating grounding techniques "
                "like entity verification and external knowledge bases can help "
                "the model anchor its responses in real-world information. Finally, "
                "implementing factual consistency checks during generation, such "
                "as cross-referencing with knowledge bases or reliable sources, "
                "can help identify and flag potential hallucinations before they "
                "reach the user. This three-pronged approach can significantly "
                "reduce the presence of hallucinations in RAG model outputs."
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
            explainer_id=RagHallucinationEvaluator.explainer_id(),
            h2o_sonar_config=self.config,
            logger=self.logger,
            explanation=e10s.LlmHeatmapLeaderboardExplanation,
            explanation_format=f5s.CustomJsonFormat,
        )
