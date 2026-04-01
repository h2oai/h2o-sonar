# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
from h2o_sonar.evaluators import rag_ragas_evaluator
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import evaluators
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import results


class FaithfulnessEvaluator(evaluators.Evaluator):
    _display_name = "Faithfulness"
    _tagline = (
        "Evaluate the consistency of the answers with retrieved contexts based on "
        "claims."
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

    _keywords = [
        evaluators.KEYWORD_LLM,
        evaluators.KEYWORD_EVALUATES_RAG,
        evaluators.KEYWORD_RQ_J,
        evaluators.KEYWORD_REQUIRES_OPENAI_KEY,
        evaluators.KEYWORD_SR_11_7_CS,
        evaluators.KEYWORD_SR_11_7_OA,
        evaluators.KEYWORD_NIST_AI_RMF_S,
        evaluators.KEYWORD_PROBLEM_TYPE_IR,
        evaluators.KEYWORD_PROBLEM_TYPE_QA,
        evaluators.KEYWORD_ES_GENERATE,
        evaluators.KEYWORD_METHOD_JUDGE,
        evaluators.KEYWORD_METHOD_TYPE_NON_DETERMINISTIC,
    ]

    _metrics_meta = commons.MetricsMeta(
        metrics=[
            commons.MetricMeta.clone(
                rag_ragas_evaluator.RagasEvaluator.METRIC_META_FAITHFULNESS,
            ),
        ]
    )

    _parameters = [
        evaluators.Evaluator._get_custom_param_metric_threshold(
            _metrics_meta.get_primary_metric()
        ),
        evaluators.Evaluator._PARAM_NAN_TOLERANCE,
        evaluators.Evaluator._PARAM_SAVE_LLM_RESULT,
        evaluators.Evaluator._PARAM_EVAL_JUDGE,
        evaluators.Evaluator._get_custom_param_min_test_case(),
    ]

    _modules_needed_by_name = ["ragas==0.1.3"]

    _brief_description = """Faithfulness Evaluator measures the factual consistency of
the generated answer with the given context.

- LLM finds claims in the actual answer and ensures that these claims present in
  the retrieved context.
- Compatibility: RAG only evaluation.
- Based on
  [RAGAs library](https://docs.ragas.io/en/latest/concepts/metrics/index.html)"""
    _description = evaluators.Evaluator._description_builder(
        brief=f"""{_brief_description}

**Method**:

- Faithfulness is calculated based on the **actual answer** and **retrieved context**.
- The evaluation assesses whether the claims made in the actual answer
  can be inferred from the retrieved context, avoiding any hallucinations.
- The **score** is determined by the **ratio** of the actual answer's claims
  present in the context to the total number of claims in the answer:

```math
                number of claims inferable from the context
faithfulness = ---------------------------------------------
                           claims in the answer
```

See also:

- 3rd party metric documentation:
  https://docs.ragas.io/en/latest/concepts/metrics/index.html
- 3rd party library used: https://github.com/explodinggradients/ragas""",
        metrics_meta=_metrics_meta,
        keywords=_keywords,
        parameters=_parameters,
        leaderboard_type=e10s.LlmHeatmapLeaderboardExplanation.explanation_type(),
    )

    def __init__(self):
        evaluators.Evaluator.__init__(self)

        self.args = None
        self.problems = []
        self.log_name = "Faithfulness"

        self.ragas = rag_ragas_evaluator.RagasEvaluator()

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        **evaluator_params,
    ) -> bool:
        evaluators.Evaluator.check_compatibility(self, params, **evaluator_params)

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

        self.ragas.progress_callback = self.progress_callback
        self.ragas.setup(model, persistence, **kwargs)

    def evaluate(self, llm_testset, explanations_types=None, **kwargs) -> list:
        return self.ragas.eval_custom_metrics(
            llm_testset,
            metrics_threshold=self.args.get(
                evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
                evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
            ),
            save_llm_result=self.args.get(
                evaluators.Evaluator.PARAM_SAVE_LLM_RESULT,
                evaluators.Evaluator.DEFAULT_SAVE_LLM_RESULT,
            ),
            custom_eval_judge_cfg_key=self._resolve_judge_key(),
            metrics_to_run=FaithfulnessEvaluator._metrics_meta,
            evaluator=self,
            nan_tolerance=self.args.get(
                evaluators.Evaluator.PARAM_NAN_TOLERANCE,
                evaluators.Evaluator.DEFAULT_NAN_TOLERANCE,
            ),
        )

    # PROBLEMS: set by RAGAs evaluator on this evaluator instance
    # INSIGHTS: set by RAGAs evaluator on this evaluator instance

    def get_result(
        self,
    ) -> results.LeaderboardResult:
        return results.LeaderboardResult(
            persistence=self.persistence,
            explainer_id=FaithfulnessEvaluator.explainer_id(),
            h2o_sonar_config=self.config,
            logger=self.logger,
            explanation=e10s.LlmHeatmapLeaderboardExplanation,
            explanation_format=f5s.CustomJsonFormat,
        )
