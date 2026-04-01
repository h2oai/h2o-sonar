# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.


from h2o_sonar.evaluators import gptscore_evaluator
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import evaluators
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import problems


class GptScoreSummaryWithoutReferenceEvaluator(gptscore_evaluator.GptScoreEvaluator):
    _display_name = "Summarization without reference (GPT Score)"
    _tagline = "Assess actual answers for coverage, factuality, and fluency."

    _keywords = [
        evaluators.KEYWORD_GPU_OPT,
        evaluators.KEYWORD_LLM,
        evaluators.KEYWORD_EVALUATES_LLM,
        evaluators.KEYWORD_EVALUATES_RAG,
        evaluators.KEYWORD_RQ_P,
        evaluators.KEYWORD_RQ_AA,
        evaluators.KEYWORD_PROBLEM_TYPE_SUM,
        evaluators.KEYWORD_ES_SUMMARIZE,
        evaluators.KEYWORD_METHOD_TYPE_DETERMINISTIC,
        evaluators.KEYWORD_METHOD_NLI,
    ]

    METRIC_SEMANTIC_COVERAGE = "semantic coverage"  # COV
    METRIC_FACTUALITY = "factuality"  # FAC
    METRIC_CONSISTENCY = "consistency"  # CON
    METRIC_INFORMATIVENESS = "informativeness"  # INF
    METRIC_COHERENCE = "coherence"  # COH
    METRIC_RELEVANCE = "relevance"  # REL
    METRIC_FLUENCY = "fluency"  # FLU

    PARENT = gptscore_evaluator.GptScoreEvaluator

    _ASPECT_DEFINITIONS = {
        METRIC_SEMANTIC_COVERAGE: "Generate a summary with as much semantic coverage as"
        " possible for the following text: {src}"
        "\n\nTl;dr\n",
        METRIC_FACTUALITY: "Generate a summary with consistent facts for the following "
        "text: {src}\n\nTl;dr\n",
        METRIC_CONSISTENCY: "Generate factually consistent summary for the following "
        "text: {src}\n\nTl;dr\n",
        METRIC_INFORMATIVENESS: "Generate an informative summary that captures the key "
        "points of the following text: {src}\n\nTl;dr\n",
        METRIC_COHERENCE: "Generate a coherent summary for the following text: "
        "{src}\n\nTl;dr\n",
        METRIC_RELEVANCE: "Generate a relevant summary with consistent details for "
        "the following text: {src}\n\nTl;dr\n",
        METRIC_FLUENCY: "Generate a fluent and grammatical summary for the following "
        "text: {src}\n\nTl;dr\n",
    }

    _metrics_meta = commons.MetricsMeta(
        metrics=[
            commons.MetricMeta(
                key=METRIC_SEMANTIC_COVERAGE,
                display_name="Semantic Coverage",
                description="How many semantic content units from the reference text "
                "are covered by the generated text?",
                higher_is_better=False,
                threshold=gptscore_evaluator.GptScoreEvaluator.DEFAULT_METRIC_THRESHOLD,
                value_range=(0, float("inf")),
                is_primary_metric=True,
            ),
            commons.MetricMeta(
                key=METRIC_FACTUALITY,
                display_name="Factuality",
                description="Does the generated text preserve the factual statements "
                "of the source text?",
                higher_is_better=False,
                threshold=gptscore_evaluator.GptScoreEvaluator.DEFAULT_METRIC_THRESHOLD,
                value_range=(0, float("inf")),
                is_primary_metric=False,
            ),
            commons.MetricMeta(
                key=METRIC_CONSISTENCY,
                display_name="Consistency",
                description="Is the generated text consistent in the information "
                "it provides?",
                higher_is_better=False,
                threshold=gptscore_evaluator.GptScoreEvaluator.DEFAULT_METRIC_THRESHOLD,
                value_range=(0, float("inf")),
                is_primary_metric=False,
            ),
            commons.MetricMeta(
                key=METRIC_INFORMATIVENESS,
                display_name="Informativeness",
                description="How well does the generated text capture the key ideas "
                "of its source text?",
                higher_is_better=False,
                threshold=gptscore_evaluator.GptScoreEvaluator.DEFAULT_METRIC_THRESHOLD,
                value_range=(0, float("inf")),
                is_primary_metric=False,
            ),
            commons.MetricMeta(
                key=METRIC_COHERENCE,
                display_name="Coherence",
                description="How much does the generated text make sense?",
                higher_is_better=False,
                threshold=gptscore_evaluator.GptScoreEvaluator.DEFAULT_METRIC_THRESHOLD,
                value_range=(0, float("inf")),
                is_primary_metric=False,
            ),
            commons.MetricMeta(
                key=METRIC_RELEVANCE,
                display_name="Relevance",
                description="How well is the generated text relevant "
                "to its source text?",
                higher_is_better=False,
                threshold=gptscore_evaluator.GptScoreEvaluator.DEFAULT_METRIC_THRESHOLD,
                value_range=(0, float("inf")),
                is_primary_metric=False,
            ),
            commons.MetricMeta(
                key=METRIC_FLUENCY,
                display_name="Fluency",
                description="Is the generated text well-written and grammatical?",
                higher_is_better=False,
                threshold=gptscore_evaluator.GptScoreEvaluator.DEFAULT_METRIC_THRESHOLD,
                value_range=(0, float("inf")),
                is_primary_metric=False,
            ),
        ]
    )

    _parameters = [
        evaluators.Evaluator._get_custom_param_metric_threshold(
            _metrics_meta.get_primary_metric()
        )
    ] + gptscore_evaluator.GptScoreEvaluator._parameters

    _brief_description = """GPT Score evaluator family is based on a novel evaluation
framework specifically designed for RAGs and LLMs. It utilizes the inherent abilities
of LLMs, particularly their ability to understand and respond to instructions, to
assess the quality of generated text.

- LLM judge based evaluation.
- Compatibility: RAG and LLM models."""
    _description = evaluators.Evaluator._description_builder(
        brief=f"""{_brief_description}

**Method**:

- The core idea of GPTScore is that a generative pre-trained model will assign
  a higher probability of high-quality generated text following a given instruction
  and context. The score corresponds to average negative log likelihood of
  the generated tokens. In this case the average negative log likelihood is calculated
   from the tokens that follow `Tl;dr\\n`.
- Instructions used by the evaluator are:
    - **Semantic coverage**:
        - Generate a summary with as much semantic coverage as possible for
          the following text: `{{src}}`
        - Tl;dr
        - `{{target}}`
    - **Factuality**:
        - Generate a summary with consistent facts for the following text: `{{src}}`
        - Tl;dr
        - `{{target}}`
    - **Consistency**:
        - Generate factually consistent summary for the following text: `{{src}}`
        - Tl;dr
        - `{{target}}`
    - **Informativeness**:
        - Generate an informative summary that captures the key points of
          the following text: `{{src}}`
        - Tl;dr
        - `{{target}}`
    - **Coherence**:
        - Generate a coherent summary for the following text: `{{src}}`
        - Tl;dr
        - `{{target}}`
    - **Relevance**:
        - Generate a relevant summary with consistent details for
          the following text: `{{src}}`
        - Tl;dr
        - `{{target}}`
    - **Fluency**:
        - Generate a fluent and grammatical summary for the following text: `{{src}}`
        - Tl;dr
        - `{{target}}`
- Where `{{src}}` corresponds to the question and `{{target}}` to the actual answer.
- **Average negative log likelihood** of the generated tokens:

```math
           -1 * sum( log( p(x_i | x_1, ..., x_{{i-1}}) ) )
   ANLL = -------------------------------------------------
                             N
```

- Where:
    - `x_i` is the i-th token in the sequence.
    - `N` is the number of tokens in the sequence.
    - `p(x_i | x_1, ..., x_{{i-1}})` is the probability of the i-th token given
       the previous tokens.
    - `Log likehood` for each token finds the probability assigned to it by the model.
       Takes the natural logarithm of this probability.
    - `Negative log likehood` converts the log likelihood from a probability to a loss.
       A higher loss indicates a less likely prediction.
    - `Average negative log likelihood` is the sum of the negative log likelihoods
      of all tokens in the sequence divided by the number of tokens in the sequence.
- The **lower** the metric value is the better.

See also:

- Paper _"GPTScore: Evaluate as You Desire"_: https://arxiv.org/abs/2302.04166
- 3rd party model: [{PARENT._e_model_gpt}](https://huggingface.co/{PARENT._e_model_gpt})
  model is used to calculate the metric values by default (can be changed).""",
        metrics_meta=_metrics_meta,
        keywords=_keywords,
        parameters=_parameters,
        leaderboard_type=e10s.LlmHeatmapLeaderboardExplanation.explanation_type(),
    )

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        **evaluator_params,
    ) -> bool:
        if not super().check_compatibility(params, **evaluator_params):
            return False

        # check that at least one row has actual answer
        if not self._check_llm_dataset_field_presence(
            params=params,
            require_actual_answer=True,
            require_expected_answer=False,
        ):
            return False

        return True

    def _prepare_inputs(
        self, row: datasets.LlmDataset.LlmDatasetRow, aspect: str
    ) -> tuple[list[str], list[str]]:
        # handle empty or invalid actual output before processing
        if not row.actual_output or not isinstance(row.actual_output, str):
            self.logger.warning(
                f"{self.log_name}: Actual output is empty or invalid for row "
                f"with prompt: '{row.i}', model: {row.model_key}."
            )
            self.add_problem_for_row(
                problems.ProblemSeverity.low,
                "Actual output is empty or invalid.",
                row,
            )
            # return minimal valid inputs to avoid scorer errors
            return [""], [""]

        src = row.i.strip()
        hypo = row.actual_output.strip()

        if len(src) == 0:
            self.add_problem_for_row(
                problems.ProblemSeverity.low, "Input is empty.", row
            )
        if len(hypo) == 0:
            self.add_problem_for_row(
                problems.ProblemSeverity.low, "Actual Output is empty.", row
            )

        return [self._ASPECT_DEFINITIONS[aspect].format(src=src)], [hypo]
