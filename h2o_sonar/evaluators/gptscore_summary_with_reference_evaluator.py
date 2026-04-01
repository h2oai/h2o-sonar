# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.


from h2o_sonar.evaluators import gptscore_evaluator
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import evaluators
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import problems


class GptScoreSummaryWithReferenceEvaluator(gptscore_evaluator.GptScoreEvaluator):
    _display_name = "Summarization with reference (GPT Score)"
    _tagline = "Assess actual answers for coverage, factuality, and informativeness."

    _keywords = [
        evaluators.KEYWORD_GPU_OPT,
        evaluators.KEYWORD_LLM,
        evaluators.KEYWORD_EVALUATES_LLM,
        evaluators.KEYWORD_EVALUATES_RAG,
        evaluators.KEYWORD_RQ_EA,
        evaluators.KEYWORD_RQ_AA,
        evaluators.KEYWORD_PROBLEM_TYPE_SUM,
        evaluators.KEYWORD_ES_SUMMARIZE,
        evaluators.KEYWORD_METHOD_TYPE_DETERMINISTIC,
        evaluators.KEYWORD_METHOD_NLI,
    ]

    METRIC_SEMANTIC_COVERAGE = "semantic coverage"  # COV
    METRIC_FACTUALITY = "factuality"  # FAC
    METRIC_INFORMATIVENESS = "informativeness"  # INF
    METRIC_COHERENCE = "coherence"  # COH
    METRIC_RELEVANCE = "relevance"  # REL
    METRIC_FLUENCY = "fluency"  # FLU

    PARENT = gptscore_evaluator.GptScoreEvaluator

    _ASPECT_DEFINITIONS = {
        METRIC_SEMANTIC_COVERAGE: "Rewrite the following text with the same semantics. "
        "{ref_hypo} In other words, ",
        METRIC_FACTUALITY: "Rewrite the following text with consistent facts. "
        "{ref_hypo} In other words, ",
        METRIC_INFORMATIVENESS: "Rewrite the following text with its core information. "
        "{ref_hypo} In other words, ",
        METRIC_COHERENCE: "Rewrite the following text into a coherent text. "
        "{ref_hypo} In other words, ",
        METRIC_RELEVANCE: "Rewrite the following text with consistent details. "
        "{ref_hypo} In other words, ",
        METRIC_FLUENCY: "Rewrite the following text into a fluent and grammatical text."
        " {ref_hypo} In other words, ",
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
                description="Does the generated text preserve the factual "
                "statements of the source text?",
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
                description="How well is the generated text relevant to "
                "its source text?",
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

    _brief_description = """GPT Score evaluator family is based on a novel  evaluation
framework specifically designed for RAGs and LLMs. It utilizes the inherent abilities
of LLMs, particularly their ability to understand and respond to instructions,
to assess the quality of generated text.

- LLM judge based evaluation.
- Compatibility: RAG and LLM models."""
    _description = evaluators.Evaluator._description_builder(
        brief=f"""{_brief_description}

**Method**:

- The core idea of GPTScore is that a generative pre-trained model will assign
  a higher probability of high-quality generated text following a given instruction
  and context. The score corresponds to average negative log likelihood of
  the generated tokens. In this case the average negative log likelihood is calculated
  from the tokens that follow `In other words,`.
- Instructions used by the evaluator are:
    - **Semantic coverage**:
        - Rewrite the following text with the same semantics.
          `{{ref_hypo}}` In other words, `{{hypo_ref}}`
    - **Factuality**:
        - Rewrite the following text with consistent facts.
          `{{ref_hypo}}` In other words, `{{hypo_ref}}`
    - **Informativeness**:
        - Rewrite the following text with its core information.
          `{{ref_hypo}}` In other words, `{{hypo_ref}}`
    - **Coherence**:
        - Rewrite the following text into a coherent text.
          `{{ref_hypo}}` In other words, `{{hypo_ref}}`
    - **Relevance**:
        - Rewrite the following text with consistent details.
          `{{ref_hypo}}` In other words, {{hypo_ref}}
    - **Fluency**:
        - Rewrite the following text into a fluent and grammatical text.
          `{{ref_hypo}}` In other words, `{{hypo_ref}}`
- Each instruction is evaluated twice - first it uses expected answer
  for `{{ref_hypo}}` and actual answer for `{{hypo_ref}}` and then it is reversed,
  the calculated scores are then averaged.
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

        # check that at least one row has both actual answer and expected answer
        if not self._check_llm_dataset_field_presence(
            params=params,
            require_actual_answer=True,
            require_expected_answer=True,
        ):
            return False

        return True

    def _prepare_inputs(
        self, row: datasets.LlmDataset.LlmDatasetRow, aspect: str
    ) -> tuple[list[str], list[str]]:
        ref = row.expected_output.strip()
        hypo = row.actual_output.strip()

        if len(ref) == 0:
            self.add_problem_for_row(
                problems.ProblemSeverity.low, "Expected Output is empty.", row
            )

        if len(hypo) == 0:
            self.add_problem_for_row(
                problems.ProblemSeverity.low, "Actual Output is empty.", row
            )
        return [
            self._ASPECT_DEFINITIONS[aspect].format(ref_hypo=ref),
            self._ASPECT_DEFINITIONS[aspect].format(ref_hypo=hypo),
        ], [hypo, ref]
