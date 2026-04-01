# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.


from h2o_sonar.evaluators import gptscore_evaluator
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import evaluators
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import problems


class GptScoreMachineTranslationEvaluator(gptscore_evaluator.GptScoreEvaluator):
    _display_name = "Machine translation (GPTScore)"
    _tagline = (
        "Assess the actual answer for accuracy, fluency, and grammatical correctness."
    )

    _keywords = [
        evaluators.KEYWORD_GPU_OPT,
        evaluators.KEYWORD_LLM,
        evaluators.KEYWORD_EVALUATES_LLM,
        evaluators.KEYWORD_EVALUATES_RAG,
        evaluators.KEYWORD_RQ_EA,
        evaluators.KEYWORD_RQ_AA,
        evaluators.KEYWORD_PROBLEM_TYPE_SUM,
        evaluators.KEYWORD_ES_GENERATE,
        evaluators.KEYWORD_METHOD_TYPE_DETERMINISTIC,
        evaluators.KEYWORD_METHOD_NLI,
    ]

    METRIC_ACCURACY = "accuracy"  # ACC
    METRIC_FLUENCY = "fluency"  # FLU
    METRIC_MULTI_QUAL_METRICS = "multidimensional quality metrics"  # MQM

    PARENT = gptscore_evaluator.GptScoreEvaluator

    _ASPECT_DEFINITIONS = {
        METRIC_ACCURACY: "Rewrite the following text with its core information "
        "and consistent facts: {ref_hypo} In other words, ",
        METRIC_FLUENCY: "Rewrite the following text to make it more grammatical "
        "and well-written: {ref_hypo} In other words, ",
        METRIC_MULTI_QUAL_METRICS: "Rewrite the following text into high-quality "
        "text with its core information: {ref_hypo} "
        "In other words, ",
    }
    _metrics_meta = commons.MetricsMeta(
        metrics=[
            commons.MetricMeta(
                key=METRIC_ACCURACY,
                display_name="Accuracy",
                description="Are there inaccuracies, missing, or unfactual "
                "content in the generated text?",
                higher_is_better=False,
                threshold=gptscore_evaluator.GptScoreEvaluator.DEFAULT_METRIC_THRESHOLD,
                value_range=(0, float("inf")),
                is_primary_metric=True,
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
            commons.MetricMeta(
                key=METRIC_MULTI_QUAL_METRICS,
                display_name="Multidimensional Quality Metrics",
                description="How is the overall quality of the generated text?",
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
of LLMs, particularly their ability to understand and respond to instructions, to
assess the quality of generated text.

- LLM judge based evaluation.
- Compatibility: RAG and LLM models."""
    _description = evaluators.Evaluator._description_builder(
        brief=f"""{_brief_description}

**Method**:

- The core idea of GPTScore is that a generative pre-trained model will assign a
higher probability of high-quality generated text following a given instruction and
context. The score corresponds to average **negative log likelihood** of the generated
tokens. In this case the average negative log likelihood is calculated from
the tokens that follow `In other words,`.
- Instructions used by the evaluator are:
    - **Accuracy**: "Rewrite the following text with its core information and consistent
      facts: `{{ref_hypo}}` In other words, `{{hypo_ref}}`"
    - **Fluency**: "Rewrite the following text to make it more grammatical and
      well-written: `{{ref_hypo}}` In other words, `{{hypo_ref}}`"
    - **Multidimensional quality metrics**: "Rewrite the following text into
      high-quality text with its core information:
      `{{ref_hypo}}` In other words, `{{hypo_ref}}`"
- Each instruction is evaluated twice - first it uses expected output for `{{ref_hypo}}`
  and actual answer for `{{hypo_ref}}` and then it is reversed, the calculated scores
  are then averaged.
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
