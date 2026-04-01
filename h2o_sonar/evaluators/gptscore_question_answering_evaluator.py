# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.


from h2o_sonar.evaluators import gptscore_evaluator
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import evaluators
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import problems
from h2o_sonar.utils import caching


class GptScoreQuestionAnsweringEvaluator(gptscore_evaluator.GptScoreEvaluator):
    _display_name = "Question answering (GPTScore)"
    _tagline = "Assess actual answers for fluency, engagement, and relevance."

    _keywords = [
        evaluators.KEYWORD_GPU_OPT,
        evaluators.KEYWORD_LLM,
        evaluators.KEYWORD_EVALUATES_LLM,
        evaluators.KEYWORD_EVALUATES_RAG,
        evaluators.KEYWORD_RQ_P,
        evaluators.KEYWORD_RQ_AA,
        evaluators.KEYWORD_PROBLEM_TYPE_SUM,
        evaluators.KEYWORD_ES_GENERATE,
        evaluators.KEYWORD_METHOD_TYPE_DETERMINISTIC,
        evaluators.KEYWORD_METHOD_NLI,
    ]

    METRIC_INTEREST = "interest"  # INT
    METRIC_ENGAGEMENT = "engagement"  # ENG
    METRIC_UNDERSTANDABILITY = "understandability"  # UND
    METRIC_RELEVANCE = "relevance"  # REL
    METRIC_SPECIFIC = "specific"  # SPE
    METRIC_CORRECTNESS = "correctness"  # COR
    METRIC_SEMANTICALLY_APPROPRIATE = "semantically appropriate"  # SEM
    METRIC_FLUENCY = "fluency"  # FLU

    PARENT = gptscore_evaluator.GptScoreEvaluator

    _ASPECT_DEFINITIONS = {
        METRIC_INTEREST: "Answer the question based on the conversation between a "
        "human and AI.\nQuestion: Are the responses of "
        "AI interesting? (a) Yes. (b) No.\nConversation: {history}\n",
        METRIC_ENGAGEMENT: "Answer the question based on the conversation between "
        "a human and AI.\nQuestion: Are the responses of "
        "AI engaging? (a) Yes. (b) No.\nConversation: {history}\n",
        METRIC_UNDERSTANDABILITY: "Answer the question based on the conversation "
        "between a human and AI.\nQuestion: Are the "
        "responses of AI understandable? "
        "(a) Yes. (b) No.\nConversation: {history}\n",
        METRIC_RELEVANCE: "Answer the question based on the conversation between a "
        "human and AI.\nQuestion: Are the responses of AI relevant "
        "to the conversation? (a) Yes. (b) No.\nConversation: "
        "{history}\n",
        METRIC_SPECIFIC: "Answer the question based on the conversation between a "
        "human and AI.\nQuestion: Are the responses of AI generic "
        "or specific to the conversation? (a) Yes. (b) No.\n"
        "Conversation: {history}\n",
        METRIC_CORRECTNESS: "Answer the question based on the conversation between a "
        "human and AI.\nQuestion: Are the responses of AI correct "
        "to conversations? (a) Yes. (b) No.\n"
        "Conversation: {history}\n",
        METRIC_SEMANTICALLY_APPROPRIATE: "Answer the question based on the "
        "conversation between a human and AI.\n"
        "Question: Are the responses of AI "
        "semantically appropriate? (a) Yes. (b) No.\n"
        "Conversation: {history}\n",
        METRIC_FLUENCY: "Answer the question based on the conversation between a "
        "human and AI.\nQuestion: Are the responses of AI fluently "
        "written? (a) Yes. (b) No.\nConversation: {history}\n",
    }

    _metrics_meta = commons.MetricsMeta(
        metrics=[
            commons.MetricMeta(
                key=METRIC_INTEREST,
                display_name="Interest",
                description="Is the generated text interesting?",
                higher_is_better=False,
                threshold=gptscore_evaluator.GptScoreEvaluator.DEFAULT_METRIC_THRESHOLD,
                value_range=(0, float("inf")),
                is_primary_metric=True,
            ),
            commons.MetricMeta(
                key=METRIC_ENGAGEMENT,
                display_name="Engagement",
                description="Is the generated text engaging?",
                higher_is_better=False,
                threshold=gptscore_evaluator.GptScoreEvaluator.DEFAULT_METRIC_THRESHOLD,
                value_range=(0, float("inf")),
                is_primary_metric=False,
            ),
            commons.MetricMeta(
                key=METRIC_UNDERSTANDABILITY,
                display_name="Understandability",
                description="Is the generated text understandable?",
                higher_is_better=False,
                threshold=gptscore_evaluator.GptScoreEvaluator.DEFAULT_METRIC_THRESHOLD,
                value_range=(0, float("inf")),
                is_primary_metric=False,
            ),
            commons.MetricMeta(
                key=METRIC_RELEVANCE,
                display_name="Relevance",
                description="How well is the generated text relevant to its source "
                "text?",
                higher_is_better=False,
                threshold=gptscore_evaluator.GptScoreEvaluator.DEFAULT_METRIC_THRESHOLD,
                value_range=(0, float("inf")),
                is_primary_metric=False,
            ),
            commons.MetricMeta(
                key=METRIC_SPECIFIC,
                display_name="Specific",
                description="Is the generated text generic or specific to the source "
                "text?",
                higher_is_better=False,
                threshold=gptscore_evaluator.GptScoreEvaluator.DEFAULT_METRIC_THRESHOLD,
                value_range=(0, float("inf")),
                is_primary_metric=False,
            ),
            commons.MetricMeta(
                key=METRIC_CORRECTNESS,
                display_name="Correctness",
                description="Is the generated text correct or was there a "
                "misunderstanding of the source text?",
                higher_is_better=False,
                threshold=gptscore_evaluator.GptScoreEvaluator.DEFAULT_METRIC_THRESHOLD,
                value_range=(0, float("inf")),
                is_primary_metric=False,
            ),
            commons.MetricMeta(
                key=METRIC_SEMANTICALLY_APPROPRIATE,
                display_name="Semantically Appropriate",
                description="Is the generated text semantically appropriate?",
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

    _model_gpt2 = caching.MODEL_GPT2_MEDIUM

    _brief_description = """GPT Score evaluator family is based on a novel  evaluation
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
  from the tokens `Answer: Yes`.
- Instructions used by the evaluator are:
    - **Interest**:
        - Answer the question based on the conversation between a human and AI.
        - Question: Are the responses of AI interesting? (a) Yes. (b) No.
        - Conversation: `{{history}}`
        - Answer: Yes
    - **Engagement**:
        - Answer the question based on the conversation between a human and AI.
        - Question: Are the responses of AI engaging? (a) Yes. (b) No.
        - Conversation: `{{history}}`
        - Answer: Yes'
    - **Understandability**:
        - Answer the question based on the conversation between a human and AI.
        - Question: Are the responses of AI understandable? (a) Yes. (b) No.
        - Conversation: `{{history}}`
        - Answer: Yes
    - **Relevance**:
        - Answer the question based on the conversation between a human and AI.
        - Question: Are the responses of AI relevant to the conversation? (a) Yes. (
          b) No.
        - Conversation: `{{history}}`
        - Answer: Yes
    - **Specific**:
        - Answer the question based on the conversation between a human and AI.
        - Question: Are the responses of AI generic or specific to the conversation?
          (a) Yes. (b) No.
        - Conversation: `{{history}}`
        - Answer: Yes'
    - **Correctness**:
        - Answer the question based on the conversation between a human and AI.
        - Question: Are the responses of AI correct to conversations? (a) Yes. (b) No.
        - Conversation: `{{history}}`
        - Answer: Yes
    - **Semantically appropriate**:
        - Answer the question based on the conversation between a human and AI.
        - Question: Are the responses of AI semantically appropriate? (a) Yes. (b) No.
        - Conversation: `{{history}}`
        - Answer: Yes
     - **Fluency**:
        - Answer the question based on the conversation between a human and AI.
        - Question: Are the responses of AI fluently written? (a) Yes. (b) No.
        - Conversation: `{{history}}`
        - Answer: Yes
- Where `{{history}}` corresponds to the conversation - question and actual answer.
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

        question = row.i.strip()
        answer = row.actual_output.strip()

        if len(question) == 0:
            self.add_problem_for_row(
                problems.ProblemSeverity.low, "Input is empty.", row
            )
        if len(answer) == 0:
            self.add_problem_for_row(
                problems.ProblemSeverity.low, "Actual Output is empty.", row
            )

        history = f"\nhuman: {question}\nAI: {answer}"

        return [self._ASPECT_DEFINITIONS[aspect].format(history=history)], [
            "Answer: Yes"
        ]
