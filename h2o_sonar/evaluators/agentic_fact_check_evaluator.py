# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
from h2o_sonar.evaluators import abc_agentic_h2ogpte_evaluator
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import evaluators
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import problems


# aliases
t_booL_leaderboard = e10s.LlmBoolLeaderboardExplanation
t_abc_evaluator = abc_agentic_h2ogpte_evaluator


class FactCheckAgenticEvaluator(t_abc_evaluator.AbcAgenticH2ogpteEvaluator):
    _display_name = "Fact-check"
    _tagline = "Agent-based fact-checking evaluator for LLM and RAG models."

    def _init_eval_instructions(self) -> str:
        return """
    Fact-check the text and determine whether the text contains false information.
    Use the internet search to verify the information if needed.
    """

    METRIC_FACT_CHECK = "fact_check"

    _metrics_meta = commons.MetricsMeta(
        metrics=[
            commons.MetricMeta(
                key=METRIC_FACT_CHECK,
                display_name="Fact-check",
                description=(
                    "Percentage of false information detected in the actual answer. "
                    "The evaluator uses h2oGPTe agents to determine whether the actual "
                    "answer contains false information."
                ),
                higher_is_better=True,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=True,
            ),
            t_booL_leaderboard.METRIC_META_MODEL_PARSE_FAILURES,
        ]
    )

    _llm = True
    _rag = True

    _keywords = [
        evaluators.KEYWORD_LLM,
        evaluators.KEYWORD_EVALUATES_LLM,
        evaluators.KEYWORD_EVALUATES_RAG,
        evaluators.KEYWORD_RQ_AA,
        evaluators.KEYWORD_RQ_P,
        evaluators.KEYWORD_SR_11_7_CS,
        evaluators.KEYWORD_SR_11_7_OGA,
        evaluators.KEYWORD_SR_11_7_OA,
        evaluators.KEYWORD_NIST_AI_RMF_S,
        evaluators.KEYWORD_NIST_AI_RMF_PE,
        evaluators.KEYWORD_NIST_AI_RMF_AT,
        evaluators.KEYWORD_NIST_AI_RMF_VR,
        evaluators.KEYWORD_PROBLEM_TYPE_IR,
        evaluators.KEYWORD_PROBLEM_TYPE_QA,
        evaluators.KEYWORD_EVALUATOR_ROLE_REGULATOR,
        evaluators.KEYWORD_METHOD_AGENTS,
        evaluators.KEYWORD_METHOD_TYPE_NON_DETERMINISTIC,
        evaluators.KEYWORD_CAP_AH,
        evaluators.KEYWORD_ES_GENERATE,  # required for H2O Eval Studio eval eye
    ]

    _brief_description = """Fact-checking evaluator which uses h2oGPTe agents to
determine whether the actual answer contains false information.

- Agents-based fact checking.
- Compatibility: RAG and LLM models."""
    _description = evaluators.Evaluator._description_builder(
        brief=f"""{_brief_description}

**Method**:

- h2oGPTe agents are used to fact-check the text and determine whether the text
  contains false information. Agents can use the internet search and various
  other tools to verify the information if needed.
    """,
        metrics_meta=_metrics_meta,
        keywords=_keywords,
        parameters=t_abc_evaluator.AbcAgenticH2ogpteEvaluator._parameters,
        leaderboard_type=e10s.LlmHeatmapLeaderboardExplanation.explanation_type(),
    )

    _PROBLEM_THRESHOLD_PROTO = problems.ProblemAndAction(
        description="",
        problem_type="fact-check",
        problem_code=problems.AVIDProblemCode.E0100_BIAS,
        actions_description=(
            "To mitigate fact-checking issues, strive to address bias in training data "
            "by including diverse and equitable examples. Additionally, filtering "
            "mechanisms can remove false information, and the model can be trained to "
            "identify and avoid generating false claims. Finally, human review can "
            "help identify and eliminate false outputs before users encounter them. "
            "These steps will lead to more accurate and reliable actual answers."
            "Additionally, consider using multiple fact-checking agents to improve the "
            "accuracy of the evaluation."
        ),
        severity=problems.ProblemSeverity.medium,
    )
