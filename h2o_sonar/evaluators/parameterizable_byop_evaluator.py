# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.


from h2o_sonar.evaluators import abc_byop_evaluator
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import evaluators
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import models


class ParameterizableByopEvaluator(abc_byop_evaluator.AbcByopEvaluator):
    _display_name = "BYOP: Bring your own prompt"
    _tagline = "Evaluate actual answers using user-supplied prompt and an LLM."

    PROMPT_TEMPLATE_PARAM: str = "prompt_template"

    _parameters = abc_byop_evaluator.AbcByopEvaluator._parameters + [
        evaluators.EvaluatorParam(
            param_name=PROMPT_TEMPLATE_PARAM,
            param_type=commons.EvaluatorParamType.str,
            description=(
                "Parameterizable BYOP evaluator evaluates prompt created from the "
                "prompt template. Prompt template is expected to guide the LLM to "
                'respond using "true" or "false". "true" corresponds to success and'
                '"false" corresponds to failure.'
                "Prompt template has to contain at least one of the "
                f"following keys: "
                f"'{abc_byop_evaluator.AbcByopEvaluator.IDENTIFIER_CONTEXT}', "
                f"'{abc_byop_evaluator.AbcByopEvaluator.IDENTIFIER_INPUT}', "
                f"'{abc_byop_evaluator.AbcByopEvaluator.IDENTIFIER_ACTUAL_OUTPUT}', "
                f"'{abc_byop_evaluator.AbcByopEvaluator.IDENTIFIER_EXPECTED_OUTPUT}'.\n"
                f"- {abc_byop_evaluator.AbcByopEvaluator.IDENTIFIER_CONTEXT} "
                "corresponds to retrieved "
                "context when used in RAG, for LLM usage it's undefined.\n"
                f"- {abc_byop_evaluator.AbcByopEvaluator.IDENTIFIER_INPUT} "
                f"corresponds to user prompt, e.g., question.\n"
                f"- {abc_byop_evaluator.AbcByopEvaluator.IDENTIFIER_ACTUAL_OUTPUT} "
                f"corresponds to output of the LLM, e.g., answer.\n"
                f"- {abc_byop_evaluator.AbcByopEvaluator.IDENTIFIER_EXPECTED_OUTPUT} "
                f"corresponds to expected output of the LLM."
            ),
        ),
    ]

    _keywords = [
        evaluators.KEYWORD_LLM,
        evaluators.KEYWORD_EVALUATES_LLM,
        evaluators.KEYWORD_EVALUATES_RAG,
        evaluators.KEYWORD_RQ_J,
        evaluators.KEYWORD_ES_GENERATE,
        evaluators.KEYWORD_METHOD_TYPE_NON_DETERMINISTIC,
        evaluators.KEYWORD_METHOD_JUDGE,
    ]

    _brief_description = (
        "Bring Your Own Prompt (BYOP) evaluator uses user supplied prompt and "
        "a judge to evaluate LLM or RAG output. Currently implemented BYOP "
        "supports only binary problems, thus the prompt has to guide the judge "
        "to output either `true` or `false`."
    )
    _description = evaluators.Evaluator._description_builder(
        brief=_brief_description,
        metrics_meta=abc_byop_evaluator.AbcByopEvaluator._metrics_meta,
        keywords=_keywords,
        parameters=abc_byop_evaluator.AbcByopEvaluator._parameters,
        leaderboard_type=e10s.LlmBoolLeaderboardExplanation.explanation_type(),
    )

    def _initialize_prompt_template(self) -> str:
        return self.args.get(self.PROMPT_TEMPLATE_PARAM)

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        model: models.ExplainableModel | None = None,
        **explainer_params,
    ) -> bool:
        evaluators.Evaluator.check_compatibility(self, params, **explainer_params)

        self._resolve_explainer_params()

        if self.args.get(ParameterizableByopEvaluator.PROMPT_TEMPLATE_PARAM):
            return True

        return False
