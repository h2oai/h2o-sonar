# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import abc
import json
import traceback

import airium


def r(
    value: float | int | str,
    precision: int = 4,
    fail_fast: bool = False,
) -> str:
    """Robust rounding function for metrics values to be used in problems,
    insights and actions messages which always returns a string to be
    used in f-strings.

    Parameters
    ----------
    value : float | int | str
        Value to be rounded.
    precision : int
        Number of decimal places to round the value.
    fail_fast : bool
        If True, then the function raises an exception if the value cannot
        be rounded. Otherwise, the original value is returned.

    Returns
    -------
    float | int | str :
        Rounded value - float if everything is OK, otherwise the original value.

    """
    if isinstance(value, bool) or not isinstance(value, (float, int, str)):
        return str(value)
    try:
        return str(round(float(value), precision))
    except ValueError as e:
        if fail_fast:
            raise ValueError(
                f"Value '{value}' cannot be rounded: {e}\n{traceback.format_exc()}"
            )

    return value


class AbcProblemInsight(abc.ABC):
    KEY_DESCRIPTION = "description"
    KEY_DESCRIPTION_HTML = "description_html"
    KEY_ACTIONS_DESCRIPTION = "actions_description"
    KEY_ACTIONS_CODES = "actions_codes"
    KEY_EXPLAINER_ID = "explainer_id"
    KEY_EXPLAINER_NAME = "explainer_name"
    KEY_EXPLANATION_TYPE = "explanation_type"
    KEY_EXPLANATION_NAME = "explanation_name"
    KEY_EXPLANATION_MIME = "explanation_mime"
    KEY_RESOURCES = "resources"

    ATTR_MODEL_NAME = "model_name"
    ATTR_EVALUATOR_NAME = "evaluator_name"
    ATTR_ROW_KEYS = "dataset_row_keys"
    ATTR_TEST_CASE_KEYS = "test_case_keys"
    ATTR_AVID_PROBLEM_CODE = "avid_problem_code"
    ATTR_AVID_PROBLEM_CODE_DESCRIPTION = "avid_problem_code_description"
    ATTR_COST = "cost"
    ATTR_NAN_PCT = "nan_pct"
    ATTR_NAN_TOLERANCE = "nan_tolerance"
    ATTR_M_ID = "metric_id"
    ATTR_M_NAME = "metric_name"
    ATTR_M_THRESHOLD = "metric_threshold"
    ATTR_M_SCORE = "metric_score"
    ATTR_CHEAPEST_MODEL_NAME = "cheapest_model_name"
    ATTR_MOST_EXPENSIVE_MODEL_NAME = "most_expensive_model_name"
    ATTR_SLOWEST_MODEL_NAME = "slowest_model_name"
    ATTR_FASTEST_MODEL_NAME = "fastest_model_name"

    @staticmethod
    def html_most_least_model_by(
        model_name: str,
        quality: str,
        evaluator_name: str,
        is_most: bool | None = True,
        model_purpose: str = "",
        extra_description: str = "",
    ) -> airium.Airium:
        most_least = "most" if is_most else ("" if is_most is None else "least")

        html = airium.Airium()
        html("Model ")
        with html.code():
            html(model_name)
        html(" was evaluated as ")
        with html.b(klass="w3-black"):
            html(f"&nbsp;the {most_least} {quality}&nbsp;")
        html(f"&nbsp; {model_purpose} model according to ")
        with html.code():
            html(evaluator_name)
        html(f" evaluator. {extra_description}")

        return html

    @staticmethod
    def html_most_least_prompt_by(
        prompt: str,
        quality: str,
        evaluator_name: str,
        is_most: bool | None = True,
    ) -> airium.Airium:
        most_least = "most" if is_most else ("" if is_most is None else "least")

        html = airium.Airium()
        html("Prompt ")
        with html.b():
            with html.i():
                html(f"'{prompt}'")
        html("&nbsp; was evaluated as ")
        with html.b(klass="w3-black"):
            html(f"&nbsp;the {most_least} ")
            html(quality)
            html("&nbsp;")
        html("&nbsp; prompt to be correctly answered according to ")
        with html.code():
            html(evaluator_name)
        html(" evaluator.")

        return html

    @staticmethod
    def html_most_difficult_prompt_by(
        prompt: str,
        evaluator_name: str,
        extra_description: str = "",
        model_purpose: str = "",
    ) -> airium.Airium:
        html = airium.Airium()
        html("Prompt ")
        with html.b():
            with html.i():
                html(f"'{prompt}'")
        html("&nbsp; was evaluated as ")
        with html.b(klass="w3-black"):
            html("&nbsp;the most difficult prompt&nbsp;")
        html("&nbsp; to be correctly answered by evaluated ")
        if model_purpose:
            with html.b(klass="w3-black"):
                html(f"&nbsp;{model_purpose}&nbsp;")
        html("&nbsp; models according to")
        with html.code():
            html(evaluator_name)
        html(" evaluator. ")
        html(extra_description)

        return html

    def __init__(
        self,
        description: str,
        description_html: airium.Airium | None = None,
        actions_description: str = "",
        actions_codes: list[str] = None,
        explainer_id: str = "",
        explainer_name: str = "",
        evaluator_id: str = "",
        evaluator_name: str = "",
        explanation_type: str = "",
        explanation_name: str = "",
        explanation_mime: str = "",
        resources: list[str] = None,
    ):
        self.description = description
        self.description_html = description_html
        self.actions_description = actions_description
        self.actions_codes = actions_codes or []
        self.explainer_id = explainer_id or evaluator_id
        self.explainer_name = explainer_name or evaluator_name
        self.explanation_type = explanation_type
        self.explanation_name = explanation_name
        self.explanation_mime = explanation_mime
        self.resources = resources or []

    def to_dict(self) -> dict:
        return {
            InsightAndAction.KEY_DESCRIPTION: self.description,
            InsightAndAction.KEY_DESCRIPTION_HTML: (
                str(self.description_html) if self.description_html else None
            ),
            InsightAndAction.KEY_ACTIONS_DESCRIPTION: self.actions_description,
            InsightAndAction.KEY_ACTIONS_CODES: self.actions_codes,
            InsightAndAction.KEY_EXPLAINER_ID: self.explainer_id,
            InsightAndAction.KEY_EXPLAINER_NAME: self.explainer_name,
            InsightAndAction.KEY_EXPLANATION_TYPE: self.explanation_type,
            InsightAndAction.KEY_EXPLANATION_NAME: self.explanation_name,
            InsightAndAction.KEY_EXPLANATION_MIME: self.explanation_mime,
            InsightAndAction.KEY_RESOURCES: self.resources,
        }

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)


class InsightAndAction(AbcProblemInsight):
    """Instance of this class represents an insight related to the interpretation
    and/or evaluation of the model. If the insight is created by an
    explainer/evaluator, then the explainer/evaluator ID and name are
    specified.
    Apart from the insight description, the entry provides also insight category
    (brief characteristic), insight attributes (dictionary of machine processable
    data describing the insight which might be used for instance as an input to
    actions), textual description of suggested actions (if any) to handle the insight
    (actionability), action codes (a standard based actions identifiers), and
    references to resources (explanations, document URLs, ...).

    """

    KEY_INSIGHT_TYPE = "insight_type"
    KEY_INSIGHT_ATTRS = "insight_attrs"

    def __init__(
        self,
        description: str,
        description_html: airium.Airium | None = None,
        insight_type: str = "problem",
        insight_attrs: dict = None,
        actions_description: str = "",
        actions_codes: list[str] = None,
        explainer_id: str = "",
        explainer_name: str = "",
        evaluator_id: str = "",
        evaluator_name: str = "",
        explanation_type: str = "",
        explanation_name: str = "",
        explanation_mime: str = "",
        resources: list[str] = None,
    ):
        """Insight constructor.

        Parameters
        ----------
        description : str,
          Insight description.
        description_html : airium.Airium | None
            Optional HTML description of the insight. If not provided, then the
            string description is used in the HTML report.
        insight_type : str
          Insight type.
        insight_attrs : dict
          Machine processable data describing the insight.
        actions_description : str
          Description of actions to handle the insight.
        actions_codes : list[str]
          List of codes of actions to handle the insight. For instance, it might
          be codes specified by NIST AI 600-1 profile.
        explainer_id : str
          ID of the explainer which created the insight. If not specified,
          then the insight is the evaluation / interpretation insight.
        explainer_name : str
          Display name of the explainer, which created the insight.
        explanation_type : str
          Type of the explanation which can clarify the insight.
        explanation_name : str
          Name of the explanation which can clarify the insight.
        explanation_mime : str
          Media type of the explanation which can clarify the insight.
        resources : list[str]
            List of resources (explanations, document URLs, ...) which can clarify
            the insight.

        """
        AbcProblemInsight.__init__(
            self,
            description=description,
            description_html=description_html,
            actions_description=actions_description,
            actions_codes=actions_codes,
            explainer_id=explainer_id,
            explainer_name=explainer_name,
            evaluator_id=evaluator_id,
            evaluator_name=evaluator_name,
            explanation_type=explanation_type,
            explanation_name=explanation_name,
            explanation_mime=explanation_mime,
            resources=resources,
        )

        self.insight_type = insight_type
        self.insight_attrs = insight_attrs or {}

    def to_dict(self) -> dict:
        as_dict = AbcProblemInsight.to_dict(self)
        as_dict[InsightAndAction.KEY_INSIGHT_TYPE] = self.insight_type
        as_dict[InsightAndAction.KEY_INSIGHT_ATTRS] = self.insight_attrs
        return as_dict

    @staticmethod
    def from_dict(problem_dict: dict) -> "InsightAndAction":
        return InsightAndAction(
            description=problem_dict.get(InsightAndAction.KEY_DESCRIPTION, ""),
            description_html=problem_dict.get(
                InsightAndAction.KEY_DESCRIPTION_HTML, None
            ),
            insight_type=problem_dict.get(InsightAndAction.KEY_INSIGHT_TYPE, "insight"),
            insight_attrs=problem_dict.get(InsightAndAction.KEY_INSIGHT_ATTRS, {}),
            actions_description=problem_dict.get(
                InsightAndAction.KEY_ACTIONS_DESCRIPTION, ""
            ),
            actions_codes=problem_dict.get(InsightAndAction.KEY_ACTIONS_CODES, []),
            explainer_id=problem_dict.get(InsightAndAction.KEY_EXPLAINER_ID, ""),
            explainer_name=problem_dict.get(InsightAndAction.KEY_EXPLAINER_NAME, ""),
            explanation_type=problem_dict.get(
                InsightAndAction.KEY_EXPLANATION_TYPE, ""
            ),
            explanation_name=problem_dict.get(
                InsightAndAction.KEY_EXPLANATION_NAME, ""
            ),
            explanation_mime=problem_dict.get(
                InsightAndAction.KEY_EXPLANATION_MIME, ""
            ),
            resources=problem_dict.get(InsightAndAction.KEY_RESOURCES, []),
        )
