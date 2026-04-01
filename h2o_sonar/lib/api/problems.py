# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import collections
import enum

import airium

from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import insights


@enum.unique
class ProblemSeverity(enum.Enum):
    high = enum.auto()
    medium = enum.auto()
    low = enum.auto()

    @staticmethod
    def compare(severity1, severity2) -> int:
        if severity1 == severity2:
            return 0
        if severity1 == ProblemSeverity.high:
            return -1
        if severity1 == ProblemSeverity.medium:
            if severity2 == ProblemSeverity.high:
                return 1
            else:
                return -1

        # severity1 == low
        return 1


@enum.unique
class ProblemCode(enum.Enum):
    pass


AVIDProblemCodeType = collections.namedtuple(
    "AVIDProblemCodeType", ["code", "description"]
)


class AVIDProblemCode(ProblemCode):
    """Problem codes from AVID https://docs.avidml.org/taxonomy/effect-sep-view"""

    # SECURITY
    S0400_MODEL_BYPASS = AVIDProblemCodeType(
        "S0400", "Intentionally try to make a model perform poorly"
    )
    S0500_EXFILTRATION = AVIDProblemCodeType(
        "S0500", "Directly or indirectly exfiltrate ML artifacts"
    )
    S0600_DATA_POISONING = AVIDProblemCodeType(
        "S0600", "Usage of poisoned data in the ML pipeline"
    )

    # ETHICS
    E0100_BIAS = AVIDProblemCodeType(
        "E0100", "Concerns of algorithms propagating societal bias"
    )
    E0200_EXPLAINABILITY = AVIDProblemCodeType(
        "E0200", "Ability to explain decisions made by AI"
    )
    E0300_TOXICITY = AVIDProblemCodeType(
        "E0300", "Perpetuating/causing/being affected by negative user actions"
    )
    E0400_MISINFORMATION = AVIDProblemCodeType(
        "E0400", "Perpetuating/causing the spread of falsehoods"
    )

    # PERFORMANCE
    P0100_DATA = AVIDProblemCodeType(
        "P0100", "Problems arising due to faults in the data pipeline"
    )
    P0200_MODEL = AVIDProblemCodeType(
        "P0200", "Ability for the AI to perform as intended"
    )
    P0300_PRIVACY = AVIDProblemCodeType(
        "P0300",
        "Protect leakage of user information as required by rules and regulations",
    )
    P0400_SAFETY = AVIDProblemCodeType("P0400", "Minimizing maximum downstream harms")


class ProblemAndAction(insights.AbcProblemInsight):
    """Instance of this class represents a problem of the interpreted model identified
    by an explainer. Apart from the problem description, the entry provides also
    problem severity, problem category (brief characteristic), problem attributes
    (dictionary of machine processable data describing the problem which might be
    used for instance as an input to actions), textual description of suggested
    actions to mitigate the problem (actionability), explainer which detected the
    problem, and references to resources (explanations, document URLs, ...).

    """

    def __init__(
        self,
        description: str,
        description_html: airium.Airium | None = None,
        severity: ProblemSeverity = ProblemSeverity.medium,
        problem_type: str = "problem",
        problem_attrs: dict = None,
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
        problem_code: ProblemCode = None,
    ):
        """Problem constructor.

        Parameters
        ----------
        description : str,
          Problem description.
        description_html : airium.Airium | None
          Optional HTML description of the problem. If not provided, then the
          string description is used in the HTML report.
        severity : ProblemSeverity
          Problem severity.
        problem_type : str
          Problem type.
        problem_attrs : dict
          Machine processable data describing the problem.
        actions_description : str
          Description of actions to mitigate the problem.
        actions_codes : list[str]
          List of codes of actions to mitigate the problem. For instance, it might
          be codes specified by NIST AI 600-1 profile.
        explainer_id : str
          ID of the explainer which identified the problem.
        explainer_name : str
          Display name of the explainer which identified the problem.
        explanation_type : str
          Type of the explanation which can clarify the problem.
        explanation_name : str
          Name of the explanation which can clarify the problem.
        explanation_mime : str
          Media type of the explanation which can clarify the problem.
        resources : list[str]
            List of resources (explanations, document URLs, ...) which can clarify
            the problem.

        """
        insights.AbcProblemInsight.__init__(
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
        self.problem_attrs = problem_attrs or {}
        if isinstance(problem_code, AVIDProblemCode):
            self.problem_attrs[ProblemAndAction.ATTR_AVID_PROBLEM_CODE] = (
                problem_code.value.code
            )
            self.problem_attrs[ProblemAndAction.ATTR_AVID_PROBLEM_CODE_DESCRIPTION] = (
                problem_code.value.description
            )

        self.severity = severity
        self.problem_type = problem_type

    KEY_SEVERITY = "severity"
    KEY_PROBLEM_TYPE = "problem_type"
    KEY_PROBLEM_ATTRS = "problem_attrs"

    def to_dict(self) -> dict:
        as_dict = insights.AbcProblemInsight.to_dict(self)
        as_dict[ProblemAndAction.KEY_SEVERITY] = str(self.severity.name).upper()
        as_dict[ProblemAndAction.KEY_PROBLEM_TYPE] = self.problem_type
        as_dict[ProblemAndAction.KEY_PROBLEM_ATTRS] = self.problem_attrs
        return as_dict

    @staticmethod
    def from_dict(problem_dict: dict) -> "ProblemAndAction":
        return ProblemAndAction(
            description=problem_dict.get(ProblemAndAction.KEY_DESCRIPTION, ""),
            description_html=problem_dict.get(
                ProblemAndAction.KEY_DESCRIPTION_HTML, None
            ),
            severity=ProblemSeverity[
                problem_dict.get(
                    ProblemAndAction.KEY_SEVERITY, ProblemSeverity.medium.name
                ).lower()
            ],
            problem_type=problem_dict.get(ProblemAndAction.KEY_PROBLEM_TYPE, "problem"),
            problem_attrs=problem_dict.get(ProblemAndAction.KEY_PROBLEM_ATTRS, {}),
            actions_description=problem_dict.get(
                ProblemAndAction.KEY_ACTIONS_DESCRIPTION, ""
            ),
            actions_codes=problem_dict.get(ProblemAndAction.KEY_ACTIONS_CODES, []),
            explainer_id=problem_dict.get(ProblemAndAction.KEY_EXPLAINER_ID, ""),
            explainer_name=problem_dict.get(ProblemAndAction.KEY_EXPLAINER_NAME, ""),
            explanation_type=problem_dict.get(
                ProblemAndAction.KEY_EXPLANATION_TYPE, ""
            ),
            explanation_name=problem_dict.get(
                ProblemAndAction.KEY_EXPLANATION_NAME, ""
            ),
            explanation_mime=problem_dict.get(
                ProblemAndAction.KEY_EXPLANATION_MIME, ""
            ),
            resources=problem_dict.get(ProblemAndAction.KEY_RESOURCES, []),
        )


def problems_for_bool_leaderboard(
    evaluator,
    leaderboard,  # explanations.LlmBoolLeaderboardExplanation
    primary_metric_meta: commons.MetricMeta,
    metric_threshold: float | None = None,
    severity: ProblemSeverity | None = None,
    problem_type: str = "accuracy",
    problem_code: ProblemCode = None,
    explanation_type: str = "",
    explanation_name: str = "",
    explanation_mime: str = "",
    actions_description: str = "",
    extra_description_actions: str = "",
) -> None:
    """Generate problems based on the heatmap leaderboard analytics.

    For models whose average `Passes` metric score is below the threshold,
    a problem is created with the description of the problem,
    severity, problem type, problem attributes, actions description,

    """
    leaderboard_data = leaderboard.as_leaderboard_dict().get(
        f5s.ExplanationFormat.KEY_DATA
    )
    if not leaderboard_data:
        return

    metric_threshold = (
        metric_threshold
        if metric_threshold is not None
        else primary_metric_meta.threshold
    )
    evaluator_name = evaluator._display_name

    for model_id in leaderboard_data:
        metric_score = leaderboard_data[model_id].get(primary_metric_meta.key)
        if metric_score is not None and metric_score < metric_threshold:
            # get all rows and test cases where the model failed
            problematic_rows = []
            problematic_test_cases = set()
            if leaderboard.m_failures and model_id in leaderboard.m_failures:
                for f in leaderboard.m_failures[model_id]:
                    problematic_rows.append((f.row_key, f.model_key))
                    problematic_test_cases.add(f.row_key)

            html = airium.Airium()
            html("Evaluated model")
            with html.code():
                html(f"{model_id}")
            html(" failed to satisfy the ")
            with html.b(klass="w3-black"):
                html("&nbsp;threshold&nbsp;")
            with html.code():
                html(f"&nbsp;{metric_threshold}")
            html(" for metric ")
            with html.code():
                html(f"{primary_metric_meta.display_name}")
            html("with average ")
            with html.b(klass="w3-black"):
                html("&nbsp;score&nbsp;")
            with html.code():
                html(f"&nbsp;{insights.r(metric_score)}.")
            html("Metric details:")
            with html.i():
                html(primary_metric_meta.description)

            problem = ProblemAndAction(
                description=(
                    f"Evaluated model {model_id} failed to satisfy the threshold "
                    f"{metric_threshold} for metric "
                    f"'{primary_metric_meta.display_name}', "
                    f"with average score {insights.r(metric_score)}. "
                    f"{primary_metric_meta.display_name} metric: "
                    f"{primary_metric_meta.description}"
                ),
                description_html=html,
                severity=(
                    severity or ProblemSeverity.medium
                    if metric_score > metric_threshold / 2.0
                    else ProblemSeverity.high
                ),
                problem_type=problem_type,
                problem_code=problem_code,
                problem_attrs={
                    ProblemAndAction.ATTR_MODEL_NAME: model_id,
                    ProblemAndAction.ATTR_M_ID: primary_metric_meta.key,
                    ProblemAndAction.ATTR_M_NAME: primary_metric_meta.display_name,
                    ProblemAndAction.ATTR_M_THRESHOLD: metric_threshold,
                    ProblemAndAction.ATTR_M_SCORE: metric_score,
                    ProblemAndAction.ATTR_ROW_KEYS: problematic_rows,
                    ProblemAndAction.ATTR_TEST_CASE_KEYS: list(problematic_test_cases),
                    ProblemAndAction.ATTR_EVALUATOR_NAME: evaluator._display_name,
                },
                actions_description=(
                    f"{actions_description} {extra_description_actions}"
                ),
                explainer_id=evaluator.explainer_id(),
                explainer_name=evaluator_name,
                explanation_type=explanation_type,
                explanation_name=explanation_name,
                explanation_mime=explanation_mime,
            )

            evaluator.add_problem(problem)


def problems_for_heat_leaderboard(
    evaluator,
    leaderboard,  # explanations.LlmHeatLeaderboardExplanation
    metric_threshold: float | None = None,
    primary_metric_meta=None,
    severity: ProblemSeverity | None = None,
    problem_type: str = "accuracy",
    explanation_type: str = "",
    explanation_name: str = "",
    explanation_mime: str = "",
    actions_description: str = "",
    extra_description_actions: str = "",
    problem_code: ProblemCode = None,
) -> None:
    """Generate problems based on the heatmap leaderboard analytics."""
    leaderboard_data = leaderboard.as_dict()[0].get(f5s.ExplanationFormat.KEY_DATA)
    if not leaderboard_data:
        return

    primary_metric_meta = primary_metric_meta or leaderboard.METRIC_META_MODEL_PASSES
    metric_threshold = (
        metric_threshold
        if metric_threshold is not None
        else primary_metric_meta.threshold
    )
    evaluator_name = evaluator._display_name

    for model_id in leaderboard_data:
        metric_score = leaderboard_data[model_id].get(primary_metric_meta.key)
        if metric_score is not None and (
            (primary_metric_meta.higher_is_better and metric_score < metric_threshold)
            or (
                not primary_metric_meta.higher_is_better
                and metric_score > metric_threshold
            )
        ):
            # get all rows and test cases where the model failed
            problematic_rows = []
            problematic_test_cases = set()
            if leaderboard.m_failures and model_id in leaderboard.m_failures:
                for _, row, _ in leaderboard.m_failures[model_id]:
                    problematic_rows.append(
                        (row.dataset_row.key, row.dataset_row.model_key)
                    )
                    problematic_test_cases.add(row.dataset_row.key)

            html = airium.Airium()
            html("Evaluated model")
            with html.code():
                html(f"{model_id}")
            html(" failed to satisfy the ")
            with html.b(klass="w3-black"):
                html("&nbsp;threshold&nbsp;")
            with html.code():
                html(f"&nbsp;{metric_threshold}")
            html(" for metric ")
            with html.code():
                html(f"{primary_metric_meta.display_name}")
            html("with average ")
            with html.b(klass="w3-black"):
                html("&nbsp;score&nbsp;")
            with html.code():
                html(f"&nbsp;{insights.r(metric_score)}.")
            html("Metric details:")
            with html.i():
                html(primary_metric_meta.description)

            problem = ProblemAndAction(
                description=(
                    f"Evaluated model {model_id} failed to satisfy the threshold "
                    f"{metric_threshold} for metric "
                    f"'{primary_metric_meta.display_name}', "
                    f"with average score {insights.r(metric_score)}. "
                    f"{primary_metric_meta.display_name} metric: "
                    f"{primary_metric_meta.description}"
                ),
                description_html=html,
                severity=(
                    severity or ProblemSeverity.medium
                    if metric_score > metric_threshold / 2.0
                    else ProblemSeverity.high
                ),
                problem_type=problem_type,
                problem_attrs={
                    ProblemAndAction.ATTR_MODEL_NAME: model_id,
                    ProblemAndAction.ATTR_M_ID: primary_metric_meta.key,
                    ProblemAndAction.ATTR_M_NAME: primary_metric_meta.display_name,
                    ProblemAndAction.ATTR_M_THRESHOLD: metric_threshold,
                    ProblemAndAction.ATTR_M_SCORE: metric_score,
                    ProblemAndAction.ATTR_ROW_KEYS: problematic_rows,
                    ProblemAndAction.ATTR_TEST_CASE_KEYS: list(problematic_test_cases),
                    ProblemAndAction.ATTR_EVALUATOR_NAME: evaluator._display_name,
                },
                problem_code=problem_code,
                actions_description=(
                    f"{actions_description} {extra_description_actions}"
                ),
                explainer_id=evaluator.explainer_id(),
                explainer_name=evaluator_name,
                explanation_type=explanation_type,
                explanation_name=explanation_name,
                explanation_mime=explanation_mime,
            )

            evaluator.add_problem(problem)


def problems_for_cls_leaderboard(
    evaluator,
    leaderboard,  # explanations.LlmClassificationLeaderboardExplanation
    metric_threshold: float | None = None,
    primary_metric_meta=None,
    severity: ProblemSeverity | None = None,
    problem_type: str = "classification",
    explanation_type: str = "",
    explanation_name: str = "",
    explanation_mime: str = "",
    actions_description: str = "",
    extra_description_actions: str = "",
    problem_code: ProblemCode = None,
) -> None:
    """Generate problems based on the classification leaderboard analytics."""
    leaderboard_data = leaderboard.as_dict()[0].get(f5s.ExplanationFormat.KEY_DATA)
    if not leaderboard_data:
        return

    primary_metric_meta = primary_metric_meta or leaderboard.METRIC_META_MODEL_PASSES
    metric_threshold = (
        metric_threshold
        if metric_threshold is not None
        else primary_metric_meta.threshold
    )
    evaluator_name = evaluator._display_name

    for model_id in leaderboard_data:
        metric_score = leaderboard_data[model_id].get(primary_metric_meta.key)
        if metric_score is not None and (
            (primary_metric_meta.higher_is_better and metric_score < metric_threshold)
            or (
                not primary_metric_meta.higher_is_better
                and metric_score > metric_threshold
            )
        ):
            html = airium.Airium()
            html("Evaluated model")
            with html.code():
                html(f"{model_id}")
            html(" failed to satisfy the ")
            with html.b(klass="w3-black"):
                html("&nbsp;threshold&nbsp;")
            with html.code():
                html(f"&nbsp;{metric_threshold}")
            html(" for metric ")
            with html.code():
                html(f"{primary_metric_meta.display_name}")
            html("with average ")
            with html.b(klass="w3-black"):
                html("&nbsp;score&nbsp;")
            with html.code():
                html(f"&nbsp;{insights.r(metric_score)}.")
            html("Metric details:")
            with html.i():
                html(primary_metric_meta.description)

            problem = ProblemAndAction(
                description=(
                    f"Evaluated model {model_id} failed to satisfy the threshold "
                    f"{metric_threshold} for metric "
                    f"'{primary_metric_meta.display_name}', "
                    f"with average score {insights.r(metric_score)}. "
                    f"{primary_metric_meta.display_name} metric: "
                    f"{primary_metric_meta.description}"
                ),
                description_html=html,
                severity=(
                    severity or ProblemSeverity.medium
                    if metric_score > metric_threshold / 2.0
                    else ProblemSeverity.high
                ),
                problem_type=problem_type,
                problem_attrs={
                    ProblemAndAction.ATTR_MODEL_NAME: model_id,
                    ProblemAndAction.ATTR_M_ID: primary_metric_meta.key,
                    ProblemAndAction.ATTR_M_NAME: primary_metric_meta.display_name,
                    ProblemAndAction.ATTR_M_THRESHOLD: metric_threshold,
                    ProblemAndAction.ATTR_M_SCORE: metric_score,
                },
                problem_code=problem_code,
                actions_description=(
                    f"{actions_description} {extra_description_actions}"
                ),
                explainer_id=evaluator.explainer_id(),
                explainer_name=evaluator_name,
                explanation_type=explanation_type,
                explanation_name=explanation_name,
                explanation_mime=explanation_mime,
            )

            evaluator.add_problem(problem)
