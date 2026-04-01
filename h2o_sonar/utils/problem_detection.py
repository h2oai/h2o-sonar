# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.


import datatable
import pandas as pd

from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import problems
from h2o_sonar.utils import normalization


def get_feature_importance_problems(
    shap_means_dict: dict[str, datatable.Frame],
    threshold: float,
    explainer_id: str,
    explainer_display_name: str,
) -> list[problems.ProblemAndAction]:
    """Get feature importance problems and suggested actions based on SHAP values
    above a specified threshold.

    Parameters
    ----------
    shap_means_dict : dict[str, datatable.Frame]
        A datatable Frame containing Shapley values.
    threshold : float
        Threshold for showing potential data leakage in the most important feature.
    explainer_id : str
        Explainer id.
    explainer_display_name: str
        Explainer display name

    Returns
    -------
    list[problems.ProblemAndAction]
        A list of problems and actions.

    """
    if threshold > 1.0 or threshold < 0.0:
        raise ValueError("Threshold value must be between 1.0 and 0.0")
    problem_list: list[problems.ProblemAndAction] = []
    # iterating for multinomial, since they contain multiple frames with FI
    for class_value, shap_frame in shap_means_dict.items():
        # 'bias' column does not contain any feature importance information
        # therefore removing it from shap_frame
        shap_frame = shap_frame[
            :, [name for name in shap_frame.names if name != "bias"]
        ]
        shap_frame_normalized: datatable.Frame = normalization.normalize_importance(
            shap_frame
        )
        # only look for greatest FI value
        pandas_frame_normalized: pd.DataFrame = (
            shap_frame_normalized.to_pandas().sort_values(by=0, ascending=False, axis=1)
        )
        cell: float = pandas_frame_normalized.iloc[0, 0]
        col_name: str = pandas_frame_normalized.columns[0]
        if cell > threshold:
            class_description: str = (
                f" for class value '{class_value}'" if len(shap_means_dict) > 1 else ""
            )
            problem_description: str = (
                f"Potential feature importance leak detected in the '{col_name}'"
                " column, which exhibits a high relative feature importance of"
                f" {cell:.1%}"
                f"{class_description}."
            )
            problem = problems.ProblemAndAction(
                description=problem_description,
                severity=problems.ProblemSeverity.high,
                problem_type="data",
                actions_description=(
                    "Consider dropping the column or using weights when training"
                    "your model to decrease emphasis the problematic feature."
                ),
                problem_attrs={
                    problems.ProblemAndAction.ATTR_EVALUATOR_NAME: (
                        explainer_display_name
                    ),
                },
                explainer_id=explainer_id,
                explainer_name=explainer_display_name,
                explanation_type=(e10s.GlobalFeatImpExplanation.explanation_type()),
                explanation_name=e10s.GlobalFeatImpExplanation.__name__,
                resources=[],
            )
            problem_list.append(problem)
    return problem_list
