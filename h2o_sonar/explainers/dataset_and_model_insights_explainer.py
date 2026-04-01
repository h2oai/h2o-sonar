# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.


import datatable

from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import problems
from h2o_sonar.lib.api import results


class DatasetAndModelInsightsExplainer(explainers.Explainer):
    """Dataset and model insights explainer."""

    _display_name = "Dataset and model insights explainer"
    _description = (
        "The explainer checks the dataset and model for various issues. "
        "For example, it provides problems and actions for missing values "
        "in the target column and a low number of unique values across "
        "columns of a dataset."
    )
    _iid = True
    _regression = True
    _binary = True
    _multiclass = True
    _requires_model = False  # dataset insights are supported only
    _global_explanation = True
    _explanation_types = [e10s.TextExplanation]

    def __init__(self):
        explainers.Explainer.__init__(self)
        self.log_name = "Dataset and model insights"

    def setup(self, model, persistence, **e_params):
        explainers.Explainer.setup(self, model, persistence, **e_params)
        self.log_name = f"Dataset and model insights {self.mli_key}/{self.key}"

    def explain(
        self, X: datatable.Frame, y=None, explanations_types=None, **kwargs
    ) -> list:
        problems_list: list[problems.ProblemAndAction] = self._calculate_problems(X)
        for p in problems_list:
            self.add_problem(p)

        text_explanation = e10s.TextExplanation(
            self, DatasetAndModelInsightsExplainer._display_name
        )
        text_explanation.add_format(
            f5s.TextCustomExplanationFormat(
                explanation=text_explanation,
                format_data=f"{len(problems_list)} dataset problems found.",
                persistence=self.persistence.store,
                format_file="",
            )
        )
        return [text_explanation]

    def explain_problems(self) -> list[problems.ProblemAndAction]:
        return super().explain_problems()

    def _calculate_problems(
        self, X: datatable.Frame
    ) -> list[problems.ProblemAndAction]:
        problem_list: list[problems.ProblemAndAction] = list()
        converted_dataset = X[:, datatable.as_type(datatable.f[:], str)]

        target_col_missing_value_problems: list[problems.ProblemAndAction] = (
            self._check_for_null_in_target_col(converted_dataset)
        )
        if len(target_col_missing_value_problems) > 0:
            problem_list.extend(target_col_missing_value_problems)

        one_unique_val_problem: list[problems.ProblemAndAction] = (
            self._check_for_unique(converted_dataset)
        )
        if len(one_unique_val_problem) > 0:
            problem_list.extend(one_unique_val_problem)

        return problem_list

    def _check_for_unique(self, X: datatable.Frame) -> list[problems.ProblemAndAction]:
        try:
            cols_to_check = self.model.used_features
        except AttributeError:
            self.logger.debug("No used features to check, checking for all features")
            cols_to_check = X.names
        valid_values_filter = (
            (datatable.f[cols_to_check] != "")
            & (datatable.f[cols_to_check] != "?")
            & (datatable.f[cols_to_check] is not None)
            & (datatable.f[cols_to_check] != "nan")
            & (datatable.f[cols_to_check] != "null")
            & (datatable.f[cols_to_check] != "NA")
            & (datatable.f[cols_to_check] != "na")
            & (datatable.f[cols_to_check] != "N/A")
            & (datatable.f[cols_to_check] != "unknown")
            & (datatable.f[cols_to_check] != "inf")
            & (datatable.f[cols_to_check] != "-inf")
            & (datatable.f[cols_to_check] != "1.7976931348623157e+308")
            & (datatable.f[cols_to_check] != "-1.7976931348623157e+308")
        )
        dataset_without_nulls: datatable.Frame = X[
            datatable.rowall(valid_values_filter), :
        ]
        # create statistics table with numbers of unique values
        unique_value_stats = dataset_without_nulls.nunique()
        total_cols: int = dataset_without_nulls.ncols
        cols_with_problems: list[str] = list()
        for c in unique_value_stats:
            if c[0, 0] <= 1:
                cols_with_problems.append(*c.names)
        problem_count = len(cols_with_problems)
        if problem_count > 0:
            severity: problems.ProblemSeverity = problems.ProblemSeverity.low
            percentage_of_cols_with_problems: float = problem_count / total_cols
            if percentage_of_cols_with_problems > 0.6:
                severity = problems.ProblemSeverity.high
            elif percentage_of_cols_with_problems > 0.3:
                severity = problems.ProblemSeverity.medium
            else:
                severity = problems.ProblemSeverity.low

            percentage_of_cols_with_problems_formatted: str = (
                f"{round(percentage_of_cols_with_problems * 100)}"
            )
            problem = problems.ProblemAndAction(
                description=(
                    f"The column{'s' if problem_count > 1 else ''} "
                    f"{cols_with_problems} "
                    f"({percentage_of_cols_with_problems_formatted} %) "
                    f"{'have' if problem_count > 1 else 'has'} only 1 valid unique "
                    f"value."
                ),
                severity=severity,
                problem_type="data",
                problem_attrs={
                    problems.ProblemAndAction.ATTR_EVALUATOR_NAME: (
                        DatasetAndModelInsightsExplainer._display_name
                    ),
                },
                actions_description=(
                    "Consider replacing missing values with valid values or removal "
                    "of the column from the dataset."
                ),
                explainer_id=self.explainer_id(),
                explainer_name=self._display_name,
                explanation_type=(e10s.TextExplanation.explanation_type()),
                explanation_name=e10s.TextExplanation.__name__,
                explanation_mime=f5s.TextCustomExplanationFormat.mime,
                resources=[],
            )
            return [problem]
        return []

    @staticmethod
    def _get_null_rows(X: datatable.Frame, col: str) -> datatable.Frame:
        if col:
            null_filter = (
                (datatable.f[col] == "")
                | (datatable.f[col] == "?")
                | (datatable.f[col] is None)
                | (datatable.f[col] == "null")
                | (datatable.f[col] == "nan")
                | (datatable.f[col] == "NA")
                | (datatable.f[col] == "na")
                | (datatable.f[col] == "N/A")
                | (datatable.f[col] == "unknown")
                | (datatable.f[col] == "inf")
                | (datatable.f[col] == "-inf")
                | (datatable.f[col] == "1.7976931348623157e+308")
                | (datatable.f[col] == "-1.7976931348623157e+308")
            )
        else:
            null_filter = (
                (datatable.f[:] == "")
                | (datatable.f[:] == "?")
                | (datatable.f[:] is None)
                | (datatable.f[:] == "null")
                | (datatable.f[:] == "nan")
                | (datatable.f[:] == "NA")
                | (datatable.f[:] == "na")
                | (datatable.f[:] == "N/A")
                | (datatable.f[:] == "unknown")
                | (datatable.f[:] == "inf")
                | (datatable.f[:] == "-inf")
                | (datatable.f[:] == "1.7976931348623157e+308")
                | (datatable.f[:] == "-1.7976931348623157e+308")
            )
        dataset = X[datatable.rowany(null_filter), :]
        return dataset

    def _check_for_null_in_target_col(
        self, X: datatable.Frame
    ) -> list[problems.ProblemAndAction]:
        total_row_count, total_col_count = X.shape
        target_col = self.params.target_col
        nulls_dataset = DatasetAndModelInsightsExplainer._get_null_rows(X, target_col)
        null_row_count, null_col_count = nulls_dataset.shape
        if null_row_count > 0:
            severity: problems.ProblemSeverity = problems.ProblemSeverity.low
            percentage_of_nulls: float = null_row_count / total_row_count
            if percentage_of_nulls > 0.9:
                severity = problems.ProblemSeverity.high
            elif percentage_of_nulls > 0.5:
                severity = problems.ProblemSeverity.medium
            else:
                severity = problems.ProblemSeverity.low

            percentage_of_nulls_formatted: str = f"{round(percentage_of_nulls * 100)}"
            problem = problems.ProblemAndAction(
                description=(
                    f"Target column contains {null_row_count}"
                    f" row{'s' if null_row_count > 1 else ''}"
                    f" ({percentage_of_nulls_formatted} %)"
                    " which have missing values."
                ),
                severity=severity,
                problem_type="data",
                actions_description=(
                    "The target column should not contain missing values. Consider "
                    "dropping these rows from the dataset."
                ),
                problem_attrs={
                    problems.ProblemAndAction.ATTR_EVALUATOR_NAME: (
                        DatasetAndModelInsightsExplainer._display_name
                    ),
                },
                explainer_id=self.explainer_id(),
                explainer_name=self._display_name,
                explanation_type=(e10s.TextExplanation.explanation_type()),
                explanation_name=e10s.TextExplanation.__name__,
                explanation_mime=f5s.TextCustomExplanationFormat.mime,
                resources=[],
            )
            return [problem]
        return []

    def get_result(self) -> results.TemplateResult:
        return results.TemplateResult(
            persistence=self.persistence,
            explainer_id=self.explainer_id(),
            explainer_name=DatasetAndModelInsightsExplainer._display_name,
            logger=self.logger,
        )
