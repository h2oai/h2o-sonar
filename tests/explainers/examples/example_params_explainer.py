# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations
from h2o_sonar.lib.api import results


class ExampleParamsExplainer(explainers.Explainer):
    PARAM_ROWS_TO_SCORE = "rows_to_score"

    _display_name = "Example Params Explainer"
    _description = "This explainer example shows how to define explainer parameters."
    _regression = True
    _global_explanation = True
    _parameters = [
        explainers.ExplainerParam(
            param_name=PARAM_ROWS_TO_SCORE,
            description="The number of dataset rows to be scored by explainer.",
            param_type=commons.ExplainerParamType.int,
            default_value=1000,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
    ]
    _explanation_types = [explanations.WorkDirArchiveExplanation]

    def __init__(self):
        explainers.Explainer.__init__(self)

        self.args = None

    def setup(self, model, persistence, **e_params):
        explainers.Explainer.setup(self, model, persistence, **e_params)

        # resolve explainer parameters to instance attributes
        self.args = explainers.ExplainerArgs(ExampleParamsExplainer._parameters)
        self.args.resolve_params(
            explainer_params=explainers.ExplainerArgs.json_str_to_dict(
                self.explainer_params_as_str
            )
        )

    def explain(self, X, y=None, explanations_types=None, **kwargs) -> list:
        # use parameter
        rows = self.args.get(self.PARAM_ROWS_TO_SCORE)
        df = X[:rows, :]
        self.logger.info(
            f"Dataset after parameter driven concatenation:\n"
            f"  shape: {df.shape}\n"
            f"  columns: {df.names}\n"
        )

        # predict
        prediction = self.model.predict(df)
        self.logger.info(f"Predictions of dataset with shape {df.shape}: {prediction}")
        return [
            self.create_explanation_workdir_archive(
                display_name=self.display_name, display_category="Demo"
            )
        ]

    def get_result(self) -> results.TemplateResult:
        return results.TemplateResult(
            persistence=self.persistence,
            explainer_id=self.explainer_id(),
            explainer_name=ExampleParamsExplainer._display_name,
        )
