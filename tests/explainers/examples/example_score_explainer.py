# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations
from h2o_sonar.lib.api import results


class ExampleScoreExplainer(explainers.Explainer):
    _display_name = "Example Score Explainer"
    _description = (
        "This is explainer example which demonstrates how to get model predict "
        "method and use it to score dataset."
    )
    _regression = True
    _global_explanation = True
    _explanation_types = [explanations.WorkDirArchiveExplanation]

    def __init__(self):
        explainers.Explainer.__init__(self)

    def setup(self, model, persistence, **e_params):
        explainers.Explainer.setup(self, model, persistence, **e_params)

    def explain(self, X, y=None, explanations_types=None, **kwargs) -> list:
        self.logger.info(f"Dataset to score: {X}")

        # model predict method
        prediction = self.model.predict(X)
        self.logger.info(
            f"Prediction (type={type(prediction)}, shape{prediction.shape}):\n"
            f"{prediction}"
        )

        return [
            self.create_explanation_workdir_archive(
                display_name=self.display_name, display_category="Demo"
            )
        ]

    def get_result(self) -> results.TemplateResult:
        return results.TemplateResult(
            persistence=self.persistence,
            explainer_id=self.explainer_id(),
            explainer_name=ExampleScoreExplainer._display_name,
        )
