# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations
from h2o_sonar.lib.api import results


class ExampleLoggingExplainer(explainers.Explainer):
    _display_name = "Example Logging Explainer"
    _description = "This is logging explainer example."
    _regression = True
    _global_explanation = True
    _explanation_types = [explanations.WorkDirArchiveExplanation]

    def __init__(self):
        explainers.Explainer.__init__(self)

    def setup(self, model, persistence, **kwargs):
        explainers.Explainer.setup(self, model, persistence, **kwargs)

        self.logger.info(f"{self.display_name} explainer initialized")

    def explain(self, X, y=None, explanations_types=None, **kwargs) -> list:
        self.logger.debug(f"explain() method invoked with args: {kwargs}")

        if not explanations_types:
            self.logger.warning(
                f"Explanation types to be returned by {self.display_name} not specified"
            )

        try:
            return [
                self.create_explanation_workdir_archive(
                    display_name=self.display_name, display_category="Demo"
                )
            ]
        except Exception as ex:
            self.logger.error(
                f"Explainer '{ExampleLoggingExplainer.__name__}' failed with: {ex}"
            )
            raise ex

    def get_result(self) -> results.TemplateResult:
        return results.TemplateResult(
            persistence=self.persistence,
            explainer_id=self.explainer_id(),
            explainer_name=ExampleLoggingExplainer._display_name,
        )
