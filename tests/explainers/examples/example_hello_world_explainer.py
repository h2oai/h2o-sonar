# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations
from h2o_sonar.lib.api import results


class ExampleHelloWorldExplainer(explainers.Explainer):
    _display_name = "Hello, World!"
    _description = "This is 'Hello, World!' explainer example."
    _regression = True
    _global_explanation = True
    _explanation_types = [explanations.WorkDirArchiveExplanation]

    def __init__(self):
        explainers.Explainer.__init__(self)

    def explain(self, X, y=None, explanations_types=None, **kwargs) -> list:
        explanation = self.create_explanation_workdir_archive(
            display_name=self.display_name, display_category="Demo"
        )

        return [explanation]

    def get_result(self) -> results.TemplateResult:
        return results.TemplateResult(
            persistence=self.persistence,
            explainer_id=self.explainer_id(),
            explainer_name=ExampleHelloWorldExplainer._display_name,
        )
