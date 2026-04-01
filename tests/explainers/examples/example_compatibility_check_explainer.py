# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations
from h2o_sonar.lib.api import results


class ExampleCompatibilityCheckExplainer(explainers.Explainer):
    _display_name = "Example Compatibility Check Explainer"
    _description = "This is explainer with compatibility check example."
    _regression = True
    _global_explanation = True
    _explanation_types = [explanations.WorkDirArchiveExplanation]

    def __init__(self):
        explainers.Explainer.__init__(self)

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        **explainer_params,
    ) -> bool:
        explainers.Explainer.check_compatibility(self, params, **explainer_params)

        # methods can explain only dataset with less than 1M rows (without sampling)
        if self.dataset_meta.row_count > 1_000_000:
            # not supported
            return False
        return True

    def explain(self, X, y=None, explanations_types=None, **kwargs) -> list:
        return [
            self.create_explanation_workdir_archive(
                display_name=self.display_name, display_category="Demo"
            )
        ]

    def get_result(self) -> results.TemplateResult:
        return results.TemplateResult(
            persistence=self.persistence,
            explainer_id=self.explainer_id(),
            explainer_name=ExampleCompatibilityCheckExplainer._display_name,
        )
