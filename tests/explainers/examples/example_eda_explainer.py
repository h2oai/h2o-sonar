# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations
from h2o_sonar.lib.api import results


class ExampleEdaExplainer(explainers.Explainer):
    _display_name = "Example Dataset Explainer"
    _description = "This is Explanatory Data Analysis explainer example."
    _regression = True
    _global_explanation = True
    _explanation_types = [explanations.WorkDirArchiveExplanation]

    def __init__(self):
        explainers.Explainer.__init__(self)

    def setup(self, model, persistence, **kwargs):
        explainers.Explainer.setup(self, model, persistence, **kwargs)

    def explain(self, X, y=None, explanations_types=None, **kwargs) -> list:
        self.logger.debug("explain() method invoked with dataset:")
        self.logger.debug(f"  type:    {type(X)}")
        self.logger.debug(f"  shape:   {X.shape}")
        self.logger.debug(f"  columns: {X.names}")
        self.logger.debug(f"  types:   {X.stypes}")
        self.logger.debug(f"  unique:  {X.nunique()}")
        self.logger.debug(f"  max:     {X.max()}")
        self.logger.debug(f"  min:     {X.min()}")

        return [
            self.create_explanation_workdir_archive(
                display_name=self.display_name, display_category="Demo"
            )
        ]

    def get_result(self) -> results.TemplateResult:
        return results.TemplateResult(
            persistence=self.persistence,
            explainer_id=self.explainer_id(),
            explainer_name=ExampleEdaExplainer._display_name,
        )
