# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations
from h2o_sonar.lib.api import results


class ExamplePersistenceExplainer(explainers.Explainer):
    _display_name = "Example Persistence Explainer"
    _description = (
        "This is explainer example which demonstrates how to use persistence object"
        "in order to access explainer file system (sandbox) - working, explanations "
        "and MLI directories."
    )
    _regression = True
    _global_explanation = True
    _explanation_types = [explanations.WorkDirArchiveExplanation]

    def __init__(self):
        explainers.Explainer.__init__(self)

    def setup(self, model, persistence, **kwargs):
        explainers.Explainer.setup(self, model, persistence, **kwargs)

    def explain(self, X, y=None, explanations_types=None, **kwargs) -> list:
        # use self.persistence object to get file system paths
        self.logger.info(f"Explainer MLI dir: {self.persistence.base_dir}")
        self.logger.info(f"Explainer dir: {self.persistence.get_explainer_dir()}")

        # save 1st row of dataset to work directory and prepare work directory archive
        df_head = X[:1, :]
        df_head.to_csv(self.persistence.get_explainer_working_file("dataset_head.csv"))

        return [
            self.create_explanation_workdir_archive(
                display_name=self.display_name, display_category="Demo"
            )
        ]

    def get_result(self) -> results.TemplateResult:
        return results.TemplateResult(
            persistence=self.persistence,
            explainer_id=self.explainer_id(),
            explainer_name=ExamplePersistenceExplainer._display_name,
        )
