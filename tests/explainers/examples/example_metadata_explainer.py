# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations
from h2o_sonar.lib.api import results


class ExampleMetaAndAttrsExplainer(explainers.Explainer):
    _display_name = "Example Explainer Metadata and Attributes"
    _description = (
        "This explainer example prints explainer metadata, instance attributes and "
        "setup() explainer parameters."
    )
    _regression = True
    _binary = True
    _multiclass = True
    _global_explanation = True
    _explanation_types = [explanations.WorkDirArchiveExplanation]

    def __init__(self):
        explainers.Explainer.__init__(self)

    def setup(self, model, persistence, **e_params):
        explainers.Explainer.setup(self, model, persistence, **e_params)

        self.logger.info("setup() explainer parameters:")
        self.logger.info(f"    {e_params}")

        self.logger.info("Explainer metadata:")
        self.logger.info(f"    display name: {self._display_name}")
        self.logger.info(f"    description: {self._description}")
        self.logger.info(f"    keywords: {self._keywords}")
        self.logger.info(f"    IID: {self._iid}")
        self.logger.info(f"    TS: {self._time_series}")
        self.logger.info(f"    image: {self._image}")
        self.logger.info(f"    regression: {self._regression}")
        self.logger.info(f"    binomial: {self._binary}")
        self.logger.info(f"    multinomial: {self._multiclass}")
        self.logger.info(f"    global: {self._global_explanation}")
        self.logger.info(f"    local: {self._local_explanation}")
        self.logger.info(f"    explanation types: {self._explanation_types}")
        self.logger.info(
            f"    optional explanation types: {self._optional_explanation_types}"
        )
        self.logger.info(f"    parameters: {self._parameters}")
        self.logger.info(f"    not standalone: {self._requires_predict_method}")
        self.logger.info(f"    Python deps: {self._modules_needed_by_name}")
        self.logger.info(f"    explainer deps: {self._depends_on}")
        self.logger.info(f"    priority: {self._priority}")

        self.logger.info("Explainer instance attributes:")
        self.logger.info(f"    explainer params: {self.explainer_params}")
        self.logger.info(f"    interpretation params: {self.params}")
        self.logger.info(f"    explainer dependencies: {self.dependencies}")
        self.logger.info(f"    model with predict method: {self.model}")
        self.logger.info(f"    features used by model: {self.model_meta.used_features}")
        self.logger.info(f"    target labels: {self.model_meta.labels}")
        self.logger.info(f"    number of target labels: {self.model_meta.num_labels}")
        self.logger.info(f"    persistence: {self.persistence}")
        self.logger.info(f"    MLI key: {self.mli_key}")
        self.logger.info(f"    model entity: {self.model_meta}")
        self.logger.info(f"    dataset entity: {self.dataset_meta}")
        self.logger.info(f"    validation dataset entity: {self.validset_meta}")
        self.logger.info(f"    test dataset entity: {self.testset_meta}")
        self.logger.info(f"    sanitization map: {self.model_meta.sanitization_map}")
        self.logger.info(f"    host runtime configuration: {self.config}")

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
            explainer_name=ExampleMetaAndAttrsExplainer._display_name,
        )
