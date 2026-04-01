# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import datatable

from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import models


class SurrogateRandomForestExplainer(explainers.Explainer):
    """Surrogate Random Forrest model explainer.

    Explainer which builds surrogate Random Forest model (on predictions of
    the interpreted model) to explain the model to be interpreted.

    This explainer is used as dependency by RF feature importance, RF LOCO
    and RF PD explainers.

    """

    @staticmethod
    def is_enabled() -> bool:
        return True

    def __init__(self):
        explainers.Explainer.__init__(self)

        self.args = None

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        model: models.ExplainableModel | None = None,
        **explainer_params,
    ) -> bool:
        explainers.Explainer.check_compatibility(self, params, **explainer_params)

        return True

    def explain(
        self,
        X: datatable.Frame,
        y: datatable.Frame | None = None,
        explanations_types: list = None,
        **e_params,
    ):
        raise NotImplementedError
