# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations
from h2o_sonar.lib.api import formats
from h2o_sonar.lib.api import results


class MyCustomExplanation(explanations.Explanation):
    """Example of a user defined explanation type."""

    _explanation_type = "user-guide-explanation-example"
    _is_global = True

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        explanations.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class ExampleCustomExplanationExplainer(explainers.Explainer):
    _display_name = "Example Custom Explanation Explainer"
    _description = "Explainer example which shows how to define custom explanation."
    _regression = True
    _global_explanation = True
    _explanation_types = [MyCustomExplanation]

    def __init__(self):
        explainers.Explainer.__init__(self)

    def setup(self, model, persistence, **e_params):
        explainers.Explainer.setup(self, model, persistence, **e_params)

    def explain(self, X, y=None, explanations_types=None, **kwargs) -> list:
        prediction = self.model.predict(X)

        # create CUSTOM explanation
        explanation = MyCustomExplanation(
            explainer=self,
            display_name="Custom Explanation Example",
            display_category="Example",
        )
        # add a text format to CUSTOM explanation
        explanation.add_format(
            formats.TextCustomExplanationFormat(
                explanation=explanation,
                format_data=f"Prediction is: {prediction}",
                format_file=None,
                persistence=self.persistence.store,
            )
        )

        return [explanation]

    def get_result(self) -> results.TemplateResult:
        return results.TemplateResult(
            persistence=self.persistence,
            explainer_id=self.explainer_id(),
            explainer_name=ExampleCustomExplanationExplainer._display_name,
        )
