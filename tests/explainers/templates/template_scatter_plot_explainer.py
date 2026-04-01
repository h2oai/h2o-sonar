# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import json

from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations
from h2o_sonar.lib.api import formats
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import results


class TemplateScatterPlotExplainer(explainers.Explainer):
    """Scatter plot explainer template.

    Use this template to create explainer which creates global and local
    explanations.

    """

    _display_name = "Template Scatter Plot"
    _description = (
        "Scatter plot explainer template which can be used to creates "
        "global and local explanations."
    )
    _regression = True
    _binary = True
    _multiclass = False
    _global_explanation = True
    _local_explanation = False
    _explanation_types = [explanations.GlobalScatterPlotExplanation]
    _keywords = [explainers.Explainer.KEYWORD_TEMPLATE]

    def setup(self, model: models.ExplainableModel, persistence, **kwargs):
        explainers.Explainer.setup(self, model=model, persistence=persistence, **kwargs)

    def explain(self, X, y=None, explanations_types: list = None, **kwargs):
        """Explainer returns result mock WITHOUT computation."""

        # explanations list
        model_explanations = list()

        # global explanations
        model_explanations.append(self._explain_global_scatter())

        return model_explanations

    def _explain_global_scatter(self):
        global_explanation = explanations.GlobalScatterPlotExplanation(
            explainer=self,
            display_name="Template Scatter Plot",
            display_category=explanations.Explanation.DISPLAY_CAT_EXAMPLE,
        )

        #
        # JSon explanation representation formed by multiple files
        #
        json_representation = formats.GlobalScatterPlotJSonFormat(
            explanation=global_explanation,
            json_data=json.dumps(TemplateScatterPlotExplainer.JSON_FORMAT_IDX),
            persistence=self.persistence.store,
        )
        # add more format files: per-feature, per-class (saved as added to format)
        # (feature and class names MUST fit names from index file ^)
        for clazz in TemplateScatterPlotExplainer.MOCK_CLASSES:
            json_representation.add_data(
                # IMPROVE: tweak values for every class
                format_data=json.dumps(TemplateScatterPlotExplainer.JSON_FORMAT_F_C),
                # filename must fit the name from index file ^
                file_name=f"scatter_{clazz}.json",
            )

        return global_explanation

    def get_result(self) -> results.TemplateResult:
        return results.TemplateResult(
            persistence=self.persistence,
            explainer_id=self.explainer_id(),
            explainer_name=TemplateScatterPlotExplainer._display_name,
        )

    #
    # JSon scatter plot mock
    #

    MOCK_CLASSES = ["class_A", "class_B", "class_C"]

    # scatter plot
    JSON_FORMAT_IDX: dict = {
        "files": {
            "class_A": "scatter_class_A.json",
            "class_B": "scatter_class_B.json",
            "class_C": "scatter_class_C.json",
        },
        "total_rows": 20,
        "metrics": [{"R2": 0.96}, {"RMSE": 0.03}],
        "documentation": _description,
    }

    # scatter plot: feature-?, class-?
    JSON_FORMAT_F_C: dict = {
        "bias": 0.15,
        "data": [
            {
                "rowId": 1,
                "responseVariable": 25,
                "limePred": 20,
                "modelPred": 30,
                "actual": 40,
            },
            {
                "rowId": 2,
                "responseVariable": 33,
                "limePred": 15,
                "modelPred": 35,
                "actual": 25,
            },
            {
                "rowId": 3,
                "responseVariable": 35,
                "limePred": 50,
                "modelPred": 30,
                "actual": 40,
            },
            {
                "rowId": 4,
                "responseVariable": 70,
                "limePred": 100,
                "modelPred": 80,
                "actual": 90,
            },
            {
                "rowId": 5,
                "responseVariable": 65,
                "limePred": 80,
                "modelPred": 70,
                "actual": 60,
            },
            {
                "rowId": 6,
                "responseVariable": 50,
                "limePred": 70,
                "modelPred": 75,
                "actual": 65,
            },
        ],
    }
