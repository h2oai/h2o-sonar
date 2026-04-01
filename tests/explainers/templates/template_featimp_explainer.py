# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import json
import random

import datatable

from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations
from h2o_sonar.lib.api import formats
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import results


class TemplateFeatureImportanceExplainer(explainers.Explainer):
    """Feature importance explainer template.

    Use this template to create explainer with global and local feature
    importance explanations.

    """

    _display_name = "Template Feature Importance"
    _description = (
        "Feature importance explainer template which can be used create "
        "explainer with global and local feature importance explanations."
    )
    _regression = True
    _binary = True
    _multiclass = True
    _global_explanation = True
    _local_explanation = True
    _explanation_types = [
        explanations.GlobalFeatImpExplanation,
        explanations.LocalFeatImpExplanation,
    ]
    _keywords = [explainers.Explainer.KEYWORD_TEMPLATE]

    def setup(self, model: models.ExplainableModel, persistence, **kwargs):
        explainers.Explainer.setup(self, model=model, persistence=persistence, **kwargs)

    def explain(self, X, y=None, explanations_types: list = None, **kwargs):
        """Create global and local (pre-computed/cached) explanations.

        Template explainer returns MOCK explainer data - replace mock data
        preparation with actual computation to create real explanations.

        """
        # explanations list
        model_explanations = list()

        # global explanation
        global_explanation = self._explain_global_featimp()
        model_explanations.append(global_explanation)

        # local explanation
        local_explanation = self._explain_local_featimp(
            features=[
                item["label"]
                for item in TemplateFeatureImportanceExplainer.GLOBAL_JSON_FORMAT_F_C[
                    "data"
                ]
            ],
            rows=X.shape[0],
        )
        # associate local explanation with global explanation
        global_explanation.has_local = local_explanation.explanation_type()
        model_explanations.append(local_explanation)

        return model_explanations

    def _explain_global_featimp(self):
        """Create global feature importance explanation with JSon format
        representation. This representation is supported by Grammar of MLI and will
        be rendered in UI.

        """
        global_explanation = explanations.GlobalFeatImpExplanation(
            explainer=self,
            # UI tile name
            display_name="Template Feature Importance",
            # UI tab name
            display_category=explanations.Explanation.DISPLAY_CAT_EXAMPLE,
        )

        # JSon explanation representation is a set of multiple files
        json_representation = formats.GlobalFeatImpJSonFormat(
            explanation=global_explanation,
            json_data=json.dumps(
                TemplateFeatureImportanceExplainer.GLOBAL_JSON_FORMAT_IDX
            ),
            persistence=self.persistence.store,
        )
        # add more format files: per-feature, per-class (saved as added to format)
        # (feature and class names MUST fit names from index file ^)
        for clazz in TemplateFeatureImportanceExplainer.MOCK_CLASSES:
            json_representation.add_data(
                # IMPROVE: tweak values for every class
                format_data=json.dumps(
                    TemplateFeatureImportanceExplainer.GLOBAL_JSON_FORMAT_F_C
                ),
                # filename must fit the name from index file ^
                file_name=f"featimp_{clazz}.json",
            )

        return global_explanation

    def get_result(
        self,
    ) -> results.FeatureImportanceResult:
        return results.FeatureImportanceResult(
            persistence=self.persistence,
            explainer_id=TemplateFeatureImportanceExplainer.explainer_id(),
            h2o_sonar_config=self.config,
        )

    MOCK_CLASSES = ["class_A", "class_B", "class_C"]

    # feature importance
    GLOBAL_JSON_FORMAT_IDX: dict = {
        "files": {
            "class_A": "featimp_class_A.json",
            "class_B": "featimp_class_B.json",
            "class_C": "featimp_class_C.json",
        },
        "total_rows": 20,
        "metrics": [{"R2": 0.96}, {"RMSE": 0.03}],
        "documentation": _description,
    }

    # feature importance: feature-?, class-?
    GLOBAL_JSON_FORMAT_F_C: dict = {
        "bias": 0.15,
        "data": [
            {"label": "PAY_0", "value": 1.00, "scope": "global"},
            {"label": "PAY_2", "value": 0.519, "scope": "global"},
            {"label": "PAY_3", "value": 0.245, "scope": "global"},
            {"label": "PAY_4", "value": 0.208, "scope": "global"},
            {"label": "PAY_5", "value": 0.140, "scope": "global"},
            {"label": "PAY_6", "value": 0.0620, "scope": "global"},
            {"label": "LIMIT_BAL", "value": 0.0406, "scope": "global"},
            {"label": "PAY_AMT1", "value": 0.0331, "scope": "global"},
            {"label": "BILL_AMT1", "value": 0.0308, "scope": "global"},
            {"label": "PAY_AMT4", "value": 0.0122, "scope": "global"},
            {"label": "BILL_AMT2", "value": 0.0113, "scope": "global"},
            {"label": "PAY_AMT2", "value": 0.00971, "scope": "global"},
            {"label": "PAY_AMT5", "value": 0.00923, "scope": "global"},
            {"label": "BILL_AMT5", "value": 0.00827, "scope": "global"},
            {"label": "BILL_AMT4", "value": 0.00800, "scope": "global"},
            {"label": "PAY_AMT3", "value": 0.00751, "scope": "global"},
            {"label": "BILL_AMT3", "value": 0.00635, "scope": "global"},
            {"label": "AGE", "value": 0.00609, "scope": "global"},
            {"label": "PAY_AMT6", "value": 0.00578, "scope": "global"},
            {"label": "BILL_AMT6", "value": 0.00382, "scope": "global"},
        ],
    }

    def _explain_local_featimp(self, features: list, rows: int):
        """Create local feature importance explanation with datatable format
        representation. This representation is supported by Grammar of MLI and will
        be rendered in UI.

        As local explanation will be precomputed and persisted (cached), it will be
        returned by Driverless AI automatically. Therefore this explanation doesn't
        have to implement `explain_local()` explanation for on-demand handling.

        Parameters
        ----------
        features: List[str]
          Feature names.
        rows: int
          Dataset row count for which mock local explanation should be created.

        """
        local_explanation = explanations.LocalFeatImpExplanation(explainer=self)

        # mock data - this data to be replaced with actual computation data,
        # determine frame format from LocalFeatImpDatatableFormat docstring
        data_dict: dict = {}
        for feature in features:
            data_dict[feature] = [random.random() for _ in range(rows)]

        #
        # JSon explanation representation is a set of multiple files
        #
        dt_format = formats.LocalFeatImpDatatableFormat(
            explanation=local_explanation,
            frame=datatable.Frame(data_dict),
            persistence=self.persistence.store,
        )
        local_explanation.add_format(dt_format)

        return local_explanation
