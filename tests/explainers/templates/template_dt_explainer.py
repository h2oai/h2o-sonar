# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import json
import random

from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations
from h2o_sonar.lib.api import formats
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import results


class TemplateDecisionTreeExplainer(explainers.Explainer):
    """Decision tree explainer template.

    Use this template to create explainer which creates global and local
    decision tree explanations.

    """

    _display_name = "Template Decision Tree"
    _description = (
        "Template decision tree explainer which can be used to create global "
        "and local decision tree explanations."
    )
    _regression = True
    _binary = True
    _multiclass = False
    _global_explanation = True
    _local_explanation = True
    _explanation_types = [explanations.GlobalDtExplanation]
    _keywords = [explainers.Explainer.KEYWORD_TEMPLATE]

    def setup(self, model: models.ExplainableModel, persistence, **kwargs):
        explainers.Explainer.setup(self, model=model, persistence=persistence, **kwargs)

    def explain(self, X, y=None, explanations_types: list = None, **kwargs):
        """Create global explanations (local will be calculated on-demand).

        Template explainer returns MOCK explanation data - replace mock data
        preparation with actual computation to create real explanations.

        """

        # explanations list
        model_explanations = list()

        # global explanations: pre-computed/cached
        global_explanation = self._explain_global_dt()
        model_explanations.append(global_explanation)

        # local explanations: on-demand
        local_explanation = self._explain_local_dt()
        # associate local explanations with the global one
        global_explanation.has_local = local_explanation.explanation_type()
        model_explanations.append(local_explanation)

        return model_explanations

    def _explain_global_dt(self):
        global_explanation = explanations.GlobalDtExplanation(
            explainer=self,
            display_name="Template Decision Tree",
            display_category=explanations.Explanation.DISPLAY_CAT_EXAMPLE,
        )

        #
        # JSon explanation representation formed by multiple files
        #
        json_representation = formats.GlobalDtJSonFormat(
            explanation=global_explanation,
            json_data=json.dumps(TemplateDecisionTreeExplainer.JSON_FORMAT_IDX),
            persistence=self.persistence.store,
        )
        # add more format files: per-feature, per-class (saved as added to format)
        # (feature and class names MUST fit names from index file ^)
        for clazz in TemplateDecisionTreeExplainer.MOCK_CLASSES:
            json_representation.add_data(
                # IMPROVE: tweak values for every class
                format_data=json.dumps(TemplateDecisionTreeExplainer.JSON_FORMAT_F_C),
                # filename must fit the name from index file ^
                file_name=f"dt_{clazz}.json",
            )

        return global_explanation

    def _explain_local_dt(self) -> explanations.LocalDtExplanation:
        """Persist local DT explanation as JSon which indicates that it will be
        created on-demand. Note passing of parameters for subsequent computation.

        """
        local_dt_explanation = explanations.LocalDtExplanation(
            explainer=self,
            display_name="Template Local DT",
            display_category=explanations.GlobalDtExplanation.DISPLAY_CAT_EXAMPLE,
        )

        json_local_idx, _ = formats.LocalDtJSonFormat.serialize_index_file(
            classes=TemplateDecisionTreeExplainer.MOCK_CLASSES,
            doc=TemplateDecisionTreeExplainer._description,
        )
        json_local_idx[formats.LocalDtJSonFormat.KEY_ON_DEMAND] = True
        on_demand_params: dict = dict()
        on_demand_params[formats.LocalDtJSonFormat.KEY_SYNC_ON_DEMAND] = True
        json_local_idx[formats.LocalDtJSonFormat.KEY_ON_DEMAND_PARAMS] = (
            on_demand_params
        )
        local_dt_explanation.add_format(
            explanation_format=formats.LocalDtJSonFormat(
                explanation=local_dt_explanation,
                json_data=json.dumps(json_local_idx, indent=4),
                persistence=self.persistence.store,
            )
        )

        return local_dt_explanation

    def explain_local(self, X, y=None, **extra_params) -> str:
        """On-demand DT surrogate local explanations."""
        dt = json.loads(json.dumps(TemplateDecisionTreeExplainer.JSON_FORMAT_F_C))
        TemplateDecisionTreeExplainer._set_local_dt_keys(
            dt=dt,
            keys_to_set=TemplateDecisionTreeExplainer._get_random_local_path(),
        )
        return json.dumps(dt)

    def get_result(self) -> results.DtResult:
        return results.DtResult(
            persistence=self.persistence,
            h2o_sonar_config=self.config,
            explainer_name="template surrogate decision tree explainer",
            explainer_id=TemplateDecisionTreeExplainer.explainer_id(),
        )

    @staticmethod
    def _get_random_local_path() -> list:
        key = "0"
        keys = [key]
        for _ in range(3):
            key = f"{key}.{random.randint(0, 1)}"
            keys.append(key)
        return keys

    @staticmethod
    def _set_local_dt_keys(dt: dict, keys_to_set: list):
        for node in dt[formats.LocalDtJSonFormat.KEY_DATA]:
            if node["key"] in keys_to_set:
                node["leaf_path"] = True

    #
    # JSon DT mock
    #

    MOCK_CLASSES = ["class_A", "class_B", "class_C"]

    # DT
    JSON_FORMAT_IDX: dict = {
        "files": {
            "class_A": "dt_class_A.json",
            "class_B": "dt_class_B.json",
            "class_C": "dt_class_C.json",
        },
    }

    # DT: feature-?, class-?
    JSON_FORMAT_F_C: dict = {
        "data": [
            {
                "key": "0",
                "name": "LIMIT_BAL",
                "parent": None,
                "edge_in": None,
                "edge_weight": None,
                "leaf_path": False,
            },
            {
                "key": "0.0",
                "name": "LIMIT_BAL",
                "parent": "0",
                "edge_in": "< 144868.000 , NA",
                "edge_weight": 0.517,
                "leaf_path": False,
            },
            {
                "key": "0.0.0",
                "name": "BILL_AMT1",
                "parent": "0.0",
                "edge_in": "< 74931.500 , NA",
                "edge_weight": 0.314,
                "leaf_path": False,
            },
            {
                "key": "0.0.0.0",
                "name": "0.001",
                "parent": "0.0.0",
                "edge_in": "< 97439.500 , NA",
                "edge_weight": 0.313,
                "leaf_path": False,
            },
            {
                "key": "0.0.0.1",
                "name": "0.012",
                "parent": "0.0.0",
                "edge_in": ">= 97439.500",
                "edge_weight": 0.001,
                "leaf_path": False,
            },
            {
                "key": "0.0.1",
                "name": "PAY_4",
                "parent": "0.0",
                "edge_in": ">= 74931.500",
                "edge_weight": 0.203,
                "leaf_path": False,
            },
            {
                "key": "0.0.1.0",
                "name": "0.006",
                "parent": "0.0.1",
                "edge_in": "< -1.500",
                "edge_weight": 0.023,
                "leaf_path": False,
            },
            {
                "key": "0.0.1.1",
                "name": "0.003",
                "parent": "0.0.1",
                "edge_in": ">= -1.500 , NA",
                "edge_weight": 0.181,
                "leaf_path": False,
            },
            {
                "key": "0.1",
                "name": "BILL_AMT4",
                "parent": "0",
                "edge_in": ">= 144868.000",
                "edge_weight": 0.483,
                "leaf_path": False,
            },
            {
                "key": "0.1.0",
                "name": "PAY_AMT1",
                "parent": "0.1",
                "edge_in": "< 19.500",
                "edge_weight": 0.079,
                "leaf_path": False,
            },
            {
                "key": "0.1.0.0",
                "name": "0.007",
                "parent": "0.1.0",
                "edge_in": "< 1972.500 , NA",
                "edge_weight": 0.062,
                "leaf_path": False,
            },
            {
                "key": "0.1.0.1",
                "name": "0.015",
                "parent": "0.1.0",
                "edge_in": ">= 1972.500",
                "edge_weight": 0.018,
                "leaf_path": False,
            },
            {
                "key": "0.1.1",
                "name": "AGE",
                "parent": "0.1",
                "edge_in": ">= 19.500 , NA",
                "edge_weight": 0.404,
                "leaf_path": False,
            },
            {
                "key": "0.1.1.0",
                "name": "0.007",
                "parent": "0.1.1",
                "edge_in": "< 30.500",
                "edge_weight": 0.117,
                "leaf_path": False,
            },
            {
                "key": "0.1.1.1",
                "name": "0.004",
                "parent": "0.1.1",
                "edge_in": ">= 30.500 , NA",
                "edge_weight": 0.287,
                "leaf_path": False,
            },
        ]
    }
