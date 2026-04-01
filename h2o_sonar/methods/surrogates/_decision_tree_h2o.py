# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
from abc import ABC

from h2o_sonar.lib.api import commons
from h2o_sonar.methods.surrogates._surrogate_tree_h2o import H2OTreeBackend
from h2o_sonar.methods.surrogates._surrogate_tree_h2o import TreeSurrogateH2O


try:
    import h2o  # noqa: F401

    HAS_H2O = True
except ImportError:
    HAS_H2O = False


class DecisionTreeH2O(TreeSurrogateH2O, ABC):
    def __init__(self, backend=H2OTreeBackend.DECISIONTREE, **kwargs):
        """Decision tree methods implementation using H2O algorithms.
        This implementation runs on CPUs and can run in distributed mode.

        """
        super().__init__(backend, **kwargs)

        # H2O specific DT parameters
        h2o_dt_params = {
            "seed": 12345,
            "ntrees": 1,
            "mtries": -2,
            "sample_rate": 1,
            "min_rows": 10,
            "categorical_encoding": "OneHotExplicit",
            "keep_cross_validation_models": True,
            "check_constant_response": False,  # to avoid testing failures in DAI ...
            "ignore_const_cols": True,
            "max_categorical_levels": 50,
        }
        self.tree_parameters.update(h2o_dt_params.items())
        self.tree_parameters.update(kwargs)

    def load_model_details(self):
        if not HAS_H2O:
            commons.raise_opt_import_err("h2o")

        # transform H2O-3 model JSON structure to expected format
        model_json = self.estimator._model_json
        output = model_json.get("output", {})

        # extract metrics from H2O-3 format and transform to expected format
        result = {}

        # add column names and types (needed for patching)
        result["_names"] = output.get("names", [])
        result["_column_types"] = output.get("column_types", [])

        # validation metrics (training metrics if validation not available)
        validation_metrics = output.get("validation_metrics")
        training_metrics = output.get("training_metrics")

        if validation_metrics:
            metrics_json = (
                validation_metrics._metric_json
                if hasattr(validation_metrics, "_metric_json")
                else validation_metrics
            )
            result["_validation_metrics"] = {
                "_MSE": metrics_json.get("MSE", float("nan")),
                "_sigma": metrics_json.get("sigma", 0.0),
            }
        elif training_metrics:
            metrics_json = (
                training_metrics._metric_json
                if hasattr(training_metrics, "_metric_json")
                else training_metrics
            )
            result["_validation_metrics"] = {
                "_MSE": metrics_json.get("MSE", float("nan")),
                "_sigma": metrics_json.get("sigma", 0.0),
            }
        else:
            result["_validation_metrics"] = {
                "_MSE": float("nan"),
                "_sigma": 0.0,
            }

        # cross-validation models and metrics
        cv_models = output.get("cross_validation_models")
        cv_metrics = output.get("cross_validation_metrics")

        result["_cross_validation_models"] = cv_models if cv_models else None

        if cv_metrics:
            cv_metrics_json = (
                cv_metrics._metric_json
                if hasattr(cv_metrics, "_metric_json")
                else cv_metrics
            )
            result["_cross_validation_metrics"] = {
                "_MSE": cv_metrics_json.get("MSE", float("nan")),
                "_sigma": cv_metrics_json.get("sigma", 0.0),
            }
        else:
            result["_cross_validation_metrics"] = None

        return result

    def save_model_details(self, path, filename=None):
        if not HAS_H2O:
            commons.raise_opt_import_err("h2o")

        import json
        import os

        # set default filename if not provided
        if filename is None:
            filename = "dtModel.json"

        # get transformed model details (with _validation_metrics structure)
        # this is what load_model_details() returns
        model_details_transformed = self.load_model_details()

        # write transformed model details to file with desired filename
        output_path = os.path.join(path, filename)
        with open(output_path, "w") as f:
            json.dump(model_details_transformed, f, indent=2)

    def _convert_h2o_objects(self, obj):
        """Recursively convert H2O objects to JSON-serializable format."""
        # import H2O classes for type checking
        try:
            from h2o.two_dim_table import H2OTwoDimTable
        except ImportError:
            H2OTwoDimTable = None

        if isinstance(obj, dict):
            return {key: self._convert_h2o_objects(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_h2o_objects(item) for item in obj]
        elif H2OTwoDimTable and isinstance(obj, H2OTwoDimTable):
            # convert H2OTwoDimTable to dict representation
            # use the internal structure which is JSON-serializable
            return {
                "_table_header": obj._table_header,
                "_table_description": getattr(obj, "_table_description", None),
                "_col_header": obj._col_header,
                "_col_types": getattr(obj, "_col_types", None),
                "_cell_values": obj._cell_values,
            }
        elif hasattr(obj, "__dict__") and hasattr(obj, "__class__"):
            # for other H2O objects, try to get their dict representation
            if hasattr(obj, "_metric_json"):
                return obj._metric_json
            elif hasattr(obj, "as_data_frame"):
                # convert to pandas DataFrame then to dict
                try:
                    df = obj.as_data_frame()
                    return df.to_dict(orient="split")
                except Exception:
                    pass
        return obj

    def load_dt_tree_json(self):
        if not HAS_H2O:
            commons.raise_opt_import_err("h2o")

        from h2o.tree import H2OTree

        # determine appropriate tree_class for the model
        tree_class = self._get_tree_class()

        # get tree structure using H2OTree class
        # decision tree is RandomForest with ntrees=1, so tree_number=0
        tree = H2OTree(model=self.estimator, tree_number=0, tree_class=tree_class)

        # convert H2OTree to JSON structure compatible with expected format
        return self._h2otree_to_json(tree)

    def _get_tree_class(self):
        """Determine the appropriate tree_class parameter for H2OTree.

        For classification models, tree_class should specify which class
        the tree predicts. For regression, it should be None.
        """
        model_json = self.estimator._model_json
        output = model_json.get("output", {})

        # check model category
        model_category = output.get("model_category", "")

        if model_category == "Regression":
            # regression models don't need tree_class
            return None
        elif model_category in ["Binomial", "Multinomial"]:
            # for classification, use class "0" or first domain value
            # H2O uses string class names for tree_class
            domains = output.get("domains")
            if domains and len(domains) > 0:
                # get response column domains (last column)
                response_domains = domains[-1] if isinstance(domains, list) else None
                if response_domains and len(response_domains) > 0:
                    # use first class for tree extraction
                    return response_domains[0]
            # fallback to "0" for binary
            return "0"

        # default: no tree_class
        return None

    def _h2otree_to_json(self, tree):
        """Convert H2OTree to JSON format expected by the explainer."""
        # get tree properties
        node_ids = tree.node_ids
        left_children = tree.left_children
        right_children = tree.right_children
        features = tree.features
        thresholds = tree.thresholds
        predictions = tree.predictions

        # build lookup maps for efficient access
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

        # build recursive tree structure starting from root (node 0)
        root_node = self._build_tree_node(
            node_id=node_ids[0],
            node_to_idx=node_to_idx,
            node_ids=node_ids,
            left_children=left_children,
            right_children=right_children,
            features=features,
            thresholds=thresholds,
            predictions=predictions,
        )

        return root_node

    def _build_tree_node(
        self,
        node_id,
        node_to_idx,
        node_ids,
        left_children,
        right_children,
        features,
        thresholds,
        predictions,
    ):
        """Recursively build tree node in the format expected by the explainer."""
        # validate node exists in tree structure
        if node_id not in node_to_idx:
            # this shouldn't happen with proper tree_class, but handle defensively
            return {
                "name": f"[Missing Node {node_id}]",
                "value": "0",
                "edgeweight": 0.0,
                "totalweight": 0.0,
                "children": None,
            }

        idx = node_to_idx[node_id]

        # check if it's a leaf node
        is_leaf = left_children[idx] == -1

        if is_leaf:
            # leaf node
            node = {
                "name": f"{node_id}",
                "value": str(predictions[idx]) if predictions[idx] is not None else "0",
                "edgeweight": 1.0,
                "totalweight": 1.0,
                "children": None,
            }
        else:
            # split node
            feature = features[idx] if features[idx] is not None else "feature"
            threshold = thresholds[idx] if thresholds[idx] is not None else 0.0

            # build children recursively
            left_child_id = left_children[idx]
            right_child_id = right_children[idx]

            children = []
            if left_child_id != -1 and left_child_id in node_to_idx:
                left_node = self._build_tree_node(
                    node_id=left_child_id,
                    node_to_idx=node_to_idx,
                    node_ids=node_ids,
                    left_children=left_children,
                    right_children=right_children,
                    features=features,
                    thresholds=thresholds,
                    predictions=predictions,
                )
                # only add if not a missing node stub
                if left_node and "Missing Node" not in left_node.get("name", ""):
                    children.append(left_node)

            if right_child_id != -1 and right_child_id in node_to_idx:
                right_node = self._build_tree_node(
                    node_id=right_child_id,
                    node_to_idx=node_to_idx,
                    node_ids=node_ids,
                    left_children=left_children,
                    right_children=right_children,
                    features=features,
                    thresholds=thresholds,
                    predictions=predictions,
                )
                # only add if not a missing node stub
                if right_node and "Missing Node" not in right_node.get("name", ""):
                    children.append(right_node)

            # if no valid children were built, treat as leaf node with prediction
            if not children:
                # don't set "name" so conversion logic will use "value" instead
                node = {
                    "value": (
                        str(predictions[idx]) if predictions[idx] is not None else "0"
                    ),
                    "edgeweight": 1.0,
                    "totalweight": 1.0,
                    "children": None,
                }
            else:
                node = {
                    "name": f"{feature} <= {threshold:.6f}",
                    "edgeweight": 1.0,
                    "totalweight": 1.0,
                    "children": children,
                }

        return node

    def save_dt_tree_json(self, path, filename=None):
        if not HAS_H2O:
            commons.raise_opt_import_err("h2o")

        import json
        import os

        # set default filename if not provided
        if filename is None:
            filename = "dtSurrogate.json"

        # get tree JSON (minimal structure for H2O-3)
        tree_json = self.load_dt_tree_json()

        # write JSON to file
        output_path = os.path.join(path, filename)
        with open(output_path, "w") as f:
            json.dump(tree_json, f, indent=2)

    def save_dt_paths_frame(self, input_df, path):
        if not HAS_H2O:
            commons.raise_opt_import_err("h2o")

        # predict leaf node assignments using H2O-3 API
        dt_paths_frame = self.estimator.predict_leaf_node_assignment(test_data=input_df)
        # combine input and predictions
        combined_frame = input_df.cbind(dt_paths_frame)

        # convert to pandas and save as CSV
        # h2o.export_file has issues with cbind expressions, so use pandas
        pandas_df = combined_frame.as_data_frame()
        pandas_df.to_csv(path, index=False)
