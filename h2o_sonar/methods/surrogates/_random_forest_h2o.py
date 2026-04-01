# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
from abc import ABC

from h2o_sonar.methods.surrogates._surrogate_tree_h2o import H2OTreeBackend
from h2o_sonar.methods.surrogates._surrogate_tree_h2o import TreeSurrogateH2O


class RandomForestH2O(TreeSurrogateH2O, ABC):
    def __init__(self, backend=H2OTreeBackend.RANDOMFOREST, **kwargs):
        """Random Forest methods implementation using H2O algorithms.
        This implementation runs on CPUs and can run in distributed mode.
        """
        super().__init__(backend, **kwargs)

        # H2O specific DT parameters
        h2o_rf_params = {
            "seed": 12345,  # seed = seed,
            "mtries": -2,
            "sample_rate": 1,
            "stopping_tolerance": 0.0001,  # stopping_tolerance = 0.0001,
            "stopping_metric": "RMSE",  # stopping_metric="RMSE",
            "min_rows": 20,  # Default for H2O-3 RF impl in DAI
            "categorical_encoding": "EnumLimited",  # categorical_encoding =
            # "EnumLimited",
            "check_constant_response": False,  # check_constant_response = False,
            "ignore_const_cols": False,  # ignore_const_cols = False,
            "max_categorical_levels": 50,  # max_categorical_levels = 50,
        }
        self.tree_parameters.update(h2o_rf_params.items())
