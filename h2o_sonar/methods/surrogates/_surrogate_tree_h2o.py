# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import copy
import inspect
import os
import pickle
from abc import ABC
from enum import Enum

import datatable
import pandas

from h2o_sonar.lib.api import commons
from h2o_sonar.methods.core._mli import InterpretableModel
from h2o_sonar.methods.surrogates._surrogate import SurrogateModel
from h2o_sonar.methods.utils import h2o_utils


try:
    import h2o
    from h2o.estimators import random_forest

    HAS_H2O = True
except ImportError:
    HAS_H2O = False


class H2OTreeBackend(Enum):
    RANDOMFOREST = 1
    DECISIONTREE = 2


class TreeSurrogateH2O(SurrogateModel, ABC):
    """Interface class for all H2O tree based surrogate models"""

    def __init__(self, backend=H2OTreeBackend.RANDOMFOREST, **kwargs):
        """Constructor.

        Parameters
        ----------
        ntrees : int
            Number of trees for a H2O tree model

        nfolds : int
            Number of folds for cross-validation. This value defaults to 0.

        max_depth : int
            The maximum tree depth. Higher values will make the model more complex and
            can lead to overfitting. Setting this value to 0 specifies no limit. This
            value defaults to 20.

        backend : h2o_sonar.surrogate.RandomForestBackend
            Backend used for training. Currently, supports Random Forest and Decision
            Tree

        """
        if not HAS_H2O:
            commons.raise_opt_import_err("h2o")

        SurrogateModel.__init__(self, **kwargs)

        self.tree_parameters = {"ntrees": 100, "nfolds": 0, "max_depth": 20}

        self._set_params(self.tree_parameters.keys(), **kwargs)

        self.backend = backend

        self.set_attributes()

        self.name = None
        self.mli = None

        self.estimator = random_forest.H2ORandomForestEstimator(**kwargs)
        self.estimator._parms = self.tree_parameters  # Set params to default for rf
        #  If kwargs passed in, then append to tree_parameters and _parms (h2o dict of
        #  params for a model)
        if len(kwargs) > 0:
            # Take on input params
            self.tree_parameters.update(kwargs.items())
            # Update params for estimator
            self.estimator._parms = self.tree_parameters

    def set_attributes(self):
        for name, val in self.tree_parameters.items():
            if name not in self._parms:
                setattr(self, name, val)

    def explain(self, model, **kwargs):
        raise NotImplementedError(
            "Explain for H2O tree surrogate models is not yet supported."
        )

    @staticmethod
    def surrogate_type():
        if H2OTreeBackend.RANDOMFOREST:
            return "random_forest"
        if H2OTreeBackend.DECISIONTREE:
            return "decision_tree"
        return "h2o_tree_model"

    @staticmethod
    def instantiate(backend, **kwargs):
        """Create a concrete instance of a H2O Tree model methods."""
        from h2o_sonar.methods.surrogates._decision_tree_h2o import DecisionTreeH2O
        from h2o_sonar.methods.surrogates._random_forest_h2o import RandomForestH2O

        if backend == H2OTreeBackend.RANDOMFOREST:
            return RandomForestH2O(backend, **kwargs)
        if backend == H2OTreeBackend.DECISIONTREE:
            return DecisionTreeH2O(backend, **kwargs)

        raise ValueError(
            f"Backend {backend} currently is not supported."
            f"Please use H2OTreeBackend.RANDOMFOREST or H2OTreeBackend.DECISIONTREE"
        )

    def _fit_impl(self, model, **kwargs):
        """Fit a H2O tree model methods for a concrete model.

        Parameters
        ----------
        model : h2o_sonar.core.InterpretableModel
          Model to be explained.

        Returns
        -------
          str :

        """

        if not TreeSurrogateH2O.can_interpret(model, **kwargs):
            raise ValueError(
                "Passed model object doesn't contain all fields "
                "necessary for H2O tree surrogate model!"
            )

        self.name = model.name
        self.mli = model.mli

        h2o_utils.ensure_h2o3_running(False)
        train_frame = model.data_as_model
        train_frame = TreeSurrogateH2O.remove_new_lines(train_frame)

        X, remove = h2o_utils.to_h2oframe(train_frame)
        # We want to be able to override some parameters per `fit()` run since
        # some parameters cannot be passed to the `train()` method and are sent
        # to the backend via `_parms` dict.
        original_parms = copy.deepcopy(self._parms)

        # Temporarily update `_parms` with arguments passed
        self._parms.update(kwargs)
        # Train doesn't allow "response_column" parameter override...
        y = kwargs.pop("response_column", None)

        if X.nrow <= 20:
            self.estimator._parms["min_rows"] = 1
            self.tree_parameters["min_rows"] = 1

        self._train_and_set_params(X, kwargs, original_parms, remove, y, model)

    @staticmethod
    def remove_new_lines(train_frame):
        # remove new lines from string columns
        pattern = r"\n"
        if isinstance(train_frame, pandas.DataFrame):
            # Remove new lines from string columns
            train_frame = train_frame.replace(pattern, " ", regex=True)
        elif isinstance(train_frame, h2o.H2OFrame):
            # Select columns with string data
            string_cols = [
                col for col, col_type in train_frame.types.items() if col_type == "enum"
            ]
            # Remove new lines from string columns
            if string_cols:
                train_frame[string_cols] = train_frame[string_cols].gsub(pattern, " ")
        elif isinstance(train_frame, datatable.Frame):
            # Select columns with string data
            string_cols = [
                col
                for col in train_frame.names
                if train_frame[:, col].ltypes[0] == datatable.ltype.str
            ]
            # Remove new lines from string columns
            for col in string_cols:
                train_frame[:, col] = datatable.Frame(
                    [x.replace(r"\n", "") for x in train_frame[:, col].to_list()[0]]
                )
        else:
            raise ValueError(
                f"Input dataset should be a H2OFrame, Pandas Dataframe, "
                f"or a datatable frame. Not a {type(train_frame)}"
            )
        return train_frame

    def predict(self, X):
        if not HAS_H2O:
            commons.raise_opt_import_err("h2o")

        if not self.mli:
            raise ValueError("fit() method not called. Run fit() first.")

        X, remove = InterpretableModel.prepare_data(X, self.mli.config)

        original_preds = self.estimator.predict(X)["predict"].as_data_frame(
            use_pandas=False, header=False
        )

        if remove:
            h2o.remove(X)

        return [float(item) for sublist in original_preds for item in sublist]

    def save(self):
        """Save the model under the MLI/model work_dir/name path.

        Returns
        -------
        str :

        """
        if not HAS_H2O:
            commons.raise_opt_import_err("h2o")

        if not self.mli:
            raise ValueError("fit() method not called. Run fit() first.")

        save_path = os.path.join(self.mli.work_dir, self.name, self.surrogate_type())

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(
            os.path.join(save_path, f"{self.surrogate_type()}.pickle"), "wb"
        ) as save_file:
            pickle.dump(self, save_file)

        h2o_tree_model_path = TreeSurrogateH2O._model_path(self.mli.work_dir, self.name)

        h2o.save_model(model=self.estimator, path=h2o_tree_model_path, force=True)

        return save_path

    def load_internal(self):
        if not HAS_H2O:
            commons.raise_opt_import_err("h2o")

        h2o_model_path = TreeSurrogateH2O._model_path(self.mli.work_dir, self.name)
        h2o_model_name = os.listdir(h2o_model_path)[0]

        h2o_model_path = os.path.join(h2o_model_path, h2o_model_name)

        # use H2O-3's load_model
        self.estimator = h2o.load_model(h2o_model_path)

    def save_mojo(self, path):
        return self.estimator.download_mojo(path=path)

    def _train_and_set_params(self, X, kwargs, original_parms, remove, y, model):
        if not HAS_H2O:
            commons.raise_opt_import_err("h2o")

        train_parameters = dict(inspect.signature(self.estimator.train).parameters)
        valid_train_parameters = dict(
            (k, kwargs[k]) for k in train_parameters.keys() if k in kwargs
        )
        if "validation_frame" in valid_train_parameters.keys():
            valid_frame = valid_train_parameters["validation_frame"]
            valid_frame = TreeSurrogateH2O.remove_new_lines(valid_frame)
            valid_train_parameters["validation_frame"] = valid_frame
        self.estimator.train(y=y, training_frame=X, **valid_train_parameters)
        self._parms = original_parms
        if remove:
            h2o.remove(X)
