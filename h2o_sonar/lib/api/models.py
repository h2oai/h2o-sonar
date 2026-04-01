# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import os.path
import pathlib
import pickle
import traceback
import uuid
from collections.abc import Callable
from enum import auto
from enum import Enum
from functools import partial

import datatable
import numpy
import pandas

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import loggers
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.methods.core import method
from h2o_sonar.utils import preprocessing
from h2o_sonar.utils import sanitization


try:
    import h2o

    HAS_PKG_H2O = True
except ImportError:
    HAS_PKG_H2O = False

try:
    import daimojo

    HAS_PKG_DAIMOJO = True
except ImportError:
    HAS_PKG_DAIMOJO = False

try:
    import requests

    HAS_PKG_REQUESTS = True
except ImportError:
    HAS_PKG_REQUESTS = False

# enable the explainable model metadata (can impact H2O ES de/serialization)
# this option can be removed when the H2O ES will be able to handle the metadata
OPT_EXPLAINABLE_MODEL_META = True


class ModelVendor:
    SKLEARN = "sklearn"
    H2O = "h2o"
    DAI = "daimojo"


class ExplainableModelType(Enum):
    """Explainable model type (extensible via inheritance)."""

    mock = auto()
    driverless_ai = auto()
    driverless_ai_rest = auto()
    h2o3 = auto()
    scikit_learn = auto()
    h2ogpte = auto()  # h2oGPTe RAG
    h2ogpte_llm = auto()  # h2oGPTe-hosted LLM
    h2ogpt = auto()  # h2oGPT-hosted LLM
    h2ollmops = auto()  # H2O LLMOps-hosted LLM
    ollama = auto()  # ollama-hosted LLM
    openai_rag = auto()  # OpenAI RAG
    openai_llm = auto()  # OpenAI-hosted LLM
    azure_openai_llm = auto()  # MS Azure hosted OpenAI LLM
    anthropic_llm = auto()  # Anthropic Claude-hosted LLM
    amazon_bedrock_rag = auto()  # Amazon Bedrock
    unknown = auto()

    @staticmethod
    def is_rag(explainable_model_type: "ExplainableModelType") -> bool:
        return explainable_model_type in [
            ExplainableModelType.h2ogpte,
            ExplainableModelType.openai_rag,
            ExplainableModelType.amazon_bedrock_rag,
        ]

    @staticmethod
    def is_llm(explainable_model_type: "ExplainableModelType") -> bool:
        return explainable_model_type in [
            ExplainableModelType.h2ogpte_llm,
            ExplainableModelType.h2ogpt,
            ExplainableModelType.h2ollmops,
            ExplainableModelType.ollama,
            ExplainableModelType.openai_llm,
            ExplainableModelType.azure_openai_llm,
            ExplainableModelType.anthropic_llm,
        ]

    @staticmethod
    def from_connection_type(
        connection_type: h2o_sonar_config.ConnectionConfigType,
    ) -> "ExplainableModelType":
        if connection_type == h2o_sonar_config.ConnectionConfigType.H2O_GPT:
            return ExplainableModelType.h2ogpt
        if connection_type == h2o_sonar_config.ConnectionConfigType.H2O_GPT_E:
            return ExplainableModelType.h2ogpte  # or h2ogpte_llm
        if connection_type == h2o_sonar_config.ConnectionConfigType.H2O_LLM_OPS:
            return ExplainableModelType.h2ollmops
        if connection_type == h2o_sonar_config.ConnectionConfigType.OLLAMA:
            return ExplainableModelType.ollama
        if connection_type == h2o_sonar_config.ConnectionConfigType.OPENAI_RAG:
            return ExplainableModelType.openai_rag
        if connection_type == h2o_sonar_config.ConnectionConfigType.OPENAI_CHAT:
            return ExplainableModelType.openai_llm
        if connection_type == h2o_sonar_config.ConnectionConfigType.AZURE_OPENAI_CHAT:
            return ExplainableModelType.azure_openai_llm
        if connection_type == h2o_sonar_config.ConnectionConfigType.ANTHROPIC_CHAT:
            return ExplainableModelType.anthropic_llm
        if connection_type == h2o_sonar_config.ConnectionConfigType.AMAZON_BEDROCK:
            return ExplainableModelType.amazon_bedrock_rag

        return ExplainableModelType.unknown

    @staticmethod
    def to_connection_type(
        explainable_model_type: "ExplainableModelType",
    ) -> h2o_sonar_config.ConnectionConfigType | None:
        if explainable_model_type == ExplainableModelType.h2ogpt:
            return h2o_sonar_config.ConnectionConfigType.H2O_GPT
        if explainable_model_type == ExplainableModelType.h2ogpte:
            return h2o_sonar_config.ConnectionConfigType.H2O_GPT_E
        if explainable_model_type == ExplainableModelType.h2ogpte_llm:
            return h2o_sonar_config.ConnectionConfigType.H2O_GPT_E
        if explainable_model_type == ExplainableModelType.h2ollmops:
            return h2o_sonar_config.ConnectionConfigType.H2O_LLM_OPS
        if explainable_model_type == ExplainableModelType.ollama:
            return h2o_sonar_config.ConnectionConfigType.OLLAMA
        if explainable_model_type == ExplainableModelType.openai_rag:
            return h2o_sonar_config.ConnectionConfigType.OPENAI_RAG
        if explainable_model_type == ExplainableModelType.openai_llm:
            return h2o_sonar_config.ConnectionConfigType.OPENAI_CHAT
        if explainable_model_type == ExplainableModelType.azure_openai_llm:
            return h2o_sonar_config.ConnectionConfigType.AZURE_OPENAI_CHAT
        if explainable_model_type == ExplainableModelType.anthropic_llm:
            return h2o_sonar_config.ConnectionConfigType.ANTHROPIC_CHAT
        if explainable_model_type == ExplainableModelType.amazon_bedrock_rag:
            return h2o_sonar_config.ConnectionConfigType.AMAZON_BEDROCK

        return None


class ExplainableModelMeta:
    """Explainable ML model metadata - this class provides uniform API to get ML model
    metadata regardless model source, provider and implementation.

    Model labels (``labels`` class field) convention:

    - Regression model: ``labels`` field to be empty list ``[]``.
    - Binomial model: ``labels`` field to be list with two strings or integers
      which represent the model labels; the positive class of interest to
      be the second list item.
    - Multinomial model: ``labels`` field to be list of strings or integers with
      the model classes.

    """

    @property
    def is_constant(self):
        """Is model constant?"""
        return self._is_constant

    @property
    def has_shapley_values(self):
        """Does model provides Shapley values?"""
        return self._has_shapley_values

    @property
    def features_metadata(self) -> method.FeaturesMetadata:
        return self._features_metadata

    @property
    def used_features(self) -> list:
        return self._used_features.copy() if self._used_features else []

    @property
    def transformed_features(self) -> list:
        return self._transformed_features.copy() if self._transformed_features else []

    @property
    def has_text_transformers(self) -> bool:
        """Does model has text transformers?"""
        return bool(self._features_metadata.text_features)

    @property
    def feature_importances(self) -> dict:
        """Return per-feature importance set by the user."""
        return self._feature_importances.copy() if self._feature_importances else {}

    def default_feature_importances(self) -> dict:
        """Construct default (fallback) feature importances - list of features used by
        the model with importances ``0.0`` - to be used if no importances were provided
        by the user.

        """
        return {f: 0.0 for f in self.used_features}

    @property
    def num_labels(self) -> int:
        return 0 if not self.labels else len(self.labels)

    @property
    def positive_label_of_interest(self):
        """In case of binomial classification it returns label of the positive class
        of interest."""
        if self.num_labels == 2:
            return self.labels[1]
        elif len(self.labels) == 1:
            return self.labels[0]
        return None

    def __init__(
        self,
        description: str = "",
        is_constant: bool = False,
        is_remote: bool = False,
        has_shapley_values: bool = False,
        target_col: str = "",
        used_features: list | None = None,
        feature_importances: dict | None = None,
        feature_meta: dict | None = None,
        transformed_features: list | None = None,
        model_path: str = "",
        model_file_size: int = 0,
        sanitization_map: sanitization.SanitizationMap | None = None,
        dataset: datasets.ExplainableDataset | None = None,
    ):
        """Create explainable model metadata.

        Parameters
        ----------
        is_constant : bool
          Constant model indicator.
        has_shapley_values : bool
          Does model provide Shapley values?
        sanitization_map : SanitizationMap | None
          Optional column names / features sanitization map.
        dataset : datasets.ExplainableDataset | None
          Optional dataset which is used e.g. to build sanitization map (if it was
          not provided).
        target_col : str
          Optional target column which is used e.g. to build sanitization map.
        used_features : list[str]
          Optional list of original feature names used by the model.
        used_features : list[str]
          Optional list of transformed feature names created by the model.
        feature_importances: dict[str, float] | None = None,
          Optional feature importances - dictionary mapping feature names to their
          importance (with respect to the model) where importance is float [0.0, 1.0].

            .. code-block:: text

                {
                    "AGE": 1.0,
                    "PAY_1": 0.7,
                    "DATE": 0.3
                }

        feature_meta: dict | None = None,
          Optional dictionary with original features metadata identified by the ML
          model - every feature used by model is marked with its type (numeric,
          categorical or both) and characteristic (date, time, datetime, text,
          image, ID).

            .. code-block:: text

                {
                  "numeric": [
                    "AGE",
                    "PAY_1"
                  ],
                  "categorical": [
                    "AGE",
                    "PAY_1"
                  ],
                  "datetime": [],
                  "date": [
                    "DATE"
                  ],
                  "time_column": [
                    "DATE"
                  ],
                  "text": [],
                  "image": [],
                  "id": [],
                  "all": [
                    "AGE",
                    "DATE",
                    "PAY_1"
                  ]
                }

        """
        # model properties
        self._is_constant = is_constant
        self.is_remote = is_remote
        self._has_shapley_values = has_shapley_values
        self.target_col = target_col or ""
        self.model_path = model_path or ""
        self.model_file_size = model_file_size or 0
        self.description = description or ""
        # original features used by the model
        self._used_features = used_features or []
        # original feature importances assessed by the model
        self._feature_importances = feature_importances or {}
        # transformed features created by the model
        self._transformed_features = transformed_features or []
        # feature metadata assessed by the model
        self._features_metadata = method.FeaturesMetadata(feature_meta)
        # labels
        self.labels = []
        # dataset
        self._dataset = dataset

        # features sanitization map
        if not sanitization_map:
            if dataset is not None and dataset.data is not None:
                sanitization_map = sanitization.SanitizationMap(
                    raw_names=list(dataset.data.names),
                    # TODO do sanitize_names() v
                    sanitized_names=list(dataset.data.names),
                )
            elif used_features:
                sanitization_map = sanitization.SanitizationMap(
                    raw_names=used_features.copy(),
                    # TODO do sanitize_names() v
                    sanitized_names=used_features.copy(),
                )
            else:
                # identity sanitization map: features will be sanitized using internal
                # fallback function, which might cause clashes in case that sanitized
                # feature name is the same for more than one raw feature name
                sanitization_map = sanitization.SanitizationMap(
                    raw_names=["IDENTITY"],
                    sanitized_names=["IDENTITY"],
                )
                # IMPROVE FAIL otherwise: unknown feature names > unable to map
                #     raise ValueError(
                #         "Unable to prepare sanitization map as neither used features "
                #         "nor dataset were provided for given interpretable model"
                #     )

        self.sanitization_map = sanitization_map

    def __str__(self):
        return str(self.to_json(2))

    def get_model_type(self) -> commons.ExperimentType:
        """Get experiment type (regression, binomial and multinomial) for model.

        Returns
        -------
        DaiExperimentType:
          DAI experiment type.

        """
        if not self.labels:
            return commons.ExperimentType.regression
        elif 1 <= len(self.labels) <= 2:
            return commons.ExperimentType.binomial
        else:
            return commons.ExperimentType.multinomial

    def to_dict(self):
        return {
            "target_col": self.target_col,
            "labels": self.labels,
            "num_labels": self.num_labels,
            "used_features": self.used_features,
            "transformed_features": self.transformed_features,
            "importances": self.feature_importances,
            "features_metadata": self._features_metadata.to_dict(),
            "model_path": self.model_path,
        }

    def to_json(self, indent=None):
        return json.dumps(self.to_dict(), indent=indent)


class TransformedFeaturesModel:
    """Transformed features model is associated with ``ExplainableModel`` which
    works on original (raw features).

    ``ExplainableModel`` may have associated transformed features model. In order to
    score a dataset using transformed features model, the dataset must be transformed
    first from the original (dataset and features) to transformed (dataset and features)
    using feature transformers.

    """

    def __init__(
        self,
        model_src,
        transformed_predict_method,
        transform_dataset_method,
        model_meta: ExplainableModelMeta | None = None,
    ):
        """Transformed features model constructor.

        Parameters
        ----------
        model_src : Any
          Model locator - like path to model on the filesystem, instance of a 3rd party
          model, pickle or any other source that can be used to create explainable
          model. Information about the model can be passed to 3rd party model
          implementations (like H2O-3) which can create the model.
        transformed_predict_method :
          Predict method which expect/uses transformed features.
        transform_dataset_method :
          Method which transforms a dataset with original features to the dataset
          with transformed features.
        model_meta : ExplainableModelMeta | None
          Optional transformed model metadata - use ``used_features`` to get names
          of model's transformed features names.

        """
        self.model_src = model_src
        self._predict_method = transformed_predict_method
        self._transform_dataset = transform_dataset_method
        self._meta = model_meta

    @property
    def meta(self) -> ExplainableModelMeta:
        return self._meta

    def predict(
        self,
        transformed_x: datasets.ExplainableDataset | datatable.Frame,
        **kwargs,
    ):
        """Score and return predictions in any format returned by the predict method."""
        if isinstance(transformed_x, datasets.ExplainableDataset):
            return self._predict_method(transformed_x.data, **kwargs)
        return self._predict_method(transformed_x, **kwargs)

    def transform_dataset(
        self, X: datasets.ExplainableDataset | datatable.Frame, **kwargs
    ) -> datasets.ExplainableDataset | datatable.Frame:
        """Transform dataset from original to transformed features."""
        if isinstance(X, datasets.ExplainableDataset):
            return self._transform_dataset(X.data, **kwargs)
        return self._transform_dataset(X, **kwargs)

    def save(self, path: str, update: bool = False):
        """Pickle the model.

        Parameters
        ----------
        path : str
          Model pickle path.
        update : bool
          Delete pickled model if it already exists on given path prior saving
          the new model.

        """
        if path and os.path.isfile(path):
            if update:
                os.remove(path)
            else:
                raise ValueError(
                    f"Unable to pickled model - file already exists on path: {path}"
                )

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        """Load model from pickle.

        Parameters
        ----------
        path : str
          Model pickle path.

        Returns
        -------
        ExplainableModel :
          Instance of the pickled model.

        """
        if path and os.path.isfile(path):
            with open(path, "rb") as f:
                unpickled_model = pickle.load(f)
            return unpickled_model

        raise ValueError(f"Unable to load pickled model - invalid path: {path}")


class ExplainableModel:
    """Explainable model - this class provides uniform API for ML models regardless
    model source, provider or implementation.

    """

    def __init__(
        self,
        model_src,
        predict_method: Callable,
        fit_method=None,
        model_type: ExplainableModelType = ExplainableModelType.unknown,
        model_meta: ExplainableModelMeta | None = None,
        transformed_model: TransformedFeaturesModel | None = None,
        label_encoder: preprocessing.MultiColumnLabelEncoderAbc | None = None,
        logger: loggers.SonarLogger | None = None,
    ):
        """Explainable model constructor. See ``ModelApi`` class for parameters
        description.

        Parameters
        ----------
        model_src : Any
          Locator of the model - like model instance, path to a pickled model or
          URL - to be interpreted.
        predict_method : Callable
          Predict method reference.
        fit_method :
          Optional fit method.
        model_type : ExplainableModelType
          Optional model type - like scikit-learn or Driverless AI MOJO - allowing to
          perform model specific optimizations.
        model_meta : ExplainableModelMeta | None
          Optional model metadata.
        transformed_model : TransformedFeaturesModel | None
          Optional reference to the (internal) model(s) of this explainable
          (original features) model which work(s) on transformed features.
        label_encoder : preprocessing.MultiColumnLabelEncoderAbc | None
          Optional label encoder instance which is used to de/encode categorical
          features - it might be required by libraries which don't support
          categorical/string/object feature types. Label encoder is used as follows
          (considering a library w/o categorical features support):
          - categorical features in the dataset are label encoded using the encoder
          - explainable model constructor (which provides the predict function) is
            initialized with the encoder
          - library works with label encoded (numeric features only) dataset
          - whenever the library calls predict method, explainable model takes care
            of decoding dataset rows to original data types (conversion back from
            numeric to categorical features) because original predict method expect
            original dataset format
          - predict method provides prediction to the library
        logger : loggers.SonarLogger | None
          Optional logger.

        """
        self.model_src = model_src
        self.predict_method = predict_method
        self.fit_method = fit_method
        self.model_type = model_type
        self._meta = model_meta or ExplainableModelMeta()
        self._transformed_model = transformed_model
        self.label_encoder = label_encoder
        self.logger = logger or loggers.SonarPrintLogger()

    def __str__(self):
        return str(self.to_json(2))

    @property
    def meta(self) -> ExplainableModelMeta:
        return self._meta

    @property
    def has_transformed_model(self) -> bool:
        """Does explainable model provides associated model which works on the
        transformed features?

        """
        return self._transformed_model is not None

    @property
    def transformed_model(self) -> TransformedFeaturesModel | None:
        """Get associated model which works on the transformed features."""
        return self._transformed_model

    def fit(
        self,
        X: datasets.ExplainableDataset | datatable.Frame,
        y=None,
        **kwargs,
    ):
        if not self.fit_method:
            raise RuntimeError(
                f"Fit method of explainable model {self} was not defined"
            )
        if isinstance(X, datasets.ExplainableDataset):
            self.fit_method(X.data, y, **kwargs)
        elif self.model_type == ExplainableModelType.h2o3:
            if not HAS_PKG_H2O:
                commons.raise_opt_import_err("h2o")
            return self.fit_method(h2o.H2OFrame(X.to_pandas()), **kwargs)
        self.fit_method(X, y, **kwargs)

    def _le_inverse_transform_pandas(self, x):
        x[self.label_encoder.encoded_columns] = x[
            self.label_encoder.encoded_columns
        ].astype(numpy.int64)
        self.label_encoder.inverse_transform(x)

    def _le_inverse_transform(self, x):
        if self.label_encoder:
            if isinstance(x, pandas.DataFrame):
                return self._le_inverse_transform_pandas(x)
            elif isinstance(x, datatable.Frame):
                # IMPROVE performance: avoid frame conversion there and back
                x_pandas = x.to_pandas()
                self._le_inverse_transform_pandas(x_pandas)
                return datatable.Frame(x_pandas)
            else:
                # TODO other frame types
                raise ValueError(
                    f"Unsupported data frame format for label encoding: '{type(x)}'"
                )

        return x

    def predict(self, X: datasets.ExplainableDataset | datatable.Frame, **kwargs):
        """Score and return predictions in any format returned by the predict method."""
        if isinstance(X, datasets.ExplainableDataset):
            return self.predict_method(
                self._le_inverse_transform(
                    datasets.ExplainableDataset.frame_2_pandas(
                        X.data, columns=self.meta.used_features
                    )
                ),
                **kwargs,
            )
        elif self.model_type == ExplainableModelType.h2o3:
            if not HAS_PKG_H2O:
                commons.raise_opt_import_err("h2o")
            return self.predict_method(
                h2o.H2OFrame(
                    self._le_inverse_transform(
                        datasets.ExplainableDataset.frame_2_pandas(
                            X, columns=self.meta.used_features
                        )
                    )
                ),
                **kwargs,
            )[:, -1]
        elif (
            self.model_type == ExplainableModelType.driverless_ai
            or self.model_type == ExplainableModelType.driverless_ai_rest
        ):
            dt_dataset = self._le_inverse_transform(
                datasets.ExplainableDataset.frame_2_datatable(
                    X, trim_to_columns=self.meta.used_features
                ),
            )
            preds = datasets.ExplainableDataset.frame_2_datatable(
                self.predict_method(dt_dataset, **kwargs)
            )
            # IF binomial & Driverless AI model, THEN return only positive class preds
            return (
                isinstance(preds, datatable.Frame) and preds[:, 1]
                if len(preds.names) == 2
                else preds
            )
        elif self.model_type == ExplainableModelType.scikit_learn:
            return self.model_src.predict_proba(self._le_inverse_transform(X), **kwargs)

        return self.predict_method(self._le_inverse_transform(X), **kwargs)

    def predict_pandas(self, X, **kwargs) -> pandas.DataFrame:
        """Score and return predictions as Pandas frame."""
        return datasets.ExplainableDataset.frame_2_pandas(self.predict(X, **kwargs))

    def predict_datatable(self, X, **kwargs) -> datatable.Frame:
        """Score and return predictions as ``datatable`` frame."""
        return datasets.ExplainableDataset.frame_2_datatable(self.predict(X, **kwargs))

    def shapley_values(self, X, original_features: bool = True, **kwargs):
        """Get Shapley values.

        Parameters
        ----------
        X : datatable.Frame
          Dataset to calculate Shapley values.
        original_features : bool
          ``True`` to get Shapley values for original features, ``False`` to get
          Shapley values for transformed features.

        Returns
        -------
        datatable.Frame :
          Shapley values based feature contributions.

        """
        raise NotImplementedError

    def to_dict(self):
        return {
            "model_type": self.model_type.name,
            "experiment_type": self._meta.get_model_type().name,
            "metadata": self._meta.to_dict(),
        }

    def to_json(self, indent=None):
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str, update: bool = False):
        """Pickle the model.

        Parameters
        ----------
        path : str
          Model pickle path.
        update : bool
          Delete pickled model if it already exists on given path prior saving
          the new model.

        """
        if path and os.path.isfile(path):
            if update:
                os.remove(path)
            else:
                raise ValueError(
                    f"Unable to pickled model - file already exists on path: {path}"
                )

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        """Load model from pickle.

        Parameters
        ----------
        path : str
          Model pickle path.

        Returns
        -------
        ExplainableModel :
          Instance of the pickled model.

        """
        if path and os.path.isfile(path):
            with open(path, "rb") as f:
                unpickled_model = pickle.load(f)
            return unpickled_model

        raise ValueError(f"Unable to load pickled model - invalid path: {path}")


def guess_model_used_features(
    dataset: datasets.ExplainableDataset | pandas.DataFrame | datatable.Frame,
    target_col: str = "",
    model_type_str: str = "scikit-learn",
) -> list[str]:
    """Guess features used by the model from the dataset.

    Parameters
    ----------
    dataset : datasets.ExplainableDataset | pandas.DataFrame | datatable.Frame
        Dataset used to train the model.
    target_col : str
        Target column name. If specified, the target column will be removed from the
        used features, otherwise it will be included.
    model_type_str : str
        Model type string to be used in exception messages.

    Returns
    -------
    list[str] :
        List of features used by the model.

    """
    used_features = []

    if dataset is not None:
        if isinstance(dataset, datasets.ExplainableDataset):
            if dataset.meta and dataset.meta.column_names:
                used_features = dataset.meta.column_names.copy()
        elif isinstance(dataset, pandas.DataFrame):
            used_features = dataset.columns.tolist()
        elif isinstance(dataset, datatable.Frame):
            used_features = list(dataset.names)
        else:
            raise ValueError(
                f"Features used by the model required for {model_type_str} models "
                f"interpretation - please set model.meta.used_features (or "
                f"provide a dataset with column names): unsupported dataset "
                f"type: {type(dataset)}"
            )

    if target_col and target_col in used_features:
        used_features.remove(target_col)

    return used_features


def guess_model_labels(
    dataset: datasets.ExplainableDataset | pandas.DataFrame | datatable.Frame,
    target_col,
    labels: list | None = None,
    model_type_str="scikit-learn",
    logger: loggers.SonarLogger | None = None,
) -> list[str] | None:
    """Guess features used by the model from the dataset.

    Parameters
    ----------
    dataset : datasets.ExplainableDataset | pandas.DataFrame | datatable.Frame
        Dataset used to train the model.
    target_col : str
        Target column name.
    labels : list[str] | None
        List of model labels value to return if not possible to determine them.
    logger : loggers.SonarLogger | None
        Logger instance.
    model_type_str : str
        Model type string to be used in exception messages.

    Returns
    -------
    list[str] | None :
        List of model labels. If ``None``, then it's not possible to determine it.

    """
    if dataset is not None and target_col:
        dataset_data = None
        if isinstance(dataset, datasets.ExplainableDataset):
            dataset_data = dataset.data[:, target_col]
        elif isinstance(dataset, pandas.DataFrame):
            dataset_data = datatable.Frame(dataset.loc[:, target_col])
        elif isinstance(dataset, datatable.Frame):
            dataset_data = dataset[:, target_col]
        else:
            if logger:
                logger.warning(
                    f"WARNING: unable to guess {model_type_str} model labels "
                    f"from the dataset: unsupported dataset type:"
                    f" {type(dataset)} and/or missing target column "
                    f"'{target_col}'"
                )
        if dataset_data is not None:
            labels = datatable.unique(dataset_data[target_col]).to_list()[0]

    return labels


class PickleFileModel(ExplainableModel):
    """Pickled explainable model."""

    EXT_PICKLE = ".pkl"

    @staticmethod
    def is_pickle_file_model(model_src) -> bool:
        return isinstance(model_src, (str, pathlib.Path)) and os.path.isfile(
            str(model_src)
        )

    @staticmethod
    def from_pickle(
        model_src,
        target_col: str = "",
        used_features: list[str] | None = None,
        sanitization_map: sanitization.SanitizationMap | None = None,
        dataset: datasets.ExplainableDataset | None = None,
    ) -> ExplainableModel:
        unpickled_model = ExplainableModel.load(str(model_src))
        if isinstance(unpickled_model, ExplainableModel):
            return unpickled_model
        elif ScikitLearnModel.is_scikit_learn_model(unpickled_model):
            return ScikitLearnModel(
                model_src=unpickled_model,
                target_col=target_col,
                used_features=used_features,
                sanitization_map=sanitization_map,
                dataset=dataset,
            )

        return ExplainableModel(
            predict_method=unpickled_model.predict,
            model_src=model_src,
        )


class DriverlessAiModel(ExplainableModel):
    """Explainable model which understands Driverless AI experiments and models thus
    it can get model metadata, ensure required sanitization and correctly construct
    predict method which accepts expected input and provides desired output.

    """

    ATTR_LABEL_NAMES = "output_names"
    ATTR_HAS_SHAPLEYS = "has_treeshap"

    EXT_MOJO = ".mojo"

    PREFIX_CLASS = "class."
    PREFIX_SHAPLEY_COLS = "contrib_"
    COL_SHAPLEY_BIAS = f"{PREFIX_SHAPLEY_COLS}{datasets.ExplainableDataset.COL_BIAS}"

    @staticmethod
    def is_dai_model(model_src) -> bool:
        return (
            model_src
            and isinstance(model_src, str)
            and model_src.endswith(DriverlessAiModel.EXT_MOJO)
        ) or (ModelVendor.DAI == commons.base_pkg(model_src)[0])

    @staticmethod
    def __create_dai_mojo(model_src):
        if not HAS_PKG_DAIMOJO:
            commons.raise_opt_import_err("daimojo")
        return daimojo.model(model_src)

    def __init__(
        self,
        model_src,
        target_col: str = "",
        used_features: list[str] | None = None,
        sanitization_map: sanitization.SanitizationMap | None = None,
        dataset: datasets.ExplainableDataset | None = None,
        logger=None,
    ):
        if not DriverlessAiModel.is_dai_model(model_src):
            raise ValueError(f"Model: '{model_src}' is not a Driverless AI model")

        if isinstance(model_src, str) and model_src.endswith(
            DriverlessAiModel.EXT_MOJO
        ):
            model_src = self.__create_dai_mojo(model_src)

        if not used_features:
            # Driverless AI Python pipeline introspection - daimojo.model.model:
            #   - features: features_names, feature_types
            #   - labels: output_names, output_types
            #   - Shapley values: has_tree_shap()
            # ... also transformed_names, mojo_version, dai_version,
            if DriverlessAiModel.is_dai_model(model_src):
                used_features = model_src.feature_names
            elif not used_features and dataset is not None:
                used_features = guess_model_used_features(
                    dataset=dataset,
                    target_col=target_col,
                    model_type_str="Driverless AI MOJO",
                )

        ExplainableModel.__init__(
            self,
            predict_method=model_src.predict,
            fit_method=None,
            model_type=ExplainableModelType.driverless_ai,
            model_meta=ExplainableModelMeta(
                target_col=target_col,
                sanitization_map=sanitization_map,
                used_features=used_features,
                dataset=dataset,
                has_shapley_values=hasattr(
                    model_src, DriverlessAiModel.ATTR_HAS_SHAPLEYS
                ),
                transformed_features=(
                    model_src.transformed_names
                    if hasattr(model_src, "transformed_names")
                    else None
                ),
            ),
            model_src=model_src,
            logger=logger,
        )

        if not self.meta.labels:
            if hasattr(model_src, DriverlessAiModel.ATTR_LABEL_NAMES):
                if not HAS_PKG_DAIMOJO:
                    commons.raise_opt_import_err("daimojo")
                do_fallback = False
                daimojo_version = commons.SemVer.from_str(daimojo.__version__)
                if not (
                    daimojo_version
                    and daimojo_version.major == 2
                    and daimojo_version.minor == 7
                    and daimojo_version.patch == 11
                ):
                    # output_names/_types initialized AFTER predict() in newer versions
                    try:
                        one_row = None
                        if dataset is not None:
                            if isinstance(dataset, datasets.ExplainableDataset):
                                one_row = dataset.data[1, :]
                            elif isinstance(dataset, pandas.DataFrame):
                                one_row = dataset.iloc[0]
                            elif isinstance(dataset, datatable.Frame):
                                one_row = dataset[1, :]
                        if one_row is not None:
                            model_src.predict(one_row)
                        else:
                            do_fallback = True
                    except Exception as ex:
                        do_fallback = True
                        if logger:
                            logger.warning(
                                f"Unable to initialize Driverless AI MOJO to "
                                f"determine model labels:{ex}"
                                f"\n{traceback.format_exc()}"
                            )
                if do_fallback:
                    if dataset is not None and target_col:
                        self.meta.labels = guess_model_labels(
                            dataset=dataset,
                            target_col=target_col,
                            labels=self.meta.labels,
                            model_type_str="Driverless AI MOJO",
                            logger=logger,
                        )
                    else:
                        if logger:
                            logger.warning(
                                "Unable to determine model labels neither from "
                                "Driverless AI MOJO nor from the dataset"
                            )
                else:  # DEFAULT branch to determine labels from MOJO
                    self.meta.labels = (
                        []
                        if len(model_src.output_names) == 1
                        else model_src.output_names.copy()
                    )
            else:
                self.meta.labels = guess_model_labels(
                    dataset=dataset,
                    target_col=target_col,
                    labels=self.meta.labels,
                    model_type_str="Driverless AI MOJO",
                    logger=logger,
                )

        # sanitize labels by removing prefixes added by Driverless AI
        sanitized_target_col = sanitization.sanitize_strings(target_col)
        if target_col:
            if all(
                label and label.startswith(f"{target_col}.")
                for label in self.meta.labels
            ):
                prefix_to_strip: str = f"{target_col}."
            elif all(
                label and label.startswith(f"{sanitized_target_col}.")
                for label in self.meta.labels
            ):
                prefix_to_strip: str = f"{sanitized_target_col}."
            else:
                prefix_to_strip = ""
        elif all(
            label and label.startswith(DriverlessAiModel.PREFIX_CLASS)
            for label in self.meta.labels
        ):
            prefix_to_strip = f"{DriverlessAiModel.PREFIX_CLASS}."
        else:
            prefix_to_strip = ""
        if prefix_to_strip:
            self.meta.labels = [
                label[len(prefix_to_strip) :] for label in self.meta.labels.copy()
            ]

    def shapley_values(
        self, X, original_features: bool = True, fast_approx: bool = False, **kwargs
    ):
        """Get Shapley values.

        Parameters
        ----------
        X : datatable.Frame
          Dataset to calculate Shapley values.
        original_features : bool
          ``True`` to get Shapley values for original features, ``False`` to get
          Shapley values for transformed features.
        fast_approx : bool
          ``True`` to use fast approximation for Shapley values calculation.

        Returns
        -------
        datatable.Frame :
          Shapley values based feature contributions.

        """
        if fast_approx:
            # IMPROVE: detect whether the scorer supports fast approximation
            raise ValueError(
                f"Fast approximation is not supported by the Driverless AI MOJO "
                f"scorer: {self.predict_method} ({type(self.predict_method)})"
            )

        return self.predict_method(
            datasets.ExplainableDataset.frame_2_datatable(X),
            pred_contribs=True,
            pred_contribs_original=original_features,
            **kwargs,
        )


DAI_REST_PATH_SAMPLE = "/sample_request"
DAI_REST_PATH_MODEL = "/model"
DAI_REST_PATH_SCORE = "/score"


def _dai_rest_server_predict_method(model_server_url: str, x: datatable.Frame):
    if not HAS_PKG_REQUESTS:
        commons.raise_opt_import_err("requests")

    request_body = {
        "fields": list(x.names),
        "rows": x.to_pandas().values.tolist(),
    }
    response = requests.post(
        url=f"{model_server_url}/score",
        json=request_body,
        verify=h2o_sonar_config.config.http_ssl_cert_verify,
    )
    if response:
        response_json = json.loads(response.text)
        if response_json.get("fields") and response_json.get("score"):
            frame_dict = {}
            for c in response_json.get("fields"):
                frame_dict[c] = []
            score = response_json.get("score", [])
            for i in range(len(score)):
                for j, c in enumerate(frame_dict.keys()):
                    frame_dict[c].append(float(response_json["score"][i][j]))
            return datatable.Frame(frame_dict)

    raise RuntimeError(
        f"Unable to parse response from Driverless AI REST model server: "
        f"'{response}' as it's empty or incomplete"
    )


class DriverlessAiRestServerModel(ExplainableModel):
    """Explainable model which represents Driverless AI experiments deployed as
    REST server. Driverless AI is moving from local REST Server to MLOps, therefore
    it is deprecated in 1.10.4 and will be removed. Anyway it is useful for existing
    Driverless AI deployments.

    See also:

    - https://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/deployment.html
    - https://h2oai.github.io/dai-deployment-templates/local-rest-scorer/

    """

    @staticmethod
    def is_dai_rest_server_model(model_src) -> bool:
        return (
            model_src
            and isinstance(model_src, str)
            and model_src.startswith("http")
            and model_src.endswith("model")
        )

    @staticmethod
    def _server_get(rest_server_url: str) -> str:
        if not HAS_PKG_REQUESTS:
            commons.raise_opt_import_err("requests")
        response = requests.get(
            f"{rest_server_url}{DAI_REST_PATH_SAMPLE}",
            verify=h2o_sonar_config.config.http_ssl_cert_verify,
        )
        return response.text

    def __init__(
        self,
        model_server_url: str,
        target_col: str = "",
        used_features: list[str] | None = None,
        sanitization_map: sanitization.SanitizationMap | None = None,
        dataset: datasets.ExplainableDataset | None = None,
    ):
        if not DriverlessAiRestServerModel.is_dai_rest_server_model(model_server_url):
            raise ValueError(
                f"Model: '{model_server_url}' is not valid Driverless AI REST server "
                f"model URL"
            )

        if not used_features:
            # Driverless AI REST server has sample request which can be used to
            # determine features used by the model
            try:
                sample_request_str = DriverlessAiRestServerModel._server_get(
                    rest_server_url=model_server_url,
                )
                sample_request = json.loads(sample_request_str)
                if sample_request:
                    fields = sample_request.get("fields", None)
                    if fields and isinstance(fields, list):
                        used_features = fields.copy()
            except Exception as ex:
                print(
                    f"WARNING: unable to determine features used by Driverless AI "
                    f"REST server model '{model_server_url}': {ex}"
                )

        model_meta = ExplainableModelMeta(
            target_col=target_col,
            sanitization_map=sanitization_map,
            used_features=used_features,
            dataset=dataset,
        )
        # REST model has no introspection - guess labels
        try:
            if dataset is not None and dataset.data is not None:
                model_meta.labels = datatable.unique(
                    dataset.data[target_col]
                ).to_list()[0]
        except Exception as ex:
            print(f"WARNING: unable to guess Driverless AI REST model labels: {ex}")
        ExplainableModel.__init__(
            self,
            predict_method=partial(_dai_rest_server_predict_method, model_server_url),
            fit_method=None,
            model_type=ExplainableModelType.driverless_ai_rest,
            model_meta=model_meta,
            model_src=model_server_url,
        )

        # IMPROVE introspection of labels in case bi/multinomial
        #  self.meta.labels = model_src.output_names.copy()

        # sanitize labels by removing class. prefix
        if all(label and label.startswith("class.") for label in self.meta.labels):
            self.meta.labels = [
                label[len("class.") :] for label in self.meta.labels.copy()
            ]


class H2o3Model(ExplainableModel):
    """H2O-3 explainable model implementation."""

    @staticmethod
    def is_h2o3_model(model_src) -> bool:
        return ModelVendor.H2O == commons.base_pkg(model_src)[0]

    def __init__(
        self,
        model_src,
        target_col: str = "",
        used_features: list[str] | None = None,
        sanitization_map: sanitization.SanitizationMap | None = None,
        dataset: datasets.ExplainableDataset | None = None,
    ):
        if not H2o3Model.is_h2o3_model(model_src):
            raise ValueError(f"Model: '{model_src}' is not h2o3 model")

        if not used_features:
            if callable(model_src.varimp):
                # H2O-3 pipeline introspection:
                used_features = [fi[0] for fi in model_src.varimp()]
            else:
                raise ValueError(
                    "Features used by the model are required for H2O-3 models "
                    "interpretation - please set model.meta.used_features "
                    "as it wasn't possible to determine used features automatically"
                )

        ExplainableModel.__init__(
            self,
            predict_method=model_src.predict,
            fit_method=model_src.fit,
            model_type=ExplainableModelType.h2o3,
            model_meta=ExplainableModelMeta(
                target_col=target_col,
                used_features=used_features,
                sanitization_map=sanitization_map,
                dataset=dataset,
            ),
            model_src=model_src,
        )


class ScikitLearnModel(ExplainableModel):
    """Scikit-learn explainable model implementation."""

    @staticmethod
    def is_scikit_learn_model(model_src) -> bool:
        return ModelVendor.SKLEARN == commons.base_pkg(model_src)[0]

    def __init__(
        self,
        model_src,
        target_col: str = "",
        used_features: list[str] | None = None,
        labels: list | None = None,
        sanitization_map: sanitization.SanitizationMap | None = None,
        dataset: (
            datasets.ExplainableDataset | datatable.Frame | pandas.DataFrame | None
        ) = None,
        logger=None,
    ):
        if not ScikitLearnModel.is_scikit_learn_model(model_src):
            raise ValueError(f"Model: '{model_src}' is not scikit-learn model")

        # INTROSPECTION: certain scikit-learn models provide feature names as attribute
        if not used_features:
            if model_src and hasattr(model_src, "feature_names_in_"):
                used_features = model_src.feature_names_in_
                used_features = model_src.feature_names = (
                    used_features.tolist()
                    if isinstance(used_features, numpy.ndarray)
                    else used_features
                )
            elif dataset is not None:
                used_features = guess_model_used_features(
                    dataset=dataset,
                    target_col=target_col,
                    model_type_str="scikit-learn",
                )

        if not used_features:
            raise ValueError(
                "Features used by the model required for scikit-learn models "
                "interpretation - please set model.meta.used_features"
            )

        # LABELS: certain scikit-learn models provide labels as attribute
        try:
            if not labels:
                if model_src and hasattr(model_src, "classes_"):
                    labels = model_src.classes_
                    labels = model_src.labels = (
                        labels.tolist() if isinstance(labels, numpy.ndarray) else labels
                    )
                elif dataset is not None and target_col:
                    labels = guess_model_labels(
                        dataset=dataset,
                        target_col=target_col,
                        labels=labels,
                        model_type_str="scikit-learn",
                        logger=logger,
                    )
        except Exception as ex:
            if logger:
                logger.warning(
                    f"WARNING: unable to guess scikit-learn model labels: {ex}"
                )

        model_meta = ExplainableModelMeta(
            target_col=target_col,
            used_features=used_features,
            sanitization_map=sanitization_map,
            dataset=(
                dataset if isinstance(dataset, datasets.ExplainableDataset) else None
            ),
        )
        model_meta.labels = labels

        ExplainableModel.__init__(
            self,
            predict_method=model_src.predict,
            fit_method=model_src.fit,
            model_type=ExplainableModelType.scikit_learn,
            model_meta=model_meta,
            model_src=model_src,
        )

    def predict(self, X: datasets.ExplainableDataset | datatable.Frame, **kwargs):
        """Score and return predictions in any format returned by the predict method.
        Scikit-learn models require specific constraint which are enforced by this
        model specific method.

        """
        data = X.data if isinstance(X, datasets.ExplainableDataset) else X
        data_target_col = None
        if isinstance(data, datatable.Frame):
            if self.meta.target_col in data.names:
                data_target_col = data[:, self.meta.target_col]
                del data[self.meta.target_col]
        elif isinstance(data, pandas.DataFrame):
            if self.meta.target_col in data:
                # IMPROVE: drop/add Pandas WITHOUT conversion to dt for (performance)
                data = datatable.Frame(data)
                del data[self.meta.target_col]
        # IMPROVE else:
        else:
            raise ValueError(
                f"Unsupported frame type to fix and score - target column must not "
                f"be present in the dataset to score: '{type(data)}'"
            )

        try:
            preds = self.predict_method(data, **kwargs)
        except ValueError as ex:
            dataset_cols = list(data.names)
            dataset_cols.sort()
            model_cols = list(self.model_src.feature_names)
            model_cols.sort()
            raise ValueError(
                f"Predict method failed with: {ex}\n"
                f"> target column    : {self.meta.target_col}\n"
                f"> data columns  ({len(dataset_cols)}): {dataset_cols}\n"
                f"> model features({len(model_cols)}) : {model_cols}\n"
                f"{traceback.format_exc()}"
            )
        finally:
            if data_target_col:
                data.cbind(data_target_col)

        return preds


class ExplainableLlmModel:
    KEY_CONNECTION = "connection"
    KEY_MODEL_TYPE = "model_type"
    KEY_NAME = "name"
    KEY_LLM_MODEL_NAME = "llm_model_name"
    KEY_LLM_MODEL_META = "llm_model_meta"
    KEY_MODEL_CFG = "model_cfg"
    KEY_KEY = "key"

    # meta keys
    KEY_H2OGPTE_STATS = "h2ogpte_perf_stats"
    KEY_H2OGPTE_VISION_M = "vision_model_name"

    KEY_STATS_SUCCESS = "success_count"
    KEY_STATS_RETRY = "retry_count"
    KEY_STATS_TIMEOUT = "timeout_count"
    KEY_STATS_FAILURE = "failure_count"
    KEY_STATS_DURATION = "duration_stats"

    def __init__(
        self,
        connection: str | h2o_sonar_config.ConnectionConfig,
        model_type: ExplainableModelType = ExplainableModelType.unknown,
        name: str = "",
        llm_model_name: str = "",
        llm_model_meta: dict | None = None,
        model_cfg: dict | None = None,
        key: str = "",
        logger: loggers.SonarLogger | None = None,
    ):
        """Constructor.

        Parameters
        ----------
        llm_model_name : str
            Name of the LLM model to be evaluated.

        """
        self.connection = connection
        self.model_type = model_type
        self.name = name or (
            f"LLM model - LLM: {llm_model_name}"
            f"{', config: ' + str(id(model_cfg)) if model_cfg else ''}"
        )
        self.llm_model_name = llm_model_name
        self.llm_model_meta = llm_model_meta or {}
        self.model_cfg = model_cfg or {}
        self.key = key or str(uuid.uuid4())
        self.logger = logger or loggers.SonarPrintLogger()

    def __str__(self):
        return str(self.to_json(2))

    def clone(self):
        return ExplainableLlmModel(
            connection=self.connection,
            model_type=self.model_type,
            name=self.name,
            llm_model_name=self.llm_model_name,
            llm_model_meta=self.llm_model_meta.copy() if self.llm_model_meta else {},
            model_cfg=self.model_cfg.copy() if self.model_cfg else {},
            key=self.key,
            logger=self.logger,
        )

    def to_dict(self):
        as_dict = {
            ExplainableLlmModel.KEY_CONNECTION: (
                self.connection.key if self.connection else None
            ),
            ExplainableLlmModel.KEY_MODEL_TYPE: self.model_type.name,
            ExplainableLlmModel.KEY_NAME: self.name,
            ExplainableLlmModel.KEY_LLM_MODEL_NAME: self.llm_model_name,
            ExplainableLlmModel.KEY_MODEL_CFG: self.model_cfg,
            ExplainableLlmModel.KEY_KEY: self.key,
        }

        if OPT_EXPLAINABLE_MODEL_META:
            as_dict[ExplainableLlmModel.KEY_LLM_MODEL_META] = self.llm_model_meta

        return as_dict

    @staticmethod
    def from_dict(as_dict: dict, connection=None) -> "ExplainableLlmModel":
        return ExplainableLlmModel(
            connection=connection
            or as_dict.get(ExplainableLlmModel.KEY_CONNECTION, ""),
            model_type=ExplainableModelType[
                as_dict.get(
                    ExplainableLlmModel.KEY_MODEL_TYPE,
                    ExplainableModelType.unknown.name,
                )
            ],
            name=as_dict.get(ExplainableLlmModel.KEY_NAME, ""),
            llm_model_name=as_dict.get(ExplainableLlmModel.KEY_LLM_MODEL_NAME, ""),
            llm_model_meta=as_dict.get(ExplainableLlmModel.KEY_LLM_MODEL_META, {}),
            model_cfg=as_dict.get(ExplainableLlmModel.KEY_MODEL_CFG, {}),
            key=as_dict.get(ExplainableLlmModel.KEY_KEY, ""),
        )

    def to_json(self, indent=None):
        return json.dumps(self.to_dict(), indent=indent)


class ExplainableRagModel:
    KEY_CONNECTION = ExplainableLlmModel.KEY_CONNECTION
    KEY_MODEL_TYPE = ExplainableLlmModel.KEY_MODEL_TYPE
    KEY_NAME = ExplainableLlmModel.KEY_NAME
    KEY_COLLECTION_ID = "collection_id"
    KEY_COLLECTION_NAME = "collection_name"
    KEY_LLM_MODEL_NAME = ExplainableLlmModel.KEY_LLM_MODEL_NAME
    KEY_LLM_MODEL_META = ExplainableLlmModel.KEY_LLM_MODEL_META
    KEY_DOCUMENTS = "documents"
    KEY_MODEL_CFG = "model_cfg"
    KEY_KEY = ExplainableLlmModel.KEY_KEY

    def __init__(
        self,
        connection: str | h2o_sonar_config.ConnectionConfig,
        model_type: ExplainableModelType = ExplainableModelType.unknown,
        name: str = "",
        collection_id: str = "",
        collection_name: str = "",
        llm_model_name: str = "",
        llm_model_meta: dict | None = None,
        documents: list[str] | None = None,
        model_cfg: dict | None = None,
        key: str = "",
        logger: loggers.SonarLogger | None = None,
    ):
        """Constructor.

        Parameters
        ----------
        documents : list[list[str]]
            For RAG collection to be used/created, the list of document paths
            (on the local filesystem) it must contain.
        llm_model_name : str
            Name of the LLM model to be used by RAG to answer questions augmentation.

        """
        from h2o_sonar.lib.integrations import genai

        self.connection = connection
        self.model_type = model_type
        self.document_names = (
            [os.path.basename(doc) for doc in documents] if documents else []
        )
        self.name = name or (
            f"RAG model - LLM: {llm_model_name}, corpus: {self.document_names}"
            f"{', config: ' + str(id(model_cfg)) if model_cfg else ''}"
        )
        # do NOT use the name in the name of collection > it is REUSED by models
        collection_name = collection_name or (
            genai.RagClient.get_collection_name(self.document_names)
        )
        self.collection_id = collection_id
        self.collection_name = collection_name
        self.llm_model_name = llm_model_name
        self.llm_model_meta = llm_model_meta or {}
        self.documents = documents
        self.model_cfg = model_cfg or {}
        self.key = key or str(uuid.uuid4())
        self.logger = logger or loggers.SonarPrintLogger()

    def clone(self):
        return ExplainableRagModel(
            connection=self.connection,
            model_type=self.model_type,
            name=self.name,
            collection_id=self.collection_id,
            collection_name=self.collection_name,
            llm_model_name=self.llm_model_name,
            llm_model_meta=self.llm_model_meta.copy() if self.llm_model_meta else {},
            documents=self.documents.copy() if self.documents else [],
            model_cfg=self.model_cfg.copy() if self.model_cfg else {},
            key=self.key,
            logger=self.logger,
        )

    def to_dict(self):
        as_dict = {
            ExplainableRagModel.KEY_CONNECTION: (
                self.connection
                if isinstance(self.connection, str)
                else (self.connection.key if self.connection else None)
            ),
            ExplainableRagModel.KEY_MODEL_TYPE: self.model_type.name,
            ExplainableRagModel.KEY_NAME: self.name,
            ExplainableRagModel.KEY_COLLECTION_ID: self.collection_id,
            ExplainableRagModel.KEY_COLLECTION_NAME: self.collection_name,
            ExplainableRagModel.KEY_LLM_MODEL_NAME: self.llm_model_name,
            ExplainableRagModel.KEY_DOCUMENTS: self.documents,
            ExplainableRagModel.KEY_MODEL_CFG: self.model_cfg,
            ExplainableRagModel.KEY_KEY: self.key,
        }

        if OPT_EXPLAINABLE_MODEL_META:
            as_dict[ExplainableRagModel.KEY_LLM_MODEL_META] = self.llm_model_meta

        return as_dict

    @staticmethod
    def from_dict(as_dict: dict, connection=None) -> "ExplainableRagModel":
        return ExplainableRagModel(
            connection=connection
            or as_dict.get(ExplainableRagModel.KEY_CONNECTION, ""),
            model_type=ExplainableModelType[
                as_dict.get(
                    ExplainableRagModel.KEY_MODEL_TYPE,
                    ExplainableModelType.unknown.name,
                )
            ],
            name=as_dict.get(ExplainableRagModel.KEY_NAME, ""),
            collection_id=as_dict.get(ExplainableRagModel.KEY_COLLECTION_ID, ""),
            collection_name=as_dict.get(ExplainableRagModel.KEY_COLLECTION_NAME, ""),
            llm_model_name=as_dict.get(ExplainableRagModel.KEY_LLM_MODEL_NAME, ""),
            llm_model_meta=as_dict.get(ExplainableRagModel.KEY_LLM_MODEL_META, {}),
            documents=as_dict.get(ExplainableRagModel.KEY_DOCUMENTS, []),
            model_cfg=as_dict.get(ExplainableRagModel.KEY_MODEL_CFG, {}),
            key=as_dict.get(ExplainableRagModel.KEY_KEY, ""),
        )


def explainable_rag_llm_model_from_json(
    json_dict: dict,
) -> ExplainableRagModel | ExplainableLlmModel:
    """Create LLM or RAG model from the JSon dictionary.

    Parameters
    ----------
    json_dict : dict
        JSon dictionary containing LLM or RAG model definition which can be found
        in ``interpretation.json::models`` section.

    Returns
    -------
    ExplainableRagModel | ExplainableLlmModel :
        Instance of the LLM or RAG model.

    """
    if not json_dict or not isinstance(json_dict, dict):
        raise ValueError(
            "Unable to load LLM / RAG model - empty/invalid JSon dictionary provided"
        )
    if ExplainableLlmModel.KEY_MODEL_TYPE not in json_dict:
        raise ValueError(
            "Unable to load LLM / RAG model - missing model type in the JSon dictionary"
        )

    model_type_str = json_dict[ExplainableRagModel.KEY_MODEL_TYPE]
    try:
        model_type_e = ExplainableModelType[model_type_str.lower()]
    except KeyError:
        raise ValueError(
            f"Unable to load LLM / RAG model - unsupported LLM or RAG model type: "
            f"'{model_type_str}'"
        )

    if ExplainableModelType.is_rag(model_type_e):
        return ExplainableRagModel.from_dict(json_dict)
    elif ExplainableModelType.is_llm(model_type_e):
        return ExplainableLlmModel.from_dict(json_dict)

    raise ValueError(
        f"Unable to load LLM / RAG model - not LLM or RAG model type: "
        f"'{model_type_str}'"
    )


class OpenAiRagModel(ExplainableRagModel):
    """OpenAI RAG model - AI Assistant with File Search/Retrieval tool enabled."""

    def __init__(
        self,
        connection: str | h2o_sonar_config.ConnectionConfig,
        name: str = "",
        thread_id: str = "",
        llm_model_name: str = "",
        documents: list[str] | None = None,
        key: str = "",
        logger: loggers.SonarLogger | None = None,
    ):
        ExplainableRagModel.__init__(
            self,
            connection=connection,
            model_type=ExplainableModelType.openai_rag,
            name=name,
            collection_id=thread_id,
            llm_model_name=llm_model_name,
            documents=documents,
            key=key,
            logger=logger,
        )


class ExplainableModelHandle(commons.ResourceHandle):
    """Handle to a REMOTE model hosted by a remote system described by its
    connection configuration.

    ``ExplainableModelHandle`` differs from the ``ExplainerModel`` in that it
    doesn't provide the actual predict function, but only the metadata required to
    access the model.

    """

    @staticmethod
    def from_string(str_handle: str, h2o_sonar_config=None) -> "ExplainableModelHandle":
        """Create a new instance of the model handle from the string."""
        (
            connection_key,
            resource_key,
            resource_version,
        ) = commons.ResourceHandle.parse_string_handle(str_handle)

        # validate connection name
        if h2o_sonar_config:
            if not h2o_sonar_config.has_connection(connection_key):
                raise ValueError(
                    f"Connection key '{connection_key}' not found in the H2O Sonar "
                    f"config"
                )

        return ExplainableModelHandle(
            connection_key=connection_key,
            model_key=resource_key,
            model_version=resource_version,
        )

    def __init__(
        self,
        connection_key: str,
        model_key: str,
        model_version: str = "",
    ):
        """Constructor.

        Parameters
        ----------
        connection_key: str
            Key of the connection configuration defined in the H2O Sonar config.
        model_key: str
            Key which uniquely identifies the model on the host system.
        model_version: str
            Optional dataset version which might be needed to uniquely identify
            the model on the host system.

        """
        commons.ResourceHandle.__init__(
            self,
            connection_key=connection_key,
            resource_key=model_key,
            version=model_version,
        )


class ModelApi:
    """Model API interface provides uniform API allowing explainers to use
    any model (scorer) regardless provider, implementation or runtime details.

    Detects model (path to model, instance of supported model, ..) and creates
    instances of the ``Model`` class.

    """

    def __init__(self, logger: loggers.SonarLogger | None = None):
        self.logger = logger or loggers.SonarPrintLogger()

    def create_model(
        self,
        model_src,
        target_col: str,
        used_features: list[str] | None = None,
        model_type: ExplainableModelType = ExplainableModelType.unknown,
        dataset: (
            datasets.ExplainableDataset
            | datatable.Frame
            | pandas.DataFrame
            | str
            | pathlib.Path
            | None
        ) = None,
        sanitization_map: sanitization.SanitizationMap | None = None,
        **extra_params,
    ) -> ExplainableModel:
        """Create explainable model.

        Parameters
        ----------
        model_src : Any
          Path to model on the filesystem, instance of a 3rd party model,
          pickle or any other source that can be used to create explainable model.
          Information about the model can be passed to 3rd party model implementations
          (like H2O-3) which can create the model.
        target_col : str
          Target column.
        used_features : list[str] | None
          Optional list of features names used by the model - it's required in case
          of all models which don't provide introspection allowing to determine
          used features.
        model_type : ExplainableModelType
          Explainable model type hint which can be used to construct the model
          correctly.
        dataset : datasets.ExplainableDataset | datatable.Frame | pandas.DataFrame
        | str | pathlib.Path | None
          Optional training dataset.
        sanitization_map : SanitizationMap | None
          Optional dataset sanitization map used by model.

        Returns
        -------
        ExplainableModel :
          Explainable model.

        """
        del extra_params

        if dataset is not None and isinstance(dataset, (str, pathlib.Path)):
            dataset = datasets.DatasetApi(logger=self.logger).create_dataset(dataset)

        if isinstance(model_src, ExplainableModel):
            return model_src
        elif ScikitLearnModel.is_scikit_learn_model(model_src):
            return ScikitLearnModel(
                model_src=model_src,
                target_col=target_col,
                used_features=used_features,
                sanitization_map=sanitization_map,
                dataset=dataset,
                logger=self.logger,
            )
        elif H2o3Model.is_h2o3_model(model_src):
            return H2o3Model(
                model_src=model_src,
                target_col=target_col,
                used_features=used_features,
                sanitization_map=sanitization_map,
                dataset=dataset,
            )
        elif DriverlessAiModel.is_dai_model(model_src):
            return DriverlessAiModel(
                model_src=model_src,
                target_col=target_col,
                sanitization_map=sanitization_map,
                used_features=used_features,
                dataset=dataset,
            )
        elif DriverlessAiRestServerModel.is_dai_rest_server_model(model_src):
            return DriverlessAiRestServerModel(
                model_server_url=model_src,
                target_col=target_col,
                sanitization_map=sanitization_map,
                used_features=used_features,
                dataset=dataset,
            )
        elif PickleFileModel.is_pickle_file_model(model_src):
            return PickleFileModel.from_pickle(
                model_src=model_src,
                target_col=target_col,
                used_features=used_features,
                sanitization_map=sanitization_map,
                dataset=dataset,
            )
        else:
            raise ValueError(
                f"Unable to create explainable model - unknown model source: "
                f"{model_src} and type: '{model_type}'"
            )
