# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import random
import re
import traceback
from functools import partial

import datatable
import numpy
import pandas

from h2o_sonar import errors
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api.models import ExplainableModel
from h2o_sonar.lib.api.models import ExplainableModelType
from h2o_sonar.methods.utils.h2o_utils import h2o_to_dt
from h2o_sonar.utils import preprocessing


try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import h2o

    HAS_H2O = True
except ImportError:
    HAS_H2O = False

H2O_SHAP_SUPPORTED_MODELS = ["gbm", "xgboost", "drf"]
SKLEARN_TREESHAP_SUPPORTED_MODELS = "ensemble"
COL_BIAS = "bias"
COL_H2O3_BIAS = "BiasTerm"


class ShapContribsSorter:
    """Sorts SHAP contributions frame columns."""

    KEY_PER_CLS_F_CONTRIBS = "per-class-feature_contributions"
    KEY_PER_CLS_BIAS = "per-class-bias"
    KEY_OVERALL_F_CONTRIBS = "overall-feature_contributions"
    KEY_OVERALL_bias = "overall-bias"

    @staticmethod
    def strip_label_from_col(col: str, label: str) -> str:
        return col[: -len(f".{label}")]

    def __init__(
        self,
        raw_shap_contribs_col_names: list | tuple,
        labels: list[str],
    ):
        self.raw_shap_contribs_col_names = raw_shap_contribs_col_names
        self.shap_contribs_col_names = {
            ShapContribsSorter.KEY_PER_CLS_F_CONTRIBS: {},
            ShapContribsSorter.KEY_PER_CLS_BIAS: {},
            ShapContribsSorter.KEY_OVERALL_F_CONTRIBS: [],
            ShapContribsSorter.KEY_OVERALL_bias: [],
        }

        for raw_col in raw_shap_contribs_col_names:
            if COL_BIAS == raw_col:
                self.shap_contribs_col_names[
                    ShapContribsSorter.KEY_OVERALL_bias
                ].append(raw_col)
                continue
            else:
                done = False
                for label in labels:
                    if f"{COL_BIAS}.{label}" == raw_col:
                        self.shap_contribs_col_names[
                            ShapContribsSorter.KEY_PER_CLS_BIAS
                        ][label] = raw_col
                        done = True
                        break
                    elif raw_col.endswith(f".{label}"):
                        if (
                            self.shap_contribs_col_names[
                                ShapContribsSorter.KEY_PER_CLS_F_CONTRIBS
                            ].get(label, None)
                            is None
                        ):
                            self.shap_contribs_col_names[
                                ShapContribsSorter.KEY_PER_CLS_F_CONTRIBS
                            ][label] = []
                        self.shap_contribs_col_names[
                            ShapContribsSorter.KEY_PER_CLS_F_CONTRIBS
                        ][label].append(raw_col)
                        done = True
                        break
            if not done:
                self.shap_contribs_col_names[
                    ShapContribsSorter.KEY_OVERALL_F_CONTRIBS
                ].append(raw_col)

    def __str__(self):
        return json.dumps(self.shap_contribs_col_names, indent=2)

    def get_bias_col_for_label(self, label: str):
        return self.shap_contribs_col_names[ShapContribsSorter.KEY_PER_CLS_BIAS].get(
            label, None
        )

    def get_cols_for_label(self, label: str, strip_label: bool = False) -> list:
        cols = self.shap_contribs_col_names[
            ShapContribsSorter.KEY_PER_CLS_F_CONTRIBS
        ].get(label, None)
        if cols is not None and strip_label:
            cols = [
                (
                    ShapContribsSorter.strip_label_from_col(col, label)
                    if col.endswith(f".{label}")
                    else col
                )
                for col in cols
            ]
        return cols


class Shap:
    """Use Shapley values to explain any machine learning model.

    This is the primary methods interface for the SHAP library.

    """

    def __init__(
        self,
        explainable_model: ExplainableModel = None,
        masker=None,
        link=None,
        algorithm: str = "auto",
        output_names: list = None,
        nsamples: int = None,
        l1: str = "auto",
    ):
        """Build a new Shap methods for the passed explainable model.

        Parameters
        ----------
        explainable_model : object or function
            User supplied function or model object that takes a dataset of samples and
            computes the output of the model for those samples.

        masker : function, numpy.array, pandas.DataFrame, tokenizer, None, or a list of
        these for each model input
            The function used to "mask" out hidden features of the form `masked_args =
            masker(*model_args, mask=mask)`. It takes input in the same form as the
            model, but for just a single sample with a binary mask, then returns an
            iterable of masked samples. These masked samples will then be evaluated
            using the model function and the outputs averaged. As a shortcut for the
            standard masking using by SHAP you can pass a background data matrix instead
            of a function and that matrix will be used for masking. Domain specific
            masking functions are available in shap such as shap.ImageMasker for images
            and shap.TokenMasker for text. In addition to determining how to replace
            hidden features, the masker can also constrain the rules of the cooperative
            game used to explain the model. For example shap.TabularMasker(data,
            hclustering="correlation") will enforce a hierarchical clustering of
            coalitions for the game (in this special case the attributions are
            known as the Owen values).

        link : function
            The link function used to map between the output units of the model and the
            SHAP value units. By default, it is shap.links.identity, but
            shap.links.logit can be useful so that expectations are computed in
            probability units while explanations remain in the (more naturally
            additive) log-odds units. For more details on how link functions work see
            any overview of link functions for generalized linear models.

        algorithm : str
            Values can be one of the following: "auto", "permutation", "partition",
            "tree", "sampling", "linear", "deep", or "gradient". The algorithm used to
            estimate the Shapley values. There are many different algorithms that
            can be used to estimate the Shapley values (and the related value for
            constrained games), each of these algorithms have various tradeoffs and are
            preferable in different situations. By default, the "auto" options attempts
            to make the best choice given the passed model and masker, but this choice
            can always be overriden by passing the name of a specific algorithm. The
            type of algorithm used will determine what type of subclass object is
            returned by this constructor, and you can also build those subclasses
            directly if you prefer or need more fine-grained control over their options.

        output_names : None or list of strings
            The names of the model outputs. For example if the model is an image
            classifier, then output_names would be the names of all the output classes.
            This parameter is optional. When output_names is None then the Explanation
            objects produced by these methods will not have any output_names, which
            could affect downstream plots.

        nsamples : "auto" or int
            Number of times to re-evaluate the model when explaining each prediction.
            More samples lead to lower variance estimates of the SHAP values. The
            "auto" setting uses `nsamples = 2 * X.shape[1] + 2048`.

            Only applies to KernelExplainer.

        l1 : "num_features(int)", "auto" (default for now, but deprecated), "aic",
            "bic", or float
            The l1 regularization to use for feature selection (the estimation
            procedure is based on a debiased lasso). The auto option currently uses
            "aic" when less that 20% of the possible sample space is enumerated,
            otherwise it uses no regularization. THE BEHAVIOR OF  "auto" WILL CHANGE
            in a future version to be based on num_features instead of AIC.
            The "aic" and "bic" options use the AIC and BIC rules for regularization.
            Using "num_features(int)" selects a fix number of top features. Passing a
            float directly sets the "alpha" parameter of the sklearn.linear_model.
            Lasso model used for feature selection.

            Only applies to KernelExplainer.

        """
        if not HAS_SHAP:
            commons.raise_opt_import_err("shap")

        self.explainable_model = explainable_model
        self.masker = masker
        self.link = link or shap.links.identity
        self.algorithm = algorithm
        self.output_names = output_names
        self.shap_values = None
        self.nsamples = nsamples
        self.l1 = l1
        self.labels = None

    def explain(self, X, experiment_type: str = ""):
        """Calculate Shapley values.

        Parameters
        ----------
        X : pandas.core.frame.DataFrame or datatable.Frame
            Dataset to use for Shapley value computation.
        experiment_type : str
            Optional experiment type to handle classes correctly.

        Returns
        -------
        datatable.Frame
            Frame that contains Shapley values with the same shape as the input dataset.
        """
        if not HAS_SHAP:
            commons.raise_opt_import_err("shap")
        if not HAS_H2O:
            commons.raise_opt_import_err("h2o")

        if not (
            isinstance(X, pandas.DataFrame)
            or isinstance(X, datatable.Frame)
            or isinstance(X, h2o.H2OFrame)
        ):
            raise ValueError(
                f"Input dataset should be of type datatable.Frame, or pandas.DataFrame,"
                f" but got type {type(X)}"
            )

        if isinstance(X, datatable.Frame):
            X = X.to_pandas()

        model_type = self.explainable_model.model_type
        if model_type == ExplainableModelType.scikit_learn:
            # categorical handling
            (X, _, _) = preprocessing.categorical_encoder(X)

            if self.explainable_model.model_src.classes_ is not None:
                self.labels = list(self.explainable_model.model_src.classes_)
            if (
                commons.base_pkg(self.explainable_model.model_src)[1]
                == SKLEARN_TREESHAP_SUPPORTED_MODELS
                and len(self.labels) <= 2
            ):
                # tree ensembles (XGBoost/LightGBM/CatBoost/scikit-learn/pyspark models)
                explainer = shap.Explainer(
                    model=self.explainable_model.model_src,  # have to use model_src
                    # otherwise shap errors out, see:
                    # https://github.com/slundberg/shap/issues/2460
                    algorithm=self.algorithm,
                    link=self.link,
                    output_names=self.output_names,
                )
                self.shap_values = explainer(X)
                self.shap_values = datatable.Frame(
                    numpy.concatenate(
                        (
                            self.shap_values.base_values,
                            self.shap_values.values,
                        ),
                        axis=1,
                    ),
                    names=(
                        [COL_BIAS] + self.output_names
                        if self.output_names
                        else [COL_BIAS] + list(X.columns)
                    ),
                )
                return self.shap_values
            else:
                background_data = shap.kmeans(X, 1)

                if experiment_type == commons.ExperimentType.regression.name:
                    labels = None
                    k_explainer = shap.KernelExplainer(
                        (
                            self.explainable_model.model_src.predict_proba
                            if self.labels
                            else self.explainable_model.model_src.predict
                        ),
                        data=background_data,
                    )
                else:
                    labels = self.labels
                    k_explainer = shap.KernelExplainer(
                        (
                            self.explainable_model.model_src.predict_proba
                            if len(self.explainable_model.model_src.classes_) > 2
                            else self.explainable_model.model_src.predict
                        ),
                        data=background_data,
                    )

                self.shap_values = calc_kernel_explainer(
                    dataset=X,
                    k_explainer=k_explainer,
                    original_features=has_output_names(
                        self.output_names, list(X.columns)
                    ),
                    nsamples=self.nsamples,
                    l1=self.l1,
                    labels=labels,
                )
                return self.shap_values
        elif model_type == ExplainableModelType.h2o3:
            if (
                self.explainable_model.model_src.algo in H2O_SHAP_SUPPORTED_MODELS
                and self.explainable_model.model_src._model_json["output"][
                    "model_category"
                ]
                != "Multinomial"
            ):
                # H2O implementation of Shap
                h2o_shap = self.explainable_model.model_src.predict_contributions(
                    h2o.H2OFrame(X) if not isinstance(X, h2o.H2OFrame) else X
                )
                # IMPORTANT: h2o_shap frame has columns ORDERED by feature importance,
                # therefore setting columns names from the dataset would assign
                # contributions to completely different features - this is wrong:
                # self.shap_values = h2o_to_dt(
                #     h2o_shap,
                #     self.output_names + [COL_BIAS]
                #     if self.output_names
                #     else list(X.columns) + [COL_BIAS],
                # )

                # keep column names and fix bias column only
                col_names = []
                for n in h2o_shap.names:
                    col_names.append(COL_BIAS if n == COL_H2O3_BIAS else n)

                self.shap_values = h2o_to_dt(X=h2o_shap, col_names=col_names)

                return self.shap_values
            else:
                # Kernel Shap
                h2o_wrapper = H2OShapWrapper(
                    self.explainable_model.model_src,
                    has_output_names(self.output_names, list(X.columns)),
                )
                background_data = h2o_shap_kmeans(X, 1)
                explainer = shap.KernelExplainer(
                    h2o_wrapper.predict, data=background_data
                )
                h2o_pd_frame = X.as_data_frame()
                self.shap_values = calc_kernel_explainer(
                    dataset=h2o_pd_frame,
                    k_explainer=explainer,
                    original_features=has_output_names(
                        self.output_names, list(X.columns)
                    ),
                    nsamples=self.nsamples,
                    l1=self.l1,
                    labels=self.explainable_model.model_src._model_json["output"][
                        "domains"
                    ][
                        self.explainable_model.model_src._model_json["output"][
                            "names"
                        ].index(self.explainable_model.meta.target_col)
                    ],
                )
                return self.shap_values
        elif model_type == ExplainableModelType.driverless_ai:
            if self.explainable_model.meta.has_shapley_values:
                self.shap_values = self.explainable_model.shapley_values(
                    datatable.Frame(X),
                )
            else:
                background_data = shap.kmeans(X, 1)
                dai_shap_predictor = MliDaiMojoPredictor(
                    scorer=self.explainable_model.predict,
                    feature_names=has_output_names(
                        self.output_names,
                        self.explainable_model.meta.used_features,
                    ),
                    num_classes=self.explainable_model.meta.num_labels,
                )
                k_explainer = shap.KernelExplainer(
                    dai_shap_predictor.predict,
                    data=background_data,
                )
                self.shap_values = calc_kernel_explainer(
                    dataset=X,
                    k_explainer=k_explainer,
                    original_features=self.explainable_model.meta.used_features,
                    nsamples=self.nsamples,
                    l1=self.l1,
                    labels=self.labels,
                )
            orig_col_names = list(self.shap_values.names)
            orig_col_names = [re.sub("contrib_", "", ele) for ele in orig_col_names]
            transformed_col_names = self.explainable_model.meta.transformed_features
            self.shap_values.names = orig_col_names

            # use blacklist to remove transformed features
            if transformed_col_names:
                blacklist = re.compile(
                    "|".join([re.escape(word) for word in transformed_col_names])
                )
                valid_orig_col_names = [
                    word for word in orig_col_names if not blacklist.search(word)
                ]
                return self.shap_values[:, valid_orig_col_names]
            return self.shap_values
        elif model_type == ExplainableModelType.driverless_ai_rest:
            # categorical handling
            (X, label_encoder, cat_variables) = preprocessing.categorical_encoder(X)

            # predict function w/ label decoder
            def _decode_predict_pandas(
                pred_fn,
                columns,
                cat_vars,
                mcle,
                dataset,
            ):
                # normalize frame
                if not isinstance(dataset, pandas.DataFrame):
                    dataset = pandas.DataFrame(dataset.tolist(), columns=columns)

                # categorical features inverse label encoding
                if mcle:
                    dataset[cat_vars] = dataset[cat_vars].astype(numpy.int64)
                    mcle.inverse_transform(dataset)

                # score
                preds = pred_fn(dataset)

                # scoring output conversion to the format expected by 3rd party library
                if isinstance(preds, pandas.core.frame.DataFrame):
                    preds = preds.to_numpy()
                if preds.ndim == 2:
                    preds = preds.flatten()

                # predictions
                return preds

            encoded_labels = []
            if self.explainable_model.meta.labels:
                encoded_labels = [
                    nl for nl in range(self.explainable_model.meta.num_labels)
                ]
            # Kernel SHAP
            predict_fn = partial(
                _decode_predict_pandas,
                self.explainable_model.predict_pandas,
                list(X.columns),
                cat_variables,
                label_encoder,
            )
            background_data = shap.kmeans(X, 1)
            k_explainer = shap.KernelExplainer(
                predict_fn,
                data=background_data,
            )
            self.shap_values = calc_kernel_explainer(
                dataset=X,
                k_explainer=k_explainer,
                original_features=has_output_names(self.output_names, list(X.columns)),
                nsamples=self.nsamples,
                l1=self.l1,
                labels=encoded_labels,
            )
            return self.shap_values
        elif model_type == ExplainableModelType.mock:
            # mock model Shapley values
            if self.explainable_model.meta.num_labels <= 2:
                features = self.explainable_model.meta.used_features
            else:
                orig_feats_contrib_names = [
                    f"{n}" for n in self.explainable_model.meta.used_features
                ]
                orig_feats_contrib_names.append(COL_BIAS)
                class_orig_feats_names = []
                for clazz in self.explainable_model.meta.labels:
                    class_names = []
                    for label in orig_feats_contrib_names:
                        class_names.append(f"{label}.{clazz}")
                    class_orig_feats_names.extend(class_names)
                features = class_orig_feats_names

            shapleys_dict = {}
            for f in features:
                shapleys_dict[f] = []
                for r in range(0, X.shape[0]):
                    s = random.random()
                    shapleys_dict[f].append(s * -1 if random.random() > 0.5 else s)
            self.shap_values = datatable.Frame(shapleys_dict)
            return self.shap_values
        else:
            raise ValueError(
                f"Model type '{self.explainable_model.model_type}' "
                f"(source='{self.explainable_model.model_src}', "
                f"source_type='{type(self.explainable_model.model_src)}') is not "
                f"supported"
            )

    def plot(self, plot_type="summary"):
        """Create Shapley plot.

        Parameters
        ----------
        plot_type: str
            Type of Shapley plot to construct, which can either be `summary` or `bar`.

        """
        if not HAS_SHAP:
            commons.raise_opt_import_err("shap")

        if plot_type == "summary":
            shap.plots.beeswarm(self.shap_values)
        elif plot_type == "bar":
            shap.plots.bar(self.shap_values)
        else:
            raise ValueError(
                f"Plot type, {plot_type}, is not supported. Please choose "
                f"either 'summary' or 'bar'"
            )


class H2OShapWrapper:
    def __init__(self, h2o_model, feature_names):
        """H2O-3 model wrapper for Kernel Explainer.

        Parameters
        ----------
        h2o_model : h2o.estimators.H2OEstimator
            H2O-3 model.

        feature_names : list
            List of features used by the h2o-3 model.

        """
        self.h2o_model = h2o_model
        self.feature_names = feature_names
        self.predictions = None

    def predict(self, X):
        import h2o

        if isinstance(X, pandas.Series):
            X = X.values.reshape(1, -1)
        frame_for_pred = h2o.H2OFrame(X)
        frame_for_pred.names = self.feature_names
        self.predictions = self.h2o_model.predict(frame_for_pred).as_data_frame().values
        if self.h2o_model._model_json["output"]["model_category"] != "Multinomial":
            return self.predictions[:, -1]
        else:
            return self.predictions[:, 1 : len(self.predictions[0])]


class MliDaiMojoPredictor:
    def __init__(
        self,
        scorer,
        feature_names,
        num_classes,
        label_encoder=None,
        cat_variables=None,
    ):
        self.scorer = scorer
        self.feature_names = feature_names
        self.num_classes = num_classes
        self.label_encoder = label_encoder
        self.cat_variables = cat_variables

    def predict(self, data_asarray):
        if self.label_encoder and self.cat_variables:
            data_asarray = pandas.DataFrame(
                data=data_asarray, columns=self.feature_names
            )
            data_asarray[self.cat_variables] = data_asarray[self.cat_variables].astype(
                numpy.int64
            )
            self.label_encoder.inverse_transform(data_asarray)

        data_asframe = datatable.Frame(data_asarray, names=self.feature_names)

        if self.num_classes > 2:  # multiclass
            return self.scorer(data_asframe).to_numpy()
        elif self.num_classes == 2:  # binary classification
            predictions = self.scorer(data_asframe)
            return (
                self.scorer(data_asframe)[:, 1].to_pandas()
                if predictions.shape[1] == 2
                else predictions.to_pandas()
            )
        else:  # regression
            return self.scorer(data_asframe).to_pandas()


def calc_kernel_explainer(
    dataset,
    k_explainer,
    original_features,
    nsamples=None,
    l1="auto",
    labels=None,
):
    """Calculate Kernel Explainer.

    Parameters
    ----------
    dataset : numpy.array or pandas.DataFrame or any scipy.sparse matrix
        Matrix of data samples to calculate Shapley (# samples x # features).

    k_explainer : shap.KernelExplainer
        Kernel Explainer object.

    original_features : list
        List of original features to use as column names for resulting frame.

    nsamples : "auto" or int
        Number of times to re-evaluate the model when explaining each prediction.
        More samples lead to lower variance estimates of the SHAP values. The
        "auto" setting uses `nsamples = 2 * X.shape[1] + 2048`.

    l1 : "num_features(int)", "auto" (default for now, but deprecated), "aic",
        "bic", or float
        The l1 regularization to use for feature selection (the estimation
        procedure is based on a debiased lasso). The auto option currently uses
        "aic" when less that 20% of the possible sample space is enumerated,
        otherwise it uses no regularization. THE BEHAVIOR OF  "auto" WILL CHANGE
        in a future version to be based on num_features instead of AIC.
        The "aic" and "bic" options use the AIC and BIC rules for regularization.
        Using "num_features(int)" selects a fix number of top features. Passing a
        float directly sets the "alpha" parameter of the sklearn.linear_model.
        Lasso model used for feature selection.

    labels : list
        List of target labels.

    Returns
    -------
    datatable.Frame :
        Frame that contains Shapley values per feature and bias column.

    """

    if isinstance(nsamples, str):
        given_nsamples = nsamples if nsamples == "auto" else int(nsamples)
    elif nsamples is not None:
        given_nsamples = nsamples
    else:
        given_nsamples = 2 * len(original_features)

    try:
        k_shap_values = k_explainer.shap_values(
            dataset[original_features],
            nsamples=given_nsamples,
            l1=l1,
        )
    except numpy.linalg.LinAlgError as ex:
        err_msg = (
            f"Kernel methods calculation for {nsamples} samples and "
            f"features: '{original_features}' failed with: {ex}: "
            f"\n{traceback.format_exc()}"
        )
        raise errors.MliError(err_msg)

    # Multinomial
    k_shap_values_mult_dt = None
    if labels and len(labels) > 2:
        idx = 0
        # Initialize Shapley frame for original features in multinomial case
        k_shap_values_mult_dt = datatable.Frame()
        kernel_explainer_frame = datatable.Frame(
            numpy.concatenate(k_shap_values, axis=1)
        )
        for label in labels:
            class_dt = kernel_explainer_frame[:, idx : idx + len(original_features)]

            old_names = class_dt.names
            new_names = [f"{s}.{label}" for s in original_features]
            old_new_names_dict = dict(zip(old_names, new_names, strict=False))
            class_dt.names = old_new_names_dict
            class_dt[:, f"{COL_BIAS}.{label}"] = k_explainer.expected_value[
                labels.index(label)
            ]
            # Cbind to create final frame
            k_shap_values_mult_dt.cbind(class_dt)
            # Export Shapley frames for original features
            idx += len(original_features)

    if labels and len(labels) == 2:  # Binary
        k_shap_values = datatable.Frame(k_shap_values, names=original_features)
        k_shap_values[:, COL_BIAS] = k_explainer.expected_value
        return k_shap_values
    elif labels is None:  # Regression
        k_shap_values = (
            k_shap_values[0] if isinstance(k_shap_values, list) else k_shap_values
        )
        k_shap_values = datatable.Frame(k_shap_values, names=original_features)
        k_shap_values[:, COL_BIAS] = k_explainer.expected_value
        return k_shap_values
    elif k_shap_values_mult_dt:  # Multinomial
        return k_shap_values_mult_dt
    else:
        raise ValueError(
            "Shapley is not built! No requirement for Shapley calculation is satisfied"
        )


def h2o_shap_kmeans(X, k):
    """Summarize a dataset with k mean samples weighted by the number of data points
    they each represent. Note, this method is different than `shap.kmeans` because
    it uses h2o's kmeans, which can handle categoricals and does not require a user
    to do any one hot encoding.

    Parameters
    ----------
    X : numpy.array or pandas.DataFrame or any scipy.sparse matrix
        Matrix of data samples to summarize (# samples x # features)

    k : int
        Number of means to use for approximation.

    Returns
    -------
    DenseData :
        DenseData object.

    """
    from h2o import estimators
    from shap.utils._legacy import DenseData

    kmeans_h2o = estimators.H2OKMeansEstimator(k=k, seed=0)
    kmeans_h2o.train(training_frame=X)
    h2o_centers = list(kmeans_h2o._model_json["output"]["centers"].cell_values[0])[1:]
    k_centers = numpy.asarray([h2o_centers])
    k_labels = numpy.concatenate(kmeans_h2o.predict(X).as_data_frame().values, axis=0)

    group_names = [str(i) for i in range(X.shape[1])]
    if str(type(X)).endswith("'pandas.core.frame.DataFrame'>"):
        group_names = X.columns

    return DenseData(k_centers, group_names, None, 1.0 * numpy.bincount(k_labels))


def has_output_names(output_names, x_names):
    """Decipher if user passed in parameter, `output_names`.

    Parameters
    ----------
    output_names : list
      List of column names to use for output Shapley frame.

    x_names : list
      List of input dataset column names

    Returns
    -------
    List :
      List of output names.

    """
    return output_names if output_names else x_names
