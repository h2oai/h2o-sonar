# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import itertools
import json
import os
import traceback

import airium
import datatable
import matplotlib.pyplot as plt
import numpy
import pandas

from h2o_sonar import errors
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api import plots
from h2o_sonar.lib.api import results
from h2o_sonar.methods import _pd
from h2o_sonar.methods.core import method
from h2o_sonar.utils import progress
from h2o_sonar.utils import sanitization


try:
    import tabulate

    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# shortcuts
FeaturesMetadata = method.FeaturesMetadata
ModelTypes = models.ExplainableModelType


class PdFor2FeaturesExplainer(explainers.Explainer):
    """PD for two features explainer."""

    PARAM_SAMPLE_SIZE = "sample_size"
    PARAM_FEATURES = "features"
    PARAM_MAX_FEATURES = "max_features"
    PARAM_GRID_RESOLUTION = "grid_resolution"
    PARAM_OOR_GRID_RESOLUTION = "oor_grid_resolution"
    PARAM_QTILE_GRID_RESOLUTION = "quantile-bin-grid-resolution"
    PARAM_QTILE_BINS = "quantile-bins"
    PARAM_PLOT_TYPE = "plot_type"

    # static option: enable/disable datatable 1-frame strategy optimization
    OPT_ICE_1_FRAME_ENABLED = True

    # calculation progress reporting range (engine 0 - 10%, normalization 90% - 100%)
    PROGRESS_MIN = 0.1
    PROGRESS_MAX = 0.9

    MAX_FEATURES = 3
    GRID_RESOLUTION = 10
    SAMPLE_SIZE = 25000

    FILE_Y_HAT = "mli_dataset_y_hat.jay"

    _display_name = "Partial Dependence Plot for Two Features"
    _description = """Partial dependence for 2 features portrays the average
prediction behavior of a model across the domains of two input variables
i.e. interaction of feature tuples with the prediction. While PD for one feature
produces 2D plot, PD for two features produces 3D plots. This explainer plots PD for
two features using heatmap, contour 3D or surface 3D.
"""
    _iid = True
    _time_series = True
    _regression = True
    _binary = True
    _global_explanation = True
    _explanation_types = [
        e10s.PartialDependenceExplanation,
    ]
    _optional_explanation_types = [e10s.GlobalHtmlFragmentExplanation]
    _keywords = [explainers.Explainer.KEYWORD_IS_SLOW]
    _parameters = [
        explainers.ExplainerParam(
            param_name=PARAM_SAMPLE_SIZE,
            description="Sample size for Partial Dependence Plot of 2 features.",
            param_type=commons.ExplainerParamType.int,
            default_value=SAMPLE_SIZE,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_MAX_FEATURES,
            description="Partial Dependence Plot number of features.",
            param_type=commons.ExplainerParamType.int,
            default_value=MAX_FEATURES,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_FEATURES,
            description=(
                "List of features from which to choose pairs to compute PD for "
                "two features."
            ),
            param_type=commons.ExplainerParamType.list,
            default_value=None,
            tags=[explainers.ExplainerParam.TAG_SRC_DATASET_COLUMN_NAMES],
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_GRID_RESOLUTION,
            description=(
                "Partial Dependence Plot observations per bin (number of"
                " equally spaced points used to create bins)."
            ),
            param_type=commons.ExplainerParamType.int,
            default_value=GRID_RESOLUTION,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_OOR_GRID_RESOLUTION,
            description="Partial Dependence Plot number of out of range bins.",
            param_type=commons.ExplainerParamType.int,
            default_value=0,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_QTILE_GRID_RESOLUTION,
            description=(
                "Partial Dependence Plot quantile binning (total quantile"
                " points used to create bins)."
            ),
            param_type=commons.ExplainerParamType.int,
            default_value=0,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_PLOT_TYPE,
            description="Plot type.",
            param_type=commons.ExplainerParamType.str,
            default_value=plots.Data3dPlot.PLOT_TYPE_HEATMAP,
            predefined=plots.Data3dPlot.PLOT_TYPES,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
    ]
    _requires_preloaded_predictor = False

    def __init__(self):
        explainers.Explainer.__init__(self)

        self.args = None

        self.log_name = "PD for 2 features"

    @staticmethod
    def _resolve_model_features(
        model: models.ExplainableModel | None,
    ) -> tuple[list, dict, list]:
        """Resolve model features to explain, features to skip and model features
        metadata.

        """
        features = []
        features_meta = {}
        skip_features = []

        model_meta: models.ExplainableModelMeta | None = model.meta if model else None

        if model_meta:
            # IMPROVE sort features by importance (if known)
            features = model_meta.used_features
            features_meta = model_meta.features_metadata.to_dict()

        return features, features_meta, skip_features

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        model: models.ExplainableModel | None = None,
        **explainer_params,
    ) -> bool:
        explainers.Explainer.check_compatibility(self, params, **explainer_params)

        if not HAS_TABULATE:
            self.logger.warning(self._check_compatibility_pckg_err_msg("tabulate"))
            return False

        # check: target column
        if not params.target_col:
            self.logger.warning(
                f"{self.log_name} explainer requires target column and "
                f"therefore it is not compatible with explanation run with parameters: "
                f"{params.dump()}",
            )
            return False

        # check: at least 1 PD/ICE compatible feature
        try:
            self.args = PdFor2FeaturesArgs(self._parameters)
            self.args.resolve_params(
                explainer_params=PdFor2FeaturesArgs.json_str_to_dict(
                    self.explainer_params_as_str
                ),
            )

            (
                features,
                features_metadata,
                skip_features,
            ) = self._resolve_model_features(model)

            if self.params.drop_cols:
                drop_features: list = sanitization.sanitize_strings(
                    self.params.drop_cols
                )
                skip_features = commons.add_string_list(skip_features, drop_features)
            if skip_features:
                features = [x for x in features if x not in skip_features]
            if not features:
                err_msg = (
                    f"{self.__class__.__name__} not compatible - no features were "
                    f"specified/used by the model ({features}) "
                )
                self.logger.warning(err_msg)
                raise errors.ExplainerCompatibilityError(err_msg)
        except Exception as ex:
            err_msg = (
                f"{self.log_name} features compatibility check failed: {ex}\n"
                f"{traceback.format_exc()}",
            )
            self.logger.error(err_msg)
            raise err_msg

        return True

    def setup(
        self,
        model: models.ExplainableModel | None,
        persistence: persistences.ExplainerPersistence,
        key: str = "",
        params: commons.CommonInterpretationParams | None = None,
        **explainer_params,
    ):
        explainers.Explainer.setup(
            self,
            model=model,
            persistence=persistence,
            key=key,
            params=params,
            **explainer_params,
        )

        self.args = PdFor2FeaturesArgs(self._parameters)
        self.args.resolve_params(
            explainer_params=PdFor2FeaturesArgs.json_str_to_dict(
                self.explainer_params_as_str
            ),
        )

        self.log_name = f"PD for 2 features {self.mli_key}/{self.key}"

    def explain(
        self,
        X: datatable.Frame,
        y: datatable.Frame | None = None,
        explanations_types: list = None,
        **e_params,
    ):
        pd = self._calculate_pd(dataset=X)
        if not pd:
            err_msg = f"No {self.log_name} result - no features for which to compute PD"
            self.logger.error(err_msg)
            raise errors.MliError(err_msg)

        return self.normalize(pd)

    def _calculate_pd(
        self,
        dataset: datatable.Frame,
    ) -> _pd.PD | None:
        """PD for 2 features calculation using scorer and original dataset.

        Parameters
        ----------
        dataset : datatable.Frame
            Dataset.

        Returns
        -------
        PD | None :
            PD method instance.

        """

        self.logger.info(f"{self.log_name} BEGIN calculation")

        model_features = self.model_meta.used_features or []
        explainer_dir_path = self.persistence.get_explainer_working_dir()
        max_features = self.args.get(PdFor2FeaturesExplainer.PARAM_MAX_FEATURES)
        grid_resolution = self.args.get(PdFor2FeaturesExplainer.PARAM_GRID_RESOLUTION)
        use_features = self.args.get(PdFor2FeaturesExplainer.PARAM_FEATURES)
        qtile_grid_resolution = self.args.get(
            PdFor2FeaturesExplainer.PARAM_QTILE_GRID_RESOLUTION
        )
        qtile_bins_dict_str = self.args.get(PdFor2FeaturesExplainer.PARAM_QTILE_BINS)

        self.logger.info(
            f"{self.log_name} loading dataset",
        )

        # TODO downsample dataset - if required by user: sample_limit parameter

        self.logger.info(
            f"{self.log_name} loaded dataset has "
            f"{dataset.shape[0]} rows and {dataset.shape[1]} columns",
        )

        self.logger.info(
            f"{self.log_name} getting features list, importance and metadata"
        )
        (
            features,
            features_meta,
            skip_features,
        ) = self._resolve_model_features(self.model)
        features_meta = features_meta or method.FeaturesMetadata.create_blank_dict()
        if not any(features_meta.values()):
            if self.dataset_meta and self.dataset_meta.columns_cat:
                features_meta[FeaturesMetadata.KEY_CATEGORICAL_FEATURES] = (
                    self.dataset_meta.columns_cat
                )
            if self.dataset_meta and self.dataset_meta.columns_cat:
                features_meta[FeaturesMetadata.KEY_NUMERIC_FEATURES] = (
                    self.dataset_meta.columns_cat
                )

        if use_features:
            self.logger.info(f"{self.log_name} user-specified features: {use_features}")
            use_features = sanitization.sanitize_strings(strings=use_features)
            features = use_features.copy()
            for use_feature in use_features:
                if use_feature not in dataset.names:
                    features.remove(use_feature)
        else:
            self.logger.info(
                f"{self.log_name} all most important model features: {features}"
            )

        # skip image features
        image_trans_features = (
            self.model_meta.features_metadata.image_features if self.model_meta else []
        )
        features = [x for x in features if x not in image_trans_features]

        if self.params.drop_cols:
            drop_features: list = sanitization.sanitize_strings(self.params.drop_cols)
            skip_features.extend(drop_features)

        if skip_features:
            features = [x for x in features if x not in skip_features]

        pdc = None
        if features:
            pdc = _pd.PD(f"Explainer: {self.mli_key}/{self.key}")

            # ensure unique values for a feature is > 1
            feat_uniq_vals = dataset[:, features].nunique().to_dict()
            # calculate meaningful PD/ICE for features with cardinality 1 by
            # adding OOR bins (if user didn't request it)
            feat_unique_1 = []
            for feature in features:
                if feat_uniq_vals[feature][0] == 1:
                    if not self.args.get(
                        PdFor2FeaturesExplainer.PARAM_OOR_GRID_RESOLUTION, 0
                    ):
                        self.logger.info(
                            f"{self.log_name}: for '{feature}' will be calculated with "
                            f"OOR bins as it has only 1 unique value",
                        )
                        # calculate PD/ICE in 2nd step with OOR (which wasn't requested)
                        feat_unique_1.append(feature)
                        features.remove(feature)
                    # else user asked for grid resolution -> do NOT discard feature
                    continue
                if feat_uniq_vals[feature][0] == 0:
                    self.logger.info(
                        f"{self.log_name}: removing {feature} from PD/ICE features "
                        f"list since it has no valid values",
                    )
                    features.remove(feature)

            if qtile_grid_resolution > 0:
                # construct qtile binning dict
                qtile_bins = [qtile_grid_resolution] * len(features)
                qtile_dict = dict(zip(features, qtile_bins, strict=False))
                features_meta[FeaturesMetadata.KEY_QUANTILE_BINS] = qtile_dict
            else:
                qtile_dict = {}

            if qtile_bins_dict_str:
                try:
                    qtile_bins_dict = json.loads(qtile_bins_dict_str)
                    if not isinstance(qtile_bins_dict, dict):
                        raise ValueError(
                            f"Invalid PD/ICE quantile bins parameter type - "
                            f"must be dictionary: "
                            f"'{qtile_bins_dict}'"
                        )
                    if not set(list(qtile_bins_dict.keys())).issubset(set(features)):
                        # quantile parameters for subsequent on-demand global run
                        # typically will not be anticipated in case of the initial
                        # PD/ICE run, therefore fail in case of initial run and
                        # fail-over in case of subsequent on-demand run (warning
                        # message only)
                        self.logger.warning(
                            f"Dictionary of quantile bins per feature must contain "
                            f"all or a subset of PD features. PD features are: "
                            f"{features} and quantile bins per feature dictionary "
                            f"is: {qtile_bins_dict}"
                        )
                    else:
                        if qtile_grid_resolution > 0:
                            # update `qtile_dict` w/ new key/values from
                            # qtile_bins_dict, if needed
                            qtile_dict_updated = {
                                key: qtile_bins_dict.get(key, qtile_dict[key])
                                for key in qtile_dict
                            }
                            features_meta[FeaturesMetadata.KEY_QUANTILE_BINS] = (
                                qtile_dict_updated
                            )
                        else:
                            features_meta[FeaturesMetadata.KEY_QUANTILE_BINS] = (
                                qtile_bins_dict
                            )
                except Exception as ex:
                    raise ValueError(
                        f"Unable to parse PD/ICE quantile bins parameter: "
                        f"'{qtile_bins_dict_str}': {ex}\n{traceback.format_exc()}"
                    )

            if not features:
                raise ValueError(
                    f"{self.log_name} requires at least 1 valid feature, however, "
                    "all experiment features were discarded (insufficient cardinality, "
                    "dropped by experiment, dropped by user in experiment or MLI, ...)."
                    " Please check explainer log for more details."
                )

            # use limit on the number of features for which to calc. PD (if specified)
            if max_features > 0:
                features = features[:max_features]

            self.logger.info(
                f"{self.log_name} features used by model: {model_features}"
            )
            self.logger.info(
                f"{self.log_name}: calculating PD for features {features}",
            )

            self.logger.info(f"{self.log_name} feature metadata: {features_meta}")

            pdc.opt_1_prediction = PdFor2FeaturesExplainer.OPT_ICE_1_FRAME_ENABLED
            self.logger.info(
                f"{self.log_name} 1 frame strategy: {pdc.opt_1_prediction}",
            )

            features = features[:max_features]
            feature_pairs = list(itertools.combinations(features, 2))
            self.logger.debug(f"{self.log_name} Feature pairs: {feature_pairs}")

            pdc.explain(
                feature_pairs,
                dataset,
                # IMPROVE: fast approx, MOJO, IS_SCORER, ...
                predict_method=self.model.predict_pandas,
                grid_resolution=int(grid_resolution),
                features_meta=features_meta,
                out_of_range_resolution=self.args.get(
                    PdFor2FeaturesExplainer.PARAM_OOR_GRID_RESOLUTION, 0
                ),
                progress_callback=progress.MethodToExplainerProgressReporter(
                    min_progress=PdFor2FeaturesExplainer.PROGRESS_MIN,
                    max_progress=PdFor2FeaturesExplainer.PROGRESS_MAX,
                    callback=self.report_progress,
                ),
            )
            self.logger.debug(f"{self.log_name} result: {pdc}")
            self.logger.debug(
                f"{self.log_name} scorer calls: {pdc.diagnostics.total_scorer_calls}",
            )

            if feat_unique_1:
                self.logger.warning(
                    f"{self.log_name} SKIPPED calculation for features with 1 unique "
                    f"value (using OOR) {feat_unique_1} with 1-frame strategy set to: "
                    f"{pdc.opt_1_prediction}",
                )

            # original row predictions (needed by ICE explanation)
            y_hat_frame = self.model.predict_datatable(dataset)
            if (
                len(y_hat_frame.shape) > 1
                and y_hat_frame.shape[1] > 2
                and self.model_meta.labels
            ):
                y_hat_frame.names = sanitization.sanitize_strings(
                    self.model_meta.labels
                )
            y_hat_frame.to_jay(
                os.path.join(explainer_dir_path, PdFor2FeaturesExplainer.FILE_Y_HAT)
            )

        return pdc

    def normalize(self, pd: _pd.PD):
        report_as_list = self._preprocess_explanations(pd)

        explanations = list()
        explanations.append(self._normalize_markdown(report_as_list))
        explanations.append(self._normalize_json_and_csv(report_as_list))
        # OPTIONAL explanation: HTML fragment
        if self.config and self.config.create_html_representations:
            try:
                explanations.append(self._normalize_html(report_as_list))
            except Exception as ex:
                self.logger.warning(
                    f"{self.log_name}: HTML representation creation failed: {ex}"
                )

        # IMPROVE: purge report(s) and images from work / work content

        return explanations

    _KEY_TITLE = "title"
    _KEY_REPORT_BODY = "report_body"
    _KEY_DATA_DICT = "data_dict"
    _KEY_FEATURE_NAMES = "feature_names"
    _KEY_X_TICS_LABELS = "x_ticks_labels"
    _KEY_Y_TICS_LABELS = "Y_ticks_labels"
    _KEY_IMAGE_FILE = "image_file_name"
    _KEY_WORK_IMAGE_PATH = "work_image_path"

    def _preprocess_explanations(self, pd: _pd.PD) -> list:
        """Pre-process for easy normalization and render charts to work."""
        pd_explanations = pd.explanations()

        # TODO rewrite plot below to plots.Data3dPlot to reuse implementation from there

        # resolve plot type
        plot_type = self.args.get(
            PdFor2FeaturesExplainer.PARAM_PLOT_TYPE,
            plots.Data3dPlot.PLOT_TYPE_HEATMAP,
        )
        if plot_type not in plots.Data3dPlot.PLOT_TYPES:
            default_plot = plots.Data3dPlot.PLOT_TYPE_HEATMAP
            self.logger.warning(
                f"{self.log_name}: unknown plot type '{plot_type}' - using fallback "
                f"to {default_plot}"
            )
            plot_type = default_plot

        # list w/ per PD explanation item pre-processed dictionary
        report_as_list = []
        for i, pd_result in enumerate(pd_explanations):
            try:
                self.logger.debug(
                    f"Chart[{i}]: {pd_result} t: {type(pd_result)}",
                )

                report_item = {
                    PdFor2FeaturesExplainer._KEY_TITLE: (
                        f"'{pd_result[0]}' and '{pd_result[1]}'"
                    ),
                    PdFor2FeaturesExplainer._KEY_FEATURE_NAMES: [
                        pd_result[0],
                        pd_result[1],
                    ],
                    PdFor2FeaturesExplainer._KEY_REPORT_BODY: "",
                    PdFor2FeaturesExplainer._KEY_IMAGE_FILE: f"image-{i}.png",
                }
                report_item[PdFor2FeaturesExplainer._KEY_WORK_IMAGE_PATH] = (
                    self.persistence.get_explainer_working_file(
                        report_item[PdFor2FeaturesExplainer._KEY_IMAGE_FILE]
                    )
                )
                report_as_list.append(report_item)

                # tics
                x_tic_labels = []
                y_tic_labels = []
                z_dict = {}
                pd_frame = pd_explanations[pd_result].get(_pd.PD.LABEL_REGRESSION)
                for col_name in pd_frame.columns.values:
                    if col_name[0] not in x_tic_labels:
                        x_tic_labels.append(col_name[0])
                    if col_name[1] not in y_tic_labels:
                        y_tic_labels.append(col_name[1])
                    if col_name[0] not in z_dict.keys():
                        z_dict[col_name[0]] = []
                    z_dict[col_name[0]].append(pd_frame.iloc[0][col_name])

                x_tic_labels.sort()
                y_tic_labels.sort(reverse=True)

                # tics vs. labels
                x_tics = results.Data3dResult._plot_labels_to_tics(x_tic_labels)
                y_tics = results.Data3dResult._plot_labels_to_tics(y_tic_labels)
                # do NOT set x and y ticks - it will be set later
                z_tics = pandas.DataFrame(z_dict).to_numpy()

                report_item[PdFor2FeaturesExplainer._KEY_X_TICS_LABELS] = x_tic_labels
                report_item[PdFor2FeaturesExplainer._KEY_Y_TICS_LABELS] = y_tic_labels

                # data: from tics to matrices w/ the same shape as z-data
                z_data = z_tics
                x_data = numpy.array([x_tics for _ in range(z_data.shape[0])])
                y_data = numpy.array([y_tics for _ in range(z_data.shape[1])]).T

                z_body = pandas.DataFrame(
                    z_dict, columns=x_tic_labels, index=y_tic_labels
                )
                report_item[PdFor2FeaturesExplainer._KEY_DATA_DICT] = z_body.to_dict()
                report_item[PdFor2FeaturesExplainer._KEY_REPORT_BODY] = (
                    f"{tabulate.tabulate(z_body, headers='keys', tablefmt='github')}"
                )

                self.logger.debug(
                    f"\n  x.labels={x_tic_labels}"
                    f"\n  x.tics  ={x_tics}"
                    f"\n  y.labels={y_tic_labels}"
                    f"\n  y.tics  ={y_tics}"
                    f"\n  z.ticks ={z_tics}"
                    f"\n  x.data:\n{x_data}"
                    f"\n  y.data:\n{y_data}"
                    f"\n  z.dict:\n{z_dict}"
                )

                # plot
                plt.figure(figsize=(8, 6), dpi=80)
                if plot_type == plots.Data3dPlot.PLOT_TYPE_HEATMAP:
                    ax = plt.axes()
                else:
                    ax = plt.axes(projection="3d")

                ax.set_title("Partial Dependence for Two Features")

                # set plot bins count:
                #   required by Matplotlib 3.4.3 for even axis bins distribution
                plt.locator_params(axis="x", nbins=len(x_tic_labels))
                plt.locator_params(axis="y", nbins=len(y_tic_labels))
                # IMPROVE try x-tics and y-tics from above
                ax.set_xticklabels(x_tic_labels)
                ax.set_yticklabels(y_tic_labels)
                # axes legend
                ax.set_xlabel(str(pd_result[0]))
                ax.set_ylabel(str(pd_result[1]))
                # color map
                color_map = "autumn"

                if plot_type == plots.Data3dPlot.PLOT_TYPE_CONTOUR:
                    ax.contour3D(x_data, y_data, z_data, 150, cmap=color_map)
                elif plot_type == plots.Data3dPlot.PLOT_TYPE_HEATMAP:
                    plt.imshow(z_tics, cmap=color_map, aspect="auto")
                    plt.colorbar()
                elif plot_type == plots.Data3dPlot.PLOT_TYPE_SURFACE:
                    ax.plot_surface(
                        x_data, y_data, z_data, cmap=color_map, edgecolor="black"
                    )
                else:
                    self.logger.error(
                        f"{self.log_name}: unable to render unknown plot type "
                        f"'{plot_type}' - valid plot types are "
                        f"{plots.Data3dPlot.PLOT_TYPES}"
                    )

                plt.savefig(
                    report_item[PdFor2FeaturesExplainer._KEY_WORK_IMAGE_PATH], dpi=300
                )

                # clear figure to get ready for next plot
                plt.clf()
            except Exception as ex:
                self.logger.error(
                    f"{self.log_name}: rendering of {pd_result} failed "
                    f"with: {ex}\n{traceback.format_exc()}",
                )
                raise ex

        return report_as_list

    def _normalize_markdown(self, report_as_list: list):
        t_pd = PdFor2FeaturesExplainer

        global_explanation = e10s.ReportExplanation(
            explainer=self,
            display_name=t_pd._display_name,
            display_category=e10s.Explanation.DISPLAY_CAT_DAI_MODEL,
        )

        # MD format
        report_text: str = (
            "# Partial Dependence Plot for Two Features\n"
            "Partial dependence plot for two features shows the marginal effect "
            "features pairs have on the predicted outcome of a machine learning model. "
            "The partial dependence plot can show whether the relationship between "
            "the target and a feature tuple is linear, monotonic or more complex."
            "\n\n"
            "**Table of Contents**\n"
        )
        for i, section in enumerate(report_as_list):
            report_text = (
                f"{report_text}* **{section[t_pd._KEY_TITLE]}**\n"
                # f"* [{section['title']}](section_{i})\n" not supported by DAI MD
                #   renderer
            )

        images_work_paths = []
        for i, section in enumerate(report_as_list):
            report_text = (
                f"{report_text}"
                f"\n"
                # f"<a name='#section_{i}'>\n" ... unsupported by DAI MD renderer
                f"# {section[t_pd._KEY_TITLE]}\n"
                f"\nPartial dependence plot for features {section[t_pd._KEY_TITLE]}."
                f"![image_{i}](./{section[t_pd._KEY_IMAGE_FILE]})\n"
                f"\n"
                f"\nChart data:"
                f"\n"
                f"```\n"
                f"{section[t_pd._KEY_REPORT_BODY]}\n"
                f"```\n"
            )

            images_work_paths.append(
                section[PdFor2FeaturesExplainer._KEY_WORK_IMAGE_PATH]
            )

        report_path = self.persistence.get_explainer_working_file("report.md")
        with open(report_path, mode="w") as file:
            file.write(report_text)

        # add MD format (will copy report and image files from work)
        global_explanation.add_format(
            f5s.MarkdownFormat(
                explanation=global_explanation,
                format_file=report_path,
                extra_format_files=images_work_paths,
            )
        )

        return global_explanation

    def _normalize_html(self, report_as_list: list):
        t_pd = PdFor2FeaturesExplainer
        e_type = e10s.GlobalHtmlFragmentExplanation

        global_explanation = e10s.GlobalHtmlFragmentExplanation(
            explainer=self,
            display_name=PdFor2FeaturesExplainer._display_name,
            display_category=e10s.Explanation.DISPLAY_CAT_DAI_MODEL,
        )

        # HTML format
        html_src = airium.Airium()

        images_work_paths = []
        for i, section in enumerate(report_as_list):
            with html_src.b():
                html_src(
                    f"Partial dependence plot for features {section[t_pd._KEY_TITLE]}:"
                )
                html_format_file = self.persistence.get_relative_path(
                    self.persistence.get_explanation_file_path(
                        explanation_type=e_type.explanation_type(),
                        explanation_format=f5s.HtmlFormat.mime,
                        explanation_file=f"./{section[t_pd._KEY_IMAGE_FILE]}",
                    )
                )
                with html_src.div():
                    html_src.img(
                        src=html_format_file,
                        alt=(
                            f"Partial dependence plot for features "
                            f"{section[t_pd._KEY_TITLE]}"
                        ),
                        # ensure that image will not overflow enclosing <div/>
                        style=(
                            "height: 100%; max-width: 100%; display: block; "
                            "margin: auto;"
                        ),
                    )
                    html_src.br()

                    # SKIPPED: chart data might overflow page width
                    # html_src("Chart data:")
                    # html_src.br()
                    # with html_src.pre():
                    #     html_src(f"{section['body']}")
                    # html_src.br()

            images_work_paths.append(
                section[PdFor2FeaturesExplainer._KEY_WORK_IMAGE_PATH]
            )

        report_text = str(html_src)
        report_file_name = f"{self.persistence.FILE_EXPLANATION}.html"
        report_path = self.persistence.get_explainer_working_file(report_file_name)
        with open(report_path, mode="w") as file:
            file.write(report_text)

        # add HTML format (will copy report and image files from work)
        html_format = f5s.HtmlFormat(
            explanation=global_explanation,
            format_data="",  # data will be updated later
            format_file=os.path.join(
                self.persistence.get_explainer_working_dir(),
                report_file_name,
            ),
            extra_format_files=images_work_paths,
            persistence=self.persistence.store,
        )
        global_explanation.add_format(html_format)

        return global_explanation

    def _normalize_json_and_csv(self, report_as_list: list):
        t_pd = PdFor2FeaturesExplainer

        global_explanation = e10s.Global3dDataExplanation(
            explainer=self,
            display_name=PdFor2FeaturesExplainer._display_name,
            display_category=e10s.Explanation.DISPLAY_CAT_DAI_MODEL,
        )

        # JSon format
        features = []
        features_names = []
        datas_by_title = {}
        for item in report_as_list:
            features.append(item[t_pd._KEY_TITLE])
            features_names.append(item[t_pd._KEY_FEATURE_NAMES])
            datas_by_title[item[t_pd._KEY_TITLE]] = item[t_pd._KEY_DATA_DICT]
        classes = [
            (
                f5s.ExplanationFormat.LABEL_REGRESSION
                if self.model_meta.num_labels == 1
                else (
                    self.model_meta.labels[1] if self.model_meta.num_labels >= 2 else ""
                )
            )
        ]

        (idx_dict, idx_json_str) = f5s.Global3dDataJSonFormat.serialize_index_file(
            features=features,
            features_names=features_names,
            classes=classes,
            keywords=PdFor2FeaturesExplainer._keywords,
            doc=PdFor2FeaturesExplainer._description,
            y_file=(
                self.persistence.get_explainer_working_file(
                    PdFor2FeaturesExplainer.FILE_Y_HAT
                )
            ),
        )
        # safe data files
        for f in idx_dict[f5s.ExplanationFormat.KEY_FEATURES]:
            f_entry = idx_dict[f5s.ExplanationFormat.KEY_FEATURES][f]
            dir_path = self.persistence.get_explanation_dir_path(
                explanation_type=e10s.Global3dDataExplanation.explanation_type(),
                explanation_format=f5s.Global3dDataJSonFormat.mime,
            )
            self.persistence.store.make_dir(dir_path)
            file_path = self.persistence.get_explanation_file_path(
                explanation_type=e10s.Global3dDataExplanation.explanation_type(),
                explanation_format=f5s.Global3dDataJSonFormat.mime,
                explanation_file=list(
                    f_entry[f5s.ExplanationFormat.KEY_FILES].values()
                )[0],
            )
            self.persistence.store.save_json(
                key=file_path,
                data=datas_by_title[f],
            )

        json_format = f5s.Global3dDataJSonFormat(
            explanation=global_explanation,
            json_data=idx_json_str,
            persistence=self.persistence.store,
        )
        global_explanation.add_format(json_format)

        # JSon+CSV format
        (idx_dict, idx_json_str) = f5s.Global3dDataJSonCsvFormat.serialize_index_file(
            features=features,
            features_names=features_names,
            classes=classes,
            keywords=PdFor2FeaturesExplainer._keywords,
            doc=PdFor2FeaturesExplainer._description,
            y_file=(
                self.persistence.get_explainer_working_file(
                    PdFor2FeaturesExplainer.FILE_Y_HAT
                )
            ),
        )
        # safe data files
        for f in idx_dict[f5s.ExplanationFormat.KEY_FEATURES]:
            f_entry = idx_dict[f5s.ExplanationFormat.KEY_FEATURES][f]
            dir_path = self.persistence.get_explanation_dir_path(
                explanation_type=e10s.Global3dDataExplanation.explanation_type(),
                explanation_format=f5s.Global3dDataJSonCsvFormat.mime,
            )
            self.persistence.store.make_dir(dir_path)
            file_path = self.persistence.get_explanation_file_path(
                explanation_type=e10s.Global3dDataExplanation.explanation_type(),
                explanation_format=f5s.Global3dDataJSonCsvFormat.mime,
                explanation_file=list(
                    f_entry[f5s.ExplanationFormat.KEY_FILES].values()
                )[0],
            )
            self.persistence.store.save(
                key=file_path,
                data=pandas.DataFrame(datas_by_title[f]).to_csv(),
                data_type=persistences.PersistenceDataType.text,
            )

        csv_format = f5s.Global3dDataJSonCsvFormat(
            explanation=global_explanation,
            json_data=idx_json_str,
            persistence=self.persistence.store,
        )
        global_explanation.add_format(csv_format)

        return global_explanation

    def get_result(self) -> results.Data3dResult:
        return results.Data3dResult(
            persistence=self.persistence,
            h2o_sonar_config=self.config,
            explainer_id=PdFor2FeaturesExplainer.explainer_id(),
            logger=self.logger,
        )


class PdFor2FeaturesArgs(explainers.ExplainerArgs):
    def __init__(self, parameters: list[explainers.ExplainerParam] = None):
        explainers.ExplainerArgs.__init__(self, parameters)

    def resolve_params(
        self,
        explainer_params: dict | None = None,
        erase: list[str] | None = None,
    ) -> dict:
        erased_config_overrides = explainers.ExplainerArgs.resolve_params(
            self,
            explainer_params=explainer_params,
        )

        max_features = self.args.get(PdFor2FeaturesExplainer.PARAM_MAX_FEATURES)
        if (
            explainer_params
            and explainer_params.get(PdFor2FeaturesExplainer.PARAM_MAX_FEATURES)
            is not None
        ):
            max_features = int(
                explainer_params.get(PdFor2FeaturesExplainer.PARAM_MAX_FEATURES)
            )
        self.args[PdFor2FeaturesExplainer.PARAM_MAX_FEATURES] = max_features

        return erased_config_overrides
