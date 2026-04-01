# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import os

import airium
import datatable
from matplotlib import pyplot as plt

from h2o_sonar import errors
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api import plots
from h2o_sonar.lib.api import results
from h2o_sonar.methods import _shap
from h2o_sonar.utils import sanitization


GlobalSummaryFeatImpExplanation = e10s.GlobalSummaryFeatImpExplanation


class SummaryShapleyExplainer(explainers.Explainer):
    """Shapley summary plot for original features."""

    #
    # IMPLEMENTATION DETAILS
    #
    # Functional architecture
    #
    # - local explanations are not supported yet
    # - categorical features drill-down charts not supported yet (encoding needed)
    #
    # Performance:
    #
    # - creation of drill-down images make explainer run 10x slower (6s > 60s+),
    #   bottleneck is matplotlib.pyplot.savefig() which takes 80%+ of image
    #   creation
    # - perhaps bins by value are not needed (see code below)
    #

    _display_name = "Shapley Summary Plot for Original Features (Kernel SHAP Method)"
    _description = (
        "Shapley explanations are a technique with credible theoretical support that "
        "presents consistent global and local feature contributions."
        "\n\n"
        "The Shapley Summary Plot shows original features versus their local Shapley "
        "values on a sample of the dataset. Feature values are binned by Shapley "
        "values and the average normalized feature value for each bin is plotted. "
        "The legend corresponds to numeric features and maps to their normalized "
        "value - yellow is the lowest value and deep orange is the highest. You can "
        "also get a scatter plot of the actual numeric features values "
        "versus their corresponding Shapley values. Categorical features are shown in "
        "grey and do not provide an actual-value scatter plot."
        "\n\n"
        "Notes:"
        "\n\n"
        "* The Shapley Summary Plot only shows original features that are used in the "
        "model."
        "\n"
        "* The dataset sample size and the number of bins can be updated in the "
        "interpretation settings."
        "\n\n"
    )
    _iid = True
    _time_series = True
    _regression = True
    _binary = True
    _multiclass = True
    _global_explanation = True
    _explanation_types = [GlobalSummaryFeatImpExplanation]
    _optional_explanation_types = [e10s.GlobalHtmlFragmentExplanation]

    # option: drill-down to show cat feature values (not 0 ~ which is consistent)
    OPT_DRILL_CAT_VALUES = False
    # option: skip creation of drill-down scatter plots of categorical features
    OPT_DRILL_SKIP_CAT = True
    # use model's feature importance
    OPT_MODEL_FI = False

    PARAM_MAX_FEATURES = "max_features"
    PARAM_SAMPLE_SIZE = "sample_size"
    PARAM_X_RESOLUTION = "x_shapley_resolution"
    PARAM_DRILLDOWN_CHARTS = "enable_drilldown_charts"
    PARAM_FAST_APPROX = "fast_approx_contribs"

    DEFAULT_MAX_FEATURES = 50
    DEFAULT_SAMPLE_SIZE = 20000
    DEFAULT_X_RESOLUTION = 500

    COLOR_GRAY = "#cccccc"

    FILE_RAW_CONTRIBS_IDX = "raw_shapley_contribs_index.json"

    # calculation progress reporting range (engine 0 - 10%, normalization 90% - 100%)
    PROGRESS_MIN = 0.1
    PROGRESS_MAX = 0.9

    _parameters = [
        explainers.ExplainerParam(
            param_name=PARAM_MAX_FEATURES,
            description="Maximum number of features to be shown in the plot.",
            param_type=commons.ExplainerParamType.int,
            default_value=DEFAULT_MAX_FEATURES,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_SAMPLE_SIZE,
            description="Sample size.",
            param_type=commons.ExplainerParamType.int,
            default_value=DEFAULT_SAMPLE_SIZE,
            value_min=100,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_X_RESOLUTION,
            description="x-axis resolution (number of Shapley values bins).",
            param_type=commons.ExplainerParamType.int,
            default_value=DEFAULT_X_RESOLUTION,
            value_min=100,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_DRILLDOWN_CHARTS,
            description=(
                "Enable creation of per-feature Shapley/feature value scatter plots."
            ),
            param_type=commons.ExplainerParamType.bool,
            default_value=True,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_FAST_APPROX,
            description=(
                "Speed up predictions with fast predictions and contributions"
                " approximations."
            ),
            param_type=commons.ExplainerParamType.bool,
            default_value=True,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
    ]
    _keywords = [
        explainers.Explainer.KEYWORD_DEFAULT,
        explainers.Explainer.KEYWORD_EXPLAINS_FEATURE_BEHAVIOR,
        explainers.Explainer.KEYWORD_H2O_SONAR,
    ]
    _requires_preloaded_predictor = False

    def __init__(self):
        explainers.Explainer.__init__(self)

        self.args = None
        self.is_on_demand_pd: bool = False
        # sanitized stringified labels
        self.labels = None
        self.log_name = "Summary Shapley explainer"

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

        model_meta = model.meta if model else None

        if model_meta:
            # IMPROVE sort features by importance
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

        if not model:
            err_msg = f"{self._display_name} requires a model"
            self.logger.warning(err_msg)
            raise errors.ExplainerCompatibilityError(err_msg)

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

        self.args = SummaryShapleyArgs(self._parameters)
        self.args.resolve_params(
            explainer_params=SummaryShapleyArgs.json_str_to_dict(
                self.explainer_params_as_str
            ),
        )
        self.labels = list(map(str, self.model_meta.labels)) if model else None
        self.log_name: str = f"Summary Shapley explainer {self.mli_key}/{self.key}"

    def explain(
        self,
        X,
        y: datatable.Frame | None = None,
        explanations_types: list = None,
        **kwargs,
    ):
        """Create global (pre-computed/cached) explanations."""
        x_cols = list(X.names)
        if self.model_meta.target_col in x_cols:
            x_cols.remove(self.model_meta.target_col)
        X = X[:, x_cols]

        # explanations list
        explanations = list()

        # global explanations
        (
            global_explanation,
            features_per_page,
            classes,
            aux_raw_idx_dict,
        ) = self._explain_global_summary_shapley(X)
        explanations.append(global_explanation)

        # OPTIONAL explanation: HTML fragment
        if self.config and self.config.create_html_representations:
            try:
                fi_html = e10s.GlobalHtmlFragmentExplanation(
                    explainer=self,
                    display_name="Shapley Summary Plot for Original Features",
                    display_category=e10s.Explanation.DISPLAY_CAT_DAI_MODEL,
                )
                fi_html.add_format(self._explain_html(fi_html))

                explanations.append(fi_html)
            except Exception as ex:
                self.logger.warning(
                    f"{self.log_name}: HTML representation creation failed: {ex}"
                )

        return explanations

    def _explain_html(
        self,
        explanation: e10s.GlobalHtmlFragmentExplanation,
    ) -> formats.HtmlFormat:
        """Build HTML representation."""
        html_format = formats.HtmlFormat(
            explanation=explanation,
            format_data=formats.HtmlFormat.MINIMAL_HTML,
            persistence=self.persistence.store,
        )

        # resolve sanitized classes
        e_type = e10s.GlobalSummaryFeatImpExplanation.explanation_type()
        h_type = e10s.GlobalHtmlFragmentExplanation.explanation_type()
        idx_dict: dict = formats.GlobalSummaryFeatImpJsonFormat.load_index_file(
            persistence=self.persistence,
            explanation_type=e_type,
        )

        # classes and drill-down files mapping
        classes = []
        features = []
        if idx_dict:
            dt_idx_files = idx_dict.get(
                formats.GlobalSummaryFeatImpJsonFormat.KEY_FILES_DETAILS, None
            )
            if dt_idx_files:
                classes = list(dt_idx_files.keys())
            if classes:
                dt_idx_features = idx_dict[
                    formats.GlobalSummaryFeatImpJsonFormat.KEY_FILES_DETAILS
                ].get(classes[0], None)
                if dt_idx_features:
                    features = list(
                        idx_dict[
                            formats.GlobalSummaryFeatImpJsonFormat.KEY_FILES_DETAILS
                        ][classes[0]].keys()
                    )

        classes = classes or [None]

        html_src = airium.Airium()

        for i, clazz in enumerate(classes):
            with html_src.b():
                html_src(
                    f"Global Shapley values for original features of class '{clazz}':"
                )

            # copy global file(s)
            file_name = f"shapley-class-{i}.png"
            to_file_path = self.persistence.get_explanation_file_path(
                explanation_type=h_type,
                explanation_format=formats.HtmlFormat.mime,
                explanation_file=file_name,
            )
            self.persistence.store.copy_file(
                from_key=self.persistence.get_explainer_working_file(file_name),
                to_key=to_file_path,
            )
            with html_src.div():
                html_src.img(
                    src=self.persistence.get_relative_path(to_file_path),
                    # style="max-width:100%; max-height:100%;",
                    # ensure that image will not overflow enclosing <div/>
                    style=(
                        "height: 100%; max-width: 100%; display: block; margin: auto;"
                    ),
                    alt=f"Global Shapley values for class '{clazz}'",
                )
            with html_src.b():
                html_src(
                    f"Local Shapley values for original features of class '{clazz}':"
                )
            # copy drill-down files
            with html_src.div():
                for j, feature in enumerate(features):
                    file_name = idx_dict[
                        formats.GlobalSummaryFeatImpJsonFormat.KEY_FILES_DETAILS
                    ][classes[0]][feature]
                    to_file_path = self.persistence.get_explanation_file_path(
                        explanation_type=h_type,
                        explanation_format=formats.HtmlFormat.mime,
                        explanation_file=file_name,
                    )
                    src_mime = formats.GlobalSummaryFeatImpJsonFormat.mime
                    self.persistence.store.copy_file(
                        from_key=self.persistence.get_explanation_file_path(
                            explanation_type=e_type,
                            explanation_format=src_mime,
                            explanation_file=file_name,
                        ),
                        to_key=to_file_path,
                    )

                    html_src.img(
                        src=self.persistence.get_relative_path(to_file_path),
                        # style="max-width:100%; max-height:100%;",
                        # ensure that image will not overflow enclosing <div/>
                        style=(
                            "height: 100%; max-width: 100%; display: block; "
                            "margin: auto;"
                        ),
                        alt=(
                            f"Local Shapley values for class {clazz} and feature "
                            f"{feature}"
                        ),
                    )
                    html_src.br()

        html_format.update_data(
            str(html_src), f"{persistences.ExplainerPersistence.FILE_EXPLANATION}.html"
        )

        return html_format

    def _explain_global_summary_shapley(self, X):
        """Create global summary feature importance explanation with JSon (index file)
        and datatable (data file) format representation. This representation is
        supported by Grammar of MLI and will be rendered in UI.

        """
        # prepare parameters
        arg_max_features = self.args.get(self.PARAM_MAX_FEATURES)
        arg_max_features = (
            SummaryShapleyExplainer.DEFAULT_MAX_FEATURES
            if arg_max_features is None or int(arg_max_features) < 0
            else int(arg_max_features)
        )

        # Summary Shap is not needed per class and multinomial case is handled in
        # implementation. However, keeping use of class for simplicity.
        if len(self.labels) <= 1:
            labels = [formats.GlobalSummaryFeatImpJsonFormat.LABEL_REGRESSION]
        elif len(self.labels) == 2:
            # bin experiment's MOJO has class of interest @ index 1 (2nd class)
            labels = [self.labels[1]]
        else:
            labels = self.labels.copy()

        # calculate RAW data
        if len(labels) <= 2:
            (
                shapley_means_dict,
                shapley_contribs_dict,
                # sampled and sanitized dataset
                dataset,
            ) = self._get_regression_or_binomial_shapley(
                dataset=X,
                label=str(labels[0]),
                max_features=arg_max_features,
            )
        else:
            (
                shapley_means_dict,
                shapley_contribs_dict,
                # sampled and sanitized dataset
                dataset,
            ) = self._get_multinomial_shapley(
                dataset=X, labels=labels, max_features=arg_max_features
            )
        self.logger.info(
            f"{self.log_name} raw MEANs ({len(shapley_means_dict)})",
        )
        self.logger.info(
            f"{self.log_name} raw CONTRIBs ({len(shapley_contribs_dict)})",
        )

        # RAW Shapley contribs (frames stored to work/ via transient index file)
        self.report_progress(0.3)
        (
            raw_shapley_contribs_idx_dict,
            _,
        ) = formats.GlobalFeatImpJSonDatatableFormat.serialize_index_file(
            classes=labels,
            doc=SummaryShapleyExplainer._description,
            data_file_prefix="raw_shapley_contribs",
        )
        self.persistence.save_json(
            data=raw_shapley_contribs_idx_dict,
            path=self.persistence.get_explainer_working_file(
                SummaryShapleyExplainer.FILE_RAW_CONTRIBS_IDX
            ),
        )
        # IMPROVE these files are currently not used, but potentially could be in UI
        (_, global_bias) = self.get_gom_from_shapley_contrib_frames(
            shapley_contribs_dict=shapley_contribs_dict,
            file_mapping_dict=raw_shapley_contribs_idx_dict,
        )

        # global explanation
        self.report_progress(0.4)
        global_explanation = GlobalSummaryFeatImpExplanation(
            explainer=self,
            # UI tile name
            display_name="Shapley Summary Plot for Original Features",
            # UI tab name
            display_category=GlobalSummaryFeatImpExplanation.DISPLAY_CAT_DAI_MODEL,
        )

        # features
        features = []
        a_means_key = next(iter(shapley_means_dict))
        num_features = shapley_means_dict[a_means_key].shape[1]

        # JSon+datatable explanation representation
        (
            idx_file_dict,
            idx_file_str,
        ) = formats.GlobalSummaryFeatImpJsonDatatableFormat.serialize_index_file(
            classes=labels,
            metrics=[{"Global bias": global_bias}],
            doc=SummaryShapleyExplainer._description,
            total_rows=num_features,
        )
        json_dt_representation = formats.GlobalSummaryFeatImpJsonDatatableFormat(
            explanation=global_explanation,
            json_data=idx_file_str,
            persistence=self.persistence.store,
        )
        # add format files: per-class (saved as added to format)
        progress_start = 0.45
        progress_end = 0.96
        self.report_progress(progress_start)
        progress_cls_inc = 0
        chart_to_path: dict = {}
        for i, clazz in enumerate(labels):
            features = list(list(shapley_means_dict.values())[0].names)
            features = [
                f for f in features if datasets.ExplainableDataset.COL_BIAS not in f
            ]
            if len(features) > arg_max_features:
                features = features[:arg_max_features]

            # NORMALIZE contrib frames to GoM summary FI
            (
                format_data,
                chart_to_filename_for_class,
            ) = self.normalize_shapleys_to_dt_gom(
                dataset=dataset,
                index_dict=raw_shapley_contribs_idx_dict,
                clazz=clazz,
                class_offset=i,
                bin_resolution=self.args.get(
                    self.PARAM_X_RESOLUTION,
                    SummaryShapleyExplainer.DEFAULT_X_RESOLUTION,
                ),
                progress_start=progress_start,
                progress_end=progress_end,
            )
            progress_start = progress_end
            progress_end += progress_cls_inc
            chart_to_path[clazz] = chart_to_filename_for_class
            json_dt_representation.add_data_frame(
                format_data=format_data,
                file_name=idx_file_dict[formats.ExplanationFormat.KEY_FILES][clazz],
            )
        global_explanation.add_format(json_dt_representation)

        # JSon explanation representation
        self.report_progress(0.97)
        (
            json_representation,
            features_per_page,
        ) = formats.GlobalSummaryFeatImpJsonFormat.from_json_datatable(
            json_dt_format=json_dt_representation,
            page_size=formats.GlobalSummaryFeatImpJsonFormat.DEFAULT_PAGE_SIZE,
            total_rows=len(features),
            index_extensions={
                formats.ExplanationFormat.KEY_FILES_DETAILS: chart_to_path
            },
        )
        global_explanation.add_format(json_representation)

        # Markdown report with image(s) representation
        (report_path, images_path) = self._create_md_report(
            dataset=dataset,
            raw_shapley_contribs_idx=raw_shapley_contribs_idx_dict,
            max_features=arg_max_features,
        )
        global_explanation.add_format(
            formats.MarkdownFormat(
                explanation=global_explanation,
                format_file=report_path,
                extra_format_files=images_path,
            )
        )

        return (
            global_explanation,
            features_per_page,
            labels,
            raw_shapley_contribs_idx_dict,
        )

    def _explain_local_summary_shapley(
        self, classes: list, features_per_page: dict
    ) -> e10s.LocalSummaryFeatImpExplanation:
        """Create on-demand local feature importance explanation - to be created
        by scorer providing both prediction and Shapley values.  This representation
        is supported by Grammar of MLI and will
        be rendered in UI.

        """
        raise NotImplementedError(
            "Shapley summary plot local explanations are not supported"
        )

    #
    # raw data
    #

    def _get_regression_or_binomial_shapley(
        self, dataset, label: str, max_features: int
    ):
        shapley_means_dict = {}
        shapley_contribs_dict = {}

        dataset = sanitization.sanitize_frame(dataset, self.model_meta.sanitization_map)
        if self.model.model_type is models.ExplainableModelType.driverless_ai:
            # subset frame to features used by the model
            dataset = dataset[:, self.model.meta.used_features]

        shap = _shap.Shap(explainable_model=self.model)
        shapley_contribs_dataset = shap.explain(
            X=dataset,
            experiment_type=(
                commons.ExperimentType.regression.name if not self.labels else ""
            ),
        )

        shapley_means = shapley_contribs_dataset[
            :,
            {
                i: (
                    datatable.f[i]
                    if i == datasets.ExplainableDataset.COL_BIAS
                    else datatable.abs(datatable.f[i])
                )
                for i in shapley_contribs_dataset.names
            },
        ].mean()

        neworder = sorted(
            range(shapley_means.ncols),
            key=lambda i: shapley_means[0, i],
            reverse=True,
        )
        ordered_features = neworder[:max_features]

        shapley_means = shapley_means[:, ordered_features]
        shapley_contribs_dataset = shapley_contribs_dataset[:, ordered_features]

        shapley_means_dict[label] = shapley_means
        shapley_contribs_dict[label] = shapley_contribs_dataset

        return shapley_means_dict, shapley_contribs_dict, dataset

    PREFIX_CONTRIB = "contrib_"

    def _get_per_class_orig_features_names(self, is_mojo: bool = False):
        prefix = SummaryShapleyExplainer.PREFIX_CONTRIB if is_mojo else ""
        orig_feats_contrib_names = [
            f"{prefix}{n}" for n in self.model_meta.used_features
        ]
        orig_feats_contrib_names.append(
            f"{prefix}{datasets.ExplainableDataset.COL_BIAS}"
        )

        per_class_orig_feats_names = dict()
        for clazz in self.labels:
            class_names = []
            for label in orig_feats_contrib_names:
                class_names.append(f"{label}.{clazz}")
            per_class_orig_feats_names[clazz] = class_names

        return per_class_orig_feats_names

    def _get_multinomial_shapley(self, dataset, labels: list[str], max_features: int):
        dataset = sanitization.sanitize_frame(dataset, self.model_meta.sanitization_map)
        if self.model.model_type is models.ExplainableModelType.driverless_ai:
            # subset frame to features used by the model
            dataset = dataset[:, self.model.meta.used_features]

        shapley_means_dict = {}
        shapley_contribs_dict = {}
        shapley_df_mult_orig_feat = datatable.Frame()

        # SHAP returns contributions of ALL classes (column suffixes)
        shap = _shap.Shap(
            explainable_model=self.model,
        )
        orig_feat_shap_contribs_all = shap.explain(dataset)

        per_class_orig_feats_names = self._get_per_class_orig_features_names()

        for label in labels:
            # keep only columns relevant for given label (class)
            orig_feat_contribs_label = orig_feat_shap_contribs_all[
                :, per_class_orig_feats_names[label]
            ]

            # filter down to features with importance > 0
            shapley_contribs_dataset = datasets.filter_importance_greater_than_zero(
                frame=orig_feat_contribs_label, label=label, skip_bias=False
            )
            # drop
            drop_cols = self.params.drop_cols or []
            features = [x for x in shapley_contribs_dataset.names if x not in drop_cols]
            shapley_contribs_dataset = shapley_contribs_dataset[:, features]

            shapley_df_mult_orig_feat.cbind(shapley_contribs_dataset)

            # format names
            shapley_contribs_dataset.names = [
                c[: -len(f".{label}")] if c.endswith(f".{label}") else c
                for c in shapley_contribs_dataset.names
            ]

            shapley_means = shapley_contribs_dataset[
                :,
                {
                    i: (
                        datatable.f[i]
                        if i == datasets.ExplainableDataset.COL_BIAS
                        else datatable.abs(datatable.f[i])
                    )
                    for i in shapley_contribs_dataset.names
                },
            ].mean()

            neworder = sorted(
                range(shapley_means.ncols),
                key=lambda i: shapley_means[0, i],
                reverse=True,
            )
            ordered_features = neworder[:max_features]

            shapley_means = shapley_means[:, ordered_features]
            shapley_contribs_dataset = shapley_contribs_dataset[:, ordered_features]

            shapley_means_dict[str(label)] = shapley_means
            shapley_contribs_dict[str(label)] = shapley_contribs_dataset

        return shapley_means_dict, shapley_contribs_dict, dataset

    def get_gom_from_shapley_frames(
        self,
        shapley_means_dict: dict[str, datatable.Frame],
        file_mapping_dict: dict[str, str],
    ) -> list[str]:
        """This method is not used by Summary Shapley feature importance
        explainer - its purpose is to be library method for other explainers.

        """
        raw_file_names = []

        for label, shapley_means in shapley_means_dict.items():
            gom_data_frame: datatable.Frame = datatable.Frame(
                {
                    formats.GlobalFeatImpJSonDatatableFormat.COL_NAME: [],
                    formats.GlobalFeatImpJSonDatatableFormat.COL_IMPORTANCE: [],
                    formats.GlobalFeatImpJSonDatatableFormat.COL_GLOBAL_SCOPE: [],
                }
            )
            for name in shapley_means.names:
                gom_data_frame.rbind(
                    datatable.Frame(
                        {
                            formats.GlobalFeatImpJSonDatatableFormat.COL_NAME: [name],
                            formats.GlobalFeatImpJSonDatatableFormat.COL_IMPORTANCE: [
                                shapley_means[0, name]
                            ],
                            formats.GlobalFeatImpJSonDatatableFormat.COL_GLOBAL_SCOPE: [
                                True
                            ],
                        }
                    )
                )

            # leverage file names from the transient index file and store them to work/
            file_name: str = file_mapping_dict[
                formats.GlobalFeatImpJSonDatatableFormat.KEY_FILES
            ][label]
            raw_file_names.append(file_name)

            # persist frame
            gom_data_frame.to_jay(
                self.persistence.get_explainer_working_file(file_name)
            )

        return raw_file_names

    def get_gom_from_shapley_contrib_frames(
        self,
        shapley_contribs_dict: dict[str, datatable.Frame],
        file_mapping_dict: dict[str, str],
    ) -> tuple[list[str], float]:
        self.logger.debug(
            "Summary Shapley contrib frames normalization:"
            f"\n  contribs={shapley_contribs_dict.keys()}"
            f"\n  files={file_mapping_dict}"
        )

        raw_file_names = []
        global_bias = None
        for label, shapley_contribs in shapley_contribs_dict.items():
            # leverage file names from the transient index file and store them to work/
            file_name: str = file_mapping_dict[
                formats.LocalSummaryFeatImplJSonFormat.KEY_FILES
            ][label]
            raw_file_names.append(file_name)

            if (
                global_bias is None
                and datasets.ExplainableDataset.COL_BIAS in shapley_contribs.names
                and shapley_contribs.shape[0]
            ):
                global_bias = shapley_contribs[
                    0:1, datasets.ExplainableDataset.COL_BIAS
                ].to_list()[0][0]

            # persist frame
            self.logger.debug(
                f"{self.log_name}: saving RAW contribs "
                f"for {label} to {file_name} with GLOBAL bias {global_bias}",
            )
            shapley_contribs.to_jay(
                self.persistence.get_explainer_working_file(file_name)
            )

        return raw_file_names, global_bias

    #
    # normalization
    #

    @staticmethod
    def _get_bare_feature_for_dataset(dataset: datatable.Frame, feature: str) -> str:
        """Robust method to determine whether the feature is dataset column."""
        if feature in dataset.names:
            return feature

        if "." in feature:
            feature_no_class = feature[0 : feature.index(".")]
            if feature_no_class in dataset.names:
                return feature_no_class
            else:
                raise errors.MliError(
                    f"Unable to find feature '{feature}' among dataset columns as "
                    f"neither it's name (nor bare name '{feature_no_class}') is "
                    f"present in the dataset:  {dataset.names}"
                )

        raise errors.MliError(
            f"Unable to find feature '{feature}' among the dataset columns: "
            f"{dataset.names}"
        )

    @staticmethod
    def _is_feature_numeric(dataset: datatable.Frame, feature: str) -> bool:
        """Robust method to determine whether the feature is numeric."""
        if feature in dataset.names:
            return dataset.ltypes[dataset.colindex(feature)] in [
                datatable.ltype.int,
                datatable.ltype.real,
            ]

        # hand multinomial of specific models (like MOJO)
        try:
            return dataset.ltypes[
                dataset.colindex(
                    SummaryShapleyExplainer._get_bare_feature_for_dataset(
                        dataset=dataset, feature=feature
                    )
                )
            ] in [datatable.ltype.int, datatable.ltype.real]
        except errors.MliError as ex:
            raise errors.MliError(
                f"Unable to determine feature '{feature}' type (categorical vs. "
                f"numeric) as neither it's name (nor bare name) is not among the "
                f"dataset columns: {dataset.names} ({ex})"
            )

    def normalize_shapleys_to_dt_gom(
        self,
        dataset,
        index_dict: dict,
        clazz: str,
        class_offset: int,
        bin_resolution: int = 500,
        progress_start: float = 0.45,
        progress_end: float = 0.96,
    ) -> tuple[datatable.Frame, dict]:
        """Get chart data for (sampled) dataset.

        Parameters
        ----------
        dataset : datatable.Frame
          Training dataset.
        index_dict : dict
          Representation index file with per-class Shapley contribs file names.
        clazz : str
          Class for which to calculate explanations.
        class_offset : int
          Class "number" alias.
        bin_resolution :
          Bin resolution.
        progress_start : float
          Method progress % range begin.
        progress_end : float
          Method progress % range end.

        Returns
        -------
        Tuple[datatable.Frame, dict]:
          GoM chart frame and map of feature names to chart (details scatter plot)
          file-system paths.

        """
        format_const = formats.GlobalSummaryFeatImpJsonDatatableFormat
        data_file_name = index_dict[format_const.KEY_FILES][clazz]
        data_file_path = self.persistence.get_explainer_working_file(data_file_name)
        progress = progress_start

        # IMPROVE keep bias once FE will be able to visualize it
        contribs = datatable.fread(data_file_path)
        if datasets.ExplainableDataset.COL_BIAS in contribs.names:
            del contribs[:, datasets.ExplainableDataset.COL_BIAS]
        contributing_features = contribs.names

        progress_divider = float(len(contributing_features))
        progress_increment = (
            (progress_end - progress_start) / progress_divider
            if progress_divider > 0
            else 0
        )

        # contribs frame min and max
        contribs_min = min([i[0] for i in contribs.min().to_list()])
        contribs_max = max([i[0] for i in contribs.max().to_list()])

        (
            bins,
            bin_ticks,
            bin_size,
        ) = SummaryShapleyExplainer._create_bins(
            contribs_min=contribs_min,
            contribs_max=contribs_max,
            bin_resolution=bin_resolution,
        )

        result_df = datatable.Frame(
            {
                format_const.KEY_FEATURE: [],
                format_const.KEY_SHAPLEY: [],
                format_const.KEY_FREQUENCY: [],
                format_const.KEY_HIGH_VALUE: [],
                format_const.KEY_ORDER: [],
            }
        )
        feature_to_chart_filename: dict = {}
        for i, feature in enumerate(contributing_features):
            if datasets.ExplainableDataset.is_bias_col(feature):
                continue

            try:
                (
                    col_frequency,
                    col_avg_height,
                    chart_filename,
                ) = self._get_chart_data_for_col(
                    feature=feature,
                    feature_offset=i,
                    class_offset=class_offset,
                    dataset=dataset,
                    contribs=contribs,
                    bin_resolution=bin_resolution,
                    bin_size=bin_size,
                    bins=bins,
                    is_num=SummaryShapleyExplainer._is_feature_numeric(
                        dataset=dataset, feature=feature
                    ),
                )
                feature_to_chart_filename[feature] = chart_filename

                # datatable frame for feature
                feature_df = datatable.Frame(
                    {
                        format_const.KEY_FEATURE: [
                            feature for _ in range(bin_resolution)
                        ],
                        format_const.KEY_SHAPLEY: bin_ticks,
                        format_const.KEY_FREQUENCY: col_frequency,
                        format_const.KEY_HIGH_VALUE: col_avg_height,
                        format_const.KEY_ORDER: [i for _ in range(bin_resolution)],
                    }
                )

                # append feature frame
                result_df.rbind(feature_df)

                # task manager progress reporting
                progress += progress_increment
                self.report_progress(progress)
            except errors.RenderingError:
                # fail over: problematic feature will be just missing in the chart
                continue
            except Exception as ex:
                raise ValueError(ex)

        return result_df, feature_to_chart_filename

    @staticmethod
    def _create_bins(contribs_min, contribs_max, bin_resolution):
        # IMPROVE: bins (list) is not perhaps needed - 1st bin lower bound (bin_min) +
        # bin size should be sufficient

        # values binning: bin_low <= X < bin_high
        if 0 == contribs_min or 0 == contribs_max or contribs_min < 0 < contribs_max:
            if contribs_min == 0 == contribs_max:
                contribs_min = -0.1
                contribs_max = 0.1

            # ensure: 0 is bin & values are symmetric around 0 & (lower bound of a) bin
            aux_bin_size = (contribs_max - contribs_min) / bin_resolution
            bin_size = (contribs_max - contribs_min + aux_bin_size * 2) / bin_resolution

            # <>
            # ensure max / min is included (at least 2 bins)
            num_n_bins = int(abs(contribs_min) / bin_size)
            num_n_bins = num_n_bins + 1 if contribs_min else num_n_bins
            num_p_bins = bin_resolution - num_n_bins
            bin_ticks = [0 - bin_size * i for i in range(num_n_bins, 0, -1)]
            for i in range(num_p_bins):
                bin_ticks.append(0 + bin_size * i)
            bins = [b - bin_size / 2 for b in bin_ticks]
            bins.append(bins[-1] + bin_size)
        else:
            # bins @ contribs: ensure MAX value is included in bin (by adding epsilon)
            bin_size = ((contribs_max + 0.01) - contribs_min) / bin_resolution
            # bins + 1 due split points vs. bins
            bins = [contribs_min + i * bin_size for i in range(bin_resolution + 1)]
            # bin tick in the middle of the bin range
            bin_ticks = [b + bin_size / 2 for b in bins[:-1]]

        return bins, bin_ticks, bin_size

    def _get_chart_data_for_col(
        self,
        feature: str,
        feature_offset: int,
        class_offset: int,
        dataset: datatable.Frame,
        contribs: datatable.Frame,
        bin_resolution: int,
        bin_size: float,
        bins: list,
        is_num: bool,
    ):
        # IMPROVE find faster calculation (numpy/vectorization/...)
        feature_no_class = SummaryShapleyExplainer._get_bare_feature_for_dataset(
            dataset=dataset,
            feature=feature,
        )
        col_data = dataset[:, feature_no_class].to_list()[0]
        col_contribs = contribs[:, feature].to_list()[0]

        # NORMALIZE feature values as percentage in feature's range
        if is_num:
            # - N/As: N/A feature value rows deleted from Shap & data ~ N/A not number
            na_rows = []
            for row, v in enumerate(col_data):
                if v is None:
                    na_rows.append(row)
            if na_rows:
                for row in reversed(na_rows):
                    del col_data[row]
                    del col_contribs[row]
            if not col_data and not col_contribs:
                # there are NO non-N/A data
                raise ValueError(
                    f"{self.log_name} feature '{feature}' will be skipped as it has "
                    f"no non-N/A rows"
                )
            min_ = min(col_data)
            data_ = (
                [d - min_ for d in col_data]
                if min_ > 0
                else [d + abs(min_) for d in col_data]
            )

            pct_ = max(data_) / 100.0
            col_normalized_data = [
                d / pct_ / 100.0 if pct_ > 0.0 else 0.0 for d in data_
            ]
        else:
            col_normalized_data = []

        # feature value aware hashing of contribs to bins
        bins_min = min(bins)
        col_frequency = [0 for _ in range(bin_resolution)]
        col_height_sum = [0 for _ in range(bin_resolution)]
        for row, c in enumerate(col_contribs):
            bin_idx = int((c - bins_min) / bin_size)
            col_frequency[bin_idx] += 1
            if is_num:
                col_height_sum[bin_idx] += col_normalized_data[row]

        if is_num:
            # calculate COLOR as average of normalized feature values within bin
            col_avg_height = [0 for _ in range(bin_resolution)]
            for i in range(len(col_frequency)):
                col_avg_height[i] = (
                    col_height_sum[i] / col_frequency[i] if col_frequency[i] > 0 else 0
                )
        else:
            col_avg_height = [None for _ in range(bin_resolution)]

        # DRILL-DOWN image
        chart_path: str = ""
        if self.args.get(self.PARAM_DRILLDOWN_CHARTS, False) and (
            is_num or not SummaryShapleyExplainer.OPT_DRILL_SKIP_CAT
        ):
            try:
                format_dir = self.persistence.get_explanation_dir_path(
                    explanation_type=GlobalSummaryFeatImpExplanation.explanation_type(),
                    explanation_format=formats.GlobalSummaryFeatImpJsonFormat.mime,
                )
                # drill-down images are created upfront
                self.persistence.make_dir(format_dir)

                chart_path = SummaryShapleyExplainer._get_scatter_path(
                    format_dir=format_dir,
                    feature_offset=feature_offset,
                    class_offset=class_offset,
                )

                SummaryShapleyExplainer._save_scatter_for_feature(
                    chart_path=chart_path,
                    data_x_shapley=col_contribs,
                    fg_color=commons.LookAndFeel.get_line_color(
                        self.config.look_and_feel
                    ),
                    bg_color=commons.LookAndFeel.get_bg_color(
                        self.config.look_and_feel
                    ),
                    edge_color=(
                        commons.LookAndFeel.get_fg_color(self.config.look_and_feel)
                        if is_num
                        else SummaryShapleyExplainer.COLOR_GRAY
                    ),
                    data_y_feature_value=(
                        col_data
                        if is_num or SummaryShapleyExplainer.OPT_DRILL_CAT_VALUES
                        else [0] * len(col_data)
                    ),
                    y_label=feature,
                )
            except Exception as ex:
                raise errors.RenderingError(str(ex))
                # even w/o drill-down chart explanation still provides enough value
        else:
            raise errors.RenderingError(
                f"Drill down chart will not be built for the feature '{feature}' "
                f"as it's not numeric or creation is disabled"
            )

        return col_frequency, col_avg_height, os.path.basename(chart_path)

    #
    # raster (non-Vega) main chart @ Markdown report
    #

    MARKDOWN_TEMPLATE: str = """# Summary Shapley Feature Importance Report
This summary Shapley feature importance explainer report.

{}

    """

    @staticmethod
    def _filter_contribs_no_classes(
        contributions_frame: datatable.Frame,
        data_names: list[str],
    ) -> datatable.Frame:
        """Filter dataset columns while ignoring classes of multinomial predictions."""
        data_feature_matches = [c for c in data_names if c in contributions_frame.names]
        if len(data_names) == len(data_feature_matches):
            return contributions_frame[:, list(data_names)]

        # pick representative for each feature regardless class
        representatives = []
        for f in data_names:
            found = False
            for c in contributions_frame.names:
                if c.startswith(f):
                    representatives.append(c)
                    found = True
                    break
            if not found:
                raise errors.MliError(
                    f"Unable to map feature {f} to a (class) extended contributions"
                    f"frame name: {contributions_frame.names}"
                )

        contributions = contributions_frame[:, representatives]
        contributions.names = data_names
        return contributions

    @staticmethod
    def _get_bias_columns(contributions) -> list:
        result = []
        for c in contributions.names:
            if datasets.ExplainableDataset.is_bias_col(c):
                result.append(c)
        return result

    def _create_md_report(
        self,
        dataset: datatable.Frame,
        raw_shapley_contribs_idx: dict,
        max_features=None,
    ) -> tuple[str, list[str]]:
        # per-class index image
        image_file_names = []
        image_paths = []
        md_fragment = ""
        for i, clazz in enumerate(
            raw_shapley_contribs_idx[formats.ExplanationFormat.KEY_FILES].keys()
        ):
            contribs_file = raw_shapley_contribs_idx[
                formats.ExplanationFormat.KEY_FILES
            ].get(clazz, "")
            if contribs_file:
                # image paths
                img_file_name = f"shapley-class-{i}.png"
                md_fragment = f"{md_fragment}![image](./{img_file_name})\n"
                image_file_names.append(img_file_name)
                work_img_path = self.persistence.get_explainer_working_file(
                    img_file_name
                )
                image_paths.append(work_img_path)

                contributions = datatable.fread(
                    self.persistence.get_explainer_working_file(contribs_file)
                )
                bias_cols = SummaryShapleyExplainer._get_bias_columns(contributions)
                if bias_cols:
                    del contributions[:, bias_cols]
                data = dataset[:, contributions.names]

                plot = plots.ScatterFeatImpPlot.plot(
                    frame=data.to_pandas(),
                    contributions=contributions.to_pandas(),
                    colormap=commons.LookAndFeel.get_colormap(
                        theme=self.config.look_and_feel if self.config else None
                    ),
                )
                # ALTERNATIVE: plt.savefig(work_img_path, dpi=300)
                plot.savefig(fname=work_img_path, dpi=300)

        # save report
        report_path = self.persistence.get_explainer_working_file("report.md")
        with open(report_path, mode="w") as file:
            file.write(SummaryShapleyExplainer.MARKDOWN_TEMPLATE.format(md_fragment))

        return report_path, image_paths

    #
    # drill-down
    #

    @staticmethod
    def _get_scatter_path(format_dir: str, feature_offset, class_offset: int):
        return os.path.join(
            format_dir, f"feature_{feature_offset}_class_{class_offset}.png"
        )

    @staticmethod
    def _save_scatter_for_feature(
        chart_path: str,
        data_x_shapley: list,
        data_y_feature_value: list,
        y_label: str,
        x_label: str = "Shapley value",
        fg_color: str = "white",
        bg_color: str = "black",
        edge_color: str = commons.LookAndFeel.COLOR_DAI_GREEN,
        chart_title: str = "",
        x_size: int = 8,
        y_size: int = 16,
        circle_size: int = 80,
        dpi: int = 100,
        alpha: float = 1.0,
    ):
        """Save scatter plot chart for given feature."""

        plt.clf()
        if bg_color == "black":
            plt.style.use("dark_background")

        fig = plt.figure(figsize=(y_size, x_size))
        # axes bg color
        fig.patch.set_facecolor(bg_color)

        ax = fig.add_subplot(111)
        ax.set_title(chart_title, color=fg_color)
        # axes ticks fg color
        ax.tick_params(axis="x", colors=fg_color)
        ax.tick_params(axis="y", colors=fg_color)
        ax.set_xlabel(x_label, color=fg_color)
        ax.set_ylabel(y_label, color=fg_color)
        # axes fg color
        ax.spines["bottom"].set_color(fg_color)
        ax.spines["left"].set_color(fg_color)
        ax.spines["top"].set_color(bg_color)
        ax.spines["right"].set_color(bg_color)
        # style plt.rcParams[*] are ignored in DAI -> args must be used instead
        plt.grid(True, color="#777777", linestyle="dashed")
        # chart bg color
        plt.rcParams["axes.facecolor"] = bg_color
        # chart fg color
        plt.scatter(
            data_x_shapley,
            data_y_feature_value,
            s=circle_size,
            facecolors="none",
            edgecolors=edge_color,
            alpha=alpha,
        )

        # IMPROVE: chart save is performance bottleneck: 10x slower than its creation
        plt.savefig(chart_path, dpi=dpi)

    def get_result(self) -> results.SummaryShapResult:
        return results.SummaryShapResult(
            persistence=self.persistence,
            raw_contribs_idx_filename=SummaryShapleyExplainer.FILE_RAW_CONTRIBS_IDX,
            explainer_id=SummaryShapleyExplainer.explainer_id(),
            h2o_sonar_config=self.config,
            logger=self.logger,
        )


class SummaryShapleyArgs(explainers.ExplainerArgs):
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

        max_features = self.args.get(SummaryShapleyExplainer.PARAM_MAX_FEATURES)
        if (
            explainer_params
            and explainer_params.get(SummaryShapleyExplainer.PARAM_MAX_FEATURES)
            is not None
        ):
            max_features = int(
                explainer_params.get(SummaryShapleyExplainer.PARAM_MAX_FEATURES)
            )
        self.args[SummaryShapleyExplainer.PARAM_MAX_FEATURES] = max_features

        return erased_config_overrides
