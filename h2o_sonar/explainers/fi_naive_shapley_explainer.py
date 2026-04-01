# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import json
import os

import datatable

from h2o_sonar import errors
from h2o_sonar.explainers.commons import explainers_commons
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api import problems
from h2o_sonar.lib.api import results
from h2o_sonar.utils import problem_detection
from h2o_sonar.utils import sampling
from h2o_sonar.utils import sanitization


class NaiveShapleyMojoFeatureImportanceExplainer(
    explainers.Explainer, explainers_commons.AbstractFeatureImportanceExplainer
):
    PARAM_SAMPLE_SIZE = "sample_size"
    PARAM_FAST_APPROX = "fast_approx_contribs"
    DEFAULT_FAST_APPROX = False
    PARAM_LEAKAGE_WARN_THRESHOLD = "leakage_warning_threshold"

    _display_name = "Shapley Values for Original Features of MOJO Models (Naive Method)"
    _description = (
        "Shapley values for original features of (Driverless AI) MOJO models are "
        "approximated from the accompanying Shapley values for transformed features "
        "with the Naive Shapley method. This method makes the assumption that input "
        "features to a transformer are independent. For example, if the transformed "
        "feature, feature1_feature2, has a Shapley value of 0.5, then the "
        "Shapley value of the original features feature1 and feature2 will "
        "be 0.25 each."
    )
    _iid = True
    _regression = True
    _binary = True
    _multiclass = True
    _global_explanation = True
    _local_explanation = True
    _requires_preloaded_predictor = True
    _explanation_types = [e10s.GlobalFeatImpExplanation, e10s.LocalFeatImpExplanation]
    _optional_explanation_types = [e10s.GlobalHtmlFragmentExplanation]
    _parameters = [
        explainers.ExplainerParam(
            param_name=PARAM_SAMPLE_SIZE,
            description="Sample size.",
            param_type=commons.ExplainerParamType.int,
            default_value=100_000,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_FAST_APPROX,
            description=(
                "Speed up predictions with fast contributions predictions "
                "approximation."
            ),
            param_type=commons.ExplainerParamType.bool,
            default_value=True,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_LEAKAGE_WARN_THRESHOLD,
            description=(
                "The threshold above which to report a potentially detected "
                "feature importance leak problem."
            ),
            param_type=commons.ExplainerParamType.float,
            default_value=0.95,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
    ]
    _keywords = [
        explainers.Explainer.KEYWORD_DEFAULT,
        explainers.Explainer.KEYWORD_EXPLAINS_O_FEATURE_IMPORTANCE,
        explainers.Explainer.KEYWORD_IS_FAST,
        explainers.Explainer.KEYWORD_H2O_SONAR,
    ]
    _priority = 40.0

    # option: in case of BINOMIAL experiment explanation render the first class ONLY
    OPT_BIN_1_CLASS = True

    def __init__(self):
        explainers.Explainer.__init__(self)
        explainers_commons.AbstractFeatureImportanceExplainer.__init__(self)

        self.args = None
        self.sanitization_map = None
        self.labels = None  # sanitized stringified labels
        self.log_name = "Naive Shapley original FI"

    def setup(
        self,
        model: models.ExplainableModel | None,
        persistence: persistences.ExplainerPersistence,
        key: str = "",
        params: commons.CommonInterpretationParams | None = None,
        explain_original_features: bool = True,
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
        explainers_commons.AbstractFeatureImportanceExplainer.setup(
            self, model=model, persistence=persistence, logger=self.logger
        )

        self.args = explainers.ExplainerArgs(self._parameters)
        self.args.resolve_params(
            explainer_params=explainers.ExplainerArgs.json_str_to_dict(
                self.explainer_params_as_str
            ),
        )
        self.sanitization_map = self.model_meta.sanitization_map
        self.labels = list(map(str, self.model_meta.labels))
        self.log_name = f"Naive Shapley original FI {self.mli_key}/{self.key}"

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        model: models.ExplainableModel | None = None,
        **explainer_params,
    ) -> bool:
        explainers.Explainer.check_compatibility(self, params, **explainer_params)

        if model:
            if models.ExplainableModelType.driverless_ai == model.model_type:
                if model.meta.has_shapley_values:
                    # model supports Shapley values
                    return True

            err_msg = (
                f"{self.log_name}: does not support model type "
                f"{model.model_type} / {type(model.model_src)} "
                f"(only Driverless AI MOJO models are explained)"
            )
            self.logger.warning(err_msg)
            raise errors.ExplainerCompatibilityError(err_msg)

        err_msg = f"{self.log_name} requires a Driverless AI MOJO model"
        self.logger.warning(err_msg)
        raise errors.ExplainerCompatibilityError(err_msg)

    def explain(
        self,
        X: datatable.Frame,
        y: datatable.Frame = None,
        explanations_types: list = None,
        **kwargs,
    ) -> list:
        del y
        del explanations_types

        this_class = NaiveShapleyMojoFeatureImportanceExplainer

        x = sampling.downsample_dataset(
            dataset=X,
            sample_size=self.args.get(
                NaiveShapleyMojoFeatureImportanceExplainer.PARAM_SAMPLE_SIZE
            ),
            runtime_sample_size=this_class.DATASET_SAMPLING_NATIVE,
            target_col=self.model_meta.target_col,
            is_classification=bool(self.model_meta.num_labels > 1),
            classes=self.model_meta.labels,
            seed=this_class.DATASET_SAMPLING_SEED,
            logger=self.logger,
        )
        self.report_progress(0.2)

        if self.model_meta.num_labels <= 2:
            (
                shapley_means_dict,
                shapley_contribs_dict,
            ) = self._get_regression_or_binomial_shap(X=x)
        else:
            (
                shapley_means_dict,
                shapley_contribs_dict,
            ) = self._get_multinomial_shap(X=x)

        sanitized_labels = [
            self.sanitization_map.sanitize_value(str(label))
            for label in shapley_means_dict.keys()
        ]

        self.report_progress(0.9)
        problems_list: list[problems.ProblemAndAction] = []
        try:
            problems_list.extend(self._calculate_problems(shapley_means_dict))
        except Exception:
            self.logger.error("Failed to generate problems.")
        if len(problems_list) > 0:
            for p in problems_list:
                self.add_problem(p)
        (
            global_idx_dict,
            _,
        ) = f5s.GlobalFeatImpJSonDatatableFormat.serialize_index_file(
            classes=sanitized_labels,
            doc=NaiveShapleyMojoFeatureImportanceExplainer._description,
        )
        (
            local_idx_dict,
            _,
        ) = f5s.LocalFeatImpWithYhatsJSonDatatableFormat.serialize_index_file(
            classes=sanitized_labels
        )

        on_demand_params: dict = dict()
        on_demand_params[self.GLOBAL_BASE_PATH] = (
            self.persistence.get_explanation_dir_path(
                explanation_type=e10s.GlobalFeatImpExplanation.explanation_type(),
                explanation_format=f5s.GlobalFeatImpJSonFormat.mime,
            )
        )
        on_demand_params[self.LOCAL_BASE_PATH] = (
            self.persistence.get_explanation_dir_path(
                explanation_type=e10s.LocalFeatImpExplanation.explanation_type(),
                explanation_format=f5s.LocalFeatImpWithYhatsJSonDatatableFormat.mime,
            )
        )

        global_idx_dict[f5s.LocalFeatImpJSonFormat.KEY_ON_DEMAND_PARAMS] = (
            on_demand_params
        )
        local_idx_dict[f5s.LocalFeatImpJSonFormat.KEY_ON_DEMAND_PARAMS] = (
            on_demand_params
        )

        local_idx_dict[f5s.LocalFeatImpWithYhatsJSonDatatableFormat.KEY_Y_HAT] = (
            f5s.LocalFeatImpWithYhatsJSonDatatableFormat.FILE_Y_HAT
        )

        # create the actual explanation
        explanations = list()
        global_explanation = e10s.GlobalFeatImpExplanation(
            explainer=self,
            display_name=NaiveShapleyMojoFeatureImportanceExplainer._display_name,
            display_category=e10s.Explanation.DISPLAY_CAT_DAI_MODEL,
        )
        json_dt_representation = self._normalize_frames_to_gom(
            explanation=global_explanation,
            shapley_means_dict=shapley_means_dict,
            file_mapping_dict=global_idx_dict,
        )

        global_explanation.add_format(explanation_format=json_dt_representation)
        global_explanation.add_format(
            explanation_format=f5s.GlobalFeatImpJSonCsvFormat.from_json_datatable(
                json_dt_representation
            )
        )
        global_explanation.add_format(
            explanation_format=f5s.GlobalFeatImpJSonFormat.from_json_datatable(
                json_dt_representation, bias_col=datasets.ExplainableDataset.COL_BIAS
            )
        )
        explanations.append(global_explanation)

        # local explanation
        self.report_progress(0.95)
        local_explanation = e10s.LocalFeatImpExplanation(
            explainer=self,
            display_name=NaiveShapleyMojoFeatureImportanceExplainer._display_name,
            display_category=e10s.Explanation.DISPLAY_CAT_DAI_MODEL,
        )

        local_json_dt_representation = self._normalize_local_shapley_frames_to_gom(
            local_explanation=local_explanation,
            shapley_contribs_dict=shapley_contribs_dict,
            file_mapping_dict=local_idx_dict,
        )

        local_explanation.add_format(explanation_format=local_json_dt_representation)

        # y hats to store along the Shapley values (full dataset for local explanations)
        y_hats = self.model.predict(
            x,
            # IMPROVE: fast approximation not supported by
            # fast_approx=self.args.get(
            #     NaiveShapleyMojoFeatureImportanceExplainer.PARAM_FAST_APPROX,
            #     NaiveShapleyMojoFeatureImportanceExplainer.DEFAULT_FAST_APPROX,
            # ),
        )
        datasets.DatasetApi.write_dataset(
            dataset=y_hats,
            path=os.path.join(
                self.persistence.get_explainer_working_dir(),
                f5s.LocalFeatImpWithYhatsJSonDatatableFormat.FILE_Y_HAT,
            ),
        )

        local_json_dt_representation.add_file(
            format_file=os.path.join(
                self.persistence.get_explainer_working_dir(),
                f5s.LocalFeatImpWithYhatsJSonDatatableFormat.FILE_Y_HAT,
            ),
            file_name=f5s.LocalFeatImpWithYhatsJSonDatatableFormat.FILE_Y_HAT,
        )

        global_explanation.has_local = local_explanation.explanation_type()

        explanations.append(local_explanation)

        # OPTIONAL explanation: HTML fragment
        if self.config and self.config.create_html_representations:
            try:
                display_name = NaiveShapleyMojoFeatureImportanceExplainer._display_name
                explanations.append(
                    e10s.GlobalHtmlFragmentExplanation.from_explanation(
                        explainer=self,
                        explanation=global_explanation,
                        display_name=display_name,
                        display_category=e10s.Explanation.DISPLAY_CAT_DAI_MODEL,
                        logger=self.logger,
                    )
                )
            except Exception as ex:
                self.logger.warning(
                    f"{self.log_name}: HTML representation creation failed: {ex}"
                )

        return explanations

    def explain_local(
        self,
        X: datatable.Frame,
        y: datatable.Frame = None,
        **extra_params,
    ) -> list:
        this_class = NaiveShapleyMojoFeatureImportanceExplainer

        global_file_mapping = self.persistence.store.load_json(
            extra_params[f5s.ExplanationFormat.KEY_ON_DEMAND_PARAMS][
                this_class.GLOBAL_BASE_PATH
            ]
        ).get(f5s.LocalFeatImpWithYhatsJSonDatatableFormat.KEY_FILES, {})

        local_index_dict = self.persistence.store.load_json(
            extra_params[f5s.ExplanationFormat.KEY_ON_DEMAND_PARAMS][
                this_class.LOCAL_BASE_PATH
            ]
        )

        local_file_mapping = local_index_dict.get(
            f5s.LocalFeatImpWithYhatsJSonDatatableFormat.KEY_FILES, {}
        )

        row = extra_params["row"]
        explain_class = extra_params[f5s.TextCustomExplanationFormat.FILTER_CLASS]

        global_featimp_file = global_file_mapping.get(explain_class, None)
        local_dt_file = local_file_mapping.get(explain_class, None)

        if not global_featimp_file or not local_dt_file:
            raise ValueError("Could not find an explanation for the desired class")

        # load backed up local importance and retrieve the desired row
        dt_shap_contribs = datatable.fread(
            os.path.join(
                extra_params[f5s.ExplanationFormat.KEY_ON_DEMAND_PARAMS][
                    this_class.LOCAL_BASE_PATH
                ],
                local_dt_file,
            )
        )

        # y-hats don't have to be available in case of migrated explanations
        if (
            local_index_dict
            and f5s.LocalFeatImpWithYhatsJSonDatatableFormat.KEY_Y_HAT
            in local_index_dict
        ):
            y_hats = datatable.fread(
                os.path.join(
                    extra_params[f5s.ExplanationFormat.KEY_ON_DEMAND_PARAMS][
                        this_class.LOCAL_BASE_PATH
                    ],
                    local_index_dict[
                        f5s.LocalFeatImpWithYhatsJSonDatatableFormat.KEY_Y_HAT
                    ],
                )
            )
        else:
            y_hats = None

        # TODO better manage bias: bias_value = 0 ~ should be same as global
        if self.dataset_meta.row_count > dt_shap_contribs.shape[0]:
            # contribs frame in contribs dict
            if self.model_meta.num_labels <= 2:
                row_data = self._get_regression_or_binomial_shap(X, export_data=False)[
                    1
                ][explain_class]
            else:
                row_data = self._get_multinomial_shap(X, export_data=False)[1][
                    explain_class
                ]
            bias_value = row_data[datasets.ExplainableDataset.COL_BIAS].to_list()[0][0]
        else:
            row_data = dt_shap_contribs[row, :]
            bias_value = row_data[datasets.ExplainableDataset.COL_BIAS].to_list()[0][0]
            del row_data[datasets.ExplainableDataset.COL_BIAS]

        lfi = f5s.LocalFeatImpJSonFormat

        # retrieve global importance
        global_featimp_dict = {}
        with open(
            os.path.join(
                extra_params[f5s.ExplanationFormat.KEY_ON_DEMAND_PARAMS][
                    this_class.GLOBAL_BASE_PATH
                ],
                global_featimp_file,
            )
        ) as fp:
            _featimp_dict = json.load(fp)
            for item in _featimp_dict[lfi.KEY_DATA]:
                global_featimp_dict[item[lfi.KEY_LABEL]] = item

        # retrieve local importance
        local_featimp_dict = {}
        feature_values: dict = X.to_dict() if X and X.shape[0] == 1 else {}
        for name in row_data.names:
            local_featimp_dict[name] = {
                lfi.KEY_LABEL: name,
                lfi.KEY_VALUE: row_data[0, name],
                lfi.KEY_FEATURE_VALUE: str(feature_values.get(name, [""])[0]),
                lfi.KEY_SCOPE: lfi.SCOPE_LOCAL,
            }
        del dt_shap_contribs

        # order each column importance with global and local importance
        combined_featimp_dict = {
            lfi.KEY_DATA: [],
            lfi.KEY_BIAS: bias_value,
            lfi.KEY_Y: (
                y_hats[row, 0]
                if y_hats and self.dataset_meta.row_count == y_hats.nrows
                else ""
            ),
            lfi.KEY_ACTUAL: X[:, self.model_meta.target_col],
        }

        for col, global_value in global_featimp_dict.items():
            combined_featimp_dict[lfi.KEY_DATA].append(global_value)
            combined_featimp_dict[lfi.KEY_DATA].append(
                local_featimp_dict.get(
                    col,
                    {
                        lfi.KEY_LABEL: col,
                        lfi.KEY_VALUE: 0,
                        lfi.KEY_FEATURE_VALUE: "",
                        lfi.KEY_SCOPE: lfi.SCOPE_LOCAL,
                    },
                )
            )

        explanation = e10s.OnDemandExplanation(explainer=self)
        explanation.add_format(
            f5s.LocalOnDemandTextFormat(
                explanation=explanation,
                format_data=json.dumps(combined_featimp_dict, indent=4),
            )
        )

        return [explanation]

    def explain_problems(self) -> list[problems.ProblemAndAction]:
        return super().explain_problems()

    def _calculate_problems(
        self, shap_means_dict: dict[str, datatable.Frame]
    ) -> list[problems.ProblemAndAction]:
        return problem_detection.get_feature_importance_problems(
            shap_means_dict,
            self.args.get(
                NaiveShapleyMojoFeatureImportanceExplainer.PARAM_LEAKAGE_WARN_THRESHOLD
            ),
            self.explainer_id(),
            self._display_name,
        )

    def _get_regression_or_binomial_shap(self, X, export_data=True):
        dataset = sanitization.sanitize_frame(X, self.model_meta.sanitization_map)

        shapley_means_dict = {}
        shapley_contribs_dict = {}

        # DAI MOJO specific parameters to get Shapley values for original features
        shapley_df_orig_feat = self.model.shapley_values(
            dataset,
            original_features=True,
        )
        # fix column names to comply with expected convention
        shapley_df_names = [
            n.replace(models.DriverlessAiModel.PREFIX_SHAPLEY_COLS, "")
            for n in list(shapley_df_orig_feat.names)
        ]
        shapley_df_orig_feat.names = shapley_df_names

        # filter down to features with importance > 0
        shapley_df_orig_feat = datasets.filter_importance_greater_than_zero(
            shapley_df_orig_feat
        )

        drop_cols = self.params.drop_cols or []

        features = [x for x in shapley_df_orig_feat.names if x not in drop_cols]
        shapley_df_orig_feat = shapley_df_orig_feat[:, features]

        if export_data:
            self._export_shapley_frames(
                shapley_df=shapley_df_orig_feat,
                target_dir=self.working_dir,
                export_original_features=True,
            )

        shapley_means = shapley_df_orig_feat[
            :,
            {
                i: (
                    datatable.f[i]
                    if i == datasets.ExplainableDataset.COL_BIAS
                    else datatable.abs(datatable.f[i])
                )
                for i in shapley_df_orig_feat.names
            },
        ].mean()

        neworder = sorted(
            range(shapley_means.ncols),
            key=lambda i: shapley_means[0, i],
            reverse=True,
        )
        shapley_means = shapley_means[:, neworder]
        if self.model_meta.num_labels:
            if (
                NaiveShapleyMojoFeatureImportanceExplainer.OPT_BIN_1_CLASS
                and self.model_meta.num_labels == 2
            ):
                # bin experiment's model has class of interest @ index 1 (2nd class)
                label = self.labels[1]
                shapley_means_dict[label] = shapley_means
                shapley_contribs_dict[label] = shapley_df_orig_feat
            else:
                for label in self.labels:
                    shapley_means_dict[str(label)] = shapley_means
                    shapley_contribs_dict[str(label)] = shapley_df_orig_feat
        else:
            label = f5s.ExplanationFormat.LABEL_REGRESSION
            shapley_means_dict[str(label)] = shapley_means
            shapley_contribs_dict[str(label)] = shapley_df_orig_feat

        return shapley_means_dict, shapley_contribs_dict

    def _get_per_class_orig_features_names(self):
        orig_feats_contrib_names = [
            f"{models.DriverlessAiModel.PREFIX_SHAPLEY_COLS}{n}"
            for n in self.model_meta.used_features
        ]
        orig_feats_contrib_names.append(
            f"{models.DriverlessAiModel.PREFIX_SHAPLEY_COLS}"
            f"{datasets.ExplainableDataset.COL_BIAS}"
        )

        per_class_orig_feats_names = dict()
        for clazz in self.labels:
            class_names = []
            for label in orig_feats_contrib_names:
                class_names.append(f"{label}.{clazz}")
            per_class_orig_feats_names[clazz] = class_names

        return per_class_orig_feats_names

    PREFIX_CONTRIB = "contrib_"

    def _get_multinomial_shap(self, X, export_data=True):
        this_class = NaiveShapleyMojoFeatureImportanceExplainer

        dataset = sanitization.sanitize_frame(X, self.model_meta.sanitization_map)

        shapley_means_dict = {}
        shapley_contribs_dict = {}
        shapley_df_mult_orig_feat = datatable.Frame()

        # predict() returns contributions of ALL classes (column suffixes)
        # (ISSUE: daimojo seems to return also TRANSFORMED features contributions,
        # therefore these must be filtered out)
        label_df_orig_feat_all = self.model.shapley_values(
            dataset,
            original_features=True,
        )
        per_class_orig_feats_names = self._get_per_class_orig_features_names()

        for label in self.labels:
            # WORKAROUND for TRANSFORMED feature contributions being mixed with
            # ORIGINAL features contributions: filter out TRANSFORMED contributions
            # and keep only columns relevant for given label (class)
            label_df_orig_feat_label = label_df_orig_feat_all[
                :, per_class_orig_feats_names[label]
            ]

            # fix column names to comply with expected convention
            shapley_df_names = [
                n.replace(NaiveShapleyMojoFeatureImportanceExplainer.PREFIX_CONTRIB, "")
                for n in list(label_df_orig_feat_label.names)
            ]
            label_df_orig_feat_label.names = shapley_df_names

            # filter down to features with importance > 0
            label_df_orig_feat = datasets.filter_importance_greater_than_zero(
                label_df_orig_feat_label, label
            )
            drop_cols = self.params.drop_cols or []
            features = [x for x in label_df_orig_feat.names if x not in drop_cols]
            label_df_orig_feat = label_df_orig_feat[:, features]

            shapley_df_mult_orig_feat.cbind(label_df_orig_feat)

            # format names
            label_df_orig_feat.names = [
                c[: -len(f".{label}")] if c.endswith(f".{label}") else c
                for c in label_df_orig_feat.names
            ]

            if export_data:
                this_class._export_shapley_frames_multinomial(
                    shapley_df=shapley_df_mult_orig_feat,
                    target_dir=self.working_dir,
                    export_original_features=True,
                    label=label,
                )

            shapley_means = label_df_orig_feat[
                :,
                {
                    i: (
                        datatable.f[i]
                        if i == datasets.ExplainableDataset.COL_BIAS
                        else datatable.abs(datatable.f[i])
                    )
                    for i in label_df_orig_feat.names
                },
            ].mean()

            neworder = sorted(
                range(shapley_means.ncols),
                key=lambda i: shapley_means[0, i],
                reverse=True,
            )
            shapley_means = shapley_means[:, neworder]

            shapley_means_dict[str(label)] = shapley_means
            shapley_contribs_dict[str(label)] = label_df_orig_feat

        return shapley_means_dict, shapley_contribs_dict

    def get_result(
        self,
    ) -> results.FeatureImportanceResult:
        return results.FeatureImportanceResult(
            persistence=self.persistence,
            explainer_id=NaiveShapleyMojoFeatureImportanceExplainer.explainer_id(),
            h2o_sonar_config=self.config,
            logger=self.logger,
        )
