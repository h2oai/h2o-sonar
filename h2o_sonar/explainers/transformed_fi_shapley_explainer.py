# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
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
from h2o_sonar.lib.api import results
from h2o_sonar.utils import sampling
from h2o_sonar.utils import sanitization


AbstractFeatureImportanceExplainer = (
    explainers_commons.AbstractFeatureImportanceExplainer
)


class ShapleyMojoTransformedFeatureImportanceExplainer(
    explainers.Explainer, AbstractFeatureImportanceExplainer
):
    DEFAULT_FAST_APPROX = True

    _display_name = "Shapley Values for Transformed Features of MOJO Models"
    _description = (
        "Shapley explanations are a technique with credible theoretical support "
        "that presents consistent global and local variable contributions. Local "
        "numeric Shapley values are calculated by tracing single rows of data "
        "through a trained tree ensemble and aggregating the contribution of each "
        "input variable as the row of data moves through the trained ensemble. "
        "For regression tasks Shapley values sum to the prediction of the "
        "(Driverless AI) MOJO model. For classification problems, Shapley values sum "
        "to the prediction of the MOJO model before applying the link function. "
        "Global Shapley values are the average of the absolute local Shapley values "
        "over every row of a data set."
    )
    _iid = True
    _time_series = True
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
            param_name=AbstractFeatureImportanceExplainer.PARAM_SAMPLE_SIZE,
            description="Sample size.",
            param_type=commons.ExplainerParamType.int,
            default_value=100_000,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=AbstractFeatureImportanceExplainer.PARAM_CALCULATE_PREDICTIONS,
            description=(
                "Score dataset and include predictions in the explanation "
                "(local explanations speed-up cache)."
            ),
            param_type=commons.ExplainerParamType.bool,
            default_value=False,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=AbstractFeatureImportanceExplainer.PARAM_FAST_APPROX,
            description=(
                "Speed up predictions with fast contributions predictions "
                "approximation."
            ),
            param_type=commons.ExplainerParamType.bool,
            default_value=True,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
    ]
    _keywords = [
        explainers.Explainer.KEYWORD_DEFAULT,
        explainers.Explainer.KEYWORD_EXPLAINS_T_FEATURE_IMPORTANCE,
        explainers.Explainer.KEYWORD_IS_FAST,
        explainers.Explainer.KEYWORD_H2O_SONAR,
    ]
    _priority = 15.5

    # option: render the first class ONLY in case of BINOMIAL experiment interpretation
    OPT_BIN_1_CLASS = True

    def __init__(self):
        explainers.Explainer.__init__(self)
        explainers_commons.AbstractFeatureImportanceExplainer.__init__(self)

        self.args: dict = {}
        self.fast_approx: bool = True
        self.labels = None  # sanitized stringified labels
        self.log_name = "Shapley transformed FI"

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
        self.log_name: str = f"Shapley transformed FI {self.mli_key}/{self.key}"

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

        this_class = ShapleyMojoTransformedFeatureImportanceExplainer

        x = sampling.downsample_dataset(
            dataset=X,
            sample_size=self.args.get(
                ShapleyMojoTransformedFeatureImportanceExplainer.PARAM_SAMPLE_SIZE
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

        (
            global_idx_dict,
            _,
        ) = f5s.GlobalFeatImpJSonDatatableFormat.serialize_index_file(
            classes=sanitized_labels,
            doc=ShapleyMojoTransformedFeatureImportanceExplainer._description,
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
            display_name=ShapleyMojoTransformedFeatureImportanceExplainer._display_name,
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
            display_name=ShapleyMojoTransformedFeatureImportanceExplainer._display_name,
            display_category=e10s.Explanation.DISPLAY_CAT_DAI_MODEL,
        )

        local_json_dt_representation = self._normalize_local_shapley_frames_to_gom(
            local_explanation=local_explanation,
            shapley_contribs_dict=shapley_contribs_dict,
            file_mapping_dict=local_idx_dict,
        )

        local_explanation.add_format(explanation_format=local_json_dt_representation)

        # y hats to store along the Shapley values (full dataset for local explanations)
        if self.args.get(this_class.PARAM_CALCULATE_PREDICTIONS):
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
        try:
            display = ShapleyMojoTransformedFeatureImportanceExplainer._display_name
            explanations.append(
                e10s.GlobalHtmlFragmentExplanation.from_explanation(
                    explainer=self,
                    explanation=global_explanation,
                    display_name=display,
                    display_category=e10s.Explanation.DISPLAY_CAT_DAI_MODEL,
                    is_raw_feature=False,
                    logger=self.logger,
                )
            )
        except Exception as ex:
            self.logger.warning(
                f"{self.log_name}: HTML representation creation failed: {ex}"
            )

        return explanations

    def _get_regression_or_binomial_shap(self, X, export_data=True):
        dataset = sanitization.sanitize_frame(X, self.model_meta.sanitization_map)

        shapley_means_dict = {}
        shapley_contribs_dict = {}

        # DAI MOJO specific parameters to get Shapley values for transformed features
        shapley_df_transformed_feat = self.model.shapley_values(
            dataset,
            original_features=False,
        )
        # fix column names to comply with expected convention
        shapley_df_names = [
            n.replace(models.DriverlessAiModel.PREFIX_SHAPLEY_COLS, "")
            for n in list(shapley_df_transformed_feat.names)
        ]
        shapley_df_transformed_feat.names = shapley_df_names

        # filter down to features with importance > 0
        shapley_df_transformed_feat = datasets.filter_importance_greater_than_zero(
            shapley_df_transformed_feat
        )

        drop_cols = self.params.drop_cols or []

        features = [x for x in shapley_df_transformed_feat.names if x not in drop_cols]
        shapley_df_transformed_feat = shapley_df_transformed_feat[:, features]

        if export_data:
            self._export_shapley_frames(
                shapley_df=shapley_df_transformed_feat,
                target_dir=self.working_dir,
                export_original_features=False,
            )

        shapley_means = shapley_df_transformed_feat[
            :,
            {
                i: (
                    datatable.f[i]
                    if i == datasets.ExplainableDataset.COL_BIAS
                    else datatable.abs(datatable.f[i])
                )
                for i in shapley_df_transformed_feat.names
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
                ShapleyMojoTransformedFeatureImportanceExplainer.OPT_BIN_1_CLASS
                and self.model_meta.num_labels == 2
            ):
                # bin experiment's model has class of interest @ index 1 (2nd class)
                label = self.labels[1]
                shapley_means_dict[label] = shapley_means
                shapley_contribs_dict[label] = shapley_df_transformed_feat
            else:
                for label in self.labels:
                    shapley_means_dict[str(label)] = shapley_means
                    shapley_contribs_dict[str(label)] = shapley_df_transformed_feat
        else:
            label = f5s.ExplanationFormat.LABEL_REGRESSION
            shapley_means_dict[str(label)] = shapley_means
            shapley_contribs_dict[str(label)] = shapley_df_transformed_feat

        return shapley_means_dict, shapley_contribs_dict

    @staticmethod
    def _get_per_class_trans_features_names(contributions: datatable.Frame, label: str):
        return [
            c
            for c in contributions.names
            if c
            and c.endswith(f".{label}")
            and c.startswith(models.DriverlessAiModel.PREFIX_SHAPLEY_COLS)
        ]

    def _get_multinomial_shap(self, X, export_data=True):
        this_class = ShapleyMojoTransformedFeatureImportanceExplainer

        dataset = sanitization.sanitize_frame(X, self.model_meta.sanitization_map)

        shapley_means_dict = {}
        shapley_contribs_dict = {}
        shapley_df_mult_transformed_feat = datatable.Frame()

        # predict() returns contributions of ALL classes (column suffixes)
        label_df_transformed_feat_all = self.model.shapley_values(
            dataset,
            original_features=False,
        )

        for label in self.labels:
            # keep only columns relevant for given label (class)
            per_class_orig_feats_names = this_class._get_per_class_trans_features_names(
                contributions=label_df_transformed_feat_all,
                label=label,
            )
            label_df_transformed_feat = label_df_transformed_feat_all[
                :, per_class_orig_feats_names
            ]

            # fix column names to comply with expected convention
            shapley_df_names = [
                n.replace(models.DriverlessAiModel.PREFIX_SHAPLEY_COLS, "")
                for n in list(label_df_transformed_feat.names)
            ]
            label_df_transformed_feat.names = shapley_df_names

            # filter down to features with importance > 0
            label_df_transformed_feat = datasets.filter_importance_greater_than_zero(
                label_df_transformed_feat, label
            )
            drop_cols = self.params.drop_cols or []
            features = [
                x for x in label_df_transformed_feat.names if x not in drop_cols
            ]
            label_df_transformed_feat = label_df_transformed_feat[:, features]

            shapley_df_mult_transformed_feat.cbind(label_df_transformed_feat)

            # format names
            label_df_transformed_feat.names = [
                c[: -len(f".{label}")] if c.endswith(f".{label}") else c
                for c in label_df_transformed_feat.names
            ]

            if export_data:
                this_class._export_shapley_frames_multinomial(
                    shapley_df=shapley_df_mult_transformed_feat,
                    target_dir=self.working_dir,
                    export_original_features=False,
                    label=label,
                )

            shapley_means = label_df_transformed_feat[
                :,
                {
                    i: (
                        datatable.f[i]
                        if i == datasets.ExplainableDataset.COL_BIAS
                        else datatable.abs(datatable.f[i])
                    )
                    for i in label_df_transformed_feat.names
                },
            ].mean()

            neworder = sorted(
                range(shapley_means.ncols),
                key=lambda i: shapley_means[0, i],
                reverse=True,
            )
            shapley_means = shapley_means[:, neworder]

            shapley_means_dict[str(label)] = shapley_means
            shapley_contribs_dict[str(label)] = label_df_transformed_feat

        return shapley_means_dict, shapley_contribs_dict

    def get_result(
        self,
    ) -> results.FeatureImportanceResult:
        this_class = ShapleyMojoTransformedFeatureImportanceExplainer
        return results.FeatureImportanceResult(
            persistence=self.persistence,
            explainer_id=this_class.explainer_id(),
            h2o_sonar_config=self.config,
            logger=self.logger,
        )
