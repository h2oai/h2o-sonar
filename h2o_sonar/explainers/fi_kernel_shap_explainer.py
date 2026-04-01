# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import os
import traceback

import datatable

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
from h2o_sonar.methods import _shap
from h2o_sonar.utils import problem_detection
from h2o_sonar.utils import sampling
from h2o_sonar.utils import sanitization


AbstractFeatureImportanceExplainer = (
    explainers_commons.AbstractFeatureImportanceExplainer
)


class KernelShapFeatureImportanceExplainer(
    explainers.Explainer, AbstractFeatureImportanceExplainer
):
    """Kernel SHAP based original feature importance explainer."""

    PARAM_NSAMPLE = "nsample"
    PARAM_L1 = "L1"
    PARAM_MAXRUNTIME = "max runtime"
    PARAM_FAST_APPROX = "fast_approx"
    PARAM_LEAKAGE_WARN_THRESHOLD = "leakage_warning_threshold"

    _display_name = "Shapley Values for Original Features (Kernel SHAP Method)"
    _description = (
        "Shapley explanations are a technique with credible theoretical support "
        "that presents consistent global and local variable contributions. "
        "Local numeric Shapley values are calculated by tracing single rows "
        "of data through a trained tree ensemble and aggregating the "
        "contribution of each input variable as the row of data moves through "
        "the trained ensemble. For regression tasks, Shapley values sum to "
        "the prediction of the Driverless AI model. For classification "
        "problems, Shapley values sum to the prediction of the Driverless AI "
        "model before applying the link function. Global Shapley values are "
        "the average of the absolute Shapley values over every "
        "row of a dataset. Shapley values for original features are calculated "
        "with the Kernel Explainer method, which uses a special weighted "
        "linear regression to compute the importance of each feature. More "
        "information about Kernel SHAP is available at "
        "http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-"
        "model-predictions.pdf."
    )
    _iid = True
    _regression = True
    _binary = True
    _multiclass = True
    _global_explanation = True
    _local_explanation = True
    _explanation_types = [e10s.GlobalFeatImpExplanation, e10s.LocalFeatImpExplanation]
    _optional_explanation_types = [e10s.GlobalHtmlFragmentExplanation]
    _keywords = [
        explainers.Explainer.KEYWORD_EXPLAINS_O_FEATURE_IMPORTANCE,
        explainers.Explainer.KEYWORD_IS_SLOW,
        explainers.Explainer.KEYWORD_H2O_SONAR,
    ]
    _modules_needed_by_name = ["shap", "scikit-image"]
    _parameters = [
        explainers.ExplainerParam(
            param_name=AbstractFeatureImportanceExplainer.PARAM_SAMPLE_SIZE,
            description="Sample size.",
            param_type=commons.ExplainerParamType.int,
            default_value=100_000,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=AbstractFeatureImportanceExplainer.PARAM_SAMPLE,
            description="Sample Kernel Shapley.",
            param_type=commons.ExplainerParamType.bool,
            default_value=True,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_NSAMPLE,
            description="Number of times to re-evaluate the model when "
            "explaining each prediction with Kernel Explainer. "
            "Default is determined internally.'auto' or int. "
            "Number of times to re-evaluate the model when explaining each "
            "prediction. More samples lead to lower variance "
            "estimates of the SHAP values. The 'auto' setting uses "
            "nsamples = 2 * X.shape[1] + 2048. This setting is disabled by "
            "default and runtime determines the right number internally.",
            param_type=commons.ExplainerParamType.int,
            default_value="",
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_L1,
            description="L1 regularization for Kernel Explainer. "
            "'num_features(int)', 'auto' (default for now, "
            "but deprecated), 'aic', 'bic', or float. The L1 regularization to "
            "use for feature selection (the estimation procedure is based on "
            "a debiased lasso). The 'auto' option currently uses aic when "
            "less that 20% of the possible sample space is enumerated, "
            "otherwise it uses no regularization. The aic and bic options use "
            "the AIC and BIC rules for regularization. Using "
            "'num_features(int)' selects a fix number of top features. Passing "
            "a float directly sets the alpha parameter of the "
            "sklearn.linear_model.Lasso model used for feature selection.",
            param_type=commons.ExplainerParamType.str,
            default_value="auto",
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_MAXRUNTIME,
            description="Max runtime for Kernel explainer in seconds.",
            param_type=commons.ExplainerParamType.int,
            default_value=900,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_FAST_APPROX,
            description="Speed up predictions with fast predictions approximation.",
            param_type=commons.ExplainerParamType.bool,
            default_value=True,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_LEAKAGE_WARN_THRESHOLD,
            description=(
                "The threshold above which to report a potentially detected"
                " feature importance leak problem."
            ),
            param_type=commons.ExplainerParamType.float,
            default_value=0.95,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
    ]
    _requires_preloaded_predictor = False

    # option: render the first class ONLY in case of BINOMIAL experiment interpretation
    OPT_BIN_1_CLASS = True

    def __init__(self):
        explainers.Explainer.__init__(self)
        explainers_commons.AbstractFeatureImportanceExplainer.__init__(self)

        self.model_key: str | None = None
        self.model_path: str | None = None
        self.actual_col: str | None = None
        self.dataset_key: str | None = None

        self.args = None
        # sanitized stringified labels
        self.labels = None

        self.log_name: str = "Kernel SHAP original FI"

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
        self.labels = (
            list(map(str, self.model_meta.labels))
            if self.model_meta and self.model_meta.labels is not None
            else None
        )

        self.log_name = f"Kernel SHAP original FI {self.mli_key}/{self.key}"

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        model: models.ExplainableModel | None = None,
        **explainer_params,
    ) -> bool:
        explainers.Explainer.check_compatibility(self, params, **explainer_params)

        return True

    def explain(
        self,
        X: datatable.Frame,
        y: datatable.Frame = None,
        explanations_types: list = None,
        **kwargs,
    ) -> list:
        del y
        del explanations_types

        this_class = KernelShapFeatureImportanceExplainer

        x = sampling.downsample_dataset(
            dataset=X,
            sample_size=self.args.get(
                KernelShapFeatureImportanceExplainer.PARAM_SAMPLE_SIZE
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
        except Exception as ex:
            self.logger.error(
                f"Failed to generate problems: {ex}\n{traceback.format_exc()}"
            )
        if len(problems_list) > 0:
            for p in problems_list:
                self.add_problem(p)
        (
            global_idx_dict,
            _,
        ) = f5s.GlobalFeatImpJSonDatatableFormat.serialize_index_file(
            classes=sanitized_labels,
            doc=KernelShapFeatureImportanceExplainer._description,
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

        explanations = list()

        # create the actual explanation
        display_name = "Shapley on Original Features (Kernel SHAP Method)"
        global_explanation = e10s.GlobalFeatImpExplanation(
            explainer=self,
            display_name=display_name,
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
            display_name=display_name,
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
            #     OriginalFeatureImportanceKernelShapExplainer.PARAM_FAST_APPROX,
            #     OriginalFeatureImportanceKernelShapExplainer.DEFAULT_FAST_APPROX,
            # ),
        )
        datasets.DatasetApi.write_dataset(
            dataset=datasets.ExplainableDataset.frame_2_datatable(y_hats),
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
                explanations.append(
                    e10s.GlobalHtmlFragmentExplanation.from_explanation(
                        explainer=self,
                        explanation=global_explanation,
                        display_name=display_name,
                        display_category=e10s.Explanation.DISPLAY_CAT_MODEL,
                        logger=self.logger,
                    )
                )
            except Exception as ex:
                self.logger.warning(
                    f"{self.log_name}: HTML representation creation failed: {ex}"
                )

        return explanations

    def _calculate_problems(
        self, shap_means_dict: dict[str, datatable.Frame]
    ) -> list[problems.ProblemAndAction]:
        return problem_detection.get_feature_importance_problems(
            shap_means_dict,
            self.args.get(
                KernelShapFeatureImportanceExplainer.PARAM_LEAKAGE_WARN_THRESHOLD
            ),
            self.explainer_id(),
            self._display_name,
        )

    def explain_problems(self) -> list[problems.ProblemAndAction]:
        return super().explain_problems()

    def _get_regression_or_binomial_shap(self, X, export_data=True):
        dataset = sanitization.sanitize_frame(X, self.model_meta.sanitization_map)

        shapley_means_dict = {}
        shapley_contribs_dict = {}

        if self.model.model_type is models.ExplainableModelType.driverless_ai:
            # subset frame to features used by the model
            dataset = dataset[:, self.model.meta.used_features]
        elif self.params.target_col in dataset.names:
            del dataset[self.params.target_col]

        shap = _shap.Shap(
            explainable_model=self.model,
        )
        try:
            shapley_df_orig_feat = shap.explain(
                X=dataset,
                experiment_type=(
                    commons.ExperimentType.regression.name if not self.labels else ""
                ),
            )
        except Exception as ex:
            model_feature_names = (
                self.model.model_src.feature_names
                if self.model
                and self.model.model_src
                and hasattr(self.model.model_src, "feature_names")
                else "<UNKNOWN>"
            )
            raise ValueError(
                f"Predict method failed with: {ex}\n"
                f"> target column: {self.params.target_col}\n"
                f">  data columns: {dataset.names}\n"
                f">         model: {model_feature_names}\n"
                f"{traceback.format_exc()}"
            )

        # fix column names to comply with expected convention
        shapley_df_names = [
            n.replace("contrib_", "") for n in list(shapley_df_orig_feat.names)
        ]
        shapley_df_orig_feat.names = shapley_df_names

        # filter down to features with importance > 0
        shapley_df_orig_feat = datasets.filter_importance_greater_than_zero(
            frame=shapley_df_orig_feat,
            skip_bias=False,
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
                KernelShapFeatureImportanceExplainer.OPT_BIN_1_CLASS
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

    def _get_multinomial_shap(self, X, export_data=True):
        this_class = KernelShapFeatureImportanceExplainer

        dataset = sanitization.sanitize_frame(X, self.model_meta.sanitization_map)
        if self.model.model_type is models.ExplainableModelType.driverless_ai:
            # subset frame to features used by the model
            dataset = dataset[:, self.model.meta.used_features]
        if (
            self.model.model_type is models.ExplainableModelType.scikit_learn
            and self.model.meta
            and self.model_meta.used_features
        ):
            # subset frame to features used by the model
            dataset = dataset[:, self.model.meta.used_features]

        shapley_means_dict = {}
        shapley_contribs_dict = {}

        # calculate Shapley values for ALL classes at once, post process per-class
        shap = _shap.Shap(
            explainable_model=self.model,
        )
        shap_contribs_raw = shap.explain(dataset)
        # filter down to features with importance > 0
        shap_contribs_raw = datasets.filter_importance_greater_than_zero(
            frame=shap_contribs_raw, label=None, skip_bias=True
        )
        drop_cols = self.params.drop_cols or []
        features = [x for x in shap_contribs_raw.names if x not in drop_cols]

        for label in self.labels:
            shapley_df_mult_orig_feat = datatable.Frame()

            shap_contribs_wo_drops = shap_contribs_raw[:, features]

            sorter = _shap.ShapContribsSorter(features, labels=self.labels)
            # insert columns w/ feature contribs of given class to the result frame
            shapley_df_mult_orig_feat.cbind(
                shap_contribs_wo_drops[
                    :, sorter.get_cols_for_label(label, strip_label=False)
                ]
            )
            # format result frame column names by removing the label suffix
            shapley_df_mult_orig_feat.names = sorter.get_cols_for_label(
                label, strip_label=True
            )

            if export_data:
                this_class._export_shapley_frames_multinomial(
                    shapley_df=shapley_df_mult_orig_feat,
                    target_dir=self.working_dir,
                    export_original_features=True,
                    label=label,
                    skip_bias=False,
                )

            shapley_means = shapley_df_mult_orig_feat[
                :,
                {
                    i: (
                        datatable.f[i]
                        if i == datasets.ExplainableDataset.COL_BIAS
                        else datatable.abs(datatable.f[i])
                    )
                    for i in shapley_df_mult_orig_feat.names
                },
            ].mean()

            neworder = sorted(
                range(shapley_means.ncols),
                key=lambda i: shapley_means[0, i],
                reverse=True,
            )
            shapley_means = shapley_means[:, neworder]

            shapley_means_dict[str(label)] = shapley_means
            shapley_contribs_dict[str(label)] = shapley_df_mult_orig_feat

        # IMPROVE Shapley sorter can be used to get per-class bias and more
        return shapley_means_dict, shapley_contribs_dict

    def get_result(
        self,
    ) -> results.FeatureImportanceResult:
        return results.FeatureImportanceResult(
            persistence=self.persistence,
            explainer_id=KernelShapFeatureImportanceExplainer.explainer_id(),
            h2o_sonar_config=self.config,
            logger=self.logger,
        )
