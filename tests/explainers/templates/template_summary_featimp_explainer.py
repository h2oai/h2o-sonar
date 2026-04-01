# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import json
import random

import datatable

from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api import results


class TemplateShapleySummaryOrigFeatExplainer(explainers.Explainer):
    """Shapley summary plot for original features of Driverless AI models
    template.

    Use this template to create explainer which creates global and local
    **summary** feature importance explanations.

    """

    _display_name = "Template Shapley Summary Plot for Original Features"
    _description = (
        "Summary feature importance explainer template which can be used "
        "create explainer with global and local summary feature importance "
        "explanations."
    )
    _regression = True
    _binary = True
    _multiclass = True
    _global_explanation = True
    _local_explanation = True
    _explanation_types = [
        e10s.GlobalSummaryFeatImpExplanation,
        e10s.LocalSummaryFeatImpExplanation,
    ]
    _keywords = [explainers.Explainer.KEYWORD_TEMPLATE]

    def setup(self, model: models.ExplainableModel, persistence, **kwargs):
        explainers.Explainer.setup(self, model=model, persistence=persistence, **kwargs)

    def explain(self, X, y=None, explanations_types: list = None, **kwargs):
        """Create global and local (pre-computed/cached) explanations.

        Template explainer returns MOCK explainer data - replace mock data
        preparation with actual computation to create real explanations.

        """
        # explanations list
        model_explanations = list()

        # global explanation
        (global_explanation, features_per_page) = self._explain_global_featimp()
        model_explanations.append(global_explanation)

        # local explanation
        local_explanation = self._explain_local_featimp(features_per_page)

        # associate local explanation with global explanation
        global_explanation.has_local = local_explanation.explanation_type()
        model_explanations.append(local_explanation)

        return model_explanations

    def get_result(
        self,
    ) -> results.TemplateResult:
        this_class = TemplateShapleySummaryOrigFeatExplainer
        return results.TemplateResult(
            persistence=self.persistence,
            explainer_id=this_class.explainer_id(),
            explainer_name=TemplateShapleySummaryOrigFeatExplainer._display_name,
        )

    def _explain_global_featimp(self):
        """Create global summary feature importance explanation with JSon
        (index file) and datatable (data file) format representation. This
        representation is supported by Grammar of MLI and will be rendered
        in UI.

        """
        global_fi_explanation = e10s.GlobalSummaryFeatImpExplanation(
            explainer=self,
            # UI tile name
            display_name="Template Shapley Summary Plot for Original Features",
            # UI tab name
            display_category=e10s.Explanation.DISPLAY_CAT_EXAMPLE,
        )

        # JSon+datatable explanation representation
        (
            idx_file_dict,
            idx_file_str,
        ) = formats.GlobalSummaryFeatImpJsonDatatableFormat.serialize_index_file(
            classes=TemplateShapleySummaryOrigFeatExplainer.MOCK_CLASSES,
            metrics=[{"R2": 0.96}, {"RMSE": 0.03}],
            doc=TemplateShapleySummaryOrigFeatExplainer._description,
            total_rows=len(TemplateShapleySummaryOrigFeatExplainer.MOCK_FEATURES),
        )
        json_dt_representation = formats.GlobalSummaryFeatImpJsonDatatableFormat(
            explanation=global_fi_explanation,
            json_data=idx_file_str,
            persistence=self.persistence.store,
        )
        # add format files: per-class (saved as added to format)
        for clazz in TemplateShapleySummaryOrigFeatExplainer.MOCK_CLASSES:
            json_dt_representation.add_data_frame(
                format_data=self._mock_create_global_json_dt_frame(
                    clazz=clazz,
                ),
                file_name=idx_file_dict["files"][clazz],
            )
        global_fi_explanation.add_format(json_dt_representation)

        # JSon explanation representation
        (
            json_representation,
            features_per_page,
        ) = formats.GlobalSummaryFeatImpJsonFormat.from_json_datatable(
            json_dt_format=json_dt_representation,
            page_size=formats.GlobalSummaryFeatImpJsonFormat.DEFAULT_PAGE_SIZE,
            persistence=self.persistence.store,
        )
        global_fi_explanation.add_format(json_representation)

        return global_fi_explanation, features_per_page

    def _mock_create_global_json_dt_frame(self, clazz) -> datatable.Frame:
        e_type = TemplateShapleySummaryOrigFeatExplainer
        dt_representation_type = formats.GlobalSummaryFeatImpJsonDatatableFormat

        # x-axis: feature Shapley-based contribution <-inf, inf>
        x_resolution = 500
        x_step = (e_type.X_SHAPLEY_MAX - e_type.X_SHAPLEY_MIN) / (x_resolution - 1)
        x_ticks = [e_type.X_SHAPLEY_MIN + x_step * i for i in range(x_resolution)]
        # y-axis: features ordered by feature importance
        y_features = TemplateShapleySummaryOrigFeatExplainer.MOCK_FEATURES
        # color: normalized feature height to [0, 1] (averaged for values @ same x-bin)

        # feature
        col_feature = []
        # per-feature value Shapley contribution
        col_shapley_value = []
        # frequency of particular Shapely contribution(s)
        col_count = []
        # high/low (red/blue) feature values averaged in given Shapley contribution
        col_avg_high_value = []
        # class
        col_clazz = []
        # order to ensure feature importance based ordering (and allow paging)
        col_order = []
        for i, feature in enumerate(y_features):
            is_categorical = random.randint(0, 1)
            for x_shap_tick in x_ticks:
                if random.random() > 0.70:
                    col_feature.append(feature)
                    col_shapley_value.append(x_shap_tick)
                    col_count.append(
                        random.randint(e_type.X_COUNT_MIN, e_type.X_COUNT_MAX)
                    )
                    col_avg_high_value.append(
                        None if is_categorical else random.random()
                    )
                    col_clazz.append(clazz)
                    col_order.append(i)

        frame_dict: dict = {
            dt_representation_type.KEY_FEATURE: col_feature,
            dt_representation_type.KEY_SHAPLEY: col_shapley_value,
            dt_representation_type.KEY_FREQUENCY: col_count,
            dt_representation_type.KEY_HIGH_VALUE: col_avg_high_value,
            dt_representation_type.KEY_ORDER: col_order,
        }

        frame = datatable.Frame(frame_dict)

        self.logger.info(
            f"Global feature importance explanation frame for class {clazz}:\n{frame}"
        )

        return frame

    def _explain_local_featimp(
        self, features_per_page: dict
    ) -> e10s.LocalSummaryFeatImpExplanation:
        """Create on-demand local feature importance explanation - to be created
        by scorer providing both prediction and Shapley values.  This representation
        is supported by Grammar of MLI and will
        be rendered in UI.

        """
        local_explanation = e10s.LocalSummaryFeatImpExplanation(
            explainer=self,
            display_name="Template Local Shapley Summary Plot for Original Features",
            display_category=e10s.LocalSummaryFeatImpExplanation.DISPLAY_CAT_EXAMPLE,
        )

        (
            json_local_idx,
            _,
        ) = formats.LocalSummaryFeatImplJSonFormat.serialize_index_file(
            classes=TemplateShapleySummaryOrigFeatExplainer.MOCK_CLASSES,
            doc=TemplateShapleySummaryOrigFeatExplainer._description,
        )
        json_local_idx[formats.LocalSummaryFeatImplJSonFormat.KEY_ON_DEMAND] = True
        on_demand_params: dict = dict()
        on_demand_params[formats.LocalSummaryFeatImplJSonFormat.KEY_SYNC_ON_DEMAND] = (
            False
        )
        on_demand_params[
            formats.GlobalSummaryFeatImpJsonFormat.KEY_FEATURES_PER_PAGE
        ] = features_per_page
        json_local_idx[formats.LocalSummaryFeatImplJSonFormat.KEY_ON_DEMAND_PARAMS] = (
            on_demand_params
        )
        local_explanation.add_format(
            explanation_format=formats.LocalSummaryFeatImplJSonFormat(
                explanation=local_explanation,
                json_data=json.dumps(json_local_idx, indent=4),
                persistence=self.persistence.store,
            )
        )

        return local_explanation

    def explain_local(
        self, X: datatable.Frame, y: datatable.Frame = None, **extra_params
    ) -> list:
        """On-demand local summary feature importance explanation."""
        del X
        del y

        row: int = extra_params.get("row", None)
        if row is None or not isinstance(row, int) or row < 0:
            raise ValueError(
                "Local explanation handler requires row index to be positive integer"
            )
        filter_clazz: str = extra_params.get(
            formats.TextCustomExplanationFormat.FILTER_CLASS, None
        )
        if filter_clazz is None:
            raise ValueError(
                f"Local explanation filter parameter "
                f"'{formats.TextCustomExplanationFormat.FILTER_CLASS}' is required"
            )
        if extra_params.get(
            formats.LocalSummaryFeatImplJSonFormat.KEY_ON_DEMAND_PARAMS, None
        ):
            page_offset = extra_params[
                formats.LocalSummaryFeatImplJSonFormat.KEY_ON_DEMAND_PARAMS
            ].get(formats.LocalSummaryFeatImplJSonFormat.KEY_PAGE_OFFSET, None)
        else:
            page_offset = None
        me = TemplateShapleySummaryOrigFeatExplainer
        dt_format_type = formats.GlobalSummaryFeatImpJsonDatatableFormat

        # MOCK prediction with Shapley contributions for particular ROW
        data: list = []
        mock_prediction: dict = {formats.LocalSummaryFeatImplJSonFormat.KEY_DATA: data}
        for feature in TemplateShapleySummaryOrigFeatExplainer.MOCK_FEATURES:
            data.append(
                {
                    dt_format_type.KEY_FEATURE: feature,
                    dt_format_type.KEY_SHAPLEY: random.uniform(
                        me.X_SHAPLEY_MIN, me.X_SHAPLEY_MAX
                    ),
                    dt_format_type.KEY_FREQUENCY: 1,
                    dt_format_type.KEY_HIGH_VALUE: random.random(),
                }
            )
        self.logger.info(f"Raw LOCAL explanation:\n{mock_prediction}")

        # paging based filtering (needed by explainers) to get only desired features
        self.logger.info(
            f"Getting LOCAL explanation for row {row}, class '{filter_clazz}' and "
            f"page offset {page_offset}"
        )
        result = {}
        if page_offset is not None:
            try:
                parent_persistence = persistences.ExplainerPersistence(
                    data_dir=self.persistence.data_dir,
                    username=self.persistence.username,
                    explainer_id=me.explainer_id(),
                    explainer_job_key=extra_params.get(
                        commons.ExplainerParamKey.KEY_E_JOB_KEY, None
                    ),
                    mli_key=extra_params.get(
                        commons.ExplainerParamKey.KEY_ON_DEMAND_MLI_KEY, None
                    ),
                )
                index_path = parent_persistence.get_explanation_file_path(
                    explanation_type=(
                        e10s.LocalSummaryFeatImpExplanation.explanation_type()
                    ),
                    explanation_format=formats.LocalSummaryFeatImplJSonFormat.mime,
                )
                index_dict = self.persistence.load_json(index_path)
                self.logger.info(
                    f"LOCAL explanation page filtering params: {index_dict}"
                )
                features_per_page = index_dict[
                    formats.LocalSummaryFeatImplJSonFormat.KEY_ON_DEMAND_PARAMS
                ][formats.GlobalSummaryFeatImpJsonFormat.KEY_FEATURES_PER_PAGE][
                    filter_clazz
                ][str(page_offset)]
                self.logger.info(
                    f"LOCAL explanation page features: {features_per_page}"
                )
                if features_per_page:
                    result: dict = {formats.LocalSummaryFeatImplJSonFormat.KEY_DATA: []}
                    for item in mock_prediction[
                        formats.LocalSummaryFeatImplJSonFormat.KEY_DATA
                    ]:
                        if (
                            item[formats.GlobalSummaryFeatImpJsonFormat.KEY_FEATURE]
                            in features_per_page
                        ):
                            item[dt_format_type.KEY_SCOPE] = dt_format_type.SCOPE_LOCAL
                            result[
                                formats.LocalSummaryFeatImplJSonFormat.KEY_DATA
                            ].append(item)
            except Exception as ex:
                self.logger.warning(
                    f"Local explanation failed to load page filtering: {ex}"
                )

        explanation = e10s.OnDemandExplanation(explainer=self)
        explanation.add_format(
            formats.LocalOnDemandTextFormat(
                explanation=explanation,
                format_data=json.dumps(result or mock_prediction),
            )
        )

        return [explanation]

    #
    # mock data
    #

    # x-axis: min and max of Shapley contributions for mock (random) data generation
    X_SHAPLEY_MIN = -3
    X_SHAPLEY_MAX = 10
    # x-axis: min and max of frequency in Shapley contributions bin (random) data
    X_COUNT_MIN = 1
    X_COUNT_MAX = 10

    MOCK_FEATURES = [
        # page offset: 0
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "LIMIT_BAL",
        "PAY_AMT1",
        "BILL_AMT1",
        "PAY_AMT4",
        # page offset: 10
        "BILL_AMT2",
        "PAY_AMT2",
        "PAY_AMT5",
        "BILL_AMT5",
        "BILL_AMT4",
        "PAY_AMT3",
        "BILL_AMT3",
        "AGE",
        "PAY_AMT6",
        "BILL_AMT6",
        # page offset: 20
        "EDUCATION",
    ]

    MOCK_CLASSES = ["class_A", "class_B", "class_C"]
