# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import json

import airium
import datatable

from h2o_sonar import errors
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api import results
from h2o_sonar.methods.fairness import _dia as dia


CUT_OFF_METRICS = ["F1", "F2", "F05", "MCC"]
NUM_CLASSES = 2


class DiaExplainer(explainers.Explainer):
    """Disparate Impact Analysis (DIA) explainer for explainable models."""

    PARAM_FEATURES = "dia_cols"
    PARAM_CUT_OFF = "cut_off"
    PARAM_MAXIMIZE_METRIC = "maximize_metric"
    PARAM_USE_HOLDOUT_PREDS = "use_holdout_preds"
    PARAM_SAMPLE_SIZE = "sample_size"
    PARAM_MAX_CARD = "max_cardinality"
    PARAM_MIN_CARD = "min_cardinality"
    PARAM_NUM_CARD = "num_card"
    PARAM_FAST_APPROX = "fast_approx"
    PARAM_FEATURE_SUMMARIES = "feature_summaries"
    PARAM_FEATURE_NAME = "feature_name"
    PARAM_NAME = "name"

    _display_name = "Disparate Impact Analysis"
    _description = (
        "Disparate Impact Analysis (DIA) is a technique that is used to evaluate "
        "fairness. Bias can be introduced "
        "to models during the process of collecting, processing, and labeling data as "
        "a result, it is important to determine whether a model is harming certain "
        "users by making a significant number of biased decisions. DIA typically "
        "works by comparing aggregate measurements of unprivileged groups to "
        "a privileged group. For instance, the proportion of the unprivileged group "
        "that receives the potentially harmful outcome is divided by the proportion "
        "of the privileged group that receives the same outcome - the resulting "
        "proportion is then used to determine whether the model is biased."
    )
    _iid = True
    _time_series = True
    _regression = True
    _binary = True
    _multiclass = False
    _global_explanation = True
    _local_explanation = False
    _explanation_types = [e10s.DiaExplanation]
    _optional_explanation_types = [e10s.GlobalHtmlFragmentExplanation]
    _parameters = [
        explainers.ExplainerParam(
            param_name=PARAM_FEATURES,
            description="List of features for which to compute DIA.",
            param_type=commons.ExplainerParamType.list,
            default_value=None,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_CUT_OFF,
            description="Cut off.",
            param_type=commons.ExplainerParamType.float,
            default_value=0.0,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_MAXIMIZE_METRIC,
            description="Maximize metric.",
            param_type=commons.ExplainerParamType.str,
            default_value="F1",
            predefined=["F1", "F05", "F2", "MCC"],
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_MAX_CARD,
            description="Max cardinality for categorical variables.",
            param_type=commons.ExplainerParamType.int,
            default_value=10,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_MIN_CARD,
            description="Minimum cardinality for categorical variables.",
            param_type=commons.ExplainerParamType.int,
            default_value=2,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_NUM_CARD,
            description=(
                "Max cardinality for numeric variables to be considered categorical."
            ),
            param_type=commons.ExplainerParamType.int,
            default_value=25,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
    ]
    _keywords = [
        explainers.Explainer.KEYWORD_DEFAULT,
        explainers.Explainer.KEYWORD_EXPLAINS_FAIRNESS,
        explainers.Explainer.KEYWORD_H2O_SONAR,
    ]
    _requires_preloaded_predictor = False
    _priority = 50.0

    @staticmethod
    def is_enabled() -> bool:
        return True

    def __init__(self):
        explainers.Explainer.__init__(self)

        self.args = None
        self.actual_col: str | None = None
        self.predict_col: str | None = None
        self.labels: list | None = None
        self.maximize_metric: str | None = None
        self.dia_cols = None
        self.weight_col = None

        self.log_name = "Disparate Impact Analysis"

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

        self.args = DiaArgs(self._parameters)
        self.args.resolve_params(
            explainer_params=DiaArgs.json_str_to_dict(self.explainer_params_as_str),
        )

        self.log_name = f"Disparate Impact Analysis {self.mli_key}/{self.key}"

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

        # check whether there are any categorical columns are available for DIA
        try:
            self.args = DiaArgs(self._parameters)
            self.args.resolve_params(
                explainer_params=DiaArgs.json_str_to_dict(self.explainer_params_as_str),
            )

            group_columns = dia.DIA.prepare_dia_features(
                dataset=None,
                dataset_meta=self.dataset_meta,
                dia_cols=[],
                target_column=params.target_col,
                predict_column=params.prediction_col,
                max_cardinality=self.args.get(DiaExplainer.PARAM_MAX_CARD),
                min_cardinality=self.args.get(DiaExplainer.PARAM_MIN_CARD),
                max_numeric_cardinality=self.args.get(DiaExplainer.PARAM_NUM_CARD),
                model_meta=explainer_params.get("model_meta", None),
                logger=self.logger,
            )
            self.logger.debug(
                f"{self.log_name} compatibility check identified group columns: "
                f"{group_columns}"
            )
        except ValueError as ex:
            raise errors.ExplainerCompatibilityError(
                f"Unable to gather categorical features to be explained: {ex}"
            )

        return True

    def explain(
        self,
        X: datatable.Frame,
        y: datatable.Frame | None = None,
        explanations_types: list = None,
        **kwargs,
    ):
        # logic for maximize metric
        self.maximize_metric = self.get_max_metric()

        y_uniques = datatable.unique(y).to_list()[0]
        if not self.labels and len(y_uniques) == NUM_CLASSES:
            self.labels = y_uniques
        self.actual_col = y.names[0]
        self.predict_col = "predictions"

        dia_entity = dia.explain_dia(
            model=self.model,
            dataset=X,
            dataset_meta=self.dataset_meta,
            classes=len(self.labels) if self.labels else 1,
            actual_col=self.actual_col,
            predict_column=self.predict_col,
            labels=self.labels,
            path=self.persistence.get_explainer_working_dir(),
            cut_off=self.args.get(DiaExplainer.PARAM_CUT_OFF),
            maximize_metric=self.args.get(DiaExplainer.PARAM_MAXIMIZE_METRIC),
            max_cardinality=self.args.get(DiaExplainer.PARAM_MAX_CARD),
            min_cardinality=self.args.get(DiaExplainer.PARAM_MIN_CARD),
            max_numeric_cardinality=self.args.get(DiaExplainer.PARAM_NUM_CARD),
            dia_cols=self.dia_cols,
            weight_col=self.weight_col,
            parameters=self.args.args,
            dia_entity=dia.DisparateImpactAnalysis(
                key=self.mli_key,  # Driverless AI GoM expect explainer key to be i. key
                name=self.display_name,
                mli_key=self.mli_key,
                path=self.persistence.get_explainer_working_dir(),
                problem_type="",
                summary=dia.DisparateImpactAnalysisSummary("", -1.0, -1.0, -1.0),
                feature_summaries=[],
                global_conf_matrix=dia.DisparateImpactAnalysisNumericTable(
                    "", [], [], []
                ),
            ),
            logger=self.logger,
        )
        dia_entity_path = self.persistence.get_explainer_working_file(
            dia.DIA_ENTITY_FILE
        )
        with open(dia_entity_path, mode="w") as json_file:
            json_file.write(json.dumps(dia_entity.dump(), indent=4))

        # preserve DIA work/ directory - DIA has no Grammar of MLI representation
        explanations = list()

        # explanation: DIA TEXT (legacy API)
        dia_explanation = e10s.DiaExplanation(
            explainer=self,
            display_name=DiaExplainer._display_name,
            display_category=e10s.DiaExplanation.DISPLAY_CAT_DAI_MODEL,
        )
        dia_explanation.add_format(
            formats.DiaTextFormat(
                explanation=dia_explanation,
                format_data="DIA representation.",
                persistence=self.persistence.store,
            )
        )
        explanations.append(dia_explanation)

        # OPTIONAL explanation: HTML fragment
        if self.config and self.config.create_html_representations:
            try:
                dia_html = e10s.GlobalHtmlFragmentExplanation(
                    explainer=self,
                    display_name=DiaExplainer._display_name,
                    display_category=e10s.DiaExplanation.DISPLAY_CAT_DAI_MODEL,
                )
                dia_html.add_format(self._explain_html(dia_html, dia_entity))

                explanations.append(dia_html)
            except Exception as ex:
                self.logger.warning(
                    f"{self.log_name}: HTML representation creation failed: {ex}"
                )

        return explanations

    def _explain_html(
        self,
        explanation: e10s.GlobalHtmlFragmentExplanation,
        dia_entity: dia.DIA,
    ) -> formats.HtmlFormat:
        """Build HTML representation."""
        html_format = formats.HtmlFormat(
            explanation=explanation,
            format_data=formats.HtmlFormat.MINIMAL_HTML,
            persistence=self.persistence.store,
        )

        html_src = airium.Airium()

        if dia_entity and dia_entity.feature_summaries:
            result = self.get_result()
            feature_name_to_key = {}
            for i, summary in enumerate(dia_entity.feature_summaries):
                feature_name = summary.feature_name.name
                feature_name_to_key[feature_name] = i
                with html_src.b():
                    html_src(f"Fairness metrics for the feature: {feature_name}")
                with html_src.div():
                    # 1st/default reference level
                    # ref_level = summary.ref_levels[0] if summary.ref_levels else ""
                    for metric in [
                        "N",
                        "Adverse Impact",
                        "Accuracy",
                        "True Positive Rate",
                        "Precision",
                        "Specificity",
                        "Negative Predicted Value",
                        "False Positive Rate",
                        "False Discovery Rate",
                        "False Negative Rate",
                        "False Omissions Rate",
                    ]:
                        try:
                            file_names = result.plot(
                                feature_name=feature_name,
                                metrics_of_interest=metric,
                                file_path=self.persistence.get_explanation_file_path(
                                    explanation_type=explanation.explanation_type(),
                                    explanation_format=html_format.mime,
                                    explanation_file=(
                                        f"dia-{i}-{metric.lower().replace(' ', '_')}"
                                        f".png"
                                    ),
                                ),
                            )
                            img_path = file_names[0]

                            html_src.img(
                                src=self.persistence.get_relative_path(img_path),
                                alt=f"{metric} for feature '{feature_name}'",
                                # ensure that image will not overflow enclosing <div/>
                                style=(
                                    "height: 100%; max-width: 100%; display: block; "
                                    "margin: auto;"
                                ),
                            )
                            html_src.br()
                        except Exception as ex:
                            self.logger.warning(
                                f"Skipping rendering of '{metric}' - unable to plot "
                                f" '{metric}': {ex}"
                            )

            html_format.update_data(
                str(html_src),
                f"{persistences.ExplainerPersistence.FILE_EXPLANATION}.html",
            )

        return html_format

    def get_max_metric(self):
        if self.args.get(DiaExplainer.PARAM_CUT_OFF):
            return "Custom Cut Off"
        elif self.args.get(DiaExplainer.PARAM_MAXIMIZE_METRIC):
            return self.args.get(DiaExplainer.PARAM_MAXIMIZE_METRIC)
        else:
            return "F1"

    @staticmethod
    def get_entry_constants():
        return results.DiaResult.DiaEntryConstant(
            dia_entity_file=dia.DIA_ENTITY_FILE,
            param_feature_summaries=DiaExplainer.PARAM_FEATURE_SUMMARIES,
            param_feature_name=DiaExplainer.PARAM_FEATURE_NAME,
            param_name=DiaExplainer.PARAM_NAME,
            param_features=DiaExplainer.PARAM_FEATURES,
            ref_levels="ref_levels",
        )

    def get_result(self) -> results.DiaResult:
        return results.DiaResult(
            persistence=self.persistence,
            h2o_sonar_config=self.config,
            dia_entry_constants=self.get_entry_constants(),
            explainer_id=DiaExplainer.explainer_id(),
            logger=self.logger,
        )


class DiaArgs(explainers.ExplainerArgs):
    def __init__(self, parameters: list[explainers.ExplainerParam] = None):
        explainers.ExplainerArgs.__init__(self, parameters)

    def resolve_params(
        self,
        explainer_params: dict | None = None,
        erase: list[str] | None = None,
    ):
        explainers.ExplainerArgs.resolve_params(
            self,
            explainer_params=explainer_params,
        )
