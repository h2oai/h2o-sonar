# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.


import datatable

from h2o_sonar import errors
from h2o_sonar.explainers import pd_ice_explainer
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api import problems
from h2o_sonar.lib.api import results


PdExplainer = pd_ice_explainer.PdIceExplainer


class ResidualPdIceExplainer(explainers.Explainer):
    """Residual PD/ICE explainer."""

    # explainer parameters
    PARAM_FEATURES = PdExplainer.PARAM_FEATURES
    PARAM_MAX_FEATURES = PdExplainer.PARAM_MAX_FEATURES
    PARAM_GRID_RESOLUTION = PdExplainer.PARAM_GRID_RESOLUTION
    PARAM_OOR_GRID_RESOLUTION = PdExplainer.PARAM_OOR_GRID_RESOLUTION
    PARAM_CENTER = PdExplainer.PARAM_CENTER
    PARAM_SORT_BINS = PdExplainer.PARAM_SORT_BINS
    PARAM_QTILE_GRID_RESOLUTION = PdExplainer.PARAM_QTILE_GRID_RESOLUTION
    PARAM_QTILE_BINS = PdExplainer.PARAM_QTILE_BINS
    PARAM_HISTOGRAMS = PdExplainer.PARAM_HISTOGRAMS
    PARAM_SAMPLE_SIZE = PdExplainer.PARAM_SAMPLE_SIZE
    PARAM_NUMCAT_NUM_CHART = PdExplainer.PARAM_NUMCAT_NUM_CHART
    PARAM_NUMCAT_THRESHOLD = PdExplainer.PARAM_NUMCAT_THRESHOLD
    PARAM_DEBUG_RESIDUALS = PdExplainer.PARAM_DEBUG_RESIDUALS

    _display_name = f"Residual {PdExplainer._display_name}"
    _description = (
        "The residual partial dependence plot (PDP) indicates which variables interact "
        "most with the error. Residuals are transformed differences between observed "
        "and predicted values: the square of the difference between observed and "
        "predicted values is used in case of regression problems; -1 * log(p) "
        "is used in case of classification problems. The residual partial "
        "dependence is created using normal partial dependence algorithm, while "
        "instead of prediction is used the residual. "
        "Individual Conditional Expectations plot (ICE) displays the interaction with "
        "error for an individual row of data when an input variable is toggled across "
        "its domain."
    )
    _iid = True
    _time_series = PdExplainer._time_series
    _regression = PdExplainer._regression
    _binary = PdExplainer._binary
    _multiclass = PdExplainer._multiclass
    _global_explanation = PdExplainer._global_explanation
    _local_explanation = PdExplainer._local_explanation
    _explanation_types = PdExplainer._explanation_types
    _optional_explanation_types = PdExplainer._optional_explanation_types
    _parameters = [
        p
        for p in PdExplainer._parameters
        if p.param_name != PdExplainer.PARAM_DEBUG_RESIDUALS
    ]
    _keywords = [
        e10s.PartialDependenceExplanation.KEYWORD_CAN_ADD_FEATURE,
        explainers.Explainer.KEYWORD_EXPLAINS_MODEL_DEBUGGING,
        explainers.Explainer.KEYWORD_H2O_SONAR,
    ]
    _requires_preloaded_predictor = PdExplainer._requires_preloaded_predictor

    def __init__(self):
        explainers.Explainer.__init__(self)

        # PD/ICE explainer
        self.pd_explainer = PdExplainer()

        self.args = None
        # sanitized stringified labels
        self.labels = None
        self.log_name = "Residual PD/ICE"
        self.pd_explainer.log_name = self.log_name

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        model: models.ExplainableModel | None = None,
        **explainer_params,
    ) -> bool:
        explainers.Explainer.check_compatibility(self, params, **explainer_params)

        self.pd_explainer.config = self.config

        if not self.pd_explainer.check_compatibility(
            params=params,
            model=model,
            **explainer_params,
        ):
            raise errors.ExplainerCompatibilityError(
                f"{self.log_name} not compatible with the model and/or dataset"
            )

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

        self.pd_explainer.setup(
            model=model,
            persistence=persistence,
            key=key,
            params=params,
            **explainer_params,
        )
        self.log_name: str = f"Residual PD/ICE {self.mli_key}/{self.key}"
        self.pd_explainer.log_name = self.log_name

        self.args = explainers.ExplainerArgs(self._parameters)
        self.args.resolve_params(
            explainer_params=explainers.ExplainerArgs.json_str_to_dict(
                self.explainer_params_as_str
            ),
        )
        # ensure residuals PD/ICE calculation
        self.args.args[PdExplainer.PARAM_DEBUG_RESIDUALS] = True
        self.pd_explainer.args = self.args

    def explain(
        self,
        X: datatable.Frame,
        y: datatable.Frame | None = None,
        explanations_types: list = None,
        **explainer_params,
    ) -> list:
        # preprocess target colum to ensure that it will be numeric in case of B and M
        (X, y) = self._explain_prepare_x(x=X, y=y)

        explanations = self.pd_explainer.explain(
            X=X,
            y=y,
            explanations_types=explanations_types,
            **explainer_params,
        )
        self._problems = self.pd_explainer.explain_problems()

        return explanations

    def _explain_prepare_x(
        self,
        x: datatable.Frame,
        y: datatable.Frame | None = None,
    ):
        if (
            x[:, self.params.target_col].ltypes[0]
            not in [
                datatable.ltype.int,
                datatable.ltype.real,
            ]
            and self.model_meta
        ):
            if x[:, self.params.target_col].ltypes[0] in [
                datatable.ltype.bool,
                datatable.ltype.str,
            ]:
                if self.model_meta.num_labels == 2:
                    y = x[
                        :,
                        datatable.ifelse(
                            datatable.f[self.params.target_col]
                            == self.model_meta.positive_label_of_interest,
                            1,
                            0,
                        ),
                    ]
                    x[:, self.params.target_col] = y
                elif self.model_meta.num_labels > 2:
                    # residuals are calculated for all classes - OHE encode labels
                    y = datatable.Frame()
                    for label in self.model_meta.labels:
                        y_col = x[
                            :,
                            datatable.ifelse(
                                datatable.f[self.params.target_col] == label,
                                1,
                                0,
                            ),
                        ]
                        y_col.names = [label]
                        y.cbind(y_col)

        return x, y

    def explain_problems(self) -> list[problems.ProblemAndAction]:
        self._problems = self.pd_explainer.explain_problems()
        for p in self._problems:
            p.explainer_id = self.explainer_id()
            p.explainer_name = self._display_name

        return self._problems

    def get_result(self) -> results.PdResult:
        return results.PdResult(
            persistence=self.persistence,
            h2o_sonar_config=self.config,
            explainer_id=ResidualPdIceExplainer.explainer_id(),
            logger=self.logger,
        )
