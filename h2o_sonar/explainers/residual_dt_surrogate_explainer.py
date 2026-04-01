# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import datatable

from h2o_sonar import errors
from h2o_sonar.explainers import dt_surrogate_explainer
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api import problems
from h2o_sonar.lib.api import results


# shortcuts
DtExplainer = dt_surrogate_explainer.DecisionTreeSurrogateExplainer


class ResidualDecisionTreeSurrogateExplainer(explainers.Explainer):
    """Residual Decision tree surrogate explainer."""

    # Implementation:
    # - implemented as DT surrogate facade w/ fixed parameters

    # explainer parameters
    PARAM_DEBUG_RESIDUALS = DtExplainer.PARAM_DEBUG_RESIDUALS
    PARAM_DEBUG_RESIDUALS_CLASS = DtExplainer.PARAM_DEBUG_RESIDUALS_CLASS
    PARAM_DT_DEPTH = DtExplainer.PARAM_DT_DEPTH
    PARAM_NFOLDS = DtExplainer.PARAM_NFOLDS
    PARAM_QBIN_COLS = DtExplainer.PARAM_QBIN_COLS
    PARAM_QBIN_COUNT = DtExplainer.PARAM_QBIN_COUNT
    PARAM_CAT_ENCODING = DtExplainer.PARAM_CAT_ENCODING

    _display_name = f"Residual {DtExplainer._display_name}"
    _description = (
        "The residual surrogate decision tree predicts which paths in the tree "
        "(paths explain approximate model behavior) lead to highest or "
        "lowest error. The residual surrogate decision tree is created by training a "
        "simple decision tree on the residuals of the predictions of the model. "
        "Residuals are differences between observed and predicted values which can be "
        "used as targets in surrogate models for the purpose of model debugging. "
        "The method used to calculate residuals varies depending on the type of "
        "problem. For classification problems, logloss residuals are calculated for "
        "a specified class (only one residual surrogate decision is created by the "
        "explainer and it is built for this class). For regression problems, residuals "
        "are determined by calculating the square of the difference between targeted "
        "and predicted values."
    )
    _iid = True
    _regression = DtExplainer._regression
    _binary = DtExplainer._binary
    _multiclass = DtExplainer._multiclass
    _time_series = DtExplainer._time_series
    _global_explanation = DtExplainer._global_explanation
    _local_explanation = DtExplainer._local_explanation
    _explanation_types = DtExplainer._explanation_types
    _optional_explanation_types = DtExplainer._optional_explanation_types
    _parameters = [
        p
        for p in DtExplainer._parameters
        if p.param_name != DtExplainer.PARAM_DEBUG_RESIDUALS
    ]
    _requires_predict_method = DtExplainer._requires_predict_method
    _priority = 12.0
    _keywords = [
        explainers.Explainer.KEYWORD_DEFAULT,
        explainers.Explainer.KEYWORD_REQUIRES_H2O3,
        explainers.Explainer.KEYWORD_EXPLAINS_MODEL_DEBUGGING,
        explainers.SurrogateExplainer.KEYWORD_SURROGATE,
        explainers.Explainer.KEYWORD_H2O_SONAR,
    ]
    _requires_preloaded_predictor = DtExplainer._requires_predict_method

    @staticmethod
    def is_enabled() -> bool:
        return True

    def __init__(self):
        explainers.Explainer.__init__(self)

        # surrogate decision tree
        self.sdt = DtExplainer()

        self.args = None
        # sanitized stringified labels
        self.labels = None
        self.log_name = ResidualDecisionTreeSurrogateExplainer._display_name
        self.sdt.log_name = self.log_name

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        model: models.ExplainableModel | None = None,
        **explainer_params,
    ) -> bool:
        explainers.Explainer.check_compatibility(self, params, **explainer_params)

        self.sdt.config = self.config

        return self.sdt.check_compatibility(
            params=params,
            model=model,
            **explainer_params,
        )

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

        self.sdt.setup(
            model=model,
            persistence=persistence,
            key=key,
            params=params,
            **explainer_params,
        )
        self.log_name: str = (
            f"{ResidualDecisionTreeSurrogateExplainer._display_name} "
            f"{self.mli_key}/{self.key}"
        )
        self.sdt.log_name = self.log_name

        self.args = explainers.ExplainerArgs(self._parameters)
        self.args.resolve_params(
            explainer_params=explainers.ExplainerArgs.json_str_to_dict(
                self.explainer_params_as_str
            ),
        )
        # ensure residuals DT calculation
        self.args.args[DtExplainer.PARAM_DEBUG_RESIDUALS] = True
        self.sdt.args = self.args
        # classification: ensure valid residuals class in case of classification
        if model and model.meta and model.meta.num_labels:
            residuals_class = self.args.get(DtExplainer.PARAM_DEBUG_RESIDUALS_CLASS, "")
            if not residuals_class:
                # default ~ class not specified > set a valid class
                self.logger.warning(
                    f"{self.log_name} setting default residuals debug class..."
                )
                if model.meta.num_labels == 2:
                    # positive class of interest
                    residuals_class = self.model.meta.labels[1]
                else:
                    residuals_class = self.model.meta.labels[0]

                self.args.args[DtExplainer.PARAM_DEBUG_RESIDUALS_CLASS] = (
                    residuals_class
                )

                self.logger.warning(
                    f"{self.log_name} residuals debug class set to '{residuals_class}'"
                )
            else:
                # validate residual class
                if residuals_class not in model.meta.labels:
                    raise errors.MliError(
                        f"{self.log_name} residuals debug class '{residuals_class}' "
                        f"is invalid - it must be one of '{model.meta.labels}'"
                    )

    def explain(
        self,
        X: datatable.Frame,
        y: datatable.Frame | None = None,
        explanations_types: list = None,
        **explainer_params,
    ) -> list:
        explanations = self.sdt.explain(
            X=X,
            y=y,
            explanations_types=explanations_types,
            **explainer_params,
        )
        self._problems = self.sdt.explain_problems()

        return explanations

    def explain_problems(self) -> list[problems.ProblemAndAction]:
        self._problems = self.sdt.explain_problems()
        for p in self._problems:
            p.explainer_id = self.explainer_id()
            p.explainer_name = self._display_name

        return self._problems

    def get_result(self) -> results.DtResult:
        return results.DtResult(
            persistence=self.persistence,
            h2o_sonar_config=self.config,
            explainer_name="residual surrogate decision tree explainer",
            explainer_id=ResidualDecisionTreeSurrogateExplainer.explainer_id(),
            highlight_highest_residual=self.sdt.args.get(
                DtExplainer.PARAM_DEBUG_RESIDUALS, False
            ),
            logger=self.logger,
        )
