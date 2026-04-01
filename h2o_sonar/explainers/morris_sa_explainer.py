# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.


import datatable
import numpy
import pandas

from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api import results
from h2o_sonar.utils import problem_detection


HAS_INTERPRET = True
try:
    from interpret import blackbox
except ImportError:
    HAS_INTERPRET = False


# Explainer MUST extend abstract Explainer class to be discovered and registered.
# In addition, it inherits common metadata and (default) functionality. The explainer
# must implement explain() method (others are optional). Explainer class provides
# easy access/handle to the dataset and model (metadata and artifacts), filesystem,
# and common utilities.
class MorrisSensitivityAnalysisExplainer(explainers.Explainer):
    """InterpretML: Morris sensitivity analysis explainer."""

    PARAM_LEAKAGE_WARN_THRESHOLD = "leakage_warning_threshold"

    # explainer display name (used e.g. in explainer listing)
    _display_name = "Morris Sensitivity Analysis"
    # explainer description (used e.g. in explanations help)
    _description = (
        "Morris sensitivity analysis (SA) explainer provides Morris sensitivity "
        "analysis based feature importance which is a measure of the contribution of "
        "an input variable to the overall predictions of the model. In applied "
        "statistics, the Morris method for global sensitivity analysis is a so-called "
        "one-step-at-a-time method (OAT), meaning that in each run only one input "
        "parameter is given a new value. This explainer is based based on InterpretML "
        "library - see http://interpret.ml"
    )
    _iid = True
    # declaration of supported experiments: regression / binary / multiclass
    _regression = True
    _binary = True
    # declaration of provided explanations: global, local or both
    _global_explanation = True
    # declaration of explanation types this explainer creates e.g. feature importance
    _explanation_types = [e10s.GlobalFeatImpExplanation]
    # optional explanation types
    _optional_explanation_types = [e10s.GlobalHtmlFragmentExplanation]
    # required Python package dependencies (can be installed using pip)
    #   pip install interpret==0.1.20 gevent==1.5.0
    _modules_needed_by_name = ["gevent==1.5.0", "interpret==0.1.20"]
    # keywords
    _keywords = [
        explainers.Explainer.KEYWORD_EXPLAINS_O_FEATURE_IMPORTANCE,
        explainers.Explainer.KEYWORD_H2O_SONAR,
    ]
    _parameters = [
        explainers.ExplainerParam(
            param_name=PARAM_LEAKAGE_WARN_THRESHOLD,
            description=(
                "The threshold above which to report a potentially detected"
                "feature importance leak problem."
            ),
            param_type=commons.ExplainerParamType.float,
            default_value=0.95,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
    ]

    # explainer constructor must not have any required parameters
    def __init__(self):
        explainers.Explainer.__init__(self)

        self.log_name = MorrisSensitivityAnalysisExplainer._display_name
        self.args = None

    # compatibility check verifies whether it's possible to run the explainer
    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        model: models.ExplainableModel | None = None,
        **explainer_params,
    ) -> bool:
        if not HAS_INTERPRET:
            self.logger.warning(self._check_compatibility_pckg_err_msg("interpret"))
            return False

        explainers.Explainer.check_compatibility(self, params, **explainer_params)

        if not self.check_required_modules(
            set(MorrisSensitivityAnalysisExplainer._modules_needed_by_name)
        ):
            self.logger.warning(
                f"{self.log_name} not compatible as the following required Python "
                f"modules are not installed: "
                f"{MorrisSensitivityAnalysisExplainer._modules_needed_by_name}"
            )
            return False

        if model.model_type == models.ExplainableModelType.h2o3:
            self.logger.warning(
                f"{self.log_name} not compatible/does not explain H2O-3 models "
            )
            return False

        return True

    # setup() method is used to initialize the explainer based on provided parameters
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

        self.args = explainers.ExplainerArgs(self._parameters)
        self.args.resolve_params(
            explainer_params=explainers.ExplainerArgs.json_str_to_dict(
                self.explainer_params_as_str
            ),
        )

    # explain() method creates the explanations
    def explain(
        self,
        X,
        explainable_x: datasets.ExplainableDataset | None = None,
        y=None,
        **kwargs,
    ) -> list:
        # DATASET: encoding of categorical features for 3rd party library which
        # support numeric features only, filtering of rows w/ missing values, and more
        (x, _, self.model.label_encoder, _) = explainable_x.prepare(
            le_cat_variables=True,
            used_features=self.model.meta.used_features,
            cleaned_frame_type=pandas.DataFrame,
        )

        # PREDICT FUNCTION: library compliant predict function which works on numpy
        def predict_function(dataset: numpy.ndarray):
            # score
            preds = self.model.predict(
                datasets.ExplainableDataset.frame_2_datatable(
                    dataset,
                    columns=self.model.meta.used_features,
                )
            )

            # scoring output conversion to the frame type required by the library
            return datasets.ExplainableDataset.frame_2_numpy(preds, flatten=True)

        # CALCULATION of the Morris SA explanation
        sensitivity: blackbox.MorrisSensitivity = blackbox.MorrisSensitivity(
            predict_function, data=x, feature_names=list(x.columns)
        )
        self.logger.debug(sensitivity)
        morris_explanation = sensitivity.explain_global(name=self.display_name)

        problem_list: list[explainers.problems.ProblemAndAction] = []
        try:
            problem_list.extend(self._calculate_problems(morris_explanation))
        except Exception:
            self.logger.error("Failed to generate problems.")

        if len(problem_list) > 0:
            for p in problem_list:
                self.add_problem(p)

        # NORMALIZATION of proprietary Morris SA library data to standard format
        global_explanation = self._normalize_to_gom(morris_explanation)
        explanations = [global_explanation]

        # OPTIONAL explanation: HTML fragment
        if self.config and self.config.create_html_representations:
            try:
                display_name = MorrisSensitivityAnalysisExplainer._display_name
                explanations.append(
                    e10s.GlobalHtmlFragmentExplanation.from_explanation(
                        explainer=self,
                        explanation=global_explanation,
                        display_name=display_name,
                        display_category=e10s.Explanation.DISPLAY_CAT_DAI_MODEL,
                        data_as_text=False,
                        logger=self.logger,
                    )
                )
            except Exception as ex:
                self.logger.warning(
                    f"{self.log_name}: HTML representation creation failed: {ex}"
                )

        # explainer MUST return explanation(s) declared in _explanation_types
        return explanations

    def _calculate_problems(
        self, morris_explanation
    ) -> list[explainers.problems.ProblemAndAction]:
        fi_scores: list[float] = list(morris_explanation.data()["scores"])
        fi_scores_cols: list[str] = morris_explanation.data()["names"]
        fi_frame: datatable.Frame = datatable.Frame(
            numpy.atleast_2d([fi_scores]), names=fi_scores_cols
        )
        fi_frame_dict: dict[str : datatable.Frame] = {"label": fi_frame}

        return problem_detection.get_feature_importance_problems(
            fi_frame_dict,
            self.args.get(
                MorrisSensitivityAnalysisExplainer.PARAM_LEAKAGE_WARN_THRESHOLD
            ),
            self.explainer_id(),
            self._display_name,
        )

    #
    # optional NORMALIZATION to Grammar of MLI (GoM)
    #

    # Normalization of the data to the Grammar of MLI defined format. Normalized data
    # can be visualized using H2O Sonar components.
    #
    # This method creates explanation (data) and its representations (JSon, datatable)
    def _normalize_to_gom(self, morris_explanation) -> e10s.GlobalFeatImpExplanation:
        # EXPLANATION
        explanation = e10s.GlobalFeatImpExplanation(
            explainer=self,
            display_name=self.display_name,
            display_category=e10s.GlobalFeatImpExplanation.DISPLAY_CAT_CUSTOM,
        )

        # FORMAT: explanation representation as JSon+datatable - JSon index file which
        # references datatable frame with sensitivity values for each class
        jdf = f5s.GlobalFeatImpJSonDatatableFormat
        # data normalization: 3rd party frame to Grammar of MLI defined frame
        # conversion - see GlobalFeatImpJSonDatatableFormat docstring for format
        # documentation and source for helpers to create the representation easily
        explanation_frame = datatable.Frame(
            {
                jdf.COL_NAME: morris_explanation.data()["names"],
                jdf.COL_IMPORTANCE: list(morris_explanation.data()["scores"]),
                jdf.COL_GLOBAL_SCOPE: [True] * len(morris_explanation.data()["scores"]),
            }
        ).sort(-datatable.f[jdf.COL_IMPORTANCE])
        # index file (of per-class data files) (JSon)
        (
            idx_dict,
            idx_str,
        ) = f5s.GlobalFeatImpJSonDatatableFormat.serialize_index_file(
            classes=["global"],
            doc=MorrisSensitivityAnalysisExplainer._description,
        )
        json_dt_format = f5s.GlobalFeatImpJSonDatatableFormat(explanation, idx_str)
        json_dt_format.update_index_file(
            idx_dict, total_rows=explanation_frame.shape[0]
        )
        # data file (datatable)
        json_dt_format.add_data_frame(
            format_data=explanation_frame,
            file_name=idx_dict[jdf.KEY_FILES]["global"],
        )
        # JSon+datatable format can be added as explanation's representation
        explanation.add_format(json_dt_format)

        # another FORMAT: explanation representation as JSon
        # Having JSon+datatable formats it's easy to get other formats like CSV,
        # datatable, ZIP, ... using helpers - adding JSon representation:
        explanation.add_format(
            explanation_format=f5s.GlobalFeatImpJSonFormat.from_json_datatable(
                json_dt_format
            )
        )

        return explanation

    def get_result(
        self,
    ) -> results.FeatureImportanceResult:
        return results.FeatureImportanceResult(
            persistence=self.persistence,
            explainer_id=MorrisSensitivityAnalysisExplainer.explainer_id(),
            chart_title=MorrisSensitivityAnalysisExplainer._display_name,
            h2o_sonar_config=self.config,
            logger=self.logger,
        )
