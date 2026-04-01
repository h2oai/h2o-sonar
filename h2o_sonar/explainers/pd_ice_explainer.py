# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import json
import os
import traceback

import datatable
import numpy
import pandas

from h2o_sonar import errors
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api import problems
from h2o_sonar.lib.api import results
from h2o_sonar.methods import _ice
from h2o_sonar.methods import _pd
from h2o_sonar.methods.core import method
from h2o_sonar.methods.utils import histogram
from h2o_sonar.utils import progress
from h2o_sonar.utils import sanitization


# shortcuts
FeaturesMetadata = method.FeaturesMetadata
ModelTypes = models.ExplainableModelType

# Constants
# Missing value
MISSING_VALUE = "missing"


class PdIceExplainer(explainers.Explainer):
    """PD/ICE explainer."""

    # Driverless AI platform specific implementation notes:
    #
    # - fast approximation and MOJO enable/disable settings are removed in this
    #   portable implementation, however, these proprietary parameters will be
    #   injected by Driverless AI container: explainable model passed to the
    #   explainer will set the parameters to predict method by getting them
    #   from server and also explainer parameters
    #

    # explainer parameters
    PARAM_FEATURES = "features"
    PARAM_MAX_FEATURES = "max_features"
    PARAM_GRID_RESOLUTION = "grid_resolution"
    PARAM_OOR_GRID_RESOLUTION = "oor_grid_resolution"
    PARAM_CENTER = "center"
    PARAM_SORT_BINS = "sort_bins"
    PARAM_QTILE_GRID_RESOLUTION = "quantile-bin-grid-resolution"
    PARAM_QTILE_BINS = "quantile-bins"
    PARAM_HISTOGRAMS = "histograms"
    PARAM_SAMPLE_SIZE = "sample_size"
    PARAM_NUMCAT_NUM_CHART = "numcat_num_chart"
    PARAM_NUMCAT_THRESHOLD = "numcat_threshold"
    PARAM_DEBUG_RESIDUALS = "debug_residuals"

    # static option: enable/disable datatable 1-frame strategy optimization
    OPT_ICE_1_FRAME_ENABLED = True

    # global explainer update options
    UPDATE_SCOPE_NUMCAT = "numcat"
    UPDATE_PARAM_NUMCAT_OVERRIDE = "numcat_override"
    # global update types
    UPDATE_TYPE_ADD_FEATURE = "add_feature"
    UPDATE_TYPE_ADD_NUMCAT = "add_num_cat"

    # calculation progress reporting range (engine 0 - 10%, normalization 90% - 100%)
    PROGRESS_MIN = 0.1
    PROGRESS_MAX = 0.9

    # PD algorithm file names
    FILE_PD_JSON = "h2o_sonar-pd-dai-model.json"
    FILE_ICE_JSON = "h2o_sonar-ice-dai-model.json"
    FILE_Y_HAT = "mli_dataset_y_hat.jay"

    # limit on number of important features to use for PD/ICE computation
    MAX_FEATURES = 10
    # bin limit for PD/ICE computation (MLI-1 via H2O-3 PD is 20, MLI-2 is 10)
    GRID_RESOLUTION = 20
    SAMPLE_SIZE = 25000
    NUMCAT_NUM_CHART = True
    NUMCAT_THRESHOLD = 11

    _display_name = "Partial Dependence Plot"
    _description = """Partial dependence plot (PDP) portrays the average prediction
behavior of the model across the domain of an input variable along with +/- 1
standard deviation bands. Individual Conditional Expectations plot (ICE) displays
the prediction behavior for an individual row of data when an input variable is
toggled across its domain.

PD binning:

**Integer** feature:

* bins in **numeric** mode:
    * bins are integers
    * (at most) `grid_resolution` integer values in between minimum and maximum
      of feature values
    * bin values are created as evenly as possible
    * minimum and maximum is included in bins
      (if `grid_resolution` is bigger or equal to 2)
* bins in **categorical** mode:
    * bins are integers
    * top `grid_resolution` values from feature values ordered by frequency
      (int values are converted to strings and most frequent values are used
      as bins)
* quantile bins in **numeric** mode:
    * bins are integers
    * bin values are created with the aim that there will be the same number of
      observations per bin
    * q-quantile used to created ``q`` bins where ``q`` is specified by PD parameter
* quantile bins in **categorical** mode:
    * not supported

**Float** feature:

* bins in **numeric** mode:
    * bins are floats
    * `grid_resolution` float values in between minimum and maximum of feature
       values
    * bin values are created as evenly as possible
    * minimum and maximum is included in bins
      (if `grid_resolution` is bigger or equal to 2)
* bins in **categorical** mode:
    * bins are floats
    * top `grid_resolution` values from feature values ordered by frequency
      (float values are converted to strings and most frequent values are used
      as bins)
* quantile bins in **numeric** mode:
    * bins are floats
    * bin values are created with the aim that there will be the same number of
      observations per bin
    * q-quantile used to created ``q`` bins where ``q`` is specified by PD parameter
* quantile bins in **categorical** mode:
    * not supported

**String** feature:

* bins in **numeric** mode:
    * not supported
* bins in **categorical** mode:
    * bins are strings
    * top `grid_resolution` values from feature values ordered by frequency
* quantile bins:
    * not supported

**Date/datetime** feature:

* bins in **numeric** mode:
    * bins are dates
    * `grid_resolution` date values in between minimum and maximum of feature
      values
    * bin values are created as evenly as possible:
        1. dates are parsed and converted to epoch timestamps i.e integers
        2. bins are created as in case of numeric integer binning
        3. integer bins are converted back to original date format
    * minimum and maximum is included in bins
      (if `grid_resolution` is bigger or equal to 2)
* bins in **categorical** mode:
    * bins are dates
    * top `grid_resolution` values from feature values ordered by frequency
      (dates are handled as opaque strings and most frequent values are used
      as bins)
* quantile bins:
    * not supported

PD out of range binning:

**Integer** feature:

* OOR bins in **numeric** mode:
    * OOR bins are integers
    * (at most) `oor_grid_resolution` integer values are added below minimum and
      above maximum
    * bin values are created by adding/substracting rounded standard deviation
      (of feature values) above and below maximum and minimum `oor_grid_resolution`
      times
        * 1 used used if rounded standard deviation would be 0
    * if feature is of unsigned integer type, then bins below 0
      are not created
        * if rounded standard deviation and/or `oor_grid_resolution` is so high
          that it would cause lower OOR bins to be negative numbers, then standard
          deviation of size 1 is tried instead
* OOR bins in **categorical** mode:
    * same as numeric mode

**Float** feature:

* OOR bins in **numeric** mode:
    * OOR bins are floats
    * `oor_grid_resolution` float values are added below minimum and above maximum
    * bin values are created by adding/substracting standard deviation
      (of feature values) above and below maximum and minimum `oor_grid_resolution`
      times
* OOR bins in **categorical** mode:
    * same as numeric mode

**String** feature:

* bins in **numeric** mode:
    * not supported
* bins in **categorical** mode:
    * OOR bins are strings
    * value `UNSEEN` is added as OOR bin

**Date** feature:

* bins in **numeric** mode:
    * not supported
* bins in **categorical** mode:
    * OOR bins are strings
    * value `UNSEEN` is added as OOR bin

"""
    _iid = True
    _time_series = True
    _regression = True
    _binary = True
    _multiclass = True
    _global_explanation = True
    _local_explanation = True
    _explanation_types = [
        e10s.PartialDependenceExplanation,
        e10s.IndividualConditionalExplanation,
    ]
    _optional_explanation_types = [e10s.GlobalHtmlFragmentExplanation]
    _parameters = [
        explainers.ExplainerParam(
            param_name=PARAM_SAMPLE_SIZE,
            description="Sample size for Partial Dependence Plot.",
            param_type=commons.ExplainerParamType.int,
            default_value=SAMPLE_SIZE,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_MAX_FEATURES,
            description=(
                "Partial Dependence Plot number of features (to see all"
                " features used by model set to -1)."
            ),
            param_type=commons.ExplainerParamType.int,
            default_value=MAX_FEATURES,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_FEATURES,
            description="Partial Dependence Plot feature list.",
            param_type=commons.ExplainerParamType.list,
            default_value=None,
            tags=[explainers.ExplainerParam.TAG_SRC_DATASET_COLUMN_NAMES],
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
            param_name=PARAM_CENTER,
            description="Center Partial Dependence Plot using ICE centered at 0.",
            param_type=commons.ExplainerParamType.bool,
            default_value=False,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_SORT_BINS,
            description="Ensure bin values sorting.",
            param_type=commons.ExplainerParamType.bool,
            default_value=True,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_HISTOGRAMS,
            description="Enable histograms.",
            param_type=commons.ExplainerParamType.bool,
            default_value=True,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_QTILE_BINS,
            description=(
                """Per-feature quantile binning (Example: if choosing features
                 F1 and F2, this parameter is '{"F1": 2,"F2": 5}'. Note, you can
                 set all features to use the same quantile binning with the
                 `Partial Dependence Plot quantile binning` parameter and then
                 adjust the quantile binning for a subset of PDP features with
                 this parameter)."""
            ),
            param_type=commons.ExplainerParamType.str,
            default_value="",
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_NUMCAT_NUM_CHART,
            description=(
                "Unique feature values count driven Partial Dependence Plot binning "
                "and chart selection."
            ),
            param_type=commons.ExplainerParamType.bool,
            default_value=NUMCAT_NUM_CHART,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_NUMCAT_THRESHOLD,
            description=(
                "Threshold for Partial Dependence Plot binning and chart selection "
                "(<=threshold categorical, >threshold numeric)."
            ),
            param_type=commons.ExplainerParamType.int,
            default_value=NUMCAT_THRESHOLD,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_DEBUG_RESIDUALS,
            description="Debug model residuals.",
            param_type=commons.ExplainerParamType.bool,
            default_value=False,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
    ]
    _keywords = [
        explainers.Explainer.KEYWORD_DEFAULT,
        e10s.PartialDependenceExplanation.KEYWORD_CAN_ADD_FEATURE,
        explainers.Explainer.KEYWORD_EXPLAINS_FEATURE_BEHAVIOR,
        explainers.Explainer.KEYWORD_H2O_SONAR,
    ]
    _requires_preloaded_predictor = False

    def __init__(self):
        explainers.Explainer.__init__(self)

        self.args = None
        self.is_on_demand_pd: bool = False

        # class > feature
        self.problematic_features: dict = {}

        self.log_name = "PD/ICE"

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

        # check: target column
        if not params.target_col:
            err_msg = (
                f"{self.log_name} explainer requires target column and therefore it is "
                f"not compatible with explainer run with parameters: {params.dump()}"
            )
            self.logger.warning(err_msg)
            raise errors.ExplainerCompatibilityError(err_msg)

        # check: at least 1 PD/ICE compatible feature
        try:
            self.args = PdIceArgs(self._parameters)
            self.args.resolve_params(
                explainer_params=PdIceArgs.json_str_to_dict(
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

        self.args = PdIceArgs(self._parameters)
        self.args.resolve_params(
            explainer_params=PdIceArgs.json_str_to_dict(self.explainer_params_as_str),
        )

        self.log_name = f"PD/ICE {self.mli_key}/{self.key}"

    def explain(
        self,
        X: datatable.Frame,
        y: datatable.Frame | None = None,
        explanations_types: list = None,
        **e_params,
    ):
        (pd, features_meta, is_sampling, dataset) = self._calculate_pd_ice(
            dataset=X,
            y=y,
            numcat_override=e_params.get(
                PdIceExplainer.UPDATE_PARAM_NUMCAT_OVERRIDE, ""
            ),
        )
        if not pd:
            err_msg = (
                f"{self.log_name}: no explainer result - no features for which to "
                f"compute PD"
            )
            self.logger.error(err_msg)
            raise errors.MliError(err_msg)

        # process raw PD/ICE algorithm output (from work/ directory) to explanations
        explanations = list()

        self.normalize_data(
            explanations=explanations,
            features_meta=features_meta,
            is_sampling=is_sampling,
            pd=pd,
            dataset=dataset,
        )

        return explanations

    def normalize_data(
        self,
        explanations,
        features_meta,
        is_sampling,
        pd=None,
        dataset: datatable.Frame | None = None,
        explainer_data_dir_path: str = "",
    ):
        if not pd and explainer_data_dir_path:
            pd = _pd.PD("PD normalization")
            pd.load_json(
                os.path.join(explainer_data_dir_path, PdIceExplainer.FILE_PD_JSON)
            )
            pd._fs = list(pd.explanations().keys())

        # RESIDUAL PD: method returns both PD and residual PD > PD will be rewritten
        # with residual PD
        if (
            self.args.get(PdIceExplainer.PARAM_DEBUG_RESIDUALS, False)
            and pd
            and pd.explanations()
        ):
            for feature in pd.explanations():
                for clazz in pd.explanations()[feature]:
                    pd_dict = pd.explanations()[feature][clazz].to_dict()

                    for bin_ in pd_dict:
                        bin_dict = pd_dict[bin_]
                        if _pd.PD.COL_RESIDUAL_MEAN in bin_dict:
                            bin_dict[_pd.PD.COL_MEAN] = bin_dict[
                                _pd.PD.COL_RESIDUAL_MEAN
                            ]
                            del bin_dict[_pd.PD.COL_RESIDUAL_MEAN]
                        if _pd.PD.COL_RESIDUAL_SD in bin_dict:
                            bin_dict[_pd.PD.COL_SD] = bin_dict[_pd.PD.COL_RESIDUAL_SD]
                            del bin_dict[_pd.PD.COL_RESIDUAL_SD]
                        if _pd.PD.COL_RESIDUAL_SEM in bin_dict:
                            bin_dict[_pd.PD.COL_SEM] = bin_dict[_pd.PD.COL_RESIDUAL_SEM]
                            del bin_dict[_pd.PD.COL_RESIDUAL_SEM]

                    pd.explanations()[feature][clazz] = pandas.DataFrame(pd_dict)

        # PD
        self.report_progress(PdIceExplainer.PROGRESS_MAX)
        pd_display_name = (
            "Partial Dependence Plot (PDP)"
            if not self.args.get(PdIceExplainer.PARAM_DEBUG_RESIDUALS, False)
            else "Residual Partial Dependence Plot"
        )
        try:
            pd_explanation = e10s.PartialDependenceExplanation(
                explainer=self,
                display_name=pd_display_name,
                display_category=e10s.Explanation.DISPLAY_CAT_DAI_MODEL,
            )
            (
                json_pd_representation,
                labels_2_pd_map,
            ) = self._work_pd_to_gom_pd_json(
                pd=pd,
                features_meta=features_meta,
                pd_explanation=pd_explanation,
                dataset=dataset,
            )
            pd_explanation.add_format(explanation_format=json_pd_representation)
            explanations.append(pd_explanation)
        except Exception as ex:
            self.logger.error(
                f"{self.log_name}: PD JSon representation creation failed for "
                f"{self.mli_key}/{self.key}: {ex}\n{traceback.format_exc()}",
            )
            # fail fast (no global normalized representations, local w/o global)
            raise ex

        # ICE
        self.report_progress(min(PdIceExplainer.PROGRESS_MAX + 5, 0.99))
        if pd_explanation:
            try:
                ice = e10s.IndividualConditionalExplanation
                ice_explanation = e10s.IndividualConditionalExplanation(
                    explainer=self,
                    display_name=(
                        "Individual Conditional Expectations (ICE)"
                        if not self.args.get(
                            PdIceExplainer.PARAM_DEBUG_RESIDUALS, False
                        )
                        else "Residual Individual Conditional Expectations (ICE)"
                    ),
                    display_category=ice.DISPLAY_CAT_DAI_MODEL,
                )
                json_ice_representation = self._work_ice_to_gom_ice_json(
                    ice_explanation=ice_explanation,
                    features_meta=features_meta,
                    pd=pd,
                    labels_2_pd_map=labels_2_pd_map,
                    is_sampling=is_sampling,
                    ice_cache_dir_path=explainer_data_dir_path,
                )
                ice_explanation.add_format(explanation_format=json_ice_representation)

                # make ICE PD's local explanations
                pd_explanation.has_local = ice_explanation.explanation_type()

                explanations.append(ice_explanation)
            except Exception as ex:
                self.logger.error(
                    f"{self.log_name}: ICE JSon/datatable representation creation "
                    f"failed for {self.mli_key}/{self.key}: "
                    f"{ex}\n{traceback.format_exc()}",
                )
                # don't fail fast as (at least) global explanation is available)

        # OPTIONAL explanation: HTML fragment
        if self.config and self.config.create_html_representations:
            try:
                t_explanation = e10s.PartialDependenceExplanation
                explanations.append(
                    e10s.GlobalHtmlFragmentExplanation.from_explanation(
                        explainer=self,
                        explanation=pd_explanation,
                        display_name=pd_display_name,
                        display_category=t_explanation.DISPLAY_CAT_DAI_MODEL,
                        problems=self.problematic_features,
                    )
                )
            except Exception as ex:
                self.logger.warning(
                    f"{self.log_name}: HTML representation creation failed: {ex}"
                )

        return explanations

    def explain_local(self, X, y=None, **e_params) -> list:
        """Execute load-from-cache or calculate on-demand ICE explanations.

        Parameters
        ----------
        X: datatable.Frame
          Dataset (whole) as datatable frame (handle).
        y: datatable.Frame
          Optional predictions as datatable frame (handle).

        Returns
        -------
        list[OnDemandExplanation]:
          On-demand explanations with single row ICE as JSon representation.

        """
        if not X:
            raise ValueError("Exactly one X row is required")
        row = e_params.get(explainers.OnDemandExplainKey.ROW, None)
        if row is None:
            raise ValueError(
                "Row (instance) is required to provide ICE local explanation."
            )
        if 1 != X.nrows:
            raise ValueError(
                "On-demand ICE: X dataset must have exactly 1 row for which is supposed"
                " to be computed local explanation"
            )
        feature = e_params.get(f5s.IceJsonDatatableFormat.FILTER_FEATURE, None)
        if not feature:
            raise ValueError(
                "At least one feature must be provided to provide ICE local explanation"
            )
        features = [feature]
        clazz = e_params.get(f5s.IceJsonDatatableFormat.FILTER_CLASS, None)
        if not clazz:
            raise ValueError(
                "At least one class must be provided to provide ICE local explanation"
            )

        numcat_aspect: str = ""
        do_debug_error: bool = False
        if e_params:
            numcat_aspect = e_params.get(f5s.IceJsonDatatableFormat.FILTER_NUMCAT, "")
            do_debug_error = e_params.get(PdIceExplainer.PARAM_DEBUG_RESIDUALS, False)

        if e_params.get(f5s.ExplanationFormat.KEY_ON_DEMAND_PARAMS, None):
            target_persistence = persistences.ExplainerPersistence(
                data_dir=self.persistence.data_dir,
                username=self.persistence.username,
                explainer_id=PdIceExplainer.explainer_id(),
                mli_key=e_params.get(explainers.OnDemandExplainKey.MLI_KEY),
                explainer_job_key=e_params.get(
                    explainers.OnDemandExplainKey.EXPLAINER_JOB_KEY
                ),
            )
            # bins: detect whether to load default bins OR aspect bins
            pd_idx_dict = f5s.PartialDependenceJSonFormat.load_index_file(
                persistence=target_persistence,
                explanation_type=e10s.PartialDependenceExplanation.explanation_type(),
            )
            if (
                not numcat_aspect
                or numcat_aspect
                == pd_idx_dict[f5s.PartialDependenceJSonFormat.KEY_FEATURES][feature][
                    f5s.PartialDependenceJSonFormat.KEY_FEATURE_TYPE
                ][0]
            ):
                # default bins
                bins_key = f5s.IceJsonDatatableFormat.KEY_BINS
            else:
                # aspect bins
                bins_key = f5s.IceJsonDatatableFormat.KEY_BINS_NUMCAT_ASPECT
            bins_dict = e_params[f5s.ExplanationFormat.KEY_ON_DEMAND_PARAMS][bins_key]
            bins = []
            for feature in features:
                bins.append(bins_dict[feature])
        else:
            raise ValueError(
                f"On-demand ICE: on-demand parameters does not contain required "
                f"parameter 'bins' ({e_params})"
            )

        # date-aware bins don't have to be constructed (got from PD via function arg)
        (_, features_meta, _) = self._resolve_model_features(self.model)

        ice_json: str = self._explain_ice_on_demand(
            x=X,
            y=y,
            features=features,
            features_meta=features_meta,
            clazz=clazz,
            bins=bins,
            base_dir_path=self.persistence.data_dir,
            labels=self.model_meta.labels,
            do_debug_error=do_debug_error,
        )

        explanation = e10s.OnDemandExplanation(explainer=self)
        explanation.add_format(
            f5s.LocalOnDemandTextFormat(explanation=explanation, format_data=ice_json)
        )

        self.logger.debug(f"On-demand DAI ICE explanation:\n{ice_json}")

        return [explanation]

    #
    # PD representation
    #

    def _work_pd_problems_f(self, cls_dict: dict):
        """Determine difference between maximum and minimum PD value for the feature."""
        pdj = f5s.PartialDependenceJSonFormat
        if self.args.get(PdIceExplainer.PARAM_DEBUG_RESIDUALS, False):
            if cls_dict and cls_dict.get(pdj.KEY_DATA):
                pds = [
                    b.get("pd", None)
                    for b in cls_dict.get(pdj.KEY_DATA)
                    if b.get("pd", None) is not None
                ]
                if pds:
                    return max(pds) - min(pds)

        return 0

    @classmethod
    def _replace_nan(cls, a_dict: dict):
        for key, val in a_dict.items():
            a_list_of_dicts = a_dict[key]
            for dic in a_list_of_dicts:
                for k, v in dic.items():
                    if isinstance(v, float) and numpy.isnan(v):
                        dic[k] = MISSING_VALUE

    def _work_pd_to_gom_pd_json(
        self,
        pd: _pd.PD,
        features_meta: dict,
        pd_explanation: e10s.PartialDependenceExplanation,
        dataset: datatable.Frame | None,
    ) -> tuple[f5s.PartialDependenceJSonFormat, dict]:
        """Conversion of raw DAI PD (work directory) assets to Grammar of MLI PD JSon
        representation.

        """
        if pd:
            exs = pd.explanations()
            pd_keys = list(exs[pd.features[0]].keys())
            num_classes: int = (
                len(self.model_meta.labels) if self.model_meta.labels else 1
            )
            pd_classes = pd_keys if num_classes > 2 else [pd_keys[0]]
            if self.model_meta.labels:
                classes = (
                    self.model_meta.labels
                    if num_classes > 2
                    else (
                        [f5s.PartialDependenceJSonFormat.LABEL_REGRESSION]
                        if num_classes == 1
                        else [self.model_meta.labels[1]]
                    )
                )
            elif num_classes == 1:
                classes = [f5s.PartialDependenceJSonFormat.LABEL_REGRESSION]
            else:
                classes = pd_classes

            # index file
            (
                index_dict,
                index_str,
            ) = f5s.PartialDependenceJSonFormat.serialize_index_file(
                features=pd.features,
                classes=classes,
                features_meta=features_meta,
                metrics=None,
                doc=PdIceExplainer._description,
            )
            json_representation = f5s.PartialDependenceJSonFormat(
                explanation=pd_explanation,
                json_data=index_str,
                persistence=self.persistence.store,
            )

            # problems identification (residual PD)
            problem_f_2_minmax = {}

            # data files
            pdj = f5s.PartialDependenceJSonFormat
            labels_2_pd_map = dict()
            for feature in pd.features:
                for i, pd_cls in enumerate(pd_classes):
                    # PD data
                    cls_dict: dict = self._pd_save_feature_json(exs[feature][pd_cls])
                    # histogram data: always try to compute cat and num variants
                    if self.args.get(PdIceExplainer.PARAM_HISTOGRAMS, False):
                        for cat_num in [True, False]:
                            self.logger.info(
                                f"{self.log_name} creating histogram: "
                                f"{feature}/{cat_num}",
                            )
                            self._inject_histogram_data(
                                data_cls_dict=cls_dict,
                                dataset=dataset,
                                feature=feature,
                                # per-class histogram ONLY for multi (r/g whole df)
                                clazz=classes[i] if num_classes > 2 else None,
                                is_categorical=cat_num,
                                target_col=self.params.target_col,
                                features_meta=features_meta,
                            )

                    problem_f_diff = self._work_pd_problems_f(cls_dict)
                    if problem_f_diff:
                        problem_f_2_minmax[feature] = (classes[i], problem_f_diff)
                    cls_dict_cpy = cls_dict.copy()
                    self._replace_nan(cls_dict_cpy)
                    format_data = json.dumps(cls_dict_cpy)
                    json_representation.add_data(
                        format_data=format_data,
                        file_name=index_dict[pdj.KEY_FEATURES][feature][pdj.KEY_FILES][
                            classes[i]
                        ],
                    )

                    labels_2_pd_map[classes[i]] = pd_cls

            # problems
            if problem_f_2_minmax:
                problem_feature = next(iter(problem_f_2_minmax.keys()))
                problem_class = problem_f_2_minmax[problem_feature][0]
                problem_max_pd = problem_f_2_minmax[problem_feature][1]
                self.problematic_features[problem_class] = []
                for f in problem_f_2_minmax:
                    if problem_f_2_minmax[f][1] > problem_max_pd:
                        problem_feature = f
                        problem_class = problem_f_2_minmax[problem_feature][0]
                        problem_max_pd = problem_f_2_minmax[problem_feature][1]

                if problem_feature:
                    self.problematic_features[problem_class].append(problem_feature)
                    class_msg = (
                        f"in class '{problem_class}' " if num_classes >= 2 else ""
                    )
                    problem = problems.ProblemAndAction(
                        description=(
                            f"The residual partial dependence plot of feature "
                            f"'{problem_feature}' {class_msg}"
                            f"indicates the highest interaction of this feature "
                            f"with the error (residual abs(max-min) ="
                            f" {problem_max_pd}) from all the features used by "
                            f"the model (or features configured for PD calculation)"
                        ),
                        severity=problems.ProblemSeverity.medium,
                        problem_type="bias",
                        problem_attrs={
                            problems.ProblemAndAction.ATTR_EVALUATOR_NAME: (
                                PdIceExplainer._display_name
                            ),
                        },
                        actions_description=(
                            f"Verify that feature `{problem_feature}' error "
                            f"interaction {class_msg} does not indicate a model bias "
                            f"or other problem."
                        ),
                        explainer_id=self.explainer_id(),
                        explainer_name=self._display_name,
                        explanation_type=(
                            e10s.PartialDependenceExplanation.explanation_type()
                        ),
                        explanation_name=e10s.PartialDependenceExplanation.__name__,
                        explanation_mime=f5s.PartialDependenceJSonFormat.mime,
                        resources=[],
                    )
                    self.add_problem(problem)

            return json_representation, labels_2_pd_map
        else:
            raise ValueError("PD is required to build its JSon representation")

    @staticmethod
    def _pd_save_feature_json(explanations) -> dict:
        cls_dict = dict()
        data = list()
        pdj = f5s.PartialDependenceJSonFormat
        for bin_ in explanations.columns.values.tolist():
            data_item = dict()
            data_item[pdj.KEY_BIN] = bin_
            data_item[pdj.KEY_PD] = explanations.at[_pd.PD.COL_MEAN, bin_]
            data_item[pdj.KEY_SD] = explanations.at[_pd.PD.COL_SD, bin_]
            data_item[pdj.KEY_OOR] = explanations.at[_pd.PD.COL_OOR, bin_]
            data.append(data_item)
        cls_dict[pdj.KEY_DATA] = data
        return cls_dict

    #
    # ICE representation
    #

    KEY_BINS = "bins"
    KEY_LABELS_MAP = "labels_2_pd_map"

    def _work_ice_to_gom_ice_json(
        self,
        ice_explanation: e10s.IndividualConditionalExplanation,
        features_meta: dict,
        pd: _pd.PD,
        labels_2_pd_map: dict,
        is_sampling: bool = True,
        ice_cache_dir_path: str = "",
    ) -> f5s.IceJsonDatatableFormat:
        """Conversion of raw DAI PD (work directory) assets to Grammar of MLI PD JSon
        representation.

        """
        if is_sampling:
            # if MLI samples, then ICE must be computed on-demand (later)

            # PD's per-feature bins MUST be used by ICE
            on_demand_params: dict = dict()
            on_demand_params[PdIceExplainer.PARAM_DEBUG_RESIDUALS] = self.args.get(
                PdIceExplainer.PARAM_DEBUG_RESIDUALS, False
            )
            on_demand_params[self.KEY_LABELS_MAP] = labels_2_pd_map
            on_demand_params[self.KEY_BINS] = dict()
            for feature in pd.features:
                feature_bins = f5s.PartialDependenceJSonFormat.get_bins(
                    persistence=self.persistence,
                    explanation_type=(
                        e10s.PartialDependenceExplanation.explanation_type()
                    ),
                    feature=feature,
                )
                on_demand_params[self.KEY_BINS][feature] = feature_bins

            json_ice_representation = f5s.IceJsonDatatableFormat(
                explanation=ice_explanation,
                json_data=f5s.IceJsonDatatableFormat.serialize_on_demand_index_file(
                    on_demand_params
                ),
            )
            on_demand_params[self.KEY_LABELS_MAP] = labels_2_pd_map
            return json_ice_representation

        # generate new index file > determine data file names from old index file >
        # rename and copy (from work to representation) dir all data files

        ice_cache_file_path: str = os.path.join(
            (
                ice_cache_dir_path
                if ice_cache_dir_path
                else self.persistence.get_explainer_working_dir()
            ),
            PdIceExplainer.FILE_ICE_JSON,
        )

        if os.path.isfile(ice_cache_file_path):
            with open(ice_cache_file_path) as json_file:
                ice_cache_dict = json.load(json_file)

            features = list(ice_cache_dict[_pd.PD.KEY_ICE_CACHE].keys())
            classes = list(ice_cache_dict[_pd.PD.KEY_ICE_CACHE][features[0]].keys())

            # index file
            (index_dict, _) = f5s.IceJsonDatatableFormat.serialize_index_file(
                features=features,
                classes=classes,
                features_meta=features_meta,
                metrics=None,
                doc=PdIceExplainer._description,
                y_file=f5s.IceJsonDatatableFormat.FILE_Y_FILE,
            )
            index_dict[self.KEY_LABELS_MAP] = labels_2_pd_map
            json_ice_representation = f5s.IceJsonDatatableFormat(
                explanation=ice_explanation,
                json_data=json.dumps(index_dict, indent=4),
                persistence=self.persistence.store,
            )

            # data files: copy and rename
            icejd = f5s.IceJsonDatatableFormat
            for feature in features:
                for cls in classes:
                    src_file: str = (
                        os.path.join(
                            ice_cache_dir_path,
                            os.path.basename(
                                ice_cache_dict[_pd.PD.KEY_ICE_CACHE][feature][cls]
                            ),
                        )
                        if ice_cache_dir_path
                        else ice_cache_dict[_pd.PD.KEY_ICE_CACHE][feature][cls]
                    )
                    dst_file: str = index_dict[icejd.KEY_FEATURES][feature][
                        icejd.KEY_FILES
                    ][cls]
                    self.logger.debug(
                        f"{self.log_name}: copying ICE JSon+datatable representation "
                        f"frames: {src_file} to {dst_file}",
                    )
                    json_ice_representation.add_file(
                        format_file=src_file, file_name=dst_file
                    )
            json_ice_representation.add_file(
                format_file=os.path.join(
                    self.persistence.get_explainer_working_dir(),
                    PdIceExplainer.FILE_Y_HAT,
                ),
                file_name=f5s.IceJsonDatatableFormat.FILE_Y_FILE,
            )
            return json_ice_representation
        else:
            raise errors.MliError(
                f"ICE cache file required to build its JSon representation does not "
                f"exist: {ice_cache_file_path}",
            )

    def explain_global(self, X, y=None, **e_params) -> list:
        """Update/re-calculate PD explanation.

        Parameters
        ----------
        X: datatable.Frame
          Dataset (whole) as datatable frame (handle).
        y: datatable.Frame
          Optional predictions as datatable frame (handle).

        Returns
        -------
        list[OnDemandExplanation]:
          Update on-demand explanations.

        """
        if not e_params.get(explainers.OnDemandExplainKey.MLI_KEY):
            raise ValueError(
                f"[{self.mli_key} / {self.key}] target MLI key is required by "
                f"explainer's global update method"
            )
        if not e_params.get(explainers.OnDemandExplainKey.EXPLAINER_JOB_KEY):
            raise ValueError(
                f"[{self.mli_key} / {self.key}] target explainer job key is required "
                f"by explainer's global update method"
            )

        if self.config.debug_log:
            self.logger.debug(
                f"\nexplain_global(): "
                f"\n  target MLI key: "
                f"{e_params.get(explainers.OnDemandExplainKey.MLI_KEY)}"
                f"\n  target job key: "
                f"{e_params.get(explainers.OnDemandExplainKey.EXPLAINER_JOB_KEY)}"
                f"\n  MLI key       : {self.mli_key}"
                f"\n  job key       : {self.key}"
                f"\n  PD features: {self.args.get(PdIceExplainer.PARAM_FEATURES)}"
                f"\n  all params :\n{e_params}",
            )

        is_numcat_update: bool = False

        # use args EXCEPT listing of features ~ fix it on ARGS object or elsewhere
        # try to load target PD/ICE explainer parameters (resolution, OOR, ...)

        # load PD/ICE explainer's params in target MLI in case of inherit strategy
        if e_params.get(explainers.OnDemandExplainKey.UPDATE_STRATEGY, {}):
            is_numcat_update = (
                True
                if e_params[explainers.OnDemandExplainKey.UPDATE_STRATEGY].get(
                    commons.UpdateGlobalExplanation.UPDATE_SCOPE, None
                )
                == PdIceExplainer.UPDATE_SCOPE_NUMCAT
                else False
            )

            if (
                e_params[explainers.OnDemandExplainKey.UPDATE_STRATEGY].get(
                    commons.UpdateGlobalExplanation.PARAMS_SOURCE, ""
                )
                == commons.UpdateGlobalExplanation.OPT_INHERIT
            ):
                try:
                    target_mli_persistence = persistences.InterpretationPersistence(
                        data_dir=self.persistence.data_dir,
                        username=self.persistence.username,
                        mli_key=e_params.get(explainers.OnDemandExplainKey.MLI_KEY),
                    )
                    target_es_params: dict = (
                        target_mli_persistence.load_explainers_params()
                    )
                    e_id = PdIceExplainer.explainer_id()
                    if (
                        target_es_params
                        and e_id in target_es_params
                        and target_es_params[e_id].get(
                            commons.ExplainerParamKey.KEY_E_PARAMS, ""
                        )
                    ):
                        target_e_params = target_es_params[e_id][
                            commons.ExplainerParamKey.KEY_E_PARAMS
                        ]
                        self.logger.debug(
                            f"explain_global() target params: {target_e_params}",
                        )
                        inherited_args = PdIceArgs(self._parameters)
                        inherited_args.resolve_params(
                            explainer_params=PdIceArgs.json_str_to_dict(
                                target_e_params
                            ),
                        )
                        self.logger.debug(
                            f"explain_global() parsed target params: "
                            f"{inherited_args.args}",
                        )

                        # merge/override parameters
                        for arg in inherited_args.args:
                            if PdIceExplainer.PARAM_FEATURES != arg:
                                self.args.args[arg] = inherited_args.args[arg]

                        self.logger.debug(
                            f"\nexplain_global() updated args: {self.args.args}",
                        )
                except Exception as ex:
                    self.logger.error(
                        f"explain_global() failed to load target PD/ICE explainer "
                        f"params from MLI "
                        f"{e_params.get(explainers.OnDemandExplainKey.MLI_KEY)}: {ex}"
                        f"\n{traceback.format_exc()}",
                    )

        target_pd_explanation_dir = ""
        try:
            self.is_on_demand_pd = True

            target_persistence = persistences.ExplainerPersistence(
                data_dir=self.persistence.data_dir,
                username=self.persistence.username,
                explainer_id=self.persistence.explainer_id,
                explainer_job_key=e_params.get(
                    explainers.OnDemandExplainKey.EXPLAINER_JOB_KEY
                ),
                mli_key=e_params.get(explainers.OnDemandExplainKey.MLI_KEY),
            )

            target_pd_explanation_dir = target_persistence.get_explanation_dir_path(
                explanation_type=e10s.PartialDependenceExplanation.explanation_type(),
                explanation_format=f5s.PartialDependenceJSonFormat.mime,
            )

            f5s.PartialDependenceJSonFormat.set_merge_status(
                dir_path=target_pd_explanation_dir,
                mli_key=self.mli_key,
                explainer_job_key=self.key,
                clear=False,
                action_type=(
                    PdIceExplainer.UPDATE_TYPE_ADD_NUMCAT
                    if is_numcat_update
                    else PdIceExplainer.UPDATE_TYPE_ADD_FEATURE
                ),
            )

            # num/cat aspect validation check
            e_params[PdIceExplainer.UPDATE_PARAM_NUMCAT_OVERRIDE] = (
                self._explain_global_numcat(target_persistence)
                if is_numcat_update
                else ""
            )

            # RUN explainer
            explanations = self.explain(X=X, y=y, **e_params)

            # merge new PD to the target one
            f5s.PartialDependenceJSonFormat.merge_format(
                from_path=self.persistence.get_explanation_dir_path(
                    explanation_type=(
                        e10s.PartialDependenceExplanation.explanation_type()
                    ),
                    explanation_format=f5s.PartialDependenceJSonFormat.mime,
                ),
                to_path=target_pd_explanation_dir,
                overwrite=True,
                discriminant=self.key,
                is_numcat_merge=is_numcat_update,
            )

            # merge new ICE to the target one
            ice_e = e10s.IndividualConditionalExplanation
            f5s.IceJsonDatatableFormat.merge_format(
                from_path=self.persistence.get_explanation_dir_path(
                    explanation_type=ice_e.explanation_type(),
                    explanation_format=f5s.IceJsonDatatableFormat.mime,
                ),
                to_path=target_persistence.get_explanation_dir_path(
                    explanation_type=ice_e.explanation_type(),
                    explanation_format=f5s.IceJsonDatatableFormat.mime,
                ),
                overwrite=True,
                discriminant=self.key,
                is_numcat_merge=is_numcat_update,
            )

            # delete work dir content to save disk space
            self.persistence.rm_dir_contents(
                dir_path=self.persistence.get_explainer_working_dir(),
                logger=self.logger,
            )
        except Exception as ex:
            self.logger.error(
                f"On-demand PD/ICE {self.mli_key} / {self.key} explainer "
                f"UPDATE failed: {ex}\n{traceback.format_exc()}",
            )
            # problem is logged, clear merge status to avoid indefinitely blocked run
            raise ex
        finally:
            if target_pd_explanation_dir:
                f5s.PartialDependenceJSonFormat.set_merge_status(
                    dir_path=target_pd_explanation_dir,
                    mli_key=self.mli_key,
                    explainer_job_key=self.key,
                    clear=True,
                    action_type=(
                        PdIceExplainer.UPDATE_TYPE_ADD_NUMCAT
                        if is_numcat_update
                        else PdIceExplainer.UPDATE_TYPE_ADD_FEATURE
                    ),
                )

        return explanations

    def _explain_global_numcat(self, target_persistence) -> str:
        # num/cat aspects: validation check
        feature = self.args.get(PdIceExplainer.PARAM_FEATURES)[0]
        explanation_format_mime = f5s.PartialDependenceJSonFormat.mime
        target_idx_path: str = target_persistence.get_explanation_file_path(
            explanation_type=e10s.PartialDependenceExplanation.explanation_type(),
            explanation_format=explanation_format_mime,
            explanation_file=(
                f"{f5s.ExplanationFormat.FILE_PREFIX_EXPLANATION_IDX}"
                f"{commons.MimeType.ext_for_mime(explanation_format_mime)}"
            ),
        )
        target_idx: dict = self.persistence.load_json(target_idx_path)
        # determine which aspect to calculate
        missing_numcat = f5s.PartialDependenceJSonFormat.get_numcat_missing_aspect(
            feature=feature, idx=target_idx
        )
        if not missing_numcat:
            raise errors.MliError(
                f"explain_global({self.mli_key}/{self.key}) for feature {feature}: "
                f"both numeric and categorical PD/ICE is already available - "
                f"nothing to compute"
            )

        return missing_numcat

    def _calculate_pd_ice(
        self,
        dataset: datatable.Frame,
        y: datatable.Frame | None = None,
        numcat_override: str = "",
    ) -> tuple[_pd.PD, dict, bool, datatable.Frame]:
        """Explainer aware PD/ICE calculation using DAI scorer and original
        dataset.

        Parameters
        ----------
        dataset : datatable.Frame
            Dataset.
        y :  datatable.Frame
            Optional labels.
        numcat_override: str
            Force feature type to be ``numerical`` or ``categorical`` regardless other
            parameters. This parameter is used to calculate PD for other than inferred
            feature aspect e.g. categorical PD for numerical feature.

        Returns
        -------
        Tuple[PD, dict]:
            PD/ICE instance, features metadata, sampling indicator.

        """
        self.logger.info(f"{self.log_name} BEGIN calculation")

        model_features = self.model_meta.used_features or []
        explainer_dir_path = self.persistence.get_explainer_working_dir()
        max_features = self.args.get(PdIceExplainer.PARAM_MAX_FEATURES)
        grid_resolution = self.args.get(PdIceExplainer.PARAM_GRID_RESOLUTION)
        use_features = self.args.get(PdIceExplainer.PARAM_FEATURES)
        qtile_grid_resolution = self.args.get(
            PdIceExplainer.PARAM_QTILE_GRID_RESOLUTION
        )
        qtile_bins_dict_str = self.args.get(PdIceExplainer.PARAM_QTILE_BINS)

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

        # num/cat check & fix
        for col in features:
            if dataset[:, col].ltypes[0] in [
                datatable.ltype.str,
                datatable.ltype.obj,
            ]:
                # num/cat OVERRIDE: cat -> num ... fail fast on undoable override
                if numcat_override == f5s.PartialDependenceJSonFormat.FEATURE_TYPE_NUM:
                    raise errors.MliError(
                        f"{self.log_name} error: numerical binning cannot be forced "
                        f"for feature '{col}' as categorical feature values are not "
                        f"of numerical type"
                    )

                # ensure that input categorical variables are in features_metadata;
                # otherwise, those variables will enter the numeric binning logic of
                # PD and may lead to an exception if OOR > number of uniques or it
                # may lead to erroneous results due to PD not handling categoricals
                # properly
                if (
                    method.Method.KEY_CATEGORICAL_FEATURES not in features_meta
                    or col not in features_meta[method.Method.KEY_CATEGORICAL_FEATURES]
                ) and (model_features and col not in model_features):
                    if method.Method.KEY_CATEGORICAL_FEATURES not in features_meta:
                        features_meta[method.Method.KEY_CATEGORICAL_FEATURES] = [col]
                    else:
                        features_meta[method.Method.KEY_CATEGORICAL_FEATURES].append(
                            col
                        )

        if self.params.drop_cols:
            drop_features: list = sanitization.sanitize_strings(self.params.drop_cols)
            skip_features.extend(drop_features)

        if skip_features:
            features = [x for x in features if x not in skip_features]

        pdc = None
        if features:
            # force num binning for num+cat features
            forced_fs_meta: dict | None = (
                features_meta.copy() if features_meta else features_meta
            )
            if (
                self.args.get(PdIceExplainer.PARAM_NUMCAT_NUM_CHART, False)
                and forced_fs_meta
                and forced_fs_meta.get(FeaturesMetadata.KEY_CATNUM_FEATURES, None)
            ):
                sanitized_cn_fs: list = sanitization.sanitize_strings(
                    strings=forced_fs_meta.get(
                        FeaturesMetadata.KEY_CATNUM_FEATURES, None
                    ),
                )
                for cn_f in sanitized_cn_fs:
                    if cn_f in dataset.names and dataset[
                        :, cn_f
                    ].nunique1() > self.args.get(PdIceExplainer.PARAM_NUMCAT_THRESHOLD):
                        # force numeric binning (and chart)
                        if (
                            cn_f
                            in forced_fs_meta[method.Method.KEY_CATEGORICAL_FEATURES]
                        ):
                            forced_fs_meta[
                                method.Method.KEY_CATEGORICAL_FEATURES
                            ].remove(cn_f)

            # PD's ICE cache path
            ice_json_file = os.path.join(
                explainer_dir_path, PdIceExplainer.FILE_ICE_JSON
            )
            self.logger.debug(f"{self.log_name}: ICE cache file: {ice_json_file}")

            pdc = _pd.PD(f"Explainer: {self.mli_key}/{self.key}")

            # ensure unique values for a feature is > 1
            feat_uniq_vals = dataset[:, features].nunique().to_dict()
            # calculate meaningful PD/ICE for features with cardinality 1 by
            # adding OOR bins (if user didn't request it)
            feat_unique_1 = []
            for feature in features:
                if feat_uniq_vals[feature][0] == 1:
                    if not self.args.get(PdIceExplainer.PARAM_OOR_GRID_RESOLUTION, 0):
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
                        err_msg = (
                            f"Dictionary of quantile bins per feature must contain "
                            f"all or a subset of PD features. PD features are: "
                            f"{features} and quantile bins per feature dictionary "
                            f"is: {qtile_bins_dict}"
                        )
                        if not self.is_on_demand_pd:
                            raise ValueError(err_msg)
                        else:
                            self.logger.warning(err_msg)
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

            # num/cat feature types override
            if numcat_override:
                self.logger.info(
                    f"{self.log_name} forcing num/cat override to '{numcat_override}' "
                    f"for features {features}... ",
                )
                if features_meta:
                    for f in features:
                        # num/cat OVERRIDE: cat -> num
                        if (
                            numcat_override
                            == f5s.PartialDependenceJSonFormat.FEATURE_TYPE_NUM
                        ):
                            # clear cat
                            for f_type in [
                                method.Method.KEY_CATEGORICAL_FEATURES,
                                f5s.PartialDependenceJSonFormat.FEATURE_TYPE_CAT_NUM,
                            ]:
                                if (
                                    f_type in features_meta
                                    and f in features_meta[f_type]
                                ):
                                    features_meta[f_type].remove(f)
                            # set num
                            if (
                                f5s.PartialDependenceJSonFormat.FEATURE_TYPE_NUM
                                not in features_meta
                            ):
                                features_meta[
                                    f5s.PartialDependenceJSonFormat.FEATURE_TYPE_NUM
                                ] = []
                            features_meta[
                                f5s.PartialDependenceJSonFormat.FEATURE_TYPE_NUM
                            ].append(f)
                        # num/cat OVERRIDE: num -> cat .. always doable (diff bins only)
                        if (
                            numcat_override
                            == f5s.PartialDependenceJSonFormat.FEATURE_TYPE_CAT
                        ):
                            # clear num
                            for f_type in [
                                f5s.PartialDependenceJSonFormat.FEATURE_TYPE_NUM,
                                f5s.PartialDependenceJSonFormat.FEATURE_TYPE_CAT_NUM,
                            ]:
                                if (
                                    f_type in features_meta
                                    and f in features_meta[f_type]
                                ):
                                    features_meta[f_type].remove(f)
                            # set cat
                            if (
                                f5s.PartialDependenceJSonFormat.FEATURE_TYPE_CAT
                                not in features_meta
                            ):
                                features_meta[
                                    f5s.PartialDependenceJSonFormat.FEATURE_TYPE_CAT
                                ] = []
                            features_meta[
                                f5s.PartialDependenceJSonFormat.FEATURE_TYPE_CAT
                            ].append(f)
            self.logger.info(f"{self.log_name} feature metadata: {features_meta}")

            pdc.opt_1_prediction = PdIceExplainer.OPT_ICE_1_FRAME_ENABLED
            self.logger.info(
                f"{self.log_name} 1 frame strategy: {pdc.opt_1_prediction}",
            )

            if self.args.get(PdIceExplainer.PARAM_DEBUG_RESIDUALS, False):
                if y is None:
                    raise errors.MliError(
                        f"{self.log_name} residual PD/ICE cannot be calculated as "
                        f"y column was not specified - it is None"
                    )
                else:
                    self.logger.info(
                        f"{self.log_name} residual PD/ICE WILL be calculated as "
                        f"the explainer parameter is "
                        f"{self.args.get(PdIceExplainer.PARAM_DEBUG_RESIDUALS, False)} "
                        f"and labels are {y}",
                    )
            elif y is not None:
                self.logger.info(
                    f"{self.log_name} residual PD/ICE should NOT be calculated, "
                    f"but y has been specified - setting it None"
                )
                y = None

            # residuals target transform function
            target_transform = (
                (lambda x: (-1 * numpy.log10(x)).fillna(0))
                if self.model_meta.num_labels >= 2
                else lambda x: x**2
            )

            pdc.explain(
                features,
                dataset,
                # IMPROVE: fast approx, MOJO, IS_SCORER, ...
                predict_method=self.model.predict_pandas,
                grid_resolution=grid_resolution,
                features_meta=features_meta,
                ice_cache_path=ice_json_file,
                bins_sort=self.args.get(PdIceExplainer.PARAM_SORT_BINS, False),
                out_of_range_resolution=self.args.get(
                    PdIceExplainer.PARAM_OOR_GRID_RESOLUTION, 0
                ),
                center=self.args.get(PdIceExplainer.PARAM_CENTER, False),
                progress_callback=progress.MethodToExplainerProgressReporter(
                    min_progress=PdIceExplainer.PROGRESS_MIN,
                    max_progress=PdIceExplainer.PROGRESS_MAX,
                    callback=self.report_progress,
                ),
                Y=(
                    y
                    if self.args.get(PdIceExplainer.PARAM_DEBUG_RESIDUALS, False)
                    else None
                ),
                target_transform=(
                    target_transform
                    if self.args.get(PdIceExplainer.PARAM_DEBUG_RESIDUALS, False)
                    else lambda z: z
                ),
            )
            self.logger.debug(f"{self.log_name} result: {pdc}")
            self.logger.debug(
                f"{self.log_name} scorer calls: {pdc.diagnostics.total_scorer_calls}",
            )

            if feat_unique_1:
                self.logger.info(
                    f"{self.log_name} calculation for features with 1 unique value "
                    f"(using OOR) {feat_unique_1} with 1-frame strategy set to: "
                    f"{pdc.opt_1_prediction}",
                )

                pdc_u1 = _pd.PD("unique()==1 features")
                pdc_u1.opt_1_prediction = PdIceExplainer.OPT_ICE_1_FRAME_ENABLED
                pdc_u1.explain(
                    feat_unique_1,
                    dataset,
                    # IMPROVE: fast approx, MOJO, IS_SCORER, ...
                    predict_method=self.model.predict_pandas,
                    grid_resolution=grid_resolution,
                    features_meta=features_meta,
                    ice_cache_path=ice_json_file,
                    bins_sort=self.args.get(PdIceExplainer.PARAM_SORT_BINS, False),
                    out_of_range_resolution=3,
                    center=self.args.get(PdIceExplainer.PARAM_CENTER, False),
                    # residuals: labels are required + x^2 residual transformation
                    Y=(
                        y
                        if self.args.get(PdIceExplainer.PARAM_DEBUG_RESIDUALS, False)
                        else None
                    ),
                    target_transform=lambda x: (
                        x**2
                        if self.args.get(PdIceExplainer.PARAM_DEBUG_RESIDUALS, False)
                        else lambda z: z
                    ),
                )

                # MERGE additional results to the original PD
                # TODO h2o_sonar-2::pd.py::pdc.merge(pdc_u1)
                for f_u1 in pdc_u1.explanations().keys():
                    self.logger.info(
                        f"{self.log_name} merging feature "
                        f"'{f_u1}' with cardinality 1 to main PD result",
                    )
                    pdc.explanations()[f_u1] = pdc_u1.explanations()[f_u1]
                    pdc.features.append(f_u1)

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
                os.path.join(explainer_dir_path, PdIceExplainer.FILE_Y_HAT)
            )

        # legacy PD ser. for backward compatibility (regression and binomial only)
        if self.model_meta.num_labels <= 2:
            self._persist_legacy_pd(
                pdc=pdc,
                mli_dir_path=explainer_dir_path,
                features_metadata=features_meta,
                num_cat_num_chart=self.args.get(
                    PdIceExplainer.PARAM_NUMCAT_NUM_CHART, False
                ),
            )

        is_sampling = True if self.dataset_meta.row_count > dataset.shape[0] else False

        return pdc, features_meta, is_sampling, dataset

    def _persist_legacy_pd(
        self,
        pdc: _pd.PD,
        mli_dir_path: str,
        features_metadata,
        num_cat_num_chart: bool = True,
    ):
        """Legacy PD persistence (regression/binomial) for backward compatibility.
        Persist either PD result OR empty chart if there are no features for which
        to compute it.

        """
        pd_json_file = os.path.join(mli_dir_path, PdIceExplainer.FILE_PD_JSON)
        if os.path.exists(pd_json_file):
            self.logger.warning(
                f"{self.log_name} file already exists - removing {pd_json_file}"
            )
            os.remove(pd_json_file)  # this deletes the file

        if pdc:
            self.logger.info(f"{self.log_name} saving PD to {pd_json_file}")
            pdc.save_json(pd_json_file)
            with open(pd_json_file) as json_file:
                pd_dict: dict = json.load(json_file)
            if num_cat_num_chart:
                pd_chart = PdIceExplainer._pd_catnum_force_chart_type(
                    pd=pd_dict,
                    numcat_features=(
                        None
                        if not features_metadata
                        else (
                            features_metadata[FeaturesMetadata.KEY_CATNUM_FEATURES]
                            if FeaturesMetadata.KEY_CATNUM_FEATURES
                            in features_metadata.keys()
                            else None
                        )
                    ),
                    numcat_as_num=num_cat_num_chart,
                )
                if pd_chart:
                    with open(pd_json_file, mode="w") as json_file:
                        json_file.seek(0)
                        json.dump(pd_chart, json_file)
                        json_file.truncate()
        else:
            self.logger.warning(
                f"{self.log_name} no (reasonable) features - saving EMPTY chart data",
            )
            empty_pdc = dict()
            empty_pdc[_pd.PD.JSON_COLS] = list()
            empty_pdc[_pd.PD.JSON_NBINS] = method.Method.DEFAULT_GRID_RESOLUTION
            empty_pdc[_pd.PD.JSON_PD_DATA] = list()
            with open(pd_json_file, "w") as fh:
                json.dump(empty_pdc, fh)

        self.logger.info(
            f"{self.log_name} computation finished & stored to: {pd_json_file}"
        )

    @staticmethod
    def _pd_catnum_force_chart_type(
        pd: dict, numcat_features: list, numcat_as_num: bool = True
    ) -> dict | None:
        """Force PD chart (numeric or categorical) for numeric&categorical
        features based on configuration. Note that PD is saved as categorical for
        num&cat features by MLI-2 PD, but DAI default is numeric chart. Chart type is
        forced only for PDs created with the configuration setting enabled i.e.
        existing PDs are not modified (PD is read as JSon file by client ~ would
        break API and/or all PD files would have to be transformed on every
        configuration change.

        """
        # Categorical vs. numerical binning:
        #
        # - model can use particular feature as:
        #   - numerical
        #   - categorical
        #   - numerical and categorical
        # - ``numcat_num_chart``
        #   - expert settings parameter which allows to enable/disable dynamic
        #     numerical vs.categorical binning and thus also default chart rendering.
        # - ``numcat_threshold``
        #   - expert settings parameter which allows to specify **threshold** for
        #     numerical vs.categorical binning - feature is rendered as numerical if
        #     cardinality is above the threshold
        # - method:
        #   - NUMERICAL binning & chart
        #       <=>
        #     model treats features as numerical OR numerical+categorical
        #       AND
        #     number of unique values > numcat_num_threshold
        #   - CATEGORICAL binning & chart
        #       <=>
        #     model treats features categorical
        #       AND
        #     number of unique values <= numcat_num_threshold

        if not numcat_features or not pd or not pd[_pd.PD.JSON_PD_DATA]:
            # no features to tweak
            return None
        if not numcat_as_num:
            # MLI-2 PD is persisted to be rendered as categorical
            return None

        # change feature type specification and bin values from string to number
        for feature in numcat_features:
            for entry in pd[_pd.PD.JSON_PD_DATA]:
                if feature == entry[_pd.PD.JSON_COLUMNS][0][_pd.PD.JSON_DESC]:
                    # entry to tweak
                    entry[_pd.PD.JSON_COLUMNS][0][_pd.PD.JSON_TYPE] = "double"
                    entry[_pd.PD.JSON_DATA][0] = entry[_pd.PD.JSON_BINS]
                    continue
        return pd

    def _inject_histogram_data(
        self,
        data_cls_dict: dict,
        dataset,
        feature: str,
        is_categorical: bool,
        target_col: str,
        clazz=None,
        features_meta: dict | None = None,
    ) -> bool:
        """Get histogram data for given feature:

        - sampled dataset used in case of sampling
        - per-class dataset sub-set used in case of multinomial
        - numerical (continuous/discrete), categorical (string/number), date features
        - original features only (PD/ICE not supported on transformed)

        Parameters
        ----------
        data_cls_dict: dict
          Dictionary where to store histogram data.
        dataset: datatable.Frame
          Datatable frame
        feature: str
          Feature/column name for which to prepare histogram data.
        is_categorical: bool
          Categorical (for discrete features) ``True``, else ``False`` for numerical.
        clazz:
          Class to be used for filtering in case of bi/multinomial.
        features_meta: dict | None
          Features metadata.

        Returns
        -------
        bool:
          ``True`` if ``data_histogram_*`` added to data dictionary, else ``False``.

        """
        try:
            if dataset:
                pdj = f5s.PartialDependenceJSonFormat

                if clazz:
                    # classification: per-class histogram
                    filtered = dataset[datatable.f[target_col] == clazz, :]
                    if not filtered.shape[0]:
                        return False
                    pd_series = filtered[:, feature]
                else:
                    # regression: no dataset filtering (histogram for the whole dataset)
                    pd_series = dataset[:, feature]

                # bins w/o OOR (OOR bins have 0 frequency)
                in_range_bins = []
                for p in data_cls_dict[pdj.KEY_DATA]:
                    if not p[pdj.KEY_OOR]:
                        in_range_bins.append(p[pdj.KEY_BIN])

                (xs, ys) = histogram.histogram_data(
                    df=pd_series,
                    bins=in_range_bins,
                    is_date=(
                        True
                        if features_meta
                        and feature
                        in features_meta.get(method.Method.KEY_DATE_FEATURES, [])
                        else False
                    ),
                    is_discrete=is_categorical,
                )

                # continuous (x > y) vs. discrete
                if len(xs) > len(ys):
                    ys.append(None)
                data = []
                cls_dict_data = data_cls_dict[pdj.KEY_DATA]
                cls_bins = [d.get(pdj.KEY_BIN, None) for d in cls_dict_data]
                if histogram.has_nan(xs) and histogram.has_nan(cls_bins):
                    na_idx_cls = histogram.find_nan_index(cls_bins)
                    na_idx_xs = histogram.find_nan_index(xs)
                    xs[na_idx_xs] = MISSING_VALUE
                    cls_bins[na_idx_cls] = MISSING_VALUE

                if self.args.get(PdIceExplainer.PARAM_OOR_GRID_RESOLUTION, 0):
                    # Ensure first half of OOR bins are before the minimum bin value
                    # and second half of OOR bins are after the maximum bin value or
                    # the missing value bin as this is the last bin if missing values
                    # are present in a column
                    oor_bins = []
                    for p in data_cls_dict[pdj.KEY_DATA]:
                        if p[pdj.KEY_OOR]:
                            oor_bins.append(p[pdj.KEY_BIN])
                    oor_bins_hist = len(oor_bins) * [0]
                    first_half_hist = oor_bins_hist[: len(oor_bins_hist) // 2]
                    second_half_hist = oor_bins_hist[len(oor_bins_hist) // 2 :]
                    ys = first_half_hist + ys
                    ys = ys + second_half_hist
                    first_half = oor_bins[: len(oor_bins) // 2]
                    second_half = oor_bins[len(oor_bins) // 2 :]
                    xs = first_half + xs
                    xs = xs + second_half

                for i, x in enumerate(xs):
                    data_item = dict()
                    data_item[pdj.KEY_X] = x
                    data_item[pdj.KEY_FREQUENCY] = ys[i]
                    data.append(data_item)

                if is_categorical:
                    data_cls_dict[pdj.KEY_DATA_HISTOGRAM_CAT] = data
                else:
                    data_cls_dict[pdj.KEY_DATA_HISTOGRAM_NUM] = data

                return True
        except Exception as ex:
            self.logger.warning(
                f"{self.log_name} unable to prepare histogram data for "
                f"{'categorical' if is_categorical else 'numerical'} feature "
                f"'{feature}' and class {clazz}: {ex}\n{traceback.format_exc()}",
            )

        return False

    def _explain_ice_on_demand(
        self,
        x: datatable.Frame,
        y: datatable.Frame,
        features: list,
        features_meta,
        clazz: str,
        bins: list[list] | None,
        base_dir_path: str,
        labels,
        do_debug_error: bool = False,
    ) -> str:
        # instance load from dataset
        dataset = x
        # TODO @ DAI model: dataset = enforce_all_types(self.logger, fitted_model, x)
        # TODO @ DAI explainable model: dataset = mli_commons.enforce_int_bool(dataset)
        # sanitize column names
        dataset.names = sanitization.sanitize_strings(dataset.names)
        if 1 != dataset.nrows:
            raise ValueError(
                f"On-demand ICE: dataset has more than 1 row/instance ({dataset.nrows})"
            )
        instance_df = dataset.to_pandas()

        # features metadata
        self.logger.info(f"{self.log_name}: on-demand ICE base dir: {base_dir_path}")

        # ICE: ensure 1 predict method invocation for whole ICE calculation
        ice = _ice.ICE("On-demand ICE explanation")
        ice.opt_1_prediction = PdIceExplainer.OPT_ICE_1_FRAME_ENABLED
        ice.explain(
            features,
            instance_df,
            bins=bins,
            # IMPROVE: fast approx, MOJO, IS_SCORER, ...
            predict_method=self.model.predict_pandas,
            features_meta=features_meta,
            # residuals on demand
            Y=y if do_debug_error else None,
            target_transform=lambda xx: xx**2 if do_debug_error else lambda z: z,
        )
        self.logger.info(f"{self.log_name}: on-demand ICE:\n{ice}")
        feature = features[0]

        # feature value
        try:
            feature_value = dataset[0, feature]
        except Exception as ex:
            self.logger.warning(
                f"{self.log_name}: on-demand ICE not able to pull feature {feature} "
                f"value: {ex}:\n{traceback.format_exc()}",
            )
            feature_value = None

        # TODO sub-optimal: if value is among bins, then skip the score() for all feats
        # IMPROVE: fast approx, MOJO, IS_SCORER, ...
        y_hat_frame = self.model.predict_datatable(x)
        if labels and len(labels) > 2:
            labels = [str(label) for label in labels]
            y_hat_frame.names = sanitization.sanitize_strings(labels)
            prediction = y_hat_frame[0, clazz]
            clazz = f"p_{labels.index(clazz)}"
        else:
            # REGRESSION: one value ~ p_0 result class
            # BINOMIAL: exactly one (positive) class always returned ~ p_0 result class
            clazz = method.Method.LABEL_REGRESSION
            prediction = y_hat_frame[0, 0]

        ice_json: str = f5s.IceJsonDatatableFormat.mli_ice_explanation_to_json(
            ice_df=ice.explanations()[feature][clazz],
            filter_feature=feature,
            prediction=prediction,
            feature_value=feature_value,
            logger=self.logger,
        )
        return ice_json

    def get_result(self) -> results.PdResult:
        return results.PdResult(
            persistence=self.persistence,
            h2o_sonar_config=self.config,
            explainer_id=PdIceExplainer.explainer_id(),
            logger=self.logger,
        )


class PdIceArgs(explainers.ExplainerArgs):
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

        max_features = self.args.get(PdIceExplainer.PARAM_MAX_FEATURES)
        if (
            explainer_params
            and explainer_params.get(PdIceExplainer.PARAM_MAX_FEATURES) is not None
        ):
            max_features = int(explainer_params.get(PdIceExplainer.PARAM_MAX_FEATURES))
        self.args[PdIceExplainer.PARAM_MAX_FEATURES] = max_features

        return erased_config_overrides
