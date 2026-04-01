# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import datetime
import json
import os
import pathlib
import re
import shutil
import time
import traceback
from collections.abc import Callable
from functools import partial

import airium
import datatable

from h2o_sonar import errors
from h2o_sonar.explainers import surrogate_rf_explainer
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api import problems
from h2o_sonar.lib.api import results
from h2o_sonar.methods.core import _mli
from h2o_sonar.methods.core import method
from h2o_sonar.methods.surrogates import _decision_tree_h2o
from h2o_sonar.methods.surrogates import _tree_traverser_h2o
from h2o_sonar.methods.surrogates.rules import rules
from h2o_sonar.methods.utils import h2o_utils
from h2o_sonar.utils import binning
from h2o_sonar.utils import sampling
from h2o_sonar.utils import sanitization


try:
    import pydot

    HAS_PYDOT = True
except ImportError:
    HAS_PYDOT = False

try:
    import h2o
    from h2o import tree as h2o_tree

    HAS_H2O = True
except ImportError:
    HAS_H2O = False


class DecisionTreeConstants:
    SEED = 12345

    # decision tree parameters
    DEFAULT_NFOLDS = 0
    DEFAULT_TREE_DEPTH = 3

    # files and directories
    FILE_WORK_DT = "dtSurrogate.json"
    FILE_METRICS_DT = "dtModel.json"
    FILE_METRICS_DT_MULTI_PREFIX = "dtModel_"
    FILE_WORK_DT_MULTI_PREFIX = "dtSurrogate_"
    FILE_DEFAULT_DETAILS = "dtModel.json"
    FILE_DEFAULT_TREE = "dtSurrogate.json"
    FILE_DRF_VAR_IMP = "varImp.json"

    DIR_DT_SURROGATE = "dt_surrogate_rules"

    # columns
    COLUMN_DT_PATH = "T1"
    COLUMN_DAI_PREDICT = "model_pred"
    COLUMN_ORIG_PRED = "orig_pred"
    COLUMN_MODEL_PRED = "model_pred"

    # dictionary keys
    KEY_LABELS_MAP = "labels_2_pd_map"

    ENC_AUTO = "AUTO"
    ENC_ONE_HOT = "One Hot Encoding"
    ENC_ENUM_LTD = "Enum Limited"
    ENC_SORT = "Sort by Response"
    ENC_LE = "Label Encoder"

    # categorical features encoding
    CAT_ENCODING_LIST = [
        ENC_AUTO,
        ENC_ONE_HOT,
        ENC_ENUM_LTD,
        ENC_SORT,
        ENC_LE,
    ]

    H2O_ENC_AUTO = "auto"
    H2O_ENC_ONE_HOT = "onehotexplicit"
    H2O_ENC_ENUM_LTD = "enumlimited"
    H2O_ENC_SORT = "sortbyresponse"
    H2O_ENC_LE = "labelencoder"

    H2O_ENCODING_NAMES = [
        H2O_ENC_AUTO,
        H2O_ENC_ONE_HOT,
        H2O_ENC_ENUM_LTD,
        H2O_ENC_SORT,
        H2O_ENC_LE,
    ]
    CAT_ENCODING_DICT = dict(zip(CAT_ENCODING_LIST, H2O_ENCODING_NAMES, strict=False))


class DecisionTreeSurrogateExplainer(explainers.Explainer, DecisionTreeConstants):
    """Surrogate decision tree explainer."""

    PARAM_DEBUG_RESIDUALS = "debug_residuals"
    PARAM_DEBUG_RESIDUALS_CLASS = "debug_residuals_class"
    PARAM_DT_DEPTH = "dt_tree_depth"
    PARAM_NFOLDS = "nfolds"
    PARAM_QBIN_COLS = "qbin_cols"
    PARAM_QBIN_COUNT = "qbin_count"
    PARAM_CAT_ENCODING = "categorical_encoding"

    _display_name = "Surrogate Decision Tree"
    _description = (
        "The surrogate decision tree is an approximate overall flow chart of "
        "the model, created by training a simple decision tree on "
        "the original inputs and the predictions of the model."
    )
    _iid = True
    _regression = True
    _binary = True
    _multiclass = True
    _time_series = True
    _global_explanation = True
    _local_explanation = True
    _explanation_types = [e10s.GlobalDtExplanation, e10s.LocalDtExplanation]
    _optional_explanation_types = [
        e10s.CustomArchiveExplanation,
        e10s.GlobalHtmlFragmentExplanation,
    ]
    _parameters = [
        explainers.ExplainerParam(
            param_name=PARAM_DEBUG_RESIDUALS,
            description="Debug model residuals.",
            param_type=commons.ExplainerParamType.bool,
            default_value=False,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_DEBUG_RESIDUALS_CLASS,
            description=(
                "Class for debugging classification model logloss residuals, "
                "empty string for debugging regression model residuals."
            ),
            param_type=commons.ExplainerParamType.str,
            # defaults:
            #   regression
            #     - empty class is the only valid value
            #   binomial
            #     - if "" default value, then positive class of interest is set,
            #       else class provided by the user is validated
            #   multinomial
            #     - if "" default value, then the first class is set,
            #       else class provided by the user is validated
            default_value="",
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_DT_DEPTH,
            description="Decision tree depth.",
            param_type=commons.ExplainerParamType.int,
            default_value=3,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_NFOLDS,
            description="Number of CV folds.",
            param_type=commons.ExplainerParamType.int,
            default_value=3,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_QBIN_COLS,
            description="Quantile binning columns.",
            param_type=commons.ExplainerParamType.list,
            default_value=None,
            tags=[explainers.ExplainerParam.TAG_SRC_DATASET_COLUMN_NAMES],
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_QBIN_COUNT,
            description="Quantile bins count.",
            param_type=commons.ExplainerParamType.int,
            default_value=0,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_CAT_ENCODING,
            description="Categorical encoding.",
            param_type=commons.ExplainerParamType.str,
            default_value=DecisionTreeConstants.H2O_ENC_ONE_HOT,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
            predefined=[
                DecisionTreeConstants.ENC_AUTO,
                DecisionTreeConstants.ENC_ONE_HOT,
                DecisionTreeConstants.ENC_ENUM_LTD,
                DecisionTreeConstants.ENC_SORT,
                DecisionTreeConstants.ENC_LE,
            ],
            comment=(
                "Specify one of the following encoding schemes for handling of "
                "categorical features:\n"
                "\n"
                "_**AUTO**_: 1 column per categorical feature.\n"
                "\n"
                "_**Enum Limited**_: Automatically reduce categorical levels to "
                "the most prevalent ones during training and only keep the top 10 most "
                "frequent levels.\n"
                "\n"
                "_**One Hot Encoding**_: N+1 new columns for categorical features with "
                "N levels.\n"
                "\n"
                "_**Label Encoder**_: Convert every enum into the integer of its index "
                "(for example, level 0 -> 0, level 1 -> 1, etc.).\n"
                "\n"
                "_**Sort by Response**_: Reorders the levels by the mean response "
                "(for example, the level with lowest response -> 0, the level "
                "with second-lowest response -> 1, etc.)."
            ),
        ),
    ]
    _requires_predict_method = False
    _priority = 11.0
    _keywords = [
        explainers.Explainer.KEYWORD_DEFAULT,
        explainers.Explainer.KEYWORD_REQUIRES_H2O3,
        explainers.SurrogateExplainer.KEYWORD_SURROGATE,
        explainers.Explainer.KEYWORD_EXPLAINS_APPROX_BEHAVIOR,
        explainers.Explainer.KEYWORD_H2O_SONAR,
    ]
    _requires_preloaded_predictor = False
    _modules_needed_by_name = ["h2o"]

    @staticmethod
    def is_enabled() -> bool:
        return True

    def __init__(self):
        explainers.Explainer.__init__(self)

        self.args = None
        # sanitized stringified labels
        self.labels = None
        self.log_name = "Surrogate decision tree"

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        model: models.ExplainableModel | None = None,
        **explainer_params,
    ) -> bool:
        explainers.Explainer.check_compatibility(self, params, **explainer_params)

        if not HAS_H2O:
            self.logger.warning(self._check_compatibility_pckg_err_msg("h2o"))
            return False

        if not HAS_PYDOT:
            self.logger.warning(
                self.logger.warning(self._check_compatibility_pckg_err_msg("pydot"))
            )
            return False

        if not self.check_required_modules(
            set(DecisionTreeSurrogateExplainer._modules_needed_by_name)
        ):
            self.logger.warning(
                f"{self.log_name} not compatible as the following required Python "
                f"modules are not installed: "
                f"{DecisionTreeSurrogateExplainer._modules_needed_by_name}"
            )
            return False

        if not model:
            err_msg = f"{self._display_name} requires a model"
            self.logger.warning(err_msg)
            raise errors.ExplainerCompatibilityError(err_msg)

        if model.meta.is_constant:
            err_msg = f"{self._display_name} is not applicable to constant model type"
            self.logger.warning(err_msg)
            raise errors.ExplainerCompatibilityError(err_msg)

        if not params.target_col:
            err_msg = (
                f"{self.log_name} explainer requires target column and therefore it is "
                f"not compatible with explanation run with parameters: {params.dump()}"
            )
            self.logger.warning(err_msg)
            raise errors.ExplainerCompatibilityError(err_msg)

        # H2O-3 requires to have at least 2 training columns for DT model
        try:
            if self.dataset_meta.column_names:
                # 1 column must be target -> at least 3 columns required
                if len(self.dataset_meta.column_names) < 3:
                    err_msg = (
                        f"{self.log_name}: explainer requires at least two training "
                        f"columns for H2O-3 model, but has "
                        f"{len(self.dataset_meta.column_names) - 1} or less"
                    )
                    self.logger.warning(err_msg)
                    raise errors.ExplainerCompatibilityError(err_msg)
        except Exception as ex:
            err_msg = f"{self.log_name}: unable to check explainer compatibility: {ex}"
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

        self.args = explainers.ExplainerArgs(self._parameters)
        self.args.resolve_params(
            explainer_params=explainers.ExplainerArgs.json_str_to_dict(
                self.explainer_params_as_str
            ),
        )
        self.log_name: str = f"Surrogate decision tree {self.mli_key}/{self.key}"

    def explain(
        self,
        X: datatable.Frame,
        y: datatable.Frame | None = None,
        explanations_types: list = None,
        **explainer_params,
    ) -> list:
        del explanations_types
        del explainer_params

        if not self.model_meta.used_features:
            raise ValueError(
                "Features used by the model must be specified in model metadata, "
                "but the list of used features is empty"
            )

        do_debug_residuals: bool = self.args.get(
            DecisionTreeSurrogateExplainer.PARAM_DEBUG_RESIDUALS, False
        )
        debug_residuals_class: str = str(
            self.args.get(
                DecisionTreeSurrogateExplainer.PARAM_DEBUG_RESIDUALS_CLASS, ""
            )
        )

        self._explain_dt(dataset=X, y=y)

        # process raw DT method output (from work/ directory) to explanations
        explanations = list()

        self._normalize_data(
            explanations,
            do_debug_residuals=do_debug_residuals,
            debug_residuals_class=debug_residuals_class,
        )

        # archive: DT rules as ZIP archive (all files provided via snapshot)
        dt_rules_zip_path = self.persistence.get_explainer_working_file(
            "dt_surrogate_rules.zip"
        )
        if (
            self.config
            and self.config.enable_dataset_downloading
            and os.path.isfile(dt_rules_zip_path)
        ):
            work_explanation = e10s.CustomArchiveExplanation(
                explainer=self,
                display_name=(
                    f"{'Residual ' if do_debug_residuals else ''}Decision tree "
                    f"surrogate rules ZIP archive"
                ),
                display_category=(
                    e10s.CustomArchiveExplanation.DISPLAY_CAT_SURROGATES_ON_RES
                    if do_debug_residuals
                    else e10s.CustomArchiveExplanation.DISPLAY_CAT_SURROGATES
                ),
            )
            work_explanation.add_format(
                f5s.CustomArchiveZipFormat(
                    explanation=work_explanation, format_file=dt_rules_zip_path
                )
            )
            explanations.append(work_explanation)

        return explanations

    def _normalize_data(
        self,
        explanations,
        do_debug_residuals: bool = False,
        explainer_data_dir_path: str = "",
        debug_residuals_class: str | int = "",
    ):
        # global explanation
        try:
            global_dt_explanation = e10s.GlobalDtExplanation(
                explainer=self,
                display_name=(
                    f"{'Residual ' if do_debug_residuals else ''}Decision Tree"
                ),
                display_category=(
                    e10s.GlobalDtExplanation.DISPLAY_CAT_SURROGATES_ON_RES
                    if do_debug_residuals
                    else e10s.GlobalDtExplanation.DISPLAY_CAT_SURROGATES
                ),
            )
            (
                json_global_representation,
                labels_2_pd_map,
            ) = self._work_dt_explanation_to_global_dt_json(
                explanation=global_dt_explanation,
                explainer_data_dir_path=explainer_data_dir_path,
                debug_model_errors_class=debug_residuals_class,
            )
            global_dt_explanation.add_format(json_global_representation)
            explanations.append(global_dt_explanation)
        except Exception as ex:
            self.logger.error(
                f"{self.log_name}: JSon representation creation failed: {ex}\n"
                f"{traceback.format_exc()}",
            )
            # fail fast (no global normalized representations, local w/o global)
            raise ex

        # local explanation
        if global_dt_explanation:
            try:
                self._work_local_explanation()

                local_dt_explanation = e10s.LocalDtExplanation(
                    explainer=self,
                    display_name="Local DT",
                    display_category=e10s.GlobalDtExplanation.DISPLAY_CAT_SURROGATES,
                )

                # persist parameters needed for on-demand calculation - if needed
                json_local_idx, _ = f5s.LocalDtJSonFormat.serialize_index_file(
                    classes=[], doc=DecisionTreeSurrogateExplainer._description
                )
                json_local_idx[f5s.LocalDtJSonFormat.KEY_ON_DEMAND] = True
                on_demand_params: dict = dict()
                on_demand_params[DecisionTreeConstants.KEY_LABELS_MAP] = labels_2_pd_map
                on_demand_params[f5s.LocalDtJSonFormat.KEY_SYNC_ON_DEMAND] = True
                json_local_idx[f5s.LocalDtJSonFormat.KEY_ON_DEMAND_PARAMS] = (
                    on_demand_params
                )
                on_demand_params[f5s.LocalDtJSonFormat.KEY_IS_MULTI] = (
                    len(self.model_meta.labels) > 2 if self.model_meta.labels else False
                )
                on_demand_params[f5s.LocalDtJSonFormat.KEY_RAW_FEATURES] = (
                    self.params.use_raw_features
                )

                local_dt_explanation.add_format(
                    explanation_format=f5s.LocalDtJSonFormat(
                        explanation=local_dt_explanation,
                        json_data=json.dumps(json_local_idx, indent=4),
                    )
                )

                # associate local explanation with the global one
                global_dt_explanation.has_local = (
                    local_dt_explanation.explanation_type()
                )

                explanations.append(local_dt_explanation)
            except Exception as ex:
                self.logger.error(
                    f"{self.log_name}: local representation creation failed for "
                    f"{self.mli_key}/{self.key}: {ex}\n{traceback.format_exc()}",
                )
                # don't fail fast as (at least) global explanation is available)

        # OPTIONAL explanation: HTML fragment
        if self.config and self.config.create_html_representations:
            try:
                dt_html = e10s.GlobalHtmlFragmentExplanation(
                    explainer=self,
                    display_name=DecisionTreeSurrogateExplainer._display_name,
                    display_category=(
                        e10s.CustomArchiveExplanation.DISPLAY_CAT_SURROGATES_ON_RES
                        if do_debug_residuals
                        else e10s.CustomArchiveExplanation.DISPLAY_CAT_SURROGATES
                    ),
                )
                dt_html.add_format(self._explain_html(dt_html))

                explanations.append(dt_html)
            except Exception as ex:
                self.logger.warning(
                    f"{self.log_name}: HTML representation creation failed: {ex}"
                )

    def _explain_html(
        self,
        explanation: e10s.GlobalHtmlFragmentExplanation,
    ) -> f5s.HtmlFormat:
        """Build HTML representation."""
        html_format = f5s.HtmlFormat(
            explanation=explanation,
            format_data=f5s.HtmlFormat.MINIMAL_HTML,
            persistence=self.persistence.store,
        )

        # resolve sanitized classes
        dt_idx_dict: dict = f5s.GlobalDtJSonFormat.load_index_file(
            persistence=self.persistence,
            explanation_type=e10s.GlobalDtExplanation.explanation_type(),
        )
        classes = []
        if dt_idx_dict:
            dt_idx_files = dt_idx_dict.get(f5s.GlobalDtJSonFormat.KEY_FILES, None)
            if dt_idx_files:
                classes = list(dt_idx_files.keys())
        classes = classes or [None]

        html_src = airium.Airium()
        result = self.get_result()

        for i, clazz in enumerate(classes):
            with html_src.b():
                msg_suffix = f" for the class '{clazz}'" if clazz else ""
                html_src(f"Approximate model behavior{msg_suffix}:")
            with html_src.div():
                graphwiz_graph = result.plot(clazz=clazz)
                dot_file_path = self.persistence.get_explainer_working_file(
                    f"dt-class-{i}.dot"
                )
                graphwiz_graph.render(filename=dot_file_path)
                (pydot_graph,) = pydot.graph_from_dot_file(dot_file_path)
                file_path = self.persistence.get_explanation_file_path(
                    explanation_type=explanation.explanation_type(),
                    explanation_format=html_format.mime,
                    explanation_file=f"dt-class-{i}.png",
                )
                pydot_graph.write_png(file_path)

                html_src.img(
                    src=self.persistence.get_relative_path(file_path),
                    alt=f"Decision tree for class '{clazz}'",
                    # ensure that image will not overflow enclosing <div/>
                    style=(
                        "height: 100%; max-width: 100%; display: block; margin: auto;"
                    ),
                )

        html_format.update_data(
            str(html_src), f"{persistences.ExplainerPersistence.FILE_EXPLANATION}.html"
        )

        return html_format

    def explain_local(
        self, X: datatable.Frame, y: datatable.Frame = None, **extra_params
    ) -> str:
        """DT surrogate local explanation requires SYNCHRONOUS execution.

        Parameters
        ----------
        X : datatable.Frame
            Dataset frame.
        y : datatable.Frame, optional
            Labels.
        extra_params : dict
            Extra parameters including:

              - persistence: Persistence object initialized for explainer/MLI run
              - explanation_type: Explanation type ~ explainer ID
              - row: Local explanation to be provided for given row
              - explanation_filter: Required filter entries (class)

        Returns
        -------
        str
            JSon representation of the local explanation.

        Notes
        -----
        JSon DT representation::

            {
                data: [
                    {
                      key: str,
                      name: str,
                      parent: str,
                      edge_in: str,
                      edge_weight: num,
                      leaf_path: bool,
                    }+
                ]
            }

        """
        # persistence and explanation type is checked by caller
        user = extra_params.get("user", None)
        if not user:
            raise ValueError("Local explanation handler is missing 'user' parameter")
        dbc = extra_params.get("dbc", None)
        if not dbc:
            raise ValueError("Local explanation handler is missing 'dbc' parameter")
        mli_key = extra_params.get("mli_key", None)
        if not mli_key:
            raise ValueError("Local explanation handler is missing 'mli_key' parameter")
        mli_get_frame_rows = extra_params.get("legacy_extractor", None)
        if not mli_get_frame_rows:
            raise errors.MliError(
                "Legacy explanation handler is missing legacy helper method"
            )
        row: int = extra_params.get("row", None)
        if row is None or not isinstance(row, int) or row < 0:
            raise ValueError(
                "Local explanation handler requires row index to be positive integer"
            )
        explanation_type = extra_params.get("explanation_type", None)
        if extra_params.get(f5s.LocalDtJSonFormat.KEY_ON_DEMAND_PARAMS, None):
            is_multinomial = extra_params[
                f5s.LocalDtJSonFormat.KEY_ON_DEMAND_PARAMS
            ].get(f5s.LocalDtJSonFormat.KEY_IS_MULTI, False)
            use_raw_features = extra_params[
                f5s.LocalDtJSonFormat.KEY_ON_DEMAND_PARAMS
            ].get(f5s.LocalDtJSonFormat.KEY_RAW_FEATURES, False)
        else:
            is_multinomial = False
            use_raw_features = False

        persistence: persistences.ExplainerPersistence = extra_params.get(
            "persistence", None
        )
        logger = extra_params.get("logger", None)
        if not logger:
            raise ValueError("Local explanation handler requires logger")
        explanation_filter: list[commons.FilterEntry] = extra_params.get(
            "explanation_filter", None
        )
        if not explanation_filter:
            raise ValueError(
                f"Local explanation filter parameters "
                f"'{f5s.LocalDtJSonFormat.FILTER_FEATURE}' and "
                f"'{f5s.LocalDtJSonFormat.FILTER_CLASS}' are required"
            )
        e_filter: dict = dict()
        for i in explanation_filter:
            e_filter[i.filter_by] = i.value
        filter_clazz: str = e_filter.get(f5s.LocalDtJSonFormat.FILTER_CLASS, None)
        if filter_clazz is None:
            raise ValueError(
                f"Local explanation filter parameter "
                f"'{f5s.LocalDtJSonFormat.FILTER_CLASS}' is required"
            )
        e_filter_clazz = persistences.Persistence.safe_name(
            sanitization.SanitizationMap.sanitize_value(filter_clazz)
        )
        frame_name: str = (
            "dtpaths_frame.bin"
            if not is_multinomial
            else f"dtpaths_frame_{e_filter_clazz}.bin"
        )
        frame_path: str = persistence.get_explainer_working_file(frame_name)
        mojo_work_path: str = persistence.get_explainer_working_file(
            f"{e_filter_clazz}_mojo"
        )
        if not is_multinomial and not os.path.isfile(frame_path):
            raise errors.MliError(
                f"Unable to load local DT surrogate explanation as frame file does "
                f"not exist (regression/binomial): {frame_path}"
            )
        elif is_multinomial and not os.path.isdir(mojo_work_path):
            raise errors.MliError(
                f"Unable to load local DT surrogate explanation as MOJO dir does "
                f"not exist (multinomial): {mojo_work_path}"
            )

        # SYNC call:
        raw_local_dict = mli_get_frame_rows(
            dbc,
            frame_name,  # wrong setting can block dispatch
            row,
            1,
            mli_key,  # get_frame_rows() uses job_key
            use_raw_features,  # wrong setting can block dispatch
            sanitization.SanitizationMap.sanitize_value(filter_clazz),
            logger,
            explainer_dir=persistence.get_explainer_working_dir(),
        )

        dt_json = json.loads(raw_local_dict)
        path = dt_json["data"]

        if self.is_time_series():
            ts_data_dict = dict(
                zip(dt_json["columns"], dt_json["data"][0], strict=False)
            )
            path = ts_data_dict[DecisionTreeConstants.COLUMN_DT_PATH]

        path = path[0] if isinstance(path, list) and len(path) else path
        path = path[0] if isinstance(path, list) and len(path) else path
        path_key = f5s.LocalDtJSonFormat.dt_path_to_node_key(path)

        # find file for class using index
        idx_dict = persistence.store.load_json(
            persistence.get_explanation_file_path(
                explanation_type=explanation_type.replace("local-", "global-"),
                explanation_format=f5s.LocalDtJSonFormat.mime,
            )
        )
        explanations_file = None
        for cls in idx_dict[f5s.GlobalDtJSonFormat.KEY_FILES]:
            if filter_clazz == cls:
                explanations_file = idx_dict[f5s.GlobalDtJSonFormat.KEY_FILES][cls]
        if explanations_file:
            local_file_path: str = persistence.get_explanation_file_path(
                explanation_type=explanation_type.replace("local-", "global-"),
                explanation_format=f5s.LocalDtJSonFormat.mime,
                explanation_file=explanations_file,
            )

            if os.path.isfile(local_file_path):
                with open(local_file_path) as json_file:
                    local_explanation_dict = json.load(json_file)
                if local_explanation_dict:
                    return json.dumps(
                        f5s.LocalDtJSonFormat.dt_set_tree_path(
                            key=path_key, tree=local_explanation_dict
                        )
                    )
            else:
                raise errors.MliError(
                    f"Local DT explanation: file not found {local_file_path}"
                )
        raise errors.MliError(
            f"Local DT explanation: class '{filter_clazz}' not found in index file "
            f"{idx_dict}"
        )

    def _diagnose_problems(self, rules_obj: rules.Rules, label=None):
        if rules_obj and rules_obj.rules:
            # find rule w/ highest value
            max_value = None
            max_rule = None
            for r in rules_obj.rules:
                if isinstance(r.rule_value, (int, float)):
                    if max_value is None or max_value < r.rule_value:
                        max_value = r.rule_value
                        max_rule = r

            if max_rule:
                label_msg = (
                    f"for class '{label}'" if label is not None and label != "" else ""
                )
                problem = problems.ProblemAndAction(
                    description=(
                        f"A path in the residual surrogate decision tree "
                        f"{label_msg if label_msg else ''} leading to the largest "
                        f"residual ({max_rule.rule_value}) may indicate a problem "
                        f"in the model."
                    ),
                    severity=problems.ProblemSeverity.low,
                    problem_type="bias",
                    problem_attrs={
                        problems.ProblemAndAction.ATTR_EVALUATOR_NAME: (
                            DecisionTreeSurrogateExplainer._display_name
                        ),
                    },
                    actions_description=(
                        f"Verify that the following surrogate decision tree path "
                        f"does not indicate a model bias or other problem: "
                        f"{max_rule}"
                    ),
                    explainer_id=self.explainer_id(),
                    explainer_name=self._display_name,
                    explanation_type=e10s.GlobalDtExplanation.explanation_type(),
                    explanation_name=e10s.GlobalDtExplanation.__name__,
                    explanation_mime=f5s.GlobalDtJSonFormat.mime,
                    resources=[],
                )
                self.add_problem(problem)

    def _work_dt_explanation_to_global_dt_json(
        self,
        explanation: e10s.GlobalDtExplanation,
        explainer_data_dir_path: str,
        debug_model_errors_class: str | int,
    ) -> tuple[f5s.GlobalDtJSonFormat, dict]:
        debug_model_errors_class = (
            None if debug_model_errors_class == "none" else debug_model_errors_class
        )

        default_class = ""
        if self.model_meta.labels:
            # do NOT sanitize labels - user must see data in non-sanitized space
            # (index file handles safe filesystem names; DT models are escaped as well)
            sanitized_labels = [str(label) for label in self.model_meta.labels]
            # bin experiment's Model has class of interest @ index 1 > make it default
            if len(sanitized_labels) == 2:
                sanitized_labels = [sanitized_labels[1]]
                default_class = (
                    sanitized_labels[0] if not debug_model_errors_class else ""
                )
        else:
            sanitized_labels = [f5s.GlobalDtJSonFormat.LABEL_REGRESSION]

        metrics: list | dict = []
        if len(sanitized_labels) <= 2:
            # Get DT metrics, which are in global state
            work_dt_metrics_path = os.path.join(
                self.persistence.get_explainer_working_dir(),
                DecisionTreeConstants.FILE_METRICS_DT,
            )
            if os.path.isfile(work_dt_metrics_path):
                with open(work_dt_metrics_path) as json_file:
                    work_dt_metrics_dict = json.load(json_file)
                metrics = self._get_dt_metrics(work_dt_metrics_dict)
            # else empty list
        else:
            metrics_per_class: dict = {}
            for label in sanitized_labels:
                e_file_label = persistences.Persistence.safe_name(
                    sanitization.SanitizationMap.sanitize_value(label)
                )
                # Get DT metrics, which are in global state
                work_dt_metrics_path = os.path.join(
                    self.persistence.get_explainer_working_dir(),
                    f"{DecisionTreeConstants.FILE_METRICS_DT_MULTI_PREFIX}"
                    f"{e_file_label}.json",
                )
                if os.path.isfile(work_dt_metrics_path):
                    with open(work_dt_metrics_path) as json_file:
                        work_dt_metrics_dict = json.load(json_file)
                    metrics_per_class[label] = self._get_dt_metrics(
                        work_dt_metrics_dict
                    )
                # else explanation does NOT exist > do NOT add item to avoid FE failure
            if len(metrics_per_class) == 1:
                # residuals
                metrics = next(iter(metrics_per_class.values()))
            else:
                # multinomial
                metrics = metrics_per_class
            # else empty list

        (idx_dict, idx_str) = f5s.GlobalDtJSonFormat.serialize_index_file(
            classes=(
                [debug_model_errors_class]
                if len(sanitized_labels) > 2 and debug_model_errors_class
                else sanitized_labels
            ),
            default_class=default_class,
            metrics=metrics,
            doc=DecisionTreeSurrogateExplainer._description,
        )

        json_representation = f5s.GlobalDtJSonFormat(
            explanation=explanation, json_data=idx_str
        )

        if not explainer_data_dir_path:
            explainer_data_dir_path = self.persistence.get_explainer_working_dir()

        labels_2_pd_map = dict()
        work_dt_path: str = ""
        if len(sanitized_labels) <= 2:
            # regression and binomial: feature importance is identical
            work_dt_path: str = os.path.join(
                explainer_data_dir_path,
                DecisionTreeConstants.FILE_WORK_DT,
            )
            if os.path.isfile(work_dt_path):
                with open(work_dt_path) as json_file:
                    work_dt_dict = json.load(json_file)
                dt_root = DecisionTreeSurrogateExplainer._work_dt_to_global_dt_json(
                    work_dt_dict
                )
                data_str = f5s.GlobalDtJSonFormat.serialize_data_file(dt_root)
                for i, label in enumerate(sanitized_labels):
                    file_name: str = idx_dict[f5s.GlobalDtJSonFormat.KEY_FILES][label]
                    json_representation.add_data(
                        format_data=data_str, file_name=file_name
                    )

                    labels_2_pd_map[label] = i

                return json_representation, labels_2_pd_map
            # else: DT compatibility check or computation failed
        else:
            debug_model_errors_class_idx = None
            if debug_model_errors_class:
                debug_model_errors_class_idx = sanitized_labels.index(
                    str(debug_model_errors_class)
                )
                sanitized_labels = [debug_model_errors_class]
            # multinomial: feature importance is identical
            valid_class_explanations = 0

            for i, label in enumerate(sanitized_labels):
                # label quoting: h2o_sonar.py::export_shapley_global_multinomial()
                e_file_label = persistences.Persistence.safe_name(
                    sanitization.SanitizationMap.sanitize_value(label)
                )

                work_dt_path = os.path.join(
                    explainer_data_dir_path,
                    (
                        f"{DecisionTreeConstants.FILE_WORK_DT_MULTI_PREFIX}"
                        f"{e_file_label}.json"
                    ),
                )

                if debug_model_errors_class_idx:
                    labels_2_pd_map[e_file_label] = debug_model_errors_class_idx
                else:
                    labels_2_pd_map[e_file_label] = i

                if os.path.isfile(work_dt_path):
                    with open(work_dt_path) as json_file:
                        work_dt_dict = json.load(json_file)
                    dt_root = DecisionTreeSurrogateExplainer._work_dt_to_global_dt_json(
                        work_dt_dict
                    )
                    json_representation.add_data(
                        format_data=f5s.GlobalDtJSonFormat.serialize_data_file(dt_root),
                        file_name=idx_dict[f5s.GlobalDtJSonFormat.KEY_FILES][label],
                    )
                    valid_class_explanations += 1
                # else: DT compatibility check or computation for given class failed

            if valid_class_explanations:
                return json_representation, labels_2_pd_map

        # DT creation failed/skipped - it can be DT compatibility check and/or
        # error which should always be logged. No explanation in this case
        raise errors.MliError(
            f"Global feature importance file not found in working directory of "
            f"{self.mli_key}/{self.explainer_id()}: {work_dt_path}",
        )

    @staticmethod
    def _get_dt_metrics(work_dt_metrics_dict):
        metrics = []

        # Define metrics
        mse = work_dt_metrics_dict["_validation_metrics"]["_MSE"]
        var = work_dt_metrics_dict["_validation_metrics"]["_sigma"] ** 2
        r2 = "undefined" if var == 0 else 1 - mse / var  # var might be 0 ~ inf v undef

        # Make dictionary of metrics and add to metrics list
        train_rmse_dict = {"Training RMSE: ": mse**0.5}
        r2_dict = {"Training R2: ": r2}
        metrics.extend([train_rmse_dict, r2_dict])

        if work_dt_metrics_dict["_cross_validation_models"]:
            # Define metrics
            nfolds = len(work_dt_metrics_dict["_cross_validation_models"])
            cv_var = work_dt_metrics_dict["_cross_validation_metrics"]["_sigma"] ** 2
            cv_mse = work_dt_metrics_dict["_cross_validation_metrics"]["_MSE"]
            cv_r2 = "undefined" if cv_var == 0 else 1 - cv_mse / cv_var

            # Make dictionary of metrics and add to metrics list
            cv_rmse_dict = {f"Mean {nfolds} fold RMSE:": cv_mse**0.5}
            cv_r2_dict = {f"Mean {nfolds} fold R2: ": cv_r2}
            metrics.extend([cv_rmse_dict, cv_r2_dict])

        return metrics

    @staticmethod
    def _work_dt_to_global_dt_json(
        work_dt_dict: dict,
    ) -> f5s.GlobalDtJSonFormat.TreeNode:
        if not work_dt_dict or not work_dt_dict.keys():
            raise errors.MliError("Work DT representation has no nodes")

        root: f5s.GlobalDtJSonFormat.TreeNode = f5s.GlobalDtJSonFormat.TreeNode(
            name=work_dt_dict.get("name", ""),
            parent=None,
            edge_in=None,
            # TODO define constants
            edge_weight=float(work_dt_dict.get("edgeweight", "0")),
            leaf_path=False,
            # TODO define constants
            total_weight=work_dt_dict.get("totalweight", ""),
            weight=work_dt_dict.get(
                "totalweight", ""
            ),  # weight == total_weight for root node
        )
        DecisionTreeSurrogateExplainer._work_dt_children_to_tree(
            parent=root, work_node=work_dt_dict
        )

        return root

    @staticmethod
    def _work_dt_node_has_children(work_node: dict) -> list:
        if not work_node:
            return []
        children = work_node.get(f5s.GlobalDtJSonFormat.KEY_CHILDREN, None)
        if not children or (
            len(children) == 2 and children[0] is None and children[1] is None
        ):
            return []

        return work_node[f5s.GlobalDtJSonFormat.KEY_CHILDREN]

    @staticmethod
    def _work_dt_children_to_tree(
        parent: f5s.GlobalDtJSonFormat.TreeNode, work_node: dict
    ):
        children: list = DecisionTreeSurrogateExplainer._work_dt_node_has_children(
            work_node
        )
        if not children:
            return

        for idx, child in enumerate(children):
            new_tree_node = f5s.GlobalDtJSonFormat.TreeNode(
                name=child.get("name", child.get("value", "")),
                parent=parent,
                edge_in=child.get("edgein", "") or '""',
                edge_weight=float(child.get("edgeweight", "0")),
                total_weight=float(child.get("totalweight", "0")),
                weight=float(child.get("weight", "0")),
                leaf_path=False,
                key=f"{parent.key}.{idx}",
            )
            DecisionTreeSurrogateExplainer._work_dt_children_to_tree(
                parent=new_tree_node, work_node=child
            )

    #
    # local explanation
    #

    def _work_local_explanation(self):
        # files prepared by parent explainer(s)
        dep_files = [
            # transformed (munged) dataset for fitted (transformed features) MOJO
            "full_fitted_pipeline.pickle",
            "train_munged.bin",
        ]

        parent_explainer_id: str = (
            surrogate_rf_explainer.SurrogateRandomForestExplainer.explainer_id()
        )

        if self.explainer_deps:
            parent_explainer_job_key: str = self.explainer_deps.get(
                parent_explainer_id, None
            )
            p_persistence: persistences.ExplainerPersistence = (
                persistences.ExplainerPersistence(
                    data_dir=self.persistence.data_dir,
                    username=self.persistence.username,
                    explainer_id=parent_explainer_id,
                    mli_key=self.mli_key,
                    explainer_job_key=parent_explainer_job_key,
                )
            )

            for dep_file in dep_files:
                dep_file_path = p_persistence.get_explainer_working_file(dep_file)
                if os.path.isfile(dep_file_path):
                    shutil.copyfile(
                        dep_file_path,
                        self.persistence.get_explainer_working_file(dep_file),
                    )
                else:
                    self.logger.warning(
                        f"{self.log_name}: unable to copy required file "
                        f"{dep_file} from parent {parent_explainer_id} work directory "
                        f"{dep_file_path} - local DT explanations for original "
                        f"features will not be available",
                    )

    def _explain_dt_residual(
        self,
        num_labels,
        labels,
        debug_model_errors_class,
        sanitized_df,
        target,
        pred_col: str = DecisionTreeConstants.COLUMN_MODEL_PRED,
        preds_mult_prob: datatable.Frame = None,
    ):
        if labels and len(labels) > 2:
            # make sure debug_model_errors_class corresponding to column name
            sanitized_df[:, DecisionTreeConstants.COLUMN_ORIG_PRED] = sanitized_df[
                :, str(debug_model_errors_class)
            ]
        else:
            sanitized_df[:, DecisionTreeConstants.COLUMN_ORIG_PRED] = sanitized_df[
                :, pred_col
            ]

        # ensure target name is sanitized
        sanitized_target = sanitization.sanitize_strings(target)
        if sanitized_target in sanitized_df.names:
            target = sanitized_target

        debug_model_errors_class = self._sanitize_debug_model_error_class(
            debug_model_errors_class=debug_model_errors_class,
            frame=sanitized_df,
            target=target,
            labels=labels,
        )

        # filter down frame to error class of interest if binomial or multinomial
        if num_labels >= 2:
            # ensure residuals class type for datatable filtering
            if (
                sanitized_df.ltypes[sanitized_df.colindex(target)]
                == datatable.ltype.bool
            ):
                dt_r_class = bool(debug_model_errors_class)
            elif (
                sanitized_df.ltypes[sanitized_df.colindex(target)]
                == datatable.ltype.int
            ):
                dt_r_class = int(debug_model_errors_class)
            elif (
                sanitized_df.ltypes[sanitized_df.colindex(target)]
                == datatable.ltype.str
            ):
                dt_r_class = str(debug_model_errors_class)
            elif (
                sanitized_df.ltypes[sanitized_df.colindex(target)]
                == datatable.ltype.real
            ):
                dt_r_class = float(debug_model_errors_class)
            else:
                dt_r_class = debug_model_errors_class

            sanitized_df = sanitized_df[datatable.f[target] == dt_r_class, :]

        debug_model_errors_class = str(debug_model_errors_class)

        if (num_labels > 2 and debug_model_errors_class and labels) or (
            num_labels > 2 and debug_model_errors_class == 0 and labels
        ):  # multinomial
            sanitized_labels = [str(label) for label in labels]
            if preds_mult_prob or all(
                label in sanitized_df.names for label in sanitized_labels
            ):
                self.logger.info(
                    f"{self.log_name}: calculating logloss residuals "
                    f"(multinomial problem) ...",
                )
                self.logger.info(
                    f"{self.log_name}: debug model errors class: "
                    f"<<<{debug_model_errors_class}>>>",
                )
                sanitized_df[:, pred_col] = sanitized_df[
                    :,
                    -1 * datatable.log(datatable.f[str(debug_model_errors_class)]),
                ]
            else:
                raise ValueError(
                    f"{self.log_name} Missing multinomial probabilities per class frame"
                )
        elif (num_labels == 2 and debug_model_errors_class and labels) or (
            num_labels == 2 and debug_model_errors_class == 0 and labels
        ):  # binomial
            sanitized_labels = [str(label) for label in labels]
            self.logger.info(
                f"{self.log_name}: calculating logloss residuals "
                "(binary classification problem) ...",
            )

            # ensure labels are sorted lexicographically/numerically(ascending)
            sanitized_labels.sort()
            self.logger.info(
                f"{self.log_name}: sorted labels for residual calculation: "
                f"<<<{sanitized_labels}>>>",
            )
            self.logger.info(
                f"{self.log_name}: debug model errors class: "
                f"<<<{debug_model_errors_class}>>>",
            )

            if debug_model_errors_class in sanitized_labels:
                target_index = sanitized_labels.index(debug_model_errors_class)
            else:
                raise errors.MliError(
                    f"{self.log_name}: residual debug class "
                    f"'{debug_model_errors_class}' ({type(debug_model_errors_class)}) "
                    f"is invalid, it must be one of: {sanitized_labels} "
                    f"{'' if not sanitized_labels else f'({type(sanitized_labels)})'}"
                )
            self.logger.info(
                f"{self.log_name}: label index for class of interest: "
                f"<<<{target_index}>>>",
            )
            sanitized_df[:, pred_col] = sanitized_df[
                :,
                -target_index * datatable.log(datatable.f[pred_col])
                - (1 - target_index) * datatable.log(1 - datatable.f[pred_col]),
            ]
        else:  # regression
            self.logger.info(
                f"{self.log_name}: calculating squared residuals "
                f"(regression problem) ..."
            )

            sanitized_df[:, DecisionTreeConstants.COLUMN_MODEL_PRED] = sanitized_df[
                :,
                datatable.math.square(
                    datatable.f[target]
                    - datatable.f[DecisionTreeConstants.COLUMN_MODEL_PRED]
                ),
            ]

        return sanitized_df, debug_model_errors_class

    def _save_all_decision_tree_details(
        self,
        dt_h2o3: _decision_tree_h2o.DecisionTreeH2O,
        mli_dir_path: str,
        is_multinomial: bool,
        e_predict_col: str,
        features_meta: dict,
    ):
        details_filename = (
            DecisionTreeConstants.FILE_DEFAULT_DETAILS
            if not is_multinomial
            else f"dtModel_{e_predict_col}.json"
        )

        tree_filename = (
            DecisionTreeConstants.FILE_DEFAULT_TREE
            if not is_multinomial
            else f"dtSurrogate_{e_predict_col}.json"
        )

        try:
            self._patch_decision_tree_value_types(
                details=self._save_decision_tree_meta(
                    mli_dir_path=mli_dir_path,
                    filename=details_filename,
                    save_method=dt_h2o3.save_model_details,
                ),
                tree_json=self._save_decision_tree_meta(
                    mli_dir_path=mli_dir_path,
                    filename=tree_filename,
                    save_method=dt_h2o3.save_dt_tree_json,
                ),
                features_metadata=features_meta,
                filepath=str(pathlib.Path(mli_dir_path) / tree_filename),
            )
        except Exception as ex:
            self.logger.warning(
                f"{self.log_name}: saving original version of decision tree as its "
                f"patching to provide human-friendly time and date/time failed with: "
                f"{ex}\n{traceback.format_exc()}",
            )
            # safe NON-patched trees
            if not os.path.isfile(os.path.join(mli_dir_path, details_filename)):
                dt_h2o3.save_model_details(path=mli_dir_path, filename=details_filename)
            if not os.path.isfile(os.path.join(mli_dir_path, tree_filename)):
                dt_h2o3.save_dt_tree_json(path=mli_dir_path, filename=tree_filename)

    def _run_dt_surrogate(
        self,
        dataset,
        dt_tree_depth: int,
        nfolds: int,
        mli_dir_path: str,
        predict_col: str,
        weights_column: str,
        ignored_columns: list,
        is_multinomial: bool,
        label: str,
        debug_model_errors: bool,
        cat_encoder: str,
    ):
        dai_config = {"h2o_url": self.config.h2o_host, "h2o_port": self.config.h2o_port}
        # TODO: get rid of _mli.MLI
        mli = _mli.MLI(mli_dir_path, DecisionTreeConstants.SEED, config=dai_config)
        # escape class/label name as it's not safe file system name
        e_predict_col = (
            persistences.Persistence.safe_name(
                sanitization.SanitizationMap.sanitize_value(label)
            )
            if is_multinomial
            else ""
        )
        response_col = (
            DecisionTreeConstants.COLUMN_DAI_PREDICT
            if not predict_col
            else sanitization.SanitizationMap.sanitize_value(predict_col)
        )
        if ignored_columns:
            ignored_columns = [
                self.model_meta.sanitization_map.to_sanitized(c)
                for c in ignored_columns.copy()
            ]
            if debug_model_errors:
                ignored_columns = ignored_columns + [
                    DecisionTreeConstants.COLUMN_ORIG_PRED
                ]
        else:
            if debug_model_errors:
                ignored_columns = [DecisionTreeConstants.COLUMN_ORIG_PRED]

        # connect to MLI instance
        self.logger.info(
            f"{self.log_name}: connecting to H2O-3 server: "
            f"{self.config.h2o_host}:{self.config.h2o_port}"
        )
        h2o.connect(ip=self.config.h2o_host, port=self.config.h2o_port)

        # model as data
        model = mli.wrap(
            "decision_tree",
            data=dataset.to_pandas(),
            data_backend=_mli.MLIDataBackend.PANDAS,
        )

        # define DT Surrogate object
        dt = _decision_tree_h2o.DecisionTreeH2O(
            max_depth=dt_tree_depth,
            nfolds=nfolds,
            categorical_encoding=cat_encoder,
        )

        # fit DT surrogate
        validation_frame = h2o.H2OFrame(
            model.data_as_model,
        )
        # ensure names
        validation_frame.names = list(dataset.names)

        # use train as validation set to get metrics for train (work around)
        try:
            dt.fit(
                model,
                response_column=response_col,
                validation_frame=validation_frame,
                weights_column=weights_column,
                ignored_columns=ignored_columns,
            )
        except Exception as ex:
            raise ex
        finally:
            # TODO everything that's being removed via h2o.remove here should really
            #   be stored somewhere in case `abort()` is invoked.
            #   In such a case 2 things need to happen:
            #   1) The DRF model task should be aborted via cancel call to H2O-3
            #   2) all keys we store for remove should be removed
            #   Not sure where to do it and how, though? Maybe the on_ handler?
            h2o.remove(validation_frame)

        features_meta = self.model_meta.features_metadata.to_dict()

        try:
            # save MOJO
            mojo_name = "dtsurr_mojo.zip"
            if is_multinomial:
                mojo_zip_path = os.path.join(
                    mli_dir_path, f"{e_predict_col}_mojo", mojo_name
                )
            else:
                mojo_zip_path = os.path.join(mli_dir_path, mojo_name)
            dt.save_mojo(path=mojo_zip_path)

            self._save_all_decision_tree_details(
                dt_h2o3=dt,
                mli_dir_path=mli_dir_path,
                is_multinomial=is_multinomial,
                e_predict_col=e_predict_col,
                features_meta=features_meta,
            )

            # save DT frames path as csv
            dt_paths_frame_name = (
                "dtPathsFrame.csv"
                if not is_multinomial
                else "dtPathsFrame_" + e_predict_col + ".csv"
            )
            dt_paths_bin_name = (
                "dtpaths_frame.bin"
                if not is_multinomial
                else "dtpaths_frame_" + e_predict_col + ".bin"
            )

            # TODO this is a conservative fix of
            #  https://github.com/h2oai/h2oai/issues/20591 which can be solved in a
            #  more efficient way once 1.9.1 is out - check:
            #    https://github.com/h2oai/h2oai/issues/20608 (described in detail)
            model_data_as_model = model.data_as_model
            try:
                paths_frame = h2o.H2OFrame(model.data_as_model)
                dt.save_dt_paths_frame(
                    input_df=paths_frame,
                    path=os.path.join(mli_dir_path, dt_paths_frame_name),
                )
                h2o.remove(paths_frame)

                # TODO save_dt_paths_frame() should do this or return the Frame
                def remove_leaf_frames(estimator_id):
                    frames = h2o.frames()["frames"]
                    for frame in frames:
                        frame_name = frame["frame_id"]["name"]
                        if frame_name and all(
                            tag in frame_name
                            for tag in [estimator_id, "leaf_node_assignment"]
                        ):
                            h2o.remove(frame_name)

                remove_leaf_frames(dt.estimator.model_id)

                # save DT frame  as bin file
                datatable.fread(os.path.join(mli_dir_path, dt_paths_frame_name)).to_jay(
                    os.path.join(mli_dir_path, dt_paths_bin_name)
                )
            except Exception as ex:
                # fallback used e.g. in case of text columns with unquoted " characters
                self.logger.warning(
                    f"{self.log_name}: ({dt_paths_frame_name}) CSV file corrupted "
                    f"(error: {ex}) - fail over...",
                )
                # .jay version of the frame just needs to be saved ~ H2O-3 workaround
                model_data_as_model.to_csv(
                    os.path.join(mli_dir_path, dt_paths_frame_name)
                )
                # save DT frame  as bin file
                datatable.fread(os.path.join(mli_dir_path, dt_paths_frame_name)).to_jay(
                    os.path.join(mli_dir_path, dt_paths_bin_name)
                )

            # save DT rules as txt
            tree = h2o_tree.H2OTree(dt.estimator, tree_number=0, tree_class=None)
            h2o_traverser = _tree_traverser_h2o.H2OTreeTraverser(
                root_node=tree.root_node, features=tree.features
            )

            # problems & actions if residual tree
            if debug_model_errors:
                self._diagnose_problems(
                    h2o_traverser.extract_rules_from_tree_as_txt(), label
                )

            # write DT rules to txt file
            self._write_dt_rules(
                suffix="txt",
                mli_dir_path=mli_dir_path,
                predict_col=e_predict_col if is_multinomial else None,
                method_of_extraction=h2o_traverser.extract_rules_from_tree_as_txt,
            )

            # write DT rules to py file
            self._write_dt_rules(
                suffix="py",
                mli_dir_path=mli_dir_path,
                predict_col=e_predict_col if is_multinomial else None,
                method_of_extraction=h2o_traverser.extract_rules_from_tree_as_py_code,
            )
        finally:
            # remove the model
            if nfolds > 0:
                model_id = dt.estimator.model_id
                for fold in range(nfolds):
                    h2o.remove(f"{model_id}_cv_{fold + 1}")

            h2o.remove(dt.estimator)

    def _explain_dt(self, dataset: datatable.Frame, y: datatable.Frame):
        self.logger.info(f"{self.log_name}: BEGIN calculation")

        # colum names resolution
        actual_col = self.model_meta.target_col
        weights_col = self.params.weight_col
        dropped_cols = self.params.drop_cols
        predict_col = ""
        qbin_cols = self.args.get(DecisionTreeSurrogateExplainer.PARAM_QBIN_COLS, [])
        # sanitization map (model features, dataset columns)
        sanitization_map = self.model_meta.sanitization_map
        # categorical features encoder
        cat_encoder = DecisionTreeConstants.CAT_ENCODING_DICT.get(
            self.args.get(DecisionTreeSurrogateExplainer.PARAM_CAT_ENCODING)
        )

        is_multinomial = (
            len(self.model_meta.labels) > 2 if self.model_meta.labels else False
        )
        if is_multinomial:
            # ensure labels are string for multinomial case to avoid issues with DT
            self.labels = list(map(str, self.model_meta.labels))

        self.logger.info(
            f"{self.log_name}: dataset {dataset.shape} loaded",
        )

        if (
            self.config
            and dataset.nrows > self.config.mli_sample_size
            and self.config.mli_sample
        ):
            self.logger.info(
                f"{self.log_name}: sampling down to {self.config.mli_sample_size} "
                f"rows...",
            )

            dataset = sampling.downsample_dataset(
                dataset=dataset,
                sample_size=self.config.mli_sample_size,
                is_classification=len(self.labels) >= 2 if self.labels else False,
                seed=DecisionTreeConstants.SEED,
                target_col=actual_col,
                logger=self.logger,
            )

        # original used features
        original_features = self.model_meta.used_features
        if actual_col in original_features:
            original_features.remove(actual_col)
        # set dropped columns to columns in original features to avoid error in
        # h2o model fit
        if dropped_cols:
            dropped_cols = [x for x in dropped_cols if x in original_features]

        # save target for later use - debug model errors on transformed features
        if actual_col in dataset.names:
            target_dt = dataset[:, actual_col]
        elif y:
            target_dt = y
        else:
            raise errors.MliError(
                f"Target column '{actual_col}' does not exist in the frame with "
                f"columns: {dataset.names}"
            )

        dataset = dataset[:, original_features]

        # predictions
        preds = self.model.predict_datatable(dataset)
        self.logger.debug(f"{self.log_name}: predictions {dataset.shape}")

        # construct predictions frame and bind to input dataset for DT surrogate
        if is_multinomial:
            dataset.cbind(datatable.Frame(preds, names=self.labels))
        else:
            dataset.cbind(
                datatable.Frame(preds, names=[DecisionTreeConstants.COLUMN_DAI_PREDICT])
            )

        # scoring finished > sanitize frame
        dataset = sanitization.sanitize_frame(dataset)
        target_dt = sanitization.sanitize_frame(target_dt)
        actual_col = sanitization_map.to_sanitized(actual_col) if actual_col else ""
        predict_col = sanitization_map.to_sanitized(predict_col) if predict_col else ""
        weights_col = sanitization_map.to_sanitized(weights_col) if weights_col else ""
        dropped_cols = (
            sanitization_map.to_sanitized(dropped_cols) if dropped_cols else []
        )
        qbin_cols = sanitization_map.to_sanitized(qbin_cols) if qbin_cols else []
        qbin_count = self.args.get(DecisionTreeSurrogateExplainer.PARAM_QBIN_COUNT, 0)
        dt_tree_depth = self.args.get(DecisionTreeSurrogateExplainer.PARAM_DT_DEPTH)
        dt_nfolds = self.args.get(DecisionTreeSurrogateExplainer.PARAM_NFOLDS)

        # handle quantile columns ...
        if qbin_count > 0 or len(qbin_cols) > 0 and self.params.use_raw_features:
            retry = 60
            check_for_drf_varimp = True
            var_imp_data = None
            drf_varimp_file_path = os.path.join(
                self.persistence.get_explainer_working_dir(),
                DecisionTreeConstants.FILE_DRF_VAR_IMP,
            )
            numeric_varimp_list = None
            if qbin_count > 0:
                # IF qbin_count > 0,
                # THEN wait for H2O DRF surrogate model to finish so DT surrogate can
                # fetch variable importance and decipher the top variables based on
                # qbin_count
                while check_for_drf_varimp:
                    if os.path.exists(drf_varimp_file_path):
                        with open(drf_varimp_file_path) as json_file:
                            var_imp_data = json.load(json_file)
                        check_for_drf_varimp = False
                    else:
                        retry = retry - 1
                        check_for_drf_varimp = retry >= 0
                        time.sleep(1)
                if var_imp_data:
                    varimp_list = var_imp_data["rowHeaders"]
                    numeric_cols = list(
                        dataset[
                            :,
                            [
                                datatable.ltype.bool,
                                datatable.ltype.int,
                                datatable.ltype.real,
                            ],
                        ].names
                    )
                    numeric_varimp_list = sorted(
                        set(varimp_list) & set(numeric_cols),
                        key=varimp_list.index,
                    )
                else:
                    self.logger.info(
                        f"{self.log_name}: could not fetch DRF variable importance "
                        "for quantile binning in time allotted ...",
                    )

            varimp_list = sanitization_map.to_sanitized(numeric_varimp_list)
            if qbin_cols or varimp_list:
                binned_list, dataset = binning.quantile_bin(
                    frame=sanitization.sanitize_frame(dataset),
                    qbin_cols=qbin_cols,
                    qbin_count=qbin_count,
                    varimp_list=varimp_list,
                    logger=self.logger,
                )
                if len(binned_list) > 0:
                    self.logger.info(
                        f"{self.log_name}: successfully binned {len(binned_list)} "
                        f"variables",
                    )
                    self.logger.data(
                        f"{self.log_name}: successfully binned variables: "
                        f"{binned_list}",
                    )

        # explanation for the TRANSFORMED features
        if not self.params.use_raw_features:
            if self.model.has_transformed_model:
                self.logger.data(dataset.head())
                self.logger.debug(
                    f"{self.log_name}: pre transform {dataset.shape}",
                )

                dataset = self.model.transformed_model.transform_dataset(
                    X=dataset,
                    input_features_frame=dataset,
                    target_frame=target_dt[:, actual_col].to_pandas()[actual_col],
                    seed=DecisionTreeConstants.SEED,
                    cols_to_keep=[weights_col] if weights_col else None,
                    logger=self.logger,
                )

                self.logger.debug(
                    f"{self.log_name}: post transform {dataset.shape} vs. "
                    f"predictions {preds.shape if preds else 'None'}",
                )

                if not is_multinomial:
                    # add back model preds as fit_transform only keeps input features
                    dataset.cbind(
                        datatable.Frame(
                            preds,
                            names=[DecisionTreeConstants.COLUMN_DAI_PREDICT],
                        )
                    )
                else:
                    dataset.cbind(datatable.Frame(preds, names=self.model_meta.labels))

            else:
                raise ValueError(
                    f"{self.log_name} cannot use transformed features since there is "
                    f"no unfitted pipeline for model {self.model}"
                )

        has_text_transformers = self.model.meta.has_text_transformers
        do_debug_model_residuals: bool = self.args.get(
            DecisionTreeSurrogateExplainer.PARAM_DEBUG_RESIDUALS, False
        )
        debug_model_residuals_class: str = str(
            self.args.get(
                DecisionTreeSurrogateExplainer.PARAM_DEBUG_RESIDUALS_CLASS, ""
            )
        )
        if is_multinomial:
            working_dir = self.persistence.get_explainer_working_dir()
            if do_debug_model_residuals:
                self.logger.info(
                    f"{self.log_name}: calculating multinomial ...",
                )
                dataset, _ = self._explain_dt_residual(
                    num_labels=self.model_meta.num_labels,
                    labels=self.model.meta.labels,
                    debug_model_errors_class=debug_model_residuals_class,
                    sanitized_df=datatable.cbind(dataset, target_dt),
                    target=actual_col,
                    pred_col=DecisionTreeConstants.COLUMN_DAI_PREDICT,
                    preds_mult_prob=dataset[:, self.labels],
                )

                if not has_text_transformers:
                    del dataset[:, actual_col]

                dataset = h2o_utils.preprocess_h2o3_data(
                    frame_for_h2o3=dataset,
                    contains_text_transformers=has_text_transformers,
                    explainer_work_path=working_dir,
                    config=self.config,
                    sanitization_utils=sanitization,
                    num_labels=self.model_meta.num_labels,
                    features_metadata=self.model.meta.features_metadata.to_dict(),
                    meta_keys=method.FeaturesMetadata,
                    persistence=self.persistence.store,
                    logger=self.logger,
                    target_col=actual_col,
                    dropped_cols=dropped_cols,
                )
                if actual_col in dataset.names:
                    del dataset[:, actual_col]

                if self.config and dataset.nrows <= self.config.mli_sample_size:
                    # IF nrows > mli_sample_size,
                    # THEN downsample_rows_dt_surrogate() already sanitized columns.
                    # ... however ...
                    # IF nrows <= mli_sample_size
                    # THEN then sanitize
                    # before DT surrogate training.
                    dataset = sanitization.sanitize_frame(dataset)

                self._run_dt_surrogate(
                    dataset=dataset,
                    dt_tree_depth=dt_tree_depth,
                    nfolds=dt_nfolds,
                    mli_dir_path=working_dir,
                    predict_col=predict_col,
                    weights_column=weights_col if weights_col else None,
                    ignored_columns=(
                        dropped_cols + self.labels if dropped_cols else self.labels
                    ),
                    is_multinomial=True,
                    label=debug_model_residuals_class,
                    debug_model_errors=do_debug_model_residuals,
                    cat_encoder=cat_encoder,
                )
            else:
                for label in self.labels:
                    dataset = h2o_utils.preprocess_h2o3_data(
                        frame_for_h2o3=(
                            datatable.cbind(dataset, target_dt)
                            if has_text_transformers
                            else dataset
                        ),
                        contains_text_transformers=has_text_transformers,
                        explainer_work_path=working_dir,
                        config=self.config,
                        sanitization_utils=sanitization,
                        num_labels=self.model_meta.num_labels,
                        features_metadata=self.model.meta.features_metadata.to_dict(),
                        meta_keys=method.FeaturesMetadata,
                        persistence=self.persistence.store,
                        logger=self.logger,
                        target_col=actual_col,
                        dropped_cols=dropped_cols,
                    )
                    if actual_col in dataset.names:
                        del dataset[:, actual_col]

                    if self.config and dataset.nrows <= self.config.mli_sample_size:
                        # IF nrows > mli_sample_size
                        # THEN downsample_rows_dt_surrogate() already sanitized columns.
                        # ... however...
                        # IF nrows <= mli_sample_size
                        # THEN sanitize before DT surrogate training
                        dataset = sanitization.sanitize_frame(dataset)

                    self._run_dt_surrogate(
                        dataset=dataset,
                        dt_tree_depth=dt_tree_depth,
                        nfolds=dt_nfolds,
                        mli_dir_path=working_dir,
                        predict_col=label,
                        weights_column=weights_col if weights_col else None,
                        ignored_columns=(
                            dropped_cols + self.labels if dropped_cols else self.labels
                        ),
                        is_multinomial=True,
                        label=label,
                        debug_model_errors=do_debug_model_residuals,
                        cat_encoder=cat_encoder,
                    )
        else:
            if do_debug_model_residuals:
                self.logger.info(
                    f"{self.log_name}: calculating binomial/regression ...",
                )
                dataset, _ = self._explain_dt_residual(
                    num_labels=self.model_meta.num_labels,
                    labels=self.model.meta.labels,
                    debug_model_errors_class=debug_model_residuals_class,
                    sanitized_df=datatable.cbind(dataset, target_dt),
                    target=actual_col,
                    pred_col=DecisionTreeConstants.COLUMN_DAI_PREDICT,
                )

                if not has_text_transformers:
                    del dataset[:, actual_col]

            elif has_text_transformers:
                dataset = datatable.cbind(dataset, target_dt)

            dataset = h2o_utils.preprocess_h2o3_data(
                frame_for_h2o3=dataset,
                contains_text_transformers=has_text_transformers,
                explainer_work_path=self.persistence.get_explainer_working_dir(),
                config=self.config,
                sanitization_utils=sanitization,
                num_labels=self.model_meta.num_labels,
                features_metadata=self.model.meta.features_metadata.to_dict(),
                meta_keys=method.FeaturesMetadata,
                persistence=self.persistence.store,
                logger=self.logger,
                target_col=actual_col,
                dropped_cols=dropped_cols,
            )
            if actual_col in dataset.names:
                del dataset[:, actual_col]

            if self.config and dataset.nrows <= self.config.mli_sample_size:
                # IF nrows > mli_sample_size,
                # THEN downsample_rows_dt_surrogate() already sanitized columns
                # ... however...
                # IF nrows <= mli_sample_size
                # THEN sanitize before DT surrogate training.
                dataset = sanitization.sanitize_frame(dataset)

            self._run_dt_surrogate(
                dataset=dataset,
                dt_tree_depth=dt_tree_depth,
                nfolds=dt_nfolds,
                mli_dir_path=self.persistence.get_explainer_working_dir(),
                predict_col=predict_col,
                weights_column=weights_col if weights_col else None,
                ignored_columns=dropped_cols if dropped_cols else None,
                is_multinomial=False,
                label="",
                debug_model_errors=do_debug_model_residuals,
                cat_encoder=cat_encoder,
            )

        # DT rules ZIP archive
        zip_archive_path = persistences.Persistence.make_key(
            self.persistence.get_explainer_working_dir(),
            f"{DecisionTreeConstants.DIR_DT_SURROGATE}.zip",
        )
        self.persistence.store.make_dir_zip_archive(
            src_key=persistences.Persistence.make_key(
                self.persistence.get_explainer_working_dir(),
                DecisionTreeConstants.DIR_DT_SURROGATE,
            ),
            zip_key=zip_archive_path,
        )
        self.logger.debug(f"{self.log_name}: dt_rules_zip_path={zip_archive_path}")

        # purge: we do not need original directory since we have a zip dir now
        self.persistence.store.delete_tree(
            persistences.Persistence.make_key(
                self.persistence.get_explainer_working_dir(),
                DecisionTreeConstants.DIR_DT_SURROGATE,
            )
        )

        self.logger.info(f"{self.log_name}: DONE calculation")

        return self.key

    def _write_dt_rules(
        self,
        suffix="txt",
        mli_dir_path=None,
        predict_col=None,
        method_of_extraction=None,
    ):
        dt_rules_dir = self.persistence.store.make_key(
            mli_dir_path, DecisionTreeConstants.DIR_DT_SURROGATE
        )
        self.persistence.store.make_dir(
            key=dt_rules_dir,
        )
        dt_rules_file_name = (
            f"dt_surrogate_rules.{suffix}"
            if not predict_col
            else f"dt_surrogate_rules_{predict_col}.{suffix}"
        )
        self.persistence.store.save(
            key=self.persistence.store.make_key(dt_rules_dir, dt_rules_file_name),
            data=method_of_extraction(rules.CodeStyle.DICT_ROW).__str__(),
            data_type=persistences.PersistenceDataType.text,
        )

    def _save_decision_tree_meta(
        self, mli_dir_path: str, filename: str, save_method: Callable
    ) -> dict:
        save_method(path=mli_dir_path, filename=filename)
        return self.persistence.store.load_json(
            persistences.Persistence.make_key(mli_dir_path, filename)
        )

    def _patch_decision_tree_node(
        self,
        tree: dict | None,
        types: dict,
        features_metadata: dict | None,
        parent: dict | None = None,
    ) -> dict | None:
        if tree is None:
            return None

        def extract_feature_name(name: str) -> str | None:
            """Extract feature name (like ``"PAY_0"``) from a split condition
            (like ``PAY_0 <= 1.500000`` while supporting operators: <=, <, >=, >, ==,
            !=, in, not in). Return ``None`` for leaf nodes.

            """
            if name is None:
                return None

            # match feature name followed by space and operator
            pattern = r"^(.+?)\s+(?:<=|<|>=|>|==|!=|in\b|not\s+in\b)"
            match = re.match(pattern, name)

            if match:
                return match.group(1).strip()

            # leaf
            return None

        try:
            current_feature = extract_feature_name(tree.get("name"))
            current_type = types[current_feature] if current_feature else None
        except KeyError as ke:
            raise errors.MliError(
                f"DT patch is unable to resolve CURRENT type: "
                f"{ke} while types are: {types}"
            )

        try:
            parent_feature = (
                extract_feature_name(parent.get("name")) if parent else None
            )
            parent_type = types[parent_feature] if parent_feature else None
        except KeyError as ke:
            raise errors.MliError(
                f"DT patch is unable to resolve PARENT type: "
                f"{ke} while types are: {types}"
            )

        if current_type == "Time":
            value = datetime.datetime.fromtimestamp(float(tree["value"]) / 1000)
            tree["raw_value"] = tree["value"]

            default_fmt = "%d %b %Y %H:%M:%S %Z"

            if (
                features_metadata is not None
                and current_feature
                and current_feature
                in features_metadata[method.Method.KEY_DATE_FEATURES]
            ):
                fmt = features_metadata[method.Method.KEY_DATE_FEATURES_FORMAT][
                    features_metadata[method.Method.KEY_DATE_FEATURES].index(
                        current_feature
                    )
                ]

                if fmt is None:
                    fmt = default_fmt
            else:
                fmt = default_fmt

            tree["value"] = value.strftime(fmt)

        if parent_type == "Time":
            tree["edgein"] = tree["edgein"].replace(
                parent["raw_value"], f'"{parent["value"]}"'
            )

        # recursively patch children (if they exist - leaf nodes have None children)
        if tree.get("children") is not None:
            tree["children"] = list(
                map(
                    partial(
                        self._patch_decision_tree_node,
                        types=types,
                        features_metadata=features_metadata,
                        parent=tree,
                    ),
                    tree["children"],
                )
            )

        return tree

    def _patch_decision_tree_value_types(
        self,
        details: dict,
        tree_json: dict,
        filepath: str,
        features_metadata: dict | None,
    ):
        patched_tree = self._patch_decision_tree_node(
            tree=tree_json,
            types=dict(zip(details["_names"], details["_column_types"], strict=False)),
            features_metadata=features_metadata,
        )

        self.persistence.save_json(path=filepath, data=patched_tree)

    @staticmethod
    def _sanitize_debug_model_error_class(
        debug_model_errors_class: str | int,
        frame: datatable.Frame,
        target: str,
        labels: list[str | int] | None,
    ):
        """Sanitize debug model error class.

        Parameters
        ----------
        debug_model_errors_class : str
            Target label as debug class.
        frame : dt.Frame
            Reference frame to sanitize from.
        target : str
            Target column name in frame.
        labels : list[str | int] | None
            Available labels of target.

        Returns
        -------
        str | int :
            Sanitized debug_model_errors_class.

        """
        # handle DT boolean handling for 0/1 values ...
        if isinstance(labels, list) and labels:
            if debug_model_errors_class == "True" and labels[1] == 1:
                debug_model_errors_class = 1
            elif debug_model_errors_class == "False" and labels[0] == 0:
                debug_model_errors_class = 0

        if (
            target
            and frame
            and target not in frame.names
            and DecisionTreeConstants.COLUMN_MODEL_PRED in frame.names
        ):
            target = DecisionTreeConstants.COLUMN_MODEL_PRED

        if (
            target
            and frame
            and debug_model_errors_class
            and isinstance(debug_model_errors_class, str)
        ):
            if target not in frame.names:
                raise ValueError(
                    f"Sanitization of debug model failed: "
                    f"'{target}' target column not in {frame.names}"
                )

            try:
                if datatable.ltype.int == frame.ltypes[frame.colindex(target)]:
                    return int(debug_model_errors_class)
                if frame.ltypes[frame.colindex(target)] in [
                    datatable.ltype.real,
                    datatable.ltype.void,
                ]:
                    return float(debug_model_errors_class)
            except ValueError:
                return debug_model_errors_class

        return debug_model_errors_class

    def get_result(self) -> results.DtResult:
        return results.DtResult(
            persistence=self.persistence,
            h2o_sonar_config=self.config,
            explainer_name="surrogate decision tree explainer",
            explainer_id=DecisionTreeSurrogateExplainer.explainer_id(),
            highlight_highest_residual=self.args.get(
                DecisionTreeSurrogateExplainer.PARAM_DEBUG_RESIDUALS, False
            ),
            logger=self.logger,
        )
