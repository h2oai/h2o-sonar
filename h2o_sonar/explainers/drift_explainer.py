# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import traceback

import datatable

from h2o_sonar import errors
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api import problems
from h2o_sonar.lib.api import results
from h2o_sonar.lib.integrations import mv_adapter


try:
    from h2o_mv.core import mv_dataset
    from h2o_mv.recipes import drift as mv_drift

    HAS_H2O_MV = True
except ImportError:
    HAS_H2O_MV = False


class DriftDetectionExplainer(
    explainers.Explainer, mv_adapter.ExplainerToMvTestAdapter
):
    """Drift detection explainer.

    @see https://docs.h2o.ai/wave-apps/h2o-model-validation/guide/tests
         /supported-validation-tests/drift-detection/drift-detection

    """

    _display_name = "Drift Detection"
    _description = (
        "Drift detection refers to a validation test that enables you to identify"
        "changes in the distribution of variables in your model's input data, "
        "preventing model performance degradation. "
        "The explainer performs drift detection using the train and another "
        "dataset captured at different times to assess how data has changed over time. "
        "The Population Stability Index (PSI) formula is applied to each variable "
        "to measure how much the variable has shifted in distribution over time. "
        "PSI is applied to numerical and categorical columns and not date columns."
    )
    _iid: bool = True
    _time_series: bool = True

    _regression: bool = True
    _binary: bool = True
    _multiclass: bool = True

    _supported_dataset_locators = [
        commons.ResourceLocatorType.local,
        commons.ResourceLocatorType.handle,
    ]
    # let Drift run if remote model is specified - container would make it incompatible
    _supported_model_locators = [
        commons.ResourceLocatorType.local,
        commons.ResourceLocatorType.handle,
    ]
    _global_explanation = True
    _explanation_types = [
        e10s.WorkDirArchiveExplanation,
        e10s.GlobalFeatImpExplanation,
    ]

    PARAM_WORKER = "worker_connection_key"
    PARAM_DROP_COLS = "drop_cols"
    PARAM_DRIFT_THRESHOLD = "drift_threshold"

    DEFAULT_DRIFT_THRESHOLD = 0.1
    DEFAULT_DROP_COLS = []

    _parameters = [
        explainers.ExplainerParam(
            param_name=PARAM_WORKER,
            description=(
                "Optional connection ID of the Driverless AI configured in the H2O "
                "Sonar configuration. Only Driverless AI servers with "
                "username and password authentication are supported."
            ),
            param_type=commons.ExplainerParamType.str,
            default_value=None,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_DROP_COLS,
            description=(
                "Defines the columns to drop during the validation test. Typically "
                "drop columns refer to columns that can indicate a drift without "
                "an impact on the model, like columns not used by the model, "
                "record IDs, time columns, etc."
            ),
            param_type=commons.ExplainerParamType.list,
            default_value=DEFAULT_DROP_COLS,
            tags=[explainers.ExplainerParam.TAG_SRC_DATASET_COLUMN_NAMES],
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_DRIFT_THRESHOLD,
            description="Drift threshold.",
            param_type=commons.ExplainerParamType.float,
            default_value=DEFAULT_DRIFT_THRESHOLD,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
    ]
    _keywords = [
        explainers.Explainer.KEYWORD_COMPLIANCE_TEST,
        explainers.Explainer.KEYWORD_EXPLAINS_FEATURE_BEHAVIOR,
        explainers.Explainer.KEYWORD_H2O_MODEL_VALIDATION,
    ]
    _modules_needed_by_name = [
        mv_adapter.ExplainerToMvTestAdapter.MV_PYTHON_MODULE_NAME
    ]

    def __init__(self):
        explainers.Explainer.__init__(self)
        mv_adapter.ExplainerToMvTestAdapter.__init__(self)

        self.args = None
        self.problems = []
        self.log_name = "Drift detection"

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        model: models.ExplainableModel | None = None,
        **explainer_params,
    ) -> bool:
        if not HAS_H2O_MV:
            self.logger.warning(
                self._check_compatibility_pckg_err_msg(mv_adapter.PACKAGE_NAME)
            )

        explainers.Explainer.check_compatibility(self, params, **explainer_params)
        mv_adapter.ExplainerToMvTestAdapter.check_mv_compatibility(self, explainer=self)

        mv_test = mv_drift.Drift(name=DriftDetectionExplainer._display_name)
        mv_test.check_compatibility()

        # check that train and another is specified
        if not params.testset:
            msg = (
                f"{self.log_name} requires training dataset and also another "
                f"dataset (specified in ``testset`` parameter) to assess drift."
            )
            self.logger.warning(msg)
            return False

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
        mv_adapter.ExplainerToMvTestAdapter.setup(
            self,
            h2o_sonar_config=self.config,
            persistence=persistence,
            logger=self.logger,
        )

        self._resolve_explainer_params()

        self.log_name = f"Drift detection {self.mli_key}/{self.key}"

    def explain(
        self,
        X: datatable.Frame,
        y: datatable.Frame | None = None,
        explanations_types: list = None,
        **e_params,
    ):
        #
        # calculation
        #
        if commons.ResourceHandle.is_handle(X):
            # REMOTE dataset(s)
            try:
                # Driverless AI connection
                worker_connection_key = self.args.get(
                    DriftDetectionExplainer.PARAM_WORKER, ""
                )
                if not worker_connection_key:
                    raise errors.MliError(
                        f"{self.log_name}: required worker connection name not "
                        f"specified in the explainer parameters"
                    )
                worker_connection = self._get_mv_dai_worker_by_key(
                    connection_key=worker_connection_key,
                    log_name=self.log_name,
                )
                self.mv_client.set_worker(worker_connection)
                self.mv_client.add_connection(
                    worker_connection
                )  # returns platform w/ mvid

                primary_dataset: mv_dataset.MVDataset = mv_dataset.MVDataset(
                    name="Training dataset",
                    platform_mvid=worker_connection.mvid,
                    platform_obj_key=X.resource_key,
                )
                primary_dataset = worker_connection.import_dataset(
                    primary_dataset.platform_obj_key
                )

                testset = e_params.get("testset", None)
                if not testset:
                    ValueError(
                        f"{self.log_name}: testset is required to specify SECONDARY "
                        f"dataset"
                    )
                self.logger.debug(f"{self.log_name}: testset: {testset}")
                secondary_dataset: mv_dataset.MVDataset = mv_dataset.MVDataset(
                    name="Test dataset",
                    platform_mvid=worker_connection.mvid,
                    platform_obj_key=testset.resource_key,
                )
                secondary_dataset = worker_connection.import_dataset(
                    secondary_dataset.platform_obj_key
                )

                mv_test = mv_drift.Drift(name=DriftDetectionExplainer._display_name)
                mv_test.settings = mv_drift.DriftSettings(
                    primary_dataset=primary_dataset,
                    secondary_dataset=secondary_dataset,
                    drop_columns=self.args.get(
                        DriftDetectionExplainer.PARAM_DROP_COLS, []
                    ),
                    drift_threshold=self.args.get(
                        DriftDetectionExplainer.PARAM_DRIFT_THRESHOLD,
                        DriftDetectionExplainer.DEFAULT_DRIFT_THRESHOLD,
                    ),
                )
            except Exception as e:
                self.logger.error(f"{self.log_name}: {e}\n{traceback.format_exc()}")
                raise e
        else:
            # LOCAL dataset(s)
            train_path = self.dataset_meta.file_path
            train_ds = self.mv_client.import_local_dataset(train_path)
            test_path = self.params.testset
            test_ds = self.mv_client.import_local_dataset(test_path)

            mv_test = mv_drift.Drift(name=DriftDetectionExplainer._display_name)
            mv_test.settings = mv_drift.DriftSettings(
                primary_dataset=train_ds,
                secondary_dataset=test_ds,
                drop_columns=self.args.get(DriftDetectionExplainer.PARAM_DROP_COLS, []),
                drift_threshold=self.args.get(
                    DriftDetectionExplainer.PARAM_DRIFT_THRESHOLD,
                    DriftDetectionExplainer.DEFAULT_DRIFT_THRESHOLD,
                ),
            )

        try:
            mv_test.run()
        finally:
            if mv_test:
                self._dump_mv_test_log_to_explainer_log(mv_test.log)

        # assert MV test integrity
        mv_adapter.ExplainerToMvTestAdapter.assert_mv_test_status(self, mv_test)
        self._assert_mv_test_result(mv_test)

        #
        # explanations
        #
        explanations = []

        # EXPLANATION: drift as feature importance
        fi_explanation = self._normalize_drift_to_gom(mv_test.results.psi_scores)
        self._identify_problems(mv_test.results.psi_scores)
        explanations.append(fi_explanation)

        # EXPLANATION: MV result ZIP archive
        mv_explanation = e10s.ModelValidationResultExplanation(
            explainer=self,
            display_name=self.display_name,
            display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
        )
        mv_zip_format = f5s.ModelValidationResultArchiveFormat(
            explanation=mv_explanation,
            mv_test_type=mv_test,
            mv_test_name=mv_test.name,
            mv_test_id=mv_test.mvid,
            mv_test_results=mv_test.results,
            mv_test_settings=mv_test.settings,
            mv_test_artifacts={},
            mv_test_log=mv_test.log,
            persistence=self.persistence.store,
            logger=self.logger,
        )
        mv_explanation.add_format(mv_zip_format)
        explanations.append(mv_explanation)

        # EXPLANATION: work/ directory ZIP archive
        archive_explanation = self.create_explanation_workdir_archive(
            display_name=self.display_name,
            display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
        )
        explanations.append(archive_explanation)
        # HTML representation
        if self.config and self.config.create_html_representations:
            try:
                display_name = DriftDetectionExplainer._display_name
                explanations.append(
                    e10s.GlobalHtmlFragmentExplanation.from_explanation(
                        explainer=self,
                        explanation=fi_explanation,
                        display_name=display_name,
                        display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
                        data_as_text=False,
                        logger=self.logger,
                    )
                )
            except Exception as ex:
                self.logger.warning(
                    f"{self.log_name}: HTML representation creation failed: {ex}"
                )

        return explanations

    def _assert_mv_test_result(self, mv_test):
        if not mv_test.results.psi_scores:
            err_msg = (
                f"{self.log_name} did not return any PSI scores - the result is empty"
            )
            self.logger.error(err_msg)
            raise errors.MliError(err_msg)

    def _normalize_drift_to_gom(
        self, psi_scores_dict: dict
    ) -> e10s.GlobalFeatImpExplanation:
        """Normalize PSI to the Grammar of MLI (GoM) format to get free charts."""

        explanation = e10s.GlobalFeatImpExplanation(
            explainer=self,
            display_name=self.display_name,
            display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
        )

        jdf = f5s.GlobalFeatImpJSonDatatableFormat
        explanation_frame = datatable.Frame(
            {
                jdf.COL_NAME: list(psi_scores_dict.keys()),
                jdf.COL_IMPORTANCE: list(psi_scores_dict.values()),
                jdf.COL_GLOBAL_SCOPE: [True] * len(psi_scores_dict),
            }
        ).sort(-datatable.f[jdf.COL_IMPORTANCE])
        # index file (of per-class data files) (JSon)
        (
            idx_dict,
            idx_str,
        ) = f5s.GlobalFeatImpJSonDatatableFormat.serialize_index_file(
            classes=["global"],
            doc=DriftDetectionExplainer._description,
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
        explanation.add_format(
            explanation_format=f5s.GlobalFeatImpJSonFormat.from_json_datatable(
                json_dt_format
            )
        )

        return explanation

    def _identify_problems(self, psi_scores_dict: dict):
        try:
            threshold = float(
                self.args.get(
                    DriftDetectionExplainer.PARAM_DRIFT_THRESHOLD,
                    DriftDetectionExplainer.DEFAULT_DRIFT_THRESHOLD,
                )
            )

            # dict: feature name -> PSI score
            if psi_scores_dict and threshold:
                for feature in psi_scores_dict.keys():
                    if float(psi_scores_dict.get(feature, threshold)) > threshold:
                        problem = problems.ProblemAndAction(
                            description=(
                                f"Detected drift for feature '{feature}': PSI score is "
                                f"{psi_scores_dict[feature]} while threshold "
                                f"(explainer parameter) is {threshold}"
                            ),
                            severity=problems.ProblemSeverity.high,
                            problem_type="data",
                            problem_attrs={
                                problems.ProblemAndAction.ATTR_EVALUATOR_NAME: (
                                    DriftDetectionExplainer._display_name
                                ),
                            },
                            actions_description=(
                                "Consider model retraining, update or weighting new "
                                "data."
                            ),
                            explainer_id=DriftDetectionExplainer.explainer_id(),
                            explainer_name=DriftDetectionExplainer._display_name,
                            explanation_type=(
                                e10s.GlobalFeatImpExplanation.explanation_type()
                            ),
                            explanation_name=e10s.GlobalFeatImpExplanation.__name__,
                            resources=[],
                        )
                        self.add_problem(problem)
        except Exception as ex:
            self.logger.warning(
                f"{self.log_name} failed while identifying problems: {ex}"
                f"\n{traceback.format_exc()}"
            )

    def get_result(
        self,
    ) -> results.FeatureImportanceResult:
        return results.FeatureImportanceResult(
            chart_title="Drift: Population Stability Indices",
            chart_x_axis="Feature",
            chart_y_axis="PSI scores",
            persistence=self.persistence,
            explainer_id=DriftDetectionExplainer.explainer_id(),
            h2o_sonar_config=self.config,
            logger=self.logger,
        )
