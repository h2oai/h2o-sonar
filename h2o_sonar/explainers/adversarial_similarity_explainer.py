# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import traceback

import airium
import datatable
from matplotlib import pyplot as plt

from h2o_sonar import errors
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api import results
from h2o_sonar.lib.api.commons import MimeType
from h2o_sonar.lib.integrations import mv_adapter


try:
    from h2o_mv.core import mv_dataset
    from h2o_mv.recipes import adversarial
    from h2o_mv.recipes.adversarial import Adversarial

    HAS_H2O_MV = True
except ImportError:
    HAS_H2O_MV = False


class AdversarialSimilarityExplainer(
    explainers.Explainer, mv_adapter.ExplainerToMvTestAdapter
):
    """Adversarial similarity explainer.

    @see https://docs.h2o.ai/wave-apps/h2o-model-validation/guide/tests/
         supported-validation-tests/adversarial-similarity/adversarial-similarity

    """

    _display_name = "Adversarial Similarity"
    _description = (
        "Adversarial similarity refers to a validation test that assists in observing "
        "similar or dissimilar segments of two different datasets. Observing the "
        "feature distribution of two different datasets can indicate similarity "
        "or dissimilarity. An adversarial similarity test can be performed rather "
        "than going over all the features individually to observe the differences. "
        "During an adversarial similarity test, decision tree algorithms are "
        "leveraged to find similar or dissimilar rows between the train dataset "
        "and any dataset with the same train columns."
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
    _global_explanation = True
    _explanation_types = [
        e10s.WorkDirArchiveExplanation,
        e10s.GlobalGroupedBarChartExplanation,
    ]

    PARAM_WORKER = "worker_connection_key"
    PARAM_DROP_COLS = "drop_cols"
    PARAM_SHAPLEY_VALUES = "shapley_values"

    DEFAULT_SHAPLEY_VALUES = False
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
            param_name=PARAM_SHAPLEY_VALUES,
            description=(
                "Determines whether to compute Shapley values "
                "for the model used to analyze the similarity between the "
                "Primary and Secondary Dataset. Test uses the generated Shapley "
                "values to create an array of visual metrics that provide valuable "
                "insights into the contribution of individual features to the overall "
                "model performance."
            ),
            param_type=commons.ExplainerParamType.bool,
            default_value=DEFAULT_SHAPLEY_VALUES,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_DROP_COLS,
            description="Defines the columns to drop during model training.",
            param_type=commons.ExplainerParamType.list,
            default_value=DEFAULT_DROP_COLS,
            tags=[explainers.ExplainerParam.TAG_SRC_DATASET_COLUMN_NAMES],
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
    ]
    _keywords = [
        explainers.Explainer.KEYWORD_COMPLIANCE_TEST,
        explainers.Explainer.KEYWORD_EXPLAINS_DATASET,
        explainers.Explainer.KEYWORD_H2O_MODEL_VALIDATION,
    ]
    _modules_needed_by_name = [
        mv_adapter.ExplainerToMvTestAdapter.MV_PYTHON_MODULE_NAME
    ]

    PLOT_TITLE = "Similar-to-Secondary Probabilities Histogram"
    CLASS_ONE_AND_ONLY = "global"

    def __init__(self):
        explainers.Explainer.__init__(self)
        mv_adapter.ExplainerToMvTestAdapter.__init__(self)

        self.args = None
        self.log_name = "Adversarial similarity"

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
            return False

        explainers.Explainer.check_compatibility(self, params, **explainer_params)
        mv_adapter.ExplainerToMvTestAdapter.check_mv_compatibility(self, explainer=self)

        self._resolve_explainer_params()

        # Driverless AI network connectivity
        try:
            self._probe_dai_connectivity(
                dai_connection_key=self.args.get(
                    AdversarialSimilarityExplainer.PARAM_WORKER, ""
                ),
                log_name=self.log_name,
                logger=self.logger,
            )
        except Exception as e:
            self.logger.warning(
                f"{self.log_name}: compatibility error {e}\n{traceback.format_exc()}"
            )
            return False

        mv_test = Adversarial(name=AdversarialSimilarityExplainer._display_name)
        mv_test.check_compatibility()

        # check that train and another is specified
        if not params.testset:
            msg = (
                f"{self.log_name} requires training dataset and also another "
                f"dataset (specified in ``testset`` parameter)."
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

        self.log_name = f"Adversarial similarity {self.mli_key}/{self.key}"

        if not self.args.get(AdversarialSimilarityExplainer.PARAM_WORKER, ""):
            raise errors.MliError(
                f"{self.log_name}: worker connection name is not specified in "
                f"the explainer parameter "
                f"{AdversarialSimilarityExplainer.PARAM_WORKER} of the explainer "
                f"{AdversarialSimilarityExplainer.__name__}."
            )

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

        # Driverless AI connection
        # get connection by name from the H2O Sonar configuration
        worker_connection_key = self.args.get(
            AdversarialSimilarityExplainer.PARAM_WORKER, ""
        )
        if not worker_connection_key:
            raise errors.MliError(
                f"{self.log_name}: required worker connection name not specified "
                f"in the explainer parameters"
            )
        worker_connection = self._get_mv_dai_worker_by_key(
            connection_key=worker_connection_key, log_name=self.log_name
        )
        if not worker_connection:
            raise errors.MliError(
                f"{self.log_name}: worker connection '{worker_connection_key}' not "
                f"found in the H2O Sonar configuration. "
            )
        self.mv_client.set_worker(worker_connection)

        if commons.ResourceHandle.is_handle(X):
            # REMOTE dataset(s)
            try:
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
            except Exception as e:
                self.logger.error(f"{self.log_name}: {e}\n{traceback.format_exc()}")
                raise e
        else:
            # LOCAL dataset(s)
            train_path = self.dataset_meta.file_path
            primary_dataset = self.mv_client.import_local_dataset(train_path)
            test_path = self.params.testset
            secondary_dataset = self.mv_client.import_local_dataset(test_path)

        mv_test = adversarial.Adversarial(
            name=AdversarialSimilarityExplainer._display_name
        )
        mv_test.settings = adversarial.AdversarialSettings(
            primary_dataset=primary_dataset,
            secondary_dataset=secondary_dataset,
            shapley_values=self.args.get(
                AdversarialSimilarityExplainer.PARAM_SHAPLEY_VALUES,
                AdversarialSimilarityExplainer.DEFAULT_SHAPLEY_VALUES,
            ),
            drop_columns=self.args.get(
                AdversarialSimilarityExplainer.PARAM_DROP_COLS, []
            ),
        )
        try:
            mv_test.run()  # type(mv_test.results) == AdversarialResults
        finally:
            if mv_test:
                self._dump_mv_test_log_to_explainer_log(mv_test.log)

        # assert MV test integrity
        mv_adapter.ExplainerToMvTestAdapter.assert_mv_test_status(self, mv_test)

        #
        # explanations
        #
        explanations = []

        # EXPLANATION: bar chart
        bar_chart_explanation = self._normalize_mv_test_results_to_gom(mv_test.results)
        explanations.append(bar_chart_explanation)

        # EXPLANATION: MV result ZIP archive
        mv_explanation = e10s.ModelValidationResultExplanation(
            explainer=self,
            display_name=self.display_name,
            display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
        )
        mv_zip_format = formats.ModelValidationResultArchiveFormat(
            explanation=mv_explanation,
            mv_test_type=mv_test,
            mv_test_name=mv_test.name,
            mv_test_id=mv_test.mvid,
            mv_test_results=mv_test.results,
            mv_test_settings=mv_test.settings,
            mv_test_log=mv_test.log,
            mv_test_artifacts=mv_test.artifacts.items() if mv_test.artifacts else {},
            mv_client=self.mv_client,
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

        # OPTIONAL explanation: HTML fragment
        if self.config and self.config.create_html_representations:
            dissimilarity_score_auc = mv_test.results.val_score
            try:
                html_explanation = e10s.GlobalHtmlFragmentExplanation(
                    explainer=self,
                    display_name=AdversarialSimilarityExplainer._display_name,
                    display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
                )

                html_format = formats.HtmlFormat(
                    explanation=html_explanation,
                    format_data=formats.HtmlFormat.MINIMAL_HTML,
                    persistence=self.persistence.store,
                )

                html_src = airium.Airium()
                with html_src.b():
                    html_src(f"Dissimilarity Score AUC: {dissimilarity_score_auc}")
                with html_src.div():
                    img_format_path = self.persistence.get_explanation_file_path(
                        explanation_type=html_explanation.explanation_type(),
                        explanation_format=html_format.mime,
                        explanation_file="fi-class-0.png",
                    )
                    self.get_result().plot(file_path=img_format_path)
                    html_src.img(
                        src=self.persistence.get_relative_path(img_format_path),
                        alt=AdversarialSimilarityExplainer.PLOT_TITLE,
                        # ensure that image will not overflow enclosing <div/>
                        style=(
                            "height: 100%; max-width: 100%; display: block; "
                            "margin: auto;"
                        ),
                    )
                    html_src.br()

                html_format.update_data(
                    str(html_src),
                    f"{persistences.ExplainerPersistence.FILE_EXPLANATION}.html",
                )

                explanations.append(html_explanation)
            except Exception as ex:
                self.logger.warning(
                    f"{self.log_name}: HTML representation creation failed: {ex}\n"
                    f"{traceback.format_exc()}"
                )

        return explanations

    def _normalize_mv_test_results_to_gom(
        self, adversarial_results
    ) -> e10s.GlobalGroupedBarChartExplanation:
        """Normalize Adversarial Similarity MV test result to the Grammar of MLI (GoM)
        format.

        """
        # ^ type == h2o_mv.recipes.adversarial.results.AdversarialResults

        primary_oof_summary = adversarial_results.primary_oof_summary
        secondary_oof_summary = adversarial_results.secondary_oof_summary
        prediction_column = primary_oof_summary.prediction_columns[1]

        #
        # create the plot data
        #

        primary_hist_input = primary_oof_summary.column_histogram_stats[
            prediction_column
        ]
        secondary_hist_input = secondary_oof_summary.column_histogram_stats[
            prediction_column
        ]

        x_data_primary = primary_hist_input["interval_mean"]
        y_data_primary = primary_hist_input[prediction_column]
        y_data_secondary = secondary_hist_input[prediction_column]

        explanation = e10s.GlobalGroupedBarChartExplanation(
            explainer=self,
            display_name=self.display_name,
            display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
        )

        jdf = formats.GlobalGroupedBarChartJSonDatatableFormat
        explanation_frame = datatable.Frame(
            {
                jdf.COL_X: x_data_primary.to_list(),
                jdf.COL_Y_GROUP_1: y_data_primary.to_list(),
                jdf.COL_Y_GROUP_2: y_data_secondary.to_list(),
            }
        )
        # index file (of per-class data files) (JSon)
        (
            idx_dict,
            idx_str,
        ) = formats.GlobalGroupedBarChartJSonDatatableFormat.serialize_index_file(
            classes=[AdversarialSimilarityExplainer.CLASS_ONE_AND_ONLY],
            doc=AdversarialSimilarityExplainer._description,
        )

        json_dt_format = formats.GlobalGroupedBarChartJSonDatatableFormat(
            explanation, idx_str
        )
        json_dt_format.update_index_file(
            idx_dict, total_rows=explanation_frame.shape[0]
        )
        # data file (datatable)
        json_dt_format.add_data_frame(
            format_data=explanation_frame,
            file_name=idx_dict[jdf.KEY_FILES][
                AdversarialSimilarityExplainer.CLASS_ONE_AND_ONLY
            ],
        )
        # JSon+datatable format can be added as explanation's representation
        explanation.add_format(json_dt_format)

        return explanation

    class Result(explainers.ExplainerResult):
        def __init__(
            self,
            persistence: persistences.ExplainerPersistence,
            explainer_id: str = "",
            h2o_sonar_config=None,
        ):
            self.explainer_id = explainer_id
            self.explanation = e10s.GlobalGroupedBarChartExplanation
            self.format = formats.GlobalGroupedBarChartJSonDatatableFormat
            self.config = h2o_sonar_config

            explainers.ExplainerResult.__init__(
                self,
                persistence=persistence,
                explainer_id=self.explainer_id,
                explanation=self.explanation,
                explanation_format=self.format,
                h2o_sonar_config=self.config,
            )

            self.log_name = "Adversarial similarity"

        def plot(
            self,
            *,
            file_path: str = "",
        ):
            jdf = formats.GlobalGroupedBarChartJSonDatatableFormat

            # data preparation
            data = self.data()
            x_data = data[jdf.COL_X].to_list()[0]
            y_data_primary = data[jdf.COL_Y_GROUP_1].to_list()[0]
            y_data_secondary = data[jdf.COL_Y_GROUP_2].to_list()[0]
            len_x_data = len(x_data) or 1

            # plot
            width = (max(x_data) - min(x_data)) / len_x_data / 3
            offset = width / 2.0

            x = x_data
            x_l_shifted = [xx - offset for xx in x]
            x_r_shifted = [xx + offset for xx in x]

            # plot data in grouped manner of bar type
            plt.bar(
                x_l_shifted,
                y_data_primary,
                width,
                color=commons.LookAndFeel.get_fg_color(self.config.look_and_feel),
            )
            plt.bar(
                x_r_shifted,
                y_data_secondary,
                width,
                color=commons.LookAndFeel.COLOR_HOT_ORANGE,
            )

            plt.title(AdversarialSimilarityExplainer.PLOT_TITLE)
            plt.xlabel("Similar-to-Secondary probability")
            plt.ylabel("Relative frequency")
            plt.legend(["Primary dataset", "Secondary dataset"])

            if file_path:
                plt.savefig(file_path)
            else:
                results.matplotlib_closing(True)

        def data(self) -> datatable.Frame:
            clazz = AdversarialSimilarityExplainer.CLASS_ONE_AND_ONLY
            idx_dict: dict = self.format.load_index_file(
                persistence=self.persistence,
                explanation_type=self.explanation.explanation_type(),
            )
            feature_file: str = idx_dict[self.format.KEY_FILES].get(clazz, "")
            if not feature_file:
                raise ValueError(
                    f"{self.log_name}Invalid class: {clazz}, available classes are: "
                    f"{list(idx_dict[self.format.KEY_FILES].keys())}"
                )
            feature_file = self.persistence.get_explanation_file_path(
                self.explanation.explanation_type(),
                MimeType.MIME_JSON_DATATABLE,
                feature_file,
            )
            frame = datatable.Frame(feature_file)

            return frame

        def _raw_data(self) -> datatable.Frame:
            return self.data()

        def summary(self, **kwargs) -> dict:
            return self.params()

    def get_result(self) -> Result:
        return AdversarialSimilarityExplainer.Result(
            persistence=self.persistence,
            explainer_id=AdversarialSimilarityExplainer.explainer_id(),
            h2o_sonar_config=self.config,
        )
