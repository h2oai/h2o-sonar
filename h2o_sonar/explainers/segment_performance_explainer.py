# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import os
import traceback

import airium
import datatable
import pandas
from matplotlib import pyplot

from h2o_sonar import errors
from h2o_sonar import loggers
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api import results
from h2o_sonar.lib.integrations import mv_adapter


try:
    from h2o_mv.core import mv_dataset
    from h2o_mv.core import mv_model
    from h2o_mv.recipes import segperf

    HAS_H2O_MV = True
except ImportError:
    HAS_H2O_MV = False


class SegmentPerformanceExplainer(
    explainers.Explainer, mv_adapter.ExplainerToMvTestAdapter
):
    """Drift detection explainer.

    @see https://docs.h2o.ai/wave-apps/h2o-model-validation/guide/tests
         /supported-validation-tests/segment-performance/segment-performance

    """

    _display_name = "Segment Performance"
    _description = (
        "A segment performance test lets you explore a model's data subsets "
        "(segments) that diverge from average scores. In other words, "
        "a segment performance test allows you to discover which data "
        "points (segments) the model struggles, outperformance, and performs "
        "with when generating accurate predictions. "
        ""
        "To run a segment performance test on a model, explainer utilizes "
        "a provided dataset to generate model predictions to assess their "
        "accuracy. Explainer splits the dataset into segments "
        "by the bins of values of every variable and every pair of variables "
        "to generate results around the ability of the model to produce "
        "accurate predictions with different data segments. These results "
        "are embedded into a bubble graph that explainer generates "
        "that enables you to observe and explore data segments the model struggles, "
        "outperformance, and performs with when generating accurate predictions. "
        "For each segment, explainer calculates its size of it "
        "relative to the size of the dataset and estimates the error "
        "the model makes on the corresponding segment. "
        ""
        "Exploring and identifying data segments in a model that do not perform "
        "as expected can lead to decisions on how you preprocess your data."
    )
    _iid: bool = True
    _time_series: bool = True

    _supported_dataset_locators = [commons.ResourceLocatorType.handle]
    _supported_model_locators = [commons.ResourceLocatorType.handle]

    _regression: bool = True
    _binary: bool = True
    _multiclass: bool = True

    _global_explanation = True
    _explanation_types = [
        e10s.WorkDirArchiveExplanation,
    ]

    # param: training dataset ~ dataset
    # param: primary dataset ~ passed using test dataset argument
    # param: model ~ model
    PARAM_WORKER = "worker_connection_key"
    PARAM_NUMBER_OF_BINS = "number_of_bins"
    PARAM_PRECISION = "precision"
    PARAM_DROP_COLS = "drop_cols"

    DEFAULT_NUMBER_OF_BINS = 5
    DEFAULT_PRECISION = 5
    DEFAULT_DROP_COLS = []

    _parameters = [
        explainers.ExplainerParam(
            param_name=PARAM_WORKER,
            description=(
                "The connection ID of the Driverless AI configured in the H2O "
                "Sonar configuration. Only Driverless AI servers with "
                "username and password authentication are supported."
            ),
            param_type=commons.ExplainerParamType.str,
            default_value=None,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_NUMBER_OF_BINS,
            description=(
                "Defines the number of bins the explainer utilizes to split the "
                "variable values of the primary dataset. In the case of a categorical "
                "column, the explainer utilizes the appropriate categories while "
                "ranging numerical columns into a specified number of bins."
                ""
                "To run a segment performance test on a model, the explainer "
                "utilizes a provided dataset to generate model predictions to "
                "assess their accuracy. Explainer splits the dataset "
                "into segments by the bins of values of every variable and every "
                "pair of variables to generate results around the ability of "
                "the model to produce accurate predictions with different "
                "data segments. These results are embedded into a bubble graph "
                "that the explainer generates that enables you to observe and "
                "explore data segments the model struggles, outperformance, "
                "and performs with when generating accurate predictions. "
                "For each segment, explainer calculates its size of "
                "it relative to the size of the dataset and estimates the error "
                "the model makes on the corresponding segment."
            ),
            param_type=commons.ExplainerParamType.int,
            default_value=DEFAULT_NUMBER_OF_BINS,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_PRECISION,
            description="Precision.",
            param_type=commons.ExplainerParamType.int,
            default_value=DEFAULT_PRECISION,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_DROP_COLS,
            description="Defines the columns to drop during the validation test.",
            param_type=commons.ExplainerParamType.list,
            default_value=DEFAULT_DROP_COLS,
            tags=[explainers.ExplainerParam.TAG_SRC_DATASET_COLUMN_NAMES],
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

    __RESULT_TITLE = "RMSE by segment (split by 2 features)"
    RESULT_FILE_CSV = "segment-performance.csv"

    def __init__(self):
        explainers.Explainer.__init__(self)
        mv_adapter.ExplainerToMvTestAdapter.__init__(self)

        self.args = None
        self.log_name = "Segment performance"

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
                    SegmentPerformanceExplainer.PARAM_WORKER, ""
                ),
                log_name=self.log_name,
                logger=self.logger,
            )
        except Exception as e:
            self.logger.warning(
                f"{self.log_name}: compatibility error {e}\n{traceback.format_exc()}"
            )
            return False

        mv_test = segperf.SegmentPerformance(
            name=SegmentPerformanceExplainer._display_name
        )
        mv_test.check_compatibility()

        # check that train and another is specified
        if not params.testset:
            msg = (
                f"{self.log_name} requires training dataset and also primary "
                f"dataset (specified in ``testset`` parameter) to assess the segment "
                f"performance."
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

        self.log_name = f"Segment performance {self.mli_key}/{self.key}"

    def explain(
        self,
        X: datatable.Frame,
        y: datatable.Frame | None = None,
        explanations_types: list = None,
        **e_params,
    ):
        testset = e_params.get("testset", None)
        if not testset:
            ValueError(
                f"{self.log_name}: testset is required to specify SECONDARY dataset"
            )
        self.logger.debug(f"{self.log_name}: testset: {testset}")

        #
        # calculation
        #
        # Imports moved to top level with try/except

        mv_test = None
        try:
            # Driverless AI connection
            # get connection by name from the H2O Sonar configuration
            worker_connection_key = self.args.get(
                SegmentPerformanceExplainer.PARAM_WORKER, ""
            )
            if not worker_connection_key:
                raise errors.MliError(
                    f"{self.log_name}: required worker connection name not specified "
                    f"in the explainer parameters"
                )
            # h2o_mv.platforms.driverless.platform.DriverlessPlatform:
            worker_connection = self._get_mv_dai_worker_by_key(
                connection_key=worker_connection_key, log_name=self.log_name
            )
            if not worker_connection:
                raise errors.MliError(
                    f"{self.log_name}: worker connection '{worker_connection_key}' "
                    f"not found in the H2O Sonar configuration. "
                )
            self.mv_client.set_worker(worker_connection)
            self.mv_client.add_connection(worker_connection)  # returns platform w/ mvid

            primary_dataset: mv_dataset.MVDataset = mv_dataset.MVDataset(
                name="Training dataset",
                platform_mvid=worker_connection.mvid,
                platform_obj_key=X.resource_key,
            )
            primary_dataset = worker_connection.import_dataset(
                primary_dataset.platform_obj_key
            )
            secondary_dataset: mv_dataset.MVDataset = mv_dataset.MVDataset(
                name="Test dataset",
                platform_mvid=worker_connection.mvid,
                platform_obj_key=testset.resource_key,
            )
            secondary_dataset = worker_connection.import_dataset(
                secondary_dataset.platform_obj_key
            )
            primary_model: mv_model.MVModel = mv_model.MVModel(
                name="Model",
                platform_mvid=worker_connection.mvid,
                training_set_mvid=primary_dataset.mvid,
                test_set_mvid=secondary_dataset.mvid,
                platform_obj_key=self.model.resource_key,
            )

            primary_model = worker_connection.import_model(
                primary_model.platform_obj_key
            )

            number_of_bins: int = self.args.get(
                SegmentPerformanceExplainer.PARAM_NUMBER_OF_BINS,
                SegmentPerformanceExplainer.DEFAULT_NUMBER_OF_BINS,
            )
            precision: int = self.args.get(
                SegmentPerformanceExplainer.PARAM_PRECISION,
                SegmentPerformanceExplainer.DEFAULT_PRECISION,
            )

            mv_test = segperf.SegmentPerformance(
                name=SegmentPerformanceExplainer._display_name
            )
            mv_test.settings = segperf.SegmentPerformanceSettings(
                model=primary_model,
                primary_dataset=primary_dataset,
                drop_columns=self.args.get(
                    SegmentPerformanceExplainer.PARAM_DROP_COLS,
                    SegmentPerformanceExplainer.DEFAULT_DROP_COLS,
                ),
                num_bins=number_of_bins,
                precision=precision,
            )
            mv_test.run()

            # assert MV test integrity
            mv_adapter.ExplainerToMvTestAdapter.assert_mv_test_status(self, mv_test)
        except Exception as e:
            self.logger.error(f"{self.log_name}: {e}\n{traceback.format_exc()}")
            raise e
        finally:
            if mv_test:
                self._dump_mv_test_log_to_explainer_log(mv_test.log)

        # normalization
        plots_paths = self.normalize_to_gom(mv_test.results.binned_df)

        #
        # explanations
        #
        explanations = []

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
            mv_test_log=mv_test.log,
            mv_test_artifacts={},
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
            try:
                explanations.append(self._normalize_html(plots_paths))
            except Exception as ex:
                self.logger.warning(
                    f"{self.log_name}: HTML representation creation failed: {ex}"
                )

        return explanations

    @staticmethod
    def normalize_scatter_plot(
        x: list,
        y: list,
        z: list,
        colors: list,
        x_axis_label: str,
        y_axis_label: str,
        plot_file_path: str,
        color_map: str = "Wistia",
        figsize=(12, 10),
        dpi=120,
    ) -> str:
        # NORMALIZE z-axis data
        min_bubble_size = 10
        ratio = min_bubble_size / min(z)
        z = [x * ratio for x in z]

        # FIGURE ~ set of plots
        figure, axes = pyplot.subplots(figsize=figsize, dpi=dpi)
        # PLOT
        scatter_plot = axes.scatter(x, y, s=z, c=colors, cmap=color_map)
        # axes LEGENDS
        axes.set_xlabel(x_axis_label)
        axes.set_ylabel(y_axis_label)
        # LEGEND(s)
        # color legend: RMSE
        handles, labels = scatter_plot.legend_elements(alpha=0.99)
        axes.legend(
            handles,
            labels,
            title="Average RMSE",
            loc="best",
            # bbox_to_anchor=(1, 0.99),  # outside the box
        )

        if plot_file_path:
            pyplot.savefig(plot_file_path, dpi=300)
        else:
            results.matplotlib_closing(True)

        # CLEAR figure to get ready for next plot
        pyplot.clf()

        return plot_file_path

    @staticmethod
    def _normalize_seg_perf_plot(
        feature1: str,
        feature2: str,
        chart_data: datatable.Frame,
        plot_file_path: str,
        logger: loggers.SonarLogger,
    ) -> str:
        x_tic_labels = chart_data[:, "value_1"].to_list()[0]
        y_tic_labels = chart_data[:, "value_2"].to_list()[0]

        x = x_tic_labels
        y = y_tic_labels
        z = chart_data[:, "Freq"].to_list()[0]
        colors = chart_data[:, "Metric"].to_list()[0]

        if x and y and z and colors:
            return SegmentPerformanceExplainer.normalize_scatter_plot(
                x=x,
                y=y,
                z=z,
                colors=colors,
                x_axis_label=feature1,
                y_axis_label=feature2,
                plot_file_path=plot_file_path,
            )

        if logger:
            logger.warning(
                f"SKIPPING chart for features {feature1} and {feature2} as "
                f"there are NO DATA"
            )
        return ""

    @staticmethod
    def _normalize_features_tuples(result_df: datatable.Frame) -> list:
        # build feature tuples
        features_1 = datatable.unique(result_df["var_1"]).to_list()[0]
        features_2 = datatable.unique(result_df["var_2"]).to_list()[0]
        features_combinations = []
        for i in range(len(features_1)):
            for j in range(len(features_2)):
                if features_1[i] != features_2[j]:
                    features_combinations.append((features_1[i], features_2[j]))

        return features_combinations

    @staticmethod
    def _normalize_plot_data_for_features(
        feature1: str, feature2: str, result_df: datatable.Frame
    ) -> datatable.Frame:
        return result_df[datatable.f["var_1"] == feature1, :][
            datatable.f["var_2"] == feature2, :
        ]

    @staticmethod
    def _normalize_plots_to_work(
        result_df: datatable.Frame,
        persistence: persistences.ExplainerPersistence,
        logger: loggers.SonarLogger,
    ) -> list:
        plots_paths = []
        features_combinations = SegmentPerformanceExplainer._normalize_features_tuples(
            result_df
        )
        for i, f_tuple in enumerate(features_combinations):
            file_name = f"scatter-{i}.png"
            feature1 = f_tuple[0]
            feature2 = f_tuple[1]
            chart_data = SegmentPerformanceExplainer._normalize_plot_data_for_features(
                feature1=feature1, feature2=feature2, result_df=result_df
            )
            plot_path = SegmentPerformanceExplainer._normalize_seg_perf_plot(
                feature1=feature1,
                feature2=feature2,
                chart_data=chart_data,
                plot_file_path=persistence.get_explainer_working_file(file_name),
                logger=logger,
            )
            if plot_path:
                plots_paths.append(plot_path)

        return plots_paths

    def normalize_to_gom(self, pd_result_df: pandas.DataFrame):
        # result frame (pandas.DataFrame) columns:
        #   value_1 / Freq / Metric / value_2 / var_1 / var_2
        # > value_1 & value_2 are x an y axis tick labels
        # > select rows: var_1 == ... && var_2 == ...
        result_df = datatable.Frame(pd_result_df)
        result_df_path = self.persistence.get_explainer_working_file(
            SegmentPerformanceExplainer.RESULT_FILE_CSV
        )
        result_df.to_csv(result_df_path)

        plots_paths = self._normalize_plots_to_work(
            result_df=result_df, persistence=self.persistence, logger=self.logger
        )
        return plots_paths

    def _normalize_html(self, plots_paths: list):
        e_type = e10s.GlobalHtmlFragmentExplanation

        global_explanation = e10s.GlobalHtmlFragmentExplanation(
            explainer=self,
            display_name=SegmentPerformanceExplainer._display_name,
            display_category=e10s.Explanation.DISPLAY_CAT_DAI_MODEL,
        )

        # HTML format
        html_src = airium.Airium()

        with html_src.b():
            html_src(SegmentPerformanceExplainer.__RESULT_TITLE)

            for plot_src_path in plots_paths:
                plot_format_path = self.persistence.get_relative_path(
                    self.persistence.get_explanation_file_path(
                        explanation_type=e_type.explanation_type(),
                        explanation_format=f5s.HtmlFormat.mime,
                        explanation_file=os.path.basename(plot_src_path),
                    )
                )
                with html_src.div():
                    html_src.img(
                        src=plot_format_path,
                        alt=SegmentPerformanceExplainer.__RESULT_TITLE,
                        # ensure that image will not overflow enclosing <div/>
                        style=(
                            "height: 100%; max-width: 100%; display: block; "
                            "margin: auto;"
                        ),
                    )
                    html_src.br()

        report_text = str(html_src)
        report_file_name = f"{self.persistence.FILE_EXPLANATION}.html"
        report_path = self.persistence.get_explainer_working_file(report_file_name)
        with open(report_path, mode="w") as file:
            file.write(report_text)

        # add HTML format (will copy report and image files from work)
        html_format = f5s.HtmlFormat(
            explanation=global_explanation,
            format_data="",  # data will be updated later
            format_file=os.path.join(
                self.persistence.get_explainer_working_dir(),
                report_file_name,
            ),
            extra_format_files=plots_paths,
            persistence=self.persistence.store,
        )
        global_explanation.add_format(html_format)

        return global_explanation

    class Result(explainers.ExplainerResult):
        def __init__(
            self,
            persistence: persistences.ExplainerPersistence,
            explainer_id: str = "",
            h2o_sonar_config=None,
        ):
            self.explainer_id = explainer_id
            self.explanation = e10s.CustomArchiveExplanation
            self.format = f5s.CustomArchiveZipFormat
            self.config = h2o_sonar_config

            explainers.ExplainerResult.__init__(
                self,
                persistence=persistence,
                explainer_id=self.explainer_id,
                explanation=self.explanation,
                explanation_format=self.format,
                h2o_sonar_config=self.config,
            )

            self.log_name = "Segment performance"

        def data(self) -> datatable.Frame:
            return datatable.fread(
                self.persistence.get_explainer_working_file(
                    SegmentPerformanceExplainer.RESULT_FILE_CSV
                )
            )

        def _raw_data(self) -> datatable.Frame:
            return self.data()

        def plot(
            self,
            *,
            feature_1: str = "",
            feature_2: str = "",
            file_path: str = "",
        ):
            explainer = SegmentPerformanceExplainer

            result_df = self.data()
            if not feature_1 or not feature_2:
                features_combinations = explainer._normalize_features_tuples(result_df)
                if features_combinations:
                    (feature_1, feature_2) = features_combinations[0]
                else:
                    raise errors.MliError(
                        f"{self.log_name} Cannot plot segment performance: no "
                        f"features found"
                    )
            SegmentPerformanceExplainer._normalize_seg_perf_plot(
                feature1=feature_1,
                feature2=feature_2,
                chart_data=explainer._normalize_plot_data_for_features(
                    feature1=feature_1,
                    feature2=feature_2,
                    result_df=result_df,
                ),
                plot_file_path=file_path,
                logger=self.logger,
            )

        def summary(self, **kwargs) -> dict:
            return self.params()

    def get_result(self) -> Result:
        return SegmentPerformanceExplainer.Result(
            persistence=self.persistence,
            explainer_id=SegmentPerformanceExplainer.explainer_id(),
            h2o_sonar_config=self.config,
        )
