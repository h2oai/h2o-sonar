# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import os
import traceback

import airium
import datatable
import pandas

from h2o_sonar import errors
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api import plots
from h2o_sonar.lib.api import results
from h2o_sonar.lib.integrations import mv_adapter


try:
    from h2o_mv.core import mv_dataset
    from h2o_mv.core import mv_model
    from h2o_mv.recipes import backtesting
    from h2o_mv.recipes.backtesting import Backtesting

    HAS_H2O_MV = True
except ImportError:
    HAS_H2O_MV = False


class BacktestingExplainer(explainers.Explainer, mv_adapter.ExplainerToMvTestAdapter):
    """Size dependency explainer.

    @see https://docs.h2o.ai/wave-apps/h2o-model-validation/guide/tests/
         /supported-validation-tests/backtesting/backtesting

    """

    _display_name = "Backtesting"
    _description = """A backtesting test enables you to assess the robustness of
a model using the existing historical training data through a series of iterative
training where training data is used from its recent to oldest collected values.

Explainer performs a backtesting test by fitting a predictive model with a training
dataset to later assess the model with a separate test dataset where the test
dataset does not overlap with the training data. Frequently, the training data is
collected over time and has an explicit time dimension. In such cases, utilizing
the most recent dataset points for the model test dataset is common practice,
as it better mimics a real model application.

By applying a backtesting test to a model, we can refit the model multiple times,
where every time, we use shorter time spans of the training data while using
a portion of that data as test data. As a result, the test dataset is replaced with
a series of values over time.

A backtesting test enables you to:

* Understand the variance of the model accuracy
* Understand and visualize how the model accuracy develops over time
* Identify potential reasons for any performance issues with the modeling approach
  in the past (for example, problems around data collection)

Each iteration during backtesting, the model is fully refitted, which includes
a rerun of feature engineering and feature selection. Not entirely refitting during
each iteration results in an incorrect backtesting outcome because the next iteration
would have selected features based on the entire data. An incorrect backtesting
outcome also leads to data leakage where information from the future is explicitly
or implicitly reflected in the current variables.
    """
    _iid: bool = True
    _time_series: bool = True

    _supported_dataset_locators = [
        commons.ResourceLocatorType.handle,
    ]
    _supported_model_locators = [
        commons.ResourceLocatorType.handle,
    ]

    _regression: bool = True
    _binary: bool = True
    _multiclass: bool = True

    _global_explanation = True
    _explanation_types = [
        e10s.WorkDirArchiveExplanation,
        e10s.GlobalFeatImpExplanation,
    ]

    # param: model ~ model
    # param: training dataset ~ dataset
    # param: time column
    PARAM_WORKER = "worker_connection_key"
    PARAM_TIME_COLUMN = "time_col"
    PARAM_SPLIT_TYPE = "split_type"
    PARAM_NUMBER_OF_SPLITS = "number_of_splits"
    PARAM_NUMBER_OF_FORECAST_PERIODS = "number_of_forecast_period"
    PARAM_FORECAST_PERIOD_UNIT = "forecast_period_unit"
    PARAM_NUMBER_OF_TRAINING_PERIODS = "number_of_training_period"
    PARAM_TRAINING_PERIOD_UNIT = "training_period_unit"
    PARAM_CUSTOM_DATES = "custom_dates"
    PARAM_PLOT_TYPE = "plot_type"

    OPT_SPLIT_TYPE_AUTO = "auto"
    OPT_SPLIT_TYPE_CUSTOM = "custom"

    DEFAULT_TIME_COLUMN = ""
    DEFAULT_NUMBER_OF_SPLITS = 2
    DEFAULT_SPLIT_TYPE = OPT_SPLIT_TYPE_AUTO
    DEFAULT_NUMBER_OF_FORECAST_PERIODS = None
    DEFAULT_FORECAST_PERIOD_UNIT = None
    DEFAULT_NUMBER_OF_TRAINING_PERIODS = None
    DEFAULT_TRAINING_PERIOD_UNIT = None
    DEFAULT_CUSTOM_DATES = None

    __RESULT_TITLE_MAPE = "Backtesting MAPE"
    __RESULT_TITLE_HEAT = "Feature importance for different split dates"
    __HEAT_LABEL_X_AXIS = "SPLIT"
    __HEAT_LABEL_Y_AXIS = "Features"

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
            param_name=PARAM_TIME_COLUMN,
            description=(
                "Defines the time column in the primary dataset that the explainer "
                "utilizes to split the primary dataset (training dataset) during "
                "the backtesting test."
            ),
            param_type=commons.ExplainerParamType.str,
            default_value=DEFAULT_TIME_COLUMN,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_SPLIT_TYPE,
            description=(
                "Split type - either 'auto' (explainer determines the split dates, "
                "which create the backtesting experiments) or 'custom' (you "
                "determine the split dates, which create the backtesting experiments)."
            ),
            param_type=commons.ExplainerParamType.str,
            default_value=DEFAULT_SPLIT_TYPE,
            predefined=[
                OPT_SPLIT_TYPE_AUTO,
                OPT_SPLIT_TYPE_CUSTOM,
            ],
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_NUMBER_OF_SPLITS,
            description=(
                "Defines the number of dates (splits) the explainer utilizes for the "
                "dataset splitting and model refitting process during the validation "
                "test. The explainer sets the specified number of dates in the past "
                "while being equally apart from one another to generate appropriate "
                "dataset splits."
            ),
            param_type=commons.ExplainerParamType.int,
            default_value=DEFAULT_NUMBER_OF_SPLITS,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_NUMBER_OF_FORECAST_PERIODS,
            description="Number of the forest periods.",
            param_type=commons.ExplainerParamType.int,
            default_value=None,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_FORECAST_PERIOD_UNIT,
            description="Forecast period unit.",
            param_type=commons.ExplainerParamType.str,
            default_value=None,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_NUMBER_OF_TRAINING_PERIODS,
            description="Number of training periods.",
            param_type=commons.ExplainerParamType.int,
            default_value=None,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_TRAINING_PERIOD_UNIT,
            description="Training period unit.",
            param_type=commons.ExplainerParamType.str,
            default_value=None,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_CUSTOM_DATES,
            description="Custom dates.",
            param_type=commons.ExplainerParamType.str,
            default_value=None,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_PLOT_TYPE,
            description=(
                f"Plot type - one of '{plots.Data3dPlot.PLOT_TYPE_HEATMAP}', "
                f"'{plots.Data3dPlot.PLOT_TYPE_CONTOUR}' or "
                f"'{plots.Data3dPlot.PLOT_TYPE_SURFACE}'."
            ),
            param_type=commons.ExplainerParamType.str,
            default_value=plots.Data3dPlot.PLOT_TYPE_HEATMAP,
            predefined=plots.Data3dPlot.PLOT_TYPES,
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

    def __init__(self):
        explainers.Explainer.__init__(self)
        mv_adapter.ExplainerToMvTestAdapter.__init__(self)

        self.args = None
        self.log_name = "Backtesting"

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
                dai_connection_key=self.args.get(BacktestingExplainer.PARAM_WORKER, ""),
                log_name=self.log_name,
                logger=self.logger,
            )
        except Exception as e:
            self.logger.warning(
                f"{self.log_name}: compatibility error {e}\n{traceback.format_exc()}"
            )
            return False

        mv_test = Backtesting(name=BacktestingExplainer._display_name)
        mv_test.check_compatibility()

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

        self.log_name = f"Backtesting {self.mli_key}/{self.key}"

    def explain(
        self,
        X,
        y=None,
        explanations_types: list = None,
        **e_params,
    ):
        #
        # calculation
        #

        mv_test = None
        try:
            # Driverless AI connection
            worker_connection_key = self.args.get(BacktestingExplainer.PARAM_WORKER, "")
            if not worker_connection_key:
                raise errors.MliError(
                    f"{self.log_name}: required worker connection name not specified "
                    f"in the explainer parameters"
                )
            worker_connection = self._get_mv_dai_worker_by_key(
                connection_key=worker_connection_key,
                log_name=self.log_name,
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
            primary_model: mv_model.MVModel = mv_model.MVModel(
                name="Model",
                platform_mvid=worker_connection.mvid,
                training_set_mvid=primary_dataset.mvid,
                platform_obj_key=self.model.resource_key,
            )

            primary_model = worker_connection.import_model(
                primary_model.platform_obj_key
            )

            mv_test = backtesting.Backtesting(name=BacktestingExplainer._display_name)
            mv_test.settings = backtesting.BacktestingSettings(
                model=primary_model,
                time_column=self.args.get(
                    BacktestingExplainer.PARAM_TIME_COLUMN,
                    BacktestingExplainer.PARAM_TIME_COLUMN,
                ),
                split_type=self.args.get(
                    BacktestingExplainer.PARAM_SPLIT_TYPE,
                    BacktestingExplainer.DEFAULT_SPLIT_TYPE,
                ),
                num_splits=self.args.get(
                    BacktestingExplainer.PARAM_NUMBER_OF_SPLITS,
                    BacktestingExplainer.DEFAULT_NUMBER_OF_SPLITS,
                ),
                num_forecast_period=self.args.get(
                    BacktestingExplainer.PARAM_NUMBER_OF_FORECAST_PERIODS,
                    BacktestingExplainer.DEFAULT_NUMBER_OF_FORECAST_PERIODS,
                ),
                forecast_period_unit=self.args.get(
                    BacktestingExplainer.PARAM_FORECAST_PERIOD_UNIT,
                    BacktestingExplainer.DEFAULT_FORECAST_PERIOD_UNIT,
                ),
                num_training_period=self.args.get(
                    BacktestingExplainer.PARAM_NUMBER_OF_TRAINING_PERIODS,
                    BacktestingExplainer.DEFAULT_NUMBER_OF_TRAINING_PERIODS,
                ),
                training_period_unit=self.args.get(
                    BacktestingExplainer.PARAM_TRAINING_PERIOD_UNIT,
                    BacktestingExplainer.DEFAULT_TRAINING_PERIOD_UNIT,
                ),
                custom_dates=self.args.get(
                    BacktestingExplainer.PARAM_CUSTOM_DATES,
                    BacktestingExplainer.DEFAULT_CUSTOM_DATES,
                ),
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

        #
        # explanations
        #
        explanations = []

        # EXPLANATION: feature importance
        self._normalize_to_gom(mv_test.results, explanations)

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
                explanations.append(self._normalize_html())
            except Exception as ex:
                self.logger.warning(
                    f"{self.log_name}: HTML representation creation failed: {ex}"
                )

        return explanations

    def _normalize_fis(
        self, fis_frames: list[pandas.DataFrame], fis_sizes: list
    ) -> e10s.Global3dDataExplanation:
        """Normalize feature importances frame to heatmap data.

        Parameters
        ----------
        fis_frames : list[pandas.DataFrame]
          Feature importances frames with columns: "Feature", "Gain" (and optionally
          "Description" in case of transformed features).

        fis_sizes : list
          Sizes of the training datasets.

        """
        # INPUT fis_frame(s) - number of rows / features might be DIFFERENT in frames:
        #
        # Feature | Gain
        # <f-1>   | <gain-1>
        # ...
        # <f-n>   | <gain-n>
        #
        # RESULT of this method is an AGGREGATED frame w/ data for all datasets sizes
        #
        # "Feature" | "<1-st dataset size (gain)>" | ... | "<n-th dataset size (gain)>"
        # f-1
        # ...
        # f-m
        #
        # METHOD:
        #
        # 1. build aggregate dictionary (as input frames might be sparse)
        #
        #   {
        #     "Feature": ["f-1", ..., "f-nm"],
        #     "1-st dataset size (gain): [gain-1, ..., gain-n],
        #     ...
        #     "n-th dataset size (gain): [gain-1, ..., gain-n],
        #   }
        #
        # 2. dictionary -> datatable.Frame
        #
        agg_dict = {
            "Feature": [],
        }
        done_fis_cols = []
        for i, fis_size in enumerate(fis_sizes):  # for each frame
            fis_col = str(fis_size)
            agg_dict[fis_col] = [0.0 for _ in range(len(agg_dict["Feature"]))]
            fis_frame_dict = datatable.Frame(fis_frames[i]).to_dict()
            for j, f in enumerate(fis_frame_dict["Feature"]):  # for each feature @ frm
                f_gain = fis_frame_dict["Gain"][j]
                if f not in agg_dict["Feature"]:
                    agg_dict["Feature"].append(f)
                    agg_dict[fis_col].append(f_gain)
                    # also complete feature gains for all previous frames
                    for done_fis_col in done_fis_cols:
                        agg_dict[done_fis_col].append(0.0)
                else:
                    f_agg_idx = agg_dict["Feature"].index(f)
                    agg_dict[fis_col][f_agg_idx] = f_gain
            done_fis_cols.append(fis_col)
        agg_frame = datatable.Frame(agg_dict)
        normalized_agg_dict = {}
        for k in agg_dict:
            if k != "Feature":
                normalized_agg_dict[k] = {}
                for i, v in enumerate(agg_dict[k]):
                    normalized_agg_dict[k][agg_dict["Feature"][i]] = v

        # normalization to the explanation
        global_explanation = e10s.Global3dDataExplanation(
            explainer=self,
            display_name=BacktestingExplainer._display_name,
            display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
        )

        # always one class
        classes = ["Global"]

        # JSon format
        features = [BacktestingExplainer.__RESULT_TITLE_HEAT]
        features_names = [
            [
                BacktestingExplainer.__HEAT_LABEL_X_AXIS,
                BacktestingExplainer.__HEAT_LABEL_Y_AXIS,
            ]
        ]

        (idx_dict, idx_json_str) = f5s.Global3dDataJSonFormat.serialize_index_file(
            features=features,
            features_names=features_names,
            classes=classes,
            keywords=BacktestingExplainer._keywords,
            doc=BacktestingExplainer._description,
        )
        # safe data files
        for f in idx_dict[f5s.ExplanationFormat.KEY_FEATURES]:
            f_entry = idx_dict[f5s.ExplanationFormat.KEY_FEATURES][f]
            dir_path = self.persistence.get_explanation_dir_path(
                explanation_type=e10s.Global3dDataExplanation.explanation_type(),
                explanation_format=f5s.Global3dDataJSonFormat.mime,
            )
            self.persistence.store.make_dir(dir_path)
            file_path = self.persistence.get_explanation_file_path(
                explanation_type=e10s.Global3dDataExplanation.explanation_type(),
                explanation_format=f5s.Global3dDataJSonFormat.mime,
                explanation_file=list(
                    f_entry[f5s.ExplanationFormat.KEY_FILES].values()
                )[0],
            )
            self.persistence.store.save_json(
                key=file_path,
                data=normalized_agg_dict,
            )

        json_format = f5s.Global3dDataJSonFormat(
            explanation=global_explanation,
            json_data=idx_json_str,
            persistence=self.persistence.store,
        )
        global_explanation.add_format(json_format)

        # JSon+CSV format
        (idx_dict, idx_json_str) = f5s.Global3dDataJSonCsvFormat.serialize_index_file(
            features=features,
            features_names=features_names,
            classes=classes,
            keywords=BacktestingExplainer._keywords,
            doc=BacktestingExplainer._description,
        )
        # safe data files
        for f in idx_dict[f5s.ExplanationFormat.KEY_FEATURES]:
            f_entry = idx_dict[f5s.ExplanationFormat.KEY_FEATURES][f]
            dir_path = self.persistence.get_explanation_dir_path(
                explanation_type=e10s.Global3dDataExplanation.explanation_type(),
                explanation_format=f5s.Global3dDataJSonCsvFormat.mime,
            )
            self.persistence.store.make_dir(dir_path)
            file_path = self.persistence.get_explanation_file_path(
                explanation_type=e10s.Global3dDataExplanation.explanation_type(),
                explanation_format=f5s.Global3dDataJSonCsvFormat.mime,
                explanation_file=list(
                    f_entry[f5s.ExplanationFormat.KEY_FILES].values()
                )[0],
            )
            self.persistence.store.save(
                key=file_path,
                data=agg_frame.to_csv(),
                data_type=persistences.PersistenceDataType.text,
            )

        csv_format = f5s.Global3dDataJSonCsvFormat(
            explanation=global_explanation,
            json_data=idx_json_str,
            persistence=self.persistence.store,
        )
        global_explanation.add_format(csv_format)

        #
        # safe heatmap figure to work/
        #
        self._normalize_to_plot(agg_frame)

        return global_explanation

    _FILE_NAME_PLOT = "heatmap.png"

    def _normalize_to_plot(
        self,
        aggregated_frame: datatable.Frame,
    ):
        # resolve PLOT TYPE
        plot_type = self.args.get(
            BacktestingExplainer.PARAM_PLOT_TYPE,
            plots.Data3dPlot.PLOT_TYPE_HEATMAP,
        )
        if plot_type not in plots.Data3dPlot.PLOT_TYPES:
            default_plot = plots.Data3dPlot.PLOT_TYPE_HEATMAP
            self.logger.warning(
                f"{self.log_name}: unknown plot type '{plot_type}' - using fallback "
                f"to {default_plot}"
            )
            plot_type = default_plot

        # labels and data
        x_axis_labels = list(aggregated_frame.names)[1:]
        y_axis_labels = aggregated_frame["Feature"].to_list()[0]
        aggregated_frame_data = aggregated_frame.copy()
        del aggregated_frame_data["Feature"]

        # save the plot to the working dir
        plots.Data3dPlot.plot(
            x_axis_labels=x_axis_labels,
            y_axis_labels=y_axis_labels,
            heatmap_data=aggregated_frame_data,
            chart_title=BacktestingExplainer.__RESULT_TITLE_HEAT,
            x_axis_label=BacktestingExplainer.__HEAT_LABEL_X_AXIS,
            y_axis_label=BacktestingExplainer.__HEAT_LABEL_Y_AXIS,
            plot_type=plot_type,
            plot_file_path=self.persistence.get_explainer_working_file(
                BacktestingExplainer._FILE_NAME_PLOT
            ),
            logger=self.logger,
            log_name=self.log_name,
        )

    def _normalize_to_gom(self, backtesting_result, explanations: list):
        """Normalize FI + MAPE to the Grammar of MLI (GoM) format to get free charts."""

        # normalize size dependency explainer FIs results (orig + trans) to the heatmap:
        # x: train dataset size
        # y: feature name
        # z: feature importance

        # original feature importances
        if backtesting_result.orig_feature_importances:
            explanations.append(
                self._normalize_fis(
                    backtesting_result.orig_feature_importances,
                    backtesting_result.train_sizes,
                )
            )
        # SKIP TRANSFORMED > no sense to make heatmap for trans. feature importances

    def _normalize_html(self):
        e_type = e10s.GlobalHtmlFragmentExplanation

        global_explanation = e10s.GlobalHtmlFragmentExplanation(
            explainer=self,
            display_name=BacktestingExplainer._display_name,
            display_category=e10s.Explanation.DISPLAY_CAT_DAI_MODEL,
        )

        # HTML format
        html_src = airium.Airium()

        img_src_path = self.persistence.get_explainer_working_file(
            BacktestingExplainer._FILE_NAME_PLOT
        )
        img_file_path = self.persistence.get_relative_path(
            self.persistence.get_explanation_file_path(
                explanation_type=e_type.explanation_type(),
                explanation_format=f5s.HtmlFormat.mime,
                explanation_file=BacktestingExplainer._FILE_NAME_PLOT,
            )
        )

        with html_src.b():
            html_src(BacktestingExplainer.__RESULT_TITLE_HEAT)
            with html_src.div():
                html_src.img(
                    src=img_file_path,
                    alt=BacktestingExplainer.__RESULT_TITLE_HEAT,
                    # ensure that image will not overflow enclosing <div/>
                    style=(
                        "height: 100%; max-width: 100%; display: block; margin: auto;"
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
            extra_format_files=[img_src_path],
            persistence=self.persistence.store,
        )
        global_explanation.add_format(html_format)

        return global_explanation

    def get_result(self) -> results.Data3dResult:
        return results.Data3dResult(
            persistence=self.persistence,
            h2o_sonar_config=self.config,
            explainer_id=BacktestingExplainer.explainer_id(),
            logger=self.logger,
        )
