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
    from h2o_mv.recipes import sizedep
    from h2o_mv.recipes.sizedep import SizeDependency

    HAS_H2O_MV = True
except ImportError:
    HAS_H2O_MV = False


class SizeDependencyExplainer(
    explainers.Explainer, mv_adapter.ExplainerToMvTestAdapter
):
    """Size dependency explainer.

    @see https://docs.h2o.ai/wave-apps/h2o-model-validation/guide/tests/
         /supported-validation-tests/size-dependency/size-dependency

    """

    _display_name = "Size Dependency"
    _description = """A size dependency test enables you to analyze the effects
different sizes of train data will have on the accuracy of a selected model.
In particular, size dependency facilitates model stability analysis and,
for example, can answer whether augmenting an existing train data seems to be
promising in terms of model accuracy.

Explainer selects an appropriate sampling technique for the size dependency
test based on the selected model type. Available sampling techniques are as follows:

* **Random sampling**
   * Explainer uses a random sampling technique to create new sub-training
     samples for independent and **identically distributed (IID)** models
* **Expanding window sampling**
   * Explainer uses a expanding window sampling technique to create new
     sub-training samples for **time series** models while utilizing time columns to
     ensure that sub-training samples grow from recent to oldest data

For either sampling technique (random or expanding window), before it is applied,
the original train data is split using folds that improve generalization and data
balance when one of the sampling techniques is applied while using folds.

In the case of **IID models**, folds and sub-training samples are created at random,
but for **time series models**, folds and sub-training samples are created using
a time column to ensure that sub-training samples grow from the most recent to
oldest data points.

Based on the number of folds (N), explainer retrains the model N times
by only updating its training dataset with the new sub-training samples while
generating a scorer for each iteration of the retraining process for further analysis.

Sampling the original training data for a model under random or expanding window
sampling can be illustrated in the below image when N (folds) equals **4**.
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

    # param: training dataset ~ dataset
    # param: test dataset ~ test dataset
    # param: model ~ model
    PARAM_WORKER = "worker_connection_key"
    PARAM_TIME_COLUMN = "time_col"
    PARAM_NUMBER_OF_SPLITS = "number_of_splits"
    PARAM_WORKER_CLEANUP = "worker_cleanup"
    PARAM_PLOT_TYPE = "plot_type"

    DEFAULT_TIME_COLUMN = ""
    DEFAULT_NUMBER_OF_SPLITS = 2
    DEFAULT_WORKER_CLEANUP = True

    __RESULT_TITLE = "Feature importance for different training data sizes"
    __LABEL_X_AXIS = "Training data size"
    __LABEL_Y_AXIS = "Features"

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
                "Defines the time column of the primary and secondary dataset, which "
                "explainer utilizes to perform time-based splits."
            ),
            param_type=commons.ExplainerParamType.str,
            default_value=DEFAULT_TIME_COLUMN,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_NUMBER_OF_SPLITS,
            description=(
                "Defines the number of splits that explainer performs on the primary "
                "dataset to assess dataset size dependency."
            ),
            param_type=commons.ExplainerParamType.int,
            default_value=DEFAULT_NUMBER_OF_SPLITS,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_WORKER_CLEANUP,
            description=(
                "Determines if the explainer should delete the artifacts created in "
                "the worker. In this case, artifacts refer to experiments and "
                "datasets generated during the backtesting validation test. "
                "By default, explainer checks this setting (enables it), "
                "and accordingly, H2O Model Validation deletes all artifacts "
                "because they are no longer needed after the validation test is "
                "complete."
            ),
            param_type=commons.ExplainerParamType.bool,
            default_value=DEFAULT_WORKER_CLEANUP,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_PLOT_TYPE,
            description="Plot type - one of 'heatmap', 'contour-3d' or 'surface-3d'.",
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
        self.log_name = "Size dependency"

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
                    SizeDependencyExplainer.PARAM_WORKER, ""
                ),
                log_name=self.log_name,
                logger=self.logger,
            )
        except Exception as e:
            self.logger.warning(
                f"{self.log_name}: compatibility error {e}\n{traceback.format_exc()}"
            )
            return False

        mv_test = SizeDependency(name=SizeDependencyExplainer._display_name)
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

        self.log_name = f"Size dependency {self.mli_key}/{self.key}"

    def explain(
        self,
        X,
        y=None,
        explanations_types: list = None,
        **e_params,
    ):
        testset = e_params.get("testset", None)
        if not testset:
            ValueError(f"{self.log_name}: testset is required, but not specified.")
        self.logger.debug(f"{self.log_name}: testset: {testset}")

        #
        # calculation
        #
        mv_test = None
        try:
            # Driverless AI connection
            # get connection by name from the H2O Sonar configuration
            worker_connection_key = self.args.get(
                SizeDependencyExplainer.PARAM_WORKER, ""
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

            time_col: str | None = self.args.get(
                SizeDependencyExplainer.PARAM_TIME_COLUMN,
                SizeDependencyExplainer.PARAM_TIME_COLUMN,
            )
            num_splits: int = self.args.get(
                SizeDependencyExplainer.PARAM_NUMBER_OF_SPLITS,
                SizeDependencyExplainer.DEFAULT_NUMBER_OF_SPLITS,
            )

            mv_test = sizedep.SizeDependency(name=SizeDependencyExplainer._display_name)
            mv_test.settings = sizedep.SizeDependencySettings(
                model=primary_model,
                time_column=time_col,
                num_splits=num_splits,
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

        # EXPLANATION: size dependency as feature importance
        explanations = self._normalize_to_gom(mv_test.results, explanations)

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
        # aggregate result to 1 datatable frame w/ data for all datasets sizes
        #
        # Feature | <1-st dataset size gain> | ... | <n-th dataset size gain>
        # f-1
        # ...
        # f-m
        names = ["Feature"]
        aggregated_frame = datatable.Frame(
            {names[0]: datatable.Frame(fis_frames[0])[names[0]].to_list()[0]}
        )
        for i, f in enumerate(fis_frames):
            names.append(str(fis_sizes[i]))
            aggregated_frame.cbind(datatable.Frame(f)[:, "Gain"])
            aggregated_frame.names = names
        aggregated_data_dict = {}
        for r in range(aggregated_frame.shape[0]):
            d = aggregated_frame[r, :].to_dict()
            d_keys = list(d.keys())[1:]
            aggregated_data_dict[d[names[0]][0]] = {k: d[k][0] for k in d_keys}

        # normalization to the explanation
        global_explanation = e10s.Global3dDataExplanation(
            explainer=self,
            display_name=SizeDependencyExplainer._display_name,
            display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
        )

        # always one class
        classes = ["Global"]

        # JSon format
        features = [SizeDependencyExplainer.__RESULT_TITLE]
        features_names = [
            [
                SizeDependencyExplainer.__LABEL_X_AXIS,
                SizeDependencyExplainer.__LABEL_Y_AXIS,
            ]
        ]

        (idx_dict, idx_json_str) = f5s.Global3dDataJSonFormat.serialize_index_file(
            features=features,
            features_names=features_names,
            classes=classes,
            keywords=SizeDependencyExplainer._keywords,
            doc=SizeDependencyExplainer._description,
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
                data=aggregated_data_dict,  # only one representation feature
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
            keywords=SizeDependencyExplainer._keywords,
            doc=SizeDependencyExplainer._description,
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
                data=aggregated_frame.to_csv(),
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
        self._normalize_to_plot(aggregated_frame)

        return global_explanation

    def _normalize_to_plot(
        self,
        aggregated_frame: datatable.Frame,
    ):
        # resolve PLOT TYPE
        plot_type = self.args.get(
            SizeDependencyExplainer.PARAM_PLOT_TYPE,
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
            chart_title=SizeDependencyExplainer.__RESULT_TITLE,
            x_axis_label=SizeDependencyExplainer.__LABEL_X_AXIS,
            y_axis_label=SizeDependencyExplainer.__LABEL_Y_AXIS,
            plot_type=plot_type,
            plot_file_path=self.persistence.get_explainer_working_file(
                SizeDependencyExplainer._FILE_NAME_PLOT
            ),
            logger=self.logger,
            log_name=self.log_name,
        )

    def _normalize_to_gom(self, sd_result, explanations) -> list:
        """Normalize PSI to the Grammar of MLI (GoM) format to get free charts."""

        # normalize size dependency explainer FIs results (orig + trans) to the heatmap:
        # x: train dataset size
        # y: feature name
        # z: feature importance

        if sd_result.training_sizes:
            # original feature importances
            if sd_result.orig_feature_importances:
                explanations.append(
                    self._normalize_fis(
                        sd_result.orig_feature_importances, sd_result.training_sizes
                    )
                )

            # SKIP TRANSFORMED > no sense to make heatmap for trans. feature importances

        return explanations

    _FILE_NAME_PLOT = "heatmap.png"

    def _normalize_html(self):
        e_type = e10s.GlobalHtmlFragmentExplanation

        img_src_path = self.persistence.get_explainer_working_file(
            SizeDependencyExplainer._FILE_NAME_PLOT
        )

        global_explanation = e10s.GlobalHtmlFragmentExplanation(
            explainer=self,
            display_name=SizeDependencyExplainer._display_name,
            display_category=e10s.Explanation.DISPLAY_CAT_DAI_MODEL,
        )

        # HTML format
        html_src = airium.Airium()

        with html_src.b():
            html_src(SizeDependencyExplainer.__RESULT_TITLE)
            img_format_file = self.persistence.get_relative_path(
                self.persistence.get_explanation_file_path(
                    explanation_type=e_type.explanation_type(),
                    explanation_format=f5s.HtmlFormat.mime,
                    explanation_file=os.path.basename(img_src_path),
                )
            )
            with html_src.div():
                html_src.img(
                    src=img_format_file,
                    alt=SizeDependencyExplainer.__RESULT_TITLE,
                    # ensure that image will not overflow enclosing <div/>
                    style=(
                        "height: 100%; max-width: 100%; display: block; margin: auto;"
                    ),
                )
                html_src.br()

                # SKIPPED: chart data
                # html_src("Chart data:")
                # html_src.br()
                # with html_src.pre():
                #     html_src(f"{section['body']}")
                # html_src.br()

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
            explainer_id=SizeDependencyExplainer.explainer_id(),
            logger=self.logger,
        )
