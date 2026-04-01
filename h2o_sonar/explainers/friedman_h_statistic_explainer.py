# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import traceback

import datatable
import matplotlib
from matplotlib import pyplot as plt

from h2o_sonar import errors
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api import results
from h2o_sonar.methods import _h_statistic
from h2o_sonar.methods import _pd
from h2o_sonar.methods.core import method
from h2o_sonar.utils import sanitization


try:
    import networkx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class FriedmanHStatisticExplainer(explainers.Explainer):
    """Friedman's H-statistic explainer.

    The explainer provides implementation of Friedman's H-statistic feature
    interactions. The statistic is 0 when there is no interaction at all and 1 if
    all the variance of the predict function is explained by a sum of the partial
    dependence functions.

    References
    ----------
    .. [1] Friedman H-statistic:
           https://christophm.github.io/interpretable-ml-book/interaction.html#theory-friedmans-h-statistic

    """

    # Driverless AI platform specific implementation notes:
    #
    # - fast approximation and MOJO enable/disable settings are removed in this
    #   portable implementation, however, these proprietary parameters will be
    #   injected by Driverless AI container: explainable model passed to the
    #   explainer will set the parameters to predict method by getting them
    #   from server and also explainer parameters
    #

    PARAM_FEATURES_NUMBER = "features_number"
    DEFAULT_FEATURES_NUMBER = 4
    PARAM_SAMPLE_SIZE = "sample_size"
    DEFAULT_SAMPLE_SIZE = 25000
    PARAM_GRID_RESOLUTION = "grid_resolution"
    DEFAULT_GRID_RESOLUTION = 3
    PARAM_FEATURES = "features"

    # explain_global() ~ PD aspect calculation parameter(s)
    UPDATE_PARAM_NUMCAT_OVERRIDE = "numcat_override"

    _display_name = "Friedman's H-statistic"
    _description = (
        "Friedman's H-statistic describes the amount of variance explained by the "
        "feature *pair*. It's expressed with a graph where most important original "
        "features are nodes and the interaction scores are edges."
        "\n"
        "When features interact with each other, then the influence of the features "
        "on the prediction does not have be additive, but more complex. For instance "
        "the contribution might be greater than the sum of contributions."
        "\n"
        "Friedman's H-statistic calculation is computationally intensive and "
        "typically requires long time to finish - calculation duration grows with the "
        "number of features and bins."
    )
    _iid = True
    _regression = True
    _binary = True
    _multiclass = False
    _global_explanation = True
    _local_explanation = False
    _explanation_types = [e10s.GlobalFeatImpExplanation]
    _optional_explanation_types = [
        e10s.ReportExplanation,
        e10s.GlobalHtmlFragmentExplanation,
    ]
    _keywords = [
        explainers.Explainer.KEYWORD_EXPLAINS_FEATURE_BEHAVIOR,
        explainers.Explainer.KEYWORD_H2O_SONAR,
        explainers.Explainer.KEYWORD_IS_SLOW,
    ]
    _parameters = [
        explainers.ExplainerParam(
            param_name=PARAM_FEATURES_NUMBER,
            description="Number of features for which to calculate H-Statistic.",
            param_type=commons.ExplainerParamType.int,
            default_value=DEFAULT_FEATURES_NUMBER,
            value_min=2,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_GRID_RESOLUTION,
            description=(
                "Observations per bin (number of equally spaced points used to "
                "create bins)."
            ),
            param_type=commons.ExplainerParamType.int,
            default_value=DEFAULT_GRID_RESOLUTION,
            value_min=1,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_FEATURES,
            description="Feature list - at least 2 features must be selected.",
            param_type=commons.ExplainerParamType.multilist,
            default_value=None,
            tags=[explainers.ExplainerParam.TAG_SRC_DATASET_COLUMN_NAMES],
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        explainers.ExplainerParam(
            param_name=PARAM_SAMPLE_SIZE,
            description="Sample size for Partial Dependence Plot",
            param_type=commons.ExplainerParamType.int,
            default_value=DEFAULT_SAMPLE_SIZE,
            src=explainers.ExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
    ]
    _requires_predict_method = False

    FILE_RAW_RESULT = "h_statistic.json"

    def __init__(self):
        explainers.Explainer.__init__(self)

        self.model_key: str | None = None
        self.actual_col: str | None = None
        self.labels: list | None = None
        self.features_meta: dict | None = None
        self.text_features: list = []
        self.config_overrides: str = ""

        self.log_name = "Friedman's H-statistic"

    def _resolve_params(self, explainer_params_as_str: str = ""):
        explainer_params = explainers.ExplainerArgs.json_str_to_dict(
            explainer_params_as_str or self.explainer_params_as_str
        )
        self.args = explainers.ExplainerArgs(self._parameters)
        self.args.resolve_params(explainer_params=explainer_params)

        # number of features specific resolution
        if (
            explainer_params
            and explainer_params.get(FriedmanHStatisticExplainer.PARAM_FEATURES_NUMBER)
            is not None
        ):
            max_features = int(
                explainer_params.get(FriedmanHStatisticExplainer.PARAM_FEATURES_NUMBER)
            )
        else:
            max_features = FriedmanHStatisticExplainer.DEFAULT_FEATURES_NUMBER
        self.args.args[FriedmanHStatisticExplainer.PARAM_FEATURES_NUMBER] = max_features

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        **explainer_params,
    ) -> bool:
        explainers.Explainer.check_compatibility(self, params, **explainer_params)

        if not HAS_NETWORKX:
            err_msg = self._check_compatibility_pckg_err_msg("networkx")
            self.logger.warning(err_msg)
            raise errors.ExplainerCompatibilityError(err_msg)

        # check: target column
        if not params.target_col:
            err_msg = (
                f"{self.log_name}: explainer requires target column and therefore it "
                f"is not compatible with explanation run with parameters: "
                f"{params.dump()}"
            )
            self.logger.warning(err_msg)
            raise errors.ExplainerCompatibilityError(err_msg)

        # check: not multinomial (could be handled by filter)
        if self.model and self.model.meta and self.model.meta.num_labels > 2:
            err_msg = (
                f"{self.log_name}: explainer does not support multinomial explanations"
            )
            self.logger.warning(err_msg)
            raise errors.ExplainerCompatibilityError(err_msg)

        # check: at least 2 compatible features
        try:
            (features, _) = self._resolve_features(explainer_params.get("X", None))
            if not features or len(features) < 2:
                return False
        except Exception as ex:
            err_msg = (
                f"{self.log_name} features compatibility check failed: {ex}\n"
                f"{traceback.format_exc()}"
            )
            self.logger.error(err_msg)
            # TODO TODO TODO raise errors.ExplainerCompatibilityError(err_msg)

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

        self.log_name = f"Friedman's H-statistic {self.mli_key}/{self.key}"
        self._resolve_params()

    def explain(self, X, y=None, explanations_types: list = None, **e_params) -> list:
        explanations = []
        if not self.check_required_modules(set("networkx")):
            report_global_explanation = e10s.ReportExplanation(
                explainer=self,
                display_name=f"{FriedmanHStatisticExplainer._display_name} report",
                display_category=e10s.Explanation.DISPLAY_CAT_DAI_MODEL,
            )

            # explanation: Report, representation: Markdown
            (
                report_path,
                images_path,
                raw_h_statistic_explanations,
            ) = self._create_md_report(X)

            report_global_explanation.add_format(
                f5s.MarkdownFormat(
                    explanation=report_global_explanation,
                    format_file=report_path,
                    extra_format_files=images_path,
                )
            )

            explanations.append(report_global_explanation)
        else:
            (
                _,
                _,
                raw_h_statistic_explanations,
            ) = self._create_md_report(X, False)

        # explanation: FI, representation: JSon, DT, CSV
        global_explanation = e10s.GlobalFeatImpExplanation(
            explainer=self,
            display_name=FriedmanHStatisticExplainer._display_name,
            display_category=e10s.Explanation.DISPLAY_CAT_DAI_MODEL,
        )

        json_dt_format = self._work_to_gom(
            h_statistic_explanations=raw_h_statistic_explanations,
            explanation=global_explanation,
        )
        global_explanation.add_format(explanation_format=json_dt_format)
        global_explanation.add_format(
            explanation_format=f5s.GlobalFeatImpJSonCsvFormat.from_json_datatable(
                json_dt_format
            )
        )
        global_explanation.add_format(
            explanation_format=f5s.GlobalFeatImpJSonFormat.from_json_datatable(
                json_dt_format
            )
        )
        explanations.append(global_explanation)

        # OPTIONAL explanation: HTML fragment
        if self.config and self.config.create_html_representations:
            try:
                display_name = FriedmanHStatisticExplainer._display_name
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

        return explanations

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

    def _resolve_features(self, dataset: datatable.Frame) -> tuple:
        model_features = (
            self.model.meta.used_features
            if self.model and self.model.meta and self.model.meta.used_features
            else []
        )
        user_features = (
            self.args.get(FriedmanHStatisticExplainer.PARAM_FEATURES)
            if self.args
            else None
        )

        self.logger.info(
            f"{self.log_name}: getting features list, importance and metadata..."
        )
        (
            features,
            features_metadata,
            skip_features,
        ) = self._resolve_model_features(self.model)
        if user_features:
            self.logger.info(
                f"{self.log_name}: user-specified features: {user_features}"
            )
            user_features = sanitization.sanitize_strings(strings=user_features)
            features = user_features.copy()
            for use_feature in user_features:
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
                numcat_override = (
                    self.args.get(
                        FriedmanHStatisticExplainer.UPDATE_PARAM_NUMCAT_OVERRIDE, ""
                    )
                    if self.args
                    else None
                )
                if numcat_override == f5s.PartialDependenceJSonFormat.FEATURE_TYPE_NUM:
                    raise errors.MliError(
                        f"{self.log_name} error: numerical binning cannot be forced "
                        f"for feature '{col}' as categorical feature values are not "
                        f"of numerical type",
                    )

                # ensure that input categorical variables are in features_metadata;
                # otherwise, those variables will enter the numeric binning logic of
                # PD and may lead to an exception if OOR > number of uniques or it
                # may lead to erroneous results due to PD not handling categoricals
                # properly
                if (
                    method.Method.KEY_CATEGORICAL_FEATURES not in features_metadata
                    or col
                    not in features_metadata[method.Method.KEY_CATEGORICAL_FEATURES]
                ) and (model_features and col not in model_features):
                    if method.Method.KEY_CATEGORICAL_FEATURES not in features_metadata:
                        features_metadata[method.Method.KEY_CATEGORICAL_FEATURES] = [
                            col
                        ]
                    else:
                        features_metadata[
                            method.Method.KEY_CATEGORICAL_FEATURES
                        ].append(col)

        if self.params.drop_cols:
            drop_features: list = sanitization.sanitize_strings(self.params.drop_cols)
            skip_features.extend(drop_features)

        if skip_features:
            features = [x for x in features if x not in skip_features]

        if not features or len(features) < 2:
            raise ValueError(
                f"{self.log_name} requires at least 2 valid feature, however, "
                f"all experiment features were discarded (insufficient cardinality, "
                f"dropped by experiment, dropped by user in experiment or MLI, ...). "
                f"Please check explainer log for more details."
            )

        # ensure unique values for a feature is > 1
        feat_uniq_vals = dataset[:, features].nunique().to_dict()
        for feature in features:
            if feat_uniq_vals[feature][0] == 1:
                self.logger.info(
                    f"{self.log_name} cannot be calculated for feature '{feature}' "
                    f"with just one unique value - it must have at least two"
                )
                features.remove(feature)
            if feat_uniq_vals[feature][0] == 0:
                self.logger.info(
                    f"{self.log_name}: removing {feature} from features list since "
                    f"it has no valid value",
                )
                features.remove(feature)

        if not features or len(features) < 2:
            raise ValueError(
                f"{self.log_name} requires at least 2 valid feature, however, "
                f"all experiment features were discarded (insufficient cardinality, "
                f"dropped by experiment, dropped by user in experiment or MLI, ...). "
                f"Please check explainer log for more details."
            )

        # use limit on the number of features for which to calc. PD (if specified)
        max_features = (
            self.args.args.get(FriedmanHStatisticExplainer.PARAM_FEATURES_NUMBER, 0)
            if self.args
            else 0
        )
        max_features = (
            max_features or FriedmanHStatisticExplainer.DEFAULT_FEATURES_NUMBER
        )
        if max_features and max_features < 2:
            raise ValueError(
                f"{self.log_name}: specified "
                f"{FriedmanHStatisticExplainer.PARAM_FEATURES_NUMBER} explainer "
                f"parameter must be at least 2 in order to perform the calculation"
            )
        # ensure maximum number of features is not exceeded
        if max_features:
            features = features[:max_features]

        self.logger.info(f"{self.log_name}: features used by model: {model_features}")
        self.logger.info(f"{self.log_name}: final features list: {features}")

        return features, features_metadata

    def _create_bins(self, features: list, features_metadata, dataset):
        if not features:
            return None

        grid_resolution = self.args.get(
            FriedmanHStatisticExplainer.PARAM_GRID_RESOLUTION
        )
        if grid_resolution == method.Method.DEFAULT_GRID_RESOLUTION:
            return None

        # build bins
        return _pd.PD.create_unique_bins(
            features=features,
            X=dataset.to_pandas(),
            features_meta=features_metadata,
            grid_resolution=grid_resolution,
        )

    def _create_md_report(
        self, dataset: datatable.Frame, generate_md: bool = True
    ) -> tuple[str, list[str], dict]:
        (features, features_metadata) = self._resolve_features(dataset)

        bins = self._create_bins(
            features=features,
            features_metadata=features_metadata,
            dataset=dataset,
        )

        self.logger.debug(
            f"{self.log_name}: run parameters:\n"
            f"grid resolution: "
            f"{self.args.get(FriedmanHStatisticExplainer.PARAM_GRID_RESOLUTION)}\n"
            f"features: {features}\n"
            f"bins: {bins}\n",
        )

        h_statistic_result = _h_statistic.HStatistic("H-statistic for pairs").explain(
            features,
            dataset,
            predict_method=self.model.predict_pandas,
            # consider explainer parameter w/ custom bins (might be too geeky)
            bins=bins,
        )

        # persist data prior normalization
        if h_statistic_result and h_statistic_result.explanations():
            self.logger.debug(
                f"{self.log_name} raw result:\n{h_statistic_result.explanations()}",
            )

        if generate_md:
            # save image
            img_file_name = "network-chart.png"
            work_img_path = self.persistence.get_explainer_working_file(img_file_name)
            FriedmanHStatisticExplainer._create_report_image(
                work_img_path, h_statistic_result, features
            )

            # MD
            report_md: str = MARKDOWN_TEMPLATE.format(image_url=img_file_name)
            try:
                report_md_listing = FriedmanHStatisticExplainer.normalize_to_md_list(
                    h_statistic_result.explanations()
                )
            except Exception as ex:
                self.logger.warning(
                    f"{self.log_name}: unable to stringify result as text: {ex}\n"
                    f"{traceback.format_exc()}",
                )
                report_md_listing = ""
            if report_md_listing:
                report_md = f"{report_md}\n{report_md_listing}"

            # save report
            report_path = self.persistence.get_explainer_working_file("report.md")
            with open(report_path, mode="w") as file:
                file.write(report_md)

            return report_path, [work_img_path], h_statistic_result.explanations()

        return "", [], h_statistic_result.explanations()

    @staticmethod
    def normalize_to_md_list(h_statistic_result: dict) -> str:
        h_statistic_list = FriedmanHStatisticExplainer.normalize_to_sorted_list(
            h_statistic_result=h_statistic_result, esc_char="`"
        )
        if h_statistic_list:
            md: str = "\nH-statistics for feature pairs:"
            for item in h_statistic_list:
                md = f"{md}\n* {item[0]}: **{item[1]}**"
            return f"{md}\n\n"

        return ""

    @staticmethod
    def normalize_to_sorted_list(
        h_statistic_result: dict,
        esc_char: str = "'",
    ) -> list:
        result = []
        if h_statistic_result:
            for item in h_statistic_result:
                result.append(
                    (
                        f"{esc_char}{item[0]}{esc_char} and "
                        f"{esc_char}{item[1]}{esc_char}",
                        h_statistic_result[item][method.Method.LABEL_REGRESSION],
                    )
                )
            result = sorted(result, key=lambda entry: entry[1], reverse=True)

        return result

    @staticmethod
    def _create_report_image(
        img_path: str,
        h_statistic_result: _h_statistic.HStatistic,
        features: list[str],
    ):
        graph = networkx.Graph()

        graph.add_nodes_from(features)
        exs = h_statistic_result.explanations()

        for i in exs:
            graph.add_edge(i[0], i[1], interaction=float("%0.4f" % exs[i]["p_0"]))

        pos = networkx.spring_layout(graph, weight="interaction")

        networkx.draw_networkx_edge_labels(graph, pos)
        networkx.draw_networkx_labels(graph, pos)
        networkx.draw(graph, pos)
        plt.savefig(img_path, dpi=300)

    def _work_to_gom(
        self,
        h_statistic_explanations: dict,
        explanation: e10s.GlobalFeatImpExplanation,
    ) -> f5s.GlobalFeatImpJSonDatatableFormat:
        if h_statistic_explanations:
            h_statistic_result = FriedmanHStatisticExplainer.normalize_to_sorted_list(
                h_statistic_explanations
            )
            features: list = [i[0] for i in h_statistic_result]
            importances: list = [i[1] for i in h_statistic_result]
            # TODO consider reuse of PD explainer pd_classes code
            num_classes: int = (
                1 if self.model_meta.num_labels == 0 else self.model_meta.num_labels
            )
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
                raise ValueError(
                    f"{self.log_name} unable to to resolve model classes: "
                    f"'{self.model_meta.labels}'"
                )

            json_datatable_format = f5s.GlobalFeatImpJSonDatatableFormat
            (
                idx_dict,
                idx_str,
            ) = json_datatable_format.serialize_index_file(
                classes=classes,
                doc=FriedmanHStatisticExplainer._description,
            )
            json_representation = json_datatable_format(
                explanation=explanation, json_data=idx_str
            )

            if num_classes <= 2:
                # regression and binomial: feature importance is identical
                gom_data_frame: datatable.Frame = datatable.Frame(
                    {
                        json_datatable_format.COL_NAME: features,
                        json_datatable_format.COL_IMPORTANCE: importances,
                        json_datatable_format.COL_GLOBAL_SCOPE: [True] * len(features),
                    }
                )
                total_rows: int = gom_data_frame.shape[0]

                for clazz in classes:
                    file_name: str = idx_dict[json_datatable_format.KEY_FILES][clazz]
                    json_representation.add_data_frame(
                        format_data=gom_data_frame, file_name=file_name
                    )

                json_representation.update_index_file(idx_dict, total_rows=total_rows)

                return json_representation
            else:
                # multinomial
                raise NotImplementedError(
                    f"{self.log_name} for multinomial experiments is not implemented"
                )
        else:
            raise ValueError(
                f"{self.log_name} is required to build its JSon representation"
            )

    class Result(explainers.ExplainerResult):
        COL_FEATURE = "feature"
        COL_INTERACTIONS = "interactions"

        def __init__(
            self,
            persistence: persistences.ExplainerPersistence,
            explainer_id: str = "",
            h2o_sonar_config=None,
        ):
            self.explanation = e10s.GlobalFeatImpExplanation
            self.format = f5s.GlobalFeatImpJSonFormat
            self.explainer_id = explainer_id
            self.config = h2o_sonar_config

            explainers.ExplainerResult.__init__(
                self,
                persistence=persistence,
                explainer_id=explainer_id,
                explanation_format=self.format,
                explanation=self.explanation,
                h2o_sonar_config=h2o_sonar_config,
            )

            self.log_name = "Friedman's H-statistic result"

        def plot(
            self,
            *,
            clazz=None,
            file_path: str = "",
        ):
            data = self.data(clazz=clazz)

            # ensure visibility of long x/y-axis ticks
            matplotlib.rcParams.update({"figure.autolayout": True})
            if file_path:
                data.to_pandas().plot.bar(
                    x=FriedmanHStatisticExplainer.Result.COL_FEATURE,
                    y=FriedmanHStatisticExplainer.Result.COL_INTERACTIONS,
                    color=commons.LookAndFeel.get_fg_color(self.config.look_and_feel),
                    title="Global Features Interactions",
                ).get_figure().savefig(file_path)
            else:
                data.to_pandas().plot.bar(
                    x=FriedmanHStatisticExplainer.Result.COL_FEATURE,
                    y=FriedmanHStatisticExplainer.Result.COL_INTERACTIONS,
                    color=commons.LookAndFeel.get_fg_color(self.config.look_and_feel),
                    title="Global Features Interactions",
                )

        def data(self, *, clazz: str | None = None) -> datatable.Frame:
            idx_dict: dict = self.format.load_index_file(
                persistence=self.persistence,
                explanation_type=self.explanation.explanation_type(),
            )
            clazz = clazz or next(iter(idx_dict[self.format.KEY_FILES].keys()))
            feature_file: str = idx_dict[self.format.KEY_FILES].get(clazz, "")
            if not feature_file:
                raise ValueError(
                    f"{self.log_name}Invalid class: {clazz}, available classes are: "
                    f"{list(idx_dict[self.format.KEY_FILES].keys())}"
                )
            feature_file = self.persistence.get_explanation_file_path(
                self.explanation.explanation_type(), self.format.mime, feature_file
            )
            data: dict = self.persistence.store.load_json(feature_file)

            frame = datatable.Frame(data[self.format.KEY_DATA])
            del frame[f5s.ExplanationFormat.KEY_SCOPE]
            frame.names = [
                FriedmanHStatisticExplainer.Result.COL_FEATURE,
                FriedmanHStatisticExplainer.Result.COL_INTERACTIONS,
            ]

            return frame

        def summary(self, **kwargs) -> dict:
            return self.params()

    def get_result(self) -> results.FeatureImportanceResult:
        return results.FeatureImportanceResult(
            persistence=self.persistence,
            explainer_id=FriedmanHStatisticExplainer.explainer_id(),
            chart_title="Global Features Interactions",
            chart_x_axis="feature",
            chart_y_axis="interactions",
            h2o_sonar_config=self.config,
            logger=self.logger,
        )


#
# Markdown report
#


MARKDOWN_TEMPLATE: str = """# Friedman's H-statistic Report
![image](./{image_url})

The diagram shows the interaction of the variables - the closer the variables are to
each other, the greater the interaction.

"""
