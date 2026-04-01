# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import enum
import math
import os
from typing import Literal

import datatable
import matplotlib
import numpy
import pandas

from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api import plots
from h2o_sonar.methods.fairness import _dia as dia


try:
    import graphviz

    HAS_PKG_GRAPHVIZ = True
except ImportError:
    HAS_PKG_GRAPHVIZ = False


# constants
MISSING_VALUE = "missing"

# disabled to avoid race condition & side effects
# matplotlib.use('Agg')


class ResultValueError(ValueError):
    pass


def matplotlib_closing(show: bool):
    if show:
        # show the plot
        matplotlib.pyplot.show(block=False)

    # save memory
    matplotlib.pyplot.close()


def list_in_english(items: list[str], quote_item=True) -> str:
    items = items.copy()
    if quote_item:
        items = [f"'{i}'" for i in items]
    if len(items) > 1:
        items.insert(-1, "and")
        comma_separated = ", ".join(items)
        return comma_separated.replace(", and,", " and")
    elif len(items) == 1:
        return items[0]
    else:
        return ""


class FeatureImportanceResult(explainers.ExplainerResult):
    def __init__(
        self,
        persistence: persistences.ExplainerPersistence,
        explainer_id: str = "",
        chart_title: str = "Global Feature Importance",
        chart_x_axis: str = "feature",
        chart_y_axis: str = "importance",
        h2o_sonar_config=None,
        logger=None,
        explanation_format: type[f5s.ExplanationFormat] = f5s.GlobalFeatImpJSonFormat,
        explanation: type[e10s.Explanation] = e10s.GlobalFeatImpExplanation,
    ):
        explainers.ExplainerResult.__init__(
            self,
            persistence=persistence,
            explainer_id=explainer_id,
            explanation_format=explanation_format,
            explanation=explanation,
            h2o_sonar_config=h2o_sonar_config,
            logger=logger,
        )

        self.chart_title = chart_title
        self.chart_x_axis_name = chart_x_axis
        self.chart_y_axis_name = chart_y_axis

    def plot(
        self,
        *,
        clazz: str | None = None,
        file_path: str = "",
    ):
        data = self.data(clazz=clazz)
        # ensure visibility of long x/y-axis ticks
        matplotlib.rcParams.update({"figure.autolayout": True})
        kwargs = dict(
            x=self.chart_x_axis_name,
            y=self.chart_y_axis_name,
            color=commons.LookAndFeel.get_fg_color(self.config.look_and_feel),
            title=self.chart_title,
        )
        if file_path:
            data.to_pandas().plot.bar(**kwargs).get_figure().savefig(file_path)
            show_plot = False
        else:
            data.to_pandas().plot.bar(**kwargs)
            show_plot = True

        matplotlib_closing(show_plot)

    @classmethod
    def help(cls) -> dict[str, list[dict[str, str | bool]]]:
        methods_doc = cls._create_methods_help()
        method_doc = methods_doc["methods"]
        method_doc["data"] = cls._create_method_help()
        method_doc["data"]["parameters"].append(cls._clazz_param_doc())
        return methods_doc

    def data(
        self,
        *,
        clazz: str | None = None,
    ) -> datatable.Frame:
        data = self._raw_data(clazz=clazz)
        frame = datatable.Frame(data)
        del frame[f5s.ExplanationFormat.KEY_SCOPE]
        frame.names = [self.chart_x_axis_name, self.chart_y_axis_name]

        return frame

    def _raw_data(
        self,
        *,
        clazz: str | None = None,
    ) -> dict:
        idx_dict: dict = self.format.load_index_file(
            persistence=self.persistence,
            explanation_type=self.explanation.explanation_type(),
        )
        clazz = clazz or next(iter(idx_dict[self.format.KEY_FILES].keys()))
        feature_file: str = idx_dict[self.format.KEY_FILES].get(clazz, "")

        if not feature_file:
            raise ResultValueError(
                f"Invalid class: '{clazz}', available classes are: "
                f"{list(idx_dict[self.format.KEY_FILES].keys())}"
            )

        feature_file = self.persistence.get_explanation_file_path(
            self.explanation.explanation_type(), self.format.mime, feature_file
        )
        data: dict = self.persistence.store.load_json(feature_file)

        return data[self.format.KEY_DATA]


class DtResult(explainers.ExplainerResult):
    def __init__(
        self,
        persistence: persistences.ExplainerPersistence,
        explainer_id: str,
        explainer_name: str,
        h2o_sonar_config=None,
        highlight_highest_residual: bool = False,
        logger=None,
    ):
        explainers.ExplainerResult.__init__(
            self,
            persistence=persistence,
            explainer_id=explainer_id,
            explanation_format=f5s.GlobalDtJSonFormat,
            explanation=e10s.GlobalDtExplanation,
            h2o_sonar_config=h2o_sonar_config,
            logger=logger,
        )
        self.explainer_name = explainer_name
        self.explainer_id = explainer_id
        self.highlight_highest_residual = highlight_highest_residual

    def plot(self, *, clazz: str | None = None):
        if not HAS_PKG_GRAPHVIZ:
            commons.raise_opt_import_err("graphviz")

        tree_data = self._raw_data(clazz=clazz)
        tree_data_dict: dict = {}

        dot = graphviz.Graph(graph_attr=dict(ratio="fill", size="25"))
        non_leaf_keys: set[str] = set()
        highlighted_path_node_keys: set[str] = set()

        if self.highlight_highest_residual:
            for node in tree_data:
                tree_data_dict[node[self.format.KEY_KEY]] = node
                parent = node[self.format.KEY_PARENT]
                if parent:
                    # non-leafs
                    non_leaf_keys.add(node[self.format.KEY_PARENT])

            # get max value node
            max_value = None
            max_value_node = None
            for node in tree_data:
                if node[self.format.KEY_KEY] not in non_leaf_keys:
                    try:
                        value = float(node[self.format.KEY_NAME])
                        if max_value is None or max_value < value:
                            max_value = value
                            max_value_node = node
                    except Exception:
                        continue

            if max_value_node:
                # print highlighted path
                curr_node = max_value_node
                while curr_node:
                    curr_node_key = curr_node[self.format.KEY_KEY]

                    dot.node(
                        name=curr_node_key,
                        label=curr_node[self.format.KEY_NAME],
                        shape="box",
                        fillcolor="#fc5d68",
                        fontcolor=commons.LookAndFeel.COLOR_BLACK,
                        style="filled",
                    )
                    highlighted_path_node_keys.add(curr_node_key)

                    parent_node_key: str = curr_node[self.format.KEY_PARENT]
                    parent_node = tree_data_dict.get(parent_node_key)
                    if parent_node:
                        # connect nodes
                        dot.edge(
                            tail_name=curr_node[self.format.KEY_PARENT],
                            head_name=curr_node_key,
                            label=curr_node[self.format.KEY_EDGE_IN],
                            color=commons.LookAndFeel.COLOR_RED,
                        )
                    curr_node = parent_node

        # fill in all other nodes
        for node in tree_data:
            if node[self.format.KEY_KEY] in highlighted_path_node_keys:
                continue

            dot.node(
                name=node[self.format.KEY_KEY],
                label=node[self.format.KEY_NAME],
                shape="box",
                fillcolor=commons.LookAndFeel.get_fg_color(self.config.look_and_feel),
                fontcolor=commons.LookAndFeel.get_line_color(self.config.look_and_feel),
                style="filled",
            )
            parent = node[self.format.KEY_PARENT]
            if parent:
                # not a root node
                dot.edge(
                    tail_name=node[self.format.KEY_PARENT],
                    head_name=node[self.format.KEY_KEY],
                    label=node[self.format.KEY_EDGE_IN],
                )
        return dot

    def data(self):
        raise NotImplementedError(
            f"This method is not supported by {self.explainer_name}"
        )

    def _raw_data(self, *, clazz: str | None = None) -> list[dict[str, str]]:
        dt_idx_dict: dict = self.format.load_index_file(
            persistence=self.persistence,
            explanation_type=self.explanation.explanation_type(),
        )
        clazz_to_file_dict: dict[str, str] = dt_idx_dict[self.format.KEY_FILES]
        clazz = (
            clazz
            or dt_idx_dict.get(self.format.KEY_DEFAULT_CLASS)
            or next(iter(clazz_to_file_dict.keys()))
        )
        dt_file: str | None = clazz_to_file_dict.get(clazz)
        if not dt_file:
            raise ResultValueError(
                f"Surrogate decision tree: Invalid class '{clazz}', available "
                f"classes are: {list(clazz_to_file_dict.keys())}"
            )
        dt_file: str = self.persistence.get_explanation_file_path(
            self.explanation.explanation_type(), self.format.mime, dt_file
        )
        return self.persistence.store.load_json(dt_file).get(self.format.KEY_DATA)

    @classmethod
    def help(cls) -> dict[str, list[dict[str, str | bool]]]:
        methods_doc = cls._create_methods_help()
        method_doc = methods_doc["methods"]
        method_doc["data"] = cls._create_method_help()
        method_doc["data"]["parameters"].append(cls._clazz_param_doc())
        return methods_doc


class PdResult(explainers.ExplainerResult):
    def __init__(
        self,
        persistence: persistences.ExplainerPersistence,
        explainer_id: str,
        h2o_sonar_config=None,
        logger=None,
    ):
        explainers.ExplainerResult.__init__(
            self,
            persistence,
            explainer_id=explainer_id,
            explanation_format=f5s.PartialDependenceJSonFormat,
            explanation=e10s.PartialDependenceExplanation,
            h2o_sonar_config=h2o_sonar_config,
            logger=logger,
        )

    def plot(
        self,
        *,
        feature_name,
        clazz=None,
        override_feature_type: (
            Literal[
                f5s.ExplanationFormat.KEY_CATEGORICAL,
                f5s.ExplanationFormat.KEY_NUMERIC,
            ]
            | None
        ) = None,
        file_path: str = "",
        is_problematic: bool = False,
    ):
        (raw_data, feature_type, raw_data_hist) = self.__data(
            feature_name=feature_name, clazz=clazz
        )
        data = datatable.Frame(raw_data)
        feature_type = override_feature_type if override_feature_type else feature_type
        data = datatable.Frame(data)
        if feature_type == self.format.KEY_NUMERIC:
            plot_func = data.to_pandas().plot.line
        elif feature_type == self.format.KEY_CATEGORICAL:
            plot_func = data.to_pandas().plot.bar
        else:
            raise ResultValueError(
                f"Invalid feature type: '{feature_type}', valid feature types are: "
                f"'{self.format.KEY_CATEGORICAL}' and '{self.format.KEY_NUMERIC}'"
            )

        kwargs = dict(
            x="bin",
            y="pd",
            title=feature_name,
            color=(
                commons.LookAndFeel.get_line_color(self.config.look_and_feel)
                if not is_problematic
                else commons.LookAndFeel.COLOR_RED
            ),
        )

        if file_path:
            plot_func(**kwargs).get_figure().savefig(file_path)
            show_plot = False
        else:
            plot_func(**kwargs)
            show_plot = True

        matplotlib_closing(show_plot)

    @classmethod
    def help(cls) -> dict[str, list[dict[str, str | bool]]]:
        methods_doc = cls._create_methods_help()
        method_doc = methods_doc["methods"]
        method_doc["data"] = cls._create_method_help()
        method_doc["data"]["parameters"].extend(
            [
                cls._feature_name_param_doc(),
                cls._clazz_param_doc(),
            ]
        )
        return methods_doc

    def _raw_data(
        self, *, feature_name: str, clazz: str | None = None
    ) -> tuple[list, str, list]:
        return self.__data(feature_name=feature_name, clazz=clazz)

    def data(self, *, feature_name: str, clazz: str | None = None) -> datatable.Frame:
        data = datatable.Frame(
            self._raw_data(feature_name=feature_name, clazz=clazz)[0]
        )
        data_hist = datatable.Frame(
            self._raw_data(feature_name=feature_name, clazz=clazz)[2]
        )
        # Ensure bins between histogram frame match bins from non-histogram frame
        if data_hist:
            data_hist.names = ["bin", "frequency"]
            data.key = "bin"
            data_join = data_hist[:, :, datatable.join(data)]
            return data_join
        else:
            return data

    def __data(
        self,
        *,
        feature_name: str,
        clazz: str | None,
    ) -> tuple[list, str, list]:
        pd_feat: dict = self.format.load_index_file(
            persistence=self.persistence,
            explanation_type=self.explanation.explanation_type(),
        )
        features: dict = pd_feat[self.format.KEY_FEATURES]
        feature: dict = features.get(feature_name)
        if not feature:
            raise ResultValueError(
                f"Invalid feature_name: '{feature_name}', available feature names "
                f"are: {list_in_english(list(features.keys()))}"
            )
        feature_classes: dict[str, str] = feature[self.format.KEY_FILES]
        clazz = (
            clazz
            or pd_feat[self.format.KEY_DEFAULT_CLASS]
            or next(iter(feature_classes.keys()))
        )
        feature_type: str = feature[self.format.KEY_FEATURE_TYPE][0]
        feature_file: str | None = feature_classes.get(str(clazz))
        if not feature_file:
            raise ResultValueError(
                f"Invalid class: '{clazz}', available classes are: "
                f"{list_in_english(list(feature_classes.keys()))}"
            )
        feature_file = self.persistence.get_explanation_file_path(
            self.explanation.explanation_type(), self.format.mime, feature_file
        )
        data: dict = self.persistence.store.load_json(feature_file)
        # return datatable.Frame(data[self.format.KEY_DATA]), feature_type
        # Need to set "missing" bins to None to avoid type mismatches in a column

        self._set_missing_none(data, self.format.KEY_DATA, "bin")
        if (
            feature_type == self.format.KEY_NUMERIC
            and self.format.KEY_DATA_HISTOGRAM_NUM in data.keys()
        ):
            self._set_missing_none(data, self.format.KEY_DATA_HISTOGRAM_NUM, "x")
            return (
                data[self.format.KEY_DATA],
                feature_type,
                data[self.format.KEY_DATA_HISTOGRAM_NUM],
            )
        elif (
            feature_type == self.format.KEY_CATEGORICAL
            and self.format.KEY_DATA_HISTOGRAM_CAT in data.keys()
        ):
            self._set_missing_none(data, self.format.KEY_DATA_HISTOGRAM_CAT, "x")
            return (
                data[self.format.KEY_DATA],
                feature_type,
                data[self.format.KEY_DATA_HISTOGRAM_CAT],
            )
        else:
            return (
                data[self.format.KEY_DATA],
                feature_type,
                [],
            )

    def _set_missing_none(self, data, key_data, key):
        dict_list = data[key_data]
        for pd_dict in dict_list:
            if pd_dict[key] == MISSING_VALUE:
                pd_dict[key] = None
                break


class SummaryShapResult(explainers.ExplainerResult):
    def __init__(
        self,
        persistence: persistences.ExplainerPersistence,
        explainer_id: str,
        raw_contribs_idx_filename: str,
        h2o_sonar_config=None,
        logger=None,
    ):
        explainers.ExplainerResult.__init__(
            self,
            persistence=persistence,
            explainer_id=explainer_id,
            explanation_format=f5s.GlobalSummaryFeatImpJsonFormat,
            explanation=e10s.GlobalSummaryFeatImpExplanation,
            h2o_sonar_config=h2o_sonar_config,
            logger=logger,
        )
        self.raw_contribs_idx_filename = raw_contribs_idx_filename

    def plot(
        self,
        *,
        feature_names: list[str] | str | None = None,
        clazz: str | None = None,
    ):
        data = self.data(feature_names=feature_names, clazz=clazz)

        data_pd = data.to_pandas()
        # IMPROVE: max_features parameter to be honored
        plots.ScatterFeatImpPlot.plot(frame=data_pd, contributions=data_pd)
        matplotlib_closing(True)

    @classmethod
    def help(cls) -> dict[str, list[dict[str, str | bool]]]:
        methods_doc = cls._create_methods_help()
        method_doc = methods_doc["methods"]
        method_doc["data"] = cls._create_method_help()
        method_doc["data"]["parameters"].extend(
            [
                cls._create_method_parameter_help(
                    name="feature_names",
                    type_name=cls._format_parameter_types(str),
                    default="All features will be used",
                    doc="A list of feature names that should be included in the "
                    "SHAP summary.",
                ),
                cls._clazz_param_doc(),
            ]
        )
        return methods_doc

    def data(
        self,
        *,
        feature_names: list[str] | str | None = None,
        clazz: str | None = None,
    ) -> datatable.Frame:
        return self._raw_data(feature_names=feature_names, clazz=clazz)

    def _raw_data(
        self,
        *,
        feature_names: list[str] | str | None = None,
        clazz: str | None = None,
    ) -> datatable.Frame:
        feature_names: list[str] = feature_names or []
        feature_names = (
            [feature_names] if isinstance(feature_names, str) else feature_names
        )
        raw_shapley_contribs_path: str = self.persistence.get_explainer_working_file(
            self.raw_contribs_idx_filename
        )
        raw_shapley_contribs_idx_dict: dict[str, str] = (
            self.persistence.store.load_json(raw_shapley_contribs_path)
        )

        if clazz:
            contributions_file: str = raw_shapley_contribs_idx_dict[
                f5s.ExplanationFormat.KEY_FILES
            ].get(clazz)
        else:
            contributions_file: str = next(
                iter(
                    raw_shapley_contribs_idx_dict[
                        f5s.ExplanationFormat.KEY_FILES
                    ].values()
                )
            )
        if contributions_file:
            contributions: datatable.Frame = datatable.fread(
                self.persistence.get_explainer_working_file(contributions_file)
            )
            if feature_names:
                all_feature_names = set(contributions.names)
                feature_names_set = set(feature_names)
                if not feature_names_set.issubset(all_feature_names):
                    invalid_feature_names = list(feature_names_set - all_feature_names)
                    raise ResultValueError(
                        f"Invalid feature names: "
                        f"{list_in_english(invalid_feature_names)}, "
                        f"available feature names are: "
                        f"{list_in_english(contributions.names)}"
                    )
                feature_names = (
                    feature_names + [self.format.KEY_BIAS]
                    if self.format.KEY_BIAS in all_feature_names
                    and self.format.KEY_BIAS not in feature_names
                    else feature_names
                )
                return contributions[:, feature_names]
            else:
                return contributions
        raise ResultValueError(
            f"Error finding the Shapley contributions file for class: '{clazz}' in "
            f"explanations index: {raw_shapley_contribs_idx_dict}"
        )


class DiaResult(explainers.ExplainerResult):
    class DiaCategory(enum.Enum):
        DIA_METRICS = os.path.splitext(dia.DIA_METRICS_FILE)[0]
        DIA_CATEGORY_CM = os.path.splitext(dia.DIA_CATEGORY_CM_FILE)[0]
        DIA_CATEGORY_DISPARITY = os.path.splitext(dia.DIA_CATEGORY_DISPARITY_FILE)[0]
        DIA_CATEGORY_PARITY = os.path.splitext(dia.DIA_CATEGORY_PARITY_FILE)[0]
        DIA_CATEGORY_ME_SMD = os.path.splitext(dia.DIA_CATEGORY_ME_SMD_FILE)[0]

    class DiaEntryConstant:
        def __init__(
            self,
            dia_entity_file: str,
            param_feature_summaries: str,
            param_feature_name: str,
            param_name: str,
            param_features: str,
            ref_levels: str,
        ):
            self.dia_entity_file = dia_entity_file
            self.param_feature_summaries = param_feature_summaries
            self.param_feature_name = param_feature_name
            self.param_name = param_name
            self.param_features = param_features
            self.ref_levels = ref_levels

    def __init__(
        self,
        persistence: persistences.ExplainerPersistence,
        explainer_id: str,
        dia_entry_constants: "DiaResult.DiaEntryConstant",
        h2o_sonar_config=None,
        logger=None,
    ):
        explainers.ExplainerResult.__init__(
            self,
            persistence=persistence,
            explainer_id=explainer_id,
            explanation_format=f5s.DiaTextFormat,
            explanation=e10s.DiaExplanation,
            h2o_sonar_config=h2o_sonar_config,
            logger=logger,
        )
        self.dia_entry_constants = dia_entry_constants

    def plot(
        self,
        *,
        feature_name: str,
        metrics_of_interest: list[str] | str | None = None,
        file_path: str = "",
    ) -> list[str]:
        data = self.data(
            feature_name=feature_name,
            category=DiaResult.DiaCategory.DIA_METRICS,
        )

        files = []
        metrics_of_interest = metrics_of_interest or []
        metrics_of_interest: list[str] = (
            [metrics_of_interest]
            if isinstance(metrics_of_interest, str)
            else metrics_of_interest
        )
        data_pd: pandas.DataFrame = data.to_pandas()
        all_column_names: list[str] = list(data_pd.columns)
        group_column_name: str = all_column_names[0]
        column_names: list[str] = all_column_names[1:]
        if metrics_of_interest:
            column_names_set = set(column_names)
            metrics_of_interest_set = set(metrics_of_interest)
            invalid_metrics = list(metrics_of_interest_set - column_names_set)
            if invalid_metrics:
                raise ResultValueError(
                    f"Invalid metric of interest name: "
                    f"{list_in_english(invalid_metrics)}"
                    f", available metric of interest names are: "
                    f"{list_in_english(column_names)}"
                )
            column_names = metrics_of_interest
        discriminator = 1
        for column_name in column_names:
            column_data = data_pd[[group_column_name, column_name]]
            kwargs = dict(
                x=group_column_name,
                y=column_name,
                title=f"{feature_name} - "
                f"{DiaResult.DiaCategory.DIA_METRICS.value.capitalize()}",
                color=commons.LookAndFeel.get_fg_color(self.config.look_and_feel),
            )
            if file_path:
                if os.path.isfile(file_path):
                    safe_file_path = (
                        f"{file_path.replace('.png', '')}-{discriminator}.png"
                    )
                    discriminator += 1
                else:
                    safe_file_path = file_path
                files.append(safe_file_path)

                column_data.plot.bar(**kwargs).get_figure().savefig(safe_file_path)
                show_plot = False
            else:
                column_data.plot.bar(**kwargs)
                show_plot = True
            matplotlib_closing(show_plot)

        return files

    @classmethod
    def help(cls) -> dict[str, list[dict[str, str | bool]]]:
        list_of_cat: str = list_in_english(
            [
                cls.DiaCategory.DIA_METRICS.value,
                cls.DiaCategory.DIA_CATEGORY_CM.value,
                cls.DiaCategory.DIA_CATEGORY_ME_SMD.value,
                cls.DiaCategory.DIA_CATEGORY_DISPARITY.value,
                cls.DiaCategory.DIA_CATEGORY_PARITY.value,
            ]
        )
        methods_doc = cls._create_methods_help()
        method_doc = methods_doc["methods"]
        method_doc["data"] = cls._create_method_help()
        method_doc["data"]["parameters"].extend(
            [
                cls._feature_name_param_doc(),
                cls._create_method_parameter_help(
                    name="category",
                    type_name=cls._format_parameter_types(str),
                    doc="The category of data to be retrieve. This can be one of the "
                    f"following: {list_of_cat}",
                ),
                cls._create_method_parameter_help(
                    name="ref_level",
                    type_name=cls._format_parameter_types(str, int),
                    required=False,
                    default="The first reference level will be selected from the set "
                    "of available reference levels.",
                    doc="The reference levels for each categorical data",
                ),
            ]
        )
        return methods_doc

    def data(
        self,
        *,
        feature_name: str,
        category: DiaCategory | str,
        ref_level: int | str | None = None,
    ) -> datatable.Frame:
        return self._raw_data(
            feature_name=feature_name, category=category, ref_level=ref_level
        )

    def _raw_data(
        self,
        *,
        feature_name: str,
        category: DiaCategory | str,
        ref_level: int | str | None = None,
    ) -> datatable.Frame:
        if isinstance(category, str):
            try:
                category = DiaResult.DiaCategory(category)
            except ValueError:
                categories = [c.value for c in DiaResult.DiaCategory]
                raise ResultValueError(
                    f"Invalid category: '{category}', valid categories are: "
                    f"{list_in_english(categories)}"
                )
        elif not isinstance(category, DiaResult.DiaCategory):
            raise ResultValueError(
                f"Invalid category: {category}, valid categories are: "
                f"{list_in_english(list(DiaResult.DiaCategory), quote_item=False)}"
            )
        if category == DiaResult.DiaCategory.DIA_METRICS and ref_level is not None:
            raise ResultValueError("Reference level is not supported for metrics")

        dia_entity_path = self.persistence.get_explainer_working_file(
            self.dia_entry_constants.dia_entity_file
        )
        dia_entity: dict[str, list[dict]] = self.persistence.store.load_json(
            dia_entity_path
        )
        feature_summaries: list[dict] = dia_entity[
            self.dia_entry_constants.param_feature_summaries
        ]
        valid_feature_names: list[str] = []
        ref_level_idx: int = 0
        feat_path: str = ""
        for feature_summary in feature_summaries:
            name: str = feature_summary[self.dia_entry_constants.param_feature_name][
                self.dia_entry_constants.param_name
            ]
            valid_feature_names.append(name)
            if name == feature_name:
                feat_path: str = persistences.Persistence.make_key(
                    self.persistence.get_explainer_working_dir(),
                    feature_name,
                )
                if not self.persistence.store.is_dir(feat_path):
                    raise ResultValueError(
                        f"Feature directory {feat_path} does not exist"
                    )
                ref_levels: list[str] = feature_summary[
                    self.dia_entry_constants.ref_levels
                ]
                if ref_level is not None:
                    ref_level = str(ref_level)
                    if ref_level not in ref_levels:
                        raise ResultValueError(
                            f"Invalid reference level: '{ref_level}', "
                            f"available reference levels are: "
                            f"{list_in_english(ref_levels)}"
                        )
                    else:
                        ref_level_idx = ref_levels.index(ref_level)
                break
        if not feat_path:
            raise ResultValueError(
                f"Invalid feature name: '{feature_name}', available "
                f"feature names are: {list_in_english(valid_feature_names)}"
            )
        if category == DiaResult.DiaCategory.DIA_METRICS:
            data_path = feat_path
        else:
            data_path = persistences.Persistence.make_key(
                feat_path,
                str(ref_level_idx),
            )
        data_file: str = persistences.Persistence.make_key(
            data_path, f"{category.value}.jay"
        )
        return datatable.fread(data_file)

    def params(self) -> dict:
        params = explainers.ExplainerResult.params(self)

        params[self.dia_entry_constants.param_features] = (
            params[self.dia_entry_constants.param_features]
            if params.get(self.dia_entry_constants.param_features) is not None
            else None
        )
        dia_entity_path = self.persistence.get_explainer_working_file(
            self.dia_entry_constants.dia_entity_file
        )
        dia_entity: dict[str, dict] = self.persistence.store.load_json(dia_entity_path)
        if params[self.dia_entry_constants.param_features] is None:
            feature_summaries = dia_entity[
                self.dia_entry_constants.param_feature_summaries
            ]
            features_used: list[str] = []
            for i in range(len(feature_summaries)):
                features_used.append(
                    feature_summaries[i][self.dia_entry_constants.param_feature_name][
                        self.dia_entry_constants.param_name
                    ]
                )
            params[self.dia_entry_constants.param_features] = features_used
        return params


class Data3dResult(explainers.ExplainerResult):
    def __init__(
        self,
        persistence: persistences.ExplainerPersistence,
        explainer_id: str,
        h2o_sonar_config=None,
        logger=None,
    ):
        explainers.ExplainerResult.__init__(
            self,
            persistence=persistence,
            explainer_id=explainer_id,
            explanation_format=f5s.GlobalSummaryFeatImpJsonFormat,
            explanation=e10s.GlobalSummaryFeatImpExplanation,
            h2o_sonar_config=h2o_sonar_config,
            logger=logger,
        )

    @classmethod
    def help(cls) -> dict[str, list[dict[str, str | bool]]]:
        methods_doc = cls._create_methods_help()
        method_doc = methods_doc["methods"]
        method_doc["data"] = cls._create_method_help()
        method_doc["data"]["parameters"].append([cls._feature_name_param_doc()])
        return methods_doc

    def data(
        self,
        *,
        feature_names: str = "",
    ) -> dict:
        return self._raw_data(feature_names=feature_names)

    def _raw_data(
        self,
        *,
        feature_names: str = "",
    ) -> dict:
        idx_file_path = self.persistence.get_explanation_file_path(
            explanation_type=e10s.Global3dDataExplanation.explanation_type(),
            explanation_format=f5s.Global3dDataJSonFormat.mime,
            explanation_file=(
                f"{f5s.ExplanationFormat.FILE_PREFIX_EXPLANATION_IDX}json"
            ),
        )
        if not self.persistence.store.exists(idx_file_path):
            raise ResultValueError(f"Unable to find explanation file: {idx_file_path}")

        idx_dict = self.persistence.store.load_json(idx_file_path)

        if not feature_names:
            feature_names = next(iter(idx_dict[f5s.ExplanationFormat.KEY_FEATURES]))

        if feature_names not in idx_dict.get(f5s.ExplanationFormat.KEY_FEATURES, []):
            ft = list(idx_dict[f5s.ExplanationFormat.KEY_FEATURES].keys())
            raise ResultValueError(
                f"Unable to find data for feature tuple '{feature_names}' - valid "
                f"feature tuples are {list_in_english(ft)}"
            )

        data_file_name = list(
            idx_dict[f5s.ExplanationFormat.KEY_FEATURES][feature_names][
                f5s.ExplanationFormat.KEY_FILES
            ].values()
        )[0]
        file_path = self.persistence.get_explanation_file_path(
            explanation_type=e10s.Global3dDataExplanation.explanation_type(),
            explanation_format=f5s.Global3dDataJSonFormat.mime,
            explanation_file=data_file_name,
        )

        return self.persistence.store.load_json(file_path)

    def plot(
        self,
        *,
        feature_names: str = "",
        plot_type: str = plots.Data3dPlot.PLOT_TYPE_SURFACE,
        title: str = "",
    ):
        # TODO rewrite plot below to plots.Data3dPlot to reuse implementation from there

        data_pd = pandas.DataFrame(self.data(feature_names=feature_names))

        x_tic_labels = list(data_pd.columns)
        y_tic_labels = list(data_pd.index)
        z_tics = z_data = data_pd.to_numpy()

        # plot
        matplotlib.pyplot.figure(figsize=(8, 6), dpi=80)
        if plot_type == plots.Data3dPlot.PLOT_TYPE_HEATMAP:
            ax = matplotlib.pyplot.axes()
        else:
            ax = matplotlib.pyplot.axes(projection="3d")

        if title:
            ax.set_title(title)

        # set plot bins count:
        #   required by Matplotlib 3.4.3 for even axis bins distribution
        x_tics = Data3dResult._plot_labels_to_tics(x_tic_labels)
        y_tics = Data3dResult._plot_labels_to_tics(y_tic_labels)
        matplotlib.pyplot.locator_params(axis="x", nbins=len(x_tic_labels))
        matplotlib.pyplot.locator_params(axis="y", nbins=len(y_tic_labels))
        ax.set_xticklabels(x_tic_labels)
        ax.set_yticklabels(y_tic_labels)
        # axes legend
        ax.set_xlabel(x_tic_labels)
        ax.set_ylabel(y_tic_labels)
        # color map
        color_map = "autumn"

        x_data = numpy.array([x_tics for _ in range(z_data.shape[0])])
        y_data = numpy.array([y_tics for _ in range(z_data.shape[1])]).T

        if plot_type == plots.Data3dPlot.PLOT_TYPE_CONTOUR:
            ax.contour3D(x_data, y_data, z_data, 150, cmap=color_map)
        elif plot_type == plots.Data3dPlot.PLOT_TYPE_HEATMAP:
            matplotlib.pyplot.imshow(z_tics, cmap=color_map, aspect="auto")
            matplotlib.pyplot.colorbar()
        elif plot_type == plots.Data3dPlot.PLOT_TYPE_SURFACE:
            ax.plot_surface(x_data, y_data, z_data, cmap=color_map, edgecolor="black")
        else:
            raise RuntimeError(
                f"Unable to render unknown plot type '{plot_type}' - valid plot types "
                f"are {plots.Data3dPlot.PLOT_TYPES}"
            )
        matplotlib_closing(True)

    @staticmethod
    def _plot_patch_tic_labels_mpl343(tic_labels):
        if tic_labels:
            # :-/ Matplotlib 3.4.3 but workaround
            patched = tic_labels.copy()
            patched.insert(0, tic_labels[0])
            return patched

        return tic_labels

    @staticmethod
    def _plot_labels_to_tics(tic_labels):
        if tic_labels and not isinstance(tic_labels[0], (int, float)):
            return [i for i in range(len(tic_labels))]

        return tic_labels


class TemplateResult(explainers.ExplainerResult):
    def __init__(
        self,
        persistence: persistences.ExplainerPersistence,
        explainer_id: str,
        explainer_name: str,
        logger=None,
    ):
        explainers.ExplainerResult.__init__(
            self,
            persistence=persistence,
            explainer_id=explainer_id,
            h2o_sonar_config=None,
            explanation_format=None,
            explanation=None,
            logger=logger,
        )
        self.explainer_name = explainer_name

    def plot(self, **kwargs):
        raise NotImplementedError(
            f"This method is not supported by {self.explainer_name} explainer"
        )

    def data(self, **kwargs) -> datatable.Frame:
        raise NotImplementedError(
            f"This method is not supported by {self.explainer_name} explainer"
        )

    def _raw_data(self, **kwargs) -> datatable.Frame:
        raise NotImplementedError(
            f"This method is not supported by {self.explainer_name} explainer"
        )


class LeaderboardResult(explainers.ExplainerResult):
    """Make (heatmap-based, bool-based, ...) leaderboard evaluator result."""

    def __init__(
        self,
        persistence: persistences.ExplainerPersistence,
        explainer_id: str = "",
        chart_title: str = "Leaderboard",
        chart_x_axis: str = "metrics",
        chart_y_axis: str = "models",
        h2o_sonar_config=None,
        logger=None,
        explanation_format: type[
            f5s.ExplanationFormat
        ] = f5s.LlmHeatmapLeaderboardJSonFormat,
        explanation: type[e10s.Explanation] = e10s.LlmHeatmapLeaderboardExplanation,
    ):
        explainers.ExplainerResult.__init__(
            self,
            persistence=persistence,
            explainer_id=explainer_id,
            explanation_format=explanation_format,
            explanation=explanation,
            h2o_sonar_config=h2o_sonar_config,
            logger=logger,
        )

        self.chart_title = chart_title
        self.chart_x_axis_name = chart_x_axis
        self.chart_y_axis_name = chart_y_axis

    @staticmethod
    def _str_nan_inf_to_math(data: dict):
        if f5s.ExplanationFormat.KEY_DATA in data:
            for m_name in data[f5s.ExplanationFormat.KEY_DATA]:
                for metric_id in data[f5s.ExplanationFormat.KEY_DATA][m_name]:
                    value = data[f5s.ExplanationFormat.KEY_DATA][m_name][metric_id]
                    if commons.SafeJavaScript.NAN == value:
                        data[f5s.ExplanationFormat.KEY_DATA][m_name][metric_id] = (
                            math.nan
                        )
                    elif commons.SafeJavaScript.INF == value:
                        data[f5s.ExplanationFormat.KEY_DATA][m_name][metric_id] = (
                            math.inf
                        )
                    elif commons.SafeJavaScript.NEG_INF == value:
                        data[f5s.ExplanationFormat.KEY_DATA][m_name][metric_id] = float(
                            "-inf"
                        )

    def _raw_data(
        self,
        *,
        metric_id: str | None = None,
    ) -> dict:
        idx_dict: dict = self.format.load_index_file(
            persistence=self.persistence,
            explanation_type=self.explanation.explanation_type(),
        )

        metric_id = metric_id or e10s.AbcHeatmapExplanation.METRIC_ALL

        metric_file: str = idx_dict[self.format.KEY_FILES].get(metric_id, "")

        if not metric_file:
            raise ResultValueError(
                f"All metrics data file not available in the index file - available "
                f"metrics data files: {list(idx_dict[self.format.KEY_FILES].keys())}"
            )

        all_metric_file = self.persistence.get_explanation_file_path(
            self.explanation.explanation_type(), self.format.mime, metric_file
        )
        data: dict = self.persistence.store.load_json(all_metric_file)

        # leaderboard data files NaN/-inf/+inf values are replaced with strings to make
        # them JSON de/serializable in JavaScript - covert these values back to ensure
        # homogeneous data type columns
        LeaderboardResult._str_nan_inf_to_math(data)

        return data[self.format.KEY_DATA] if self.format.KEY_DATA in data else data

    def data(
        self,
        *,
        metric_id: str | None = None,
    ) -> dict:
        return self._raw_data(metric_id=metric_id)

    def plot(
        self,
        *,
        metric_id: str | None = None,
        file_path: str = "",
    ):
        data_dict = self.data(metric_id=metric_id)
        data_dt_dict = {}
        for model in data_dict:
            for metric_id in data_dict[model]:
                if metric_id not in data_dt_dict:
                    data_dt_dict[metric_id] = []
                data_dt_dict[metric_id].append(data_dict[model][metric_id])
        data_dt = datatable.Frame(data_dt_dict)

        # ensure visibility of long x/y-axis ticks
        matplotlib.rcParams.update({"figure.autolayout": True})
        # plot heatmap
        # IMPROVE: add X and Y axis labels
        matplotlib.pyplot.imshow(data_dt.to_numpy(), cmap="hot")
        matplotlib.pyplot.colorbar()

        if file_path:
            matplotlib.pyplot.savefig(file_path, dpi=80)
        else:
            matplotlib_closing(True)

    @classmethod
    def help(cls) -> dict[str, list[dict[str, str | bool]]]:
        methods_doc = cls._create_methods_help()
        method_doc = methods_doc["methods"]
        method_doc["data"] = cls._create_method_help()
        method_doc["data"]["parameters"].append(cls._clazz_param_doc())
        return methods_doc
