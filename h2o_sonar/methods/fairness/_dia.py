# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import os
import traceback

import datatable
import numpy as np
from datatable import f
from datatable.lib._datatable import aggregate

from h2o_sonar import loggers
from h2o_sonar.lib.api import commons
from h2o_sonar.methods.fairness import _disparate_impact_analysis as _dia
from h2o_sonar.methods.utils.fairness_utils import get_r2_rmse


try:
    from sklearn.metrics import f1_score
    from sklearn.metrics import fbeta_score
    from sklearn.metrics import matthews_corrcoef

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


DIA_INTERPRETER_ID = "DIA"
DIA_ENTITY_FILE = "dia_entity.json"
DIA_ACTUALS_FILE = "actuals.jay"
DIA_PREDICTIONS_FILE = "predictions.jay"
DIA_ROW_CATEGORIES_FILE = "row_categories.jay"
DIA_CATEGORY_PARITY_FILE = "parity.jay"
DIA_CATEGORY_DISPARITY_FILE = "disparity.jay"
DIA_METRICS_FILE = "metrics.jay"
DIA_CATEGORY_CM_FILE = "cm.jay"
GROUPS = "Groups"
N = "N"
ME = "Marginal Error"
SMD = "Standardized Mean Difference"
DEFAULT_PROB_THRESHOLD = 0.5
DIA_CATEGORY_ME_SMD_FILE = "me_smd.jay"
COL_MODEL_PRED = "model_pred"


class DisparateImpactAnalysis:
    """Disparate impact analysis bean class."""

    def __init__(
        self,
        key,
        name,
        mli_key,
        path,
        problem_type,
        summary,
        feature_summaries,
        global_conf_matrix,
    ) -> None:
        self.key = key
        self.name = name
        self.mli_key = mli_key
        self.path = path
        self.problem_type = problem_type
        self.summary = summary
        self.feature_summaries = feature_summaries
        self.global_conf_matrix = global_conf_matrix

    def dump(self) -> dict:
        d = {k: v for k, v in vars(self).items()}
        d["summary"] = self.summary.dump()
        d["feature_summaries"] = [a.dump() for a in self.feature_summaries]
        d["global_conf_matrix"] = self.global_conf_matrix.dump()
        return d

    def clone(self) -> "DisparateImpactAnalysis":
        return DisparateImpactAnalysis(
            self.key,
            self.name,
            self.mli_key,
            self.path,
            self.problem_type,
            self.summary,
            self.feature_summaries,
            self.global_conf_matrix,
        )

    @staticmethod
    def load(d: dict) -> "DisparateImpactAnalysis":
        d["summary"] = DisparateImpactAnalysisSummary.load(d["summary"])
        d["feature_summaries"] = [
            DisparateImpactAnalysisFeatureSummary.load(a)
            for a in d["feature_summaries"]
        ]
        d["global_conf_matrix"] = DisparateImpactAnalysisNumericTable.load(
            d["global_conf_matrix"]
        )
        return DisparateImpactAnalysis(**d)


class DisparateImpactAnalysisSummary:
    """Disparate impact analysis (DIA) summary."""

    def __init__(self, max_metric: str = "", cut_off=-1, rmse=-1, r2=-1) -> None:
        """Create DIA summary.

        Parameters
        ----------
          max_metric :
            Binomial.
          cut_off :
            Binomial.
          rmse :
            Regression.
          r2 :
            Regression.

        """
        self.max_metric = max_metric
        self.cut_off = cut_off
        self.rmse = rmse
        self.r2 = r2

    def dump(self) -> dict:
        d = {k: v for k, v in vars(self).items()}
        return d

    def clone(self) -> "DisparateImpactAnalysisSummary":
        return DisparateImpactAnalysisSummary(
            self.max_metric, self.cut_off, self.rmse, self.r2
        )

    @staticmethod
    def load(d: dict) -> "DisparateImpactAnalysisSummary":
        return DisparateImpactAnalysisSummary(**d)


class DIA:
    def __init__(
        self,
        model,
        dataset,
        dataset_meta,
        dia_cols,
        classes,
        actual_col,
        predict_column,
        weight_col,
        labels=None,
        cut_off=None,
        maximize_metric=None,
        path="",
        parameters=None,
        dia_entity: DisparateImpactAnalysis | None = None,
    ):
        self.model = model
        self.dataset = dataset
        self.dataset_meta = dataset_meta
        self.dia_cols = dia_cols
        self.classes = classes
        self.labels = labels
        self.cut_off = cut_off
        self.actual_col = actual_col
        self.predict_column = predict_column
        self.maximize_metric = maximize_metric
        self.weight_col = weight_col
        self.path = path

        self.summary = None
        self.problem_type = "binomial"
        self.global_conf_matrix = None
        self.feature_summaries = []
        self.parameters = parameters

        self.dia_entity = dia_entity

    def calculate(
        self,
        max_cardinality=10,
        min_cardinality=2,
        max_numeric_cardinality=25,
    ):
        """
        Calculate DIA for a given model.

        Parameters
        ----------
        max_cardinality : int
            Maximum cardinality.
        min_cardinality : int
            Minimum cardinality.
        max_numeric_cardinality : int
            Maximum cardinality for numeric features.


        Returns
        -------
        str :
            DIA task key.

        """
        if not HAS_SKLEARN:
            commons.raise_opt_import_err("scikit-learn")

        predictions = None

        group_columns = DIA.prepare_dia_features(
            dataset=self.dataset,
            dataset_meta=self.dataset_meta,
            dia_cols=self.dia_cols,
            target_column=self.actual_col,
            predict_column="",
            max_cardinality=max_cardinality,
            min_cardinality=min_cardinality,
            max_numeric_cardinality=max_numeric_cardinality,
            model_meta=self.model.meta if self.model else None,
        )

        dataset = self.dataset[~datatable.isna(f[self.actual_col]), :]  # noqa: E711
        if self.predict_column in dataset.names:
            dataset = dataset[~datatable.isna(f[self.predict_column]), :]

        if not predictions:
            # TODO This should not be needed ...
            x_cols = list(dataset.names)
            target = dataset[:, self.actual_col].names[0]
            if target in x_cols:
                x_cols.remove(target)

            predictions = self.model.predict_datatable(dataset[:, x_cols])[:, -1]

        if self.predict_column not in predictions.names:
            predictions.names = [self.predict_column]
        dataset.cbind(predictions)
        self.predict_column = dataset.names[-1]

        if self.classes == 2:
            if self.cut_off == 0.0:
                (actual_col_np, y_pred) = self._int_label_predict_cols_encode(dataset)

                if self.labels:
                    pos_label = (
                        self.labels[1] if len(self.labels) == 2 else self.labels[0]
                    )
                else:
                    pos_label = 1  # _classification.*() default
                if self.maximize_metric == "F2":
                    self.cut_off = fbeta_score(
                        y_true=actual_col_np,
                        y_pred=y_pred,
                        sample_weight=(
                            dataset[self.weight_col].to_numpy()
                            if self.weight_col
                            else None
                        ),
                        labels=self.labels,
                        pos_label=pos_label,
                        beta=2,
                    )
                elif self.maximize_metric == "F05":
                    self.cut_off = fbeta_score(
                        y_true=actual_col_np,
                        y_pred=y_pred,
                        sample_weight=(
                            dataset[self.weight_col].to_numpy()
                            if self.weight_col
                            else None
                        ),
                        labels=self.labels,
                        pos_label=pos_label,
                        beta=0.5,
                    )
                elif self.maximize_metric == "MCC":
                    self.cut_off = matthews_corrcoef(
                        y_true=actual_col_np,
                        y_pred=y_pred,
                        sample_weight=(
                            dataset[self.weight_col].to_numpy()
                            if self.weight_col
                            else None
                        ),
                    )
                else:
                    # Default ...
                    self.maximize_metric = "F1"
                    self.cut_off = f1_score(
                        y_true=actual_col_np,
                        y_pred=y_pred,
                        sample_weight=(
                            dataset[self.weight_col].to_numpy()
                            if self.weight_col
                            else None
                        ),
                        labels=self.labels,
                        pos_label=pos_label,
                    )
        else:
            self.maximize_metric = "None (Regression)"
            self.cut_off = "None (Regression)"

        self.dia_entity.summary = self.summary = self._prep_summary(dataset)

        aggr_frame = datatable.Frame()
        if self.classes < 2:
            columns_ = [self.actual_col, self.predict_column] + [
                gc.name for gc in group_columns
            ]
            [aggr_frame, _] = aggregate(dataset[:, columns_], min_rows=500, seed=1234)

            aggr_frame[:, self.actual_col].to_jay(
                os.path.join(self.path, DIA_ACTUALS_FILE)
            )

            aggr_frame[:, self.predict_column].to_jay(
                os.path.join(self.path, DIA_PREDICTIONS_FILE)
            )

        if self.classes == 2:
            global_group_column = (
                group_columns[0].name if group_columns else dataset.names[-1]
            )
            self._prep_global_confusion_matrix(
                dia=_dia.BinaryDisparateImpactAnalysis(
                    actual_column=self.actual_col,
                    predict_column=self.predict_column,
                    group_column=global_group_column,
                    cutoff=self.cut_off,
                    labels=self.labels,
                    sample_weight=self.weight_col,
                ),
                dataset=dataset,
            )

        for group_column in group_columns:
            group_dir = make_tmp_dir(str(group_column.name), tmp_dir_base=self.path)

            if self.classes == 2:
                self.dia_entity.problem_type = self.problem_type = "binomial"
                dia = _dia.BinaryDisparateImpactAnalysis(
                    actual_column=self.actual_col,
                    predict_column=self.predict_column,
                    group_column=group_column.name,
                    cutoff=self.cut_off,
                    labels=self.labels,
                    sample_weight=self.weight_col,
                )
            else:
                self.dia_entity.problem_type = self.problem_type = "regression"
                dia = _dia.RegressionDisparateImpactAnalysis(
                    actual_column=self.actual_col,
                    predict_column=self.predict_column,
                    group_column=group_column.name,
                )

                aggr_frame[:, group_column.name].to_jay(
                    os.path.join(
                        self.path,
                        str(group_column.name),
                        DIA_ROW_CATEGORIES_FILE,
                    )
                )

            ref_levels = (
                datatable.unique(dataset[:, group_column.name])
                .sort(group_column.name)
                .to_list()[0]
            )

            for i in range(len(ref_levels)):
                make_tmp_dir(str(i), tmp_dir_base=group_dir)

            if self.classes == 2:
                metrics, confusion_matrices = dia.get_metrics(dataset, return_cm=True)
                self._prep_confusion_matrices(
                    group_column=group_column.name,
                    confusion_matrices=confusion_matrices,
                    ref_levels=ref_levels,
                )
            else:
                metrics = dia.get_metrics(dataset)

            metrics.to_jay(
                os.path.join(self.path, str(group_column.name), DIA_METRICS_FILE)
            )

            feature_summary = self._prep_feature_summaries(group_column, ref_levels)

            self._prep_disparity_parity(
                dia,
                metrics,
                ref_levels,
                group_column.name,
                dataset[:, [self.predict_column, group_column.name]],
            )

            self.feature_summaries.append(feature_summary)
            self.dia_entity.feature_summaries.append(feature_summary)

        return self

    def _int_label_predict_cols_encode(self, dataset: datatable.Frame):
        y_hat = dataset[self.predict_column]
        actual_col = dataset[self.actual_col]

        if y_hat.ltypes[0] in [datatable.ltype.str, datatable.ltype.bool]:
            label1 = 1.0
            label0 = 0.0
            y_hat_np = (
                y_hat.to_pandas()
                .replace(self.labels[1], 1)
                .replace(self.labels[0], 0)
                .to_numpy()
            )
            actual_col_np = (
                actual_col.to_pandas()
                .replace(self.labels[1], 1)
                .replace(self.labels[0], 0)
                .to_numpy()
            )
        else:
            label1 = self.labels[1]
            label0 = self.labels[0]
            y_hat_np = y_hat.to_numpy()
            actual_col_np = actual_col.to_numpy()

        y_pred = np.where(
            y_hat_np > DEFAULT_PROB_THRESHOLD,
            label1,
            label0,
        )

        return actual_col_np, y_pred

    def check_metric_in_cm_stats(
        self, ens_cm_stats_valid_max_metric, ens_cm_stats_valid_max_metric_value
    ):
        if (
            ens_cm_stats_valid_max_metric
            and ens_cm_stats_valid_max_metric_value is not None
            and ens_cm_stats_valid_max_metric_value >= 0.0
        ):
            if self.maximize_metric.lower() in ens_cm_stats_valid_max_metric:
                self.cut_off = ens_cm_stats_valid_max_metric_value
            else:
                self.cut_off = -1
        else:
            self.cut_off = -1

    @staticmethod
    def _guess_column_type(column_name: str, dataset_entity, dataset: datatable.Frame):
        if (
            dataset_entity
            and dataset_entity.column_names
            and column_name in dataset.names
            and dataset_entity.column_types
        ):
            return str(
                dataset_entity.column_types[
                    dataset_entity.column_names.index(column_name)
                ]
            )
        elif dataset and column_name in dataset.names:
            return str(dataset.ltypes[dataset.names.index(column_name)]).replace(
                "ltype.", ""
            )

        return None

    @staticmethod
    def prepare_dia_features(
        dataset,
        dataset_meta,
        dia_cols,
        target_column,
        predict_column,
        max_cardinality=10,
        min_cardinality=2,
        max_numeric_cardinality=25,  # TODO configuration
        drop_cols: list[str] | None = None,
        model_meta=None,  # models.ExplainableModelMeta
        logger=None,
    ):
        logger = logger or loggers.SonarPrintLogger()
        logger.debug(
            f"Preparing and checking DIA features ({dia_cols}):"
            f" dataset={dataset}"
            f" dataset_meta={dataset_meta}"
        )
        # columns choose by users
        user_selected = dia_cols or set()
        # drop columns
        drop_cols = drop_cols or []
        skip_cols = (
            [target_column, COL_MODEL_PRED] + drop_cols
            if drop_cols
            else [target_column, COL_MODEL_PRED]
        )
        if predict_column:
            skip_cols.append(predict_column)
        # IMPROVE: sanitization of skip_cols
        skip_cols = set(skip_cols)

        dataset_group_columns = set()
        invalid_categorical_columns = set()

        # if MODEL METADATA are available, then use them to determine
        #  categorical features and then complete checks using dataset columns meta
        if (
            model_meta
            and model_meta.features_metadata
            and model_meta.features_metadata.categorical_features
            and dataset_meta
            and dataset_meta.columns_meta
        ):
            logger.debug(
                f"Using MODEL METADATA to prepare DIA features:"
                f" meta={model_meta}"
                f" column_names={dataset_meta.column_names}"
                f" column_uniques={dataset_meta.column_uniques}"
            )
            model_group_columns = {
                feature
                for feature in model_meta.features_metadata.categorical_features
                if feature not in skip_cols
            }
            if model_group_columns:
                for column_meta in dataset_meta.columns_meta:
                    # IMPROVE sanitize column name
                    if (
                        column_meta.name not in skip_cols
                        and column_meta.name not in model_group_columns
                        and (not user_selected or column_meta.name in user_selected)
                    ):
                        num_classes = column_meta.unique
                        column_type = str(column_meta.data_type)
                        if (
                            any(x in column_type for x in ["int", "real", "bool"])
                            and max_numeric_cardinality
                            >= num_classes
                            >= min_cardinality
                            and (user_selected or num_classes <= max_cardinality)
                        ):
                            # do not run DIA for columns that do not meet the minimum
                            # cardinality requirement (default is >= 2)
                            dataset_group_columns.add(column_meta.name)
                        elif (
                            "str" in column_type
                            and num_classes >= min_cardinality
                            and (user_selected or num_classes <= max_cardinality)
                        ):
                            # do not run DIA for columns that do not meet the minimum
                            # cardinality requirement (default is >= 2)
                            dataset_group_columns.add(column_meta.name)
                        else:
                            invalid_categorical_columns[
                                f"Feature: '{column_meta.name}'"
                            ] = f"Cardinality: '{num_classes}'"

                        if column_meta.name in dataset_group_columns:
                            max_cardinality = (
                                max_numeric_cardinality
                                if column_meta.is_numeric
                                else max_cardinality
                            )
                            if num_classes < min_cardinality or (
                                column_meta.name not in user_selected
                                and num_classes > max_cardinality
                            ):
                                dataset_group_columns.remove(column_meta.name)

                if user_selected:
                    dataset_group_columns = dataset_group_columns.intersection(
                        user_selected
                    )

                if dataset_group_columns:
                    logger.debug(
                        f"DIA group columns prepared using MODEL METADATA:"
                        f" {dataset_group_columns}"
                    )
                    return dataset_group_columns
                # else: try to lookup categorical columns using other methods ...

        max_numeric_enum_cardinality = max_numeric_cardinality
        min_dia_cardinality = min_cardinality
        max_dia_cardinality = max_cardinality
        if dataset_meta and dataset_meta.column_names and dataset_meta.column_uniques:
            logger.debug(
                f"Using dataset ENTITY to prepare DIA features:"
                f" column_names={dataset_meta.column_names}"
                f" column_uniques={dataset_meta.column_uniques}"
            )
            for i in range(0, len(dataset_meta.column_names)):
                if (
                    any(x in str(dataset_meta.column_types[i]) for x in ["int", "real"])
                    and dataset_meta.column_uniques[i] <= max_numeric_enum_cardinality
                ):
                    # do not run DIA for columns that do not meet the minimum
                    # cardinality requirement (default is >= 2)
                    if dataset_meta.column_uniques[i] >= min_dia_cardinality:
                        dataset_group_columns.add(dataset_meta.column_names[i])
                elif (
                    "str" in str(dataset_meta.column_types[i])
                    and dataset_meta.column_uniques[i] <= max_dia_cardinality
                ):
                    # do not run DIA for columns that do not meet the minimum
                    # cardinality requirement (default is >= 2)
                    if dataset_meta.column_uniques[i] >= min_dia_cardinality:
                        dataset_group_columns.add(dataset_meta.column_names[i])
            logger.debug(
                f"DIA group columns prepared using dataset ENTITY:"
                f" {dataset_group_columns}"
            )
        elif dataset:
            logger.debug(
                f"Using DATASET to prepare DIA features: column_names={dataset.names}"
            )
            for i in range(0, len(dataset.names)):
                if (
                    any(x in str(dataset[:, i].ltypes) for x in ["int", "real"])
                    and dataset[:, i].nunique()[0, 0] <= max_numeric_enum_cardinality
                ):
                    # do not run DIA for columns that do not meet the minimum
                    # cardinality requirement (default is >= 2)
                    if dataset[:, i].nunique()[0, 0] >= min_dia_cardinality:
                        dataset_group_columns.add(dataset.names[i])
                elif (
                    "str" in str(dataset[:, i].ltypes)
                    and dataset[:, i].nunique()[0, 0] <= max_dia_cardinality
                ):
                    # do not run DIA for columns that do not meet the minimum
                    # cardinality requirement (default is >= 2)
                    if dataset[:, i].nunique()[0, 0] >= min_dia_cardinality:
                        dataset_group_columns.add(dataset.names[i])
            logger.debug(
                f"DIA group columns prepared using DATASET: {dataset_group_columns}"
            )

        logger.debug(f"DIA group columns to SKIP: {skip_cols}")
        dataset_group_columns.difference_update(skip_cols)
        # IMPROVE sanitize dia columns
        if dia_cols:
            user_selected = dia_cols
            dataset_group_columns = dataset_group_columns.intersection(user_selected)
        group_columns = [BoolEntry(e, True) for e in dataset_group_columns]
        logger.debug(f"DIA group columns as BOOLs: {group_columns}")
        if not group_columns:
            raise ValueError(
                "DIA cannot be ran - no categorical columns available. "
                "This is expected and not an error."
            )

        return group_columns

    def _prep_global_confusion_matrix(self, dia, dataset):
        global_cm = dia.get_confusion_matrix(dataset)
        global_cm_pandas = global_cm.to_pandas().fillna("NaN")

        labels = [x.split("actual")[1] for x in global_cm_pandas.columns]
        rows = ["Predicted " + str(x) for x in labels]
        columns = ["Actual " + str(x) for x in labels]
        matrix = DisparateImpactAnalysisNumericTable(
            "", columns, rows, global_cm_pandas.values.tolist()
        )
        self.dia_entity.global_conf_matrix = self.global_conf_matrix = matrix

    def _prep_confusion_matrices(self, group_column, confusion_matrices, ref_levels):
        # per reference level confusion matrix
        for idx, ref_level in enumerate(ref_levels):
            confusion_matrices[ref_level].to_jay(
                os.path.join(
                    self.path, str(group_column), str(idx), DIA_CATEGORY_CM_FILE
                )
            )

    def _prep_disparity_parity(
        self, dia, metrics, ref_levels, group_column, pred_group_frame
    ):
        for idx, ref_level in enumerate(ref_levels):
            disparity = dia.get_disparity(
                metrics,
                ref_level=ref_level,
                pred_group_frame=pred_group_frame,
                fetch_metrics=False,
            )

            if dia.problem_type == "binomial":
                me_smd = disparity[:, [GROUPS, N, ME, SMD]]
            else:
                me_smd = disparity[:, [GROUPS, N, SMD]]
            me_smd.to_jay(
                os.path.join(
                    self.path,
                    str(group_column),
                    str(idx),
                    DIA_CATEGORY_ME_SMD_FILE,
                )
            )

            parity = dia.get_parity(disparity, ref_level=ref_level, get_disparity=False)
            parity.to_jay(
                os.path.join(
                    self.path,
                    str(group_column),
                    str(idx),
                    DIA_CATEGORY_PARITY_FILE,
                )
            )

            disparity.to_jay(
                os.path.join(
                    self.path,
                    str(group_column),
                    str(idx),
                    DIA_CATEGORY_DISPARITY_FILE,
                )
            )

    @staticmethod
    def _prep_feature_summaries(group_column, ref_levels):
        return DisparateImpactAnalysisFeatureSummary(
            group_column, [str(ref_level) for ref_level in ref_levels]
        )

    def compute_parity(
        self,
        frame,
        group_column,
        ref_level,
        low_threshold=None,
        high_threshold=None,
        offset=0,
        count=10,
    ):
        if self.problem_type == "binomial":
            dia = _dia.BinaryDisparateImpactAnalysis(
                actual_column=self.actual_col,
                predict_column=self.predict_column,
                group_column=group_column,
                cutoff=self.cut_off,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                sample_weight=self.weight_col,
            )
        else:
            dia = _dia.RegressionDisparateImpactAnalysis(
                actual_column=self.actual_col,
                predict_column=self.predict_column,
                group_column=group_column,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
            )

        additional_parity_metrics = [
            "Marginal Error",
            "Standardized Mean Difference",
        ]
        if self.problem_type == "regression":
            additional_parity_metrics.remove("Marginal Error")
            additional_parity_frame = datatable.repeat(
                datatable.Frame(
                    [[0]] * len(additional_parity_metrics),
                    names=additional_parity_metrics,
                ),
                frame.shape[0],
            )
        else:
            additional_parity_frame = datatable.repeat(
                datatable.Frame(
                    [[0]] * len(additional_parity_metrics),
                    names=additional_parity_metrics,
                ),
                frame.shape[0],
            )
        # Check if additional metrics are not in input frame names
        if not set(additional_parity_metrics).issubset(set(list(frame.names))):
            # cbind() missing metrics, which are needed to avoid BE error for old
            # (1.8.*) experiments
            frame.cbind(additional_parity_frame)
            # Need to preserve column order as mli-2 does not check for subset of
            # expected column names. Rather, it compares lists, which is strict.
            groups_n_air = ["Groups", "N", "Adverse Impact Disparity"]
            old_col_names = list(frame.names)
            filtered_col_names = [
                x
                for x in old_col_names
                if x not in additional_parity_metrics + groups_n_air
            ]
            new_col_order = (
                groups_n_air + additional_parity_metrics + filtered_col_names
            )
            frame = frame[:, new_col_order]
        parity = dia.get_parity(frame, ref_level=ref_level, get_disparity=False)
        return parity[offset : offset + count, :]

    def _prep_summary(self, dataset):
        r2 = rmse = -1
        if self.classes == 1:
            r2_rmse = get_r2_rmse(
                dataset[:, [self.actual_col, self.predict_column]].to_pandas(),
                self.actual_col,
                self.predict_column,
            )

            r2 = r2_rmse["R2"] if not np.isnan(r2_rmse["R2"]) else ""
            rmse = r2_rmse["RMSE"] if not np.isnan(r2_rmse["RMSE"]) else ""

        return DisparateImpactAnalysisSummary(
            max_metric=self.maximize_metric,
            cut_off=self.cut_off,
            rmse=rmse,
            r2=r2,
        )

    def dump(self) -> dict:
        # d = {k: v for k, v in vars(self).items()}
        d = {}
        d["summary"] = self.summary.dump()
        d["feature_summaries"] = [a.dump() for a in self.feature_summaries]
        d["global_conf_matrix"] = (
            self.global_conf_matrix.dump() if self.global_conf_matrix else None
        )

        return d


def explain_dia(
    model,
    dataset,
    dataset_meta,
    classes,
    actual_col,
    predict_column,
    labels,
    path,
    cut_off,
    dia_cols,
    weight_col,
    maximize_metric,
    max_cardinality=10,
    min_cardinality=2,
    max_numeric_cardinality=25,
    parameters=None,
    dia_entity=None,
    logger=None,
):
    dia = DIA(
        model=model,
        dataset=dataset,
        dataset_meta=dataset_meta,
        classes=classes,
        actual_col=actual_col,
        predict_column=predict_column,
        labels=labels,
        cut_off=cut_off,
        maximize_metric=maximize_metric,
        path=path,
        dia_cols=dia_cols,
        weight_col=weight_col,
        parameters=parameters,
        dia_entity=dia_entity,
    )

    try:
        dia.calculate(
            max_cardinality=max_cardinality,
            min_cardinality=min_cardinality,
            max_numeric_cardinality=max_numeric_cardinality,
        )
        return dia.dia_entity
    except Exception as ex:
        logger.error(f"Unable to calculate DIA: {ex}\n{traceback.format_exc()}")
        raise ex


class DisparateImpactAnalysisFeatureSummary:
    def __init__(self, feature_name, ref_levels) -> None:
        self.feature_name = feature_name
        self.ref_levels = ref_levels

    def dump(self) -> dict:
        d = {k: v for k, v in vars(self).items()}
        d["feature_name"] = self.feature_name.dump()
        return d

    def clone(self) -> "DisparateImpactAnalysisFeatureSummary":
        return DisparateImpactAnalysisFeatureSummary(self.feature_name, self.ref_levels)

    @staticmethod
    def load(d: dict) -> "DisparateImpactAnalysisFeatureSummary":
        d["feature_name"] = BoolEntry.load(d["feature_name"])
        return DisparateImpactAnalysisFeatureSummary(**d)


class DisparateImpactAnalysisNumericTable:
    def __init__(
        self,
        name: str = "",
        col_names: list | None = None,
        row_names: list | None = None,
        values: list | None = None,
    ) -> None:
        self.name = name
        self.col_names = col_names or []
        self.row_names = row_names or []
        self.values = values or []

    def dump(self) -> dict:
        d = {k: v for k, v in vars(self).items()}
        return d

    def clone(self) -> "DisparateImpactAnalysisNumericTable":
        return DisparateImpactAnalysisNumericTable(
            self.name, self.col_names, self.row_names, self.values
        )

    @staticmethod
    def load(d: dict) -> "DisparateImpactAnalysisNumericTable":
        return DisparateImpactAnalysisNumericTable(**d)


class DiaSummary:
    """Disparate Impact Analysis Summary domain object."""

    def __init__(
        self, dia_features, mli_key, dia_key, problem_type, global_confusion_matrix
    ) -> None:
        self.dia_features = dia_features
        self.mli_key = mli_key
        self.dia_key = dia_key
        self.problem_type = problem_type
        self.global_confusion_matrix = global_confusion_matrix

    def dump(self) -> dict:
        d = {k: v for k, v in vars(self).items()}
        d["dia_features"] = [a.dump() for a in self.dia_features]
        d["global_confusion_matrix"] = self.global_confusion_matrix.dump()
        return d

    def clone(self) -> "DiaSummary":
        return DiaSummary(
            self.dia_features,
            self.mli_key,
            self.dia_key,
            self.problem_type,
            self.global_confusion_matrix,
        )

    @staticmethod
    def load(d: dict) -> "DiaSummary":
        d["dia_features"] = [BoolEntry.load(a) for a in d["dia_features"]]
        d["global_confusion_matrix"] = DiaMatrix.load(d["global_confusion_matrix"])
        return DiaSummary(**d)


class DiaAvp:
    """Disparate Impact Analysis AvP domain object."""

    def __init__(self, category_summary, metrics, avp) -> None:
        self.category_summary = category_summary
        self.metrics = metrics
        self.avp = avp

    def dump(self) -> dict:
        d = {k: v for k, v in vars(self).items()}
        d["category_summary"] = [a.dump() for a in self.category_summary]
        d["metrics"] = [a.dump() for a in self.metrics]
        d["avp"] = [a.dump() for a in self.avp]
        return d

    def clone(self) -> "DiaAvp":
        return DiaAvp(self.category_summary, self.metrics, self.avp)

    @staticmethod
    def load(d: dict) -> "DiaAvp":
        d["category_summary"] = [
            DiaCategorySummary.load(a) for a in d["category_summary"]
        ]
        d["metrics"] = [DiaMetric.load(a) for a in d["metrics"]]
        d["avp"] = [DiaAvpEntry.load(a) for a in d["avp"]]
        return DiaAvp(**d)


class DiaCategorySummary:
    def __init__(self, name, count, value) -> None:
        self.name = name
        self.count = count
        self.value = value

    def dump(self) -> dict:
        d = {k: v for k, v in vars(self).items()}
        return d

    def clone(self) -> "DiaCategorySummary":
        return DiaCategorySummary(self.name, self.count, self.value)

    @staticmethod
    def load(d: dict) -> "DiaCategorySummary":
        return DiaCategorySummary(**d)


class DiaMetric:
    def __init__(self, name, levels) -> None:
        self.name = name
        self.levels = levels

    def dump(self) -> dict:
        d = {k: v for k, v in vars(self).items()}
        d["levels"] = [a.dump() for a in self.levels]
        return d

    def clone(self) -> "DiaMetric":
        return DiaMetric(self.name, self.levels)

    @staticmethod
    def load(d: dict) -> "DiaMetric":
        d["levels"] = [FloatEntry.load(a) for a in d["levels"]]
        return DiaMetric(**d)


class DiaAvpEntry:
    def __init__(self, actual, predicted, category) -> None:
        self.actual = actual
        self.predicted = predicted
        self.category = category

    def dump(self) -> dict:
        d = {k: v for k, v in vars(self).items()}
        return d

    def clone(self) -> "DiaAvpEntry":
        return DiaAvpEntry(self.actual, self.predicted, self.category)

    @staticmethod
    def load(d: dict) -> "DiaAvpEntry":
        return DiaAvpEntry(**d)


class DiaFeatureSummary:
    def __init__(
        self, dia_experiment, maximized_metric, cut_off, rmse, r2, ref_levels
    ) -> None:
        self.dia_experiment = dia_experiment
        self.maximized_metric = maximized_metric
        self.cut_off = cut_off
        self.rmse = rmse
        self.r2 = r2
        self.ref_levels = ref_levels

    def dump(self) -> dict:
        d = {k: v for k, v in vars(self).items()}
        return d

    def clone(self) -> "DiaFeatureSummary":
        return DiaFeatureSummary(
            self.dia_experiment,
            self.maximized_metric,
            self.cut_off,
            self.rmse,
            self.r2,
            self.ref_levels,
        )

    @staticmethod
    def load(d: dict) -> "DiaFeatureSummary":
        return DiaFeatureSummary(**d)


class DiaMatrix:
    def __init__(self, matrix) -> None:
        self.matrix = matrix

    def dump(self) -> dict:
        d = {k: v for k, v in vars(self).items()}
        d["matrix"] = [a.dump() for a in self.matrix]
        return d

    def clone(self) -> "DiaMatrix":
        return DiaMatrix(self.matrix)

    @staticmethod
    def load(d: dict) -> "DiaMatrix":
        d["matrix"] = [DiaTableRow.load(a) for a in d["matrix"]]
        return DiaMatrix(**d)


class DiaTableRow:
    def __init__(self, values) -> None:
        self.values = values

    def dump(self) -> dict:
        d = {k: v for k, v in vars(self).items()}
        d["values"] = [a.dump() for a in self.values]
        return d

    def clone(self) -> "DiaTableRow":
        return DiaTableRow(self.values)

    @staticmethod
    def load(d: dict) -> "DiaTableRow":
        d["values"] = [DiaTableColumn.load(a) for a in d["values"]]
        return DiaTableRow(**d)


class DiaTableColumn:
    def __init__(self, col_name, col_value) -> None:
        self.col_name = col_name
        self.col_value = col_value

    def dump(self) -> dict:
        d = {k: v for k, v in vars(self).items()}
        return d

    def clone(self) -> "DiaTableColumn":
        return DiaTableColumn(self.col_name, self.col_value)

    @staticmethod
    def load(d: dict) -> "DiaTableColumn":
        return DiaTableColumn(**d)


class FloatEntry:
    """"""

    def __init__(self, name, value) -> None:
        self.name = name
        self.value = value

    def dump(self) -> dict:
        d = {k: v for k, v in vars(self).items()}
        return d

    def clone(self) -> "FloatEntry":
        return FloatEntry(self.name, self.value)

    @staticmethod
    def load(d: dict) -> "FloatEntry":
        return FloatEntry(**d)


class BoolEntry:
    def __init__(
        self,
        name,
        value,
    ) -> None:
        self.name = name
        self.value = value

    def dump(self) -> dict:
        d = {k: v for k, v in vars(self).items()}
        return d

    def clone(self) -> "BoolEntry":
        return BoolEntry(self.name, self.value)

    @staticmethod
    def load(d: dict) -> "BoolEntry":
        return BoolEntry(**d)


def make_tmp_dir(name: str, tmp_dir_base: str = None):  # -> str:
    tmp_dir = os.path.join(tmp_dir_base, name)
    if not os.path.exists(tmp_dir):
        try:
            makedirs(tmp_dir, exist_ok=True)
        except FileExistsError:
            pass
    return tmp_dir


# IMPROVE: reuse the method from file system persistence
def makedirs(path, exist_ok=True):
    """Avoid some inefficiency in os.makedirs().

    path :
      Directory to create.
    exist_ok : bool
      Fail if directory already exists (``False``), or ignore (``True``).

    """
    if os.path.isdir(path) and os.path.exists(path):
        assert exist_ok, f"Path '{path}' already exists"
        return path

    os.makedirs(path, exist_ok=exist_ok)
