# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import datatable as dt
import numpy as np
from datatable import by
from datatable import f
from datatable import mean
from datatable import sd

from h2o_sonar.lib.api import commons
from h2o_sonar.methods.fairness._abstract_disparate_impact_analysis import (
    AbstractDisparateImpactAnalysis,
)
from h2o_sonar.methods.utils.fairness_utils import check_cm_input
from h2o_sonar.methods.utils.fairness_utils import check_frame
from h2o_sonar.methods.utils.fairness_utils import cm_exp_parser
from h2o_sonar.methods.utils.fairness_utils import get_binary_metric_dict
from h2o_sonar.methods.utils.fairness_utils import get_group_levels
from h2o_sonar.methods.utils.fairness_utils import get_r2_rmse


try:
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import LabelEncoder

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class BinaryDisparateImpactAnalysis(AbstractDisparateImpactAnalysis):
    def __init__(
        self,
        actual_column=None,
        predict_column=None,
        group_column=None,
        high_threshold=1.25,
        low_threshold=0.8,
        cutoff=None,
        labels=None,
        sample_weight=None,
    ):
        """Implementation of Disparate Impact Analysis (DIA) for binary
        classification.

        Disparate Impact Analysis (DIA) is a practical way to discuss and handle
        observational fairness or simply put, how your model predictions affect
        different groups. DIA is far from perfect, as it relies heavily on
        user-defined thresholds and reference levels for disparity and does not
        attempt to remediate disparity or provide information on sources of
        disparity, but it is a fairly straightforward method to quantify your
        model’s behavior across sensitive demographic segments or other
        potentially interesting groups of observations. DIA is also an accepted,
        regulation-compliant tool for fair-lending purposes in the U.S.
        financial services industry.

        Parameters
        ----------
        actual_column: str
            Column that contains the true value for the outcome of interest
        predict_column: str
            Column that contains the predicted probabilities for the outcome of
            interest.
        group_column: str
            Column that contains certain groups of interest for DIA, e.g.,
            {female, male, other}, {high school, college, graduate school,
            other}
        high_threshold: float
            Allowed upper bound for disparity. Default is 1.25.
        low_threshold: float
            Allowed lower bound for disparity. Default is 0.8.
        cutoff: float
            Numeric cutoff which act's as a decision boundary for the outcome
            of interest, e.g., above the cutoff we say a customer will default
            and below we say they will not default.
        labels: list[int or str]
            Unique outcomes for target. Should be a list of length 2.
        sample_weight: str
            Sample weights

        """
        super().__init__(
            actual_col=actual_column,
            predict_col=predict_column,
            group_col=group_column,
            high_thresh=high_threshold,
            low_thresh=low_threshold,
            cutoff=cutoff,
            problem_type="binomial",
        )

        if labels is None:
            labels = [0, 1]

        self.labels = labels
        self.sample_weight = sample_weight

    def get_parity(self, frame, ref_level=None, get_disparity=True):
        """
        A binary indication of parity for metrics is reported by simply checking
        whether disparity values are within the user-defined thresholds. Further
        parity indicators are defined as combinations of other disparity values:

            - Type I Parity: Fairness in both FDR Parity and FPR Parity
            - Type II Parity: Fairness in both FOR Parity and FNR Parity
            - Equalized Odds: Fairness in both FPR Parity and TPR Parity
            - Supervised Fairness: Fairness in both Type I and Type II Parity
            - Overall Fairness: Fairness across all parities for all metrics

        Parameters
        ----------
        frame: datatable.Frame
            Datatable for DIA which should either be the original frame with an
            actual, predict, and group column or a pre-calculated disparity
            frame.
        ref_level
            Reference group level used for disparity calculation.
        get_disparity: bool
            Indicate if disparity needs to be calculated before calculating
            parity.

        Returns
        -------
        par_frame: datatable.Frame
            A frame in which the first column contains each group level and each
            column after is a boolean indicator of parity across all disparity
            metrics + additional parity metrics:
                - Type I Parity: Fairness in both FDR Parity and FPR Parity
                - Type II Parity: Fairness in both FOR Parity and FNR Parity
                - Equalized Odds: Fairness in both FPR Parity and TPR Parity
                - Supervised Fairness: Fairness in both Type I and Type II
                  Parity
                - Overall Fairness: Fairness across all parities for all metrics
        """
        par_frame, par_levels = super().get_parity(
            frame=frame, ref_level=ref_level, get_disparity=get_disparity
        )
        return self.__get_overall_parity_checks(par_frame, par_levels)

    def get_metrics(self, frame, return_cm=False):
        """
        Calculate metrics of interest, which are further used for disparity
        and parity analysis:

            Overall performance:
                "Accuracy": "(tp + tn) / (tp + tn +fp + fn)"
                Example: How often the model predicts default and non-default
                correctly for this group.

            Predicting outcome of interest will happen (correctly):
                "True Positive Rate": "tp / (tp + fn)"
                Example: Out of the people in the group *that did* default,
                how many the model predicted *correctly* would default.

            "Precision": "tp / (tp + fp)":
                Example:  Out of the people in the group the model *predicted*
                would default, how many the model predicted *correctly* would
                default.

            "Specificity": "tn / (tn + fp)":
                Predicting outcome of interest won't happen (correctly)
                Example: Out of the people in the group *that did not* default,
                 how many the model predicted *correctly* would not default.

            "Negative Predicted Value": "tn / (tn + fn)":
                Example: Out of the people in the group the model *predicted*
                would not default, how many the model predicted *correctly*
                would not default.

            Analyzing errors:
                Type I:
                    False Accusations:
                        "False Positive Rate": "fp / (tn + fp)":
                            Example: Out of the people in the group *that did
                            not* default, how many the model predicted
                            *incorrectly* would default.

                        "False Discovery Rate": "fp / (tp + fp)":
                            Example: Out of the people in the group the model
                            *predicted* would default, how many the model
                            predicted *incorrectly* would default.

                Type II:
                    Costly omissions:
                        "False Negative Rate": "fn / (tp + fn)":
                            Example: Out of the people in the group *that did*
                            default, how many the model predicted *incorrectly*
                            would not default.

                        "False Omissions Rate": "fn / (tn + fn)":
                            Example: Out of the people in the group the model
                            *predicted* would not default, how many the model
                            predicted *incorrectly* would not default.

        Parameters
        ----------
        frame: datatable.Frame
            Datatable for DIA which should either be the original frame with an
            actual, predict, and group column or a pre-calculated disparity
            frame.

        return_cm: bool
            Whether to return the computed confusion matrices for each level

        Returns
        -------
        metrics_frame: datatable.Frame
            A frame in which the first column contains each group level and each
            column after contains all metrics of interest.

        confusion_matrices: dict
            A dictionary of confusion matrices containing pairs level->cm

        """
        check_frame(self.actual_column, self.predict_column, self.group_column, frame)
        group_levels = get_group_levels(self.group_column, frame)

        # initialize dict of confusion matrices and corresponding rows of
        # datatable
        cm_dict = dict.fromkeys(group_levels)
        for level in group_levels:
            cm_dict[level] = self.get_confusion_matrix(frame, level=level)
        metrics = dt.Frame(
            names=self.metrics, stypes=[dt.stype.float32] * len(self.metrics)
        )
        metrics.nrows = len(group_levels)
        groups_counts = frame[:, dt.count(), dt.by(self.group_column)]
        groups_counts.names = [
            AbstractDisparateImpactAnalysis.GROUPS_COL_NAME,
            AbstractDisparateImpactAnalysis.GROUP_COUNT_NAME,
        ]
        for i, level in enumerate(group_levels):
            for j, metric in enumerate(get_binary_metric_dict().keys()):
                expression = cm_exp_parser(
                    get_binary_metric_dict()[metric], cm_dict, level
                )

                try:
                    metrics[i, j] = eval(expression)
                except ZeroDivisionError:
                    metrics[i, j] = float("nan")

        metrics_frame = dt.cbind(groups_counts, metrics)

        if return_cm:
            return metrics_frame, cm_dict

        return metrics_frame

    def get_confusion_matrix(
        self,
        frame,
        level=None,
        print_frame=False,
        get_global_cm=False,
        convert_to_pandas=False,
        predict_column_precision=np.float32,
    ):
        """Calculate the confusion matrix for a particular group level of interest.

        Parameters
        ----------
        frame: datatable.Frame
            Datatable with an actual, predict, and group column.
        level
            Variable to slice frame before creating confusion matrix.
        print_frame: bool
            Boolean if output frame should be printed to the console or not.
        get_global_cm: bool
            Boolean if the global confusion matrix should be calculated.
        convert_to_pandas: bool
            Boolean if confusion matrix should be converted from datatable.Frame
            to pandas.DataFrame.
        predict_column_precision: numpy.type
            Data precision type of predicted column.

        Returns
        -------
        cm_frame: datatable.Frame or pandas.DataFrame
            A frame that contains all relevant information of a confusion
            matrix.

        """
        if not HAS_SKLEARN:
            commons.raise_opt_import_err("scikit-learn")

        check_frame(self.actual_column, self.predict_column, self.group_column, frame)
        group_levels = get_group_levels(self.group_column, frame)

        check_cm_input(
            get_global_cm, group_levels, level, print_frame, self.group_column
        )

        level_list = dt.unique(frame[:, self.actual_column]).to_list()[0]
        if level_list:
            if len(level_list) > 2:
                raise ValueError(
                    "Actuals column has more than 2 unique values for binomial DIA: "
                    f"{level_list}"
                )
            if len(level_list) == 1:
                raise ValueError(
                    "Actuals column has only 1 unique values for binomial DIA: "
                    f"{level_list}"
                )

        level_list.sort(reverse=True)
        cm_frame = dt.Frame(
            {
                "actual" + str(level_list[0]): [np.nan],
                "actual" + str(level_list[1]): [np.nan],
            }
        )
        cm_frame.rbind(cm_frame)

        # avoid manipulating input frame and only use columns that are
        # necessary for DIA
        if not self.sample_weight:
            frame_ = frame[
                :,
                [
                    f[self.group_column],
                    f[self.actual_column],
                    f[self.predict_column],
                ],
            ]
        else:
            frame_ = frame[
                :,
                [
                    f[self.group_column],
                    f[self.actual_column],
                    f[self.predict_column],
                    f[self.sample_weight],
                ],
            ]

        if self.group_column and level:
            frame_ = frame_[f[self.group_column] == level, :]

        encoder = LabelEncoder()
        enc_label = list(encoder.fit_transform(level_list))
        # using numpy to compare with cutoff, be consistent with driverless
        y_true = encoder.transform(
            frame_[:, self.actual_column].to_numpy(),
        )
        y_pred = (
            frame_[:, self.predict_column].to_numpy().astype(predict_column_precision)
        )
        # encoded predictions with same type of actual
        level_dtype = np.array(level_list).dtype
        enc_pred = np.empty(y_pred.shape, dtype=level_dtype)
        mask = y_pred >= self.cutoff
        enc_pred[mask] = level_list[0]
        enc_pred[~mask] = level_list[1]

        dname = "d_" + str(self.actual_column)
        frame_[:, dname] = enc_pred
        labels_type = dt.Frame(level_list).stypes[0]
        # in case labels[0] will create a column with a too
        # "narrow" type for labels[1] i.e. [1,2]
        # would create bool8 because of 1...
        frame_[:, dname] = frame_[:, labels_type(f[dname])]

        # calculate size of each confusion matrix value
        for i, lev_i in enumerate(level_list):
            for j, lev_j in enumerate(level_list):
                cm_frame[j, i] = frame_[
                    (f[self.actual_column] == lev_i) & (f[dname] == lev_j), :
                ].shape[0]

        if self.sample_weight:
            y_pred = np.where(y_pred >= self.cutoff, 1, 0)
            sample_weights = frame_[:, self.sample_weight].to_numpy().ravel()
            cm = dt.Frame(
                confusion_matrix(
                    y_true,
                    y_pred,
                    labels=enc_label,
                    sample_weight=sample_weights,
                )
            )

            # reset frame as sklearn CM is a different setup
            #
            # h2o_sonar-2 CM:
            # --- ---
            # TP | FP
            # -------
            # FN | TN
            #
            # sklearn CM:
            # --- ---
            # TN | FP
            # -------
            # FN | TP
            cm_frame[0, 0] = cm[1, 1]
            cm_frame[0, 1] = cm[0, 1]
            cm_frame[1, 0] = cm[1, 0]
            cm_frame[1, 1] = cm[0, 0]

        if convert_to_pandas:
            cm_frame = cm_frame.to_pandas()
            cm_frame.columns = ["actual: " + str(i) for i in level_list]
            cm_frame.index = ["predicted: " + str(i) for i in level_list]

        if print_frame:
            if self.group_column is None or get_global_cm:
                if not convert_to_pandas:
                    print("Datatable Confusion Matrix:")
                    cm_frame.view(False)
                else:
                    print("Pandas Confusion Matrix:")
                    print(cm_frame)
            else:
                if not convert_to_pandas:
                    print(
                        "Datatable Confusion Matrix by "
                        + self.group_column
                        + "="
                        + level
                    )
                    cm_frame.view(False)
                else:
                    print("Pandas Confusion Matrix:")
                    print(cm_frame)

        return cm_frame

    def __get_overall_parity_checks(self, par_frame, par_levels):
        par_frame[:, "Type I Parity"] = par_frame[
            :,
            f["False Discovery Rate Parity"] & f["False Positive Rate Parity"],
        ]
        par_frame[:, "Type II Parity"] = par_frame[
            :,
            f["False Omissions Rate Parity"] & f["False Negative Rate Parity"],
        ]
        par_frame[:, "Equalized Odds"] = par_frame[
            :, f["False Positive Rate Parity"] & f["True Positive Rate Parity"]
        ]
        par_frame[:, "Supervised Fairness"] = par_frame[
            :, f["Type I Parity"] & f["Type II Parity"]
        ]
        par_frame[:, "Overall Fairness"] = par_frame[
            :,
            f["Type I Parity"]
            & f["Type II Parity"]
            & f["False Positive Rate Parity"]
            & f["True Positive Rate Parity"]
            & f["False Omissions Rate Parity"]
            & f["False Negative Rate Parity"]
            & f["False Discovery Rate Parity"]
            & f["False Positive Rate Parity"],
        ]
        return super().format_overall_parity_checks(
            par_frame=par_frame, par_levels=par_levels
        )


class RegressionDisparateImpactAnalysis(AbstractDisparateImpactAnalysis):
    def __init__(
        self,
        actual_column=None,
        predict_column=None,
        group_column=None,
        high_threshold=1.25,
        low_threshold=0.8,
    ):
        """
        Implementation of Disparate Impact Analysis (DIA) for regression.

        Disparate Impact Analysis (DIA) is a practical way to discuss and handle
        observational fairness or simply put, how your model predictions affect
        different groups. DIA is far from perfect, as it relies heavily on user-
        defined thresholds and reference levels for disparity and does not
        attempt to remediate disparity or provide information on sources of
        disparity, but it is a fairly straightforward method to quantify your
        model’s behavior across sensitive demographic segments or other
        potentially interesting groups of observations. DIA is also an accepted,
        regulation-compliant tool for fair-lending purposes in the U.S.
        financial services industry.

        Parameters
        ----------
        actual_column: str
            Column that contains the true value for the outcome of interest
        predict_column: str
            Column that contains the predicted outcome of interest.
        group_column: str
            Column that contains certain groups of interest for DIA, e.g.,
            {female, male, other}, {high school, college, graduate school,
            other}
        high_threshold: float
            Allowed upper bound for disparity. Default is 1.25.
        low_threshold: float
            Allowed lower bound for disparity. Default is 0.8.
        """
        super().__init__(
            actual_col=actual_column,
            predict_col=predict_column,
            group_col=group_column,
            high_thresh=high_threshold,
            low_thresh=low_threshold,
            problem_type="regression",
        )

    def get_parity(self, frame, ref_level=None, get_disparity=True):
        """
        A binary indication of parity for metrics is reported by simply checking
        whether disparity values are within the user-defined thresholds. Further
        parity indicator(s) are defined as combinations of other disparity
        values:

            - Overall Fairness: Fairness across all parities for all metrics

        Parameters
        ----------
        frame: datatable.Frame
            Datatable for DIA which should either be the original frame with an
            actual, predict, and group column or a pre-calculated disparity
            frame.
        ref_level
            Reference group level used for disparity calculation.
        get_disparity: bool
            Indicate if disparity needs to be calculated before calculating
            parity.

        Returns
        -------
        par_frame: datatable.Frame
            A frame in which the first column contains each group level and each
            column after is a boolean indicator of parity across all disparity
            metrics.
        """
        par_frame, par_levels = super().get_parity(
            frame=frame, ref_level=ref_level, get_disparity=get_disparity
        )
        return self.__get_overall_parity_checks(par_frame, par_levels)

    def get_metrics(self, frame, **kwargs):
        """
        Calculate metrics of interest, which are further used for disparity
        and parity analysis:

            - Mean prediction
            - Standard deviation of predictions
            - Max of predictions
            - Min of predictions
            - R2 of predictions
            - RMSE of predictions

        Parameters
        ----------
        frame: datatable.Frame
            Datatable for DIA which should either be the original frame with an
            actual, predict, and group column or a pre-calculated disparity
            frame.

        Returns
        -------
        metrics_frame: datatable.Frame
            A frame in which the first column contains each group level and each
            column after contains all metrics of interest.

        """
        check_frame(self.actual_column, self.predict_column, self.group_column, frame)
        metrics_frame = frame[:, mean(f[self.predict_column]), by(f[self.group_column])]
        metrics_frame.names = [
            AbstractDisparateImpactAnalysis.GROUPS_COL_NAME,
            "Mean Prediction",
        ]
        groups_counts = frame[:, dt.count(), dt.by(self.group_column)]
        groups_counts.names = [
            AbstractDisparateImpactAnalysis.GROUPS_COL_NAME,
            AbstractDisparateImpactAnalysis.GROUP_COUNT_NAME,
        ]
        metrics_frame = dt.cbind(groups_counts, metrics_frame[:, "Mean Prediction"])

        for metric in self.metrics:
            if metric == "Std.Dev Prediction":
                metrics_frame[:, metric] = frame[
                    :, sd(f[self.predict_column]), by(f[self.group_column])
                ][:, 1]
            elif metric == "Maximum Prediction":
                metrics_frame[:, metric] = frame[
                    :, dt.max(f[self.predict_column]), by(f[self.group_column])
                ][:, 1]
            elif metric == "Minimum Prediction":
                metrics_frame[:, metric] = frame[
                    :, dt.min(f[self.predict_column]), by(f[self.group_column])
                ][:, 1]

        # calculate R2 and RMSE utilizing Sklearn and Pandas for now until DT
        # is able to do this
        pd_frame = frame[
            :, [self.group_column, self.actual_column, self.predict_column]
        ].to_pandas()
        r2_rmse = dt.Frame(
            pd_frame.groupby(self.group_column)
            .apply(
                get_r2_rmse,
                actual_column=self.actual_column,
                predict_column=self.predict_column,
            )
            .reset_index()[["R2", "RMSE"]]
        )
        # Pandas groupby will not consider NaN/None/missing as a group
        # whereas datatable will so the above will generate 1 too few rows;
        # this assumes datatable always puts the empty category as first
        # after grouping.
        if frame[:, self.group_column].countna().to_list()[0][0] > 0:
            null_r2_rmse = dt.Frame(
                get_r2_rmse(
                    pd_frame[pd_frame[self.group_column].isnull()],
                    self.actual_column,
                    self.predict_column,
                )
                .to_frame()
                .T.reset_index()[["R2", "RMSE"]]
            )
            r2_rmse = dt.rbind(null_r2_rmse, r2_rmse)

        metrics_frame = dt.cbind(metrics_frame, r2_rmse)

        return metrics_frame

    def __get_overall_parity_checks(self, par_frame, par_levels):
        par_frame[:, "Overall Fairness"] = par_frame[
            :,
            f["Mean Prediction Parity"]
            & f["Std.Dev Prediction Parity"]
            & f["Maximum Prediction Parity"]
            & f["Minimum Prediction Parity"]
            & f["R2 Parity"]
            & f["RMSE Parity"],
        ]
        return super().format_overall_parity_checks(
            par_frame=par_frame, par_levels=par_levels
        )
