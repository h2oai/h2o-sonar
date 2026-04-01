# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
from abc import abstractmethod
from functools import partial

import datatable

from h2o_sonar.methods.utils import fairness_utils


class AbstractDisparateImpactAnalysis:
    GROUPS_COL_NAME = "Groups"
    ALL_COL_NAME = "all"
    GROUP_COUNT_NAME = "N"
    BINOMIAL_PROBLEM_TYPE = "binomial"
    REGRESSION_PROBLEM_TYPE = "regression"

    def __init__(
        self,
        actual_col=None,
        predict_col=None,
        group_col=None,
        high_thresh=1.25,
        low_thresh=0.8,
        cutoff=None,  # only used for binary DIA
        problem_type=None,
    ):
        """Disparate Impact Analysis (DIA) commons

        Parameters
        ----------
        actual_col: str
            Column that contains the true value for the outcome of interest
        predict_col: str
            Column that contains the predicted probabilities for the outcome of
            interest.
        group_col: str
            Column that contains certain groups of interest for DIA, e.g.,
            {female, male, other}, {high school, college, graduate school,
            other}
        high_thresh: float
            Allowed upper bound for disparity. Default is 1.25.
        low_thresh: float
            Allowed lower bound for disparity. Default is 0.8.
        cutoff: float
            Numeric cutoff which act's as a decision boundary for the outcome
            of interest, e.g., above the cutoff we say a customer will default
            and below we say they will not default. Only apply's to binary DIA.
        problem_type: str
            Problem type for DIA. Should either be `regression` or `binomial`.

        """
        fairness_utils.check_dia_input(
            actual_column=actual_col,
            high_threshold=high_thresh,
            low_threshold=low_thresh,
            predict_column=predict_col,
            cutoff=cutoff,
        )

        self.actual_column = actual_col
        self.predict_column = predict_col
        self.group_column = group_col
        self.high_threshold = high_thresh
        self.low_threshold = low_thresh
        self.cutoff = cutoff
        self.problem_type = problem_type
        self.metrics = fairness_utils.get_metrics_list(problem_type)

    def get_parity(self, frame, ref_level=None, get_disparity=True):
        """
        A binary indication of parity for metrics is reported by simply checking
        whether disparity values are within the user-defined thresholds.

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
        par_levels: list
            Group level's for a particular group column, e.g, {male, female}

        """
        fairness_utils.check_frame_type(frame)

        if not isinstance(get_disparity, bool):
            raise ValueError(
                f"`get_disparity` should be of type `bool` but got type "
                f"{type(get_disparity)}"
            )
        if get_disparity:
            if ref_level is None:
                raise ValueError("`ref_level` is needed if `get_disparity` is True.")
            frame = self.get_disparity(
                frame=frame, ref_level=ref_level, fetch_metrics=True
            )
        else:
            expected_disp_col_names = (
                [AbstractDisparateImpactAnalysis.GROUPS_COL_NAME]
                + [AbstractDisparateImpactAnalysis.GROUP_COUNT_NAME]
                + [
                    "Adverse Impact Disparity",
                    "Marginal Error",
                    "Standardized Mean Difference",
                ]
                + [
                    x
                    for x in [col + " Disparity" for col in self.metrics]
                    if x != "Adverse Impact Disparity"
                ]
            )
            if (
                self.problem_type
                is AbstractDisparateImpactAnalysis.REGRESSION_PROBLEM_TYPE
            ):
                expected_disp_col_names.remove("Marginal Error")
                expected_disp_col_names.remove("Adverse Impact Disparity")
            if list(frame.names) != expected_disp_col_names:
                raise ValueError(
                    f"Input frame should have column names "
                    f"{expected_disp_col_names} but has column names"
                    f"{frame.names} and `get_disparity` is set to "
                    f"{get_disparity}. Please either make a "
                    f"disparity frame by calling `disparity_frame = "
                    f"self.get_disparity(frame=frame, ref_level=ref_level"
                    f", get_metrics=True)` and pass that `disparity_frame` "
                    f"to `get_parity()` with `get_disparity=False`"
                    f"or pass in the original frame with the actual column, "
                    f"the predicted column, and the group column "
                    f"to `get_parity()` and set `get_disparity` to True."
                )

        group_levels = fairness_utils.get_group_levels(
            AbstractDisparateImpactAnalysis.GROUPS_COL_NAME, frame
        )

        par_col_names = [col + " Parity" for col in self.metrics]
        # par_frame = dt.Frame(
        #     names=par_col_names, stypes=[dt.stype.bool8] * len(par_col_names)
        # )
        par_levels = group_levels
        # par_frame.nrows = len(par_levels)

        disp_frame_names = list(frame.names)
        disp_frame_names.remove(AbstractDisparateImpactAnalysis.GROUP_COUNT_NAME)
        disp_frame_names.remove(AbstractDisparateImpactAnalysis.GROUPS_COL_NAME)

        if self.problem_type is AbstractDisparateImpactAnalysis.BINOMIAL_PROBLEM_TYPE:
            disp_frame_names.remove("Marginal Error")
            if "Standardized Mean Difference" in disp_frame_names:
                disp_frame_names.remove("Standardized Mean Difference")
        else:
            if "Standardized Mean Difference" in disp_frame_names:
                disp_frame_names.remove("Standardized Mean Difference")

        def to_parity(lower_threshold, higher_threshold, value):
            if value is None:
                return None
            return lower_threshold <= value <= higher_threshold

        partial_to_parity = partial(to_parity, self.low_threshold, self.high_threshold)
        par_frame = datatable.Frame(
            frame[:, disp_frame_names].to_pandas().applymap(partial_to_parity),
            names=par_col_names,
        )

        par_frame = datatable.cbind(
            frame[:, AbstractDisparateImpactAnalysis.GROUP_COUNT_NAME],
            par_frame,
        )
        return par_frame, par_levels

    def get_disparity(
        self, frame, ref_level, fetch_metrics=True, pred_group_frame=None
    ):
        """
        Compare metrics for each group level to the metrics for a user-defined
        reference level.

        Parameters
        ----------
        frame: datatable.Frame
            Datatable for DIA which should either be the original frame with an
            actual, predict, and group column or a pre-calculated metrics
            frame.
        ref_level
            Reference group level used for disparity calculation.
        fetch_metrics: bool
            Indicate if metrics need to be calculated before calculating
            disparity.
        pred_group_frame: datatable.Frame
            Frame containing prediction column and group column. This is used to
            calculated SMD when the `frame` argument is a metrics frame i.e. not the
            original frame.

        Returns
        -------
        disparity_frame: datatable.Frame
            A frame in which the first column contains each group level and
            each column after contains all disparity values across metrics of
            interest.

        """
        fairness_utils.check_frame_type(frame)

        if not isinstance(fetch_metrics, bool):
            raise ValueError(
                f"`fetch_metrics` should be of type `bool` but got type "
                f"{type(fetch_metrics)}"
            )

        if pred_group_frame and not fetch_metrics:
            pred_mean_per_group_frame = self.get_pred_mean_per_group(pred_group_frame)
            pred_column_sigma = pred_group_frame[
                :, datatable.sd(pred_group_frame[:, self.predict_column])
            ]
            calculate_smd = True
        elif not pred_group_frame and fetch_metrics:
            if self.group_column in frame.names:
                pred_mean_per_group_frame = self.get_pred_mean_per_group(frame)
            else:
                raise ValueError(
                    f"Group column, {self.group_column}, is not in the input frame"
                )
            if self.predict_column in frame.names:
                pred_column_sigma = frame[
                    :, datatable.sd(frame[:, self.predict_column])
                ]
            else:
                raise ValueError(
                    f"Prediction column, {self.predict_column}, is not in the input "
                    f"frame"
                )
            calculate_smd = True
        else:
            raise ValueError(
                f"Passing in {pred_group_frame} for `pred_group_frame`"
                f"frame` and {fetch_metrics} for `fetch_metrics` is not supported. "
                f"Either pass in `pred_group_frame` & set `fetch_metrics` to False or "
                f"do not pass in `pred_group_frame` & set `fetch_metrics` to True"
            )

        if fetch_metrics:
            frame = self.get_metrics(frame=frame)
        else:
            expected_metric_col_names = (
                [AbstractDisparateImpactAnalysis.GROUPS_COL_NAME]
                + [AbstractDisparateImpactAnalysis.GROUP_COUNT_NAME]
                + self.metrics
            )
            if list(frame.names) != expected_metric_col_names:
                raise ValueError(
                    f"Input frame should have column names "
                    f"{expected_metric_col_names} but has column names"
                    f"{frame.names} and `get_metrics` is set to "
                    f"{fetch_metrics}. Please either make a "
                    f"metrics frame by calling `metrics_frame = "
                    f"self.get_metrics(frame=frame)` and pass that "
                    f"`metrics_frame` to `get_disparity()` with "
                    f"`get_metrics=False` or pass in the original "
                    f"frame with the actual column, the predicted column, "
                    f"and the group column to `get_disparity()` "
                    f"and set `fetch_metrics` to True."
                )

        # Asserts
        group_levels = fairness_utils.get_group_levels(
            AbstractDisparateImpactAnalysis.GROUPS_COL_NAME, frame
        )
        if ref_level not in group_levels:
            raise ValueError(
                f"`ref_level`, {ref_level}, is not in `Group` column. "
                f"Accepted group levels are: {group_levels}"
            )

        if ref_level is None and "str" in str(
            frame[:, AbstractDisparateImpactAnalysis.GROUPS_COL_NAME].stypes
        ):
            ref_level = frame[
                :, AbstractDisparateImpactAnalysis.GROUPS_COL_NAME
            ].stypes[0](ref_level)

        ref_group = frame[datatable.f.Groups == ref_level, :]
        ref_group = ref_group[:, self.metrics]

        disp_col_names = [col + " Disparity" for col in self.metrics]
        disp_frame = frame[
            :, [AbstractDisparateImpactAnalysis.GROUPS_COL_NAME] + self.metrics
        ]

        if self.problem_type is AbstractDisparateImpactAnalysis.BINOMIAL_PROBLEM_TYPE:
            disp_frame[:, "Marginal Error"] = disp_frame[
                :,
                ref_group[:, datatable.f["Adverse Impact"]]
                - datatable.f["Adverse Impact"],
            ]

        if calculate_smd and pred_mean_per_group_frame and pred_column_sigma:
            disp_frame = disp_frame[:, :, datatable.join(pred_mean_per_group_frame)]
            disp_frame[:, "Standardized Mean Difference"] = disp_frame[
                :,
                (
                    (
                        datatable.f["mean_pred"]
                        - pred_mean_per_group_frame[datatable.f.Groups == ref_level, :][
                            "mean_pred"
                        ]
                    )
                    / pred_column_sigma
                ),
            ]
            disp_col_names = ["Standardized Mean Difference"] + disp_col_names

        for disparity_measure in disp_col_names:
            if disparity_measure != "Standardized Mean Difference":
                measure = disparity_measure.rsplit(" Disparity", 1)[0]
                disp_frame[:, disparity_measure] = disp_frame[
                    :, datatable.f[measure] / ref_group[:, datatable.f[measure]]
                ]

        if self.problem_type is AbstractDisparateImpactAnalysis.BINOMIAL_PROBLEM_TYPE:
            disp_col_names = ["Adverse Impact Disparity", "Marginal Error"] + [
                x for x in disp_col_names if x != "Adverse Impact Disparity"
            ]

        disp_frame = datatable.cbind(
            frame[:, datatable.f.Groups],
            frame[:, AbstractDisparateImpactAnalysis.GROUP_COUNT_NAME],
            disp_frame[:, disp_col_names],
        )

        return disp_frame

    def get_pred_mean_per_group(self, frame):
        pred_mean_per_group_frame = frame[
            :,
            datatable.mean(datatable.f[self.predict_column]),
            datatable.by(datatable.f[self.group_column]),
        ]
        pred_mean_per_group_frame.names = [
            AbstractDisparateImpactAnalysis.GROUPS_COL_NAME,
            "mean_pred",
        ]
        pred_mean_per_group_frame.key = AbstractDisparateImpactAnalysis.GROUPS_COL_NAME
        return pred_mean_per_group_frame

    @abstractmethod
    def get_metrics(self, frame, **kwargs):
        pass

    @staticmethod
    def format_overall_parity_checks(par_frame, par_levels):
        """

        Parameters
        ----------
        par_frame: datatable.Frame
            A frame in which the first column contains each group level and each
            column after is a boolean indicator of parity across all disparity
            metrics.
        par_levels: list
            Group level's for a particular group column, e.g, {male, female}

        Returns
        -------
        par_frame: datatable.Frame
            Formatted parity frame
        """
        par_frame_groups = datatable.Frame(
            Groups=[str(par_level) for par_level in par_levels]
        )
        par_frame_groups.nrows = len(par_levels)
        par_frame = datatable.cbind(par_frame_groups, par_frame)
        par_frame_all = datatable.cbind(
            datatable.Frame(datatable.Frame(par_frame.sum()))
        )

        par_frame_all[:, AbstractDisparateImpactAnalysis.GROUPS_COL_NAME] = (
            par_frame_all[
                :,
                datatable.stype.str32(
                    datatable.f[AbstractDisparateImpactAnalysis.GROUPS_COL_NAME]
                ),
            ]
        )
        par_frame_all[0, AbstractDisparateImpactAnalysis.GROUPS_COL_NAME] = (
            AbstractDisparateImpactAnalysis.ALL_COL_NAME
        )
        par_frame_all_metrics = list(par_frame.names)
        par_frame_all_metrics.remove(AbstractDisparateImpactAnalysis.GROUPS_COL_NAME)
        for j in range(2, len(par_frame_all_metrics) + 1):
            if par_frame_all[0, j] == len(par_levels):
                par_frame_all[0, j] = 1
            else:
                par_frame_all[0, j] = 0
        types = [
            (name, datatable.stype(stype)(datatable.f[name]))
            for (name, stype) in zip(par_frame.names, par_frame.stypes, strict=False)
        ]
        par_frame_all = par_frame_all[:, dict(types)]
        par_frame = datatable.rbind(par_frame, par_frame_all)
        return par_frame
