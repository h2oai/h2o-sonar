# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import datatable as dt
import numpy as np
import pandas as pd


def cm_exp_parser(expression, cm_dict, level):
    """Small utility function that translates abbreviated metric expressions into
    executable Python statements:

    tp | fp       cm_dict[level][0, 0] | cm_dict[level][0, 1]
    -------  ==>  --------------------------------------------
    fn | tn       cm_dict[level][1, 0] | cm_dict[level][1, 1]

    """

    expression = (
        expression.replace("tp", f"{cm_dict[level][0, 0]}")
        .replace("fp", f"{cm_dict[level][0, 1]}")
        .replace("fn", f"{cm_dict[level][1, 0]}")
        .replace("tn", f"{cm_dict[level][1, 1]}")
    )

    return expression


def get_metrics_list(problem_type):
    """Get DIA metrics for a given problem type (regression or binomial)."""
    if not problem_type:
        raise ValueError("Need to set problem_type")
    if problem_type == "regression":
        return get_reg_metrics_list()
    if problem_type == "binomial":
        return list(get_binary_metric_dict().keys())
    raise ValueError(
        f"Problem type should either be `regression` or `binomial`"
        f"but got {problem_type}"
    )


def get_binary_metric_dict():
    """Dictionary of metrics utilized by binary DIA."""
    # Dictionary of metrics used in DIA
    metric_dict = {
        # Adverse Impact
        # Example: how often the model predicted default for each group
        "Adverse Impact": "(tp + fp) / (tp + tn + fp + fn)",
        # Overall performance
        # Example: How often the model predicts default and non-default
        # correctly for this group
        "Accuracy": "(tp + tn) / (tp + tn + fp + fn)",
        # Predicting outcome of interest will happen (correctly)
        # Example: Out of the people in the group *that did* default,
        # how many the model predicted *correctly* would default
        "True Positive Rate": "tp / (tp + fn)",
        # Example:  Out of the people in the group the model *predicted* would
        # default, how many the model predicted *correctly* would default
        "Precision": "tp / (tp + fp)",
        # Predicting outcome of interest won't happen (correctly)
        # Example: Out of the people in the group *that did not* default,
        # how many the model predicted *correctly* would not default
        "Specificity": "tn / (tn + fp)",
        # Example: Out of the people in the group the model *predicted*
        # would not default, how many the model predicted *correctly* would not
        # default
        "Negative Predicted Value": "tn / (tn + fn)",
        # Analyzing errors
        #
        # Type I
        # False Accusations
        # Example: Out of the people in the group *that did not* default,
        # how many the model predicted *incorrectly* would default
        "False Positive Rate": "fp / (tn + fp)",
        # Example: Out of the people in the group the model *predicted*
        # would default, how many the model predicted *incorrectly* would
        # default
        "False Discovery Rate": "fp / (tp + fp)",
        # Type II
        # Costly omissions
        # Example: Out of the people in the group *that did* default,
        # how many the model predicted *incorrectly* would not default
        "False Negative Rate": "fn / (tp + fn)",
        # Example: Out of the people in the group the model *predicted*
        # would not default, how many the model predicted *incorrectly* would
        # not default
        "False Omissions Rate": "fn / (tn + fn)",
    }

    return metric_dict


def get_reg_metrics_list():
    """List of metrics utilized by regression DIA."""
    metrics = [
        "Mean Prediction",
        "Std.Dev Prediction",
        "Maximum Prediction",
        "Minimum Prediction",
        "R2",
        "RMSE",
    ]
    return metrics


def get_r2_rmse(frame, actual_column, predict_column):
    """Calculate R2 and RMSE between actual and predicted columns in a Pandas
    frame.

    """
    r_square = r_squared(frame[actual_column], frame[predict_column])
    rmse = root_mean_squared_error(frame[actual_column], frame[predict_column])
    return pd.Series(dict(R2=r_square, RMSE=rmse))


def get_group_levels(group_column, frame):
    """
    Get level's for a particular group column, e.g, {male, female}

    """
    # Get group levels
    if group_column not in frame.names:
        raise ValueError(
            f"Group column, {group_column}, is not in frame column names. "
            f"Expected one of the following: {frame.names}"
        )
    return dt.unique(frame[:, group_column]).to_list()[0]


def check_dia_input(
    actual_column, high_threshold, low_threshold, predict_column, cutoff=None
):
    """
    Check input of binary DIA class initialization

    """
    if not isinstance(actual_column, str):
        raise ValueError(
            f"`actual_column` should be of type `str` but got type "
            f"{type(actual_column)}"
        )
    if not isinstance(predict_column, str):
        raise ValueError(
            f"`predict_column` should be of type `str` but got type "
            f"{type(predict_column)}"
        )
    if not isinstance(high_threshold, float):
        raise ValueError(
            f"`high_threshold` should be of type `float` but got type "
            f"{type(high_threshold)}"
        )
    if not isinstance(low_threshold, float):
        raise ValueError(
            f"`low_threshold` should be of type `float` but got type "
            f"{type(low_threshold)}"
        )
    if cutoff:  # Only used for binary DIA
        if not isinstance(cutoff, float):
            raise ValueError(
                f"`cutoff` should be of type `float` but got type {type(cutoff)}"
            )
        if cutoff < 0:
            raise ValueError(f"`cutoff` should be greater than zero but got {cutoff}")
        if cutoff > 1.0:
            raise ValueError(
                f"`cutoff` should not be greater than 1.0 but got {cutoff}"
            )
    if high_threshold < 0:
        raise ValueError(
            f"`high_threshold` should be greater than zero but got {high_threshold}"
        )
    if low_threshold < 0:
        raise ValueError(
            f"`low_threshold` should be greater than zero but got {low_threshold}"
        )
    if low_threshold > high_threshold:
        raise ValueError(
            f"`low_threshold` should not be greater than `high_threshold` "
            f"but got {low_threshold} for `low_threshold` and "
            f"{high_threshold} for `high_threshold`"
        )
    if high_threshold == low_threshold:
        raise ValueError(
            f"`low_threshold` and `high_threshold` should not be equal to each "
            f"other but got {low_threshold} for `low_threshold` and "
            f"{high_threshold} for `high_threshold`"
        )


def check_frame(actual_column, predict_column, group_column, frame):
    """
    Sanity checks for input frame to DIA

    """
    check_frame_type(frame)

    if actual_column not in frame.names:
        raise ValueError(
            f"Actual column, {actual_column}, is not in frame column names. "
            f"Expected one of the following: {frame.names}"
        )
    if predict_column not in frame.names:
        raise ValueError(
            f"Predict column, {predict_column}, is not in frame column names. "
            f"Expected one of the following: {frame.names}"
        )
    if group_column not in frame.names:
        raise ValueError(
            f"Group column, {group_column}, is not in frame column names. "
            f"Expected one of the following: {frame.names}"
        )


def check_frame_type(frame):
    """
    Check frame type for DIA

    """
    if not isinstance(frame, dt.Frame):
        raise ValueError(
            f"Input frame should be of type `datatable.Frame` but got type "
            f"{type(frame)}"
        )


def check_cm_input(get_global_cm, group_levels, level, print_frame, group_column):
    """
    Check input to confusion matrix for binary DIA

    """
    if level:
        if level not in group_levels:
            raise ValueError(
                f"`level`, {level}, is not in `{group_column}` "
                f"column. Expected levels are: {group_levels}"
            )
    if not isinstance(print_frame, bool):
        raise ValueError(
            f"`print_frame` should be of type `bool` but got type {type(print_frame)}"
        )
    if not isinstance(get_global_cm, bool):
        raise ValueError(
            f"`get_global_cm` should be of type `bool` but got type "
            f"{type(get_global_cm)}"
        )


def r_squared(actual, predicted):
    """
    Computes R^2 (coefficient of determination) regression score function.

    """
    wma = np.average(actual)
    wmp = np.average(predicted)
    wva = np.average(np.power((actual - wma), 2))
    wvp = np.average(np.power((predicted - wmp), 2))
    wcv = np.average((actual - wma) * (predicted - wmp))
    return np.power(wcv / (np.sqrt(wva * wvp) + 1e-15), 2)


def squared_error(actual, predicted):
    """
    Computes the squared error.

    """
    return np.power(np.array(actual) - np.array(predicted), 2)


def mean_squared_error(actual, predicted):
    """
    Computes the mean squared error.

    """
    return np.mean(squared_error(actual, predicted))


def root_mean_squared_error(actual, predicted):
    """
    Computes the root mean squared error.

    """
    return np.sqrt(mean_squared_error(actual, predicted))


def get_prroc_dt(frame, y, yhat, pos=1, neg=0, res=0.01):
    """Calculates precision, recall, and f1 for a datatable of y and yhat values.

    Args:
        frame: Datatable of actual (y) and predicted (yhat) values.
        y: Name of actual value column.
        yhat: Name of predicted value column.
        pos: Primary target value, default 1.
        neg: Secondary target value, default 0.
        res: Resolution by which to loop through cutoffs, default 0.01.

    Returns:
        Datatable of precision, recall, and f1 values.
    """

    dname = "d_" + str(y)  # column for predicted decisions
    eps = 1e-20  # for safe numerical operations

    # init p-r roc frame
    prroc_frame = dt.Frame({"cutoff": [], "recall": [], "precision": [], "f1": []})

    # loop through cutoffs to create p-r roc frame
    for cutoff in np.arange(0, 1 + res, res):
        # binarize decision to create confusion matrix values
        frame[:, dname] = 0
        frame[dt.f[yhat] > cutoff, dname] = 1

        # calculate confusion matrix values
        true_pos = frame[(dt.f[dname] == pos) & (dt.f[y] == pos), :].shape[0]
        false_pos = frame[(dt.f[dname] == pos) & (dt.f[y] == neg), :].shape[0]
        # true_neg = frame[(dt.f[dname] == neg) & (dt.f[y] == neg), :].shape[0]
        false_neg = frame[(dt.f[dname] == neg) & (dt.f[y] == pos), :].shape[0]

        # calculate precision, recall, and f1
        recall = (true_pos + eps) / ((true_pos + false_neg) + eps)
        precision = (true_pos + eps) / ((true_pos + false_pos) + eps)
        f_1 = 2 / ((1 / (recall + eps)) + (1 / (precision + eps)))

        # add new values to frame
        prroc_frame.rbind(
            dt.Frame(
                {
                    "cutoff": [cutoff],
                    "recall": [recall],
                    "precision": [precision],
                    "f1": [f_1],
                }
            )
        )

        del frame[:, dname]

    return prroc_frame


def smd_multinomial(frame, y, group_col, ref_level):
    """

    Parameters
    ----------
    frame: datatable.Frame
        Datatable that contains target, group column, and multinomial predictions
        (probabilities) as columns for each class outcome. For example:

            target | group_col | class_1_prob | class_2_prob | ... |
    y: str
        Column that contains the true value for the outcome of interest
    group_col: str
        Column that contains certain groups of interest for DIA, e.g.,
        {female, male, other}, {high school, college, graduate school,
        other}
    ref_level: str
        Reference group level used for disparity calculation.

    Returns
    -------
    smd_frame: datatable.Frame
        A frame in which the first column contains each group level and each column
        after contains the standardized mean difference between each class outcome
        and the reference level.


    """
    assert y in frame.names, (
        f"Target, {y}, should be in input frame names. However, column names are "
        f"currently {frame.names}"
    )

    # Get unique values of target
    pred_labels = dt.unique(frame[:, y]).to_list()[0]
    assert all(x in frame.names for x in pred_labels), (
        f"Prediction labels, {pred_labels}, should be in input frame names. However, "
        f"column names are currently {frame.names}"
    )

    assert group_col in frame.names, (
        f"Group column, {group_col}, should be in input frame names. However, column "
        f"names are currently: {frame.names}"
    )

    # Get classes from group column
    protected_labels = dt.unique(frame[:, group_col]).to_list()[0]
    assert ref_level in protected_labels, (
        f"Reference level, {ref_level}, should be in group column, {group_col}, "
        f"unique values. However, the group column, {group_col}, "
        f"only has the following unique values: {protected_labels}"
    )

    # Calculate sigma across all label predictions
    sigma = np.std(frame[:, pred_labels].to_numpy())

    # Initialize SMD frame
    group_frame = dt.Frame(group=protected_labels)
    preds_frame = dt.Frame(
        [[0]] * len(pred_labels),
        names=pred_labels,
        stypes=[dt.float32] * len(pred_labels),
    )
    preds_frame = dt.repeat(preds_frame, len(protected_labels))
    smd_frame = dt.cbind(group_frame, preds_frame)

    # Calculate SMD across group levels and class outcomes
    for protected in protected_labels:
        for label in pred_labels:
            # yhat mean for protected at specified label
            mean_protected = frame[dt.f[group_col] == protected, :][:, label].mean()[
                0, 0
            ]

            # yhat mean for reference at specified label
            mean_reference = frame[dt.f[group_col] == ref_level, :][:, label].mean()[
                0, 0
            ]

            # Calculate SMD
            # Note, bottom if statement is needed bc for some reason the precision of
            # the mean changes, causing a very small difference even if
            # protected == reference
            if protected != ref_level:
                smd = (mean_protected - mean_reference) / sigma
            else:
                smd = 0

            smd_frame[dt.f.group == protected, label] = smd

    return smd_frame
