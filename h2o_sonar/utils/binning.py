# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import gc

import datatable
import numpy
import pandas

from h2o_sonar import config
from h2o_sonar import errors


def qbin_column(frame: datatable.Frame, column: str, logger):
    """Quantile bin a column in a frame and substitute it in that frame with
    quantile group ranges for each row.

    Parameters
    ----------
    frame : datatable.Frame
        Frame containing the data. One of the column names must correspond
        to the column parameter.
    column : str
        Name of the column to be checked.
    logger : Logger
        Logger.

    """
    series_to_qcut = pandas.Series(frame[:, column].to_list()[0]).astype(numpy.float)

    if series_to_qcut.isnull().values.any(axis=0):
        if series_to_qcut.isnull().values.all(axis=0):
            logger.data(
                f"Columns, <<<{column}>>>, contains all null values and therefore "
                f"cannot be binned ...",
            )
            return
        elif len(series_to_qcut.dropna().unique()) <= 1:
            logger.data(
                f"Columns, <<<{column}>>>, contains only one unique value after "
                f"dropping null values and therefore cannot be binned ...",
            )
            return

    firstpass = pandas.qcut(
        series_to_qcut, config.config.mli_num_quantiles, duplicates="drop"
    )
    frame[:, column] = datatable.Frame(list(firstpass.astype(str)))
    frame.names = {column: column + "_mli_qtile"}
    del firstpass
    gc.collect()


def quantile_bin(
    frame: datatable.Frame = None,
    qbin_cols: list[str] | None = None,
    qbin_count: int = 0,
    varimp_list: list[str] | None = None,
    logger=None,
):
    """Quantile binning.

    Parameters
    ----------
    frame : dt.Frame
      Input frame for quantile binning.
    qbin_cols : list
      Column(s) to use for quantile binning
    qbin_count : int
      Number of top numeric variables to use from model's variable
      importance list.
    varimp_list : list
      Variable importance list from model.
    logger : Logger
      Logger.

    Returns
    -------
    Tuple[list, Pandas Dataframe] :
      List of columns that were binned and Dataframe with quantile binned columns.

    """
    # input checks
    qbin_cols = qbin_cols if qbin_cols is not None else []
    varimp_list = varimp_list if varimp_list is not None else []

    # input frame check
    if frame is None:
        raise ValueError(
            "Quantile binning should not be called without an input frame!"
        )
    if not isinstance(frame, datatable.Frame):
        raise ValueError(
            f"Input frame for quantile binning should be a datatable.Frame but "
            f"got {type(frame)}"
        )

    # qbin_cols and qbin_count checks
    if qbin_count < 0:
        raise errors.InvalidDataError(
            f"Quantile binning requires the top n features to be non-negative "
            f"but got {qbin_count}"
        )
    if not qbin_cols and qbin_count == 0:
        raise errors.InvalidDataError(
            "Quantile binning requires at least a list of columns to bin or "
            "top n from variable importance list!"
        )
    if not isinstance(qbin_cols, list):
        raise errors.InvalidDataError(
            f"Quantile binning requires selected column(s) to be of type `list` "
            f"but got <<<{type(qbin_cols)}>>>"
        )
    if not isinstance(qbin_count, int):
        raise errors.InvalidDataError(
            f"Quantile binning requires count of quantile bin variables to be "
            f"of type `int` but got <<<{type(qbin_cols)}>>>"
        )

    # varimp_list check
    if not varimp_list and qbin_count > 0 and not qbin_cols:
        raise errors.InvalidDataError(
            "Quantile binning requires a list of variables importances as an "
            "input if no column(s) are selected to bin and top n variable to "
            "choose from variable importance list is greater than zero ..."
        )
    if varimp_list:
        if not isinstance(varimp_list, list):
            raise ValueError(
                f"Quantile binning requires variable importance list to be of type "
                f"`list` but got <<<{type(varimp_list)}>>>"
            )

    count_done = 0
    idx = 0
    binned_list = []
    # create the quantile bins from variable list
    while count_done < qbin_count:
        if varimp_list:
            if idx >= len(varimp_list):
                logger.warning("Could not quantile bin all requested variables")
                break
            logger.info(f"Attempting to bin variable: {varimp_list[idx]}")
            qbin_column(frame, varimp_list[idx], logger)
            count_done += 1
            binned_list.append(varimp_list[idx])
            logger.info(
                f"Successfully binned variable: {varimp_list[idx]}",
            )
            idx += 1
        else:
            logger.warning(
                "Could not quantile bin empty list of requested variables",
            )
            break
    # bin requested columns if they haven't been done already
    for i in range(0, len(qbin_cols)):
        if qbin_cols[i] in binned_list:
            logger.data(f"Feature: {qbin_cols[i]} already binned")
            continue
        else:
            logger.data(
                f"Attempting to bin requested variable: <<<{qbin_cols[i]}>>>",
            )
            qbin_column(frame, qbin_cols[i], logger)
            binned_list.append(qbin_cols[i])
            logger.data(
                f"Successfully binned requested variable: <<<{qbin_cols[i]}>>>",
            )

    return binned_list, frame


def build_qtile_bins(bins: list, X: pandas.DataFrame, feature: str, quantile: int):
    """Build quantile bins and append back to input bins list.

    Parameters
    ----------
    bins : list
        List of bins.
    X : pandas.DataFrame
        Input frame to PD/ICE.
    feature : str
        Feature to create quantile bins for.
    quantile : int
        The decile to compute.

    """
    bins.append(
        X[feature]
        .quantile(
            [
                x / quantile
                for x in range(
                    1,
                    quantile + 1,
                )
            ]
        )
        .values.tolist()
    )
