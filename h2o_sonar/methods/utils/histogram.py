# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import math
from enum import auto
from enum import Enum

import datatable
import numpy
import pandas as pd


class HistogramBackend(Enum):
    MLI = auto()


def histogram_data(
    df,
    bins: list = None,
    grid_resolution: int = 20,
    is_discrete: bool = False,
    is_date: bool = False,
    discrete_threshold: float = 0.06,
    backend: HistogramBackend = HistogramBackend.MLI,
    logger=None,
) -> tuple:
    """Get histogram data for given feature.

    Supported feature data types:

    - ``integer`` (continuous)
    - ``float`` (continuous)
    - ``string`` (discrete)
    - ``date/time`` (continuous)

    This implementation provides MLI and backend:

    - **MLI histogram backend** is used by default. It calculates histograms for int,
      float, string and date features using Numpy (Pandas is used only to convert dates
      for Numpy). MLI histogram backend allows bins specification and provides valid
      x-axis labels for all feature types.

    Method can decide which backend to use.

    Parameters
    ----------
    df: datatable.Frame:
      Data for which to calculate histogram represented as frame with one column
      (target feature).
    bins: list
      Optional bins / split points for which to compute histogram (unsupported by
      AutoReport backend).
    grid_resolution: int
      Optional grid resolution - the number of equal-width bins / split points in the
      given range.
    is_discrete: bool
      Optional specification to override continuous histogram default (``False``) of
      integer / float features and create discrete (categorical) histogram instead.
    is_date: bool:
      Optional specification to force date/time histogram for a string feature.
    discrete_threshold: float
      Optional threshold for relative difference between min/max gap, to get
      histogram for numeric ``df`` as discrete (if ``min_gap/max_gap < threshold``,
      plot histogram).
    backend: HistogramBackend
      Backend to calculate histograms, ``HistogramBackend.MLI`` is default.
    logger:
      Logger for testability.

    Returns
    -------
    list, list:
      x-axis (bins/split point values) and y-axis (histogram frequencies).
      The number of x-axis and y-axis ticks are the same in case of discrete
      (categorical) features, but different in case of histogram for continuous
      features: ``len(x) = len(frequencies)+1``.

    """
    if backend is HistogramBackend.MLI:
        (x_ticks, y_ticks) = _histogram_data(
            df=df,
            bins=bins,
            grid_resolution=grid_resolution,
            is_discrete=is_discrete,
            is_date=is_date,
        )
        if df.shape[0] - df[datatable.f[0] != None, :].shape[0] != 0:  # noqa: E711
            if None not in x_ticks and float("nan") not in x_ticks:
                x_ticks.append(float("nan"))
                y_ticks.append(df[datatable.f[0] == None, :].shape[0])  # noqa: E711
        return x_ticks, y_ticks

    raise ValueError(f"Unknown histogram backend: {backend}")


def _histogram_data(
    df: datatable.Frame,
    bins: list = None,
    grid_resolution: int = 20,
    is_discrete: bool = False,
    is_date: bool = False,
    drop_nas: bool = False,
) -> tuple:
    if drop_nas:
        nan_filter = None  # linter
        df = df[datatable.f[0] != nan_filter, :]

    if bins:
        if is_discrete:
            if has_nan(bins):
                bins.remove(bins[find_nan_index(bins)])
        if is_date:
            if not is_discrete:
                # parse bins to int (nanoseconds)
                int_bins = [pd.Timestamp(str(b)).value for b in bins]
                # compute date histogram
                x, f = _histogram_data_date(
                    df=df, grid_resolution=grid_resolution, int_bins=int_bins
                )
                return bins, f.tolist()

        # int / float WITH bins specification
        if df.ltypes[0] in [datatable.ltype.int, datatable.ltype.real]:
            if not is_discrete:
                # continuous histogram
                (f, x) = numpy.histogram(
                    a=df.to_numpy(),
                    bins=bins or grid_resolution,
                )
                x = x.tolist()
                f = f.tolist()
                if has_nan(x):
                    x.remove(x[find_nan_index(x)])
                return x, f

        # string (or discrete int / float / date)
        df = df[datatable.f[df.names[0]] != None, :]  # noqa: E711
        (np_x, np_f) = numpy.unique(df, return_counts=True)
        x_all = np_x.tolist()
        f_all = np_f.tolist()
        x = []
        f = []
        # return a value for every bin value
        for b in bins:
            x.append(b)
            f.append(f_all[x_all.index(b)] if b in x_all else 0)
        return x, f

    # date WITHOUT bins specification
    if is_date:
        x, f = _histogram_data_date(df, grid_resolution)
        return x, f.tolist()

    # int / float WITHOUT bins specification
    if df.ltypes[0] in [datatable.ltype.int, datatable.ltype.real]:
        if not is_discrete:
            (f, x) = numpy.histogram(
                a=df.to_numpy(),
                bins=grid_resolution,
            )
            return x.tolist(), f.tolist()

    # string WITHOUT bins specification
    (x, f) = numpy.unique(df, return_counts=True)
    return x.tolist(), f.tolist()


def _histogram_data_date(
    df,
    grid_resolution: int,
    int_bins: list | None = None,
):
    """Date, time and datatime histograms."""
    col = df.to_pandas()
    # pandas frame with successfully parsed date
    pd_date_frame = col.astype("datetime64")
    # convert date objects to int to compute histogram
    pd_date_as_int_frame = pd_date_frame.astype(numpy.int64)
    # transfer it to numpy histogram computation
    np_date_as_int_frame = pd_date_as_int_frame.to_numpy()
    # int histogram using numpy: frequency, x as integer
    (f, x) = numpy.histogram(
        a=np_date_as_int_frame,
        bins=int_bins or grid_resolution,
    )
    # x-axis: get dates for integer timestamp(s)
    d = [str(pd.to_datetime(i)) for i in x]
    return d, f


def has_nan(lst):
    for item in lst:
        if isinstance(item, float) and math.isnan(item):
            return True
    return False


def find_nan_index(lst):
    for i, x in enumerate(lst):
        if x != x or x is None:  # Check if element is NaN or None
            return i
    return None
