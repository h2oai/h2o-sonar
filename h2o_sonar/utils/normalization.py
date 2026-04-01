# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import datatable


def normalize_importance(frame: datatable.Frame) -> datatable.Frame:
    """Normalize local feature importance values to global as percentage.

    Parameters
    ----------
    frame : datatable.Frame
        Frame with local feature importance values.

    Returns
    -------
    datatable.Frame
        Normalized frame with global feature importance values.

    """
    if frame.nrows < 1:
        raise ValueError("Frame needs to have at least one row to be normalized")
    column_totals_frame: datatable.Frame = frame.sum()
    total_frame: datatable.Frame = column_totals_frame[
        0, datatable.rowsum(datatable.f[:])
    ]
    total: float = total_frame[0, 0]
    output: datatable.Frame = frame[:, datatable.f[:] / total]
    output.names = column_totals_frame.names
    return output
