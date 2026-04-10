# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import traceback

import datatable
import numpy
from matplotlib import pyplot
from pandas.api.types import is_numeric_dtype

from h2o_sonar.lib.api import commons


def safe_plot_names(column_list: list[str]) -> list:
    """Return a list of column names that exclude problematic special characters
    for matplotlib plotting functions.

    Parameters
    ----------
    column_list: list[str]
      List of column names.

    Returns
    -------
    List:
      List with column names that are safe to plot.

    """
    safe_column_list = [col.replace("$", "_") for col in column_list]
    return safe_column_list


class ScatterFeatImpPlot:
    """Scatter plot feature importance representation is based on chart from:

    https://github.com/slundberg/shap

    """

    @staticmethod
    def plot(
        contributions,
        frame,
        alpha: float = 1.0,
        colormap: str | None = None,
        figsize=(12, 12),  # type: Tuple[float] | list[float] | None
        jitter: float = 0.35,
        chart_title: str = "Feature importance summary plot",
        x_label: str = "Value",
        y_label: str = "Feature",
        thermometer_label: str = "Normalized feature value",
        columns=None,  # type: list[int] | list[str] | None
        top_n_features: int = 20,
        samples: int | None = None,
        colorize_factors: bool = True,
        drop_zero_contribs=True,
        hard_asserts=False,
        logger=None,
    ) -> pyplot.Figure:
        """Feature importance summary plot.

        Summary plot shows contribution of features for each instance. The sum of
        the feature contributions and the bias term is equal to the raw prediction
        of the model, i.e., prediction before applying inverse link function.

        Parameters
        ----------
        contributions :
          Pandas contributions frame with coefficients. Frame column names to be
          (sanitized) feature names, rows to correspond to dataset rows, cells to be
          coefficients.
        frame :
          Pandas dataset frame with values. Frame column names to be (sanitized)
          feature names, rows to correspond to dataset rows, cells to be values.
        columns :
          Either a list of columns or column indices to show. If specified
          parameter ``top_n_features`` will be ignored.
        top_n_features : int
          A number of columns to pick using variable importance
          (where applicable). Set to ``-1`` to show all features.
        samples :
          Maximum number of observations to use; if lower than number of rows in the
          frame, take a random sample.
        colorize_factors :
          If ``True``, use colors from the colormap to colorize the factors; otherwise
          all levels will have same color.
        alpha :
          Transparency of the points.
        colormap :
          Colormap to use instead of the default blue to red colormap.
        figsize :
          Figure size - passed directly to ``matplotlib``.
        jitter :
          Amount of jitter used to show the point density.
        chart_title : str
          Chart title.
        x_label : str
          Chart x-axis label.
        y_label : str
          Chart y-axis label.
        thermometer_label : str
          Chart thermometer label.
        drop_zero_contribs :
          Whether to drop features that have zero contribution. Features that are not
          used in the final model will have zero contribution.
        hard_asserts : bool
          Used in testing to raise exception in try except statements.
        logger :
          Optional logger object.

        Returns
        -------
        pyplot.Figure :
          A ``matplotlib`` figure object which can be saved or displayed.

        """
        # TODO trim frame columns using contributions
        # TODO sample rows when over limit
        # TODO limit features (limited but not based on varimp yet)
        # TODO use columns parameter
        # TODO implement colorize for categoricals

        colormap = commons.LookAndFeel.get_colormap(colormap)

        non_zero_df = contributions.loc[:, (contributions != 0).any(axis=0)]
        # only drop features with zero contribs if contrib df is not all zero
        if drop_zero_contribs and not non_zero_df.empty:
            # drop features that were not used in final model (all contribs=0)
            contributions = non_zero_df

        # sum up contributions of each feature and sort highest to lowest
        cols = contributions.abs().sum(0).sort_values(ascending=False).index

        pyplot.figure(figsize=figsize)
        pyplot.grid(True)
        pyplot.axvline(0, c="black")
        if top_n_features > 0:
            cols = cols[:top_n_features]
        # we reverse the list so that top features are at top of plot (i.e.,
        # i doesn't start at 0 and increase, but instead decreases.
        ytick_index_list = []
        plot_col_name_list = []
        for i, plot_col_name in zip(range(len(cols), -1, -1), cols, strict=False):
            try:
                pd_series = frame[plot_col_name]
                col = contributions[plot_col_name]
                dens = ScatterFeatImpPlot._density(col)
                color_values = (
                    ScatterFeatImpPlot._uniformize(pd_series)
                    if colorize_factors or is_numeric_dtype(pd_series)
                    else numpy.full(pd_series.shape, -1.0)
                )
                if color_values is None:
                    # _uniformize returns None for non-numeric (string) columns;
                    # fall back to the grey sentinel so points still render
                    color_values = numpy.full(pd_series.shape, -1.0)
                c = (color_values,)
                pyplot.scatter(
                    col,
                    i + dens * numpy.random.uniform(-jitter, jitter, size=len(col)),
                    alpha=alpha,
                    c=c,
                    cmap=colormap,
                )
                pyplot.clim(0, 1)
            except Exception as ex:
                msg = (
                    f"Shapley plot rendering failed with {ex}:"
                    f"\n{traceback.format_exc()}"
                )
                if logger:
                    logger.error(msg)
                    if hard_asserts:
                        raise
            # get ordered ytick index and labels, skipping problematic cols.
            ytick_index_list.append(i)
            plot_col_name_list.append(plot_col_name)
        colormap.set_under(color="grey")
        cbar = pyplot.colorbar()
        cbar.set_label(thermometer_label, rotation=270)
        cbar.ax.get_yaxis().labelpad = 15
        # special characters will break the pyplot.yticks
        plot_col_name_list = safe_plot_names(plot_col_name_list)
        pyplot.yticks(ticks=ytick_index_list, labels=plot_col_name_list)
        pyplot.xlabel(x_label)
        pyplot.ylabel(y_label)
        pyplot.title(chart_title)
        pyplot.tight_layout()
        fig = pyplot.gcf()

        return fig

    @staticmethod
    def _density(xs: numpy.ndarray, bins: int = 100) -> numpy.ndarray:
        """Make an approximate density estimation by blurring a histogram:

        Parameters
        ----------
        xs: np.ndarray
          Numpy vector.
        bins: int
          Number of bins.

        Returns
        -------
        np.ndarray:
          Density values.

        """
        hist = list(numpy.histogram(xs, bins=bins))
        # gaussian blur
        hist[0] = numpy.convolve(
            hist[0],
            [
                0.00598,
                0.060626,
                0.241843,
                0.383103,
                0.241843,
                0.060626,
                0.00598,
            ],
        )[3:-3]
        hist[0] = hist[0] / numpy.max(hist[0])
        hist[1] = (hist[1][:-1] + hist[1][1:]) / 2
        return numpy.interp(xs, hist[1], hist[0])

    @staticmethod
    def _uniformize(col) -> numpy.ndarray | None:
        """Convert to quantiles.

        Parameters
        ----------
        col: pandas.core.series.Series
          Pandas series with a column name.

        Returns
        -------
         np.ndarray | None :
          Quantile values of individual points in the column.

        """
        if not is_numeric_dtype(col.dtype):
            return None

        xs = numpy.linspace(0, 1, 100)
        quantiles = numpy.nanquantile(col.astype(float), xs)
        res = numpy.interp(col, quantiles, xs)
        col_min, col_max = numpy.nanmin(res), numpy.nanmax(res)
        if col_min == col_max:
            # constant-value column: no color variation possible; use the
            # under-range sentinel so points render in grey via set_under()
            return numpy.full(len(col), -1.0)
        res = (res - col_min) / (col_max - col_min)
        return res


class Data3dPlot:
    """Plot 3D data:

    - heatmap
    - 3D surface plot
    - 3D contour plot

    """

    PLOT_TYPE_HEATMAP = "heatmap"
    PLOT_TYPE_CONTOUR = "contour-3d"
    PLOT_TYPE_SURFACE = "surface-3d"

    PLOT_TYPES = [
        PLOT_TYPE_HEATMAP,
        PLOT_TYPE_CONTOUR,
        PLOT_TYPE_SURFACE,
    ]

    @staticmethod
    def plot(
        x_axis_labels: list,
        y_axis_labels: list,
        heatmap_data: datatable.Frame,
        chart_title: str = "",
        x_axis_label: str = "",
        y_axis_label: str = "",
        plot_type: str = PLOT_TYPE_HEATMAP,
        color_map: str = "autumn",
        figsize=(12, 10),
        dpi=120,
        plot_file_path: str = "",
        logger=None,
        log_name: str = "",
    ):
        """Heatmap plot.

        Parameters
        ----------
        x_axis_labels : list
          Horizontal axes labels.
        y_axis_labels : list
          Vertical axes labels.
        heatmap_data : datatable.Frame
          Datable frame with heatmap data (column names don't matter, only data are
          relevant).
        chart_title : str
          Chart title.
        x_axis_label : str
          Horizontal axis label.
        y_axis_label : str
          Vertical axis label.
        plot_type : str
          Plot type, one of PLOT_TYPES.
        color_map : str
          Matplotlib color map name.
        figsize : tuple
          Figure size.
        dpi : int
          Dots per inch.
        plot_file_path : str
          Path to save the plot to.
        logger :
          Logger instance.
        log_name : str
          Name of the logger.

        """
        # this plot type implementation is the most stable 3D data plot in H2O Sonar
        # with fixed ticks/labels/data handling and customization

        # DO plot
        try:
            # TICS: axes points
            x_tics = numpy.arange(len(x_axis_labels))
            y_tics = numpy.arange(len(y_axis_labels))
            z_tics = heatmap_data.to_numpy()
            # DATA: from tics to matrices w/ the same shape as z-data
            z_data = z_tics
            x_data = numpy.array([x_tics for _ in range(z_data.shape[0])])
            y_data = numpy.array([y_tics for _ in range(z_data.shape[1])]).T

            # PLOT
            pyplot.figure(figsize=figsize, dpi=dpi)
            if plot_type == Data3dPlot.PLOT_TYPE_HEATMAP:
                ax = pyplot.axes()
            else:
                ax = pyplot.axes(projection="3d")

            if chart_title:
                ax.set_title(chart_title)

            # set plot bins count:
            #   required by Matplotlib 3.4.3 for even axis bins distribution
            pyplot.locator_params(axis="x", nbins=len(x_axis_labels))
            pyplot.locator_params(axis="y", nbins=len(y_axis_labels))
            # SET TICS
            ax.set_xticks(x_tics)
            ax.set_yticks(y_tics)
            # SET tics LABELS
            ax.set_xticklabels(x_axis_labels)
            ax.set_yticklabels(y_axis_labels)
            # axes legend
            ax.set_xlabel(x_axis_label)
            ax.set_ylabel(y_axis_label)

            if plot_type == Data3dPlot.PLOT_TYPE_CONTOUR:
                ax.contour3D(x_data, y_data, z_data, 150, cmap=color_map)
            elif plot_type == Data3dPlot.PLOT_TYPE_HEATMAP:
                # ROTATE the tick LABELS and set their alignment.
                pyplot.setp(
                    ax.get_xticklabels(),
                    rotation=45,
                    ha="right",
                    rotation_mode="anchor",
                )
                pyplot.setp(
                    ax.get_yticklabels(),
                    rotation=45,
                    ha="right",
                    rotation_mode="anchor",
                )

                pyplot.imshow(z_tics, cmap=color_map, aspect="auto")
                pyplot.colorbar()
            elif plot_type == Data3dPlot.PLOT_TYPE_SURFACE:
                ax.plot_surface(
                    x_data, y_data, z_data, cmap=color_map, edgecolor="black"
                )
            else:
                if logger:
                    logger.error(
                        f"{log_name}: unable to render unknown plot type "
                        f"'{plot_type}' - valid plot types are "
                        f"{Data3dPlot.PLOT_TYPES}"
                    )

            if plot_file_path:
                pyplot.savefig(plot_file_path, dpi=300)

            # CLEAR figure to get ready for next plot
            pyplot.clf()
        except Exception as ex:
            if logger:
                logger.error(
                    f"{log_name}: rendering of the heatmap {plot_type} failed "
                    f"with: {ex}\n{traceback.format_exc()}",
                )
            raise ex
