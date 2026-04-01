# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json as js
import os
import statistics as stat
import traceback
from collections.abc import Callable
from typing import Any

import datatable
import numpy as np
import pandas as pd

from h2o_sonar import errors
from h2o_sonar import loggers
from h2o_sonar.lib.api import persistences
from h2o_sonar.methods._abstract_ice_pd import AbstractIcePd
from h2o_sonar.methods._ice import ICE
from h2o_sonar.methods._pd_cache import PdCache
from h2o_sonar.methods.core import _mli
from h2o_sonar.methods.core import method
from h2o_sonar.utils import binning
from h2o_sonar.utils import progress


class PD(method.Method, AbstractIcePd, PdCache):
    """Implementation of Partial Dependence (PD) methods. Partial dependence
    explains the marginal effect of a target feature(s) on the predicted outcome of
    a previously fit model.

    PD calculation: the prediction function is fixed at a few values of the chosen
    target feature(s) and averaged over the other features. As PD is the average of
    the local ICEs for all instances, PD calculation implementation is be based on ICE.

    PD binning: see ICE for binning and out of range binning documentation.

    PD result: dictionary with item for every target feature. Item key is feature
    name and item value is Pandas DataFrame. DataFrame columns correspond to values
    where PD was computed. DataFrame has 3 rows: PD values (averaged ICE), standard
    deviation (of ICE) and unbiased standard error of the mean (for ICE).

    Examples
    --------
    >>> explanations = PD("Fluent").explain(
    ...     ['Feature'],
    ...     X,
    ...     predict_method=scorer).explanations()
    >>>
    >>> explanations = PD("PD and identity() residuals PD").explain(
    ...     ['Feature'],
    ...     X,
    ...     Y = actual_values,
    ...     predict_method=scorer).explanations()
    >>>
    >>> explanations = PD("PD and abs() residuals PD").explain(
    ...     ['Feature'],
    ...     X,
    ...     Y = actual_values,
    ...     predict_method=scorer,
    ...     target_transform=abs).explanations()
    >>>
    >>> explanations = PD("PD and out of range PD").explain(
    ...     ['Feature'],
    ...     X,
    ...     predict_method=scorer,
    ...     out_of_range_resolution=3).explanations()
    >>>
    >>> explanations = PD("Out of range PD only").explain(
    ...     ['Feature'], X,
    ...     predict_method=scorer,
    ...     grid_resolution=0,
    ...     out_of_range_resolution=3).explanations()
    >>>
    >>> explanations = PD("Load from JSon file").load_json(
    ...     "pd.json").explanations()
    >>>
    >>> pd = PD("Step by step PD calculation and persistence")
    >>> pd.explain(["Feature"], X, predict_method=scorer)
    >>> # ...
    >>> explanations = pd.explanations()
    >>> pd.save_json("pd.json")
    >>>
    >>> pd = PD("Single feature with bins")
    >>> pd.explain(['F1'], X, predict_method=scorer, bins=[1,2])
    >>> explanations = pd.explanations()
    >>> # ...
    >>> pd = PD("Multiple features with bins")
    >>> pd.explain(
    ...     ['F1','F2'],
    ...     X,
    ...     predict_method=scorer,
    ...     bins=[[1,2],[5,6,7]])
    >>> explanations = pd.explanations()
    >>>
    >>> pd = PD("Indicate that a feature is categorical")
    >>> pd.explain(
    ...     ['F1'],
    ...     X,
    ...     predict_method=scorer,
    ...     features_metadata = {'categorical': ['F1']})
    >>> explanations = pd.explanations()
    >>>
    >>> pd = PD("Ask for quantiles-based bin")
    >>> pd.explain(
    ...     ['F1', 'F2'],
    ...     X,
    ...     predict_method=scorer,
    ...     features_metadata = {'quantile-bin': {'F1': 2, 'F2': 5}}
    >>> explanations = pd.explanations()
    >>>
    >>> pd = PD("Ask for quantiles-based bin but use default grid resolution for all "
                variables")
    >>> pd.explain(
    ...     ['F1', 'F2'],
    ...     X,
    ...     predict_method=scorer,
    ...     features_metadata = {'quantile-bin': ['F1', 'F2']}
    >>> explanations = pd.explanations()
    >>>
    >>> pd = PD("Ask for quantiles-based and use same bin value for input variables")
    >>> features = ['F1', 'F2']
    >>> qtile_bins = [8] * len(features)
    >>> qtile_dict = dict(zip(features, qtile_bins))
    >>> pd.explain(
    ...     ['F1'],
    ...     X,
    ...     predict_method=scorer,
    ...     features_metadata = {'quantile-bin': qtile_dict}
    >>> explanations = pd.explanations()
    >>>
    >>> explanations = PD("Multidimensional PD").explain(
    ...     [('F1','F2'), 'F3'],
    ...     X,
    ...     predict_method=scorer).explanations()
    >>>
    >>> explanations = PD("Multidimensional PD").explain(
    ...     [('F1','F2'), 'F3'],
    ...     X,
    ...     predict_method=scorer,
    ...     bins=[([1, 2], [3, 4, 5]), [8, 9]]).explanations()
    >>>
    >>> pd = PD("ICE cached by PD").explain(
    ...     ['F1','F2'],
    ...     X,
    ...     predict_method=scorer,
    ...     ice_cache={
    ...            'F1': {'p_0': [0, 2]},
    ...            'F2': {'p_0': [1]}})
    >>> pdp_explanations = pd.explanations()
    >>> ice_explanations = pd.explanations(kind='ice')
    >>>
    >>> pd = PD("Bins constructed from features' unique values")
    >>> features = ['F1','F2']
    >>> bins = PD.create_unique_bins(features, X)
    >>> pd.explain(features, X, predict_method=scorer, bins=bins)
    >>> explanations = pd.explanations()
    >>>
    >>> pdp = PD("Fluent PD").load_json("pdp.json").explanations()
    >>> # ...
    >>> pdp = PD("Step by step PD loading")
    >>> pdp.load_json("pdp.json")
    >>> explanations = pdp.explanations()
    >>>
    >>> pd = PD("Date aware bins w/ format specification")
    >>> pd.explain(
    ...     ['DATE_INT','DATE_STR'],
    ...     X,
    ...     features_metadata = {
    ...         PD.KEY_DATE_FEATURES: ['DATE_INT','DATE_STR'],
    ...         PD.KEY_DATE_FEATURES_FORMAT: [
    ...             PD.DEFAULT_DATE_FEATURE_FORMAT,
    ...             "%Y-%m-%d",
    ...         ],
    ... })
    >>> explanations = pdp.explanations()

    See also
    --------
    https://pandas.pydata.org

    """

    EXPLAINER_TYPE = "pd"

    COL_MEAN = "mean"
    COL_SD = "sd"
    COL_SEM = "sem"
    COL_RESIDUAL_MEAN = "residuals_mean"
    COL_RESIDUAL_SD = "residuals_sd"
    COL_RESIDUAL_SEM = "residuals_sem"
    COL_OOR = "oor"

    @property
    def opt_1_prediction(self):
        """Option which controls ICE computation strategy for PD.

        If this attribute is set to ``False``, then predict method will be
        called for every feature's bin. If this option is set to ``True``,
        then ICE may call predict method only once per feature. See
        ICE documentation for more details.

        """
        return self._ice.opt_1_prediction

    @opt_1_prediction.setter
    def opt_1_prediction(self, allow_1_prediction: bool):
        self._ice.opt_1_prediction = allow_1_prediction

    def __init__(self, name, interpretable_model=None):
        """Create partial dependence instance.

        Parameters
        ----------
        name: str
            name of this partial dependence methods
        interpretable_model: InterpretableModel
            interpretable model whose predict_method
            and current directory should be used

        """
        method.Method.__init__(
            self,
            method_name=name,
            method_type=PD.EXPLAINER_TYPE,
            interpretable_model=interpretable_model,
        )
        AbstractIcePd.__init__(self)
        PdCache.__init__(self)
        self._ice = ICE("PD backend")
        self._bins = None
        self._stats = {"mins": None, "maxs": None, "stds": None}
        self._target_transform = None
        self._oor = None
        self._oor_catnum = []
        self._progress_callback: progress.AbstractProgressCallbackContext | None = None

    # IMPROVE: explain() method signatures don't fit yet
    def explain(
        self,
        features: list,
        X: Any,
        Y: Any = None,
        predict_method: Callable | None = None,
        bins: list | None = None,
        grid_resolution=method.Method.DEFAULT_GRID_RESOLUTION,
        features_meta: dict | None = None,
        target_transform=lambda x: x,
        out_of_range_resolution: int = 0,
        out_of_range_blacklist: list | None = None,
        stats: bool = True,
        center: bool = False,
        bins_sort: bool = False,
        ice_cache: list | dict | None = None,
        ice_cache_path: str = None,
        progress_callback: progress.AbstractProgressCallbackContext | None = None,
    ):
        """Calculate partial dependence and cache it in memory so that it can be
        subsequently obtained using function :func:`explanations() <h2o_sonar.
        methods.pdp.PD.explanations>`.

        Parameters
        ----------
        features: list[int or str or tuple]
            A list of features (strings or numerical ids) for which partial
            dependence is supposed to be calculated. If list item is tuple,
            then n-dimensional (where n is ``len(tuple)``) PD is calculated.
        X: pandas.core.frame.DataFrame or datatable.Frame
            Original data for which should be partial dependence computed.
        Y: pandas.core.frame.DataFrame or datatable.Frame
            Actual values - if provided, then PD on residuals is *also*
            calculated. The number of ``Y`` columns must correspond to the
            number of columns returned by predict method.
        predict_method: function
            A lambda function which takes instances (a set of rows) and
            outputs predictions as Pandas Series or DataFrame.
            If this parameter is not specified, then interpretable model's
            predict_method is used.
        bins: list[list[object]]
            Data values for each target feature for which we want to compute
            partial dependence, vector if for single target feature,
            otherwise a matrix.
        bins_sort: bool
            Ensure bins values sorting.
        grid_resolution: int
            The number of equally spaced points used to create bins if bins
            are not specified.
        features_meta: dict
            Features metadata allowing to indicate that given feature is
            categorical (use ``categorical`` key and list of feature names),
            (use ``date`` key and list of feature names, to specify format use
            ``date-format`` and list of Python date formats) or numerical
            (default).
            Use ``quantiles`` dict key and dict of feature names and quantiles to
            ensure bins construction using quantiles instead of even split. Using
            `None` value for quantiles will fallback to default grid resolution.
        target_transform: function
            A lambda to be applied on residuals - instead of identity
            consider e.g. ``abs()`` or ``sqrt()``
        out_of_range_resolution: int
            Calculate out of range values if `out_of_range_resolution`
            parameter is bigger than 0. For instance, if
            `out_of_range_resolution` is set to 3, then the
            expression below is used to construct six bins (2*3) where
            n = `[1, 2, 3]`.

            .. math::
            min(feature) - n*sd(feature)
            max(feature) + n*sd(feature)

        out_of_range_blacklist: list
            Features for which to skip OOR computation.
        stats: bool
            Compute standard deviation and standard error of the mean
            apart to partial dependence.
        center: bool
            Set this parameter to ``True`` to compute c-PD (centered PD).
        ice_cache: dict
            Use empty dictionary to cache all ICEs computed as a part of PD
            computation in memory. Use feature/class/list of instances to
            filter and cache ICE e.g. ``{'F': {'p_0': [3, 7]}`` to keep ICE for
            instances ``3`` and ``7`` of feature ``F``. Get cached values
            using ``explanations(kind='ice')``. Note that ICE is not cached
            by default.
        ice_cache_path: str
            File path where to store all computed ICEs - ``ice_cache`` filter is
            not used. If cache index file already exists, then new ICEs are
            appended.
        progress_callback : progress.AbstractProgressCallbackContext | None
            Progress callback allowing the progress of PD calculation.

        Returns
        -------
        h2o_sonar.methods.PD
            PD instance to get computed explanations using `explanations()`
            method.

        Raises
        ------
        MliUnsupportedDataFormatError
            If input parameters are not in expected format.
        MliUnsupportedOperationError
            If PD is required to be computed on unsupported feature types
        MliPredictMethodError
            If predict method fails.

        See also
        --------
        https://pandas.pydata.org
        https://github.com/h2oai/datatable

        """
        super()._check_and_set_features(features, features_meta)
        self.__check_out_of_range_resolution(
            out_of_range_resolution, out_of_range_blacklist
        )
        predict_method = super()._method_precondition(predict_method)
        super()._check_resolution(X, grid_resolution)
        self._bins = bins
        self._target_transform = (
            (lambda x: x) if not target_transform else target_transform
        )
        self._center = center
        self._stats = stats
        self._bins_sort = bins_sort

        self.diagnostics.add_scorer_calls_slot()

        self._progress_callback = progress_callback

        if Y is not None:
            if isinstance(Y, datatable.Frame):
                Y = _mli.InterpretableModel.to_pandas(Y)
            elif not isinstance(Y, pd.DataFrame):
                raise errors.MliUnsupportedDataFormatError(
                    f"Unsupported Y data type: {type(Y)}",
                    "Use Pandas DataFrame or datatable Frame",
                )
            # Y columns count must correspond to predict method return value
            if X.shape[0] != Y.shape[0]:
                raise ValueError(
                    f"X's and Y's number of rows must fit, but it is "
                    f"{X.shape[0]} and {Y.shape[0]}"
                )
        else:
            if not bins and not grid_resolution and not out_of_range_resolution:
                raise ValueError(
                    "No bins, resolution, Y and out of range - nothing to do"
                )

        self._extra_explanations_cache = self._check_ice_cache(
            self._fs, ice_cache, ice_cache_path
        )

        if isinstance(X, pd.DataFrame):
            if X.empty:
                raise ValueError("Data cannot be empty")
            return self._explain_pdp(X=X, Y=Y, predict_method=predict_method)
        if isinstance(X, datatable.Frame):
            X = _mli.InterpretableModel.to_pandas(X)
            if X.empty:
                raise ValueError("Data cannot be empty")
            return self._explain_pdp(X=X, Y=Y, predict_method=predict_method)

        raise errors.MliUnsupportedDataFormatError(
            f"Unsupported X data type: {type(X)}",
            "Use Pandas DataFrame or datatable Frame",
        )

    def _explain_pdp_cache_ice(self, feature, ice):
        if self._ice_cache is not None:
            if self._ice_cache and feature in self._ice_cache:
                self._extra_explanations_cache[PdCache.KEY_ICE_CACHE][feature] = {}
                for clazz in self._ice_cache[feature]:
                    if clazz in ice[feature]:
                        self._extra_explanations_cache[PdCache.KEY_ICE_CACHE][feature][
                            clazz
                        ] = ice[feature][clazz].take(self._ice_cache[feature][clazz])
            else:
                self._extra_explanations_cache[PdCache.KEY_ICE_CACHE][feature] = ice[
                    feature
                ]

    def _explain_pdp(self, X, Y, predict_method):
        """Pandas DataFrame based Partial Dependence (PD) implementation.

        Parameters
        ----------
        X: pandas.core.frame.DataFrame
            The original data for which we want to compute partial dependence.
        Y: pandas.core.frame.DataFrame
            Actual values - if provided, then PD on residuals is also
            calculated
        predict_method: function
            A lambda which takes instances (a set of rows) and outputs
            predictions

        Returns
        -------
        h2o_sonar.methods.PD
            PD instance to get computed explanations using `explanations()`
            method.

        Raises
        ------
        MliPredictMethodError
            If predict method fails.
        MliUnsupportedOperationError
            If PD is required to be computed on unsupported feature types

        See also
        --------
        https://pandas.pydata.org

        """
        super()._check_dataset_features(X)

        self._explanations_cache = {}
        self._json_cache = {}

        self.__init_stats(X)

        progress_callback = (
            progress.ProgressCallbackContext(
                total_steps=len(self._fs),
                parent_callback=self._progress_callback,
            )
            if self._progress_callback
            else None
        )

        for i, feature in enumerate(self._fs):
            # PERFORMANCE: ICE is intentionally computed per feature
            # (not for all features at once to fit in memory in case of big
            # datasets, consider 1G examples x 10 bins x 100 feats (~1TB)
            # vs 1G x 10 bins loop (10GB RAM)

            # IMPROVE: turn feature loop to parallel tasks when the time comes (opt)

            # MAP(X, pm)
            c_ices = self._ice.explain(
                [feature],
                X,
                predict_method=predict_method,
                bins=[self._bins[i]] if self._bins else None,
                bins_sort=self._bins_sort,
                mins=[self._stats["mins"][i]] if self._stats["mins"] else None,
                maxs=[self._stats["maxs"][i]] if self._stats["maxs"] else None,
                out_of_range_resolution=self._oor,
                out_of_range_blacklist=self._oor_blacklist,
                stds=[self._stats["stds"][i]] if self._oor else None,
                features_meta=self._fs_meta,
                grid_resolution=self._g_resolution,
                progress_callback=(
                    progress.ProgressCallbackStackingBridge(progress_callback)
                    if progress_callback
                    else None
                ),
            ).explanations()

            if self._ice_cache_path is not None:
                self._ice.save(self._ice_cache_path, append=True)

            self.diagnostics.add_scorer_calls(
                self._ice.diagnostics.scorer_calls_history[-1]
            )

            # REDUCE(ICE, mean/pd/sem)
            self._explanations_cache[feature] = {}
            for col, clazz in enumerate(c_ices[feature]):
                cols = (
                    [PD.COL_MEAN, PD.COL_SD, PD.COL_SEM]
                    if self._stats
                    else [PD.COL_MEAN]
                )
                feature_pdp = pd.DataFrame(
                    index=c_ices[feature][clazz].columns.values, columns=cols
                )

                # PD
                feature_pdp[PD.COL_MEAN] = c_ices[feature][clazz].mean()
                if self._stats:
                    feature_pdp[PD.COL_SD] = c_ices[feature][clazz].std()
                    feature_pdp[PD.COL_SEM] = c_ices[feature][clazz].sem()
                    feature_pdp[PD.COL_OOR] = False

                if self._center:
                    feature_pdp[PD.COL_MEAN] = PD._center_column(
                        feature_pdp[PD.COL_MEAN]
                    )

                # PD on residuals
                self.__explain_residuals(
                    ices=c_ices,
                    feature=feature,
                    clazz=clazz,
                    col=col,
                    Y=Y,
                    feature_pdp=feature_pdp,
                )

                if feature not in self._oor_blacklist:
                    self.__add_oor_result_row(
                        feature=feature,
                        feature_pdp=feature_pdp,
                        oor_bins=self._ice.oor_bins,
                    )

                self._explanations_cache[feature][clazz] = feature_pdp.transpose()

            # cache ICE
            self._explain_pdp_cache_ice(feature=feature, ice=c_ices)

        # evict cached ICE to safe memory
        self._ice.evict_explanations()

        if progress_callback:
            progress_callback.set_progress(1.0)

        # fluent API
        return self

    def __add_oor_result_row(self, feature, feature_pdp, oor_bins):
        if self._oor:
            bins = feature_pdp.index.values
            oor_bins = oor_bins[feature] if oor_bins and feature in oor_bins else []
            for row in range(feature_pdp.shape[0]):
                if bins is not None and row < len(bins) and bins[row] in oor_bins:
                    feature_pdp[PD.COL_OOR].iat[row] = True
                else:
                    feature_pdp[PD.COL_OOR].iat[row] = False

    def __explain_residuals(self, ices, feature, clazz, col, Y, feature_pdp):
        if Y is not None:
            try:
                col = 0 if len(Y.columns) == 1 else col
                residuals = -ices[feature][clazz].sub(Y[Y.columns[col]], axis=0)
                residuals = residuals.apply(self._target_transform)

                feature_pdp[PD.COL_RESIDUAL_MEAN] = residuals.mean()
                if self._stats:
                    feature_pdp[PD.COL_RESIDUAL_SD] = residuals.std()
                    feature_pdp[PD.COL_RESIDUAL_SEM] = residuals.sem()

                if self._center:
                    feature_pdp[PD.COL_RESIDUAL_MEAN] = PD._center_column(
                        feature_pdp[PD.COL_RESIDUAL_MEAN]
                    )
            except Exception as ex:
                raise errors.MliError(
                    f"ICE residuals calculation failed for feature `{feature}`, "
                    f"class '{clazz}', column '{col}', labels {Y} "
                    f"{Y.shape if Y is not None else '(0, 0)'} with message: {ex}:\n"
                    f"{traceback.format_exc()}"
                )

    def __check_out_of_range_resolution(self, oor_resolution: int, oor_blacklist):
        if oor_resolution:
            if not isinstance(oor_resolution, int):
                raise TypeError("Out of range parameter must be integer")
            self._oor = oor_resolution

        # IMPROVE consider check ensuring valid feature names in blacklist
        self._oor_blacklist = oor_blacklist if oor_blacklist else []

    def __init_stats(self, X):
        """Compute minima, maxima (and standard deviation if OOR to be
        computed) either from dataset or bins. PD calls ICE which
        constructs bins (in case they are not present) using maxima and
        minima provided by PD.

        If bins are specified, then bins to calculate stats, else use dataset
        to calculate it.

        Do NOT construct bins - ICE will do that (DRY).

        """
        self._stats = {"mins": [], "maxs": [], "stds": []}

        # process feature by feature: normal/n-dim, num/cat, ...
        for i, feature in enumerate(self._fs):
            if isinstance(feature, tuple):
                t_mins = []
                t_maxs = []
                t_stds = []
                for t_i, t_feature in enumerate(feature):
                    _bin = None if not self._bins else self._bins[i][t_i]
                    t_min, t_max, t_std = self.__init_stats_feature(t_feature, _bin, X)
                    t_mins.append(t_min)
                    t_maxs.append(t_max)
                    t_stds.append(t_std)
                self._stats["mins"].append(tuple(t_mins))
                self._stats["maxs"].append(tuple(t_maxs))
                self._stats["stds"].append(tuple(t_stds))
            else:
                _bin = None if not self._bins else self._bins[i]
                t_min, t_max, t_std = self.__init_stats_feature(feature, _bin, X)
                self._stats["mins"].append(t_min)
                self._stats["maxs"].append(t_max)
                self._stats["stds"].append(t_std)

    def __init_stats_feature(self, feature, bin_, X):
        assert not isinstance(feature, tuple), "Non n-dim feature required"

        f_min = None
        f_max = None
        f_std = None

        if not self._bins:
            # if there are no bins and feature is cat, then use a FOO value
            # as ICE will ignore maxs/mins and compute bins for PD
            if not method.Method._is_categorical(X[feature]):
                # if there are no bins & no cat feature, frame to give mins/maxs
                f_min = X[feature].min()
                f_max = X[feature].max()
                if self._oor:
                    if self._check_can_std(feature, f_min, f_max):
                        f_std = X[feature].std()
        else:
            # if there are bins, then use bins to initialize mins/maxs
            none_free_bin = self.strip_none(bin_)
            f_min = min(none_free_bin)
            f_max = max(none_free_bin)
            if self._oor:
                if self._check_can_std(feature, f_min, f_max):
                    f_std = stat.stdev(none_free_bin)

        return f_min, f_max, f_std

    def _check_can_std(self, feature, f_min, f_max) -> bool:
        if f_min == f_max:
            loggers.warn(
                f"Constant feature has no standard deviation - OOR "
                f"will not be computed for feature '{feature}'"
            )
            self._oor_blacklist.append(feature)
            return False

        return True

    @staticmethod
    def _center_column(column):
        return column - column.mean()

    def merge(self, pd_to_merge) -> list:
        """Merge other PD results to this one.

        Parameters
        ----------
        pd_to_merge : PD
          Other PD instance to be merge into this one.

        Returns
        -------
        list :
          List of merged features.

        """
        merged_features: list = []
        self._fs = self._fs or []
        if pd_to_merge and pd_to_merge.features and pd_to_merge.explanations():
            explanations_to_merge = pd_to_merge.explanations()
            for feature in explanations_to_merge.keys():
                if self._explanations_cache is None:
                    self._explanations_cache = explanations_to_merge
                else:
                    self._explanations_cache[feature] = explanations_to_merge[feature]
                self.features.append(feature)
                merged_features.append(feature)

        return merged_features

    JSON_COLS = "cols"
    JSON_COLUMNS = "columns"
    JSON_NBINS = "nbins"
    JSON_PD_DATA = "partial_dependence_data"
    JSON_NAME = "name"
    JSON_DESC = "description"
    JSON_TYPE = "type"
    JSON_ROWCOUNT = "rowcount"
    JSON_DATA = "data"
    JSON_BINS = "bins"
    JSON_LABEL_MEAN = "mean_response"
    JSON_LABEL_SD = "stddev_response"
    JSON_LABEL_MSE = "std_error_mean_response"
    JSON_LABEL_R_MEAN = "residuals_mean_response"
    JSON_LABEL_R_SD = "residuals_stddev_response"
    JSON_LABEL_R_MSE = "residuals_std_error_mean_response"
    JSON_LABEL_OOR = "oor_response"

    def to_json(self):
        if not self._explanations_cache:
            return {}

        persistences.JsonPersistableExplanations.check_explanations_serializability(
            self._explanations_cache
        )

        if not self._json_cache:
            # build
            json = {
                PD.JSON_COLS: self._fs,
                PD.JSON_NBINS: self._g_resolution,
                PD.JSON_PD_DATA: [],
            }
            # nbins is supposed to be default number of bins
            for feature in self._fs:
                json_f = {
                    PD.JSON_NAME: "PartialDependencePlot",
                    PD.JSON_DESC: f"Partial Dependence Plot on column '{feature}'",
                    PD.JSON_COLUMNS: [],
                }

                if (
                    self._fs_meta
                    and PD.KEY_CATEGORICAL_FEATURES in self._fs_meta
                    and feature in self._fs_meta[PD.KEY_CATEGORICAL_FEATURES]
                ):
                    f_type = "string"
                else:
                    f_type = "double"

                json_f[PD.JSON_COLUMNS].append(
                    {
                        PD.JSON_NAME: feature.lower(),
                        PD.JSON_DESC: feature,
                        PD.JSON_TYPE: f_type,
                    }
                )
                for i in [
                    PD.JSON_LABEL_MEAN,
                    PD.JSON_LABEL_SD,
                    PD.JSON_LABEL_MSE,
                    PD.JSON_LABEL_OOR,
                ]:
                    json_f[PD.JSON_COLUMNS].append(
                        {
                            PD.JSON_NAME: i,
                            PD.JSON_DESC: i,
                            PD.JSON_TYPE: "double",
                        }
                    )

                if len(self._explanations_cache[feature]) == 2:
                    positive_clazz: str = PD.LABEL_PREFIX_CLASS + str(1)
                else:
                    positive_clazz = PD.LABEL_REGRESSION

                if positive_clazz not in self._explanations_cache[feature]:
                    raise NotImplementedError(
                        "JSon serialization for multinomial experiment type "
                        "predictions is not supported."
                    )

                exs = self._explanations_cache[feature][positive_clazz]
                if len(exs.index) == 7:
                    for i in [
                        PD.JSON_LABEL_R_MEAN,
                        PD.JSON_LABEL_R_SD,
                        PD.JSON_LABEL_R_MSE,
                    ]:
                        json_f[PD.JSON_COLUMNS].append(
                            {PD.JSON_NAME: i, PD.JSON_DESC: i}
                        )

                json_f[PD.JSON_ROWCOUNT] = len(exs.columns)

                json_f[PD.JSON_DATA] = []
                # bins
                # original bin item types must be protected to keep consistency
                json_f[PD.JSON_BINS] = exs.columns.values.tolist()
                # bins MUST have int/str type for MLI@DAI UI to decide cat/num
                json_f[PD.JSON_DATA].append(
                    PD.__to_json_bins(exs.columns.values.tolist(), f_type)
                )
                # PD, SD, SEM and OOR
                pd_data = exs.values.tolist()
                json_f[PD.JSON_DATA].extend(pd_data)

                json[PD.JSON_PD_DATA].append(json_f)

            # cache JSon
            self._json_cache = json

        return self._json_cache

    @staticmethod
    def __to_json_bins(bins: list, feature_type: bool):
        try:
            if feature_type == "string":
                return list(map(str, bins))
            return list(map(float, bins))
        except ValueError:
            loggers.warn(
                f"Unable to convert bins to {feature_type} based "
                f"on feature type: {bins}"
            )
        return bins

    def __resolve_json_path(self, path):
        if path is None:
            if self._i_model is None:
                path = self.default_json_file_name
            else:
                path = os.path.join(
                    self._i_model.mli.work_dir, self.default_json_file_name
                )
        return path

    # override
    def save_json(self, path=None):
        """WARNING: PD (de)serialization can be used only in case of regression
        prediction type.

        Save cached PD explanations as JSon file.

        Parameters
        ----------
        path: str
            Local file path where to store explanations. If path isn't
            specified, then explanations are stored to 'explanations.json' in
            the MLI working or current

        Returns
        -------
        h2o_sonar.methods.PD
            PD instance.

        """
        path = self.__resolve_json_path(path)

        if not self._explanations_cache:
            raise errors.MliJsonSerializationError(
                "No explanations - run explain() method first"
            )

        self.to_json()

        # save
        self._save_json(self._json_cache, path)

        # fluent API
        return self

    # override
    def load_json(self, path=None):
        """WARNING: PD (de)serialization can be used only in case of regression
        prediction type.

        Load cached PD explanations from a JSon file.

        Parameters
        ----------
        path: str
            Local file path where to store explanations. If path isn't
            specified, then explanations are stored to 'explanations.json' in
            the MLI working or current directory
        h2o_sonar.methods.PD
            PD instance

        """
        path = self.__resolve_json_path(path)

        if not os.path.exists(path):
            raise FileNotFoundError("File with explanations doesn't exist: " + path)

        with open(path, encoding="utf-8") as fp:
            json = js.load(fp)

            if not isinstance(json, dict):
                raise errors.MliJsonDeserializationError(
                    "Root object of JSon file to be 'dict': " + path
                )
            if not isinstance(json[PD.JSON_COLS], list):
                raise errors.MliJsonDeserializationError(
                    "'cols' value to be 'list' in JSon file: " + path
                )
            if not isinstance(json[PD.JSON_PD_DATA], list):
                raise errors.MliJsonDeserializationError(
                    "PD data value to be 'list' in JSon file: " + path
                )

            self._json_cache = json
            self._explanations_cache = {}
            frame_idx = [PD.COL_MEAN, PD.COL_SD, PD.COL_SEM, PD.COL_OOR]
            if (
                json[PD.JSON_PD_DATA]
                and len(json[PD.JSON_PD_DATA][0][PD.JSON_DATA]) >= 6
            ):
                # IMPROVE if not stat, than not all columns might be available
                frame_idx.extend(
                    [
                        PD.COL_RESIDUAL_MEAN,
                        PD.COL_RESIDUAL_SD,
                        PD.COL_RESIDUAL_SEM,
                    ]
                )

            for i, feature in enumerate(json[PD.JSON_COLS]):
                # IMPROVE: consider type check on PD entries: safety vs. perf.
                frame = pd.DataFrame(
                    data=[
                        json[PD.JSON_PD_DATA][i][PD.JSON_DATA][d]
                        for d in range(1, len(json[PD.JSON_PD_DATA][i][PD.JSON_DATA]))
                    ],
                    index=frame_idx,
                    columns=json[PD.JSON_PD_DATA][i][PD.JSON_BINS],
                )

                self._explanations_cache[feature] = {PD.LABEL_REGRESSION: frame}

        # fluent API
        return self

    @staticmethod
    def create_unique_bins(
        features,
        X,
        features_meta=None,
        grid_resolution=method.Method.DEFAULT_GRID_RESOLUTION,
    ):
        """Create bins using unique feature values where possible. As fallback
        use grid resolution.

        Parameters
        ----------
        X: pandas.core.frame.DataFrame
            Original data for which should be partial dependence computed.
        features: list[int or str]
            A list of features for which partial bins should be created.
        grid_resolution: int
            The number of equally spaced points used to create bins if the
            number of unique values is big.
        features_meta: dict
            Features metadata allowing to indicate whether given feature is
            categorical (use ``categorical`` key and list of feature names).
            Use ``quantiles`` dict key and dict of feature names and quantiles to
            ensure bins construction using quantiles instead of even split. Using
            `None` value for quantiles will fallback to default grid resolution.

        Returns
        -------
        bins: list[list[object]]
            Data values for each target feature for which we want to compute
            partial dependence, vector if for single target feature,
            otherwise a matrix.

        """
        if isinstance(X, pd.DataFrame):
            if X.empty:
                raise ValueError("Data cannot be empty")
        else:
            raise errors.MliUnsupportedDataFormatError(
                f"Unsupported X data type: {type(X)}",
                "Use Pandas DataFrame",
            )
        if not features:
            raise ValueError("At least one feature must be specified")
        if grid_resolution < 1:
            raise ValueError("Grid resolution must be positive integer")

        bins = []
        for feature in features:
            if feature not in X.columns.values:
                raise ValueError(
                    f"Feature '{feature}' is not label of any input data column"
                )

            x_uniq = X[feature].unique()
            if x_uniq.size <= grid_resolution:
                bins.append(x_uniq.tolist())
            else:
                if (
                    features_meta
                    and PD.KEY_CATEGORICAL_FEATURES in features_meta
                    and feature in features_meta[PD.KEY_CATEGORICAL_FEATURES]
                ) or method.Method._is_categorical(X[feature]):
                    # make most frequent cats bins
                    bins.append(
                        X.groupby(feature, dropna=False)
                        .size()
                        .sort_values(ascending=False)
                        .head(grid_resolution)
                        .index.tolist()
                    )
                else:
                    PD.__create_unique_numerical_bins(
                        feature=feature,
                        X=X,
                        features_meta=features_meta,
                        bins=bins,
                        grid_resolution=grid_resolution,
                    )
        return bins

    @staticmethod
    def __create_unique_numerical_bins(
        feature, X, features_meta, bins, grid_resolution
    ):
        if features_meta and PD.KEY_QUANTILE_BINS in features_meta:
            if isinstance(features_meta[PD.KEY_QUANTILE_BINS], list):
                features_meta[PD.KEY_QUANTILE_BINS] = dict(
                    zip(
                        features_meta[PD.KEY_QUANTILE_BINS],
                        [None] * len(features_meta[PD.KEY_QUANTILE_BINS]),
                        strict=False,
                    )
                )
            if feature in features_meta[PD.KEY_QUANTILE_BINS].keys():
                quantile = features_meta[PD.KEY_QUANTILE_BINS].get(feature)
                binning.build_qtile_bins(
                    bins=bins,
                    X=X,
                    feature=feature,
                    quantile=quantile if quantile else PD.DEFAULT_GRID_RESOLUTION,
                )
            else:
                PD.build_numerical_bins(X, bins, feature, grid_resolution)
        else:
            PD.build_numerical_bins(X, bins, feature, grid_resolution)

    @staticmethod
    def build_numerical_bins(X, bins, feature, grid_resolution):
        min_f = X[feature].min()
        max_f = X[feature].max()
        resolution = 1 if grid_resolution == 1 else grid_resolution - 1
        step = (max_f - min_f) / resolution
        if min_f == max_f:
            bins.append([max_f])
        elif step <= 0:
            bins.append([(min_f + max_f) / 2.0])
        else:
            bins.append(np.append(np.arange(min_f, max_f, step), max_f).tolist())
