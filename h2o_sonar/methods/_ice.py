# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import math
import os
from collections import OrderedDict

import datatable
import numpy
import pandas
import psutil

from h2o_sonar import errors
from h2o_sonar import loggers
from h2o_sonar.lib.api import commons
from h2o_sonar.methods import _abstract_ice_pd
from h2o_sonar.methods.core import _mli
from h2o_sonar.methods.core import method
from h2o_sonar.methods.utils import histogram
from h2o_sonar.utils import binning
from h2o_sonar.utils import progress


try:
    import h2o

    HAS_H2O = True
except ImportError:
    HAS_H2O = False


class ICE(method.Method, _abstract_ice_pd.AbstractIcePd):
    """Implementation of Individual Conditional Expectation (ICE) methods.
    For a given instance and feature, ICE explains how the instance’s prediction
    changes when the feature changes.

    ICE calculation: the ICE values for given instance are computed by leaving
    all other features than the target feature the same, creating variants of
    this instance by replacing the feature’s value with values from the dataset
    and making the predictions with these newly created instances. The result
    is a set of points for an instance with the feature value from the dataset
    and the respective predictions.

    ICE result: dictionary with item for every target feature. Item key
    is feature name and item value is Pandas DataFrame. DataFrame columns
    correspond to values where ICE was computed. DataFrame rows correspond
    to dataset instances. Values in DataFrame rows are ICE values - prediction
    for given value and instance.

    ICE binning:

    **Integer** feature:
    * bins in **numeric** mode:
        * bins are integers
        * (at most) `grid_resolution` integer values in between minimum and maximum
          of feature values
        * bin values are created as evenly as possible
        * minimum and maximum is included in bins
          (if `grid_resolution` is bigger or equal to 2)
    * bins in **categorical** mode:
        * bins are integers
        * top `grid_resolution` values from feature values ordered by _frequency_
          (int values are converted to strings and most frequent values are used
          as bins)

    **Float** feature:
    * bins in **numeric** mode:
        * bins are floats
        * `grid_resolution` float values in between minimum and maximum of feature
           values
        * bin values are created as evenly as possible
        * minimum and maximum is included in bins
          (if `grid_resolution` is bigger or equal to 2)
    * bins in **categorical** mode:
        * bins are floats
        * top `grid_resolution` values from feature values ordered by _frequency_
          (float values are converted to strings and most frequent values are used
          as bins)

    **String** feature:
    * bins in **numeric** mode:
        * not supported
    * bins in **categorical** mode:
        * bins are strings
        * top `grid_resolution` values from feature values ordered by _frequency_

    **Date/datetime** feature:
    * bins in **numeric** mode:
        * bins are dates
        * `grid_resolution` date values in between minimum and maximum of feature
          values
        * bin values are created as evenly as possible:
            1. dates are parsed and converted to epoch timestamps i.e integers
            2. bins are created as in case of numeric integer binning
            3. integer bins are converted back to original date format
        * minimum and maximum is included in bins
          (if `grid_resolution` is bigger or equal to 2)
    * bins in **categorical** mode:
        * bins are dates
        * top `grid_resolution` values from feature values ordered by _frequency_
          (dates are handled as opaque strings and most frequent values are used
          as bins)

    ICE out of range binning:

    **Integer** feature:
    * OOR bins in **numeric** mode:
        * OOR bins are integers
        * (at most) `oor_grid_resolution` integer values are added below minimum and
          above maximum
        * bin values are created by adding/substracting rounded standard deviation
          (of feature values) above and below maximum and minimum `oor_grid_resolution`
          times
            * 1 used used if rounded standard deviation would be 0
        * if feature is of _unsigned_ integer type, then bins below 0
          are not created
            * if rounded standard deviation and/or `oor_grid_resolution` is so high
              that it would cause lower OOR bins to be negative numbers, then standard
              deviation of size 1 is tried instead
    * OOR bins in **categorical** mode:
        * same as numeric mode

    **Float** feature:
    * OOR bins in **numeric** mode:
        * OOR bins are floats
        * `oor_grid_resolution` float values are added below minimum and above maximum
        * bin values are created by adding/substracting standard deviation
          (of feature values) above and below maximum and minimum `oor_grid_resolution`
          times
    * OOR bins in **categorical** mode:
        * same as numeric mode

    **String** feature:
    * bins in **numeric** mode:
        * not supported
    * bins in **categorical** mode:
        * OOR bins are strings
        * value `UNSEEN` is added as OOR bin

    **Date** feature:

    * bins in **numeric** mode:
        * not supported
    * bins in **categorical** mode:
        * OOR bins are strings
        * value `UNSEEN` is added as OOR bin

    Examples
    --------
    >>> explanations = ICE("Fluent").explain(
    ...     ['Feature'],
    ...     X,
    ...     predict_method=scorer,
    ...     mins=[1], maxs=[10]).explanations()
    >>>
    >>> explanations = ICE("Load JSon file").load_json(
    ...     "ice.json").explanations()
    >>>
    >>> ice = ICE("Step by step ICE calculation and persistence")
    >>> ice.explain(
    ...     ['Feature'],
    ...     X,
    ...     predict_method=scorer,
    ...     mins=[1], maxs=[6])
    >>> # ...
    >>> explanations = ice.explanations()
    >>> ice.save_json("ice.json")
    >>>
    >>> ice = ICE("Single feature ICE with bins")
    >>> ice = ice.explain(
    ...     ['F1'],
    ...     X,
    ...     predict_method=scorer,
    ...     bins=[[1,2]])
    >>> explanations = ice.explanations()
    >>> # ...
    >>> ice = ICE("Multiple features ICE with bins")
    >>> explanations = ice.explain(
    ...     ['F1','F2'],
    ...     X,
    ...     predict_method=scorer,
    ...     bins=[[1,2],[5,6,7]])
    >>>
    >>> explanations = ICE("ICE and out of range ICE").explain(
    ...     ['Feature'],
    ...     X,
    ...     predict_method=scorer,
    ...     mins=[1], maxs=[6], stds=[3],
    ...     out_of_range_resolution=3).explanations()
    >>>
    >>> explanations = ICE("Out of range ICE only").explain(
    ...     ['Feature'],
    ...     X,
    ...     predict_method=scorer,
    ...     mins=[1], maxs=[6], stds=[3],
    ...     grid_resolution=0,
    ...     out_of_range_resolution=3).explanations()
    >>>
    >>> explanations = ICE("Multidimensional ICE").explain(
    ...     [('F1','F2'), F3],
    ...     X,
    ...     predict_method=scorer,
    ...     mins=[(1,3), 8], maxs=[(2,5), 9]).explanations()
    >>>
    >>> explanations = ICE("Multidimensional ICE").explain(
    ...     [('F1','F2'), F3],
    ...     X,
    ...     predict_method=scorer,
    ...     bins=[([1, 2], [3, 4, 5]), [8, 9]]).explanations()
    >>>
    >>> explanations = ICE("Fluent ICE").load_json("ice.json").explanations()
    >>> # ...
    >>> ice = ICE("Step by step ICE loading")
    >>> ice.load_json("ice.json")
    >>> explanations = ice.explanations()
    >>>

    See also
    --------
    https://pandas.pydata.org

    """

    EXPLAINER_TYPE = "ice"

    @property
    def opt_1_prediction(self):
        """Option which controls ICE computation strategy. This option is set
        to ``True`` by default i.e. one prediction strategy is allowed.

        If this option is set to ``False``, then predict method will be
        called for every feature's bin - which might be slow as it requires
        ``len(features) * len(bins)`` predict function invocations, but it
        preserves memory and enables processing of reasonably big datatsets.

        If it is set to ``True``, then it indicates that if
        ``(frame size in bytes * len(features) * len(bins)) < free RAM/2``
        then single predict method invocation strategy will be used i.e. ICE
        for all features and bins will be computed using ONE predict method
        invocation. Note that this is memory intensive operation which
        requires input dataset to be ``len(features) * len(bins)``
        times in the memory.

        In case of multidimensional ICE will be reduced only the number
        of predict method invocations within each dimension, not across all
        dimensions.


        Returns
        -------
        bool:
          ``True`` if 1 prediction strategy **may** be used, ``False`` if
          it's forbidden.

        """
        return self._1_prediction

    @opt_1_prediction.setter
    def opt_1_prediction(self, allow_1_prediction: bool):
        """ICE computation strategy setter."""
        self._1_prediction = allow_1_prediction

    def __init__(self, name, interpretable_model=None):
        """Create ICE instance.

        Parameters
        ----------
        name: str
            Name of this ICE methods.
        interpretable_model: h2o_sonar.core.h2o_sonar.InterpretableModel
            Interpretable model whose predict_method and current directory
            should be used.

        """
        method.Method.__init__(
            self,
            method_name=name,
            method_type=ICE.EXPLAINER_TYPE,
            interpretable_model=interpretable_model,
        )
        _abstract_ice_pd.AbstractIcePd.__init__(self)

        self._1_prediction = True

        self._progress_callback: progress.AbstractProgressCallbackContext | None = None

    # IMPROVE: explain() method signatures don't fit yet
    def explain(
        self,
        features: list,
        X,
        Y=None,
        predict_method=None,
        bins: list = None,
        mins: list = None,
        maxs: list = None,
        out_of_range_resolution: int = 0,
        out_of_range_blacklist: list | None = None,
        stds: list | None = None,
        grid_resolution: int = method.Method.DEFAULT_GRID_RESOLUTION,
        center: bool = False,
        target_transform=lambda x: x,
        features_meta: dict | None = None,
        bins_sort: bool = False,
        progress_callback: progress.AbstractProgressCallbackContext | None = None,
    ):
        """
        Calculate ICE and cache it in memory so that it can be
        subsequently obtained using function :func:`explanations() <h2o_sonar.
        methods.ice.ICE.explanations>`.

        Parameters
        ----------
        features : list[int or str or tuple]
            A list of features (strings or numerical ids) for which ICE is
            supposed to be calculated. If list item is tuple, then
            n-dimensional (where n is `len(tuple)`) ICE is calculated.
        X : pandas.core.frame.DataFrame or datatable.Frame
            Original data for which we want to compute ICE
        Y : pandas.core.frame.DataFrame or datatable.Frame
            Actual values - if provided, then residuals ICE is
            calculated. The number of ``Y`` columns must correspond to the
            number of columns returned by predict method.
        mins : list[int or float]
            If `bins` are not specified, then minimum for every target feature
            in `X` must be provided. If ICE n-dimensional (computed for a
            feature tuple), then mins list item must be also tuple with minima
            for every feature within feature tuple.
        maxs : list[int or float]
            If `bins` are not specified, then maximum for every target feature
            in `X` must be provided. If ICE n-dimensional (computed for a
            feature tuple), then maxs list item must be also tuple with maxima
            for every feature within feature tuple.
        predict_method : function
            A lambda function which takes instances (a set of rows)
            and outputs predictions as Pandas Series or DataFrame. If this
            parameter is not specified, then interpretable model's
            predict_method is used.
        bins : list[list[object]]
            Data values for each target feature for which we want to compute
            ICE, vector if for single target_feature, otherwise a matrix.
        bins_sort : bool
            Ensure bins values sorting.
        grid_resolution : int
            The number of equally spaced points used to create bins if bins
            are not specified.
        center : bool
            Set this parameter to ``True`` to compute c-ICE (centered ICE).
        target_transform : function
            A lambda to be applied on residuals - instead of identity
            consider e.g. ``abs()`` or ``sqrt()``
        features_meta : dict
            Features metadata allowing to indicate whether given feature is
            categorical (use ``categorical`` key and list of feature names),
            date (use ``date`` key and list of feature names, to specify format use
            ``date-format`` and list of Python date formats) or numerical
            (default). Use ``quantiles`` dict key and dict key of feature names and
            quantiles to ensure bins construction using quantiles instead of even split.
        out_of_range_resolution : int
            Calculate out of range values if `out_of_range_resolution` parameter
            is bigger than 0. For instance, if `out_of_range_resolution` is set
            to 3, then the expression below is used to construct six bins (2*3)
            where n = `[1, 2, 3]`.

            .. math::
            min(feature) - n*sd(feature)
            max(feature) + n*sd(feature)

        out_of_range_blacklist : list
            Skip OOR computation for given features.
        stds : list[int or float]
            If out of range is computed, then standard deviation for every
            target feature in ``X`` must be provided.
        progress_callback : progress.AbstractProgressCallbackContext | None
            Progress callback allowing the progress of PD calculation.

        Returns
        -------
        h2o_sonar.methods.ICE :
            ICE instance to get computed explanations using `explanations()`
            function.

        Raises
        ------
        MliUnsupportedDataFormatError
            If input parameters are not in expected format.
        MliPredictMethodError
            If predict method fails.

        See also
        --------
        https://pandas.pydata.org
        https://github.com/h2oai/datatable

        """
        if not HAS_H2O:
            commons.raise_opt_import_err("h2o")

        if not bins and not grid_resolution and not out_of_range_resolution:
            raise ValueError(
                "Cannot compute ICE - no bins, no resolution and no out of range"
            )
        super()._check_and_set_features(features, features_meta)
        self.__check_out_of_range(
            out_of_range_resolution, out_of_range_blacklist, mins, maxs, stds
        )
        predict_method = super()._method_precondition(predict_method)
        super()._check_resolution(X, grid_resolution)
        self._target_transform = (
            (lambda x: x) if not target_transform else target_transform
        )
        self._center = center
        self._bins_sort = bins_sort

        self.diagnostics.add_scorer_calls_slot()

        self._progress_callback = progress_callback

        if not bins:
            if not mins:
                raise ValueError("If bins not specified, then minima are mandatory")
            if not isinstance(mins, list):
                raise ValueError("Minima must be of list type")
            if len(self.features) != len(mins):
                raise ValueError(
                    f"Minimum for every feature in '{features}' must be provided"
                )
            if not maxs:
                raise ValueError("If bins not specified, then maxima is mandatory")
            if not isinstance(maxs, list):
                raise ValueError("Maxima must be of list type")
            if len(self.features) != len(maxs):
                raise ValueError(
                    f"Maximum for every feature in '{features}' must be provided"
                )

        if isinstance(X, pandas.DataFrame):
            if X.empty:
                raise ValueError("Data cannot be empty")
            return self._choose_strategy(
                X=X,
                Y=Y,
                bins=bins,
                mins=mins,
                maxs=maxs,
                predict_method=predict_method,
            )

        if isinstance(X, (datatable.Frame, h2o.H2OFrame)):
            frame = _mli.InterpretableModel.to_pandas(X)
            if frame.empty:
                raise ValueError("Data cannot be empty")
            return self._choose_strategy(
                X=frame,
                Y=Y,
                bins=bins,
                mins=mins,
                maxs=maxs,
                predict_method=predict_method,
            )

        raise errors.MliUnsupportedDataFormatError(
            message=f"Unsupported X data type: {type(X)}.",
            suggestion="Use Pandas DataFrame, H2OFrame or datatable frame.",
        )

    def _choose_strategy(self, X, Y, bins, mins, maxs, predict_method):
        """Choose ICE calculation strategy (memory vs. predict method invocations)."""

        assert isinstance(X, pandas.DataFrame)
        super()._check_dataset_features(X)
        bins_ = self._check_bins(bins)

        # choose computation strategy ~ predict method invocations count
        if self._1_prediction:
            # assess (mem) safety of 1 prediction strategy
            go_one = True
            for feature in self._fs:
                if isinstance(feature, tuple):
                    # strategy not supported for n-dim ICE
                    go_one = False
                    break

            if go_one:
                for i, feature in enumerate(self._fs):
                    self.__prepare_bins(
                        feature, X, i, bins, bins_, maxs, mins, self._stds
                    )
                total_bins = 0
                for bin_ in bins_:
                    total_bins = total_bins + len(bin_)

                ram_free = psutil.virtual_memory().available
                mem_frame = X.memory_usage(index=True, deep=True).sum()

                if (mem_frame * total_bins) < (ram_free / 2):
                    ice_1_prediction = self._explain_ice_1_prediction(
                        X, Y, bins_, predict_method
                    )
                    if self._progress_callback:
                        self._progress_callback.set_progress(1.0)
                    return ice_1_prediction

                # reuse bins computed above (1-dim ICE)
                ice_1_prediction = self._explain_ice(
                    X, Y, bins_, bins_, mins, maxs, predict_method
                )
                if self._progress_callback:
                    self._progress_callback.set_progress(1.0)
                return ice_1_prediction

        # else strategy: per feature's bin prediction
        return self._explain_ice(X, Y, bins, bins_, mins, maxs, predict_method)

    def _explain_ice(self, X, Y, bins, bins_, mins, maxs, predict_method):
        """Pandas DataFrame based Individual Conditional Expectation (ICE)
        implementation.

        """
        loggers.debug("ICE strategy: MANY predict method invocations")

        progress_callback = (
            progress.ProgressCallbackContext(
                total_steps=len(self._fs),
                parent_callback=self._progress_callback,
            )
            if self._progress_callback
            else None
        )

        self._explanations_cache = {}
        for i, feature in enumerate(self._fs):
            if isinstance(feature, tuple):
                # nD-ICE
                self.__n_prepare_bins(feature, X, i, bins, bins_, maxs, mins)

                # calculate ICE for the features tuple
                self.__n_explain_ice(
                    features=feature,
                    X=X,
                    predict_method=predict_method,
                    Y=Y,
                    bins=bins[i] if bins else bins_[0],
                    features_progress_callback=progress_callback,
                )
            else:
                # 1D-ICE
                # cache original feature's column values
                cols_cache = X.loc[:, feature].copy(deep=True)

                self.__prepare_bins(feature, X, i, bins, bins_, maxs, mins, self._stds)

                # calculate ICE for the feature
                self._explain_ice_feature(
                    features=feature,
                    X=X,
                    Y=Y,
                    bins_=bins_[i],
                    predict_method=predict_method,
                    features_progress_callback=progress_callback,
                )

                # return input frame to original state
                X.loc[:, feature] = cols_cache

        self._center_ice()

        if self._progress_callback:
            self._progress_callback.set_progress(1.0)

        return self

    def __n_prepare_bins(self, f_tuple, X, i, bins, bins_, maxs, mins):
        if bins is None:
            if not isinstance(maxs[i], tuple):
                raise ValueError(f"Maxima for feature tuple {f_tuple} must be tuple")
            if len(maxs[i]) != len(f_tuple):
                raise ValueError(f"Every feature in tuple {f_tuple} must have maximum")
            if not isinstance(mins[i], tuple):
                raise ValueError(f"Minima for feature tuple {f_tuple} must be tuple")
            if len(mins[i]) != len(f_tuple):
                raise ValueError(f"Every feature in tuple {f_tuple} must have minimum")

            tuple_bins = []
            for j, ft_ in enumerate(f_tuple):
                self.__prepare_bins(
                    ft_,
                    X,
                    j,
                    None,
                    tuple_bins,
                    maxs[i],
                    mins[i],
                    None if self._stds is None else self._stds[i],
                )
            bins_.append(tuple(tuple_bins))
        else:
            # bins exist > add just OOR bins if needed
            if self._oor:
                for j, ft_ in enumerate(f_tuple):
                    t_bins = list(bins[i])
                    self.__prepare_bins(
                        ft_, X, j, bins, t_bins, maxs[i], mins[i], self._stds[i]
                    )
                    bins[i] = tuple(t_bins)

    def __prepare_bins(self, feature, X, i, bins, bins_, maxs, mins, stds):
        if self._oor and (feature not in self._oor_blacklist):
            if bins is None and self._g_resolution:
                self._add_new_bin(X, feature, i, mins, maxs, stds, bins, bins_)
            else:
                oor_bin = _abstract_ice_pd.AbstractIcePd.create_oor_bins(
                    feature_dtype=X[feature].dtype.kind,
                    min_=mins[i],
                    max_=maxs[i],
                    std_dev=stds[i],
                    out_of_range_resolution=self._oor,
                )
                self._add_oor_bin(feature, oor_bin)
                if not bins and not self._g_resolution:
                    bins_.append(oor_bin)
                else:
                    bins_[i].extend(oor_bin)
        else:
            self._add_new_bin(X, feature, i, mins, maxs, stds, bins, bins_)

        # remove bin duplicates
        bins_[i] = list(OrderedDict.fromkeys(bins_[i]))

        # sort
        if self._bins_sort:
            if (
                self._fs_meta is not None
                and ICE.KEY_CATEGORICAL_FEATURES in self._fs_meta
                and feature in self._fs_meta[ICE.KEY_CATEGORICAL_FEATURES]
            ) and self._oor:
                cat_levels = bins_[i][:-1]
                nan_val = None
                if histogram.has_nan(cat_levels):
                    nan_val = cat_levels[histogram.find_nan_index(cat_levels)]
                    cat_levels.remove(nan_val)
                cat_levels.sort()
                if nan_val:
                    cat_levels.append(nan_val)
                bins_[i] = cat_levels + bins_[i][-1:]
            else:
                # ensure type compatibility in case of OOR bins
                if (
                    bins_[i]
                    and len(bins_[i]) == 2
                    and isinstance(bins_[i][0], bool)
                    and isinstance(bins_[i][1], str)
                ):
                    bins_[i] = [bins_[i][0], False if bins_[i][0] else True]

                # Account for nan values in list of bins, e.g., ['a', 'b', nan, ...]
                bins_[i] = sorted(
                    bins_[i],
                    key=lambda x: (
                        (math.isnan(x), x) if isinstance(x, float) else (False, x)
                    ),
                )

    def _explain_ice_feature(
        self, features, X, Y, bins_, predict_method, features_progress_callback
    ):
        """Calculate ICE for single feature or n-dimensional set of
        features (tuple)

        """
        # dictionary: { p_0/p_0+1/p_n: frame }
        p_dict = {}
        p_cols = None

        # bins is either list (1-dim) or list of lists (n-dim)
        if self._bins_sort and bins_ and not isinstance(bins_[0], list):
            bins_.sort()

        if features_progress_callback:
            bins_progress_callback = (
                features_progress_callback.get_sub_callback_for_steps(len(bins_))
            )
        else:
            bins_progress_callback = None

        for i_b, bin_ in enumerate(bins_):
            # set target feature col(s) values to bin value(s) for all examples
            if isinstance(bin_, list):
                bin_ = tuple(bin_)
                for i, ft_ in enumerate(features):
                    X.loc[:, ft_] = bin_[i]
            else:
                X.loc[:, features] = bin_

            # predict all examples w/ feature fixed to bin value
            # RQ: no mem overhead > cannot duplicate rows for bigger batch
            try:
                ice_b = predict_method(X)
                self.diagnostics.add_scorer_calls()
            except Exception as e:
                loggers.error(f"Predict method failed: {e}")
                raise errors.MliPredictMethodError(f"Predict method failed: {e}") from e

            p_cols = self._explain_ice_feature_post_predict(
                p_dict, p_cols, ice_b, Y, bins_, bin_
            )

            if bins_progress_callback:
                bins_progress_callback.set_steps(i_b + 1)

        self._explanations_cache[features] = p_dict

    def _explain_ice_feature_post_predict(
        self, p_dict: dict, p_cols: int, ice_b, Y, bins_: list, bin_
    ):
        ICE.__check_prediction_result(ice_b)

        # result init
        if not p_dict:
            p_cols = ICE._init_ice_result(ice_b, p_dict, bins_)

        # check that predict m. always returns p. w/ same dimensions
        if isinstance(ice_b, pandas.DataFrame) and p_cols != ice_b.shape[1]:
            raise errors.MliPredictMethodError(
                f"Predict must always return the same number of "
                f"classes - expected {p_cols}, got {ice_b.shape[1]}"
            )

        for col in range(p_cols):
            p_class = ice_b if isinstance(ice_b, pandas.Series) else ice_b.iloc[:, col]

            if Y is not None:
                p_dict[ICE.LABEL_PREFIX_CLASS + str(col)][bin_] = (
                    self.__explain_residuals(p_class, Y, Y.columns[col])
                )
            else:
                p_class.index = p_dict[ICE.LABEL_PREFIX_CLASS + str(col)].index
                p_dict[ICE.LABEL_PREFIX_CLASS + str(col)][bin_] = p_class

        return p_cols

    def _explain_ice_1_prediction(self, X, Y, bins, predict_method):
        """Calculate ICE using ONE prediction method invocation strategy.
        Assemble one frame with all the target features and bins,
        make prediction by passing this frame to predict method,
        then disassemble the frame to get the result.

        IMPROVE: All features must be 1-dimensional ICE (for now), otherwise
        this method fails to fallback to default computation strategy.

        """
        loggers.debug("ICE strategy: 1 predict method invocation")

        if bins is None:
            raise ValueError(
                "Bins must be defined for one prediction computation strategy"
            )

        # build one frame: use original frame to backup/substitute/copy
        # start = time.time()

        p_callback = (
            progress.ProgressCallbackContext(
                total_steps=2 * len(self._fs) + 1,
                parent_callback=self._progress_callback,
            )
            if self._progress_callback
            else None
        )

        x_1 = []
        x_index = X.index
        X = datatable.Frame(X)
        count = 0
        for i, feature in enumerate(self._fs):
            cols_cache = X[:, feature].copy()
            for bin_ in bins[i]:
                X[:, feature] = datatable.Frame([bin_])
                x_1.append(X.copy())  # make copy to avoid reuse below
                count += 1
            X[:, feature] = cols_cache
            if p_callback:
                p_callback.set_steps(i)
        x_1 = datatable.rbind(*x_1, force=True).to_pandas()
        x_1.index = numpy.concatenate([x_index] * count)

        # loggers.debug(
        #    f"1 frame for {self._fs} built in: {time.time() - start}s")

        # 1 prediction
        try:
            ice_x_1 = predict_method(x_1)
            self.diagnostics.add_scorer_calls()
        except Exception as e:
            loggers.error(f"Predict method failed: {e}")
            raise errors.MliPredictMethodError(f"Predict method failed: {e}") from e

        if p_callback:
            p_callback.set_steps(len(self._fs) + 1)

        self.__check_prediction_result(ice_x_1)

        # build result
        self._explanations_cache = {}
        row = p_cols = 0
        for i, feature in enumerate(self._fs):
            # dictionary: { p_0/p_0+1/p_n: frame }
            p_dict = {}
            for bin_ in bins[i]:
                # build prediction bin
                if isinstance(ice_x_1, pandas.Series):
                    ice_b = ice_x_1.iloc[row : row + X.shape[0]]
                else:
                    ice_b = ice_x_1.iloc[row : row + X.shape[0], :]
                row = row + X.shape[0]

                p_cols = self._explain_ice_feature_post_predict(
                    p_dict, p_cols, ice_b, Y, bins[i], bin_
                )
                self._explanations_cache[feature] = p_dict

            if p_callback:
                p_callback.set_steps(len(self._fs) + 1 + i)

        self._center_ice()

        return self

    def __explain_residuals(self, prediction, Y, col):
        if Y is not None:
            residuals = -prediction.sub(Y[col], axis=0)
            residuals = residuals.apply(self._target_transform)
            return residuals
        return prediction

    def __check_out_of_range(self, oor_resolution, oor_blacklist, mins, maxs, stds):
        if oor_resolution:
            if not isinstance(oor_resolution, int):
                raise TypeError(
                    "Out of range resolution parameter must be of integer type"
                )

            self._oor = oor_resolution
            self._oor_bins = {} if self._oor else None

            if not isinstance(mins, list):
                raise ValueError("Out of range requires minima for each feature")
            if not isinstance(maxs, list):
                raise ValueError("Out of range requires maxima for each feature")
            if not isinstance(stds, list):
                raise ValueError(
                    "Out of range requires list of standard deviations for each feature"
                )

            self._stds = stds

        # IMPROVE consider a check of valid feature names in the blacklist
        self._oor_blacklist = oor_blacklist if oor_blacklist else []

    @staticmethod
    def _init_ice_result(ice_b, p_dict, bin_):
        p_cols = ice_b.shape[1] if isinstance(ice_b, pandas.DataFrame) else 1
        if p_cols == 1:
            if isinstance(bin_[0], list):
                p_dict[ICE.LABEL_REGRESSION] = pandas.DataFrame(
                    index=ice_b.index, columns=[tuple(bb) for bb in bin_]
                )
            else:
                p_dict[ICE.LABEL_REGRESSION] = pandas.DataFrame(
                    index=ice_b.index, columns=bin_
                )
        else:
            for col in range(p_cols):
                p_dict[ICE.LABEL_PREFIX_CLASS + str(col)] = (
                    pandas.DataFrame(
                        index=ice_b.index, columns=[tuple(bb) for bb in bin_]
                    )
                    if isinstance(bin_[0], list)
                    else pandas.DataFrame(index=ice_b.index, columns=bin_)
                )
        return p_cols

    def _center_ice(self):
        if self._center:
            for ftr in self._explanations_cache:
                if (
                    self._fs_meta is not None
                    and ftr in self._fs_meta[ICE.KEY_CATEGORICAL_FEATURES]
                ):
                    continue  # skip center for categorical
                for clazz in self._explanations_cache[ftr]:
                    ice_mean = pandas.DataFrame(
                        self._explanations_cache[ftr][clazz].mean(axis=1)
                    )
                    self._explanations_cache[ftr][clazz] = pandas.DataFrame(
                        self._explanations_cache[ftr][clazz].values - ice_mean.values,
                        columns=self._explanations_cache[ftr][clazz].columns,
                    )

    def _add_new_bin(self, X, feature, i, mins, maxs, stds, bins, bins_):
        if (
            self._fs_meta
            and ICE.KEY_DATE_FEATURES in self._fs_meta
            and feature in self._fs_meta[ICE.KEY_DATE_FEATURES]
        ):
            if not bins:
                try:
                    self.__add_new_bin_date(X, feature, bins_)
                except (
                    RuntimeError,
                    TypeError,
                    ValueError,
                    NameError,
                    LookupError,
                ) as e:
                    loggers.debug(
                        f"Fallback: handle date features as categorical due to: {e}"
                    )
                    self.__add_new_bin_categorical(X, feature, bins_)
        elif (
            self._fs_meta
            and ICE.KEY_CATEGORICAL_FEATURES in self._fs_meta
            and feature in self._fs_meta[ICE.KEY_CATEGORICAL_FEATURES]
        ) or method.Method._is_categorical(X[feature]):
            if not bins:
                self.__add_new_bin_categorical(X, feature, bins_)

            if self._oor and feature not in self._oor_blacklist:
                if self.__can_num_bins(i, mins, maxs, stds, bins_):
                    # create num OOR bins for cat feature ~ more relevant than UNSEEN
                    self.__add_cat_with_num_bins(feature)
                    oor_bin = _abstract_ice_pd.AbstractIcePd.create_oor_bins(
                        feature_dtype=X[feature].dtype.kind,
                        min_=mins[i],
                        max_=maxs[i],
                        std_dev=stds[i],
                        out_of_range_resolution=self._oor,
                    )
                    self._add_oor_bin(feature, oor_bin)
                    bins_[i].extend(oor_bin)
                elif bins_ and len(bins_) == 1 and isinstance(bins_[0], bool):
                    oor_bin = [False if bins_[0] else True]
                    self._add_oor_bin(feature, oor_bin)
                    bins_[i].extend(oor_bin)
                else:
                    unseen_value = ICE.get_unseen_string_value(X[feature], "UNSEEN")
                    oor_bin = [f"{unseen_value}"]
                    self._add_oor_bin(feature, oor_bin)
                    bins_[i].extend(oor_bin)
        else:
            if not bins:
                self.__add_new_bin_numerical(X, feature, i, mins, maxs, bins_)
            if self._oor and feature not in self._oor_blacklist:
                oor_bin = _abstract_ice_pd.AbstractIcePd.create_oor_bins(
                    feature_dtype=X[feature].dtype.kind,
                    min_=mins[i],
                    max_=maxs[i],
                    std_dev=stds[i],
                    out_of_range_resolution=self._oor,
                )
                self._add_oor_bin(feature, oor_bin)
                bins_[i].extend(oor_bin)

    def __add_cat_with_num_bins(self, feature):
        if not self._fs_meta:
            self._fs_meta = {}
        if ICE.KEY_CAT_WITH_NUM_BIN not in self._fs_meta:
            self._fs_meta[ICE.KEY_CAT_WITH_NUM_BIN] = []
        if feature not in self._fs_meta[ICE.KEY_CAT_WITH_NUM_BIN]:
            self._fs_meta[ICE.KEY_CAT_WITH_NUM_BIN].append(feature)

    @staticmethod
    def __can_num_bins(i, mins, maxs, stds, bins) -> bool:
        try:
            if mins and maxs and stds and bins:
                if (
                    mins[i] is not None
                    and maxs[i] is not None
                    and stds[i] is not None
                    and bins[i] is not None
                ):
                    return True
        except (NameError, RuntimeError, TypeError) as e:
            loggers.debug(f"Categorical feature with numeric binning failed: {e}")
        return False

    @staticmethod
    def get_unseen_string_value(column, base_string):
        """Return an unseen value, not present in the ``column`` containing
        ``base_string``, f.e. "UNSEEN" might be present in column, unlike
        "UNSEEN_[1]".

        Parameters
        ----------
        column : pd.Series
          Input pandas Series.
        base_string : str
          Default value to try.

        Returns
        -------
        str :
          Value not present in the ``column``.

        """
        unique_values = column.unique()
        unseen_value = base_string
        if base_string in unique_values:
            random_index = "1"
            unseen_value = f"{base_string}_[{random_index}]"
            while unseen_value in unique_values:
                random_index = f"{random_index}{numpy.random.randint(10)}"
            unseen_value = f"{base_string}_[{random_index}]"
        return unseen_value

    def __add_new_bin_date(self, X, feature, bins_):
        date_format = None
        if (
            ICE.KEY_DATE_FEATURES_FORMAT in self._fs_meta
            and feature in self._fs_meta[ICE.KEY_DATE_FEATURES]
            and len(self._fs_meta[ICE.KEY_DATE_FEATURES_FORMAT])
            == len(self._fs_meta[ICE.KEY_DATE_FEATURES])
        ):
            for i, date_f in enumerate(self._fs_meta[ICE.KEY_DATE_FEATURES]):
                if self._fs_meta[ICE.KEY_DATE_FEATURES][i] == date_f:
                    date_format = self._fs_meta[ICE.KEY_DATE_FEATURES_FORMAT][i]
                    break

        date_bins, _ = method.Method.create_date_aware_bins(
            [feature],
            X,
            grid_resolution=self._g_resolution,
            date_format=(
                date_format
                if date_format
                else method.Method.DEFAULT_DATE_FEATURE_FORMAT
            ),
        )
        # note that bins are already sorted
        bins_.append(date_bins[0])

    def __add_new_bin_categorical(self, X, feature, bins_):
        # make most frequent cats bins
        gbf = X.groupby(feature, dropna=False)
        bins_.append(
            gbf.size()
            .sort_values(ascending=False)
            .head(self._g_resolution)
            .index.tolist()
        )

    def __add_new_bin_numerical(self, X, feature, i, mins, maxs, bins_):
        if self._fs_meta and ICE.KEY_QUANTILE_BINS in self._fs_meta:
            if isinstance(self._fs_meta[ICE.KEY_QUANTILE_BINS], list):
                self._fs_meta[ICE.KEY_QUANTILE_BINS] = dict(
                    zip(
                        self._fs_meta[ICE.KEY_QUANTILE_BINS],
                        [None] * len(self._fs_meta[ICE.KEY_QUANTILE_BINS]),
                        strict=False,
                    )
                )
            if feature in self._fs_meta[ICE.KEY_QUANTILE_BINS].keys():
                quantile = self._fs_meta[ICE.KEY_QUANTILE_BINS].get(feature)
                binning.build_qtile_bins(
                    bins=bins_,
                    X=X,
                    feature=feature,
                    quantile=quantile if quantile else ICE.DEFAULT_GRID_RESOLUTION,
                )
            else:
                self.create_numerical_bins(
                    feature_dtype=X[feature].dtype.kind,
                    bins=bins_,
                    idx=i,
                    maxs=maxs,
                    mins=mins,
                )
        else:
            self.create_numerical_bins(
                feature_dtype=X[feature].dtype.kind,
                bins=bins_,
                idx=i,
                maxs=maxs,
                mins=mins,
            )
        has_nans = True if X[feature].isnull().values.any() else False
        if has_nans:
            bins_[i].append(float("nan"))

    def create_numerical_bins(self, feature_dtype: str, bins, idx: int, maxs, mins):
        min_ = (
            mins[idx] if mins else (min(self.strip_none(bins[idx])) if bins else None)
        )
        max_ = (
            maxs[idx] if maxs else (max(self.strip_none(bins[idx])) if bins else None)
        )

        if feature_dtype in "iu":
            _abstract_ice_pd.AbstractIcePd.create_numerical_bins_int(
                grid_resolution=self._g_resolution,
                bins=bins,
                max_=max_,
                min_=min_,
            )
        else:
            _abstract_ice_pd.AbstractIcePd.create_numerical_bins_float(
                grid_resolution=(
                    1 if self._g_resolution == 1 else self._g_resolution - 1
                ),
                bins=bins,
                max_=max_,
                min_=min_,
            )

    @staticmethod
    def __check_prediction_result(ice_b):
        if ice_b is not None:
            if isinstance(ice_b, pandas.DataFrame) and ice_b.shape[1] > 0:
                pass
            elif not isinstance(ice_b, pandas.Series):
                raise errors.MliPredictMethodError(
                    f"Predict method return value data type must be Pandas Series "
                    f"or DataFrame, but it's: {type(ice_b)}"
                )
        else:
            raise errors.MliPredictMethodError("Predict method result cannot be empty")

    def __n_explain_ice(
        self, features, X, predict_method, Y, bins, features_progress_callback
    ):
        """Minimalistic n-dimensional ICE computation.

        Parameters
        ----------
        features : Tuple
            Features for which to compute n-dimensional ICE
            (where n is len(features)).
        X :
            Original data for which we want to compute ICE
        predict_method :
            A lambda which takes instances (a set of rows) and outputs
            predictions
        Y :
            Actual values (labels).
        bins :
           Per feature bin.
        features_progress_callback :
           Callback to report progress.

        """
        if bins is not None:
            if not isinstance(bins, tuple):
                raise ValueError(f"Bins for {features} must be of tuple type")
            if not bins:
                raise ValueError("Bins cannot be empty")
            if len(features) != len(bins):
                raise ValueError(f"Bin count must correspond to features {features}")
        else:
            raise errors.MliUnsupportedOperationError(
                "Bins must be specified for N-dimensional ICE"
            )

        # bins ready (passed or calculated)

        # build points where to calculate ICE as combination of bin values
        points = ICE.__create_points_for_bins(bins)

        # cache previous cols values
        cols_cache = {}
        for ft_ in features:
            cols_cache[ft_] = X.loc[:, ft_].copy(deep=True)

        self._explain_ice_feature(
            features=features,
            X=X,
            Y=Y,
            bins_=points,
            predict_method=predict_method,
            features_progress_callback=features_progress_callback,
        )

        # return input frame to original state using cached values
        for ft_ in features:
            X.loc[:, ft_] = cols_cache[ft_]

        # explanations cache is updated w/ ICE calculation for the feat tuple

    @staticmethod
    def __create_points_for_bins(features_bins):
        points = [[]]
        for feature_bins in features_bins:
            new_points = []
            for point in points:
                for v in feature_bins:
                    point_c = point.copy()
                    point_c.append(v)
                    new_points.append(point_c)
            points = new_points
        return points

    def __resolve_json_path(self, path):
        if not path and self._i_model is not None:
            path = os.path.join(
                self._i_model.mli.work_dir, self._default_json_file_name
            )
        return path

    # override
    def save_json(self, path=None):
        """Save cached ICE explanations as JSon file.

        Parameters
        ----------
        path: str
            Local file path where to store explanations. If path isn't
            specified, then explanations are stored to ```explanations.json```
            in the current directory.

        Returns
        -------
        h2o_sonar.methods.ICE
            ICE instance

        """
        self.check_explanations_serializability(self._explanations_cache)

        super()._save_json(self._explanations_cache, self.__resolve_json_path(path))

        # fluent API
        return self

    # override
    def load_json(self, path=None):
        """Load cached ICE explanations from a JSon file.

        Parameters
        ----------
        path: str
            Local file path where to store explanations. If path isn't
            specified, then explanations are stored to ```explanations.json```
            in the current directory.

        Returns
        -------
        h2o_sonar.methods.ICE
            ICE instance

        """
        self._explanations_cache = super()._load_json(self.__resolve_json_path(path))

        # fluent API
        return self

    JSON_ICE = "ice"
    JSON_NEXT_ID = "next_id"

    def save(self, path: str, append: bool = False):
        """EXPERIMENTAL: persistence format may change without deprecation.

        Save cached ICE explanations as JSon index file and per feature binary
        frames with ICE.  If path isn't specified, then explanations index is
        stored to ```explanations.json``` in the current directory

        Please note that ICE frames columns and index is transformed (columns
        are converted to strings and index is normalized).

        Parameters
        ----------
        path: str
            Index file path where to store explanations.
        append: bool
            In case that index file exists, don't fail but append new
            features to that index file.

        Returns
        -------
        h2o_sonar.methods.ICE
            ICE instance

        """
        if self._explanations_cache is None:
            raise errors.MliError("No explanations - call run() function first")

        path = self.__resolve_json_path(path)
        path = self._default_json_file_name if not path else path

        # save index file
        if os.path.exists(path):
            if append:
                index_json, next_id = self.__load_index_json(path)
                ice_json = index_json[ICE.JSON_ICE]
                sequence = next_id
            else:
                raise FileExistsError(f"Index file already exists {path}")
        else:
            index_json = {}
            ice_json = {}
            index_json[ICE.JSON_ICE] = ice_json
            sequence = 1

        # save frames
        df_path_prefix = path if not path.endswith(".json") else path[:-5]
        for feature in self._explanations_cache:
            ice_json[feature] = {}
            for clazz in self._explanations_cache[feature]:
                df_ice = self._explanations_cache[feature][clazz]

                # save
                df_path = f"{df_path_prefix}-{sequence}.jay"
                ice_json[feature][clazz] = df_path
                # IMPROVE Pandas 2 datatable conversion ~ ICE data 2x in memory

                # Need to preserve column names as DT will set some values to `C*`,
                # e.g., C0.
                col_names_to_preserve = [str(x) for x in list(df_ice.columns)]
                dt_ice = datatable.Frame(df_ice)
                dt_ice.names = col_names_to_preserve
                dt_ice.to_jay(df_path)

                sequence += 1

        index_json[ICE.JSON_NEXT_ID] = sequence

        super()._save_json(index_json, path, overwrite=True)

        # fluent API
        return self

    def load(self, path: str, row_index: int = None):
        """EXPERIMENTAL: persistence format may change without deprecation.

        Load cached ICE explanations from JSon index file and per feature binary
        frames with ICE.  If path isn't specified, then explanations index is
        loaded from ``explanations.json`` in the current directory

        Parameters
        ----------
        path : str
            Index file path from where to load explanations.
        row_index : int
            Load only ICE for given dataset instance identified by row.

        Returns
        -------
        h2o_sonar.methods.ICE :
            ICE instance.

        """
        if row_index is not None:
            if not isinstance(row_index, int):
                raise ValueError("Row index must be integer")

        self.evict_explanations()
        self._explanations_cache = {}
        index_json, _ = self.__load_index_json(path)
        ice_json = index_json[ICE.JSON_ICE]
        for feature in ice_json:
            self._explanations_cache[feature] = {}
            for clazz in ice_json[feature]:
                df_path = ice_json[feature][clazz]
                # TODO potential PROBLEM .jay turns all column
                #   label types to string e.g. 1.3 label becomes `1.3`
                df_ice = datatable.fread(df_path)
                if row_index is None:
                    df_ice = df_ice.to_pandas()
                    self._explanations_cache[feature][clazz] = df_ice
                else:
                    row = df_ice[row_index, :]
                    self._explanations_cache[feature][clazz] = row.to_pandas()

        # fluent API
        return self

    def __load_index_json(self, path):
        path = self.__resolve_json_path(path)
        path = self._check_json_path_existence(path)

        with open(path, encoding="utf-8") as fp:
            index_json = json.load(fp)
            if ICE.JSON_ICE not in index_json:
                raise errors.MliJsonDeserializationError(
                    f"JSon file key {ICE.JSON_ICE} is missing in index"
                )
            if ICE.JSON_NEXT_ID not in index_json:
                raise errors.MliJsonDeserializationError(
                    f"JSon file key {ICE.JSON_NEXT_ID} is missing in index"
                )
            next_id = index_json[ICE.JSON_NEXT_ID]
            if not isinstance(next_id, int):
                raise errors.MliJsonDeserializationError(
                    f"JSon file key {ICE.JSON_NEXT_ID} must be integer"
                )

        return index_json, next_id
