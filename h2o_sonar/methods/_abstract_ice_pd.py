# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import copy
import math
from abc import abstractmethod

import numpy
import pandas as pd

from h2o_sonar.lib.api import persistences
from h2o_sonar.methods.core import _caching_method


class AbstractIcePd(
    _caching_method.CachingMethod, persistences.JsonPersistableExplanations
):
    """ICE and PD commons.

    Attributes
    ----------
    features: str
        Target features for which will be ICE/PD calculated.

    """

    @property
    def features(self):
        return self._fs

    def __init__(self):
        """Initialize common ICE and PD properties."""
        _caching_method.CachingMethod.__init__(self)
        persistences.JsonPersistableExplanations.__init__(self)

        self._fs = None
        self._fs_meta = None
        self._bins_sort = False
        self._g_resolution = None
        self._center = None
        self._oor = None
        self._oor_blacklist = None
        self._oor_bins = {}
        self._stds = None
        self._target_transform = None

    def __str__(self):
        as_str = ""

        if self._explanations_cache is not None:
            for cached_feature in self._explanations_cache:
                as_str = f"{as_str}\n'{cached_feature}':"
                for p_type in self._explanations_cache[cached_feature]:
                    as_str = (
                        f"{as_str}\n '{p_type}':\n"
                        f"{self._explanations_cache[cached_feature][p_type]}"
                    )

        as_str += "\n"

        return as_str

    def _check_and_set_features(self, features, features_meta):
        if not isinstance(features, list):
            raise ValueError("Features must be of list type")
        if not features:
            raise ValueError("At least one target feature must be specified")
        if not isinstance(features[0], list) and len(features) != len(set(features)):
            # PD/ICE must fail: feature is key in result dict, etc.
            seen = set()
            for feature in features:
                if feature not in seen:
                    seen.add(feature)
                else:
                    raise ValueError(f"Duplicate target feature(s) detected: {feature}")
        self._fs = features
        self._fs_meta = features_meta

    def _check_dataset_features(self, X):
        for feature in self._fs:
            if isinstance(feature, tuple):
                for tuple_f in feature:
                    if tuple_f not in X.columns.values:
                        raise ValueError(
                            f"Tuple feature '{tuple_f}' is not label of any input "
                            f"data column"
                        )
            else:
                if feature not in X.columns.values:
                    raise ValueError(
                        f"Feature '{feature}' is not label of any input data column"
                    )

    @staticmethod
    def _check_bin_duplicates(bin_):
        if len(bin_) != len(set(bin_)):
            raise ValueError("Bins cannot contain duplicate values: " + str(bin_))

    def _check_bins(self, bins):
        if bins is not None:
            if not isinstance(bins, list):
                raise ValueError("Bins must be of list type")
            if not bins:
                raise ValueError("Bins cannot be empty")
            if len(bins) != len(self._fs):
                raise ValueError("Bin count must correspond to target features")

            bins_ = copy.deepcopy(bins)

            for i, feature in enumerate(self._fs):
                if isinstance(feature, tuple):
                    self._check_n_bin(feature, bins_[i])
                else:
                    self._check_bin_duplicates(bins_[i])
        else:
            bins_ = []

        return bins_

    def _check_n_bin(self, feature, n_bin):
        if not isinstance(n_bin, tuple):
            raise ValueError(
                "For multidimensional feature '" + str(feature) + "' bin must be tuple"
            )
        for bin_ in n_bin:
            self._check_bin_duplicates(bin_)

    def _check_resolution(self, X, grid_resolution):
        if X is None:
            raise ValueError("Data cannot be undefined")

        if grid_resolution is not None and not isinstance(grid_resolution, int):
            raise ValueError("Resolution must be integer")
        if grid_resolution < 0:
            raise ValueError("Resolution must not be negative integer")
        self._g_resolution = grid_resolution

    @staticmethod
    def get_pandas_super_type(dtype1, dtype2):
        """Infer the most generic dtype: bool > int > float > cat"""
        # IMPROVE this function is inefficient, but I didn't find better way
        # how to implement it, ideally I would need this method signature, but
        # from Pandas (or numpy, and later from datatable).
        i_1 = AbstractIcePd.__pandas_instance_for_dtype(dtype1)
        i_2 = AbstractIcePd.__pandas_instance_for_dtype(dtype2)
        return pd.Series([i_1, i_2]).dtype

    @staticmethod
    def __pandas_instance_for_dtype(dtype):
        if pd.api.types.is_bool_dtype(dtype):
            return True
        if pd.api.types.is_integer_dtype(dtype):
            return 42
        if pd.api.types.is_float_dtype(dtype):
            return 3.14
        if pd.api.types.is_string_dtype(dtype):
            return "str"
        # date is inferred to object

        return {}

    @staticmethod
    def create_numerical_bins_int(grid_resolution: int, bins, max_, min_):
        """Create of int bins for int feature."""

        if max_ == min_:
            bins.append([max_])
            return bins
        if grid_resolution == 1:
            bins.append([max_])
            return bins
        if grid_resolution == 2:
            bins.append([min_, max_])
            return bins

        feature_bins = []

        if (max_ - min_ + 1) <= grid_resolution:
            for bin_ in range(min_, max_ + 1):
                feature_bins.append(bin_)
            bins.append(feature_bins)
            return bins

        # float bins > round (-- bin on clash) > complete w/ iteration if not enough
        float_inc: float = (float(max_) - float(min_)) / float(grid_resolution)
        watermark: float = float(min_)
        feature_bins.append(min_)
        while len(feature_bins) < grid_resolution and watermark < max_:
            watermark += float_inc

            bin_ = int(watermark + float_inc)
            if bin_ not in feature_bins:
                feature_bins.append(bin_)
            elif (bin_ - 1) not in feature_bins:
                feature_bins.append(bin_ - 1)
            # else: will try to complete missing bins at the end

        if len(feature_bins) < grid_resolution:
            for bin_ in range(min_, max_ + 1):
                if bin_ not in feature_bins:
                    feature_bins.append(bin_)
                    if len(feature_bins) >= grid_resolution:
                        break

        bins.append(feature_bins)
        return bins

    @staticmethod
    def create_numerical_bins_float(grid_resolution: int, bins, max_, min_):
        # re-calibrate bins for every feature
        step = (max_ - min_) / grid_resolution
        if min_ == max_:
            bins.append([max_])
        elif step <= 0:
            bins.append([(min_ + max_) / 2.0])
        elif math.isnan(min_) or math.isnan(max_) or math.isnan(step):
            bins.append([float("NaN")])
        else:
            bins.append(numpy.append(numpy.arange(min_, max_, step), max_).tolist())

    @staticmethod
    def create_oor_bins(
        feature_dtype: str, min_, max_, std_dev, out_of_range_resolution
    ):
        """Create out of range bins.

        Parameters
        ----------
        feature_dtype : str
            Feature data type: ``"u"`` (unsigned integer), ``"i"`` (signed integer),
            ``"f"`` (float).
        min_: int or float
            Regular bin minimum.
        max_: int or float
            Regular bin maximum.
        std_dev :
            Regular bin standard deviation.
        out_of_range_resolution : int
            Number of out of range bins to create below/above regular bin.

        Returns
        -------
        list :
            Out of range bin.

        """
        if feature_dtype in "iu":
            return AbstractIcePd.create_oor_bins_int(
                feature_dtype=feature_dtype,
                min_=min_,
                max_=max_,
                std_dev=std_dev,
                out_of_range_resolution=out_of_range_resolution,
            )

        return AbstractIcePd.create_oor_bins_float(
            min_=min_,
            max_=max_,
            std_dev=std_dev,
            out_of_range_resolution=out_of_range_resolution,
        )

    @staticmethod
    def create_oor_bins_int(
        feature_dtype: str, min_, max_, std_dev, out_of_range_resolution
    ):
        oor_bins = []
        int_std = max(int(std_dev), 1)
        if feature_dtype in "u":
            if min_ - int_std * out_of_range_resolution >= 0:
                for i in range(1, out_of_range_resolution + 1):
                    oor_bins.append(min_ - int_std * i)
            elif min_ - out_of_range_resolution >= 0:
                for i in range(1, out_of_range_resolution + 1):
                    oor_bin = min_ - i
                    if oor_bin:
                        oor_bins.append(oor_bin)

        elif feature_dtype in "i":
            for i in range(1, out_of_range_resolution + 1):
                oor_bins.append(min_ - int_std * i)

        for i in range(1, out_of_range_resolution + 1):
            oor_bins.append(max_ + int_std * i)

        return oor_bins

    @staticmethod
    def create_oor_bins_float(min_, max_, std_dev, out_of_range_resolution):
        # min(feature) - n*sd(feature)
        oor_bins = [min_ - i * std_dev for i in range(1, out_of_range_resolution + 1)]
        # max(feature) + n*sd(feature)
        oor_bins.extend(
            [max_ + i * std_dev for i in range(1, out_of_range_resolution + 1)]
        )

        return oor_bins

    @property
    def oor_bins(self):
        return self._oor_bins

    def _add_oor_bin(self, feature, oor_bin):
        """Add OOR bin to OOR bins index. OOR bins are same for all classes (in case
        of binomial and multinomial classification).

        Parameters
        ----------
        feature: str
          Name of the feature.
        oor_bin:
          OOR bin(s) of the feature.

        """
        if feature not in self._oor_bins:
            self._oor_bins[feature] = []
        if oor_bin:
            if isinstance(oor_bin, list):
                self._oor_bins[feature].extend(oor_bin)
            else:
                self._oor_bins[feature].append(oor_bin)

    @staticmethod
    def strip_none(bin_):
        assert isinstance(bin_, list), "Parameter must be list"
        return [i for i in bin_ if i is not None]

    @abstractmethod
    def save_json(self, path=None):
        pass

    @abstractmethod
    def load_json(self, path=None):
        pass
