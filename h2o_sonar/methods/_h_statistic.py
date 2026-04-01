# Copyright (C) 2018-2026 H2O.ai, Inc. All rights reserved
import itertools
import statistics as stat

import numpy

from h2o_sonar import loggers as logging
from h2o_sonar.methods import _pd
from h2o_sonar.methods.core import _caching_method
from h2o_sonar.methods.core import method
from h2o_sonar.utils import progress


class HStatistic(method.Method, _caching_method.CachingMethod):
    """Implementation of Friedman's H-statistic feature interactions methods.

    The statistic is 0 when there is no interaction at all and 1 if all of the
    variances of the predict function is explained by the sum of the partial
    dependence functions.

    H-statistic binning: see ICE for binning and out of range binning documentation.

    See also
    --------
    https://christophm.github.io/interpretable-ml-book/interaction.html#\
    theory-friedmans-h-statistic

    """

    METHOD_TYPE = "h-statistics"

    @property
    def opt_1_prediction(self):
        """Option which controls ICE computation strategy for H-statistic.
        See PD documentation for more details.

        """
        return self._1_prediction

    @opt_1_prediction.setter
    def opt_1_prediction(self, allow_1_predict: bool):
        self._1_prediction = allow_1_predict

    def __init__(self, name, interpretable_model=None):
        """Create H-statistic instance.

        Parameters
        ----------
        name : str
            Name of this H-statistic methods.
        interpretable_model : InterpretableModel
            Interpretable model whose predict_method and current directory
            should be used.

        """
        method.Method.__init__(
            self,
            method_name=name,
            method_type=HStatistic.METHOD_TYPE,
            interpretable_model=interpretable_model,
        )
        _caching_method.CachingMethod.__init__(self)

        self._bins = None
        self._bins_dict = None
        self._fs_meta = None
        self._1_prediction = True

        self._progress_callback: progress.AbstractProgressCallbackContext | None = None

        logging.setLevel(logging.DEBUG)

    def __str__(self):
        result = ""
        if self._explanations_cache is not None:
            for feature in self._explanations_cache:
                result = f"{result}\n'{feature}':"
                for p_type in self._explanations_cache[feature]:
                    result = (
                        f"{result}\n  '{feature}':\n"
                        f"    {self._explanations_cache[feature][p_type]}"
                    )
        result += "\n"

        return result

    # IMPROVE: explain() method signatures don't fit yet
    def explain(
        self,
        features,
        X,
        predict_method=None,
        bins=None,
        features_meta=None,
        progress_callback: progress.AbstractProgressCallbackContext | None = None,
    ):
        """Calculate H-statistic for pairs of given features and cache it in
        memory so  that it can be subsequently obtained using
        `explanations()` method.

        Parameters
        ----------
        features : list[int or str]
            A list of features (strings or numerical ids) for which
            H-statistic is supposed to be calculated. H-statistic is computed
            for all feature pairs.
        X : pandas.core.frame.DataFrame or h2o.H2OFrame or datatable.Frame
            Original data for which should be partial dependence computed
        predict_method : function
            A lambda function which takes instances (a set of rows) and
            outputs predictions. If not specified, then interpretable model's
            predict_method is used.
        bins : list[list[object]]
            Per feature values.
        features_meta : dict
            Features metadata allowing to indicate whether given feature is
            categorical (use ``categorical`` key and list of feature names).
            Use ``quantiles`` dict key and list of feature names to
            ensure bins construction using quantiles instead of even split.
        progress_callback : progress.AbstractProgressCallbackContext | None
            Progress callback allowing the progress of PD calculation.

        Returns
        -------
        h2o_sonar.methods.HStatistic :
            HStatistic instance to get computed explanations using
            `explanations()` method.

        Raises
        ------
        MliUnsupportedDataFormatError :
            If input parameters are not in expected format.
        MliUnsupportedOperationError :
            If H-statistic is required to be computed on unsupported
            feature types
        MliPredictMethodError :
            If predict method fails.

        """
        predict_method = super()._method_precondition(predict_method)

        if not features:
            raise ValueError("Features must be specified")
        if not isinstance(features, list):
            raise ValueError("Feature names must be in list")
        if len(features) < 2:
            raise ValueError(
                "Provide at least two features to calculate the interactions"
            )

        self._fs_meta = features_meta

        if bins:
            if not isinstance(bins, list):
                raise ValueError("Bins must be in list")
            if len(features) != len(bins):
                raise ValueError("Every feature must have bin")

            self._bins = bins
            self._bins_dict = {}
            for i, _ in enumerate(features):
                self._bins_dict[features[i]] = bins[i]

        self.diagnostics.add_scorer_calls_slot()
        self._progress_callback = progress_callback

        h_pairs = self._explain_pairs(features, X, predict_method)

        self._explanations_cache = h_pairs

        return self

    def _explain_pairs(self, features, X, predict_method):
        """Compute H-statistic for pairs."""

        feature_pairs = []
        for i in itertools.combinations(features, 2):
            feature_pairs.append(i)
        pairs_bins = None
        if self._bins:
            pairs_bins = []
            for fp in feature_pairs:
                pairs_bins.append(
                    tuple([self._bins_dict[fp[0]], self._bins_dict[fp[1]]])
                )

        # logging.debug("Feature pairs: {}".format(feature_pairs))
        # logging.debug("Pairs bins: {}".format(pairs_bins))

        progress_callback = (
            progress.ProgressCallbackContext(
                total_steps=2,  # two stages: PD for pairs + features
                parent_callback=self._progress_callback,
            )
            if self._progress_callback
            else None
        )

        # compute PD for all pairs
        pdp = _pd.PD("Pairs PD")
        pdp.opt_1_prediction = self._1_prediction
        pds_2 = pdp.explain(
            feature_pairs,
            X,
            predict_method=predict_method,
            bins=pairs_bins,
            center=True,
            stats=False,
            features_meta=self._fs_meta,
            progress_callback=(
                progress.ProgressCallbackStackingBridge(progress_callback)
                if progress_callback
                else None
            ),
        )
        # logging.debug("PD(f,g):\n{}".format(pds_2))

        # compute PD for single features
        pdp = _pd.PD("Features PD")
        pdp.opt_1_prediction = self._1_prediction
        pds_1 = pdp.explain(
            features,
            X,
            predict_method=predict_method,
            bins=self._bins,
            center=True,
            stats=False,
            features_meta=self._fs_meta,
            progress_callback=(
                progress.ProgressCallbackStackingBridge(progress_callback)
                if progress_callback
                else None
            ),
        )
        # logging.debug("PD(f) PD(g):\n{}".format(pds_1))

        # diagnostics
        self.diagnostics.add_scorer_calls(pds_1.diagnostics.total_scorer_calls)
        self.diagnostics.add_scorer_calls(pds_2.diagnostics.total_scorer_calls)

        # compute H-statistic for each pair
        result = {}
        for fp in feature_pairs:
            # logging.debug("Computing H-stat for: {}".format(fp))
            result[fp] = {}

            num = 0
            den = 0

            result[fp] = {}
            clasess_stats = []
            for clazz in pds_2.explanations()[fp]:
                p_pd = pds_2.explanations()[fp][clazz]
                for col in pds_2.explanations()[fp][clazz]:
                    num = num + numpy.square(
                        p_pd.loc["mean"][col]
                        - pds_1.explanations()[fp[0]][clazz].loc["mean"][col[0]]
                        - pds_1.explanations()[fp[1]][clazz].loc["mean"][col[1]]
                    )
                    den = den + numpy.square(p_pd.loc["mean"][col])
                h_stat = num / den if den else 0

                # if logging.get_level() == logging.DEBUG:
                #    logging.debug(
                #        "Numerator:\n{}\n{}".format(num, type(num)))
                #    logging.debug(
                #        "Denominator:\n{}\n{}".format(den, type(den)))
                #    logging.debug(
                #        "H-stat {}/{}:\n{}".format(fp, clazz, h_stat))

                result[fp][clazz] = h_stat
                clasess_stats.append(h_stat)

            # mean/std/sem for bi/multiclass h-stats
            if len(clasess_stats) > 1:
                result[fp][_pd.PD.COL_MEAN] = stat.mean(clasess_stats)
                result[fp][_pd.PD.COL_SD] = stat.stdev(clasess_stats)
                result[fp][_pd.PD.COL_SEM] = result[fp][_pd.PD.COL_SD] / (
                    len(clasess_stats) ** (1 / 2)
                )

        return result

    def explanations(self):
        return self._explanations_cache
