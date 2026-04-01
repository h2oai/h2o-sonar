# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import collections

import numpy as np

from h2o_sonar import loggers


KolmogorovSmirnovResult = collections.namedtuple(
    typename="KolmogorovSmirnovResult",
    field_names=["statistic", "p_value", "same_distribution", "p_value_method"],
)


def kolmogorov_smirnov(
    sample_u: list,
    sample_v: list,
    p_calc_method: str = "",
    logger: loggers.SonarLogger | None = None,
) -> KolmogorovSmirnovResult:
    """Discrete Kolmogorov-Smirnov (KS) test for two samples.

    Compare two samples and understand if they come from the same discrete distribution.

    KS metric interpretation: The KS statistic is the maximum distance between the
    empirical cumulative distribution functions (ECDFs) of the two samples. A larger
    KS statistic indicates a greater difference between the two distributions.
    The p-value indicates the probability of observing a KS statistic at least as
    extreme as the one calculated, assuming the null hypothesis (that the two samples
    come from the same distribution) is true. A small p-value (typically < 0.05)
    suggests that the null hypothesis can be rejected, indicating a significant
    difference between the two distributions.
    The KS test is sensitive to differences in both location and shape of the
    empirical cumulative distribution functions of the two samples.
    The KS test is non-parametric, meaning it does not assume any specific
    distribution for the data. This makes it a versatile tool for comparing
    distributions, especially when the underlying distributions are unknown or
    not normally distributed.

    Parameters
    ----------
    sample_u : list
        First sample.
    sample_v : list
       Second sample.
    p_calc_method : str
       Method to calculate p-value - options are "auto", "exact", and "asymp".

       - ``exact`` method: exact distribution of test statistic - used w/
         "auto" for small samples
       - ``asymp`` method: asymptotic distribution of test statistic - used w/
         "auto" for large samples

    logger : loggers.SonarLogger | None
        Logger.

    Returns
    -------
    KolmogorovSmirnovResult :
        Kolmogorov-Smirnov statistic (``0.0`` meaning perfect agreement,
        ``1.0`` disagreement), p-value (hypothesis testing), same distribution flag,
        and p-value method.

    References
    ----------
    .. [1] scipy.stats.ks_2samp:
           https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html

    """
    from scipy import stats as scipy_stats

    logger = logger or loggers.SonarPrintLogger()

    # IMPROVE return bool w/ guess whether is the same distribution or not (p-value)
    p_calc_method = p_calc_method or "auto"
    try:
        result = scipy_stats.ks_2samp(sample_u, sample_v, method=p_calc_method)
        return KolmogorovSmirnovResult(
            result.statistic, result.pvalue, bool(result.pvalue >= 0.05), p_calc_method
        )
    except ValueError as e:
        if p_calc_method == "auto":
            raise e

        logger.warning(
            f"Could not use p-value calculation method '{p_calc_method}' - falling "
            f"back to 'auto' method: {e} "
        )
        result = scipy_stats.ks_2samp(sample_u, sample_v, method="auto")
        return KolmogorovSmirnovResult(
            result.statistic, result.pvalue, bool(result.pvalue >= 0.05), p_calc_method
        )


def wasserstein_distance(
    sample_u: list,
    sample_v: list,
) -> float:
    """Calculate the Wasserstein distance between two distributions.

    - The function assumes values are sorted, but handles sorting internally if not.
    - The function works correctly even if the value arrays don't perfectly overlap.

     Wasserstein distance interpretation: the distance represents the minimum "cost"
     (amount of probability mass multiplied by the distance moved) required to
     transform distribution 1 into distribution 2.

     Parameters
    ----------
    sample_u : list
        First sample.
    sample_v : list
       Second sample.

    Returns
    -------
    float
        Wasserstein distance between the two distributions (lower value meaning higher
        agreement).

    References
    ----------
    .. [1] scipy.stats.wasserstein_distance:
           https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html

    """
    from scipy import stats as scipy_stats

    # get unique values and their frequencies
    values_u, counts_u = np.unique(sample_u, return_counts=True)
    values_v, counts_v = np.unique(sample_v, return_counts=True)
    # normalize counts to probabilities
    weights_u = counts_u / np.sum(counts_u)
    weights_v = counts_v / np.sum(counts_v)

    w_distance = scipy_stats.wasserstein_distance(
        u_values=values_u.tolist(),
        v_values=values_v.tolist(),
        u_weights=weights_u,
        v_weights=weights_v,
    )

    return w_distance


def jensen_shannon_divergence(
    sample_u: list,
    sample_v: list,
) -> float:
    """Calculate the Jensen-Shannon divergence (not distance) between two distributions.

    Parameters
    ----------
    sample_u : list
        First probability distribution.
    sample_v : list
        Second probability distribution.

    Returns
    -------
    float
        Jensen-Shannon divergence between the two distributions.

    """
    from scipy.spatial import distance as scipy_distance

    # transform the samples to distributions of probabilities
    samples = sample_u + sample_v
    values = np.unique(samples)
    # bet counts for each sample, adding missing values with zero counts
    counts_u = np.array([sample_u.count(v) for v in values])
    counts_v = np.array([sample_v.count(v) for v in values])
    # normalize counts to probabilities
    p = counts_u / counts_u.sum()
    q = counts_v / counts_v.sum()

    # ensure they are valid probability distributions
    assert len(p) == len(q), "Probability distributions must have the same length"
    assert np.isclose(np.sum(p), 1.0), (
        "Probabilities distribution of the sample u must sum to 1.0"
    )
    assert np.isclose(np.sum(q), 1.0), (
        "Probabilities distribution of the sample v must sum to 1.0"
    )

    # calculate the Jensen-Shannon distance and convert to divergence
    return float(scipy_distance.jensenshannon(p=p, q=q)) ** 2
