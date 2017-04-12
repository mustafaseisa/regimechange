"""Local and non-parametric methods for identifying when discrete regime
changes have occurred in a bivariate time series setting."""

import numpy as np
from scipy.stats import norm as gaussian

METRICS = { # library of common metrics which define a state change
    'correlation': lambda ts, w: _weighted_corr(ts[:, 0], ts[:, 1], w),
    'tracking error': lambda ts, w: _weighted_std(ts[:, 0] - ts[:, 1], w),
    'excess return': lambda ts, w: np.average(ts[:, 0] - ts[:, 1], weights=w),
    'excess volatility': lambda ts, w: _weighted_2ndmom(ts[:, 0] - ts[:, 1], w)
}

KERNELS = { # library of kernels for estimating local regime changes
    'gaussian': lambda age, bw: gaussian.pdf(range(age), scale=bw),
    'hyperbolic': lambda age, bw: 1/np.arange(1, age+1)**(1/bw),
    'triangular': lambda age, bw: np.maximum(0, 1-(1/bw)*np.arange(age)),
    'uniform': lambda age, bw: np.array([1]*min(int(bw), age) + \
                                            [0]*max(0, age-int(bw)))
}

def _weighted_2ndmom(array, weights):
    """Weighted uncentralized second moment."""

    weighted_var = np.average(array**2, weights=weights)
    return np.sqrt(weighted_var)

def _weighted_std(array, weights):
    """Weighted standard deviation."""

    weighted_mean = np.average(array, weights=weights)
    weighted_var = np.average((array - weighted_mean)**2, weights=weights)
    return np.sqrt((array.shape[0]/(array.shape[0]-1)) * weighted_var)

def _weighted_corr(array_x, array_y, weights):
    """Weighted standard deviation."""

    mean_x = np.average(array_x, weights=weights)
    mean_y = np.average(array_y, weights=weights)

    rss_x = np.linalg.norm(weights*(array_x - mean_x), 2)
    rss_y = np.linalg.norm(weights*(array_y - mean_y), 2)

    inner_prod = (array_x - mean_x).T.dot(weights * (array_y - mean_y))

    return inner_prod/(rss_x * rss_y)

def kernel_split(time_series, metric, kernel, bandwidth=10, pad=5):
    """Detection of some instantaneous, potentially local state change.

    Given a bivariate time series, metric defining a state change, and
    a weighting kernel controling fidelity to local information,
    estimates the date at which a regime change has occurred with respect
    to the provided metric. Specifically, the function returns the date a
    the new regime begins.

    @arg    {np.array}  time_series 2D array containing time series data
                                    with dates in ascending order along
                                    axis 0 and assets along axis 1.

    @arg    {function}  metric      A metric of interest that will define
                                    the state change between the two time
                                    series.

            @arg        {np.array}  Bivariate time series; same format as
                                    time_series above. Will be used as
                                    data for which metric is calculated.

            @arg        {np.array}  Flat np.array of same length as
                                    previous argument; used to weight
                                    observations in calculation of metric
                                    of interest.

            @arg        {float}     The metric of interest returned as a
                                    scalar.

    @arg    {function}  kernel      A kernel of defining fidelity to
                                    local regime changes.

            @arg        {int}       The length of the sequence of weights
                                    outputted by the kernel function.

            @arg        {float}     Bandwidth controlling fidelity of
                                    the kernel to local information.

            @return     {np.array}  Array of positive floats defining
                                    sequence of (typically decaying)
                                    kernel weights. Need not be
                                    normalized as kernel_split method
                                    will perform the normalization
                                    internally.

    @arg    {float}     bandwidth   Kernel bandwidth to be passed to the
                                    supplied kernel function. Forced to
                                    be be greater than or equal to one to
                                    accomodate KNN and other discrete
                                    kernels.

    @arg    {int}       pad         The number of observations on each
                                    end of the time series that are not
                                    considered to be possible points of
                                    state change. Minimum is two to allow
                                    statistical estimators like standard
                                    deviation to have sufficient degrees
                                    of freedom. Maximum is such that
                                    there after padding, there are at
                                    least two dates under consideration
                                    as points of state change.

    @return {int}       The index corresponding to the date the new
                        regime begins.
    """

    # typechecks and fail safety:

    assert bandwidth >= 1, 'Bandwidth parameter must be greater than or equal '\
                            'to one.'

    assert isinstance(pad, int), 'Argument pad must be a positive integer.'
    assert pad >= 2, 'At least two observations must be padded on each end ' \
                        'of the time series array.'

    assert isinstance(time_series, np.ndarray), 'Time series must be numpy ' \
                                                     'array.'
    assert np.issubdtype(time_series.dtype, np.number), 'Time series array ' \
                                    'can only contain only numerical values.'
    assert time_series.ndim == 2, 'Time series array must be 2D with dates ' \
                        'ascending along axis 0 and assets along axis 1.'
    assert (time_series != np.nan).all() and (time_series != np.infty).all(), \
                'Time series array cannot contain missing or infinite values.'
    num_dates, num_assets = time_series.shape # dimensions of data
    assert num_assets == 2, 'Time series array can only contain two assets.'
    assert num_dates - 2*pad >= 3, 'Time series must have at least five ' \
                                        'observations after padding.'

    # estimating breakpoint

    regime_discrepancy = [] # will store diff between every split of regimes

    for partition in range(pad, num_dates-pad):

        regime_a = time_series[:partition] # left partition of time series
        weights_a = kernel(partition, bandwidth)[::-1]

        regime_b = time_series[partition:] # right partition of time series
        weights_b = kernel(num_dates - partition, bandwidth)

        regime_discrepancy.append(
            abs(
                metric(regime_a, weights_a) - metric(regime_b, weights_b)
                )
            )

    return np.argmax(regime_discrepancy) + pad
