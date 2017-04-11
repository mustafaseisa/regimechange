"""Tools for identifying when one or more discrete regime changes
has occurred in a bivariate time series setting."""

import numpy as np
from scipy.stats import norm as gaussian

METRICS = { # library of common metrics which define a state change
    'correlation': lambda ts: np.corrcoef(ts[:, 0], ts[:, 1])[0, 1],
    'tracking error': lambda ts: np.std(ts[:, 0]-ts[:, 1]),
    'excess return': lambda ts: np.mean(ts[:, 0] - ts[:, 1]),
    'excess volatility': lambda ts: (1/ts.shape[0]) * np.linalg.norm(
        ts[:, 0] - ts[:, 1], 2
        )
}

KERNELS = { # library of kernels for estimating local regime changes
    'gaussian': lambda age, bw: gaussian.pdf(range(age), scale=bw),
    'triangular': lambda age, bw: np.maximum(0, 1-(1/bw)*np.arange(age)),
    'hyperbolic': lambda age, bw: 1/np.arange(1, age+1)**(1/bw),
    'uniform': lambda age, bw: np.array([1]*min(bw, age) + [0]*max(0, age-bw))
}

def kernel_split(time_series, metric, kernel, bandwidth=10, pad=5):
    """Detection of some instantaneous, potentially local state change.

    Given bivariate time series, metric defining a state change, and
    a weighting kernel defining local fidelity, estimates the date at
    which the two regimes are maximally different with respect to the
    provided metric. Function then returns the date a new regime
    begins.

    @arg    {np.array}  time_series 2D array containing time series data
                                    with dates in ascending order along
                                    axis 0 and assets along axis 1.'

    @arg    {function}  metric      A metric of interest that will define
                                    the state change between the two time
                                    series. Must be a function that
                                    takes a 2D array of the same format
                                    as the time_series argument and
                                    returns a scalar value defining the
                                    state between the two series (eg.
                                    correlation coefficient).

    @arg    {int}       pad         The number of observations on each
                                    end of the time series that are not
                                    considered to be possible points of
                                    state change. Minimum is two to allow
                                    statisitcal estimators like Pearson
                                    correlation coefficient to have
                                    sufficient degrees of freedom.
                                    Maximum is such that there after
                                    padding, there are at least two dates
                                    under consideration as points of
                                    state change.

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

        regime_a = time_series[:partition]
        regime_b = time_series[partition:]

        regime_discrepancy.append(
            abs(
                metric(regime_a) - metric(regime_b)
                )
            )

    return np.argmax(regime_discrepancy) + pad
