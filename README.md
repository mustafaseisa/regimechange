
# REGIME CHANGE

The package `regimechange` contains tools for estimating regime changes in a bivariate time series setting. Regime changes can be defined with respect to any given metric (eg. correlation or tracking error) and a kernel weighting parameter that controls the fidelity of the estimator to more local changes.

```python
import regimechange as rg
help(rg)
```

    Help on module regimechange:
    
    NAME
        regimechange
    
    DESCRIPTION
        Tools for identifying when one or more discrete regime changes
        occurred in a bivariate time series setting.
    
    FUNCTIONS
        kernel_split(time_series, metric, kernel='Uniform', pad=5)
            Detection of some instantaneous, potentially local state change.
            
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
    
    DATA
        METRICS = {'beta': <function <lambda>>, 'tracking error': <function <l...
    
    FILE
        /Users/mustafameisa/Desktop/regimechange.py
    
    

The `METRICS` dictionary contains a set of key metrics that define a state change.

### Usage

Let's go through a few examples to make this clear.

```python
from matplotlib import pyplot as plt
import numpy as np
```

Consider a discrete regime change that occurs in with respect to the CAPM Beta metric. Specifically, we'll generate data where one time series is almost perfectly correlated with the other and then, at day 68, the correlation flips.


```python
benchmark = np.random.normal(size=(100,1)) # some benchmark index
tracking = benchmark.copy() + .5*np.random.normal(size=(100,1)) # fund tracking benchmark
tracking[68:] = -1*tracking[68:] # flip relationship at day 68

plt.figure(figsize=(12, 6))
plt.axvline(x=68, color = 'orange', label='regime change', linewidth=3)
plt.plot(benchmark, label='benchmark index', linewidth=2)
plt.plot(tracking, label='tracking fund', linestyle='--')
plt.legend()
plt.show()
```


![beta_regime_change](https://cloud.githubusercontent.com/assets/13667067/24891341/1822beee-1e2a-11e7-8185-a3e65f0eb18e.png)


We can estimate when this regime change occured using the `kernel_split` method:

```python
data = np.hstack((benchmark, tracking))
rg.kernel_split(data, rg.METRICS.get('beta'))

# 68
```

Another example is with the metric tracking error. We'll generate data where one time series tacks the other well then suddenly tracks poorly after day 40.


```python
# tracking error blows up at day 40

benchmark = np.random.normal(size=(100,1)) # some benchmark index
tracking = benchmark.copy() + .1*np.random.normal(size=(100,1)) # fund tracking benchmark
tracking[40:] = tracking[40:] + np.random.normal(size=(60,1)) # tracking error blows up at day 40

plt.figure(figsize=(12, 6))
plt.axvline(x=40, color = 'orange', label='regime change', linewidth=3)
plt.plot(benchmark, label='benchmark index', linewidth=2)
plt.plot(tracking, label='tracking fund', linestyle='--')
plt.legend()
plt.show()
```


![tracking_regime_change](https://cloud.githubusercontent.com/assets/13667067/24891342/1833b410-1e2a-11e7-99b9-88ff995825b5.png)


We can again estimate when this regime change occured using the `kernel_split` method:

```python
data = np.hstack((benchmark, tracking))
rg.kernel_split(data, rg.METRICS.get('tracking error'))

# 41
```



### Speed Test


```python
%timeit rg.kernel_split(data, rg.METRICS.get('tracking error'))
```

    100 loops, best of 3: 4.91 ms per loop


### Future Updates

The following items are scheduled to be included:
    * Kernel parameter for estimating local regime changes
    * Regularization for cases when the two regimes have significantly different number of observations used to estimate the metric of interest (unequal variance)
    * Beyond bivariate regime change
