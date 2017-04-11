
# The REGIME CHANGE Package
**MUSTAFA S EISA** \ 10 APRIL 2017


```python
from matplotlib import pyplot as plt
import regimechange as rg
import numpy as np

plt.style.use('ggplot')
```


```python
%matplotlib inline
```

The package `regimechange` contains the tools for estimating regime changes in a bivariate time series setting.


```python
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
    
    


The `METRICS` dictionary contains a set of key metrics that define a state change. Let's go through a few examples to make this clear.

### Example: Regime change with respect to $\beta$


```python
benchmark = np.random.normal(size=(100,1)) # some benchmark index
tracking = benchmark.copy() + .5*np.random.normal(size=(100,1)) # fund tracking benchmark
tracking[68:] = -1*tracking[68:] # flip relationship at day 68
```


```python
plt.figure(figsize=(12, 6))
plt.axvline(x=68, color = 'orange', label='regime change', linewidth=3)
plt.plot(benchmark, label='benchmark index', linewidth=2)
plt.plot(tracking, label='tracking fund', linestyle='--')
plt.legend()
plt.show()
```


![png](output_8_0.png)



```python
# we can estimate regime change date this using the kernel_split method

data = np.hstack((benchmark, tracking))
rg.kernel_split(data, rg.METRICS.get('beta'))
```




    68



### Example: Regime change with respect to Tracking Error


```python
# tracking error blows up at day 40

benchmark = np.random.normal(size=(100,1)) # some benchmark index
tracking = benchmark.copy() + .1*np.random.normal(size=(100,1)) # fund tracking benchmark
tracking[40:] = tracking[40:] + np.random.normal(size=(60,1)) # tracking error blows up at day 40
```


```python
plt.figure(figsize=(12, 6))
plt.axvline(x=40, color = 'orange', label='regime change', linewidth=3)
plt.plot(benchmark, label='benchmark index', linewidth=2)
plt.plot(tracking, label='tracking fund', linestyle='--')
plt.legend()
plt.show()
```


![png](output_12_0.png)



```python
# we can estimate regime change date this using the kernel_split method

data = np.hstack((benchmark, tracking))
rg.kernel_split(data, rg.METRICS.get('tracking error'))
```




    41



### Speed Test


```python
%timeit rg.kernel_split(data, rg.METRICS.get('tracking error'))
```

    100 loops, best of 3: 4.91 ms per loop

