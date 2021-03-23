# Signal Processing Algorithms

A suite of algorithms implementing [Energy Statistics](https://en.wikipedia.org/wiki/Energy_distance), 
[E-Divisive with Means](https://arxiv.org/pdf/1306.4933.pdf) and 
[Generalized ESD Test for Outliers](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm) in python.

## Getting Started - Users
```
pip install signal-processing-algorithms
```

## Getting Started - Developers

Getting the code:

```
$ git clone git@github.com:mongodb/signal-processing-algorithms.git
$ cd signal-processing-algorithms
```

Installation
```
$ pip install poetry
$ poetry install
```
Testing/linting:
```
$ poetry run pytest
```

Running the slow tests:
```
$ poetry run pytest --runslow
```

**Some of the larger tests can take a significant amount of time (more than 2 hours).**

## Energy statistics
[Energy Statistics](https://en.wikipedia.org/wiki/Energy_distance) is the statistical concept of Energy Distance 
and can be used to measure how similar/different two distributions are.

For statistical samples from two random variables X and Y:
x1, x2, ..., xn and y1, y2, ..., yn

E-Statistic is defined as:

<a href="https://www.codecogs.com/eqnedit.php?latex=E_{n,m}(X,Y):=2A-B-C" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E_{n,m}(X,Y):=2A-B-C" title="E_{n,m}(X,Y):=2A-B-C" /></a>

where,

<a href="https://www.codecogs.com/eqnedit.php?latex=A:={\frac&space;{1}{nm}}\sum&space;_{i=1}^{n}\sum&space;_{j=1}^{m}\|x_{i}-y_{j}\|,B:={\frac&space;{1}{n^{2}}}\sum&space;_{i=1}^{n}\sum&space;_{j=1}^{n}\|x_{i}-x_{j}\|,C:={\frac&space;{1}{m^{2}}}\sum&space;_{i=1}^{m}\sum&space;_{j=1}^{m}\|y_{i}-y_{j}\|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A:={\frac&space;{1}{nm}}\sum&space;_{i=1}^{n}\sum&space;_{j=1}^{m}\|x_{i}-y_{j}\|,B:={\frac&space;{1}{n^{2}}}\sum&space;_{i=1}^{n}\sum&space;_{j=1}^{n}\|x_{i}-x_{j}\|,C:={\frac&space;{1}{m^{2}}}\sum&space;_{i=1}^{m}\sum&space;_{j=1}^{m}\|y_{i}-y_{j}\|" title="A:={\frac {1}{nm}}\sum _{i=1}^{n}\sum _{j=1}^{m}\|x_{i}-y_{j}\|,B:={\frac {1}{n^{2}}}\sum _{i=1}^{n}\sum _{j=1}^{n}\|x_{i}-x_{j}\|,C:={\frac {1}{m^{2}}}\sum _{i=1}^{m}\sum _{j=1}^{m}\|y_{i}-y_{j}\|" /></a>

T-statistic is defined as: 

<a href="https://www.codecogs.com/eqnedit.php?latex=T={\frac&space;{nm}{n&plus;m}}E_{{n,m}}(X,Y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T={\frac&space;{nm}{n&plus;m}}E_{{n,m}}(X,Y)" title="T={\frac {nm}{n+m}}E_{{n,m}}(X,Y)" /></a>

E-coefficient of inhomogeneity is defined as:

<a href="https://www.codecogs.com/eqnedit.php?latex=H=\frac{2E||X-Y||&space;-&space;E||X-X'||&space;-&space;E||Y-Y'||}{2E||X-Y||}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?H=\frac{2E||X-Y||&space;-&space;E||X-X'||&space;-&space;E||Y-Y'||}{2E||X-Y||}" title="H=\frac{2E||X-Y|| - E||X-X'|| - E||Y-Y'||}{2E||X-Y||}" /></a>


```
from signal_processing_algorithms.energy_statistics import energy_statistics
from some_module import series1, series2

# To get Energy Statistics of the distributions.
es = energy_statistics.get_energy_statistics(series1, series2)

# To get Energy Statistics and permutation test results of the distributions.
es_with_probabilities = energy_statistics.get_energy_statistics_and_probabilities(series1, series2, permutations=100)

```

## Intro to E-Divisive

Detecting distributional changes in a series of numerical values can be surprisingly difficult. Simple systems based on thresholds or
 mean values can be yield false positives due to outliers in the data, and will fail to detect changes in the noise
 profile of the series you are analyzing.
 
One robust way of detecting many of the changes missed by other approaches is to use [E-Divisive with Means](https://arxiv.org/pdf/1306.4933.pdf), an energy
 statistic based approach that compares the expected distance (Euclidean norm) between samples of two portions of the
 series with the expected distance between samples within those portions.
 
That is to say, assuming that the two portions can each be modeled as i.i.d. samples drawn from distinct random variables
 (X for the first portion, Y for the second portion), you would expect the E-statistic to be non-zero if there is a
 difference between the two portions: 
 
 <a href="https://www.codecogs.com/eqnedit.php?latex=E_{n,m}(X,Y):=2A-B-C" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E_{n,m}(X,Y):=2A-B-C" title="E_{n,m}(X,Y):=2A-B-C" /></a>
 where A, B and C are as defined in the Energy Statistics above.

One can prove that <a href="https://www.codecogs.com/eqnedit.php?latex={E_{n,m}(X,Y)\geq&space;0}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?{E_{n,m}(X,Y)\geq&space;0}" title="{E_{n,m}(X,Y)\geq 0}" /></a> and that the corresponding population value is zero if and only if X and Y have the same distribution. Under this null hypothesis the test statistic

<a href="https://www.codecogs.com/eqnedit.php?latex=T={\frac&space;{nm}{n&plus;m}}E_{{n,m}}(X,Y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T={\frac&space;{nm}{n&plus;m}}E_{{n,m}}(X,Y)" title="T={\frac {nm}{n+m}}E_{{n,m}}(X,Y)" /></a>

converges in distribution to a quadratic form of independent standard normal random variables. Under the alternative hypothesis T tends to infinity. This makes it possible to construct a consistent statistical test, the energy test for equal distributions
  
Thus for a series Z of length L,

<a href="https://www.codecogs.com/eqnedit.php?latex=Z&space;=&space;\{Z_{1},&space;...,&space;Z_{\tau}&space;,&space;...&space;,&space;Z_{L}\},&space;X&space;=\{Z_{1},...,Z_{\tau}\},&space;Y=\{Z_{\tau&plus;1}\,...,Z_{L}\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Z&space;=&space;\{Z_{1},&space;...,&space;Z_{\tau}&space;,&space;...&space;,&space;Z_{L}\},&space;X&space;=\{Z_{1},...,Z_{\tau}\},&space;Y=\{Z_{\tau&plus;1}\,...,Z_{L}\}" title="Z = \{Z_{1}, ..., Z_{\tau} , ... , Z_{L}\}, X =\{Z_{1},...,Z_{\tau}\}, Y=\{Z_{\tau+1}\,...,Z_{L}\}" /></a>

we find the most likely change point by solving the following for &tau; such that it has the maximum T statistic value.

### Multiple Change Points

The algorithm for finding multiple change points is also simple.

Assuming you have some k known change points:
1. Partition the series into segments between/around these change points.
2. Find the maximum value of our divergence metric _within_ each partition.
3. Take the maximum of the maxima we have just found --> this is our k+1th change point.
4. Return to step 1 and continue until reaching your stopping criterion.

### Stopping Criterion

In this package we have implemented a permutation based test as a stopping criterion:

After step 3 of the multiple change point procedure above, randomly permute all of the data _within_ each cluster, and
 find the most likely change point for this permuted data using the procedure laid out above. 
 
After performing this operation z times, count the number of
 permuted change points z' that have higher divergence metrics than the change point you calculated with un-permuted data.
 The significance level of your change point is thus z'/(z+1). 

We allow users to configure a permutation tester with `pvalue`
 and `permutations` representing the significance cutoff for algorithm termination and permutations to perform for each
 test, respectively.
 
### Example
```
from signal_processing_algorithms.energy_statistics import energy_statistics
from some_module import series

change_points = energy_statistics.e_divisive(series, pvalue=0.01, permutations=100)
```

