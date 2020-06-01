# Signal Processing Algorithms

A suite of algorithms implementing [E-Divisive with Means](https://arxiv.org/pdf/1306.4933.pdf) and
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

## Intro to E-Divisive

Detecting distributional changes in a series of numerical values can be surprisingly difficult. Simple systems based on thresholds or
 mean values can be yield false positives due to outliers in the data, and will fail to detect changes in the noise
 profile of the series you are analyzing.
 
One robust way of detecting many of the changes missed by other approaches is to use [E-Divisive with Means](https://arxiv.org/pdf/1306.4933.pdf), an energy
 statistic based approach that compares the expected distance (Euclidean norm) between samples of two portions of the
 series with the expected distance between samples within those portions.
 
That is to say, assuming that the two portions can each be modeled as i.i.d. samples drawn from distinct random variables
 (X for the first portion, Y for the second portion), you would expect the following to be non-zero if there is a
 sdifference between the two portions: 
 
 <a href="https://www.codecogs.com/eqnedit.php?latex=\varepsilon&space;(X,&space;Y;&space;\alpha&space;)&space;=&space;2E|X-Y|^\alpha&space;-&space;E|X-X'|^\alpha&space;-&space;E|Y-Y'|^\alpha" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\varepsilon&space;(X,&space;Y;&space;\alpha&space;)&space;=&space;2E|X-Y|^\alpha&space;-&space;E|X-X'|^\alpha&space;-&space;E|Y-Y'|^\alpha" title="\varepsilon (X, Y; \alpha ) = 2E|X-Y|^\alpha - E|X-X'|^\alpha - E|Y-Y'|^\alpha" /></a>

Where alpha is some fixed constant in (0, 2).
This can be calculated empirically with samples from the portions corresponding to X, Y as follows:
 
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{2}{mn}&space;\underset{i=1}{\overset{n}{\sum}}\underset{j=1}{\overset{m}{\sum}}|X_{i}-Y_{j}|^\alpha&space;-\binom{n}{2}^{-1}\underset{1\leq&space;i<k\leq&space;n}{\sum}|X_{i}-X_{k}|^\alpha&space;-&space;\binom{m}{2}^{-1}\underset{1&space;\leq&space;j<k&space;\leq&space;m}{\sum}|Y_{j}-Y_{k}|^\alpha" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{2}{mn}&space;\underset{i=1}{\overset{n}{\sum}}\underset{j=1}{\overset{m}{\sum}}|X_{i}-Y_{j}|^\alpha&space;-\binom{n}{2}^{-1}\underset{1\leq&space;i<k\leq&space;n}{\sum}|X_{i}-X_{k}|^\alpha&space;-&space;\binom{m}{2}^{-1}\underset{1&space;\leq&space;j<k&space;\leq&space;m}{\sum}|Y_{j}-Y_{k}|^\alpha" title="\frac{2}{mn} \underset{i=1}{\overset{n}{\sum}}\underset{j=1}{\overset{m}{\sum}}|X_{i}-Y_{j}|^\alpha -\binom{n}{2}^{-1}\underset{1\leq i<k\leq n}{\sum}|X_{i}-X_{k}|^\alpha - \binom{m}{2}^{-1}\underset{1 \leq j<k \leq m}{\sum}|Y_{j}-Y_{k}|^\alpha" /></a>
 
Thus for a series Z of length L, we find the most likely change point by solving the following for argmax(&tau;) (with a scaling factor of mn/(m+n) and &alpha;=1 for simplicity):

<a href="https://www.codecogs.com/eqnedit.php?latex=Z&space;=&space;\{Z_{1},&space;...,&space;Z_{\tau}&space;,&space;...&space;,&space;Z_{L}\},&space;X&space;=\{Z_{1},...,Z_{\tau}\},&space;Y=\{Z_{\tau&plus;1}\,...,Z_{L}\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Z&space;=&space;\{Z_{1},&space;...,&space;Z_{\tau}&space;,&space;...&space;,&space;Z_{L}\},&space;X&space;=\{Z_{1},...,Z_{\tau}\},&space;Y=\{Z_{\tau&plus;1}\,...,Z_{L}\}" title="Z = \{Z_{1}, ..., Z_{\tau} , ... , Z_{L}\}, X =\{Z_{1},...,Z_{\tau}\}, Y=\{Z_{\tau+1}\,...,Z_{L}\}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{2}{L}(&space;\underset{i=1}{\overset{\tau}{\sum}}\underset{j=\tau&plus;1}{\overset{L}{\sum}}|X_{i}-Y_{j}|&space;-\frac{L-\tau}{\tau-1}\underset{1\leq&space;i<k\leq&space;\tau}{\sum}|X_{i}-X_{k}|&space;-&space;\frac{\tau}{L-\tau-1}\underset{\tau&space;<&space;j<k&space;\leq&space;L}{\sum}|Y_{j}-Y_{k}|)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{2}{L}(&space;\underset{i=1}{\overset{\tau}{\sum}}\underset{j=\tau&plus;1}{\overset{L}{\sum}}|X_{i}-Y_{j}|&space;-\frac{L-\tau}{\tau-1}\underset{1\leq&space;i<k\leq&space;\tau}{\sum}|X_{i}-X_{k}|&space;-&space;\frac{\tau}{L-\tau-1}\underset{\tau&space;<&space;j<k&space;\leq&space;L}{\sum}|Y_{j}-Y_{k}|)" title="\frac{2}{L}( \underset{i=1}{\overset{\tau}{\sum}}\underset{j=\tau+1}{\overset{L}{\sum}}|X_{i}-Y_{j}| -\frac{L-\tau}{\tau-1}\underset{1\leq i<k\leq \tau}{\sum}|X_{i}-X_{k}| - \frac{\tau}{L-\tau-1}\underset{\tau < j<k \leq L}{\sum}|Y_{j}-Y_{k}|)" /></a>

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
from signal_processing_algorithms.e_divisive import EDivisive
from signal_processing_algorithms.e_divisive.calculators import cext_calculator
from signal_processing_algorithms.e_divisive.significance_test import QHatPermutationsSignificanceTester
from some_module import series

// Use C-Extension calculator for calculating divergence metrics
calculator = cext_calculator
// Permutation tester with 1% significance threshold performing 100 permutations for each change point candidate
tester = QHatPermutationsSignificanceTester(
    calculator=calculator, pvalue=0.01, permutations=100
)
algo = EDivisive(calculator=calculator, significance_tester=tester)

change_points = algo.get_change_points(series)
```

## Interactive Documentation

In addition to the package itself and this readme, we have a set of interactive documents that you can use to recreate experiments and investigations of this package, play with them, and make your own!

The requirement for running these documents are:
* Docker
* Docker Compose

Once you have these, simply navigate to [`$REPO/docs`](./docs), execute `docker-compose up` and follow the link!

You can also view these documents in non-interactive form w/o docker+compose:
* [Profiling](./docs/profiling/algorithm_implementations.ipynb)