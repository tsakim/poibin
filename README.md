# Poisson Binomial Distribution for Python

## About
The module contains a Python implementation of functions related to the Poisson Binomial probability distribution, which describes the probability distribution of the sum of independent Bernoulli random variables with non-uniform success probabilities. For further information, see reference.

The implemented methods are:
* `pmf`: probability mass function
* `cdf`: cumulative distribution function
* `pval`: p-value for right tailed tests

## Dependencies
[NumPy](http://www.numpy.org/)

## Usage
Consider `n` independent and non-identically distributed random variables and be `p` a list/NumPy array of the corresponding Bernoulli success probabilities.
In order to create the Poisson Binomial distributions, use

```
$ from poibin import PoiBin
$ pb = PoiBin(p)
```

Be `x` a list/NumPy array of different numbers of success. Use the following methods to obtain the corresponding quantities:

* Probability mass function
```
$ pb.pmf(x)
```
* Cumulative distribution function
```
$ pb.cdf(x)
```
* P-values for right tailed tests
```
$ pb.pval(x)
```

All three methods accept single integers as well as lists/NumPy arrays of integers. Note that `x[i]` must be smaller than `len(p)`.

## Testing
The methods have been implemented using the `doctest` module. To run the tests, execute

```
$ python -m doctest poibin_tests.txt
```
in the command line. For verbose mode, use

```
$ python -m doctest -v poibin_tests.txt
```

## Reference
Yili Hong, On computing the distribution function for the Poisson binomial distribution,                                                               
Computational Statistics & Data Analysis, Volume 59, March 2013, pages 41-51, ISSN 0167-9473,

http://dx.doi.org/10.1016/j.csda.2012.10.006.  

