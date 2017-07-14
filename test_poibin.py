# -*- coding: utf-8 -*-
"""
Test file for the module poibin

Created on Tue May 08, 2017

Author:
    Mika Straka

Description:
    This file contains the test cases for the functions and methods
    defined in ``poibin.py``. The tests can be run with ``pytest``.

Usage:
    To run the tests, execute
        $ pytest test_poibin.py
    in the command line. If you want to run the tests in verbose mode, use
        $ pytest -v test_poibin.py

References:
.. [Hong2013] Yili Hong, On computing the distribution function for the Poisson
    binomial distribution,
    Computational Statistics & Data Analysis, Volume 59, March 2013,
    Pages 41-51, ISSN 0167-9473,
    http://dx.doi.org/10.1016/j.csda.2012.10.006.

.. [Rpoibin] Yili Hong, The Poisson Binomial Distribution,
    R package,
    https://cran.r-project.org/package=poibin
"""

################################################################################
# Tests
################################################################################

import numpy as np
import pytest
from poibin import PoiBin
from scipy.stats import binom


# PoiBin.pmf -------------------------------------------------------------------

def test_pmf():
    """Test the probability mass function.

    The outcomes of some results are compared with the poibin R package
    [Rpoibin]_.
    """
    p = [1, 1]
    pb = PoiBin(p)
    assert pb.pmf([1, 2]).size == 2

    # Compare results with the ones obtained with the R poibin package
    # [Rpoibin]_
    p = [0.4163448, 0.3340270, 0.9689613]
    pb = PoiBin(p)
    res = pb.pmf([0, 1, 2, 3])
    res_ref = np.array([0.0120647, 0.39129134, 0.46189012, 0.13475384])
    assert np.all(np.abs(res - res_ref) < 1e-8)

    p = [0.9955901, 0.5696224, 0.8272597, 0.3818746, 0.4290036, 0.8707646,
         0.8858267, 0.7557183]
    pb = PoiBin(p)
    res = pb.pmf([0, 2, 7, 8])
    res_ref = np.array([4.17079659e-07, 2.46250608e-03, 2.02460933e-01,
                        4.48023378e-02])
    assert np.all(np.abs(res - res_ref) < 1e-8)

def test_pmf_pb_binom():
    """Compare the probability mass function with the binomial limit case."""
    # For equal probabilites p_j, the Poisson Binomial distribution reduces to
    # the Binomial one:
    p = [0.5, 0.5]
    pb = PoiBin(p)
    bn = binom(n=2, p=p[0])

    # Compare to four digits behind the comma
    assert int(bn.pmf(0) * 10000) == int(pb.pmf(0) * 10000)

    # For different probabilities p_j, the Poisson Binomial distribution and
    # the Binomial distribution are different:
    pb = PoiBin([0.5, 0.8])
    bn = binom(2, p=0.5)
    assert int(bn.pmf(0) * 10000) != int(pb.pmf(0) * 10000)

def test_pmf_accuracy():
    """Compare accuracy of the probability mass function.

    Compare the results with the accuracy check proposed in [Hong2013]_,
    equation (15).
    """
    [p1, p2, p3] = np.around(np.random.random_sample(size=3), decimals=2)
    [n1, n2, n3] = np.random.random_integers(1, 10, size=3)
    nn = n1 + n2 + n3
    l1 = [p1 for i in range(n1)]
    l2 = [p2 for i in range(n2)]
    l3 = [p3 for i in range(n3)]
    p = l1 + l2 + l3
    b1 = binom(n=n1, p=p1)
    b2 = binom(n=n2, p=p2)
    b3 = binom(n=n3, p=p3)
    k = np.random.randint(0, nn + 1)
    chi_bn = 0
    for j in range(0, k+1):
        for i in range(0, j+1):
            chi_bn += b1.pmf(i) * b2.pmf(j - i) * b3.pmf(k - j)
    pb = PoiBin(p)
    chi_pb = pb.pmf(k)
    assert np.all(np.around(chi_bn, decimals=10) == np.around(chi_pb,
                                                              decimals=10))

# PoiBin.cdf ------------------------------------------------------------------

def test_cdf():
    """Test the cumulative distribution function."""
    p = [1, 1]
    pb = PoiBin(p)
    assert np.all(pb.cdf([1, 2]) - np.array([0., 1.]) < 4 * np.finfo(float).eps)
    assert (pb.cdf(2) - 1.) < 4 * np.finfo(float).eps

def test_cdf_pb_binom():
    """Compare the cumulative distribution function with the binomial limit
    case.
    """
    # For equal probabilites p_j, the Poisson Binomial distribution reduces
    # to the Binomial one:
    p = [0.5, 0.5]
    pb = PoiBin(p)
    bn = binom(n=2, p=p[0])
    # Compare to four digits behind the comma
    assert int(bn.cdf(0) * 10000) == int(pb.cdf(0) * 10000)

    # For different probabilities p_j, the Poisson Binomial distribution and
    # the Binomial distribution are different:
    pb = PoiBin([0.5, 0.8])
    bn = binom(2, p=0.5)
    assert int(bn.cdf(0) * 10000) != int(pb.cdf(0) * 10000)

def test_cdf_accuracy():
    """Compare accuracy of the cumulative distribution function.

    Compare the results with the ones obtained with the R poibin package
    [Rpoibin]_.
    """
    p = [0.1, 0.1]
    pb = PoiBin(p)
    assert np.all(np.abs(pb.cdf([0, 2]) - np.array([0.81, 1.])) < 1e-10)
    p = [0.5, 1.0]
    pb = PoiBin(p)
    assert np.all(np.abs(pb.cdf([1, 2]) == np.array([0.5, 1.])) < 1e-10)
    p = [0.1, 0.5]
    pb = PoiBin(p)
    assert np.all(np.abs(pb.cdf([0, 1, 2]) == np.array([0.45, 0.95, 1.])) <
                  1e-10)
    p = [0.1, 0.5, 0.7]
    pb = PoiBin(p)
    assert np.all(np.abs(pb.cdf([0, 1, 2]) == np.array([0.135, 0.6, 0.965])) <
                  1e-10)

# PoiBin.pval ------------------------------------------------------------------

def test_pval():
    """Test the p-values function."""
    p = [1, 1]
    pb = PoiBin(p)

    assert np.all(pb.pval([1, 2]) - np.array([1., 1.]) <
           4 * np.finfo(float).eps)
    assert (pb.pval(2) - 1.) < 4 * np.finfo(float).eps

def test_pval_pb_binom():
    """Compare the p-values with the binomial limit case.

    Test that the p-values of the Poisson Binomial distribution are the same
    as the ones of the Binomial distribution when all the probabilities are
    equal.
    """
    pi = np.around(np.random.random_sample(), decimals=2)
    ni = np.random.randint(5, 500)
    pp = [pi for i in range(ni)]
    bn = binom(n=ni, p=pi)
    k = np.random.randint(0, ni)
    pval_bn = 1 - bn.cdf(k) + bn.pmf(k)
    pb = PoiBin(pp)
    pval_pb = pb.pval(k)
    assert np.all(np.around(pval_bn, decimals=10) == np.around(pval_pb,
                                                               decimals=10))

# PoiBin.get_cdf ---------------------------------------------------------------

def test_get_cdf():
    """Test that the right cumulative distribution function is obtained."""
    p = [1, 1]
    pb = PoiBin(p)
    assert np.all(pb.get_cdf([1, 1, 1]) == np.array([1., 2., 3.]))

# PoiBin.get_pmf_xi ------------------------------------------------------------

def test_get_pmf_xi():
    """Test that the correct pmf elements are obtained."""
    p = [0.2, 0.5]
    pb = PoiBin(p)
    assert np.all(np.abs(pb.get_pmf_xi() - np.array([0.4, 0.5, 0.1])) <
                  1e-10)
    p = [0.3, 0.8]
    pb = PoiBin(p)
    assert np.all(np.abs(pb.get_pmf_xi() - np.array([0.14, 0.62, 0.24])) <
                  1e-10)
    p = [0.3, 0.8, 0.3]
    pb = PoiBin(p)
    assert np.all(np.abs(pb.get_pmf_xi() - np.array([0.098, 0.476, 0.354,
                                                     0.072])) < 1e-10)
# PoiBin.check_rv_input --------------------------------------------------------

def test_check_rv_input():
    """Test tat inputs are positive integers."""
    p = [1, 1]
    pb = PoiBin(p)
    assert pb.check_rv_input([1, 2])
    assert pb.check_rv_input(2)

    with pytest.raises(AssertionError,
                       message="Input value cannot be negative."):
        pb.check_rv_input(-1)
    with pytest.raises(AssertionError,
                       message="Input value must be an integer."):
        pb.check_rv_input(1.7)

# PoiBin.check_xi_are_real -----------------------------------------------------

def test_check_xi_are_real():
    """Test the check that the ``xi`` values are real."""
    pb = PoiBin([0])
    xi = np.array([1 + 0j, 1.8 + 0j], dtype=complex)
    assert pb.check_xi_are_real(xi)
    xi = np.array([1 + 99j, 1.8 + 0j], dtype=complex)
    assert not pb.check_xi_are_real(xi)

# PoiBin.check_input_prob ------------------------------------------------------

def test_check_input_prob():
    """Test the check that input probabilities are between 0 and 1."""
    with pytest.raises(ValueError,
                       message="Input must be an one-dimensional array or a"\
                                + "list."):
        pb = PoiBin([[1, 1], [1, 2]])
    with pytest.raises(ValueError,
                       message="Input probabilities have to be non negative."):
        pb = PoiBin([1, -1])
    with pytest.raises(ValueError,
                       message="Input probabilities have to be smaller"\
                               + "than 1."):
        pb = PoiBin([1, 2])

