#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pytest
from mytf.util import NumpyTuple

# NOTE: there's a lotta intended jokes below... none are actually funny

TbT = (2,3)
oTbT = (1,2,3)
tTbT = (2,2,3)

def chi_square_random_variable_matrix(s=TbT):
    return np.random.chisquare(3.141592653, s)

csrm = chi_square_random_variable_matrix

@pytest.fixture(scope='function')
def a():
    return NumpyTuple( (csrm() for _ in range(3)) )

@pytest.fixture(scope='function')
def b():
    return NumpyTuple( (csrm() for _ in range(3)) )

@pytest.fixture(scope='function')
def c():
    return NumpyTuple( (csrm() for _ in range(3)) ).promote()

@pytest.fixture(scope='function')
def d():
    return NumpyTuple( (csrm() for _ in range(3)) ).promote().promote()

def test_initial_shape(a,b,c,d):
    assert a.shape == ( TbT, ) * 3
    assert b.shape == ( TbT, ) * 3
    assert c.shape == (1,*(( TbT, ) * 3))
    assert d.shape == (1,*(( TbT, ) * 3))
