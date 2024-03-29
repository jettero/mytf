#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pytest
from mytf.util import NumpyTuple
from collections import namedtuple

TbT = (2, 3)
t1TbT = (1, *TbT)
t2TbT = (2, *TbT)
t4TbT = (4, *TbT)


def chi_square_random_variable_matrix(s=TbT):
    return np.random.chisquare(3.141592653, s)


csrm = chi_square_random_variable_matrix


@pytest.fixture(scope="function")
def a():
    return NumpyTuple(*(csrm() for _ in range(3)))


@pytest.fixture(scope="function")
def b():
    return NumpyTuple(*(csrm() for _ in range(3)))


def test_initial_shape(a, b):
    assert a.shape == (TbT,) * 3
    assert b.shape == (TbT,) * 3


def test_promote(a, b):
    assert a.promote().shape == (t1TbT,) * 3
    assert b.promote().shape == (t1TbT,) * 3


def test_too_promote(a, b):
    assert a.promote().shape == (t1TbT,) * 3
    assert b.promote().shape == (t1TbT,) * 3


def test_cat(a, b):
    assert a.cat(b).shape == (t2TbT,) * 3


def test_more_cat(a, b):
    assert a.cat(b).cat(b.cat(a)).shape == (t4TbT,) * 3


############################# subclassing
TSC = namedtuple("TSC", ["a", "b"])


@pytest.fixture(scope="module", params=("first", "last"))
def ntcls(request):
    if request.param == "first":

        class ntcls(NumpyTuple, TSC):
            pass

        return ntcls

    class ntcls(TSC, NumpyTuple):
        pass

    return ntcls


def test_nttsc1(ntcls, a, b):
    ab = ntcls(a, b)
    assert np.all(ab.a == a)
    assert np.all(ab.b == b)
