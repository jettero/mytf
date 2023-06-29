#!/usr/bin/env python
# coding: utf-8

import pytest
import numpy as np
from mytf import features_last, features_first
import tensorflow as tf

IMAGE_SIZE = (7,7)
COLOR_CHANNELS = 3
N = np.prod((*IMAGE_SIZE, COLOR_CHANNELS))

@pytest.fixture(scope='function', params=(None,1,3))
def np_cxy(request):
    if request.param is None:
        return np.reshape(list(range(N)), (COLOR_CHANNELS, *IMAGE_SIZE))
    return np.reshape(list(range(request.param*N)), (request.param, COLOR_CHANNELS, *IMAGE_SIZE))

@pytest.fixture(scope='function')
def np_xyc():
    return np.reshape(list(range(N)), (*IMAGE_SIZE, COLOR_CHANNELS))

@pytest.fixture(scope='function')
def tf_cxy(np_cxy):
    return tf.convert_to_tensor(np_cxy)

@pytest.fixture(scope='function')
def tf_xyc(np_xyc):
    return tf.convert_to_tensor(np_xyc)

@pytest.fixture(scope='function', params=['np', 'tf'])
def cxy(request, np_cxy, tf_cxy):
    if request.param == 'np':
        return np_cxy
    return tf_cxy

@pytest.fixture(scope='function', params=['np', 'tf'])
def xyc(request, np_xyc, tf_xyc):
    if request.param == 'np':
        return np_xyc
    return tf_xyc

def test_cxy2xyc(cxy,xyc):
    swapped = features_last(cxy)
    if len(cxy.shape) == 4:
        for sw,b in zip(swapped,cxy):
            assert sw.shape == xyc.shape
            assert sw.shape == (*IMAGE_SIZE, COLOR_CHANNELS)
            for x in range(IMAGE_SIZE[0]):
                for y in range(IMAGE_SIZE[1]):
                    for c in range(COLOR_CHANNELS):
                        assert sw[x][y][c] == b[c][x][y]
    else:
        assert swapped.shape == xyc.shape
        assert swapped.shape == (*IMAGE_SIZE, COLOR_CHANNELS)
        for x in range(IMAGE_SIZE[0]):
            for y in range(IMAGE_SIZE[1]):
                for c in range(COLOR_CHANNELS):
                    assert swapped[x][y][c] == cxy[c][x][y]

def test_last_then_first(cxy):
    swapped = features_last(cxy)
    unswapped = features_first(swapped)
    assert swapped.shape[-3:] == (*IMAGE_SIZE, COLOR_CHANNELS)
    assert unswapped.shape[-3:] == (COLOR_CHANNELS, *IMAGE_SIZE)
    assert np.all(unswapped == cxy)

def test_first_then_last(xyc):
    swapped = features_first(xyc)
    unswapped = features_last(swapped)
    assert swapped.shape[-3:] == (COLOR_CHANNELS, *IMAGE_SIZE)
    assert unswapped.shape[-3:] == (*IMAGE_SIZE, COLOR_CHANNELS)
    assert np.all(unswapped == xyc)
