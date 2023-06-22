#!/usr/bin/env python
# coding: utf-8

import pytest
from mytf.grid_world import Cell, GridWorld, SuperHardRoom, Map
from space.map import Room, Cell, Wall

@pytest.fixture(params=[SuperHardRoom, GridWorld])
def gw(request):
    yield request.param()

def test_tview(gw):
    for (x,y), cell in gw.R:
        if isinstance(cell, Cell):
            gw.s = x,y
            tview = gw.tview
            assert tview.shape == (5,7,7)

def test_encode_padding(gw):
    assert gw.encode().shape[-1] < 18
    assert gw.encode().shape[-2] < 18
    assert gw.encode(pad=(18,18)).shape == (5,18,18)

def test_decode_tview_map(gw):
    tview = gw.tview
    m = Map()
