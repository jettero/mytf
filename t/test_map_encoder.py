#!/usr/bin/env python
# coding: utf-8

import pytest
from mytf.grid_world import Cell, GridWorld, SuperHardRoom

@pytest.fixture(params=[SuperHardRoom, GridWorld])
def gw(request):
    yield request.param()

def test_tview(gw):
    for (x,y), cell in gw.R:
        if isinstance(cell, Cell):
            gw.s = x,y
            tview = gw.tview
            assert tview.shape == (5,7,7)
