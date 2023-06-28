#!/usr/bin/env python
# coding: utf-8

from .const import TURTLE, GOAL

def goal_slice_to_shorest_path(one_hot_map):
    mg = np.argmax(tview[GOAL])
    ss = tview.shape[-2:]

    if mg > 0.3:
        one_hot_map[GOAL] = np.zeros( ss )

