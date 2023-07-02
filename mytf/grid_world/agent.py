#!/usr/bin/env python
# coding: utf-8

import numpy as np
from space.map import Cell
from .const import CELL, GOAL, Actions


def adjacent_cells(cells, ii, threshold=0.3):
    to_check = (
        (ii[0] + 0, ii[1] - 1),
        (ii[0] - 0, ii[1] + 1),
        (ii[0] + 1, ii[1] - 0),
        (ii[0] - 1, ii[1] + 0),
        (ii[0] + 1, ii[1] + 1),
        (ii[0] - 1, ii[1] - 1),
        (ii[0] + 1, ii[1] - 1),
        (ii[0] - 1, ii[1] + 1),
    )

    for loc in to_check:
        if cells[loc] >= threshold:
            yield loc


def shortest_path_slice(one_hot_map, threshold=0.3, maxdist=None, normalize=True):
    mg = np.argmax(one_hot_map[GOAL])  # the index in the flattened array
    ss = one_hot_map[GOAL].shape
    ii = np.unravel_index(mg, ss)

    if one_hot_map[GOAL][ii] >= threshold:
        ret = np.zeros(ss)
        ret[ii] = 1
        x = 1
        did_something = True
        while did_something and (maxdist is None or x < maxdist):
            did_something = False
            todo = np.argwhere(np.equal(ret, x))
            x += 1
            for ii in todo:
                for loc in adjacent_cells(one_hot_map[CELL], ii):
                    if ret[loc] == 0:
                        ret[loc] = x
                        did_something = True

        if not normalize:
            return ret

        mv = np.max(ret)
        ret = mv - ret
        ret *= one_hot_map[CELL]
        ret /= mv - 1
        return ret


class Agent:
    def __init__(self, gw):
        self.gw = gw
        self.sps = shortest_path_slice(gw.encode())

    def best_move(self, p=None):
        if p is None:
            p = self.gw.s
        score = 0
        winner = None
        for dir, cell in self.gw.R[p].iter_neighbors(dirs=Actions):
            cx, cy = cell.pos
            if self.sps[cy, cx] > score:
                score = self.sps[cy, cx]
                winner = dir
        return winner

    def shortest_path_to_goal(self):
        p = self.gw.s
        ret = list()
        while p != self.gw.g:
            d = self.best_move(p=p)
            if d is not None:
                ret.append(d)
                p = self.gw.R[p].dpos(ret[-1])
            if d is None:
                break
        return ret

    def do_move(self, dir):
        self.gw.do_move(dir)

    def one_hot_shortrange_goal(self, goal_only=False):
        v = self.gw.view
        t = self.gw.tview
        g = np.zeros(t[GOAL].shape, dtype=np.float32)
        m = np.max(self.sps)
        for vpos, apos in [(p, x.pos) for p, x in v if isinstance(x, Cell)]:
            g[vpos] = self.sps[apos]
        if goal_only:
            return g
        t[GOAL] = np.array(g == np.max(g), dtype=np.int32)
        return t
