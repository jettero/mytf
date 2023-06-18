#!/usr/bin/env python
# coding: utf-8

import logging
import functools
from collections import namedtuple
import numpy as np

from ..util.misc import terminal_size, write_now
from space.map import Room, Cell, Wall
from space.living import Human
from space.item import Ubi

import space.exceptions as E

log = logging.getLogger(__name__)
Actions = tuple('nsew') + tuple('NE SE NW SW'.split())

class Goal(Ubi):
    a = '*'

class Turtle(Human):
    a = '☺'
    def __init__(self):
        super().__init__('Grid World Turtle', 'Turtle')

def encode_map(map, turtle, goal, one_hot=True):
    def _inner():
        for (x,y), cell in map:
            if isinstance(cell, Wall):
                yield 1
            elif isinstance(cell, Cell):
                if turtle in cell:
                    yield 4
                elif goal in cell:
                    yield 3
                else:
                    yield 2
            else:
                yield 0

    b = map.bounds
    r = np.array(list(_inner()), dtype=np.int32).reshape( (b.YY, b.XX) )
    if one_hot:
        r = np.eye(5, dtype=np.int32)[r].transpose(2,0,1)
        r[2] = np.maximum(r[2], np.maximum(r[3], r[4]))
    return r

class GridWorld:
    def __init__(self, room=None, maxdist=3):
        self.maxdist = maxdist

        if room is None:
            room = Room()

        self.R = room
        self.B = room.bounds
        self.T = Turtle()
        self.G = Goal()

        self.R.drop_item_randomly(self.T)
        self.R.drop_item_randomly(self.G)
        while self.s == self.g:
            self.R.drop_item_randomly(self.G)

        self.save_initial()

    def save_initial(self):
        self.s0 = self.s
        self.g0 = self.g

    def shake_the_board(self):
        while True:
            self.R.drop_item_randomly(self.T)
            self.R.drop_item_randomly(self.G)
            self.s0 = self.s
            self.g0 = self.g
            if self.s0 != self.g0:
                break
        self.emit_blindness()

    def emit_blindness(self):
        # This thing isn't a game at all... we use the visibility flag of the
        # cell to show path information instead of visibility information. This
        # resets all the cells to be unseen -- at least when we're drawing the
        # map in ascii.
        for c in self.R.iter_cells():
            c.visible = False

    def reset(self):
        if self.s != self.s0:
            self.R[self.s0] = self.T
        if self.g != self.g0:
            self.R[self.g0] = self.G
        self.emit_blindness()

    def pos(self, what):
        return self.R.find_obj(what).pos

    def put(self, what, p):
        p = tuple(p[0:2])
        w = self.pos(what)
        if self.pos(what) != p:
            self.R[p] = what

    @property
    def s(self):
        return self.pos(self.T)
    @s.setter
    def s(self, p):
        self.put(self.T, p)

    @property
    def g(self):
        return self.pos(self.G)
    @g.setter
    def g(self, p):
        self.put(self.G, p)

    def __str__(self):
        return str(self.R)

    def __repr__(self):
        return f'GridWorld[{self.B.XX} × {self.B.YY}: s={self.s} g={self.g}]'

    def print_scroll_head(self, comment=tuple(), visicalc=True, visible_area_only=False):
        # join comments on space in case they're ('thing=thing', 'otherthing=thingy')
        # then splitlines and lstrip them in case they were lines afterall
        comment = ' '.join(comment)
        comment = tuple( x.lstrip() for x in comment.splitlines() )

        a_map = self.R
        if visicalc or visible_area_only:
            # compute the visicalc so the view is colored
            a_map_ = self.R.visicalc_submap( self.T, maxdist=self.maxdist )
            if visible_area_only:
                # even if we don't actually use the submap
                a_map = a_map_

        # +3: column indicator line, blank line, plus one more cuz scroll
        # region starts below the blank line
        non_scroll_height = self.B.YY + len(comment) + 3
        ts = terminal_size()

        # erase lines to scroll_height
        for lno in range(1, non_scroll_height):
            print(f'\x1b[{lno}H\x1b[1G\x1b[K', end='')

        # goto line H, row G
        print(f'\x1b[1H\x1b[1G', end='')
        print(a_map)
        for c in comment:
            print(c)

        # set scrolling region
        write_now(f'\x1b[{non_scroll_height};{ts.lines}r')

        # go to the bottom of the screen
        write_now(f'\x1b[{ts.lines}H')

    @property
    def view(self):
        return self.R.visicalc_submap( self.T, maxdist=self.maxdist )

    @property
    def tview(self):
        return self.encode_view()

    @property
    def tmap(self):
        return self.encode()

    def encode_view(self):
        return encode_map(self.view, self.T, self.G)

    def encode(self):
        return encode_map(self.R, self.T, self.G)

    def dist2goal(self, pos=None, goal_pos=None):
        return self.distnorm2goal(pos=pos, goal_pos=goal_pos)[-1]

    def dir2goal(self, pos=None, goal_pos=None):
        return self.distnorm2goal(pos=pos, goal_pos=goal_pos)[0]

    def distnorm2goal(self, pos=None, goal_pos=None):
        if pos is None:
            pos = self.state
        if goal_pos is None:
            goal_pos = self.goal
        try:
            d = np.array(goal_pos) - np.array(pos)
            n = np.linalg.norm(d)
        except TypeError:
            return (0,0), 0.000001
        if n != 0:
            return d/n, n
        return d, 0.0

def EasyRoom(x=5,y=5, s=None, g=None):
    room = Room(x, y)
    gw = GridWorld(room=room)
    if s is None:
        s = (1,2)
    if g is None:
        g = (x, y-1)
    gw.s0 = s
    gw.g0 = g
    gw.reset()
    return gw

def HardRoom():
    r = Room(7,3)
    r[5,4] = Room(2,5)
    r.cellify_partitions()
    gw = GridWorld(r)
    gw.s0 = gw.s = 2,3
    gw.g0 = gw.g = 7,9
    gw.reset()
    return gw

def SuperHardRoom():
    room = Room(5,5)
    room[10,8] = Room(5,5)
    for i in range(6,13):
        room[i,4] = Wall()
        room[i,5] = Cell()
        room[i,6] = Wall()
    for i in range(6,9):
        room[11,i] = Wall()
        room[12,i] = Cell()
        room[13,i] = Wall()
    room[13,4] = Wall()
    room[13,5] = Wall()
    gw = GridWorld(room=room)
    gw.s0 = (2,2)
    gw.g0 = (11,12)
    gw.reset()
    return gw

def run_check():
    gw = SuperHardRoom()
    print(repr(gw))
    print(gw)

def vectorize_action(a):
    v = np.zeros(len(Actions))
    try:
        v[ Actions.index(a) ] = 1
    except ValueError:
        pass
    return v


if __name__ == '__main__':
    run_check();
