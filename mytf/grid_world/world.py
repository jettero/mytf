#!/usr/bin/env python
# coding: utf-8

import numpy as np

from space.map import Room, Cell, Wall
from space.living import Human
from space.item import Ubi

from .const import Actions
from .util import ViewActionView
from .util import ScrollHeadTrait, EncoderTrait

class Goal(Ubi):
    a = '*'

class Turtle(Human):
    a = '☺'
    def __init__(self):
        super().__init__('Grid World Turtle', 'Turtle')

class GridWorld(EncoderTrait, ScrollHeadTrait):
    largest_size = [0,0]

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

        if self.R.bounds.XX > self.largest_size[0]:
            self.largest_size[0] = self.R.bounds.XX

        if self.R.bounds.YY > self.largest_size[1]:
            self.largest_size[1] = self.R.bounds.YY

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

    @property
    def view(self):
        return self.R.maxdist_submap( self.T, maxdist=self.maxdist )

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

    def do_move(self, a):
        """
        we return

            ViewActionView(lob, act, rob) / tuple(lob,act,rob)
        """

        lob = rob = self.tview
        act = vectorize_action(a)

        can_move, err_json = self.T.can_move_words(a)
        if can_move:
            self.T.move(a)
            rob = self.tview

        return ViewActionView(lob, act, rob)


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
