#!/usr/bin/env python
# coding: utf-8

import logging
import functools
from collections import namedtuple
import numpy as np
from .util.misc import terminal_size, write_now
from space.map import Room, Cell, Wall
from space.living import Human
from space.item import Ubi
import space.exceptions as E

log = logging.getLogger(__name__)
A = tuple('nsew') + tuple('NE SE NW SW'.split())

CE = namedtuple('CellEncoding', 'null wall goal turtle'.split())(
    null   = (0,0,0),
    turtle = (0,0,1),
    goal   = (0,1,0),
    wall   = (1,0,0),
)

class Goal(Ubi):
    a = '*'

class Turtle(Human):
    a = '☺'
    def __init__(self):
        super().__init__('Grid World Turtle', 'Turtle')

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

    def reset(self):
        if self.s != self.s0:
            self.R[self.s0] = self.T
        if self.g != self.g0:
            self.R[self.g0] = self.G
        # make sure if we abuse .visible to show pathing information
        # that we clear that back out on reset
        for c in self.R.iter_cells():
            c.visible = False

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


    @functools.lru_cache(maxsize=1024)
    def _cachable_matrix_view(self, s, g):
        mi = self.maxdist * 2 + 1
        sv = self.view
        sb = self.view.bounds
        tx,ty = sv[ self.T ].pos
        dx,dy = tx - sb.x, ty - sb.y
        ax = max(0, self.maxdist - dx)
        ay = max(0, self.maxdist - dy)
        ret = np.zeros( (mi,mi,len(CE.null)), dtype=np.int32 )
        for (x,y), cell in sv:
            x += ax
            y += ay
            if cell is None:
                continue
            if isinstance(cell, Wall):
                ret[y,x] = CE.wall
            elif isinstance(cell, Cell):
                if self.G in cell:
                    ret[y,x] += CE.goal
                if self.T in cell:
                    ret[y,x] += CE.turtle
        return ret

    @property
    def matrix_view(self):
        return self._cachable_matrix_view(self.s, self.g)

    def matrix_view_for(self, *p):
        if len(p) == 1 and isinstance(p[0], (list,tuple)):
            p = p[0]
        o = self.s
        self.s = p
        ret = self.matrix_view
        self.s = o
        return ret

    def tview_for(self, *p):
        return self.matrix_view_for(*p).transpose( (2,0,1) )

    @property
    def tview(self):
        return self.matrix_view.transpose( (2,0,1) )

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

def run_check():
    gw = GridWorld()
    print(repr(gw))
    print(gw)

if __name__ == '__main__':
    run_check();
