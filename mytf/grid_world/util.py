#!/usr/bin/env python
# coding: utf-8

import numpy as np

from collections import namedtuple
from ..util.misc import terminal_size, write_now
from space.map import Cell, Wall, Map
from .const import VOID, WALL, CELL, GOAL, TURTLE, MAX_TYPE

class ViewActionView(namedtuple('ViewActionView', ['lob', 'act', 'rob'])):
    @property
    def shape(self):
        return tuple( x.shape for x in self )

    def cat(self, *x):
        items = [ y.promote() for y in (self, *x) ]
        return ViewActionView(
            np.concatenate([x.lob for x in items]),
            np.concatenate([x.act for x in items]),
            np.concatenate([x.rob for x in items]),
        )

    @property
    def depth(self):
        return len(self.act.shape)

    def promote(self, depth=2):
        ret = self
        while ret.depth < depth:
            ret = ViewActionView(
            ret.lob.reshape((1,*ret.lob.shape)),
            ret.act.reshape((1,*ret.act.shape)),
            ret.rob.reshape((1,*ret.rob.shape)),
        )
        return ret

    def slice(self, x):
        return ViewActionView(*(y[x] for y in self))

def encode_map(map, turtle, goal, one_hot=True, min_size=None, pad=None):
    def _inner():
        for (x,y), cell in map:
            if isinstance(cell, Wall):
                yield WALL
            elif isinstance(cell, Cell):
                if turtle in cell:
                    yield TURTLE
                elif goal in cell:
                    yield GOAL
                else:
                    yield CELL
            else:
                yield VOID

    b = map.bounds
    r = np.array(list(_inner()), dtype=np.int32).reshape( (b.YY, b.XX) )
    if isinstance(min_size, (list,tuple)) and len(min_size) == 2:
        while r.shape[-2] < min_size[0]:
            r = np.append(r, [0], 0, axis=0)

    if isinstance(pad, (list,tuple)) and len(pad) == 2:
        pad_x,pad_y = pad

        while pad_x > r.shape[1]:
            r = np.append(r, np.zeros((r.shape[0], 1), dtype=np.int32), axis=1)
        while pad_y > r.shape[0]:
            r = np.append(r, np.zeros((1, r.shape[1]), dtype=np.int32), axis=0)

    if one_hot:
        # 1. make a one hot matrix by taking the indices of an eye.
        # 2. not really sure why this gives the (x,y,MAX_TYPE) shape, but it does
        # 3. then transpose/swap axis 2 and 1 so we get our more human readable
        #    (MAX_TYPE,x,y) shape
        r = np.eye(MAX_TYPE, dtype=np.int32)[r].transpose(2,0,1)

        # 4. make sure the TURTLE and GOAL spots are also CELL spots through
        #    "clever" use of min/max
        r[CELL] = np.maximum(r[CELL], np.maximum(r[GOAL], r[TURTLE]))
    return r

def decode_map(tview, with_goal=True, with_turtle=False):

    tview = np.array(tview)

    ss = tview.shape[-2:]
    mt = np.unravel_index( np.argmax(tview[TURTLE]), ss )
    mg = np.unravel_index( np.argmax(tview[GOAL]), ss )

    tview[TURTLE] = np.zeros( ss )
    tview[GOAL] = np.zeros( ss )

    tview[TURTLE][ mt ] = 1
    tview[GOAL][ mg ] = 1

    t = np.argmax(tview, axis=0)
    ret = Map(*ss)

    # NOTE: the matrix looks right in ipython, but either the one-hot transform
    # is rotating on a diagonal, or np[x,y] is rotated from how I'd expect the
    # printed appearance to look... whatever. Map[x,y] vs np.array[y,x]

    from .world import Turtle, Goal # late import to avoid circular imports

    for (x,y), _ in ret:
        if t[y,x] == WALL:
            ret[x,y] = Wall() 

        elif t[y,x] == CELL:
            ret[x,y] = Cell() 

        elif t[y,x] == TURTLE:
            ret[x,y] = Cell() 
            if with_turtle:
                ret[x,y] = Turtle()

        elif t[y,x] == GOAL:
            ret[x,y] = Cell() 
            if with_goal:
                ret[x,y] = Goal()

    return ret

class EncoderTrait:
    @property
    def tview(self):
        return self.encode_view()

    def encode_view(self, one_hot=True):
        return encode_map(self.view, self.T, self.G, one_hot=one_hot)

    def encode(self, min_size=None, one_hot=True, pad=None):
        # In [1]: /print e.shape
        # ...: e = np.append(e, np.zeros((e.shape[0], 1)), axis=1)
        # ...: e = np.append(e, np.zeros((e.shape[0], 1)), axis=1)
        # ...: e = np.append(e, np.zeros((1, e.shape[1])), axis=0)
        # ...: e.shape
        # (15, 17)
        # Out[1]: (16, 19)
        return encode_map(self.R, self.T, self.G, one_hot=one_hot, min_size=min_size, pad=pad)

class ScrollHeadTrait:
    def print_scroll_head(self, comment=tuple(), visicalc=True, visible_area_only=False):
        # join comments on space in case they're ('thing=thing', 'otherthing=thingy')
        # then splitlines and lstrip them in case they were lines afterall
        comment = ' '.join(comment)
        comment = tuple( x.lstrip() for x in comment.splitlines() )

        a_map = self.R
        if visicalc or visible_area_only:
            # compute the visicalc so the view is colored
            a_map_ = self.R.maxdist_submap( self.T, maxdist=self.maxdist )
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
