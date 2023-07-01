#!/usr/bin/env python
# coding: utf-8

import numpy as np

from collections import namedtuple
from space.map import Cell, Wall, Map
from ..util.misc import terminal_size, write_now, NumpyTuple
from .const import VOID, WALL, CELL, GOAL, TURTLE, MAX_TYPE, Actions

class ViewView(namedtuple('ViewView', ['v1', 'v2']), NumpyTuple):
    pass

class ViewActionView(namedtuple('ViewActionView', ['lob', 'act', 'rob']), NumpyTuple):
    pass

def encode_view(map, turtle, goal, one_hot=True, min_size=None, pad=None):
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

def decode_view(tview, with_goal=True, with_turtle=False, unravel_only=False, threshold=0.3):
    # TODO: this assumes a one-hot encoding ... it should probably have a mode
    # where we skip the unravel

    tview = np.array(tview)

    ss = tview.shape[-2:]
    vg = np.max(tview[GOAL]) # the highest value in the slice
    vt = np.max(tview[TURTLE])
    mg = np.argmax(tview[GOAL]) # the index in the flat reshape of tview[GOAL] with the max value
    mt = np.argmax(tview[TURTLE])

    tview[TURTLE] = np.zeros( ss )
    tview[GOAL] = np.zeros( ss )

    if vg > threshold:
        tview[GOAL][ np.unravel_index( mg, ss ) ] = 2
        # tview[][]=1 might seem to make more sense, but then it can "lose" in
        # the argmax below.

    if vt > threshold:
        tview[TURTLE][ np.unravel_index( mt, ss ) ] = 2

    t = np.argmax(tview, axis=0)

    if unravel_only:
        return t

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

def vectorize_action(a):
    v = np.zeros(len(Actions))
    try:
        v[ Actions.index(a) ] = 1
    except ValueError:
        pass
    return v

class EncoderTrait:
    @property
    def tview(self):
        return self.encode_view()

    def encode_view(self, one_hot=True):
        return encode_view(self.view, self.T, self.G, one_hot=one_hot)

    def encode(self, min_size=None, one_hot=True, pad=None):
        # In [1]: /print e.shape
        # ...: e = np.append(e, np.zeros((e.shape[0], 1)), axis=1)
        # ...: e = np.append(e, np.zeros((e.shape[0], 1)), axis=1)
        # ...: e = np.append(e, np.zeros((1, e.shape[1])), axis=0)
        # ...: e.shape
        # (15, 17)
        # Out[1]: (16, 19)
        return encode_view(self.R, self.T, self.G, one_hot=one_hot, min_size=min_size, pad=pad)


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
