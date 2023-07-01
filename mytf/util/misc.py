# coding: utf-8

import sys, shutil, argparse
import numpy as np

class NumpyTuple(tuple):
    D = None

    @property
    def shape(self):
        return tuple( x.shape for x in self )

    def cat(self, *x, depth=None):
        items = [ y.promote(depth=depth) for y in (self, *x) ]
        return self.make(*(
            np.concatenate([x[i] for x in items]) for i in range(len(self))
        ))

    @property
    def depth(self):
        return len(self[0].shape)

    def promote(self, depth=None):
        """
        Promote is intended to add a batch dimension to all the numpy arrays in
        the tuple ...  It's intended to automatically sense when this has
        already been done, but it's not always going to get it right.  supply
        the depth kwarg to promote() and/or cat() when it seems to be doing the
        wrong thing. The depth is the len() of the shape of the first numpy
        array in the tuple.
        """
        if depth is None:
            if self.D is None:
                self.D = self.depth
            depth = self.D
        ret = self
        while ret.depth < depth:
            ret = self.make(*(x.reshape((1,*x.shape)) for x in ret))
        return ret

    def slice(self, x):
        return self.make(*(y[x] for y in self))

    @classmethod
    def make(cls, *a):
        # NOTE: tried to determine the right args with isinstance/issubclass,
        # but telling the difference between a subclass of tuple and a subclass
        # of namedtuple is surprisingly difficult since they're both tuples and
        # have different argument signatures. I think in principle you could
        # make it work with `import inspect`; and then analyzing the ArgSpecs
        # ... but ... it's not really worth it.
        try:
            return cls(*a)
        except TypeError:
            pass
        return cls(a)

def bnd(lb, val, ub, swap=True):
    ''' bnd(lb,val,ub)
        if val is none, bnd() returns a lambda x: bnd(lb,x,ub)
        if lb is greater than ub and swap=True (default),
            lb and ub are swapped
    '''
    if swap and lb > ub:
        lb,ub = ub,lb
    if val is None:
        return lambda x: min(ub, max(lb, x))
    return min(ub, max(lb, val))

class Bound:
    def __init__(self, lb, ub, swap=True):
        self._cb = bnd(lb, None, ub, swap=swap)

    def __call__(self, x):
        return self._cb(x)

def dmax(o, γ=0.97, β=1.0):
    r = set([ o.argmax() ])
    m = o.max()
    if 0.0 <= β < 1.0:
        l = γ * m
        d = (m-l)
        l += d * β
    else:
        l = m
    for i,v in enumerate(o):
        if v >= l:
            r.add(i)
    return tuple(r)

def fnameify(fname, ext='.h5'):
    if not ext.startswith('.'):
        ext = '.' + ext
    if '/' not in fname:
        fname = f'save/{fname}'
    if not fname.endswith(ext):
        fname += ext
    return fname

def yes_no_threshold(e=0.5):
    if e <= 0: return False
    if e >= 1: return True
    return bool( e >= np.random.uniform(0,1) )

def iterate_anything(*x):
    for i in x:
        if isinstance(i, (list,tuple)):
            yield from iterate_anything(*i)
        else:
            yield i

def generate_forever(*a):
    while True:
        yield from a

def ema(*x, a=0.5):
    if not x:
        return None
    g = iterate_anything(x)
    try:
        r = next(g)
    except StopIteration:
        return None
    for i in x[1:]:
        if r is None:
            r = i
            continue
        if i is None:
            continue
        r = a*i + (1-a)*r
    return r

def format_output_vector(vec, action=None, γ=None, β=None,
        f='5.2f', A=tuple('nsew') + tuple('NE SE NW SW'.split())):
    dm = dmax(vec, γ=γ, β=β)
    def generate_items():
        for an,x in enumerate(vec):
            aw = A[an]
            xf = f'{x:{f}}'
            fs = f'{aw}={xf}'
            c = (93,99) if an in dm else (234,238)
            c = c[ 1 if an == action else 0 ]
            yield f'\x1b[38;5;{c}m{fs}\x1b[m'
    return ' '.join(generate_items())

def write_now(*txt):
    for t in txt:
        sys.stdout.write(t)
    sys.stdout.flush()

def terminal_size(fallback=(80,24)):
    return shutil.get_terminal_size(fallback=fallback)
