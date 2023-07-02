# coding: utf-8

import sys, shutil, argparse
import numpy as np
import tensorflow as tf


def bnd(lb, val, ub, swap=True):
    """bnd(lb,val,ub)
    if val is none, bnd() returns a lambda x: bnd(lb,x,ub)
    if lb is greater than ub and swap=True (default),
        lb and ub are swapped
    """
    if swap and lb > ub:
        lb, ub = ub, lb
    if val is None:
        return lambda x: min(ub, max(lb, x))
    return min(ub, max(lb, val))


class Bound:
    def __init__(self, lb, ub, swap=True):
        self._cb = bnd(lb, None, ub, swap=swap)

    def __call__(self, x):
        return self._cb(x)


def dmax(o, γ=0.97, β=1.0):
    r = set([o.argmax()])
    m = o.max()
    if 0.0 <= β < 1.0:
        l = γ * m
        d = m - l
        l += d * β
    else:
        l = m
    for i, v in enumerate(o):
        if v >= l:
            r.add(i)
    return tuple(r)


def fnameify(fname, ext=".h5"):
    if not ext.startswith("."):
        ext = "." + ext
    if "/" not in fname:
        fname = f"save/{fname}"
    if not fname.endswith(ext):
        fname += ext
    return fname


def yes_no_threshold(e=0.5):
    if e <= 0:
        return False
    if e >= 1:
        return True
    return bool(e >= np.random.uniform(0, 1))


def iterate_anything(*x):
    for i in x:
        if isinstance(i, (list, tuple)):
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
        r = a * i + (1 - a) * r
    return r


def features_last(x):
    """ move features from -3 to the end """
    s = x.shape
    p = list(range(len(s)))
    p.append(p.pop(-3))

    if isinstance(x, np.ndarray):
        return x.transpose(p)
    return tf.transpose(x, perm=p)


def features_first(x):
    """ move features from -1 to -3 """
    s = x.shape
    p = list(range(len(s)))
    p.insert(-2, p.pop(-1))

    if isinstance(x, np.ndarray):
        return x.transpose(p)
    return tf.transpose(x, perm=p)


def format_output_vector(
    vec,
    action=None,
    γ=None,
    β=None,
    f="5.2f",
    A=tuple("nsew") + tuple("NE SE NW SW".split()),
):
    dm = dmax(vec, γ=γ, β=β)

    def generate_items():
        for an, x in enumerate(vec):
            aw = A[an]
            xf = f"{x:{f}}"
            fs = f"{aw}={xf}"
            c = (93, 99) if an in dm else (234, 238)
            c = c[1 if an == action else 0]
            yield f"\x1b[38;5;{c}m{fs}\x1b[m"

    return " ".join(generate_items())


def write_now(*txt):
    for t in txt:
        sys.stdout.write(t)
    sys.stdout.flush()


def terminal_size(fallback=(80, 24)):
    return shutil.get_terminal_size(fallback=fallback)
