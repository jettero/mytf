#!/usr/bin/env python
# coding: utf-8

import re
import tensorflow as tf

@tf.function
def bytes_to_bits(x):
    return strings_to_bits(encoding=None)

@tf.function
def strings_to_bits(x, encoding='UTF-8', depth=8, dtype=tf.int32):
    """
    convert a single string to bits

        In [1]: string_to_tf_bits('supz')
        Out[1]:
        <tf.Tensor: shape=(4, 8), dtype=int32, numpy=
        array([[1, 1, 0, 0, 1, 1, 1, 0],
               [1, 0, 1, 0, 1, 1, 1, 0],
               [0, 0, 0, 0, 1, 1, 1, 0],
               [0, 1, 0, 1, 1, 1, 1, 0]], dtype=int32)>

    or convert a bunch of strings to a ragged tensor

        In [4]: string_to_tf_bits(['supz', 'a test'])                                                                                                             [16/431]
        Out[4]:
        <tf.RaggedTensor [[[1, 1, 0, 0, 1, 1, 1, 0],
          [1, 0, 1, 0, 1, 1, 1, 0],
          [0, 0, 0, 0, 1, 1, 1, 0],
          [0, 1, 0, 1, 1, 1, 1, 0]], [[1, 0, 0, 0, 0, 1, 1, 0],
                                      [0, 0, 0, 0, 0, 1, 0, 0],
                                      [0, 0, 1, 0, 1, 1, 1, 0],
                                      [1, 0, 1, 0, 0, 1, 1, 0],
                                      [1, 1, 0, 0, 1, 1, 1, 0],
                                      [0, 0, 1, 0, 1, 1, 1, 0]]]>
    """
    if encoding is not None:
        x = tf.strings.unicode_decode(x, encoding)
    m = tf.bitwise.left_shift(tf.ones([], dtype=tf.int32), tf.range(depth, dtype=tf.int32))
    x = tf.bitwise.bitwise_and(tf.expand_dims(x, -1), m)
    x = tf.cast(tf.not_equal(x, 0), dtype)
    return x

@tf.function
def fuzzy_bits_to_bytes(x, encoding='UTF-8', depth=8):
    """
    recode fuzzy output bits into strings of bytes (not python strings)

        In [1]: fuzzy_bits_to_bytes(strings_to_bits('supz'))
        Out[1]: b'supz'
    """
    x = tf.round(x)
    x = tf.cast(x, tf.int32)
    x = tf.maximum(x, 0)
    x = tf.minimum(x, 1)
    x *= tf.bitwise.left_shift(tf.ones([], dtype=tf.int32), tf.range(depth, dtype=tf.int32))
    x = tf.reduce_sum(x, axis=-1)
    x = tf.strings.unicode_encode(x, encoding)
    return x

def stringify_bytes(x, encoding='UTF-8', depth=8):
    """
    Go all the way back to actual strings (not bytes). This is not decorated as
    a TF function. It's probably only useful for printing to humans.
    """
    if x.dtype != tf.string:
        x = fuzzy_bits_to_bytes(x)
    x = x.numpy()
    return x.decode(encoding) if isinstance(x, (bytes,)) else [ y.decode() for y in x ]

def sans_color(x):
    return re.sub(r'\x1b\[[\d;]*m', '', str(x))

def colorless_len(x):
    return len(sans_color(x))

def _side_by_side(x, y):
    x = x.splitlines()
    y = y.splitlines()
    m = max(colorless_len(i) for i in x) + 2

    while len(x) < len(y):
        x.append('')

    while len(y) < len(x):
        y.append('')

    for xl, yl in zip(x,y):
        yield xl + (" " * (m - colorless_len(xl))) + yl

def side_by_side(x, y):
    return "\n".join( _side_by_side(x,y) ) + "\n"
