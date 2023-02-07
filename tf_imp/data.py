#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf

@tf.function
def string_to_tf_bits(x, encoding='UTF-8'):
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
    d = tf.strings.unicode_decode(x, encoding)
    m = tf.bitwise.left_shift(tf.ones([], dtype=d.dtype), tf.range(8, dtype=d.dtype))
    M = tf.bitwise.bitwise_and(tf.expand_dims(d, -1), m)
    b = tf.cast(tf.not_equal(M, 0), tf.int32)
    return b
