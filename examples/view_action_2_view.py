#!/usr/bin/env python
# coding: utf-8

import multiprocessing
import mytf
import tensorflow as tf
import numpy as np

from mytf.grid_world.util import decode_view
from mytf.strings import side_by_side

NUM_CPU = multiprocessing.cpu_count()

print( ", ".join( x for x in dir(mytf) if not x.startswith('_') ))
shr = mytf.grid_world.SuperHardRoom()
gw = shr
s0 = gw.do_move('blah')

actions = "e SE SE SE e e e e e SE s s s s SW ".split()
x_7, x_a, y_7 = s0.cat( *(gw.do_move(x) for x in actions) )

x_7 = tf.transpose(x_7, perm=[0, 2,3,1])
y_7 = tf.transpose(y_7, perm=[0, 2,3,1])

print('x_70\n', x_7[0])
print('y_70\n', y_7[0])

print(f'x_7 = {x_7.shape}')
print(f'x_a = {x_a.shape}')
print(f'y_7 = {y_7.shape}')

lob = tf.keras.layers.Input(shape=x_7.shape[1:], name='left of bang')
act = tf.keras.layers.Input(shape=x_a.shape[1:], name='action')
o = tf.keras.layers.Concatenate(name='mixed')([
    tf.keras.layers.Flatten(name='flat-lob')(lob),
    tf.keras.layers.Flatten(name='flat-act')(act),
])

N = np.prod(x_7.shape[1:])

o = tf.keras.layers.Dense(20 * N, activation='relu')(o)
o = tf.keras.layers.Dense(20 * N, activation='relu')(o)
o = tf.keras.layers.Reshape((*x_7.shape[1:3], 100))(o)
print(f'merged input: {o.shape}')

o = tf.keras.layers.Conv2D(kernel_size=3, filters=5*N, activation='relu')(o)
o = tf.keras.layers.Dense(2*N, activation='relu')(o)
o = tf.keras.layers.Dense(2*N, activation='relu')(o)
o = tf.keras.layers.Conv2DTranspose(kernel_size=3, filters=5*N, activation='relu')(o)
print(f'thought: {o.shape}')

o = tf.keras.layers.Dense(1024, activation='relu')(o)
o = tf.keras.layers.Dense(1024, activation='relu')(o)
o = tf.keras.layers.Dense(5, activation='relu')(o)
print(f'final: {o.shape}')

PredictAction = tf.keras.Model(inputs=(lob,act), outputs=(o,), name='PredictAction')
PredictAction.compile(loss='mse', optimizer='adam')
PredictAction.summary()

example_input = (x_7, x_a)
y_pred, = PredictAction(example_input)
print(f'y_pred: {y_pred.shape}')

def do_lap(epochs=100, workers=NUM_CPU):
    PredictAction.fit(x=(x_7,x_a), y=y_7, epochs=epochs, verbose=True, use_multiprocessing=True, workers=workers)
    y_pred, = PredictAction((x_7,x_a))
    for y,y_ in zip(y_7, y_pred):
        y = tf.transpose(y, perm=[2,0,1])
        y_ = tf.transpose(y_, perm=[2,0,1])
        d1 = decode_view(y, with_turtle=True)
        d2 = decode_view(y_, with_turtle=True)
        print( side_by_side(str(d1), str(d2)) )

# print("use /do_lap to run the network 100 times and print things")
