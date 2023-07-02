#!/usr/bin/env python
# coding: utf-8

import multiprocessing
import mytf
import tensorflow as tf
import numpy as np

from mytf.grid_world.agent import Agent
from mytf.grid_world.util import decode_view
from mytf.strings import side_by_side

NUM_CPU = multiprocessing.cpu_count()
gw = mytf.grid_world.SuperHardRoom()
agent = Agent(gw)

x0 = gw.tview
y0 = agent.one_hot_shortrange_goal()
xs = (1, *x0.shape)
ys = (1, *y0.shape)
x_data = x0.reshape(xs)
y_true = y0.reshape(ys)

actions = agent.shortest_path_to_goal()
for a in actions:
    agent.do_move(a)
    x_data = np.concatenate((x_data, gw.tview.reshape(xs)))
    y_true = np.concatenate((y_true, agent.one_hot_shortrange_goal().reshape(ys)))

x_data = tf.transpose(x_data, perm=[0, 2, 3, 1])
y_true = tf.transpose(y_true, perm=[0, 2, 3, 1])

print(f"x_data = {x_data.shape}")
print(f"y_true = {y_true.shape}")

N = np.prod(x_data.shape[1:])
M = np.prod(y_true.shape[1:])
I = tf.keras.layers.Input(shape=x_data.shape[1:], name='left of bang')

o = I
o = tf.keras.layers.Conv2D(kernel_size=3, filters=4*N, activation='relu')(o)
o = tf.keras.layers.Dense(4*N, activation='relu')(o)
o = tf.keras.layers.Dropout(0.1)(o)
o = tf.keras.layers.Dense(4*N, activation='relu')(o)
o = tf.keras.layers.Conv2DTranspose(kernel_size=3, filters=4*N, activation='relu')(o)
print(f'thought: {o.shape}')

o = tf.keras.layers.Dense(M, activation='relu')(o)
o = tf.keras.layers.Dropout(0.1)(o)
o = tf.keras.layers.Dense(M, activation='relu')(o)
o = tf.keras.layers.Dense(y_true.shape[-1], activation='relu')(o)
print(f'final: {o.shape}')

SenseGoal = tf.keras.Model(inputs=(I,), outputs=(o,), name='SenseGoal')
SenseGoal.compile(loss='mse', optimizer='adam')
SenseGoal.summary()

y_pred, = SenseGoal(x_data)
print(f'y_pred: {y_pred.shape}')

def do_lap(epochs=100, workers=NUM_CPU):
    SenseGoal.fit(x=x_data, y=y_true, epochs=epochs, verbose=True, use_multiprocessing=True, workers=workers)
    y_pred, = SenseGoal(x_data)
    for t,p in zip(y_true, y_pred):
        d1 = decode_view(y, with_turtle=True)
        d2 = decode_view(yp, with_turtle=True)
        print( side_by_side(str(d1), str(d2)) )

if __name__ == '__main__':
    do_lap()
else:
    print("use /do_lap to run the network 100 times and print things")
