#!/usr/bin/env python
# coding: utf-8

import sys, os
import multiprocessing
import mytf
import tensorflow as tf
import numpy as np

from mytf.grid_world.const import GOAL
from mytf.grid_world.agent import Agent
from mytf.grid_world.util import decode_view
from mytf.strings import side_by_side
from mytf.misc import features_last, features_first
from mytf.util import NumpyTuple

try:
    IPYTHON = get_ipython()
except:
    IPYTHON = False

NUM_CPU = multiprocessing.cpu_count()

def compute_xy(*worlds):
    for gw in worlds:
        agent = Agent(gw)

        x0 = gw.tview
        y0 = agent.one_hot_shortrange_goal()
        xs = (1, *x0.shape)
        ys = (1, *y0.shape)
        x_data = x0.reshape(xs)

        actions = agent.shortest_path_to_goal()
        for a in actions:
            agent.do_move(a)
            x_data = np.concatenate((x_data, gw.tview.reshape(xs)))

        x_data = features_last(x_data)
        y_true = agent.sps

        yield NumpyTuple(x_data, y_true, nb_depth=3)

data = list(compute_xy(
    mytf.grid_world.EasyRoom(),
    mytf.grid_world.HardRoom(),
    mytf.grid_world.SuperHardRoom(),
))

# max_group_size = max( x.shape[0] for x,y in data )
# print(f'max_group_size = {max_group_size}')

x_data = tf.ragged.stack([ i[0] for i in data ], axis=0).to_tensor()
y_true = tf.ragged.stack([ i[1] for i in data ], axis=0).to_tensor()

x_shape = x_data.shape[1:]
y_shape = y_true.shape[1:]

N = np.prod(x_shape[1:]) # skip the steps dimension giving 7,7,5
M = np.prod(y_shape) # 15,17

print(f'x_shape: {x_shape}')
print(f'y_shape: {y_shape}')
print(f'x_data: {x_data.shape}')
print(f'y_true: {y_true.shape}')
print(f'\nN={N}; M={M}')

def exit_if_true(x):
    try:
        if int(os.environ[x]) > 0:
            sys.exit(0)
    except (KeyError, TypeError):
        pass

exit_if_true('DATA_ONLY')

I = tf.keras.layers.Input(shape=x_shape, name="left of bang")

o = I
o = tf.keras.layers.ConvLSTM2D(kernel_size=3, filters=4*N, activation="relu")(o)
print(f"thought: {o.shape}")
exit_if_true('THOUGHT_ONLY')

while np.prod(o.shape[1:3]) < M:
    s0 = o.shape[1:]
    o = tf.keras.layers.Conv2DTranspose(kernel_size=3, filters=4*M, activation='relu')(o)
    print(f'{s0} -> {o.shape[1:]}')

o = tf.keras.layers.Flatten()(o)
o = tf.keras.layers.Dense(M, activation='sigmoid')(o)
o = tf.keras.layers.Dropout(0.1)(o)
o = tf.keras.layers.Reshape(y_shape)(o)
print(f"final-reshape: {o.shape}")
exit_if_true('FINAL_ONLY')

SenseGoal = tf.keras.Model(inputs=(I,), outputs=(o,), name="SenseGoal")
SenseGoal.compile(loss="mse", optimizer="adam")
SenseGoal.summary()

exit_if_true('SUMMARY_ONLY')

(y_pred,) = SenseGoal(x_data)
print(f"y_pred: {y_pred.shape}")

exit_if_true('PRED_ONLY')

def do_lap(epochs=100, workers=NUM_CPU, skip_print=False):
    global y_pred, ypff
    SenseGoal.fit(x=x_data, y=y_true, epochs=epochs, verbose=True, use_multiprocessing=True, workers=workers)
    (y_pred,) = SenseGoal(x_data)
    if not skip_print:
        for t, p in zip(y_true, y_pred):
            dt = str(np.array(tf.round(100 * t) / 10, dtype=np.int32))
            dp = str(np.array(tf.round(100 * p) / 10, dtype=np.int32))
            print(side_by_side(str(dt), str(dp)))


if __name__ == "__main__" and not IPYTHON:
    while True:
        try:
            epochs = int(input("epochs (0: exit): "))
            if epochs < 1:
                break
        except (KeyboardInterrupt, EOFError):
            print('\n.oO( ^C/^D counts as 0 )')
            break
        except:
            continue
        do_lap(epochs=epochs)
else:
    do_lap(epochs=1, skip_print=True)
    print("use /do_lap [epochs=100] to run the network, populate ypff, and print things")
