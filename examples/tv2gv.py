#!/usr/bin/env python
# coding: utf-8

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
        y_true = y0.reshape(ys)

        actions = agent.shortest_path_to_goal()
        for a in actions:
            agent.do_move(a)
            x_data = np.concatenate((x_data, gw.tview.reshape(xs)))
            y_true = np.concatenate((y_true, agent.one_hot_shortrange_goal().reshape(ys)))

        x_data = features_last(x_data)
        y_true = y_true[:, GOAL]

        yield NumpyTuple(x_data, y_true, nb_depth=3)

data = list(compute_xy(
    mytf.grid_world.EasyRoom(),
    mytf.grid_world.HardRoom(),
    mytf.grid_world.SuperHardRoom(),
))

max_group_size = max( x.shape[0] for x,y in data )

print(f'max_group_size = {max_group_size}')

import sys
sys.exit(0)

print(f'x_data: {x_data.shape}')
print(f'y_true: {y_true.shape}')

N = np.prod(x_data.shape[1:])
M = np.prod(y_true.shape[1:])

print(f'\nN={N}; M={M}')

I = tf.keras.layers.Input(shape=x_data.shape[1:], name="left of bang")

o = I
o = tf.keras.layers.Conv2D(kernel_size=3, filters=4 * N, activation="relu")(o)
o = tf.keras.layers.Dense(4 * N, activation="relu")(o)
o = tf.keras.layers.Dropout(0.1)(o)
o = tf.keras.layers.Dense(4 * N, activation="relu")(o)
o = tf.keras.layers.Conv2DTranspose(kernel_size=3, filters=4 * N, activation="relu")(o)
print(f"thought: {o.shape}")

o = tf.keras.layers.Dense(M, activation="relu")(o)
o = tf.keras.layers.Dropout(0.1)(o)
o = tf.keras.layers.Dense(M, activation="sigmoid")(o)
o = tf.keras.layers.Conv2DTranspose(kernel_size=1, filters=1, activation='sigmoid')(o)
o = tf.keras.layers.Reshape(y_true.shape[1:])(o)
print(f"final: {o.shape}")

SenseGoal = tf.keras.Model(inputs=(I,), outputs=(o,), name="SenseGoal")
SenseGoal.compile(loss="mse", optimizer="adam")
SenseGoal.summary()

(y_pred,) = SenseGoal(x_data)
print(f"y_pred: {y_pred.shape}")


def do_lap(epochs=100, workers=NUM_CPU, skip_print=False):
    global y_pred, ypff
    SenseGoal.fit(x=x_data, y=y_true, epochs=epochs, verbose=True, use_multiprocessing=True, workers=workers)
    (y_pred,) = SenseGoal(x_data)
    if not skip_print:
        for x, y in zip(x_data, y_pred):
            dx = decode_view(features_first(x), with_turtle=True)
            dy = str(tf.round(100 * y) / 100)
            print(side_by_side(str(dx), str(dy)))


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
