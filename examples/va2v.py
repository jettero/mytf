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

shr = mytf.grid_world.SuperHardRoom()
gw = shr
s0 = gw.do_move("blah")

agent = Agent(gw)

actions = agent.shortest_path_to_goal()
x_7o, x_a, y_7o = s0.cat(*(gw.do_move(x) for x in actions))

x_7 = tf.transpose(x_7o, perm=[0, 2, 3, 1])
y_7 = tf.transpose(y_7o, perm=[0, 2, 3, 1])

print(f"x_7 = {x_7.shape}")
print(f"x_a = {x_a.shape}")
print(f"y_7 = {y_7.shape}")

lob = tf.keras.layers.Input(shape=x_7.shape[1:], name="left of bang")
act = tf.keras.layers.Input(shape=x_a.shape[1:], name="action")
o = tf.keras.layers.Concatenate(name="mixed")(
    [
        tf.keras.layers.Flatten(name="flat-lob")(lob),
        tf.keras.layers.Flatten(name="flat-act")(act),
    ]
)

N = np.prod(x_7.shape[1:])

o = tf.keras.layers.Dense(20 * N, activation="relu")(o)
o = tf.keras.layers.Dropout(0.1)(o)
o = tf.keras.layers.Dense(20 * N, activation="relu")(o)
o = tf.keras.layers.Reshape((*x_7.shape[1:3], 100))(o)
print(f"merged input: {o.shape}")

o = tf.keras.layers.Conv2D(kernel_size=3, filters=5 * N, activation="relu")(o)
o = tf.keras.layers.Dense(2 * N, activation="relu")(o)
o = tf.keras.layers.Dropout(0.1)(o)
o = tf.keras.layers.Dense(2 * N, activation="relu")(o)
o = tf.keras.layers.Conv2DTranspose(kernel_size=3, filters=5 * N, activation="relu")(o)
print(f"thought: {o.shape}")

o = tf.keras.layers.Dense(1024, activation="relu")(o)
o = tf.keras.layers.Dropout(0.1)(o)
o = tf.keras.layers.Dense(1024, activation="relu")(o)
o = tf.keras.layers.Dense(5, activation="relu")(o)
print(f"final: {o.shape}")

PredictAction = tf.keras.Model(inputs=(lob, act), outputs=(o,), name="PredictAction")
PredictAction.compile(loss="mse", optimizer="adam")
PredictAction.summary()

input_data = (x_7, x_a)
(y_pred,) = PredictAction(input_data)
print(f"y_pred: {y_pred.shape}")

y_predo = np.transpose(y_pred, axes=[0, 3, 1, 2])


def do_lap(epochs=100, workers=NUM_CPU):
    global y_pred, y_predo
    PredictAction.fit(
        x=input_data,
        y=y_7,
        epochs=epochs,
        verbose=True,
        use_multiprocessing=True,
        workers=workers,
    )
    (y_pred,) = PredictAction(input_data)
    y_predo = np.transpose(y_pred, axes=[0, 3, 1, 2])
    for y, yp in zip(y_7o, y_predo):
        d1 = decode_view(y, with_turtle=True)
        d2 = decode_view(yp, with_turtle=True)
        print(side_by_side(str(d1), str(d2)))


# print("use /do_lap to run the network 100 times and print things")
