#!/usr/bin/env python

try:
    import mytf
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, '.')
    import mytf

import tensorflow as tf
import numpy as np

shr = mytf.grid_world.SuperHardRoom()
gw = shr
s0 = gw.do_move('blah')

# create data
actions = "e SE SE SE e e e e e SE s s s s SW ".split()
x_input = s0.cat( *(gw.do_move(x) for x in actions) ).promote(depth=3)
y_true  = gw.encode(pad=(20,20))
y_true  = y_true.reshape((1,*y_true.shape))

ims = x_input[0].shape[1:]
vms = x_input[1].shape[1:]
yms = y_true.shape[1:]
fdr = np.prod(x_input[0].shape[1:])
fcl = np.prod(yms[1:]) # fcl=20*20 rather than 5*20*20, cuz 2000 is too much LSTM
fdl = np.prod(yms)

lob = tf.keras.layers.Input(shape=ims, name="left of bang")
rob = tf.keras.layers.Input(shape=ims, name="right of bang")
act = tf.keras.layers.Input(shape=vms, name="action")
print(f'lob: {lob.shape}')
print(f'act: {act.shape}')
print(f'rob: {rob.shape}')

# input merging
o = tf.keras.layers.Concatenate()([ tf.keras.layers.Flatten()(x) for x in (lob,act,rob) ])
o = tf.keras.layers.Dense(fdr, activation='relu')(o)
o = tf.keras.layers.Reshape(ims)(o)
print(f'cdr: {o.shape}')

# Conv ⇒ LSTM ⇒ Dense
o = tf.keras.layers.ConvLSTM2D(kernel_size=2, filters=fcl, activation='relu')(o)
o = tf.keras.layers.Dense(fdl, activation='relu')(o)
print(f'cl2: {o.shape}')

# final reshape
o = tf.keras.layers.Flatten()(o)
o = tf.keras.layers.Dense(fdl, activation='relu')(o)
o = tf.keras.layers.Reshape(yms)(o)
print(f'fdr: {o.shape}')

# compile model
MapImaginerModel = tf.keras.Model(inputs=(lob,act,rob), outputs=(o,), name='MapImaginerModel')
MapImaginerModel.compile(loss='mse', optimizer='adam')
MapImaginerModel.summary()

y_pred, = MapImaginerModel(x_input)

print(f'x_input: {x_input.shape}')
print(f'y_true:  {y_true.shape}')
print(f'y_pred:  {y_pred.shape}')

# training/fitting
