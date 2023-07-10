#!/usr/bin/env python
# coding: utf-8

import re
from collections import defaultdict

NAMES = defaultdict(lambda: 0)
KIHE = {"kernel_initializer": "HeNormal"}


def numberize_name(name):
    if m := re.match("^(.+?)\d+$"):
        name = m.group(1)
    x = NAMES[name]
    NAMES[name] += 1
    return f"{name}{x}"


def rcm(inputs, f=32, k=3, s=1, activation="relu", name="rcm"):
    name = numberize_name(name)
    x = inputs
    for i in range(2):
        x = tf.keras.layers.Conv2D(f, k, s, padding="same", **KIHE, name=f"{name}-{i}/3x3")(x)
        x = tf.keras.layers.Activation(activation, name=f"{name}-{i}/{activation}")(x)
        x = tf.keras.layers.Conv2D(f, k, s, padding="same", **KIHE, name=f"{name}-{i}/3x3")(x)
        x = Add(name=f"{name}-{i}/plus")([x, inputs])
        x = tf.keras.layers.Activation(activation, name=f"{name}-{i}/{activation}")(x)
    return x
