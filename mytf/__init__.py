#!/usr/bin/env python
# coding: utf-8

from .stfu import stfu_tensorflow

stfu_tensorflow()

from .strings import *
from .misc import *

from .grid_world import run_check as gw_run_check, GridWorld, Room, Actions
