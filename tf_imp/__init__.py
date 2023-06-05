#!/usr/bin/env python
# coding: utf-8

from .stfu import stfu_tensorflow

stfu_tensorflow()

from .strings import *

# there's actually not much to import here mostly we're invoking the egglib
# __init__.py what adds all the eggs to the import path.
from .egglib import *

from .grid_world import run_check as gw_run_check, GridWorld, Room, A as GWActions
