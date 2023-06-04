# coding: utf-8

import sys
import os
import glob

egglib_dir = os.path.dirname( __file__ )
for _egg in glob.glob( os.path.join(egglib_dir, '*.egg') ):
    _egg = os.path.abspath(_egg)
    if _egg not in sys.path:
        sys.path.append(_egg)
