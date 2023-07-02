#!/usr/bin/env python
# coding: utf-8

import os
import re
import logging
import warnings

APPLICABLE_LOGGERS = {"tensorflow", "tensorboard", "h5py", "numpy"}

DEFAULT_PATTERNS = (
    "is slow compared to the",
    "np.bool8",
)


def relevant_package(x):
    for item in APPLICABLE_LOGGERS:
        if item in x:
            return True


def unique_patterns(x):
    already = set()
    for item in x:
        if item not in already:
            already.add(item)
            if item in ("+",):
                yield from DEFAULT_PATTERNS
            else:
                yield item


def stfu_tensorflow(*patterns):
    """
    patterns are things we don't realy want to hear about. They're computed as
    regex, but simple strings will work fine too

    If the pattern '+' is given, it will be replaces with the DEFAULT_PATTERNS
    from this package.

    If no patterns are given, we'll just use DEFAULT_PATTERNS directly.
    """

    if not patterns:
        patterns = DEFAULT_PATTERNS
    else:
        patterns = tuple(unique_patterns(patterns))

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # this doesn't help as much as you'd hope

    # This applies more to numpy than anything else tf.dtypes use np.bool8
    # (still), which is deprecated, but I can't do anything about it so the
    # warning is more than useless to me; it's pure annoyance
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    try:
        from tensorflow.python.util import deprecation

        # Monkey patching deprecation utils to shut it up!
        def deprecated(date, instructions, warn_once=True):
            def deprecated_wrapper(func):
                return func

            return deprecated_wrapper

        deprecation.deprecated = deprecated
    except ImportError:
        pass

    class STFUFilter(logging.Filter):
        def filter(self, record):
            # https://docs.python.org/3/library/logging.html#logging.LogRecord

            if relevant_package(record.name):
                # If any of words in the APPLICABLE_LOGGERS appear in
                # record.name, then consider our patterns
                for pat in patterns:
                    if hasattr(pat, "search") and pat.search(record.msg):
                        return False  # matched, don't emit this log

                    elif re.search(pat, record.msg):
                        return False  # matched, don't emit this log

            return True  # go ahead and emit the log

    for handler in logging.root.handlers:
        handler.addFilter(STFUFilter())
