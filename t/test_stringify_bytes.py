#!/usr/bin/env python
# coding: utf-8

import pytest
import tf_imp as tfi
import hashlib

@pytest.fixture
def x_data():
    yield "string"
    yield b'test bytes'
    yield list( "list of strings".split() )
    yield list( b'list of bytes'.split() )

@pytest.fixture
def y_debitsed(x_data):
    yield b"string"
    yield b'test bytes'
    yield list( b"list of strings".split() )
    yield list( b'list of bytes'.split() )

@pytest.fixture
def y_stringified(x_data):
    yield "string"
    yield 'test bytes'
    yield list( "list of strings".split() )
    yield list( 'list of bytes'.split() )

# def test_strings_to_bits_to_bytes(x_data, y_debitsed):
#     for lhs,rhs in zip(x_data, y_debitsed):
#         b = tfi.strings_to_bits(lhs)
#         B = tfi.fuzzy_bits_to_bytes(b)

#         assert B == rhs

# def test_strings_to_bits_to_strings(x_data, y_stringified):
#     for lhs,rhs in zip(x_data, y_debitsed):
#         b = tfi.strings_to_bits(lhs)
#         B = tfi.fuzzy_bits_to_bytes(b)

#         sb = tfi.stringify_bytes(b)   
#         sB = tfi.stringify_bytes(B) 

#         assert sb == rhs
#         assert sB == rhs
