#!/usr/bin/env python
# coding: utf-8

import pytest
import mytf as tfi

@pytest.fixture
def x_data():
    return [
        list( "list of strings".split() ),
        list( b'list of bytes'.split() ),
    ]

@pytest.fixture
def y_debitsed():
    return [
        list( b"list of strings".split() ),
        list( b'list of bytes'.split() ),
    ]

@pytest.fixture
def y_stringified():
    return [
        list( "list of strings".split() ),
        list( "list of bytes".split() ),
    ]

def test_fixtures_work(x_data, y_debitsed, y_stringified):
    assert len(x_data) == len(y_debitsed)
    assert len(y_debitsed) == len(y_stringified)

def test_strings_to_bits(x_data):
    bits = tfi.strings_to_bits('supz')
    assert tuple(bits.shape) == (4,8)

    bits = tfi.strings_to_bits(['supz', 'mang'])
    assert tuple(bits.shape) == (2,None,8)

    bits = tfi.strings_to_bits(['supz', 'longer'])
    assert tuple(bits.shape) == (2,None,8)

def test_strings_to_bits_to_bytes(x_data, y_debitsed):
    for lhs,rhs in zip(x_data, y_debitsed):
        b = tfi.strings_to_bits(lhs)
        B = tfi.fuzzy_bits_to_bytes(b)

        # B == rhs is vectorized e.g., tf.array([True, True, True]) ... 
        # originally we had bool(B == rhs), which crashes, "truth value of an
        # array with more than one element is ambiguous."
        assert all(B == rhs)

def test_strings_to_bits_to_strings(x_data, y_stringified):
    for lhs,rhs in zip(x_data, y_stringified):
        b = tfi.strings_to_bits(lhs)
        B = tfi.fuzzy_bits_to_bytes(b)

        sb = tfi.stringify_bytes(b)   
        sB = tfi.stringify_bytes(B) 

        assert sb == rhs
        assert sB == rhs
