import pytest
import gsim_py


def test_sum_as_string():
    assert gsim_py.sum_as_string(1, 1) == "2"
