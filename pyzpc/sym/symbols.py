import sympy
from sympy import Matrix, symarray, Symbol
from numpy import empty, ndindex
from functools import reduce

def mat(name: str, *args):
    for arg in args:
        assert(isinstance(arg, int))
    assert('@' not in name)

    arr = empty(args, dtype=object)
    for index in ndindex(args):
        arr[index] = Symbol(f'{name}@{"@".join(map(str, index))}')
    return Matrix(arr)

def mat_size(mat):
    return reduce(lambda x, y: x * y, mat.shape)

