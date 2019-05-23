import numpy as np
from numba import types
from numba.extending import overload


@overload(np.linalg.norm)
def overload_norm(A, axis):
    """ implement axis keyword """
    if isinstance(A, types.Array) or isinstance(axis, types.Integer):
        def norm_impl(A, axis):
            return np.sqrt(np.sum(A ** 2, axis=axis))
        return norm_impl
