import numpy as np
import sys

try:
    from numba import types
    from numba.extending import overload


    @overload(np.linalg.norm)
    def overload_norm(A, axis):
        """ implement axis keyword """
        if isinstance(A, types.Array) or isinstance(axis, types.Integer):
            def norm_impl(A, axis):
                return np.sqrt(np.sum(A ** 2, axis=axis))
            return norm_impl
except ImportError:
    print("WARNING: no numba found, iterations can be considerably slowed down", file=sys.stderr)

    def empty_decorator(*args, **kwargs):

        def wrapper(func):
            return func

        return wrapper

    class Numba:
        pass

    # if numba is not installed, skip the jit compilation
    Numba.njit = empty_decorator
    Numba.jit = empty_decorator

    # mock the numby module with this empty shell
    sys.modules.update({"numba": Numba})
