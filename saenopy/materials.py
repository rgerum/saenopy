import numpy as np
from numba import njit
import numba
from saenopy.saveable import Saveable


def sample_and_integrate_function(func, min, max, step, zero_point=0, maximal_value=10e10):
    def xToI(x):
        return np.ceil((x - min) / step).astype(int)

    x = np.arange(min, max, step)
    y = func(x)

    if maximal_value is not None:
        y[y > maximal_value] = maximal_value

    # integrate
    int_y = np.cumsum(y * step)
    int_y -= int_y[xToI(zero_point)]

    # integrate again
    int_int_y = np.cumsum(int_y * step)
    int_int_y -= int_int_y[xToI(zero_point)]

    @njit()
    def look_up_y(x):
        shape = x.shape
        x = x.flatten()
        # we now have to pass this though the non-linearity function w (material model)
        # this function has been discretized and we interpolate between these discretisation steps

        # the discretisation step
        li = np.floor((x - min) / step)

        # the part between the two steps
        dli = (x - min) / step - li

        # if we are at the border of the discretisation, we stick to the end
        max_index = li > ((max - min) / step) - 2
        li[max_index] = int(((max - min) / step) - 2)
        dli[max_index] = 0

        # convert now to int after fixing the maximum
        lii = li.astype(np.int64)

        # interpolate between the two discretisation steps
        res0 = (1 - dli) * int_int_y[lii] + dli * int_int_y[lii + 1]
        res1 = (1 - dli) * int_y[lii] + dli * int_y[lii + 1]
        res2 = (1 - dli) * y[lii] + dli * y[lii + 1]

        return res0.reshape(shape), res1.reshape(shape), res2.reshape(shape)

    return look_up_y


class Material:
    """
    The base class for all material models.
    """
    parameters = {}
    min = -1.0
    max = 4.0
    step = 0.000001

    def stiffness(self, s):
        # to be overloaded by a material implementation
        raise NotImplementedError

    def energy(self, param):
        # to be overloaded by a material implementation
        raise NotImplementedError

    def force(self, param):
        # to be overloaded by a material implementation
        raise NotImplementedError

    def generate_look_up_table(self):
        return sample_and_integrate_function(self.stiffness, self.min, self.max, self.step)

    def __str__(self):
        return self.__class__.__name__+"("+", ".join(key+"="+str(value) for key, value in self.parameters.items())+")"


class SemiAffineFiberMaterial(Material, Saveable):
    """
    This class defines the default material of saenopy. The fibers show buckling (i.e. decaying stiffness) for
    -1 < lambda < 0, a linear stiffness response for small strains 0 < lambda < lambda_s, and strain stiffening for
    large strains lambda_s < lambda.

    Parameters
    ----------
    k : float
        The stiffness of the material in the linear regime.
    d_0 : float, optional
        The decay parameter in the buckling regime. If omitted the material shows no buckling but has a linear response
        for compression.
    lambda_s : float, optional
        The stretching where the strain stiffening starts. If omitted the material shows no strain stiffening.
    d_s : float, optional
        The parameter specifying how strong the strain stiffening is. If omitted the material shows no strain
        stiffening.
    """
    __save_parameters__ = ["k", "d_0", "lambda_s", "d_s"]
    k: float = None
    d_0: float = None
    lambda_s: float = None
    d_s: float = None

    def __init__(self, k, d_0=None, lambda_s=None, d_s=None):
        super().__init__()
        # parameters
        self.k = k
        self.d_0 = d_0 if d_0 is not None and d_0 >= 0 else None
        # buckling None (constant stiffness) and buckling zero (drop in stiffness) is not the same 
        if self.d_0 is not None and self.d_0 < 1e-30:  # approximate the zero case
            self.d_0 = 1e-30
        self.lambda_s = lambda_s if lambda_s is not None and lambda_s >= 0 else None
        self.d_s = d_s if d_s is not None and d_s >= 0 else None
        self.parameters = dict(k=k, d_0=d_0, lambda_s=lambda_s, d_s=d_s)

    def stiffness(self, s):
        # the linear spring regime (1 < s < s1)
        stiff = np.ones_like(s) * self.k

        # buckling for compression
        if self.d_0 is not None:
            buckling = s < 0
            stiff[buckling] = self.k * np.exp(s[buckling] / self.d_0)

        # and exponential stretch for overstretching fibers
        if self.d_s is not None and self.lambda_s is not None:
            stretching = s > self.lambda_s
            stiff[stretching] = self.k * np.exp((s[stretching] - self.lambda_s) / self.d_s)

        return stiff

    def energy(self, x0):
        # generate an empty target array
        x = x0.ravel()
        y = np.zeros_like(x)

        # find the buckling range
        if self.d_0 is not None:
            buckling = x < 0
        else:
            buckling = np.zeros_like(x) == 1
        # find the stretching range
        if self.d_s is not None and self.lambda_s is not None:
            stretching = self.lambda_s <= x
        else:
            stretching = np.zeros_like(x) == 1
        # and the rest is the linear range
        linear = (~buckling) & (~stretching)

        if self.d_0 is not None:
            # calculate the buckling energy
            y[buckling] = self.k * self.d_0 ** 2 * np.exp(x[buckling] / self.d_0) - self.k * self.d_0 * x[
                buckling] - self.k * self.d_0 ** 2

        # calculate the energy in the linear range
        y[linear] = 0.5 * self.k * x[linear] ** 2

        if self.d_s is not None and self.lambda_s is not None:
            # and in the stretching range
            dk = self.d_s * self.k
            sk = self.lambda_s * self.k
            d2k = self.d_s * dk
            y[stretching] = - 0.5 * self.lambda_s ** 2 * self.k + self.d_s * self.k * self.lambda_s - d2k \
                            + d2k * np.exp((x[stretching] - self.lambda_s) / self.d_s) - dk * x[stretching] + sk * x[
                                stretching]

        # return the resulting energy
        return y.reshape(x0.shape)

    def force(self, x0):
        # generate an empty target array
        x = x0.ravel()
        y = np.zeros_like(x)

        # find the buckling range
        if self.d_0 is not None:
            buckling = x < 0
        else:
            buckling = np.zeros_like(x) == 1
        # find the stretching range
        if self.d_s is not None and self.lambda_s is not None:
            stretching = self.lambda_s <= x
        else:
            stretching = np.zeros_like(x) == 1
        # and the rest is the linear range
        linear = (~buckling) & (~stretching)

        if self.d_0 is not None:
            # calculate the buckling energy
            y[buckling] = self.k * self.d_0 * np.exp(x[buckling] / self.d_0) - self.d_0 * self.k

        # calculate the energy in the linear range
        y[linear] = self.k * x[linear]
        if self.d_s is not None and self.lambda_s is not None:
            y[stretching] = self.k * self.lambda_s - self.d_s * self.k + self.d_s * self.k * np.exp(
                (x[stretching] - self.lambda_s) / self.d_s)

        # return the resulting energy
        return y.reshape(x0.shape)

    def generate_look_up_table(self):
        d_0 = self.d_0
        lambda_s = self.lambda_s
        d_s = self.d_s
        k = self.k

        buckling = self.d_0 is not None
        strain_stiffening = (self.lambda_s is not None and self.d_s is not None)

        # no buckling but strain stiffening
        if not buckling and strain_stiffening:
            @njit(numba.core.types.containers.UniTuple(numba.float64[:, :], 3)(numba.float64[:, :]))
            def get_all(s):
                shape = s.shape
                s = s.flatten()

                stiff = np.zeros_like(s)
                force = np.zeros_like(s)
                energy = np.zeros_like(s)

                dk = d_s * k
                sk = lambda_s * k
                d2k = d_s * dk

                for i, x in enumerate(s):
                    if x < lambda_s:
                        stiff[i] = k
                        force[i] = k * x
                        energy[i] = 0.5 * k * x ** 2
                    else:
                        stiff[i] = k * np.exp((x - lambda_s) / d_s)
                        force[i] = k * lambda_s - d_s * k + d_s * k * np.exp((x - lambda_s) / d_s)
                        energy[i] = - 0.5 * lambda_s ** 2 * k + d_s * k * lambda_s - d2k + d2k * np.exp(
                            (x - lambda_s) / d_s) - dk * x + sk * x

                return energy.reshape(shape), force.reshape(shape), stiff.reshape(shape)

            return get_all

        # buckling but no strain stiffening
        if buckling and not strain_stiffening:
            @njit(numba.core.types.containers.UniTuple(numba.float64[:, :], 3)(numba.float64[:, :]))
            def get_all(s):
                shape = s.shape
                s = s.flatten()

                stiff = np.zeros_like(s)
                force = np.zeros_like(s)
                energy = np.zeros_like(s)

                for i, x in enumerate(s):
                    if x < 0:
                        stiff[i] = k * np.exp(x / d_0)
                        force[i] = k * d_0 * np.exp(x / d_0) - d_0 * k
                        energy[i] = k * d_0 ** 2 * np.exp(x / d_0) - k * d_0 * x - k * d_0 ** 2
                    else:
                        stiff[i] = k
                        force[i] = k * x
                        energy[i] = 0.5 * k * x ** 2

                return energy.reshape(shape), force.reshape(shape), stiff.reshape(shape)

            return get_all

        # no buckling and no strain stiffening
        if not buckling and not strain_stiffening:
            @njit(numba.core.types.containers.UniTuple(numba.float64[:, :], 3)(numba.float64[:, :]))
            def get_all(s):
                shape = s.shape
                s = s.flatten()

                stiff = np.zeros_like(s)
                force = np.zeros_like(s)
                energy = np.zeros_like(s)

                for i, x in enumerate(s):
                    stiff[i] = k
                    force[i] = k * x
                    energy[i] = 0.5 * k * x ** 2

                return energy.reshape(shape), force.reshape(shape), stiff.reshape(shape)

            return get_all

        @njit(numba.core.types.containers.UniTuple(numba.float64[:, :], 3)(numba.float64[:, :]))
        def get_all(s):
            shape = s.shape
            s = s.flatten()

            stiff = np.zeros_like(s)
            force = np.zeros_like(s)
            energy = np.zeros_like(s)

            dk = d_s * k
            sk = lambda_s * k
            d2k = d_s * dk

            for i, x in enumerate(s):
                if x < 0:
                    stiff[i] = k * np.exp(x / d_0)
                    force[i] = k * d_0 * np.exp(x / d_0) - d_0 * k
                    energy[i] = k * d_0 ** 2 * np.exp(x / d_0) - k * d_0 * x - k * d_0 ** 2
                elif x < lambda_s:
                    stiff[i] = k
                    force[i] = k * x
                    energy[i] = 0.5 * k * x ** 2
                else:
                    stiff[i] = k * np.exp((x - lambda_s) / d_s)
                    force[i] = k * lambda_s - d_s * k + d_s * k * np.exp((x - lambda_s) / d_s)
                    energy[i] = - 0.5 * lambda_s ** 2 * k + d_s * k * lambda_s - d2k + d2k * np.exp(
                        (x - lambda_s) / d_s) - dk * x + sk * x

            return energy.reshape(shape), force.reshape(shape), stiff.reshape(shape)

        return get_all


class LinearMaterial(Material):
    """
    A material that has a linear stiffness.

    Parameters
    ----------
    k : float
        The stiffness of the material.
    """
    def __init__(self, k):
        # parameters
        self.k = k
        self.parameters = dict(k=k)

    def stiffness(self, s):
        # the linear spring regime (0 < s < s1)
        stiff = np.ones_like(s) * self.k

        return stiff

    def force(self, x):
        return self.k * x

    def energy(self, x):
        # calculate the energy in the linear range
        return 0.5 * self.k * x ** 2
