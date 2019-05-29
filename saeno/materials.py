import numpy as np
from numba import njit

def saveEpsilon(epsilon, fname, CFG):
    """
    Save the material function to a file, e.g. for further plotting.
    """
    imax = int(np.ceil((CFG["EPSMAX"] + 1.0) / CFG["EPSSTEP"]))

    with open(fname, "w") as fp:
        for i in range(imax):
            lambd = (i * CFG["EPSSTEP"]) - 1.0

            fp.write(str(lambd) + " " + str(epsilon[i]) + "\n")

    lambd = (np.arange(imax) * CFG["EPSSTEP"]) - 1.0
    np.save(fname.replace(".dat", ".npy"), np.array([lambd, epsilon]).T)


def sampleAndIntegrateFunction(func, min, max, step, zero_point=0, maximal_value=10e10):
    def iToX(i):
        return i * step + min

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
    def lookUpY(x):
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

    return lookUpY


class Material:
    """
    The base class for all material models.
    """
    parameters = {}
    min = -1
    max = 4.0
    step = 0.000001

    def stiffness(self, s):
        # to be overloaded by a material implementation
        return s

    def generate_look_up_table(self):
        return sampleAndIntegrateFunction(self.stiffness, self.min, self.max, self.step)

    def __str__(self):
        return self.__class__.__name__+"("+", ".join(key+"="+str(value) for key, value in self.parameters.items())+")"


class SemiAffineFiberMaterial(Material):
    """
    A material that has a linear stiffness range from 0 to s1 which then goes to an exponential stiffening regime. For
    compression the material responds with an exponential decay of the stiffness, a buckling response.

    Parameters
    ----------
    k1 : float
        The stiffness of the material in the linear regime.
    ds0 : float, optional
        The decay parameter in the buckling regime. If omitted the material shows no buckling but has a linear response
        for compression.
    s1 : float, optional
        The stretching where the exponential stiffening starts. If omitted the material shows no exponential stiffening.
    ds1 : float, optional
        The parameter specifying how strong the exponential stiffening is. If omitted the material shows no exponential
        stiffening.
    """
    def __init__(self, k1, ds0=None, s1=None, ds1=None):
        # parameters
        self.k1 = k1
        self.ds0 = ds0
        self.s1 = s1
        self.ds1 = ds1
        self.parameters = dict(k1=k1, ds0=ds0, s1=s1, ds1=ds1)

    def stiffness(self, s):
        # the linear spring regime (0 < s < s1)
        stiff = np.ones_like(s) * self.k1

        # buckling for compression
        if self.ds0 is not None:
            buckling = s < 0
            stiff[buckling] = self.k1 * np.exp(s[buckling] / self.ds0)

        # and exponential stretch for overstretching fibers
        if self.ds1 is not None and self.s1 is not None:
            stretching = s > self.s1
            stiff[stretching] = self.k1 * np.exp((s[stretching] - self.s1) / self.ds1)

        return stiff

    def energy(self, x0):
        # generate an empty target array
        x = x0.ravel()
        y = np.zeros_like(x)

        # find the buckling range
        if self.ds0 is not None:
            buckling = x < 0
        else:
            buckling = np.zeros_like(x) == 1
        # find the stretching range
        if self.ds1 is not None and self.s1 is not None:
            stretching = self.s1 <= x
        else:
            stretching = np.zeros_like(x) == 1
        # and the rest is the linear range
        linear = (~buckling) & (~stretching)

        if self.ds0 is not None:
            # calculate the buckling energy
            y[buckling] = self.k1 * self.ds0 ** 2 * np.exp(x[buckling] / self.ds0) - self.k1 * self.ds0 * x[buckling] - self.k1 * self.ds0 ** 2

        # calculate the energy in the linear range
        y[linear] = 0.5 * self.k1 * x[linear] ** 2

        if self.ds1 is not None and self.s1 is not None:
            # and in the stretching range
            dk = self.ds1 * self.k1
            sk = self.s1 * self.k1
            d2k = self.ds1 * dk
            y[stretching] = - 0.5 * self.s1 ** 2 * self.k1 + self.ds1 * self.k1 * self.s1 - d2k \
                            + d2k * np.exp((x[stretching] - self.s1) / self.ds1) - dk * x[stretching] + sk * x[stretching]

        # return the resulting energy
        return y.reshape(x0.shape)


class LinearMaterial(Material):
    """
    A material that has a linear stiffness.

    Parameters
    ----------
    k1 : float
        The stiffness of the material.
    """
    def __init__(self, k1):
        # parameters
        self.k1 = k1
        self.parameters = dict(k1=k1)

    def stiffness(self, s):
        # the linear spring regime (0 < s < s1)
        stiff = np.ones_like(s) * self.k1

        return stiff

    def energy(self, x):
        # calculate the energy in the linear range
        return 0.5 * self.k1 * x**2
