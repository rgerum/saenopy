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


def sampleAndIntegrateFunction(func, min, max, step, zero_point=1, maximal_value=10e10):
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
    min = 0.0
    max = 5.0
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
    This class defines the default material of saenopy. The fibers show buckling (i.e. decaying stiffness) for
    0 < lambda < 1, a linear stiffness response for small strains 1 < lambda < lambda_s, and strain stiffening for
    large strains lambda_s < lambda.

    Parameters
    ----------
    k : float
        The stiffness of the material in the linear regime.
    d0 : float, optional
        The decay parameter in the buckling regime. If omitted the material shows no buckling but has a linear response
        for compression.
    lambda_s : float, optional
        The stretching where the exponential stiffening starts. If omitted the material shows no exponential stiffening.
    ds : float, optional
        The parameter specifying how strong the exponential stiffening is. If omitted the material shows no exponential
        stiffening.
    """

    def __init__(self, k, d0=None, lambda_s=None, ds=None):
        # parameters
        self.k = k
        self.d0 = d0 if d0 is not None and d0 > 0 else None
        self.lambda_s = lambda_s if lambda_s is not None and lambda_s > 0 else None
        self.ds = ds if ds is not None and ds > 0 else None
        self.parameters = dict(k=k, d0=d0, lambda_s=lambda_s, ds=ds)

    def stiffness(self, s):
        self._check_parameters_valid()

        # the linear spring regime (1 < s < s1)
        stiff = np.ones_like(s) * self.k

        # buckling for compression
        if self.d0 is not None:
            buckling = s < 1
            stiff[buckling] = self.k * np.exp((s[buckling] - 1) / self.d0)

        # and exponential stretch for overstretching fibers
        if self.ds is not None and self.lambda_s is not None:
            stretching = s > self.lambda_s
            stiff[stretching] = self.k * np.exp((s[stretching] - self.lambda_s) / self.ds)

        return stiff

    def energy(self, x0):
        self._check_parameters_valid()

        # generate an empty target array
        x = x0.ravel()
        y = np.zeros_like(x)

        # find the buckling range
        if self.d0 is not None:
            buckling = x < 1
        else:
            buckling = np.zeros_like(x) == 1
        # find the stretching range
        if self.ds is not None and self.lambda_s is not None:
            stretching = self.lambda_s <= x
        else:
            stretching = np.zeros_like(x) == 1
        # and the rest is the linear range
        linear = (~buckling) & (~stretching)

        k = self.k
        d0 = self.d0
        lambda_s = self.lambda_s
        ds = self.ds

        if self.d0 is not None:
            # calculate the buckling energy

            y[buckling] = k * d0 ** 2 * np.exp((x[buckling] - 1) / self.d0) - k * d0 * x[buckling] - k * d0 ** 2 + k * d0
        y[linear] = 0.5 * k * x[linear] ** 2 - k * x[linear] + 0.5 * k
        if self.ds is not None and self.lambda_s is not None:
            y[stretching] = 0.5 * k * lambda_s ** 2 - k * lambda_s + 0.5 * k \
                            + (k * lambda_s - k - k * ds) * x[stretching] + k * ds ** 2 * np.exp((x[stretching] - lambda_s) / ds) \
                            - (k * lambda_s - k - k * ds) * lambda_s - k * ds ** 2

        # return the resulting energy
        return y.reshape(x0.shape)

    def force(self, x0):
        self._check_parameters_valid()

        # generate an empty target array
        x = x0.ravel()
        y = np.zeros_like(x)

        # find the buckling range
        if self.d0 is not None:
            buckling = x < 1
        else:
            buckling = np.zeros_like(x) == 1
        # find the stretching range
        if self.ds is not None and self.lambda_s is not None:
            stretching = self.lambda_s <= x
        else:
            stretching = np.zeros_like(x) == 1
        # and the rest is the linear range
        linear = (~buckling) & (~stretching)

        if self.d0 is not None:
            # calculate the buckling energy
            y[buckling] = self.k * self.d0 * np.exp((x[buckling] - 1) / self.d0) - self.d0 * self.k

        # calculate the energy in the linear range
        y[linear] = self.k * x[linear] - self.k
        if self.ds is not None and self.lambda_s is not None:
            y[stretching] = self.k * self.lambda_s - self.k - self.ds * self.k + self.ds * self.k * np.exp(
                (x[stretching] - self.lambda_s) / self.ds)

        # return the resulting energy
        return y.reshape(x0.shape)

    def _check_parameters_valid(self):
        # stiffening is not allowed in the buckling regime
        if self.lambda_s is not None and self.lambda_s <= 1:
            self.lambda_s = 1


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

    def energy(self, x):
        # calculate the energy in the linear range
        return 0.5 * self.k * x ** 2
