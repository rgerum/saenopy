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
        max_index = li > ((min - min) / step) - 2
        li[max_index] = int(((min - min) / step) - 2)
        dli[max_index] = 0

        # convert now to int after fixing the maximum
        lii = li.astype(np.int64)

        # interpolate between the two discretisation steps
        epsilon_b = (1 - dli) * y[lii] + dli * y[lii + 1]
        epsbar_b = (1 - dli) * int_y[lii] + dli * int_y[lii + 1]
        epsbarbar_b = (1 - dli) * int_int_y[lii] + dli * int_int_y[lii + 1]

        return epsilon_b.reshape(shape), epsbar_b.reshape(shape), epsbarbar_b.reshape(shape)

    return lookUpY


def semiAffineFiberMaterial(k1, ds0=None, s1=None, ds1=None, min=-1, max=4.0, step=0.000001):
    """
    A material that has a linear stiffness range from 0 to s1 which then goes to an exponential stiffening regime. For
    compression the material responds with an exponential deay of the stiffness, a buckling response.

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
    min : float, optional
        Where to start to sample the function.
    max : float, optional
        Where to stop to sample the function.
    step : float, optional
        In which steps to sample the function.

    Returns
    -------
    func : function
        A function which returns, the stiffness, force, and energy for a given stretching.
    """

    def stiffness(s):
        # the linear spring regime (0 < s < s1)
        stiff = np.ones_like(s) * k1

        # buckling for compression
        if ds0 is not None:
            buckling = s < 0
            stiff[buckling] = k1 * np.exp(s[buckling] / ds0)

        # and exponential stretch for overstretching fibers
        if ds1 is not None and s1 is not None:
            stretching = s > s1
            stiff[stretching] = k1 * np.exp((s[stretching] - s1) / ds1)

        return stiff

    return sampleAndIntegrateFunction(stiffness, min, max, step)


def linearMaterial(k1, min=-1, max=4.0, step=0.000001):
    """
    A material that has a linear stiffness.

    Parameters
    ----------
    k1 : float
        The stiffness of the material.
    min : float, optional
        Where to start to sample the function.
    max : float, optional
        Where to stop to sample the function.
    step : float, optional
        In which steps to sample the function.

    Returns
    -------
    func : function
        A function which returns, the stiffness, force, and energy for a given stretching.
    """

    def stiffness(s):
        # the linear spring regime (0 < s < s1)
        stiff = np.ones_like(s) * k1

        return stiff

    return sampleAndIntegrateFunction(stiffness, min, max, step)
