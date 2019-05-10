import numpy as np


def saveEpsilon(epsilon, fname, CFG):
    """
    Save the material function to a file, e.g. for further plotting.
    """
    imax = np.ceil((CFG["EPSMAX"] + 1.0) / CFG["EPSSTEP"])

    with open(fname) as fp:
        for i in range(imax):
            lambd = (i * CFG["EPSSTEP"]) - 1.0

            fp.write(str(lambd) + " " + str(epsilon[i]) + "\n")


def buildEpsilon(k1, ds0, s1, ds1, CFG):
    """
    Build a lookup table for the material function.
    """
    print("EPSILON PARAMETERS", k1, ds0, s1, ds1)

    # calculate the number of steps (+1.0 because EPSMIN is defined as -1)
    # TODO define EPSMIN in the config?
    # epsilon cannot be smaller than -1 because for lambda = -1 the fiber already has 0 length
    imax = int(np.ceil((CFG["EPSMAX"] + 1.0) / CFG["EPSSTEP"]))

    # initialize the lists
    epsilon = np.zeros(imax)
    epsbar = np.zeros(imax)
    epsbarbar = np.zeros(imax)

    # iterate over all steps
    for i in range(imax):
        # calculate the current argument (-1 because we start with EPSMIN = -1)
        lambd = (i * CFG["EPSSTEP"]) - 1.0

        epsbarbar[i] = 0

        # define epsbarbar according to p 43 eq 4.1.5
        if lambd > 0:
            # constant
            epsbarbar[i] += k1

            # of larger than s1, it grows exponentially
            if lambd > s1 and ds1 > 0.0:
                epsbarbar[i] += k1 * (np.exp((lambd - s1) / ds1) - 1.0)
        else:
            # for negative values it decays exponentially
            if ds0 != 0.0:
                epsbarbar[i] += k1 * np.exp(lambd / ds0)
            else:
                epsbarbar[i] += k1

        # savety to prevent overflow
        if epsbarbar[i] > 10e10:
            epsbarbar[i] = 10e10

    # epsbar is the integral over epsbarbar
    # TODO perhaps we can do this with cumsum
    sum = 0.0
    for i in range(imax):
        sum += epsbarbar[i] * CFG["EPSSTEP"]
        epsbar[i] = sum

    # define the offset so that for lambda = 0 epsbar = 0
    imid = int(np.ceil(1.0 / CFG["EPSSTEP"]))
    off = epsbar[imid]
    for i in range(imax):
        epsbar[i] -= off

    # epsilon is the integral over epsbar
    # TODO simplify with cumsum?
    sum = 0.0
    for i in range(imax):
        sum += epsbar[i] * CFG["EPSSTEP"]
        epsilon[i] = sum

    # define the offset so that for lambda = 0 epsilon = 0
    off = epsilon[imid]
    for i in range(imax):
        epsilon[i] -= off

    return epsilon, epsbar, epsbarbar