import numpy as np


def MakeFromPolar(r, theta, phi):
    """
    Convert from polar coordinates to cartesian coordinates
    """
    # get sine and cosine of the angles
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    # and convert to cartesian coordinates
    x = r * st * cp
    y = r * st * sp
    z = r * ct
    # create the array
    return np.array([x, y, z])


def buildBeams(N):
    """
    Builds a sphere of unit vectors with N beams in the xy plane.
    """

    # start with an empty list
    beams = []

    # iterate over the whole angle in the xy plane
    for i in range(N):
        # get the Nth part of the total rotation
        theta = (2 * np.pi / N) * i

        # estimate how many vectors we need to cover the phi angle (for the z direction)
        jmax = int(np.floor(N * np.sin(theta) + 0.5))

        # iterate over those angles to get beams in every direction
        for j in range(jmax):
            # get the phi angle
            phi = (2 * np.pi / jmax) * j

            # and create a unit vector from the polar coordinates theta and phi
            beams.append(MakeFromPolar(1.0, theta, phi))

    # return all the vectors
    return np.array(beams)


def saveBeams(beams, fname):
    np.savetxt(fname, beams)
