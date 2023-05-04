import numpy as np


def make_from_polar(r: float, theta: float, phi: float) -> np.ndarray:
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


def build_beams(n: int) -> np.ndarray:
    """
    Builds a sphere of unit vectors with N beams in the xy plane.
    """
    n = int(np.floor(np.sqrt(int(n) * np.pi + 0.5)))

    # start with an empty list
    beams = []

    # iterate over the whole angle in the xy plane
    for i in range(n):
        # get the Nth part of the total rotation
        theta = (2 * np.pi / n) * i

        # estimate how many vectors we need to cover the phi angle (for the z direction)
        j_max = int(np.floor(n * np.sin(theta) + 0.5))

        # iterate over those angles to get beams in every direction
        for j in range(j_max):
            # get the phi angle
            phi = (2 * np.pi / j_max) * j

            # and create a unit vector from the polar coordinates theta and phi
            beams.append(make_from_polar(1.0, theta, phi))

    # return all the vectors
    return np.array(beams)
