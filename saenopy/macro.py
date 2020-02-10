import numpy as np
from .buildBeams import buildBeams
from .materials import Material
from typing import Sequence


def getQuadrature(N: int, xmin: float, xmax: float) -> (np.ndarray, np.ndarray):
    """
    Provides N quadrature points for an integration from xmin to xmax together with their weights.

    Parameters
    ----------
    N : int
        The number of quadrature points to use. Has to be 1 <= N <= 5.
    xmin : float
        The start of the integration range
    xmax : float
        The end of the integration range

    Returns
    -------
    points : np.ndarray
        The points of the quadrature
    w : np.ndarray
        The weights of the points
    """
    if N < 1:
        raise ValueError()

    if N == 1:
        points = [0]
        w = [2]

    if N == 2:
        points = [-np.sqrt(1 / 3), np.sqrt(1 / 3)]
        w = [1, 1]

    if N == 3:
        points = [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)]
        w = [5 / 9, 8 / 9, 5 / 9]

    if N == 4:
        points = [-np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)), +np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)),
                  -np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5)), +np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5))]
        w = [(18 + np.sqrt(30)) / 36, (18 + np.sqrt(30)) / 36, (18 - np.sqrt(30)) / 36, (18 - np.sqrt(30)) / 36]

    if N == 5:
        points = [0,
                  -1 / 3 * np.sqrt(5 - 2 * np.sqrt(10 / 7)), +1 / 3 * np.sqrt(5 - 2 * np.sqrt(10 / 7)),
                  -1 / 3 * np.sqrt(5 + 2 * np.sqrt(10 / 7)), +1 / 3 * np.sqrt(5 + 2 * np.sqrt(10 / 7))]
        w = [128 / 225, (322 + 13 * np.sqrt(70)) / 900, (322 + 13 * np.sqrt(70)) / 900, (322 - 13 * np.sqrt(70)) / 900,
             (322 - 13 * np.sqrt(70)) / 900]

    if N > 5:
        raise ValueError()

    points = np.array(points)
    w = np.array(w)
    factor = (xmax - xmin) / 2
    points = factor * points + (xmax + xmin) / 2
    w = w * factor
    return points, w


def combineQuadrature(p1_w1: Sequence, p2_w2: Sequence) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Combine the quadratures of two different axes.

    Parameters
    ----------
    p1_w1 : tuple
        the points and weights for the first axis
    p2_w2 : tuple
        the points and weights for the second axis

    Returns
    -------
    x : np.ndarray
        the points for the first axis
    y : np.ndarray
        the points for the second axis
    w : np.ndarray
        the combined weights for the points
    """
    p1, w1 = p1_w1
    p2, w2 = p2_w2
    x, y = [f.ravel() for f in np.meshgrid(p1, p2)]
    w = (w1[:, None] * w2[None, :]).ravel()
    return x, y, w


def getShearRheometerStress(gamma: np.ndarray, material: Material, s: np.ndarray = None) -> (np.ndarray, np.ndarray):
    r"""
    Get the stress for a given strain of the material in a shear rheometer.

    The following shear deformation :math:`\mathbf{F}` is applied to the material:

    .. math::
        \mathbf{F}(\gamma) =
        \begin{pmatrix}
            1 & \gamma & 0 \\
            0 & 1 & 0 \\
            0 & 0 & 1 \\
        \end{pmatrix}

    and the resulting stress is obtained by calculating numerically the derivative of the energy density :math:`W` with
    respect to the strain :math:`\gamma`:

    .. math::
        \sigma(\gamma) = \frac{dW(\mathbf{F}(\gamma))}{d\gamma}

    Parameters
    ----------
    gamma : ndarray
        The applied strain.
    material : :py:class:`~.materials.Material`
        The material model to use.

    Returns
    -------
    strain : ndarray
        The strain values.
    stress : ndarray
        The resulting stress.
    """
    if s is None:
        s = buildBeams(30)

    F = np.eye(3)
    F = np.tile(F, (gamma.shape[0], 1, 1))
    F[:, 0, 1] = np.tan(gamma)

    s_bar = F @ s.T

    s_abs = np.linalg.norm(s_bar, axis=-2)

    eps = material.energy(s_abs - 1)

    W = np.mean(eps, axis=-1)
    dW = np.diff(W) / np.diff(gamma)
    return gamma[:-1] + np.diff(gamma) / 2, dW


def getShearRheometerStressRotation(gamma, material, H=1e-3, R=10e-3, s=30, q=2):
    if isinstance(s, int):
        s = buildBeams(s)

    x_r, z_h, w = combineQuadrature(getQuadrature(q, 0, 1), getQuadrature(q, 0, 1))

    F = np.zeros((gamma.shape[0], len(z_h), 3, 3))
    theta = gamma * H / R
    theta_p = theta[:, None] * z_h[None, :]

    cos, sin = np.cos(theta_p), np.sin(theta_p)
    xtheta_h = x_r * theta[:, None] * R / H
    F[:, :, 0, 0], F[:, :, 0, 1], F[:, :, 0, 2] = cos, -sin, -sin * xtheta_h
    F[:, :, 1, 0], F[:, :, 1, 1], F[:, :, 1, 2] = sin, cos, cos * xtheta_h
    F[:, :, 2, 2] = 1

    s_bar = F @ s.T

    s_abs = np.linalg.norm(s_bar, axis=-2)
    eps = material.energy(s_abs - 1)

    W = np.mean(eps, axis=-1)
    W = np.average(W, axis=-1, weights=w)
    dW = np.diff(W) / np.diff(gamma)

    return gamma[:-1] + np.diff(gamma) / 2, dW


def getStretchThinning(lambda_h, lambda_v, material, s=None):
    r"""
    Get the thinning of the material for streching.

    The following deformation :math:`\mathbf{F}` is applied to the material, composed of a horizontal and a vertical
    stretching:

    .. math::
        \mathbf{F}(\gamma) =
        \begin{pmatrix}
            \lambda_h & 0 & 0 \\
            0 & 1 & 0 \\
            0 & 0 & \lambda_v \\
        \end{pmatrix}

    the resulting energy density :math:`W(\mathbf{F}(\lambda_h,\lambda_v))` is then minimized numerically for every
    :math:`\lambda_h` to obtain the :math:`\lambda_v` that results in the lowest energy of the system.

    Parameters
    ----------
    lambda_h : ndarray
        The applied stretching in horizontal direction.
    lambda_v : ndarray
        The different values for thinning to test. The value with the lowest energy for each horizontal stretch is
        returned.
    material : :py:class:`~.materials.Material`
        The material model to use.

    Returns
    -------
    lambda_h : ndarray
        The horizontal stretching values.
    lambda_v : ndarray
        The vertical stretching that minimizes the energy for the horizontal stretching.
    """
    if s is None:
        s = buildBeams(30)

    F00, F22 = np.meshgrid(lambda_v, lambda_h)
    F11 = np.ones_like(F00)
    F = np.dstack((F00, F11, F22))

    s_bar = np.einsum("hvj,bj->hvjb", F, s)
    s_abs = np.linalg.norm(s_bar, axis=-2)
    eps = material.energy(s_abs - 1)
    W = np.mean(eps, axis=-1)

    index = np.argmin(W, axis=1)
    return lambda_h, lambda_v[index]


def getExtensionalRheometerStress(epsilon, material, s=None):
    r"""
    Get the stress for a given strain of the material in an extensional rheometer.

    The following deformation :math:`\mathbf{F}` is applied to the material:

    .. math::
        \mathbf{F}(\gamma) =
        \begin{pmatrix}
            \epsilon & 0 & 0 \\
            0 & 1 & 0 \\
            0 & 0 & 1 \\
        \end{pmatrix}

    and the resulting stress is obtained by calculating numerically the derivative of the energy density :math:`W` with
    respect to the strain :math:`\epsilon`:

    .. math::
        \sigma(\gamma) = \frac{dW(\mathbf{F}(\gamma))}{d\epsilon}


    Parameters
    ----------
    epsilon : ndarray
        The applied strain.
    material : :py:class:`~.materials.Material`
        The material model to use.

    Returns
    -------
    strain : ndarray
        The strain values.
    stress : ndarray
        The resulting stress.
    """
    if s is None:
        s = buildBeams(30)

    F = np.eye(3)
    F = np.tile(F, (epsilon.shape[0], 1, 1))
    F[:, 0, 0] = epsilon

    s_bar = F @ s.T

    s_abs = np.linalg.norm(s_bar, axis=-2)

    eps = material.energy(s_abs - 1)

    W = np.mean(eps, axis=-1)
    dW = np.diff(W) / np.diff(epsilon)
    return epsilon[:-1] + np.diff(epsilon) / 2, dW
