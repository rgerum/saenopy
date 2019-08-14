import numpy as np
from saeno.buildBeams import buildBeams


def getShearRheometerStress(gamma, material, s=None):
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
