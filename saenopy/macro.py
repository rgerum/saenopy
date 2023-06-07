import numpy as np
from .build_beams import build_beams
from .materials import Material
from typing import Sequence
from scipy.interpolate import interp1d


def get_shear_rheometer_stress(gamma: np.ndarray, material: Material, s: np.ndarray = None) -> (np.ndarray, np.ndarray):
    r"""
    This function returns the stress the material model generates when subjected to a shear strain,
    as seen in a shear rheometer.

    The shear strain is described using the following deformation gradient :math:`\mathbf{F}`:

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
        The strain values for which to calculate the stress.
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
        s = build_beams(30)

    F = np.eye(3)
    F = np.tile(F, (gamma.shape[0], 1, 1))
    F[:, 0, 1] = gamma

    s_bar = F @ s.T

    s_abs = np.linalg.norm(s_bar, axis=-2)

    eps = material.energy(s_abs - 1)

    W = np.mean(eps, axis=-1)
    dW = np.diff(W) / np.diff(gamma)
    return gamma[:-1] + np.diff(gamma) / 2, dW


def get_stretch_thinning(gamma_h, gamma_v, material, s=None):
    r"""
    This function returns the vertical thinning (strain in z direction) of the material model
    when the material model is stretched horizontally (strain in x direction), as seen in a stretcher device.

    The strain in x and z direction is described using the following deformation gradient :math:`\mathbf{F}`:

    .. math::
        \mathbf{F}(\gamma) =
        \begin{pmatrix}
            \gamma_h & 0 & 0 \\
            0 & 1 & 0 \\
            0 & 0 & \gamma_v \\
        \end{pmatrix}

    the resulting energy density :math:`W(\mathbf{F}(\gamma_h,\gamma_v))` is then minimized numerically for every
    :math:`\gamma_h` to obtain the :math:`\gamma_v` that results in the lowest energy of the system.

    Parameters
    ----------
    gamma_h : ndarray
        The applied strain in horizontal direction.
    gamma_v : ndarray
        The different values for thinning to test. The value with the lowest energy for each horizontal strain is
        returned.
    material : :py:class:`~.materials.Material`
        The material model to use.

    Returns
    -------
    gamma_h : ndarray
        The horizontal strain values.
    gamma_v : ndarray
        The vertical strain that minimizes the energy for the horizontal strain.
    """
    if s is None:
        s = build_beams(30)

    F00, F22 = np.meshgrid(gamma_v, gamma_h)
    F11 = np.ones_like(F00)
    F = np.dstack((F00, F11, F22))

    s_bar = np.einsum("hvj,bj->hvjb", F, s)
    s_abs = np.linalg.norm(s_bar, axis=-2)
    eps = material.energy(s_abs - 1)
    W = np.mean(eps, axis=-1)

    index = np.argmin(W, axis=1)
    return gamma_h, gamma_v[index]


def get_extensional_rheometer_stress(gamma, material, s=None):
    r"""
    This function returns the stress the material model generates when subjected to an extensional strain,
    as seen in an extensional rheometer.

    The extensional strain is described using the following deformation gradient :math:`\mathbf{F}`:

    .. math::
        \mathbf{F}(\gamma) =
        \begin{pmatrix}
            \gamma & 0 & 0 \\
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
        The strain values for which to calculate the stress.
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
        s = build_beams(30)

    F = np.eye(3)
    F = np.tile(F, (gamma.shape[0], 1, 1))
    F[:, 0, 0] = gamma

    s_bar = F @ s.T

    s_abs = np.linalg.norm(s_bar, axis=-2)

    eps = material.energy(s_abs - 1)

    W = np.mean(eps, axis=-1)
    dW = np.diff(W) / np.diff(gamma)
    return gamma[:-1] + np.diff(gamma) / 2, dW


import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from saenopy.materials import SemiAffineFiberMaterial


def get_mapping(p, func, indices):
    m = func(p)
    mapping = []
    for i in range(len(p)):
        pp = p.copy()
        pp[i] = pp[i]+1
        mm = func(pp)
        for i in indices:
            if mm[i] != m[i]:
                mapping.append(True)
                break
        else:
            mapping.append(False)
    return mapping


def minimize(cost_data: list, parameter_start: Sequence, method='Powell', maxfev:int = 1e4, MaterialClass=SemiAffineFiberMaterial, x_sample=20, colors=None, **kwargs):
    parameter_start = np.array(parameter_start)

    costs_shear = []
    mapping_shear = np.array([False] * len(parameter_start))
    plots_shear = []
    costs_stretch = []
    mapping_stretch = np.array([False] * len(parameter_start))
    plots_stretch = []

    index = 0
    for func, data, params in cost_data:
        color = None
        if colors is not None:
            color = colors[index]
            index += 1
        if func == get_stretch_thinning:
            mapping_stretch |= get_mapping(parameter_start, params, [1])

            def getCost(func, data, params):
                stretchx = data[:, 0]
                stretchy = data[:, 1]

                ###lambda_h = np.arange(1 - 0.05, 1 + 0.07, 0.01)
                lambda_h = np.linspace(np.min(stretchx), np.max(stretchx), x_sample) ## fit complete input data regime
                lambda_v = np.arange(0, 1.1, 0.001)

                def cost(p):
                    nonlocal parameter_start
                    parameter_start = parameter_start.copy()
                    parameter_start[mapping_stretch] = p
                    p = params(parameter_start)
                    material1 = MaterialClass(*p)
                    x, y = get_stretch_thinning(lambda_h, lambda_v, material1)
                    stretchy2 = interp1d(x, y, fill_value=np.nan, bounds_error=False)(stretchx)
                    cost = np.nansum((stretchy2 - stretchy) ** 2)
                    return cost

                def plot_me(color=color):
                    material = MaterialClass(*params(parameter_start))
                    plt.plot(stretchx, stretchy, "o", label="data", color=color)

                    x, y = get_stretch_thinning(lambda_h, lambda_v, material)
                    plt.plot(x, y, "r-", lw=3, label="model")
                return cost, plot_me
            cost, plot = getCost(func, data, params)
            costs_stretch.append(cost)
            plots_stretch.append(plot)

        if func == get_extensional_rheometer_stress:
            mapping_shear |= get_mapping(parameter_start, params, [0, 2, 3])

            def get_cost(func, data, params):
                shearx = data[:, 0]
                sheary = data[:, 1]

                x0 = shearx
                dx = x0[1] - x0[0]
                weights = np.diff(np.log(x0), append=np.log(
                    x0[-1] + dx)) ** 2  # needs to be improved (based on spacing of data points in logarithmic space)
                gamma = np.linspace(np.min(x0), np.max(x0), x_sample)

                def cost(p):
                    nonlocal parameter_start
                    parameter_start = parameter_start.copy()
                    parameter_start[mapping_shear] = p
                    p = params(parameter_start)
                    material1 = MaterialClass(*p)
                    x, y = get_extensional_rheometer_stress(gamma, material1)
                    stretchy2 = interp1d(x, y, fill_value=np.nan, bounds_error=False)(shearx)
                    cost = np.nansum((np.log(stretchy2) - np.log(sheary)) ** 2 * weights)
                    return cost

                def plot_me(color=color):
                    material = MaterialClass(*params(parameter_start))
                    plt.loglog(shearx, sheary, "o", label="data", color=color)

                    x, y = get_extensional_rheometer_stress(gamma, material)
                    plt.loglog(x, y, "r-", lw=3, label="model")

                return cost, plot_me

            cost, plot = get_cost(func, data, params)
            costs_shear.append(cost)
            plots_shear.append(plot)

        if func == get_shear_rheometer_stress:
            mapping_shear |= get_mapping(parameter_start, params, [0, 2, 3])

            def get_cost(func, data, params):
                shearx = data[:, 0]
                sheary = data[:, 1]

                x0 = shearx
                dx = x0[1] - x0[0]
                weights = np.diff(np.log(x0), append=np.log(
                    x0[-1] + dx)) ** 2  # needs to be improved (based on spacing of data points in logarithmic space)
                gamma = np.linspace(np.min(x0), np.max(x0), x_sample)

                def cost(p):
                    nonlocal parameter_start
                    parameter_start = parameter_start.copy()
                    parameter_start[mapping_shear] = p
                    p = params(parameter_start)
                    material1 = MaterialClass(*p)
                    x, y = get_shear_rheometer_stress(gamma, material1)
                    stretchy2 = interp1d(x, np.clip(y, 1e-9, None), fill_value=np.nan, bounds_error=False)(shearx)
                    valid_indices = ~np.isnan(stretchy2)
                    cost = np.nansum((np.log(stretchy2[valid_indices]) - np.log(sheary[valid_indices])) ** 2 * weights[valid_indices])
                    return cost

                def plot_me(color=color):
                    material = MaterialClass(*params(parameter_start))
                    plt.loglog(shearx, sheary, "o", label="data", color=color)

                    x, y = get_shear_rheometer_stress(gamma, material)
                    plt.loglog(x, y, "r-", lw=3, label="model")

                return cost, plot_me

            cost, plot = get_cost(func, data, params)
            costs_shear.append(cost)
            plots_shear.append(plot)

    for i in range(5):
        for mapping, costs in [[mapping_shear, costs_shear], [mapping_stretch, costs_stretch]]:
            if len(costs) == 0:
                continue
            # define the cost function
            def cost(p):
                return sum([c(p) for c in costs])

            # minimize the cost with reasonable start parameters
            from scipy.optimize import minimize
            sol = minimize(cost, parameter_start[mapping], method=method, options={'maxfev': maxfev}, **kwargs)
            parameter_start[mapping] = sol["x"]

        if len(costs_shear) == 0 or len(costs_stretch) == 0:
            break

    def plot_all():
        subplot_count = (len(plots_stretch) > 0) + (len(plots_shear) > 0)
        if len(plots_shear):
            plt.subplot(1, subplot_count, 1)
            for plot in plots_shear:
                plot()
            plt.xlabel("strain")
            plt.ylabel("stress")
        if len(plots_stretch):
            plt.subplot(1, subplot_count, 1+(len(plots_stretch)>0))
            for plot in plots_stretch:
                plot()
            plt.xlabel("horizontal stretch")
            plt.ylabel("vertical contraction")

    return parameter_start, plot_all