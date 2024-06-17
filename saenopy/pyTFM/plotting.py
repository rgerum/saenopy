from contextlib import suppress

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm


def plot_continuous_boundary_stresses(
    plot_values,
    mask_boundaries=None,
    plot_t_vecs=False,
    plot_n_arrows=False,
    figsize=(10, 7),
    scale_ratio=0.2,
    border_arrow_filter=1,
    cbar_str="line tension in N/m",
    vmin=None,
    vmax=None,
    cbar_width="2%",
    cbar_height="50%",
    cbar_axes_fraction=0.2,
    # cbar_tick_label_size=20,
    background_color="white",
    cbar_borderpad=0.1,
    linewidth=4,
    cmap="jet",
    plot_cbar=True,
    cbar_style="clickpoints",
    boundary_resolution=3,
    cbar_title_pad=1,
    outer_cb_color="grey",
    outer_cb_style="-",
):
    """
    plotting the line stresses (total transmitted force of cell boundaries), colored by their absolute values
    as continuous lines.
    """

    if not isinstance(plot_values[0], list):
        plot_values = [plot_values]

    min_v = np.min([pv[3] for pv in plot_values])  # minimum over all objects
    max_v = np.max([pv[4] for pv in plot_values])  # maximum over all objects
    shape = plot_values[0][0]  # image shape, should be the same for all objects
    print("plotting cell border stresses")
    min_v = vmin if isinstance(vmin, (float, int)) else min_v
    max_v = vmax if isinstance(vmax, (float, int)) else max_v
    mask_boundaries = (
        np.zeros(shape)
        if not isinstance(mask_boundaries, np.ndarray)
        else mask_boundaries
    )

    fig = plt.figure(figsize=figsize)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    background_color = (
        plt.get_cmap(cmap)(0)
        if background_color == "cmap_0"
        else background_color
    )
    fig.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    ax.set_axis_off()
    fig.add_axes(ax)

    scale = 1
    for shape, edge_lines, lines_interpol, *rest in plot_values:
        all_t_vecs = np.vstack(
            [subdict["t_vecs"] for subdict in lines_interpol.values()]
        )
        if plot_t_vecs:
            scale = scale_for_quiver(
                all_t_vecs[:, 0],
                all_t_vecs[:, 1],
                dims=mask_boundaries.shape,
                scale_ratio=scale_ratio,
                return_scale=True,
            )
        for line_id, interp in tqdm(
            lines_interpol.items(), total=len(lines_interpol.values())
        ):
            p_new = interp["points_new"]
            x_new = p_new[:, 0]
            y_new = p_new[:, 1]
            t_norm = interp["t_norm"]
            t_vecs = interp["t_vecs"]
            n_vecs = interp["n_vecs"]
            # plotting line segments
            c = plt.get_cmap(cmap)(
                (t_norm - min_v) / (max_v - min_v)
            )  # normalization and creating a color range
     
                
            # see how well that works
            if line_id in edge_lines:  # plot lines at the edge
                plt.plot(
                    x_new,
                    y_new,
                    outer_cb_style,
                    color=outer_cb_color,
                    linewidth=linewidth,
                )
            else:
                for i in range(
                    0, len(x_new) - boundary_resolution, boundary_resolution
                ):
                    plt.plot(
                        [x_new[i], x_new[i + boundary_resolution]],
                        [y_new[i], y_new[i + boundary_resolution]],
                        color=c[i],
                        linewidth=linewidth,
                    )

            # plotting stress vectors
            if plot_t_vecs:
                t_vecs_scale = t_vecs * scale
                for i, (xn, yn, t) in enumerate(zip(x_new, y_new, t_vecs_scale)):
                    if i % border_arrow_filter == 0:
                        plt.arrow(xn, yn, t[0], t[1], head_width=0.5)
            # plotting normal vectors
            if plot_n_arrows:
                for i in range(len(x_new) - 1):
                    if i % border_arrow_filter == 0:
                        plt.arrow(
                            x_new[i],
                            y_new[i],
                            n_vecs[i][0],
                            n_vecs[i][1],
                            head_width=0.5,
                        )

    plt.gca().invert_yaxis()  # to get the typicall imshow orientation
    plt.xlim(0, shape[1])
    plt.ylim(shape[0], 0)
    # background_color=matplotlib.cm.get_cmap(cmap)(0) if background_color=="cmap_0" else background_color
    # ax.set_facecolor(background_color)
    if plot_cbar:
        add_colorbar(
            min_v,
            max_v,
            cmap,
            ax=ax,
            cbar_style=cbar_style,
            cbar_width=cbar_width,
            cbar_height=cbar_height,
            cbar_borderpad=cbar_borderpad,
            # v=cbar_tick_label_size,
            cbar_str=cbar_str,
            cbar_axes_fraction=cbar_axes_fraction,
            cbar_title_pad=cbar_title_pad,
        )
    return fig, ax


def add_colorbar(
    vmin,
    vmax,
    cmap="rainbow",
    ax=None,
    cbar_style="not-clickpoints",
    cbar_width="2%",
    cbar_height="50%",
    cbar_borderpad=0.1,
    cbar_tick_label_size=15,
    cbar_str="",
    cbar_axes_fraction=0.2,
    shrink=0.8,
    aspect=20,
    cbar_title_pad=1,
):
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=norm)
    
    
    sm.set_array([])  # bug fix for lower matplotlib version
    if cbar_style == "clickpoints":  # colorbar inside of the plot
        cbaxes = inset_axes(
            ax,
            width=cbar_width,
            height=cbar_height,
            loc=5,
            borderpad=cbar_borderpad * 30,
        )
        cb0 = plt.colorbar(sm, cax=cbaxes)
        with suppress(TypeError, AttributeError):
            cbaxes.set_title(cbar_str, color="white", pad=cbar_title_pad)
        cbaxes.tick_params(colors="white", labelsize=cbar_tick_label_size)
    else:  # colorbar outide of the plot
        cb0 = plt.colorbar(
            sm,
            aspect=aspect,
            shrink=shrink,
            fraction=cbar_axes_fraction,
            pad=cbar_borderpad,
            ax=plt.gca(),
        )  # just exploiting the axis generation by a plt.colorbar
        cb0.outline.set_visible(False)
        cb0.ax.tick_params(labelsize=cbar_tick_label_size)
        with suppress(TypeError, AttributeError):
            cb0.ax.set_title(cbar_str, color="black", pad=cbar_title_pad)
    return cb0


def show_quiver(
    fx,
    fy,
    filter=None,
    scale_ratio=0.2,
    headwidth=None,
    headlength=None,
    headaxislength=None,
    width=None,
    cmap="rainbow",
    figsize=None,
    cbar_str="",
    ax=None,
    fig=None,
    vmin=None,
    vmax=None,
    cbar_axes_fraction=0.2,
    # cbar_tick_label_size=15,
    cbar_width="2%",
    cbar_height="50%",
    cbar_borderpad=0.1,
    cbar_style="not-clickpoints",
    plot_style="not-clickpoints",
    cbar_title_pad=1,
    plot_cbar=True,
    alpha=1,
    ax_origin="upper",
    filter_method="regular",
    filter_radius=5,
):
    # list of all necessary quiver parameters
    if filter is None:
        filter = [0, 1]
    quiver_parameters = {
        "headwidth": headwidth,
        "headlength": headlength,
        "headaxislength": headaxislength,
        "width": width,
        "scale_units": "xy",
        "angles": "xy",
        "scale": None,
    }
    quiver_parameters = {
        key: value for key, value in quiver_parameters.items() if value is not None
    }

    fx = fx.astype("float64")
    fy = fy.astype("float64")
    dims = fx.shape  # needed for scaling
    if not isinstance(ax, matplotlib.axes.Axes):
        fig = plt.figure(figsize=figsize)
        ax = plt.axes()
    map_values = np.sqrt(fx**2 + fy**2)
    vmin, vmax = set_vmin_vmax(map_values, vmin, vmax)
    plt.imshow(
        map_values, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, origin=ax_origin
    )
    if plot_style == "clickpoints":
        ax.set_position([0, 0, 1, 1])
    ax.set_axis_off()
    # plotting arrows
    # filtering every n-th value and every value smaller then x
    fx, fy, xs, ys = filter_values(
        fx,
        fy,
        abs_filter=filter[0],
        f_dist=filter[1],
        filter_method=filter_method,
        radius=filter_radius,
    )
    if scale_ratio:  # optional custom scaling with the image axis lenght
        fx, fy = scale_for_quiver(fx, fy, dims=dims, scale_ratio=scale_ratio)
        quiver_parameters["scale"] = 1  # disabeling the auto-scaling behavior of quiver
    plt.quiver(xs, ys, fx, fy, **quiver_parameters)  # plotting the arrows
    if plot_cbar:
        add_colorbar(
            vmin,
            vmax,
            cmap,
            ax=ax,
            cbar_style=cbar_style,
            cbar_width=cbar_width,
            cbar_height=cbar_height,
            cbar_borderpad=cbar_borderpad,
            # v=cbar_tick_label_size,
            cbar_str=cbar_str,
            cbar_axes_fraction=cbar_axes_fraction,
            cbar_title_pad=cbar_title_pad,
        )
    return fig, ax


def set_vmin_vmax(x, vmin, vmax):
    if not isinstance(vmin, (float, int)):
        vmin = np.nanmin(x)
    if not isinstance(vmax, (float, int)):
        vmax = np.nanmax(x)
    if isinstance(vmax, (float, int)) and not isinstance(vmin, (float, int)):
        vmin = vmax - 1 if vmin > vmax else None
    return vmin, vmax


def find_maxima(ar1, ar2, radius=5, shape="circle"):
    # generating circle

    ys, xs = np.indices((radius * 2 + 1, radius * 2 + 1))
    xs = (xs - radius).astype(float)
    ys = (ys - radius).astype(float)
    if shape == "circle":
        out = np.sqrt(xs**2 + ys**2) <= radius
        xs[~out] = np.nan
        ys[~out] = np.nan
    vector_abs = np.sqrt(ar1**2 + ar2**2)
    lmax = np.unravel_index(np.nanargmax(vector_abs), shape=vector_abs.shape)
    maxis = [lmax]
    while True:
        x_exclude = (lmax[1] + xs).flatten()
        y_exclude = (lmax[0] + ys).flatten()
        outside_image = (
            (x_exclude >= vector_abs.shape[1])
            | (x_exclude < 0)
            | (y_exclude >= vector_abs.shape[0])
            | (y_exclude < 0)
            | (np.isnan(x_exclude))
            | (np.isnan(y_exclude))
        )
        x_exclude = x_exclude[~outside_image]
        y_exclude = y_exclude[~outside_image]
        vector_abs[y_exclude.astype(int), x_exclude.astype(int)] = np.nan
        try:
            lmax = np.unravel_index(np.nanargmax(vector_abs), shape=vector_abs.shape)
        except ValueError:
            break
        maxis.append(lmax)

    maxis_y = [i[0] for i in maxis]
    maxis_x = [i[1] for i in maxis]
    return maxis_y, maxis_x


def filter_values(ar1, ar2, abs_filter=0, f_dist=3, filter_method="regular", radius=5):
    """
    function to filter out values from an array for better display
    """

    if filter_method == "regular":
        pixx = np.arange(np.shape(ar1)[0])
        pixy = np.arange(np.shape(ar1)[1])
        xv, yv = np.meshgrid(pixy, pixx)

        def_abs = np.sqrt((ar1**2 + ar2**2))
        select_x = ((xv - 1) % f_dist) == 0
        select_y = ((yv - 1) % f_dist) == 0
        select_size = def_abs > abs_filter
        select = select_x * select_y * select_size
        s1 = ar1[select]
        s2 = ar2[select]
        x_ind = xv[select]
        y_ind = yv[select]
    elif filter_method == "local_maxima":
        y_ind, x_ind = find_maxima(ar1, ar2, radius=radius, shape="circle")
        s1 = ar1[y_ind, x_ind]
        s2 = ar2[y_ind, x_ind]
    elif filter_method == "local_maxima_square":
        y_ind, x_ind = find_maxima(ar1, ar2, radius=radius, shape="square")
        s1 = ar1[y_ind, x_ind]
        s2 = ar2[y_ind, x_ind]
    else:
        raise ValueError("Filter method unknown", filter_method)
    return s1, s2, x_ind, y_ind


def scale_for_quiver(ar1, ar2, dims, scale_ratio=0.2, return_scale=False):
    scale = scale_ratio * np.max(dims) / np.nanmax(np.sqrt(ar1 ** 2 + ar2 ** 2))
    if return_scale:
        return scale
    return ar1 * scale, ar2 * scale
