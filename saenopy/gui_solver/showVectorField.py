import numpy as np
import pyvista as pv


def showVectorField(plotter, obj, field, name, center=None, show_nan=True, show_all_points=False, factor=.1, scalebar_max=None):
    #try:
    #    field = getattr(obj, name)
    #except AttributeError:
    #    field = obj.getNodeVar(name)
    nan_values = np.isnan(field[:, 0])

    #plotter.clear()

    point_cloud = pv.PolyData(obj.R)
    point_cloud.point_data[name] = field
    point_cloud.point_data[name + "_mag"] = np.linalg.norm(field, axis=1)
    point_cloud.point_data[name + "_mag2"] = point_cloud.point_data[name + "_mag"].copy()
    point_cloud.point_data[name + "_mag2"][nan_values] = 0
    if getattr(plotter, "nan_actor", None) is not None:
        plotter.remove_actor(plotter.nan_actor)
    if show_all_points:
        plotter.add_mesh(point_cloud, colormap="turbo", scalars=name + "_mag2")
    elif show_nan:
        R = obj.R[nan_values]
        if R.shape[0]:
            point_cloud2 = pv.PolyData(R)
            point_cloud2.point_data["nan"] = obj.R[nan_values, 0] * np.nan
            plotter.nan_actor = plotter.add_mesh(point_cloud2, colormap="turbo", scalars="nan", show_scalar_bar=False)

    norm_stack_size = np.abs(np.max(obj.R) - np.min(obj.R))
    if scalebar_max is None:
        factor = factor * norm_stack_size / np.nanmax(point_cloud[name + "_mag2"])#np.nanpercentile(point_cloud[name + "_mag2"], 99.9)
    else:
        factor = factor * norm_stack_size / scalebar_max

    # generate the arrows
    arrows = point_cloud.glyph(orient=name, scale=name + "_mag2", factor=factor)

    title = name
    if name == "U_measured" or name == "U_target" or name == "U":
        title = "Deformations (m)"
    elif name == "f":
        title = "Forces (N)"

    sargs = dict(#position_x=0.05, position_y=0.95,
                 title_font_size=15,
                 label_font_size=9,
                 n_labels=3,
                 title=title,
                 #italic=True,  ##height=0.25, #vertical=True,
                 fmt="%.1e",
                 color=plotter._theme.font.color,
                 font_family="arial")
    plotter.add_mesh(arrows, scalar_bar_args=sargs, colormap="turbo", name="arrows")

    plotter.auto_value = np.nanpercentile(point_cloud[name + "_mag2"], 99.9)
    if scalebar_max is None:
        plotter.update_scalar_bar_range([0, np.nanpercentile(point_cloud[name + "_mag2"], 99.9)])
    else:
        plotter.update_scalar_bar_range([0, scalebar_max])

    if getattr(plotter, "center_actor", None) is not None:
        plotter.remove_actor(plotter.center_actor)
    if center is not None:
        # plot center points if desired
        plotter.center_actor = plotter.add_points(np.array([center]), color='m', point_size=10)

    plotter.show_grid(color=plotter._theme.font.color)
    #plotter.renderer.show_bounds(color=plotter._theme.font.color)
    plotter.show()
