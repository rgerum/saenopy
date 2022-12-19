import imageio
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from saenopy.solver import Solver
import numpy as np


def showVectorField(plotter: QtInteractor, obj: Solver, field: np.ndarray, name: str, center=None, show_nan=True,
                    show_all_points=False, factor=.1, scalebar_max=None, display_image=None):
    # force rendering to be disabled while updating content to prevent flickering
    render = plotter.render
    plotter.render = lambda *args: None
    try:
        # get positions of nan values
        nan_values = np.isnan(field[:, 0])

        # create a point cloud
        point_cloud = pv.PolyData(obj.R)
        point_cloud.point_data[name] = field
        point_cloud.point_data[name + "_mag"] = np.linalg.norm(field, axis=1)
        point_cloud.point_data[name + "_mag2"] = point_cloud.point_data[name + "_mag"].copy()
        point_cloud.point_data[name + "_mag2"][nan_values] = 0

        if not show_all_points and show_nan:
            R = obj.R[nan_values]
            if R.shape[0]:
                point_cloud2 = pv.PolyData(R)
                point_cloud2.point_data["nan"] = obj.R[nan_values, 0] * np.nan

        # remove a previous nan_actor if present
        if getattr(plotter, "nan_actor", None) is not None:
            plotter.remove_actor(plotter.nan_actor, render=False)

        # remove a previous center_actor if present
        if getattr(plotter, "center_actor", None) is not None:
            plotter.remove_actor(plotter.center_actor, render=False)

        plotter.renderer.remove_bounds_axes()

        # scalebar scaling factor
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

        # show the points
        if show_all_points:
            plotter.add_mesh(point_cloud, colormap="turbo", scalars=name + "_mag2", render=False)
        elif show_nan:
            if R.shape[0]:
                plotter.nan_actor = plotter.add_mesh(point_cloud2, colormap="turbo", scalars="nan",
                                                     show_scalar_bar=False, render=False)

        # add the arrows
        plotter.add_mesh(arrows, scalar_bar_args=sargs, colormap="turbo", name="arrows", render=False)

        # update the scalebar
        plotter.auto_value = np.nanpercentile(point_cloud[name + "_mag2"], 99.9)
        if scalebar_max is None:
            plotter.update_scalar_bar_range([0, np.nanpercentile(point_cloud[name + "_mag2"], 99.9)])
        else:
            plotter.update_scalar_bar_range([0, scalebar_max])

        # plot center points if desired
        if center is not None:
            plotter.center_actor = plotter.add_points(np.array([center]), color='m', point_size=10, render=False)

        if getattr(plotter, "_image_mesh", None) is not None:
            plotter.remove_actor(plotter._image_mesh, render=False)
        if display_image is not None:
            img, voxel_size, z_pos = display_image
            xmin = (-img.shape[1]/2)*voxel_size[0]*1e-6
            ymin = (-img.shape[0]/2)*voxel_size[1]*1e-6
            x = np.linspace(xmin, -xmin, 10)
            y = np.linspace(ymin, -ymin, 10)
            x, y = np.meshgrid(x, y)
            z = z_pos*voxel_size[2]*1e-6+0*x

            curvsurf = pv.StructuredGrid(x, y, z)

            # Map the curved surface to a plane - use best fitting plane
            curvsurf.texture_map_to_plane(inplace=True)

            tex = pv.numpy_to_texture(img)
            mesh = plotter.add_mesh(curvsurf, texture=tex)
            plotter._image_mesh = mesh

        xmin, ymin, zmin = obj.R.min(axis=0)
        xmax, ymax, zmax = obj.R.max(axis=0)
        plotter.show_grid(bounds=[xmin, xmax, ymin, ymax, zmin, zmax], color=plotter._theme.font.color, render=False)
    finally:
        plotter.render = render
        plotter.render()
