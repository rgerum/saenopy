import imageio
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from saenopy.solver import Solver
import numpy as np


def getVectorFieldImage(self):
    image = self.vtk_toolbar.show_image.value()
    if image:
        stack = self.result.stack[self.t_slider.value()]
        im = stack[:, :, :, self.z_slider.value(), self.vtk_toolbar.channel_select.value()]
        if self.vtk_toolbar.button_z_proj.value():
        #if self.result.stack_parameter["z_project_name"] == "maximum":
            z_range = [0, 5, 10, 1000][self.vtk_toolbar.button_z_proj.value()]
            start = np.clip(self.z_slider.value() - z_range, 0,
                            stack.shape[2])
            end = np.clip(self.z_slider.value() + z_range, 0, stack.shape[2])
            im = stack[:, :, :, start:end, self.vtk_toolbar.channel_select.value()]
            im = np.max(im, axis=3)
        else:
            (min, max) = np.percentile(im, (1, 99))
            im = im.astype(np.float32) - min
            im = im.astype(np.float64) * 255 / (max - min)
            im = np.clip(im, 0, 255).astype(np.uint8)

        display_image = [im, stack.voxel_size, self.z_slider.value() - stack.shape[2] / 2]
        if self.vtk_toolbar.show_image.value() == 2:
            display_image[2] = -stack.shape[2] / 2
    else:
        display_image = None
    return display_image

def showVectorField2(self, M, points_name):
    display_image = getVectorFieldImage(self)
    if display_image[0].shape[2] == 1:
        display_image[0] = display_image[0][:, :, 0]

    try:
        field = getattr(M, points_name)
    except AttributeError:
        field = M.getNodeVar(points_name)
    showVectorField(self.plotter, M, field, points_name,
                    scalebar_max=self.vtk_toolbar.getScaleMax(), show_nan=self.vtk_toolbar.use_nans.value(),
                    display_image=display_image, show_grid=self.vtk_toolbar.show_grid.value(),
                    stack_shape=np.array(self.result.stack[0].shape[:3])*np.array(self.result.stack[0].voxel_size))


def showVectorField(plotter: QtInteractor, obj: Solver, field: np.ndarray, name: str, center=None, show_nan=True, stack_shape=None,
                    show_all_points=False, factor=.1, scalebar_max=None, display_image=None, show_grid=True):
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
        plotter.renderer.remove_bounding_box()

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
        if getattr(plotter, "_box", None) is not None:
            plotter.remove_actor(plotter._box, render=False)
        if display_image is not None:
            img, voxel_size, z_pos = display_image  
            # adjust the direction of the underlying image 
            # the combination of following both operations does the job
            img_adjusted = img[:, ::-1]                             # mirror the image
            img_adjusted = np.swapaxes(img_adjusted, 1,0)   # switch axis
            # get coords
            xmin = (-img_adjusted.shape[1]/2)*voxel_size[0]*1e-6
            ymin = (-img_adjusted.shape[0]/2)*voxel_size[1]*1e-6
            x = np.linspace(xmin, -xmin, 10)
            y = np.linspace(ymin, -ymin, 10)
            x, y = np.meshgrid(x, y)
            z = z_pos*voxel_size[2]*1e-6+0*x
            # structureGrid
            curvsurf = pv.StructuredGrid(x, y, z)
            # Map the curved surface to a plane - use best fitting plane
            curvsurf.texture_map_to_plane(inplace=True)   
            tex = pv.numpy_to_texture(img_adjusted)
            # add image below arrow field            
            mesh = plotter.add_mesh(curvsurf, texture=tex)
            plotter._image_mesh = mesh

        xmin, ymin, zmin = obj.R.min(axis=0)
        xmax, ymax, zmax = obj.R.max(axis=0)
        plotter.remove_bounds_axes()

        if show_grid == 2:
            #plotter.show_bounds(bounds=[xmin, xmax, ymin, ymax, zmin, zmax], grid='front', location='outer', all_edges=True,
            #                    show_xlabels=False, show_ylabels=False, show_zlabels=False,
            #                    xlabel=" ", ylabel=" ", zlabel=" ", render=False)
            corners = np.asarray([[xmin, ymin, zmin], [xmax, ymin, zmin], [xmin, ymax, zmin], [xmax, ymax, zmin],
                                   [xmin, ymin, zmax], [xmax, ymin, zmax], [xmin, ymax, zmax], [xmax, ymax, zmax]])
            grid = pv.ExplicitStructuredGrid(np.asarray([2, 2, 2]), corners)
            plotter._box = plotter.add_mesh(grid, style='wireframe', render_lines_as_tubes=True, line_width=2, show_edges=True, name="border")
        elif show_grid == 3:
            xmin, xmax = -stack_shape[0]/2*1e-6, stack_shape[0]/2*1e-6
            print(xmin,ymin)
            ymin, ymax = -stack_shape[1]/2*1e-6, stack_shape[1]/2*1e-6
            zmin, zmax = -stack_shape[2]/2*1e-6, stack_shape[2]/2*1e-6
            corners = np.asarray([[xmin, ymin, zmin], [xmax, ymin, zmin], [xmin, ymax, zmin], [xmax, ymax, zmin],
                                  [xmin, ymin, zmax], [xmax, ymin, zmax], [xmin, ymax, zmax], [xmax, ymax, zmax]])
            grid = pv.ExplicitStructuredGrid(np.asarray([2, 2, 2]), corners)
            plotter._box = plotter.add_mesh(grid, style='wireframe', render_lines_as_tubes=True, line_width=2,
                                            show_edges=True, name="border")
        elif show_grid:
            plotter.show_grid(bounds=[xmin, xmax, ymin, ymax, zmin, zmax], color=plotter._theme.font.color, render=False)
    finally:
        plotter.render = render
        plotter.render()
