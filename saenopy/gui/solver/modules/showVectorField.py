import imageio
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
import numpy as np
import matplotlib.pyplot as plt
from saenopy.solver import Solver


def getVectorFieldImage(self, use_fixed_contrast_if_available=False, use_2D=False):
    try:
        image = self.vtk_toolbar.show_image.value()
        if use_2D:
            image = 1
        if image and self.t_slider.value() < len(self.result.stacks):
            if getattr(self, "input_reference_stack", None) and self.input_reference_stack.value() and self.result.stack_reference:
                stack = self.result.stack_reference
            else:
                stack = self.result.stacks[self.t_slider.value()]
            if self.vtk_toolbar.channel_select.value() >= len(stack.channels):
                self.vtk_toolbar.channel_select.setValue(0)
                im = stack[:, :, :, self.z_slider.value(), 0]
            else:
                im = stack[:, :, :, self.z_slider.value(), self.vtk_toolbar.channel_select.value()]
            if self.vtk_toolbar.button_z_proj.value():
                z_range = [0, 5, 10, 1000][self.vtk_toolbar.button_z_proj.value()]
                start = np.clip(self.z_slider.value() - z_range, 0,
                                stack.shape[2])
                end = np.clip(self.z_slider.value() + z_range, 0, stack.shape[2])
                im = stack[:, :, :, start:end, self.vtk_toolbar.channel_select.value()]
                im = np.max(im, axis=3)

            if self.vtk_toolbar.contrast_enhance.value():
                if use_fixed_contrast_if_available and self.vtk_toolbar.contrast_enhance_values.value():
                    (min, max) = self.vtk_toolbar.contrast_enhance_values.value()
                else:
                    (min, max) = np.percentile(im, (1, 99))
                    self.vtk_toolbar.contrast_enhance_values.setValue((min, max))
                im = im.astype(np.float32) - min
                im = im.astype(np.float64) * 255 / (max - min)
                im = np.clip(im, 0, 255).astype(np.uint8)

            display_image = [im, stack.voxel_size, self.z_slider.value() - stack.shape[2] / 2]
            if self.vtk_toolbar.show_image.value() == 2:
                display_image[2] = -stack.shape[2] / 2
        else:
            display_image = None
    except FileNotFoundError:
        display_image = None
    return display_image

def showVectorField2(self, M, points_name):
    display_image = getVectorFieldImage(self)

    try:
        field = getattr(M, points_name)
    except AttributeError:
        field = M.get_node_var(points_name)

    if len(self.result.stacks):
        stack_shape = np.array(self.result.stacks[0].shape[:3]) * np.array(self.result.stacks[0].voxel_size)
    else:
        stack_shape = None
    showVectorField(self.plotter, M, field, points_name,
                    scalebar_max=self.vtk_toolbar.getScaleMax(), show_nan=self.vtk_toolbar.use_nans.value(),
                    display_image=display_image, show_grid=self.vtk_toolbar.show_grid.value(),
                    factor=0.1*self.vtk_toolbar.arrow_scale.value(),
                    colormap=self.vtk_toolbar.colormap_chooser.value(),
                    colormap2=self.vtk_toolbar.colormap_chooser2.value(),
                    stack_shape=stack_shape)


def showVectorField(plotter: QtInteractor, obj: Solver, field: np.ndarray, name: str, center=None, show_nan=True, stack_shape=None,
                    show_all_points=False, factor=.1, scalebar_max=None, display_image=None, show_grid=True,
                    colormap="turbo", colormap2=None, stack_min_max=None, arrow_opacity=1, skip=1):
    # ensure that the image is either with color channels or no channels
    if (display_image is not None) and (display_image[0].shape[2] == 1):
        display_image[0] = display_image[0][:, :, 0]

    # force rendering to be disabled while updating content to prevent flickering
    render = plotter.render
    plotter.render = lambda *args: None
    try:
        plotter.renderer.remove_bounds_axes()
        plotter.renderer.remove_bounding_box()

        scale = 1  # 1e-6

        if field is not None:
            obj_R = obj.nodes*1e6

            if skip != 1:
                N = int(np.sqrt(obj_R.shape[0]))
                x_unique = len(np.unique(obj_R[:, 0]))
                y_unique = len(np.unique(obj_R[:, 1]))
                z_unique = len(np.unique(obj_R[:, 2]))
                obj_R = obj_R.reshape(x_unique, y_unique, z_unique, 3)[::skip, ::skip, ::skip].reshape(-1, 3)
                field = field.reshape(x_unique, y_unique, z_unique, 3)[::skip, ::skip, ::skip].reshape(-1, 3)

            # get positions of nan values
            nan_values = np.isnan(field[:, 0])

            # create a point cloud
            point_cloud = pv.PolyData(obj_R)
            point_cloud.point_data[name] = field
            point_cloud.point_data[name + "_mag"] = np.linalg.norm(field, axis=1)
            # convert to common units
            if name == "displacements_measured" or name == "displacements_target" or name == "displacements":
                  # scale deformations to µN
                  point_cloud.point_data[name + "_mag2"] = 1e6*point_cloud.point_data[name + "_mag"].copy()
            if name == "forces":
                  # scale forces to pN
                  point_cloud.point_data[name + "_mag2"] = 1e12*point_cloud.point_data[name + "_mag"].copy()
            # hide nans
            point_cloud.point_data[name + "_mag2"][nan_values] = 0
            # show nans
            if not show_all_points and show_nan:
                R = obj_R[nan_values]
                if R.shape[0]:
                    point_cloud2 = pv.PolyData(R)
                    point_cloud2.point_data["nan"] = obj_R[nan_values, 0] * np.nan
            R = obj_R[nan_values]
            if name == "forces" and getattr(obj, "cell_boundary_mask", None) is not None:
                R = obj_R[obj.cell_boundary_mask]
                if R.shape[0]:
                    point_cloud2 = pv.PolyData(R)
                    point_cloud2.point_data["nan"] = R[:, 0] * np.nan

            # scalebar scaling factor
            norm_stack_size = np.abs(np.max(obj_R) - np.min(obj_R))
            if scalebar_max is None:
                factor = factor * norm_stack_size / np.nanmax(point_cloud[name + "_mag2"])#np.nanpercentile(point_cloud[name + "_mag2"], 99.9)
            else:
                factor = factor * norm_stack_size / scalebar_max

            # generate the arrows
            arrows = point_cloud.glyph(orient=name, scale=name + "_mag2", factor=factor)

            title = name
            if name == "displacements_measured" or name == "displacements_target" or name == "displacements":
                title = "Deformations (µm)"
            elif name == "forces":
                title = "Forces (pN)"

            sargs = dict(#position_x=0.05, position_y=0.95,
                         title_font_size=15,
                         label_font_size=15,
                         n_labels=3,
                         title=title,
                         #italic=True,  ##height=0.25, #vertical=True,
                         fmt="%.1e",
                         color=plotter._theme.font.color,
                         font_family="arial")

            # show the points
            plotter.remove_actor("nans")
            if show_all_points:
                plotter.add_mesh(point_cloud, colormap=colormap, scalars=name + "_mag2", render=False)
            elif show_nan:
                if R.shape[0]:
                    plotter.add_mesh(point_cloud2, colormap=colormap, scalars="nan",
                                                         show_scalar_bar=False, render=False, name="nans")

            # add the arrows
            plotter.add_mesh(arrows, scalar_bar_args=sargs, colormap=colormap, name="arrows", opacity=arrow_opacity, render=False)

            # update the scalebar
            plotter.auto_value = np.nanpercentile(point_cloud[name + "_mag2"], 99.9)
            if scalebar_max is None:
                plotter.update_scalar_bar_range([0, np.nanpercentile(point_cloud[name + "_mag2"], 99.9)])
            else:
                plotter.update_scalar_bar_range([0, scalebar_max])
        else:
            plotter.remove_actor("arrows")

        # plot center points if desired
        if center is not None:
            plotter.add_points(np.array([center])*1e6, color='m', point_size=10, render=False, name="center")
        else:
            plotter.remove_actor("center")

        if display_image is not None:
            img, voxel_size, z_pos = display_image  
            # adjust the direction of the underlying image 
            # the combination of following both operations does the job
            img_adjusted = img[:, ::-1]                             # mirror the image
            img_adjusted = np.swapaxes(img_adjusted, 1,0)   # switch axis
            if len(img_adjusted.shape) == 2 and colormap2 is not None and colormap2 != "gray":
                cmap = plt.get_cmap(colormap2)
                #print(img_adjusted.shape, img_adjusted.dtype, img_adjusted.min(), img_adjusted.mean(), img_adjusted.max())
                img_adjusted = (cmap(img_adjusted)*255).astype(np.uint8)[:,:, :3]
                #print(img_adjusted.shape, img_adjusted.dtype, img_adjusted.min(), img_adjusted.mean(), img_adjusted.max())
            # get coords
            xmin = (-img_adjusted.shape[1]/2)*voxel_size[0]*scale
            ymin = (-img_adjusted.shape[0]/2)*voxel_size[1]*scale
            x = np.linspace(xmin, -xmin, 10)
            y = np.linspace(ymin, -ymin, 10)
            x, y = np.meshgrid(x, y)
            z = z_pos*voxel_size[2]*scale+0*x
            # structureGrid
            curvsurf = pv.StructuredGrid(x, y, z)
            # Map the curved surface to a plane - use best fitting plane
            curvsurf.texture_map_to_plane(inplace=True)   
            tex = pv.numpy_to_texture(img_adjusted)
            # add image below arrow field            
            mesh = plotter.add_mesh(curvsurf, texture=tex, name="image_mesh")
        else:
            plotter.remove_actor("image_mesh")

        plotter.remove_bounds_axes()

        if show_grid == 2 and (field is not None or stack_min_max is not None):
            if field is not None:
                xmin, ymin, zmin = obj_R.min(axis=0)
                xmax, ymax, zmax = obj_R.max(axis=0)
            else:
                ((xmin, ymin, zmin), (xmax, ymax, zmax)) = stack_min_max
            corners = np.asarray([[xmin, ymin, zmin], [xmax, ymin, zmin], [xmin, ymax, zmin], [xmax, ymax, zmin],
                                   [xmin, ymin, zmax], [xmax, ymin, zmax], [xmin, ymax, zmax], [xmax, ymax, zmax]])
            grid = pv.ExplicitStructuredGrid(np.asarray([2, 2, 2]), corners)
            plotter.add_mesh(grid, style='wireframe', render_lines_as_tubes=True, line_width=2, show_edges=True, name="border")
        elif show_grid == 3 and stack_shape is not None:
            xmin, xmax = -stack_shape[0]/2*scale, stack_shape[0]/2*scale
            ymin, ymax = -stack_shape[1]/2*scale, stack_shape[1]/2*scale
            zmin, zmax = -stack_shape[2]/2*scale, stack_shape[2]/2*scale
            corners = np.asarray([[xmin, ymin, zmin], [xmax, ymin, zmin], [xmin, ymax, zmin], [xmax, ymax, zmin],
                                  [xmin, ymin, zmax], [xmax, ymin, zmax], [xmin, ymax, zmax], [xmax, ymax, zmax]])
            grid = pv.ExplicitStructuredGrid(np.asarray([2, 2, 2]), corners)
            plotter.add_mesh(grid, style='wireframe', render_lines_as_tubes=True, line_width=2,
                                            show_edges=True, name="border")
        else:
            plotter.remove_actor("border")
        if show_grid == 1 and (field is not None or stack_min_max is not None):
            if field is not None:
                xmin, ymin, zmin = obj_R.min(axis=0)
                xmax, ymax, zmax = obj_R.max(axis=0)
            else:
                ((xmin, ymin, zmin), (xmax, ymax, zmax)) = stack_min_max
            plotter.show_grid(bounds=[xmin, xmax, ymin, ymax, zmin, zmax], color=plotter._theme.font.color, render=False)
    finally:
        plotter.render = render
        plotter.render()
