import time
import pyvista
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

from saenopy.gui.solver.modules.exporter.FiberViewer import process_stack, join_stacks
from saenopy.gui.solver.modules.exporter.ExportRenderCommon import get_time_text, getVectorFieldImage, get_mesh_arrows, get_mesh_extent


def render_3d(params, result, plotter, exporter=None):

    render = plotter.render
    plotter.render = lambda *args: None
    try:
        render_3d_arrows(params, result, plotter)

        render_3d_image(params, result, plotter, exporter)

        render_3d_bounds(params, result, plotter)

        render_3d_fibers(params, result, plotter, exporter)

        render_3d_text(params, result, plotter)

        render_3d_camera(params, result, plotter, exporter, double_render=False)

        plotter.previous_plot_params = params
        plotter.previous_plot_result = result
    finally:
        plotter.render = render

def filter_params(params, used_values, previous_params):
    difference = False
    new_params = {}
    if isinstance(used_values, str):
        used_values = [used_values]
    for key in used_values:
        if isinstance(key, str):
            if key in previous_params:
                if previous_params[key] != params[key]:
                    difference = True
            else:
                difference = True
            new_params[key] = params[key]
        elif isinstance(key, tuple):
            key, sub_keys = key
            p, d = filter_params(params[key], sub_keys, previous_params.get(key, {}))
            new_params[key] = p
            if d:
                difference = d
    return new_params, difference


def render_3d_fibers(params, result, plotter, exporter):
    used_values = [
        ("crop", ('y', 'x', 'z')),
        ("time", ("t",)),
        ("stack", "use_reference_stack"),
        ("channel0", ("show", "sigma_sato", "sigma_gauss", "range", "alpha", "cmap")),
        ("channel1", ("show", "sigma_sato", "sigma_gauss", "range", "alpha", "cmap", "channel")),
        "channel_thresh",
    ]
    params, changed = filter_params(params, used_values, getattr(plotter, "previous_plot_params", {}))
    if not changed and result == getattr(plotter, "previous_plot_result", {}):
        return

    crops = []
    for value in [params["crop"]["y"], params["crop"]["x"], params["crop"]["z"]]:
        crops.extend(value)

    t = params["time"]["t"]
    t_start = time.time()
    stack_data = None
    stack = result.stacks[t]

    if params["stack"]["use_reference_stack"]:
        stack = result.stack_reference

    if params["channel0"]["show"] and crops[0] != crops[1] and crops[2] != crops[3] and crops[4] != \
            crops[5]:
        stack_data1 = process_stack(stack, 0,
                                    crops=crops,
                                    **params["channel0"])
        stack_data = stack_data1
        if exporter is not None:
            exporter.channel0_properties.sigmoid.p.set_im(stack_data1["original"])
    else:
        stack_data1 = None

    if params["channel1"]["show"] and crops[0] != crops[1] and crops[2] != crops[3] and crops[4] != \
            crops[5]:

        if isinstance(params["channel1"]["channel"], str):
            params["channel1"]["channel"] = result.stacks[0].channels.index(params["channel1"]["channel"])
        stack_data2 = process_stack(stack,
                                    crops=crops,
                                    **params["channel1"])
        if exporter is not None:
            exporter.channel1_properties.sigmoid.p.set_im(stack_data2["original"])
        if stack_data1 is not None:
            stack_data = join_stacks(stack_data1, stack_data2, params["channel_thresh"])
        else:
            stack_data = stack_data2

    if stack_data is not None:
        dataset = stack_data["data"]
        mesh = pyvista.UniformGrid(dimensions=dataset.shape, spacing=stack_data["resolution"],
                                   origin=stack_data["center"])
        mesh['values'] = dataset.ravel(order='F')
        mesh.active_scalars_name = 'values'

        plotter.add_volume(mesh,
                          cmap=stack_data["cmap"], opacity=stack_data["opacity"],
                          blending="composite", name="fiber", render=False)  # 1.0*x
        plotter.remove_scalar_bar("values")
    else:
        plotter.remove_actor("fiber")
    #print("plot time", f"{time.time() - t_start:.3f}s")


def render_3d_text(params, result, plotter):
    used_values = [
        ("time", ("t", "display", "fontsize", "start", "format")),
        ("image", "height"),
    ]
    params, changed = filter_params(params, used_values, getattr(plotter, "previous_plot_params", {}))
    if not changed and result == getattr(plotter, "previous_plot_result", {}):
        return

    if result is not None and result.time_delta is not None and params["time"]["display"]:
        plotter.add_text(get_time_text(params, result), name="time_text", font_size=params["time"]["fontsize"],
                         position=(20, params["image"]["height"] - 20 - params["time"]["fontsize"] * 2))
    else:
        plotter.remove_actor("time_text")


def render_3d_arrows(params, result, plotter):
    used_values = [
        "use_nans",
        "arrows",
        ("time", "t"),
        ("deformation_arrows", ("scale_max", "autoscale", "skip", "arrow_opacity", "colormap", "arrow_scale")),
        ("force_arrows", ("scale_max", "autoscale", "skip", "arrow_opacity", "colormap", "arrow_scale", "use_center")),
    ]
    params, changed = filter_params(params, used_values, getattr(plotter, "previous_plot_params", {}))

    if not changed and result == getattr(plotter, "previous_plot_result", {}):
        return

    obj, field, params_arrows, name = get_mesh_arrows(params, result)

    if obj is None:
        plotter.remove_actor("arrows")
        plotter.remove_actor("nans")
        plotter.remove_actor("center")
        return

    show_nan = params["use_nans"]
    scalebar_max = params_arrows["scale_max"] if params_arrows["autoscale"] else None
    colormap = params_arrows["colormap"]
    skip = params_arrows["skip"]
    arrow_opacity = params_arrows["arrow_opacity"]

    if field is not None:
        obj_R = obj.nodes * 1e6

        if skip != 1:
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
            factor = 0.1 * params_arrows["arrow_scale"]
        if name == "forces":
            # scale forces to pN
            point_cloud.point_data[name + "_mag2"] = 1e12*point_cloud.point_data[name + "_mag"].copy()
            factor = 0.15 * params_arrows["arrow_scale"]
        # hide nans
        point_cloud.point_data[name + "_mag2"][nan_values] = 0

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

        # show the nan points
        if show_nan:
            R = obj_R[nan_values]
            if R.shape[0]:
                point_cloud2 = pv.PolyData(R)
                point_cloud2.point_data["nan"] = obj_R[nan_values, 0] * np.nan
                plotter.add_mesh(point_cloud2, colormap=colormap, scalars="nan", show_scalar_bar=False, render=False, name="nans")
            else:
                plotter.remove_actor("nans")
        else:
            plotter.remove_actor("nans")

        # add the arrows
        sargs = dict(title_font_size=15, label_font_size=15, n_labels=3, title=title,
                     fmt="%.1e", color=plotter._theme.font.color, font_family="arial")
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
    if params_arrows.get("use_center", False):
        center = obj.get_center(mode="Force")
        plotter.add_points(np.array([center])*1e6, color='m', point_size=10, render=False, name="center")
    else:
        plotter.remove_actor("center")


def render_3d_image(params, result, plotter, exporter=None):
    used_values = [
        ("stack", ("image", "colormap", "use_reference_stack", "channel",
                   "z", "z_proj", "contrast_enhance", "use_contrast_enhance",
                   "channel_B", "colormap_B")),
        ("time", "t"),
    ]
    params, changed = filter_params(params, used_values, getattr(plotter, "previous_plot_params", {}))
    if not changed and result == getattr(plotter, "previous_plot_result", {}):
        return
    scale = 1
    colormap2 = params["stack"]["colormap"]

    display_image = getVectorFieldImage(result, params, use_fixed_contrast_if_available=True, exporter=exporter)
    params["stack"]["channel"] = params["stack"]["channel_B"]
    if params["stack"]["channel_B"] != "":
        display_imageB = getVectorFieldImage(result, params, use_fixed_contrast_if_available=True, exporter=exporter)
    else:
        display_imageB = None

    def adjust_im(img, colormap2):
        # adjust the direction of the underlying image
        # the combination of following both operations does the job
        img_adjusted = img[:, ::-1]  # mirror the image
        img_adjusted = np.swapaxes(img_adjusted, 1, 0)  # switch axis

        if (len(img_adjusted.shape) == 2 or img_adjusted.shape[
            2] == 1) and colormap2 is not None and colormap2 != "gray":
            if len(img_adjusted.shape) == 3:
                img_adjusted = img_adjusted[:, :, 0]
            cmap = plt.get_cmap(colormap2)
            img_adjusted = (cmap(img_adjusted) * 255).astype(np.uint8)[:, :, :3]
        elif len(img_adjusted.shape) == 2 or img_adjusted.shape[2] == 1:
            img_adjusted = np.tile(np.squeeze(img_adjusted)[:, :, None], (1, 1, 3))
        return img_adjusted

    if display_image is not None:
        img, voxel_size, z_pos = display_image
        img_adjusted = adjust_im(img, colormap2)
        if display_imageB is not None:
            img_adjusted = img_adjusted.astype(np.int16) + adjust_im(display_imageB[0], params["stack"]["colormap_B"])
            img_adjusted = np.clip(img_adjusted, 0, 255).astype(np.uint8)
        # get coords
        xmin = (-img_adjusted.shape[1] / 2) * voxel_size[0] * scale
        ymin = (-img_adjusted.shape[0] / 2) * voxel_size[1] * scale
        x = np.linspace(xmin, -xmin, 10)
        y = np.linspace(ymin, -ymin, 10)
        x, y = np.meshgrid(x, y)
        z = z_pos * voxel_size[2] * scale * np.ones_like(x)
        # structureGrid
        curvsurf = pv.StructuredGrid(x, y, z)
        # Map the curved surface to a plane - use best fitting plane
        curvsurf.texture_map_to_plane(inplace=True)
        tex = pv.numpy_to_texture(img_adjusted)
        # add image below arrow field
        plotter.add_mesh(curvsurf, texture=tex, name="image_mesh")
    else:
        plotter.remove_actor("image_mesh")


def render_3d_bounds(params, result, plotter):
    used_values = [
        "show_grid",
        "use_nans",
        "arrows",
        ("time", "t"),
        ("deformation_arrows", ("scale_max", "autoscale", "skip", "arrow_opacity", "colormap", "arrow_scale")),
        ("force_arrows", ("scale_max", "autoscale", "skip", "arrow_opacity", "colormap", "arrow_scale", "use_center")),
    ]
    params, changed = filter_params(params, used_values, getattr(plotter, "previous_plot_params", {}))
    if not changed and result == getattr(plotter, "previous_plot_result", {}):
        return

    show_grid = params["show_grid"]

    # show the mesh border cube
    if show_grid == 2:
        # get the mesh extent
        stack_min_max = get_mesh_extent(params, result)
        # if present draw a cube
        if stack_min_max is not None:
            ((xmin, ymin, zmin), (xmax, ymax, zmax)) = stack_min_max
            corners = np.asarray([[xmin, ymin, zmin], [xmax, ymin, zmin], [xmin, ymax, zmin], [xmax, ymax, zmin],
                                  [xmin, ymin, zmax], [xmax, ymin, zmax], [xmin, ymax, zmax], [xmax, ymax, zmax]])
            grid = pv.ExplicitStructuredGrid(np.asarray([2, 2, 2]), corners)
            plotter.add_mesh(grid, style='wireframe', render_lines_as_tubes=True, line_width=2, show_edges=True,
                             name="border")
        # or remove the current cube
        else:
            plotter.remove_actor("border")
    # show the image stack border cube
    elif show_grid == 3:
        # get the stack shape
        if len(result.stacks):
            stack_shape = np.array(result.stacks[0].shape[:3]) * np.array(
                result.stacks[0].voxel_size)
        else:
            stack_shape = None
        # draw a cube around it
        if stack_shape is not None:
            xmin, xmax = -stack_shape[0] / 2, stack_shape[0] / 2
            ymin, ymax = -stack_shape[1] / 2, stack_shape[1] / 2
            zmin, zmax = -stack_shape[2] / 2, stack_shape[2] / 2
            corners = np.asarray([[xmin, ymin, zmin], [xmax, ymin, zmin], [xmin, ymax, zmin], [xmax, ymax, zmin],
                                  [xmin, ymin, zmax], [xmax, ymin, zmax], [xmin, ymax, zmax], [xmax, ymax, zmax]])
            grid = pv.ExplicitStructuredGrid(np.asarray([2, 2, 2]), corners)
            plotter.add_mesh(grid, style='wireframe', render_lines_as_tubes=True, line_width=2,
                             show_edges=True, name="border")
        # or remove the border
        else:
            plotter.remove_actor("border")
    else:
        plotter.remove_actor("border")

    # show the coordinate axes
    if show_grid == 1:
        # get the extent of the mesh
        stack_min_max = get_mesh_extent(params, result)
        # show the grid or remove it
        if stack_min_max is not None:
            ((xmin, ymin, zmin), (xmax, ymax, zmax)) = stack_min_max
            plotter.show_grid(bounds=[xmin, xmax, ymin, ymax, zmin, zmax], color=plotter._theme.font.color, render=False)
        else:
            plotter.remove_bounds_axes()
    else:
        plotter.remove_bounds_axes()


def render_3d_camera(params, result, plotter, exporter=None, double_render=False):
    used_values = [
        ("camera", ("elevation", "azimuth", "distance", "offset_x", "offset_y", "roll")),
    ]
    params, changed = filter_params(params, used_values, getattr(plotter, "previous_plot_params", {}))
    if not changed and result == getattr(plotter, "previous_plot_result", {}):
        return

    # reset the camera before rotating
    plotter.camera_position = "yz"

    # if the distance is not set use a reasonable default based on the current stack
    if params["camera"]["distance"] == 0:
        distance = plotter.camera.position[0]
        params["camera"]["distance"] = distance
        if exporter is not None:
            exporter.input_distance.setValue(distance)

    # define the offsets, they might need to be rotated by roll an azimuth
    dx = params["camera"]["offset_x"]
    dz = params["camera"]["offset_y"]
    dx, dz = rotate((dx, dz), params["camera"]["roll"])
    dx, dy = rotate((dx, 0), params["camera"]["azimuth"])
    # position of the camera defined by offset and distance
    plotter.camera.position = (params["camera"]["distance"] - dy, -dx, -dz)
    plotter.camera.focal_point = (0 - dy, -dx, -dz)
    # the rotation angles of the camera
    plotter.camera.azimuth = params["camera"]["azimuth"]
    plotter.camera.elevation = params["camera"]["elevation"]
    plotter.camera.roll += params["camera"]["roll"]


def rotate(pos, angle):
    x, y = pos
    angle = np.deg2rad(angle)
    s, c = np.sin(angle), np.cos(angle)
    return x * c + y * s, -x * s + y * c
