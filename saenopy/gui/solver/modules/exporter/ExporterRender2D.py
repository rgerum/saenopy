import numpy as np
import pandas as pd
from copy import deepcopy
from PIL import Image, ImageDraw, ImageFont
import matplotlib
import matplotlib.pyplot as plt
from saenopy.gui.common.resources import resource_path

from saenopy.gui.solver.modules.showVectorField import getVectorFieldImage
from saenopy.gui.solver.modules.exporter.ExportRenderCommon import get_time_text, getVectorFieldImage, get_mesh_arrows

default_data = {'use2D': True, 'image': {'logo_size': 0, 'scale': 1.0, 'antialiase': True, 'scale_overlay': 1.0}, 'camera': {'elevation': 35.0, 'azimuth': 45.0, 'distance': 0, 'offset_x': 0, 'offset_y': 0, 'roll': 0}, 'theme': 'dark', 'show_grid': True, 'use_nans': False, 'arrows': 'deformation', 'averaging_size': 1.0, 'deformation_arrows': {'autoscale': True, 'scale_max': 10.0, 'colormap': 'turbo', 'arrow_scale': 1.0, 'arrow_opacity': 1.0, 'skip': 1}, 'force_arrows': {'autoscale': True, 'scale_max': 1000.0, 'use_center': False, 'use_log': True, 'colormap': 'turbo', 'arrow_scale': 1.0, 'arrow_opacity': 1.0, 'skip': 1}, 'maps': 'None', 'maps_cmap': 'turbo', 'maps_alpha': 0.5, 'stack': {'image': True, 'channel': 'cells', 'z_proj': False, 'use_contrast_enhance': True, 'contrast_enhance': None, 'colormap': 'gray', 'z': 0, 'use_reference_stack': False, 'alpha': 1.0, 'channel_B': '', 'colormap_B': 'gray', 'alpha_B': 1.0}, 'scalebar': {'hide': False, 'length': 0.0, 'width': 5.0, 'xpos': 15.0, 'ypos': 10.0, 'fontsize': 18.0}, 'colorbar': {'hide': False, 'length': 150.0, 'width': 10.0, 'xpos': 15.0, 'ypos': 10.0, 'fontsize': 18.0}, '2D_arrows': {'width': 2.0, 'headlength': 5.0, 'headheight': 5.0}, 'crop': {'x': (913, 1113), 'y': (893, 1093), 'z': (0, 1)}, 'channel0': {'show': False, 'skip': 1, 'sigma_sato': 2, 'sigma_gauss': 0, 'percentiles': (0, 1), 'range': (0, 1), 'alpha': (0.1, 0.5, 1), 'cmap': 'pink'}, 'channel1': {'show': False, 'skip': 1, 'sigma_sato': 0, 'sigma_gauss': 7, 'percentiles': (0, 1), 'range': (0, 1), 'alpha': (0.1, 0.5, 1), 'cmap': 'Greens', 'channel': 1}, 'channel_thresh': 1.0, 'time': {'t': 0, 'format': '%d:%H:%M', 'start': 0.0, 'display': True, 'fontsize': 18}}

def update_dict(new_dict, old_dict, key):
    if isinstance(old_dict[key], dict):
        for key2 in old_dict[key].keys():
            update_dict(new_dict[key], old_dict[key], key2)
    else:
        new_dict[key] = old_dict[key]

def render_2d(params, result, exporter=None):
    default_copy = deepcopy(default_data)
    for key in params.keys():
        update_dict(default_copy, params, key)
    params = default_copy

    pil_image, display_image, im_scale, aa_scale = render_2d_image(params, result, exporter)
    if pil_image is None:
        return np.zeros((10, 10))

    pil_image, disp_params = render_2d_arrows(params, result, pil_image, im_scale, aa_scale, display_image, return_scale=True)

    if aa_scale == 2:
        pil_image = pil_image.resize([pil_image.width // 2, pil_image.height // 2])
        aa_scale = 1

    if params.get("maps", "None") != "None":
        pil_image, disp_params = render_map(params, result, pil_image, im_scale, aa_scale, display_image, return_scale=True)

    if params["scalebar"]["hide"] is False:
        pil_image = render_2d_scalebar(params, result, pil_image, im_scale, aa_scale)
    if disp_params != None and params["colorbar"]["hide"] is False:
        pil_image = render_2d_colorbar(params, result, pil_image, im_scale, aa_scale, scale_min=disp_params.get("scale_min", 0), scale_max=disp_params["scale_max"], colormap=disp_params["colormap"], unit=disp_params["scalebar_unit"])

    pil_image = render_2d_time(params, result, pil_image)

    pil_image = render_2d_logo(params, result, pil_image, aa_scale)

    return np.asarray(pil_image)


def render_2d_image(params, result, exporter):
    display_image = getVectorFieldImage(result, params, use_fixed_contrast_if_available=True, use_2D=True, exporter=exporter)
    if display_image is None:
        return None, None, 1, 1
    if params["stack"].get("channel_B", "") != "":
        params["stack"]["channel"] = params["stack"]["channel_B"]
        display_imageB = getVectorFieldImage(result, params, use_fixed_contrast_if_available=True, exporter=exporter)
    else:
        display_imageB = None

    im_scale = params["image"]["scale"]
    aa_scale = params["image"]["antialiase"] + 1

    im = np.squeeze(display_image[0])

    def adjust_img(im, colormap2):
        if len(im.shape) == 3 and (colormap2 is None or colormap2 == "gray"):
            if im.dtype == np.uint8:
                im = im.astype(np.float32) / 255
            return im
        if len(im.shape) == 3:
            if im.shape[2] == 1:
                im = im[:, :, 0]
            else:
                im = np.mean(im[:, :, :3], axis=2)
        if colormap2 is None:
            return im
        cmap = plt.get_cmap(colormap2)
        im = cmap(im)
        return im

    colormap2 = params["stack"]["colormap"]
    if True: #colormap2 is not None and colormap2 != "gray":
        im = adjust_img(im, colormap2)
        im = im * params["stack"].get("alpha", 1)

        if display_imageB is not None:
            im2 = adjust_img(display_imageB[0], params["stack"]["colormap_B"])
            im2 = im2 * params["stack"].get("alpha_B", 1)
            im += im2
            im = np.clip(im, 0, 1)
        im = (im * 255).astype(np.uint8)[:, :, :3]

    pil_image = Image.fromarray(im).convert("RGB")
    pil_image = pil_image.resize([int(pil_image.width * im_scale * aa_scale), int(pil_image.height * im_scale * aa_scale)])

    return pil_image, display_image, im_scale, aa_scale


def render_map(params, result, pil_image, im_scale, aa_scale, display_image, return_scale=False):
    data = result.get_data_structure()

    if "maps" not in data:
        if return_scale:
            return pil_image, None
        return pil_image

    map_name = params["maps"]
    map_im = result.get_map_data(params["maps"])
    cmap = data["maps"][map_name]["colormap"]
    vmin, vmax = data["maps"][map_name]["lim"]
    #cmap = params["maps_cmap"]
    alpha = params["maps_alpha"]
    if alpha is None:
        alpha = 0.5

    nan_mask = np.isnan(map_im)[:, :, None]

    pil_image = pil_image.convert("RGBA")

    map_im = map_im - vmin
    map_im = map_im / (vmax-vmin)
    map_im = plt.get_cmap(cmap)(map_im)

    map_im = (map_im*255).astype(np.uint8)
    map_im = np.where(nan_mask, np.asarray(pil_image), map_im)

    map_im = Image.fromarray(map_im, mode="RGBA")

    # Composite the overlay on top of the base image
    pil_image = Image.blend(pil_image, map_im, alpha).convert("RGB")
    if return_scale:
        return pil_image, {"scale_min": vmin, "scale_max": vmax, "colormap": cmap, "scalebar_unit": None}
    return pil_image


def render_2d_arrows(params, result, pil_image, im_scale, aa_scale, display_image, return_scale=False):
    def project_data(R, field, skip=1):
        length2 = np.linalg.norm(field[:, :2], axis=1)
        length3 = np.linalg.norm(field[:, :3], axis=1)
        angle = np.arctan2(field[:, 1], field[:, 0])
        data = pd.DataFrame(np.hstack((R, length2[:, None], length3[:, None], angle[:, None])),
                            columns=["x", "y", "length2", "length", "angle"])
        data = data.sort_values(by="length2", ascending=False)
        d2 = data.groupby(["x", "y"]).first()
        # optional slice
        if skip > 1:
            d2 = d2.loc[(slice(None, None, skip), slice(None, None, skip)), :]
        return np.array([i for i in d2.index]), d2[["length2", "angle", "length"]]

    mesh, field, params_arrows, name = get_mesh_arrows(params, result)

    if params_arrows is None:
        scale_max = None
    else:
        scale_max = params_arrows["scale_max"] if not params_arrows["autoscale"] else None
        colormap = params_arrows["colormap"]
        skip = params_arrows["skip"]
        alpha = params_arrows["arrow_opacity"]

    if mesh is None:
        if return_scale:
            if scale_max is None:
                return pil_image, None
            else:
                return pil_image, {"scale_max": scale_max, "colormap": colormap}
            return pil_image, None
        return pil_image

    if field is not None:
        # rescale and offset
        scale = 1e6 / display_image[1][0]
        offset = np.array(display_image[0].shape[0:2]) / 2    
        R = mesh.nodes.copy()
        is3D = R.shape[1] == 3
        field = field.copy()
        max_length = np.nanpercentile(np.linalg.norm(field, axis=1), 99.9)

        field_to_pixel_factor = 1
        scale_max_to_pixel_factor = 1
        scalebar_unit = getattr(mesh, "scalebar_unit", "px")

        if getattr(mesh, "units", None) == "pixels":
            R = R[:, :2]
            R[:, 1] = display_image[0].shape[0] - R[:, 1]
            field = field[:, :2] 
            field_to_pixel_factor = params_arrows["arrow_scale"] * getattr(mesh, "display_scale", 1)
            field[:, 1] = -field[:, 1]
        else:  # "microns" + 3D
            R = R[:, :2][:, ::-1] * scale + offset[::-1]
            field = field[:, :][:, [1, 0, 2]]
            field_to_pixel_factor = scale * params_arrows["arrow_scale"]

            if name == "forces":
                scalebar_unit = "pN"
                max_length *= 1e12
                scale_max_to_pixel_factor = 1e12
                field_to_pixel_factor *= 1e6
            else:
                scalebar_unit = "µm"
                max_length *= 1e6
                scale_max_to_pixel_factor = 1e6

            norm_stack_size = np.abs(np.max(R) - np.min(R))
            scalebar_max = params["deformation_arrows"]["scale_max"]
            if params["deformation_arrows"]["autoscale"]:
                field_to_pixel_factor *= 0.1 * norm_stack_size / np.nanmax(field*scale_max_to_pixel_factor)  # np.nanpercentile(point_cloud[name + "_mag2"], 99.9)
            else:
                field_to_pixel_factor *= 0.1 * norm_stack_size / scalebar_max

        if scale_max is None:
            scale_max = max_length

        if is3D:
            z_center = (params["stack"]["z"] - result.stacks[0].shape[2] / 2) * display_image[1][2] * 1e-6
            z_min = z_center - params["averaging_size"] * 1e-6
            z_max = z_center + params["averaging_size"] * 1e-6

            index = (z_min < mesh.nodes[:, 2]) & (mesh.nodes[:, 2] < z_max)

            R = R[index]
            field = field[index]
            R, field = project_data(R, field, skip=skip)
        else:
            length = np.linalg.norm(field, axis=1)
            angle = np.arctan2(field[:, 1], field[:, 0])
            field = pd.DataFrame(np.hstack((length[:, None], angle[:, None])), columns=["length", "angle"])
        # safety check if all arrows where filtered out
        if R.shape[0] != 0:
            # get the colormap
            cmap = plt.get_cmap(colormap)
            # calculate the colors of the arrows
            colors = cmap(field.length * scale_max_to_pixel_factor / scale_max)
            # set the transparency
            colors[:, 3] = alpha
            # make colors uint8
            colors = (colors * 255).astype(np.uint8)

            pil_image = add_quiver(pil_image, R, field.length * field_to_pixel_factor, field.angle, colors,
                                   scale=im_scale * aa_scale,
                                   width=params["2D_arrows"]["width"],
                                   headlength=params["2D_arrows"]["headlength"],
                                   headheight=params["2D_arrows"]["headheight"])
    if return_scale:
        return pil_image, {"scale_max": scale_max, "colormap": colormap, "scalebar_unit": scalebar_unit}
    return pil_image


def render_2d_scalebar(params, result, pil_image, im_scale, aa_scale):
    data = result.get_data_structure()

    def getBarParameters(pixtomu, scale=1):
        mu = 200 * pixtomu / scale
        values = [1, 5, 10, 25, 50, 75, 100, 150, 200, 250, 500, 1000, 1500, 2000, 2500, 5000, 10000]
        old_v = mu
        for v in values:
            if mu < v:
                mu = old_v
                break
            old_v = v
        pixel = mu / pixtomu * scale
        return pixel, mu

    if params["scalebar"]["length"] == 0:
        pixel, mu = getBarParameters(data["voxel_size"][0])
    else:
        mu = params["scalebar"]["length"]
        pixel = mu / data["voxel_size"][0]

    scale = params["image"]["scale_overlay"] * params["image"]["scale"]

    color = "k"
    if params["theme"] == "dark":
        color = "w"

    pil_image = add_scalebar(pil_image, scale=1, image_scale=im_scale * aa_scale,
                             width=params["scalebar"]["width"] * aa_scale * scale,
                             xpos=params["scalebar"]["xpos"] * aa_scale * scale,
                             ypos=params["scalebar"]["ypos"] * aa_scale * scale,
                             fontsize=params["scalebar"]["fontsize"] * aa_scale * scale, pixel_width=pixel,
                             size_in_um=mu, color=color, unit="µm")
    return pil_image

def render_2d_colorbar(params, result, pil_image, im_scale, aa_scale, colormap="viridis", scale_min=0, scale_max=1, unit="µm"):
    color = "k"
    if params["theme"] == "dark":
        color = "w"

    pil_image = add_colorbar(pil_image, scale=params["image"]["scale_overlay"] * im_scale,
                             colormap=colormap,#params["colorbar"]["colorbar"],
                             bar_width=params["colorbar"]["length"] * aa_scale,
                             bar_height=params["colorbar"]["width"] * aa_scale,
                             #tick_height=params["colorbar"]["tick_height"] * aa_scale,
                             #tick_count=params["colorbar"]["tick_count"],
                             #min_v=params["scalebar"]["min_v"],
                             min_v=scale_min,
                             max_v=scale_max,#params["colorbar"]["max_v"],
                             offset_x=params["colorbar"]["xpos"] * aa_scale,
                             offset_y=-params["colorbar"]["ypos"] * aa_scale,
                             fontsize=params["colorbar"]["fontsize"] * aa_scale,
                             color=color,
                             unit=unit
                             )

    return pil_image


def render_2d_time(params, result, pil_image):
    data = result.get_data_structure()
    color = "k"
    if params["theme"] == "dark":
        color = "w"
    if result is not None and data["time_delta"] is not None and params["time"]["display"]:
        pil_image = add_text(pil_image, get_time_text(params, result), position=(10, 10), color=color)
    return pil_image


def render_2d_logo(params, result, pil_image, aa_scale):
    scale = params["image"]["scale_overlay"] * params["image"]["scale"]

    if params["image"]["logo_size"] >= 10:
        if params["theme"] == "dark":
            im_logo = Image.open(resource_path("Logo_black.png"))
        else:
            im_logo = Image.open(resource_path("Logo.png"))
        scale = params["image"]["logo_size"] / im_logo.width * scale  # im.width/400*0.2
        im_logo = im_logo.resize([int(400 * scale * aa_scale), int(200 * scale * aa_scale)])
        padding = int(im_logo.width * 0.1)

        pil_image = pil_image.convert("RGBA")
        pil_image.alpha_composite(im_logo, dest=(pil_image.width - im_logo.width - padding, padding))
    return pil_image


def add_text(pil_image, text, position, color="w", fontsize=18):
    image = ImageDraw.ImageDraw(pil_image)
    font_size = int(round(fontsize * 4 / 3))  # the 4/3 appears to be a factor of "converting" screel dpi to image dpi
    try:
        font = ImageFont.truetype("arial", font_size)  # ImageFont.truetype("tahoma.ttf", font_size)
    except IOError:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)

    length_number = image.textlength(text, font=font)
    x, y = position

    if x < 0:
        x = pil_image.width + x - length_number[0]
    if y < 0:
        y = pil_image.height + y - length_number[1]
    color = tuple((matplotlib.colors.to_rgba_array(color)[0, :3] * 255).astype("uint8"))
    if pil_image.mode != "RGB":
        color = int(np.mean(color))

    image.text((x, y), text, color, font=font)
    return pil_image

def add_colorbar(pil_image,
                 colormap="viridis",
                 bar_width=150,
                 bar_height=10,
                 tick_height=5,
                 tick_width=1,
                 tick_count=3,
                 min_v=0,
                 max_v=10,
                 offset_x=15,
                 offset_y=-10,
                 scale=1, fontsize=16, color="w", unit="m"):
    cmap = plt.get_cmap(colormap)
    offset_x = int(offset_x * scale)
    offset_y = int(offset_y * scale)

    bar_width = int(bar_width*scale)
    bar_height = int(bar_height*scale)
    tick_height = int(tick_height*scale)
    tick_width = int(tick_width*scale)

    if offset_x < 0:
        offset_x = pil_image.size[0] + offset_x
    if offset_y < 0:
        offset_y = pil_image.size[1] + offset_y

    color = tuple((matplotlib.colors.to_rgba_array(color)[0, :3] * 255).astype("uint8"))
    if pil_image.mode != "RGB":
        color = int(np.mean(color))

    colors = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)
    for i in range(bar_width):
        c = plt.get_cmap(cmap)(int(i / bar_width * 255))
        colors[:, i, :] = [c[0] * 255, c[1] * 255, c[2] * 255]
    pil_image.paste(Image.fromarray(colors), (offset_x, offset_y - bar_height))

    image = ImageDraw.ImageDraw(pil_image)
    import matplotlib.ticker as ticker

    font_size = int(
        round(fontsize * scale * 4 / 3))  # the 4/3 appears to be a factor of "converting" screel dpi to image dpi
    if font_size == 0:
        font = None
    else:
        try:
            font = ImageFont.truetype("arial", font_size)  # ImageFont.truetype("tahoma.ttf", font_size)
        except IOError:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)

    locator = ticker.MaxNLocator(nbins=tick_count - 1)
    #tick_positions = locator.tick_values(min_v, max_v)
    tick_positions = np.linspace(min_v, max_v, tick_count)

    max_value = tick_positions[-1]
    factor = 1
    power = 0
    if unit is not None:
        if max_value:
            power = int(np.floor(np.log10(max_value)/3)*3)
            factor = 10**power
        base_factor = 0
        factor_prefix = {6: "M", 3: "k", -3: "m", -6: "µ", -9: "n", -12: "p"}
        if unit.startswith("p"):
            unit = unit[1:]
            base_power = -12
            unit = factor_prefix[power+base_power]+unit
        if unit.startswith("µ") or unit.startswith("u"):
            unit = unit[1:]
            base_power = -6
            unit = factor_prefix[power + base_power]+unit

    for i, pos in enumerate(tick_positions):
        x0 = offset_x + (bar_width - tick_width - 1) / (tick_count - 1) * i
        y0 = offset_y - bar_height - 1

        image.rectangle([x0, y0-tick_height, x0+tick_width, y0], fill=color)

        text = "%d" % (pos/factor)
        length_number = image.textlength(text, font=font)
        height_number = image.textbbox((0, 0), text, font=font)[3]

        x = x0 - length_number * 0.5 + 1
        y = y0 - height_number - tick_height - int(np.ceil(tick_height/2))
        # draw the text for the number and the unit
        if font is not None:
            image.text((x, y), text, color, font=font)
    if unit:
        height_number = image.textbbox((0, 0), unit, font=font)[3]
        x0 = offset_x + bar_width + 10
        y0 = offset_y - bar_height  / 2 - height_number /2
        if font is not None:
            image.text((x0, y0), unit, color, font=font)
    #image.rectangle([pil_image.size[0]-10, 0, pil_image.size[0], 10], fill="w")
    return pil_image

def add_scalebar(pil_image, scale, image_scale, width, xpos, ypos, fontsize, pixel_width, size_in_um, color="w", unit="µm"):
    image = ImageDraw.ImageDraw(pil_image)
    pixel_height = width
    pixel_offset_x = xpos
    pixel_offset_y = ypos
    pixel_offset2 = 3
    font_size = int(round(fontsize*scale*4/3))  # the 4/3 appears to be a factor of "converting" screel dpi to image dpi

    #pixel_width, size_in_um = self.getBarParameters(1)
    pixel_width *= image_scale
    color = tuple((matplotlib.colors.to_rgba_array(color)[0, :3]*255).astype("uint8"))
    if pil_image.mode != "RGB":
        color = int(np.mean(color))

    if pixel_offset_x > 0:
        image.rectangle([pil_image.size[0] -pixel_offset_x - pixel_width, pil_image.size[1] -pixel_offset_y - pixel_height, pil_image.size[0] -pixel_offset_x, pil_image.size[1] -pixel_offset_y], color)
    else:
        image.rectangle([-pixel_offset_x,
                         pil_image.size[1] - pixel_offset_y - pixel_height,
                         -pixel_offset_x + pixel_width,
                         pil_image.size[1] - pixel_offset_y], color)
    if True:
        # get the font
        if font_size == 0:
            return pil_image
        try:
            font = ImageFont.truetype("arial", font_size)#ImageFont.truetype("tahoma.ttf", font_size)
        except IOError:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        # width and height of text elements
        text = "%d" % size_in_um
        length_number = image.textlength(text, font=font)
        length_space = 0.5*image.textlength(" ", font=font)  # here we emulate a half-sized whitespace
        length_unit = image.textlength(unit, font=font)
        height_number = image.textbbox((0, 0), text+unit, font=font)[3]

        total_length = length_number + length_space + length_unit

        # find the position for the text to have it centered and bottom aligned
        if pixel_offset_x > 0:
            x = pil_image.size[0] - pixel_offset_x - pixel_width * 0.5 - total_length * 0.5
        else:
            x = - pixel_offset_x + pixel_width * 0.5 - total_length * 0.5
        y = pil_image.size[1] - pixel_offset_y - pixel_offset2 - pixel_height - height_number
        # draw the text for the number and the unit
        image.text((x, y), text, color, font=font)
        image.text((x+length_number+length_space, y), unit, color, font=font)
        return pil_image




def getarrow(length, angle, scale=1, width=2, headlength=5, headheight=5, offset=None):
    length *= scale
    width *= scale
    headlength *= scale
    headheight *= scale

    headlength = headlength*np.ones(len(length))
    headheight = headheight*np.ones(len(length))
    width = width*np.ones(len(length))
    index_small = length < headlength
    if np.any(index_small):
        headheight[index_small] = headheight[index_small] * length[index_small] / headlength[index_small]
        headlength[index_small] = length[index_small]
        width[index_small] = headheight[index_small]

    # generate the arrow points
    arrow = [(0, width / 2), (length - headlength, width / 2), (length - headlength, headheight / 2), (length, 0),
            (length - headlength, -headheight / 2), (length - headlength, -width / 2), (0, -width / 2)]
    # and distribute them for each point
    arrows = np.zeros([length.shape[0], 7, 2])
    for p in range(7):
        for i in range(2):
            arrows[:, p, i] = arrow[p][i]

    # rotate the arrow
    #angle = np.deg2rad(angle)
    rot = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    arrows = np.einsum("ijk,kli->ijl", arrows, rot)

    # add the offset
    arrows += offset[:, None, :]

    return arrows


def add_quiver(pil_image, R, lengths, angles, colors, scale=1, width=2, headlength=5, headheight=5):
    # get the arrows
    arrows = getarrow(lengths, angles, scale=scale, width=width, headlength=headlength, headheight=headheight, offset=R*scale)

    # draw the arrows
    image = ImageDraw.ImageDraw(pil_image, "RGBA")
    for a, c in zip(arrows, colors):
        image.polygon(list(a.flatten()), fill=tuple(c), outline=tuple(c))

    return pil_image
