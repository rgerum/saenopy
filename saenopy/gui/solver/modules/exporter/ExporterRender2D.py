import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib
import matplotlib.pyplot as plt
from saenopy.gui.common.resources import resource_path

from saenopy.gui.solver.modules.showVectorField import getVectorFieldImage
from saenopy.gui.solver.modules.exporter.ExportRenderCommon import get_time_text, getVectorFieldImage, get_mesh_arrows


def render_2d(params, result, exporter=None):
    pil_image, display_image, im_scale, aa_scale = render_2d_image(params, result, exporter)
    if pil_image is None:
        return np.zeros((10, 10))

    pil_image, disp_params = render_2d_arrows(params, result, pil_image, im_scale, aa_scale, display_image, return_scale=True)

    if aa_scale == 2:
        pil_image = pil_image.resize([pil_image.width // 2, pil_image.height // 2])
        aa_scale = 1

    if params["scalebar"]["hide"] is False:
        pil_image = render_2d_scalebar(params, result, pil_image, im_scale, aa_scale)
    if disp_params != None and params["colorbar"]["hide"] is False:
        pil_image = render_2d_colorbar(params, result, pil_image, im_scale, aa_scale, scale_max=disp_params["scale_max"], colormap=disp_params["colormap"])

    pil_image = render_2d_time(params, result, pil_image)

    pil_image = render_2d_logo(params, result, pil_image, aa_scale)

    return np.asarray(pil_image)


def render_2d_image(params, result, exporter):
    display_image = getVectorFieldImage(result, params, use_fixed_contrast_if_available=True, use_2D=True, exporter=exporter)
    if display_image is None:
        return None, None, 1, 1
    if params["stack"]["channel_B"] != "":
        params["stack"]["channel"] = params["stack"]["channel_B"]
        display_imageB = getVectorFieldImage(result, params, use_fixed_contrast_if_available=True, exporter=exporter)
    else:
        display_imageB = None

    im_scale = params["image"]["scale"]
    aa_scale = params["image"]["antialiase"] + 1

    im = np.squeeze(display_image[0])

    def adjust_img(im, colormap2):
        if len(im.shape) == 3:
            if im.shape[2] == 1:
                im = im[:, :, 0]
            else:
                im = np.mean(im[:, :, :3], axis=2)
        cmap = plt.get_cmap(colormap2)
        im = cmap(im)
        return im

    colormap2 = params["stack"]["colormap"]
    if colormap2 is not None and colormap2 != "gray":
        im = adjust_img(im, colormap2)

        if display_imageB is not None:
            print("add second", params["stack"]["channel_B"], params["stack"]["colormap_B"])
            im += adjust_img(display_imageB[0], params["stack"]["colormap_B"])
            im = np.clip(im, 0, 1)
        im = (im * 255).astype(np.uint8)[:, :, :3]

    pil_image = Image.fromarray(im).convert("RGB")
    pil_image = pil_image.resize([int(pil_image.width * im_scale * aa_scale), int(pil_image.height * im_scale * aa_scale)])

    return pil_image, display_image, im_scale, aa_scale


def render_2d_arrows(params, result, pil_image, im_scale, aa_scale, display_image, return_scale=False):
    def project_data(R, field, skip=1):
        length = np.linalg.norm(field, axis=1)
        angle = np.arctan2(field[:, 1], field[:, 0])
        data = pd.DataFrame(np.hstack((R, length[:, None], angle[:, None])),
                            columns=["x", "y", "length", "angle"])
        data = data.sort_values(by="length", ascending=False)
        d2 = data.groupby(["x", "y"]).first()
        # optional slice
        if skip > 1:
            d2 = d2.loc[(slice(None, None, skip), slice(None, None, skip)), :]
        return np.array([i for i in d2.index]), d2[["length", "angle"]]

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
        if getattr(mesh, "units", None) == "pixels":
            R = R[:, :2]
            R[:, 1] = display_image[0].shape[0] - R[:, 1]
            field = field[:, :2] * params_arrows["arrow_scale"]
            field[:, 1] = -field[:, 1]
        else:  # "microns" + 3D
            R = R[:, :2][:, ::-1] * scale + offset
            field = field[:, :2][:, ::-1] * scale * params_arrows["arrow_scale"]

            if name == "forces":
                field *= 1e4

        if scale_max is None:
            max_length = np.nanmax(np.linalg.norm(field, axis=1))# * params_arrows["arrow_scale"]
            scale_max = max_length / params_arrows["arrow_scale"]
        else:
            max_length = scale_max * params_arrows["arrow_scale"]

        if is3D:
            z_center = (params["averaging_size"] - result.stacks[0].shape[2] / 2) * display_image[1][2] * 1e-6
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
        pil_image = add_quiver(pil_image, R, field.length, field.angle, max_length=max_length, cmap=colormap,
                               alpha=alpha,
                               scale=im_scale * aa_scale,
                               width=params["2D_arrows"]["width"],
                               headlength=params["2D_arrows"]["headlength"],
                               headheight=params["2D_arrows"]["headheight"])
    if return_scale:
        return pil_image, {"scale_max": scale_max, "colormap": colormap}
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

def render_2d_colorbar(params, result, pil_image, im_scale, aa_scale, colormap="viridis", scale_max=1):
    color = "k"
    if params["theme"] == "dark":
        color = "w"

    pil_image = add_colorbar(pil_image, scale=params["image"]["scale_overlay"] * params["image"]["scale"],
                             colormap=colormap,#params["colorbar"]["colorbar"],
                             bar_width=params["colorbar"]["length"] * aa_scale,
                             bar_height=params["colorbar"]["width"] * aa_scale,
                             #tick_height=params["colorbar"]["tick_height"] * aa_scale,
                             #tick_count=params["colorbar"]["tick_count"],
                             #min_v=params["scalebar"]["min_v"],
                             max_v=scale_max,#params["colorbar"]["max_v"],
                             offset_x=params["colorbar"]["xpos"] * aa_scale,
                             offset_y=-params["colorbar"]["ypos"] * aa_scale,
                             fontsize=params["colorbar"]["fontsize"] * aa_scale,
                             color=color,
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


def add_quiver(pil_image, R, lengths, angles, max_length, cmap, alpha=1, scale=1):
    cmap = plt.get_cmap(cmap)
    image = ImageDraw.ImageDraw(pil_image, "RGBA")
    def getarrow(length, width=2, headlength=5, headheight=5):
        length *= scale
        width *= scale
        headlength *= scale
        headheight *= scale
        if length < headlength:
            headheight = headheight*length/headlength
            headlength = length
            return [(length - headlength, headheight / 2),
                    (length, 0),
                    (length - headlength, -headheight / 2)]
        return [(0, width/2), (length-headlength, width/2), (length-headlength, headheight/2), (length, 0),
                (length-headlength, -headheight/2), (length-headlength, -width/2), (0, -width/2)]

    def get_offset(arrow, pos, angle):
        arrow = np.array(arrow)
        rot = [[np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))], [-np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]]
        arrow = arrow @ rot
        r = np.array(arrow) + np.array(pos)*scale
        return [tuple(i) for i in r]

    #max_length = np.nanmax(lengths)
    for i in range(len(R)):
        angle = angles.iloc[i]
        length = lengths.iloc[i]
        color = tuple((np.asarray(cmap(length/max_length))*255).astype(np.uint8))
        color = (color[0], color[1], color[2], int(alpha*255))
        image.polygon(get_offset(getarrow(length), R[i], np.rad2deg(angle)), fill=color, outline=color)
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
                 scale=1, fontsize=16, color="w"):
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
    try:
        font = ImageFont.truetype("arial", font_size)  # ImageFont.truetype("tahoma.ttf", font_size)
    except IOError:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)

    locator = ticker.MaxNLocator(nbins=tick_count - 1)
    #tick_positions = locator.tick_values(min_v, max_v)
    tick_positions = np.linspace(min_v, max_v, tick_count)
    for i, pos in enumerate(tick_positions):
        x0 = offset_x + (bar_width - tick_width - 1) / (tick_count - 1) * i
        y0 = offset_y - bar_height - 1

        image.rectangle([x0, y0-tick_height, x0+tick_width, y0], fill=color)

        text = "%d" % pos
        length_number = image.textlength(text, font=font)
        height_number = image.textbbox((0, 0), text, font=font)[3]

        x = x0 - length_number * 0.5 + 1
        y = y0 - height_number - tick_height - int(np.ceil(tick_height/2))
        # draw the text for the number and the unit
        image.text((x, y), text, color, font=font)
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


def add_quiver(pil_image, R, lengths, angles, max_length, cmap, alpha=1, scale=1, width=2, headlength=5, headheight=5):
    # get the colormap
    cmap = plt.get_cmap(cmap)
    # calculate the colors of the arrows
    colors = cmap(lengths / max_length)
    # set the transparancy
    colors[:, 3] = alpha
    # make colors uint8
    colors = (colors*255).astype(np.uint8)

    # get the arrows
    arrows = getarrow(lengths, angles, scale=scale, width=width, headlength=headlength, headheight=headheight, offset=R*scale)

    # draw the arrows
    image = ImageDraw.ImageDraw(pil_image, "RGBA")
    for a, c in zip(arrows, colors):
        image.polygon(list(a.flatten()), fill=tuple(c), outline=tuple(c))

    return pil_image
