import datetime
import numpy as np

def get_current_arrow_data(params, result):
    M = None
    field = None
    center = None
    name = ""
    colormap = None
    factor = None
    scale_max = params["deformation_arrows"]["scale_max"] if params["deformation_arrows"]["autoscale"] else None
    stack_min_max = None
    skip = params["deformation_arrows"]["skip"]
    alpha = params["deformation_arrows"]["arrow_opacity"] if params["arrows"] != "fitted forces" else params["force_arrows"]["arrow_opacity"]

    if params["arrows"] == "piv":
        if result is None:
            M = None
        else:
            M = result.mesh_piv[params["time"]["t"]]

        if M is not None:
            if M.hasNodeVar("U_measured"):
                # showVectorField2(self, M, "U_measured")
                field = M.getNodeVar("U_measured")
                factor = 0.1 * params["deformation_arrows"]["arrow_scale"]
                name = "U_measured"
                colormap = params["deformation_arrows"]["colormap"]
                stack_min_max = [M.R.min(axis=0) * 1e6, M.R.max(axis=0) * 1e6]
    elif params["arrows"] == "target deformations":
        M = result.solver[params["time"]["t"]]
        # showVectorField2(self, M, "U_target")
        if M is not None:
            field = M.U_target
            factor = 0.1 * params["deformation_arrows"]["arrow_scale"]
            name = "U_target"
            colormap = params["deformation_arrows"]["colormap"]
            stack_min_max = [M.R.min(axis=0) * 1e6, M.R.max(axis=0) * 1e6]
    elif params["arrows"] == "fitted deformations":
        M = result.solver[params["time"]["t"]]
        if M is not None:
            field = M.U
            factor = 0.1 * params["deformation_arrows"]["arrow_scale"]
            name = "U"
            colormap = params["deformation_arrows"]["colormap"]
            stack_min_max = [M.R.min(axis=0) * 1e6, M.R.max(axis=0) * 1e6]
    elif params["arrows"] == "fitted forces":
        M = result.solver[params["time"]["t"]]
        if M is not None:
            center = None
            if params["force_arrows"]["use_center"] is True:
                center = M.getCenter(mode="Force")
            field = -M.f * M.reg_mask[:, None]
            factor = 0.15 * params["force_arrows"]["arrow_scale"]
            name = "f"
            colormap = params["force_arrows"]["colormap"]
            scale_max = params["force_arrows"]["scale_max"] if params["force_arrows"]["autoscale"] else None
            stack_min_max = [M.R.min(axis=0) * 1e6, M.R.max(axis=0) * 1e6]
            skip = params["force_arrows"]["skip"]
    else:
        # get min/max of stack
        M = result.solver[params["time"]["t"]]
        if M is not None:
            stack_min_max = [M.R.min(axis=0) * 1e6, M.R.max(axis=0) * 1e6]
        else:
            M = result.mesh_piv[params["time"]["t"]]
            if M is not None:
                stack_min_max = [M.R.min(axis=0) * 1e6, M.R.max(axis=0) * 1e6]
            else:
                stack_min_max = None
    return M, field, center, name, colormap, factor, scale_max, stack_min_max, skip, alpha


def getVectorFieldImage(result, params, use_fixed_contrast_if_available=False, use_2D=False):
    try:
        image = params["stack"]["image"]
        if use_2D:
            image = 1
        if image and params["time"]["t"] < len(result.stack):
            if params["stack"]["use_reference_stack"] and result.stack_reference:
                stack = result.stack_reference
            else:
                stack = result.stack[params["time"]["t"]]
            channel = params["stack"]["channel"]
            if isinstance(channel, str):
                channel = result.stack[0].channels.index(channel)
            if channel >= len(stack.channels):
                im = stack[:, :, :, params["stack"]["z"], 0]
            else:
                im = stack[:, :, :, params["stack"]["z"], channel]
            if params["stack"]["z_proj"]:
                z_range = [0, 5, 10, 1000][params["stack"]["z_proj"]]
                start = np.clip(params["stack"]["z"] - z_range, 0,
                                stack.shape[2])
                end = np.clip(params["stack"]["z"] + z_range, 0, stack.shape[2])
                im = stack[:, :, :, start:end, params["stack"]["channel"]]
                im = np.max(im, axis=3)

            if params["stack"]["contrast_enhance"]:
                print("contreast!")
                if use_fixed_contrast_if_available and getattr(result, "contrast_enhance_values", None):
                    (min, max) = result.contrast_enhance_values
                    print("current contrast", min, max)
                else:
                    print("new contrast")
                    (min, max) = np.percentile(im, (1, 99))
                    result.contrast_enhance_values = (min, max)
                im = im.astype(np.float32) - min
                im = im.astype(np.float64) * 255 / (max - min)
                im = np.clip(im, 0, 255).astype(np.uint8)

            display_image = [im, stack.voxel_size, params["stack"]["z"] - stack.shape[2] / 2]
            if params["stack"]["image"] == 2:
                display_image[2] = -stack.shape[2] / 2
        else:
            display_image = None
    except FileNotFoundError:
        display_image = None
    return display_image


def get_time_text(params, result):
    print("time", float(params["time"]["t"] * result.time_delta) + params["time"]["start"], float(params["time"]["t"] * result.time_delta), params["time"]["start"])
    return formatTimedelta(datetime.timedelta(seconds=float(params["time"]["t"] * result.time_delta) + params["time"]["start"]),
                           params["time"]["format"])

def formatTimedelta(t: datetime.timedelta, fmt: str) -> str:
    sign = 1
    if t.total_seconds() < 0:
        sign = -1
        t = -t
    seconds = t.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    parts = {"d": t.days, "H": hours, "M": minutes, "S": seconds,
             "s": t.total_seconds(), "m": t.microseconds // 1000, "f": t.microseconds}

    max_level = None
    if fmt.find("%d") != -1:
        max_level = "d"
    elif fmt.find("%H") != -1:
        max_level = "H"
    elif fmt.find("%M") != -1:
        max_level = "M"
    elif fmt.find("%S") != -1:
        max_level = "S"
    elif fmt.find("%m") != -1:
        max_level = "m"
    elif fmt.find("%f") != -1:
        max_level = "f"

    fmt = fmt.replace("%d", str(parts["d"]))
    if max_level == "H":
        fmt = fmt.replace("%H", "%d" % (parts["H"] + parts["d"] * 24))
    else:
        fmt = fmt.replace("%H", "%02d" % parts["H"])
    if max_level == "M":
        fmt = fmt.replace("%M", "%2d" % (parts["M"] + parts["H"] * 60 + parts["d"] * 60 * 24))
    else:
        fmt = fmt.replace("%M", "%02d" % parts["M"])

    if max_level == "S":
        fmt = fmt.replace("%S", "%d" % parts["s"])
    else:
        fmt = fmt.replace("%S", "%02d" % parts["S"])

    if max_level == "m":
        fmt = fmt.replace("%m", "%3d" % (parts["m"] + parts["s"] * 1000))
    else:
        fmt = fmt.replace("%m", "%03d" % parts["m"])
    if max_level == "f":
        fmt = fmt.replace("%f", "%6d" % (parts["f"] + parts["s"] * 1000 * 1000))
    else:
        fmt = fmt.replace("%f", "%06d" % parts["f"])
    if sign == -1:
        for i in range(len(fmt)):
            if fmt[i] != " ":
                break
        if i == 0:
            fmt = "-" + fmt
        else:
            fmt = fmt[:i - 1] + "-" + fmt[i:]
    return fmt

