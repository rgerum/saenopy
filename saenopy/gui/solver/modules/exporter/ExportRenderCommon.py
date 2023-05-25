import datetime
import numpy as np


def get_mesh_arrows(params, result):
    if params["arrows"] == "piv":
        if result is not None:
            mesh = result.mesh_piv[params["time"]["t"]]
            if mesh is not None and mesh.displacements_measured is not None:
                return mesh, mesh.displacements_measured, params["deformation_arrows"], "displacements_measured"
    elif params["arrows"] == "target deformations":
        M = result.solvers[params["time"]["t"]]
        if M is not None:
            return M.mesh, M.mesh.displacements_target, params["deformation_arrows"], "displacements_target"
    elif params["arrows"] == "fitted deformations":
        M = result.solvers[params["time"]["t"]]
        if M is not None:
            return M.mesh, M.mesh.displacements, params["deformation_arrows"], "displacements"
    elif params["arrows"] == "fitted forces":
        M = result.solvers[params["time"]["t"]]
        if M is not None:
            return M.mesh, -M.mesh.forces * M.mesh.regularisation_mask[:, None], params["force_arrows"], "forces"
    return None, None, {}, ""


def get_mesh_extent(params, result):
    if params["arrows"] == "piv":
        mesh = result.mesh_piv[params["time"]["t"]]
        if mesh is not None and mesh.displacements_measured is not None:
            return [mesh.nodes.min(axis=0) * 1e6, mesh.nodes.max(axis=0) * 1e6]
    elif params["arrows"] == "target deformations":
        M = result.solvers[params["time"]["t"]]
        if M is not None:
            return [M.mesh.nodes.min(axis=0) * 1e6, M.mesh.nodes.max(axis=0) * 1e6]
    elif params["arrows"] == "fitted deformations":
        M = result.solvers[params["time"]["t"]]
        if M is not None:
            return [M.mesh.nodes.min(axis=0) * 1e6, M.mesh.nodes.max(axis=0) * 1e6]
    elif params["arrows"] == "fitted forces":
        M = result.solvers[params["time"]["t"]]
        if M is not None:
            return [M.mesh.nodes.min(axis=0) * 1e6, M.mesh.nodes.max(axis=0) * 1e6]
    else:
        M = result.solvers[params["time"]["t"]]
        if M is not None:
            return [M.mesh.nodes.min(axis=0) * 1e6, M.mesh.nodes.max(axis=0) * 1e6]
        else:
            M = result.mesh_piv[params["time"]["t"]]
            if M is not None:
                return [M.mesh.nodes.min(axis=0) * 1e6, M.mesh.nodes.max(axis=0) * 1e6]
    return None


def getVectorFieldImage(result, params, use_fixed_contrast_if_available=False, use_2D=False, exporter=None):
    try:
        image = params["stack"]["image"]
        if use_2D:
            image = 1
        if image and params["time"]["t"] < len(result.stacks):
            if params["stack"]["use_reference_stack"] and result.stack_reference:
                stack = result.stack_reference
            else:
                stack = result.stacks[params["time"]["t"]]
            channel = params["stack"]["channel"]
            if isinstance(channel, str):
                try:
                    channel = result.stacks[0].channels.index(channel)
                except ValueError:
                    channel = 0
            if channel >= len(stack.channels):
                im = stack[:, :, :, params["stack"]["z"], 0]
            else:
                im = stack[:, :, :, params["stack"]["z"], channel]
            if params["stack"]["z_proj"]:
                z_range = [0, 5, 10, 1000][params["stack"]["z_proj"]]
                start = np.clip(params["stack"]["z"] - z_range, 0, stack.shape[2])
                end = np.clip(params["stack"]["z"] + z_range, 0, stack.shape[2])
                im = stack[:, :, :, start:end, channel]
                im = np.max(im, axis=3)

            if params["stack"]["use_contrast_enhance"]:
                if use_fixed_contrast_if_available and params["stack"]["contrast_enhance"]:
                    (min, max) = params["stack"]["contrast_enhance"]
                else:
                    (min, max) = np.percentile(im, (1, 99))
                    if exporter:
                        exporter.vtk_toolbar.contrast_enhance_values.setValue((min, max))
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
