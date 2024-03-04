import datetime
import numpy as np
from saenopy.gui.spheroid.modules.result import ResultSpheroid

class Mesh2D:
    pass

def get_mesh_arrows(params, result):
    data = result.get_data_structure()
    if params["arrows"] not in data["fields"]:
        return None, None, {}, ""
    mesh, field = result.get_field_data(params["arrows"], params["time"]["t"])
    if data["fields"][params["arrows"]]["measure"] == "deformation":
        if mesh is not None and field is not None:
            return mesh, field, params["deformation_arrows"], data["fields"][params["arrows"]]["name"]
        else:
            return None, None, params["deformation_arrows"], data["fields"][params["arrows"]]["name"]
    if data["fields"][params["arrows"]]["measure"] == "force":
        if mesh is not None and field is not None:
            return mesh, field, params["force_arrows"], data["fields"][params["arrows"]]["name"]
        else:
            return None, None, params["force_arrows"], data["fields"][params["arrows"]]["name"]
    return None, None, {}, ""


def get_mesh_extent(params, result): 
    mesh, field = result.get_field_data(params["arrows"], params["time"]["t"]) 
    if mesh is None:
        return None
    else:
        return [mesh.nodes.min(axis=0) * 1e6, mesh.nodes.max(axis=0) * 1e6]
  




def getVectorFieldImage(result, params, use_fixed_contrast_if_available=False, use_2D=False, exporter=None):
    data = result.get_data_structure()
    try:
        image = params["stack"]["image"]
        if use_2D:
            image = 1
        if image and params["time"]["t"] < data["time_point_count"]:
            stack = result.get_image_data(params["time"]["t"], params["stack"]["channel"], params["stack"]["use_reference_stack"])
            if params["stack"]["z_proj"]:
                z_range = [0, 5, 10, 1000][params["stack"]["z_proj"]]
                start = np.clip(params["stack"]["z"] - z_range, 0, stack.shape[3])
                end = np.clip(params["stack"]["z"] + z_range, 0, stack.shape[3])
                im = stack[:, :, :, start:end]
                im = np.max(im, axis=3)
            else:
                im = stack[:, :, :, params["stack"]["z"]]

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

            display_image = [im, data["voxel_size"], params["stack"]["z"] - data["z_slices_count"] / 2]
            if params["stack"]["image"] == 2:
                display_image[2] = -stack.shape[3] / 2
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
