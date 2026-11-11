"""Image-axis statistics from saved maps; no tensor calculation or file writes."""
import numpy as np


def selection_mask(result):
    shape = result.angle_to_x_map.shape
    if result.validity_mask is not None:
        return np.asarray(result.validity_mask, dtype=bool)
    if result.analysis_mode == "full":
        # Without a saved cell mask the region cannot be identified safely,
        # especially with ignore_cell_outline. Leave those metrics unavailable.
        if result.segmentation is None:
            return None
        edge = (result.orientation_parameters or {}).get("edge", 0)
        crop = slice(edge, -edge) if edge else slice(None)
        mask = ~np.asarray(result.segmentation["mask"], dtype=bool)[crop, crop]
        if mask.shape != shape:
            return None
        parameters = result.orientation_parameters or {}
        if parameters.get("max_dist") or parameters.get("min_dist"):
            x, y = result.segmentation["centroid"]
            yy, xx = np.indices(shape)
            distance = np.hypot(xx - x + edge, yy - y + edge)
            if parameters.get("max_dist"):
                mask &= distance <= parameters["max_dist"]
            if parameters.get("min_dist"):
                mask &= distance >= parameters["min_dist"]
        return mask
    if result.analysis_mode == "axis_orientation_masked":
        return None
    return np.ones(shape, dtype=bool)


def axis_statistics(result):
    """Return unweighted means and the common 36-bin X/Y histogram, or None."""
    if result.angle_to_x_map is None:
        return None
    mask = selection_mask(result)
    if mask is None:
        return None
    x = np.asarray(result.angle_to_x_map)
    y = result.angle_to_y_map
    if y is None:
        y = (x + 180) % 180 - 90
    valid = mask & np.isfinite(x) & np.isfinite(y)
    stats = {"Selected pixels": int(mask.sum()), "Valid pixels": int(valid.sum())}
    counts = []
    bins = np.linspace(-90, 90, 37)
    for axis, angles in (("x", x), ("y", y)):
        deviation = float(np.abs(angles[valid]).mean()) if valid.any() else float("nan")
        stats[f"Mean absolute deviation {axis}"] = deviation
        stats[f"Alignment {axis}"] = 1 - deviation / 45
        counts.append(np.histogram(angles[valid], bins=bins)[0])
    return stats, np.column_stack([bins[:-1], bins[1:], *counts])
