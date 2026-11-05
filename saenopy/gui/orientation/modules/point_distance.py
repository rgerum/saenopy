"""Compute point-distance courses from existing maps, including older results."""
import numpy as np
from CompactionAnalyzer.distance import point_distance_profile


PROFILE_COLUMNS = {
    'distance_start_px': 'Shell_start (px)',
    'distance_end_px': 'Shell_end (px)',
    'distance_px': 'Shell_mid (px)',
    'distance_um': 'Shell_mid (µm)',
    'mean_alignment_point': 'Point alignment (shell)',
    'mean_angle_point': 'Point angular deviation (shell)',
    'mean_coherency': 'Point coherency (shell)',
    'valid_pixels': 'Valid pixels',
    'coherency_pixels': 'Coherency pixels',
}


def shell_width_pixels(parameters, pixel_size):
    width = parameters.get('shell_width', 5.)
    unit = parameters.get('shell_width_type', 'µm')
    if unit not in ('pixel', 'µm'):
        raise ValueError('shell_width_type must be pixel or µm')
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError('pixel_size must be positive and finite')
    return width / pixel_size if unit == 'µm' else width


def profile_records(profile):
    return [{column: profile[key][i].item() for key, column in PROFILE_COLUMNS.items()}
            for i in range(len(profile['distance_px']))]


def ensure_point_distance(result):
    """Fill missing profiles in memory; loading never rewrites the result file."""
    if result.analysis_mode != 'point_orientation' or result.results_distance:
        return
    parameters = result.orientation_parameters or {}
    if (result.orientation_map is None or result.coherence_map is None
            or any(parameters.get(key) is None for key in ('reference_x', 'reference_y'))):
        return
    edge = parameters.get('edge', 0)
    center = (parameters['reference_x']-edge, parameters['reference_y']-edge)
    if result.uses_mask and result.validity_mask is None:
        return  # An old masked result must not silently become whole-image analysis.
    profile = point_distance_profile(result.orientation_map, result.coherence_map, center,
        mask=result.validity_mask, shell_width=shell_width_pixels(parameters, result.pixel_size),
        scale=result.pixel_size)
    result.results_distance = profile_records(profile)
