"""Analysis shared by the GUI and its exported Python scripts."""
import inspect
import copy
from pathlib import Path
import numpy as np
from CompactionAnalyzer.CompactionFunctions import StuctureAnalysisMain


def analyze_orientation(result, orientation_parameters, segmentation_parameters=None, *, save=True):
    """Publish new maps and settings together after calculation and saving succeed."""
    candidate = copy.copy(result)
    _analyze_orientation(candidate, orientation_parameters, segmentation_parameters)
    if save:
        candidate.save()
    result.__dict__.update(candidate.__dict__)
    return result


def _analyze_orientation(result, orientation_parameters, segmentation_parameters=None):
    parameters = orientation_parameters.copy()
    segmentation = dict(segmentation_parameters or {})
    sigma = parameters["sigma_tensor"]
    if parameters["sigma_tensor_type"] == "µm":
        sigma /= result.pixel_size
    common = dict(fiber_list=[result.get_absolute_path_fiber()],
                  cell_list=None if result.image_cell is None else [result.get_absolute_path_cell()],
                  out_list=[str(Path(result.output).with_suffix(""))],
                  scale=result.pixel_size, sigma_tensor=sigma, edge=parameters["edge"],
                  sigma_first_blur=parameters["sigma_first_blur"], mode_saenopy=True)
    if result.analysis_mode == "full":
        # Full GUI results remain in the Saenopy file, as before. These two
        # optional external exports are exposed only in the new axis workflows.
        parameters["SaveNumpy"] = False
        parameters["plotting"] = False
        shell_width = parameters["shell_width"]
        if parameters["shell_width_type"] == "µm":
            shell_width /= result.pixel_size
        values = StuctureAnalysisMain(**common,
            max_dist=parameters["max_dist"], ignore_cell_outline=parameters["ignore_cell_outline"],
            segmention_thres=segmentation["thresh"], seg_gaus1=segmentation["gauss1"],
            seg_gaus2=segmentation["gauss2"], seg_invert=segmentation["invert"],
            angle_sections=parameters["angle_sections"], shell_width=shell_width,
            SaveNumpy=False, SaveExcel=False, plotting=False, return_segmentation=True)
        angle, orientation, coherence, vectors, total, angles, distance, cell_segmentation = values
        result.segmentation = cell_segmentation
        result.segmentation_preview = None
        result.angle_to_x_map = angle
        result.angle_to_y_map = (angle + 180) % 180 - 90
        result.orientation_map = orientation
        result.coherence_map = coherence
        result.orientation_vectors = vectors
        result.results_total = total.to_dict(orient="records")
        result.results_distance = distance.to_dict(orient="records")
        from .axis_statistics import selection_mask, axis_statistics
        result.orientation_parameters = parameters
        result.validity_mask = None
        result.validity_mask = selection_mask(result)
        statistics = axis_statistics(result)
        result.axis_histogram = None
        if statistics is not None:
            stats, result.axis_histogram = statistics
            result.results_total[0].update(stats)
    elif result.analysis_mode in ("point_orientation", "line_orientation"):
        from CompactionAnalyzer import analyze_image
        from CompactionAnalyzer.plotting import line_reference_angle
        from .axis_statistics import axis_statistics
        from .ReferenceOverlay import parameter_line, LINE_COLUMNS
        line = result.analysis_mode == "line_orientation"
        profile_arguments = {}
        profile_outputs = []
        if not line:
            from .point_distance import shell_width_pixels, profile_records
            parameters.setdefault("shell_width", 5.)
            parameters.setdefault("shell_width_type", "µm")
            profile_arguments["shell_width"] = shell_width_pixels(parameters, result.pixel_size)
            profile_outputs = ["distance_profile"]
        suffix = "line" if line else "point"
        deviation = "mean_deviation_line" if line else "mean_angle_point"
        reference = parameter_line(parameters) if line else (parameters.get("reference_x"), parameters.get("reference_y"))
        reference_argument = {"reference_line" if line else "reference_point": reference}
        mask_arguments = {}
        if result.uses_mask:
            mask_arguments = dict(mask_image=result.get_absolute_path_cell(),
                mask_sigma=segmentation.get("mask_sigma", 5),
                mask_threshold=segmentation.get("mask_threshold", .5),
                mask_invert=segmentation.get("mask_invert", False))
        values = analyze_image(result.get_absolute_path_fiber(), mode=result.analysis_mode,
            **mask_arguments,
            **reference_argument, **profile_arguments, scale=result.pixel_size, sigma_tensor=sigma,
            edge=parameters["edge"], sigma_first_blur=parameters["sigma_first_blur"],
            outputs=["angle_x", "angle_y", f"alignment_{suffix}", "coherency", "selection_mask",
                     f"mean_alignment_{suffix}", deviation, "mean_coherency", *profile_outputs])
        result.angle_to_x_map = values["angle_x"]
        result.angle_to_y_map = values["angle_y"]
        result.orientation_map = values[f"alignment_{suffix}"]
        result.coherence_map = values["coherency"]
        result.validity_mask = values["selection_mask"]
        result.orientation_vectors = None
        result.segmentation = None
        result.mask_preview = None
        result.results_distance = [] if line else profile_records(values["distance_profile"])
        summary, result.axis_histogram = axis_statistics(result)
        summary.update({f"Alignment {suffix}": values[f"mean_alignment_{suffix}"],
                        "Mean deviation line" if line else "Mean angle point": values[deviation],
                        "Mean Coherency": values["mean_coherency"]})
        if line:
            summary.update(zip(LINE_COLUMNS, reference.ravel().tolist()))
            summary["Reference line angle (deg)"] = line_reference_angle(reference)
        else:
            summary.update({"Reference x (px)": float(reference[0]), "Reference y (px)": float(reference[1])})
        result.results_total = [summary]
        parameters["SaveNumpy"] = parameters["plotting"] = False
    else:
        # Axis workflows need the companion CompactionAnalyzer update.
        if "mask_sigma" not in inspect.signature(StuctureAnalysisMain).parameters:
            raise RuntimeError("Axis analysis requires CompactionAnalyzer >= 1.0.8 with axis_orientation_masked support. Update CompactionAnalyzer in this Python environment and restart Saenopy.")
        values = StuctureAnalysisMain(**common, analysis_mode=result.analysis_mode,
            mask_sigma=segmentation.get("mask_sigma", 5),
            mask_threshold=segmentation.get("mask_threshold", 0.5),
            mask_invert=segmentation.get("mask_invert", False),
            SaveNumpy=parameters.get("SaveNumpy", True), SaveExcel=False,
            plotting=parameters.get("plotting", True))
        arrays = values["arrays"]
        result.angle_to_x_map = arrays["min_x"]
        result.angle_to_y_map = arrays["min_y"]
        result.coherence_map = arrays["coherency"]
        result.orientation_vectors = np.stack([arrays["min_vx"], arrays["min_vy"]], axis=-1)
        result.validity_mask = values["validity_mask"]
        result.orientation_map = result.get_map_data("alignment_x_axis")
        result.axis_histogram = values["histogram"]
        result.results_total = [values["summary"]]
        result.results_distance = []
        result.segmentation = None  # A validity mask has no radial cell centre.
        result.mask_preview = None  # Display the mask from this completed run.
    result.orientation_parameters = parameters
    result.segmentation_parameters = segmentation
    return result
