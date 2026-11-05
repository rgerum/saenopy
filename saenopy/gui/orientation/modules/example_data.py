"""Create example measurements with their own mode and documented parameters."""
from copy import deepcopy
from pathlib import Path
from .result import get_orientation_files


def get_orientation_example_files(example, exist_overwrite_callback=None):
    results = []
    cancelled = False

    def overwrite(filename):
        nonlocal cancelled
        decision = exist_overwrite_callback(filename)
        cancelled = decision == 0
        return decision

    for variant in example.get("variants", [{}]):
        settings = {**example, **variant}
        output = Path(example["output_path"])
        if variant.get("name"):
            output /= variant["name"]
        measurements = get_orientation_files(
            fiber_list_string=str(settings["input_fiber"]),
            cell_list_string=settings.get("input_cell"),
            output_path=str(output), pixel_size=settings["pixel_size"],
            analysis_mode=settings.get("analysis_mode", "full"),
            exist_overwrite_callback=overwrite if exist_overwrite_callback else None)
        for result in measurements:
            # Respect the user's decision to read an existing result unchanged.
            if not getattr(result, "is_read", False):
                for name in ("orientation_parameters", "segmentation_parameters"):
                    if name in settings:
                        setattr(result, name, deepcopy(settings[name]))
                result.save()
        results.extend(measurements)
        if cancelled:
            break
    return results
