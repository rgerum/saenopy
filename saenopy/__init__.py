import importlib
import importlib.metadata


__version__ = importlib.metadata.metadata("saenopy")["version"]

_LAZY_ATTRIBUTES = {
    "Solver": ("saenopy.solver", "Solver"),
    "load": ("saenopy.solver", "load"),
    "load_results": ("saenopy.solver", "load_results"),
    "subtract_reference_state": ("saenopy.solver", "subtract_reference_state"),
    "interpolate_mesh": ("saenopy.solver", "interpolate_mesh"),
    "get_stacks": ("saenopy.result_file", "get_stacks"),
    "Result": ("saenopy.result_file", "Result"),
    "get_displacements_from_stacks": ("saenopy.get_deformations", "get_displacements_from_stacks"),
    "load_example": ("saenopy.examples", "load_example"),
    "render_image": ("saenopy.gui.solver.modules.exporter.Exporter", "render_image"),
}

_LAZY_MODULES = {
    "macro": "saenopy.macro",
    "pyTFM": "saenopy.pyTFM",
}

__all__ = sorted([*_LAZY_ATTRIBUTES, *_LAZY_MODULES, "__version__"])


def __getattr__(name):
    if name in _LAZY_ATTRIBUTES:
        module_name, attribute_name = _LAZY_ATTRIBUTES[name]
        value = getattr(importlib.import_module(module_name), attribute_name)
        globals()[name] = value
        return value
    if name in _LAZY_MODULES:
        value = importlib.import_module(_LAZY_MODULES[name])
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
