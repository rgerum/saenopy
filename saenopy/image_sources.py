"""Additional, co-centred images for surface segmentation and target masks.

Image arrays use YXZ axes; voxel sizes are XYZ in micrometres. A single
source is reused at every time point. Series are naturally sorted, just like
matrix stacks, and matched by position (the first image of each stack pair).
"""
from pathlib import Path
import re
from typing import List

import natsort
import numpy as np

from .saveable import Saveable
from .stack import Stack, format_glob


class ImageSource(Saveable):
    __save_parameters__ = ["name", "kind", "template", "time_mode", "stacks"]
    name: str
    kind: str
    template: str
    time_mode: str
    stacks: List[Stack]

    def paths_absolute(self, parent):
        from .result_file import make_path_absolute
        self.template = str(make_path_absolute(self.template, Path(parent.output).parent))
        for stack in self.stacks:
            stack.paths_absolute()

    def paths_relative(self, parent):
        from .result_file import make_path_relative
        self.template = str(make_path_relative(self.template, Path(parent.output).parent))
        for stack in self.stacks:
            stack.paths_relative(parent)


def _load_stacks(filename, voxel_size):
    from .result_file import get_channel_placeholder, process_line
    voxel_size = np.asarray(voxel_size, dtype=float)
    if voxel_size.shape != (3,) or not np.all(np.isfinite(voxel_size)) or np.any(voxel_size <= 0):
        raise ValueError("Voxel size must contain three positive finite XYZ values (µm).")
    voxel_size = tuple(voxel_size)
    # Keep the existing Leica reader and its selected channel/time semantics.
    if filename.endswith(".lif"):
        entries = process_line(filename, ".")
        if len(entries) != 1:
            raise ValueError("Select images for exactly one measurement.")
        entry = entries[0]
        if "times" in entry:
            stacks = [Stack(entry["filename"].replace(".lif", f"{{t:{t}}}.lif"), voxel_size)
                      for t in entry["times"]]
        else:
            time = re.search(r"\{t:(\d+)\}", filename)
            template = entry["filename"]
            if time:
                template = template.replace(".lif", f"{{t:{time[1]}}}.lif")
            stacks = [Stack(template, voxel_size)]
        for stack in stacks:
            stack.channels = stack.channels[:1]
        return stacks
    pattern, selected_channel = get_channel_placeholder(filename)
    table, _ = format_glob(pattern)
    if table.empty:
        raise ValueError("No images match the selected path.")
    if table.template.nunique() != 1:
        raise ValueError("Select one measurement; use {t} for time and {z} for slices.")
    for axis in ("z", "c", "t"):
        if axis not in table:
            table[axis] = "0"
    channels = natsort.natsorted(table.c.unique())
    if selected_channel is None and len(channels) > 1:
        raise ValueError("Select exactly one image channel, using a fixed channel path or {c:channel}.")
    if selected_channel is not None:
        if selected_channel not in channels:
            raise ValueError("The selected image channel is missing.")
        channels.remove(selected_channel)
        channels.insert(0, selected_channel)
    # Each added source is one selected channel, independently of the matrix.
    channels = channels[:1]
    table = table[table.c == channels[0]]
    slices = natsort.natsorted(table.z.unique())
    stacks = []
    for time in natsort.natsorted(table.t.unique()):
        frame = table[table.t == time]
        files = []
        for z in slices:
            row = []
            for channel in channels:
                matches = frame[(frame.z == z) & (frame.c == channel)]
                if len(matches) != 1:
                    raise ValueError("Each time point needs every selected slice and channel exactly once.")
                row.append(matches.iloc[0].filename)
            files.append(row)
        stacks.append(Stack(filename.replace("{t}", str(time)), tuple(voxel_size),
                            image_filenames=np.asarray(files), channels=list(channels)))
    return stacks


def load_2d_deformation_masks(filename, voxel_size):
    """Load 2D image-file masks for ``interpolate_mesh(ignore_mask_2d=...)``.

    Return a list of single-plane Stacks: one for a plain image path, or one
    per naturally sorted ``{t}`` time point. Voxel size is XYZ in micrometres,
    usually ``result.stacks[0].voxel_size``. Nonzero pixels exclude complete Z
    columns of target deformations; the mask and matrix share their centre.
    Reuse ``masks[0]`` for every frame, or pass ``masks[i]`` for a time series.
    This loader does not modify or save a Result.
    """
    stacks = _load_stacks(str(Path(filename).absolute()), voxel_size)
    if any(stack.shape[2] != 1 for stack in stacks):
        raise ValueError("Custom masks must be 2D: select one plane per time point; it applies through all Z.")
    return stacks


def add_image_source(result, filename, *, voxel_size=None, name=None,
                     kind="surface", time_mode="single"):
    """Attach a 3D surface stack or 2D exclusion mask to a Result.

    Paths support the existing ``{z}``, ``{t}``, ``{c:channel}`` and TIFF
    ``[z]`` syntax. ``voxel_size=None`` uses the matrix voxel size. Images
    share the matrix centre, even when their shape or voxel size differs.
    ``time_mode='single'`` reuses one stack/image for every frame; ``'series'``
    requires one source per matrix frame (or per deformation for masks).
    The returned ``local:N`` key selects the source in ``seg_channel`` or
    ``mesh_parameters['custom_mask']``. Call ``result.save()`` to persist it.
    """
    if kind not in ("surface", "mask") or time_mode not in ("single", "series"):
        raise ValueError("Choose kind='surface' or 'mask' and time_mode='single' or 'series'.")
    if not result.stacks:
        raise ValueError("Load the matrix images first.")
    voxel = result.stacks[0].voxel_size if voxel_size is None else voxel_size
    filename = str(Path(filename).absolute())
    stacks = (load_2d_deformation_masks(filename, voxel) if kind == "mask"
              else _load_stacks(filename, voxel))
    expected = {len(result.stacks)}
    if kind == "mask":
        expected.add(len(result.mesh_piv))
    if time_mode == "single" and len(stacks) != 1:
        raise ValueError("Select one image/stack to reuse for all time points, or choose a time series.")
    if time_mode == "series" and len(stacks) not in expected:
        raise ValueError(f"Image series has {len(stacks)} time points; expected "
                         + " or ".join(map(str, sorted(expected))) + ".")
    for stack in stacks:
        if kind == "surface" and stack.shape[2] < 2:
            raise ValueError("Surface segmentation needs a 3D stack; select the Z slices.")
    source = ImageSource(name=name or Path(filename).name, kind=kind,
                         template=filename, time_mode=time_mode, stacks=stacks)
    if result.output:
        source.paths_relative(result)
    result.image_sources.append(source)
    return f"local:{len(result.image_sources) - 1}"


def get_image_source(result, selection, index):
    """Resolve an attached source at a matrix/deformation time index."""
    try:
        prefix, number = selection.split(":")
        source_index = int(number)
        if prefix != "local" or source_index < 0:
            raise ValueError
        source = result.image_sources[source_index]
    except (AttributeError, ValueError, IndexError, TypeError) as exc:
        raise ValueError(f"Image source {selection!r} is unavailable; select or add local images.") from exc
    expected = {len(result.stacks)}
    if source.kind == "mask":
        expected.add(len(result.mesh_piv))
    count = len(source.stacks)
    if count != 1 and count not in expected:
        raise ValueError("Image source time series no longer matches the matrix images.")
    if not 0 <= index < len(result.stacks):
        raise ValueError("Image time index is outside the matrix series.")
    return source.stacks[0 if count == 1 else index]


def get_custom_mask(result, index, parameters=None):
    """Get the selected 2D mask Stack, or None when exclusion is disabled."""
    params = result.mesh_parameters if parameters is None else parameters
    selection = (params or {}).get("custom_mask")
    return None if selection is None else get_image_source(result, selection, index)


def image_source_definitions(result):
    """Portable import settings for the GUI's generated Python script."""
    from .result_file import make_path_absolute
    return [dict(filename=str(make_path_absolute(source.template, Path(result.output).parent)),
                 voxel_size=tuple(float(v) for v in source.stacks[0].voxel_size),
                 name=source.name, kind=source.kind, time_mode=source.time_mode)
            for source in result.image_sources]
