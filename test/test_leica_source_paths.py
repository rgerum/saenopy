"""Leica sources keep their container paths and selected data when saved or packed."""
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

import saenopy
from saenopy.gui.common import lif_reader
from saenopy.gui.solver.modules import path_editor
from saenopy.stack import Stack


@pytest.fixture
def leica_reader(monkeypatch):
    opened = []

    class Image:
        dims = SimpleNamespace(y=4, x=7, z=3, t=2)
        channels = 3

        def __init__(self, filename, folder):
            self.filename, self.folder = filename, int(folder)

        def get_frame(self, z, t, c):
            seed = int(self.filename.read_text())
            return (np.arange(28).reshape(4, 7) + seed + self.folder * 1000
                    + t * 100 + c * 10 + z).astype(np.uint16)

    class Reader:
        def __init__(self, filename):
            self.filename = Path(filename)
            self.filename.read_text()  # Resolve the actual file, including missing/wrong-CWD cases.
            opened.append(self.filename)

        def get_image(self, folder):
            return Image(self.filename, folder)

    monkeypatch.setattr(lif_reader, "LifFile", Reader)
    return opened


def leica_result(folder, time_mode="series"):
    folder.mkdir()
    (folder / "measurement.lif").write_text("5")
    template = str(folder / "measurement{f:2}{c:1}.lif")
    stacks = [Stack(template.replace(".lif", f"{{t:{t}}}.lif"), (1, 2, 3)) for t in range(2)]
    reference = Stack(str(folder / "measurement{f:2}{c:2}{t:0}.lif"), (1, 2, 3))
    result = saenopy.Result(output=folder / "result.saenopy", template=template,
                            stack=stacks, stack_reference=reference, time_delta=1)
    source = str(folder / ("measurement{f:2}{c:2}{t:1}.lif" if time_mode == "single"
                          else "measurement{f:2}{c:2}.lif"))
    saenopy.add_image_source(result, source, voxel_size=(.5, .75, 2), time_mode=time_mode)
    return result


def all_stacks(result):
    return result.stacks + [result.stack_reference] + result.image_sources[0].stacks


def assert_pixels(result, seed):
    for stack in all_stacks(result):
        assert stack.shape == (4, 7, 3, len(stack.channels))
        for channel, selected in enumerate(stack.channels):
            expected = (np.arange(28).reshape(4, 7, 1) + seed + 2000
                        + stack.leica_time * 100 + int(selected) * 10 + np.arange(3))
            np.testing.assert_array_equal(stack[:, :, 0, :, channel], expected)
            np.testing.assert_array_equal(stack[:, :, 0, 2, channel], expected[:, :, 2])
    source = result.image_sources[0]
    assert all(stack.channels == ["2"] and tuple(stack.voxel_size) == (.5, .75, 2)
               for stack in source.stacks)
    assert result.stacks[0].channels == ["1", "0", "2"]
    assert result.stack_reference.channels == ["2", "0", "1"]


@pytest.mark.parametrize("time_mode", ["single", "series"])
def test_leica_save_as_and_relocation_resolve_before_open(tmp_path, monkeypatch, leica_reader, time_mode):
    original = tmp_path / "original"
    result = leica_result(original, time_mode)
    result.save()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    # A matching basename in the working directory must never supply the data.
    (elsewhere / "measurement.lif").write_text("900")
    monkeypatch.chdir(elsewhere)
    leica_reader.clear()
    loaded = saenopy.Result.load(result.output)
    assert not leica_reader  # Opening waits until Result.on_load has assigned the base path.
    assert_pixels(loaded, 5)
    loaded.save(original / "other" / "copy.saenopy")
    relocated = tmp_path / "relocated"
    shutil.move(str(original), relocated)
    leica_reader.clear()
    moved = saenopy.Result.load(relocated / "other" / "copy.saenopy")
    assert not leica_reader
    assert_pixels(moved, 5)
    assert all(path.resolve() == relocated / "measurement.lif" for path in leica_reader)
    assert moved.image_sources[0].time_mode == time_mode


def test_leica_path_editor_relocates_packs_and_reloads_without_files(tmp_path, monkeypatch, leica_reader):
    result = leica_result(tmp_path / "original")
    assert_pixels(result, 5)  # Populate cached readers before changing paths.
    relocated = tmp_path / "a_longer_relocated_directory"
    relocated.mkdir()
    (relocated / "new.lif").write_text("50")

    def value(item):
        return SimpleNamespace(value=lambda: item)

    dialog = SimpleNamespace(
        exec=lambda: True,
        input_folder=value(str(relocated / "new{f:2}{c:1}.lif")),
        input_folder2=value(str(relocated / "new{f:2}{c:2}{t:0}.lif")),
        input_sources=[value(str(relocated / "new.lif"))],
        input_pack=value(True), input_save=value(True))
    monkeypatch.setattr(path_editor, "PathEditor", lambda *args: dialog)
    path_editor.start_path_change(None, result)
    assert_pixels(result, 50)
    assert result.image_sources[0].template.endswith("new{f:2}{c:2}.lif")
    for filename in (tmp_path / "original" / "measurement.lif", relocated / "new.lif"):
        filename.unlink()
    leica_reader.clear()
    loaded = saenopy.Result.load(result.output)
    assert_pixels(loaded, 50)
    for stack in all_stacks(loaded):
        stack.pack_files()  # Packing an already portable result needs no original container.
    assert not leica_reader


def test_leica_path_change_retains_packed_data(tmp_path, leica_reader):
    result = leica_result(tmp_path / "original")
    stack = result.stacks[0]
    stack.pack_files()
    packed = stack.packed_files
    changer = path_editor.PathChanger(result.template, tmp_path / "missing.lif")
    changer.stack_change(stack)
    assert stack.packed_files is packed
    assert stack._leica_file is None
    np.testing.assert_array_equal(stack[:, :, 0, :, 0],
                                  np.arange(28).reshape(4, 7, 1) + 2015 + np.arange(3))
