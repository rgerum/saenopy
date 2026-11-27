from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile

import saenopy
from saenopy.stack import Stack
from saenopy.gui.solver.modules import path_editor


def test_path_editor_relocates_and_packs_additional_sources(tmp_path, monkeypatch):
    original = tmp_path / "old"
    relocated = tmp_path / "a_much_longer_image_directory"
    original.mkdir()
    relocated.mkdir()
    pixels = np.zeros((5, 7), np.uint8)
    pixels[2, 3] = 255
    for folder in (original, relocated):
        tifffile.imwrite(folder / "matrix_z0.tif", pixels)
        tifffile.imwrite(folder / "mask.tif", pixels)
    matrix = Stack(str(original / "matrix_z{z}.tif"), (1, 1, 1))
    result = saenopy.Result(output=tmp_path / "results" / "sample.saenopy",
                            template=str(original / "matrix_z{z}.tif"), stack=[matrix])
    key = saenopy.add_image_source(result, original / "mask.tif", kind="mask")
    result.mesh_parameters = {"custom_mask": key}

    def value(item):
        return SimpleNamespace(value=lambda: item)

    dialog = SimpleNamespace(
        exec=lambda: True,
        input_folder=value(str(relocated / "matrix_z{z}.tif")),
        input_folder2=None,
        input_sources=[value(str(relocated / "mask.tif"))],
        input_pack=value(True), input_save=value(True))
    monkeypatch.setattr(path_editor, "PathEditor", lambda *args: dialog)
    path_editor.start_path_change(None, result)
    assert result.image_sources[0].template == str(relocated / "mask.tif")
    assert result.image_sources[0].stacks[0].image_filenames[0, 0] == str(relocated / "mask.tif")
    for folder in (original, relocated):
        for image in folder.glob("*.tif"):
            image.unlink()
    loaded = saenopy.Result.load(result.output)
    np.testing.assert_array_equal(loaded.stacks[0][:, :, 0, 0, 0], pixels)
    np.testing.assert_array_equal(saenopy.get_custom_mask(loaded, 0)[:, :, 0, 0, 0], pixels)


def test_path_editor_understands_tiff_time_pages():
    changer = path_editor.PathChanger("masks.tif[t]", "relocated/masks.tif[t]")
    assert changer.change_path("masks.tif[3]") == "relocated/masks.tif[3]"
