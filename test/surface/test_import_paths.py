"""Exercise browsing and TIFF time selection through the shared GUI widgets."""
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tifffile

from test_image_source_gui import app


def test_folder_browse_preserves_templates(app, tmp_path, monkeypatch):
    from qtpy import QtWidgets
    import saenopy
    from saenopy.gui.solver.modules import path_editor

    original, relocated = tmp_path / "old", tmp_path / "relocated"
    for folder in (original, relocated):
        folder.mkdir()
        for t in range(2):
            iio.imwrite(folder / f"mask_t{t}.png", np.full((5, 7), t, np.uint8))
            tifffile.imwrite(folder / f"cell_t{t}.tiff", np.full((2, 5, 7), t + 5, np.uint8),
                             photometric="minisblack")
            for z in range(2):
                for c in range(2):
                    pixels = np.full((5, 7), 100 * t + 10 * c + z, np.uint8)
                    tifffile.imwrite(folder / f"matrix_t{t}_z{z}_c{c}.tif", pixels)
                    if t == 0:
                        tifffile.imwrite(folder / f"reference_z{z}_c{c}.tif", pixels)
    result = saenopy.get_stacks(original / "matrix_t{t}_z{z}_c{c:0}.tif", tmp_path / "r",
                                (1, 1, 1), time_delta=1,
                                reference_stack=original / "reference_z{z}_c{c:0}.tif")[0]
    mask = saenopy.add_image_source(result, original / "mask_t{t}.png", kind="mask", time_mode="series")
    surface = saenopy.add_image_source(result, original / "cell_t{t}.tiff[z]", time_mode="series")
    dialog = path_editor.PathEditor(None, result)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", lambda *args: str(relocated))
    for widget in [dialog.input_folder, dialog.input_folder2, *dialog.input_sources]:
        filename = Path(widget.value()).name
        widget.button.click()
        assert Path(widget.value()) == relocated / filename
    dialog.input_save.setValue(True)
    monkeypatch.setattr(dialog, "exec", lambda: QtWidgets.QDialog.Accepted)
    monkeypatch.setattr(path_editor, "PathEditor", lambda *args: dialog)
    path_editor.start_path_change(None, result)
    for image in original.iterdir():
        image.unlink()
    loaded = saenopy.Result.load(result.output)
    np.testing.assert_array_equal(loaded.stacks[1][0, 0, 0, :, 1], [110, 111])
    np.testing.assert_array_equal(loaded.stack_reference[0, 0, 0, :, 1], [10, 11])
    assert saenopy.get_image_source(loaded, mask, 1)[:, :, 0, 0, 0].all()
    np.testing.assert_array_equal(saenopy.get_image_source(loaded, surface, 1)[:, :, 0, :, 0], 6)
    # A manually edited filename is retained by the next folder selection too.
    widget = dialog.input_sources[0]
    widget.line.setText(str(original / "manual_t{t}.png"))
    widget.line.editingFinished.emit()
    widget.button.click()
    assert Path(widget.value()) == relocated / "manual_t{t}.png"
    dialog.close()


def test_shared_tiff_selector_emits_time_pages(app, tmp_path):
    from qtpy import QtWidgets
    from saenopy.gui.common.stack_selector import StackSelector
    from saenopy.stack import format_glob

    for z in range(2):
        tifffile.imwrite(tmp_path / f"volume_z{z}.tiff",
                         np.stack([np.full((5, 7), t * 10 + z, np.uint8) for t in range(2)]),
                         photometric="minisblack")
    holder = QtWidgets.QWidget()
    selector = StackSelector(QtWidgets.QVBoxLayout(holder), "matrix", use_time=True)
    patterns = []
    selector.glob_string_changed.connect(lambda _, value: patterns.append(value))
    selector.file_changed(str(tmp_path / "volume_z0.tiff"))
    selector.active.z_prop.setValue("z", send_signal=True)
    selector.active.t_prop.setValue("tiff pages", send_signal=True)
    assert patterns[-1].endswith("volume_z{z}.tiff[t]")
    table, _ = format_glob(patterns[-1])
    assert len(table) == 4 and set(table.t) == {0, 1} and set(table.z) == {"0", "1"}
    assert selector.get_t_count() == selector.get_z_count() == 2
    np.testing.assert_array_equal(selector.get_image(1, 1), 11)
    from saenopy import get_stacks
    result = get_stacks(patterns[-1], tmp_path / "results", (1, 1, 1), time_delta=1)[0]
    np.testing.assert_array_equal(result.stacks[1][0, 0, 0, :, 0], [10, 11])
    holder.close()


def test_shared_tiff_selector_imports_channel_pages(app, tmp_path):
    from qtpy import QtWidgets
    from saenopy import get_stacks
    from saenopy.gui.common.stack_selector import StackSelector

    for t in range(2):
        for z in range(2):
            tifffile.imwrite(tmp_path / f"volume_t{t}_z{z}.tiff",
                             np.stack([np.full((5, 7), 100 * t + 10 * c + z, np.uint8) for c in range(2)]),
                             photometric="minisblack")
    holder = QtWidgets.QWidget()
    selector = StackSelector(QtWidgets.QVBoxLayout(holder), "matrix", use_time=True)
    patterns = []
    selector.glob_string_changed.connect(lambda _, value: patterns.append(value))
    selector.file_changed(str(tmp_path / "volume_t0_z0.tiff"))
    selector.active.t_prop.setValue("t", send_signal=True)
    selector.active.c_prop.setValue("tiff pages", send_signal=True)
    page_choice = next(prop for prop in selector.active.property_selectors if prop.name == "tiff pages")
    for page in (0, 1):
        page_choice.setValue(str(page), send_signal=True)
        assert patterns[-1].endswith("volume_t{t}_z{z}.tiff[{c:" + str(page) + "}]")
        result = get_stacks(patterns[-1], tmp_path / f"results{page}", (1, 1, 1), time_delta=1)[0]
        assert result.stacks[1].channels == [str(page), str(1 - page)]
        np.testing.assert_array_equal(result.stacks[1][0, 0, 0, :, 0], [100 + 10 * page, 101 + 10 * page])
        np.testing.assert_array_equal(selector.get_image(1, 1), 101 + 10 * page)
    holder.close()
