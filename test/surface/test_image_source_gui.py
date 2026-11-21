"""Exercise local-image imports using the real wildcard selection widgets."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
import tifffile


@pytest.fixture
def app(tmp_path, monkeypatch):
    from qtpy import QtCore, QtGui, QtWidgets
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    if font_path.exists():
        QtGui.QFontDatabase.addApplicationFont(str(font_path))
        application.setFont(QtGui.QFont("Segoe UI", 10))
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(tmp_path))
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args: pytest.fail(str(args[-1])))
    yield application
    application.processEvents()


def matrix_result(tmp_path, count=2):
    from saenopy import Result
    from saenopy.stack import Stack
    stacks = [Stack("matrix.tif", (0.5, 0.75, 2), channels=["matrix"],
                    image_filenames=np.array([["matrix.tif"]]), _shape=(8, 12, 1, 1))
              for _ in range(count)]
    result = Result(stack=stacks, stack_reference=stacks[0])
    result.output = str(tmp_path / "result.saenopy")
    return result


def test_add_action_never_emits_a_parameter(app):
    from saenopy.gui.solver.modules.image_source_dialog import ImageSourceChoice
    choice = ImageSourceChoice(None, "channel", 0, [0, "local:0", ImageSourceChoice.ADD],
                               ["matrix", "cell", "Add local images…"])
    parameters, requests = [], []
    choice.valueChanged.connect(parameters.append)
    choice.add_requested.connect(lambda: requests.append(True))
    choice.combobox.setCurrentIndex(1)
    choice.combobox.setCurrentIndex(2)
    assert choice.value() == "local:0"
    assert parameters == ["local:0"]
    assert requests == [True]
    choice.close()


def test_plain_2d_mask_uses_matrix_voxels(app, tmp_path):
    from qtpy import QtWidgets
    from saenopy.gui.solver.modules.image_source_dialog import ImageSourceDialog
    from saenopy import get_custom_mask
    path = tmp_path / "mask.png"
    pixels = np.zeros((8, 12), np.uint8)
    pixels[2:4, 5:8] = 255
    iio.imwrite(path, pixels)
    result = matrix_result(tmp_path)
    dialog = ImageSourceDialog(None, result, "mask")
    dialog.stack_selector.file_changed(str(path))
    assert Path(dialog.pattern.text()) == path
    assert dialog.input_same_voxel_size.value()
    assert not dialog.voxel_inputs[0].isEnabled()
    dialog.show()
    app.processEvents()
    assert dialog.grab().save(str(tmp_path / "mask_import.png"))
    dialog.accept()
    assert dialog.result() == QtWidgets.QDialog.Accepted
    assert dialog.source_key == "local:0"
    mask = get_custom_mask(result, 1, {"custom_mask": dialog.source_key})
    np.testing.assert_array_equal(mask[:, :, 0, 0, 0], pixels)
    assert mask.voxel_size == (0.5, 0.75, 2)
    dialog.close()


def test_saved_sources_restore_in_controls_and_lock_during_tasks(app, tmp_path, monkeypatch):
    from qtpy import QtCore, QtGui, QtWidgets
    from pyvistaqt import QtInteractor
    import qtawesome
    from saenopy import add_image_source, get_custom_mask
    from saenopy.gui.common.PipelineModule import StateEnum
    from saenopy.gui.solver.modules.BatchEvaluate import BatchEvaluate
    monkeypatch.setattr(QtInteractor, "render", lambda self: None)
    monkeypatch.setattr(qtawesome, "icon", lambda *args, **kwargs: QtGui.QIcon())
    result = matrix_result(tmp_path)
    tifffile.imwrite(tmp_path / "cell.tif", np.ones((5, 8, 12), np.uint8), photometric="minisblack")
    iio.imwrite(tmp_path / "mask.png", np.ones((8, 12), np.uint8))
    cell_key = add_image_source(result, str(tmp_path / "cell.tif") + "[z]", name="Cell zoom")
    mask_key = add_image_source(result, str(tmp_path / "mask.png"), kind="mask", name="Exclude bead")
    result.solve_parameters = {"seg_channel": cell_key}
    result.mesh_parameters = {"custom_mask": mask_key, "reference_stack": "next"}
    window = BatchEvaluate()
    regularizer, mesh = window.sub_module_regularize, window.sub_module_mesh
    try:
        regularizer.setResult(result)
        mesh.setResult(result)
        assert regularizer.input_seg_channel.value() == cell_key
        assert mesh.input_custom_mask.value() == mask_key
        assert mesh.input_use_custom_mask.isChecked()
        assert not mesh.input_custom_mask.isHidden()
        assert regularizer.input_seg_channel.valueName() == "Cell zoom"
        assert mesh.input_custom_mask.valueName() == "Exclude bead"
        window.show()
        app.processEvents()
        assert regularizer.grab().save(str(tmp_path / "surface_controls.png"))
        assert mesh.grab().save(str(tmp_path / "mask_controls.png"))
        mesh.input_use_custom_mask.setChecked(False)
        assert mesh.input_custom_mask.isHidden()
        assert result.mesh_parameters_tmp["custom_mask"] is None
        # Disabling affects the next interpolation, not the recorded result.
        assert result.mesh_parameters["custom_mask"] == mask_key
        assert get_custom_mask(result, 0, result.mesh_parameters_tmp) is None
        assert mesh.grab().save(str(tmp_path / "mask_controls_collapsed.png"))
        mesh.input_use_custom_mask.setChecked(True)
        assert not mesh.input_custom_mask.isHidden()
        assert mesh.input_custom_mask.isEnabled()
        assert result.mesh_parameters_tmp["custom_mask"] is None
        mesh.input_custom_mask.setValue(mask_key, send_signal=True)
        assert result.mesh_parameters_tmp["custom_mask"] == mask_key
        for state in (StateEnum.scheduled, StateEnum.running, StateEnum.cancelling):
            regularizer.set_result_state(result, state)
            mesh.set_result_state(result, state)
            regularizer.state_changed(result)
            mesh.state_changed(result)
            assert not regularizer.input_seg_channel.isEnabled()
            assert not mesh.input_custom_mask.isEnabled()
            assert not mesh.input_use_custom_mask.isEnabled()
        regularizer.set_result_state(result, StateEnum.idle)
        mesh.set_result_state(result, StateEnum.idle)
        monkeypatch.setattr(window, "has_scheduled_tasks", lambda: True)
        window.tasks_changed.emit()
        assert not regularizer.input_seg_channel.isEnabled()
        assert not mesh.input_custom_mask.isEnabled()
        assert not mesh.input_use_custom_mask.isEnabled()
        regularizer.add_local_images()
        mesh.add_custom_mask()
        assert len(result.image_sources) == 2
        monkeypatch.setattr(window, "has_scheduled_tasks", lambda: False)
        window.tasks_changed.emit()
        assert regularizer.input_seg_channel.isEnabled()
        assert mesh.input_custom_mask.isEnabled()
        fresh = matrix_result(tmp_path)
        regularizer.setResult(fresh)
        mesh.setResult(fresh)
        assert regularizer.input_seg_channel.value() == 0
        assert mesh.input_custom_mask.value() is None
        assert not mesh.input_use_custom_mask.isChecked()
        assert mesh.input_custom_mask.isHidden()
        assert not mesh.input_custom_mask.isEnabled()
        assert fresh.mesh_parameters_tmp["custom_mask"] is None
    finally:
        for plotter in window.findChildren(QtInteractor):
            plotter.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        app.processEvents()


@pytest.mark.parametrize("packed", [False, True])
def test_initializer_and_mesh_export_restore_and_use_sources(tmp_path, packed):
    from types import SimpleNamespace
    import saenopy
    from saenopy.get_deformations import PivMesh
    from saenopy.gui.solver.modules.Initializer import Initializer
    from saenopy.gui.solver.modules.MeshCreator import MeshCreator
    matrix = tmp_path / "matrix.tif"
    reference = tmp_path / "reference.tif"
    image = np.ones((5, 9, 9), np.uint8)
    tifffile.imwrite(matrix, image, photometric="minisblack")
    tifffile.imwrite(reference, image, photometric="minisblack")
    result = saenopy.get_stacks(str(matrix) + "[z]", tmp_path / "out",
                                (1, 1, 1), reference_stack=str(reference) + "[z]")[0]
    mask_path = tmp_path / "user's mask.png"
    mask = np.zeros((9, 9), np.uint8)
    mask[3:6, 3:6] = 255
    iio.imwrite(mask_path, mask)
    cell_path = tmp_path / "cell.tif"
    tifffile.imwrite(cell_path, image, photometric="minisblack")
    saenopy.add_image_source(result, str(cell_path) + "[z]", voxel_size=(2, 2, 2), name="Cell zoom")
    key = saenopy.add_image_source(result, str(mask_path), kind="mask")
    result.mesh_parameters_tmp = dict(reference_stack="next", element_size=2,
                                      mesh_size="piv", custom_mask=key)
    init_import, init_code = Initializer.get_code(SimpleNamespace(result=result))
    mesh_import, mesh_code = MeshCreator.get_code(SimpleNamespace(result=result))
    if packed:
        for source in result.image_sources:
            for stack in source.stacks:
                stack.pack_files()
        result.save()
        mask_path.unlink()
        cell_path.unlink()
    else:
        Path(result.output).unlink()
    namespace = {"saenopy": saenopy}
    exec(init_import + init_code, namespace)
    restored = namespace["results"][0]
    assert len(restored.image_sources) == 2
    assert restored.image_sources[0].stacks[0].voxel_size == (2, 2, 2)
    nodes = np.stack(np.meshgrid(*([np.arange(-4, 5) * 1e-6] * 3), indexing="ij"), axis=-1).reshape(-1, 3)
    restored.mesh_piv = [PivMesh(nodes=nodes, displacements_measured=np.full_like(nodes, 1e-6))]
    exec(mesh_import + mesh_code, namespace)
    mesh = restored.solvers[0].mesh
    excluded = np.all(np.abs(mesh.nodes[:, :2]) < 1.5e-6, axis=1)
    assert excluded.any() and (~excluded).any()
    assert np.isnan(mesh.displacements_target[excluded]).all()
    assert np.isfinite(mesh.displacements_target[~excluded]).all()
    assert len(saenopy.Result.load(restored.output).image_sources) == 2


def test_mask_series_uses_time_property_and_optional_pixel_size(app, tmp_path):
    from saenopy.gui.solver.modules.image_source_dialog import ImageSourceDialog
    from saenopy import get_custom_mask
    for time in range(2):
        iio.imwrite(tmp_path / f"mask_t{time}.png", np.full((8, 12), time, np.uint8))
    result = matrix_result(tmp_path)
    dialog = ImageSourceDialog(None, result, "mask")
    dialog.stack_selector.file_changed(str(tmp_path / "mask_t0.png"))
    dialog.stack_selector.active.t_prop.setValue("t", send_signal=True)
    dialog.input_time.setValue("series")
    dialog.input_same_voxel_size.setValue(False, send_signal=True)
    assert dialog.voxel_inputs[0].isEnabled()
    dialog.voxel_inputs[0].setValue(1.5)
    dialog.voxel_inputs[1].setValue(2.5)
    assert "{t}" in dialog.pattern.text()
    dialog.accept()
    source = result.image_sources[0]
    assert len(source.stacks) == 2
    mask = get_custom_mask(result, 1, {"custom_mask": dialog.source_key})
    np.testing.assert_array_equal(mask[:, :, 0, 0, 0], np.ones((8, 12)))
    assert mask.voxel_size == (1.5, 2.5, 2)
    dialog.close()


def test_surface_multipage_tiff_has_z_and_layout(app, tmp_path):
    from saenopy.gui.solver.modules.image_source_dialog import ImageSourceDialog
    result = matrix_result(tmp_path)
    path = tmp_path / "cell.tif"
    tifffile.imwrite(path, np.arange(5 * 8 * 12, dtype=np.uint16).reshape(5, 8, 12), photometric="minisblack")
    dialog = ImageSourceDialog(None, result, "surface")
    dialog.stack_selector.file_changed(str(path))
    assert dialog.pattern.text().endswith("[z]")
    dialog.show()
    app.processEvents()
    assert dialog.grab().save(str(tmp_path / "surface_import.png"))
    dialog.accept()
    assert result.image_sources[0].stacks[0].shape == (8, 12, 5, 1)
    assert result.image_sources[0].time_mode == "single"
    dialog.close()
