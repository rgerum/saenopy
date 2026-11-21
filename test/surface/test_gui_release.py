"""Offscreen Qt integration: official result loading and the actual export handler."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from pathlib import Path
import appdirs
import pytest

pytestmark = pytest.mark.usefixtures("isolated_qt_settings")


def test_official_file_in_gui_and_full_code_export(tmp_path, monkeypatch):
    from qtpy import QtWidgets, QtCore
    from saenopy.gui.solver.modules.BatchEvaluate import BatchEvaluate
    # Test Qt state, geometry and export independently of a headless OpenGL
    # driver. Set SAENOPY_TEST_RENDER=1 for an interactive rendering check.
    if os.environ.get("SAENOPY_TEST_RENDER") != "1":
        from pyvistaqt import QtInteractor
        monkeypatch.setattr(QtInteractor, "render", lambda self: None)

    source = Path(appdirs.user_data_dir("saenopy", "rgerum")) / "1_ClassicSingleCellTFM" / "example_output" / "Pos007_S001_z{z}_ch{c00}_eval.saenopy"
    if not source.exists():
        pytest.skip("local official example is not installed")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    class TestWindow(BatchEvaluate):
        settings_key = "saenopy_surface_release_test"

    window = TestWindow()
    regularizer = window.sub_module_regularize
    from saenopy.surface_regularization import DEFAULT_SEG_CHANNEL
    assert regularizer.input_seg_channel.value() == DEFAULT_SEG_CHANNEL == 0
    assert regularizer.input_thr_factor.value() == pytest.approx(0.6)
    assert not hasattr(regularizer, "input_physical_normalization")
    assert regularizer.input_dilate_layers.spin_box.minimumWidth() >= 90
    def fail_dialog(*args):
        pytest.fail(str(args[-1]))
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", fail_dialog)
    try:
        window.load_from_path([str(source)])
        assert len(window.data) == 1
        window.list.setCurrentRow(0)
        app.processEvents()
        result = window.data[0][2]
        assert result.___save_version__ == "1.9"
        assert result.solve_parameters["physical_normalization"] is False
        assert result.solve_parameters_tmp["physical_normalization"] is True
        assert result.solve_parameters_tmp["surface"] is False
        assert window.sub_module_regularize.input_seg_channel.value() == result.solve_parameters_tmp["seg_channel"]
        assert not hasattr(window.sub_module_regularize, "input_surface_alpha")
        # Re-select after showing Surface: loading an official Classic file must
        # not inherit the previously visible Surface checkbox state.
        window.sub_module_regularize.input_surface.setValue(True)
        window.load_from_path([str(source)])
        window.list.setCurrentRow(len(window.data) - 1)
        app.processEvents()
        current = window.data[-1][2]
        # Duplicate-file handling may retain the selected instance; explicitly
        # load a fresh Result for the selection-transition assertion.
        import saenopy
        fresh = saenopy.Result.load(str(source))
        window.sub_module_regularize.setResult(fresh)
        assert fresh.solve_parameters["physical_normalization"] is False
        assert fresh.solve_parameters_tmp["physical_normalization"] is True
        window.list.setCurrentRow(0)
        window.set_current_result.emit(result)
        destination = tmp_path / "full_export.py"
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a: (str(destination), ""))
        window.generate_code()
        code = destination.read_text(encoding="utf-8")
        compile(code, str(destination), "exec")
        assert "fit_result" in code
        assert "interpolate_mesh" in code
        assert "get_displacements_from_stacks" in code
        assert "get_stacks" in code
        assert "surface_alpha" not in code
        assert "'physical_normalization': True" in code
        assert "'cg_maxiter_factor': 2" in code
        assert "'solver_precision': 1e-18" in code

        # Preview uses the current inputs and opens the enabled Forces view.
        import numpy as np
        import importlib
        from saenopy import surface_regularization as sr
        from saenopy.gui.common.PipelineModule import StateEnum
        module = importlib.import_module("saenopy.gui.solver.modules.Regularizer")
        calls = []
        def fake_segment(res, index, params):
            from types import SimpleNamespace
            calls.append(dict(params))
            body = np.zeros((8, 8, 8), bool)
            body[2:6, 2:6, 2:6] = True
            return SimpleNamespace(voxel_size=(1., 1., 1.)), body, body, np.zeros((1, 3)), 42.0
        monkeypatch.setattr(module, "segment_with_params", fake_segment)
        monkeypatch.setattr(sr, "surface_node_mask", lambda nodes, *a, **kw: np.ones(len(nodes), bool))
        monkeypatch.setattr(window.tab5, "update_display", lambda: None)
        regularizer.input_thr_factor.setValue(0.6, send_signal=True)
        regularizer.input_dilate_layers.setValue(2, send_signal=True)
        window.tabs.setCurrentIndex(0)
        regularizer.preview_segmentation()
        assert len(calls) == 1
        assert calls[0]["seg_threshold_factor"] == pytest.approx(0.6)
        assert calls[0]["seg_dilate_layers"] == 2
        assert window.tabs.currentWidget() is window.tab5.tab.parent()
        assert window.tab5.vtk_toolbar.use_surface.value() is True
        assert window.tabs.isTabEnabled(window.tabs.currentIndex())
        assert "segmentation failed" not in regularizer.input_button_preview_text.text()
        assert result.solvers[0].mesh._segmentation_preview is not None
        # Failed threshold calculation must retain and label the previous
        # preview, without applying partial geometry to the solver.
        previous_preview = result.solvers[0].mesh._segmentation_preview
        previous_mask = result.solvers[0].mesh.cell_boundary_mask
        if previous_mask is not None:
            previous_mask = previous_mask.copy()
        def fail_segment(*args, **kwargs):
            raise sr.SegmentationError("Li threshold timed out; no new mask created.")
        monkeypatch.setattr(module, "segment_with_params", fail_segment)
        regularizer.preview_segmentation()
        assert "timed out" in regularizer.input_button_preview_text.text()
        assert "unchanged" in regularizer.input_button_preview_text.text()
        assert result.solvers[0].mesh._segmentation_preview is previous_preview
        np.testing.assert_array_equal(result.solvers[0].mesh.cell_boundary_mask, previous_mask)
        monkeypatch.setattr(module, "segment_with_params", fake_segment)
        window.show()
        app.processEvents()
        spin = regularizer.input_dilate_layers.spin_box
        assert spin.width() >= 90
        editor = spin.findChild(QtWidgets.QLineEdit)
        assert editor.width() > editor.fontMetrics().horizontalAdvance("99999")
        factor_editor = regularizer.input_thr_factor.line_edit
        assert factor_editor.width() >= 90
        assert regularizer.grab().save(str(tmp_path / "regularizer_controls.png"))

        # A short window and larger font must scroll, not squash the Surface
        # inputs. Test actual allocated heights, not only minimum widths.
        font = window.font()
        font.setPointSize(12)
        window.setFont(font)
        window.resize(1200, 650)
        app.processEvents()
        scroll = window.parameter_scroll
        assert scroll.verticalScrollBar().maximum() > 0
        scroll.ensureWidgetVisible(regularizer.input_button_preview)
        app.processEvents()
        for control in (spin, factor_editor, regularizer.input_button_preview):
            assert control.height() >= control.minimumSizeHint().height()
            assert control.height() >= control.fontMetrics().height()
        rows = [regularizer.input_surface, regularizer.input_seg_channel,
                regularizer.input_thr_method, regularizer.input_thr_factor,
                regularizer.input_dilate_layers, regularizer.input_button_preview]
        # The compact form may share or wrap rows, but must never clip
        # labels, overlap controls or place them outside the Surface group.
        group = regularizer.surface_parameters
        rectangles = []
        for control in rows:
            rect = QtCore.QRect(control.mapTo(group, QtCore.QPoint()), control.size())
            assert group.rect().contains(rect)
            assert all(not rect.intersects(other) for other in rectangles)
            rectangles.append(rect)
            if hasattr(control, "label"):
                label = control.label
                assert label.width() >= label.fontMetrics().horizontalAdvance(label.text())
        assert regularizer.grab().save(str(tmp_path / "regularizer_controls_large_font.png"))

        # Scheduled, running and cancelling fits block even direct handler calls.
        for state in (StateEnum.scheduled, StateEnum.running, StateEnum.cancelling):
            regularizer.set_result_state(result, state)
            regularizer.processing_state_changed.emit(result)
            assert not regularizer.input_button_preview.isEnabled()
            assert not regularizer.input_thr_factor.isEnabled()
            regularizer.preview_segmentation()
            assert len(calls) == 1
        regularizer.set_result_state(result, StateEnum.idle)
        regularizer.processing_state_changed.emit(result)
        assert regularizer.input_button_preview.isEnabled()

        # Any queued pipeline task also blocks preview and re-enables it on completion.
        monkeypatch.setattr(window, "run_next", lambda: None)
        window.addTask(None, result, {}, "test")
        assert not regularizer.input_button_preview.isEnabled()
        regularizer.preview_segmentation()
        assert len(calls) == 1
        window.run_finished()
        assert regularizer.input_button_preview.isEnabled()

        # A fresh fit opens Forces and enables Fitted Deformations immediately,
        # before the first iteration. Live fields never alias worker arrays.
        from saenopy.gui.solver.modules.live_fit import displayed_solver, solver_snapshot
        result.solvers[0].regularisation_results = None
        result.solvers[0].relrec = None
        result.solvers[0].mesh.cell_boundary_mask = None
        monkeypatch.setattr(window.tab4, "update_display", lambda: None)
        regularizer.start_process()
        assert window.tabs.currentWidget() is window.tab5.tab.parent()
        for viewer in (window.tab4, window.tab5):
            assert viewer.checkTabEnabled(result)
            assert window.tabs.isTabEnabled(window.tabs.indexOf(viewer.tab.parent()))
        snapshot = solver_snapshot(result.solvers[0])
        regularizer.live_field_ready.emit(result, 0, snapshot)
        assert displayed_solver(result, 0) is snapshot
        assert not np.shares_memory(snapshot.mesh.forces, result.solvers[0].mesh.forces)
        assert not np.shares_memory(snapshot.mesh.displacements, result.solvers[0].mesh.displacements)
        window.tab4.current_result_plotted = True
        window.tab4.current_tab_selected = False
        regularizer.show_live_field(result, 0, snapshot)
        assert not window.tab4.current_result_plotted
        regularizer.set_result_state(result, StateEnum.failed)
        regularizer.processing_state_changed.emit(result)
        regularizer.show_live_field(result, 0, snapshot)
        assert displayed_solver(result, 0) is result.solvers[0]
        assert not result._live_fit_solvers
        window.run_finished()
        regularizer.set_result_state(result, StateEnum.idle)
        regularizer.processing_state_changed.emit(result)

        # Loading a legacy result never changes its stored fit settings, but
        # the next GUI job (including a batch job) always requests normalization.
        fresh.solve_parameters_tmp["physical_normalization"] = False
        regularizer.start_process(result=fresh)
        assert window.tasks[-1][2]["solve_parameters"]["physical_normalization"] is True
        assert fresh.solve_parameters["physical_normalization"] is False
        window.run_finished()
        # Create a current-format file for this test; never depend on stale
        # outputs from an unsupported internal development format.
        fresh.solve_parameters["surface"] = True
        surface_file = tmp_path / "current_surface.saenopy"
        fresh.save(surface_file)
        window.load_from_path([str(surface_file)])
        window.list.setCurrentRow(len(window.data) - 1)
        app.processEvents()
        assert window.data[-1][2].solve_parameters_tmp["surface"] is True
    finally:
        from pyvistaqt import QtInteractor
        for plotter in window.findChildren(QtInteractor):
            plotter.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        app.processEvents()


def test_missing_surface_mode_does_not_inherit_previous_dataset(tmp_path, monkeypatch):
    from qtpy import QtCore, QtWidgets
    from pyvistaqt import QtInteractor
    from saenopy import Result
    from saenopy.gui.solver.modules.BatchEvaluate import BatchEvaluate

    monkeypatch.setattr(QtInteractor, "render", lambda self: None)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    class TestWindow(BatchEvaluate):
        settings_key = "saenopy_surface_mode_default_test"

    window = TestWindow()
    control = window.sub_module_regularize
    surface_result = Result()
    surface_result.solve_parameters = {"surface": True}
    try:
        for saved in (None, {}, {"alpha": 1e10}, {"surface": False}, {"surface": True}):
            control.setResult(surface_result)
            assert control.input_surface.value() is True
            result = Result()
            result.solve_parameters = None if saved is None else saved.copy()
            control.setResult(result)
            expected = bool(saved and saved.get("surface", False))
            assert control.input_surface.value() is expected
            assert result.solve_parameters_tmp["surface"] is expected
            assert result.solve_parameters == saved
            imports, code = control.get_code()
            assert f"'surface': {expected}" in code
            compile(imports + code, "<surface-mode-export>", "exec")

            # Edits belong to this dataset and survive selecting another one.
            control.input_surface.setValue(not expected, send_signal=True)
            control.setResult(surface_result)
            control.setResult(result)
            assert control.input_surface.value() is (not expected)
            assert bool(result.solve_parameters_tmp["surface"]) is (not expected)
            assert result.solve_parameters == saved

        # A partial temporary dictionary must still respect a saved mode.
        for saved_mode in (False, True):
            result = Result()
            result.solve_parameters = {"surface": saved_mode}
            result.solve_parameters_tmp = {"alpha": 1e9}
            control.setResult(result)
            assert control.input_surface.value() is saved_mode
            assert result.solve_parameters_tmp["alpha"] == 1e9

        # An unselected batch dataset must not inherit the visible Surface mode.
        control.setResult(surface_result)
        batch_result = Result()
        queued = []
        monkeypatch.setattr(window, "addTask", lambda *args: queued.append(args))
        control.start_process(result=batch_result)
        assert queued[0][2]["solve_parameters"]["surface"] is False
        assert batch_result.solve_parameters is None
        assert control.input_surface.value() is True
    finally:
        window.close()
        app.processEvents()


def test_cell_channel_default_preserves_saved_and_edited_choices(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from qtpy import QtCore, QtWidgets
    from pyvistaqt import QtInteractor
    from saenopy import Result
    from saenopy.gui.solver.modules.BatchEvaluate import BatchEvaluate

    monkeypatch.setattr(QtInteractor, "render", lambda self: None)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    class TestWindow(BatchEvaluate):
        settings_key = "saenopy_cell_channel_default_test"

    window = TestWindow()
    regularizer = window.sub_module_regularize
    try:
        # Switch repeatedly between datasets: choose by position, not channel name,
        # and never inherit the previous dataset's selection.
        for channels, saved, expected in [
            (["00", "01"], {}, 1), (["00"], {}, 0),
            (["fibres", "cells", "nuclei"], {}, 1),
            (["00", "01"], {"seg_channel": 0}, 0),
            (["00", "01", "02"], {"seg_channel": 2}, 2),
        ]:
            result = Result()
            result.stacks = [SimpleNamespace(channels=channels)]
            result.solve_parameters = saved.copy()
            regularizer.setResult(result)
            assert regularizer.input_seg_channel.value() == expected
            assert result.solve_parameters_tmp["seg_channel"] == expected
            assert result.solve_parameters == saved
            regularizer.input_seg_channel.setValue(0, send_signal=True)
            regularizer.setResult(Result())
            regularizer.setResult(result)
            assert regularizer.input_seg_channel.value() == 0
            assert result.solve_parameters_tmp["seg_channel"] == 0
            assert result.solve_parameters == saved
        # An official legacy file did not select a segmentation channel.
        # Loading it must not fabricate channel 0 as a saved user choice.
        legacy_data = Result().to_dict()
        legacy_data["___save_version__"] = "1.7"
        legacy_data["solve_parameters"] = {"alpha": 1e10}
        legacy = Result.from_dict(legacy_data)
        legacy.stacks = [SimpleNamespace(channels=["matrix", "nuclei"])]
        regularizer.setResult(legacy)
        assert regularizer.input_seg_channel.value() == 1
        assert "seg_channel" not in legacy.solve_parameters
        # Batch processing also initializes an unselected dataset by its own channels.
        batch_result = Result()
        batch_result.stacks = [SimpleNamespace(channels=["matrix", "cells"])]
        regularizer._prepare_gui_parameters(batch_result)
        assert batch_result.solve_parameters_tmp["seg_channel"] == 1
    finally:
        window.close()
        app.processEvents()


def test_gui_uses_internal_cg_without_rewriting_stored_results(tmp_path, monkeypatch):
    from qtpy import QtCore, QtWidgets
    from pyvistaqt import QtInteractor
    from saenopy import Result
    from saenopy.gui.solver.modules.BatchEvaluate import BatchEvaluate
    monkeypatch.setattr(QtInteractor, "render", lambda self: None)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = BatchEvaluate()
    control = window.sub_module_regularize
    try:
        assert not hasattr(control, "input_cg_maxiter_factor")
        for saved in [{"cg_maxiter_factor": 16}, {}, {"cg_maxiter_factor": 8}, {}]:
            result = Result()
            result.solve_parameters = saved.copy()
            control.setResult(result)
            assert result.solve_parameters_tmp["cg_maxiter_factor"] == 2
            assert result.solve_parameters == saved
            control.setResult(Result())
            control.setResult(result)
            assert result.solve_parameters_tmp["cg_maxiter_factor"] == 2
            assert result.solve_parameters == saved
            imports, code = control.get_code()
            assert "'cg_maxiter_factor': 2" in code
            compile(imports + code, '<internal-cg-export>', 'exec')
    finally:
        window.close()
        app.processEvents()
