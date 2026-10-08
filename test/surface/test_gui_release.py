"""Offscreen Qt integration: official result loading and the actual export handler."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from pathlib import Path
import appdirs
import pytest


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
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(tmp_path))

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
        assert result.___save_version__ == "1.8"
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
        assert "'cg_maxiter_factor': 16" in code
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
