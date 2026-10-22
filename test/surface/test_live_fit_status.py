"""Viewer status follows the selected result, including hidden tabs and failures."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from types import SimpleNamespace
import pytest


@pytest.mark.parametrize("state,active,status", [
    ("", False, None), ("finished", False, None),
    ("scheduled", True, "queued..."), ("running", True, "in progress..."),
    ("cancelling", True, "cancelling..."), ("failed", False, "fit failed"),
    ("finished", True, "in progress..."), ("failed", True, "in progress..."),
])
def test_fit_status_text(state, active, status):
    from saenopy.gui.solver.modules.live_fit import fit_status_label
    result = SimpleNamespace(solve_parameters_state=state, _live_fit_active=active)
    expected = "Field." if status is None else f"Field. <b>({status})</b>"
    assert fit_status_label(result, "Field.") == expected
    assert fit_status_label(None, "Field.") == "Field."


def test_status_follows_gui_fit_lifecycle(tmp_path, monkeypatch):
    from qtpy import QtCore, QtWidgets
    from pyvistaqt import QtInteractor
    from saenopy import Result
    from saenopy.gui.common.PipelineModule import StateEnum
    from saenopy.gui.solver.modules.BatchEvaluate import BatchEvaluate

    monkeypatch.setattr(QtInteractor, "render", lambda self: None)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(tmp_path))

    class TestWindow(BatchEvaluate):
        settings_key = "saenopy_live_status_test"

    window = TestWindow()
    viewers = (window.tab4, window.tab5)
    regularizer = window.sub_module_regularize
    result, other = Result(), Result()
    try:
        regularizer.setResult(result)
        for viewer in viewers:
            viewer.current_tab_selected = False
            viewer.setResult(result)
            assert viewer.label_tab.text() == viewer.field_description

        result._live_fit_active = True
        result._live_fit_solvers = {}
        for state, text in [(StateEnum.scheduled, "queued..."),
                            (StateEnum.running, "in progress..."),
                            (StateEnum.cancelling, "cancelling...")]:
            regularizer.set_result_state(result, state)
            regularizer.processing_state_changed.emit(result)
            for viewer in viewers:
                assert text in viewer.label_tab.text()
                viewer.setResult(other)
                assert viewer.label_tab.text() == viewer.field_description
                viewer.resultChanged(result)
                assert viewer.label_tab.text() == viewer.field_description
                viewer.setResult(result)
                assert text in viewer.label_tab.text()

        # Cancellation returns to idle; a normal finish clears the notice;
        # failure reports failure instead of leaving an eternal progress label.
        for state, text in [(StateEnum.idle, None), (StateEnum.finished, None),
                            (StateEnum.failed, "fit failed")]:
            result._live_fit_active = True
            regularizer.set_result_state(result, state)
            if state in (StateEnum.idle, StateEnum.finished):
                window.result_changed.emit(result)
            else:
                regularizer.processing_state_changed.emit(result)
            assert not result._live_fit_active
            for viewer in viewers:
                if text:
                    assert text in viewer.label_tab.text()
                else:
                    assert viewer.label_tab.text() == viewer.field_description

        # A saved cancelled last frame stays resumable, and each viewer labels
        # the selected partial frame even while its tab is hidden.
        import copy
        import numpy as np
        from test_release import example_result
        partial = example_result()
        partial.solvers.append(copy.deepcopy(partial.solvers[0]))
        for solver in partial.solvers:
            solver.regularisation_results = np.ones((2, 3))
            solver.regularisation_parameters = {}
        partial.solvers[1].regularisation_parameters["cancelled"] = True
        regularizer.setResult(partial)
        assert regularizer.input_button.isEnabled()
        assert regularizer.input_button.text() == "continue"
        for viewer in viewers:
            viewer.setResult(partial)
            viewer.t_slider.setRange(0, 1)
            viewer.t_slider.setValue(1)
            assert "fit cancelled; partial result" in viewer.label_tab.text()
            viewer.t_slider.setValue(0)
            assert viewer.label_tab.text() == viewer.field_description
    finally:
        # Drain queued matplotlib draws before deleting their Qt canvases.
        for _ in range(3):
            app.processEvents()
        for plotter in window.findChildren(QtInteractor):
            plotter.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        app.processEvents()
