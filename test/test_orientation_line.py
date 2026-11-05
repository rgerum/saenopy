"""Two-click line reference, result persistence and shared statistics."""
from pathlib import Path
import numpy as np
import pytest
import tifffile
from test_orientation_workflow import window, make_result


def test_import_line_and_switch_reference_type(window, make_result, tmp_path):
    from saenopy.gui.orientation.modules.AddFilesDialog import AddFilesDialog
    source = make_result('axis_orientation_only')
    window.workflow_choice.buttons['point'].click()
    dialog = AddFilesDialog(window.deformations, window.deformations.settings)
    dialog.fiberText.setValue(source.image_fiber)
    dialog.outputText.setValue(str(tmp_path/'line'))
    dialog.show()
    dialog.reference_type.buttons['line'].click()
    assert dialog.selected_analysis_mode() == 'line_orientation'
    assert dialog.selected_cell_pattern() is None
    assert dialog.button_addList1.isEnabled()
    assert not dialog.cellText.isVisible() and dialog.mask_source.isVisible()
    assert 'two points' in dialog.label.text()
    dialog.reference_type.buttons['point'].click()
    assert dialog.selected_analysis_mode() == 'point_orientation'
    dialog.close()


def make_line_result(tmp_path):
    from saenopy.gui.orientation.modules.result import get_orientation_files
    yy, xx = np.indices((128, 160))
    path = tmp_path/'image.tif'
    tifffile.imwrite(path, (20000+18000*np.cos(xx*.3+yy*.12)).astype(np.uint16))
    return get_orientation_files(tmp_path/'out', str(path), None, .5, analysis_mode='line_orientation')[0]


def click_image(tab, x, y):
    from qtpy import QtCore, QtTest
    pos = tab.label.mapFromScene(tab.label.origin.mapToScene(QtCore.QPointF(x+.5, y+.5)))
    QtTest.QTest.mouseClick(tab.label.viewport(), QtCore.Qt.LeftButton, pos=pos)


def test_two_clicks_cancel_preview_and_restore_saved_results(window, tmp_path, monkeypatch):
    from qtpy import QtCore, QtTest, QtWidgets
    from saenopy.gui.orientation.modules.result import ResultOrientation
    from saenopy.gui.orientation.modules.analysis import analyze_orientation
    from saenopy.gui.orientation.modules.ReferenceOverlay import parameter_line
    from saenopy.gui.orientation.analyze.PlottingWindow import PlottingWindow
    result = make_line_result(tmp_path)
    batch = window.deformations
    batch.add_data(result)
    tab, module = batch.tab1, batch.sub_module_deformation
    module.edge.setValue(10, send_signal=True)
    module.sigma_tensor.setValue(3., send_signal=True)
    module.sigma_tensor_type.setValue('pixel', send_signal=True)
    batch.tabs.setCurrentWidget(tab.tab.parent())
    QtWidgets.QApplication.processEvents()
    assert batch.sub_module_segmentation.isHidden()
    assert not module.input_button.isEnabled()
    assert batch.tabs.tabText(batch.tabs.indexOf(tab.tab.parent())) == 'Reference line'
    tab.label.scaler.setScale(1.25)
    tab.label.translater.setPos(11, 7)
    tab.choose_point.click()
    click_image(tab, 30, 70)
    assert tab._line_start == (30, 70)
    assert not module.input_button.isEnabled()
    assert parameter_line(result.orientation_parameters_tmp) is None
    click_image(tab, 30, 70)  # A single pixel does not define a line.
    assert tab.choose_point.isChecked()
    QtTest.QTest.keyClick(tab.label.viewport(), QtCore.Qt.Key_Escape)
    assert not tab.choose_point.isChecked()
    assert parameter_line(result.orientation_parameters_tmp) is None
    tab.choose_point.click()
    click_image(tab, 30, 70)
    click_image(tab, 110, 30)
    expected = np.array([[30, 70], [110, 30]])
    np.testing.assert_array_equal(parameter_line(result.orientation_parameters_tmp), expected)
    assert module.input_button.isEnabled()
    assert tab.reference_overlay.extension.isVisible()
    analyze_orientation(result, result.orientation_parameters_tmp.copy())
    loaded = ResultOrientation.load(result.output)
    np.testing.assert_array_equal(parameter_line(loaded.orientation_parameters), expected)
    assert loaded.analysis_mode == 'line_orientation'
    assert loaded.orientation_map.shape == (108, 140)
    assert loaded.segmentation is None and loaded.results_distance == []
    np.testing.assert_allclose(loaded.orientation_map, 1-np.abs(loaded.get_map_data('angle_line'))/45)
    assert list(Path(result.output).parent.iterdir()) == [Path(result.output)]
    batch.set_current_result.emit(loaded)
    batch.tabs.setCurrentWidget(batch.tab4.tab.parent())
    assert batch.tab4.axis.value() == 'line'
    assert batch.tab2.axis.value() == 'line'
    assert set(batch.tab2.axis.buttons) == {'line', 'x', 'y'}
    assert batch.tab4.reference_overlay.markers[0].pos() == QtCore.QPointF(20.5, 60.5)
    old_angles = loaded.get_map_data('angle_line').copy()
    tab.set_reference_line((10, 10), (100, 10))
    np.testing.assert_array_equal(parameter_line(loaded.orientation_parameters), expected)
    np.testing.assert_array_equal(loaded.get_map_data('angle_line'), old_angles)
    assert batch.tab4.reference_overlay.markers[0].pos() == QtCore.QPointF(20.5, 60.5)
    # Cancelling a replacement leaves the complete pending line intact.
    tab.choose_point.click()
    click_image(tab, 20, 20)
    tab.choose_point.click()
    np.testing.assert_array_equal(parameter_line(loaded.orientation_parameters_tmp), [[10, 10], [100, 10]])
    imports, code = module.get_code()
    calls = []
    monkeypatch.setattr('saenopy.gui.orientation.modules.analysis.analyze_orientation', lambda r,p,s: calls.append((r,p)))
    exec(imports+code, {})
    assert calls[0][0].analysis_mode == 'line_orientation'
    assert calls[0][1]['line_x2'] == 100
    data = PlottingWindow.load_file(None, result.output).resulting_data
    assert data['Alignment line'][0] == pytest.approx(np.nanmean(loaded.orientation_map))
    assert data['Mean deviation line'][0] == pytest.approx(np.nanmean(np.abs(old_angles)))
    plotter = window.plotting_window
    plotter.add_files([result.output])
    enabled = plotter.update_metric_availability()
    assert {'line alignment', 'line angular deviation', 'alignment x', 'coherency', 'angle histogram y'} <= set(enabled)
    assert 'point alignment' not in enabled and 'global orientation' not in enabled
    plotter.type.setValue('line alignment', send_signal=True)
    assert plotter.canvas.axes.get_ylim() == (-1, 1)
    plotter.type.setValue('line angular deviation', send_signal=True)
    assert plotter.canvas.axes.get_ylim() == (0, 90)
    plotter.type.setValue('angle histogram line', send_signal=True)
    histogram = plotter.get_histogram_data()
    assert histogram.fraction.sum() == pytest.approx(1)
    counts, _ = np.histogram(old_angles[np.isfinite(old_angles)], bins=np.linspace(-90, 90, 37))
    np.testing.assert_allclose(histogram.fraction, counts/counts.sum())
    assert 'reference line' in plotter.canvas.axes.get_xlabel()
    assert 'Probability' == plotter.canvas.axes.get_ylabel()


def test_failed_line_rerun_preserves_calculated_reference(window, tmp_path, monkeypatch):
    from qtpy import QtWidgets
    from saenopy.gui.orientation.modules.analysis import analyze_orientation
    result = make_line_result(tmp_path)
    params = dict(edge=10, sigma_tensor=4, sigma_tensor_type='pixel', sigma_first_blur=.5,
                  line_x1=10., line_y1=10., line_x2=100., line_y2=10.)
    analyze_orientation(result, params)
    window.deformations.add_data(result)
    tab, module = window.deformations.tab1, window.deformations.sub_module_deformation
    previous = result.orientation_parameters.copy()
    old_map = result.orientation_map.copy()
    tab.set_reference_line((30, 30), (30, 70))
    def fail(*a, **k):
        raise ValueError('Invalid new parameters')
    monkeypatch.setattr(module, 'process', fail)
    monkeypatch.setattr(QtWidgets.QMessageBox, 'critical', lambda *a, **k: None)
    module.process_thread(result, {'orientation_parameters': result.orientation_parameters_tmp})
    assert result.orientation_parameters == previous
    assert result.orientation_parameters_tmp['line_y2'] == 70
    np.testing.assert_array_equal(result.orientation_map, old_map)
