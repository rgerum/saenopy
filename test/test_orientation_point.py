"""Point import, click geometry, persistence and statistics in every reference mode."""
from pathlib import Path
import numpy as np
import pytest
import tifffile
from test_orientation_workflow import window, make_result


def test_point_import_whole_image_choice(window, make_result, tmp_path, monkeypatch):
    from saenopy.gui.orientation.modules.AddFilesDialog import AddFilesDialog
    source = make_result('full')
    batch = window.deformations
    window.workflow_choice.buttons['point'].click()
    assert batch.analysis_mode == 'point_orientation'
    def accept(dialog):
        dialog.show()
        assert dialog.selected_analysis_mode() == 'point_orientation'
        dialog.mask_source.setValue('whole_image', send_signal=True)
        dialog.cellText.setValue('missing.tif')
        dialog.fiberText.setValue(source.image_fiber)
        dialog.outputText.setValue(str(tmp_path/'point'))
        assert dialog.button_addList1.isEnabled()
        assert not dialog.cellText.isVisible() and dialog.mask_source.isVisible()
        assert dialog.selected_cell_pattern() is None
        dialog.accept_new()
        return 1
    monkeypatch.setattr(AddFilesDialog,'exec',accept)
    batch.add_measurement()
    result = batch.data[-1][2]
    assert result.analysis_mode == 'point_orientation' and result.image_cell is None
    assert batch.sub_module_segmentation.isHidden()
    assert not batch.sub_module_deformation.input_button.isEnabled()
    assert batch.tabs.tabText(batch.tabs.indexOf(batch.tab1.tab.parent())) == 'Reference point'


def test_point_click_analysis_reload_and_export(window, tmp_path, monkeypatch):
    from qtpy import QtCore, QtTest, QtWidgets
    from saenopy.gui.orientation.modules.result import get_orientation_files, ResultOrientation
    from saenopy.gui.orientation.analyze.PlottingWindow import PlottingWindow
    from saenopy.gui.orientation.modules.analysis import analyze_orientation
    yy,xx = np.indices((128,160))
    image = tmp_path/'image.tif'
    tifffile.imwrite(image, (20000+18000*np.cos(xx*.3+yy*.12)).astype(np.uint16))
    result = get_orientation_files(tmp_path/'out', str(image), None, .5, analysis_mode='point_orientation')[0]
    batch = window.deformations
    batch.add_data(result)
    tab, module = batch.tab1, batch.sub_module_deformation
    module.edge.setValue(10, send_signal=True)
    module.sigma_tensor.setValue(3., send_signal=True)
    module.sigma_tensor_type.setValue('pixel', send_signal=True)
    batch.tabs.setCurrentWidget(tab.tab.parent())
    QtWidgets.QApplication.processEvents()
    # Real viewport clicks through origin/scene transforms (including zoom/pan).
    tab.label.scaler.setScale(1.25)
    tab.label.translater.setPos(11,7)
    pos = tab.label.mapFromScene(tab.label.origin.mapToScene(QtCore.QPointF(80.5,64.5)))
    tab.choose_point.click()
    QtTest.QTest.mouseClick(tab.label.viewport(), QtCore.Qt.LeftButton, pos=pos)
    assert result.orientation_parameters_tmp['reference_x'] == 80
    assert result.orientation_parameters_tmp['reference_y'] == 64
    assert module.input_button.isEnabled()
    assert 'run find orientations' in tab.point_status.text()
    parameters = result.orientation_parameters_tmp.copy()
    analyze_orientation(result, parameters)
    loaded = ResultOrientation.load(result.output)
    assert loaded.orientation_parameters['reference_x'] == 80
    assert loaded.orientation_parameters['reference_y'] == 64
    assert loaded.orientation_map.shape == (108,140)
    assert np.isnan(loaded.orientation_map[54,70])
    assert loaded.segmentation is None and loaded.image_cell is None
    assert loaded.results_distance
    assert sum(row['Valid pixels'] for row in loaded.results_distance) == np.isfinite(loaded.orientation_map).sum()
    assert list(Path(result.output).parent.iterdir()) == [Path(result.output)]
    data = PlottingWindow.load_file(None, result.output).resulting_data
    assert data['Alignment point'][0] == pytest.approx(np.nanmean(loaded.orientation_map))
    assert data['Mean angle point'][0] == pytest.approx(45*(1-data['Alignment point'][0]))
    batch.set_current_result.emit(loaded)
    batch.tabs.setCurrentWidget(batch.tab4.tab.parent())
    assert batch.tab4.axis.isHidden()
    assert batch.tab4.point_marker.pos() == QtCore.QPointF(70.5,54.5)
    tab.set_reference_point(40,30)
    assert loaded.orientation_parameters['reference_x'] == 80
    assert loaded.orientation_parameters_tmp['reference_x'] == 40
    assert 'run find orientations' in tab.point_status.text()
    assert batch.tab4.point_marker.pos() == QtCore.QPointF(70.5,54.5)
    # Changing measurements retains each pending reference independently.
    batch.set_current_result.emit(result)
    assert result.orientation_parameters_tmp['reference_x'] == 80
    batch.set_current_result.emit(loaded)
    assert loaded.orientation_parameters_tmp['reference_x'] == 40
    imports, code = module.get_code()
    assert "'reference_x': 40.0" in code
    calls=[]
    monkeypatch.setattr('saenopy.gui.orientation.modules.analysis.analyze_orientation', lambda r,p,s: calls.append((r,p)))
    exec(imports+code,{})
    assert calls[0][0].analysis_mode == 'point_orientation'
    assert calls[0][1]['reference_x'] == 40
    plotter = window.plotting_window
    plotter.add_files([result.output])
    enabled = plotter.update_metric_availability()
    assert {'point alignment','point angular deviation','alignment x','angle histogram y','coherency'} <= set(enabled)
    assert 'global orientation' not in enabled
    plotter.type.setValue('point alignment', send_signal=True)
    assert plotter.canvas.axes.get_ylim() == (-1,1)
    plotter.type.setValue('point angular deviation', send_signal=True)
    assert plotter.canvas.axes.get_ylim() == (0,90)


def test_cell_axis_stats_use_saved_region_without_reanalysis(window, make_result, monkeypatch):
    from saenopy.gui.orientation.analyze.PlottingWindow import PlottingWindow
    result = make_result('full')
    mask = np.zeros((80,120),dtype=bool)
    mask[30:50,50:70] = True
    result.segmentation = dict(mask=mask, centroid=(60.,40.), radius=10.)
    result.orientation_parameters['max_dist'] = 20
    result.angle_to_x_map[:] = 80
    yy,xx = np.indices((72,112))
    valid = ~mask[4:-4,4:-4] & ((xx-56)**2+(yy-36)**2 <= 20**2)
    result.angle_to_x_map[valid] = 15
    result.angle_to_y_map = None  # Version 1.0 files also work.
    result.save()
    before = Path(result.output).read_bytes()
    monkeypatch.setattr('CompactionAnalyzer.CompactionFunctions.analyze_local', lambda *a, **k: pytest.fail('Must not recompute tensor'))
    loaded = PlottingWindow.load_file(None,result.output)
    assert loaded.resulting_data['Mean absolute deviation x'][0] == pytest.approx(15)
    assert loaded.resulting_data['Mean absolute deviation y'][0] == pytest.approx(75)
    assert loaded.resulting_data['Alignment x'][0] == pytest.approx(2/3)
    assert loaded.axis_histogram[:,2].sum() == valid.sum()
    assert Path(result.output).read_bytes() == before
    window.deformations.add_data(result)
    batch = window.deformations
    assert batch.tabs.isTabVisible(batch.tabs.indexOf(batch.tab2.tab.parent()))
    assert batch.tabs.indexOf(batch.tab2.tab.parent()) < batch.tabs.indexOf(batch.tab3.tab.parent())
    plotter = window.plotting_window
    plotter.add_files([result.output])
    assert {'alignment x','alignment y','mean absolute deviation x','angle histogram y','global orientation'} <= set(plotter.update_metric_availability())
    plotter.type.setValue('angle histogram x',send_signal=True)
    assert plotter.get_histogram_data().fraction.sum() == pytest.approx(1)


def test_failed_point_rerun_keeps_previous_reference(window, make_result, monkeypatch):
    from qtpy import QtWidgets
    result = make_result('axis_orientation_only')
    result.analysis_mode = 'point_orientation'
    result.orientation_parameters.update(reference_x=60., reference_y=40.)
    result.results_total[0].update({'Reference x (px)':60., 'Reference y (px)':40.})
    window.deformations.add_data(result)
    module = window.deformations.sub_module_deformation
    previous = result.orientation_parameters.copy()
    old_map = result.orientation_map.copy()
    window.deformations.tab1.set_reference_point(20,10)
    def fail(*a,**k):
        raise ValueError('Invalid new parameters')
    monkeypatch.setattr(module,'process',fail)
    monkeypatch.setattr(QtWidgets.QMessageBox,'critical',lambda *a,**k:None)
    module.process_thread(result, {'orientation_parameters':result.orientation_parameters_tmp})
    assert result.orientation_parameters == previous
    assert result.orientation_parameters_tmp['reference_x'] == 20
    np.testing.assert_array_equal(result.orientation_map,old_map)
