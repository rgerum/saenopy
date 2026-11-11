"""Regression checks for imported images, references and failed reruns."""
import copy
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from test_orientation_workflow import window, make_result
from test_orientation_point_distance import measurement


@pytest.mark.parametrize('mode', ['point_orientation', 'line_orientation'])
@pytest.mark.parametrize('failure', ['statistics', 'save'])
def test_failed_rerun_preserves_all_completed_fields(window, tmp_path, monkeypatch, mode, failure):
    from qtpy import QtWidgets
    from saenopy.gui.orientation.modules.analysis import analyze_orientation
    from saenopy.gui.orientation.modules.result import ResultOrientation
    result = measurement(tmp_path, 'previous')
    if mode == 'line_orientation':
        result.analysis_mode = mode
        parameters = dict(result.orientation_parameters, line_x1=20., line_y1=70., line_x2=100., line_y2=30.)
        analyze_orientation(result, parameters)
    batch = window.deformations
    batch.add_data(result)
    previous = copy.deepcopy(result.to_dict())
    saved = Path(result.output).read_bytes()
    pending = dict(result.orientation_parameters, sigma_tensor=5., reference_x=25., line_y2=50.)
    result.orientation_parameters_tmp = pending

    def fail(*args, **kwargs):
        raise OSError('Simulated late failure')

    monkeypatch.setattr(QtWidgets.QMessageBox, 'critical', lambda *args: None)
    if failure == 'statistics':
        monkeypatch.setattr('saenopy.gui.orientation.modules.axis_statistics.axis_statistics', fail)
    else:
        monkeypatch.setattr(ResultOrientation, 'save', fail)
    batch.sub_module_deformation.process_thread(result, {'orientation_parameters': pending})
    np.testing.assert_equal(result.to_dict(), previous)
    assert result.orientation_parameters_tmp == pending
    assert Path(result.output).read_bytes() == saved


def test_direct_analysis_preserves_previous_maps_when_save_fails(tmp_path, monkeypatch):
    from saenopy.gui.orientation.modules.analysis import analyze_orientation
    from saenopy.gui.orientation.modules.result import ResultOrientation
    result = measurement(tmp_path, 'direct')
    previous = copy.deepcopy(result.to_dict())
    saved = Path(result.output).read_bytes()
    def fail(*args, **kwargs):
        raise OSError('Cannot write result')
    monkeypatch.setattr(ResultOrientation, 'save', fail)
    with pytest.raises(OSError, match='Cannot write'):
        analyze_orientation(result, dict(result.orientation_parameters, reference_x=20.))
    np.testing.assert_equal(result.to_dict(), previous)
    assert Path(result.output).read_bytes() == saved


def test_failed_atomic_overwrite_keeps_previous_file_and_removes_temp(tmp_path, monkeypatch):
    result = measurement(tmp_path, 'atomic')
    path = Path(result.output)
    before = path.read_bytes()
    files = set(tmp_path.iterdir())
    def fail(handle, **kwargs):
        handle.write(b'partial archive')
        raise OSError('Disk full')
    monkeypatch.setattr(np, 'savez_compressed', fail)
    with pytest.raises(OSError, match='Disk full'):
        result.save()
    assert path.read_bytes() == before
    assert set(tmp_path.iterdir()) == files


@pytest.mark.skipif(os.name != 'posix', reason='POSIX mode bits and umask are not available on Windows')
def test_archive_save_retains_permissions_and_respects_umask(tmp_path):
    import stat
    result = measurement(tmp_path, 'permissions')
    path = Path(result.output)
    path.chmod(0o640)
    result.save()
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    new_path = tmp_path / 'new.saenopyOrientation'
    previous_umask = os.umask(0o027)
    try:
        result.save(new_path)
    finally:
        os.umask(previous_umask)
    assert stat.S_IMODE(new_path.stat().st_mode) == 0o640


@pytest.mark.parametrize('mode', ['full', 'axis_orientation_masked'])
def test_failed_live_preview_clears_stale_overlay(window, make_result, monkeypatch, mode):
    result = make_result(mode)
    window.deformations.add_data(result)
    detector = window.deformations.sub_module_segmentation
    result.mask_preview = np.ones((72, 112), dtype=bool)
    result.mask_preview_edge = 4
    result.segmentation_preview = {'mask': np.ones((80, 120), dtype=bool)}
    detector._preview_target = result
    redraws = []
    monkeypatch.setattr(window.deformations.tab1, 'update_display', lambda: redraws.append(True))
    detector.finish_live_preview(detector._preview_revision, None, 'Invalid threshold')
    assert result.mask_preview is None and result.segmentation_preview is None
    assert result.mask_preview_edge == 0 and redraws == [True]
    assert 'unavailable' in detector.preview_status.text()


def test_coincident_line_in_pasted_parameters_is_reported_in_tab(window, make_result):
    result = make_result('axis_orientation_only')
    result.analysis_mode = 'line_orientation'
    result.orientation_parameters.update(line_x1=30., line_y1=30., line_x2=30., line_y2=30.)
    window.deformations.add_data(result)
    tab = window.deformations.tab1
    tab.update_display()
    assert 'two distinct points' in tab.point_status.text()
    assert not window.deformations.sub_module_deformation.check_available(result)


@pytest.mark.parametrize('point', [(np.nan, 5), (np.inf, 5), (-1, 5), (120, 5), (5, 80)])
def test_invalid_point_cannot_start_analysis(window, make_result, point):
    result = make_result('axis_orientation_only')
    result.analysis_mode = 'point_orientation'
    result.shape = (80, 120)
    result.orientation_parameters.update(reference_x=point[0], reference_y=point[1])
    assert not window.deformations.sub_module_deformation.check_available(result)


def test_import_filters_nonimages_and_resolves_relative_paths(tmp_path, monkeypatch):
    from saenopy.gui.orientation.modules.result import get_orientation_files, ResultOrientation
    source = tmp_path / 'inputs'
    source.mkdir()
    pixels = np.arange(120, dtype=np.uint16).reshape(10, 12)
    tifffile.imwrite(source / 'valid.tif', pixels)
    (source / 'folder.tif').mkdir()
    (source / 'broken.png').write_bytes(b'not an image')
    (source / 'notes.txt').write_text('not an image')
    monkeypatch.chdir(tmp_path)
    results = get_orientation_files('results', 'inputs/*', None, .5, analysis_mode='axis_orientation_only')
    assert len(results) == 1
    absolute_output = Path(results[0].output).resolve()
    monkeypatch.chdir(source)
    loaded = ResultOrientation.load(str(absolute_output))
    np.testing.assert_array_equal(loaded.get_image(1), pixels)


def test_npz_analysis_exports_use_complete_filename_stem(tmp_path, monkeypatch):
    import importlib
    from saenopy.gui.orientation.modules.result import ResultOrientation
    analysis = importlib.import_module('saenopy.gui.orientation.modules.analysis')
    path = tmp_path / 'a.npz'
    result = ResultOrientation(output=str(path), image_fiber=str(tmp_path / 'image.tif'),
                               image_cell=None, pixel_size=1., analysis_mode='axis_orientation_only')
    calls = []
    def calculate(*, mask_sigma=5, **kwargs):
        calls.append(kwargs['out_list'])
        values = np.zeros((3, 4))
        return {'arrays': dict(min_x=values, min_y=values, coherency=values, min_vx=values, min_vy=values),
                'validity_mask': np.ones_like(values, dtype=bool), 'histogram': np.zeros((36,4)), 'summary': {}}
    monkeypatch.setattr(analysis, 'StuctureAnalysisMain', calculate)
    analysis.analyze_orientation(result, dict(sigma_tensor=3., sigma_tensor_type='pixel', edge=0, sigma_first_blur=.5))
    assert calls == [[str(tmp_path / 'a')]]


def test_coherency_tab_uses_its_own_map(window, make_result):
    result = make_result('axis_orientation_only')
    result.orientation_map = None
    tab = window.deformations.tab3
    assert tab.checkTabEnabled(result)
    result.orientation_map = np.zeros((72,112))
    result.coherence_map = None
    assert not tab.checkTabEnabled(result)


def test_whole_image_axis_maps_apply_saved_validity_mask(make_result):
    result = make_result('axis_orientation_only')
    result.validity_mask = np.ones((72,112), dtype=bool)
    result.validity_mask[0,0] = False
    for key in ['orientation_x_axis','orientation_y_axis','alignment_x_axis','alignment_y_axis','coherence']:
        assert np.isnan(result.get_map_data(key)[0,0])


@pytest.mark.parametrize('width', [0, -1, np.nan, np.inf])
def test_invalid_shell_width_is_rejected(width):
    from saenopy.gui.orientation.modules.point_distance import shell_width_pixels
    with pytest.raises(ValueError, match='positive and finite'):
        shell_width_pixels({'shell_width':width, 'shell_width_type':'pixel'}, .5)


def test_nearly_equal_calibrations_share_rings_and_sem(window, tmp_path):
    first = measurement(tmp_path, 'first', pixel_size=.5)
    second = measurement(tmp_path, 'second', pixel_size=.5 + 2e-11)
    for result, value in [(first, .2), (second, .8)]:
        for row in result.results_distance:
            row['Point alignment (shell)'] = value
        result.save()
    plotter = window.plotting_window
    plotter.add_files([first.output, second.output])
    plotter.type.setValue('point alignment over distance', send_signal=True)
    curve = plotter.canvas.axes.lines[0]
    assert 'n=2' in curve.get_label()
    np.testing.assert_allclose(curve.get_ydata(), .5)
    assert len(curve.get_xdata()) == len(first.results_distance)
    vertices = np.concatenate([path.vertices for path in plotter.canvas.axes.collections[0].get_paths()])
    np.testing.assert_allclose([vertices[:,1].min(), vertices[:,1].max()], [.2,.8])
