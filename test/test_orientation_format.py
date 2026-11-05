"""End-to-end archive, measurement-view and image-export compatibility."""
from pathlib import Path
import shutil
import numpy as np
import pytest
import tifffile
from test_orientation_workflow import window

CASES = {
    'cell': ('full', 'separate_image'),
    'whole': ('axis_orientation_only', 'whole_image'),
    'same_mask': ('axis_orientation_masked', 'same_image'),
    'second_mask': ('axis_orientation_masked', 'separate_image'),
    'point': ('point_orientation', 'whole_image'),
    'line': ('line_orientation', 'whole_image'),
}


@pytest.fixture(scope='module')
def archives(tmp_path_factory):
    from saenopy.gui.orientation.modules.result import get_orientation_files, ResultOrientation
    from saenopy.gui.orientation.modules.analysis import analyze_orientation
    root = tmp_path_factory.mktemp('all-modes')
    yy, xx = np.indices((160, 200))
    cell = ((xx-100)**2 + (yy-80)**2 < 32**2)
    fiber = (20000 + 17000*np.cos(xx*.22 + yy*.11)) * (.15 + .85*cell)
    tifffile.imwrite(root/'orientation.tif', fiber.astype(np.uint16))
    tifffile.imwrite(root/'mask.tif', (1000+50000*cell).astype(np.uint16))
    results = {}
    for key, (mode, source) in CASES.items():
        mask = None if source == 'whole_image' else root/('orientation.tif' if source == 'same_image' else 'mask.tif')
        result = get_orientation_files(root/key, str(root/'orientation.tif'),
            None if mask is None else str(mask), .5, analysis_mode=mode)[0]
        params = dict(edge=10, sigma_tensor=3., sigma_tensor_type='pixel', sigma_first_blur=.5,
            SaveNumpy=False, plotting=False, max_dist=None, ignore_cell_outline=False,
            angle_sections=5, shell_width=10, shell_width_type='pixel')
        if key == 'point':
            params.update(reference_x=100., reference_y=80.)
        if key == 'line':
            params.update(line_x1=30., line_y1=100., line_x2=170., line_y2=40.)
        seg = dict(thresh=1., gauss1=1., gauss2=40., invert=False,
            mask_sigma=3., mask_threshold=.5, mask_invert=False)
        analyze_orientation(result, params, seg)
        results[key] = ResultOrientation.load(result.output)
    return results


@pytest.mark.parametrize('case', CASES)
def test_all_modes_save_reload_and_import(window, archives, case, tmp_path, monkeypatch):
    from qtpy import QtWidgets
    from saenopy.gui.orientation.modules.result import ResultOrientation
    from saenopy.gui.orientation.analyze.PlottingWindow import PlottingWindow
    result = archives[case]
    path = Path(result.output)
    original = path.read_bytes()
    assert result.___save_version__ == '1.1'
    assert result.analysis_mode == CASES[case][0]
    if case in ('whole', 'same_mask', 'second_mask'):
        assert result.mask_source == CASES[case][1]
    with np.load(path, allow_pickle=False) as data:
        assert all(not data[key].dtype.hasobject for key in data.files)
        assert str(data['analysis_mode']) == result.analysis_mode
    maps = result.get_data_structure()['maps']
    for name in maps:
        values = result.get_map_data(name)
        assert values.shape == (140, 180), name
        assert np.isfinite(values).any(), name
    if case == 'point':
        assert np.isnan(result.orientation_map[70, 90])
        assert result.orientation_parameters['reference_x'] == 100
    if case == 'line':
        assert result.orientation_parameters['line_x2'] == 170
        np.testing.assert_allclose(result.orientation_map, 1-np.abs(result.get_map_data('angle_line'))/45)
    # Exercise the same loader used by measurement drag-and-drop.
    monkeypatch.setattr(QtWidgets.QMessageBox, 'critical', lambda *args: pytest.fail(str(args)))
    window.deformations.load_from_path(str(path))
    assert window.deformations.data[-1][2].analysis_mode == result.analysis_mode
    plotted = PlottingWindow.load_file(None, str(path))
    assert np.isfinite(plotted.resulting_data['Mean Coherency']).any()
    assert np.isfinite(plotted.resulting_data['Alignment x']).any()
    assert np.isfinite(plotted.resulting_data['Alignment y']).any()
    window.plotting_window.add_files([str(path)])
    window.plotting_window.type.setValue('angle histogram x', send_signal=True)
    assert window.plotting_window.get_histogram_data().fraction.sum() == pytest.approx(1)
    # Save to a fresh subdirectory, and verify numerical contents are unchanged.
    copy = tmp_path/'saved-again'/'result.saenopyOrientation'
    result.save(copy)
    loaded = ResultOrientation.load(copy)
    for name in maps:
        np.testing.assert_array_equal(loaded.get_map_data(name), result.get_map_data(name))
    assert path.read_bytes() == original


def test_relative_image_paths_survive_package_relocation(archives, tmp_path):
    from saenopy.gui.orientation.modules.result import ResultOrientation
    package = tmp_path/'package'
    package.mkdir()
    for case, result in archives.items():
        loaded = ResultOrientation.load(result.output)
        for attr, resolve in [('image_fiber', loaded.get_absolute_path_fiber),
                              ('image_cell', loaded.get_absolute_path_cell)]:
            source = resolve()
            if source is not None:
                source = Path(source)
                shutil.copyfile(source, package/source.name)
                setattr(loaded, attr, source.name)
        loaded.output = str(package/(case+'.saenopyOrientation'))
        loaded.save()
    moved = tmp_path/'relocated'
    package.rename(moved)
    for case in CASES:
        loaded = ResultOrientation.load(moved/(case+'.saenopyOrientation'))
        assert loaded.get_image(1).shape == (160, 200)
        if loaded.image_cell is not None:
            assert loaded.get_image(0).shape == (160, 200)
        if case in ('same_mask', 'second_mask'):
            assert loaded.mask_source == CASES[case][1]


@pytest.mark.parametrize('saved_mask', [True, False])
def test_version_10_archive_still_loads_without_reanalysis(window, archives, tmp_path, saved_mask):
    from saenopy.saveable import flatten_dict
    from saenopy.gui.orientation.modules.result import ResultOrientation
    from saenopy.gui.orientation.analyze.PlottingWindow import PlottingWindow
    legacy = archives['cell'].to_dict()
    for key in ('analysis_mode', 'validity_mask', 'angle_to_y_map', 'axis_histogram'):
        legacy.pop(key)
    legacy['___save_version__'] = '1.0'
    if not saved_mask:
        legacy['segmentation'] = '__NONE__'
    legacy['results_total'] = [{k:v for k,v in legacy['results_total'][0].items()
        if not k.startswith(('Alignment ', 'Mean absolute deviation '))}]
    path = tmp_path/'legacy.saenopyOrientation'
    with path.open('wb') as stream:
        np.savez_compressed(stream, **flatten_dict(legacy))
    before = path.read_bytes()
    loaded = ResultOrientation.load(path)
    assert loaded.analysis_mode == 'full'
    np.testing.assert_array_equal(loaded.get_map_data('orientation_y_axis'), archives['cell'].angle_to_y_map)
    window.deformations.load_from_path(str(path))
    plotted = PlottingWindow.load_file(None, str(path))
    assert np.isfinite(plotted.resulting_data['Orientation']).any()
    assert bool(np.isfinite(plotted.resulting_data['Alignment x']).any()) == saved_mask
    window.plotting_window.add_files([str(path)])
    enabled = window.plotting_window.update_metric_availability()
    assert 'global orientation' in enabled
    assert ('alignment x' in enabled) == saved_mask
    assert path.read_bytes() == before


@pytest.mark.parametrize('scale', [1., .6, 1.5])
def test_all_maps_export_at_requested_scale(archives, scale):
    from saenopy.gui.solver.modules.exporter.ExporterRender2D import render_2d
    for result in archives.values():
        for name in result.get_data_structure()['maps']:
            image = render_2d(dict(maps=name, stack=dict(channel='fibers'),
                image=dict(scale=scale), scalebar=dict(hide=True), colorbar=dict(hide=True),
                time=dict(display=False)), result)
            assert image.shape == (int(140*scale), int(180*scale), 3)


def test_mixed_modes_have_common_and_reference_specific_statistics(window, archives):
    plot = window.plotting_window
    plot.add_files([result.output for result in archives.values()])
    enabled = plot.update_metric_availability()
    assert {'global orientation', 'point alignment', 'line alignment', 'alignment x',
            'alignment y', 'angle histogram line', 'coherency'} <= set(enabled)
    for metric in ('coherency', 'global orientation', 'point alignment', 'line alignment',
                   'angle histogram x', 'angle histogram y', 'angle histogram line'):
        plot.type.setValue(metric, send_signal=True)
        if metric.startswith('angle histogram'):
            data = plot.get_histogram_data()
            assert not data.empty
            assert np.isfinite(data.fraction).all()
            np.testing.assert_allclose(data.groupby(['group', 'filename']).fraction.sum(), 1.)


@pytest.mark.parametrize('scale', [.5, 1., 2.])
def test_scaled_map_preserves_missing_pixels_and_angle_colors(scale):
    from PIL import Image
    from matplotlib import pyplot as plt
    from saenopy.gui.solver.modules.exporter.ExporterRender2D import render_map
    values = np.full((40, 60), np.nan)
    values[10:30, 10:30] = -89
    values[10:30, 30:50] = 89
    class Result:
        def get_data_structure(self):
            return dict(maps=dict(angle=dict(colormap='gist_rainbow', lim=[-90,90])))
        def get_map_data(self, name):
            return values
    background = Image.new('RGB',(int(60*scale),int(40*scale)),(10,20,30))
    image = render_map(dict(maps='angle',maps_alpha=1.),Result(),background,scale,1,None)
    pixels = np.asarray(image)
    np.testing.assert_array_equal(pixels[0,0], [10,20,30])
    for x, angle in [(20,-89),(40,89)]:
        expected = (np.asarray(plt.get_cmap('gist_rainbow')((angle+90)/180))[:3]*255).astype(np.uint8)
        np.testing.assert_array_equal(pixels[int(20*scale),int(x*scale)],expected)
