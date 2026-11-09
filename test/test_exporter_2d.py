from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from saenopy.gui.solver.modules.exporter import ExporterRender2D as renderer
from saenopy.gui.spheroid.modules.result import ResultSpheroid


class Result:
    time_delta = 90

    def __init__(self):
        self.image_calls = 0
        x, y = np.meshgrid(np.arange(8, 73, 8), np.arange(8, 57, 8))
        self.nodes = np.column_stack([x.ravel(), y.ravel()]).astype(float)
        self.field = np.tile([3., -2.], (len(self.nodes), 1))

    def get_data_structure(self):
        return dict(time_point_count=3, voxel_size=[1, 1, 1], z_slices_count=1,
                    time_delta=self.time_delta, fields={
                        'deformation': dict(measure='deformation', name='displacements_measured')})

    def get_image_data(self, t, channel, reference):
        self.image_calls += 1
        return np.full((64, 80, 1, 1), 30+t+int(reference)+10*(channel=='second'), dtype=np.uint8)

    def get_field_data(self, name, t):
        mesh = SimpleNamespace(nodes=self.nodes, units='pixels')
        return mesh, self.field


def parameters():
    params = deepcopy(renderer.default_data)
    params['stack']['channel'] = 'default'
    params['stack']['contrast_enhance'] = (0, 255)
    params['image']['antialiase'] = False
    params['time']['display'] = False
    params['scalebar']['hide'] = True
    params['colorbar']['hide'] = True
    params['deformation_arrows']['autoscale'] = False
    return params


def test_spheroid_without_force_reconstruction_roundtrips(tmp_path):
    path = tmp_path / 'unfitted.saenopySpheroid'
    result = ResultSpheroid('*.tif', [], path, pixel_size=3.45, time_delta=90,
                            piv_parameters=dict(window_size=50, overlap=50,
                                                n_min=None, n_max=None,
                                                thresh_segmentation=0.62,
                                                continuous_segmentation=False))
    result.save()
    loaded = ResultSpheroid.load(path)
    assert loaded.force_parameters == {}
    assert loaded.res_data == {}
    assert loaded.pixel_size == 3.45
    assert loaded.time_delta == 90


def test_arrow_edits_reuse_background_without_accumulating_overlays():
    result, exporter, params = Result(), SimpleNamespace(), parameters()
    renderer.render_2d(params, result, exporter)
    params['deformation_arrows']['arrow_scale'] = 4
    second = renderer.render_2d(params, result, exporter)
    assert result.image_calls == 1
    expected = renderer.render_2d(params, result)
    np.testing.assert_array_equal(second, expected)
    repeated = renderer.render_2d(params, result, exporter)
    np.testing.assert_array_equal(repeated, expected)
    assert result.image_calls == 2  # one initial read and one uncached reference


@pytest.mark.parametrize('group,key,value', [
    ('time', 't', 1), ('stack', 'channel', 'second'),
    ('stack', 'use_reference_stack', True),
    ('stack', 'contrast_enhance', (10, 240)), ('stack', 'colormap', 'turbo'),
    ('stack', 'alpha', 0.5), ('stack', 'channel_B', 'second'),
    ('image', 'scale', 0.5), ('image', 'antialiase', True),
])
def test_background_edits_invalidate_cache(group, key, value):
    result, exporter, params = Result(), SimpleNamespace(), parameters()
    renderer.render_2d(params, result, exporter)
    params[group][key] = value
    actual = renderer.render_2d(params, result, exporter)
    assert result.image_calls > 1
    np.testing.assert_array_equal(actual, renderer.render_2d(params, result))


def test_cache_is_local_to_exporter_and_result():
    first, second, params = Result(), Result(), parameters()
    exporter = SimpleNamespace()
    renderer.render_2d(params, first, exporter)
    renderer.render_2d(params, second, exporter)
    renderer.render_2d(params, first, SimpleNamespace())
    assert first.image_calls == 2
    assert second.image_calls == 1


def test_second_channel_does_not_mutate_parameters():
    result, params = Result(), parameters()
    params['stack']['channel_B'] = 'second'
    expected = deepcopy(params)
    renderer.render_2d_image(params, result, None)
    assert params == expected


@pytest.mark.parametrize('skip', [1, 2, 3])
def test_native_2d_skip_subsamples_both_axes(skip):
    result, params = Result(), parameters()
    params['deformation_arrows']['skip'] = skip
    with patch.object(renderer, 'add_quiver', side_effect=lambda image, *args, **kw: image) as draw:
        renderer.render_2d(params, result)
    positions, lengths, angles, colors = draw.call_args.args[1:5]
    expected = np.array([(x, 64-y) for y in range(8, 57, 8*skip) for x in range(8, 73, 8*skip)])
    np.testing.assert_array_equal(positions, expected)
    assert len(lengths) == len(angles) == len(colors) == len(expected)


@pytest.mark.parametrize('image_scale,antialias', [(1, False), (0.5, False), (0.5, True), (2, True)])
def test_compact_arrows_cap_display_length_without_changing_colors_or_directions(image_scale, antialias):
    result, exporter, params = Result(), SimpleNamespace(), parameters()
    result.field[:] = np.linspace(0, 200, len(result.field))[:, None] * [0.6, -0.8]
    original_field, original_nodes = result.field.copy(), result.nodes.copy()
    params['image'].update(scale=image_scale, antialiase=antialias)
    params['deformation_arrows']['arrow_scale'] = 6
    with patch.object(renderer, 'add_quiver', side_effect=lambda image, *args, **kw: image) as draw:
        renderer.render_2d(params, result, exporter)
        linear = draw.call_args
        params['2D_arrows'].update(compact=True, max_length=60, width_percent=0.2)
        renderer.render_2d(params, result, exporter)
        compact = draw.call_args
    np.testing.assert_allclose(compact.args[2], np.minimum(linear.args[2], 60))
    assert np.max(compact.args[2]) == 60
    for index in (1, 3, 4):  # positions, angles and colors are identical
        np.testing.assert_array_equal(compact.args[index], linear.args[index])
    assert compact.kwargs['width'] == pytest.approx(80 * 0.002)
    assert compact.kwargs['headlength'] == pytest.approx(3.5 * 80 * 0.002)
    assert compact.kwargs['headheight'] == pytest.approx(3 * 80 * 0.002)
    assert compact.kwargs['scale'] == image_scale * (2 if antialias else 1)
    assert result.image_calls == 1
    np.testing.assert_array_equal(result.field, original_field)
    np.testing.assert_array_equal(result.nodes, original_nodes)


def test_old_arrow_parameters_keep_linear_geometry():
    result, params = Result(), parameters()
    params['2D_arrows'] = dict(width=2, headlength=5, headheight=5)
    params['deformation_arrows']['arrow_scale'] = 100
    with patch.object(renderer, 'add_quiver', side_effect=lambda image, *args, **kw: image) as draw:
        renderer.render_2d(params, result)
    assert np.min(draw.call_args.args[2]) > 60
    assert draw.call_args.kwargs == dict(scale=1, width=2, headlength=5, headheight=5)


def test_compact_geometry_matches_matplotlib_quiver_including_short_and_zero_arrows():
    import matplotlib.pyplot as plt
    width = 8.224
    lengths = np.array([0., 4., 20., 60.])
    angles = np.array([0., np.pi/4, -np.pi/2, np.pi])
    nodes = np.array([[10., 20.]] * 4)
    original = lengths.copy()
    actual = renderer.getarrow(lengths, angles, scale=0.5, width=width,
                               headlength=3.5*width, headheight=3*width,
                               headaxislength=3.1*width, offset=nodes*0.5)
    fig, ax = plt.subplots()
    try:
        quiver = ax.quiver(nodes[:, 0], nodes[:, 1], lengths, np.zeros(4),
                           width=0.002, headwidth=3, headlength=3.5,
                           headaxislength=3.1, minlength=0, pivot='tail')
        # Matplotlib's horizontal template is expressed in units of shaft width.
        x, y = quiver._h_arrows(lengths/width)
        expected = np.stack([x[:, :7], y[:, :7]], axis=2) * width
        rotation = np.array([[np.cos(angles), np.sin(angles)],
                             [-np.sin(angles), np.cos(angles)]])
        expected = np.einsum('ijk,kli->ijl', expected, rotation)
        np.testing.assert_allclose(actual, (expected + nodes[:, None, :])*0.5, atol=1e-12)
        np.testing.assert_array_equal(lengths, original)
    finally:
        plt.close(fig)


def test_quiver_skips_nonfinite_vectors():
    from PIL import Image
    image = Image.new('RGB', (80, 64))
    renderer.add_quiver(image, np.array([[10., 20.], [20., 30.]]),
                        np.array([10., np.nan]), np.array([0., 1.]),
                        np.array([[255, 0, 0, 255], [0, 255, 0, 255]], dtype=np.uint8))
    assert np.asarray(image)[:, :, 0].any()
    assert not np.asarray(image)[:, :, 1].any()


def test_export_gui_compact_controls_and_saved_parameters():
    from qtpy import QtWidgets
    from saenopy.gui.solver.modules.exporter.Exporter import ExportViewer
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    viewer = ExportViewer(None, None)
    try:
        initial = viewer.get_parameters()['2D_arrows']
        assert initial['compact'] is True
        assert initial['max_length'] == 60
        assert initial['width_percent'] == 0.2
        assert viewer.input_2darrow_max_length.isEnabled()
        assert not viewer.input_2darrow_width.isEnabled()
        modified = dict(initial, max_length=75, width_percent=0.3)
        viewer.set_parameters({'2D_arrows': modified})
        assert viewer.get_parameters()['2D_arrows'] == modified
        viewer.set_parameters({'2D_arrows': dict(width=3, headlength=7, headheight=6)})
        assert viewer.input_2darrow_compact.value() is False
        assert viewer.input_2darrow_width.isEnabled()
        assert not viewer.input_2darrow_max_length.isEnabled()
        viewer.set_parameters({'2D_arrows': modified})
        assert viewer.get_parameters()['2D_arrows'] == modified
        assert viewer.input_2darrow_max_length.isEnabled()
    finally:
        viewer.plotter.close()
        viewer.export_window.close()
        viewer.close()
        app.processEvents()
