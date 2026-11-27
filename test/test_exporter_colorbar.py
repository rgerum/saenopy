from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image, ImageDraw

from saenopy.gui.solver.modules.exporter import ExporterRender2D as renderer
from saenopy.gui.solver.modules.exporter import ExporterRender3D as renderer3d


@pytest.mark.parametrize('unit,minimum,maximum,expected', [
    ('µm', 0, 1e-12, ['0', '5e-13', '1e-12', 'µm']),  # formerly KeyError -18
    ('µm', 0, 1e-15, ['0', '5e-16', '1e-15', 'µm']),  # formerly KeyError -21
    ('µm', 0, 1e-27, ['0', '5e-28', '1e-27', 'µm']),  # formerly KeyError -33
    ('pN', 0, 1e-6, ['0', '5e-07', '1e-06', 'pN']),
    ('pN', 0, 1e6, ['0', '0.5', '1', 'µN']),  # convert only once
    ('pN', 0, 1e12, ['0', '0.5', '1', 'N']),  # no SI prefix
    ('µm', 0, 1e6, ['0', '0.5', '1', 'm']),
    ('µm', 0, 1, ['0', '0.5', '1', 'µm']),
    ('um', 0, 0, ['0', '0', '0', 'um']),
    ('px', 0, 1000, ['0', '500', '1000', 'px']),
    ('log10(F/pN)', 0, 6, ['0', '3', '6', 'log10(F/pN)']),
    (None, -0.1, 0.1, ['-0.1', '0', '0.1']),
    ('µm', -10, 0, ['-10', '-5', '0', 'µm']),
])
def test_colorbar_labels_preserve_values_and_units(unit, minimum, maximum, expected):
    image = Image.new('RGB', (500, 160))
    with patch.object(ImageDraw.ImageDraw, 'text', autospec=True) as draw:
        assert renderer.add_colorbar(image, min_v=minimum, max_v=maximum, unit=unit) is image
    assert [call.args[2] for call in draw.call_args_list] == expected


@pytest.mark.parametrize('force,logarithmic,unit', [
    (True, False, 'pN'),
    (True, True, 'log10(F/pN)'),
    (False, False, 'µm'),
])
def test_3d_colorbar_unit_matches_rendered_field(force, logarithmic, unit):
    params = deepcopy(renderer.default_data)
    params['force_arrows']['use_log'] = logarithmic
    name = 'forces' if force else 'displacements'
    arrow_params = params['force_arrows' if force else 'deformation_arrows']
    obj = SimpleNamespace(nodes=np.array([[0., 0., 0.], [10e-6, 0., 0.]]))
    field = np.array([[100e-12, 0., 0.], [1000e-12, 0., 0.]]) if force else np.array([[1e-6, 0., 0.], [2e-6, 0., 0.]])
    plotter = SimpleNamespace(_theme=SimpleNamespace(font=SimpleNamespace(color='white')),
                             add_mesh=Mock(), remove_actor=Mock(),
                             update_scalar_bar_range=Mock(), remove_scalar_bar=Mock())
    with patch.object(renderer3d, 'get_mesh_arrows', return_value=(obj, field, arrow_params, name)):
        display = renderer3d.render_3d_arrows(params, object(), plotter)
    assert display['scalebar_unit'] == unit
    expected_max = np.percentile(np.log10([100., 1000.]) if logarithmic else ([100., 1000.] if force else [1., 2.]), 99.9)
    assert display['scale_max'] == pytest.approx(expected_max)
    with patch.object(ImageDraw.ImageDraw, 'text', autospec=True) as draw:
        renderer.render_2d_colorbar(params, None, Image.new('RGB', (500, 160)), 1, 1,
                                   scale_max=display['scale_max'], unit=display['scalebar_unit'])
    assert draw.call_args_list[-1].args[2] == unit
