import numpy as np
import pytest
from saenopy.deformation_filter import filter_target_outliers, local_outliers
from saenopy.get_deformations import PivMesh, interpolate_different_mesh


def grid(n=5):
    return np.stack(np.meshgrid(*([np.arange(n)*1e-6]*3), indexing='ij'), axis=-1).reshape(-1, 3)


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_signed_component_reversal_excludes_whole_vector(axis):
    nodes = grid(); u = np.full_like(nodes, 1e-6)
    center = len(nodes)//2; u[center, axis] = -1e-6
    before = u.copy()
    cleaned, bad = filter_target_outliers(nodes, u, componentwise=True, eps_um=0, min_abs_um=0)
    assert bad.sum() == 1 and bad[center]
    assert np.isnan(cleaned[center]).all()
    np.testing.assert_array_equal(cleaned[~bad], before[~bad])
    np.testing.assert_array_equal(u, before)


def test_xy_variability_does_not_hide_small_z_spike():
    nodes = grid(7); center = len(nodes)//2
    u = np.zeros_like(nodes); u[:, 0] = nodes[:, 0]*10; u[center, 2] = .1e-6
    _, vector = filter_target_outliers(nodes, u, eps_um=0, min_abs_um=0)
    _, components = filter_target_outliers(nodes, u, componentwise=True, eps_um=0, min_abs_um=0)
    assert components[center] and not vector[center]


def test_nonfinite_vectors_do_not_support_statistics():
    nodes = grid(); u = np.zeros_like(nodes)
    u[0] = np.nan; u[1, 1] = np.inf; u[len(u)//2, 2] = 1e-6
    clean, bad = filter_target_outliers(nodes, u, componentwise=True, eps_um=0, min_abs_um=0)
    assert not bad[:2].any() and bad[len(u)//2]
    assert np.isnan(clean[0]).all() and np.isinf(clean[1, 1])
    _, small = filter_target_outliers(nodes[:1], u[:1], componentwise=True)
    assert not small.any()


def test_zero_scatter_and_relative_scale_invariance():
    nodes = grid(); u = np.zeros_like(nodes)
    _, constant = filter_target_outliers(nodes, u, componentwise=True, eps_um=0, min_abs_um=0)
    assert not constant.any()
    u[len(u)//2, 2] = 1e-7
    _, expected = filter_target_outliers(nodes, u, componentwise=True, eps_um=0, min_abs_um=0)
    for scale in (.001, 1e3):
        _, actual = filter_target_outliers(nodes, u*scale, componentwise=True, eps_um=0, min_abs_um=0)
        np.testing.assert_array_equal(actual, expected)


def test_current_nearest_fallback_fills_only_from_nearest_sample():
    nodes = grid(3); u = nodes.copy(); u[13] = np.nan
    points = np.array([[.2,.2,.2], [.8,.8,.8], [3,3,3]])*1e-6
    actual = interpolate_different_mesh(nodes, u, points)
    # Both interior queries have invalid linear support. The first has a valid
    # nearest sample; the second's nearest sample is itself invalid.
    np.testing.assert_array_equal(actual[0], u[0])
    assert np.isnan(actual[1:]).all()


def test_mesh_filters_piv_copy_before_interpolation(monkeypatch):
    import saenopy
    from saenopy import deformation_filter as df
    nodes = grid(7); u = np.zeros_like(nodes); u[len(u)//2, 2] = .1e-6
    before = u.copy(); mesh = PivMesh(); mesh.nodes = nodes
    calls = []; real_filter = df.filter_target_outliers
    real_interpolate = saenopy.get_deformations.interpolate_different_mesh
    def detect(p, values, **kw):
        assert len(p) == len(nodes) and kw['componentwise']
        calls.append('filter')
        return real_filter(p, values, **kw)
    def interpolate(p, values, targets):
        assert calls == ['filter'] and np.isnan(values[len(u)//2]).all()
        calls.append('interpolate')
        return real_interpolate(p, values, targets)
    monkeypatch.setattr(df, 'filter_target_outliers', detect)
    monkeypatch.setattr(saenopy.get_deformations, 'interpolate_different_mesh', interpolate)
    result = saenopy.interpolate_mesh(mesh, u, dict(mesh_size='piv', element_size=2,
        outlier_filter=True, outlier_eps_um=0., outlier_min_abs_um=0.))
    assert calls == ['filter', 'interpolate']
    np.testing.assert_array_equal(u, before)
    assert result.mesh.displacements_target is not None
