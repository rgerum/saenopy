import numpy as np
import pytest
from saenopy.deformation_filter import filter_target_outliers, _component_neighbourhoods


def grid(n=7, steps=(1, 1, 1)):
    return np.stack(np.meshgrid(*[np.arange(n)*h*1e-6 for h in steps],
                               indexing="ij"), axis=-1).reshape(-1, 3)


def detect(nodes, u, **kwargs):
    return filter_target_outliers(nodes, u, componentwise=True, eps_um=0,
                                  min_abs_um=0, **kwargs)[1]


def test_exact_124_offsets_exclude_center_and_do_not_duplicate_edges():
    nodes = grid()
    u = np.tile(np.arange(len(nodes))[:, None], (1, 3)).astype(float)
    ids, nb = next(_component_neighbourhoods(nodes, u, np.ones(len(nodes), bool)))
    center = len(nodes)//2
    expected = np.flatnonzero((np.abs(nodes - nodes[center]) < 2.01e-6).all(axis=1))
    expected = expected[expected != center]
    np.testing.assert_array_equal(np.sort(nb[center, :, 0]), expected)
    assert len(expected) == 124
    corner = nb[0, :, 0]
    assert np.isfinite(corner).sum() == 26
    assert len(np.unique(corner[np.isfinite(corner)])) == 26
    assert not np.any(corner == 0)


def test_nan_hole_does_not_expand_window_and_one_neighbor_is_used():
    nodes = grid(9); center = len(nodes)//2
    local = np.flatnonzero((np.abs(nodes-nodes[center]) < 2.01e-6).all(axis=1))
    local = local[local != center]
    u = np.zeros_like(nodes); u[local] = np.nan; u[center, 2] = 1e-6
    assert not detect(nodes, u)[center]
    u[local[0]] = 0
    assert detect(nodes, u)[center]


def test_six_neighbors_use_the_same_decision_limits():
    nodes = grid(); center = len(nodes)//2
    neighbours = np.flatnonzero((np.abs(nodes-nodes[center]) < 2.01e-6).all(axis=1))
    neighbours = neighbours[neighbours != center][:6]
    u = np.full_like(nodes, np.nan)
    u[neighbours] = 0
    # Neighbour median=1, MAD=0.1: threshold limit=.4 regardless of support.
    u[neighbours, 2] = np.array([.9]*3 + [1.1]*3)*1e-6
    u[center] = [0, 0, 1.3e-6]
    assert not detect(nodes, u, min_mult=0)[center]
    u[center, 2] = 1.6e-6
    assert detect(nodes, u, min_mult=0)[center]
    # The relative amplitude floor is still 0.5 um, not doubled.
    u[neighbours, 2] = 1e-6
    u[center, 2] = 1.4e-6
    assert not detect(nodes, u)[center]
    u[center, 2] = 1.8e-6
    assert detect(nodes, u)[center]


def test_small_coherent_cluster_does_not_define_its_own_reference():
    nodes = grid(); center = len(nodes)//2; u = np.zeros_like(nodes)
    u[np.linalg.norm(nodes-nodes[center], axis=1) < 1.01e-6, 2] = 1e-6
    assert detect(nodes, u)[center]


def test_axis_spacing_and_row_order_do_not_change_node_window():
    nodes = grid(); rng = np.random.default_rng(428)
    u = rng.normal(size=nodes.shape)*1e-6
    expected = detect(nodes, u)
    order = rng.permutation(len(nodes))
    np.testing.assert_array_equal(detect(nodes[order]*[1, 10, .01], u[order]), expected[order])


def test_known_smooth_affine_field_is_retained_at_edges_too():
    nodes = grid(9)
    matrix = np.array([[.12, -.08, .03], [.1, .2, -.09], [.02, -.01, .05]])
    u = nodes @ matrix + np.array([2, -3, 1])*1e-6
    assert not detect(nodes, u, thresh=4).any()


def test_legacy_k_does_not_limit_component_window():
    nodes = grid(); rng = np.random.default_rng(429); u = rng.normal(size=nodes.shape)*1e-6
    np.testing.assert_array_equal(detect(nodes, u, k=12), detect(nodes, u, k=26))


def test_missing_grid_node_is_an_error_not_a_larger_window():
    nodes = grid()
    with pytest.raises(ValueError, match="complete Cartesian"):
        detect(nodes[:-1], np.zeros_like(nodes[:-1]))
    nodes[0] = nodes[1]
    with pytest.raises(ValueError, match="unique Cartesian"):
        detect(nodes, np.zeros_like(nodes))


def test_no_neighbors_means_no_decision_and_no_fill():
    nodes = grid(); center = len(nodes)//2
    u = np.full_like(nodes, np.nan); u[center] = [1, 2, 3]
    cleaned, bad = filter_target_outliers(nodes, u, componentwise=True)
    assert not bad.any()
    np.testing.assert_array_equal(cleaned, u)
