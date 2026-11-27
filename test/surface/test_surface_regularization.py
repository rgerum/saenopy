"""
Lightweight unit tests for saenopy.surface_regularization (pure-geometry parts,
no heavy data needed).  Run: pytest test_surface_regularization.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from saenopy import surface_regularization as sr


def test_surface_node_mask():
    # shell = a few points on a plane at x=0; nodes at various distances
    shell = np.array([[0, 0, 0], [0, 0, 10e-6], [0, 10e-6, 0]], float)
    nodes = np.array([[0, 0, 0],          # on shell -> in
                      [0, 0, 3e-6],       # 3 um away -> in (radius 5 um)
                      [0, 0, 20e-6]], float)  # 20 um away -> out
    m = sr.surface_node_mask(nodes, shell, element_size_um=10.0)  # radius 5 um
    assert list(m) == [True, True, False]


def test_inside_cell_mask():
    # 10x10x10 body, a solid 4..6 cube marked True
    vol = np.zeros((10, 10, 10), bool); vol[4:7, 4:7, 4:7] = True
    voxel = (1.0, 1.0, 1.0)            # 1 um isotropic
    shape = (10, 10, 10)
    # center is (shape-1)/2 = 4.5; node at physical 0 -> voxel index 4.5 -> ~4/5 (inside)
    nodes = np.array([[0, 0, 0],                    # center -> inside cube
                      [4e-6, 4e-6, 4e-6]], float)   # near edge of grid -> outside cube
    inside = sr.inside_cell_mask(nodes, vol, voxel, shape)
    assert inside[0] == True
    assert inside[1] == False


def test_segment_coordinates_match_anisotropic_voxels():
    image = np.zeros((9, 11, 13))
    image[3:6, 4:7, 5:8] = 10
    body, shell = sr.segment_cell(image, [2., 3., 4.], threshold=5,
                                  smooth=0, close_radius=0)
    from skimage.morphology import erosion
    expected = np.argwhere(body & ~erosion(body)).astype(float)
    expected -= (np.array(image.shape) - 1) / 2
    expected *= np.array([3., 2., 4.]) * 1e-6
    np.testing.assert_allclose(shell, expected)
