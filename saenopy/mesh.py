from typing import Union

import numpy as np
from nptyping import NDArray, Shape, Float, Int
from valid8 import ValidationFailure

from saenopy.saveable import Saveable


class Mesh(Saveable):
    __save_parameters__ = ['nodes', 'tetrahedra']
    nodes: NDArray[Shape["N_c, 3"], Float] = None  # the 3D positions of the vertices, dimension: N_c x 3
    tetrahedra: NDArray[Shape["N_t, 4"], Int] = None  # the tetrahedra' 4 corner vertices (defined by index), dimensions: N_T x 4

    def __init__(self, nodes: NDArray[Shape["N_c, 3"], Float] = None, tetrahedra: NDArray[Shape["N_t, 4"], Int] = None,
                 **kwargs):
        super().__init__(**kwargs)
        if nodes is not None:
            self.set_nodes(nodes)
        if tetrahedra is not None:
            self.set_tetrahedra(tetrahedra)

    def set_nodes(self, data: NDArray[Shape["N_c, 3"], Float]):
        """
        Provide mesh coordinates.

        Parameters
        ----------
        data : ndarray
            The coordinates of the vertices. Dimensions Nx3
        """
        # check the input
        data = np.asarray(data)
        assert len(data.shape) == 2, "Mesh node data needs to be Nx3."
        assert data.shape[1] == 3, "Mesh vertices need to have 3 spacial coordinate."

        # store the loaded node coordinates
        self.nodes = data.astype(np.float64)

    def set_tetrahedra(self, data: NDArray[Shape["N_t, 4"], Int]):
        """
        Provide mesh connectivity. Nodes have to be connected by tetrahedra. Each tetrahedron consts of the indices of
        the 4 vertices which it connects.

        Parameters
        ----------
        data : ndarray
            The node indices of the 4 corners. Dimensions Nx4
        """
        # check the input
        data = np.asarray(data)
        assert len(data.shape) == 2, "Mesh tetrahedra needs to be Nx4."
        assert data.shape[1] == 4, "Mesh tetrahedra need to have 4 corners."
        assert 0 <= data.min(), "Mesh tetrahedron node indices are not allowed to be negative."
        assert data.max() < self.nodes.shape[0], "Mesh tetrahedron node indices cannot be bigger than the number of vertices."

        # store the tetrahedron data (needs to be int indices)
        self.tetrahedra = data.astype(int)


class InvalidShape(ValidationFailure):
    help_msg = '{msg} (shape {target_shape}) found shape ({data_shape})'


def check_tetrahedra_scalar_field(self, data: NDArray[Shape["N_t"], Float]):
    if data is None or self.tetrahedra is None:
        return True
    data = np.asarray(data)
    if len(data.shape) != 1 or (self.tetrahedra is not None and data.shape[0] != self.tetrahedra.shape[0]):
        raise InvalidShape(data, data_shape=data.shape, target_shape=(self.tetrahedra.shape[0],),
                           msg="Tetrahedral field needs to be scalar")


def check_node_scalar_field(self, data: NDArray[Shape["N_c"], Float]):
    if data is None or self.nodes is None:
        return True
    data = np.asarray(data)
    if len(data.shape) != 1 or (self.nodes is not None and data.shape[0] != self.nodes.shape[0]):
        raise InvalidShape(data, data_shape=data.shape, target_shape=(self.nodes.shape[0],),
                           msg="Node field needs to be scalar")


def check_node_vector_field(self, data: NDArray[Shape["N_c"], Float]):
    if data is None or self.nodes is None:
        return True
    data = np.asarray(data)
    if len(data.shape) != 2 or (self.nodes is not None and data.shape[0] != self.nodes.shape[0]) or data.shape[1] != 3:
        raise InvalidShape(data, data_shape=data.shape, target_shape=(self.nodes.shape[0], 3),
                           msg="Node field needs to be vectorial")
