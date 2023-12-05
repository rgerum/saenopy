import matplotlib.pyplot as plt
from qtpy import QtCore, QtWidgets, QtGui
from saenopy.gui.common import QtShortCuts, QExtendedGraphicsView
from qimage2ndarray import array2qimage
import sys
import traceback
from PIL import Image, ImageDraw
from .PipelineModule import PipelineModule
from tifffile import imread
from saenopy.gui.common.gui_classes import CheckAbleGroup, QProcess, ProcessSimple
from .result import Result2D
from pyTFM.TFM_functions import TFM_tractions
from pyTFM.plotting import show_quiver
import numpy as np
from pyTFM.TFM_functions import strain_energy_points, contractillity
from scipy.ndimage.morphology import binary_fill_holes
from pyTFM.grid_setup_solids_py import interpolation # a simple function to resize the mask
from pyTFM.grid_setup_solids_py import prepare_forces
from pyTFM.grid_setup_solids_py import grid_setup, FEM_simulation
from pyTFM.grid_setup_solids_py import find_borders
from pyTFM.stress_functions import lineTension
from pyTFM.plotting import plot_continuous_boundary_stresses


class CalculateStress(PipelineModule):

    def __init__(self, parent=None, layout=None):
        super().__init__(parent, layout)
        self.parent = parent
        #layout.addWidget(self)
        with self.parent.tabs.createTab("Line Tension") as self.tab:
            pass

        with QtShortCuts.QVBoxLayout(self) as layout:
            layout.setContentsMargins(0, 0, 0, 0)
            with CheckAbleGroup(self, "stress", url="https://saenopy.readthedocs.io/en/latest/interface_solver.html#detect-deformations").addToLayout() as self.group:
                with QtShortCuts.QVBoxLayout():
                    self.label = QtWidgets.QLabel(
                        "draw a mask with the red color to select an area slightly larger then the colony. Draw a mask with the green color to circle every single cell and mark their boundaries.").addToLayout()
                    self.label.setWordWrap(True)
                    self.input_button = QtShortCuts.QPushButton(None, "calculate stress & line tensions", self.start_process)

        self.setParameterMapping("stress_parameters", {})

    def valueChanged(self):
        if self.check_available(self.result):
            im = imread(self.result.reference_stack).shape
            #voxel_size1 = self.result.stacks[0].voxel_size
            #stack_deformed = self.result.stacks[0]
            #overlap = 1 - (self.input_element_size.value() / self.input_win.value())
            #stack_size = np.array(stack_deformed.shape)[:3] * voxel_size1 - self.input_win.value()
            #self.label.setText(
            #    f"""Overlap between neighbouring windows\n(size={self.input_win.value()}µm or {(self.input_win.value() / np.array(voxel_size1)).astype(int)} px) is choosen \n to {int(overlap * 100)}% for an element_size of {self.input_element_size.value():.1f}μm elements.\nTotal region is {stack_size}.""")
        else:
            self.label.setText("")

    def check_available(self, result):
        return result.tx is not None

    def check_evaluated(self, result: Result2D) -> bool:
        return result.im_tension is not None

    def tabChanged(self, tab):
        if self.tab is not None and self.tab.parent() == tab:
            if self.check_evaluated(self.result):
                im = self.result.im_tension
                self.parent.draw.setImage(im*255)


    def process(self, result: Result2D, stress_parameters: dict): # type: ignore
        ps1 = result.pixel_size  # pixel size of the image of the beads
        # dimensions of the image of the beads
        ps2 = ps1 * np.mean(np.array(result.shape) / np.array(result.u.shape))  # pixel size of the deformation field

        # first mask: The area used for Finite Elements Methods
        # it should encircle all forces generated by the cell colony
        mask_FEM = binary_fill_holes(result.mask == 1)  # the mask should be a single patch without holes
        # changing the masks dimensions to fit to the deformation and traction field:
        mask_FEM = interpolation(mask_FEM, dims=result.tx.shape)

        # second mask: The area of the cells. Average stresses and other values are calculated only
        # on the actual area of the cell, represented by this mask.
        mask_cells = binary_fill_holes(result.mask == 2)
        mask_cells = interpolation(mask_cells, dims=result.tx.shape)

        # converting tractions (forces per surface area) to actual forces
        # and correcting imbalanced forces and torques
        # tx->traction forces in x direction, ty->traction forces in y direction
        # ps2->pixel size of the traction field, mask_FEM-> mask for FEM
        fx, fy = prepare_forces(result.tx, result.ty, ps2, mask_FEM)
        result.fx = fx
        result.fy = fy

        # constructing the FEM grid
        nodes, elements, loads, mats = grid_setup(mask_FEM, -fx, -fy, sigma=0.5)
        # performing the FEM analysis
        # verbose prints the progress of numerically solving the FEM system of equations.
        UG_sol, stress_tensor = FEM_simulation(nodes, elements, loads, mats, mask_FEM, verbose=True)
        # UG_sol is a list of deformations for each node. We don't need it here.

        # mean normal stress
        ms_map = ((stress_tensor[:, :, 0, 0] + stress_tensor[:, :, 1, 1]) / 2) / (ps2 * 10 ** -6)
        # average on the area of the cell colony.
        ms = np.mean(ms_map[mask_cells])  # 0.0043 N/m

        # coefficient of variation
        cv = np.nanstd(ms_map[mask_cells]) / np.abs(np.nanmean(ms_map[mask_cells]))  # 0.41 no unit

        result.ms = ms
        result.cv = cv

        """ Calculating the Line Tension """
        # identifying borders, counting cells, performing spline interpolation to smooth the borders
        borders = find_borders(result.mask == 2, result.tx.shape)
        # we can for example get the number of cells from the "borders" object
        n_cells = borders.n_cells  # 8

        # calculating the line tension along the cell borders
        lt, min_v, max_v = lineTension(borders.lines_splines, borders.line_lengths, stress_tensor, pixel_length=ps2)
        # lt is a nested dictionary. The first key is the id of a cell border.
        # For each cell border the line tension vectors ("t_vecs"), the normal
        # and shear component of the line tension ("t_shear") and the normal
        # vectors of the cell border ("n_vecs") are calculated at a large number of points.

        # average norm of the line tension. Only borders not at colony edge are used
        lt_vecs = np.concatenate([lt[l_id]["t_vecs"] for l_id in lt.keys() if l_id not in borders.edge_lines])
        avg_line_tension = np.mean(np.linalg.norm(lt_vecs, axis=1))  # 0.00569 N/m

        # average normal component of the line tension
        lt_normal = np.concatenate([lt[l_id]["t_normal"] for l_id in lt.keys() if l_id not in borders.edge_lines])
        avg_normal_line_tension = np.mean(np.abs(lt_normal))  # 0.00566 N/m,
        # here you can see that almost the line tensions act almost exclusively perpendicular to the cell borders.

        # plotting the line tension
        fig3, ax = plot_continuous_boundary_stresses([borders.inter_shape, borders.edge_lines, lt, min_v, max_v],
                                                     cbar_style="outside")
        plt.savefig("line_tension.png")
        ax.set_position([0, 0, 1, 1])
        fig3.set_dpi(100)
        fig3.set_size_inches(result.shape[1] / 100, result.shape[0] / 100)
        plt.savefig("tension.png")
        im = plt.imread("tension.png")
        result.im_tension = im

        result.save()