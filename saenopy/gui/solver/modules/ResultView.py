import os
import qtawesome as qta
from qtpy import QtWidgets
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

import imageio
from saenopy.gui.common import QtShortCuts
from saenopy import Result, Solver
from saenopy.materials import SemiAffineFiberMaterial

from .PipelineModule import PipelineModule
from .QTimeSlider import QTimeSlider


result_view = None
class ResultView(PipelineModule):
    M: Solver = None

    def __init__(self, parent: "BatchEvaluate", layout):
        global result_view
        super().__init__(parent, layout)
        result_view = self

        with self.parent.tabs.createTab("View") as self.tab:
            with QtShortCuts.QVBoxLayout() as vlayout:
                with QtShortCuts.QHBoxLayout() as layout_vert_plot:
                    self.input_checks = {}
                    for name, dislay_name in {"displacements_target": "Target Deformations", "displacements": "Fitted Deformations", "forces": "Forces", "stiffness": "Stiffness"}.items():
                        input_bool = QtShortCuts.QInputBool(layout_vert_plot, dislay_name, name != "stiffness")
                        input_bool.valueChanged.connect(self.replot)
                        self.input_checks[name] = input_bool
                    layout_vert_plot.addStretch()
                    self.button_export = QtWidgets.QPushButton(qta.icon("mdi.floppy"), "")
                    self.button_export.setToolTip("save image")
                    layout_vert_plot.addWidget(self.button_export)
                    self.button_export.clicked.connect(self.saveScreenshot)

                # add the pyvista interactor object
                self.plotter_layout = QtWidgets.QHBoxLayout()
                self.plotter_layout.setContentsMargins(0, 0, 0, 0)
                self.frame = QtWidgets.QFrame().addToLayout()
                self.frame.setLayout(self.plotter_layout)

                self.plotter = QtInteractor(self.frame, auto_update=False)
                self.plotter_layout.addWidget(self.plotter.interactor)
                #vlayout.addLayout(self.plotter_layout)
                #return
                self.t_slider = QTimeSlider(connected=self.update_display).addToLayout()
                self.tab.parent().t_slider = self.t_slider

    def check_evaluated(self, result: Result) -> bool:
        try:
            return getattr(self.result.solvers[0], "regularisation_results", None) is not None
        except (AttributeError, IndexError, TypeError):
            return False

    def update_display(self):
        if self.check_evaluated(self.result):
            self.M = self.result.solvers[self.t_slider.value()]
            mesh = self.M.mesh
            R = mesh.nodes
            minR = np.min(R, axis=0)
            maxR = np.max(R, axis=0)

            if mesh.regularisation_mask is None:
                border = (R[:, 0] < minR[0] + 0.5e-6) | (R[:, 0] > maxR[0] - 0.5e-6) | \
                         (R[:, 1] < minR[1] + 0.5e-6) | (R[:, 1] > maxR[1] - 0.5e-6) | \
                         (R[:, 2] < minR[2] + 0.5e-6) | (R[:, 2] > maxR[2] - 0.5e-6)
                mesh.regularisation_mask = ~border

            self.point_cloud = pv.PolyData(mesh.nodes)
            self.point_cloud.point_data["forces"] = -mesh.forces * mesh.regularisation_mask[:, None]
            self.point_cloud["forces_mag"] = np.linalg.norm(mesh.forces * mesh.regularisation_mask[:, None], axis=1)
            self.point_cloud.point_data["displacements"] = mesh.displacements
            self.point_cloud["displacements_mag"] = np.linalg.norm(mesh.displacements, axis=1)
            self.point_cloud.point_data["displacements_target"] = mesh.displacements_target
            self.point_cloud["displacements_target_mag"] = np.linalg.norm(mesh.displacements_target, axis=1)
            nan_values = np.isnan(mesh.displacements_target[:, 0])
            self.point_cloud["displacements_target_mag"][nan_values] = 0

            self.point_cloud2 = None

            self.offset = np.min(mesh.nodes, axis=0)
            self.replot()
        else:
            self.plotter.interactor.setToolTip("")

    def calculateStiffness(self):
        self.point_cloud2 = pv.PolyData(np.mean(self.M.mesh.nodes[self.M.mesh.tetrahedra], axis=1))
        # self.M.setMaterialModel(SemiAffineFiberMaterial(1645, 0.0008, 0.0075, 0.033), generate_lookup=False)
        if self.M.material_model is None:
            print("Warning using default material parameters")
            self.M.set_material_model(SemiAffineFiberMaterial(1449, 0.00215, 0.055, 0.032), generate_lookup=False)
        self.M._check_relax_ready()
        self.M._prepare_temporary_quantities()
        self.point_cloud2["stiffness"] = self.M.get_max_tet_stiffness() / 6

    point_cloud = None

    theme = None
    def setTheme(self, x):
        self.theme = x
        self.current_result_plotted = False

    def replot(self):
        names = [name for name, input_widget in self.input_checks.items() if input_widget.value()]
        if len(names) == 0:
            return
        if len(names) <= 3:
            shape = (len(names), 1)
        else:
            shape = (2, 2)
        if self.plotter.shape != shape:
            self.plotter_layout.removeWidget(self.plotter)
            self.plotter.close()

            self.plotter = QtInteractor(self.frame, shape=shape, border=False, auto_update=False)

            self.plotter.set_background("black")
            # pv.set_plot_theme("document")
            self.plotter_layout.addWidget(self.plotter.interactor)

        if self.theme is not None:
            self.plotter._theme = self.theme
            self.plotter.set_background(self.theme.background)

        plotter = self.plotter
        # force rendering to be disabled while updating content to prevent flickering
        render = plotter.render
        plotter.render = lambda *args: None
        try:
            xmin, ymin, zmin = self.M.mesh.nodes.min(axis=0)
            xmax, ymax, zmax = self.M.mesh.nodes.max(axis=0)
            # color bar design properties
            # Set a custom position and size
            sargs = dict(#position_x=0.05, position_y=0.95,
                         title_font_size=15,
                         label_font_size=9,
                         n_labels=3,
                         #italic=True,  ##height=0.25, #vertical=True,
                         fmt="%.1e",
                         font_family="arial")

            for i, name in enumerate(names):
                plotter.subplot(i // plotter.shape[1], i % plotter.shape[1])
                # scale plot with axis length later
                norm_stack_size = np.abs(np.max(self.M.mesh.nodes) - np.min(self.M.mesh.nodes))

                if name == "stiffness":
                    if self.point_cloud2 is None:
                        self.calculateStiffness()
                    # clim =  np.nanpercentile(self.point_cloud2["stiffness"], [50, 99.9])
                    sargs2 = sargs.copy()
                    sargs2["title"] = "Stiffness (Pa)"
                    plotter.add_mesh(self.point_cloud2, colormap="turbo", point_size=4., render_points_as_spheres=True,
                                     scalar_bar_args=sargs2, opacity="linear")
                    plotter.update_scalar_bar_range(np.nanpercentile(self.point_cloud2["stiffness"], [50, 99.9]))
                elif name == "forces":
                    arrows = self.point_cloud.glyph(orient="forces", scale="forces_mag",
                                                    # Automatically scale maximal force to 15% of axis length
                                                    factor=0.15 * norm_stack_size / np.nanmax(
                                                        np.linalg.norm(self.M.mesh.forces * self.M.mesh.regularisation_mask[:, None], axis=1)))
                    sargs2 = sargs.copy()
                    sargs2["title"] = "Force (N)"
                    plotter.add_mesh(arrows, colormap='turbo', name="arrows", scalar_bar_args=sargs2)
                    plotter.update_scalar_bar_range(np.nanpercentile(self.point_cloud["forces_mag"], [50, 99.9]))
                    # plot center points if desired
                    # plotter.add_points(np.array([self.M.getCenter(mode="Force")]), color='r')

                elif name == "displacements_target":
                    arrows2 = self.point_cloud.glyph(orient=name, scale=name + "_mag",
                                                     # Automatically scale maximal force to 10% of axis length
                                                     factor=0.1 * norm_stack_size / np.nanmax(
                                                         np.linalg.norm(self.M.mesh.displacements_target, axis=1)))
                    sargs2 = sargs.copy()
                    sargs2["title"] = "Deformations (m)"
                    plotter.add_mesh(arrows2, colormap='turbo', name="arrows2", scalar_bar_args=sargs2)  #

                    # plot center if desired
                    # plotter.add_points(np.array([self.M.getCenter(mode="deformation_target")]), color='w')

                    plotter.update_scalar_bar_range(np.nanpercentile(self.point_cloud[name + "_mag"], [50, 99.9]))
                    # plotter.update_scalar_bar_range([0,1.5e-6])
                elif name == "displacements":
                    arrows3 = self.point_cloud.glyph(orient=name, scale=name + "_mag",
                                                     # Automatically scale maximal force to 10% of axis length
                                                     factor=0.1 * norm_stack_size / np.nanmax(
                                                         np.linalg.norm(self.M.mesh.displacements, axis=1)))
                    sargs2 = sargs.copy()
                    sargs2["title"] = "Fitted Deformations [m]"
                    plotter.add_mesh(arrows3, colormap='turbo', name="arrows3", scalar_bar_args=sargs2)
                    plotter.update_scalar_bar_range(np.nanpercentile(self.point_cloud[name + "_mag"], [50, 99.9]))
                    # plotter.update_scalar_bar_range([0,1.5e-6])

                # plot center points if desired
                # plotter.add_points(np.array([self.M.getCenter(mode="Deformation")]), color='w')
                # plotter.add_points(np.array([self.M.getCenter(mode="Force")]), color='r')

            # print(names)
            plotter.link_views()

            for i, name in enumerate(names):
                plotter.subplot(i // plotter.shape[1], i % plotter.shape[1])
                if self.theme is not None:
                    plotter.show_grid(bounds=[xmin, xmax, ymin, ymax, zmin, zmax], color=self.theme.font.color,
                                      render=False)
                else:
                    plotter.show_grid(bounds=[xmin, xmax, ymin, ymax, zmin, zmax], render=False)
        finally:
            plotter.render = render
            plotter.render()

    def saveScreenshot(self):
        new_path = QtWidgets.QFileDialog.getSaveFileName(self, "Save Image", os.getcwd(), "Image Files (*.jpg, *.png)")
        # if we got one, set it
        if new_path:
            imageio.imsave(new_path, self.plotter.image)
            print("saved", new_path)

