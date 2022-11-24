import os
import qtawesome as qta
from qtpy import QtWidgets
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

import imageio
from saenopy.gui import QtShortCuts
from saenopy.solver import Result, Solver

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
                    for name, dislay_name in {"U_target": "Target Deformations", "U": "Fitted Deformations", "f": "Forces", "stiffness": "Stiffness"}.items():
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

                self.plotter = QtInteractor(self.frame)
                self.plotter_layout.addWidget(self.plotter.interactor)
                vlayout.addLayout(self.plotter_layout)

                self.t_slider = QTimeSlider(connected=self.update_display).addToLayout()
                self.tab.parent().t_slider = self.t_slider
        self.setParameterMapping(None, {})

    def check_evaluated(self, result: Result) -> bool:
        return self.result is not None and self.result.solver is not None and getattr(self.result.solver[0], "regularisation_results", None) is not None

    def update_display(self):
        if self.check_evaluated(self.result):
            self.M = self.result.solver[self.t_slider.value()]
            R = self.M.R
            minR = np.min(R, axis=0)
            maxR = np.max(R, axis=0)

            if self.M.reg_mask is None:
                border = (R[:, 0] < minR[0] + 0.5e-6) | (R[:, 0] > maxR[0] - 0.5e-6) | \
                         (R[:, 1] < minR[1] + 0.5e-6) | (R[:, 1] > maxR[1] - 0.5e-6) | \
                         (R[:, 2] < minR[2] + 0.5e-6) | (R[:, 2] > maxR[2] - 0.5e-6)
                self.M.reg_mask = ~border

            self.point_cloud = pv.PolyData(self.M.R)
            self.point_cloud.point_data["f"] = -self.M.f * self.M.reg_mask[:, None]
            self.point_cloud["f_mag"] = np.linalg.norm(self.M.f * self.M.reg_mask[:, None], axis=1)
            self.point_cloud.point_data["U"] = self.M.U
            self.point_cloud["U_mag"] = np.linalg.norm(self.M.U, axis=1)
            self.point_cloud.point_data["U_target"] = self.M.U_target
            self.point_cloud["U_target_mag"] = np.linalg.norm(self.M.U_target, axis=1)
            nan_values = np.isnan(self.M.U_target[:, 0])
            self.point_cloud["U_target_mag"][nan_values] = 0

            self.point_cloud2 = None

            self.offset = np.min(self.M.R, axis=0)
            self.replot()
        else:
            self.plotter.interactor.setToolTip("")

    def calculateStiffness(self):
        self.point_cloud2 = pv.PolyData(np.mean(self.M.R[self.M.T], axis=1))
        from saenopy.materials import SemiAffineFiberMaterial
        # self.M.setMaterialModel(SemiAffineFiberMaterial(1645, 0.0008, 0.0075, 0.033), generate_lookup=False)
        if self.M.material_model is None:
            print("Warning using default material parameters")
            self.M.setMaterialModel(SemiAffineFiberMaterial(1449, 0.00215, 0.055, 0.032), generate_lookup=False)
        self.M._check_relax_ready()
        self.M._prepare_temporary_quantities()
        self.point_cloud2["stiffness"] = self.M.getMaxTetStiffness() / 6

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

            self.plotter = QtInteractor(self.frame, shape=shape, border=False)

            self.plotter.set_background("black")
            # pv.set_plot_theme("document")
            self.plotter_layout.addWidget(self.plotter.interactor)

        if self.theme is not None:
            self.plotter._theme = self.theme
            self.plotter.set_background(self.theme.background)

        plotter = self.plotter
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
            norm_stack_size = np.abs(np.max(self.M.R) - np.min(self.M.R))

            if name == "stiffness":
                if self.point_cloud2 is None:
                    self.calculateStiffness()
                # clim =  np.nanpercentile(self.point_cloud2["stiffness"], [50, 99.9])
                sargs2 = sargs.copy()
                sargs2["title"] = "Stiffness (Pa)"
                plotter.add_mesh(self.point_cloud2, colormap="turbo", point_size=4., render_points_as_spheres=True,
                                 scalar_bar_args=sargs2, opacity="linear")
                plotter.update_scalar_bar_range(np.nanpercentile(self.point_cloud2["stiffness"], [50, 99.9]))
            elif name == "f":
                arrows = self.point_cloud.glyph(orient="f", scale="f_mag",
                                                # Automatically scale maximal force to 15% of axis length
                                                factor=0.15 * norm_stack_size / np.nanmax(
                                                    np.linalg.norm(self.M.f * self.M.reg_mask[:, None], axis=1)))
                sargs2 = sargs.copy()
                sargs2["title"] = "Force (N)"
                plotter.add_mesh(arrows, colormap='turbo', name="arrows", scalar_bar_args=sargs2)
                plotter.update_scalar_bar_range(np.nanpercentile(self.point_cloud["f_mag"], [50, 99.9]))
                # plot center points if desired
                # plotter.add_points(np.array([self.M.getCenter(mode="Force")]), color='r')

            elif name == "U_target":
                arrows2 = self.point_cloud.glyph(orient=name, scale=name + "_mag",
                                                 # Automatically scale maximal force to 10% of axis length
                                                 factor=0.1 * norm_stack_size / np.nanmax(
                                                     np.linalg.norm(self.M.U_target, axis=1)))
                sargs2 = sargs.copy()
                sargs2["title"] = "Deformations (m)"
                plotter.add_mesh(arrows2, colormap='turbo', name="arrows2", scalar_bar_args=sargs2)  #

                # plot center if desired
                # plotter.add_points(np.array([self.M.getCenter(mode="deformation_target")]), color='w')

                plotter.update_scalar_bar_range(np.nanpercentile(self.point_cloud[name + "_mag"], [50, 99.9]))
                # plotter.update_scalar_bar_range([0,1.5e-6])
            elif name == "U":
                arrows3 = self.point_cloud.glyph(orient=name, scale=name + "_mag",
                                                 # Automatically scale maximal force to 10% of axis length
                                                 factor=0.1 * norm_stack_size / np.nanmax(
                                                     np.linalg.norm(self.M.U, axis=1)))
                sargs2 = sargs.copy()
                sargs2["title"] = "Fitted Deformations [m]"
                plotter.add_mesh(arrows3, colormap='turbo', name="arrows3", scalar_bar_args=sargs2)
                plotter.update_scalar_bar_range(np.nanpercentile(self.point_cloud[name + "_mag"], [50, 99.9]))
                # plotter.update_scalar_bar_range([0,1.5e-6])

            # plot center points if desired
            # plotter.add_points(np.array([self.M.getCenter(mode="Deformation")]), color='w')
            # plotter.add_points(np.array([self.M.getCenter(mode="Force")]), color='r')

            if self.theme is not None:
                plotter.show_grid(color=self.theme.font.color)
            else:
                plotter.show_grid()

        # print(names)
        plotter.link_views()
        plotter.show()

    def saveScreenshot(self):
        new_path = QtWidgets.QFileDialog.getSaveFileName(self, "Save Image", os.getcwd(), "Image Files (*.jpg, *.png)")
        # if we got one, set it
        if new_path:
            if isinstance(new_path, tuple):
                new_path = new_path[0]
            else:
                new_path = str(new_path)
            imageio.imsave(new_path, self.plotter.image)
            print("saved", new_path)

