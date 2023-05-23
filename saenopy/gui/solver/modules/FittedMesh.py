import numpy as np
from qtpy import QtCore, QtWidgets
from pyvistaqt import QtInteractor
from saenopy.gui.common import QtShortCuts
from saenopy.gui.common.gui_classes import MatplotlibWidget, NavigationToolbar
from saenopy import Result

from .PipelineModule import PipelineModule
from .QTimeSlider import QTimeSlider
from .VTK_Toolbar import VTK_Toolbar
from .showVectorField import showVectorField2
from .DeformationDetector import CamPos


class FittedMesh(PipelineModule):
    pipeline_name = "fit forces"
    iteration_finished = QtCore.Signal(object, object)

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with self.parent.tabs.createTab("Fitted Deformations") as self.tab:
            with QtShortCuts.QVBoxLayout() as layout:
                self.label_tab = QtWidgets.QLabel("The fitted mesh deformations.").addToLayout()

                with QtShortCuts.QHBoxLayout() as layout:
                    self.plotter = QtInteractor(self, auto_update=False)  # , theme=pv.themes.DocumentTheme())
                    self.tab.parent().plotter = self.plotter
                    self.plotter.set_background("black")
                    layout.addWidget(self.plotter.interactor)

                    self.z_slider = QTimeSlider("z", self.z_slider_value_changed, "set z position",
                                                QtCore.Qt.Vertical).addToLayout()
                    self.z_slider.t_slider.valueChanged.connect(
                        lambda value: parent.shared_properties.change_property("z_slider", value, self))
                    parent.shared_properties.add_property("z_slider", self)

                self.vtk_toolbar = VTK_Toolbar(self.plotter, self.update_display, shared_properties=self.parent.shared_properties).addToLayout()

                self.t_slider = QTimeSlider(connected=self.update_display).addToLayout()
                self.tab.parent().t_slider = self.t_slider

    def z_slider_value_changed(self):
        self.update_display()

    def check_available(self, result: Result):
        return result is not None and result.solvers is not None

    def check_evaluated(self, result: Result) -> bool:
        try:
            return getattr(self.result.solvers[0], "regularisation_results", None) is not None
        except (AttributeError, IndexError, TypeError):
            return False

    def property_changed(self, name, value):
        if name == "z_slider":
            self.z_slider.setValue(value)

    def setResult(self, result: Result):
        super().setResult(result)
        if result and result.stacks and result.stacks[0]:
            self.z_slider.setRange(0, result.stacks[0].shape[2] - 1)
            self.z_slider.setValue(self.result.stacks[0].shape[2] // 2)

            if result.stacks[0].channels:
                self.vtk_toolbar.channel_select.setValues(np.arange(len(result.stacks[0].channels)), result.stacks[0].channels)
                self.vtk_toolbar.channel_select.setVisible(True)
            else:
                self.vtk_toolbar.channel_select.setValue(0)
                self.vtk_toolbar.channel_select.setVisible(False)

    def update_display(self):
        if self.current_tab_selected is False:
            self.current_result_plotted = False
            return

        if self.check_evaluated(self.result):
            cam_pos = None
            if self.plotter.camera_position is not None and CamPos.cam_pos_initialized is True:
                cam_pos = self.plotter.camera_position
            CamPos.cam_pos_initialized = True
            M = self.result.solvers[self.t_slider.value()]
            mesh = M.mesh
            self.plotter.interactor.setToolTip(str(self.result.solve_parameters) + f"\nNodes {mesh.nodes.shape[0]}\nTets {mesh.tetrahedra.shape[0]}")
            showVectorField2(self, mesh, "displacements")
            #showVectorField(self.plotter, M, M.U, "displacements", factor=0.1, scalebar_max=self.vtk_toolbar.getScaleMax(), show_nan=self.vtk_toolbar.use_nans.value())
            if cam_pos is not None:
                self.plotter.camera_position = cam_pos
        else:
            self.plotter.interactor.setToolTip("")

