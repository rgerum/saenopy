from qtpy import QtWidgets, QtCore
import numpy as np
from pyvistaqt import QtInteractor

from saenopy import Result
from saenopy.gui.common import QtShortCuts

from saenopy.gui.common.QTimeSlider import QTimeSlider
from .VTK_Toolbar import VTK_Toolbar
from .showVectorField import showVectorField2
from saenopy.gui.common.TabModule import TabModule


class CamPos:
    cam_pos_initialized = False


class TabPiv(TabModule):

    def __init__(self, parent: "BatchEvaluate"):
        super().__init__(parent)

        with self.parent.tabs.createTab("PIV Deformations") as self.tab:
            with QtShortCuts.QVBoxLayout() as layout:
                self.label_tab = QtWidgets.QLabel(
                    "The deformations from the piv algorithm at every window where the crosscorrelation was evaluated.").addToLayout()

                with QtShortCuts.QHBoxLayout() as layout:
                    self.plotter = QtInteractor(None, auto_update=False)  # , theme=pv.themes.DocumentTheme())
                    self.tab.parent().plotter = self.plotter
                    layout.addWidget(self.plotter.interactor)

                    self.z_slider = QTimeSlider("z", self.z_slider_value_changed, "set z position",
                                                QtCore.Qt.Vertical).addToLayout()
                    self.z_slider.t_slider.valueChanged.connect(
                        lambda value: parent.shared_properties.change_property("z_slider", value, self))
                    parent.shared_properties.add_property("z_slider", self)

                self.vtk_toolbar = VTK_Toolbar(self.plotter, self.update_display, "deformation", z_slider=self.z_slider, shared_properties=self.parent.shared_properties).addToLayout()

                self.t_slider = QTimeSlider(connected=self.update_display).addToLayout()
                self.tab.parent().t_slider = self.t_slider

    def z_slider_value_changed(self):
        self.update_display()

    def check_evaluated(self, result: Result) -> bool:
        try:
            return self.result.mesh_piv[0] is not None
        except (AttributeError, IndexError):
            return False

    def property_changed(self, name, value):
        if name == "z_slider":
            self.z_slider.setValue(value)

    def setResult(self, result: Result):
        super().setResult(result)
        if result and result.stacks and result.stacks[0]:
            self.z_slider.setRange(0, result.stacks[0].shape[2] - 1)
            self.z_slider.setValue(result.stacks[0].shape[2] // 2)

            if result.stacks[0].channels:
                value = self.vtk_toolbar.channel_select.value()
                self.vtk_toolbar.channel_select.setValues(np.arange(len(result.stacks[0].channels)), result.stacks[0].channels)
                self.vtk_toolbar.channel_select.setValue(value)
                self.vtk_toolbar.channel_select.setVisible(True)
            else:
                self.vtk_toolbar.channel_select.setValue(0)
                self.vtk_toolbar.channel_select.setVisible(False)

    def update_display(self, *, plotter=None):
        if self.current_tab_selected is False:
            self.current_result_plotted = False
            return

        if plotter is None:
            plotter = self.plotter
        cam_pos = None
        if plotter.camera_position is not None and CamPos.cam_pos_initialized is True:
            cam_pos = self.plotter.camera_position
        CamPos.cam_pos_initialized = True
        plotter.interactor.setToolTip("")
        if self.result is None:
            mesh = None
        else:
            mesh = self.result.mesh_piv[self.t_slider.value()]

        if mesh is None:
            plotter.show()
            return

        plotter.interactor.setToolTip(
            str(self.result.piv_parameters) + f"\nNodes {mesh.nodes.shape[0]}\nTets {mesh.tetrahedra.shape[0]}")

        if mesh.displacements_measured is not None:
            showVectorField2(self, mesh, "displacements_measured")

        if cam_pos is not None:
            plotter.camera_position = cam_pos
