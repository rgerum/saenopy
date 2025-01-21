from qtpy import QtCore, QtWidgets
import numpy as np
from pyvistaqt import QtInteractor

from saenopy import Result
from saenopy.gui.common import QtShortCuts

from saenopy.gui.common.QTimeSlider import QTimeSlider
from saenopy.gui.common.TabModule import TabModule
from .VTK_Toolbar import VTK_Toolbar
from .showVectorField import showVectorField, getVectorFieldImage
from .DeformationDetector import CamPos

import matplotlib.ticker as ticker


class OmitLast30PercentLocator(ticker.AutoLocator):
    def __call__(self):
        ticks = super(OmitLast30PercentLocator, self).__call__()
        # Safely fetch the axis limits without modifying them
        lim = self.axis.get_view_interval()
        cutoff = lim[0] + (lim[1] - lim[0]) * 0.7
        return [t for t in ticks if t < cutoff]


class TabForces(TabModule):
    pipeline_name = "fit forces"
    iteration_finished = QtCore.Signal(object, object, int, int)

    def __init__(self, parent: "BatchEvaluate"):
        super().__init__(parent)

        with self.parent.tabs.createTab("Forces") as self.tab:
            with QtShortCuts.QVBoxLayout() as layout:
                self.label_tab = QtWidgets.QLabel("The fitted regularized forces.").addToLayout()

                with QtShortCuts.QHBoxLayout() as layout:
                    self.plotter = QtInteractor(None, auto_update=False)  # , theme=pv.themes.DocumentTheme())
                    self.tab.parent().plotter = self.plotter
                    layout.addWidget(self.plotter.interactor)

                    self.z_slider = QTimeSlider("z", self.z_slider_value_changed, "set z position",
                                                QtCore.Qt.Vertical).addToLayout()
                    self.z_slider.t_slider.valueChanged.connect(
                        lambda value: parent.shared_properties.change_property("z_slider", value, self))
                    parent.shared_properties.add_property("z_slider", self)

                self.vtk_toolbar = VTK_Toolbar(self.plotter, self.update_display, center=True,
                                               shared_properties=self.parent.shared_properties).addToLayout()

                self.t_slider = QTimeSlider(connected=self.update_display).addToLayout()
                self.tab.parent().t_slider = self.t_slider

    def z_slider_value_changed(self):
        self.update_display()

    def checkTabEnabled(self, result: Result) -> bool:
        try:
            if self.result is not None and self.result.solvers is not None:
                relrec = getattr(self.result.solvers[self.t_slider.value()], "relrec", None)
                if relrec is not None:
                    return True
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
                self.vtk_toolbar.channel_select.setValues(np.arange(len(result.stacks[0].channels)),
                                                          result.stacks[0].channels)
                self.vtk_toolbar.channel_select.setVisible(True)
            else:
                self.vtk_toolbar.channel_select.setValue(0)
                self.vtk_toolbar.channel_select.setVisible(False)

    def update_display(self):
        if self.current_tab_selected is False:
            self.current_result_plotted = False
            return

        if self.checkTabEnabled(self.result):
            cam_pos = None
            if self.plotter.camera_position is not None and CamPos.cam_pos_initialized is True:
                cam_pos = self.plotter.camera_position
            CamPos.cam_pos_initialized = True
            M = self.result.solvers[self.t_slider.value()]
            mesh = M.mesh
            self.plotter.interactor.setToolTip(
                str(self.result.solve_parameters) + f"\nNodes {mesh.nodes.shape[0]}\nTets {mesh.tetrahedra.shape[0]}")
            center = None
            center_color = "m"
            if self.vtk_toolbar.use_center.value() == 1:
                center = M.get_center(mode="Force")
                center_color = "m"
            if self.vtk_toolbar.use_center.value() == 2:
                center = M.get_center(mode="Deformation")
                center_color = "c"
            display_image = getVectorFieldImage(self)
            if len(self.result.stacks):
                stack_shape = np.array(self.result.stacks[0].shape[:3]) * np.array(self.result.stacks[0].voxel_size)
            else:
                stack_shape = None

            if M.mesh.regularisation_mask is not None:
                f = -M.mesh.forces * M.mesh.regularisation_mask[:, None]
            else:
                f = -M.mesh.forces

            showVectorField(self.plotter, M.mesh, f, "forces", center=center, center_color=center_color,
                            factor=0.15 * self.vtk_toolbar.arrow_scale.value(),
                            colormap=self.vtk_toolbar.colormap_chooser.value(),
                            colormap2=self.vtk_toolbar.colormap_chooser2.value(),
                            scalebar_max=self.vtk_toolbar.getScaleMax(), show_nan=self.vtk_toolbar.use_nans.value(),
                            display_image=display_image, show_grid=self.vtk_toolbar.show_grid.value(),
                            stack_shape=stack_shape, log_scale=self.vtk_toolbar.use_log.value())
            if cam_pos is not None:
                self.plotter.camera_position = cam_pos
        else:
            self.plotter.interactor.setToolTip("")
