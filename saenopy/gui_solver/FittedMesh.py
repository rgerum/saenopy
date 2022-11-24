from qtpy import QtCore, QtWidgets
from pyvistaqt import QtInteractor
from saenopy.gui import QtShortCuts
from saenopy.gui.gui_classes import MatplotlibWidget, NavigationToolbar
from saenopy.solver import Result

from .PipelineModule import PipelineModule
from .QTimeSlider import QTimeSlider
from .VTK_Toolbar import VTK_Toolbar
from .showVectorField import showVectorField
from .DeformationDetector import CamPos


class FittedMesh(PipelineModule):
    pipeline_name = "fit forces"
    iteration_finished = QtCore.Signal(object, object)

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with self.parent.tabs.createTab("Fitted Deformations") as self.tab:
            with QtShortCuts.QVBoxLayout() as layout:
                self.label_tab = QtWidgets.QLabel("The fitted mesh deformations.").addToLayout()

                self.plotter = QtInteractor(self)
                self.plotter.set_background("black")
                self.tab.parent().plotter = self.plotter
                layout.addWidget(self.plotter.interactor)

                self.vtk_toolbar = VTK_Toolbar(self.plotter, self.update_display).addToLayout()

                self.t_slider = QTimeSlider(connected=self.update_display).addToLayout()
                self.tab.parent().t_slider = self.t_slider

        self.setParameterMapping(None, {})

    def check_available(self, result: Result):
        return result is not None and result.solver is not None

    def check_evaluated(self, result: Result) -> bool:
        return self.result is not None and self.result.solver is not None and getattr(self.result.solver[0], "regularisation_results", None) is not None

    def update_display(self):
        if self.check_evaluated(self.result):
            cam_pos = None
            if self.plotter.camera_position is not None and CamPos.cam_pos_initialized is True:
                cam_pos = self.plotter.camera_position
            CamPos.cam_pos_initialized = True
            self.plotter.interactor.setToolTip(str(self.result.solve_parameter)+f"\nNodes {self.result.solver[self.t_slider.value()].R.shape[0]}\nTets {self.result.solver[self.t_slider.value()].T.shape[0]}")
            M = self.result.solver[self.t_slider.value()]
            showVectorField(self.plotter, M, M.U, "U", factor=0.1, scalebar_max=self.vtk_toolbar.getScaleMax(), show_nan=self.vtk_toolbar.use_nans.value())
            if cam_pos is not None:
                self.plotter.camera_position = cam_pos
        else:
            self.plotter.interactor.setToolTip("")

