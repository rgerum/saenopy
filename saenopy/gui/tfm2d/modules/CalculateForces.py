import matplotlib.pyplot as plt
from qtpy import QtWidgets
from typing import Tuple

from saenopy.gui.common import QtShortCuts
from saenopy.gui.common.gui_classes import CheckAbleGroup
from saenopy.gui.common.code_export import get_code

from .PipelineModule import PipelineModule
from .result import Result2D

from saenopy.pyTFM.calculate_forces import calculate_forces
from saenopy.pyTFM.plotting import show_quiver


class Force(PipelineModule):

    def __init__(self, parent=None, layout=None):
        super().__init__(parent, layout)
        self.parent = parent

        with self.parent.tabs.createTab("Forces") as self.tab:
            pass

        with QtShortCuts.QVBoxLayout(self) as layout:
            layout.setContentsMargins(0, 0, 0, 0)
            with CheckAbleGroup(self, "calculate forces").addToLayout() as self.group:
                with QtShortCuts.QVBoxLayout():
                    with QtShortCuts.QHBoxLayout():
                        self.input_young = QtShortCuts.QInputNumber(None, "young", 49000, float=False,
                                                                    value_changed=self.valueChanged, unit="Pa",
                                                                    tooltip="the size of the volume to look for a match")
                        self.input_sigma = QtShortCuts.QInputNumber(None, "poisson ratio", 0.49, step=1, float=True,
                                                                          value_changed=self.valueChanged,
                                                                          tooltip="the overlap of windows")
                        self.input_h = QtShortCuts.QInputNumber(None, "h", 300, step=1, float=True,
                                                                      value_changed=self.valueChanged, unit="µm",
                                                                      tooltip="the overlap of windows")
                    self.label = QtWidgets.QLabel().addToLayout()
                    self.input_button = QtShortCuts.QPushButton(None, "calculate traction forces", self.start_process)

        self.setParameterMapping("force_parameters", {
            "young": self.input_young,
            "sigma": self.input_sigma,
            "h": self.input_h,
        })

    def check_available(self, result):
        return result.u is not None

    def check_evaluated(self, result: Result2D) -> bool:
        return result.tx is not None

    def tabChanged(self, tab):
        if self.tab is not None and self.tab.parent() == tab:
            if self.check_evaluated(self.result):
                im = self.result.get_force_field()
                self.parent.draw.setImage(im*255)

    def process(self, result: Result2D, force_parameters: dict):  # type: ignore
        tx, ty = calculate_forces(result.u, result.v, pixel_size=result.pixel_size, shape=result.shape,
                                  h=force_parameters["h"], young=force_parameters["young"],
                                  sigma=force_parameters["sigma"])
        result.tx = tx
        result.ty = ty
        result.im_force = None
        result.save()
        show_quiver(tx, ty, cbar_str="tractions\n[Pa]")
        plt.savefig("force.png")

    def get_code(self) -> Tuple[str, str]:
        import_code = "from saenopy.pyTFM.calculate_forces import calculate_forces\n"

        results = []
        def code(my_force_parameters):  # pragma: no cover
            # define the parameters for the piv deformation detection
            force_parameters = my_force_parameters

            # iterate over all the results objects
            for result in results:
                # set the parameters
                result.piv_parameters = force_parameters

                tx, ty = calculate_forces(result.u, result.v, pixel_size=result.pixel_size, shape=result.shape,
                                          h=force_parameters["h"], young=force_parameters["young"],
                                          sigma=force_parameters["sigma"])
                result.tx = tx
                result.ty = ty

        data = {
            "my_force_parameters": self.result.force_parameters_tmp
        }

        code = get_code(code, data)

        return import_code, code