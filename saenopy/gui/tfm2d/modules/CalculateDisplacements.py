import matplotlib.pyplot as plt
from qtpy import QtWidgets
from saenopy.gui.common import QtShortCuts
from typing import Tuple

from saenopy.gui.common.gui_classes import CheckAbleGroup
from saenopy.gui.common.code_export import get_code
from .PipelineModule import PipelineModule
from .result import Result2D

from saenopy.pyTFM.calculate_deformation import calculate_deformation
from saenopy.pyTFM.plotting import show_quiver


class DeformationDetector3(PipelineModule):

    def __init__(self, parent=None, layout=None):
        super().__init__(parent, layout)
        self.parent = parent
        #layout.addWidget(self)
        with self.parent.tabs.createTab("Deformations") as self.tab:
            pass

        with QtShortCuts.QVBoxLayout(self) as layout:
            layout.setContentsMargins(0, 0, 0, 0)
            with CheckAbleGroup(self, "find deformations (piv)").addToLayout() as self.group:
                with QtShortCuts.QVBoxLayout() as layout:
                    with QtShortCuts.QHBoxLayout():
                        self.input_win = QtShortCuts.QInputNumber(None, "window size", 100, float=False,
                                                                  value_changed=self.valueChanged, unit="px",
                                                                  tooltip="the size of the volume to look for a match")
                        self.input_overlap = QtShortCuts.QInputNumber(None, "overlap", 60, step=1, float=False,
                                                                          value_changed=self.valueChanged, unit="px",
                                                                          tooltip="the overlap of windows")
                        self.input_std = QtShortCuts.QInputNumber(None, "std_factor", 15, step=1, float=True,
                                                                      value_changed=self.valueChanged, unit="px",
                                                                      tooltip="additional filter for extreme values in deformation field")
                    self.label = QtWidgets.QLabel().addToLayout()
                    self.input_button = QtShortCuts.QPushButton(None, "detect deformations", self.start_process)

        self.setParameterMapping("piv_parameters", {
            "window_size": self.input_win,
            "overlap": self.input_overlap,
            "std_factor": self.input_std
        })

    def check_available(self, result):
        return True

    def check_evaluated(self, result: Result2D) -> bool:
        return result.u is not None

    def tabChanged(self, tab):
        if self.tab is not None and self.tab.parent() == tab:
            if self.check_evaluated(self.result):
                im = self.result.get_deformation_field()
                self.parent.draw.setImage(im*255)


    def process(self, result: Result2D, piv_parameters: dict): # type: ignore
        # result.reference_stack, result.input
        u, v, mask_val, mask_std = calculate_deformation(result.get_image(0), result.get_image(1), window_size=piv_parameters["window_size"], overlap=piv_parameters["overlap"], std_factor=piv_parameters["std_factor"])
        result.u = -u
        result.v = v
        result.mask_val = mask_val
        result.mask_std = mask_std
        result.im_displacement = None
        result.save()
        print(u)
        print(v)
        print(mask_std)
        print(mask_val)
        fig1, ax = show_quiver(u, v, cbar_str="deformations\n[pixels]")
        plt.savefig("deformation.png")

    def get_code(self) -> Tuple[str, str]:
        import_code = "from saenopy.pyTFM.calculate_deformation import calculate_deformation\n"

        results = []
        def code(my_piv_params):  # pragma: no cover
            # define the parameters for the piv deformation detection
            piv_parameters = my_piv_params

            # iterate over all the results objects
            for result in results:
                # set the parameters
                result.piv_parameters = piv_parameters

                # calculate the deformation between the two images
                u, v, mask_val, mask_std = calculate_deformation(result.get_image(0), result.get_image(1),
                                                                 window_size=piv_parameters["window_size"],
                                                                 overlap=piv_parameters["overlap"],
                                                                 std_factor=piv_parameters["std_factor"])
                result.u = -u
                result.v = v
                result.mask_val = mask_val
                result.mask_std = mask_std

        data = {
            "my_piv_params": self.result.piv_parameters_tmp
        }

        code = get_code(code, data)

        return import_code, code
