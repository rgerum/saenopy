import matplotlib.pyplot as plt
from qtpy import QtWidgets
from tifffile import imread
import numpy as np
from typing import Tuple

from saenopy.gui.common import QtShortCuts
from saenopy.gui.common.gui_classes import CheckAbleGroup
from saenopy.gui.common.code_export import get_code

from .PipelineModule import PipelineModule
from .result import Result2D

from saenopy.pyTFM.TFM_tractions import TFM_tractions
from saenopy.pyTFM.plotting import show_quiver


class Force(PipelineModule):

    def __init__(self, parent=None, layout=None):
        super().__init__(parent, layout)
        self.parent = parent
        #layout.addWidget(self)
        with self.parent.tabs.createTab("Forces") as self.tab:
            pass

        with QtShortCuts.QVBoxLayout(self) as layout:
            layout.setContentsMargins(0, 0, 0, 0)
            with CheckAbleGroup(self, "calculate forces").addToLayout() as self.group:
                with QtShortCuts.QVBoxLayout() as layout:
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
        return result.u is not None

    def check_evaluated(self, result: Result2D) -> bool:
        return result.tx is not None

    def tabChanged(self, tab):
        if self.tab is not None and self.tab.parent() == tab:
            if self.check_evaluated(self.result):
                im = self.result.get_force_field()
                self.parent.draw.setImage(im*255)


    def process(self, result: Result2D, force_parameters: dict): # type: ignore
        ps1 = result.pixel_size  # pixel size of the image of the beads
        # dimensions of the image of the beads
        im1_shape = result.shape
        print("process", force_parameters)
        ps2 = ps1 * np.mean(np.array(im1_shape) / np.array(result.u.shape))  # pixel size of the deformation field
        tx, ty = TFM_tractions(result.u, result.v, pixelsize1=ps1, pixelsize2=ps2,
                               h=force_parameters["h"], young=force_parameters["young"], sigma=force_parameters["sigma"])

        result.tx = tx
        result.ty = ty
        result.im_force = None
        result.save()
        fig2, ax = show_quiver(tx, ty, cbar_str="tractions\n[Pa]")
        plt.savefig("force.png")

    def get_code(self) -> Tuple[str, str]:
        import_code = "from saenopy.pyTFM.TFM_functions import TFM_tractions\n"

        results = []
        def code(my_force_parameters):  # pragma: no cover
            # define the parameters for the piv deformation detection
            force_parameters = my_force_parameters

            # iterate over all the results objects
            for result in results:
                # set the parameters
                result.piv_parameters = force_parameters

                ps1 = result.pixel_size  # pixel size of the image of the beads
                # dimensions of the image of the beads
                im1_shape = result.shape
                ps2 = ps1 * np.mean(
                    np.array(im1_shape) / np.array(result.u.shape))  # pixel size of the deformation field
                tx, ty = TFM_tractions(result.u, result.v, pixelsize1=ps1, pixelsize2=ps2,
                                       h=force_parameters["h"], young=force_parameters["young"],
                                       sigma=force_parameters["sigma"])

                result.tx = tx
                result.ty = ty

        data = {
            "my_force_parameters": self.result.force_parameters_tmp
        }

        code = get_code(code, data)

        return import_code, code