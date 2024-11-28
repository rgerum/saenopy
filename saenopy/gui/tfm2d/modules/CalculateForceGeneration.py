from qtpy import QtWidgets
from typing import List, Tuple

from saenopy.gui.common import QtShortCuts
from saenopy.gui.common.gui_classes import CheckAbleGroup
from saenopy.gui.common.code_export import get_code, export_as_string

from .result import Result2D
from .PipelineModule import PipelineModule

from saenopy.pyTFM.calculate_strain_energy import calculate_strain_energy


class ForceGeneration(PipelineModule):

    def __init__(self, parent=None, layout=None):
        super().__init__(parent, layout)
        self.parent = parent
        #layout.addWidget(self)
        #with self.parent.tabs.createTab("Forces") as self.tab:
        #    pass

        with QtShortCuts.QVBoxLayout(self) as layout:
            layout.setContentsMargins(0, 0, 0, 0)
            with CheckAbleGroup(self, "force generation").addToLayout() as self.group:
                with QtShortCuts.QVBoxLayout():
                    self.label = QtWidgets.QLabel("draw a mask with the red color to select the area where deformations and tractions that are generated by the colony.").addToLayout()
                    self.label.setWordWrap(True)
                    self.input_button = QtShortCuts.QPushButton(None, "calculate force generation", self.start_process)

        self.setParameterMapping("force_gen_parameters", {})

    def check_available(self, result):
        return result.tx is not None

    def check_evaluated(self, result: Result2D) -> bool:
        return result.tx is not None

    def tabChanged(self, tab):
        pass

    def process(self, result: Result2D, force_gen_parameters: dict): # type: ignore

        results_dict = calculate_strain_energy(result.mask, result.pixel_size, result.shape,
                                               result.u, result.v, result.tx, result.ty)

        result.res_dict.update(results_dict)
        result.save()

    def get_code(self) -> Tuple[str, str]:
        import_code = "\n".join([
            "from saenopy.pyTFM.calculate_strain_energy import calculate_strain_energy",
        ])+"\n"

        results: List[Result2D] = []
        @export_as_string
        def code():  # pragma: no cover
            # iterate over all the results objects
            for result in results:
                result.get_mask()
                results_dict = calculate_strain_energy(result.mask, result.pixel_size, result.shape,
                                                       result.u, result.v, result.tx, result.ty)

                result.res_dict.update(results_dict)
                result.save()

        data = {}

        code = get_code(code, data)

        return import_code, code
