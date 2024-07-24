from qtpy import QtWidgets
from saenopy.gui.common import QtShortCuts
from typing import List, Tuple

from saenopy.gui.common.gui_classes import CheckAbleGroup
from saenopy.gui.common.code_export import get_code

from .PipelineModule import PipelineModule
from .result import Result2D

from saenopy.pyTFM.calculate_stress import calculate_stress


class CalculateStress(PipelineModule):

    def __init__(self, parent=None, layout=None):
        super().__init__(parent, layout)
        self.parent = parent
        #layout.addWidget(self)
        with self.parent.tabs.createTab("Line Tension") as self.tab:
            pass

        with QtShortCuts.QVBoxLayout(self) as layout:
            layout.setContentsMargins(0, 0, 0, 0)
            with CheckAbleGroup(self, "colony stress (optional)").addToLayout() as self.group:
                with QtShortCuts.QVBoxLayout():
                    self.label = QtWidgets.QLabel(
                        "draw a mask with the red color to select an area slightly larger then the colony. Draw a mask with the green color to circle every single cell and mark their boundaries.").addToLayout()
                    self.label.setWordWrap(True)
                    self.input_button = QtShortCuts.QPushButton(None, "calculate stress & line tensions", self.start_process)
            self.group.setValue(False)
        self.setParameterMapping("stress_parameters", {})

    def check_available(self, result):
        return result.tx is not None

    def check_evaluated(self, result: Result2D) -> bool:
        return result.lt is not None

    def tabChanged(self, tab):
        if self.tab is not None and self.tab.parent() == tab:
            if self.check_evaluated(self.result):
                im = self.result.get_line_tensions()
                self.parent.draw.setImage(im*255)


    def process(self, result: Result2D, stress_parameters: dict): # type: ignore
        res_dict, attributes = calculate_stress(result.mask, result.pixel_size, result.shape,
                                                result.u, result.tx, result.ty)
        result.res_dict.update(res_dict)
        for attr, value in attributes.items():
            setattr(result, attr, value)
        result.save()

    def get_code(self) -> Tuple[str, str]:
        import_code = "\n".join([
            "from saenopy.pyTFM.calculate_stress import calculate_stress",
        ])+"\n"

        results: List[Result2D] = []
        def code():  # pragma: no cover
            # iterate over all the results objects
            for result in results:
                result.get_mask()
                res_dict, attributes = calculate_stress(result.mask, result.pixel_size, result.shape,
                                                        result.u, result.tx, result.ty)
                result.res_dict.update(res_dict)
                for attr, value in attributes.items():
                    setattr(result, attr, value)
                result.save()

        data = {}

        code = get_code(code, data)

        return import_code, code
