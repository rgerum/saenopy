from typing import Tuple

from saenopy.gui.common.code_export import get_code
from saenopy.gui.common.gui_classes import CheckAbleGroup
from saenopy.gui.common import QtShortCuts

from .PipelineModule import PipelineModule
from .result import Result2D

from saenopy.pyTFM.correct_stage_drift import correct_stage_drift


class DeformationDetector2(PipelineModule):

    def __init__(self, parent=None, layout=None):
        super().__init__(parent, layout)
        self.parent = parent
        with self.parent.tabs.createTab("Deformed") as self.tab:
            pass

        with QtShortCuts.QVBoxLayout(self) as layout:
            layout.setContentsMargins(0, 0, 0, 0)
            with CheckAbleGroup(self, "drift").addToLayout() as self.group:
                with QtShortCuts.QVBoxLayout():
                    self.input_button = QtShortCuts.QPushButton(None, "calculate drift correction", self.start_process)

        self.setParameterMapping("drift_parameters", {})

    def check_evaluated(self, result):
        return True

    def tabChanged(self, tab):
        if self.tab is not None and self.tab.parent() == tab:
            im = self.result.get_image(0)
            self.parent.draw.setImage(im, self.result.shape)

    def check_available(self, result):
        return True

    def process(self, result: Result2D, drift_parameters: dict): # type: ignore
        b_save, a_save, [], drift = correct_stage_drift(result.get_image(1, corrected=False), result.get_image(0, corrected=False))

        a_save.save(result.input_corrected)
        b_save.save(result.reference_stack_corrected)

    def get_code(self) -> Tuple[str, str]:
        import_code = "from saenopy.pyTFM.correct_stage_drift import correct_stage_drift\n"

        results = []
        def code():  # pragma: no cover
            # iterate over all the results objects
            for result in results:
                # make the drift correction
                b_save, a_save, [], drift = correct_stage_drift(result.get_image(1, corrected=False),
                                                                result.get_image(0, corrected=False))

                a_save.save(result.input_corrected)
                b_save.save(result.reference_stack_corrected)

        data = {}

        code = get_code(code, data)

        return import_code, code