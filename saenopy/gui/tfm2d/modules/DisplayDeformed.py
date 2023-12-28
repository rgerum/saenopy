from .PipelineModule import PipelineModule
from tifffile import imread
from qtpy import QtCore, QtWidgets, QtGui
from saenopy.gui.common.gui_classes import CheckAbleGroup, QProcess, ProcessSimple
from saenopy.gui.common import QtShortCuts, QExtendedGraphicsView
from pyTFM.frame_shift_correction import correct_stage_drift
from .result import Result2D
from pathlib import Path
from saenopy.gui.common.code_export import get_code
from typing import List, Tuple

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
            im = self.result.get_image(1)
            self.parent.draw.setImage(im, self.result.shape)

    def check_available(self, result):
        return True

    def process(self, result: Result2D, drift_parameters: dict): # type: ignore
        b_save, a_save, [], drift = correct_stage_drift(result.get_image(1, corrected=False), result.get_image(0, corrected=False))

        b_save.save(result.input_corrected)
        a_save.save(result.reference_stack_corrected)

    def get_code(self) -> Tuple[str, str]:
        import_code = "from pyTFM.frame_shift_correction import correct_stage_drift\n"

        results = []
        def code():  # pragma: no cover
            # iterate over all the results objects
            for result in results:
                # make the drift correction
                b_save, a_save, [], drift = correct_stage_drift(result.get_image(1, corrected=False),
                                                                result.get_image(0, corrected=False))

                b_save.save(result.input_corrected)
                a_save.save(result.reference_stack_corrected)

        data = {}

        code = get_code(code, data)

        return import_code, code