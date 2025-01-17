from qtpy import QtWidgets
from saenopy.gui.common import QtShortCuts
from typing import List, Tuple

from saenopy.gui.common.gui_classes import CheckAbleGroup
from saenopy.gui.common.code_export import get_code, export_as_string

from .PipelineModule import PipelineModule
from .result import Result2D

from saenopy.pyTFM.calculate_stress import calculate_stress

from saenopy.gui.common.TabModule import TabModule

class TabStress(TabModule):

    def __init__(self, parent=None):
        super().__init__(parent)

        with self.parent.tabs.createTab("Line Tension") as self.tab:
            pass

    def check_evaluated(self, result: Result2D) -> bool:
        return result.lt is not None

    def tabChanged(self, tab):
        if self.tab is not None and self.tab.parent() == tab:
            if self.check_evaluated(self.result):
                im = self.result.get_line_tensions()
                self.parent.draw.setImage(im*255)
