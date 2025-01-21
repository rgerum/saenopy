from saenopy.gui.common.TabModule import TabModule
from .result import Result2D


class TabStress(TabModule):
    result: Result2D

    def __init__(self, parent: "BatchEvaluate"):
        super().__init__(parent)

        with self.parent.tabs.createTab("Line Tension") as self.tab:
            pass

    def checkTabEnabled(self, result: Result2D) -> bool:
        return result.lt is not None

    def tabChanged(self, tab):
        if self.tab is not None and self.tab.parent() == tab:
            if self.checkTabEnabled(self.result):
                im = self.result.get_line_tensions()
                self.parent.draw.setImage(im*255)
