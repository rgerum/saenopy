from saenopy.gui.common.TabModule import TabModule
from .result import Result2D


class TabDeformed(TabModule):
    result: Result2D

    def __init__(self, parent: "BatchEvaluate"):
        super().__init__(parent)

        with self.parent.tabs.createTab("Deformed") as self.tab:
            pass

    def tabChanged(self, tab):
        if self.tab is not None and self.tab.parent() == tab:
            im = self.result.get_image(0)
            self.parent.draw.setImage(im, self.result.shape)
