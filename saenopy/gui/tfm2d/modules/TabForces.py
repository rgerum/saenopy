from .result import Result2D
from saenopy.gui.common.TabModule import TabModule


class TabForces(TabModule):

    def __init__(self, parent=None):
        super().__init__(parent)

        with self.parent.tabs.createTab("Forces") as self.tab:
            pass

    def check_evaluated(self, result: Result2D) -> bool:
        return result.tx is not None

    def tabChanged(self, tab):
        if self.tab is not None and self.tab.parent() == tab:
            if self.check_evaluated(self.result):
                im = self.result.get_force_field()
                self.parent.draw.setImage(im*255)
