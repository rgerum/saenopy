from saenopy.gui.common.TabModule import TabModule


class TabDeformed(TabModule):

    def __init__(self, parent=None):
        super().__init__(parent)

        with self.parent.tabs.createTab("Deformed") as self.tab:
            pass

    def tabChanged(self, tab):
        if self.tab is not None and self.tab.parent() == tab:
            im = self.result.get_image(0)
            self.parent.draw.setImage(im, self.result.shape)
