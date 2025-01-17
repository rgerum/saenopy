from saenopy.gui.common.TabModule import TabModule

class TabRelaxed(TabModule):

    def __init__(self, parent=None):
        super().__init__(parent)
        with self.parent.tabs.createTab("Referenece") as self.tab:
            pass

    def tabChanged(self, tab):
        if self.tab is not None and self.tab.parent() == tab:
            im = self.result.get_image(1)
            self.parent.draw.setImage(im, self.result.shape)
