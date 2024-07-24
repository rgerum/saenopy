from .PipelineModule import PipelineModule


class DeformationDetector(PipelineModule):

    def __init__(self, parent=None, layout=None):
        super().__init__(parent, layout)
        self.parent = parent
        with self.parent.tabs.createTab("Referenece") as self.tab:
            pass

    def check_evaluated(self, result):
        return True

    def tabChanged(self, tab):
        if self.tab is not None and self.tab.parent() == tab:
            im = self.result.get_image(1)
            self.parent.draw.setImage(im, self.result.shape)
