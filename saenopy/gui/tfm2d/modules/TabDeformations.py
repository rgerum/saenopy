from .result import Result2D
from saenopy.gui.common.TabModule import TabModule
from saenopy.gui.solver.modules.exporter.ExporterRender2D import render_2d


class TabDeformation(TabModule):
    result: Result2D

    def __init__(self, parent: "BatchEvaluate"):
        super().__init__(parent)

        with self.parent.tabs.createTab("Deformations") as self.tab:
            pass

    def checkTabEnabled(self, result: Result2D) -> bool:
        return result.u is not None

    def tabChanged(self, tab):
        if self.tab is not None and self.tab.parent() == tab:
            if self.checkTabEnabled(self.result):

                if self.result is None:
                    return
                im = render_2d({
                    "stack": {
                        "channel": "cells",
                    },
                    "arrows": "deformation",

                    "colorbar": {
                        "hide": True,
                    },

                    "scalebar": {
                        "hide": True,
                    },
                }, self.result)

                self.parent.draw.setImage(im)
