from saenopy.gui.common.TabModule import TabModule
from saenopy.gui.orientation.modules.result import ResultOrientation
from qtpy import QtCore, QtWidgets, QtGui
from qimage2ndarray import array2qimage
from saenopy.gui.common import QtShortCuts, QExtendedGraphicsView

from saenopy.gui.common.ModuleScaleBar import ModuleScaleBar
from saenopy.gui.common.ModuleColorBar import ModuleColorBar
from saenopy.gui.solver.modules.exporter.ExporterRender2D import render_2d


class TabCoherence(TabModule):
    result: ResultOrientation = None

    def __init__(self, parent: "BatchEvaluate"):
        super().__init__(parent)

        with self.parent.tabs.createTab("Coherence") as self.tab:
            with QtShortCuts.QVBoxLayout() as layout:
                #self.label_tab = QtWidgets.QLabel(
                #    "The deformations from the piv algorithm at every window where the crosscorrelation was evaluated.").addToLayout()

                with QtShortCuts.QHBoxLayout() as layout:
                    self.label = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
                    self.scale1 = ModuleScaleBar(self, self.label)
                    self.color1 = ModuleColorBar(self, self.label)

                    self.pixmap = QtWidgets.QGraphicsPixmapItem(self.label.origin)
                    self.contour = QtWidgets.QGraphicsPathItem(self.label.origin)
                    pen = QtGui.QPen(QtGui.QColor(255, 0, 0))
                    pen.setCosmetic(True)
                    pen.setWidth(2)
                    self.contour.setPen(pen)

    def checkTabEnabled(self, result: ResultOrientation) -> bool:
        # whether the tab should be enabled for this Result object
        return self.result.orientation_map is not None

    def update_display(self, *, plotter=None):
        if self.result is None:
            return
        im = render_2d({
            "image": {
                "scale": 1,
                "scale_overlay": 1,
                "antialiase": 1,
                "logo_size": 0,
            },
            "stack": {
                "image": 1,
                "channel": "cells",
                "colormap": "gray",
                "alpha": 1,
                "channel_B": "fibers",
                "colormap_B": "gray",
                "alpha_B": 1,
                "use_reference_stack": False,
                "contrast_enhance": None,
                "z_proj": False,
                "z": 0,
                "use_contrast_enhance": True,
            },
            "time": {
                "t": 0
            },
            "maps": "coherence",
            "maps_cmap": "turbo",
            "maps_alpha": 0.5,

            "colorbar": {
                "hide": True,
            },

            "scalebar": {
                "hide": True,
            },
            "theme": "dark",
        }, self.result)
        self.pixmap.setPixmap(QtGui.QPixmap(array2qimage(im)))
        self.label.setExtend(im.shape[1], im.shape[0])
        self.scale1.setScale([self.result.pixel_size])
        self.color1.setScale(0, 1, "turbo")
