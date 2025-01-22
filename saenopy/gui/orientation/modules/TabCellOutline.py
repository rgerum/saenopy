import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import jointforces as jf
from pathlib import Path

from saenopy.gui.common.TabModule import TabModule
from saenopy.gui.orientation.modules.result import ResultOrientation, SegmentationParametersDict
from qtpy import QtCore, QtWidgets, QtGui
from qimage2ndarray import array2qimage
from saenopy.gui.common import QtShortCuts, QExtendedGraphicsView

from saenopy.gui.common.gui_classes import QHLine, CheckAbleGroup
from saenopy.gui.common.PipelineModule import PipelineModule
from saenopy.gui.common.QTimeSlider import QTimeSlider
from saenopy.gui.common.resources import resource_icon
from saenopy.gui.common.code_export import get_code
from saenopy.gui.common.ModuleScaleBar import ModuleScaleBar
from saenopy.gui.common.ModuleColorBar import ModuleColorBar


class TabCellOutline(TabModule):
    result: ResultOrientation = None

    def __init__(self, parent: "BatchEvaluate"):
        super().__init__(parent)
      #  self.update_display()
        with self.parent.tabs.createTab("Segmentation Deformations") as self.tab:
            with QtShortCuts.QVBoxLayout() as layout:
                #self.label_tab = QtWidgets.QLabel(
                #    "The deformations from the piv algorithm at every window where the crosscorrelation was evaluated.").addToLayout()

                with QtShortCuts.QHBoxLayout() as layout:
                    #self.plotter = QtInteractor(self, auto_update=False)  # , theme=pv.themes.DocumentTheme())
                    #self.tab.parent().plotter = self.plotter
                    #self.plotter.set_background("black")
                    #layout.addWidget(self.plotter.interactor)
                    self.label = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
                    self.scale1 = ModuleScaleBar(self, self.label)
                    self.color1 = ModuleColorBar(self, self.label)
                    #self.label.setMinimumWidth(300)
                    self.pixmap = QtWidgets.QGraphicsPixmapItem(self.label.origin)
                    self.contour = QtWidgets.QGraphicsPathItem(self.label.origin)
                    pen = QtGui.QPen(QtGui.QColor(255, 0, 0))
                    pen.setCosmetic(True)
                    pen.setWidth(2)
                    self.contour.setPen(pen)
                    self.center = QtWidgets.QGraphicsEllipseItem(self.label.origin)
                    brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
                    self.center.setBrush(brush)

    def checkTabEnabled(self, result: ResultOrientation) -> bool:
            # whether the tab should be enabled for this Result object
            return self.result.orientation_map is not None


    def setResult(self, result: ResultOrientation):
        super().setResult(result)
        self.update_display()

    def update_display(self, *, plotter=None):
        # if self.result is None:
        #     return
        #if self.current_tab_selected is False:
        #    return

        im = self.result.get_image(0)
        self.pixmap.setPixmap(QtGui.QPixmap(array2qimage(im)))
        self.label.setExtend(im.shape[1], im.shape[0])

        if self.result.segmentation is not None:
            from skimage import measure
            # Find contours at a constant value of 0.8
            contours = measure.find_contours(self.result.segmentation["mask"], 0.5)

            path = QtGui.QPainterPath()
            for c in contours:
                path.moveTo(c[0][1],  c[0][0])
                for cc in c:
                    path.lineTo(cc[1],  cc[0])
            self.contour.setPath(path)
            x, y = self.result.segmentation["centroid"]
            self.center.setRect(x-3, y-3, 6, 6)
            self.center.setVisible(True)
        else:
            path = QtGui.QPainterPath()
            self.contour.setPath(path)
            self.center.setVisible(False)
        return None
