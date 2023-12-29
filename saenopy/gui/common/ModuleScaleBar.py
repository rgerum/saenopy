import numpy as np
from qtpy import QtCore, QtWidgets, QtGui


class ModuleScaleBar(QtWidgets.QGroupBox):
    pixtomu = None

    def __init__(self, parent, view):
        QtWidgets.QWidget.__init__(self)
        self.parent = parent

        self.font = QtGui.QFont()
        self.font.setPointSize(16)

        self.scale = 1

        self.scalebar = QtWidgets.QGraphicsRectItem(0, 0, 1, 1, view.hud_lowerRight)
        self.scalebar.setBrush(QtGui.QBrush(QtGui.QColor("white")))
        self.scalebar.setPen(QtGui.QPen(QtGui.QColor("white")))
        self.scalebar.setPos(-20, -20)
        self.scalebar_text = QtWidgets.QGraphicsTextItem("", view.hud_lowerRight)
        self.scalebar_text.setFont(self.font)
        self.scalebar_text.setDefaultTextColor(QtGui.QColor("white"))

        self.time_text = QtWidgets.QGraphicsTextItem("", view.hud_upperRight)
        self.time_text.setFont(self.font)
        self.time_text.setDefaultTextColor(QtGui.QColor("white"))

        view.signal_zoom.connect(self.zoomEvent)

        self.updateStatus()

    def updateStatus(self):
        self.updateBar()

    def zoomEvent(self, scale, pos):
        self.scale = scale
        self.updateBar()

    def setScale(self, voxel_size):
        self.pixtomu = voxel_size[0]
        self.updateBar()

    def updateBar(self):
        if self.scale == 0 or self.pixtomu is None:
            return
        mu = 100*self.pixtomu/self.scale
        values = [1, 5, 10, 25, 50, 75, 100, 150, 200, 250, 500, 1000, 1500, 2000, 2500, 5000, 10000]
        old_v = mu
        for v in values:
            if mu < v:
                mu = old_v
                break
            old_v = v
        if np.abs(self.pixtomu) < 1e-10:
            pixel = 0
        else:
            pixel = mu/(self.pixtomu)*self.scale
        self.scalebar.setRect(0, 0, -pixel, 5)
        self.scalebar_text.setPos(-pixel-20-25, -20-30)
        self.scalebar_text.setTextWidth(pixel+50)
        self.scalebar_text.setHtml(u"<center>%d&thinsp;µm</center>" % mu)
