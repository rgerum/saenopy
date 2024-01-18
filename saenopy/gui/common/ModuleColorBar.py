import matplotlib.pyplot as plt
import numpy as np
from qtpy import QtCore, QtWidgets, QtGui
from qimage2ndarray import array2qimage


class ModuleColorBar(QtWidgets.QGroupBox):
    min_v = None
    max_v = None
    cmap = None
    tick_count = 3

    def __init__(self, parent, view):
        QtWidgets.QWidget.__init__(self)
        self.parent = parent

        self.font = QtGui.QFont()
        self.font.setPointSize(16)

        self.tick = []
        for i in range(self.tick_count):
            tick_ticks = QtWidgets.QGraphicsTextItem("", view.hud_lowerLeft)
            tick_ticks.setFont(self.font)
            tick_ticks.setDefaultTextColor(QtGui.QColor("white"))

            tick_line = QtWidgets.QGraphicsRectItem(0, 0, 1, 5, tick_ticks)
            tick_line.setBrush(QtGui.QBrush(QtGui.QColor("white")))
            tick_line.setPen(QtGui.QPen(QtGui.QColor("white")))
            self.tick.append([tick_ticks, tick_line])
        self.scalebar = QtWidgets.QGraphicsPixmapItem(view.hud_lowerLeft)
        self.scalebar.setPos(20, -20)

        self.updateStatus()

    def updateStatus(self):
        self.updateBar()

    def setScale(self, min_v, max_v, cmap):
        self.min_v = min_v
        self.max_v = max_v
        self.cmap = cmap
        self.updateBar()

    def updateBar(self):
        if self.min_v is None or self.max_v is None or self.cmap is None:
            return
        bar_width = 200
        ofset_x = 20
        ofset_y = 25
        colors = np.zeros((10, bar_width, 3), dtype=np.uint8)
        for i in range(bar_width):
            c = plt.get_cmap(self.cmap)(int(i/bar_width*255))
            colors[:, i, :] = [c[0]*255, c[1]*255, c[2]*255]
        self.scalebar.setPixmap(QtGui.QPixmap(array2qimage(colors)))
        self.scalebar.setPos(ofset_x, -ofset_y)

        import matplotlib.ticker as ticker

        locator = ticker.MaxNLocator(nbins=self.tick_count-1)
        tick_positions = locator.tick_values(self.min_v, self.max_v)
        tick_positions = np.linspace(self.min_v, self.max_v, self.tick_count)
        for i, pos in enumerate(tick_positions):
            self.tick[i][0].setPos(ofset_x - 50 + (bar_width-1)/(self.tick_count-1)*i, -ofset_y - 33)
            self.tick[i][1].setPos(+ 50, 33 - 5)
            self.tick[i][0].setTextWidth(100)
            self.tick[i][0].setHtml(f"<center>{int(pos):d}</center>")
