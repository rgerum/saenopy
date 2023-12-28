from qtpy import QtCore, QtWidgets, QtGui


class QSlider(QtWidgets.QSlider):
    min = None
    max = None
    evaluated = 0

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        if self.maximum() - self.minimum():

            if (self.min is not None) and (self.min != "None"):
                self.drawRect(0, self.min, "gray", 0.2)
            if (self.max is not None) and (self.max != "None"):
                value = self.max
                if self.max < 0:
                    value = self.maximum() + self.max
                self.drawRect(value, self.maximum(), "gray", 0.2)
            if (self.evaluated is not None) and (self.min != "None"):
                self.drawRect(self.min if self.min is not None else 0, self.evaluated, "lightgreen", 0.3)
        super().paintEvent(ev)

    def drawRect(self, start, end, color, border):
        p = QtGui.QPainter(self)
        p.setPen(QtGui.QPen(QtGui.QColor("transparent")))
        p.setBrush(QtGui.QBrush(QtGui.QColor(color)))

        if (self.min is not None) and (end != "None") and (start != "None"):
            s = self.width() * (start - self.minimum()) / (self.maximum() - self.minimum() + 1e-5)
            e = self.width() * (end - self.minimum()) / (self.maximum() - self.minimum() + 1e-5)
            p.drawRect(int(s), int(self.height() * border),
                       int(e - s), int(self.height() * (1 - border * 2)))

    def setEvaluated(self, value):
        self.evaluated = value
        self.update()

