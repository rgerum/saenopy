"""Point/line overlays shared by input and computed maps; image coordinates only."""
import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

LINE_KEYS = ('line_x1', 'line_y1', 'line_x2', 'line_y2')
LINE_COLUMNS = ('Reference line x1 (px)', 'Reference line y1 (px)',
                'Reference line x2 (px)', 'Reference line y2 (px)')


def parameter_line(parameters):
    values = [parameters.get(key) for key in LINE_KEYS]
    if any(value is None for value in values):
        return None
    return np.asarray(values, dtype=float).reshape(2, 2)


def calculated_line(result):
    row = (result.results_total or [{}])[0]
    values = [row.get(key) for key in LINE_COLUMNS]
    return None if any(value is None for value in values) else np.asarray(values).reshape(2, 2)


class ReferenceOverlay:
    def __init__(self, origin):
        self.extension = QtWidgets.QGraphicsLineItem(origin)
        self.segment = QtWidgets.QGraphicsLineItem(origin)
        for item, style in ((self.extension, QtCore.Qt.DashLine), (self.segment, QtCore.Qt.SolidLine)):
            pen = QtGui.QPen(QtGui.QColor('yellow'), 2, style)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setZValue(3)
        self.markers = []
        for _ in range(2):
            marker = QtWidgets.QGraphicsEllipseItem(-5, -5, 10, 10, origin)
            marker.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
            marker.setPen(QtGui.QPen(QtGui.QColor('black'), 2))
            marker.setBrush(QtGui.QBrush(QtGui.QColor('yellow')))
            marker.setZValue(4)
            self.markers.append(marker)
        self.hide()

    def hide(self):
        for item in (self.extension, self.segment, *self.markers):
            item.hide()

    def show(self, points, shape, edge=0):
        self.hide()
        if points is None:
            return
        points = np.asarray(points, dtype=float).reshape(-1, 2) - edge + .5
        if not np.isfinite(points).all():
            return
        for point, marker in zip(points, self.markers):
            marker.setPos(*point)
            marker.show()
        if len(points) < 2:
            return
        start, end = points
        delta = end - start
        if np.hypot(*delta) <= 1e-12:
            return
        self.segment.setLine(QtCore.QLineF(*start, *end))
        self.segment.show()
        # Extend the reference to the image border. Segment length/location
        # do not restrict the analyzed region.
        low, high = -np.inf, np.inf
        for coordinate, direction, bound in zip(start, delta, (shape[1], shape[0])):
            if abs(direction) < 1e-12:
                if not 0 <= coordinate <= bound:
                    return
            else:
                limits = sorted((-coordinate / direction, (bound-coordinate) / direction))
                low, high = max(low, limits[0]), min(high, limits[1])
        if low <= high:
            self.extension.setLine(QtCore.QLineF(*(start+low*delta), *(start+high*delta)))
            self.extension.show()
