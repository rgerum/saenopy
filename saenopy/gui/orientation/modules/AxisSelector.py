"""A shared X/Y reference switch for angle and alignment maps."""
from qtpy import QtCore, QtWidgets, QtGui


class SlidingAxisTrack(QtWidgets.QWidget):
    """Animated reference track with accessible, keyboard-operable buttons."""
    def __init__(self, selector):
        super().__init__(selector)
        self.selector = selector
        self._position = 0.
        self._press_position = None
        self._dragging = False
        self.animation = QtCore.QPropertyAnimation(self, b"position", self)
        self.animation.setDuration(140)
        self.animation.setEasingCurve(QtCore.QEasingCurve.InOutCubic)
        self.setMinimumWidth(170)
        self.setMaximumWidth(260)
        self.setFixedHeight(36)

    def get_position(self):
        return self._position

    def set_position(self, value):
        self._position = float(value)
        self.update()

    position = QtCore.Property(float, get_position, set_position)

    def move_to(self, value, animate):
        self.animation.stop()
        target = list(self.selector.buttons).index(value) / (len(self.selector.buttons) - 1)
        if animate:
            self.animation.setStartValue(self._position)
            self.animation.setEndValue(target)
            self.animation.start()
        else:
            self.set_position(target)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        palette = self.palette()
        painter.setPen(QtGui.QPen(palette.color(QtGui.QPalette.Mid), 1))
        painter.setBrush(palette.color(QtGui.QPalette.Midlight))
        painter.drawRoundedRect(QtCore.QRectF(self.rect()).adjusted(.5, .5, -.5, -.5), 18, 18)
        width = (self.width() - 4) / len(self.selector.buttons)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(palette.color(QtGui.QPalette.Highlight))
        painter.drawRoundedRect(QtCore.QRectF(2 + self._position * (self.width()-4-width), 2, width, self.height()-4), 16, 16)

    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
            self._press_position = watched.mapTo(self, event.pos())
            self._dragging = False
        elif event.type() == QtCore.QEvent.MouseMove and self._press_position is not None:
            point = watched.mapTo(self, event.pos())
            if abs(point.x() - self._press_position.x()) >= QtWidgets.QApplication.startDragDistance():
                self._dragging = True
            if self._dragging:
                self.animation.stop()
                width = self.width() / len(self.selector.buttons)
                self.set_position(min(1., max(0., (point.x() - width/2) / (self.width()-width))))
                return True
        elif event.type() == QtCore.QEvent.MouseButtonRelease and self._press_position is not None:
            self._press_position = None
            if self._dragging:
                self._dragging = False
                point = watched.mapTo(self, event.pos())
                watched.setDown(False)
                keys = list(self.selector.buttons)
                index = max(0, min(len(keys)-1, int(point.x() / self.width() * len(keys))))
                self.selector.setValue(keys[index], send_signal=True)
                return True
        return super().eventFilter(watched, event)


class AxisSelector(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(str)

    def __init__(self, value="x", parent=None, *, choices=None, title="Reference axis"):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel(title))
        self.track = SlidingAxisTrack(self)
        self.track_layout = QtWidgets.QHBoxLayout(self.track)
        self.track_layout.setContentsMargins(2, 2, 2, 2)
        self.track_layout.setSpacing(0)
        layout.addWidget(self.track, 1)
        self.buttons = {}
        self.group = QtWidgets.QButtonGroup(self)
        self.group.setExclusive(True)
        self.setStyleSheet("""
            QPushButton { border: 1px solid transparent; border-radius: 14px; padding: 3px 8px; background: transparent; }
            QPushButton:checked { color: palette(highlighted-text); font-weight: 600; }
            QPushButton:focus { border: 1px dotted palette(text); }
        """)
        self.set_choices(choices or [("x", "X axis"), ("y", "Y axis")], value)

    def set_choices(self, choices, value=None):
        for button in self.buttons.values():
            self.group.removeButton(button)
            self.track_layout.removeWidget(button)
            button.hide()
            button.deleteLater()
        self.buttons = {}
        for axis, label in choices:
            button = QtWidgets.QPushButton(label)
            button.setCheckable(True)
            button.setMinimumHeight(30)
            button.setCursor(QtCore.Qt.PointingHandCursor)
            button.clicked.connect(lambda checked=False, axis=axis: self.setValue(axis, send_signal=True))
            button.installEventFilter(self.track)
            button.setAccessibleName(f"Reference {label}")
            self.group.addButton(button)
            self.buttons[axis] = button
            self.track_layout.addWidget(button, 1)
        self.track.setMinimumWidth(85 * len(self.buttons))
        self.track.setMaximumWidth(130 * len(self.buttons))
        self.setValue(value or choices[0][0])

    def value(self):
        return next(axis for axis, button in self.buttons.items() if button.isChecked())

    def setValue(self, value, send_signal=False):
        self.buttons[value].setChecked(True)
        self.track.move_to(value, animate=send_signal)
        if send_signal:
            self.valueChanged.emit(value)
