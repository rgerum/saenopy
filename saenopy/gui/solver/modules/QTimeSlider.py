from qtpy import QtCore, QtWidgets
from saenopy.gui.common import QtShortCuts


class QTimeSlider(QtWidgets.QWidget):
    def __init__(self, name="t", connected=None, tooltip="set time", orientation=QtCore.Qt.Horizontal):
        super().__init__()
        self.tooltip_name = tooltip
        with (QtShortCuts.QHBoxLayout(self) if orientation == QtCore.Qt.Horizontal else QtShortCuts.QVBoxLayout(self)) as layout:
            layout.setContentsMargins(0, 0, 0, 0)
            self.label = QtWidgets.QLabel(name).addToLayout()
            self.label.setAlignment(QtCore.Qt.AlignCenter)
            self.t_slider = QtWidgets.QSlider(orientation).addToLayout()
            self.t_slider.valueChanged.connect(connected)
            self.t_slider.valueChanged.connect(self.value_changed)
            self.t_slider.setToolTip(tooltip)
        self.value = self.t_slider.value
        self.setValue = self.t_slider.setValue
        self.setRange = self.t_slider.setRange

    def value_changed(self):
        self.t_slider.setToolTip(self.tooltip_name+f"\n{self.t_slider.value()+1}/{self.t_slider.maximum()}")