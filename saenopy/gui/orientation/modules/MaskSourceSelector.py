"""Choose the region used for axis analysis when importing a measurement."""
from qtpy import QtCore, QtWidgets


class MaskSourceSelector(QtWidgets.QGroupBox):
    valueChanged = QtCore.Signal(str)

    def __init__(self, value="whole_image", parent=None):
        super().__init__("Where should orientation be measured?", parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setSpacing(0)
        self.buttons = {}
        self.button_group = QtWidgets.QButtonGroup(self)
        self.button_group.setExclusive(True)
        for index, (name, label) in enumerate([
            ("whole_image", "Whole image\nNo mask"),
            ("same_image", "Mask from the same image"),
            ("separate_image", "Mask from a second image"),
        ]):
            button = QtWidgets.QPushButton(label)
            button.setObjectName(f"maskSource{index}")
            button.setCheckable(True)
            button.setMinimumHeight(65)
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            button.setCursor(QtCore.Qt.PointingHandCursor)
            button.clicked.connect(lambda checked=False, name=name: self.valueChanged.emit(name))
            self.button_group.addButton(button)
            layout.addWidget(button, 1)
            self.buttons[name] = button
        self.setStyleSheet("""
            QPushButton { border: 1px solid palette(mid); padding: 7px; }
            QPushButton#maskSource0 { border-top-left-radius: 7px; border-bottom-left-radius: 7px; }
            QPushButton#maskSource2 { border-top-right-radius: 7px; border-bottom-right-radius: 7px; }
            QPushButton:checked { background-color: palette(highlight); color: palette(highlighted-text); font-weight: 600; }
            QPushButton:focus { border: 2px solid palette(highlight); }
        """)
        self.setValue(value)

    def value(self):
        return next(name for name, button in self.buttons.items() if button.isChecked())

    def setValue(self, value, send_signal=False):
        self.buttons[value].setChecked(True)
        if send_signal:
            self.valueChanged.emit(value)
