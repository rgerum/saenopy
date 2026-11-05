"""Per-measurement choice of the reference used for orientation."""
from qtpy import QtCore, QtWidgets


class AnalysisModeSelector(QtWidgets.QGroupBox):
    mode_selected = QtCore.Signal(str)

    def __init__(self, mode="full", parent=None):
        super().__init__("Choose an analysis", parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setSpacing(8)
        self.buttons = {}
        self.button_group = QtWidgets.QButtonGroup(self)
        self.button_group.setExclusive(True)
        for index, (value, title, subtitle) in enumerate([
            ("full", "Relative to Cell", "Orientation as a marker of <b>contractility</b>"),
            ("axis", "Relative to X and Y Axes", "Orientation in the whole image or a mask"),
            ("point", "Relative to a Point or Line", "Orientation towards a point or along a line"),
        ]):
            button = QtWidgets.QPushButton()
            button.setObjectName(f"analysisMode{index}")
            button.setAccessibleName(title)
            button.setAccessibleDescription(subtitle.replace('<b>', '').replace('</b>', ''))
            button.setCheckable(True)
            button.setMinimumHeight(108)
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            button.setCursor(QtCore.Qt.PointingHandCursor)
            content = QtWidgets.QVBoxLayout(button)
            content.setContentsMargins(10, 10, 10, 10)
            heading = QtWidgets.QLabel(title)
            font = heading.font()
            font.setBold(True)
            heading.setFont(font)
            detail = QtWidgets.QLabel(subtitle)
            for label in (heading, detail):
                label.setWordWrap(True)
                label.setAlignment(QtCore.Qt.AlignCenter)
                label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
                content.addWidget(label, 1)
            button.clicked.connect(lambda checked=False, value=value: self.mode_selected.emit(value))
            def colors(checked, labels=(heading, detail)):
                color = 'highlighted-text' if checked else 'button-text'
                for label in labels:
                    label.setStyleSheet(f'color: palette({color}); background: transparent; border: none;')
            button.toggled.connect(colors)
            colors(False)
            self.button_group.addButton(button)
            layout.addWidget(button, 1)
            self.buttons[value] = button
        self.setStyleSheet("""
            QPushButton { border: 1px solid palette(mid); border-radius: 8px; }
            QPushButton:checked { background-color: palette(highlight); }
            QPushButton:focus { border: 2px solid palette(highlight); }
        """)
        self.setValue(mode)

    def value(self):
        return next(mode for mode, button in self.buttons.items() if button.isChecked())

    def setValue(self, mode):
        key = {"full": "full", "point": "point", "point_orientation": "point", "line_orientation": "point"}.get(mode, "axis")
        self.buttons[key].setChecked(True)
