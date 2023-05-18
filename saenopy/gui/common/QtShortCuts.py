#!/usr/bin/env python
# -*- coding: utf-8 -*-
# QtShortCuts.py

# Copyright (c) 2015-2020, Richard Gerum, Sebastian Richter, Alexander Winterl
#
# This file is part of ClickPoints.
#
# ClickPoints is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ClickPoints is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ClickPoints. If not, see <http://www.gnu.org/licenses/>

import colorsys
import os

import matplotlib as mpl
import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

current_layout = None

def setCurrentLayout(layout):
    global current_layout
    current_layout = layout

def currentLayout():
    return current_layout

def addToLayout(self, layout=None):
    if layout is None and current_layout is not None:
        layout = current_layout
    layout.addWidget(self)
    return self

QtWidgets.QWidget.addToLayout = addToLayout
def connect(self, target):
    super().connect(target)
    return self
QtCore.Signal.connect = connect

class QInput(QtWidgets.QWidget):
    """
    A base class for input widgets with a text label and a unified API.

    - The valueChanged signal is emitted when the user has changed the input.

    - The value of the input element get be set with setValue(value) and queried by value()

    """
    # the signal when the user has changed the value
    valueChanged = QtCore.Signal('PyQt_PyObject')

    no_signal = False

    last_emited_value = None

    def __init__(self, layout=None, name=None, tooltip=None, stretch=False, value_changed=None, settings=None, settings_key=None):
        # initialize the super widget
        super(QInput, self).__init__()

        # initialize the layout of this widget
        QtWidgets.QHBoxLayout(self)
        self.layout().setContentsMargins(0, 0, 0, 0)

        if layout is None and current_layout is not None:
            layout = current_layout

        # add me to a parent layout
        if layout is not None:
            if stretch is True:
                self.wrapper_layout = QtWidgets.QHBoxLayout()
                self.wrapper_layout.setContentsMargins(0, 0, 0, 0)
                self.wrapper_layout.addWidget(self)
                self.wrapper_layout.addStretch()
                layout.addLayout(self.wrapper_layout)
            else:
                layout.addWidget(self)

        # add a label to this layout
        if name is not None and name != "":
            self.label = QtWidgets.QLabel(name)
            self.layout().addWidget(self.label)

        if tooltip is not None and not isinstance(tooltip, list):
            self.setToolTip(tooltip)

        if value_changed is not None:
            self.valueChanged.connect(value_changed)
        self.settings = settings
        self.settings_key = settings_key

    def setLabel(self, text):
        # update the label
        self.label.setText(text)

    def _emitSignal(self):
        if self.value() != self.last_emited_value:
            self.valueChanged.emit(self.value())
            self.last_emited_value = self.value()

    def _valueChangedEvent(self, value):
        if self.no_signal:
            return
        self.setValue(value)
        self._emitSignal()

    def setValue(self, value, send_signal=False):
        self.no_signal = True
        if self.settings is not None:
            self.settings.setValue(self.settings_key, value)
        try:
            self._doSetValue(value)
            if send_signal is True:
                self.valueChanged.emit(value)
        finally:
            self.no_signal = False

    def _doSetValue(self, value):
        # dummy method to be overloaded by child classes
        pass

    def value(self):
        # dummy method to be overloaded by child classes
        pass

class QRangeSlider(QInput):
    editingFinished = QtCore.Signal()

    def __init__(self, layout, name, min, max, **kwargs):
        # initialize the super widget
        QInput.__init__(self, layout, name, **kwargs)

        self.input_min = QtWidgets.QSpinBox()
        self.input_min.valueChanged.connect(self._inputBoxChange)
        self.layout().addWidget(self.input_min)

        from qtrangeslider import QRangeSlider
        self.slider = QRangeSlider(QtCore.Qt.Horizontal).addToLayout()
        self.slider.valueChanged.connect(self._valueChangedEvent)
        self.layout().addWidget(self.slider)

        self.input_max = QtWidgets.QSpinBox()
        self.input_max.valueChanged.connect(self._inputBoxChange)
        self.layout().addWidget(self.input_max)

        self.setRange(min, max)

        self.input_min.editingFinished.connect(self.editingFinished)
        self.input_max.editingFinished.connect(self.editingFinished)
        self.slider.sliderReleased.connect(self.editingFinished)

    def _inputBoxChange(self):
        self.setValue((self.input_min.value(), self.input_max.value()))
        self._emitSignal()

    def _valueChangedEvent(self, value):
        if self.no_signal:
            return
        self.setValue(value)
        self._emitSignal()

    def _doSetValue(self, value):
        self.slider.setValue(value)
        self.input_min.setValue(value[0])
        self.input_max.setValue(value[1])

    def value(self):
        return self.slider.value()

    def setRange(self, min, max):
        self.input_min.setRange(min, max)
        self.input_max.setRange(min, max)
        self.slider.setRange(min, max)
        self._range = (min, max)

    def range(self):
        return self._range


cast_float = float
class QInputNumber(QInput):
    slider_dragged = False

    def __init__(self, layout=None, name=None, value=0, min=None, max=None, use_slider=False, float=True, decimals=2,
                 unit=None, step=None, name_post=None, log_slider=False, **kwargs):
        # initialize the super widget
        QInput.__init__(self, layout, name, **kwargs)

        if self.settings is not None:
            value = cast_float(self.settings.value(self.settings_key, value))

        if float is False:
            self.decimals = 0
        else:
            if decimals is None:
                decimals = 2
            self.decimals = decimals
        self.decimal_factor = 10**self.decimals

        if use_slider and min is not None and max is not None:
            # slider
            self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.layout().addWidget(self.slider)
            self.log_slider = log_slider
            if log_slider:
                self.slider.setRange(int(np.log10(min) * self.decimal_factor), int(np.log10(max) * self.decimal_factor))
                self.slider.valueChanged.connect(lambda x: self._valueChangedEvent(10**(x / self.decimal_factor)))
            else:
                self.slider.setRange(int(min * self.decimal_factor), int(max * self.decimal_factor))
                self.slider.valueChanged.connect(lambda x: self._valueChangedEvent(x / self.decimal_factor))
            self.slider.sliderPressed.connect(lambda: self._setSliderDragged(True))
            self.slider.sliderReleased.connect(lambda: self._setSliderDragged(False))
        else:
            self.slider = None

        # add spin box
        self.use_float = float
        if float:
            self.spin_box = QtWidgets.QDoubleSpinBox()
            self.spin_box.setDecimals(decimals)
        else:
            self.spin_box = QtWidgets.QSpinBox()
        if unit is not None:
            self.spin_box.setSuffix(" " + unit)
        self.layout().addWidget(self.spin_box)
        self.spin_box.valueChanged.connect(self._valueChangedEvent)

        if min is not None:
            self.spin_box.setMinimum(min)
        else:
            self.spin_box.setMinimum(-99999)
        if max is not None:
            self.spin_box.setMaximum(max)
        else:
            self.spin_box.setMaximum(+99999)
        if step is not None:
            self.spin_box.setSingleStep(step)

        if name_post is not None:
            self.label2 = QtWidgets.QLabel(name_post)
            self.layout().addWidget(self.label2)

        self.setValue(value)

    def setRange(self, min, max):
        self.spin_box.setMinimum(min)
        self.spin_box.setMaximum(max)
        if self.slider:
            if self.log_slider:
                self.slider.setRange(int(np.log10(min) * self.decimal_factor), int(np.log10(max) * self.decimal_factor))
                self.slider.valueChanged.connect(lambda x: self._valueChangedEvent(10 ** (x / self.decimal_factor)))
            else:
                self.slider.setRange(int(min * self.decimal_factor), int(max * self.decimal_factor))

    def _setSliderDragged(self, value):
        self.slider_dragged = value
        if value is False:
            self._emitSignal()

    def _valueChangedEvent(self, value):
        if self.no_signal:
            return
        self.setValue(value)
        if not self.slider_dragged:
            self._emitSignal()

    def _doSetValue(self, value):
        if self.use_float:
            self.spin_box.setValue(value)
        else:
            self.spin_box.setValue(int(value))
        if self.slider is not None:
            if self.log_slider:
                self.slider.setValue(int(np.log10(value) * self.decimal_factor))
            else:
                self.slider.setValue(int(value * self.decimal_factor))

    def value(self):
        return self.spin_box.value()


class QInputString(QInput):
    error = None

    def __init__(self, layout=None, name=None, value="", allow_none=True, none_value="None", type=str, unit=None, name_post=None, validator=None, **kwargs):
        # initialize the super widget
        QInput.__init__(self, layout, name, **kwargs)
        self.none_value = none_value

        if self.settings is not None:
            value = self.settings.value(self.settings_key, value)

        self.line_edit = QtWidgets.QLineEdit()
        self.layout().addWidget(self.line_edit)
        self.line_edit.editingFinished.connect(self.editingFinishedCall)
        if type is int or type is float or type == "exp":
            self.line_edit.setAlignment(QtCore.Qt.AlignRight)
        if unit is not None:
            self.label_unit = QtWidgets.QLabel(unit)
            self.layout().addWidget(self.label_unit)

        self.allow_none = allow_none
        self.type = type
        self.validator = validator

        if name_post is not None:
            self.label2 = QtWidgets.QLabel(name_post)
            self.layout().addWidget(self.label2)

        self.setValue(value)
        self.emitValueChanged()

        self.line_edit.textChanged.connect(self.emitValueChanged)

    def editingFinishedCall(self):
        try:
            self._valueChangedEvent(self.value())
        except ValueError:
            return

    def emitValueChanged(self):
        """ connected to the textChanged signal """
        try:
            value = self.value()
            if self.validator is not None:
                if self.validator(value) is False:
                    raise ValueError
            self.line_edit.setStyleSheet("")
        except ValueError as err:
            self.line_edit.setStyleSheet("background: #d56060")

    def _doSetValue(self, value):
        if self.type == "exp":
            self.line_edit.setText(f"10**{np.format_float_positional(np.log10(float(value)), precision=2, unique=True, fractional=True, trim='-')}")
        else:
            self.line_edit.setText(str(value))

    def value(self):
        text = self.line_edit.text()
        if self.allow_none is True and (text == "None" or text == ""):
            return self.none_value
        if self.type == int:
            return int(text)
        if self.type == float:
            return float(text)
        if self.type == "exp":
            if text.startswith("1e"):
                return 10**float(text[2:])
            if text.startswith("10**"):
                return 10**float(text[4:])
            return float(text)
        return text


class QInputBool(QInput):
    button = None
    my_value = None
    icon = None
    buttons = None

    def __init__(self, layout=None, name=None, value=False, icon=None, group=False, tooltip=None, **kwargs):
        # initialize the super widget
        QInput.__init__(self, layout, name, tooltip=tooltip, **kwargs)
        self.tooltip = tooltip

        if self.settings is not None:
            value = self.settings.value(self.settings_key, value) == "true"

        if group is True and isinstance(icon, list):
            self.button_group = QtWidgets.QButtonGroup()
            self.icon = icon
            self.buttons = []
            for button_icon in icon:
                button = QtWidgets.QPushButton()
                if isinstance(button_icon, str):
                    button.setText(button_icon)
                else:
                    button.setIcon(button_icon)
                button.setCheckable(True)
                if tooltip and isinstance(tooltip, list):
                    button.setToolTip(tooltip[len(self.buttons)])
                self.layout().addWidget(button)
                #self.button_group.addButton(button)
                self.buttons.append(button)
                button.clicked.connect(lambda _, index=len(self.buttons): self.button_group_clicked(index-1))
        elif icon is not None:
            self.icon = icon
            self.button = QtWidgets.QPushButton()
            if isinstance(icon, list):
                if isinstance(icon[0], str):
                    self.button.setText(icon[0])
                else:
                    self.button.setIcon(icon[0])
            else:
                self.button.setIcon(icon)
            self.button.setCheckable(True)
            self.layout().addWidget(self.button)
            if isinstance(icon, list) and len(icon) > 2:
                self.button.clicked.connect(self.button_clicked)
            else:
                self.button.clicked.connect(lambda: self._valueChangedEvent(self.value()))
        else:
            self.checkbox = QtWidgets.QCheckBox()
            self.layout().addWidget(self.checkbox)
            self.checkbox.stateChanged.connect(lambda: self._valueChangedEvent(self.value()))

        self.setValue(value)

    def button_group_clicked(self, index):
        if self.no_signal:
            return
        self.my_value = index
        #self._doSetValue(self.my_value)
        self._valueChangedEvent(self.my_value)

    def button_clicked(self):
        self.my_value = (self.my_value + 1) % len(self.icon)
        self._doSetValue(self.my_value)
        self._valueChangedEvent(self.my_value)

    def _doSetValue(self, value):
        self.my_value = value
        if self.button is not None:
            self.button.setChecked(bool(value))
            if isinstance(self.icon, list):
                if isinstance(self.icon[value], str):
                    self.button.setText(self.icon[value])
                else:
                    self.button.setIcon(self.icon[value])
                if isinstance(self.tooltip, list):
                    self.button.setToolTip(self.tooltip[value])
        elif self.buttons is not None:
            self.no_signal = True
            for button in self.buttons:
                button.setChecked(False)
            self.buttons[value].setChecked(True)
            self.no_signal = False
        else:
            self.checkbox.setChecked(bool(value))

    def value(self):
        if isinstance(self.icon, list) and (len(self.icon) > 2 or self.buttons):
            return self.my_value
        if self.button is not None:
            return self.button.isChecked()
        return self.checkbox.isChecked()


class QInputChoice(QInput):

    def __init__(self, layout=None, name=None, value=None, values=None, value_names=None, reference_by_index=False, **kwargs):
        # initialize the super widget
        QInput.__init__(self, layout, name, **kwargs)

        if self.settings is not None:
            value = self.settings.value(self.settings_key, value)

        self.reference_by_index = reference_by_index
        self.values = values
        self.value_names = value_names if value_names is not None else values

        self.combobox = QtWidgets.QComboBox()
        self.layout().addWidget(self.combobox)

        if self.value_names is not None:
            self.combobox.addItems(self.value_names)

        self.combobox.currentIndexChanged.connect(lambda: self._valueChangedEvent(self.value()))

        if value is not None:
            self.setValue(value)

    def setValues(self, new_values, value_names=None):
        self.no_signal = True
        try:
            self.value_names = list(value_names) if value_names is not None else [str(v) for v in new_values]

            if self.values is not None:
                for i in range(len(self.values)):
                    self.combobox.removeItem(0)

            self.values = list(new_values)

            self.combobox.addItems(self.value_names)
        finally:
            self.no_signal = False

    def _doSetValue(self, value):
        if self.reference_by_index is True:
            self.combobox.setCurrentIndex(value)
        else:
            try:
                self.combobox.setCurrentIndex(self.values.index(value))
            except ValueError:
                try:
                    self.combobox.setCurrentIndex(self.value_names.index(value))
                except ValueError:
                    return

    def value(self):
        if self.reference_by_index is True:
            return self.combobox.currentIndex()
        else:
            if self.values is None or len(self.values) == 0:
                return None
            return self.values[self.combobox.currentIndex()]

    def valueName(self):
        if self.reference_by_index is True:
            return self.combobox.currentIndex()
        else:
            if self.values is None:
                return None
            return self.value_names[self.combobox.currentIndex()]


class QInputColor(QInput):

    def __init__(self, layout=None, name=None, value=None, **kwargs):
        # initialize the super widget
        QInput.__init__(self, layout, name, **kwargs)

        if self.settings is not None:
            value = self.settings.value(self.settings_key, value)

        self.button = QtWidgets.QPushButton()
        self.button.setMaximumWidth(40)
        self.layout().addWidget(self.button)
        self.button.clicked.connect(self._openDialog)

        # set the color
        self.setValue(value)

    def changeEvent(self, event):
        if event.type() == QtCore.QEvent.EnabledChange:
            if not self.isEnabled():
                self.button.setStyleSheet("background-color: #f0f0f0;")
            else:
                self.setValue(self.color)

    def _openDialog(self):
        # get new color from color picker
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(*tuple(mpl.colors.to_rgba_array(self.value())[0] * 255)),
                                                self.parent(), self.label.text() + " choose color")
        # if a color is set, apply it
        if color.isValid():
            color = mpl.colors.to_hex(color.getRgbF())
            self.setValue(color)
            self._emitSignal()

    def _doSetValue(self, value):
        # display and save the new color
        if value is None:
            value = "#FF0000FF"
        self.button.setStyleSheet("background-color: %s;" % value)
        self.color = value

    def value(self):
        # return the color
        return self.color


class QInputFilename(QInput):
    last_folder = None

    def __init__(self, layout=None, name=None, value=None, dialog_title="Choose File", file_type="All", filename_checker=None, existing=False, allow_edit=False, **kwargs):
        # initialize the super widget
        QInput.__init__(self, layout, name, **kwargs)

        if self.settings is not None:
            value = self.settings.value(self.settings_key, value)

        self.dialog_title = dialog_title
        self.file_type = file_type
        self.filename_checker = filename_checker
        self.existing = existing

        self.line = QtWidgets.QLineEdit()
        self.layout().addWidget(self.line)
        if allow_edit is False:
            self.line.setEnabled(False)
        else:
            self.line.editingFinished.connect(lambda: self.setValue(self.line.text()))

        self.button = QtWidgets.QPushButton("choose file")
        self.layout().addWidget(self.button)
        self.button.clicked.connect(self._openDialog)

        # set the color
        self.setValue(value)
        if value is None:
            self.last_folder = os.getcwd()

    def _openDialog(self):
        # open an new files
        if not self.existing:
            if "PYCHARM_HOSTED" in os.environ:
                filename = QtWidgets.QFileDialog.getSaveFileName(None, self.dialog_title, self.last_folder, self.file_type, options=QtWidgets.QFileDialog.DontUseNativeDialog)
            else:
                filename = QtWidgets.QFileDialog.getSaveFileName(None, self.dialog_title, self.last_folder, self.file_type)
        # or choose an existing file
        else:
            if "PYCHARM_HOSTED" in os.environ:
                filename = QtWidgets.QFileDialog.getOpenFileName(None, self.dialog_title, self.last_folder, self.file_type, options=QtWidgets.QFileDialog.DontUseNativeDialog)
            else:
                filename = QtWidgets.QFileDialog.getOpenFileName(None, self.dialog_title, self.last_folder, self.file_type)

        # get the string
        if isinstance(filename, tuple):  # Qt5
            filename = filename[0]
        else:  # Qt4
            filename = str(filename)

        # optical check the filename
        if self.filename_checker and filename:
            filename = self.filename_checker(filename)

        # set the filename
        if filename:
            self.setValue(filename)
            self._emitSignal()

    def _doSetValue(self, value):
        if value is None:
            return
        self.last_folder = os.path.dirname(value)
        self.line.setText(str(value))

    def value(self):
        # return the color
        return self.line.text()


class QInputFolder(QInput):
    last_folder = None

    def __init__(self, layout=None, name=None, value=None, dialog_title="Choose Folder", filename_checker=None, allow_edit=False, **kwargs):
        # initialize the super widget
        QInput.__init__(self, layout, name, **kwargs)

        if self.settings is not None:
            value = self.settings.value(self.settings_key, value)

        self.dialog_title = dialog_title
        self.filename_checker = filename_checker

        self.line = QtWidgets.QLineEdit()
        self.layout().addWidget(self.line)
        if allow_edit is False:
            self.line.setEnabled(False)
        else:
            self.line.editingFinished.connect(lambda: self.setValue(self.line.text()))

        self.button = QtWidgets.QPushButton("choose folder")
        self.layout().addWidget(self.button)
        self.button.clicked.connect(self._openDialog)

        # set the color
        self.setValue(str(value))
        if value is None:
            self.last_folder = os.getcwd()

    def _openDialog(self):
        # choose an existing file
        filename = QtWidgets.QFileDialog.getExistingDirectory(None, self.dialog_title, self.last_folder)

        # get the string
        if isinstance(filename, tuple):  # Qt5
            filename = filename[0]
        else:  # Qt4
            filename = str(filename)

        # optical check the filename
        if self.filename_checker and filename:
            filename = self.filename_checker(filename)

        # set the filename
        if filename:
            self.setValue(filename)
            self._emitSignal()

    def _doSetValue(self, value):
        self.last_folder = value
        self.line.setText(value)

    def value(self):
        # return the color
        return self.line.text()

class QPushButton(QtWidgets.QPushButton):
    def __init__(self, layout, name, connect=None, icon=None, tooltip=None):
        super().__init__(name)
        if layout is None and current_layout is not None:
            layout = current_layout
        layout.addWidget(self)
        if connect is not None:
            self.clicked.connect(connect)
        if icon is not None:
            self.setIcon(icon)
        if tooltip is not None:
            self.setToolTip(tooltip)

class QGroupBox(QtWidgets.QGroupBox):
    def __init__(self, layout, name):
        super().__init__(name)
        if layout is None and current_layout is not None:
            layout = current_layout
        layout.addWidget(self)
        self.layout = QVBoxLayout(self)

    def __enter__(self):
        return self, self.layout.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.layout.__exit__(exc_type, exc_val, exc_tb)

class QTabWidget(QtWidgets.QTabWidget):

    def __init__(self, layout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if layout is None and current_layout is not None:
            layout = current_layout
        layout.addWidget(self)

    def createTab(self, name):
        tab_stack = QtWidgets.QWidget()
        self.addTab(tab_stack, name)
        v_layout = QVBoxLayout(tab_stack)
        return v_layout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class EnterableLayout:
    def __enter__(self):
        global current_layout
        self.old_layout = current_layout
        current_layout = self.layout
        return self.layout

    def __exit__(self, exc_type, exc_val, exc_tb):
        global current_layout
        current_layout = self.old_layout


class QVBoxLayout(QtWidgets.QVBoxLayout, EnterableLayout):
    def __init__(self, parent=None, no_margins=False):
        if parent is None and current_layout is not None:
            parent = current_layout
        if getattr(parent, "addLayout", None) is None:
            super().__init__(parent)
        else:
            super().__init__()
            parent.addLayout(self)
        self.layout = self
        if no_margins is True:
            self.setContentsMargins(0, 0, 0, 0)


class QHBoxLayout(QtWidgets.QHBoxLayout, EnterableLayout):
    def __init__(self, parent=None, no_margins=False):
        if parent is None and current_layout is not None:
            parent = current_layout
        if getattr(parent, "addLayout", None) is None:#isinstance(parent, QtWidgets.QWidget):
            super().__init__(parent)
        else:
            super().__init__()
            parent.addLayout(self)
        self.layout = self
        if no_margins is True:
            self.setContentsMargins(0, 0, 0, 0)


def QVLine(layout=None):
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.VLine)
    line.setFrameShadow(QtWidgets.QFrame.Sunken)

    if current_layout is not None:
        current_layout.addWidget(line)
    else:
        layout.addWidget(line)
    return line


def QHLine(layout=None):
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.HLine)
    line.setFrameShadow(QtWidgets.QFrame.Sunken)

    if current_layout is not None:
        current_layout.addWidget(line)
    else:
        layout.addWidget(line)
    return line


class QSplitter(QtWidgets.QSplitter, EnterableLayout):
    def __init__(self, *args):
        super().__init__(*args)
        if current_layout is not None:
            current_layout.addWidget(self)
        self.layout = self
        self.widgets = []

    def addLayout(self, layout):
        widget = QtWidgets.QWidget()
        self.widgets.append(widget)
        self.addWidget(widget)
        widget.setLayout(layout)
        return layout


def AddQSpinBox(layout, text, value=0, float=True, strech=False):
    horizontal_layout = QtWidgets.QHBoxLayout()
    layout.addLayout(horizontal_layout)
    text = QtWidgets.QLabel(text)
    if float:
        spinBox = QtWidgets.QDoubleSpinBox()
    else:
        spinBox = QtWidgets.QSpinBox()
    spinBox.label = text
    spinBox.setRange(-99999, 99999)
    spinBox.setValue(value)
    spinBox.setHidden_ = spinBox.setHidden

    def setHidden(hidden):
        spinBox.setHidden_(hidden)
        text.setHidden(hidden)

    spinBox.setHidden = setHidden
    horizontal_layout.addWidget(text)
    horizontal_layout.addWidget(spinBox)
    spinBox.managingLayout = horizontal_layout
    if strech:
        horizontal_layout.addStretch()
    return spinBox


def AddQLineEdit(layout, text, value=None, strech=False, editwidth=None):
    horizontal_layout = QtWidgets.QHBoxLayout()
    layout.addLayout(horizontal_layout)
    text = QtWidgets.QLabel(text)
    lineEdit = QtWidgets.QLineEdit()
    if editwidth:
        lineEdit.setFixedWidth(editwidth)
    if value:
        lineEdit.setText(value)
    lineEdit.label = text
    horizontal_layout.addWidget(text)
    horizontal_layout.addWidget(lineEdit)
    lineEdit.managingLayout = horizontal_layout
    if strech:
        horizontal_layout.addStretch()
    return lineEdit


def AddQSaveFileChoose(layout, text, value=None, dialog_title="Choose File", file_type="All", filename_checker=None,
                       strech=False):
    horizontal_layout = QtWidgets.QHBoxLayout()
    layout.addLayout(horizontal_layout)
    text = QtWidgets.QLabel(text)
    lineEdit = QtWidgets.QLineEdit()
    if value:
        lineEdit.setText(value)
    lineEdit.label = text
    lineEdit.setEnabled(False)

    def OpenDialog():
        srcpath = QtWidgets.QFileDialog.getSaveFileName(None, dialog_title, os.getcwd(), file_type)
        if isinstance(srcpath, tuple):
            srcpath = srcpath[0]
        else:
            srcpath = str(srcpath)
        if filename_checker and srcpath:
            srcpath = filename_checker(srcpath)
        if srcpath:
            lineEdit.setText(srcpath)

    button = QtWidgets.QPushButton("Choose File")
    button.pressed.connect(OpenDialog)
    horizontal_layout.addWidget(text)
    horizontal_layout.addWidget(lineEdit)
    horizontal_layout.addWidget(button)
    lineEdit.managingLayout = horizontal_layout
    if strech:
        horizontal_layout.addStretch()
    return lineEdit


def AddQOpenFileChoose(layout, text, value=None, dialog_title="Choose File", file_type="All", filename_checker=None,
                       strech=False):
    horizontal_layout = QtWidgets.QHBoxLayout()
    layout.addLayout(horizontal_layout)
    text = QtWidgets.QLabel(text)
    lineEdit = QtWidgets.QLineEdit()
    if value:
        lineEdit.setText(value)
    lineEdit.label = text
    lineEdit.setEnabled(False)

    def OpenDialog():
        srcpath = QtWidgets.QFileDialog.getOpenFileName(None, dialog_title, os.getcwd(), file_type)
        if isinstance(srcpath, tuple):
            srcpath = srcpath[0]
        else:
            srcpath = str(srcpath)
        if filename_checker and srcpath:
            srcpath = filename_checker(srcpath)
        if srcpath:
            lineEdit.setText(srcpath)

    button = QtWidgets.QPushButton("Choose File")
    button.pressed.connect(OpenDialog)
    horizontal_layout.addWidget(text)
    horizontal_layout.addWidget(lineEdit)
    horizontal_layout.addWidget(button)
    lineEdit.managingLayout = horizontal_layout
    if strech:
        horizontal_layout.addStretch()
    return lineEdit


def AddQColorChoose(layout, text, value=None, strech=False):
    # add a layout
    horizontal_layout = QtWidgets.QHBoxLayout()
    layout.addLayout(horizontal_layout)
    # add a text
    text = QtWidgets.QLabel(text)
    button = QtWidgets.QPushButton("")
    button.label = text

    def OpenDialog():
        # get new color from color picker
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(*HTMLColorToRGB(button.getColor())))
        # if a color is set, apply it
        if color.isValid():
            color = "#%02x%02x%02x" % color.getRgb()[:3]
            button.setColor(color)

    def setColor(value):
        # display and save the new color
        button.setStyleSheet("background-color: %s;" % value)
        button.color = value

    def getColor():
        # return the color
        return button.color

    # default value for the color
    if value is None:
        value = "#FF0000"
    # add functions to button
    button.pressed.connect(OpenDialog)
    button.setColor = setColor
    button.getColor = getColor
    # set the color
    button.setColor(value)
    # add widgets to the layout
    horizontal_layout.addWidget(text)
    horizontal_layout.addWidget(button)
    # add a strech if requested
    if strech:
        horizontal_layout.addStretch()
    return button


def AddQComboBox(layout, text, values=None, selectedValue=None):
    horizontal_layout = QtWidgets.QHBoxLayout()
    layout.addLayout(horizontal_layout)
    text = QtWidgets.QLabel(text)
    comboBox = QtWidgets.QComboBox()
    comboBox.label = text
    for value in values:
        comboBox.addItem(value)
    if selectedValue:
        comboBox.setCurrentIndex(values.index(selectedValue))
    comboBox.values = values

    def setValues(new_values):
        for i in range(len(comboBox.values)):
            comboBox.removeItem(0)
        for value in new_values:
            comboBox.addItem(value)
        comboBox.values = new_values

    comboBox.setValues = setValues
    horizontal_layout.addWidget(text)
    horizontal_layout.addWidget(comboBox)
    comboBox.managingLayout = horizontal_layout
    return comboBox


def AddQCheckBox(layout, text, checked=False, strech=False):
    horizontal_layout = QtWidgets.QHBoxLayout()
    layout.addLayout(horizontal_layout)
    text = QtWidgets.QLabel(text)
    checkBox = QtWidgets.QCheckBox()
    checkBox.label = text
    checkBox.setChecked(checked)
    horizontal_layout.addWidget(text)
    horizontal_layout.addWidget(checkBox)
    if strech:
        horizontal_layout.addStretch()
    return checkBox


def AddQLabel(layout, text=None):
    text = QtWidgets.QLabel(text)
    if text:
        layout.addWidget(text)
    return text


def AddQHLine(layout):
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.HLine)
    line.setFrameShadow(QtWidgets.QFrame.Sunken)
    layout.addWidget(line)
    return line


def GetColorByIndex(index):
    colors = np.linspace(0, 1, 16, endpoint=False).tolist() * 3  # 16 different hues
    saturations = [1] * 16 + [0.5] * 16 + [1] * 16  # in two different saturations
    value = [1] * 16 + [1] * 16 + [0.5] * 16  # and two different values
    return "#%02x%02x%02x" % tuple((np.array(
        colorsys.hsv_to_rgb((np.array(colors[index]) * 3) % 1, saturations[index], value[index])) * 255).astype(int))


# set the standard colors for the color picker dialog
colors = np.linspace(0, 1, 16, endpoint=False).tolist() * 3  # 16 different hues
saturations = [1] * 16 + [0.5] * 16 + [1] * 16  # in two different saturations
value = [1] * 16 + [1] * 16 + [0.5] * 16  # and two different values
for index, (color, sat, val) in enumerate(zip(colors, saturations, value)):
    # deform the index, as the dialog fills them column wise and we want to fill them row wise
    index = index % 8 * 6 + index // 8
    # convert color from hsv to rgb, to an array, to an tuple, to a hex string then to an integer
    color_integer = int("%02x%02x%02x" % tuple((np.array(colorsys.hsv_to_rgb(color, sat, val)) * 255).astype(int)), 16)
    try:
        QtWidgets.QColorDialog.setStandardColor(index, QtGui.QColor(color_integer))  # for Qt5
    except TypeError:
        QtWidgets.QColorDialog.setStandardColor(index, color_integer)  # for Qt4


import matplotlib.pyplot as plt
class ColorMapChoose(QtWidgets.QDialog):
    """ A dialog to select a colormap """
    result = ""

    def __init__(self, parent: QtWidgets.QWidget, map):
        """ initialize the dialog with all the colormap of matplotlib """
        QtWidgets.QDialog.__init__(self, parent)
        main_layout = QtWidgets.QVBoxLayout(self)
        self.layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(self.layout)
        button_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(button_layout)
        self.button_cancel = QtWidgets.QPushButton("Cancel")
        self.button_cancel.clicked.connect(lambda _: self.done(0))
        button_layout.addStretch()
        button_layout.addWidget(self.button_cancel)

        self.maps = plt.colormaps()
        self.buttons = []
        self.setWindowTitle("Select colormap")

        # Have colormaps separated into categories:
        # http://matplotlib.org/examples/color/colormaps_reference.html
        cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma']),
                 ('Sequential', [
                     'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                     'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                     'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
                 ('Simple Colors', [
                     'gray', 'red', 'orange', 'yellow', 'lime', 'green', 'mint', 'cyan', 'navy', 'blue', 'purple', 'magenta', 'grape']),
                 ('Sequential (2)', [
                     'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                     'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
                     'hot', 'afmhot', 'gist_heat', 'copper']),
                 ('Diverging', [
                     'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                     'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
                 ('Qualitative', [
                     'Pastel1', 'Pastel2', 'Paired', 'Accent',
                     'Dark2', 'Set1', 'Set2', 'Set3',
                     'tab10', 'tab20', 'tab20b', 'tab20c']),
                 ('Miscellaneous', [
                     'turbo', 'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
                     'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
                     'gist_rainbow', 'rainbow', 'nipy_spectral', 'gist_ncar'])]

        for cmap_category, cmap_list in cmaps:
            layout = QtWidgets.QVBoxLayout()
            label = QtWidgets.QLabel(cmap_category)
            layout.addWidget(label)
            label.setFixedWidth(150)
            for cmap in cmap_list:
                button = QtWidgets.QPushButton(cmap)
                button.setStyleSheet("text-align: center; border: 2px solid black; "+self.getBackground(cmap))
                button.clicked.connect(lambda _, cmap=cmap: self.buttonClicked(cmap))
                self.buttons.append(button)
                layout.addWidget(button)
            layout.addStretch()
            self.layout.addLayout(layout)

    def buttonClicked(self, text: str):
        """ the used as selected a colormap, we are done """
        self.result = text
        self.done(1)

    def exec(self):
        """ execute the dialog and return the result """
        result = QtWidgets.QDialog.exec(self)
        return self.result, result == 1

    def getBackground(self, color: str) -> str:
        """ convert a colormap to a gradient background """
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        try:
            cmap = plt.get_cmap(color)
        except:
            return ""
        text = "background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, "
        N = 10
        for i in range(N):
            i = i / (N - 1)
            text += f"stop: {i:.2f} {mpl.colors.to_hex(cmap(i))}, "
        text = text[:-2] + ");"
        return text


class QDragableColor(QtWidgets.QLabel):
    """ a color widget that can be dragged onto another QDragableColor widget to exchange the two colors.
    Alternatively it can be right-clicked to select either a color or a colormap through their respective menus.
    The button can represent either a single color or a colormap.
    """

    color_changed = QtCore.Signal(str)
    color_changed_by_color_picker = QtCore.Signal(bool)
    valueChanged = QtCore.Signal(str)

    def __init__(self, value: str):
        """ initialize with a color """
        super().__init__(value)
        import matplotlib.pyplot as plt
        self.maps = plt.colormaps()
        self.setAlignment(QtCore.Qt.AlignHCenter)
        self.setColor(value, True)

    def getBackground(self) -> str:
        """ get the background of the color button """

        try:
            cmap = plt.get_cmap(self.color)
        except:
            return ""
        text = "background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, "
        N = 10
        for i in range(N):
            i = i / (N - 1)
            text += f"stop: {i:.2f} {mpl.colors.to_hex(cmap(i))}, "
        text = text[:-2] + ");"
        return text

    def setColor(self, value: str, no_signal=False):
        """ set the current color """
        # display and save the new color
        self.color = value
        self.setText(value)
        self.color_changed.emit(value)
        self.valueChanged.emit(value)
        if value in self.maps:
            self.setStyleSheet("text-align: center; border: 2px solid black; padding: 0.1em; "+self.getBackground())
        else:
            self.setStyleSheet(f"text-align: center; background-color: {value}; border: 2px solid black; padding: 0.1em; ")

    def getColor(self) -> str:
        """ get the current color """
        # return the color
        return self.color

    def value(self):
        return self.color

    def setValue(self, value):
        self.setColor(value)

    def mousePressEvent(self, event):
        """ when a mouse button is pressed """
        # a mouse button opens a color choose menu
        if event.button() == QtCore.Qt.LeftButton:
            self.openDialog()

    def openDialog(self):
        """ open a color chooser dialog """
        if self.color in self.maps:
            dialog = ColorMapChoose(self.parent(), self.color)
            colormap, selected = dialog.exec()
            if selected is False:
                return
            self.setColor(colormap)
        else:
            # get new color from color picker
            qcolor = QtGui.QColor(*tuple(int(x * 255) for x in mpl.colors.to_rgb(self.getColor())))
            color = QtWidgets.QColorDialog.getColor(qcolor, self.parent())
            # if a color is set, apply it
            if color.isValid():
                color = "#%02x%02x%02x" % color.getRgb()[:3]
                self.setColor(color)
                self.color_changed_by_color_picker.emit(True)


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
try:
    mpl.colormaps.register(LinearSegmentedColormap('red', {'red': ((0,0,0),(1,1,1)), 'green': ((0,0,0),(1,0,0)), 'blue': ((0.0,  0.0, 0.0), (1,  0, 0))}))
    mpl.colormaps.register(LinearSegmentedColormap('orange', {'red': ((0,0,0),(1,1,1)), 'green': ((0,0,0),(1,0.5,0.5)), 'blue': ((0.0,  0.0, 0.0), (1,  0, 0))}))
    mpl.colormaps.register(LinearSegmentedColormap('yellow', {'red': ((0,0,0),(1,1,1)), 'green': ((0,0,0),(1,1,1)), 'blue': ((0.0,  0.0, 0.0), (1,  0, 0))}))
    mpl.colormaps.register(LinearSegmentedColormap('lime', {'red': ((0,0,0),(1,0.5,0.5)), 'green': ((0,0,0),(1,1,1)), 'blue': ((0.0,  0.0, 0.0), (1,  0, 0))}))
    mpl.colormaps.register(LinearSegmentedColormap('green', {'red': ((0,0,0),(1,0,0)), 'green': ((0,0,0),(1,1,1)), 'blue': ((0.0,  0.0, 0.0), (1,  0, 0))}))
    mpl.colormaps.register(LinearSegmentedColormap('mint', {'red': ((0,0,0),(1,0,0)), 'green': ((0,0,0),(1,1,1)), 'blue': ((0.0,  0.0, 0.0), (1,  0.5, 0.5))}))
    mpl.colormaps.register(LinearSegmentedColormap('cyan', {'red': ((0,0,0),(1,0,0)), 'green': ((0,0,0),(1,1,1)), 'blue': ((0.0,  0.0, 0.0), (1, 1, 1))}))
    mpl.colormaps.register(LinearSegmentedColormap('navy', {'red': ((0,0,0),(1,0,0)), 'green': ((0,0,0),(1,0.5,0.5)), 'blue': ((0.0,  0.0, 0.0), (1, 1, 1))}))
    mpl.colormaps.register(LinearSegmentedColormap('blue', {'red': ((0,0,0),(1,0,0)), 'green': ((0,0,0),(1,0,0)), 'blue': ((0.0,  0.0, 0.0), (1,  1, 1))}))
    mpl.colormaps.register(LinearSegmentedColormap('purple', {'red': ((0,0,0),(1,0.5,0.5)), 'green': ((0,0,0),(1,0,0)), 'blue': ((0.0,  0.0, 0.0), (1,  1, 1))}))
    mpl.colormaps.register(LinearSegmentedColormap('magenta', {'red': ((0,0,0),(1,1,1)), 'green': ((0,0,0),(1,0,0)), 'blue': ((0.0,  0.0, 0.0), (1,  1, 1))}))
    mpl.colormaps.register(LinearSegmentedColormap('grape', {'red': ((0,0,0),(1,1,1)), 'green': ((0,0,0),(1,0,0)), 'blue': ((0.0,  0.0, 0.0), (1,  0.5, 0.5))}))
    #mpl.colormaps.register(LinearSegmentedColormap('redd', {'red': ((0.0,  0.0, 0.0), (1,  1, 1))}))
    #mpl.colormaps.register(LinearSegmentedColormap('greenn', {'green': ((0.0,  0.0, 0.0), (1,  1, 1))}))
except:
    print ("Did not update colormaps since colormaps with identical names already existing. ")

class SuperQLabel(QtWidgets.QLabel):
    def __init__(self, *args, **kwargs):
        super(SuperQLabel, self).__init__(*args, **kwargs)

        self.textalignment = QtCore.Qt.AlignLeft | QtCore.Qt.TextWrapAnywhere
        self.isTextLabel = True
        self.align = None

    def paintEvent(self, event):

        opt = QtWidgets.QStyleOption()
        opt.initFrom(self)
        painter = QtGui.QPainter(self)

        self.style().drawPrimitive(QtWidgets.QStyle.PE_Widget, opt, painter, self)

        self.style().drawItemText(painter, self.rect(),
                                  self.textalignment, self.palette(), True, self.text())

