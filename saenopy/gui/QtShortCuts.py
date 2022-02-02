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

        if tooltip is not None:
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

    def setValue(self, value):
        self.no_signal = True
        if self.settings is not None:
            self.settings.setValue(self.settings_key, value)
        try:
            self._doSetValue(value)
        finally:
            self.no_signal = False

    def _doSetValue(self, value):
        # dummy method to be overloaded by child classes
        pass

    def value(self):
        # dummy method to be overloaded by child classes
        pass

cast_float = float
class QInputNumber(QInput):
    slider_dragged = False

    def __init__(self, layout=None, name=None, value=0, min=None, max=None, use_slider=False, float=True, decimals=2,
                 unit=None, step=None, name_post=None, **kwargs):
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
            self.slider.setRange(min * self.decimal_factor, max * self.decimal_factor)
            self.slider.valueChanged.connect(lambda x: self._valueChangedEvent(x / self.decimal_factor))
            self.slider.sliderPressed.connect(lambda: self._setSliderDragged(True))
            self.slider.sliderReleased.connect(lambda: self._setSliderDragged(False))
        else:
            self.slider = None

        # add spin box
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
        self.spin_box.setValue(value)
        if self.slider is not None:
            self.slider.setValue(value * self.decimal_factor)

    def value(self):
        return self.spin_box.value()


class QInputString(QInput):
    error = None

    def __init__(self, layout=None, name=None, value="", allow_none=True, type=str, unit=None, name_post=None, validator=None, **kwargs):
        # initialize the super widget
        QInput.__init__(self, layout, name, **kwargs)

        if self.settings is not None:
            value = self.settings.value(self.settings_key, value)

        self.line_edit = QtWidgets.QLineEdit()
        self.layout().addWidget(self.line_edit)
        self.line_edit.editingFinished.connect(lambda: self._valueChangedEvent(self.value()))
        if type is int or type is float:
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
        self.line_edit.setText(str(value))

    def value(self):
        text = self.line_edit.text()
        if self.allow_none is True and text == "None":
            return None
        if self.type == int:
            return int(text)
        if self.type == float:
            return float(text)
        return text


class QInputBool(QInput):

    def __init__(self, layout=None, name=None, value=False, **kwargs):
        # initialize the super widget
        QInput.__init__(self, layout, name, **kwargs)

        if self.settings is not None:
            value = self.settings.value(self.settings_key, value) == "true"

        self.checkbox = QtWidgets.QCheckBox()
        self.layout().addWidget(self.checkbox)
        self.checkbox.stateChanged.connect(lambda: self._valueChangedEvent(self.value()))

        self.setValue(value)

    def _doSetValue(self, value):
        self.checkbox.setChecked(value)

    def value(self):
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
        self.value_names = value_names if value_names is not None else [str(v) for v in new_values]

        if self.values is not None:
            for i in range(len(self.values)):
                self.combobox.removeItem(0)

        self.values = new_values

        self.combobox.addItems(self.value_names)

    def _doSetValue(self, value):
        if self.reference_by_index is True:
            self.combobox.setCurrentIndex(value)
        else:
            try:
                self.combobox.setCurrentIndex(self.values.index(value))
            except ValueError:
                self.combobox.setCurrentIndex(self.value_names.index(value))

    def value(self):
        if self.reference_by_index is True:
            return self.combobox.currentIndex()
        else:
            return self.values[self.combobox.currentIndex()]


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
        self.line.setText(value)

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
        self.setValue(value)
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
    def __init__(self, layout, name, connect=None):
        super().__init__(name)
        if layout is None and current_layout is not None:
            layout = current_layout
        layout.addWidget(self)
        if connect is not None:
            self.clicked.connect(connect)

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
