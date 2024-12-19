import os
import traceback

import qtawesome as qta
from qtpy import QtCore, QtWidgets, QtGui
import numpy as np

from qimage2ndarray import array2qimage
import imageio

from pathlib import Path

import tifffile

from saenopy.gui.common import QtShortCuts, QExtendedGraphicsView
from saenopy import Result
from saenopy.gui.common.TabModule import TabModule
from saenopy.gui.common.resources import resource_icon
from saenopy.gui.common.QTimeSlider import QTimeSlider
from saenopy.gui.common.ModuleScaleBar import ModuleScaleBar


class TabStackDisplay(TabModule):
    view_single = False
    view_single_timer = None
    view_single_switch = 0

    def __init__(self, parent: "BatchEvaluate"):
        super().__init__(parent)

        self.view_single_timer = QtCore.QTimer()
        self.view_single_timer.timeout.connect(self.viewSingleTimer)
        self.view_single_timer.setInterval(400)

        with self.parent.tabs.createTab("Stacks") as self.tab:
            with QtShortCuts.QHBoxLayout() as layout:
                with QtShortCuts.QVBoxLayout() as layout:
                    with QtShortCuts.QHBoxLayout() as layout:
                        self.label1 = QtWidgets.QLabel("reference").addToLayout()
                        layout.addStretch()

                    with QtShortCuts.QHBoxLayout() as layout:
                        with QtShortCuts.QVBoxLayout() as layout:
                            self.view1 = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
                            self.view1.setMinimumWidth(300)
                            self.pixmap1 = QtWidgets.QGraphicsPixmapItem(self.view1.origin)
                            self.scale1 = ModuleScaleBar(self, self.view1)

                            self.label2 = QtWidgets.QLabel("active").addToLayout()
                            self.view2 = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
                            # self.label2.setMinimumWidth(300)
                            self.pixmap2 = QtWidgets.QGraphicsPixmapItem(self.view2.origin)
                            self.scale2 = ModuleScaleBar(self, self.view2)

                            self.views = [self.view1, self.view2]
                            self.pixmaps = [self.pixmap1, self.pixmap2]

                        self.z_slider = QTimeSlider("z", self.z_slider_value_changed, "set z position",
                                                    QtCore.Qt.Vertical).addToLayout()
                        self.z_slider.t_slider.valueChanged.connect(
                            lambda value: parent.shared_properties.change_property("z_slider", value, self))
                        parent.shared_properties.add_property("z_slider", self)

                    with QtShortCuts.QHBoxLayout() as layout:
                        layout.addStretch()

                        self.button_display_single = QtShortCuts.QInputBool(None, "", icon=[
                            resource_icon("view_two.ico"),
                            resource_icon("view_single.ico"),
                        ], group=True, tooltip=["Parallel view of the two stacks", "View only one stack and alternate view between them"])
                        self.button_display_single.valueChanged.connect(self.setSingle)
                        QtShortCuts.QVLine()

                        self.channel_select = QtShortCuts.QInputChoice(None, "", 0, [0], ["       "], tooltip="From which channel to show.")
                        self.channel_select.valueChanged.connect(self.update_display)
                        self.channel_select.valueChanged.connect(
                            lambda value: parent.shared_properties.change_property("channel_select", value, self))
                        parent.shared_properties.add_property("channel_select", self)
                        self.channel_select.valueChanged.connect(self.z_slider_value_changed)

                        self.button_z_proj = QtShortCuts.QInputBool(None, "", icon=[
                            resource_icon("slice0.ico"),
                            resource_icon("slice1.ico"),
                            resource_icon("slice2.ico"),
                            resource_icon("slice_all.ico"),
                        ], group=False, tooltip=["Show only the current z slice",
                                                 "Show a maximum intensity projection over +-5 z slices",
                                                 "Show a maximum intensity projection over +-10 z slices",
                                                 "Show a maximum intensity projection over all z slices"])
                        self.button_z_proj.valueChanged.connect(lambda value: self.setZProj([0, 5, 10, 1000][value]))
                        self.button_z_proj.valueChanged.connect(
                            lambda value: parent.shared_properties.change_property("button_z_proj", value, self))
                        parent.shared_properties.add_property("button_z_proj", self)

                        self.contrast_enhance = QtShortCuts.QInputBool(None, "", icon=[
                            resource_icon("contrast0.ico"),
                            resource_icon("contrast1.ico"),
                        ], group=False, tooltip="Toggle contrast enhancement")
                        self.contrast_enhance.valueChanged.connect(self.z_slider_value_changed)
                        self.contrast_enhance.valueChanged.connect(
                            lambda value: parent.shared_properties.change_property("contrast_enhance", value, self))
                        parent.shared_properties.add_property("contrast_enhance", self)
                        QtShortCuts.QVLine()

                        self.button = QtWidgets.QPushButton(qta.icon("fa5s.home"), "").addToLayout()
                        self.button.setToolTip("reset view")
                        self.button.clicked.connect(lambda x: (self.view1.fitInView(), self.view2.fitInView()))
                        self.button2 = QtWidgets.QPushButton(qta.icon("mdi.floppy"), "").addToLayout()
                        self.button2.setToolTip("save image")
                        self.button2.clicked.connect(self.export)

                    self.t_slider = QTimeSlider(connected=self.z_slider_value_changed).addToLayout()
                    self.tab.parent().t_slider = self.t_slider

        self.view1.link(self.view2)
        self.current_tab_selected = True

    def setZProj(self, value):
        if self.result:
            if value == 0:
                self.result.stack_parameters["z_project_name"] = None
            else:
                self.result.stack_parameters["z_project_name"] = "maximum"
            self.result.stack_parameters["z_project_range"] = value
            self.result.stack_parameters["z_project_range"] = value
            self.z_slider_value_changed()

    def setSingle(self, use_single):
        self.view_single = use_single
        self.view2.setVisible(not self.view_single)
        self.label1.setVisible(not self.view_single)
        self.label2.setVisible(not self.view_single)
        if use_single:
            self.view_single_timer.start()
        else:
            self.view_single_timer.stop()
            self.view_single_switch = 0
        self.update_display()

    def viewSingleTimer(self):
        self.view_single_switch = 1-self.view_single_switch
        self.z_slider_value_changed()

    def check_available(self, result: Result) -> bool:
        if result is not None and result.stacks is not None and len(result.stacks) > 0:
            return True
        return False

    def export(self):
        if self.result is None:
            return
        new_path = QtWidgets.QFileDialog.getSaveFileName(None, "Save Images", os.getcwd())[0]
        # if we got one, set it
        if new_path:
            new_path = new_path.strip(".gif").strip("_relaxed.tif").strip("_deformed.tif")
            new_path = Path(new_path)
            print(new_path.parent / (new_path.stem + "_deformed.tif"))
            t = self.t_slider.value()
            z = self.z_slider.value()
            c = self.channel_select.value()
            if self.result.stack_reference is not None:
                stack1, stack2 = self.result.stack_reference[:, :, :, z, c], self.result.stacks[t][:, :, :, z, c]
            else:
                stack1, stack2 = self.result.stacks[t][:, :, :, z, c], self.result.stacks[t + 1][:, :, :, z, c]
            tifffile.imwrite(new_path.parent / (new_path.stem + "_relaxed.tif"), stack1)
            tifffile.imwrite(new_path.parent / (new_path.stem + "_deformed.tif"), stack2)
            if len(stack1.shape) == 3 and stack1.shape[2] == 1:
                stack1 = stack1[:, :, 0]
            if len(stack2.shape) == 3 and stack2.shape[2] == 1:
                stack2 = stack2[:, :, 0]
            imageio.mimsave(new_path.parent / (new_path.stem + ".gif"), [stack1, stack2], duration=1000 * 1./2)

    def update_display(self):
        return

    def property_changed(self, name, value):
        if name == "z_slider":
            if value != self.z_slider.value():
                self.z_slider.setValue(value)
                self.z_slider_value_changed()
        if name == "button_z_proj":
            if value != self.button_z_proj.value():
                self.button_z_proj.setValue(value)
                self.z_slider_value_changed()
        if name == "channel_select":
            if value != self.channel_select.value():
                if value < len(self.channel_select.value_names):
                    self.channel_select.setValue(value)
                else:
                    self.channel_select.setValue(0)
                self.z_slider_value_changed()
        if name == "contrast_enhance":
            if value != self.contrast_enhance.value():
                self.contrast_enhance.setValue(value)
                self.update_display()

    def setResult(self, result: Result):
        super().setResult(result)
        if result and result.stacks and result.stacks[0]:
            self.z_slider.setRange(0, result.stacks[0].shape[2] - 1)
            self.z_slider.setValue(self.result.stacks[0].shape[2] // 2)
            self.z_slider_value_changed()

            if result.stacks[0].channels:
                self.channel_select.setValues(np.arange(len(result.stacks[0].channels)),
                                              result.stacks[0].channels)
                self.channel_select.setVisible(True)
            else:
                self.channel_select.setValue(0)
                self.channel_select.setVisible(False)

    def z_slider_value_changed(self):
        if self.result is not None and len(self.result.stacks):
            for i in range(2 - self.view_single):
                if self.result.stack_reference is not None:
                    if i + self.view_single_switch == 0:
                        stack = self.result.stack_reference
                    else:
                        stack = self.result.stacks[self.t_slider.value()]
                else:
                    stack = self.result.stacks[self.t_slider.value() + i + self.view_single_switch]
                [self.scale1, self.scale2][i].setScale(stack.voxel_size)

                c = self.channel_select.value()
                z = self.z_slider.value()

                try:
                    self.views[i].setToolTip(f"stack\n{stack.description(z)}")

                    if self.result.stack_parameters["z_project_name"] == "maximum":
                        start = np.clip(z - self.result.stack_parameters["z_project_range"], 0, stack.shape[2])
                        end = np.clip(z + self.result.stack_parameters["z_project_range"], 0, stack.shape[2])
                        im = stack[:, :, :, start:end, c]
                        im = np.max(im, axis=3)
                    else:
                        im = stack[:, :, :, z, c]
                    if self.contrast_enhance.value():
                        (min, max) = np.percentile(im, (1, 99))
                        im = im.astype(np.float32)-min
                        im = im.astype(np.float64) * 255 / (max-min)
                        im = np.clip(im, 0, 255).astype(np.uint8)
                except FileNotFoundError as err:
                    traceback.print_exception(err)
                    im = np.zeros([255, 255, 3], dtype=np.uint8)
                    im[:, :, 0] = 255
                    im[:, :, 2] = 255
                    self.views[i].setToolTip(f"stack information not found")
                self.pixmaps[i].setPixmap(QtGui.QPixmap(array2qimage(im)))
                self.views[i].setExtend(im.shape[1], im.shape[0])
                self.parent.display_image = (im, stack.voxel_size)

            self.z_slider.setToolTip(f"set z position\ncurrent position {self.z_slider.value()}")
