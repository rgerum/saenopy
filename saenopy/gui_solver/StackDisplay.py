import os
import qtawesome as qta
from qtpy import QtCore, QtWidgets, QtGui
import numpy as np

from qimage2ndarray import array2qimage
import imageio
import inspect

from pathlib import Path

import saenopy
import saenopy.multigridHelper
from saenopy.gui import QtShortCuts, QExtendedGraphicsView
import saenopy.getDeformations
import saenopy.multigridHelper
import saenopy.materials
from saenopy import Result

from typing import Tuple

from .PipelineModule import PipelineModule
from .QTimeSlider import QTimeSlider


class StackDisplay(PipelineModule):
    view_single = False
    view_single_timer = None
    view_single_switch = 0

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        self.view_single_timer = QtCore.QTimer()
        self.view_single_timer.timeout.connect(self.viewSingleTimer)
        self.view_single_timer.setInterval(400)

        with self.parent.tabs.createTab("Stacks") as self.tab:
            with QtShortCuts.QHBoxLayout() as layout:
                with QtShortCuts.QVBoxLayout() as layout:
                    with QtShortCuts.QHBoxLayout() as layout:
                        self.label1 = QtWidgets.QLabel("reference").addToLayout()
                        layout.addStretch()

                        self.button_display_single = QtShortCuts.QInputBool(None, "", icon=[
                            QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "view_two.ico")),
                            QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "view_single.ico")),
                        ], group=True, tooltip=["Parallel view of the two stacks", "View only one stack and alternate view between them"])
                        self.button_display_single.valueChanged.connect(self.setSingle)
                        QtShortCuts.QVLine()

                        self.channel_select = QtShortCuts.QInputChoice(None, "", 0, [0], [""])
                        self.channel_select.valueChanged.connect(self.update_display)
                        self.channel_select.valueChanged.connect(
                            lambda value: parent.shared_properties.change_property("channel_select", value, self))
                        parent.shared_properties.add_property("channel_select", self)
                        self.channel_select.valueChanged.connect(self.z_slider_value_changed)

                        self.button_z_proj = QtShortCuts.QInputBool(None, "", icon=[
                            QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "slice0.ico")),
                            QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "slice1.ico")),
                            QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "slice2.ico")),
                            QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "slice_all.ico")),
                        ], group=False, tooltip=["Show only the current z slice",
                                                "Show a maximum intensity projection over +-5 z slices",
                                                "Show a maximum intensity projection over +-10 z slices",
                                                "Show a maximum intensity projection over all z slices"])
                        self.button_z_proj.valueChanged.connect(lambda value: self.setZProj([0, 5, 10, 1000][value]))
                        self.button_z_proj.valueChanged.connect(
                            lambda value: parent.shared_properties.change_property("button_z_proj", value, self))
                        parent.shared_properties.add_property("button_z_proj", self)
                        QtShortCuts.QVLine()

                        self.contrast_enhance = QtShortCuts.QInputBool(None, "contrast enhance", False,
                                                                       settings=self.parent.settings,
                                                                       settings_key="stack_contrast_enhance")
                        self.contrast_enhance.valueChanged.connect(self.z_slider_value_changed)
                        self.button = QtWidgets.QPushButton(qta.icon("fa5s.home"), "").addToLayout()
                        self.button.setToolTip("reset view")
                        self.button.clicked.connect(lambda x: (self.view1.fitInView(), self.view2.fitInView()))
                        self.button2 = QtWidgets.QPushButton(qta.icon("mdi.floppy"), "").addToLayout()
                        self.button2.setToolTip("save image")
                        self.button2.clicked.connect(self.export)
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

                    self.t_slider = QTimeSlider(connected=self.z_slider_value_changed).addToLayout()
                    self.tab.parent().t_slider = self.t_slider
                self.z_slider = QTimeSlider("z", self.z_slider_value_changed, "set z position",
                                            QtCore.Qt.Vertical).addToLayout()
                self.z_slider.t_slider.valueChanged.connect(
                    lambda value: parent.shared_properties.change_property("z_slider", value, self))
                parent.shared_properties.add_property("z_slider", self)

        self.view1.link(self.view2)
        self.current_tab_selected = True
        self.setParameterMapping(None, {})

    def setZProj(self, value):
        if self.result:
            if value == 0:
                self.result.stack_parameter["z_project_name"] = None
            else:
                self.result.stack_parameter["z_project_name"] = "maximum"
            self.result.stack_parameter["z_project_range"] = value
            self.result.stack_parameter["z_project_range"] = value
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

    def check_evaluated(self, result: Result) -> bool:
        return self.result is not None

    def check_available(self, result: Result) -> bool:
        if result is not None and result.stack is not None and len(result.stack) > 0:
            return True
        return False

    def export(self):
        if self.result is None:
            return
        import tifffile
        new_path = QtWidgets.QFileDialog.getSaveFileName(None, "Save Images", os.getcwd())
        # if we got one, set it
        if new_path:
            if isinstance(new_path, tuple):
                new_path = new_path[0]
            else:
                new_path = str(new_path)
            new_path = new_path.strip(".gif").strip("_relaxed.tif").strip("_deformed.tif")
            new_path = Path(new_path)
            print(new_path.parent / (new_path.stem + "_deformed.tif"))
            t = self.t_slider.value()
            z = self.z_slider.value()
            c = self.channel_select.value()
            if self.result.stack_reference is not None:
                stack1, stack2 = self.result.stack_reference[:, :, :, z, c], self.result.stack[t][:, :, :, z, c]
            else:
                stack1, stack2 = self.result.stack[t][:, :, :, z, c], self.result.stack[t + 1][:, :, :, z, c]
            tifffile.imsave(new_path.parent / (new_path.stem + "_relaxed.tif"), stack1)
            tifffile.imsave(new_path.parent / (new_path.stem + "_deformed.tif"), stack2)
            imageio.mimsave(new_path.parent / (new_path.stem + ".gif"), [stack1, stack2], fps=2)

    def update_display(self):
        return
        if self.check_available(self.result):
            self.scale1.setScale(self.result.stack[0].voxel_size)
            self.scale2.setScale(self.result.stack[1].voxel_size)

    def property_changed(self, name, value):
        if name == "z_slider":
            self.z_slider.setValue(value)
            self.z_slider_value_changed()
        if name == "button_z_proj":
            self.button_z_proj.setValue(value)
            self.z_slider_value_changed()
        if name == "channel_select":
            print("channel_select", value, self.channel_select.value_names)
            if value < len(self.channel_select.value_names):
                self.channel_select.setValue(value)
            else:
                self.channel_select.setValue(0)
            print(self.channel_select.value())
            self.z_slider_value_changed()

    def setResult(self, result: Result):
        super().setResult(result)
        if result and result.stack and result.stack[0]:
            self.z_slider.setRange(0, result.stack[0].shape[2] - 1)
            self.z_slider.setValue(self.result.stack[0].shape[2] // 2)
            self.z_slider_value_changed()

            if result.stack[0].channels:
                self.channel_select.setValues(np.arange(len(result.stack[0].channels)),
                                                          result.stack[0].channels)
                self.channel_select.setVisible(True)
            else:
                self.channel_select.setValue(0)
                self.channel_select.setVisible(False)

    def z_slider_value_changed(self):
        if self.result is not None:
            for i in range(2 - self.view_single):
                if self.result.stack_reference is not None:
                    if i + self.view_single_switch == 0:
                        stack = self.result.stack_reference
                    else:
                        stack = self.result.stack[self.t_slider.value()]
                else:
                    stack = self.result.stack[self.t_slider.value() + i + self.view_single_switch]
                [self.scale1, self.scale2][i].setScale(stack.voxel_size)

                self.views[i].setToolTip(
                    f"stack\n{stack.description(self.z_slider.value())}")

                c = self.channel_select.value()

                if self.result.stack_parameter["z_project_name"] == "maximum":
                    start = np.clip(self.z_slider.value()-self.result.stack_parameter["z_project_range"], 0, stack.shape[2])
                    end = np.clip(self.z_slider.value()+self.result.stack_parameter["z_project_range"], 0, stack.shape[2])
                    im = stack[:, :, :, start:end, c]
                    im = np.max(im, axis=3)
                else:
                    im = stack[:, :, :, self.z_slider.value(), c]
                if self.contrast_enhance.value():
                    (min, max) = np.percentile(im, (1, 99))
                    im = im.astype(np.float32)-min
                    im = im.astype(np.float64) * 255 / (max-min)
                    im = np.clip(im, 0, 255).astype(np.uint8)
                self.pixmaps[i].setPixmap(QtGui.QPixmap(array2qimage(im)))
                self.views[i].setExtend(im.shape[0], im.shape[0])
                self.parent.display_image = (im, stack.voxel_size)

            self.z_slider.setToolTip(f"set z position\ncurrent position {self.z_slider.value()}")

    def get_code(self) -> Tuple[str, str]:
        from saenopy.solver import common_start, common_end
        def filename_to_string(filename, insert="{z}"):
            if isinstance(filename, list):
                return str(Path(common_start(filename) + insert + common_end(filename)))
            return str(Path(filename)).replace("*", insert)

        from saenopy.solver import get_stacks
        import_code = ""
        if self.result.time_delta is None:
            def code(stack1, stack2, output, voxel_size1):
                # load the relaxed and the contracted stack, {z} is the placeholder for the z stack
                # use * as a placeholder to import multiple experiments at once
                results = saenopy.get_stacks([
                    stack1,
                    stack2,
                ], output, voxel_size=voxel_size1)

            data = dict(
                stack1=filename_to_string(self.result.stack[0].filename),
                stack2=filename_to_string(self.result.stack[1].filename),
                output=str(Path(self.result.output).parent),
                voxel_size1=self.result.stack[0].voxel_size,
            )
        else:
            def code(stack1, output, voxel_size1, time_delta1):
                # load the time series stack, {z} is the placeholder for the z stack, {t} is the placeholder for the time steps
                # use * as a placeholder to import multiple experiments at once
                results = saenopy.get_stacks(stack1,
                                             output,
                                             voxel_size=voxel_size1, time_delta=time_delta1)

            stack_filenames = filename_to_string([filename_to_string(stack.filename) for stack in self.result.stack],
                                                 insert="{t}")
            data = dict(
                stack1=stack_filenames,
                output=str(Path(self.result.output).parent),
                voxel_size1=self.result.stack[0].voxel_size,
                time_delta1=self.result.time_delta,
            )

        code_lines = inspect.getsource(code).split("\n")[1:]
        indent = len(code_lines[0]) - len(code_lines[0].lstrip())
        code = "\n".join(line[indent:] for line in code_lines)

        for key, value in data.items():
            if isinstance(value, str):
                if "\\" in value:
                    code = code.replace(key, "r'" + value + "'")
                else:
                    code = code.replace(key, "'" + value + "'")
            else:
                code = code.replace(key, str(value))
        return import_code, code


class ModuleScaleBar(QtWidgets.QGroupBox):
    pixtomu = None

    def __init__(self, parent, view):
        QtWidgets.QWidget.__init__(self)
        self.parent = parent

        self.font = QtGui.QFont()
        self.font.setPointSize(16)

        self.scale = 1

        self.scalebar = QtWidgets.QGraphicsRectItem(0, 0, 1, 1, view.hud_lowerRight)
        self.scalebar.setBrush(QtGui.QBrush(QtGui.QColor("white")))
        self.scalebar.setPen(QtGui.QPen(QtGui.QColor("white")))
        self.scalebar.setPos(-20, -20)
        self.scalebar_text = QtWidgets.QGraphicsTextItem("", view.hud_lowerRight)
        self.scalebar_text.setFont(self.font)
        self.scalebar_text.setDefaultTextColor(QtGui.QColor("white"))

        self.time_text = QtWidgets.QGraphicsTextItem("", view.hud_upperRight)
        self.time_text.setFont(self.font)
        self.time_text.setDefaultTextColor(QtGui.QColor("white"))

        view.signal_zoom.connect(self.zoomEvent)

        self.updateStatus()

    def updateStatus(self):
        self.updateBar()

    def zoomEvent(self, scale, pos):
        self.scale = scale
        self.updateBar()

    def setScale(self, voxel_size):
        self.pixtomu = voxel_size[0]
        self.updateBar()

    def updateBar(self):
        if self.scale == 0 or self.pixtomu is None:
            return
        mu = 100*self.pixtomu/self.scale
        values = [1, 5, 10, 25, 50, 75, 100, 150, 200, 250, 500, 1000, 1500, 2000, 2500, 5000, 10000]
        old_v = mu
        for v in values:
            if mu < v:
                mu = old_v
                break
            old_v = v
        if np.abs(self.pixtomu) < 1e-10:
            pixel = 0
        else:
            pixel = mu/(self.pixtomu)*self.scale
        self.scalebar.setRect(0, 0, -pixel, 5)
        self.scalebar_text.setPos(-pixel-20-25, -20-30)
        self.scalebar_text.setTextWidth(pixel+50)
        self.scalebar_text.setHtml(u"<center>%d&thinsp;µm</center>" % mu)



