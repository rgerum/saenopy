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
from saenopy.solver import Result

from typing import Tuple

from .PipelineModule import PipelineModule
from .QTimeSlider import QTimeSlider


class StackDisplay(PipelineModule):
    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with self.parent.tabs.createTab("Stacks") as self.tab:
            with QtShortCuts.QHBoxLayout() as layout:
                with QtShortCuts.QVBoxLayout() as layout:
                    with QtShortCuts.QHBoxLayout() as layout:
                        self.label1 = QtWidgets.QLabel("relaxed").addToLayout()
                        layout.addStretch()
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

                    self.label2 = QtWidgets.QLabel("deformed").addToLayout()
                    self.view2 = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
                    # self.label2.setMinimumWidth(300)
                    self.pixmap2 = QtWidgets.QGraphicsPixmapItem(self.view2.origin)
                    self.scale2 = ModuleScaleBar(self, self.view2)

                    self.views = [self.view1, self.view2]
                    self.pixmaps = [self.pixmap1, self.pixmap2]

                    self.t_slider = QTimeSlider(connected=self.update_display).addToLayout()
                    self.tab.parent().t_slider = self.t_slider
                self.z_slider = QTimeSlider("z", self.z_slider_value_changed, "set z position",
                                            QtCore.Qt.Vertical).addToLayout()

        self.view1.link(self.view2)
        self.current_tab_selected = True
        self.setParameterMapping(None, {})

    def check_evaluated(self, result: Result) -> bool:
        return self.result is not None

    def check_available(self, result: Result) -> bool:
        if result is not None and result.stack is not None and len(result.stack)>0:
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
            tifffile.imsave(new_path.parent / (new_path.stem + "_relaxed.tif"),
                            self.result.stack[0][:, :, self.z_slider.value()])
            tifffile.imsave(new_path.parent / (new_path.stem + "_deformed.tif"),
                            self.result.stack[1][:, :, self.z_slider.value()])
            imageio.mimsave(new_path.parent / (new_path.stem + ".gif"),
                            [self.result.stack[0][:, :, self.z_slider.value()],
                             self.result.stack[1][:, :, self.z_slider.value()]], fps=2)

    def update_display(self):
        if self.check_available(self.result):
            self.scale1.setScale(self.result.stack[0].voxel_size)
            self.scale2.setScale(self.result.stack[1].voxel_size)
            self.z_slider.setRange(0, self.result.stack[0].shape[2] - 1)
            self.z_slider.setValue(self.result.stack[0].shape[2] // 2)
            self.z_slider_value_changed()

    def z_slider_value_changed(self):
        if self.result is not None:
            for i in range(2):
                self.views[i].setToolTip(
                    f"stack\n{self.result.stack[self.t_slider.value() + i].description(self.z_slider.value())}")

                im = self.result.stack[self.t_slider.value() + i][:, :, self.z_slider.value()]
                if self.contrast_enhance.value():
                    im -= im.min()
                    im = (im.astype(np.float64) * 255 / im.max()).astype(np.uint8)
                self.pixmaps[i].setPixmap(QtGui.QPixmap(array2qimage(im)))
                self.views[i].setExtend(im.shape[0], im.shape[0])

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
        pixel = mu/(self.pixtomu)*self.scale
        self.scalebar.setRect(0, 0, -pixel, 5)
        self.scalebar_text.setPos(-pixel-20-25, -20-30)
        self.scalebar_text.setTextWidth(pixel+50)
        self.scalebar_text.setHtml(u"<center>%d&thinsp;µm</center>" % mu)



