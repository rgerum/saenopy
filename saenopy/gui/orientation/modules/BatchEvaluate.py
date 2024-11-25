import sys

# Setting the Qt bindings for QtPy
import os

import pandas as pd
import qtawesome as qta
from qtpy import QtCore, QtWidgets, QtGui
import numpy as np
from saenopy.gui.common import QtShortCuts, QExtendedGraphicsView
from saenopy.gui.common.gui_classes import Spoiler, CheckAbleGroup, QHLine, QVLine, MatplotlibWidget, NavigationToolbar, execute, kill_thread, ListWidget
import imageio
from qimage2ndarray import array2qimage
import matplotlib.pyplot as plt
import glob
import imageio
import threading
import glob
import re
from pathlib import Path

from saenopy.examples import get_examples_orientation
from .AddFilesDialog import AddFilesDialog

settings = QtCore.QSettings("FabryLab", "CompactionAnalyzer")

class BatchEvaluate(QtWidgets.QWidget):
    progress_signal = QtCore.Signal(int, int, int, int)
    measurement_evaluated_signal = QtCore.Signal(int, int)
    finished_signal = QtCore.Signal()
    thread = None

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumWidth(800)
        self.setMinimumHeight(400)
        self.setWindowTitle("Saenopy Viewer")

        with QtShortCuts.QHBoxLayout(self) as main_layout:
            with QtShortCuts.QSplitter() as lay:
                with QtShortCuts.QVBoxLayout() as layout:
                    layout.setContentsMargins(0, 0, 0, 0)
                    self.list = ListWidget(layout, add_item_button="add measurements")
                    self.list.addItemClicked.connect(self.show_files)
                    self.list.itemSelectionChanged.connect(self.listSelected)
                    self.progress1 = QtWidgets.QProgressBar()
                    layout.addWidget(self.progress1)

                with QtShortCuts.QVBoxLayout() as layout:
                    self.label_text = QtWidgets.QLabel().addToLayout()

                    self.label = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
                    self.label.setMinimumWidth(300)
                    self.pixmap = QtWidgets.QGraphicsPixmapItem(self.label.origin)
                    self.contour = QtWidgets.QGraphicsPathItem(self.label.origin)
                    pen = QtGui.QPen(QtGui.QColor(255, 0, 0))
                    pen.setCosmetic(True)
                    self.contour.setPen(pen)

                    self.label2 = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
                    self.pixmap2 = QtWidgets.QGraphicsPixmapItem(self.label2.origin)

                    self.contour2 = QtWidgets.QGraphicsPathItem(self.label2.origin)
                    self.contour2.setPen(pen)

                    # self.label_text2 = QtWidgets.QLabel().addToLayout()
                    # self.progress2 = QtWidgets.QProgressBar().addToLayout()

                frame = QtWidgets.QFrame().addToLayout()
                frame.setMaximumWidth(300)
                with QtShortCuts.QVBoxLayout(frame) as layout:
                    frame2 = QtWidgets.QFrame().addToLayout()
                    with QtShortCuts.QVBoxLayout(frame2, no_margins=True) as layout:
                        with QtShortCuts.QHBoxLayout():
                            self.scale = QtShortCuts.QInputString(None, "scale", "1.0", type=float, settings=settings,
                                                                  settings_key="orientation/scale")
                            QtWidgets.QLabel("um/px").addToLayout()
                        with QtShortCuts.QHBoxLayout():
                            self.sigma_tensor = QtShortCuts.QInputString(None, "sigma_tensor", "7.0", type=float,
                                                                         settings=settings,
                                                                         settings_key="orientation/sigma_tensor")
                            self.sigma_tensor_type = QtShortCuts.QInputChoice(None, "", "um", ["um", "pixel"],
                                                                              settings=settings,
                                                                              settings_key="orientation/sigma_tensor_unit")
                            self.sigma_tensor_button = QtShortCuts.QPushButton(None, "detect",
                                                                               self.sigma_tensor_button_clicked)
                            self.sigma_tensor_button.setDisabled(True)
                        with QtShortCuts.QHBoxLayout():
                            self.edge = QtShortCuts.QInputString(None, "edge", "40", type=int, settings=settings,
                                                                 settings_key="orientation/edge",
                                                                 tooltip="How many pixels to cut at the edge of the image.")
                            QtWidgets.QLabel("px").addToLayout()
                            self.max_dist = QtShortCuts.QInputString(None, "max_dist", "None", type=int,
                                                                     settings=settings,
                                                                     settings_key="orientation/max_dist",
                                                                     tooltip="Optional: specify the maximal distance around the cell center",
                                                                     none_value=None)
                            QtWidgets.QLabel("px").addToLayout()

                        with QtShortCuts.QHBoxLayout():
                            self.sigma_first_blur = QtShortCuts.QInputString(None, "sigma_first_blur", "0.5",
                                                                             type=float, settings=settings,
                                                                             settings_key="orientation/sigma_first_blur")
                            QtWidgets.QLabel("px").addToLayout()
                        with QtShortCuts.QHBoxLayout():
                            self.angle_sections = QtShortCuts.QInputString(None, "angle_sections", "5", type=int,
                                                                           settings=settings,
                                                                           settings_key="orientation/angle_sections")
                            QtWidgets.QLabel("deg").addToLayout()

                        with QtShortCuts.QHBoxLayout():
                            self.shell_width = QtShortCuts.QInputString(None, "shell_width", "5", type=float,
                                                                        settings=settings,
                                                                        settings_key="orientation/shell_width")
                            self.shell_width_type = QtShortCuts.QInputChoice(None, "", "um", ["um", "pixel"],
                                                                             settings=settings,
                                                                             settings_key="orientation/shell_width_type")

                        with QtShortCuts.QGroupBox(None, "Segmentation Parameters"):
                            self.segmention_thres = QtShortCuts.QInputString(None, "segmention_thresh", "1.0",
                                                                             type=float,
                                                                             settings=settings,
                                                                             settings_key="orientation/segmention_thres")
                            self.segmention_thres.valueChanged.connect(self.listSelected)
                            with QtShortCuts.QHBoxLayout():
                                self.seg_gaus1 = QtShortCuts.QInputString(None, "seg_gauss1", "0.5", type=float,
                                                                          settings=settings,
                                                                          settings_key="orientation/seg_gaus1")
                                self.seg_gaus1.valueChanged.connect(self.listSelected)
                                self.seg_gaus2 = QtShortCuts.QInputString(None, "seg_gauss2", "100", type=float,
                                                                          settings=settings,
                                                                          settings_key="orientation/seg_gaus2")
                                self.seg_gaus2.valueChanged.connect(self.listSelected)

                            with CheckAbleGroup(self, "individual segmentation").addToLayout() as self.individual_data:
                                with QtShortCuts.QVBoxLayout() as layout2:
                                    self.segmention_thres_indi = QtShortCuts.QInputString(None, "segmention_thresh",
                                                                                          None, type=float,
                                                                                          allow_none=True)
                                    self.segmention_thres_indi.valueChanged.connect(self.listSelected)
                                    with QtShortCuts.QHBoxLayout():
                                        self.seg_gaus1_indi = QtShortCuts.QInputString(None, "seg_gauss1", None,
                                                                                       type=float, allow_none=True)
                                        self.seg_gaus1_indi.valueChanged.connect(self.listSelected)
                                        self.seg_gaus2_indi = QtShortCuts.QInputString(None, "seg_gauss2", None,
                                                                                       type=float, allow_none=True)
                                        self.seg_gaus2_indi.valueChanged.connect(self.listSelected)

                    layout.addStretch()

                    self.button_run = QtShortCuts.QPushButton(None, "run", self.run)
        self.images = []
        self.data = []
        self.list.setData(self.data)

        self.input_list = [
            frame2,
        ]

        self.individual_data.value_changed.connect(self.changedCheckBox)

        self.progress_signal.connect(self.progress_callback)
        self.measurement_evaluated_signal.connect(self.measurement_evaluated)
        self.finished_signal.connect(self.finished)

    def sigma_tensor_button_clicked(self):
        parent = self

        class SigmaRange(QtWidgets.QDialog):
            def __init__(self, parent):
                super().__init__(parent)
                self.setWindowTitle("Determine Sigma Tensor")
                with QtShortCuts.QVBoxLayout(self) as layout:
                    self.output_folder = QtShortCuts.QInputFolder(None, "output folder", settings=settings,
                                                                  settings_key="orientation/sigma_tensor_range_output")
                    self.label_scale = QtWidgets.QLabel(f"Scale is {parent.scale.value()} px/um").addToLayout(layout)
                    with QtShortCuts.QHBoxLayout() as layout2:
                        self.sigma_tensor_min = QtShortCuts.QInputString(None, "min", "1.0", type=float,
                                                                         settings=settings,
                                                                         settings_key="orientation/sigma_tensor_min")
                        self.sigma_tensor_max = QtShortCuts.QInputString(None, "max", "15", type=float,
                                                                         settings=settings,
                                                                         settings_key="orientation/sigma_tensor_max")
                        self.sigma_tensor_step = QtShortCuts.QInputString(None, "step", "1", type=float,
                                                                          settings=settings,
                                                                          settings_key="orientation/sigma_tensor_step")
                        self.sigma_tensor_type = QtShortCuts.QInputChoice(None, "", "um", ["um", "pixel"],
                                                                          settings=settings,
                                                                          settings_key="orientation/sigma_tensor_unit")

                    self.progresss = QtWidgets.QProgressBar().addToLayout(layout)

                    self.canvas = MatplotlibWidget(self)
                    layout.addWidget(self.canvas)
                    layout.addWidget(NavigationToolbar(self.canvas, self))

                    with QtShortCuts.QHBoxLayout() as layout3:
                        # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                        layout3.addStretch()
                        self.button_addList2 = QtShortCuts.QPushButton(None, "cancel", self.reject)
                        self.button_addList1 = QtShortCuts.QPushButton(None, "run", self.run)

            def run(self):
                from natsort import natsorted
                fiber, cell, output, attr = parent.data[parent.list.currentRow()][2]

                output_folder = self.output_folder.value()

                sigma_tensor_min = self.sigma_tensor_min.value()
                sigma_tensor_max = self.sigma_tensor_max.value()
                sigma_tensor_step = self.sigma_tensor_step.value()
                if self.sigma_tensor_type.value() == "um":
                    sigma_tensor_min /= parent.scale.value()
                    sigma_tensor_max /= parent.scale.value()
                    sigma_tensor_step /= parent.scale.value()
                shell_width = parent.shell_width.value()
                if parent.shell_width_type.value() == "um":
                    shell_width /= parent.scale.value()

                sigma_list = np.arange(sigma_tensor_min, sigma_tensor_max + sigma_tensor_step, sigma_tensor_step)
                self.progresss.setRange(0, len(sigma_list))

                from CompactionAnalyzer.CompactionFunctions import StuctureAnalysisMain, generate_lists
                for index, sigma in enumerate(sigma_list):
                    sigma = float(sigma)
                    self.progresss.setValue(index)
                    app.processEvents()
                    # Create outputfolder
                    output_sub = os.path.join(output_folder,
                                              rf"Sigma{str(sigma * parent.scale.value()).zfill(3)}")  # subpath to store results
                    fiber_list, cell_list, out_list = generate_lists(fiber, cell,
                                                                     output_main=output_sub)

                    StuctureAnalysisMain(fiber_list=fiber_list,
                                         cell_list=cell_list,
                                         out_list=out_list,
                                         scale=parent.scale.value(),
                                         sigma_tensor=sigma,
                                         edge=parent.edge.value(),
                                         max_dist=parent.max_dist.value(),
                                         segmention_thres=parent.segmention_thres.value() if attr[
                                                                                                 "segmention_thres"] is None else
                                         attr["segmention_thres"],
                                         seg_gaus1=parent.seg_gaus1.value() if attr["seg_gaus1"] is None else attr[
                                             "seg_gaus1"],
                                         seg_gaus2=parent.seg_gaus2.value() if attr["seg_gaus2"] is None else attr[
                                             "seg_gaus2"],
                                         sigma_first_blur=parent.sigma_first_blur.value(),
                                         angle_sections=parent.angle_sections.value(),
                                         shell_width=shell_width,
                                         regional_max_correction=True,
                                         seg_iter=1,
                                         SaveNumpy=False,
                                         plotting=True,
                                         dpi=100
                                         )
                    self.progresss.setValue(index + 1)

                    ### plot results
                    # read in all creates result folder
                    result_folders = natsorted(
                        glob.glob(os.path.join(output_folder, "Sigma*", "*", "results_total.xlsx")))

                    import yaml
                    sigmas = []
                    orientation = []
                    for folder in result_folders:
                        with (Path(folder).parent / "parameters.yml").open() as fp:
                            parameters = yaml.load(fp, Loader=yaml.SafeLoader)["Parameters"]
                            sigmas.append(parameters["sigma_tensor"][0] * parameters["scale"][0])
                        orientation.append(pd.read_excel(folder)["Orientation (weighted by intensity and coherency)"])

                    self.canvas.setActive()
                    plt.cla()
                    plt.axis("auto")

                    plt.plot(sigmas, orientation, "o-")
                    plt.ylabel("Orientation", fontsize=12)
                    plt.xlabel("Windowsize (μm)", fontsize=12)
                    plt.grid()
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_folder, "Results.png"), dpi=500)
                    self.canvas.draw()

        dialog = SigmaRange(self)
        if not dialog.exec():
            return

    def changedCheckBox(self):
        for widget in [self.segmention_thres, self.seg_gaus1, self.seg_gaus2]:
            widget.setDisabled(self.individual_data.value())
        if not self.individual_data.value():
            for widget in [self.segmention_thres_indi, self.seg_gaus1_indi, self.seg_gaus2_indi]:
                widget.setValue("None")

    def show_files(self):

        dialog = AddFilesDialog(self, settings)
        if not dialog.exec():
            return

        import glob
        import re
        if dialog.mode == "new":
            fiber_list_string = os.path.normpath(dialog.fiberText.value())
            cell_list_string = os.path.normpath(dialog.cellText.value())
            output_folder = os.path.normpath(dialog.outputText.value())
        elif dialog.mode == "example":
            # get the date from the example referenced by name
            example = get_examples_orientation()[dialog.mode_data]
            fiber_list_string = str(example["input_fiber"])
            cell_list_string = str(example["input_cell"])
            output_folder = str(example["output_path"])
            print(fiber_list_string)
            print(cell_list_string)
            print(output_folder)

        from CompactionAnalyzer.CompactionFunctions import generate_lists
        fiber_list, cell_list, out_list = generate_lists(fiber_list_string, cell_list_string, output_main=output_folder)

        import matplotlib as mpl
        for fiber, cell, out in zip(fiber_list, cell_list, out_list):
            self.list.addData(fiber, True,
                              [fiber, cell, out, {"segmention_thres": None, "seg_gaus1": None, "seg_gaus2": None}],
                              mpl.colors.to_hex(f"gray"))

    def clear_files(self):
        self.list.clear()
        self.data = {}

    last_cell = None

    def listSelected(self):
        def get_pixmap(im_cell, cmap="viridis"):
            im_cell = im_cell.astype(np.float64)
            im_cell -= np.min(im_cell)
            im_cell /= np.max(im_cell)
            im_cell = plt.get_cmap(cmap)(im_cell)
            im_cell = (im_cell * 255).astype(np.uint8)

            return QtGui.QPixmap(array2qimage(im_cell))

        if len(self.list.selectedItems()):
            self.sigma_tensor_button.setDisabled(False)
            data = self.data[self.list.currentRow()][2]
            attr = data[3]
            if self.last_cell == self.list.currentRow():
                attr["segmention_thres"] = self.segmention_thres_indi.value()
                attr["seg_gaus1"] = self.seg_gaus1_indi.value()
                attr["seg_gaus2"] = self.seg_gaus2_indi.value()
            else:
                self.segmention_thres_indi.setValue(attr["segmention_thres"])
                self.seg_gaus1_indi.setValue(attr["seg_gaus1"])
                self.seg_gaus2_indi.setValue(attr["seg_gaus2"])
                # print("->", [v is None for v in attr.values()])
                if np.all([v is None for v in attr.values()]):
                    self.individual_data.setValue(False)
                else:
                    self.individual_data.setValue(True)
            self.last_cell = self.list.currentRow()
            im_cell = imageio.v2.imread(data[1])
            from CompactionAnalyzer.CompactionFunctions import segment_cell, normalize
            im_cell = normalize(im_cell, 1, 99)

            self.pixmap.setPixmap(get_pixmap(im_cell))
            self.label.setExtend(im_cell.shape[1], im_cell.shape[0])

            im_fiber = imageio.v2.imread(data[0])
            im_fiber = normalize(im_fiber, 1, 99)
            self.pixmap2.setPixmap(get_pixmap(im_fiber))
            self.label2.setExtend(im_fiber.shape[1], im_fiber.shape[0])

            result = segment_cell(im_cell,
                                  thres=self.segmention_thres.value() if attr["segmention_thres"] is None else attr[
                                      "segmention_thres"],
                                  seg_gaus1=self.seg_gaus1.value() if attr["seg_gaus1"] is None else attr["seg_gaus1"],
                                  seg_gaus2=self.seg_gaus2.value() if attr["seg_gaus2"] is None else attr["seg_gaus2"])
            mask = result["mask"]
            from skimage import measure
            # Find contours at a constant value of 0.8
            contours = measure.find_contours(mask, 0.5)

            path = QtGui.QPainterPath()
            for c in contours:
                path.moveTo(c[0][1], c[0][0])
                for cc in c:
                    path.lineTo(cc[1], cc[0])
            self.contour.setPath(path)
            self.contour2.setPath(path)

            self.label_text.setText(data[2])

            self.link_views()

    def link_views(self):

        def changes1(*args):
            self.label2.setOriginScale(self.label.getOriginScale() * self.label.view_rect[0] / self.label2.view_rect[0])
            start_x, start_y, end_x, end_y = self.label.GetExtend()
            center_x, center_y = start_x + (end_x - start_x) / 2, start_y + (end_y - start_y) / 2
            center_x = center_x / self.label.view_rect[0] * self.label2.view_rect[0]
            center_y = center_y / self.label.view_rect[1] * self.label2.view_rect[1]
            self.label2.centerOn(center_x, center_y)

        def zoomEvent(scale, pos):
            changes1()

        self.label.zoomEvent = zoomEvent
        self.label.panEvent = changes1

        def changes2(*args):
            self.label.setOriginScale(self.label2.getOriginScale() * self.label2.view_rect[0] / self.label.view_rect[0])
            start_x, start_y, end_x, end_y = self.label2.GetExtend()
            center_x, center_y = start_x + (end_x - start_x) / 2, start_y + (end_y - start_y) / 2
            center_x = center_x / self.label2.view_rect[0] * self.label.view_rect[0]
            center_y = center_y / self.label2.view_rect[1] * self.label.view_rect[1]
            self.label.centerOn(center_x, center_y)

        def zoomEvent(scale, pos):
            changes2()

        self.label2.zoomEvent = zoomEvent
        self.label2.panEvent = changes2
        changes2()

    def run(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self.run_thread, daemon=True)
            self.thread.start()
            self.setState(True)
        else:
            kill_thread(self.thread)
            self.thread = None
            self.setState(False)

    def setState(self, running):
        if running:
            self.button_run.setText("stop")
            for widget in self.input_list:
                widget.setDisabled(True)
        else:
            self.button_run.setText("run")
            for widget in self.input_list:
                widget.setDisabled(False)

    def finished(self):
        self.thread = None
        self.setState(False)

    def progress_callback(self, i, n, ii, nn):
        self.progress1.setRange(0, n)
        self.progress1.setValue(i)
        # self.progress2.setRange(0, nn-1)
        # self.progress2.setValue(ii)
        self.list.setCurrentRow(i)

    def measurement_evaluated(self, index, state):
        if state == 1:
            self.list.item(index).setIcon(qta.icon("fa5s.check", options=[dict(color="darkgreen")]))
        elif state == -1:
            self.list.item(index).setIcon(qta.icon("fa5s.times", options=[dict(color="red")]))
        else:
            self.list.item(index).setIcon(qta.icon("fa5.circle", options=[dict(color="white")]))

    def run_thread(self):
        try:
            print("compute orientations")
            n = len([1 for d in self.data if d[1]])
            counter = 0
            self.progress_signal.emit(0, n, 0, 1)
            for i in range(n):
                try:
                    if not self.data[i][1]:
                        continue

                    fiber, cell, output, attr = self.data[i][2]

                    sigma_tensor = self.sigma_tensor.value()
                    if self.sigma_tensor_type.value() == "um":
                        sigma_tensor /= self.scale.value()
                    shell_width = self.shell_width.value()
                    if self.shell_width_type.value() == "um":
                        shell_width /= self.scale.value()

                    from CompactionAnalyzer.CompactionFunctions import StuctureAnalysisMain
                    StuctureAnalysisMain(fiber_list=[fiber],
                                         cell_list=[cell],
                                         out_list=[output],
                                         scale=self.scale.value(),
                                         sigma_tensor=sigma_tensor,
                                         edge=self.edge.value(),
                                         max_dist=self.max_dist.value(),
                                         segmention_thres=self.segmention_thres.value() if attr[
                                                                                               "segmention_thres"] is None else
                                         attr["segmention_thres"],
                                         seg_gaus1=self.seg_gaus1.value() if attr["seg_gaus1"] is None else attr[
                                             "seg_gaus1"],
                                         seg_gaus2=self.seg_gaus2.value() if attr["seg_gaus2"] is None else attr[
                                             "seg_gaus2"],
                                         sigma_first_blur=self.sigma_first_blur.value(),
                                         angle_sections=self.angle_sections.value(),
                                         shell_width=shell_width,
                                         )

                    self.measurement_evaluated_signal.emit(i, 1)
                except Exception as err:
                    import traceback
                    traceback.print_exc()
                    self.measurement_evaluated_signal.emit(i, -1)
                counter += 1
                self.progress_signal.emit(counter, n, 0, 1)
        finally:
            self.finished_signal.emit()

