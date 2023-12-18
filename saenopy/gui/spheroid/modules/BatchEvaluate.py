import sys

# Setting the Qt bindings for QtPy
import os

import pandas as pd
import qtawesome as qta

os.environ["QT_API"] = "pyqt5"

from qtpy import QtCore, QtWidgets, QtGui

import numpy as np
from natsort import natsorted


from saenopy.gui.common import QtShortCuts, QExtendedGraphicsView
from qimage2ndarray import array2qimage
import matplotlib.pyplot as plt
import imageio
import threading
import glob


from matplotlib.figure import Figure
import jointforces as jf
import urllib
from pathlib import Path

import ctypes

from .ListWidget import ListWidget
from .QSlider import QSlider
from .AddFilesDialog import AddFilesDialog
from .helper import kill_thread, execute
from saenopy.examples import get_examples_spheriod
from saenopy.gui.spheroid.modules.LookupTable import SelectLookup
from saenopy.gui.common.gui_classes import QVLine, QHLine, Spoiler, CheckAbleGroup

class BatchEvaluate(QtWidgets.QWidget):
    progress_signal = QtCore.Signal(int, int, int, int)
    measurement_evaluated_signal = QtCore.Signal(int, int)
    finished_signal = QtCore.Signal()
    thread = None

    def __init__(self, parent=None):
        super().__init__(parent)

        # QSettings
        self.settings = QtCore.QSettings("Saenopy", "Saenopy")

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
                    self.slider = QSlider().addToLayout()
                    self.slider.setRange(0, 0)
                    self.slider.valueChanged.connect(self.slider_changed)
                    self.slider.setOrientation(QtCore.Qt.Horizontal)
                    # layout.addWidget(self.slider)

                    self.label_text = QtWidgets.QLabel()
                    layout.addWidget(self.label_text)

                    self.label = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
                    self.label.setMinimumWidth(300)
                    self.pixmap = QtWidgets.QGraphicsPixmapItem(self.label.origin)
                    self.contour = QtWidgets.QGraphicsPathItem(self.label.origin)
                    pen = QtGui.QPen(QtGui.QColor(255, 0, 0))
                    pen.setCosmetic(True)
                    self.contour.setPen(pen)

                    self.label2 = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
                    # self.label2.setMinimumWidth(300)
                    self.pixmap2 = QtWidgets.QGraphicsPixmapItem(self.label2.origin)

                    self.label_text2 = QtWidgets.QLabel().addToLayout()
                    self.progress2 = QtWidgets.QProgressBar().addToLayout()

                frame = QtWidgets.QFrame().addToLayout()
                frame.setMaximumWidth(300)
                with QtShortCuts.QVBoxLayout(frame) as layout:
                    with CheckAbleGroup(self, "Detect Deformations").addToLayout() as self.deformation_data:
                        with QtShortCuts.QVBoxLayout() as layout2:
                            self.window_size = QtShortCuts.QInputNumber(layout2, "window size", 50,
                                                                        float=False, name_post='px',
                                                                        settings=self.settings,
                                                                        settings_key="spheriod/deformation/window_size")

                            QHLine().addToLayout()
                            with QtShortCuts.QHBoxLayout(None):
                                self.n_min = QtShortCuts.QInputString(None, "n_min", "None", allow_none=True, type=int,
                                                                      settings=self.settings,
                                                                      settings_key="spheriod/deformation/n_min")
                                self.n_max = QtShortCuts.QInputString(None, "n_max", "None", allow_none=True, type=int,
                                                                      settings=self.settings, name_post='frames',
                                                                      settings_key="spheriod/deformation/n_max")

                            self.thres_segmentation = QtShortCuts.QInputNumber(None, "segmentation threshold", 0.9,
                                                                               float=True,
                                                                               min=0.2, max=1.5, step=0.1,
                                                                               use_slider=False,
                                                                               settings=self.settings,
                                                                               settings_key="spheriod/deformation/thres_segmentation2")
                            self.thres_segmentation.valueChanged.connect(
                                lambda: self.param_changed("thres_segmentation", True))
                            self.continous_segmentation = QtShortCuts.QInputBool(None, "continous_segmentation", False,
                                                                                 settings=self.settings,
                                                                                 settings_key="spheriod/deformation/continous_segemntation")
                            self.continous_segmentation.valueChanged.connect(
                                lambda: self.param_changed("continous_segmentation", True))
                            self.n_min.valueChanged.connect(lambda: self.param_changed("n_min"))
                            self.n_max.valueChanged.connect(lambda: self.param_changed("n_max"))

                            with CheckAbleGroup(self, "individual segmentation").addToLayout() as self.individual_data:
                                with QtShortCuts.QVBoxLayout() as layout2:
                                    self.segmention_thres_indi = QtShortCuts.QInputString(None, "segmention threshold",
                                                                                          None,
                                                                                          type=float, allow_none=True)
                                    self.segmention_thres_indi.valueChanged.connect(self.listSelected)

                            self.individual_data.value_changed.connect(self.changedCheckBox)

                    # QHLine().addToLayout()
                    if 1:
                        with CheckAbleGroup(self, "Plot").addToLayout() as self.plot_data:
                            with QtShortCuts.QVBoxLayout() as layout2:
                                with QtShortCuts.QHBoxLayout() as layout2:
                                    self.color_norm = QtShortCuts.QInputString(None, "color norm", 75., type=float,
                                                                               settings=self.settings, name_post='µm',
                                                                               settings_key="spheriod/deformation/color_norm")

                                    self.cbar_um_scale = QtShortCuts.QInputString(None, "pixel_size", None,
                                                                                  allow_none=True,
                                                                                  type=float, settings=self.settings,
                                                                                  name_post='µm/px',
                                                                                  settings_key="spheriod/deformation/cbar_um_scale")
                                with QtShortCuts.QHBoxLayout() as layout2:
                                    self.quiver_scale = QtShortCuts.QInputString(None, "quiver_scale", 1, type=float,
                                                                                 settings=self.settings,
                                                                                 name_post='a.u.',
                                                                                 settings_key="spheriod/deformation/quiver_scale")

                                with QtShortCuts.QHBoxLayout() as layout2:
                                    self.dpi = QtShortCuts.QInputString(None, "dpi", 150, allow_none=True, type=int,
                                                                        settings=self.settings,
                                                                        settings_key="spheriod/deformation/dpi")
                                    self.dt_min = QtShortCuts.QInputString(None, "dt", None, allow_none=True,
                                                                           type=float,
                                                                           settings=self.settings, name_post='min',
                                                                           settings_key="spheriod/deformation/dt_min")

                    with CheckAbleGroup(self, "Calculate Forces").addToLayout() as self.force_data:
                        with QtShortCuts.QVBoxLayout():
                            with QtShortCuts.QHBoxLayout():
                                self.lookup_table = QtShortCuts.QInputString(None, "Lookup Table")
                                self.lookup_table.line_edit.setDisabled(True)
                                self.button_lookup = QtShortCuts.QPushButton(None, "choose file", self.choose_lookup)
                            # self.output = QtShortCuts.QInputFolder(None, "Result Folder")
                            # self.lookup_table = QtShortCuts.QInputFilename(None, "Lookup Table", 'lookup_example.pkl',
                            #                                               file_type="Pickle Lookup Table (*.pkl)",
                            #                                               existing=True)

                            self.pixel_size = QtShortCuts.QInputString(None, "pixel_size", "1.29", name_post='µm/px',
                                                                       type=float)

                            with QtShortCuts.QHBoxLayout():
                                self.x0 = QtShortCuts.QInputString(None, "r_min", "2", type=float)
                                self.x1 = QtShortCuts.QInputString(None, "r_max", "None", type=float,
                                                                   name_post='spheriod radii', allow_none=True)

                    layout.addStretch()

                    self.button_run = QtShortCuts.QPushButton(None, "run", self.run)
        self.images = []
        self.data = []
        self.list.setData(self.data)

        self.input_list = [
            # self.inputText,
            # self.outputText,
            # self.button_clear,
            # self.button_addList,
            self.force_data,
            self.deformation_data,
            self.plot_data,
        ]

        self.progress_signal.connect(self.progress_callback)
        self.measurement_evaluated_signal.connect(self.measurement_evaluated)
        self.finished_signal.connect(self.finished)

    def changedCheckBox(self):
        for widget in [self.thres_segmentation]:
            widget.setDisabled(self.individual_data.value())
        if not self.individual_data.value():
            for widget in [self.segmention_thres_indi]:
                widget.setValue("None")

    def choose_lookup(self):

        self.lookup_gui = SelectLookup()
        self.lookup_gui.exec()

        if self.lookup_gui.result is not None:
            self.lookup_table.setValue(self.lookup_gui.result)

    def show_files(self):
        settings = self.settings



        dialog = AddFilesDialog(self, settings)
        if not dialog.exec():
            return

        # create a new measurement object
        if dialog.mode == "new":
            input_path = dialog.inputText.value()
            output_path = dialog.outputText.value()
        elif dialog.mode == "example":
            # get the date from the example referenced by name
            example = get_examples_spheriod()[dialog.mode_data]
            input_path = example["input"]
            output_path = example["output_path"]

        import glob
        import re
        text = os.path.normpath(input_path)
        glob_string = text.replace("?", "*")
        # print("globbing", glob_string)
        files = natsorted(glob.glob(glob_string))

        output_base = glob_string
        while "*" in str(output_base):
            output_base = Path(output_base).parent

        regex_string = re.escape(text).replace(r"\*", "(.*)").replace(r"\?", ".*")

        data = {}
        for file in files:
            file = os.path.normpath(file)
            print(file, regex_string)
            match = re.match(regex_string, file).groups()
            reconstructed_file = regex_string
            for element in match:
                reconstructed_file = reconstructed_file.replace("(.*)", element, 1)
            reconstructed_file = reconstructed_file.replace(".*", "*")
            reconstructed_file = re.sub(r'\\(.)', r'\1', reconstructed_file)

            if reconstructed_file not in data:
                output = Path(output_path) / os.path.relpath(file, output_base)
                output = output.parent / output.stem
                data[reconstructed_file] = dict(
                    images=[],
                    output=output,
                    thres_segmentation=None,
                    continous_segmentation=None,
                    custom_mask=None,
                    n_min=None,
                    n_max=None,
                )
            data[reconstructed_file]["images"].append(file)
            # if len(data[reconstructed_file]["images"]) > 4:
            #    data[reconstructed_file]["images"] = data[reconstructed_file]["images"][:4]
        # data.update(self.data)
        # self.data = data
        # self.list.clear()
        # self.list.addItems(list(data.keys()))
        import matplotlib as mpl
        for reconstructed_file, d in data.items():
            self.list.addData(reconstructed_file, True, d, mpl.colors.to_hex(f"gray"))

    def clear_files(self):
        self.list.clear()
        self.data = {}

    last_cell = None

    def listSelected(self):
        if len(self.list.selectedItems()):
            data = self.data[self.list.currentRow()][2]

            attr = data  # [3]
            if self.last_cell == self.list.currentRow():
                attr["thres_segmentation"] = self.segmention_thres_indi.value()
                # attr["seg_gaus1"] = self.seg_gaus1_indi.value()
                # attr["seg_gaus2"] = self.seg_gaus2_indi.value()
            else:
                self.segmention_thres_indi.setValue(attr["thres_segmentation"])
                # self.seg_gaus1_indi.setValue(attr["seg_gaus1"])
                # self.seg_gaus2_indi.setValue(attr["seg_gaus2"])
                # print("->", [attr[v] is None for v in ["thres_segmentation"]])
                if np.all([attr[v] is None for v in ["thres_segmentation"]]):
                    self.individual_data.setValue(False)
                else:
                    self.individual_data.setValue(True)
            self.last_cell = self.list.currentRow()
            self.images = data["images"]
            self.last_image = None
            self.last_seg = None
            # for name in ["thres_segmentation", "continous_segmentation", "n_min", "n_max"]:
            #    if data[name] is not None:
            #        getattr(self, name).setValue(data[name])
            self.slider.setRange(0, len(self.images) - 1)
            self.slider_changed(self.slider.value())
            self.label_text2.setText(str(data["output"]))
            self.slider.min = self.n_min.value()
            self.slider.max = self.n_max.value()
            self.slider.update()

    def param_changed(self, name, update_image=False):
        if len(self.list.selectedItems()):
            data = self.data[self.list.currentRow()][2]
            data[name] = getattr(self, name).value()
            if update_image:
                self.slider_changed(self.slider.value())
            self.slider.min = self.n_min.value()
            self.slider.max = self.n_max.value()
            self.slider.update()

    last_image = None
    last_seg = None

    def slider_changed(self, i):
        data = self.data[self.list.currentRow()][2]

        thres_segmentation = self.thres_segmentation.value() if data["thres_segmentation"] is None else data[
            "thres_segmentation"]

        if self.last_image is not None and self.last_image[0] == i:
            i, im, im0 = self.last_image
            # print("cached")
        else:
            im = imageio.v2.imread(self.images[i]).astype(float)
            if self.continous_segmentation.value() is True:
                im0 = im
            else:
                im0 = imageio.v2.imread(self.images[0]).astype(float)
            self.last_image = [i, im, im0]

        if self.last_seg is not None and \
                self.last_seg[1] == thres_segmentation and \
                self.continous_segmentation.value() is False:
            pass
            # print("cached")
        else:
            print(self.last_seg, i, thres_segmentation)
            seg0 = jf.piv.segment_spheroid(im0, True, thres_segmentation)
            from skimage import measure
            # Find contours at a constant value of 0.8
            contours = measure.find_contours(seg0["mask"], 0.5)

            path = QtGui.QPainterPath()
            for c in contours:
                path.moveTo(c[0][1], im.shape[0] - c[0][0])
                for cc in c:
                    path.lineTo(cc[1], im.shape[0] - cc[0])
            self.contour.setPath(path)
            self.last_seg = [i, thres_segmentation, seg0]
        im = im - im.min()
        im = (im / im.max() * 255).astype(np.uint8)

        self.pixmap.setPixmap(QtGui.QPixmap(array2qimage(im)))
        self.label.setExtend(im.shape[1], im.shape[0])
        self.label_text.setText(f"{i + 1}/{len(self.images)} {self.images[i]}")

        # from jointforces.piv import save_displacement_plot
        # import io
        # buf = io.BytesIO()
        # import time
        # t = time.time()
        # dis_sum = np.load(str(data["output"]) + '/def' + str(i).zfill(6) + '.npy', allow_pickle=True).item()
        # print("loadtime", time.time()-t)

        try:
            # im = imageio.v2.imread(buf)
            im = imageio.v2.imread(str(data["output"]) + '/plot' + str(i).zfill(6) + '.png')
        except FileNotFoundError:
            im = np.zeros(im.shape)
        self.pixmap2.setPixmap(QtGui.QPixmap(array2qimage(im)))
        self.label2.setExtend(im.shape[1], im.shape[0])

        self.line_views()

    def line_views(self):

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
        if self.lookup_table.value() == '' and self.force_data.value() is True:
            QtWidgets.QMessageBox.critical(self, 'Error - Saenopy',
                                           'No lookup table for force reconstruction specified. Either provide one or disable force calculation.',
                                           QtWidgets.QMessageBox.Ok)
            return
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
        self.progress2.setRange(0, nn - 1)
        self.progress2.setValue(ii)
        for j in range(self.list.count()):
            if j < i:
                self.list.item(j).setIcon(qta.icon("fa5s.check", options=[dict(color="darkgreen")]))
            else:
                self.list.item(j).setIcon(qta.icon("fa5.circle", options=[dict(color="white")]))
        self.list.setCurrentRow(i)
        self.slider.setEvaluated(ii)
        self.slider.setValue(ii)
        return
        # when plotting show the slider
        if self.plot.value() is True:
            # set the range for the slider
            self.slider.setRange(1, i)
            # it the slider was at the last value, move it to the new maximum
            if self.slider.value() == i - 1:
                self.slider.setValue(i)

    def run_thread(self):
        try:
            # print("compute displacements")
            n = self.list.count() - 1
            for i in range(n):
                try:
                    if not self.data[i][1]:
                        continue
                    data = self.data[i][2]
                    self.progress_signal.emit(i, n, 0, len(data["images"]))
                    folder, file = os.path.split(self.data[i][0])

                    continous_segmentation = self.continous_segmentation.value()
                    thres_segmentation = data["thres_segmentation"] or self.thres_segmentation.value()

                    # set proper None values if no number set
                    try:
                        n_min = int(self.n_min.value())
                    except:
                        n_min = None
                    try:
                        n_max = int(self.n_max.value())
                    except:
                        n_max = None
                    try:
                        cbar_um_scale = float(self.cbar_um_scale.value())
                    except:
                        cbar_um_scale = None
                    try:
                        r_max = float(self.x1.value())
                    except:
                        r_max = None
                    try:
                        r_min = float(self.x0.value())
                    except:
                        r_min = None

                    if self.deformation_data.value() is True:
                        jf.piv.compute_displacement_series(str(folder),
                                                           str(file),
                                                           str(data["output"]),
                                                           n_max=n_max,
                                                           n_min=n_min,
                                                           plot=self.plot_data.value(),
                                                           # plot=self.plot.value(),
                                                           draw_mask=False,
                                                           color_norm=self.color_norm.value(),
                                                           cbar_um_scale=cbar_um_scale,
                                                           quiver_scale=self.quiver_scale.value(),
                                                           dpi=(self.dpi.value()),
                                                           continous_segmentation=continous_segmentation,
                                                           thres_segmentation=thres_segmentation,
                                                           window_size=(self.window_size.value()),
                                                           dt_min=(self.dt_min.value()),
                                                           cutoff=None, cmap="turbo",
                                                           callback=lambda ii, nn: self.progress_signal.emit(i, n, ii,
                                                                                                             nn))

                    elif self.plot_data.value() is True:
                        images = data["images"]
                        for ii in range(0, len(images)):
                            im = imageio.v2.imread(images[i]).astype(float)
                            if ii == 0 or self.continous_segmentation.value() is True:
                                seg0 = jf.piv.segment_spheroid(im, True, self.thres_segmentation.value())
                            if ii > 0:
                                # print("self.dt_min.value()*ii if self.dt_min.value() is not None else None", self.dt_min.value()*ii if self.dt_min.value() is not None else None)
                                from jointforces.piv import save_displacement_plot
                                dis_sum = np.load(str(data["output"]) + '/def' + str(ii).zfill(6) + '.npy',
                                                  allow_pickle=True).item()
                                save_displacement_plot(str(data["output"]) + '/plot' + str(ii).zfill(6) + '.png', im,
                                                       seg0, dis_sum,
                                                       quiver_scale=(self.quiver_scale.value()),
                                                       color_norm=self.color_norm.value(),
                                                       cbar_um_scale=(self.cbar_um_scale.value()),
                                                       dpi=(self.dpi.value()),
                                                       t=self.dt_min.value() * ii if self.dt_min.value() is not None else None)
                            self.progress_signal.emit(i, n, ii, len(images))

                    if self.force_data.value() is True:
                        jf.force.reconstruct(str(data["output"]),  # PIV output folder
                                             str(self.lookup_table.value()),  # lookup table
                                             self.pixel_size.value(),  # pixel size (µm)
                                             None, r_min=r_min, r_max=r_max)
                except Exception as err:
                    import traceback
                    traceback.print_exc()
                    self.measurement_evaluated_signal.emit(i, -1)
                self.progress_signal.emit(i + 1, n, 0, len(data["images"]))
        finally:
            self.finished_signal.emit()

    def measurement_evaluated(self, index, state):
        if state == 1:
            self.list.item(index).setIcon(qta.icon("fa5s.check", options=[dict(color="darkgreen")]))
        elif state == -1:
            self.list.item(index).setIcon(qta.icon("fa5s.times", options=[dict(color="red")]))
        else:
            self.list.item(index).setIcon(qta.icon("fa5.circle", options=[dict(color="white")]))

