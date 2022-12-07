import sys
import os
import qtawesome as qta
from qtpy import QtCore, QtWidgets, QtGui
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

import imageio
from qimage2ndarray import array2qimage
import matplotlib.pyplot as plt
import glob
import imageio
import threading
import inspect

import natsort

from pathlib import Path
import re
import pandas as pd
import matplotlib as mpl

import saenopy
import saenopy.multigridHelper
from saenopy.gui import QtShortCuts, QExtendedGraphicsView
from saenopy.gui.gui_classes import Spoiler, CheckAbleGroup, QHLine, QVLine, MatplotlibWidget, NavigationToolbar, execute, kill_thread, ListWidget, QProcess, ProcessSimple
import saenopy.getDeformations
import saenopy.multigridHelper
import saenopy.materials
from saenopy.gui.stack_selector import StackSelector
from saenopy.getDeformations import getStack, Stack, format_glob
from saenopy.multigridHelper import getScaledMesh, getNodesWithOneFace
from saenopy.loadHelpers import Saveable
from saenopy.solver import Result

from typing import List, Tuple

"""REFERENCE FOLDERS"""
#\\131.188.117.96\biophysDS2\dboehringer\Platte_4\SoftwareWorkinProgess\TFM-Example-Data-3D\a127-tom-test-set\20170914_A172_rep1-bispos3\Before
#\\131.188.117.96\biophysDS\lbischof\tif_and_analysis_backup\2021-06-02-NK92-Blebb-Rock\Blebb-round1\Mark_and_Find_001
from .DeformationDetector import DeformationDetector
from .FittedMesh import FittedMesh
from .MeshCreator import MeshCreator
from .Regularizer import Regularizer
from .ResultView import ResultView
from .StackDisplay import StackDisplay


class BatchEvaluate(QtWidgets.QWidget):
    result_changed = QtCore.Signal(object)
    tab_changed = QtCore.Signal(object)
    set_current_result = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.settings = QtCore.QSettings("Saenopy", "Seanopy_deformation")

        with QtShortCuts.QHBoxLayout(self) as main_layout:
            main_layout.setContentsMargins(0, 0, 0, 0)
            with QtShortCuts.QSplitter() as lay:
                with QtShortCuts.QVBoxLayout() as layout:
                    layout.setContentsMargins(0, 0, 0, 0)
                    self.list = ListWidget(layout, add_item_button="add measurements")
                    self.list.addItemClicked.connect(self.show_files)
                    self.list.itemSelectionChanged.connect(self.listSelected)
                    self.progressbar = QProgressBar().addToLayout()
                #           self.label = QtWidgets.QLabel(
                #               "Select a path as an input wildcard. Use * to specify a placeholder. All paths that match the wildcard will be added.").addToLayout()
                with QtShortCuts.QHBoxLayout() as layout:
                    layout.setContentsMargins(0, 0, 0, 0)
                    with QtShortCuts.QTabWidget(layout) as self.tabs:
                        self.tabs.setMinimumWidth(500)
                        old_tab = None
                        cam_pos = None
                        def tab_changed(x):
                            nonlocal old_tab, cam_pos
                            tab = self.tabs.currentWidget()
                            if old_tab is not None and getattr(old_tab, "plotter", None):
                                cam_pos = old_tab.plotter.camera_position
                            if cam_pos is not None and getattr(tab, "plotter", None):
                                tab.plotter.camera_position = cam_pos
                            if old_tab is not None:
                                tab.t_slider.setValue(old_tab.t_slider.value())
                            old_tab = tab
                            self.tab_changed.emit(tab)
                        self.tabs.currentChanged.connect(tab_changed)
                        pass
                with QtShortCuts.QVBoxLayout() as layout0:
                    layout0.parent().setMaximumWidth(420)
                    layout0.setContentsMargins(0, 0, 0, 0)
                    self.sub_module_stacks = StackDisplay(self, layout0)
                    self.sub_module_deformation = DeformationDetector(self, layout0)
                    self.sub_module_mesh = MeshCreator(self, layout0)
                    self.sub_module_fitted_mesh = FittedMesh(self, layout0)
                    self.sub_module_regularize = Regularizer(self, layout0)
                    self.sub_module_view = ResultView(self, layout0)
                    layout0.addStretch()
                    self.button_start_all = QtShortCuts.QPushButton(None, "run all", self.run_all)
                    self.button_code = QtShortCuts.QPushButton(None, "export code", self.generate_code)

        self.data = []
        self.list.setData(self.data)

        # self.list.addData("foo", True, [], mpl.colors.to_hex(f"C0"))

        # data = Result.load(r"..\test\TestData\output3\Mark_and_Find_001_Pos001_S001_z_ch00.npz")
        # self.list.addData("test", True, data, mpl.colors.to_hex(f"gray"))

        self.setAcceptDrops(True)

        self.tasks = []
        self.current_task_id = 0
        self.thread = None
        self.signal_task_finished.connect(self.run_finished)

    def progress(self, tup):
        n, total = tup
        self.progressbar.setMaximum(total)
        self.progressbar.setValue(n)

    def generate_code(self):
        new_path = QtWidgets.QFileDialog.getSaveFileName(None, "Save Session as Script", os.getcwd(), "Python File (*.py)")
        if new_path:
            if isinstance(new_path, tuple):
                new_path = new_path[0]
            else:
                new_path = str(new_path)

            import_code = ""
            run_code = ""
            for module in [self.sub_module_stacks, self.sub_module_deformation, self.sub_module_mesh, self.sub_module_regularize]:
                code1, code2 = module.get_code()
                import_code += code1
                run_code += code2 +"\n"
            run_code = import_code + "\n\n" + run_code
            print(run_code)
            with open(new_path, "w") as fp:
                fp.write(run_code)

    def run_all(self):
        for i in range(len(self.data)):
            if not self.data[i][1]:
                continue
            result = self.data[i][2]
            if self.sub_module_deformation.group.value() is True:
                self.sub_module_deformation.start_process(result=result)
            if self.sub_module_mesh.group.value() is True:
                self.sub_module_mesh.start_process(result=result)
            if self.sub_module_regularize.group.value() is True:
                self.sub_module_regularize.start_process(result=result)

    def addTask(self, task, result, params, name):
        print("add task", task, result, params, name)
        self.tasks.append([task, result, params, name])
        if self.thread is None:
            self.run_next()

    signal_task_finished = QtCore.Signal()

    def run_next(self):
        task, result, params, name = self.tasks[self.current_task_id]
        self.thread = threading.Thread(target=self.run_thread, args=(task, result, params, name))
        self.thread.start()

    def run_thread(self, task, result, params, name):
        result.state = True
        self.update_icons()
        task(result, params)
        self.signal_task_finished.emit()
        result.state = False
        self.update_icons()

    def run_finished(self):
        self.current_task_id += 1
        self.thread = None
        if self.current_task_id < len(self.tasks):
            self.run_next()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        # accept url lists (files by drag and drop)
        for url in event.mimeData().urls():
            # if str(url.toString()).strip().endswith(".npz"):
            event.accept()
            return
        event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        event.acceptProposedAction()

    def dropEvent(self, event: QtCore.QEvent):
        for url in event.mimeData().urls():

            url = url.toLocalFile()  # path()

            if url[0] == "/" and url[2] == ":":
                url = url[1:]
            if url.endswith(".npz"):
                urls = [url]
            else:
                urls = glob.glob(url +"/**/*.npz", recursive=True)
            for url in urls:
                data = Result.load(url)
                self.list.addData(data.output, True, data, mpl.colors.to_hex(f"gray"))
                # app.processEvents()
        self.update_icons()

    def update_icons(self):
        for j in range(self.list.count( ) -1):
            if self.data[j][2].state is True:
                self.list.item(j).setIcon(qta.icon("fa5s.hourglass-half", options=[dict(color="orange")]))
            else:
                self.list.item(j).setIcon(qta.icon("fa5.circle", options=[dict(color="gray")]))

    def show_files(self):
        settings = self.settings

        class AddFilesDialog(QtWidgets.QDialog):
            def __init__(self, parent):
                super().__init__(parent)
                self.setMinimumWidth(800)
                self.setMinimumHeight(600)
                self.setWindowTitle("Add Files")
                with QtShortCuts.QVBoxLayout(self) as layout:
                    with QtShortCuts.QTabWidget(layout) as self.tabs:
                        with self.tabs.createTab("Pair Stacks") as self.tab:
                            self.outputText = QtShortCuts.QInputFolder(None, "output", settings=settings,
                                                                       settings_key="batch/wildcard2", allow_edit=True)
                            with QtShortCuts.QHBoxLayout() as layout3:
                                self.stack_relaxed = StackSelector(layout3, "relaxed")
                                self.stack_relaxed.glob_string_changed.connect \
                                    (lambda x, y: (print("relaxed, y"), self.input_relaxed.setText(y)))
                                self.stack_deformed = StackSelector(layout3, "deformed", self.stack_relaxed)
                                self.stack_deformed.glob_string_changed.connect \
                                    (lambda x, y: (print("deformed, y") ,self.input_deformed.setText(y)))
                            with QtShortCuts.QHBoxLayout() as layout3:
                                self.input_relaxed = QtWidgets.QLineEdit().addToLayout()
                                self.input_deformed = QtWidgets.QLineEdit().addToLayout()
                            with QtShortCuts.QHBoxLayout() as layout3:
                                # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                                layout3.addStretch()
                                self.button_addList2 = QtShortCuts.QPushButton(None, "cancel", self.reject)
                                def accept():
                                    self.mode = "pair"
                                    if not self.stack_relaxed.validator():
                                        QtWidgets.QMessageBox.critical(self, "Deformation Detector", "Enter a valid voxel size for the relaxed stack.")
                                        return
                                    if not self.stack_deformed.validator():
                                        QtWidgets.QMessageBox.critical(self, "Deformation Detector", "Enter a valid voxel size for the deformed stack.")
                                        return
                                    self.accept()
                                self.button_addList1 = QtShortCuts.QPushButton(None, "ok", accept)
                        with self.tabs.createTab("Time Stacks") as self.tab2:
                            self.outputText2 = QtShortCuts.QInputFolder(None, "output", settings=settings,
                                                                        settings_key="batch/wildcard2", allow_edit=True)
                            with QtShortCuts.QHBoxLayout() as layout3:
                                self.stack_before2 = StackSelector(layout3, "time series", use_time=True)
                                self.stack_before2.glob_string_changed.connect \
                                    (lambda x, y: self.input_relaxed2.setText(y))
                            with QtShortCuts.QHBoxLayout() as layout3:
                                self.input_relaxed2 = QtWidgets.QLineEdit().addToLayout()
                            with QtShortCuts.QHBoxLayout() as layout3:
                                # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                                layout3.addStretch()
                                self.button_addList3 = QtShortCuts.QPushButton(None, "cancel", self.reject)
                                def accept():
                                    if not self.stack_before2.validator():
                                        QtWidgets.QMessageBox.critical(self, "Deformation Detector", "Enter a valid voxel size for the stack.")
                                        return
                                    if not self.stack_before2.validator_time():
                                        QtWidgets.QMessageBox.critical(self, "Deformation Detector",
                                                                       "Enter a valid time delta.")
                                        return

                                    self.mode = "time"
                                    self.accept()
                                self.button_addList4 = QtShortCuts.QPushButton(None, "ok", accept)

                        with self.tabs.createTab("Existing Files") as self.tab3:
                            self.outputText3 = QtShortCuts.QInputFilename(None, "output", settings=settings, file_type="Results Files (*.npz)",
                                                                          settings_key="batch/wildcard_existing", allow_edit=True, existing=True)
                            with QtShortCuts.QHBoxLayout() as layout3:
                                # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                                layout3.addStretch()
                                self.button_addList6 = QtShortCuts.QPushButton(None, "cancel", self.reject)
                                def accept():
                                    self.mode = "existing"
                                    self.accept()
                                self.button_addList5 = QtShortCuts.QPushButton(None, "ok", accept)

                        with self.tabs.createTab("Examples") as self.tab4:
                            def loadexample1():
                                import appdirs
                                saenopy.loadExample("ClassicSingleCellTFM", appdirs.user_data_dir("saenopy", "rgerum"))
                                self.mode = "example"
                                self.mode_data = 1
                                self.accept()
                            with QtShortCuts.QGroupBox(None, "ClassicSingleCellTFM"):
                                QtWidgets.QLabel \
                                    ("This example evaluates three hepatic stellate cells in 1.2mg/ml collagen with relaxed and deformed stacks.\nThe relaxed stacks were recorded with cytochalasin D treatment of the cells.").addToLayout()
                                self.button_example1 = QtShortCuts.QPushButton(None, "Open", loadexample1)

                            def loadexample2():
                                import appdirs
                                saenopy.loadExample("DynamicalSingleCellTFM", appdirs.user_data_dir("saenopy", "rgerum"))
                                self.mode = "example"
                                self.mode_data = 2
                                self.accept()

                            with QtShortCuts.QGroupBox(None, "DynamicalSingleCellTFM"):
                                QtWidgets.QLabel(
                                    "This example evaluates a single natural killer cell that migrated through 1.2mg/ml collagen, recorded for 24min.").addToLayout()

                                self.button_example2 = QtShortCuts.QPushButton(None, "Open", loadexample2)
                            self.tab4.addStretch()


        class FileExistsDialog(QtWidgets.QDialog):
            def __init__(self, parent, filename):
                super().__init__(parent)
                # self.setMinimumWidth(800)
                # self.setMinimumHeight(600)
                self.setWindowTitle("File Exists")
                with QtShortCuts.QVBoxLayout(self) as layout:
                    self.label = QtWidgets.QLabel(f"A file with the name {filename} already exists.").addToLayout()
                    with QtShortCuts.QHBoxLayout() as layout3:
                        layout3.addStretch()
                        self.use_for_all = QtShortCuts.QInputBool(None, "remember decision for all files", False)
                    with QtShortCuts.QHBoxLayout() as layout3:
                        # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                        layout3.addStretch()
                        self.button_addList2 = QtShortCuts.QPushButton(None, "cancel", self.reject)
                        def accept():
                            self.mode = "overwrite"
                            self.accept()
                        self.button_addList1 = QtShortCuts.QPushButton(None, "overwrite", accept)
                        def accept2():
                            self.mode = "read"
                            self.accept()
                        self.button_addList1 = QtShortCuts.QPushButton(None, "read", accept2)

        last_decision = None
        def do_overwrite(filename):
            nonlocal last_decision
            if last_decision is not None:
                return last_decision
            dialog = FileExistsDialog(self, filename)
            result = dialog.exec()
            if not result:
                return 0
            if dialog.use_for_all.value():
                last_decision = dialog.mode
            return dialog.mode

        # getStack
        dialog = AddFilesDialog(self)
        if not dialog.exec():
            return

        from saenopy.solver import get_stacks
        if dialog.mode == "pair":
            results = get_stacks(
                [dialog.input_relaxed.text(), dialog.input_deformed.text()],
                output_path=dialog.outputText.value(),
                voxel_size=dialog.stack_relaxed.getVoxelSize(),
                exist_overwrite_callback=do_overwrite,
            )
            for data in results:
                self.list.addData(data.output, True, data, mpl.colors.to_hex(f"gray"))
        elif dialog.mode == "time":
            results = get_stacks(
                dialog.input_relaxed2.text(),
                output_path=dialog.outputText2.value(),
                voxel_size=dialog.stack_before2.getVoxelSize(),
                time_delta=dialog.stack_before2.getTimeDelta(),
                exist_overwrite_callback=do_overwrite,
            )
            for data in results:
                self.list.addData(data.output, True, data, mpl.colors.to_hex(f"gray"))
        elif dialog.mode == "existing":
            for file in glob.glob(dialog.outputText3.value(), recursive=True):
                data = Result.load(file)
                self.list.addData(data.output, True, data, mpl.colors.to_hex(f"gray"))
        elif dialog.mode == "example":
            import appdirs
            if dialog.mode_data == 1:
                results = get_stacks(
                    [str(Path(appdirs.user_data_dir("saenopy", "rgerum")) / '1_ClassicSingleCellTFM/Relaxed/Mark_and_Find_001/Pos*_S001_z{z}_ch00.tif'),
                     str(Path(appdirs.user_data_dir("saenopy", "rgerum")) / '1_ClassicSingleCellTFM/Deformed/Mark_and_Find_001/Pos*_S001_z{z}_ch00.tif'),
                     ],
                    output_path=str
                        (Path(appdirs.user_data_dir("saenopy", "rgerum")) / '1_ClassicSingleCellTFM/example_output'),
                    voxel_size=[0.7211, 0.7211, 0.988],
                    exist_overwrite_callback=do_overwrite,
                )
                for data in results:
                    self.list.addData(data.output, True, data, mpl.colors.to_hex(f"gray"))
            if dialog.mode_data == 2:
                results = get_stacks(
                    str(Path(appdirs.user_data_dir("saenopy", "rgerum")) / '2_DynamicalSingleCellTFM/data/Pos*_S001_t{t}_z{z}_ch00.tif'),
                    output_path=str
                        (Path(appdirs.user_data_dir("saenopy", "rgerum")) / '2_DynamicalSingleCellTFM/example_output'),
                    voxel_size=[0.2407, 0.2407, 1.0071],
                    time_delta=60,
                    exist_overwrite_callback=do_overwrite,
                )
                for data in results:
                    self.list.addData(data.output, True, data, mpl.colors.to_hex(f"gray"))

        self.update_icons()
        # import matplotlib as mpl
        # for fiber, cell, out in zip(fiber_list, cell_list, out_list):
        #    self.list.addData(fiber, True, [fiber, cell, out, {"segmention_thres": None, "seg_gaus1": None, "seg_gaus2": None}], mpl.colors.to_hex(f"gray"))

    def listSelected(self):
        if self.list.currentRow() is not None:
            pipe = self.data[self.list.currentRow()][2]
            self.set_current_result.emit(pipe)




class QProgressBar(QtWidgets.QProgressBar):
    signal_start = QtCore.Signal(int)
    signal_progress = QtCore.Signal(int)

    def __init__(self, layout=None):
        super().__init__()
        self.setOrientation(QtCore.Qt.Horizontal)
        if layout is not None:
            layout.addWidget(self)
        self.signal_start.connect(lambda i: self.setRange(0, i))
        self.signal_progress.connect(lambda i: self.setValue(i))

    def iterator(self, iter):
        print("iterator", iter)
        self.signal_start.emit(len(iter))
        for i, v in enumerate(iter):
            yield i
            print("emit", i)
            self.signal_progress.emit(i+1)




