import json
import sys
import os

import qtawesome as qta
from qtpy import QtCore, QtWidgets, QtGui
import numpy as np
import glob
import threading
from pathlib import Path
import matplotlib as mpl

import traceback

from saenopy import get_stacks
from saenopy import Result
from saenopy.gui.common import QtShortCuts
from saenopy.gui.common.gui_classes import ListWidget
from saenopy.gui.common.stack_selector_tif import add_last_voxel_size, add_last_time_delta

from saenopy.gui.common.AddFilesDialog import FileExistsDialog
from saenopy.gui.spheroid.modules.result import ResultSpheroid
from saenopy.gui.tfm2d.modules.result import Result2D


class SharedProperties:
    properties = None

    def __init__(self):
        self.properties = {}

    def add_property(self, name, target):
        if name not in self.properties:
            self.properties[name] = []
        self.properties[name].append(target)

    def change_property(self, name, value, target):
        if name in self.properties:
            for t in self.properties[name]:
                if t != target:
                    t.property_changed(name, value)


class BatchEvaluateBase(QtWidgets.QWidget):
    settings_key = "saenopy"
    result_changed = QtCore.Signal(object)
    tab_changed = QtCore.Signal(object)
    set_current_result = QtCore.Signal(object)

    file_extension = None

    result_params = []

    def add_modules(self):
        self.modules = []

    def add_tabs(self):
        layout = QtShortCuts.currentLayout()
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

    def set_plot_window(self, plot_window):
        self.plot_window = plot_window

    def get_copy_to_menu(self):
        result = self.list.data[self.list.currentRow()][2]
        return self.plot_window.get_copy_to_menu(result.output)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.shared_properties = SharedProperties()

        self.settings = QtCore.QSettings("Saenopy", self.settings_key)

        with QtShortCuts.QHBoxLayout(self) as main_layout:
            main_layout.setContentsMargins(0, 0, 0, 0)
            with QtShortCuts.QSplitter() as lay:
                with QtShortCuts.QVBoxLayout() as layout:
                    layout.setContentsMargins(0, 0, 0, 0)
                    self.list = ListWidget(None, add_item_button="add measurements", copy_params=True, allow_paste_callback=self.allow_paste, copy_to_callback=self.get_copy_to_menu)
                    layout.addWidget(self.list, 2)
                    self.list.addItemClicked.connect(self.add_measurement)
                    self.list.signal_act_copy_clicked.connect(self.copy_params)
                    self.list.signal_act_paste_clicked.connect(self.paste_params)
                    self.list.signal_act_paste2_clicked.connect(self.paste_params2)
                    self.list.signal_act_paths_clicked.connect(self.path_editor)
                    self.list.signal_act_paths2_clicked.connect(self.path_open)
                    self.list.signal_act_paths3_clicked.connect(self.path_copy)
                    self.list.itemSelectionChanged.connect(self.listSelected)

                    self.results_pane = QtWidgets.QVBoxLayout()
                    layout.addLayout(self.results_pane)

                    self.progress_label2 = QtWidgets.QLabel().addToLayout()

                    self.progressbar = QtWidgets.QProgressBar().addToLayout()
                    self.progressbar.setOrientation(QtCore.Qt.Horizontal)

                    self.progress_label = QtWidgets.QLabel().addToLayout()
                with QtShortCuts.QHBoxLayout() as layout:
                    layout.setContentsMargins(0, 0, 0, 0)
                    self.add_tabs()
                with QtShortCuts.QVBoxLayout() as layout0:
                    layout0.parent().setMaximumWidth(420)
                    layout0.setContentsMargins(0, 0, 0, 0)
                    self.add_modules()
                    layout0.addStretch()
                    self.button_start_all = QtShortCuts.QPushButton(None, "run all", self.run_all)
                    with QtShortCuts.QHBoxLayout():
                        self.button_code = QtShortCuts.QPushButton(None, "export code", self.generate_code)
                        if getattr(self, "generate_data", None):
                            self.button_excel = QtShortCuts.QPushButton(None, "export data", self.generate_data)
                        if getattr(self, "sub_module_export", None):
                            self.button_export = QtShortCuts.QPushButton(None, "export images", lambda x: self.sub_module_export.show_window())

        self.data = []
        self.list.setData(self.data)

        self.setAcceptDrops(True)

        self.tasks = []
        self.current_task_id = 0
        self.thread = None
        self.signal_task_finished.connect(self.run_finished)

        # disable all tabs
        for i in range(self.tabs.count()-1, -1, -1):
            self.tabs.setTabEnabled(i, False)

        # load paths
        self.load_from_path([arg for arg in sys.argv if arg.endswith(self.file_extension)])

    def copy_params(self):
        result = self.list.data[self.list.currentRow()][2]
        params = {name: getattr(result, name+"_tmp") for name in self.result_params}
        print(params)
        for group in params:
            if params[group] is None:
                continue
            for g in params[group]:
                if type(params[group][g]) == np.bool_:
                    params[group][g] = bool(params[group][g])
                if type(params[group][g]) == np.int64:
                    params[group][g] = int(params[group][g])
                if type(params[group][g]) == np.int32:
                    params[group][g] = int(params[group][g])
        text = json.dumps(params, indent=2)
        cb = QtGui.QGuiApplication.clipboard()
        cb.setText(text, mode=cb.Clipboard)

    def allow_paste(self):
        cb = QtGui.QGuiApplication.clipboard()
        text = cb.text(mode=cb.Clipboard)
        try:
            data = json.loads(text)
            if all([name in data for name in self.result_params]):
                return True
        except (ValueError, TypeError):
            return False
        return False

    def paste_params(self):
        cb = QtGui.QGuiApplication.clipboard()
        text = cb.text(mode=cb.Clipboard)
        try:
            data = json.loads(text)
        except ValueError:
            return False
        result = self.list.data[self.list.currentRow()][2]
        params = self.result_params
        for par in params:
            if par in data:
                setattr(result, par+"_tmp", data[par])
        self.set_current_result.emit(result)

    def paste_params2(self):
        cb = QtGui.QGuiApplication.clipboard()
        text = cb.text(mode=cb.Clipboard)
        try:
            data = json.loads(text)
        except ValueError:
            return False
        for i in range(len(self.list.data)):
            result = self.list.data[i][2]
            params = self.result_params
            for par in params:
                if par in data:
                    setattr(result, par + "_tmp", data[par])
        result = self.list.data[self.list.currentRow()][2]
        self.set_current_result.emit(result)

    def path_editor(self):
        pass

    def path_open(self):
        result = self.list.data[self.list.currentRow()][2]
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(Path(result.output).parent)))

    def path_copy(self):
        result = self.list.data[self.list.currentRow()][2]
        cb = QtGui.QGuiApplication.clipboard()
        cb.setText(str(result.output), mode=cb.Clipboard)

    def progress(self, tup):
        n, total = tup
        self.progressbar.setMaximum(total)
        self.progressbar.setValue(n)

    def generate_code(self):
        try:
            result = self.list.data[self.list.currentRow()][2]
            if result is None:
                return
        except IndexError:
            return
        new_path = QtWidgets.QFileDialog.getSaveFileName(None, "Save Session as Script", os.getcwd(), "Python File (*.py)")
        if new_path:
            # ensure filename ends in .py
            if not new_path.endswith(".py"):
                new_path += ".py"

            import_code = ""
            run_code = ""
            for module in self.modules:
                code1, code2 = module.get_code()
                import_code += code1
                run_code += code2 +"\n"
            run_code = import_code + "\n\n" + run_code
            #print(run_code)
            with open(new_path, "w") as fp:
                fp.write(run_code)

    def run_all(self):
        for i in range(len(self.data)):
            if not self.data[i][1]:
                continue
            result = self.data[i][2]
            for module in self.modules:
                if getattr(module, "group", None) and module.group.value() is True:
                    module.start_process(result=result)

    def addTask(self, task, result, params, name):
        self.tasks.append([task, result, params, name])
        if self.thread is None:
            self.run_next()

    signal_task_finished = QtCore.Signal()

    def run_next(self):
        task, result, params, name = self.tasks[self.current_task_id]
        self.thread = threading.Thread(target=self.run_thread, args=(task, result, params, name), daemon=True)
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
        urls = []
        for url in event.mimeData().urls():

            url = url.toLocalFile()  # path()

            if url[0] == "/" and url[2] == ":":
                url = url[1:]
            urls.append(url)
        self.load_from_path(urls)

    def load_from_path(self, paths):
        # make sure that paths is a list
        if isinstance(paths, (str, Path)):
            paths = [paths]

        # iterate over all paths
        for path in paths:
            # if it is a directory search all saenopy files in it
            path = Path(path)
            if path.is_dir():
                path = str(path) + "/**/*" + self.file_extension
            # glob over the path (or just use the path if it does not contain a *)
            for p in sorted(glob.glob(str(path), recursive=True)):
                print(p)
                try:
                    if self.file_extension == ".saenopySpheroid":
                        self.add_data(ResultSpheroid.load(p))
                        pass
                    elif self.file_extension == ".saenopy2D":
                        self.add_data(Result2D.load(p))
                    else:
                        self.add_data(Result.load(p))
                except Exception as err:
                    QtWidgets.QMessageBox.critical(self, "Open Files", f"File {p} is not a valid Saenopy file.")
                    traceback.print_exc()
        #self.update_icons()

    def add_data(self, data):
        self.list.addData(data.output, True, data, mpl.colors.to_hex(f"gray"))
        self.list.setCurrentRow(len(self.data) - 1)

    def update_icons(self):
        for j in range(self.list.count( ) -1):
            if self.data[j][2].state is True:
                self.list.item(j).setIcon(qta.icon("fa5s.hourglass-half", options=[dict(color="orange")]))
            else:
                self.list.item(j).setIcon(qta.icon("fa5.circle", options=[dict(color="gray")]))

    def add_measurement(self):
        pass

    def listSelected(self):
        if self.list.currentRow() is not None and self.list.currentRow() < len(self.data):
            pipe = self.data[self.list.currentRow()][2]
            self.set_current_result.emit(pipe)
