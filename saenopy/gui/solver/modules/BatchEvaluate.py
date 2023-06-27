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

from .DeformationDetector import DeformationDetector
from .FittedMesh import FittedMesh
from .MeshCreator import MeshCreator
from .Regularizer import Regularizer
from .ResultView import ResultView
from .StackDisplay import StackDisplay
from saenopy.gui.solver.modules.exporter.Exporter import ExportViewer
from .load_measurement_dialog import AddFilesDialog, FileExistsDialog
from .path_editor import start_path_change
from saenopy.examples import get_examples


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


class BatchEvaluate(QtWidgets.QWidget):
    result_changed = QtCore.Signal(object)
    tab_changed = QtCore.Signal(object)
    set_current_result = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.shared_properties = SharedProperties()

        self.settings = QtCore.QSettings("Saenopy", "Seanopy_deformation")

        with QtShortCuts.QHBoxLayout(self) as main_layout:
            main_layout.setContentsMargins(0, 0, 0, 0)
            with QtShortCuts.QSplitter() as lay:
                with QtShortCuts.QVBoxLayout() as layout:
                    layout.setContentsMargins(0, 0, 0, 0)
                    self.list = ListWidget(layout, add_item_button="add measurements", copy_params=True, allow_paste_callback=self.allow_paste)
                    self.list.addItemClicked.connect(self.add_measurement)
                    self.list.signal_act_copy_clicked.connect(self.copy_params)
                    self.list.signal_act_paste_clicked.connect(self.paste_params)
                    self.list.signal_act_paths_clicked.connect(self.path_editor)
                    self.list.itemSelectionChanged.connect(self.listSelected)
                    self.progressbar = QtWidgets.QProgressBar().addToLayout()
                    self.progressbar.setOrientation(QtCore.Qt.Horizontal)
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
                    #self.sub_module_fiber = FiberViewer(self, layout0)
                    self.sub_module_export = ExportViewer(self, layout0)
                    layout0.addStretch()
                    self.button_start_all = QtShortCuts.QPushButton(None, "run all", self.run_all)
                    with QtShortCuts.QHBoxLayout():
                        self.button_code = QtShortCuts.QPushButton(None, "export code", self.generate_code)
                        self.button_export = QtShortCuts.QPushButton(None, "export images", lambda x: self.sub_module_export.export_window.show())

        self.data = []
        self.list.setData(self.data)

        self.setAcceptDrops(True)

        self.tasks = []
        self.current_task_id = 0
        self.thread = None
        self.signal_task_finished.connect(self.run_finished)

        # load paths
        self.load_from_path([arg for arg in sys.argv if arg.endswith(".saenopy")])

        # disable all tabs
        for i in range(self.tabs.count()-1, -1, -1):
            self.tabs.setTabEnabled(i, False)

    def copy_params(self):
        result = self.list.data[self.list.currentRow()][2]
        params = {
            "piv_parameters": result.piv_parameters_tmp,
            "mesh_parameters": result.mesh_parameters_tmp,
            "material_parameters": result.material_parameters_tmp,
            "solve_parameters": result.solve_parameters_tmp,
        }
        print(params)
        for group in params:
            if params[group] is None:
                continue
            for g in params[group]:
                if type(params[group][g]) == np.bool_:
                    params[group][g] = bool(params[group][g])
                if type(params[group][g]) == np.int64:
                    params[group][g] = int(params[group][g])
        text = json.dumps(params, indent=2)
        cb = QtGui.QGuiApplication.clipboard()
        cb.setText(text, mode=cb.Clipboard)

    def allow_paste(self):
        cb = QtGui.QGuiApplication.clipboard()
        text = cb.text(mode=cb.Clipboard)
        try:
            data = json.loads(text)
            if "piv_parameters" in data and \
                "mesh_parameters" in data and \
                "material_parameters" in data and \
                "solve_parameters" in data:
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
        params = ["piv_parameters", "mesh_parameters", "material_parameters", "solve_parameters"]
        for par in params:
            if par in data:
                setattr(result, par+"_tmp", data[par])
        self.set_current_result.emit(result)

    def path_editor(self):
        result = self.list.data[self.list.currentRow()][2]
        start_path_change(self, result)

    def progress(self, tup):
        n, total = tup
        self.progressbar.setMaximum(total)
        self.progressbar.setValue(n)

    def generate_code(self):
        new_path = QtWidgets.QFileDialog.getSaveFileName(None, "Save Session as Script", os.getcwd(), "Python File (*.py)")
        if new_path:
            # ensure filename ends in .py
            if not new_path.endswith(".py"):
                new_path += ".py"

            import_code = ""
            run_code = ""
            for module in [self.sub_module_stacks, self.sub_module_deformation, self.sub_module_mesh, self.sub_module_regularize]:
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
            if self.sub_module_deformation.group.value() is True:
                self.sub_module_deformation.start_process(result=result)
            if self.sub_module_mesh.group.value() is True:
                self.sub_module_mesh.start_process(result=result)
            if self.sub_module_regularize.group.value() is True:
                self.sub_module_regularize.start_process(result=result)

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
                path = str(path) + "/**/*.saenopy"
            # glob over the path (or just use the path if it does not contain a *)
            for p in sorted(glob.glob(str(path), recursive=True)):
                print(p)
                try:
                    self.add_data(Result.load(p))
                except Exception as err:
                    QtWidgets.QMessageBox.critical(self, "Open Files", f"File {p} is not a valid Saenopy file.")
                    traceback.print_exc()
        self.update_icons()

    def add_data(self, data):
        self.list.addData(data.output, True, data, mpl.colors.to_hex(f"gray"))

    def update_icons(self):
        for j in range(self.list.count( ) -1):
            if self.data[j][2].state is True:
                self.list.item(j).setIcon(qta.icon("fa5s.hourglass-half", options=[dict(color="orange")]))
            else:
                self.list.item(j).setIcon(qta.icon("fa5.circle", options=[dict(color="gray")]))

    def add_measurement(self):
        last_decision = None
        def do_overwrite(filename):
            nonlocal last_decision

            # if we are in demo mode always load the files
            if os.environ.get("DEMO") == "true":  # pragma: no cover
                return "read"

            # if there is a last decistion stored use that
            if last_decision is not None:
                return last_decision

            # ask the user if they want to overwrite or read the existing file
            dialog = FileExistsDialog(self, filename)
            result = dialog.exec()
            # if the user clicked cancel
            if not result:
                return 0
            # if the user wants to remember the last decision
            if dialog.use_for_all.value():
                last_decision = dialog.mode
            # return the decision
            return dialog.mode

        # getStack
        dialog = AddFilesDialog(self, self.settings)
        if not dialog.exec():
            return

        # create a new measurement object
        if dialog.mode == "new":
            # if there was a reference stack selected
            if dialog.reference_choice.value() == 1:
                reference_stack = dialog.stack_reference_input.text()
            else:
                reference_stack = None

            # the active selected stack
            active_stack = dialog.stack_data_input.text()

            # if there was a time specified, get the time delta
            if "{t}" in active_stack:
                time_delta = dialog.stack_data.getTimeDelta()
                add_last_time_delta(dialog.stack_data.getTimeDelta())
            else:
                time_delta = None

            try:
                # load the stack
                results = get_stacks(
                    active_stack,
                    reference_stack=reference_stack,
                    output_path=dialog.outputText.value(),
                    voxel_size=dialog.stack_data.getVoxelSize(),
                    time_delta=time_delta,
                    crop=dialog.stack_data.get_crop(),
                    exist_overwrite_callback=do_overwrite,
                )
            except Exception as err:
                # notify the user if errors occured
                QtWidgets.QMessageBox.critical(self, "Load Stacks", str(err))
                traceback.print_exc()
            else:
                # store the last voxel size
                add_last_voxel_size(dialog.stack_data.getVoxelSize())
                # add the loaded measruement objects
                for data in results:
                    self.add_data(data)

        # load existing files
        elif dialog.mode == "existing":
            self.load_from_path(dialog.outputText3.value())

        # load from the examples database
        elif dialog.mode == "example":
            # get the date from the example referenced by name
            example = get_examples()[dialog.mode_data]

            # generate a stack with the examples data
            results = get_stacks(
                example["stack"],
                reference_stack=example.get("reference_stack", None),
                output_path=example["output_path"],
                voxel_size=example["voxel_size"],
                time_delta=example.get("time_delta", None),
                crop=example.get("crop", None),
                exist_overwrite_callback=do_overwrite,
            )
            # load all the measurement objects
            for data in results:
                if getattr(data, "is_read", False) is False:
                    data.piv_parameters = example["piv_parameters"]
                    data.mesh_parameters = example["mesh_parameters"]
                    data.material_parameters = example["material_parameters"]
                    data.solve_parameters = example["solve_parameters"]
                self.add_data(data)
        elif dialog.mode == "example_evaluated":
                self.load_from_path(dialog.examples_output)

        # update the icons
        self.update_icons()

    def listSelected(self):
        if self.list.currentRow() is not None and self.list.currentRow() < len(self.data):
            pipe = self.data[self.list.currentRow()][2]
            self.set_current_result.emit(pipe)
