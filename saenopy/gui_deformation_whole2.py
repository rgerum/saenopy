import sys

# Setting the Qt bindings for QtPy
import os

os.environ["QT_API"] = "pyqt5"

from qtpy import QtCore, QtWidgets, QtGui

import numpy as np

import pyvista as pv
import vtk
from pyvistaqt import QtInteractor

import saenopy
import saenopy.multigridHelper
from saenopy.gui import QtShortCuts, QExtendedGraphicsView
from saenopy.gui.stack_selector import StackSelector
from saenopy.gui.gui_classes import Spoiler, CheckAbleGroup, QHLine, QVLine, MatplotlibWidget, execute, kill_thread, ListWidget
import imageio
from qimage2ndarray import array2qimage
import matplotlib.pyplot as plt
import glob
import imageio
import threading
import saenopy.getDeformations
import saenopy.multigridHelper
import saenopy.materials
from saenopy.gui.stack_selector import StackSelector
from saenopy.getDeformations import getStack, Stack
import matplotlib as mpl
from saenopy.solver import Solver
from pathlib import Path
import re
from saenopy.loadHelpers import Saveable

""" some magic to prevent PyQt5 from swallowing exceptions """
# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook
# Set the exception hook to our wrapping function
sys.excepthook = lambda *args: sys._excepthook(*args)


class Result(Saveable):
    __save_parameters__ = ['output', 'stack_deformed', 'stack_relaxed', 'piv_parameter', 'mesh_piv',
                           'iterpolate_parameter', 'solve_parameter', 'solver']
    stack_deformed: Stack = None
    stack_relaxed: Stack = None
    output: str = None
    solver = None
    piv_parameter: dict = None
    mesh_piv: saenopy.solver.Mesh = None

    iterpolate_parameter: dict = None
    solve_parameter: dict = None
    solver: saenopy.solver.Solver = None

    def __init__(self, output, stack_deformed, stack_relaxed, piv_parameter=None, **kwargs):
        self.output = output

        self.stack_deformed = stack_deformed
        self.stack_relaxed = stack_relaxed

        if piv_parameter is not None:
            self.piv_parameter = piv_parameter
        else:
            self.piv_parameter = {}

        super().__init__(**kwargs)

    def __str__(self):
        return f"""
from saenopy.getDeformations import Stack
stack_deformed = Stack({self.deformed})
stack_relaxed = Stack({self.relaxed})
"""
    def save(self):
        super().save(self.output)


def double_glob(text):
    glob_string = text.replace("?", "*")
    print("globbing", glob_string)
    files = glob.glob(glob_string)

    output_base = glob_string
    while "*" in str(output_base):
        output_base = Path(output_base).parent

    regex_string = re.escape(text).replace("\*", "(.*)").replace("\?", ".*")

    results = []
    for file in files:
        file = os.path.normpath(file)
        print(file, regex_string)
        match = re.match(regex_string, file).groups()
        reconstructed_file = regex_string
        for element in match:
            reconstructed_file = reconstructed_file.replace("(.*)", element, 1)
        reconstructed_file = reconstructed_file.replace(".*", "*")
        reconstructed_file = re.sub(r'\\(.)', r'\1', reconstructed_file)
        if reconstructed_file not in results:
            results.append(reconstructed_file)
    return results, output_base


class QProgressBar(QtWidgets.QProgressBar):
    signal_start = QtCore.Signal(int)
    signal_progress = QtCore.Signal(int)

    def __init__(self, layout):
        super().__init__()
        self.setOrientation(QtCore.Qt.Horizontal)
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


class StackDisplay(QtWidgets.QWidget):
    def __init__(self, parent, layout):
        super().__init__()
        if layout is not None:
            layout.addWidget(self)
        self.parent = parent

        self.parent.result_changed.connect(self.setResult)
        self.setResult(None)

        with self.parent.tabs.createTab("Stacks") as self.tab:
            with QtShortCuts.QHBoxLayout() as layout:
                with QtShortCuts.QVBoxLayout() as layout:
                    self.label1 = QtWidgets.QLabel("deformed").addToLayout()
                    self.view1 = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
                    self.view1.setMinimumWidth(300)
                    self.pixmap1 = QtWidgets.QGraphicsPixmapItem(self.view1.origin)

                    self.label2 = QtWidgets.QLabel("relaxed").addToLayout()
                    self.view2 = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
                    # self.label2.setMinimumWidth(300)
                    self.pixmap2 = QtWidgets.QGraphicsPixmapItem(self.view2.origin)
                self.z_slider = QtWidgets.QSlider(QtCore.Qt.Vertical).addToLayout()
                self.z_slider.valueChanged.connect(self.z_slider_value_changed)

        self.link_views()

    def setResult(self, result: Result):
        self.result = result
        if self.result is not None:
            self.parent.tabs.setTabEnabled(0, True)
            self.z_slider.setRange(0, result.stack_deformed.shape[2] - 1)
            self.z_slider.setValue(result.stack_deformed.shape[2] // 2)
        else:
            self.parent.tabs.setTabEnabled(0, False)

    def z_slider_value_changed(self):
        if self.result is not None:
            print(self.z_slider.value(), self.result.stack_deformed.shape)
            im = self.result.stack_deformed[:, :, self.z_slider.value()]
            self.pixmap1.setPixmap(QtGui.QPixmap(array2qimage(im)))
            self.view1.setExtend(im.shape[1], im.shape[0])

            im = self.result.stack_relaxed[:, :, self.z_slider.value()]
            self.pixmap2.setPixmap(QtGui.QPixmap(array2qimage(im)))
            self.view2.setExtend(im.shape[1], im.shape[0])

    def link_views(self):
        def changes1(*args):
            self.view2.setOriginScale(self.view1.getOriginScale() * self.view1.view_rect[0] / self.view2.view_rect[0])
            start_x, start_y, end_x, end_y = self.view1.GetExtend()
            center_x, center_y = start_x + (end_x - start_x) / 2, start_y + (end_y - start_y) / 2
            center_x = center_x / self.view1.view_rect[0] * self.view2.view_rect[0]
            center_y = center_y / self.view1.view_rect[1] * self.view2.view_rect[1]
            self.view2.centerOn(center_x, center_y)

        def zoomEvent(scale, pos):
            changes1()

        self.view1.zoomEvent = zoomEvent
        self.view1.panEvent = changes1

        def changes2(*args):
            self.view1.setOriginScale(self.view2.getOriginScale() * self.view2.view_rect[0] / self.view1.view_rect[0])
            start_x, start_y, end_x, end_y = self.view2.GetExtend()
            center_x, center_y = start_x + (end_x - start_x) / 2, start_y + (end_y - start_y) / 2
            center_x = center_x / self.view2.view_rect[0] * self.view1.view_rect[0]
            center_y = center_y / self.view2.view_rect[1] * self.view1.view_rect[1]
            self.view1.centerOn(center_x, center_y)

        def zoomEvent(scale, pos):
            changes2()

        self.view2.zoomEvent = zoomEvent
        self.view2.panEvent = changes2
        changes2()

class DeformationDetector(QtWidgets.QWidget):
    detection_finished = QtCore.Signal()
    detection_error = QtCore.Signal(str)
    #mesh_size_changed = QtCore.Signal(float, float, float)
    result = None

    def __init__(self, parent, layout):
        super().__init__()
        if layout is not None:
            layout.addWidget(self)
        self.parent = parent

        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

        with QtShortCuts.QVBoxLayout(self) as layout:
            with QtShortCuts.QHBoxLayout():
                self.input_overlap = QtShortCuts.QInputNumber(None, "overlap", 0.6, step=0.1, value_changed=self.valueChanged)
                self.input_win = QtShortCuts.QInputNumber(None, "window size", 30, value_changed=self.valueChanged, unit="μm")
            with QtShortCuts.QHBoxLayout():
                self.input_signoise = QtShortCuts.QInputNumber(None, "signoise", 1.3, step=0.1)
                self.input_driftcorrection = QtShortCuts.QInputBool(None, "driftcorrection", True)
            self.label = QtWidgets.QLabel().addToLayout()
            self.input_button = QtShortCuts.QPushButton(None, "detect deformations", self.start_detect)

            self.progressbar = QProgressBar(layout).addToLayout()

        self.parameter_dict = {
            "win_um": self.input_win,
            "fac_overlap": self.input_overlap,
            "signoise_filter": self.input_signoise,
            "drift_correction": self.input_driftcorrection,
        }
        for name, widget in self.parameter_dict.items():
            widget.valueChanged.connect(lambda x, name=name: self.setParameter(name, x))

        with self.parent.tabs.createTab("Deformations") as self.tab:
            with QtShortCuts.QHBoxLayout() as layout:
                self.plotter = QtInteractor(self)
                self.plotter.set_background("black")
                layout.addWidget(self.plotter.interactor)

        self.detection_finished.connect(self.finished_detection)
        self.detection_error.connect(self.errored_detection)

        self.parent.result_changed.connect(self.setResult)
        self.setResult(None)

    def setResult(self, result: Result):
        self.result = result
        if self.result is None or self.result.stack_deformed is None or self.result.stack_relaxed is None:
            for name, widget in self.parameter_dict.items():
                widget.setDisabled(True)
            self.label.setText("")
        else:
            for name, widget in self.parameter_dict.items():
                widget.setDisabled(False)
                if name in result.piv_parameter:
                    widget.setValue(result.piv_parameter[name])
                else:
                    result.piv_parameter[name] = widget.value()
            self.valueChanged()
        self.update_plot()

    def update_plot(self):
        if self.result is not None and self.result.mesh_piv is not None:
            self.parent.tabs.setTabEnabled(1, True)
            mesh = self.result.mesh_piv
            self.point_cloud = pv.PolyData(mesh.R)
            self.point_cloud.point_arrays["U_measured"] = mesh.getNodeVar("U_measured")
            self.point_cloud["U_measured_mag"] = np.linalg.norm(mesh.getNodeVar("U_measured"), axis=1)
            arrows = self.point_cloud.glyph(orient="U_measured", scale="U_measured_mag", factor=5)
            self.plotter.add_mesh(arrows, colormap='turbo', name="arrows")
            self.plotter.update_scalar_bar_range(np.nanpercentile(self.point_cloud["U_measured_mag"], [1, 99.9]))
            self.plotter.show_grid()
            self.plotter.show()
        else:
            self.parent.tabs.setTabEnabled(1, False)

    def setParameter(self, name, value):
        if self.result is not None:
            self.result.piv_parameter[name] = value

    def valueChanged(self):
        voxel_size1 = self.result.stack_deformed.voxel_size
        voxel_size2 = self.result.stack_relaxed.voxel_size
        stack_deformed = self.result.stack_deformed
        stack_relaxed = self.result.stack_relaxed

        unit_size = (1-self.input_overlap.value())*self.input_win.value()
        stack_size = np.array(stack_deformed.shape)*voxel_size1 - self.input_win.value()
        self.label.setText(f"Deformation grid with {unit_size:.1f}μm elements. Total region is {stack_size}.")
        #self.mesh_size_changed.emit(stack_size)

    def start_detect(self):
        self.input_button.setEnabled(False)
        self.progressbar.setRange(0, 0)
        self.thread = threading.Thread(target=self.detection_thread)
        self.thread.start()

    def detection_thread(self):
        try:
            import tqdm
            t = tqdm.tqdm
            n = tqdm.tqdm.__new__
            if 0:
                def new(cls, iter):
                    print("new", cls, iter)
                    instance = n(cls, iter)
                    init = instance.__init__
                    def init_new(iter, *args):
                        print("do init")
                        print(iter, args)
                        init(self.progressbar.iterator(iter))
                    instance.__init__ = init_new
                    #print("instace", instance)
                    return instance
                tqdm.tqdm.__new__ = new# lambda cls, iter: n(cls, self.progressbar.iterator(iter))
            tqdm.tqdm.__new__ = lambda cls, iter: self.progressbar.iterator(iter)
            #for i in tqdm.tqdm(range(10)):
            #    import time
            #    time.sleep(0.1)

            self.result.mesh_piv = saenopy.getDeformations.getDisplacementsFromStacks2(self.result.stack_deformed, self.result.stack_relaxed,
                                                                                       **self.result.piv_parameter)
            self.result.save()
            self.parent.result_changed.emit(self.result)
            self.detection_finished.emit()
        except IndexError as err:
            print(err, file=sys.stderr)
            self.detection_error.emit(str(err))

    def finished_detection(self):
        self.input_button.setEnabled(True)
        self.progressbar.setRange(0, 1)

    def errored_detection(self, text):
        QtWidgets.QMessageBox.critical(self, "Deformation Detector", text)
        self.input_button.setEnabled(True)
        self.progressbar.setRange(0, 1)


class MeshCreator(QtWidgets.QWidget):
    detection_finished = QtCore.Signal()
    result = None

    mesh_size = [200, 200, 200]

    def __init__(self, parent, layout):
        super().__init__()
        if layout is not None:
            layout.addWidget(self)
        self.parent = parent

        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

        with QtShortCuts.QVBoxLayout(self) as layout:
            with QtShortCuts.QHBoxLayout():
                self.input_element_size = QtShortCuts.QInputNumber(None, "element_size", 7, unit="μm")
            with QtShortCuts.QHBoxLayout() as layout2:
                self.input_inner_region = QtShortCuts.QInputNumber(None, "inner_region", 100, unit="μm")
                self.input_thinning_factor = QtShortCuts.QInputNumber(None, "thinning factor", 0.2, step=0.1)
            with QtShortCuts.QHBoxLayout() as layout2:

                self.input_mesh_size_same = QtShortCuts.QInputBool(None, "same as stack", True)
                self.input_mesh_size_same.valueChanged.connect(self.changed_same_as)
                self.input_mesh_size_x = QtShortCuts.QInputNumber(None, "mesh size x", 200, step=1)
                self.input_mesh_size_y = QtShortCuts.QInputNumber(None, "y", 200, step=1)
                self.input_mesh_size_z = QtShortCuts.QInputNumber(None, "z", 200, step=1)
                self.input_mesh_size_label = QtWidgets.QLabel("μm").addToLayout()

                #self.deformation_detector.mesh_size_changed.connect(deformation_detector_mesh_size_changed)
                self.changed_same_as()
            self.input_button = QtWidgets.QPushButton("interpolate mesh").addToLayout()
            self.input_button.clicked.connect(self.start_interpolation)

            self.progressbar = QProgressBar(layout).addToLayout()

        self.parameter_dict = {
            "element_size": self.input_element_size,
            "inner_region": self.input_inner_region,
            "thinning_factor": self.input_thinning_factor,
            "mesh_size_same": self.input_mesh_size_same,
            "mesh_size_x": self.input_mesh_size_x,
            "mesh_size_y": self.input_mesh_size_y,
            "mesh_size_z": self.input_mesh_size_z,
        }
        for name, widget in self.parameter_dict.items():
            widget.valueChanged.connect(lambda x, name=name: self.setParameter(name, x))

        with self.parent.tabs.createTab("Mesh") as self.tab:
            with QtShortCuts.QHBoxLayout() as layout:
                self.plotter = QtInteractor(self)
                self.plotter.set_background("black")
                layout.addWidget(self.plotter.interactor)

        self.detection_finished.connect(self.finished_detection)

        self.parent.result_changed.connect(self.setResult)
        self.setResult(None)

    def setParameter(self, name, value):
        if self.result is not None:
            self.result.solve_parameter[name] = value

    def changed_same_as(self):
        self.input_mesh_size_x.setDisabled(self.input_mesh_size_same.value())
        self.input_mesh_size_y.setDisabled(self.input_mesh_size_same.value())
        self.input_mesh_size_z.setDisabled(self.input_mesh_size_same.value())
        self.deformation_detector_mesh_size_changed()

    def deformation_detector_mesh_size_changed(self):
        if self.input_mesh_size_same.value():
            if self.result is not None and self.result.mesh_piv is not None:
                x, y, z = self.result.mesh_piv.R.max(axis=0) - self.result.mesh_piv.R.min(axis=0)
                self.input_mesh_size_x.setValue(x*1e6)
                self.setParameter("mesh_size_x", x*1e6)
                self.input_mesh_size_y.setValue(y*1e6)
                self.setParameter("mesh_size_y", y*1e6)
                self.input_mesh_size_z.setValue(z*1e6)
                self.setParameter("mesh_size_z", z*1e6)


    def setResult(self, result: Result):
        self.result = result
        if self.result is None or self.result.mesh_piv is None:
            for name, widget in self.parameter_dict.items():
                widget.setDisabled(True)
            #self.label.setText("")
        else:
            if result.solve_parameter is None:
                result.solve_parameter = {}
            for name, widget in self.parameter_dict.items():
                widget.setDisabled(False)
                if name in result.solve_parameter:
                    widget.setValue(result.solve_parameter[name])
                else:
                    result.solve_parameter[name] = widget.value()
            self.changed_same_as()
            #self.valueChanged()
        self.update_plot()

    def update_plot(self):
        if self.result is not None and self.result.mesh_piv is not None:
            self.parent.tabs.setTabEnabled(2, True)
            mesh = self.result.solver
            self.point_cloud = pv.PolyData(mesh.R)
            self.point_cloud.point_arrays["U_target"] = mesh.U_target#getNodeVar("U_measured")
            self.point_cloud["U_target_mag"] = np.linalg.norm(mesh.U_target, axis=1)
            arrows = self.point_cloud.glyph(orient="U_target", scale="U_target_mag", factor=5)
            self.plotter.add_mesh(arrows, colormap='turbo', name="arrows")
            self.plotter.update_scalar_bar_range(np.nanpercentile(self.point_cloud["U_target_mag"], [1, 99.9]))
            self.plotter.show_grid()
            self.plotter.show()
        else:
            self.parent.tabs.setTabEnabled(2, False)

    def start_interpolation(self):
        self.input_button.setEnabled(False)
        self.progressbar.setRange(0, 0)
        self.thread = threading.Thread(target=self.interpolation_thread)
        self.thread.start()

    def interpolation_thread(self):
        if self.result is None or self.result.mesh_piv is None:
            return
        M = self.result.mesh_piv
        points, cells = saenopy.multigridHelper.getScaledMesh(self.result.solve_parameter["element_size"]*1e-6,
                                      self.result.solve_parameter["inner_region"]*1e-6,
                                      np.array([self.result.solve_parameter["mesh_size_x"], self.result.solve_parameter["mesh_size_y"],
                                                 self.result.solve_parameter["mesh_size_z"]])*1e-6 / 2,
                                      [0, 0, 0], self.result.solve_parameter["thinning_factor"])
        print(np.max(points, axis=0)*1e6, np.min(points, axis=0)*1e6)
        print(np.max(points, axis=0)*1e6 - np.min(points, axis=0)*1e6)
        print(np.max(M.R, axis=0) * 1e6, np.min(M.R, axis=0) * 1e6)
        print(np.max(M.R, axis=0)*1e6 - np.min(M.R, axis=0)*1e6)

        R = (M.R - np.min(M.R, axis=0)) - (np.max(M.R, axis=0) - np.min(M.R, axis=0)) / 2
        U_target = saenopy.getDeformations.interpolate_different_mesh(R, M.getNodeVar("U_measured"), points)
        print(np.min(U_target), np.max(U_target), np.sum(np.isnan(U_target)))
        self.M = saenopy.Solver()
        self.M.setNodes(points)
        self.M.setTetrahedra(cells)
        self.M.setTargetDisplacements(U_target)

        self.result.solver = self.M
        self.result.save()
        self.parent.result_changed.emit(self.result)

        self.detection_finished.emit()

    def finished_detection(self):
        self.input_button.setEnabled(True)
        self.progressbar.setRange(0, 1)
        #self.replot()

    def replot(self):
        if self.result is None or self.result.solver is None:
            self.parent.tabs.setTabEnabled(2, False)
        else:
            self.parent.tabs.setTabEnabled(2, True)
            self.M = self.result.solver
            self.point_cloud = pv.PolyData(self.M.R)
            #self.point_cloud.point_arrays["f"] = -self.M.f
            #self.point_cloud["f_mag"] = np.linalg.norm(self.M.f, axis=1)
            #self.point_cloud.point_arrays["U"] = self.M.U
            #self.point_cloud["U_mag"] = np.linalg.norm(self.M.U, axis=1)
            self.point_cloud.point_arrays["U_target"] = self.M.U_target
            self.point_cloud["U_target_mag"] = np.linalg.norm(self.M.U_target, axis=1)

            arrows = self.point_cloud.glyph(orient="U_target", scale="U_target_mag", factor=5)
            self.plotter.add_mesh(arrows, colormap='turbo', name="arrows")
            self.plotter.update_scalar_bar_range(np.nanpercentile(self.point_cloud["U_target_mag"], [1, 99.9]))
            self.plotter.show_grid()
            self.plotter.show()


class BatchEvaluate(QtWidgets.QWidget):
    result_changed = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.settings = QtCore.QSettings("Saenopy", "Seanopy_deformation")

        with QtShortCuts.QHBoxLayout(self) as main_layout:
            with QtShortCuts.QSplitter() as lay:
                with QtShortCuts.QVBoxLayout() as layout:
                    layout.setContentsMargins(0, 0, 0, 0)
                    self.list = ListWidget(layout, add_item_button="add measurements")
                    self.list.addItemClicked.connect(self.show_files)
                    self.list.itemSelectionChanged.connect(self.listSelected)
                    self.progress1 = QtWidgets.QProgressBar()
                    layout.addWidget(self.progress1)
         #           self.label = QtWidgets.QLabel(
         #               "Select a path as an input wildcard. Use * to specify a placeholder. All paths that match the wildcard will be added.").addToLayout()
                with QtShortCuts.QHBoxLayout() as layout:
                    layout.setContentsMargins(0, 0, 0, 0)
                    with QtShortCuts.QTabWidget(layout) as self.tabs:
                        pass
                with QtShortCuts.QVBoxLayout() as layout:
                    self.sub_module_stacks = StackDisplay(self, layout)
                    with CheckAbleGroup(self, "find deformations").addToLayout() as self.find_deformations:
                        with QtShortCuts.QVBoxLayout() as layout:
                            self.sub_module_deformation = DeformationDetector(self, layout)
                    with CheckAbleGroup(self, "interpolate mesh").addToLayout() as self.find_deformations:
                        with QtShortCuts.QVBoxLayout() as layout:
                            self.sub_module_mesh = MeshCreator(self, layout)

        self.data = []
        self.list.setData(self.data)

        #self.list.addData("foo", True, [], mpl.colors.to_hex(f"C0"))

        data = Result.load(r"E:\saenopy\test\TestData\output\Mark_and_Find_001_Pos001_S001_z_ch00.npz")
        self.list.addData("test", True, data, mpl.colors.to_hex(f"gray"))

    def show_files(self):
        settings = self.settings

        class AddFilesDialog(QtWidgets.QDialog):
            def __init__(self, parent):
                super().__init__(parent)
                self.setMinimumWidth(800)
                self.setMinimumHeight(600)
                self.setWindowTitle("Add Files")
                with QtShortCuts.QVBoxLayout(self) as layout:
                    self.outputText = QtShortCuts.QInputFolder(None, "output", settings=settings,
                                                               settings_key="batch/wildcard2", allow_edit=True)
                    with QtShortCuts.QHBoxLayout() as layout3:
                        self.stack_before = StackSelector(layout3, "deformed")
                        self.stack_before.glob_string_changed.connect(lambda x, y: self.input_deformed.setText(y.replace("*", "?")))
                        self.stack_after = StackSelector(layout3, "relaxed", self.stack_before)
                        self.stack_after.glob_string_changed.connect(lambda x, y: self.input_relaxed.setText(y.replace("*", "?")))
                    with QtShortCuts.QHBoxLayout() as layout3:
                        self.input_deformed = QtWidgets.QLineEdit().addToLayout()
                        self.input_relaxed = QtWidgets.QLineEdit().addToLayout()
                        self.input_relaxed.text()
                    with QtShortCuts.QHBoxLayout() as layout3:
                        # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                        layout3.addStretch()
                        self.button_addList2 = QtShortCuts.QPushButton(None, "cancel", self.reject)
                        self.button_addList1 = QtShortCuts.QPushButton(None, "ok", self.accept)
        #getStack
        dialog = AddFilesDialog(self)
        if not dialog.exec():
            return

        results1, output_base = double_glob(dialog.input_deformed.text())
        results2, _ = double_glob(dialog.input_relaxed.text())
        voxel_size1 = dialog.stack_before.getVoxelSize()
        voxel_size2 = dialog.stack_after.getVoxelSize()

        for r1, r2 in zip(results1, results2):
            output = Path(dialog.outputText.value()) / os.path.relpath(r1, output_base)
            output = output.parent / output.stem
            output = Path(str(output).replace("*", "")+".npz")

            if output.exists():
                print('exists')
                data = Result.load(output)
            else:
                print("new")
                data = Result(
                    output=output,
                    stack_deformed=Stack(r1, voxel_size1),
                    stack_relaxed=Stack(r2, voxel_size2),
                )
            print(r1, r2, output)
            self.list.addData(r1, True, data, mpl.colors.to_hex(f"gray"))

        #import matplotlib as mpl
        #for fiber, cell, out in zip(fiber_list, cell_list, out_list):
        #    self.list.addData(fiber, True, [fiber, cell, out, {"segmention_thres": None, "seg_gaus1": None, "seg_gaus2": None}], mpl.colors.to_hex(f"gray"))

    def listSelected(self):
        if self.list.currentRow() is not None:
            pipe = self.data[self.list.currentRow()][2]
            self.result_changed.emit(pipe)



class MainWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # QSettings
        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

        self.setMinimumWidth(800)
        self.setMinimumHeight(400)
        self.setWindowTitle("Saenopy Viewer")

        main_layout = QtWidgets.QHBoxLayout(self)

        with QtShortCuts.QTabWidget(main_layout) as self.tabs:
            """ """
            with self.tabs.createTab("Compaction") as v_layout:
                with QtShortCuts.QHBoxLayout() as h_layout:
                    # self.deformations = Deformation(h_layout, self)
                    self.deformations = BatchEvaluate(self)
                    h_layout.addWidget(self.deformations)
                    self.description = QtWidgets.QTextEdit()
                    self.description.setDisabled(True)
                    self.description.setMaximumWidth(300)
                    h_layout.addWidget(self.description)
                    self.description.setText("""
                    <h1>Start Evaluation</h1>
                     """.strip())
                v_layout.addWidget(QHLine())
                with QtShortCuts.QHBoxLayout() as h_layout:
                    h_layout.addStretch()
                    #self.button_previous = QtShortCuts.QPushButton(None, "back", self.previous)
                    #self.button_next = QtShortCuts.QPushButton(None, "next", self.next)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    print(sys.argv)
    window = MainWindow()
    if len(sys.argv) >= 2:
        window.loadFile(sys.argv[1])
    window.show()
    sys.exit(app.exec_())
