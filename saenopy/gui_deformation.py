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
import imageio
from qimage2ndarray import array2qimage
import matplotlib.pyplot as plt
import glob
import imageio
import threading
import saenopy.getDeformations
import saenopy.multigridHelper
import saenopy.materials

if QtCore.qVersion() >= "5.":
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure


""" some magic to prevent PyQt5 from swallowing exceptions """
# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook
# Set the exception hook to our wrapping function
sys.excepthook = lambda *args: sys._excepthook(*args)



class DeformationDetector(QtWidgets.QWidget):
    detection_finished = QtCore.Signal()
    detection_error = QtCore.Signal(str)

    def __init__(self, layout, stack_deformed, stack_relaxed):
        super().__init__()
        layout.addWidget(self)

        main_layout = QtWidgets.QVBoxLayout(self)

        self.stack_deformed = stack_deformed
        self.stack_relaxed = stack_relaxed

        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

        self.input_overlap = QtShortCuts.QInputNumber(main_layout, "overlap", 0.6, step=0.1)
        self.input_win = QtShortCuts.QInputNumber(main_layout, "window size", 30)
        self.input_signoise = QtShortCuts.QInputNumber(main_layout, "signoise", 1.3, step=0.1)
        self.input_driftcorrection = QtShortCuts.QInputBool(main_layout, "driftcorrection", True)
        self.input_button = QtWidgets.QPushButton("detect deformations")
        main_layout.addWidget(self.input_button)
        self.input_button.clicked.connect(self.start_detect)

        self.progressbar = QProgressBar(main_layout)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("black")
        main_layout.addWidget(self.plotter.interactor)

        self.detection_finished.connect(self.finished_detection)
        self.detection_error.connect(self.errored_detection)

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
            voxel_size1 = self.stack_deformed.getVoxelSize()
            voxel_size2 = self.stack_relaxed.getVoxelSize()
            if voxel_size1 is None:
                return self.detection_error.emit("The no stack for the deformed state selected.")
            if voxel_size2 is None:
                return self.detection_error.emit("The no stack for the relaxed state selected.")
            if np.any(np.array(voxel_size1) != np.array(voxel_size2)):
                return self.detection_error.emit("The two stacks do not have the same voxel size.")
            if 1:
                stack_deformed = self.stack_deformed.getStack()
                stack_relaxed = self.stack_relaxed.getStack()
                if np.any(np.array(stack_deformed.shape) != np.array(stack_relaxed.shape)):
                    return self.detection_error.emit("The two stacks do not have the same voxel count.")
                self.M = saenopy.getDeformations.getDisplacementsFromStacks(stack_deformed, stack_relaxed,
                                           voxel_size1, win_um=self.input_win.value(), fac_overlap=self.input_overlap.value(),
                                           signoise_filter=self.input_signoise.value(), drift_correction=self.input_driftcorrection.value())
                self.M.save("tmp_deformation.npz")
            else:
                self.M = saenopy.load("tmp_deformation.npz")
                self.M.R += (np.max(self.M.R, axis=0) - np.min(self.M.R, axis=0))
            self.point_cloud = pv.PolyData(self.M.R)
            self.point_cloud.point_arrays["f"] = -self.M.f
            self.point_cloud["f_mag"] = np.linalg.norm(self.M.f, axis=1)
            self.point_cloud.point_arrays["U"] = self.M.U
            self.point_cloud["U_mag"] = np.linalg.norm(self.M.U, axis=1)
            self.point_cloud.point_arrays["U_target"] = self.M.U_target
            self.point_cloud["U_target_mag"] = np.linalg.norm(self.M.U_target, axis=1)

            self.detection_finished.emit()
        except Exception as err:
            traceback.print_exc()
            self.detection_error.emit(str(err))

    def finished_detection(self):
        self.input_button.setEnabled(True)
        self.progressbar.setRange(0, 1)
        self.replot()

    def errored_detection(self, text):
        QtWidgets.QMessageBox.critical(self, "Deformation Detector", text)
        self.input_button.setEnabled(True)
        self.progressbar.setRange(0, 1)

    def replot(self):
        arrows = self.point_cloud.glyph(orient="U_target", scale="U_target_mag", factor=5)
        self.plotter.add_mesh(arrows, colormap='turbo', name="arrows")
        self.plotter.update_scalar_bar_range(np.nanpercentile(self.point_cloud["U_target_mag"], [1, 99.9]))
        self.plotter.show_grid()
        self.plotter.show()


class MeshCreator(QtWidgets.QWidget):
    detection_finished = QtCore.Signal()

    def __init__(self, layout, deformation_detector):
        super().__init__()
        layout.addWidget(self)

        main_layout = QtWidgets.QVBoxLayout(self)

        self.deformation_detector = deformation_detector

        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

        self.input_element_size = QtShortCuts.QInputString(main_layout, "element_size", "7")
        self.input_inner_region = QtShortCuts.QInputString(main_layout, "inner_region", "100")
        self.input_thinning_factor = QtShortCuts.QInputNumber(main_layout, "thinning factor", 0.2, step=0.1)
        self.input_button = QtWidgets.QPushButton("interpolate mesh")
        main_layout.addWidget(self.input_button)
        self.input_button.clicked.connect(self.start_interpolation)

        self.progressbar = QProgressBar(main_layout)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("black")
        main_layout.addWidget(self.plotter.interactor)

        self.detection_finished.connect(self.finished_detection)

    def start_interpolation(self):
        self.input_button.setEnabled(False)
        self.progressbar.setRange(0, 0)
        self.thread = threading.Thread(target=self.interpolation_thread)
        self.thread.start()

    def interpolation_thread(self):
        M = self.deformation_detector.M
        points, cells = saenopy.multigridHelper.getScaledMesh(float(self.input_element_size.value())*1e-6,
                                      float(self.input_inner_region.value())*1e-6,
                                      (np.max(M.R, axis=0) - np.min(M.R, axis=0)) / 2,
                                      [0, 0, 0], self.input_thinning_factor.value())

        U_target = saenopy.getDeformations.interpolate_different_mesh(M.R, M.U_target, points)
        self.M = saenopy.Solver()
        self.M.setNodes(points)
        self.M.setTetrahedra(cells)
        self.M.setTargetDisplacements(U_target)

        self.point_cloud = pv.PolyData(self.M.R)
        self.point_cloud.point_arrays["f"] = -self.M.f
        self.point_cloud["f_mag"] = np.linalg.norm(self.M.f, axis=1)
        self.point_cloud.point_arrays["U"] = self.M.U
        self.point_cloud["U_mag"] = np.linalg.norm(self.M.U, axis=1)
        self.point_cloud.point_arrays["U_target"] = self.M.U_target
        self.point_cloud["U_target_mag"] = np.linalg.norm(self.M.U_target, axis=1)

        self.detection_finished.emit()

    def finished_detection(self):
        self.input_button.setEnabled(True)
        self.progressbar.setRange(0, 1)
        self.replot()

    def replot(self):
        arrows = self.point_cloud.glyph(orient="U_target", scale="U_target_mag", factor=5)
        self.plotter.add_mesh(arrows, colormap='turbo', name="arrows")
        self.plotter.update_scalar_bar_range(np.nanpercentile(self.point_cloud["U_target_mag"], [1, 99.9]))
        self.plotter.show_grid()
        self.plotter.show()


class Regularizer(QtWidgets.QWidget):
    detection_finished = QtCore.Signal()
    iteration_finished = QtCore.Signal(object)

    def __init__(self, layout, mesh_creator):
        super().__init__()
        layout.addWidget(self)

        main_layout = QtWidgets.QVBoxLayout(self)

        self.mesh_creator = mesh_creator

        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

        self.input_k = QtShortCuts.QInputString(main_layout, "k", "1645")
        self.input_d0 = QtShortCuts.QInputString(main_layout, "d0", "0.0008")
        self.input_lamda_s = QtShortCuts.QInputString(main_layout, "lamdba_s", "0.0075")
        self.input_ds = QtShortCuts.QInputString(main_layout, "ds", "0.033")
        self.input_alpha = QtShortCuts.QInputString(main_layout, "alpha", "9")
        self.input_stepper = QtShortCuts.QInputString(main_layout, "stepper", "0.33")
        self.input_imax = QtShortCuts.QInputNumber(main_layout, "i_max", 100, float=False)
        self.input_button = QtWidgets.QPushButton("calculate forces")
        main_layout.addWidget(self.input_button)
        self.input_button.clicked.connect(self.start_interpolation)

        self.progressbar = QProgressBar(main_layout)

        self.canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.canvas.figure.add_axes([0.2, 0.2, 0.8, 0.8])
        main_layout.addWidget(self.canvas)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("black")
        main_layout.addWidget(self.plotter.interactor)

        self.detection_finished.connect(self.finished_detection)
        self.iteration_finished.connect(self.iteration_callback)

        self.iteration_finished.emit(np.ones([10, 3]))

    def start_interpolation(self):
        self.input_button.setEnabled(False)
        self.progressbar.setRange(0, 0)
        self.thread = threading.Thread(target=self.interpolation_thread)
        self.thread.start()

    def iteration_callback(self, relrec):
        relrec = np.array(relrec)
        self.canvas.figure.axes[0].cla()
        self.canvas.figure.axes[0].semilogy(relrec[:, 0])
        self.canvas.figure.axes[0].semilogy(relrec[:, 1])
        self.canvas.figure.axes[0].semilogy(relrec[:, 2])
        self.canvas.figure.axes[0].set_xlabel("iteration")
        self.canvas.figure.axes[0].set_ylabel("error")
        self.canvas.draw()


    def interpolation_thread(self):
        M = self.mesh_creator.M

        def callback(M, relrec):
            self.iteration_finished.emit(relrec)

        M.setMaterialModel(saenopy.materials.SemiAffineFiberMaterial(
                           float(self.input_k.value()),
                           float(self.input_d0.value()),
                           float(self.input_lamda_s.value()),
                           float(self.input_ds.value()),
                           ))

        M.solve_regularized(stepper=float(self.input_stepper.value()), i_max=self.input_imax.value(),
                            alpha=float(self.input_alpha.value()), callback=callback, verbose=True)
        self.M = M
        self.point_cloud = pv.PolyData(self.M.R)
        self.point_cloud.point_arrays["f"] = -self.M.f
        self.point_cloud["f_mag"] = np.linalg.norm(self.M.f, axis=1)
        self.point_cloud.point_arrays["U"] = self.M.U
        self.point_cloud["U_mag"] = np.linalg.norm(self.M.U, axis=1)
        self.point_cloud.point_arrays["U_target"] = self.M.U_target
        self.point_cloud["U_target_mag"] = np.linalg.norm(self.M.U_target, axis=1)

        self.detection_finished.emit()

    def finished_detection(self):
        self.input_button.setEnabled(True)
        self.progressbar.setRange(0, 1)
        self.replot()

    def replot(self):
        arrows = self.point_cloud.glyph(orient="f", scale="f_mag", factor=3e3)
        self.plotter.add_mesh(arrows, colormap='turbo', name="arrows")
        #self.plotter.update_scalar_bar_range(np.nanpercentile(self.point_cloud["U_target_mag"], [1, 99.9]))
        self.plotter.show_grid()
        self.plotter.show()


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



class MainWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # QSettings
        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

        self.setMinimumWidth(800)
        self.setMinimumHeight(400)
        self.setWindowTitle("Saenopy Viewer")

        main_layout = QtWidgets.QHBoxLayout(self)

        h_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(h_layout)

        self.stack_before = StackSelector(h_layout, "deformed")
        self.stack_after = StackSelector(h_layout, "relaxed", self.stack_before)
        self.deformation = DeformationDetector(main_layout, self.stack_before, self.stack_after)
        self.mesh = MeshCreator(main_layout, self.deformation)
        self.regularizer = Regularizer(main_layout, self.mesh)

    def load(self):
        files = glob.glob(self.input_filename.value())
        self.input_label.setText("\n".join(files))
#        self.input_filename


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    print(sys.argv)
    window = MainWindow()
    if len(sys.argv) >= 2:
        window.loadFile(sys.argv[1])
    window.show()
    sys.exit(app.exec_())
