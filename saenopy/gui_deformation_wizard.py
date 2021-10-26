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

class QHLine(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


class DeformationDetector(QtWidgets.QWidget):
    detection_finished = QtCore.Signal()
    detection_error = QtCore.Signal(str)
    mesh_size_changed = QtCore.Signal(float, float, float)
    M = None

    def __init__(self, layout, stack_deformed, stack_relaxed):
        super().__init__()
        if layout is not None:
            layout.addWidget(self)

        main_layout = QtWidgets.QVBoxLayout(self)

        self.stack_deformed = stack_deformed
        self.stack_relaxed = stack_relaxed

        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

        with QtShortCuts.QHBoxLayout(main_layout) as layout2:
            self.input_overlap = QtShortCuts.QInputNumber(None, "overlap", 0.6, step=0.1, value_changed=self.valueChanged)
            self.input_win = QtShortCuts.QInputNumber(None, "window size", 30, value_changed=self.valueChanged, unit="μm")
        with QtShortCuts.QHBoxLayout(main_layout) as layout2:
            self.input_signoise = QtShortCuts.QInputNumber(None, "signoise", 1.3, step=0.1)
            self.input_driftcorrection = QtShortCuts.QInputBool(None, "driftcorrection", True)
        self.label = QtWidgets.QLabel()
        main_layout.addWidget(self.label)
        self.input_button = QtWidgets.QPushButton("detect deformations")
        main_layout.addWidget(self.input_button)
        self.input_button.clicked.connect(self.start_detect)

        self.progressbar = QProgressBar(main_layout)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("black")
        main_layout.addWidget(self.plotter.interactor)

        self.detection_finished.connect(self.finished_detection)
        self.detection_error.connect(self.errored_detection)

    def valueChanged(self):
        voxel_size1 = self.stack_deformed.getVoxelSize()
        voxel_size2 = self.stack_relaxed.getVoxelSize()
        stack_deformed = self.stack_deformed.getStack()
        stack_relaxed = self.stack_relaxed.getStack()

        unit_size = self.input_overlap.value()*self.input_win.value()
        stack_size = np.array(stack_deformed.shape)*voxel_size1 - self.input_win.value()
        self.label.setText(f"Deformation grid with {unit_size}μm elements. Total region is {stack_size}.")
        self.mesh_size_changed.emit(stack_size)

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
                return self.detection_error.emit("No stack for the deformed state has been  selected.")
            if voxel_size2 is None:
                return self.detection_error.emit("No stack for the relaxed state has been selected.")
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
            print(err, file=sys.stderr)
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

    mesh_size = [200, 200, 200]

    def __init__(self, layout, deformation_detector):
        super().__init__()
        if layout is not None:
            layout.addWidget(self)

        main_layout = QtWidgets.QVBoxLayout(self)

        self.deformation_detector = deformation_detector

        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

        self.input_element_size = QtShortCuts.QInputNumber(main_layout, "element_size", 7, unit="μm")
        with QtShortCuts.QHBoxLayout(main_layout) as layout2:
            self.input_inner_region = QtShortCuts.QInputNumber(None, "inner_region", 100, unit="μm")
            self.input_thinning_factor = QtShortCuts.QInputNumber(None, "thinning factor", 0.2, step=0.1)
        with QtShortCuts.QHBoxLayout(main_layout) as layout2:
            def changed_same_as():
                self.input_mesh_size_x.setDisabled(self.input_mesh_size_same.value())
                self.input_mesh_size_y.setDisabled(self.input_mesh_size_same.value())
                self.input_mesh_size_z.setDisabled(self.input_mesh_size_same.value())
                deformation_detector_mesh_size_changed(*self.mesh_size)
            def deformation_detector_mesh_size_changed(x, y, z):
                self.mesh_size = [x, y, z]
                if self.input_mesh_size_same.value():
                    self.input_mesh_size_x.setValue(x)
                    self.input_mesh_size_y.setValue(y)
                    self.input_mesh_size_z.setValue(z)
            self.input_mesh_size_same = QtShortCuts.QInputBool(None, "same as stack", True)
            self.input_mesh_size_same.valueChanged.connect(changed_same_as)
            self.input_mesh_size_x = QtShortCuts.QInputNumber(None, "mesh size x", 200, step=1)
            self.input_mesh_size_y = QtShortCuts.QInputNumber(None, "y", 200, step=1)
            self.input_mesh_size_z = QtShortCuts.QInputNumber(None, "z", 200, step=1)
            self.input_mesh_size_label = QtWidgets.QLabel("μm").addToLayout()

            self.deformation_detector.mesh_size_changed.connect(deformation_detector_mesh_size_changed)
            changed_same_as()
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
        if self.deformation_detector is None:
            return
        M = self.deformation_detector.M1
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

        self.material_parameters = QtWidgets.QGroupBox("Material Parameters")
        main_layout.addWidget(self.material_parameters)
        with QtShortCuts.QVBoxLayout(self.material_parameters) as layout:
            with QtShortCuts.QHBoxLayout(None) as layout2:
                self.input_k = QtShortCuts.QInputString(None, "k", "1645", type=float)
                self.input_d0 = QtShortCuts.QInputString(None, "d0", "0.0008", type=float)
            with QtShortCuts.QHBoxLayout(None) as layout2:
                self.input_lamda_s = QtShortCuts.QInputString(None, "lamdba_s", "0.0075", type=float)
                self.input_ds = QtShortCuts.QInputString(None, "ds", "0.033", type=float)

        self.material_parameters = QtWidgets.QGroupBox("Regularisation Parameters")
        main_layout.addWidget(self.material_parameters)
        with QtShortCuts.QVBoxLayout(self.material_parameters) as layout:
            self.input_alpha = QtShortCuts.QInputString(None, "alpha", "9", type=float)
            with QtShortCuts.QHBoxLayout(None) as layout:
                self.input_stepper = QtShortCuts.QInputString(None, "stepper", "0.33", type=float)
                self.input_imax = QtShortCuts.QInputNumber(None, "i_max", 100, float=False)

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

        self.tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(self.tabs)


        self.tab_stack = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_stack, "Stack")
        v_layout = QtWidgets.QVBoxLayout(self.tab_stack)
        h_layout = QtWidgets.QHBoxLayout()
        v_layout.addLayout(h_layout)

        self.stack_before = StackSelector(h_layout, "deformed")
        self.stack_after = StackSelector(h_layout, "relaxed", self.stack_before)
        self.description = QtWidgets.QTextEdit()
        self.description.setDisabled(True)
        h_layout.addWidget(self.description)
        self.description.setText("""
<h1>Step 1: Load the Image Stacks</h1>
First we need to load the image data of the recorded stacks.<br/>
<br/>
Select the image stacks for the <b>deformed</b> and the <b>relaxed</b> state. In the deformed state the cell should pull on the 
fibers and in the relaxed state there should be no cell or a non pulling cell (e.g. treated with CytoD).<br/>
<br/>
Images can either be a single Leica file or a .tif file with a naming sceme that allows to find the other images of the 
stack in the same folder.<br/>
<br/>
You can select which image <b>channel</b> to use for evaluation.<br/>
<br/>
If you load a custom stack, you need to specifiy which dimension is the <b>z-component</b>.
And you need to supply a <b>voxel size</b>.<br/>
<br/>
<br/>
<b>Hint</b>: you can optionally <b>export</b> the current z slices as images or as a gif. You can also right click and drag on 
the slider to make a <b>z-projection</b> over the selected z range.
         """.strip())
#TODO make image loader also with wildcard

        v_layout.addWidget(QHLine())
        h_layout = QtWidgets.QHBoxLayout()
        v_layout.addLayout(h_layout)

        h_layout.addStretch()
        self.button_next = QtWidgets.QPushButton("next")
        self.button_next.clicked.connect(self.next)
        h_layout.addWidget(self.button_next)

        """ """
        self.tab_stack = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_stack, "Deformation")
        v_layout = QtWidgets.QVBoxLayout(self.tab_stack)
        h_layout = QtWidgets.QHBoxLayout()
        v_layout.addLayout(h_layout)

        self.deformation = DeformationDetector(h_layout, self.stack_before, self.stack_after)
        self.description = QtWidgets.QTextEdit()
        self.description.setDisabled(True)
        h_layout.addWidget(self.description)
        self.description.setText("""
<h1>Step 2: Deformation Detection</h1>
Now we need to extract the deformations from the recorded image data.<br/>
<br/>
This part uses a 3D PIV (particle image velocimetry) algorithm to determine how much each voxel has shifted.

<h2>Parameters</h2>
<ul>
<li><b>Overlap</b>: the fraction by wich two neighbouring windows overlap.
Higher overlap increases the resolution (and calculation time). Default is 0.6.
</li>
<li><b>Window size</b>: 
A cube of this size (μm) will be shifted in space until the best match is found. 
The window size should be bigger
than the expected deformation. Default is 30μm.</li>
<li><b>Signoise</b>: above which signal to noice level to discatd deformations.</li>
<li><b>Driftcorrection</b>: whether to perform drift correction or not.</li>
</ul>
                 """.strip())
# TODO: check the error message for Out of Memory

        v_layout.addWidget(QHLine())
        h_layout = QtWidgets.QHBoxLayout()
        v_layout.addLayout(h_layout)

        h_layout.addStretch()
        self.button_next = QtWidgets.QPushButton("next")
        self.button_next.clicked.connect(self.next)
        self.button_previous = QtWidgets.QPushButton("back")
        self.button_previous.clicked.connect(self.previous)
        h_layout.addWidget(self.button_previous)
        h_layout.addWidget(self.button_next)

        """ """
        self.tab_stack = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_stack, "Mesh")
        v_layout = QtWidgets.QVBoxLayout(self.tab_stack)
        h_layout = QtWidgets.QHBoxLayout()
        v_layout.addLayout(h_layout)

        self.mesh = MeshCreator(h_layout, self.deformation)
        self.description = QtWidgets.QTextEdit()
        self.description.setDisabled(True)
        h_layout.addWidget(self.description)
        self.description.setText("""
        <h1>Step 3: Mesh Generation</h1>
        Now we need to create a mesh and interpolate the displacements onto this mesh.<br/>
        <br/>

        <h2>Parameters</h2>
        <ul>
        <li><b>Element size</b>: spacing between nodes of the mesh in μm.</li>
        <li><b>Inner Region</b>: the spacing will be exactly the one given in element size in a centerl region of this size.</li>
        <li><b>Thinning Factor</b>: how much to thin the elements outside the inner region. 1</li>
        </ul>
                         """.strip())
#TODO add better description of the thining
        v_layout.addWidget(QHLine())
        h_layout = QtWidgets.QHBoxLayout()
        v_layout.addLayout(h_layout)

        h_layout.addStretch()
        self.button_next = QtWidgets.QPushButton("next")
        self.button_next.clicked.connect(self.next)
        self.button_previous = QtWidgets.QPushButton("back")
        self.button_previous.clicked.connect(self.previous)
        h_layout.addWidget(self.button_previous)
        h_layout.addWidget(self.button_next)

        """ """

        self.tab_stack = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_stack, "Regularisation")
        v_layout = QtWidgets.QVBoxLayout(self.tab_stack)
        h_layout = QtWidgets.QHBoxLayout()
        v_layout.addLayout(h_layout)

        self.regularizer = Regularizer(h_layout, self.mesh)
        self.description = QtWidgets.QTextEdit()
        self.description.setDisabled(True)
        h_layout.addWidget(self.description)
        self.description.setText("""
                <h1>Step 4: Force Regularisation</h1>
                Now we need to calculate the forces with a regulraisation method.<br/>
                <br/>
                <h2>Parameters</h2>
                <ul>
                <li><b>Alpha</b>: how strong the regularisation should be. Default value is 9.</li>
                <li><b>Stepper</b>: how wide to step after each iteration.</li>
                <li><b>imax</b>: the maximum number of iterations.</li>
                </ul>
                                 """.strip())

        v_layout.addWidget(QHLine())
        h_layout = QtWidgets.QHBoxLayout()
        v_layout.addLayout(h_layout)

        h_layout.addStretch()
        self.button_next = QtWidgets.QPushButton("next")
        self.button_next.clicked.connect(self.next)
        self.button_previous = QtWidgets.QPushButton("back")
        self.button_previous.clicked.connect(self.previous)
        h_layout.addWidget(self.button_previous)
        h_layout.addWidget(self.button_next)
        self.button_next.setDisabled(True)

    def load(self):
        files = glob.glob(self.input_filename.value())
        self.input_label.setText("\n".join(files))
#        self.input_filename

    def next(self):
        self.tabs.setCurrentIndex(self.tabs.currentIndex()+1)

    def previous(self):
        self.tabs.setCurrentIndex(self.tabs.currentIndex()-1)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    print(sys.argv)
    window = MainWindow()
    if len(sys.argv) >= 2:
        window.loadFile(sys.argv[1])
    window.show()
    sys.exit(app.exec_())
