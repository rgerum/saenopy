import json
import sys
import traceback

from qtpy import QtCore, QtWidgets, QtGui
import multiprocessing
# keep import for pyinstaller
import skimage.exposure.exposure
import skimage.filters.ridges
import skimage.filters.thresholding

from saenopy.gui.common import QtShortCuts
from saenopy.gui.solver.gui_solver import MainWindowSolver as SolverMain
from saenopy.gui.spheroid.gui_deformation_spheroid import MainWindow as SpheroidMain
from saenopy.gui.orientation.gui_orientation import MainWindow as OrientationMain
from saenopy.gui.code.gui_code import MainWindowCode
from saenopy.gui.material_fit.gui_fit import MainWindowFit
from saenopy.gui.tfm2d.gui_2d import MainWindow2D
from saenopy.gui.common.resources import resource_path, resource_icon


class InfoBox(QtWidgets.QWidget):
    def __init__(self, name, func):
        super().__init__()
        self.setMinimumWidth(200)
        self.setMaximumHeight(500)
        with QtShortCuts.QHBoxLayout(self) as l:
            with QtShortCuts.QGroupBox(l, name):
                with QtShortCuts.QVBoxLayout() as l2:
                    if name == "3D TFM":
                        self.text = QtWidgets.QLabel("Calculate the forces from a\n3D stack or a series of 3D stacks.").addToLayout()
                    elif name == "2.5D Spheroid":
                        self.text = QtWidgets.QLabel("Calculate the forces of multicellular\naggregates from a timeseries of\n2D images in 3D matrices.").addToLayout()
                    elif name == "2D TFM":
                        self.text = QtWidgets.QLabel("Calculate the forces from\n2D images using PyTFM.").addToLayout()
                    else:
                        self.text = QtWidgets.QLabel("Measure the orientation\nof matrix fibers in 2D images as\na proxy for cellular force.").addToLayout()
                    self.button1 = QtShortCuts.QPushButton(None, name, func)


class MainWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # QSettings
        self.settings = QtCore.QSettings("Saenopy", "Saenopy")

        self.setMinimumWidth(800)
        self.setMinimumHeight(400)
        self.setWindowTitle("Saenopy")
        self.setWindowIcon(resource_icon("Icon.ico"))

        with QtShortCuts.QHBoxLayout(self) as main_layout:
            main_layout.setContentsMargins(0, 0, 0, 0)
            with QtShortCuts.QTabWidget(main_layout) as self.tabs:
                self.tabs.currentChanged.connect(self.changedTab)
                self.tabs.setTabPosition(QtWidgets.QTabWidget.West)
                with self.tabs.createTab("Home") as layout:
                    layout.addStretch()
                    with QtShortCuts.QHBoxLayout() as layout2:
                        layout.addStretch()
                        self.image = QtWidgets.QLabel("x").addToLayout()
                        self.image_timer = QtCore.QTimer()
                        timer_index = 0
                        def timer():
                            nonlocal timer_index
                            if timer_index == 14:
                                #self.image.setPixmap(QtGui.QPixmap(resource_path("Logo.png")))
                                self.image_timer.stop()
                                return
                            self.image.setPixmap(QtGui.QPixmap(resource_path(f"animation/frame{timer_index:02d}.png")))
                            timer_index += 1
                        self.image_timer.timeout.connect(timer)
                        self.image_timer.start(100)
                        timer()
                        #self.image.setPixmap(QtGui.QPixmap(resource_path("Logo.png")))
                        self.image.setScaledContents(True)
                        self.image.setMaximumWidth(400)
                        self.image.setMaximumHeight(200)
                        layout.addStretch()
                    with QtShortCuts.QHBoxLayout() as layout2:
                        layout2.addStretch()
                        InfoBox("3D TFM", lambda: self.setTab(2)).addToLayout()
                        layout2.addStretch()
                        InfoBox("2.5D Spheroid", lambda: self.setTab(3)).addToLayout()
                        layout2.addStretch()
                        InfoBox("2.5D Orientation", lambda: self.setTab(4)).addToLayout()
                        layout2.addStretch() 
                        InfoBox("2D TFM", lambda: self.setTab(5)).addToLayout()
                        layout2.addStretch()
                    layout.addStretch()
                with self.tabs.createTab("Material Fit") as self.layout_code:
                    QtShortCuts.currentLayout().setContentsMargins(0, 0, 0, 0)
                    self.fitter = MainWindowFit().addToLayout()

                with self.tabs.createTab("3D TFM") as self.layout_solver:
                    QtShortCuts.currentLayout().setContentsMargins(0, 0, 0, 0)

                with self.tabs.createTab("2.5D Spheroid") as self.layout_spheroid:
                    QtShortCuts.currentLayout().setContentsMargins(0, 0, 0, 0)

                with self.tabs.createTab("2.5D Orientation") as self.layout_orientation:
                    QtShortCuts.currentLayout().setContentsMargins(0, 0, 0, 0)

                with self.tabs.createTab("2D TFM") as self.layout_pytfm:
                    QtShortCuts.currentLayout().setContentsMargins(0, 0, 0, 0)

                with self.tabs.createTab("Code") as self.layout_code:
                    QtShortCuts.currentLayout().setContentsMargins(0, 0, 0, 0)
                    self.coder = MainWindowCode().addToLayout()

        #self.tabs.setCurrentIndex(self.settings.value("master_tab", 0))
        self.first_tab_change = False

        for file in sys.argv[1:]:
            print(file)
            if file.endswith(".json"):
                try:
                    with open(file, "r") as fp:
                        data = json.load(fp)
                    filename = data[0]["paths"][0]["path"]
                    if filename.endswith(".saenopy"):
                        self.setTab(2)
                        self.solver.tabs.setCurrentIndex(1)
                        self.solver.plotting_window.load(file)
                    elif filename.endswith("saenopy2D"):
                        self.setTab(5)
                        self.pytfm2d.tabs.setCurrentIndex(1)
                        self.pytfm2d.plotting_window.load(file)
                    elif filename.endswith(".saenopySpheroid"):
                        self.setTab(3)
                        self.spheroid.tabs.setCurrentIndex(1)
                        self.spheroid.plotting_window.load(file)
                    elif filename.endswith(".saenopyOrientation"):
                        self.setTab(4)
                        self.orientation.tabs.setCurrentIndex(1)
                        self.orientation.plotting_window.load(file)
                    continue
                except (IndexError, KeyError):
                    continue
            if file.endswith(".py"):
                self.setTab(6)
            elif file.endswith(".saenopy"):
                self.setTab(2)
            elif file.endswith(".saenopy2D"):
                self.setTab(5)
            elif file.endswith(".saenopySpheroid"):
                self.setTab(3)
            elif file.endswith(".saenopyOrientation"):
                self.setTab(4)
            else:
                raise ValueError("Unknown file type")

    first_tab_change = True
    solver = None
    spheroid = None
    orientation = None
    pytfm2d = None
    def changedTab(self, value):
        if self.first_tab_change is False:
            self.settings.setValue("master_tab", value)

        if value == 2 and self.solver is None:
            self.solver = SolverMain().addToLayout(self.layout_solver)
            self.setMinimumWidth(1600)
            self.setMinimumHeight(900)
        if value == 3 and self.spheroid is None:  # pragma: no cover
            self.spheroid = SpheroidMain().addToLayout(self.layout_spheroid)
            self.setMinimumWidth(1600)
            self.setMinimumHeight(900)
        if value == 4 and self.orientation is None:  # pragma: no cover
            self.orientation = OrientationMain().addToLayout(self.layout_orientation)
            self.setMinimumWidth(1600)
            self.setMinimumHeight(900)
        if value == 5 and self.pytfm2d is None:  # pragma: no cover
            self.pytfm2d = MainWindow2D().addToLayout(self.layout_pytfm)
            self.setMinimumWidth(1600)
            self.setMinimumHeight(900)

    def setTab(self, value):
        self.tabs.setCurrentIndex(value)


def main():  # pragma: no cover
    app = QtWidgets.QApplication(sys.argv)
    if sys.platform.startswith('win'):
        import ctypes
        myappid = 'fabrylab.saenopy.master'  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    window = MainWindow()
    window.show()
    try:
        import pyi_splash

        # Update the text on the splash screen
        pyi_splash.update_text("PyInstaller is a great software!")
        pyi_splash.update_text("Second time's a charm!")

        # Close the splash screen. It does not matter when the call
        # to this function is made, the splash screen remains open until
        # this function is called or the Python program is terminated.
        pyi_splash.close()
    except (ImportError, RuntimeError):
        pass

    from traceback import format_exception
    def except_hook(type_, value, tb):
        print(*format_exception(type_, value, tb), file=sys.stderr)
        QtWidgets.QMessageBox.critical(window, "Error", f"An Error occurred:\n{type_.__name__}: {value}")
        return

    sys.excepthook = except_hook

    res = app.exec_()
    sys.exit(res)


if __name__ == '__main__':  # pragma: no cover
    # On Windows calling this function is necessary.
    multiprocessing.freeze_support()

    if len(sys.argv) >= 3 and sys.argv[1] == "run" and sys.argv[2].endswith(".py"):
        source = open(sys.argv[2]).read()
        code = compile(source, sys.argv[1], 'exec')
        exec(code)
        exit(0)

    for arg in sys.argv:
        if arg == "--demo":
            import os
            os.environ["DEMO"] = "true"

    main()
