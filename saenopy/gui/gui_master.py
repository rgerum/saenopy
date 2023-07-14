import sys
import traceback

from qtpy import QtCore, QtWidgets, QtGui
import multiprocessing
# keep import for pyinstaller
import skimage.filters.ridges
import skimage.filters.thresholding

from saenopy.gui.common import QtShortCuts
from saenopy.gui.solver.gui_solver import MainWindowSolver as SolverMain
from saenopy.gui.spheroid.gui_deformation_spheroid import MainWindow as SpheroidMain
from saenopy.gui.orientation.gui_orientation import MainWindow as OrientationMain
from saenopy.gui.code.gui_code import MainWindowCode
from saenopy.gui.material_fit.gui_fit import MainWindowFit
from saenopy.gui.common.resources import resource_path, resource_icon


class InfoBox(QtWidgets.QWidget):
    def __init__(self, name, func):
        super().__init__()
        self.setMinimumWidth(200)
        self.setMaximumHeight(500)
        with QtShortCuts.QHBoxLayout(self) as l:
            with QtShortCuts.QGroupBox(l, name):
                with QtShortCuts.QVBoxLayout() as l2:
                    if name == "Solver":
                        self.text = QtWidgets.QLabel("Calculate the forces from a\n3D stack or a series of 3D stacks.").addToLayout()
                    elif name == "Spheroid":
                        self.text = QtWidgets.QLabel("Calculate the forces of\nmulticellular aggregates\nfrom a timeseries of 2D images.").addToLayout()
                    else:
                        self.text = QtWidgets.QLabel("Measure the orientations\nof fiberes in 2D images.\n\nAs a proxy for contractility.").addToLayout()
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
                        InfoBox("Solver", lambda: self.setTab(2)).addToLayout()
                        layout2.addStretch()
                        InfoBox("Spheroid", lambda: self.setTab(3)).addToLayout()
                        layout2.addStretch()
                        InfoBox("Orientation", lambda: self.setTab(4)).addToLayout()
                        layout2.addStretch()
                    layout.addStretch()
                with self.tabs.createTab("Material Fit") as self.layout_code:
                    QtShortCuts.current_layout.setContentsMargins(0, 0, 0, 0)
                    self.fitter = MainWindowFit().addToLayout()

                with self.tabs.createTab("Solver") as self.layout_solver:
                    QtShortCuts.current_layout.setContentsMargins(0, 0, 0, 0)

                with self.tabs.createTab("Spheroid") as self.layout_spheroid:
                    QtShortCuts.current_layout.setContentsMargins(0, 0, 0, 0)

                with self.tabs.createTab("Orientation") as self.layout_orientation:
                    QtShortCuts.current_layout.setContentsMargins(0, 0, 0, 0)

                with self.tabs.createTab("Code") as self.layout_code:
                    QtShortCuts.current_layout.setContentsMargins(0, 0, 0, 0)
                    self.coder = MainWindowCode().addToLayout()

        #self.tabs.setCurrentIndex(self.settings.value("master_tab", 0))
        self.first_tab_change = False

    first_tab_change = True
    solver = None
    spheroid = None
    orientation = None
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

    while True:
        try:
            res = app.exec_()
            break
        except Exception as err:
            traceback.print_traceback(err)
            QtWidgets.QMessageBox.critical(window, "Error", f"An Error occurred:\n{err}")
            continue
    sys.exit(res)


if __name__ == '__main__':  # pragma: no cover
    # On Windows calling this function is necessary.
    multiprocessing.freeze_support()

    if len(sys.argv) >= 2 and sys.argv[1].endswith(".py"):
        source = open(sys.argv[1]).read()
        code = compile(source, sys.argv[1], 'exec')
        exec(code)
        exit(0)


    """ some magic to prevent PyQt5 from swallowing exceptions """
    # Back up the reference to the exceptionhook
    sys._excepthook = sys.excepthook
    # Set the exception hook to our wrapping function
    sys.excepthook = lambda *args: sys._excepthook(*args)

    for arg in sys.argv:
        if arg == "--demo":
            import os
            os.environ["DEMO"] = "true"

    main()
