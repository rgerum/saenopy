import sys
from qtpy import QtCore, QtWidgets, QtGui
import multiprocessing

from saenopy.gui.common import QtShortCuts
from saenopy.gui_deformation_whole2 import MainWindow as SolverMain
from saenopy.gui_deformation_spheriod import MainWindow as SpheriodMain
from saenopy.gui_orientation import MainWindow as OrientationMain
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
                    elif name == "Spheriod":
                        self.text = QtWidgets.QLabel("Calculate the forces of\nmulticellular aggregates\nfrom a timeseries of 2D images.").addToLayout()
                    else:
                        self.text = QtWidgets.QLabel("Measure the orientations\nof fiberes in 2D images.\n\nAs a proxy for contractility.").addToLayout()
                    self.button1 = QtShortCuts.QPushButton(None, name, func)


class MainWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # QSettings
        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

        self.setMinimumWidth(800)
        self.setMinimumHeight(400)
        self.setWindowTitle("Saenopy")
        self.setWindowIcon(resource_icon("Icon.ico"))

        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

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
                        self.image.setPixmap(QtGui.QPixmap(resource_path("Logo.png")))
                        self.image.setScaledContents(True)
                        self.image.setMaximumWidth(400)
                        self.image.setMaximumHeight(200)
                        layout.addStretch()
                    with QtShortCuts.QHBoxLayout() as layout2:
                        layout2.addStretch()
                        InfoBox("Solver", lambda: self.setTab(1)).addToLayout()
                        layout2.addStretch()
                        InfoBox("Spheriod", lambda: self.setTab(2)).addToLayout()
                        layout2.addStretch()
                        InfoBox("Orientation", lambda: self.setTab(3)).addToLayout()
                        layout2.addStretch()
                    layout.addStretch()
                with self.tabs.createTab("Solver") as self.layout_solver:
                    self.layout_solver.setContentsMargins(0, 0, 0, 0)
                with self.tabs.createTab("Spheriod") as self.layout_spheriod:
                    self.layout_spheriod.setContentsMargins(0, 0, 0, 0)
                with self.tabs.createTab("Orientation") as self.layout_orientation:
                    self.layout_orientation.setContentsMargins(0, 0, 0, 0)

        #self.tabs.setCurrentIndex(self.settings.value("master_tab", 0))
        self.first_tab_change = False

    first_tab_change = True
    solver = None
    spheriod = None
    orientation = None
    def changedTab(self, value):
        if self.first_tab_change is False:
            self.settings.setValue("master_tab", value)

        if value == 1 and self.solver is None:
            self.solver = SolverMain().addToLayout(self.layout_solver)
            self.setMinimumWidth(1600)
            self.setMinimumHeight(900)
        if value == 2 and self.spheriod is None:
            self.spheriod = SpheriodMain().addToLayout(self.layout_spheriod)
            self.setMinimumWidth(1600)
            self.setMinimumHeight(900)
        if value == 3 and self.orientation is None:
            self.orientation = OrientationMain().addToLayout(self.layout_orientation)
            self.setMinimumWidth(1600)
            self.setMinimumHeight(900)

    def setTab(self, value):
        self.tabs.setCurrentIndex(value)


def main():
    app = QtWidgets.QApplication(sys.argv)
    if sys.platform.startswith('win'):
        import ctypes
        myappid = 'fabrylab.saenopy.master'  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    print(sys.argv)
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
    sys.exit(app.exec_())


if __name__ == '__main__':
    # On Windows calling this function is necessary.
    multiprocessing.freeze_support()

    main()
