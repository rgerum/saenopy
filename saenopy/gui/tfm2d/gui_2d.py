import sys
from qtpy import QtCore, QtWidgets

from saenopy.gui.common import QtShortCuts
from saenopy.gui.common.resources import resource_icon
from saenopy.gui.tfm2d.analyze.PlottingWindow import PlottingWindow
from saenopy.gui.tfm2d.modules.BatchEvaluate import BatchEvaluate


class MainWindow2D(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # QSettings
        self.settings = QtCore.QSettings("Saenopy", "Saenopy")

        main_layout = QtWidgets.QHBoxLayout(self)

        with QtShortCuts.QTabWidget(main_layout) as self.tabs:
            with self.tabs.createTab("Analyse Measurements"):
                with QtShortCuts.QHBoxLayout():
                    self.deformations = BatchEvaluate(self)
                    QtShortCuts.currentLayout().addWidget(self.deformations)

            with self.tabs.createTab("Data Analysis"):
                with QtShortCuts.QHBoxLayout():
                    self.plotting_window = PlottingWindow(self, self.deformations).addToLayout()


if __name__ == '__main__':  # pragma: no cover
    app = QtWidgets.QApplication(sys.argv)
    if sys.platform.startswith('win'):
        import ctypes
        myappid = 'fabrylab.saenopy.master'  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    print(sys.argv)
    window = MainWindow2D()
    window.setMinimumWidth(1600)
    window.setMinimumHeight(900)
    window.setWindowTitle("Saenopy Viewer")
    window.setWindowIcon(resource_icon("Icon.ico"))
    window.show()
    sys.exit(app.exec_())
