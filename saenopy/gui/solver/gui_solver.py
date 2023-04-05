import sys
from qtpy import QtCore, QtWidgets

from saenopy.gui.common import QtShortCuts
from saenopy.gui.common.resources import resource_icon
from saenopy.gui.solver.analyze.plot_window import PlottingWindow
from saenopy.gui.solver.modules.BatchEvaluate import BatchEvaluate


class MainWindowSolver(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # QSettings
        self.settings = QtCore.QSettings("Saenopy", "Saenopy")

        main_layout = QtWidgets.QHBoxLayout(self)

        with QtShortCuts.QTabWidget(main_layout) as self.tabs:
            with self.tabs.createTab("Analyse Measurements"):
                with QtShortCuts.QHBoxLayout():
                    self.deformations = BatchEvaluate(self)
                    QtShortCuts.current_layout.addWidget(self.deformations)

            with self.tabs.createTab("Data Analysis"):
                with QtShortCuts.QHBoxLayout():
                    self.plotting_window = PlottingWindow(self).addToLayout()


if __name__ == '__main__':  # pragma: no cover
    app = QtWidgets.QApplication(sys.argv)
    if sys.platform.startswith('win'):
        import ctypes
        myappid = 'fabrylab.saenopy.master'  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    print(sys.argv)
    window = MainWindowSolver()
    window.setMinimumWidth(1600)
    window.setMinimumHeight(900)
    window.setWindowTitle("Saenopy Viewer")
    window.setWindowIcon(resource_icon("Icon.ico"))
    window.show()
    sys.exit(app.exec_())
