import sys
from qtpy import QtCore, QtWidgets

from saenopy.gui.common import QtShortCuts
from saenopy.gui.common.resources import resource_icon
from saenopy.gui.solver.analyze.plot_window import PlottingWindow
from saenopy.gui.solver.modules.BatchEvaluate import BatchEvaluate


class MainWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # QSettings
        self.settings = QtCore.QSettings("Saenopy", "Saenopy")

        self.setMinimumWidth(1600)
        self.setMinimumHeight(900)
        self.setWindowTitle("Saenopy Viewer")
        self.setWindowIcon(resource_icon("Icon.ico"))

        main_layout = QtWidgets.QHBoxLayout(self)

        with QtShortCuts.QTabWidget(main_layout) as self.tabs:
            """ """
            with self.tabs.createTab("Analyse Measurements") as v_layout:
                with QtShortCuts.QHBoxLayout() as h_layout:
                    # self.deformations = Deformation(h_layout, self)
                    self.deformations = BatchEvaluate(self)
                    h_layout.addWidget(self.deformations)
                    if 0:
                        self.description = QtWidgets.QTextEdit()
                        self.description.setDisabled(True)
                        self.description.setMaximumWidth(300)
                        h_layout.addWidget(self.description)
                        self.description.setText("""
                        <h1>Start Evaluation</h1>
                         """.strip())
                #v_layout.addWidget(QHLine())
                #with QtShortCuts.QHBoxLayout() as h_layout:
                #    h_layout.addStretch()
                    #self.button_previous = QtShortCuts.QPushButton(None, "back", self.previous)
                    #self.button_next = QtShortCuts.QPushButton(None, "next", self.next)
            with self.tabs.createTab("Data Analysis") as v_layout:
                with QtShortCuts.QHBoxLayout() as h_layout:
                    # self.deformations = Deformation(h_layout, self)
                    self.plotting_window = PlottingWindow(self).addToLayout()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    if sys.platform.startswith('win'):
        import ctypes
        myappid = 'fabrylab.saenopy.master'  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    print(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
