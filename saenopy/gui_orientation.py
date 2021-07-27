import sys

# Setting the Qt bindings for QtPy
import os

import pandas as pd
import qtawesome as qta

os.environ["QT_API"] = "pyqt5"

from qtpy import QtCore, QtWidgets, QtGui

import numpy as np

from saenopy.gui import QtShortCuts, QExtendedGraphicsView
import imageio
from qimage2ndarray import array2qimage
import matplotlib.pyplot as plt
import glob
import imageio
import threading
import glob
import re


if QtCore.qVersion() >= "5.":
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

import urllib
from pathlib import Path

""" some magic to prevent PyQt5 from swallowing exceptions """
# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook
# Set the exception hook to our wrapping function
sys.excepthook = lambda *args: sys._excepthook(*args)

import ctypes

from qtpy import API_NAME as QT_API_NAME
if QT_API_NAME.startswith("PyQt4"):
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as Canvas
    from matplotlib.backends.backend_qt4agg import FigureManager
    from matplotlib.backends.backend_qt4 import NavigationToolbar2QT as NavigationToolbar
else:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
    from matplotlib.backends.backend_qt5agg import FigureManager
    from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar


class MatplotlibWidget(Canvas):

    def __init__(self, parent=None, width=4, height=3, dpi=100):
        from matplotlib import _pylab_helpers
        plt.ioff()
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.figure.patch.set_facecolor([0, 1, 0, 0])
        self.axes = self.figure.add_subplot(111)

        Canvas.__init__(self, self.figure)
        self.setStyleSheet("background-color:transparent;")
        self.setParent(parent)

        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)

        self.manager = FigureManager(self, 1)
        self.manager._cidgcf = self.figure

        """
        _pylab_helpers.Gcf.figs[num] = canvas.manager
        # get the canvas of the figure
        manager = _pylab_helpers.Gcf.figs[num]
        # set the size if it is defined
        if figsize is not None:
            _pylab_helpers.Gcf.figs[num].window.setGeometry(100, 100, figsize[0] * 80, figsize[1] * 80)
        # set the figure as the active figure
        _pylab_helpers.Gcf.set_active(manager)
        """
        _pylab_helpers.Gcf.set_active(self.manager)

    def setActive(self):
        from matplotlib import _pylab_helpers
        self.manager._cidgcf = self.figure
        _pylab_helpers.Gcf.set_active(self.manager)

def execute(func, *args, **kwargs):
    func(*args, **kwargs)
    import inspect
    code_lines = inspect.getsource(func).split("\n")[1:]
    indent = len(code_lines[0]) - len(code_lines[0].lstrip())
    code = "\n".join(line[indent:] for line in code_lines)
    for key, value in kwargs.items():
        if isinstance(value, str):
            code = code.replace(key, "'"+value+"'")
        else:
            code = code.replace(key, str(value))
    code = code.replace("self.canvas.draw()", "plt.show()")
    return code

def kill_thread(thread):
    """
    thread: a threading.Thread object
    """
    thread_id = thread.ident
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(SystemExit))
    if res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
        print('Exception raise failure')

class QHLine(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)

class QVLine(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.VLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)

class Spoiler(QtWidgets.QWidget):
    def __init__(self, parent=None, title='', animationDuration=300):
        """
        References:
            # Adapted from c++ version
            http://stackoverflow.com/questions/32476006/how-to-make-an-expandable-collapsable-section-widget-in-qt
        """
        super(Spoiler, self).__init__(parent=parent)

        self.animationDuration = animationDuration
        self.toggleAnimation = QtCore.QParallelAnimationGroup()
        self.contentArea = QtWidgets.QScrollArea()
        self.headerLine = QtWidgets.QFrame()
        self.toggleButton = QtWidgets.QToolButton()
        self.mainLayout = QtWidgets.QGridLayout()

        toggleButton = self.toggleButton
        toggleButton.setStyleSheet("QToolButton { border: none; }")
        toggleButton.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        toggleButton.setArrowType(QtCore.Qt.RightArrow)
        toggleButton.setText(str(title))
        toggleButton.setCheckable(True)
        toggleButton.setChecked(False)

        headerLine = self.headerLine
        headerLine.setFrameShape(QtWidgets.QFrame.HLine)
        headerLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        headerLine.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)

        self.contentArea.setStyleSheet("QScrollArea { background-color: white; border: none; }")
        self.contentArea.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        # start out collapsed
        self.contentArea.setMaximumHeight(0)
        self.contentArea.setMinimumHeight(0)
        # let the entire widget grow and shrink with its content
        toggleAnimation = self.toggleAnimation
        toggleAnimation.addAnimation(QtCore.QPropertyAnimation(self, b"minimumHeight"))
        toggleAnimation.addAnimation(QtCore.QPropertyAnimation(self, b"maximumHeight"))
        toggleAnimation.addAnimation(QtCore.QPropertyAnimation(self.contentArea, b"maximumHeight"))
        # don't waste space
        mainLayout = self.mainLayout
        mainLayout.setVerticalSpacing(0)
        mainLayout.setContentsMargins(0, 0, 0, 0)
        row = 0
        mainLayout.addWidget(self.toggleButton, row, 0, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(self.headerLine, row, 2, 1, 1)
        row += 1
        mainLayout.addWidget(self.contentArea, row, 0, 1, 3)
        self.setLayout(self.mainLayout)

        def start_animation(checked):
            arrow_type = QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow
            direction = QtCore.QAbstractAnimation.Forward if checked else QtCore.QAbstractAnimation.Backward
            toggleButton.setArrowType(arrow_type)
            self.toggleAnimation.setDirection(direction)
            self.toggleAnimation.start()

        self.toggleButton.clicked.connect(start_animation)

    def setContentLayout(self, contentLayout):
        # Not sure if this is equivalent to self.contentArea.destroy()
        self.contentArea.destroy()
        self.contentArea.setLayout(contentLayout)
        collapsedHeight = self.sizeHint().height() - self.contentArea.maximumHeight()
        contentHeight = contentLayout.sizeHint().height()
        for i in range(self.toggleAnimation.animationCount()-1):
            spoilerAnimation = self.toggleAnimation.animationAt(i)
            spoilerAnimation.setDuration(self.animationDuration)
            spoilerAnimation.setStartValue(collapsedHeight)
            spoilerAnimation.setEndValue(collapsedHeight + contentHeight)
        contentAnimation = self.toggleAnimation.animationAt(self.toggleAnimation.animationCount() - 1)
        contentAnimation.setDuration(self.animationDuration)
        contentAnimation.setStartValue(0)
        contentAnimation.setEndValue(contentHeight)


class CheckAbleGroup(QtWidgets.QWidget, QtShortCuts.EnterableLayout):
    value_changed = QtCore.Signal(bool)
    main_layout = None
    def __init__(self, parent=None, title='', animationDuration=300):
        super().__init__(parent=parent)

        self.headerLine = QtWidgets.QFrame()
        self.checkbox = QtWidgets.QCheckBox()
        self.toggleButton = QtWidgets.QToolButton()
        self.mainLayout = QtWidgets.QGridLayout()

        with QtShortCuts.QVBoxLayout(self) as self.main_layout:
            self.main_layout.setContentsMargins(0, 0, 0, 0)
            with QtShortCuts.QHBoxLayout() as layout:
                layout.setContentsMargins(12, 0, 0, 0)
                self.toggleButton = QtShortCuts.QInputBool(None, "", True)
                self.label = QtWidgets.QPushButton(title).addToLayout()
                self.label.setStyleSheet("QPushButton { border: none; }")
                self.label.clicked.connect(self.toggle)

                headerLine = QtWidgets.QFrame().addToLayout()
                headerLine.setFrameShape(QtWidgets.QFrame.HLine)
                headerLine.setFrameShadow(QtWidgets.QFrame.Sunken)
                headerLine.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
            self.child_widget = QtWidgets.QWidget().addToLayout()
        self.layout = self
        self.value = self.toggleButton.value
        #self.setValue = self.toggleButton.setValue
        self.valueChanged = self.toggleButton.valueChanged
        self.toggleButton.valueChanged.connect(self.changedActive)

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setPen(QtGui.QPen(QtGui.QColor("gray")))
        #p.setBrush(QtGui.QBrush(QtGui.QColor("gray")))
        top = 5
        p.drawRect(0, self.height()-1, self.width(), 0)
        p.drawRect(0, top, 0, self.height())
        p.drawRect(0, top, 7, 0)
        p.drawRect(self.width()-1, top, 0, self.height())
        super().paintEvent(ev)

    def toggle(self):
        self.setValue(not self.value())
        self.changedActive()

    def setValue(self, value):
        self.toggleButton.setValue(value)
        self.changedActive()

    def changedActive(self):
        self.value_changed.emit(self.value())
        self.child_widget.setEnabled(self.value())

    def addLayout(self, layout):
        if self.main_layout is None:
            self.setLayout(layout)
        else:
            self.child_widget.setLayout(layout)
        return layout


################

# QSettings
settings = QtCore.QSettings("FabryLab", "CompactionAnalyzer")

class MainWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumWidth(800)
        self.setMinimumHeight(400)
        self.setWindowTitle("CompactionAnalyzer")

        main_layout = QtWidgets.QHBoxLayout(self)

        with QtShortCuts.QTabWidget(main_layout) as self.tabs:

            """ """
            with self.tabs.createTab("Compaction") as v_layout:
                with QtShortCuts.QHBoxLayout() as h_layout:
                    #self.deformations = Deformation(h_layout, self)
                    self.deformations = BatchEvaluate(self)
                    h_layout.addWidget(self.deformations)
                    self.description = QtWidgets.QTextEdit()
                    self.description.setDisabled(True)
                    h_layout.addWidget(self.description)
                    self.description.setText("""
                    <h1>Start Evaluation</h1>
                    
                    As a measure of the contractile strength for cells/organoids in fibrous materials, we quantify the tissue 
                    compaction by the re-orientation of the fiber structure towards the cell centre and 
                    additionally by the increased intensity around the cell.<br/>
                     
                    <h2>Parameters</h2>
                    <ul>
                    <li><b>Fiber Images, Cell Images</b>: Per cell we need to specify paths to an image of the fiber structure & an image of the cell outline. 
                    The *-placeholder enables to read in lists of fiber- and cell-images for batch analysis. </li>
                    <li><b>output</b>: Output folder to store the results (substructure is created automatically).</li>
                    <li><b>scale</b>: Image scale in µm per pixel. </li>
                    <li><b>sigma_tensor</b>: Windowsize, in which individual structure elements are calculated. Should be in the range of the underlying structure and can be optimised per setup by performing a test-series. </li>
                    <li><b>angle_sections</b>: Angle sections around the cell in degree, that can be used for polarity analysis. Default is 5.
                    <li><b>shell_width</b>: Distance shells around the cell for analyzing Intensity & Orientation propagation over distance. Default is 5µm.
                    <li><b>segmentation_thres</b>: Threshold for cell segemntation. Default is 1 (Otsu's threshold).</li>
                    <li><b>seg_gauss1, seg_gauss2</b>: Set of two gauss filters applied before segmentation (bandpass filter, Default is 0.5 and 100). </li>
                    <li><b>sigma_first_blur</b>: Initial slight gaussian blur on the fiber image, before structure analysis is conducted. Default is 0.5. </li>
                    <li><b>edge</b>:  Pixelwidth at the edges that are left blank because no alignment can be calculated at the edges. Default is 40 px. </li>
                    <li><b>Individual Segmentation</b>: If you select this check box, the specified segmentation value here (instead of the global value) will be applied to this individual cell within a batch analysis.</li>
                    </ul>


                     """.strip())
                v_layout.addWidget(QHLine())
                with QtShortCuts.QHBoxLayout() as h_layout:
                    h_layout.addStretch()
                    self.button_previous = QtShortCuts.QPushButton(None, "back", self.previous)
                    self.button_next = QtShortCuts.QPushButton(None, "next", self.next)

            """ """
            with self.tabs.createTab("Data Analysis") as v_layout:
                with QtShortCuts.QHBoxLayout() as h_layout:

                    #self.deformations = Force(h_layout, self)
                    self.deformations = PlottingWindow(self)
                    h_layout.addWidget(self.deformations)

                    self.description = QtWidgets.QTextEdit()
                    self.description.setDisabled(True)
                    h_layout.addWidget(self.description)
                    self.description.setText("""
                            <h1>Data Analysis</h1>
                           
                            In this step we can analyze our results. <br/>
                            <br/>                    
                            
                            For each  <b>individually</b> cell/organoid the <b>global orientation</b> or the
                            <b>normalized intensity</b> within the first distance shell can be inspected. 
                            In addition, the propagation of <b>orientation and intensity
                            over distance </b> can be viewed.  <br/>
                             <br/>
                            Several cells/organoids
                            can be merged in <b>groups</b>, for which then the
                            mean value and standard error of these quantities can be visualized groupwise. <br/>
                            <br/>
                                
                            Different groups can be created and to each group individual experiments 
                            are added by using a path-placeholder combination. In particular,                                            
                            the "*" is used to scan across sub-directories to find different "results_total.xlsx"-files.<br/>                  
                            <i> Example: "C:/User/ExperimentA/well1/Pos*/result.xlsx" to read 
                            in all positions in a certain directory </i><br/>
                            <br/>     
                            
                            Export stores the data as CSV file or an embedded table within a python script 
                            that allows to re-plot and adjust the figures later on. <br/>
                            """)
                v_layout.addWidget(QHLine())
                with QtShortCuts.QHBoxLayout() as h_layout:
                    h_layout.addStretch()
                    self.button_previous = QtShortCuts.QPushButton(None, "back", self.previous)
                    self.button_next = QtShortCuts.QPushButton(None, "next", self.next)

    def load(self):
        files = glob.glob(self.input_filename.value())
        self.input_label.setText("\n".join(files))
#        self.input_filename

    def next(self):
        self.tabs.setCurrentIndex(self.tabs.currentIndex()+1)

    def previous(self):
        self.tabs.setCurrentIndex(self.tabs.currentIndex()-1)





class BatchEvaluate(QtWidgets.QWidget):
    progress_signal = QtCore.Signal(int, int, int, int)
    finished_signal = QtCore.Signal()
    thread = None

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumWidth(800)
        self.setMinimumHeight(400)
        self.setWindowTitle("Saenopy Viewer")

        with QtShortCuts.QHBoxLayout(self) as main_layout:
            with QtShortCuts.QSplitter() as lay:
                with QtShortCuts.QVBoxLayout() as layout:
                    layout.setContentsMargins(0, 0, 0, 0)
                    self.list = ListWidget(layout, add_item_button="add measurements")
                    self.list.addItemClicked.connect(self.show_files)
                    self.list.itemSelectionChanged.connect(self.listSelected)
                    self.progress1 = QtWidgets.QProgressBar()
                    layout.addWidget(self.progress1)

                with QtShortCuts.QVBoxLayout() as layout:
                    self.label_text = QtWidgets.QLabel().addToLayout()

                    self.label = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
                    self.label.setMinimumWidth(300)
                    self.pixmap = QtWidgets.QGraphicsPixmapItem(self.label.origin)
                    self.contour = QtWidgets.QGraphicsPathItem(self.label.origin)
                    pen = QtGui.QPen(QtGui.QColor(255, 0, 0))
                    pen.setCosmetic(True)
                    self.contour.setPen(pen)

                    self.label2 = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
                    #self.label2.setMinimumWidth(300)
                    self.pixmap2 = QtWidgets.QGraphicsPixmapItem(self.label2.origin)

                    self.contour2 = QtWidgets.QGraphicsPathItem(self.label2.origin)
                    self.contour2.setPen(pen)

                    self.label_text2 = QtWidgets.QLabel().addToLayout()
                    self.progress2 = QtWidgets.QProgressBar().addToLayout()

                with QtShortCuts.QVBoxLayout() as layout:

                    self.scale = QtShortCuts.QInputString(None, "scale", "1.0", type=float, settings=settings, settings_key="orientation/scale")
                    with QtShortCuts.QHBoxLayout():
                        self.sigma_tensor = QtShortCuts.QInputString(None, "sigma_tensor", "7.0", type=float, settings=settings, settings_key="orientation/sigma_tensor")
                        self.sigma_tensor_type = QtShortCuts.QInputChoice(None, "", "um", ["um", "pixel"], settings=settings, settings_key="orientation/sigma_tensor_unit")
                    self.edge = QtShortCuts.QInputString(None, "edge", "40", type=int, settings=settings, settings_key="orientation/edge")
                    self.segmention_thres = QtShortCuts.QInputString(None, "segmention_thres", "1.0", type=float, settings=settings, settings_key="orientation/segmention_thres")
                    self.segmention_thres.valueChanged.connect(self.listSelected)
                    self.seg_gaus1 = QtShortCuts.QInputString(None, "seg_gaus1", "0.5", type=float, settings=settings, settings_key="orientation/seg_gaus1")
                    self.seg_gaus1.valueChanged.connect(self.listSelected)
                    self.seg_gaus2 = QtShortCuts.QInputString(None, "seg_gaus2", "100", type=float, settings=settings, settings_key="orientation/seg_gaus2")
                    self.seg_gaus2.valueChanged.connect(self.listSelected)
                    self.sigma_first_blur = QtShortCuts.QInputString(None, "sigma_first_blur", "0.5", type=float, settings=settings, settings_key="orientation/sigma_first_blur")
                    self.angle_sections = QtShortCuts.QInputString(None, "angle_sections", "5", type=int, settings=settings, settings_key="orientation/angle_sections")
                    with QtShortCuts.QHBoxLayout():
                        self.shell_width = QtShortCuts.QInputString(None, "shell_width", "5", type=float, settings=settings, settings_key="orientation/shell_width")
                        self.shell_width_type = QtShortCuts.QInputChoice(None, "", "um", ["um", "pixel"],
                                                                      settings=settings,
                                                                      settings_key="orientation/shell_width_type")

                    with CheckAbleGroup(self, "individual segmentation").addToLayout() as self.individual_data:
                     with QtShortCuts.QVBoxLayout() as layout2:
                        self.segmention_thres_indi = QtShortCuts.QInputString(None, "segmention_thres", None, type=float, allow_none=True)
                        self.segmention_thres_indi.valueChanged.connect(self.listSelected)
                        self.seg_gaus1_indi = QtShortCuts.QInputString(None, "seg_gaus1", None, type=float, allow_none=True)
                        self.seg_gaus1_indi.valueChanged.connect(self.listSelected)
                        self.seg_gaus2_indi = QtShortCuts.QInputString(None, "seg_gaus2", None, type=float, allow_none=True)
                        self.seg_gaus2_indi.valueChanged.connect(self.listSelected)

                    layout.addStretch()

                    self.button_run = QtShortCuts.QPushButton(None, "run", self.run)
        self.images = []
        self.data = []
        self.list.setData(self.data)

        self.input_list = [
            #self.inputText,
            #self.outputText,
            #self.button_clear,
            #self.button_addList,
        ]



        self.individual_data.value_changed.connect(self.changedCheckBox)

        self.progress_signal.connect(self.progress_callback)
        self.finished_signal.connect(self.finished)

    def changedCheckBox(self):
        print("changedCheckBox")
        for widget in [self.segmention_thres, self.seg_gaus1, self.seg_gaus2]:
            widget.setDisabled(self.individual_data.value())
        if not self.individual_data.value():
            for widget in [self.segmention_thres_indi, self.seg_gaus1_indi, self.seg_gaus2_indi]:
                widget.setValue("None")

    def choose_lookup(self):

        self.lookup_gui = SelectLookup()
        self.lookup_gui.exec()

        if self.lookup_gui.result is not None:
            self.lookup_table.setValue(self.lookup_gui.result)

    def show_files(self):

        class AddFilesDialog(QtWidgets.QDialog):
            def __init__(self, parent):
                super().__init__(parent)
                self.setWindowTitle("Add Files")
                with QtShortCuts.QVBoxLayout(self) as layout:
                    self.label = QtWidgets.QLabel(
                        "Select two paths as an input wildcard. Use * to specify a placeholder. One should be for the fiber images and one for the cell images.")
                    layout.addWidget(self.label)

                    self.fiberText = QtShortCuts.QInputFilename(None, "Fiber Images", file_type="Image (*.tif, *.png, *.jpg)", settings=settings,
                                                                settings_key="batch/wildcard_fiber", existing=True,
                                                                allow_edit=True)
                    self.cellText = QtShortCuts.QInputFilename(None, "Cell Images", file_type="Image (*.tif, *.png, *.jpg)", settings=settings,
                                                                settings_key="batch/wildcard_cell", existing=True,
                                                                allow_edit=True)
                    self.outputText = QtShortCuts.QInputFolder(None, "output", settings=settings,
                                                               settings_key="batch/output_folder", allow_edit=True)

                    with QtShortCuts.QHBoxLayout() as layout3:
                        # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                        layout3.addStretch()
                        self.button_addList = QtShortCuts.QPushButton(None, "cancel", self.reject)
                        self.button_addList = QtShortCuts.QPushButton(None, "ok", self.accept)

        dialog = AddFilesDialog(self)
        if not dialog.exec():
            return

        import glob
        import re
        fiber_list_string = os.path.normpath(dialog.fiberText.value())
        cell_list_string = os.path.normpath(dialog.cellText.value())
        output_folder = os.path.normpath(dialog.outputText.value())

        from CompactionAnalyzer.CompactionFunctions import generate_lists
        fiber_list, cell_list, out_list = generate_lists(fiber_list_string, cell_list_string, output_main=output_folder)

        import matplotlib as mpl
        for fiber, cell, out in zip(fiber_list, cell_list, out_list):
            self.list.addData(fiber, True, [fiber, cell, out, {"segmention_thres": None, "seg_gaus1": None, "seg_gaus2": None}], mpl.colors.to_hex(f"gray"))

    def clear_files(self):
        self.list.clear()
        self.data = {}

    last_cell = None
    def listSelected(self):
        def get_pixmap(im_cell, cmap="viridis"):
            im_cell = im_cell.astype(np.float64)
            im_cell -= np.min(im_cell)
            im_cell /= np.max(im_cell)
            im_cell = plt.get_cmap(cmap)(im_cell)
            im_cell = (im_cell*255).astype(np.uint8)

            return QtGui.QPixmap(array2qimage(im_cell))

        if len(self.list.selectedItems()):
            data = self.data[self.list.currentRow()][2]
            attr = data[3]
            if self.last_cell == self.list.currentRow():
                attr["segmention_thres"] = self.segmention_thres_indi.value()
                attr["seg_gaus1"] = self.seg_gaus1_indi.value()
                attr["seg_gaus2"] = self.seg_gaus2_indi.value()
            else:
                self.segmention_thres_indi.setValue(attr["segmention_thres"])
                self.seg_gaus1_indi.setValue(attr["seg_gaus1"])
                self.seg_gaus2_indi.setValue(attr["seg_gaus2"])
                print("->", [v is None for v in attr.values()])
                if np.all([v is None for v in attr.values()]):
                    self.individual_data.setValue(False)
                else:
                    self.individual_data.setValue(True)
            self.last_cell = self.list.currentRow()
            im_cell = imageio.imread(data[1])
            from CompactionAnalyzer.CompactionFunctions import segment_cell, normalize
            im_cell = normalize(im_cell, 1, 99)

            self.pixmap.setPixmap(get_pixmap(im_cell))
            self.label.setExtend(im_cell.shape[1], im_cell.shape[0])

            im_fiber = imageio.imread(data[0])
            im_fiber = normalize(im_fiber, 1, 99)
            self.pixmap2.setPixmap(get_pixmap(im_fiber))
            self.label2.setExtend(im_fiber.shape[1], im_fiber.shape[0])

            result = segment_cell(im_cell,
                                  thres=self.segmention_thres.value() if attr["segmention_thres"] is None else attr["segmention_thres"],
                                  seg_gaus1=self.seg_gaus1.value() if attr["seg_gaus1"] is None else attr["seg_gaus1"],
                                  seg_gaus2=self.seg_gaus2.value() if attr["seg_gaus2"] is None else attr["seg_gaus2"])
            mask = result["mask"]
            from skimage import measure
            # Find contours at a constant value of 0.8
            contours = measure.find_contours(mask, 0.5)

            path = QtGui.QPainterPath()
            for c in contours:
                path.moveTo(c[0][1], c[0][0])
                for cc in c:
                    path.lineTo(cc[1], cc[0])
            self.contour.setPath(path)
            self.contour2.setPath(path)

            self.label_text.setText(data[2])

            self.line_views()


    def param_changed(self, name, update_image=False):
        if len(self.list.selectedItems()):
            data = self.data[self.list.currentRow()][2]
            data[name] = getattr(self, name).value()
            if update_image:
                self.slider_changed(self.slider.value())
            self.slider.min = self.n_min.value()
            self.slider.max = self.n_max.value()
            self.slider.update()

    last_image = None
    last_seg = None
    def slider_changed(self, i):
        if self.last_image is not None and self.last_image[0] == i:
            i, im, im0 = self.last_image
            print("cached")
        else:
            im = imageio.imread(self.images[i]).astype(np.float)
            if self.continous_segmentation.value() is True:
                im0 = im
            else:
                im0 = imageio.imread(self.images[0]).astype(np.float)
            self.last_image = [i, im, im0]

        if self.last_seg is not None and \
                self.last_seg[1] == self.thres_segmentation.value() and \
                self.continous_segmentation.value() is False:
            pass
            print("cached")
        else:
            print(self.last_seg, i, self.thres_segmentation.value())
            seg0 = jf.piv.segment_spheroid(im0, True, self.thres_segmentation.value())
            from skimage import measure
            # Find contours at a constant value of 0.8
            contours = measure.find_contours(seg0["mask"], 0.5)

            path = QtGui.QPainterPath()
            for c in contours:
                path.moveTo(c[0][1], im.shape[0]-c[0][0])
                for cc in c:
                    path.lineTo(cc[1], im.shape[0]-cc[0])
            self.contour.setPath(path)
            self.last_seg = [i, self.thres_segmentation.value()]
        im = im - im.min()
        im = (im/im.max()*255).astype(np.uint8)

        self.pixmap.setPixmap(QtGui.QPixmap(array2qimage(im)))
        self.label.setExtend(im.shape[1], im.shape[0])
        self.label_text.setText(f"{i+1}/{len(self.images)} {self.images[i]}")

        data = self.data[self.list.currentRow()][2]
        try:
            im = imageio.imread(str(data["output"]) + '/plot' + str(i).zfill(6) + '.png')
        except FileNotFoundError:
            im = np.zeros(im.shape)
        self.pixmap2.setPixmap(QtGui.QPixmap(array2qimage(im)))
        self.label2.setExtend(im.shape[1], im.shape[0])

        self.line_views()

    def line_views(self):

        def changes1(*args):
            self.label2.setOriginScale(self.label.getOriginScale() * self.label.view_rect[0] / self.label2.view_rect[0])
            start_x, start_y, end_x, end_y = self.label.GetExtend()
            center_x, center_y = start_x + (end_x - start_x) / 2, start_y + (end_y - start_y) / 2
            center_x = center_x / self.label.view_rect[0] * self.label2.view_rect[0]
            center_y = center_y / self.label.view_rect[1] * self.label2.view_rect[1]
            self.label2.centerOn(center_x, center_y)

        def zoomEvent(scale, pos):
            changes1()

        self.label.zoomEvent = zoomEvent
        self.label.panEvent = changes1

        def changes2(*args):
            self.label.setOriginScale(self.label2.getOriginScale() * self.label2.view_rect[0] / self.label.view_rect[0])
            start_x, start_y, end_x, end_y = self.label2.GetExtend()
            center_x, center_y = start_x + (end_x - start_x) / 2, start_y + (end_y - start_y) / 2
            center_x = center_x / self.label2.view_rect[0] * self.label.view_rect[0]
            center_y = center_y / self.label2.view_rect[1] * self.label.view_rect[1]
            self.label.centerOn(center_x, center_y)

        def zoomEvent(scale, pos):
            changes2()

        self.label2.zoomEvent = zoomEvent
        self.label2.panEvent = changes2
        changes2()

    def run(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self.run_thread, daemon=True)
            self.thread.start()
            self.setState(True)
        else:
            kill_thread(self.thread)
            self.thread = None
            self.setState(False)

    def setState(self, running):
        if running:
            self.button_run.setText("stop")
            for widget in self.input_list:
                widget.setDisabled(True)
        else:
            self.button_run.setText("run")
            for widget in self.input_list:
                widget.setDisabled(False)

    def finished(self):
        self.thread = None
        self.setState(False)

    def progress_callback(self, i, n, ii, nn):
        self.progress1.setRange(0, n)
        self.progress1.setValue(i)
        self.progress2.setRange(0, nn-1)
        self.progress2.setValue(ii)
        for j in range(self.list.count()):
            if j < i:
                self.list.item(j).setIcon(qta.icon("fa.check", options=[dict(color="darkgreen")]))
            else:
                self.list.item(j).setIcon(qta.icon("fa.circle", options=[dict(color="white")]))
        self.list.setCurrentRow(i)
        #self.slider.setEvaluated(ii)
        #self.slider.setValue(ii)
        return
        # when plotting show the slider
        if self.plot.value() is True:
            # set the range for the slider
            self.slider.setRange(1, i)
            # it the slider was at the last value, move it to the new maximum
            if self.slider.value() == i-1:
                self.slider.setValue(i)

    def run_thread(self):
        try:
            print("compute displacements")
            n = self.list.count() - 1
            self.progress_signal.emit(0, n, 0, 1)
            for i in range(n):
                if not self.data[i][1]:
                    continue

                fiber, cell, output, attr = self.data[i][2]


                sigma_tensor = self.sigma_tensor.value()
                if self.sigma_tensor_type.value() == "um":
                    sigma_tensor /= self.scale.value()
                shell_width = self.shell_width.value()
                if self.shell_width_type.value() == "um":
                    shell_width /= self.scale.value()

                from CompactionAnalyzer.CompactionFunctions import StuctureAnalysisMain
                StuctureAnalysisMain(fiber_list=[fiber],
                                     cell_list=[cell],
                                     out_list=[output],
                                     scale=self.scale.value(),
                                     sigma_tensor=sigma_tensor,
                                     edge=self.edge.value(),
                                     segmention_thres=self.segmention_thres.value() if attr["segmention_thres"] is None else attr["segmention_thres"],
                                     seg_gaus1=self.seg_gaus1.value() if attr["seg_gaus1"] is None else attr["seg_gaus1"],
                                     seg_gaus2=self.seg_gaus2.value() if attr["seg_gaus2"] is None else attr["seg_gaus2"],
                                     sigma_first_blur=self.sigma_first_blur.value(),
                                     angle_sections=self.angle_sections.value(),
                                     shell_width=shell_width,
                                     )

                self.progress_signal.emit(i+1, n, 0, 1)
        finally:
            self.finished_signal.emit()


class ListWidget(QtWidgets.QListWidget):
    itemSelectionChanged2 = QtCore.Signal()
    itemChanged2 = QtCore.Signal()
    addItemClicked = QtCore.Signal()

    data = []
    def __init__(self, layout, editable=False, add_item_button=False, color_picker=False):
        super().__init__()
        layout.addWidget(self)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.list2_context_menu)
        self.itemChanged.connect(self.list2_checked_changed)
        self.itemChanged = self.itemChanged2
        self.act_delete = QtWidgets.QAction(qta.icon("fa.trash"), "Delete", self)
        self.act_delete.triggered.connect(self.delete_item)

        self.act_color = None
        if color_picker is True:
            self.act_color = QtWidgets.QAction(qta.icon("fa.paint-brush"), "Change Color", self)
            self.act_color.triggered.connect(self.change_color)

        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)

        self.flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable
        if editable:
            self.flags |= QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsDragEnabled

        self.add_item_button = add_item_button
        self.addAddItem()
        self.itemSelectionChanged.connect(self.listSelected)
        self.itemSelectionChanged = self.itemSelectionChanged2

        self.itemClicked.connect(self.item_clicked)
        self.model().rowsMoved.connect(self.rowsMoved)

    def rowsMoved(self, parent, start, end, dest, row):
        if row == self.count():
            if self.add_item is not None:
                self.takeItem(self.count() - 2)
            self.addAddItem()
            return
        if row > start:
            row -= 1
        self.data.insert(row, self.data.pop(start))

    def item_clicked(self, item):
        if item == self.add_item:
            self.addItemClicked.emit()

    def listSelected(self):
        if self.no_list_change is True:
            return
        self.no_list_change = True
        if self.currentItem() == self.add_item:
            self.no_list_change = False
            return
        self.itemSelectionChanged.emit()
        self.no_list_change = False

    add_item = None
    def addAddItem(self):
        if self.add_item_button is False:
            return
        if self.add_item is not None:
            del self.add_item
        self.add_item = QtWidgets.QListWidgetItem(qta.icon("fa.plus"), self.add_item_button, self)
        self.add_item.setFlags(QtCore.Qt.ItemIsEnabled)

    def list2_context_menu(self, position):
        if self.currentItem() and self.currentItem() != self.add_item:
            # context menu
            menu = QtWidgets.QMenu()

            if self.act_color is not None:
                menu.addAction(self.act_color)

            menu.addAction(self.act_delete)

            # open menu at mouse click position
            if menu:
                menu.exec_(self.viewport().mapToGlobal(position))

    def change_color(self):
        import matplotlib as mpl
        index = self.currentRow()

        # get new color from color picker
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(*mpl.colors.to_rgb(self.data[index][3])))
        # if a color is set, apply it
        if color.isValid():
            self.data[index][3] = "#%02x%02x%02x" % color.getRgb()[:3]
            self.item(index).setIcon(qta.icon("fa.circle", options=[dict(color=color)]))

    def delete_item(self):
        index = self.currentRow()
        self.data.pop(index)
        self.takeItem(index)

    def setData(self, data):
        self.no_list_change = True
        self.data = data
        self.clear()
        for d, checked, _, color in data:
            self.customAddItem(d, checked, color)

        self.addAddItem()
        self.no_list_change = False

    no_list_change = False
    def list2_checked_changed(self, item):
        if self.no_list_change is True:
            return
        data = self.data
        for i in range(len(data)):
            item = self.item(i)
            data[i][0] = item.text()
            data[i][1] = item.checkState()
        self.itemChanged.emit()

    def addData(self, d, checked, extra=None, color=None):
        self.no_list_change = True
        if self.add_item is not None:
            self.takeItem(self.count()-1)
        self.data.append([d, checked, extra, color])
        item = self.customAddItem(d, checked, color)
        self.addAddItem()
        self.no_list_change = False
        return item

    def customAddItem(self, d, checked, color):
        item = QtWidgets.QListWidgetItem(d, self)
        if color is not None:
            item.setIcon(qta.icon("fa.circle", options=[dict(color=color)]))
        item.setFlags(self.flags)
        item.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
        return item


class PlottingWindow(QtWidgets.QWidget):
    progress_signal = QtCore.Signal(int, int, int, int)
    finished_signal = QtCore.Signal()
    thread = None

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumWidth(800)
        self.setMinimumHeight(400)
        self.setWindowTitle("Saenopy Evaluation")

        self.images = []
        self.data_folders = []
        self.current_plot_func = lambda: None

        with QtShortCuts.QHBoxLayout(self) as main_layout:
            with QtShortCuts.QVBoxLayout() as layout:
                with QtShortCuts.QGroupBox(None, "Groups") as (_, layout2):
                    layout2.setContentsMargins(0, 3, 0, 1)
                    self.list = ListWidget(layout2, True, add_item_button="add group", color_picker=True)
                    self.list.setStyleSheet("QListWidget{border: none}")
                    self.list.itemSelectionChanged.connect(self.listSelected)
                    self.list.itemChanged.connect(self.replot)
                    self.list.itemChanged.connect(self.update_group_name)
                    self.list.addItemClicked.connect(self.addGroup)

                with QtShortCuts.QGroupBox(layout, "Group") as (self.box_group, layout2):
                    layout2.setContentsMargins(0, 3, 0, 1)
                    self.list2 = ListWidget(layout2, add_item_button="add files")
                    self.list2.setStyleSheet("QListWidget{border: none}")
                    self.list2.itemSelectionChanged.connect(self.run2)
                    self.list2.itemChanged.connect(self.replot)
                    self.list2.addItemClicked.connect(self.addFiles)

            with QtShortCuts.QGroupBox(main_layout, "Plot Forces") as (_, layout):
                self.type = QtShortCuts.QInputChoice(None, "type", "global orientation", ["global orientation", "normalized intensity (first shell)", "orientation over distance", "normed intensity over distance"])
                self.type.valueChanged.connect(self.replot)

                self.canvas = MatplotlibWidget(self)
                layout.addWidget(self.canvas)
                layout.addWidget(NavigationToolbar(self.canvas, self))

                with QtShortCuts.QHBoxLayout() as layout2:
                    self.button_export = QtShortCuts.QPushButton(layout2, "Export", self.export)
                    layout2.addStretch()
                    self.button_run = QtShortCuts.QPushButton(layout2, "Single Plot", self.run2)
                    self.button_run2 = QtShortCuts.QPushButton(layout2, "Grouped Plot", self.plot_groups)
                    self.plot_buttons = [self.button_run, self.button_run2]
                    for button in self.plot_buttons:
                        button.setCheckable(True)

        self.list.setData(self.data_folders)
        self.addGroup()
        self.current_plot_func = self.run2

    def update_group_name(self):
        if self.list.currentItem() is not None:
            self.box_group.setTitle(f"Files for '{self.list.currentItem().text()}'")
            self.box_group.setEnabled(True)
        else:
            self.box_group.setEnabled(False)

    def addGroup(self):
        import matplotlib as mpl
        text = f"Group{1+len(self.data_folders)}"
        item = self.list.addData(text, True, [], mpl.colors.to_hex(f"C{len(self.data_folders)}"))
        self.list.setCurrentItem(item)
        self.list.editItem(item)

    def addFiles(self):
        class AddFilesDialog(QtWidgets.QDialog):
            def __init__(self, parent):
                super().__init__(parent)
                self.setWindowTitle("Add Files")
                with QtShortCuts.QVBoxLayout(self) as layout:
                    self.label = QtWidgets.QLabel("Select a path as an input wildcard. Use * to specify a placeholder. All paths that match the wildcard will be added.")
                    layout.addWidget(self.label)
                    def checker(filename):
                        return filename + "/**/results_total.xlsx"
                    self.inputText = QtShortCuts.QInputFolder(None, None, settings=settings, filename_checker=checker,
                                                                settings_key="batch_eval/wildcard", allow_edit=True)
                    with QtShortCuts.QHBoxLayout() as layout3:
                        # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                        layout3.addStretch()
                        self.button_addList = QtShortCuts.QPushButton(None, "cancel", self.reject)
                        self.button_addList = QtShortCuts.QPushButton(None, "ok", self.accept)

        dialog = AddFilesDialog(self)
        if not dialog.exec():
            return

        text = os.path.normpath(dialog.inputText.value())
        files = glob.glob(text, recursive=True)

        current_group = self.list2.data
        current_files = [d[0] for d in current_group]
        for file in files:
            if file in current_files:
                print("File already in list", file)
                continue
            if self.list2.data is current_group:
                data = {"results_total": pd.read_excel(file),
                        "results_distance": pd.read_excel(Path(file).parent / "results_distance.xlsx"),
                        "image": Path(file).parent / "Figures\overlay2.png"}
                self.list2.addData(file, True, data)

    def listSelected(self):
        try:
            data = self.data_folders[self.list.currentRow()]
        except IndexError:
            return
        self.update_group_name()
        self.list2.setData(data[2])

    def getAllCurrentPandasData(self, key, only_first_line=False):
        results = []
        for name, checked, files, color in self.data_folders:
            if checked != 0:
                for name2, checked2, res, color in files:
                    if checked2 != 0:
                        res[key]["group"] = name
                        if only_first_line is True:
                            results.append(res[key].iloc[0:1])
                        else:
                            results.append(res[key])
        res = pd.concat(results)
        res.to_csv("tmp_pandas.csv")
        return res

    def replot(self):
        if self.current_plot_func is not None:
            self.current_plot_func()

    def plot_groups(self):
        for button in self.plot_buttons:
            button.setChecked(False)
        self.button_run2.setChecked(True)
        self.current_plot_func = self.plot_groups

        self.button_export.setDisabled(False)

        self.canvas.setActive()
        plt.cla()
        plt.axis("auto")
        if self.type.value() == "global orientation":
            res = self.getAllCurrentPandasData("results_total")
            code_data = [res, ["group", 'Orientation (weighted by intensity and coherency)']]
            def plot(res, color_dict2):
                # define the colors
                color_dict = color_dict2

                # iterate over the groups
                for name, d in res.groupby("group")['Orientation (weighted by intensity and coherency)']:
                    plt.bar(name, d.mean(), yerr=d.sem(), color=color_dict[name])

                # add ticks and labels
                plt.ylabel("orientation")
                # despine the axes
                plt.gca().spines["top"].set_visible(False)
                plt.gca().spines["right"].set_visible(False)
                plt.tight_layout()
                # show the plot
                self.canvas.draw()

        elif self.type.value() == "normalized intensity (first shell)":

            res = self.getAllCurrentPandasData("results_distance", only_first_line=True)
            code_data = [res, ["group", 'Intensity Norm (individual)']]

            def plot(res, color_dict2):
                # define the colors
                color_dict = color_dict2

                # iterate over the groups
                for name, d in res.groupby("group")['Intensity Norm (individual)']:
                    plt.bar(name, d.mean(), yerr=d.sem(), color=color_dict[name])

                # add ticks and labels
                plt.ylabel("normalized intensity")
                # despine the axes
                plt.gca().spines["top"].set_visible(False)
                plt.gca().spines["right"].set_visible(False)
                plt.tight_layout()
                # show the plot
                self.canvas.draw()

        elif self.type.value() == "orientation over distance":
            res = self.getAllCurrentPandasData("results_distance")
            code_data = [res, ["group", "Shell_mid (µm)", "Orientation (individual)"]]

            def plot(res, color_dict2):
                # define the colors
                color_dict = color_dict2

                # iterate over the groups
                for name, d in res.groupby("group"):
                    d = d.groupby("Shell_mid (µm)")["Orientation (individual)"].agg(["mean", "sem"])
                    plt.plot(d.index,d["mean"], color=color_dict[name])
                    plt.fill_between(d.index, d["mean"]-d["sem"], d["mean"]+d["sem"], color=color_dict[name], alpha=0.5)

                # add ticks and labels
                plt.xlabel("shell mid (µm)")
                plt.ylabel("individual orientation")
                # despine the axes
                plt.gca().spines["top"].set_visible(False)
                plt.gca().spines["right"].set_visible(False)
                plt.tight_layout()
                # show the plot
                self.canvas.draw()

        elif self.type.value() == "normed intensity over distance":
            res = self.getAllCurrentPandasData("results_distance")
            code_data = [res, ["group", "Shell_mid (µm)", 'Intensity Norm (individual)']]

            def plot(res, color_dict2):
                # define the colors
                color_dict = color_dict2

                # iterate over the groups
                for name, d in res.groupby("group"):
                    d = d.groupby("Shell_mid (µm)")['Intensity Norm (individual)'].agg(["mean", "sem"])
                    plt.plot(d.index, d["mean"], color=color_dict[name])
                    plt.fill_between(d.index, d["mean"] - d["sem"], d["mean"] + d["sem"], color=color_dict[name], alpha=0.5)

                # add ticks and labels
                plt.xlabel("shell mid (µm)")
                plt.ylabel("normalized intensity")
                # despine the axes
                plt.gca().spines["top"].set_visible(False)
                plt.gca().spines["right"].set_visible(False)
                plt.tight_layout()
                # show the plot
                self.canvas.draw()

        self.canvas.setActive()
        plt.cla()

        color_dict = {d[0]: d[3] for d in self.data_folders}

        code = execute(plot, code_data[0][code_data[1]], color_dict2=color_dict)

        self.export_data = [code, code_data]
        return

    def run2(self):
        for button in self.plot_buttons:
            button.setChecked(False)
        self.button_run.setChecked(True)
        self.current_plot_func = self.run2

        self.button_export.setDisabled(True)

        data = self.list2.data[self.list2.currentRow()][2]
        im = imageio.imread(data["image"])

        plot_color = self.list.data[self.list.currentRow()][3]

        self.canvas.setActive()
        plt.cla()
        plt.axis("auto")
        if self.type.value() == "global orientation":
            plt.imshow(im)
            plt.title(f"Orientation {data['results_total']['Orientation (weighted by intensity and coherency)'].iloc[0]:.3f}")
            plt.axis("off")
        elif self.type.value() == "normalized intensity (first shell)":
            plt.imshow(im)
            plt.title(f"Normalized Intensity {data['results_distance']['Intensity Norm (individual)'].iloc[0]:.3f}")
            plt.axis("off")
        elif self.type.value() == "normed intensity over distance":
            plt.plot(data["results_distance"]["Shell_mid (µm)"],
                     data["results_distance"]["Intensity Norm (individual)"], color=plot_color)
            plt.xlabel("shell mid (µm)")
            plt.ylabel("intensity norm")
        elif self.type.value() == "orientation over distance":
            plt.plot(data["results_distance"]["Shell_mid (µm)"],
                     data["results_distance"]["Orientation (individual)"], color=plot_color)
            plt.xlabel("shell mid (µm)")
            plt.ylabel("individual orientation")
        # despine the axes
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.tight_layout()
        self.canvas.draw()

    def export(self):
        class AddFilesDialog(QtWidgets.QDialog):
            def __init__(self, parent):
                super().__init__(parent)
                self.setWindowTitle("Export Plot")
                with QtShortCuts.QVBoxLayout(self) as layout:
                    self.label = QtWidgets.QLabel("Select a path to export the plot script with the data.")
                    layout.addWidget(self.label)
                    self.inputText = QtShortCuts.QInputFilename(None, None, file_type="Python Script (*.py)", settings=settings,
                                                                settings_key="batch_eval/export_plot", existing=False)
                    self.strip_data = QtShortCuts.QInputBool(None, "export only essential data columns", True, settings=settings, settings_key="batch_eval/export_complete_df")
                    self.include_df = QtShortCuts.QInputBool(None, "include dataframe in script", True, settings=settings, settings_key="batch_eval/export_include_df")
                    with QtShortCuts.QHBoxLayout() as layout3:
                        # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                        layout3.addStretch()
                        self.button_addList = QtShortCuts.QPushButton(None, "cancel", self.reject)
                        self.button_addList = QtShortCuts.QPushButton(None, "ok", self.accept)

        dialog = AddFilesDialog(self)
        if not dialog.exec():
            return

        with open(str(dialog.inputText.value()), "wb") as fp:
            code = ""
            code += "import matplotlib.pyplot as plt\n"
            code += "import pandas as pd\n"
            code += "import io\n"
            code += "\n"
            code += "# the data for the plot\n"
            res, columns = self.export_data[1]
            if dialog.strip_data.value() is False:
                columns = None
            if dialog.include_df.value() is True:
                code += "csv_data = r'''" + res.to_csv(columns=columns) + "'''\n"
                code += "# load the data as a DataFrame\n"
                code += "res = pd.read_csv(io.StringIO(csv_data))\n\n"
            else:
                csv_file = str(dialog.inputText.value()).replace(".py", "_data.csv")
                res.to_csv(csv_file, columns=columns)
                code += "# load the data from file\n"
                code += f"res = pd.read_csv('{csv_file}')\n\n"
            code += self.export_data[0]
            fp.write(code.encode("utf8"))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    print(sys.argv)
    window = MainWindow()
    if len(sys.argv) >= 2:
        window.loadFile(sys.argv[1])
    window.show()
    sys.exit(app.exec_())
