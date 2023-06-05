import sys

# Setting the Qt bindings for QtPy
import os

import pandas as pd
import qtawesome as qta
from qtpy import QtCore, QtWidgets, QtGui
import numpy as np
from saenopy.gui.common import QtShortCuts, QExtendedGraphicsView
from saenopy.gui.common.gui_classes import Spoiler, CheckAbleGroup, QHLine, QVLine, MatplotlibWidget, NavigationToolbar, execute, kill_thread, ListWidget
import imageio
from qimage2ndarray import array2qimage
import matplotlib.pyplot as plt
import glob
import imageio
import threading
import glob
import re
from pathlib import Path



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
                    self.description.setMaximumWidth(300)
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
                    <li><b>max_dist</b>:   Specify (optionally) a maximal distance around the cell center for analysis (in px). Default is None. </li>
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
                            <i> Example: "C:/User/ExperimentA/well1/Pos*/results_total.xlsx" to read 
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
    measurement_evaluated_signal = QtCore.Signal(int, int)
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
                    self.pixmap2 = QtWidgets.QGraphicsPixmapItem(self.label2.origin)

                    self.contour2 = QtWidgets.QGraphicsPathItem(self.label2.origin)
                    self.contour2.setPen(pen)

                    #self.label_text2 = QtWidgets.QLabel().addToLayout()
                    #self.progress2 = QtWidgets.QProgressBar().addToLayout()

                frame = QtWidgets.QFrame().addToLayout()
                frame.setMaximumWidth(300)
                with QtShortCuts.QVBoxLayout(frame) as layout:
                    frame2 = QtWidgets.QFrame().addToLayout()
                    with QtShortCuts.QVBoxLayout(frame2, no_margins=True) as layout:
                        with QtShortCuts.QHBoxLayout():
                            self.scale = QtShortCuts.QInputString(None, "scale", "1.0", type=float, settings=settings, settings_key="orientation/scale")
                            QtWidgets.QLabel("um/px").addToLayout()
                        with QtShortCuts.QHBoxLayout():
                            self.sigma_tensor = QtShortCuts.QInputString(None, "sigma_tensor", "7.0", type=float, settings=settings, settings_key="orientation/sigma_tensor")
                            self.sigma_tensor_type = QtShortCuts.QInputChoice(None, "", "um", ["um", "pixel"], settings=settings, settings_key="orientation/sigma_tensor_unit")
                            self.sigma_tensor_button = QtShortCuts.QPushButton(None, "detect", self.sigma_tensor_button_clicked)
                            self.sigma_tensor_button.setDisabled(True)
                        with QtShortCuts.QHBoxLayout():
                            self.edge = QtShortCuts.QInputString(None, "edge", "40", type=int, settings=settings, settings_key="orientation/edge", tooltip="How many pixels to cut at the edge of the image.")
                            QtWidgets.QLabel("px").addToLayout()
                            self.max_dist = QtShortCuts.QInputString(None, "max_dist", "None", type=int, settings=settings, settings_key="orientation/max_dist", tooltip="Optional: specify the maximal distance around the cell center")
                            QtWidgets.QLabel("px").addToLayout()

                        with QtShortCuts.QHBoxLayout():
                            self.sigma_first_blur = QtShortCuts.QInputString(None, "sigma_first_blur", "0.5", type=float, settings=settings, settings_key="orientation/sigma_first_blur")
                            QtWidgets.QLabel("px").addToLayout()
                        with QtShortCuts.QHBoxLayout():
                            self.angle_sections = QtShortCuts.QInputString(None, "angle_sections", "5", type=int, settings=settings, settings_key="orientation/angle_sections")
                            QtWidgets.QLabel("deg").addToLayout()

                        with QtShortCuts.QHBoxLayout():
                            self.shell_width = QtShortCuts.QInputString(None, "shell_width", "5", type=float,
                                                                        settings=settings,
                                                                        settings_key="orientation/shell_width")
                            self.shell_width_type = QtShortCuts.QInputChoice(None, "", "um", ["um", "pixel"],
                                                                             settings=settings,
                                                                             settings_key="orientation/shell_width_type")

                        with QtShortCuts.QGroupBox(None, "Segmentation Parameters"):
                            self.segmention_thres = QtShortCuts.QInputString(None, "segmention_thresh", "1.0", type=float,
                                                                             settings=settings,
                                                                             settings_key="orientation/segmention_thres")
                            self.segmention_thres.valueChanged.connect(self.listSelected)
                            with QtShortCuts.QHBoxLayout():
                                self.seg_gaus1 = QtShortCuts.QInputString(None, "seg_gauss1", "0.5", type=float, settings=settings,
                                                                          settings_key="orientation/seg_gaus1")
                                self.seg_gaus1.valueChanged.connect(self.listSelected)
                                self.seg_gaus2 = QtShortCuts.QInputString(None, "seg_gauss2", "100", type=float, settings=settings,
                                                                          settings_key="orientation/seg_gaus2")
                                self.seg_gaus2.valueChanged.connect(self.listSelected)

                            with CheckAbleGroup(self, "individual segmentation").addToLayout() as self.individual_data:
                             with QtShortCuts.QVBoxLayout() as layout2:
                                self.segmention_thres_indi = QtShortCuts.QInputString(None, "segmention_thresh", None, type=float, allow_none=True)
                                self.segmention_thres_indi.valueChanged.connect(self.listSelected)
                                with QtShortCuts.QHBoxLayout():
                                    self.seg_gaus1_indi = QtShortCuts.QInputString(None, "seg_gauss1", None, type=float, allow_none=True)
                                    self.seg_gaus1_indi.valueChanged.connect(self.listSelected)
                                    self.seg_gaus2_indi = QtShortCuts.QInputString(None, "seg_gauss2", None, type=float, allow_none=True)
                                    self.seg_gaus2_indi.valueChanged.connect(self.listSelected)

                    layout.addStretch()

                    self.button_run = QtShortCuts.QPushButton(None, "run", self.run)
        self.images = []
        self.data = []
        self.list.setData(self.data)

        self.input_list = [
            frame2,
        ]

        self.individual_data.value_changed.connect(self.changedCheckBox)

        self.progress_signal.connect(self.progress_callback)
        self.measurement_evaluated_signal.connect(self.measurement_evaluated)
        self.finished_signal.connect(self.finished)

    def sigma_tensor_button_clicked(self):
        parent = self
        class SigmaRange(QtWidgets.QDialog):
            def __init__(self, parent):
                super().__init__(parent)
                self.setWindowTitle("Determine Sigma Tensor")
                with QtShortCuts.QVBoxLayout(self) as layout:
                    self.output_folder = QtShortCuts.QInputFolder(None, "output folder", settings=settings, settings_key="orientation/sigma_tensor_range_output")
                    self.label_scale = QtWidgets.QLabel(f"Scale is {parent.scale.value()} px/um").addToLayout(layout)
                    with QtShortCuts.QHBoxLayout() as layout2:
                        self.sigma_tensor_min = QtShortCuts.QInputString(None, "min", "1.0", type=float, settings=settings, settings_key="orientation/sigma_tensor_min")
                        self.sigma_tensor_max = QtShortCuts.QInputString(None, "max", "15", type=float, settings=settings, settings_key="orientation/sigma_tensor_max")
                        self.sigma_tensor_step = QtShortCuts.QInputString(None, "step", "1", type=float, settings=settings, settings_key="orientation/sigma_tensor_step")
                        self.sigma_tensor_type = QtShortCuts.QInputChoice(None, "", "um", ["um", "pixel"], settings=settings, settings_key="orientation/sigma_tensor_unit")

                    self.progresss = QtWidgets.QProgressBar().addToLayout(layout)

                    self.canvas = MatplotlibWidget(self)
                    layout.addWidget(self.canvas)
                    layout.addWidget(NavigationToolbar(self.canvas, self))

                    with QtShortCuts.QHBoxLayout() as layout3:
                        # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                        layout3.addStretch()
                        self.button_addList2 = QtShortCuts.QPushButton(None, "cancel", self.reject)
                        self.button_addList1 = QtShortCuts.QPushButton(None, "run", self.run)

            def run(self):
                from natsort import natsorted
                fiber, cell, output, attr = parent.data[parent.list.currentRow()][2]

                output_folder = self.output_folder.value()

                sigma_tensor_min = self.sigma_tensor_min.value()
                sigma_tensor_max = self.sigma_tensor_max.value()
                sigma_tensor_step = self.sigma_tensor_step.value()
                if self.sigma_tensor_type.value() == "um":
                    sigma_tensor_min /= parent.scale.value()
                    sigma_tensor_max /= parent.scale.value()
                    sigma_tensor_step /= parent.scale.value()
                shell_width = parent.shell_width.value()
                if parent.shell_width_type.value() == "um":
                    shell_width /= parent.scale.value()

                sigma_list = np.arange(sigma_tensor_min, sigma_tensor_max+sigma_tensor_step, sigma_tensor_step)
                self.progresss.setRange(0, len(sigma_list))

                from CompactionAnalyzer.CompactionFunctions import StuctureAnalysisMain, generate_lists
                for index, sigma in enumerate(sigma_list):
                    sigma = float(sigma)
                    self.progresss.setValue(index)
                    app.processEvents()
                    # Create outputfolder
                    output_sub = os.path.join(output_folder, rf"Sigma{str(sigma*parent.scale.value()).zfill(3)}")  # subpath to store results
                    fiber_list, cell_list, out_list = generate_lists(fiber, cell,
                                                                     output_main=output_sub)


  
                    StuctureAnalysisMain(fiber_list=fiber_list,
                                         cell_list=cell_list,
                                         out_list=out_list,
                                         scale=parent.scale.value(),
                                         sigma_tensor=sigma,
                                         edge=parent.edge.value(),
                                         max_dist=parent.max_dist.value(),
                                         segmention_thres=parent.segmention_thres.value() if attr[
                                                                                               "segmention_thres"] is None else
                                         attr["segmention_thres"],
                                         seg_gaus1=parent.seg_gaus1.value() if attr["seg_gaus1"] is None else attr[
                                             "seg_gaus1"],
                                         seg_gaus2=parent.seg_gaus2.value() if attr["seg_gaus2"] is None else attr[
                                             "seg_gaus2"],
                                         sigma_first_blur=parent.sigma_first_blur.value(),
                                         angle_sections=parent.angle_sections.value(),
                                         shell_width=shell_width,
                                         regional_max_correction=True,
                                         seg_iter=1,
                                         SaveNumpy=False,
                                         plotting=True,
                                         dpi=100
                                         )
                    self.progresss.setValue(index+1)

                    ### plot results
                    # read in all creates result folder
                    result_folders = natsorted(glob.glob(os.path.join(output_folder, "Sigma*", "*", "results_total.xlsx")))

                    import yaml
                    sigmas = []
                    orientation = []
                    for folder in result_folders:
                        with (Path(folder).parent / "parameters.yml").open() as fp:
                            parameters = yaml.load(fp, Loader=yaml.SafeLoader)["Parameters"]
                            sigmas.append(parameters["sigma_tensor"][0] * parameters["scale"][0])
                        orientation.append(pd.read_excel(folder)["Orientation (weighted by intensity and coherency)"])

                    self.canvas.setActive()
                    plt.cla()
                    plt.axis("auto")

                    plt.plot(sigmas, orientation, "o-")
                    plt.ylabel("Orientation", fontsize=12)
                    plt.xlabel("Windowsize (μm)", fontsize=12)
                    plt.grid()
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_folder, "Results.png"), dpi=500)
                    self.canvas.draw()

        dialog = SigmaRange(self)
        if not dialog.exec():
            return

    def changedCheckBox(self):
        for widget in [self.segmention_thres, self.seg_gaus1, self.seg_gaus2]:
            widget.setDisabled(self.individual_data.value())
        if not self.individual_data.value():
            for widget in [self.segmention_thres_indi, self.seg_gaus1_indi, self.seg_gaus2_indi]:
                widget.setValue("None")

    def show_files(self):
        from CompactionAnalyzer.CompactionFunctions import generate_lists

        class AddFilesDialog(QtWidgets.QDialog):
            def __init__(self, parent):
                super().__init__(parent)
                self.setWindowTitle("Add Files")
                with QtShortCuts.QVBoxLayout(self) as layout:
                    self.label = QtWidgets.QLabel(
                        "Select two paths as an input wildcard. Use * to specify a placeholder. One should be for the fiber images and one for the cell images.")
                    layout.addWidget(self.label)

                    self.cellText = QtShortCuts.QInputFilename(None, "Cell Images", file_type="Image (*.tif *.png *.jpg)", settings=settings,
                                                                settings_key="batch/wildcard_cell", existing=True,
                                                                allow_edit=True)
                    self.fiberText = QtShortCuts.QInputFilename(None, "Fiber Images", file_type="Image (*.tif *.png *.jpg)", settings=settings,
                                                                settings_key="batch/wildcard_fiber", existing=True,
                                                                allow_edit=True)
                    self.outputText = QtShortCuts.QInputFolder(None, "output", settings=settings,
                                                               settings_key="batch/output_folder", allow_edit=True)

                    def changed():
                        fiber_list_string = os.path.normpath(self.fiberText.value())
                        cell_list_string = os.path.normpath(self.cellText.value())
                        output_folder = os.path.normpath(self.outputText.value())
                        fiber_list, cell_list, out_list = generate_lists(fiber_list_string, cell_list_string,
                                                                         output_main=output_folder)
                        if self.fiberText.value() == "" or self.cellText.value() == "":
                            self.label2.setText("")
                            self.label2.setStyleSheet("QLabel { color : red; }")
                            self.button_addList1.setDisabled(True)
                        elif len(fiber_list) != len(cell_list):
                            self.label2.setText(f"Warning: {len(fiber_list)} fiber images found and {len(cell_list)} cell images found. Numbers do not match.")
                            self.label2.setStyleSheet("QLabel { color : red; }")
                            self.button_addList1.setDisabled(True)
                        else:
                            if "*" not in fiber_list_string:
                                if len(fiber_list) == 0:
                                    self.label2.setText(f"'Fiber Images' not found")
                                    self.label2.setStyleSheet("QLabel { color : red; }")
                                    self.button_addList1.setDisabled(True)
                                elif len(cell_list) == 0:
                                    self.label2.setText(f"'Cell Images' not found")
                                    self.label2.setStyleSheet("QLabel { color : red; }")
                                    self.button_addList1.setDisabled(True)
                                else:
                                    self.label2.setText(f"No * found in 'Fiber Images', will only import a single image.")
                                    self.label2.setStyleSheet("QLabel { color : orange; }")
                                    self.button_addList1.setDisabled(False)
                            else:
                                self.label2.setText(
                                    f"{len(fiber_list)} fiber images found and {len(cell_list)} cell images found.")
                                self.label2.setStyleSheet("QLabel { color : green; }")
                                self.button_addList1.setDisabled(False)
                    self.fiberText.line.textChanged.connect(changed)
                    self.cellText.line.textChanged.connect(changed)
                    self.label2 = QtWidgets.QLabel()#.addToLayout()
                    layout.addWidget(self.label2)

                    with QtShortCuts.QHBoxLayout() as layout3:
                        # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                        layout3.addStretch()
                        self.button_addList2 = QtShortCuts.QPushButton(None, "cancel", self.reject)
                        self.button_addList1 = QtShortCuts.QPushButton(None, "ok", self.accept)
                    changed()

        dialog = AddFilesDialog(self)
        if not dialog.exec():
            return

        import glob
        import re
        fiber_list_string = os.path.normpath(dialog.fiberText.value())
        cell_list_string = os.path.normpath(dialog.cellText.value())
        output_folder = os.path.normpath(dialog.outputText.value())

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
            self.sigma_tensor_button.setDisabled(False)
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
            im_cell = imageio.v2.imread(data[1])
            from CompactionAnalyzer.CompactionFunctions import segment_cell, normalize
            im_cell = normalize(im_cell, 1, 99)

            self.pixmap.setPixmap(get_pixmap(im_cell))
            self.label.setExtend(im_cell.shape[1], im_cell.shape[0])

            im_fiber = imageio.v2.imread(data[0])
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

            self.link_views()


    def link_views(self):

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
        #self.progress2.setRange(0, nn-1)
        #self.progress2.setValue(ii)
        self.list.setCurrentRow(i)

    def measurement_evaluated(self, index, state):
        if state == 1:
            self.list.item(index).setIcon(qta.icon("fa5s.check", options=[dict(color="darkgreen")]))
        elif state == -1:
            self.list.item(index).setIcon(qta.icon("fa5s.times", options=[dict(color="red")]))
        else:
            self.list.item(index).setIcon(qta.icon("fa5.circle", options=[dict(color="white")]))

    def run_thread(self):
        try:
            print("compute displacements")
            n = len([1 for d in self.data if d[1]])
            counter = 0
            self.progress_signal.emit(0, n, 0, 1)
            for i in range(n):
                try:
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
                                         max_dist=self.max_dist.value(),
                                         segmention_thres=self.segmention_thres.value() if attr["segmention_thres"] is None else attr["segmention_thres"],
                                         seg_gaus1=self.seg_gaus1.value() if attr["seg_gaus1"] is None else attr["seg_gaus1"],
                                         seg_gaus2=self.seg_gaus2.value() if attr["seg_gaus2"] is None else attr["seg_gaus2"],
                                         sigma_first_blur=self.sigma_first_blur.value(),
                                         angle_sections=self.angle_sections.value(),
                                         shell_width=shell_width,
                                         )

                    self.measurement_evaluated_signal.emit(i, 1)
                except Exception as err:
                    import traceback
                    traceback.print_exc()
                    self.measurement_evaluated_signal.emit(i, -1)
                counter += 1
                self.progress_signal.emit(counter, n, 0, 1)
        finally:
            self.finished_signal.emit()




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
                        self.button_addList0 = QtShortCuts.QPushButton(None, "cancel", self.reject)
                        self.button_addList1 = QtShortCuts.QPushButton(None, "ok", self.accept)

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
                        "image": Path(file).parent / "Figures" / "overlay2.png"}
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
        im = imageio.v2.imread(data["image"])

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
