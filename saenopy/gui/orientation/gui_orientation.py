import sys

from qtpy import QtCore, QtWidgets, QtGui
from saenopy.gui.common import QtShortCuts
from saenopy.gui.common.gui_classes import QHLine
import glob

from saenopy.gui.orientation.analyze.PlottingWindow import PlottingWindow
from saenopy.gui.orientation.modules.BatchEvaluate import BatchEvaluate


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
            with self.tabs.createTab("Analyse Measurements") as v_layout:
                v_layout.setContentsMargins(0, 0, 0, 0)
                with QtShortCuts.QHBoxLayout() as h_layout:
                    h_layout.setContentsMargins(0, 0, 0, 0)
                    #self.deformations = Deformation(h_layout, self)
                    self.deformations = BatchEvaluate(self)
                    h_layout.addWidget(self.deformations)
                    self.description = QtWidgets.QTextEdit()
                    self.description.setReadOnly(True)
                    self.description.setStyleSheet("QTextEdit { background: #f0f0f0}")
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
                v_layout.setContentsMargins(0, 0, 0, 0)
                with QtShortCuts.QHBoxLayout() as h_layout:
                    h_layout.setContentsMargins(0, 0, 0, 0)

                    #self.deformations = Force(h_layout, self)
                    self.deformations = PlottingWindow(self)
                    h_layout.addWidget(self.deformations)

                    self.description = QtWidgets.QTextEdit()
                    self.description.setReadOnly(True)
                    self.description.setStyleSheet("QTextEdit { background: #f0f0f0}")
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


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    print(sys.argv)
    window = MainWindow()
    if len(sys.argv) >= 2:
        window.loadFile(sys.argv[1])
    window.show()
    sys.exit(app.exec_())
