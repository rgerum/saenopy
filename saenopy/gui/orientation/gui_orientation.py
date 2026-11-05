import sys

from qtpy import QtCore, QtWidgets, QtGui
from saenopy.gui.common import QtShortCuts
from saenopy.gui.common.gui_classes import QHLine
import glob

from saenopy.gui.orientation.analyze.PlottingWindow import PlottingWindow
from saenopy.gui.orientation.modules.BatchEvaluate import BatchEvaluate
from saenopy.gui.orientation.modules.AnalysisModeSelector import AnalysisModeSelector
from saenopy.gui.orientation.modules.api_help import python_api_help


################

# QSettings
settings = QtCore.QSettings("FabryLab", "CompactionAnalyzer")

class MainWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumWidth(800)
        self.setMinimumHeight(400)
        self.setWindowTitle("CompactionAnalyzer")

        outer_layout = QtWidgets.QVBoxLayout(self)
        self.workflow_choice = AnalysisModeSelector()
        outer_layout.addWidget(self.workflow_choice)
        main_layout = QtWidgets.QHBoxLayout()
        outer_layout.addLayout(main_layout)

        with QtShortCuts.QTabWidget(main_layout) as self.tabs:

            """ """
            with self.tabs.createTab("Analyse Measurements") as v_layout:
                v_layout.setContentsMargins(0, 0, 0, 0)
                with QtShortCuts.QHBoxLayout() as h_layout:
                    h_layout.setContentsMargins(0, 0, 0, 0)
                    #self.deformations = Deformation(h_layout, self)
                    self.deformations = BatchEvaluate(self)
                    self.measurement_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
                    h_layout.addWidget(self.measurement_splitter)
                    self.measurement_splitter.addWidget(self.deformations)
                    self.description = QtWidgets.QTextEdit()
                    self.description.setReadOnly(True)
                    self.configure_help(self.description)
                    self.measurement_splitter.addWidget(self.description)
                    self.measurement_splitter.setCollapsible(0, False)
                    self.measurement_splitter.setStretchFactor(0, 1)
                    self.measurement_splitter.setSizes([900, 260])
                    self.description.setText("""
                    <h1>Start Evaluation</h1>
                    
                    As a measure of the contractile strength for cells/organoids in fibrous materials, we quantify the tissue 
                    compaction by the re-orientation of the fiber structure towards the cell centre and 
                    additionally by the increased intensity around the cell.<br/>
                     
                    <h2>Parameters</h2>
                    <ul>
                    <li><b>Orientation Images, Cell Images</b>: Per cell we need to specify paths to an image of the fiber structure & an image of the cell outline.
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
                self.analysis_description = self.description
                v_layout.addWidget(QHLine())
                with QtShortCuts.QHBoxLayout() as h_layout:
                    self.measurement_help_button = self.add_help_button(h_layout, self.analysis_description)
                    h_layout.addStretch()
                    self.button_previous = QtShortCuts.QPushButton(None, "back", self.previous)
                    self.button_next = QtShortCuts.QPushButton(None, "next", self.next)

            """ """
            with self.tabs.createTab("Data Analysis") as v_layout:
                v_layout.setContentsMargins(0, 0, 0, 0)
                with QtShortCuts.QHBoxLayout() as h_layout:
                    h_layout.setContentsMargins(0, 0, 0, 0)

                    #self.deformations = Force(h_layout, self)
                    self.plotting_window = PlottingWindow(self, self.deformations)
                    self.statistics_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
                    h_layout.addWidget(self.statistics_splitter)
                    self.statistics_splitter.addWidget(self.plotting_window)

                    self.description = QtWidgets.QTextEdit()
                    self.description.setReadOnly(True)
                    self.configure_help(self.description)
                    self.statistics_splitter.addWidget(self.description)
                    self.statistics_splitter.setCollapsible(0, False)
                    self.statistics_splitter.setStretchFactor(0, 1)
                    self.statistics_splitter.setSizes([900, 260])
                    self.description.setHtml("""
                        <h1>Data Analysis</h1>
                        <p>Create a group, then add evaluated <b>.saenopyOrientation</b> files.
                        Drag files or a folder into this area, use <b>add files</b>,
                        or copy measurements from the analysis list into a group.</p>
                        <p>Compare X/Y alignment, angular deviation and angle histograms in all
                        measurement modes. Cell measurements also provide cell-relative alignment,
                        deviation and distance courses. Point measurements provide alignment and
                        deviation towards the selected point. Line measurements provide alignment
                        and angular deviation relative to the line defined by two clicks.</p>
                        <p><b>Coherency</b> is available in all modes. It averages local directional
                        consistency within the analyzed region, independent of the X/Y axis or
                        reference direction: 0 is isotropic and 1 is strongly directional.</p>
                        <p>Group summaries give each image equal weight. Angle histograms are
                        normalized to a sum of 1 per image before calculating the group mean and SEM.
                        Measurements without the selected metric are excluded; unavailable metrics are greyed out.</p>
                        <p>Alignment is shown from −1 to +1. Angular deviation is the mean absolute
                        angle to the chosen axis, from 0° (parallel) to 90° (perpendicular).</p>
                        <p><b>Export</b> saves a Python plotting script with embedded data or a
                        separate CSV. <b>save</b> and <b>load</b> store or reopen the group session.</p>
                        <p>Drag the dividers to resize the lists, plot and help. Use <b>Help</b>
                        below to hide this text and give the plot more space.</p>
                        """)
                self.statistics_description = self.description
                v_layout.addWidget(QHLine())
                with QtShortCuts.QHBoxLayout() as h_layout:
                    self.statistics_help_button = self.add_help_button(h_layout, self.statistics_description)
                    h_layout.addStretch()
                    self.button_previous = QtShortCuts.QPushButton(None, "back", self.previous)
                    self.button_next = QtShortCuts.QPushButton(None, "next", self.next)

        self.workflow_choice.mode_selected.connect(self.select_analysis_mode)
        self.deformations.set_current_result.connect(self.show_loaded_result)
        self.tabs.currentChanged.connect(lambda *_: self.tabs.show())
        self.original_analysis_help = self.analysis_description.toHtml()
        self.tabs.setVisible(bool(self.deformations.data))
        if self.deformations.data:
            self.show_loaded_result(self.deformations.data[self.deformations.list.currentRow()][2])

    @staticmethod
    def configure_help(widget):
        # Use Qt's matching text/background roles in both light and dark themes.
        widget.setMinimumWidth(0)
        widget.setMaximumWidth(280)
        widget.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)

    @staticmethod
    def add_help_button(layout, description):
        button = QtWidgets.QPushButton("Help")
        button.setCheckable(True)
        button.setChecked(True)
        button.toggled.connect(description.setVisible)
        layout.addWidget(button)
        return button

    def select_workflow(self, workflow):
        # Compatibility for callers selecting the earlier two-workflow UI.
        self.select_analysis_mode("full" if workflow == "full" else "axis_orientation_masked")

    def select_analysis_mode(self, mode):
        if mode == "axis":
            mode = "axis_orientation_only"
        if mode == "point":
            mode = "point_orientation"
        self.deformations.analysis_mode = mode
        self.deformations.analysis_workflow = {"full": "full", "point_orientation": "point", "line_orientation": "point"}.get(mode, "axis")
        self.workflow_choice.hide()
        self.tabs.show()
        self.tabs.setCurrentIndex(0)
        self.show_analysis_help(mode)
        if not self.deformations.data:
            self.deformations.update_mode_tabs(None)

    def show_analysis_help(self, mode):
        if mode == "line_orientation":
            self.analysis_description.setHtml("""
                <h1>Orientation relative to a line</h1>
                <p>Add an orientation image and choose <b>Line</b> under the point/line category.
                In <b>Reference line</b>, click <b>Choose reference line</b>, then two distinct
                points in the image. A preview follows the cursor after the first click.
                Press Esc, right-click or click the button again to cancel.</p>
                <p>The line defines one direction across the whole image. Its length and position
                do not restrict the measured area. Reversing the endpoints gives the same result.</p>
                <p>Click <b>find orientations</b> to apply a new line. <b>Angle</b> and
                <b>Alignment (line)</b> switch between Line, X and Y. Alignment is +1 parallel,
                0 at 45° and −1 perpendicular.</p>
                <p><b>Data Analysis</b> includes line alignment and angular deviation, X/Y statistics,
                coherency and a line-angle histogram. Each image uses its own saved reference line.
                The two points and results are saved in the .saenopyOrientation file.</p>""")
        elif mode == "point_orientation":
            self.analysis_description.setHtml("""
                <h1>Orientation towards a point</h1>
                <p>In the import dialog, select <b>Point</b> or <b>Line</b> to choose the reference.</p>
                <p>Add an orientation image, then use <b>Choose reference point</b> and click
                the image. Set a separate point for each measurement. X increases to the right,
                Y downwards; coordinates refer to the original image.</p>
                <p>Click <b>find orientations</b> after choosing or moving the point.
                The reference and results are saved in the .saenopyOrientation file.</p>
                <p>Point-relative alignment is +1 radial, 0 at 45 degrees and −1 tangential.
                The reference point itself has no radial direction.</p>
                <p><b>Angle</b> switches between X and Y. <b>Data Analysis</b> includes
                point-relative alignment and angular deviation, X/Y statistics,
                coherency and axis angle histograms.</p>
                <p>Set <b>shell_width</b> to average in concentric rings around the selected point.
                In <b>Data Analysis</b>, choose <b>Point alignment over distance</b>,
                <b>Angular deviation to point over distance</b>, or <b>Coherency over distance from point</b>.
                Plot one measurement or grouped courses, in pixels or µm. Only selected pixels
                contribute; empty rings have no value. Grouped courses require the same ring width.</p>""")
        elif mode != "full":
            self.analysis_description.setHtml("""
                <h1>Axis orientation</h1>
                <p>When adding a measurement, choose <b>Whole image</b>, <b>Mask from the same image</b>
                or <b>Mask from a second image</b>. Orientation is always calculated from
                <b>Orientation Images</b>. Separate <b>Mask Images</b> only select the analyzed area.</p>
                <p>The mask is applied after the structure tensor. All selected regions are retained.
                Mask blur and threshold can be previewed before analysis; use <b>find orientations</b>
                to update the results after changing parameters.</p>
                <p>Use the X/Y switches in <b>Angle</b> and <b>Axis alignment</b> to choose the
                reference axis. Alignment is +1 parallel, 0 at 45 degrees, and −1 perpendicular.
                Selected mask regions appear as a transparent red overlay.</p>
                <p>Histograms are available in <b>Data Analysis</b>, alongside alignment and angular
                deviation to X/Y. Each image has equal weight within a group; histogram probabilities sum to 1 per image.</p>
                <p>The image edge is cropped in both modes. Save plots and NumPy/CSV files independently.
                The .saenopyOrientation result is always saved.</p>
                <p>Loaded measurements retain their own analysis mode.</p>""")
        else:
            self.analysis_description.setHtml(self.original_analysis_help + "<p>The <b>Angle</b> tab shows X/Y angles. Data Analysis also includes X/Y alignment, angular deviation and angle histograms from the analyzed region outside the cell.</p>")
        self.analysis_description.append(python_api_help(mode))
        # Keep the GUI instructions first; the direct call is further down.
        self.analysis_description.moveCursor(QtGui.QTextCursor.Start)
        self.analysis_description.verticalScrollBar().setValue(0)

    def show_loaded_result(self, result):
        if result is not None:
            self.workflow_choice.hide()
            self.tabs.show()
            self.show_analysis_help(result.analysis_mode)

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
    #if len(sys.argv) >= 2:
    #    window.loadFile(sys.argv[1])
    window.show()
    sys.exit(app.exec_())
