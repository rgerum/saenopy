import sys
from qtpy import QtCore, QtWidgets, QtGui
from saenopy.gui.common import QtShortCuts
import glob

from saenopy.gui.spheroid.analyze.PlottingWindow import PlottingWindow
from saenopy.gui.spheroid.modules.BatchEvaluate import BatchEvaluate

settings = QtCore.QSettings("Saenopy", "Saenopy")


class MainWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # QSettings
        self.settings = QtCore.QSettings("Saenopy", "Saenopy")

        self.setMinimumWidth(800)
        self.setMinimumHeight(400)
        self.setWindowTitle("Saenopy Viewer")

        main_layout = QtWidgets.QHBoxLayout(self)

        with QtShortCuts.QTabWidget(main_layout) as self.tabs:

            """ """
            with self.tabs.createTab("Analyse Measurements"):
                with QtShortCuts.QHBoxLayout() as h_layout:
                    self.deformations = BatchEvaluate(self)
                    h_layout.addWidget(self.deformations)
                    self.description = QtWidgets.QTextEdit()
                    self.description.setReadOnly(True)
                    self.description.setStyleSheet("QTextEdit { background: #f0f0f0}")
                    h_layout.addWidget(self.description)
                    self.description.setMaximumWidth(300)
                    self.description.setText("""
                    <h1>Deformation Detection</h1>
                    <p>
                        We need to extract the deformations from the recorded image data.
                    </p>
                    <p>
                        Therefore we use the 2D OpenPIV (particle image velocimetry) algorithm to determine the movement 
                        of beads that surround the spheroid in the equatorial plane. By exploiting spherical symmetry this deformation 
                        can then simply compared to the 3D simulations by using our material lookup-table.
                    </p>
                    
                     <h1>Force Reconstruction</h1>
                     <p>
                        For all matrix deformation a pressure & force value can be assigned by the relative deformation and
                        the relative distance to the center with regards to the used material lookup-table. The overall force is then 
                        calculated from all matrix deformations that lie in the specified distance between <b>r_min</b> and <b>r_max</b> away
                        to the spheroid center (default is r_min=2 and r_max=None).
                     </p>
                     <p>
                        Both steps can be executed individually or joint.
                     </p>
                     
                     <h2>Parameters</h2>
                     <h3>Deformation</h3>
                     <ul>
                         <li><b>Raw Images</b>: Path to the folder containing the raw image data.</li>
                         <li><b>Wildcard</b>: Wildcard to selection elements within this folder (Example: Pos01_t*.tif; star will allow all timesteps). </li>
                         <li><b>n_min, n_max</b>: Set a minimum or maximum element if first or last time steps from this selection should be ignored (default is None). </li>
                         <li><b>thresh_segmentation</b>: Factor to change the threshold for spheroid segmentation (default is 0.9). </li>
                         <li><b>continuous_segmentation</b>: If active, the segmentation is repeated for every timestep individually.</li>
                     </ul>
                     <h3>Force</h3>
                     <ul>
                         <li><b>r_min, r_max</b>: Distance range (relative radii towards center) in which deformations are 
                         included for the force calculation</li>
                     </ul>                 
                     """.strip())

            """ """
            with self.tabs.createTab("Data Analysis"):
                with QtShortCuts.QHBoxLayout() as h_layout:
                    self.plotting_window = PlottingWindow(self, self.deformations)
                    h_layout.addWidget(self.plotting_window)

                    self.description = QtWidgets.QTextEdit()
                    self.description.setReadOnly(True)
                    self.description.setStyleSheet("QTextEdit { background: #f0f0f0}")
                    h_layout.addWidget(self.description)
                    self.description.setText("""
                        <h1>Data Analysis</h1>
                        <p>
                            In this step we can analyze our results.
                        </p>
                        <p>
                            For each  <b>individually</b> spheroid/organoid the <b>force</b> or the <b>pressure</b> development 
                            can be visualized over time. In addition, different spheroids or organoids
                            can be merged in <b>groups</b>, for which then the
                            mean value and standard error can be visualized groupwise.
                        </p>
                        <p>
                            Different groups can be created and to each group individual experiments 
                            are added by using a path-placeholder combination. In particular,                                            
                            the "*" is used to scan across sub-directories to find different ".saenopySpheroid"-files.<br/>                  
                            <i> Example: "C:/User/ExperimentA/well1/Mic_rep{t}_pos*.saenopySpheroid" to read 
                            in all positions in certain directory </i>
                        </p>
                        <p>
                            Finally, we need to specify the time between consecutive images in <b>dt</b>. 
                            To display a bar chart for a particular time point, we select the desired
                            time point using <b>Comparison time</b>.
                        </p>
                        <p>
                            Export allows to store the data as CSV file or an embedded table within a python script 
                            allowing to re-plot and adjust the figures later on.
                        </p>
                            """)

    def load(self):
        files = glob.glob(self.input_filename.value())
        self.input_label.setText("\n".join(files))
#        self.input_filename


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    print(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
