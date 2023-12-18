import sys

# Setting the Qt bindings for QtPy
import os

import pandas as pd
import qtawesome as qta

os.environ["QT_API"] = "pyqt5"

from qtpy import QtCore, QtWidgets, QtGui

import numpy as np
from natsort import natsorted


from saenopy.gui.common import QtShortCuts, QExtendedGraphicsView
from qimage2ndarray import array2qimage
import matplotlib.pyplot as plt
import imageio
import threading
import glob


from matplotlib.figure import Figure
import jointforces as jf
import urllib
from pathlib import Path

import ctypes


from saenopy.gui.spheroid.modules.MatplotlibWidget import MatplotlibWidget, NavigationToolbar
from saenopy.gui.spheroid.analyze.plot_window import PlottingWindow
from saenopy.gui.spheroid.modules.LookupTable import SelectLookup
from saenopy.gui.spheroid.modules.BatchEvaluate import BatchEvaluate
from saenopy.gui.common.gui_classes import QVLine, QHLine, Spoiler, CheckAbleGroup

settings = QtCore.QSettings("Saenopy", "Saenopy")



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
                    h_layout.addWidget(self.description)
                    self.description.setMaximumWidth(300)
                    self.description.setText("""
                    <h1>Deformation Detection</h1>
                    Now we need to extract the deformations from the recorded image data. <br/>
                    <br/>
                    Therefore we use the 2D OpenPIV (particle image velocimetry) algorithm to determine the movement 
                    of beads that surround the spheroid in the equatorial plane. By exploiting spherical symmetry this deformation 
                    can then simply compared to the 3D simulations by using our material lookup-table. <br/>
                    <br/>
                                        
                    
                     <h1>Force Reconstruction</h1>
                     For all matrix deformation a pressure & force value can be assigned by the relative deformation and
                     the relative distance to the center with regards to the used material lookup-table.The overall force is then 
                     calculated from all matrix deformations that lie in the specified distance between <b>r_min</b> and <b>r_max</b> away
                     to the spheroid center (default is r_min=2 and r_max=None). <br/>
                    <br/>
                    
                     Both steps can be executed individually or joint.<br/>
                    <br/>
                     
                    <h2>Parameters</h2>
                    <ul>
                    <br/>Deformation<br/>
                    <li><b>Raw Images</b>: Path to the folder containing the raw image data.</li>
                    <li><b>Wildcard</b>: Wildcard to selection elements within this folder (Example: Pos01_t*.tif; star will allow all timesteps). </li>
                    <li><b>n_min, n_max</b>: Set a minimum or maximum element if first or last time steps from this selection should be ignored (default is None). </li>
                    <li><b>thres_segmentation</b>: Factor to change the threshold for spheroid segmentation (default is 0.9). </li>
                    <li><b>continous_segmentation</b>: If active, the segmentation is repeated for every timestep individually.
                    By default we use the segmentation of the first time step (less sensitive to fluctuations)  </li>
                    <li><b>thres_segmentation</b>: Factor to change the threshold for spheroid segmentation (default is 0.9).</li>
                    <br/>Force<br/>
                    <li><b>r_min, r_max</b>:Distance range (relativ radii towards center) in which deformations are included for the force caluclation/li>
                    </ul>
                 
                     """.strip())

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
                            
                            For each  <b>individually</b> spheroid/organoid the <b>force</b> or the <b>pressure</b> development 
                            can be visualized over time. In addition, different spheroids or organoids
                            can be merged in <b>groups</b> , for which then the
                            mean value and standard error can be visualized groupwise. <br/>
                            <br/>
                                
                            Different groups can be created and to each group individual experiments 
                            are added by using a path-placeholder combination. In particular,                                            
                            the "*" is used to scan across sub-directories to find different "result.xlsx"-files.<br/>                  
                            <i> Example: "C:/User/ExperimentA/well1/Pos*/result.xlsx" to read 
                            in all positions in certain directory </i><br/>
                            <br/>
                            
                            Finally, we need to specify the time between consecutive images in <b>dt</b>. 
                            To display a bar chart for a particular time point, we select the desired
                            time point using <b>Comparison time</b>.<br/>
                            <br/>
                            
                            Export allows to store the data as CSV file or an embedded table within a python script 
                            allowing to re-plot and adjust the figures later on. <br/>
                            """)

    def load(self):
        files = glob.glob(self.input_filename.value())
        self.input_label.setText("\n".join(files))
#        self.input_filename





if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    print(sys.argv)
    window = MainWindow()
    if len(sys.argv) >= 2:
        window.loadFile(sys.argv[1])
    window.show()
    sys.exit(app.exec_())
