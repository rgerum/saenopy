import json
import sys
import os
import matplotlib as mpl

import matplotlib.pyplot as plt
from qtpy import QtCore, QtWidgets
import numpy as np

from saenopy.gui.common import QtShortCuts
from saenopy.gui.common.resources import resource_icon
from saenopy.gui.solver.analyze.plot_window import PlottingWindow
from saenopy.gui.solver.modules.BatchEvaluate import BatchEvaluate
from saenopy.gui.common.gui_classes import ListWidget
from saenopy.gui.common.gui_classes import CheckAbleGroup, MatplotlibWidget, NavigationToolbar
from saenopy import macro


class MainWindowFit(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # QSettings
        self.settings = QtCore.QSettings("Saenopy", "Saenopy")

        with QtShortCuts.QHBoxLayout(self):
            with QtShortCuts.QVBoxLayout():
                self.input_type = QtShortCuts.QInputChoice(None, "type", "none", ["none", "shear rheometer", "stretch thinning", "extensional rheometer"]).addToLayout()
                self.input_type.setDisabled(True)
                self.input_type.valueChanged.connect(lambda x: self.set_value(x, "type"))

                self.input_params = QtShortCuts.QInputString(None, "params", "").addToLayout()
                self.input_params.setDisabled(True)
                self.input_params.valueChanged.connect(lambda x: self.set_value(x, "params"))

                self.list = ListWidget(QtShortCuts.current_layout, add_item_button="add measurements", color_picker=True)
                self.list.addItemClicked.connect(self.add_measurement)
                self.list.itemSelectionChanged.connect(self.listSelected)

            with QtShortCuts.QVBoxLayout():
                self.current_params = QtShortCuts.QInputString(None, "params names", "").addToLayout()
                self.current_param_values = QtShortCuts.QInputString(None, "start values", "").addToLayout()
                self.current_param_values.valueChanged.connect(self.current_param_values_changed)
                self.final_param_values = QtShortCuts.QInputString(None, "fitted values", "").addToLayout()
                self.final_param_values.line_edit.setReadOnly(True)
                self.canvas = MatplotlibWidget(self).addToLayout()
                self.button_run = QtShortCuts.QPushButton(None, "run", self.run).addToLayout()

        self.params_index = 1
        self.data = []
        self.start_params = {}
        self.list.setData(self.data)

    def current_param_values_changed(self):
        params = self.current_param_values.value()
        try:
            self.start_params.update(json.loads(params))
            self.current_param_values.line_edit.setStyleSheet("")
            print(self.start_params)
        except ValueError as err:
            self.current_param_values.line_edit.setStyleSheet("background: #d56060")

    def set_value(self, x, key):
        if key == "params":
            try:
                x = [i.strip() for i in x.split(",")]
                assert len(x) == 4
                self.input_params.line_edit.setStyleSheet("")
                for i, xx in enumerate(x):
                    if xx not in self.start_params:
                        print(xx, "not in start", self.data[self.list.currentRow()][2][key], i)
                        self.start_params[xx] = self.start_params[self.data[self.list.currentRow()][2][key][i]]
            except AssertionError:
                self.input_params.line_edit.setStyleSheet("background: #d56060")
                return
        print("set", self.list.currentRow(), x, key)
        self.data[self.list.currentRow()][2][key] = x
        self.update_params()

    def listSelected(self):
        print("listSelected", self.list.currentRow(), self.list.currentRow() < len(self.data))
        if self.list.currentRow() is not None and self.list.currentRow() < len(self.data):
            extra = self.data[self.list.currentRow()][2]
            self.input_type.setDisabled(False)
            self.input_type.setValue(extra["type"])

            self.input_params.setDisabled(False)
            self.input_params.setValue(", ".join(extra["params"]))
            #self.set_current_result.emit(pipe)

            self.update_params()

    def update_params(self):
        param_names = {}
        for d in self.data:
            if d[1] and d[2]["type"] != "none":
                params = d[2]["params"]
                indices = []
                for p in params:
                    p = p.strip()
                    if p not in param_names:
                        param_names[p] = len(param_names)
                    indices.append(param_names[p])
        self.current_params.setValue(", ".join(param_names.keys()))
        print(", ".join(param_names.keys()))
        self.current_param_values.setValue(json.dumps({k: v for k, v in self.start_params.items() if k in param_names}))

    def run(self):
        parts = []
        colors = []
        param_names = {}
        for d in self.data:
            if d[1] and d[2]["type"] != "none":
                colors.append(d[3])
                params = d[2]["params"]
                indices = []
                for p in params:
                    p = p.strip()
                    if p not in param_names:
                        param_names[p] = len(param_names)
                    indices.append(param_names[p])
                def set(p, indices=indices):
                    return tuple(p[i] for i in indices)

                if d[2]["type"] == "shear rheometer":
                    parts.append([macro.get_shear_rheometer_stress, d[2]["data"], set])
                if d[2]["type"] == "stretch thinning":
                    parts.append([macro.get_stretch_thinning, d[2]["data"], set])
                if d[2]["type"] == "extensional rheometer":
                    parts.append([macro.get_extensional_rheometer_stress, d[2]["data"], set])

        parameters, plot = macro.minimize(parts,
            [self.start_params[s] for s in param_names.keys()],
            colors=colors
        )
        results = {k: parameters[v] for k, v in param_names.items()}
        print(results)
        self.final_param_values.setValue(json.dumps(results))
        plt.figure(self.canvas.figure)
        plt.clf()
        plot()
        self.canvas.draw()

    def add_measurement(self):
        new_path = QtWidgets.QFileDialog.getOpenFileName(None, "Load Session", os.getcwd(), "JSON File (*.json)")
        if new_path:
            self.add_file(new_path)

    def add_file(self, new_path):
        data = np.loadtxt(new_path)
        print(data.shape)
        params = [f"k{self.params_index}", f"d_0{self.params_index}", f"lambda_s{self.params_index}", f"d_s{self.params_index}"]
        self.list.addData(new_path, True, dict(data=data, type="none", params=params), mpl.colors.to_hex(f"C{len(self.data)}"))
        for i, param in enumerate(params):
            self.start_params[param.strip()] = [900, 0.0004, 0.075, 0.33][i]
        self.params_index += 1
        self.update_params()


if __name__ == '__main__':  # pragma: no cover

    data0_6 = np.array(
        [[4.27e-06, -2.26e-03], [1.89e-02, 5.90e-01], [3.93e-02, 1.08e+00], [5.97e-02, 1.57e+00], [8.01e-02, 2.14e+00],
         [1.00e-01, 2.89e+00], [1.21e-01, 3.83e+00], [1.41e-01, 5.09e+00], [1.62e-01, 6.77e+00], [1.82e-01, 8.94e+00],
         [2.02e-01, 1.17e+01], [2.23e-01, 1.49e+01], [2.43e-01, 1.86e+01], [2.63e-01, 2.28e+01], [2.84e-01, 2.71e+01]])
    data1_2 = np.array(
        [[1.22e-05, -1.61e-01], [1.71e-02, 2.57e+00], [3.81e-02, 4.69e+00], [5.87e-02, 6.34e+00], [7.92e-02, 7.93e+00],
         [9.96e-02, 9.56e+00], [1.20e-01, 1.14e+01], [1.40e-01, 1.35e+01], [1.61e-01, 1.62e+01], [1.81e-01, 1.97e+01],
         [2.02e-01, 2.41e+01], [2.22e-01, 2.95e+01], [2.42e-01, 3.63e+01], [2.63e-01, 4.43e+01], [2.83e-01, 5.36e+01],
         [3.04e-01, 6.37e+01], [3.24e-01, 7.47e+01], [3.44e-01, 8.61e+01], [3.65e-01, 9.75e+01], [3.85e-01, 1.10e+02],
         [4.06e-01, 1.22e+02], [4.26e-01, 1.33e+02]])
    data2_4 = np.array(
        [[2.02e-05, -6.50e-02], [1.59e-02, 8.46e+00], [3.76e-02, 1.68e+01], [5.82e-02, 2.43e+01], [7.86e-02, 3.34e+01],
         [9.90e-02, 4.54e+01], [1.19e-01, 6.11e+01], [1.40e-01, 8.16e+01], [1.60e-01, 1.06e+02], [1.80e-01, 1.34e+02],
         [2.01e-01, 1.65e+02], [2.21e-01, 1.96e+02], [2.41e-01, 2.26e+02]])
    np.savetxt("6.txt", data0_6)
    np.savetxt("2.txt", data1_2)
    np.savetxt("4.txt", data2_4)

    app = QtWidgets.QApplication(sys.argv)
    if sys.platform.startswith('win'):
        import ctypes
        myappid = 'fabrylab.saenopy.master'  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    print(sys.argv)
    window = MainWindowFit()
    window.setMinimumWidth(1600)
    window.setMinimumHeight(900)
    window.setWindowTitle("Saenopy Viewer")
    window.setWindowIcon(resource_icon("Icon.ico"))
    window.show()
    window.add_file("6.txt")
    window.add_file("2.txt")
    window.add_file("4.txt")
    sys.exit(app.exec_())
