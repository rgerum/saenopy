import numpy as np
import pandas as pd

from saenopy.gui.spheroid.modules.result import ResultSpheroid
from saenopy.gui.common import QtShortCuts

from saenopy.gui.common.PlottingWindowBase import PlottingWindowBase


class PlottingWindow(PlottingWindowBase):
    settings_key = "saenopy_spheroid"
    file_extension = ".saenopySpheroid"

    dt = 1

    def add_parameters(self):
        self.type = QtShortCuts.QInputChoice(None, "type", "Pressure", ["Pressure", "Contractility"])
        self.type.valueChanged.connect(self.replot)
        self.input_tbar = QtShortCuts.QInputString(None, "Comparison Time", 2, type=float)
        self.input_tbar_unit = QtShortCuts.QInputChoice(self.input_tbar.layout(), None, "min", ["steps", "min", "h"])
        self.input_tbar_unit.valueChanged.connect(self.replot)
        self.input_tbar.valueChanged.connect(self.replot)

    def load_file(self, file):
        res: ResultSpheroid = ResultSpheroid.load(file)
        data_list = []
        self.dt = res.time_delta 
        for i in range(len(res.res_data["pressure_mean"])):
            data_list.append({})
            for name in res.res_data.keys():
                data_list[i][name] = res.res_data[name][i]
            data_list[i]["filename"] = file
            data_list[i]["t"] = res.time_delta * i 
        res.resulting_data = pd.DataFrame(data_list)
        return res

    def get_label(self):
        if self.type.value() == "Contractility":
            mu_name = 'contractility_mean'
            y_label = 'Contractility (µN)'
        else:
            mu_name = 'pressure_mean'
            y_label = 'Pressure (Pa)'
        return mu_name, y_label

    def get_comparison_index(self):
        if self.input_tbar.value() is None:
            return None
        if self.input_tbar_unit.value() == "steps":
            index = int(np.floor(self.input_tbar.value() + 0.5))
        elif self.input_tbar_unit.value() == "min":
            index = int(np.floor(self.input_tbar.value() *  60 / self.dt + 0.5))
        else:
            index = int(np.floor(self.input_tbar.value() * 60 * 60 / self.dt + 0.5))
        return index
