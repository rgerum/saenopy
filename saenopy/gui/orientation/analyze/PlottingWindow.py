import pandas as pd

from saenopy.gui.common import QtShortCuts

from saenopy.gui.common.PlottingWindowBase import PlottingWindowBase
from saenopy.gui.orientation.modules.result import ResultOrientation


class PlottingWindow(PlottingWindowBase):
    settings_key = "saenopy_orientation"
    file_extension = ".saenopyOrientation"

    def add_parameters(self):
        self.type = QtShortCuts.QInputChoice(None, "type", "global orientation",
                                             ["global orientation", "normalized intensity (first shell)", "orientation over distance", "normed intensity over distance"])
        self.type.valueChanged.connect(self.replot)
        self.agg = QtShortCuts.QInputChoice(None, "aggregate", "mean",
                                            ["mean", "max", "min", "median"])
        self.agg.valueChanged.connect(self.replot)

    def __init__(self, parent=None, batch_evluate_instance=None):
        super().__init__(parent, batch_evluate_instance)
        self.button_run.setText("Single Distance Course")
        self.button_run2.setText("Grouped Distance Courses")

    def get_time_factor(self, maxtime):
        return 1, "Distance (µm)"

    def check_results_with_time(self):
        time_values = False
        if self.type.value() == "global orientation":
            time_values = False
        elif self.type.value() == "normalized intensity (first shell)":
            time_values = False
        elif self.type.value() == "orientation over distance":
            time_values = True
        elif self.type.value() == "normed intensity over distance":
            time_values = True

        if time_values is False:
            self.barplot()
        if getattr(self, "agg", None) is not None:
            self.agg.setEnabled(time_values)
        self.button_run.setEnabled(time_values)
        self.button_run2.setEnabled(time_values)

    def load_file(self, file):
        res: ResultOrientation = ResultOrientation.load(file)
        res.resulting_data = []

        global_data = res.results_total[0].copy()
        global_data["normalized intensity (first shell)"] = res.results_distance[0]["Intensity Norm (individual)"]

        for i in res.results_distance:
            i = i.copy()
            i.update(global_data)
            i["filename"] = file
            i["t"] = i['Shell_mid (µm)']
            res.resulting_data.append(i)
        res.resulting_data = pd.DataFrame(res.resulting_data)
        return res

    def get_label(self):
        if self.type.value() == "global orientation":
            mu_name = 'Orientation'
            y_label = 'Orientation'
        elif self.type.value() == "normalized intensity (first shell)":
            mu_name = 'normalized intensity (first shell)'
            y_label = 'normalized intensity (first shell)'
        elif self.type.value() == "orientation over distance":
            mu_name = 'Orientation (individual)'
            y_label = 'Orientation (individual)'
        elif self.type.value() == "normed intensity over distance":
            mu_name = 'Intensity Norm (individual)'
            y_label = 'Intensity Norm (individual)'
        return mu_name, y_label
