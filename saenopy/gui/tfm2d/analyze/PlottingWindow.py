import pandas as pd

from saenopy.gui.tfm2d.modules.result import Result2D
from saenopy.gui.common import QtShortCuts

from saenopy.gui.common.PlottingWindowBase import PlottingWindowBase


class PlottingWindow(PlottingWindowBase):
    settings_key = "Seanopy_2d"
    file_extension = ".saenopy2D"

    def add_parameters(self):
        self.type = QtShortCuts.QInputChoice(None, "type", "area",
                                             ["contractility",
                                              "strain energy",
                                              "---\narea",
                                              "cell number",
                                              "mean normal stress",
                                              "max normal stress",
                                              "max shear stress",
                                              "cv mean normal stress",
                                              "cv max normal stress",
                                              "cv max shear stress",
                                              "average magnitude line tension",
                                              "std magnitude line tension",
                                              "average normal line tension",
                                              "std normal line tension",
                                              "average shear line tension",
                                              "std shear line tension",

                                              "average cell force",
                                              "average cell pressure",
                                              "average cell shear",
                                              "std cell force",
                                              "std cell pressure",
                                              "std cell shear",                                             
                                              ])
        self.type.valueChanged.connect(self.replot)
        self.agg = QtShortCuts.QInputChoice(None, "aggregate", "mean",
                                            ["mean", "max", "min", "median"])
        self.agg.valueChanged.connect(self.replot)
        self.agg.setHidden(True)

    def load_file(self, file):
        print("load file", file)
        res: Result2D = Result2D.load(file)
        res.resulting_data = pd.DataFrame([res.res_dict])
        res.resulting_data["filename"] = file
        print(res.resulting_data)
        print(res.res_dict)
        return res

    def get_label(self):
        if self.type.value() == "---\narea":
            mu_name = 'area Cell Area'
            y_label = 'area (m$^2$)'
        elif self.type.value() == "cell number":
            mu_name = 'cell number'
            y_label = 'cell number'
        elif self.type.value() == "mean normal stress":
            mu_name = 'mean normal stress Cell Area'
            y_label = 'mean normal stress (N/m)'
        elif self.type.value() == "max normal stress":
            mu_name = 'max normal stress Cell Area'
            y_label = 'max normal stress (N/m)'
        elif self.type.value() == "max shear stress":
            mu_name = 'max shear stress Cell Area'
            y_label = 'max shear stress (N/m)'
        elif self.type.value() == "cv mean normal stress":
            mu_name = 'cv mean normal stress Cell Area'
            y_label = 'cv mean normal stress'
        elif self.type.value() == "cv max normal stress":
            mu_name = 'cv max normal stress Cell Area'
            y_label = 'cv max normal stress'
        elif self.type.value() == "cv max shear stress":
            mu_name = 'cv max shear stress Cell Area'
            y_label = 'cv max shear stress'
        elif self.type.value() == "average magnitude line tension":
            mu_name = 'average magnitude line tension'
            y_label = 'average magnitude line tension (N/m)'
        elif self.type.value() == "std magnitude line tension":
            mu_name = 'std magnitude line tension'
            y_label = 'std magnitude line tension'
        elif self.type.value() == "average normal line tension":
            mu_name = 'average normal line tension'
            y_label = 'average normal line tension (N/m)'
        elif self.type.value() == "std normal line tension":
            mu_name = 'std normal line tension'
            y_label = 'std normal line tension'
        elif self.type.value() == "average shear line tension":
            mu_name = 'average shear line tension'
            y_label = 'average shear line tension (N/m)'
        elif self.type.value() == "std shear line tension":
            mu_name = 'std shear line tension'
            y_label = 'std shear line tension'
        elif self.type.value() == "average cell force":
            mu_name = 'average cell force'
            y_label = 'average cell force (N/m)'
        elif self.type.value() == "average cell pressure":
            mu_name = 'average cell pressure'
            y_label = 'average cell pressure (N/m)'
        elif self.type.value() == "average cell shear":
            mu_name = 'average cell shear'
            y_label = 'average cell shear (N/m)'
        elif self.type.value() == "std cell force":
            mu_name = 'std cell force'
            y_label = 'std cell force'
        elif self.type.value() == "std cell pressure":
            mu_name = 'std cell pressure'
            y_label = 'std cell pressure'
        elif self.type.value() == "std cell shear":
            mu_name = 'std cell shear'
            y_label = 'std cell shear'
        elif self.type.value() == "contractility":
            mu_name = 'contractility'
            y_label = 'contractility (N)'
        elif self.type.value() == "strain energy":
            mu_name = 'strain energy'
            y_label = 'strain energy (J)'
        elif self.type.value() == "":
            mu_name = ''
            y_label = ''
        return mu_name, y_label

