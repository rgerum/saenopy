import numpy as np
import pandas as pd

from saenopy import Result
from saenopy.gui.common import QtShortCuts

from saenopy.gui.common.PlottingWindowBase import PlottingWindowBase


class PlottingWindow(PlottingWindowBase):
    settings_key = "Seanopy_deformation"
    file_extension = ".saenopy"

    def add_parameters(self):
        self.type = QtShortCuts.QInputChoice(None, "type", "strain_energy",
                                             ["strain_energy", "contractility (force center)", "contractility (deformations center)",
                                              "contractility (force center t0)", "contractility (deformations center t0)",
                                              "polarity", "99_percentile_deformation",
                                              "99_percentile_force"])
        self.type.valueChanged.connect(self.replot)
        self.agg = QtShortCuts.QInputChoice(None, "aggregate", "mean",
                                            ["mean", "max", "min", "median"])
        self.agg.valueChanged.connect(self.replot)

    def load_file(self, file):
        res: Result = Result.load(file)
        res.resulting_data = []
        if len(res.solvers) == 0 or res.solvers[0] is None or res.solvers[0].regularisation_results is None:
            return
        center_f_t0 = res.solvers[0].get_center(mode="force")
        center_d_t0 = res.solvers[0].get_center(mode="deformation")
        for i, M in enumerate(res.solvers):
            res.resulting_data.append({
                "t": i * res.time_delta if res.time_delta else 0,
                "strain_energy_J": M.mesh.strain_energy,
                "contractility_nN (force center)": M.get_contractility(center_mode="force")*1e9,
                "contractility_nN (deformations center)": M.get_contractility(center_mode="deformation")*1e9,
                "contractility_nN (force center t0)": M.get_contractility(center_mode=center_f_t0)*1e9,
                "contractility_nN (deformations center t0)": M.get_contractility(center_mode=center_d_t0)*1e9,
                "polarity": M.get_polarity(),
                "99_percentile_deformation_µm": np.nanpercentile(
                    np.linalg.norm(M.mesh.displacements_target[M.mesh.regularisation_mask], axis=1), 99)*1e6,
                "99_percentile_force_nN": np.nanpercentile(
                    np.linalg.norm(M.mesh.forces[M.mesh.regularisation_mask], axis=1), 99)*1e9,
                "filename": file,
            })
        res.resulting_data = pd.DataFrame(res.resulting_data)
        return res
               
        
    def get_label(self):
        if self.type.value() == "strain_energy":
            mu_name = 'strain_energy_J'
            y_label = 'Strain Energy (J)'
        elif self.type.value() == "contractility (force center)":
            mu_name = 'contractility_nN (force center)'
            y_label = 'Contractility (nN)'
        elif self.type.value() == "contractility (deformations center)":
            mu_name = 'contractility_nN (deformations center)'
            y_label = 'Contractility (nN)'
        elif self.type.value() == "contractility (force center t0)":
            mu_name = 'contractility_nN (force center t0)'
            y_label = 'Contractility (nN)'
        elif self.type.value() == "contractility (deformations center t0)":
            mu_name = 'contractility_nN (deformations center t0)'
            y_label = 'Contractility (nN)'
        elif self.type.value() == "polarity":
            mu_name = 'polarity'
            y_label = 'Polarity'
        elif self.type.value() == "99_percentile_deformation":
            mu_name = '99_percentile_deformation_µm'
            y_label = 'Deformation (µm)'
        elif self.type.value() == "99_percentile_force":
            mu_name = '99_percentile_force_nN'
            y_label = 'Force (nN)'
        return mu_name, y_label