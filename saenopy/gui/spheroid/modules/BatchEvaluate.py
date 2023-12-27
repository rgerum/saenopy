import json
import sys
import os

import qtawesome as qta
from qtpy import QtCore, QtWidgets, QtGui
import numpy as np
import glob
import threading
from pathlib import Path
import matplotlib as mpl

import traceback

from saenopy import get_stacks
from saenopy import Result
from saenopy.gui.common import QtShortCuts
from saenopy.gui.common.gui_classes import ListWidget
from saenopy.gui.common.stack_selector_tif import add_last_voxel_size, add_last_time_delta

from .AddFilesDialog import AddFilesDialog
from saenopy.gui.common.AddFilesDialog import do_overwrite
from .DeformationDetector import DeformationDetector
from saenopy.gui.solver.modules.exporter.Exporter import ExportViewer
#from .load_measurement_dialog import AddFilesDialog
from saenopy.gui.common.AddFilesDialog import FileExistsDialog
#from .path_editor import start_path_change
from saenopy.examples import get_examples
from saenopy.gui.common.BatchEvaluate import BatchEvaluate
from saenopy.examples import get_examples_spheroid
from saenopy.gui.spheroid.modules.result import ResultSpheroid, get_stacks_spheroid
from .ForceCalculator import ForceCalculator


class BatchEvaluate(BatchEvaluate):
    settings_key = "Spheroid"
    file_extension = ".saenopySpheroid"

    result_params = params = ["piv_parameters", "force_parameters"]

    def add_modules(self):
        layout0 = QtShortCuts.currentLayout()
        self.sub_module_deformation = DeformationDetector(self, layout0)
        self.sub_module_forces = ForceCalculator(self, layout0)
        self.sub_module_export = ExportViewer(self, layout0)
        self.modules = [self.sub_module_deformation, self.sub_module_forces, self.sub_module_export]

    def path_editor(self):
        return
        #result = self.list.data[self.list.currentRow()][2]
        #start_path_change(self, result)

    def add_measurement(self):
        settings = self.settings

        dialog = AddFilesDialog(self, settings)
        if not dialog.exec():
            return

        # create a new measurement object
        if dialog.mode == "new":
            input_path = dialog.inputText.value()
            output_path = dialog.outputText.value()
            pixel_size = dialog.pixel_size.value()
            time_delta = dialog.time_delta.value()
        elif dialog.mode == "example":
            # get the date from the example referenced by name
            example = get_examples_spheroid()[dialog.mode_data]
            input_path = example["input"]
            output_path = example["output_path"]
            pixel_size = example["pixel_size"]
            time_delta = example["time_delta"]

        results = get_stacks_spheroid(
            input_path,
            output_path,
            pixel_size=pixel_size,
            time_delta=time_delta,
            exist_overwrite_callback=lambda x: do_overwrite(x, self),
        )

        for data in results:
            self.add_data(data)
