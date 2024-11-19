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

from .DeformationDetector import DeformationDetector
from .FittedMesh import FittedMesh
from .MeshCreator import MeshCreator
from .Regularizer import Regularizer
from .ResultView import ResultView
from .StackDisplay import StackDisplay
from saenopy.gui.solver.modules.exporter.Exporter import ExportViewer
from .load_measurement_dialog import AddFilesDialog
from saenopy.gui.common.AddFilesDialog import FileExistsDialog
from .path_editor import start_path_change
from saenopy.examples import get_examples
from saenopy.gui.common.BatchEvaluateBase import BatchEvaluateBase


class BatchEvaluate(BatchEvaluateBase):
    settings_key = "Seanopy_deformation"
    file_extension = ".saenopy"

    result_params = ["piv_parameters", "mesh_parameters", "material_parameters", "solve_parameters"]

    def add_modules(self):
        layout0 = QtShortCuts.currentLayout()
        self.sub_module_stacks = StackDisplay(self, layout0)
        self.sub_module_deformation = DeformationDetector(self, layout0)
        self.sub_module_mesh = MeshCreator(self, layout0)
        self.sub_module_fitted_mesh = FittedMesh(self, layout0)
        self.sub_module_regularize = Regularizer(self, layout0)
        self.sub_module_view = ResultView(self, layout0)
        # self.sub_module_fiber = FiberViewer(self, layout0)
        self.sub_module_export = ExportViewer(self, layout0)
        self.modules = [self.sub_module_stacks, self.sub_module_deformation, self.sub_module_mesh, self.sub_module_regularize]

    def path_editor(self):
        result = self.list.data[self.list.currentRow()][2]
        start_path_change(self, result)

    def add_measurement(self):
        last_decision = None
        def do_overwrite(filename):
            nonlocal last_decision

            # if we are in demo mode always load the files
            if os.environ.get("DEMO") == "true":  # pragma: no cover
                return "read"

            # if there is a last decistion stored use that
            if last_decision is not None:
                return last_decision

            # ask the user if they want to overwrite or read the existing file
            dialog = FileExistsDialog(self, filename)
            result = dialog.exec()
            # if the user clicked cancel
            if not result:
                return 0
            # if the user wants to remember the last decision
            if dialog.use_for_all.value():
                last_decision = dialog.mode
            # return the decision
            return dialog.mode

        # getStack
        dialog = AddFilesDialog(self, self.settings)
        if not dialog.exec():
            return

        # create a new measurement object
        if dialog.mode == "new":
            # if there was a reference stack selected
            if dialog.reference_choice.value() == 1:
                reference_stack = dialog.stack_reference_input.text()
            else:
                reference_stack = None

            # the active selected stack
            active_stack = dialog.stack_data_input.text()

            # if there was a time specified, get the time delta
            if "{t}" in active_stack:
                time_delta = dialog.stack_data.getTimeDelta()
                add_last_time_delta(dialog.stack_data.getTimeDelta())
            else:
                time_delta = None

            try:
                # load the stack
                results = get_stacks(
                    active_stack,
                    reference_stack=reference_stack,
                    output_path=dialog.outputText.value(),
                    voxel_size=dialog.stack_data.getVoxelSize(),
                    time_delta=time_delta,
                    crop=dialog.stack_data.get_crop(),
                    exist_overwrite_callback=do_overwrite,
                )
            except Exception as err:
                # notify the user if errors occured
                QtWidgets.QMessageBox.critical(self, "Load Stacks", str(err))
                traceback.print_exc()
            else:
                # store the last voxel size
                add_last_voxel_size(dialog.stack_data.getVoxelSize())
                # add the loaded measruement objects
                for data in results:
                    self.add_data(data)

        # load existing files
        elif dialog.mode == "existing":
            self.load_from_path(dialog.outputText3.value())

        # load from the examples database
        elif dialog.mode == "example":
            # get the date from the example referenced by name
            example = get_examples()[dialog.mode_data]

            # generate a stack with the examples data
            results = get_stacks(
                example["stack"],
                reference_stack=example.get("reference_stack", None),
                output_path=example["output_path"],
                voxel_size=example["voxel_size"],
                time_delta=example.get("time_delta", None),
                crop=example.get("crop", None),
                exist_overwrite_callback=do_overwrite,
            )
            # load all the measurement objects
            for data in results:
                if getattr(data, "is_read", False) is False:
                    data.piv_parameters = example["piv_parameters"]
                    data.mesh_parameters = example["mesh_parameters"]
                    data.material_parameters = example["material_parameters"]
                    data.solve_parameters = example["solve_parameters"]
                self.add_data(data)
        elif dialog.mode == "example_evaluated":
                self.load_from_path(dialog.examples_output)

        # update the icons
        self.update_icons()