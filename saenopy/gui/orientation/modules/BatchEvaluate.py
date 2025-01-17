# Setting the Qt bindings for QtPy
import os
from qtpy import QtCore, QtWidgets, QtGui
from saenopy.gui.common import QtShortCuts
import traceback

from saenopy.examples import get_examples_orientation
from .AddFilesDialog import AddFilesDialog
from saenopy.gui.common.AddFilesDialog import FileExistsDialog
from .TabAngle import TabAngle
from .TabCellOutline import TabCellOutline
from .TabCoherence import TabCoherence
from .TabOrientationField import TabOrientationField
from .result import ResultOrientation, get_orientation_files
from .Detection import DeformationDetector
from .Segmentation import SegmentationDetector
from saenopy.gui.common.BatchEvaluateBase import BatchEvaluateBase
from saenopy.gui.solver.modules.exporter.Exporter import ExportViewer

settings = QtCore.QSettings("FabryLab", "CompactionAnalyzer")


class BatchEvaluate(BatchEvaluateBase):
    settings_key = "saenopy_orientation"
    file_extension = ".saenopyOrientation"
    result: ResultOrientation = None

    result_params = ["segmentation_parameters", "orientation_parameters"]

    def add_modules(self):
        layout0 = QtShortCuts.currentLayout()
        self.sub_module_segmentation = SegmentationDetector(self, layout0)
        self.sub_module_deformation = DeformationDetector(self, layout0)
        self.sub_module_export = ExportViewer(self, layout0)
        self.modules = [self.sub_module_segmentation, self.sub_module_deformation]

        self.tab1 = TabCellOutline(self)
        self.tab2 = TabAngle(self)
        self.tab3 = TabCoherence(self)
        self.tab4 = TabOrientationField(self)

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
            # if there was a bf stack selected
            fiberText = dialog.fiberText.value()

            # the active selected stack
            cellText = dialog.cellText.value()

            try:
                results = get_orientation_files(dialog.outputText.value(),
                    fiberText, cellText, pixel_size=dialog.pixel_size.value(),
                   exist_overwrite_callback=do_overwrite,
                )
            except Exception as err:
                # notify the user if errors occured
                QtWidgets.QMessageBox.critical(self, "Load Stacks", str(err))
                traceback.print_exc()
            else:
                # add the loaded measruement objects
                for data in results:
                    self.add_data(data)

        # load existing files
        elif dialog.mode == "existing":
            self.load_from_path(dialog.outputText3.value())

        # load from the examples database
        elif dialog.mode == "example":
            # get the date from the example referenced by name
            example = get_examples_orientation()[dialog.mode_data]

            # generate a stack with the examples data
            results = get_orientation_files(
                fiber_list_string=str(example["input_fiber"]),
                cell_list_string = str(example["input_cell"]),
                output_path = str(example["output_path"]),
                pixel_size = example["pixel_size"],
                exist_overwrite_callback=do_overwrite,
            )
            # load all the measurement objects
            for data in results:
                #if getattr(data, "is_read", False) is False:
                #    data.piv_parameters = example["piv_parameters"]
                #    data.force_parameters = example["force_parameters"]
                self.add_data(data)
        elif dialog.mode == "example_evaluated":
                self.load_from_path(dialog.examples_output)

        # update the icons
        self.update_icons()
