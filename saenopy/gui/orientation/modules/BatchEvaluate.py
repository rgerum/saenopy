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

    analysis_workflow = "full"
    analysis_mode = "full"
    scroll_parameters = True

    result_params = ["segmentation_parameters", "orientation_parameters"]

    def __init__(self, parent=None):
        super().__init__(parent)
        # QSplitter.replaceWidget leaves the new scroll area hidden in Qt5.
        # Keep this adjustment local to the orientation interface.
        self.parameter_scroll.show()
        self.tabs.setMinimumWidth(250)
        self.parameter_scroll.setMinimumWidth(260)
        if not self.data:  # Command-line result files may already be loaded.
            self.sub_module_segmentation.setResult(None)
            self.sub_module_deformation.setResult(None)
        self.set_current_result.connect(self.update_mode_tabs)
        row = self.list.currentRow()
        self.update_mode_tabs(self.data[row][2] if 0 <= row < len(self.data) else None)

    def update_mode_tabs(self, result):
        mode = result.analysis_mode if result is not None else self.analysis_mode
        self.tabs.setTabVisible(self.tabs.indexOf(self.tab2.tab.parent()), True)

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
            self.analysis_mode = dialog.selected_analysis_mode()
            self.axis_mask_source = dialog.mask_source.value()
            self.analysis_workflow = {"full": "full", "point_orientation": "point", "line_orientation": "point"}.get(self.analysis_mode, "axis")
            # if there was a bf stack selected
            fiberText = dialog.fiberText.value()

            # the active selected stack
            cellText = dialog.selected_cell_pattern()

            try:
                results = get_orientation_files(dialog.outputText.value(),
                    fiberText, cellText, pixel_size=dialog.pixel_size.value(),
                   exist_overwrite_callback=do_overwrite,
                   analysis_mode=dialog.selected_analysis_mode(),
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
            from .example_data import get_orientation_example_files
            results = get_orientation_example_files(example, do_overwrite)
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
