from saenopy.gui.common import QtShortCuts

from .AddFilesDialog import AddFilesDialog
from saenopy.gui.common.AddFilesDialog import do_overwrite
from .DeformationDetector import DeformationDetector
from saenopy.gui.solver.modules.exporter.Exporter import ExportViewer
from saenopy.gui.common.BatchEvaluateBase import BatchEvaluateBase
from .path_editor import start_path_change
from saenopy.examples import get_examples_spheroid
from saenopy.gui.spheroid.modules.result import get_stacks_spheroid
from .ForceCalculator import ForceCalculator


class BatchEvaluate(BatchEvaluateBase):
    settings_key = "Spheroid"
    file_extension = ".saenopySpheroid"

    result_params = ["piv_parameters", "force_parameters"]

    def add_modules(self):
        layout0 = QtShortCuts.currentLayout()
        self.sub_module_deformation = DeformationDetector(self, layout0)
        self.sub_module_forces = ForceCalculator(self, layout0)
        self.sub_module_export = ExportViewer(self, layout0)
        self.modules = [self.sub_module_deformation, self.sub_module_forces, self.sub_module_export]

    def path_editor(self):
        result = self.list.data[self.list.currentRow()][2]
        start_path_change(self, result)

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
        elif dialog.mode == "example_evaluated":
            self.load_from_path(dialog.examples_output)
            return
        # load existing files
        elif dialog.mode == "existing":
            self.load_from_path(dialog.outputText3.value())
            return

        results = get_stacks_spheroid(
            input_path,
            output_path,
            pixel_size=pixel_size,
            time_delta=time_delta,
            exist_overwrite_callback=lambda x: do_overwrite(x, self),
        )

        for data in results:
            self.add_data(data)
