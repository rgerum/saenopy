from typing import Tuple
import jointforces as jf

from saenopy.gui.spheroid.modules.result import ResultSpheroid
from saenopy.gui.spheroid.modules.LookupTable import SelectLookup
from saenopy.gui.common import QtShortCuts

from saenopy.gui.common.gui_classes import CheckAbleGroup
from saenopy.gui.common.PipelineModule import PipelineModule
from saenopy.gui.common.code_export import get_code


class ForceCalculator(PipelineModule):
    pipeline_name = "calculate forces"
    use_thread = False

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with QtShortCuts.QVBoxLayout(self) as layout:
            layout.setContentsMargins(0, 0, 0, 0)
            with CheckAbleGroup(self, "Calculate Forces").addToLayout() as self.group:
                with QtShortCuts.QVBoxLayout() as layout:
                    with QtShortCuts.QVBoxLayout() as layout2:
                        with QtShortCuts.QHBoxLayout():
                            self.lookup_table = QtShortCuts.QInputString(None, "Lookup Table")
                            self.lookup_table.line_edit.setDisabled(True)
                            self.button_lookup = QtShortCuts.QPushButton(None, "choose file", self.choose_lookup)
                        # self.output = QtShortCuts.QInputFolder(None, "Result Folder")
                        # self.lookup_table = QtShortCuts.QInputFilename(None, "Lookup Table", 'lookup_example.pkl',
                        #                                               file_type="Pickle Lookup Table (*.pkl)",
                        #                                               existing=True)

                        #self.pixel_size = QtShortCuts.QInputString(None, "pixel_size", "1.29", name_post='µm/px',
                        #                                           type=float)

                        with QtShortCuts.QHBoxLayout():
                            self.x0 = QtShortCuts.QInputString(None, "r_min", "2", type=float)
                            self.x1 = QtShortCuts.QInputString(None, "r_max", "None", type=float,
                                                               name_post='spheroid radii', allow_none=True)

                    self.input_button = QtShortCuts.QPushButton(None, "calculate forces", self.start_process)


        self.setParameterMapping("force_parameters", {
            "lookup_table": self.lookup_table,
            "r_min": self.x0,
            "r_max": self.x1,
        })

    def choose_lookup(self):

        self.lookup_gui = SelectLookup()
        self.lookup_gui.exec()

        if self.lookup_gui.result is not None:
            self.lookup_table.setValue(self.lookup_gui.result)
        self.input_button.setEnabled(self.check_available(self.result))

    def check_available(self, result: ResultSpheroid) -> bool:
        if self.lookup_table.value() == "None" or result.displacements is None or len(result.displacements) == 0:
            return False
        return True

    def check_evaluated(self, result: ResultSpheroid) -> bool:
        return True

    def setResult(self, result: ResultSpheroid):
        super().setResult(result)

    def process(self, result: ResultSpheroid, force_parameters: dict):
        jf.force.reconstruct_gui(result,  # PIV output folder
                             str(self.lookup_table.value()),  # lookup table
                             result.pixel_size,  # pixel size (µm)
                             r_min=force_parameters["r_min"],
                             r_max=force_parameters["r_max"])
        result.save()

    def get_code(self) -> Tuple[str, str]:
        import_code = ""

        results = []
        def code(force_parameters1):  # pragma: no cover
            for result in results:
                result.force_parameters = force_parameters1
                jf.force.reconstruct_gui(result,
                                         result.force_parameters["lookup_table"],
                                         result.pixel_size,
                                         r_min=result.force_parameters["r_min"],
                                         r_max=result.force_parameters["r_max"])
                result.save()

        data = dict(
            force_parameters1=self.result.force_parameters,
        )

        code = get_code(code, data)
        return import_code, code
