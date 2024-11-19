import os
import qtawesome as qta
from qtpy import QtCore, QtWidgets, QtGui
import numpy as np
import traceback

from saenopy.gui.tfm2d.modules.result import get_stacks2D, Result2D
from saenopy.gui.common import QtShortCuts

from .load_measurement_dialog import AddFilesDialog
from saenopy.gui.common.AddFilesDialog import FileExistsDialog
from .draw import DrawWindow
from .DisplayCellImage import DisplayCellImage
from .DisplayRelaxed import DeformationDetector
from .DisplayDeformed import DeformationDetector2
from .CalculateDisplacements import DeformationDetector3
from .CalculateForces import Force
from .CalculateForceGeneration import ForceGeneration
from .CalculateStress import CalculateStress
from .path_editor import start_path_change
from saenopy.examples import get_examples_2D

from saenopy.gui.common.BatchEvaluateBase import BatchEvaluateBase
from ...solver.modules.exporter.Exporter import ExportViewer
from saenopy.gui.common.ModuleScaleBar import ModuleScaleBar


class BatchEvaluate(BatchEvaluateBase):
    settings_key = "Seanopy_deformation"
    file_extension = ".saenopy2D"
    result: Result2D = None

    result_params = ["piv_parameters", "force_parameters"]

    def add_modules(self):
        layout0 = QtShortCuts.currentLayout()
        layout0.parent().setMaximumWidth(420)
        layout0.setContentsMargins(0, 0, 0, 0)
        self.sub_bf = DisplayCellImage(self, layout0)
        self.sub_draw = DeformationDetector(self, layout0)
        self.sub_draw2 = DeformationDetector2(self, layout0)
        self.sub_draw3 = DeformationDetector3(self, layout0)
        self.sub_force = Force(self, layout0)
        self.sub_force_gen = ForceGeneration(self, layout0)
        self.sub_stress = CalculateStress(self, layout0)
        self.sub_module_export = ExportViewer(self, layout0)
        layout0.addStretch()

        box = QtWidgets.QGroupBox("painting").addToLayout()
        with QtShortCuts.QVBoxLayout(box) as layout:
            self.slider_cursor_width = QtShortCuts.QInputNumber(None, "cursor width", 10, 1, 100, True, float=False)
            self.slider_cursor_width.valueChanged.connect(lambda x: self.draw.setCursorSize(x))
            self.slider_cursor_opacity = QtShortCuts.QInputNumber(None, "mask opacity", 0.5, 0, 1, True, float=True)
            self.slider_cursor_opacity.valueChanged.connect(lambda x: self.draw.setOpacity(x))
            with QtShortCuts.QHBoxLayout():
                self.button_red = QtShortCuts.QPushButton(None, "tractions", lambda x: self.draw.setColor(1),
                                                          icon=qta.icon("fa5s.circle", color="red"))
                self.button_green = QtShortCuts.QPushButton(None, "cell boundary (optional)", lambda x: self.draw.setColor(2),
                                                            icon=qta.icon("fa5s.circle", color="green"))
                # self.button_blue = QtShortCuts.QPushButton(None, "blue", lambda x: self.draw.setColor(3), icon=qta.icon("fa5s.circle", color="blue"))
            QtWidgets.QLabel("hold 'alt' key for eraser").addToLayout()
        self.modules = [self.sub_bf, self.sub_draw2, self.sub_draw3, self.sub_force, self.sub_force_gen, self.sub_stress]

    def add_tabs(self):
        with QtShortCuts.QVBoxLayout() as layout:
            with QtShortCuts.QTabBarWidget(layout) as self.tabs:
                self.tabs.setMinimumWidth(500)
                old_tab = None
                cam_pos = None

                def tab_changed(x):
                    nonlocal old_tab, cam_pos
                    tab = self.tabs.currentWidget()
                    self.tab_changed.emit(tab)

                self.tabs.currentChanged.connect(tab_changed)
                pass
            self.draw = DrawWindow(self, QtShortCuts.currentLayout())
            self.draw.signal_mask_drawn.connect(self.on_mask_drawn)
            self.scale1 = ModuleScaleBar(self, self.draw.view1)

    def generate_data(self):
        new_path = QtWidgets.QFileDialog.getSaveFileName(None, "Save Data CSV", os.getcwd(),
                                                         "Comma Separated File (*.csv)")
        if not new_path:
            return None
        # ensure filename ends in .py
        if not new_path.endswith(".csv"):
            new_path += ".csv"

        key_units = {
            "filename": "",
            
            "contractility": "N",
            "area Traction Area": "m2",
            "strain energy": "J",
            
            "area Cell Area": "m2",
            "cell number": "",
            "center of object": "",

            "mean normal stress Cell Area": "N/m",
            "max normal stress Cell Area": "N/m",
            "max shear stress Cell Area": "N/m",
            "cv mean normal stress Cell Area": "",
            "cv max normal stress Cell Area": "",
            "cv max shear stress Cell Area": "",

            "average magnitude line tension": "N/m",
            "std magnitude line tension": "",
            "average normal line tension": "N/m",
            "std normal line tension": "",
            "average shear line tension": "N/m",
            "std shear line tension": "",

            "average cell force": "N/m",
            "average cell pressure": "N/m",
            "average cell shear": "N/m",
            "std cell force": "",
            "std cell pressure": "",
            "std cell shear": "",
        }
        data = ""
        data += ",".join(key_units.keys())
        data += "\n"
        data += ",".join(key_units.values())
        data += "\n"
        for row in self.list.data:
            result = row[2]
            if result is None:
                continue
            data += result.bf
            data += ",".join([str(result.res_dict.get(key, "")) for key in key_units.keys()])
            data += "\n"
            print(result.res_dict)
        with open(new_path, "w") as f:
            f.write(data)
        print(data)

    def on_mask_drawn(self):
        try:
            result = self.list.data[self.list.currentRow()][2]
        except IndexError:
            return
        if result:
            result.mask = self.draw.get_image()

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
            # if there was a bf stack selected
            bf_stack = dialog.stack_bf_input.value()

            # if there was a reference stack selected
            reference_stack = dialog.stack_reference_input.value()

            # the active selected stack
            active_stack = dialog.stack_data_input.value()

            try:
                results = get_stacks2D(dialog.outputText.value(),
                    bf_stack, active_stack, reference_stack, pixel_size=dialog.pixel_size.value(),
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
            example = get_examples_2D()[dialog.mode_data]

            # generate a stack with the examples data
            results = get_stacks2D(
                example["output_path"],
                example["bf"],
                example["deformed"],
                example["reference"],
                example["pixel_size"],
                exist_overwrite_callback=do_overwrite,
            )
            # load all the measurement objects
            for data in results:
                if getattr(data, "is_read", False) is False:
                    data.piv_parameters = example["piv_parameters"]
                    data.force_parameters = example["force_parameters"]
                self.add_data(data)
        elif dialog.mode == "example_evaluated":
                self.load_from_path(dialog.examples_output)

        # update the icons
        self.update_icons()

    def listSelected(self):
        if self.list.currentRow() is not None and self.list.currentRow() < len(self.data):
            pipe = self.data[self.list.currentRow()][2]
            if pipe.mask is None:
                self.draw.setMask(np.zeros(pipe.shape, dtype=np.uint8))
            else:
                self.draw.setMask(pipe.mask.astype(np.uint8))
            self.scale1.setScale([pipe.pixel_size])
            self.set_current_result.emit(pipe)
            tab = self.tabs.currentWidget()
            self.tab_changed.emit(tab)
