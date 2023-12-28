from qtpy import QtCore, QtWidgets, QtGui

from saenopy.gui.common import QtShortCuts
from saenopy.gui.common.stack_selector import StackSelector
from saenopy.gui.common.stack_selector_crop import StackSelectorCrop
from saenopy.gui.common.stack_preview import StackPreview
from saenopy.examples import get_examples
from saenopy.gui.common.AddFilesDialog import AddFilesDialog


class AddFilesDialog(AddFilesDialog):
    file_extension = ".saenopy"
    settings_group = "batch"

    examples_list = get_examples()

    def add_new_measurement_tab(self):
        with self.tabs.createTab("New Measurement") as self.tab:
            with QtShortCuts.QHBoxLayout():
                with QtShortCuts.QVBoxLayout():
                    with QtShortCuts.QHBoxLayout():
                        self.reference_choice = QtShortCuts.QInputChoice(None, "Reference", 0, [0, 1],
                                                                         ["difference between time points",
                                                                          "single stack"])
                        QtShortCuts.currentLayout().addStretch()
                    with QtShortCuts.QHBoxLayout():
                        with QtShortCuts.QHBoxLayout():
                            with QtShortCuts.QVBoxLayout():
                                QtShortCuts.currentLayout().setContentsMargins(0, 0, 0, 0)
                                self.place_holder_widget = QtWidgets.QWidget().addToLayout()
                                layout_place_holder = QtWidgets.QVBoxLayout(self.place_holder_widget)
                                layout_place_holder.addStretch()

                                def ref_changed():
                                    self.stack_reference.setVisible(self.reference_choice.value())
                                    self.place_holder_widget.setVisible(not self.reference_choice.value())

                                self.reference_choice.valueChanged.connect(ref_changed)
                                self.stack_reference = StackSelector(QtShortCuts.currentLayout(), "reference")
                                self.stack_reference.glob_string_changed.connect \
                                    (lambda x, y: self.stack_reference_input.setText(y))
                                self.stack_reference.setVisible(self.reference_choice.value())

                                self.stack_reference_input = QtWidgets.QLineEdit().addToLayout()
                            with QtShortCuts.QVBoxLayout():
                                QtShortCuts.currentLayout().setContentsMargins(0, 0, 0, 0)
                                self.stack_data = StackSelector(QtShortCuts.currentLayout(), "active stack(s)",
                                                                self.stack_reference, use_time=True)
                                self.stack_data.setMinimumWidth(300)
                                self.stack_reference.setMinimumWidth(300)
                                self.place_holder_widget.setMinimumWidth(300)
                                self.stack_data.glob_string_changed.connect(
                                    lambda x, y: self.stack_data_input.setText(y))
                                self.stack_data_input = QtWidgets.QLineEdit().addToLayout()
                    self.stack_crop = StackSelectorCrop(self.stack_data, self.reference_choice,
                                                        self.stack_reference).addToLayout()
                    self.stack_data.stack_crop = self.stack_crop
                self.stack_preview = StackPreview(QtShortCuts.currentLayout(), self.reference_choice,
                                                  self.stack_reference, self.stack_data)
            self.outputText = QtShortCuts.QInputFolder(None, "output", settings=self.settings,
                                                       settings_key=f"{self.settings_group}/wildcard2", allow_edit=True)
            with QtShortCuts.QHBoxLayout():
                # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                QtShortCuts.currentLayout().addStretch()
                self.button_addList00 = QtShortCuts.QPushButton(None, "cancel", self.reject)

                self.button_addList0 = QtShortCuts.QPushButton(None, "ok", self.accept_new)

    def accept_new(self):
        self.mode = "new"
        if self.reference_choice.value() == 1 and self.stack_reference.active is None:
                QtWidgets.QMessageBox.critical(self, "Deformation Detector",
                                               "Provide a stack for the reference state.")
        elif self.stack_data.active is None:
            QtWidgets.QMessageBox.critical(self, "Deformation Detector",
                                           "Provide a stack for the deformed state.")
        elif self.stack_data.get_t_count() <= 1 and self.stack_reference.active is None:
            QtWidgets.QMessageBox.critical(self, "Deformation Detector",
                                           "Provide either a reference stack or a time sequence.")
        elif not self.stack_crop.validator():
            QtWidgets.QMessageBox.critical(self, "Deformation Detector",
                                           "Enter a valid voxel size.")
        elif "{t}" in self.stack_data_input.text() and not self.stack_crop.validator_time():
            QtWidgets.QMessageBox.critical(self, "Deformation Detector",
                                           "Enter a valid time delta.")
        else:
            self.accept()
