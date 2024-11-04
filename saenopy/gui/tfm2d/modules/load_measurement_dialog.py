from qtpy import QtCore, QtWidgets, QtGui

from saenopy.gui.common import QtShortCuts
from saenopy.examples import get_examples_2D
from saenopy.gui.common.AddFilesDialog import AddFilesDialog


class AddFilesDialog(AddFilesDialog):
    file_extension = ".saenopy2D"
    settings_group = "open_2d"

    examples_list = get_examples_2D()

    def add_new_measurement_tab(self):
        with self.tabs.createTab("New Measurement") as self.tab:
            with QtShortCuts.QHBoxLayout():
                with QtShortCuts.QVBoxLayout():
                    label1 = QtWidgets.QLabel("cell image").addToLayout()
                    self.stack_bf_input = QtShortCuts.QInputFilename(None, None, file_type="Image Files (*.tif *.tiff *.jpg *.jpeg *.png)",
                                                                     settings=self.settings,
                                                                     settings_key=f"{self.settings_group}/input0", allow_edit=True,
                                                                     existing=True)
                with QtShortCuts.QVBoxLayout():
                    label1 = QtWidgets.QLabel("reference").addToLayout()
                    self.stack_reference_input = QtShortCuts.QInputFilename(None, None, file_type="Image Files (*.tif *.tiff *.jpg *.jpeg *.png)",
                                                                            settings=self.settings,
                                                                            settings_key=f"{self.settings_group}/input1", allow_edit=True,
                                                                            existing=True)
                with QtShortCuts.QVBoxLayout():
                    label1 = QtWidgets.QLabel("deformed").addToLayout()
                    self.stack_data_input = QtShortCuts.QInputFilename(None, None, file_type="Image Files (*.tif *.tiff *.jpg *.jpeg *.png)",
                                                                       settings=self.settings,
                                                                       settings_key=f"{self.settings_group}/input2", allow_edit=True,
                                                                       existing=True)
            self.pixel_size = QtShortCuts.QInputString(None, "pixel size", 0.201, settings=self.settings, name_post='µm/px',
                                                       settings_key=f"{self.settings_group}/pixel_size", allow_none=False, type=float)
            QtShortCuts.currentLayout().addStretch()
            self.outputText = QtShortCuts.QInputFolder(None, "output", settings=self.settings,
                                                       settings_key=f"{self.settings_group}/wildcard", allow_edit=True)
            with QtShortCuts.QHBoxLayout():
                # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                QtShortCuts.currentLayout().addStretch()
                self.button_addList00 = QtShortCuts.QPushButton(None, "cancel", self.reject)

                self.button_addList0 = QtShortCuts.QPushButton(None, "ok", self.accept_new)
