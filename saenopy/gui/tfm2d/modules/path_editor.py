from qtpy import QtWidgets
from pathlib import Path
import re

from saenopy.gui.common import QtShortCuts
from saenopy.gui.solver.modules.path_editor import PathChanger


def start_path_change(parent, result):
    path_editor = PathEditor(parent, result)
    if not path_editor.exec():
        return

    result.bf = path_editor.input_folder.value()
    result.reference_stack = path_editor.input_folder2.value()
    result.input = path_editor.input_folder3.value()

    #if path_editor.input_pack.value():
    #    for stack in result.stacks:
    #        stack.pack_files()
    #    if result.stack_reference:
    #        result.stack_reference.pack_files()

    if path_editor.input_save.value():
        result.save()
        #print("saved")
        return


class PathEditor(QtWidgets.QDialog):
    def __init__(self, parent, result):
        super().__init__(parent)
        self.setWindowTitle("Change Path")
        with QtShortCuts.QVBoxLayout(self) as layout:
            QtWidgets.QLabel("Change path where the images are stored.").addToLayout()
            self.label = QtShortCuts.SuperQLabel(
                f"The current path is {result.bf}.").addToLayout()
            self.label.setWordWrap(True)
            #self.input_pack = QtShortCuts.QInputBool(None, "pack files", False)
            self.input_save = QtShortCuts.QInputBool(None, "save", False)
            self.input_folder = QtShortCuts.QInputFolder(None, "", Path(result.bf), allow_edit=True)
            self.input_folder2 = QtShortCuts.QInputFolder(None, "", Path(result.reference_stack), allow_edit=True)
            self.input_folder3 = QtShortCuts.QInputFolder(None, "", Path(result.input), allow_edit=True)

            with QtShortCuts.QHBoxLayout():
                self.button_addList0 = QtShortCuts.QPushButton(None, "cancel", self.reject)
                self.button_addList0 = QtShortCuts.QPushButton(None, "ok", self.accept)

