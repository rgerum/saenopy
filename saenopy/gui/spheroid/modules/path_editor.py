from qtpy import QtWidgets
from pathlib import Path
import re

from saenopy.gui.common import QtShortCuts
from saenopy.gui.solver.modules.path_editor import PathChanger
from saenopy.gui.spheroid.modules.result import ResultSpheroid


def start_path_change(parent, result: ResultSpheroid):
    path_editor = PathEditor(parent, result)
    if not path_editor.exec():
        return

    path_changer = PathChanger(result.template.replace("*", "{t}"), path_editor.input_folder.value().replace("*", "{t}"))
    print(result.images)
    result.images = [path_changer.change_path(path) for path in result.images]
    print(result.images)
    result.template = path_changer.change_path(result.template.replace("*", "{t}")).replace("{t}", "*")

    #if path_editor.input_folder2 is not None:
    #    path_changer = PathChanger(result.stack_reference.template, path_editor.input_folder2.value())
    #    path_changer.stack_change(result.stack_reference)

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
                f"The current path is {result.template}.").addToLayout()
            self.label.setWordWrap(True)
            #self.input_pack = QtShortCuts.QInputBool(None, "pack files", False)
            self.input_save = QtShortCuts.QInputBool(None, "save", False)
            self.input_folder = QtShortCuts.QInputFolder(None, "", Path(result.template), allow_edit=True)
            #if result.stack_reference:
            #    self.input_folder2 = QtShortCuts.QInputFolder(None, "", Path(result.stack_reference.template),
            #                                                  allow_edit=True)
            #else:
            #    self.input_folder2 = None
            with QtShortCuts.QHBoxLayout():
                self.button_addList0 = QtShortCuts.QPushButton(None, "cancel", self.reject)
                self.button_addList0 = QtShortCuts.QPushButton(None, "ok", self.accept)

